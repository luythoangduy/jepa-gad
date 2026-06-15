# CONAD-JEPA: Contrastive Attributed Network Anomaly Detection
#             with JEPA-style Latent Prediction
# Refactored: removed PPR/ego/clean-gcn dead code paths,
#             unified recon error, vectorized structure loss.

"""Neural components for CONAD-JEPA."""

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def inject_anomalies(x, edge_index, num_nodes, anomaly_ratio=0.1, seed=42):
    """Inject CONAD-style pseudo anomalies into a graph (vectorized).

    Four augmentation types (each ~25% of selected nodes):
        0: Add random edges (high-degree)
        1: Remove most edges (outlying)
        2: Replace features with distant node (deviated)
        3: Scale features by 10x or 0.1x (disproportionate)

    Parameters
    ----------
    x : torch.Tensor
        Node feature matrix of shape ``[N, F]``.
    edge_index : torch.Tensor
        Edge indices with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    anomaly_ratio : float, optional
        Ratio of nodes to corrupt. Default: ``0.1``.
    seed : int, optional
        Random seed. Default: ``42``.

    Returns
    -------
    tuple
        ``(x_ano, edge_index_ano, y_pseudo)``.
    """
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    x_ano = x.clone()
    y_pseudo = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
    num_anomalies = max(1, int(anomaly_ratio * num_nodes))
    selected = torch.randperm(num_nodes, generator=generator,
                              device=x.device)[:num_anomalies]
    y_pseudo[selected] = 1

    # Assign augmentation types
    aug_types = torch.randint(0, 4, (num_anomalies,),
                              generator=generator, device=x.device)

    # --- Type 2: deviated features (independent candidates per node) ---
    type2_nodes = selected[aug_types == 2]
    if type2_nodes.numel() > 0:
        sample_size = min(50, num_nodes)
        # rand_idx: [n_type2, sample_size]
        rand_idx = torch.randint(0, num_nodes,
                                 (type2_nodes.shape[0], sample_size),
                                 generator=generator, device=x.device)
        x_candidates = x[rand_idx]  # [n_type2, sample_size, F]
        x_target = x[type2_nodes].unsqueeze(1)  # [n_type2, 1, F]
        dist = torch.norm(x_target - x_candidates, dim=2)  # [n_type2, sample_size]
        farthest = torch.argmax(dist, dim=1)  # [n_type2]
        farthest_idx = rand_idx[torch.arange(type2_nodes.shape[0]), farthest]
        x_ano[type2_nodes] = x[farthest_idx]

    # --- Type 3: disproportionate (vectorized scale) ---
    type3_nodes = selected[aug_types == 3]
    if type3_nodes.numel() > 0:
        coin = torch.rand(type3_nodes.shape[0], generator=generator,
                          device=x.device)
        scales = torch.where(coin < 0.5,
                             torch.tensor(10.0, device=x.device),
                             torch.tensor(0.1, device=x.device))
        x_ano[type3_nodes] = x_ano[type3_nodes] * scales.unsqueeze(1)

    # --- Type 0: add random edges (vectorized) ---
    type0_nodes = selected[aug_types == 0]
    new_edges_parts = []
    if type0_nodes.numel() > 0:
        num_to_add = min(10, max(1, num_nodes // 20))
        # For each type0 node, sample num_to_add random destinations
        rand_dsts = torch.randint(0, num_nodes,
                                  (type0_nodes.shape[0], num_to_add),
                                  generator=generator, device=x.device)
        src_expand = type0_nodes.unsqueeze(1).expand_as(rand_dsts)
        # Forward edges: src -> dst
        fwd_src = src_expand.reshape(-1)
        fwd_dst = rand_dsts.reshape(-1)
        # Remove self-loops
        not_self = fwd_src != fwd_dst
        fwd_src = fwd_src[not_self]
        fwd_dst = fwd_dst[not_self]
        # Add both directions
        new_src = torch.cat([fwd_src, fwd_dst])
        new_dst = torch.cat([fwd_dst, fwd_src])
        new_edges_parts.append(torch.stack([new_src, new_dst], dim=0))

    # --- Type 1: remove edges (vectorized boolean mask) ---
    type1_nodes = selected[aug_types == 1]
    if type1_nodes.numel() > 0:
        # Remove all edges incident to type1 nodes
        incident = (torch.isin(edge_index[0], type1_nodes) |
                    torch.isin(edge_index[1], type1_nodes))
        edge_index_kept = edge_index[:, ~incident]
    else:
        edge_index_kept = edge_index

    # --- Assemble final edge_index and remove duplicates ---
    parts = [edge_index_kept]
    parts.extend(new_edges_parts)
    edge_index_ano = torch.cat(parts, dim=1)
    edge_index_ano = torch.unique(edge_index_ano, dim=1)

    return x_ano, edge_index_ano, y_pseudo


class NodeEncoder(nn.Module):
    """GCN node encoder used by CONAD-JEPA."""

    def __init__(self, in_dim, out_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        dims = [in_dim] + [out_dim] * num_layers
        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.acts = nn.ModuleList([
            nn.PReLU() for _ in range(max(0, num_layers - 1))
        ])

    def forward(self, x, edge_index, edge_weight=None):
        """Encode nodes into latent representations."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i != len(self.convs) - 1:
                x = self.acts[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Predictor(nn.Module):
    """MLP predictor from context embeddings to target embeddings."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.PReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, z_context):
        """Predict target latent embeddings."""
        h = self.linear1(z_context)
        if h.shape[0] > 1:
            h = self.bn(h)
        h = self.act(h)
        return self.linear2(h)


class Decoder(nn.Module):
    """Structure and attribute decoder for CONAD-JEPA."""

    def __init__(self, in_dim, out_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        attr_dims = [in_dim] + [in_dim] * max(0, num_layers - 1) + [out_dim]
        self.attr_convs = nn.ModuleList([
            GCNConv(attr_dims[i], attr_dims[i + 1])
            for i in range(len(attr_dims) - 1)
        ])
        self.attr_acts = nn.ModuleList([
            nn.PReLU() for _ in range(max(0, len(attr_dims) - 2))
        ])
        self.struct_decoder = NodeEncoder(in_dim, in_dim,
                                          max(1, num_layers - 1),
                                          dropout)

    def forward(self, z, edge_index):
        """Decode latent embeddings into adjacency and attributes."""
        x_hat = z
        for i, conv in enumerate(self.attr_convs):
            x_hat = conv(x_hat, edge_index)
            if i != len(self.attr_convs) - 1:
                x_hat = self.attr_acts[i](x_hat)
                x_hat = F.dropout(x_hat, p=self.dropout,
                                  training=self.training)
        z_struct = self.struct_decoder(z, edge_index)
        return x_hat, z_struct


class UncertaintyWeighting(nn.Module):
    """Learned homoscedastic uncertainty weighting for multiple losses."""

    def __init__(self, num_tasks=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """Return the uncertainty-weighted total loss."""
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total

    def get_weights(self):
        """Return normalized task weights."""
        weights = torch.exp(-self.log_vars.detach())
        return weights / weights.sum().clamp(min=1e-12)


class CONADJEPAModel(nn.Module):
    """CONAD with JEPA-style latent prediction (feature target mode).

    Architecture:
        - context_encoder (GCN): encodes corrupted graph features
        - feature_encoder (MLP): online feature projection
        - feature_target_encoder (MLP, EMA): target feature projection
        - predictor (MLP): predicts target from context embeddings
        - decoder: reconstructs node attributes and adjacency structure
        - uncertainty_weighting: learned task balancing

    The JEPA path detects contextual anomalies (feature mismatch),
    while the reconstruction path detects structural anomalies
    (topology mismatch). Both are complementary.
    """

    def __init__(self, in_dim, hid_dim=64, num_layers=2, dropout=0.0,
                 attr_loss_weight=1.0, struct_loss_weight=1.0,
                 jepa_loss_weight=1.0, struct_row_all=True):
        super().__init__()
        self.context_encoder = NodeEncoder(in_dim, hid_dim, num_layers,
                                           dropout)
        self.feature_encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        self.feature_target_encoder = copy.deepcopy(self.feature_encoder)
        for param in self.feature_target_encoder.parameters():
            param.requires_grad = False
        self.predictor = Predictor(hid_dim, hid_dim, hid_dim)
        self.decoder = Decoder(hid_dim, in_dim, num_layers, dropout)
        self.uncertainty_weighting = UncertaintyWeighting(num_tasks=3)
        self.attr_loss_weight = attr_loss_weight
        self.struct_loss_weight = struct_loss_weight
        self.jepa_loss_weight = jepa_loss_weight
        self.struct_row_all = struct_row_all

    def _vectorized_structure_target(self, node_indices, edge_index,
                                     num_cols):
        """Build sparse adjacency target matrix using vectorized ops.

        Parameters
        ----------
        node_indices : torch.Tensor
            Indices of center (row) nodes.
        edge_index : torch.Tensor
            Full graph edge index ``[2, E]``.
        num_cols : int
            Number of columns (all nodes if struct_row_all, else batch).

        Returns
        -------
        target : torch.Tensor
            Dense target matrix of shape ``[len(node_indices), num_cols]``.
        """
        sorted_idx, sort_perm = node_indices.sort()

        if self.struct_row_all:
            # rows = batch nodes, cols = all nodes
            pos_mask = torch.isin(edge_index[0], node_indices)
            if not torch.any(pos_mask):
                return torch.zeros(node_indices.shape[0], num_cols,
                                   device=node_indices.device)
            src = edge_index[0, pos_mask]
            dst = edge_index[1, pos_mask]
            # Map src node IDs to local row indices
            insert_pos = torch.searchsorted(sorted_idx, src)
            rows = sort_perm[insert_pos.clamp(max=sort_perm.shape[0] - 1)]
            cols = dst
        else:
            # rows and cols both within batch
            pos_mask = (torch.isin(edge_index[0], node_indices) &
                        torch.isin(edge_index[1], node_indices))
            if not torch.any(pos_mask):
                return torch.zeros(node_indices.shape[0], num_cols,
                                   device=node_indices.device)
            src = edge_index[0, pos_mask]
            dst = edge_index[1, pos_mask]
            insert_pos_r = torch.searchsorted(sorted_idx, src)
            rows = sort_perm[insert_pos_r.clamp(max=sort_perm.shape[0] - 1)]
            insert_pos_c = torch.searchsorted(sorted_idx, dst)
            cols = sort_perm[insert_pos_c.clamp(max=sort_perm.shape[0] - 1)]

        target = torch.zeros(node_indices.shape[0], num_cols,
                              device=node_indices.device)
        target[rows, cols] = 1.0
        return target

    def _compute_recon_error(self, x_hat_batch, x_batch, z_struct_center,
                             z_struct_all, node_indices, edge_index, alpha):
        """Compute per-node reconstruction error (attr + struct).

        This function is used by both training (for loss) and
        inference (for scoring), ensuring consistency.

        Parameters
        ----------
        x_hat_batch : torch.Tensor
            Reconstructed features for batch nodes.
        x_batch : torch.Tensor
            Original features for batch nodes.
        z_struct_center : torch.Tensor
            Structure embeddings for batch nodes.
        z_struct_all : torch.Tensor or None
            Structure embeddings for all nodes (if struct_row_all).
        node_indices : torch.Tensor
            Indices of batch nodes.
        edge_index : torch.Tensor
            Full graph edge index.
        alpha : float
            Weight for structure error.

        Returns
        -------
        recon_error : torch.Tensor
            Per-node reconstruction error of shape ``[batch_size]``.
        """
        # Attribute error
        diff_attr = torch.pow(x_batch - x_hat_batch, 2)
        attr_err = torch.sqrt(torch.sum(diff_attr, dim=1).clamp(min=1e-12))

        # Structure error (vectorized)
        if z_struct_all is not None and self.struct_row_all:
            pred = torch.matmul(z_struct_center, z_struct_all.t())
            num_cols = z_struct_all.shape[0]
        else:
            pred = torch.matmul(z_struct_center, z_struct_center.t())
            num_cols = z_struct_center.shape[0]

        target = self._vectorized_structure_target(
            node_indices, edge_index, num_cols)
        diff_struct = torch.pow(target - pred, 2)
        struct_err = torch.sqrt(
            torch.sum(diff_struct, dim=1).clamp(min=1e-12))

        return alpha * struct_err + (1.0 - alpha) * attr_err

    def forward(self, x, edge_index, x_ano=None, edge_index_ano=None,
                y_pseudo=None, node_indices=None):
        """Compute CONAD-JEPA outputs.

        During training (``self.training == True``), computes loss and
        per-node scores. During eval, computes only per-node scores.

        Parameters
        ----------
        x : torch.Tensor
            Clean node features.
        edge_index : torch.Tensor
            Clean edge index.
        x_ano : torch.Tensor, optional
            Corrupted node features (training only).
        edge_index_ano : torch.Tensor, optional
            Corrupted edge index (training only).
        y_pseudo : torch.Tensor, optional
            Pseudo anomaly labels (training only).
        node_indices : torch.Tensor, optional
            Batch node indices.

        Returns
        -------
        dict
            Keys: ``'residual'``, ``'recon_error'``.
            During training, also includes ``'loss'`` and ``'logs'``.
        """
        if node_indices is None:
            node_indices = torch.arange(x.shape[0], device=x.device)

        # --- JEPA target: always clean features through EMA MLP ---
        with torch.no_grad():
            z_t_center = self.feature_target_encoder(x[node_indices])

        # --- Context encoder ---
        if self.training:
            # Train: encode corrupted graph (no center masking,
            # validated by ablation H5)
            z_context_all = self.context_encoder(x_ano, edge_index_ano)
            decode_edge = edge_index_ano
        else:
            # Test: encode clean graph (standard transductive eval)
            z_context_all = self.context_encoder(x, edge_index)
            decode_edge = edge_index

        z_c_center = z_context_all[node_indices]

        # --- JEPA prediction + residual (MSE on normalized vectors,
        #     validated by ablation H6) ---
        pred = self.predictor(z_c_center)
        pred_norm = F.normalize(pred, dim=-1)
        z_t_norm = F.normalize(z_t_center, dim=-1)
        residual = F.mse_loss(pred_norm, z_t_norm,
                              reduction='none').sum(dim=-1)

        # --- Feature online loss (critical for AP,
        #     validated by ablation H7) ---
        z_t_online = self.feature_encoder(x[node_indices])
        z_t_online_norm = F.normalize(z_t_online, dim=-1)
        feature_online_loss = F.mse_loss(
            z_t_online_norm, pred_norm.detach(),
            reduction='none').sum(dim=-1).mean()

        # --- Reconstruction ---
        x_hat_all, z_struct_all = self.decoder(z_context_all, decode_edge)
        x_hat_batch = x_hat_all[node_indices]
        z_struct_center = z_struct_all[node_indices]
        z_struct_ref = z_struct_all if self.struct_row_all else None

        recon_error = self._compute_recon_error(
            x_hat_batch, x[node_indices], z_struct_center,
            z_struct_ref, node_indices, edge_index, alpha=0.5)

        result = {
            'residual': residual,
            'recon_error': recon_error,
        }

        if self.training:
            # --- Margin-based JEPA loss ---
            y_batch = y_pseudo[node_indices]
            normal_mask = y_batch == 0
            anomaly_mask = y_batch == 1
            if torch.any(normal_mask):
                normal_res = residual[normal_mask].detach()
                m_adaptive = (normal_res.mean() +
                              2.0 * normal_res.std(unbiased=False))
                l_jepa_normal = residual[normal_mask].mean()
            else:
                m_adaptive = residual.detach().mean()
                l_jepa_normal = residual.mean()
            if torch.any(anomaly_mask):
                l_jepa_margin = F.relu(
                    m_adaptive - residual[anomaly_mask]).pow(2).mean()
            else:
                l_jepa_margin = residual.new_tensor(0.0)
            l_jepa = l_jepa_normal + l_jepa_margin + feature_online_loss

            # --- Reconstruction losses ---
            l_attr = recon_error.mean()  # already combined attr+struct
            # For uncertainty weighting, we split attr and struct
            diff_attr = torch.pow(x[node_indices] - x_hat_batch, 2)
            attr_loss = torch.sqrt(
                torch.sum(diff_attr, dim=1).clamp(min=1e-12)).mean()
            struct_loss = recon_error.mean() - (1.0 - 0.5) * attr_loss
            # Simpler: just use recon_error decomposition
            # Actually let's compute struct error separately for weighting
            if z_struct_ref is not None:
                pred_s = torch.matmul(z_struct_center, z_struct_all.t())
                num_cols = z_struct_all.shape[0]
            else:
                pred_s = torch.matmul(z_struct_center, z_struct_center.t())
                num_cols = z_struct_center.shape[0]
            target_s = self._vectorized_structure_target(
                node_indices, edge_index, num_cols)
            diff_s = torch.pow(target_s - pred_s, 2)
            struct_loss = torch.sqrt(
                torch.sum(diff_s, dim=1).clamp(min=1e-12)).mean()

            weighted_attr = self.attr_loss_weight * attr_loss
            weighted_struct = self.struct_loss_weight * struct_loss
            weighted_jepa = self.jepa_loss_weight * l_jepa
            total_loss = self.uncertainty_weighting([
                weighted_attr, weighted_struct, weighted_jepa])

            result['loss'] = total_loss
            result['logs'] = {
                'loss_attr': float(attr_loss.detach().cpu()),
                'loss_struct': float(struct_loss.detach().cpu()),
                'loss_jepa': float(l_jepa.detach().cpu()),
                'loss_feature_online': float(
                    feature_online_loss.detach().cpu()),
                'weighted_attr': float(weighted_attr.detach().cpu()),
                'weighted_struct': float(weighted_struct.detach().cpu()),
                'weighted_jepa': float(weighted_jepa.detach().cpu()),
                'margin': float(m_adaptive.detach().cpu()),
            }

        return result

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.99):
        """Update feature target encoder with EMA."""
        for param_o, param_t in zip(self.feature_encoder.parameters(),
                                    self.feature_target_encoder.parameters()):
            param_t.data.mul_(momentum)
            param_t.data.add_(param_o.data, alpha=1.0 - momentum)
