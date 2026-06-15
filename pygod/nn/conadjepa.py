# CONAD-JEPA: Contrastive Attributed Network Anomaly Detection
#             with JEPA-style Latent Prediction
# Gap fixes: PPR dual-view target, uncertainty weighting,
#            adaptive margin, z-score score normalization

"""Neural components for CONAD-JEPA."""

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph, subgraph


def inject_anomalies(x, edge_index, num_nodes, anomaly_ratio=0.1, seed=42):
    """Inject CONAD-style pseudo anomalies into a graph.

    Parameters
    ----------
    x : torch.Tensor
        Node feature matrix.
    edge_index : torch.Tensor
        Edge indices with shape ``[2, num_edges]``.
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
    rng = random.Random(seed)

    x_ano = x.clone()
    edges = edge_index.clone()
    y_pseudo = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
    num_anomalies = max(1, int(anomaly_ratio * num_nodes))
    selected = torch.randperm(num_nodes, generator=generator,
                              device=x.device)[:num_anomalies]
    y_pseudo[selected] = 1

    edge_list = edges.t().tolist()
    existing = {tuple(edge) for edge in edge_list}
    remove_edges = set()

    for node_tensor in selected:
        node = int(node_tensor.item())
        aug_type = rng.randrange(4)

        if aug_type == 0:
            candidates = torch.randperm(num_nodes, generator=generator,
                                        device=x.device)
            added = 0
            for dst_tensor in candidates:
                dst = int(dst_tensor.item())
                if dst == node or (node, dst) in existing:
                    continue
                edge_list.append([node, dst])
                edge_list.append([dst, node])
                existing.add((node, dst))
                existing.add((dst, node))
                added += 1
                if added >= min(10, max(1, num_nodes // 20)):
                    break

        elif aug_type == 1:
            incident = [
                i for i, (src, dst) in enumerate(edge_list)
                if src == node or dst == node
            ]
            keep = set(rng.sample(incident, k=1)) if incident else set()
            remove_edges.update(i for i in incident if i not in keep)

        elif aug_type == 2:
            sample_size = min(50, num_nodes)
            candidates = torch.randperm(num_nodes, generator=generator,
                                        device=x.device)[:sample_size]
            dist = torch.norm(x[node].view(1, -1) - x[candidates], dim=1)
            x_ano[node] = x[candidates[torch.argmax(dist)]]

        else:
            scale = 10.0 if rng.random() < 0.5 else 0.1
            x_ano[node] = x_ano[node] * scale

    if remove_edges:
        edge_list = [edge for i, edge in enumerate(edge_list)
                     if i not in remove_edges]

    if edge_list:
        edge_index_ano = torch.tensor(edge_list, dtype=torch.long,
                                      device=x.device).t().contiguous()
    else:
        edge_index_ano = torch.empty((2, 0), dtype=torch.long,
                                     device=x.device)
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

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.attr_decoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.PReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, z):
        """Decode latent embeddings into adjacency and attributes."""
        a_hat = torch.sigmoid(torch.matmul(z, z.t()))
        x_hat = self.attr_decoder(z)
        return a_hat, x_hat


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
    """CONAD with JEPA-style latent prediction and PPR targets."""

    def __init__(self, in_dim, hid_dim=64, num_layers=2, dropout=0.0,
                 ppr_k=32, target_mode='ppr', ego_hops=1,
                 fast_batch=True, context_mask_rate=1.0):
        super().__init__()
        self.context_encoder = NodeEncoder(in_dim, hid_dim, num_layers,
                                           dropout)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.feature_target_encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        self.predictor = Predictor(hid_dim, hid_dim, hid_dim)
        self.decoder = Decoder(hid_dim, in_dim)
        self.uncertainty_weighting = UncertaintyWeighting(num_tasks=3)
        self.ppr_k = ppr_k
        self.target_mode = target_mode
        self.ego_hops = ego_hops
        self.fast_batch = fast_batch
        self.context_mask_rate = context_mask_rate

    def _context_center(self, v, x_ano, edge_index_ano):
        subset, edge_index_ctx, mapping, _ = k_hop_subgraph(
            int(v), 1, edge_index_ano, relabel_nodes=True,
            num_nodes=x_ano.shape[0])
        x_ctx = x_ano[subset].clone()
        center_idx = int(mapping.item())
        x_ctx[center_idx] = 0.0
        z_ctx = self.context_encoder(x_ctx, edge_index_ctx)
        return z_ctx[center_idx]

    def _target_center(self, v, x, edge_index, topk_indices, topk_values):
        if self.target_mode == 'feature':
            return self.feature_target_encoder(x[v].view(1, -1)).squeeze(0)

        if self.target_mode == 'ego':
            subset, edge_index_sub, mapping, _ = k_hop_subgraph(
                int(v), self.ego_hops, edge_index, relabel_nodes=True,
                num_nodes=x.shape[0])
            z_t = self.target_encoder(x[subset], edge_index_sub)
            return z_t[int(mapping.item())]

        nodes = topk_indices[v]
        values = topk_values[v]
        if not torch.any(nodes == v):
            nodes = nodes.clone()
            values = values.clone()
            nodes[-1] = v
            values[-1] = values.max()

        edge_index_sub, _ = subgraph(nodes, edge_index, relabel_nodes=True)
        x_sub = x[nodes]
        center_idx = torch.nonzero(nodes == v, as_tuple=False)[0].item()
        weights = values / values.sum().clamp(min=1e-12)
        edge_weight = None
        if edge_index_sub.numel() > 0:
            edge_weight = weights[edge_index_sub[1]]
        z_t = self.target_encoder(x_sub, edge_index_sub, edge_weight)
        return z_t[center_idx]

    @staticmethod
    def _build_ppr_edges(topk_indices, topk_values):
        num_nodes, k = topk_indices.shape
        target = torch.arange(num_nodes, device=topk_indices.device)
        target = target.view(-1, 1).expand(-1, k).reshape(-1)
        source = topk_indices.reshape(-1)
        edge_index = torch.stack([source, target], dim=0)
        edge_weight = topk_values.reshape(-1)
        return edge_index, edge_weight

    def _fast_target_centers(self, x, edge_index, topk_indices, topk_values,
                             node_indices):
        if self.target_mode == 'feature':
            return self.feature_target_encoder(x[node_indices])

        if self.target_mode == 'ppr':
            ppr_edge_index, ppr_edge_weight = self._build_ppr_edges(
                topk_indices, topk_values)
            z_target_all = self.target_encoder(x, ppr_edge_index,
                                               ppr_edge_weight)
            return z_target_all[node_indices]

        z_target_all = self.target_encoder(x, edge_index)
        return z_target_all[node_indices]

    def forward(self, x, edge_index, x_ano, edge_index_ano, Pi,
                topk_indices, topk_values, y_pseudo, node_indices=None):
        """Compute CONAD-JEPA loss and reconstruction outputs."""
        del Pi
        if node_indices is None:
            node_indices = torch.arange(x.shape[0], device=x.device)

        if self.target_mode == 'feature':
            z_context_all = self.context_encoder(x_ano, edge_index_ano)
            z_c_center = z_context_all[node_indices]
            z_t_center = self.feature_target_encoder(x[node_indices])
        elif self.fast_batch:
            if self.training and self.context_mask_rate < 1.0:
                mask_prob = torch.rand(node_indices.shape[0],
                                       device=x.device)
                keep = mask_prob < self.context_mask_rate
                if not torch.any(keep):
                    keep[torch.randint(0, keep.shape[0], (1,),
                                       device=x.device)] = True
                active_indices = node_indices[keep]
            else:
                active_indices = node_indices

            x_ctx = x_ano.clone()
            x_ctx[active_indices] = 0.0
            z_context_all = self.context_encoder(x_ctx, edge_index_ano)
            z_c_center = z_context_all[active_indices]
            with torch.no_grad():
                z_t_center = self._fast_target_centers(
                    x, edge_index, topk_indices, topk_values,
                    active_indices)
            node_indices = active_indices
        else:
            z_c_center = torch.stack([
                self._context_center(v, x_ano, edge_index_ano)
                for v in node_indices.tolist()
            ], dim=0)
            with torch.no_grad():
                z_t_center = torch.stack([
                    self._target_center(v, x, edge_index, topk_indices,
                                        topk_values)
                    for v in node_indices.tolist()
                ], dim=0)

        pred = self.predictor(z_c_center)
        a_hat, x_hat = self.decoder(z_c_center)

        if self.target_mode == 'feature':
            pred_norm = F.normalize(pred, dim=-1)
            z_t_norm = F.normalize(z_t_center, dim=-1)
            residual = F.mse_loss(pred_norm, z_t_norm,
                                  reduction='none').sum(dim=-1)
        else:
            pred_norm = F.normalize(pred, dim=-1)
            z_t_norm = F.normalize(z_t_center, dim=-1)
            residual = 1.0 - (pred_norm * z_t_norm).sum(dim=-1)

        y_batch = y_pseudo[node_indices]
        normal_mask = y_batch == 0
        anomaly_mask = y_batch == 1
        if torch.any(normal_mask):
            normal_res = residual[normal_mask].detach()
            m_adaptive = normal_res.mean() + 2.0 * normal_res.std(
                unbiased=False)
            l_jepa_normal = residual[normal_mask].mean()
        else:
            m_adaptive = residual.detach().mean()
            l_jepa_normal = residual.mean()
        if torch.any(anomaly_mask):
            l_jepa_margin = F.relu(
                m_adaptive - residual[anomaly_mask]).pow(2).mean()
        else:
            l_jepa_margin = residual.new_tensor(0.0)
        l_jepa = l_jepa_normal + l_jepa_margin

        x_target = x[node_indices]
        l_attr = F.mse_loss(x_hat, x_target)
        pos_mask = torch.isin(edge_index[0], node_indices) & \
            torch.isin(edge_index[1], node_indices)
        if torch.any(pos_mask):
            local = {int(node): i for i, node in enumerate(
                node_indices.tolist())}
            rows = torch.tensor([local[int(n)] for n in
                                 edge_index[0, pos_mask].tolist()],
                                device=x.device)
            cols = torch.tensor([local[int(n)] for n in
                                 edge_index[1, pos_mask].tolist()],
                                device=x.device)
            l_struct = F.binary_cross_entropy(
                a_hat[rows, cols],
                torch.ones(rows.shape[0], device=x.device))
        else:
            l_struct = a_hat.new_tensor(0.0)

        total_loss = self.uncertainty_weighting([l_attr, l_struct, l_jepa])
        logs = {
            'loss_attr': float(l_attr.detach().cpu()),
            'loss_struct': float(l_struct.detach().cpu()),
            'loss_jepa': float(l_jepa.detach().cpu()),
            'margin': float(m_adaptive.detach().cpu()),
        }
        return total_loss, residual, a_hat, x_hat, logs

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.99):
        """Update target encoder parameters with EMA."""
        for param_c, param_t in zip(self.context_encoder.parameters(),
                                    self.target_encoder.parameters()):
            param_t.data.mul_(momentum)
            param_t.data.add_(param_c.data, alpha=1.0 - momentum)
