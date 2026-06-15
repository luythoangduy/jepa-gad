# CONAD-JEPA: Contrastive Attributed Network Anomaly Detection
#             with JEPA-style Latent Prediction
# Gap fixes: PPR dual-view target, uncertainty weighting,
#            adaptive margin, z-score score normalization

"""CONAD-JEPA detector."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import Detector
from ..nn import CONADJEPAModel
from ..nn.conadjepa import inject_anomalies
from ..utils.ppr import compute_ppr

try:
    from tqdm.auto import trange
except ImportError:
    trange = None


class CONADJEPA(Detector):
    """CONAD with JEPA-style latent prediction over PPR target views."""

    def __init__(self,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.0,
                 lr=1e-3,
                 epoch=100,
                 batch_size=0,
                 alpha=0.5,
                 ppr_alpha=0.15,
                 ppr_k=32,
                 ppr_iter=10,
                 anomaly_ratio=0.1,
                 ema_momentum=0.99,
                 target_mode='feature',
                 ego_hops=1,
                 grad_clip=5.0,
                 fast_batch=True,
                 context_mask_rate=1.0,
                 refresh_anomaly_every=1,
                 attr_loss_weight=1.0,
                 struct_loss_weight=1.0,
                 jepa_loss_weight=1.0,
                 struct_row_all=True,
                 mask_nodes_per_epoch=512,
                 contamination=0.1,
                 device='cpu',
                 verbose=False,
                 seed=42):
        super(CONADJEPA, self).__init__(contamination=contamination,
                                        verbose=int(verbose))
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.alpha = alpha
        self.ppr_alpha = ppr_alpha
        self.ppr_k = ppr_k
        self.ppr_iter = ppr_iter
        self.anomaly_ratio = anomaly_ratio
        self.ema_momentum = ema_momentum
        self.target_mode = target_mode
        self.ego_hops = ego_hops
        self.grad_clip = grad_clip
        self.fast_batch = fast_batch
        self.context_mask_rate = context_mask_rate
        self.refresh_anomaly_every = refresh_anomaly_every
        self.attr_loss_weight = attr_loss_weight
        self.struct_loss_weight = struct_loss_weight
        self.jepa_loss_weight = jepa_loss_weight
        self.struct_row_all = struct_row_all
        self.mask_nodes_per_epoch = mask_nodes_per_epoch
        self.device = torch.device(device)
        self.seed = seed
        self.model = None
        self._fit_cache = None

    def process_graph(self, data):
        """Validate graph input."""
        if data.x is None:
            raise ValueError('CONADJEPA requires node features in data.x.')
        if data.edge_index is None:
            raise ValueError('CONADJEPA requires edge_index.')

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _prepare_inputs(self, data, inject=True):
        self.process_graph(data)
        x_cpu = data.x.detach().cpu().float()
        edge_index_cpu = data.edge_index.detach().cpu().long()
        num_nodes = x_cpu.shape[0]

        if self.target_mode == 'ppr':
            if self.verbose:
                print('CONADJEPA: computing dense PPR matrix...')
            pi = compute_ppr(edge_index_cpu, num_nodes,
                             alpha=self.ppr_alpha,
                             num_iter=self.ppr_iter).float()
            if self.verbose:
                print('CONADJEPA: extracting top-k PPR neighbors...')
            k = min(self.ppr_k, num_nodes)
            topk_values, topk_indices = torch.topk(pi, k=k, dim=1)
        elif self.target_mode in ('ego', 'feature', 'clean-gcn'):
            if self.verbose:
                if self.target_mode == 'ego':
                    print('CONADJEPA: using ego-graph target view...')
                elif self.target_mode == 'feature':
                    print('CONADJEPA: using feature-only target view...')
                else:
                    print('CONADJEPA: using clean-GCN target view...')
            pi = torch.empty((0, 0), dtype=torch.float32)
            topk_indices = torch.empty((num_nodes, 0), dtype=torch.long)
            topk_values = torch.empty((num_nodes, 0), dtype=torch.float32)
        else:
            raise ValueError("target_mode must be 'ppr', 'ego', "
                             "'feature', or 'clean-gcn'.")

        x = x_cpu.to(self.device)
        edge_index = edge_index_cpu.to(self.device)
        topk_indices = topk_indices.to(self.device)
        topk_values = topk_values.to(self.device)
        pi_device = pi.to(self.device)
        if inject:
            if self.verbose:
                print('CONADJEPA: injecting pseudo anomalies...')
            x_ano, edge_index_ano, y_pseudo = inject_anomalies(
                x, edge_index, num_nodes, anomaly_ratio=self.anomaly_ratio,
                seed=self.seed)
        else:
            x_ano = x
            edge_index_ano = edge_index
            y_pseudo = torch.zeros(num_nodes, dtype=torch.long,
                                   device=self.device)

        return {
            'x': x,
            'edge_index': edge_index,
            'Pi': pi_device,
            'topk_indices': topk_indices,
            'topk_values': topk_values,
            'x_ano': x_ano,
            'edge_index_ano': edge_index_ano,
            'y_pseudo': y_pseudo,
        }

    def _refresh_anomalies(self, inputs, epoch):
        if self.verbose and self.target_mode == 'feature':
            print('CONADJEPA feature debug: refreshing pseudo anomalies '
                  'for epoch {}'.format(epoch + 1))
        x_ano, edge_index_ano, y_pseudo = inject_anomalies(
            inputs['x'], inputs['edge_index'], inputs['x'].shape[0],
            anomaly_ratio=self.anomaly_ratio,
            seed=self.seed + epoch)
        inputs['x_ano'] = x_ano
        inputs['edge_index_ano'] = edge_index_ano
        inputs['y_pseudo'] = y_pseudo

    @staticmethod
    def _zscore(score):
        return (score - score.mean()) / score.std(unbiased=False).clamp(
            min=1e-12)

    def _resolve_batch_size(self, num_nodes, scoring=False):
        if scoring and self.batch_size == 0 and self.fast_batch and \
                self.target_mode in ('ppr', 'ego', 'clean-gcn'):
            return min(512, num_nodes)
        if self.batch_size == 0 and self.fast_batch and \
                self.target_mode in ('ppr', 'ego', 'clean-gcn') and \
                self.mask_nodes_per_epoch and \
                self.mask_nodes_per_epoch > 0:
            return num_nodes
        if self.batch_size == 0 and self.fast_batch and \
                self.target_mode in ('ppr', 'ego', 'clean-gcn') and \
                (self.context_mask_rate >= 1.0 or scoring):
            return min(512, num_nodes)
        if self.batch_size == 0:
            return num_nodes
        return self.batch_size

    def _batch_reconstruction_error(self, x_hat, x, edge_index,
                                    node_indices, a_hat):
        diff_attr = torch.pow(x[node_indices] - x_hat, 2)
        attr_err = torch.sqrt(torch.sum(diff_attr, dim=1).clamp(min=1e-12))
        z_center = self.model.last_z_struct_center
        z_all = self.model.last_z_struct_all
        if z_all is not None and self.struct_row_all:
            pred = torch.matmul(z_center, z_all.t())
            target = torch.zeros_like(pred)
            pos_mask = torch.isin(edge_index[0], node_indices)
            if torch.any(pos_mask):
                local = {int(node): i for i, node in enumerate(
                    node_indices.tolist())}
                rows = torch.tensor([local[int(n)] for n in
                                     edge_index[0, pos_mask].tolist()],
                                    device=x.device)
                cols = edge_index[1, pos_mask]
                target[rows, cols] = 1.0
            diff_struct = torch.pow(target - pred, 2)
            struct_err = torch.sqrt(torch.sum(diff_struct, dim=1).clamp(
                min=1e-12))
        else:
            target = torch.zeros_like(a_hat)
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
                target[rows, cols] = 1.0
            diff_struct = torch.pow(target - a_hat, 2)
            struct_err = torch.sqrt(torch.sum(diff_struct, dim=1).clamp(
                min=1e-12))
        return self.alpha * struct_err + (1.0 - self.alpha) * attr_err

    def fit(self, data, label=None):
        """Fit CONAD-JEPA on a PyG graph."""
        del label
        self._set_seed()
        inputs = self._prepare_inputs(data, inject=True)
        x = inputs['x']
        num_nodes, in_dim = x.shape

        self.model = CONADJEPAModel(
            in_dim=in_dim,
            hid_dim=self.hid_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            ppr_k=self.ppr_k,
            target_mode=self.target_mode,
            ego_hops=self.ego_hops,
            fast_batch=self.fast_batch,
            context_mask_rate=self.context_mask_rate,
            attr_loss_weight=self.attr_loss_weight,
            struct_loss_weight=self.struct_loss_weight,
            jepa_loss_weight=self.jepa_loss_weight,
            struct_row_all=self.struct_row_all,
            mask_nodes_per_epoch=self.mask_nodes_per_epoch).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        batch_size = self._resolve_batch_size(num_nodes)
        loader = DataLoader(torch.arange(num_nodes), batch_size=batch_size,
                            shuffle=True)

        self.model.train()
        if self.verbose and trange is not None:
            epoch_iter = trange(self.epoch, desc='CONADJEPA training')
        else:
            epoch_iter = range(self.epoch)
        for epoch in epoch_iter:
            if epoch > 0 and self.refresh_anomaly_every and \
                    self.refresh_anomaly_every > 0 and \
                    epoch % self.refresh_anomaly_every == 0:
                self._refresh_anomalies(inputs, epoch)
            epoch_loss = 0.0
            epoch_attr = 0.0
            epoch_struct = 0.0
            epoch_jepa = 0.0
            epoch_feature_online = 0.0
            epoch_w_attr = 0.0
            epoch_w_struct = 0.0
            epoch_w_jepa = 0.0
            epoch_margin = 0.0
            epoch_residual = 0.0
            epoch_count = 0
            for node_indices in loader:
                node_indices = node_indices.to(self.device)
                optimizer.zero_grad()
                loss, residual, _, _, logs = self.model(
                    inputs['x'], inputs['edge_index'],
                    inputs['x_ano'], inputs['edge_index_ano'],
                    inputs['Pi'], inputs['topk_indices'],
                    inputs['topk_values'], inputs['y_pseudo'],
                    node_indices=node_indices)
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)
                optimizer.step()
                self.model.update_target_encoder(self.ema_momentum)
                batch_count = residual.numel()
                epoch_loss += loss.item() * batch_count
                epoch_attr += logs['loss_attr'] * batch_count
                epoch_struct += logs['loss_struct'] * batch_count
                epoch_jepa += logs['loss_jepa'] * batch_count
                epoch_feature_online += logs['loss_feature_online'] * \
                    batch_count
                epoch_w_attr += logs['weighted_attr'] * batch_count
                epoch_w_struct += logs['weighted_struct'] * batch_count
                epoch_w_jepa += logs['weighted_jepa'] * batch_count
                epoch_margin += logs['margin'] * batch_count
                epoch_residual += residual.detach().mean().item() * \
                    batch_count
                epoch_count += batch_count

            denom = max(1, epoch_count)
            loss_value = epoch_loss / denom
            attr_value = epoch_attr / denom
            struct_value = epoch_struct / denom
            jepa_value = epoch_jepa / denom
            feature_online_value = epoch_feature_online / denom
            w_attr_value = epoch_w_attr / denom
            w_struct_value = epoch_w_struct / denom
            w_jepa_value = epoch_w_jepa / denom
            margin_value = epoch_margin / denom
            residual_value = epoch_residual / denom
            if self.verbose and trange is not None:
                if self.target_mode == 'feature':
                    epoch_iter.set_postfix(
                        loss='{:.4f}'.format(loss_value),
                        attr='{:.4f}'.format(attr_value),
                        struct='{:.4f}'.format(struct_value),
                        jepa='{:.4f}'.format(jepa_value),
                        resid='{:.4f}'.format(residual_value),
                        w_attr='{:.4f}'.format(w_attr_value),
                        w_struct='{:.4f}'.format(w_struct_value),
                        w_jepa='{:.4f}'.format(w_jepa_value),
                        feat_online='{:.4f}'.format(
                            feature_online_value))
                else:
                    epoch_iter.set_postfix(loss='{:.6f}'.format(loss_value))
            elif self.verbose:
                if self.target_mode == 'feature':
                    print('Epoch {:04d}: loss={:.6f} | attr={:.6f} | '
                          'struct={:.6f} | jepa={:.6f} | residual={:.6f} | '
                          'margin={:.6f} | weighted_attr={:.6f} | '
                          'weighted_struct={:.6f} | weighted_jepa={:.6f} | '
                          'feature_online={:.6f}'
                          .format(
                              epoch + 1, loss_value, attr_value,
                              struct_value, jepa_value, residual_value,
                              margin_value, w_attr_value, w_struct_value,
                              w_jepa_value, feature_online_value))
                else:
                    print('Epoch {:04d}: loss={:.6f}'.format(
                        epoch + 1, loss_value))

        self._fit_cache = inputs
        if self.verbose:
            print('CONADJEPA: computing final anomaly scores...')
        self.decision_score_ = self.decision_function(data)
        if self.verbose and self.target_mode == 'feature':
            print('CONADJEPA feature debug: final score mean={:.6f} '
                  'std={:.6f} min={:.6f} max={:.6f}'.format(
                      self.decision_score_.mean().item(),
                      self.decision_score_.std(unbiased=False).item(),
                      self.decision_score_.min().item(),
                      self.decision_score_.max().item()))
        self._process_decision_score()
        return self

    def decision_function(self, data, label=None):
        """Return raw anomaly scores for a PyG graph."""
        del label
        if self.model is None:
            raise RuntimeError('CONADJEPA must be fitted before scoring.')

        inputs = self._prepare_inputs(data, inject=False)
        num_nodes = inputs['x'].shape[0]
        batch_size = self._resolve_batch_size(num_nodes, scoring=True)
        loader = DataLoader(torch.arange(num_nodes), batch_size=batch_size,
                            shuffle=False)
        residual_all = torch.zeros(num_nodes, device=self.device)
        recon_all = torch.zeros(num_nodes, device=self.device)
        self.model.eval()
        with torch.no_grad():
            for node_indices in loader:
                node_indices = node_indices.to(self.device)
                _, residual, a_hat, x_hat, _ = self.model(
                    inputs['x'], inputs['edge_index'],
                    inputs['x_ano'], inputs['edge_index_ano'],
                    inputs['Pi'], inputs['topk_indices'],
                    inputs['topk_values'], inputs['y_pseudo'],
                    node_indices=node_indices)
                recon = self._batch_reconstruction_error(
                    x_hat, inputs['x'], inputs['edge_index'], node_indices,
                    a_hat)
                residual_all[node_indices] = residual.detach()
                recon_all[node_indices] = recon.detach()
        score = self._zscore(residual_all) + self._zscore(recon_all)
        if self.verbose and self.target_mode == 'feature':
            print('CONADJEPA feature debug: residual mean={:.6f} '
                  'std={:.6f} min={:.6f} max={:.6f}'.format(
                      residual_all.mean().item(),
                      residual_all.std(unbiased=False).item(),
                      residual_all.min().item(),
                      residual_all.max().item()))
            print('CONADJEPA feature debug: reconstruction mean={:.6f} '
                  'std={:.6f} min={:.6f} max={:.6f}'.format(
                      recon_all.mean().item(),
                      recon_all.std(unbiased=False).item(),
                      recon_all.min().item(),
                      recon_all.max().item()))
        return score.detach().cpu()


__all__ = ['CONADJEPA']
