# CONAD-JEPA: Contrastive Attributed Network Anomaly Detection
#             with JEPA-style Latent Prediction
# Refactored: removed PPR/ego/clean-gcn params, simplified
#             train/test flow, unified reconstruction scoring.

"""CONAD-JEPA detector."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import Detector
from ..nn import CONADJEPAModel
from ..nn.conadjepa import inject_anomalies

try:
    from tqdm.auto import trange
except ImportError:
    trange = None


class CONADJEPA(Detector):
    """CONAD with JEPA-style latent prediction (feature target mode).

    Parameters
    ----------
    hid_dim : int, optional
        Hidden dimension. Default: ``64``.
    num_layers : int, optional
        Number of GCN layers. Default: ``2``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``.
    lr : float, optional
        Learning rate. Default: ``1e-3``.
    epoch : int, optional
        Training epochs. Default: ``100``.
    batch_size : int, optional
        Mini-batch size, 0 for full-batch. Default: ``0``.
    alpha : float, optional
        Weight for structure vs attribute error in scoring.
        Default: ``0.5``.
    anomaly_ratio : float, optional
        Ratio of pseudo anomalies to inject. Default: ``0.1``.
    ema_momentum : float, optional
        EMA momentum for feature target encoder. Default: ``0.99``.
    grad_clip : float, optional
        Gradient clipping norm. Default: ``5.0``.
    refresh_anomaly_every : int, optional
        Re-inject pseudo anomalies every N epochs. 0 to keep fixed.
        Default: ``1``.
    attr_loss_weight : float, optional
        Weight for attribute loss. Default: ``1.0``.
    struct_loss_weight : float, optional
        Weight for structure loss. Default: ``1.0``.
    jepa_loss_weight : float, optional
        Weight for JEPA loss. Default: ``1.0``.
    struct_row_all : bool, optional
        If True, structure loss uses batch-to-all adjacency.
        Default: ``True``.
    contamination : float, optional
        Proportion of outliers for threshold. Default: ``0.1``.
    device : str, optional
        Device. Default: ``'cpu'``.
    verbose : bool, optional
        Verbosity. Default: ``False``.
    seed : int, optional
        Random seed. Default: ``42``.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.0,
                 lr=1e-3,
                 epoch=100,
                 batch_size=0,
                 alpha=0.5,
                 anomaly_ratio=0.1,
                 ema_momentum=0.99,
                 grad_clip=5.0,
                 refresh_anomaly_every=1,
                 attr_loss_weight=1.0,
                 struct_loss_weight=1.0,
                 jepa_loss_weight=1.0,
                 struct_row_all=True,
                 contamination=0.1,
                 device='cpu',
                 verbose=False,
                 seed=42,
                 # Legacy params (ignored, kept for backward compat)
                 target_mode='feature',
                 ppr_alpha=0.15,
                 ppr_k=32,
                 ppr_iter=10,
                 ego_hops=1,
                 fast_batch=True,
                 context_mask_rate=1.0,
                 mask_nodes_per_epoch=512,
                 **kwargs):
        super(CONADJEPA, self).__init__(contamination=contamination,
                                        verbose=int(verbose))
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.alpha = alpha
        self.anomaly_ratio = anomaly_ratio
        self.ema_momentum = ema_momentum
        self.grad_clip = grad_clip
        self.refresh_anomaly_every = refresh_anomaly_every
        self.attr_loss_weight = attr_loss_weight
        self.struct_loss_weight = struct_loss_weight
        self.jepa_loss_weight = jepa_loss_weight
        self.struct_row_all = struct_row_all
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
        """Prepare input tensors for training or scoring."""
        self.process_graph(data)
        x = data.x.detach().cpu().float().to(self.device)
        edge_index = data.edge_index.detach().cpu().long().to(self.device)
        num_nodes = x.shape[0]

        if inject:
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
            'x_ano': x_ano,
            'edge_index_ano': edge_index_ano,
            'y_pseudo': y_pseudo,
        }

    def _refresh_anomalies(self, inputs, epoch):
        """Re-inject pseudo anomalies with a different seed."""
        x_ano, edge_index_ano, y_pseudo = inject_anomalies(
            inputs['x'], inputs['edge_index'], inputs['x'].shape[0],
            anomaly_ratio=self.anomaly_ratio,
            seed=self.seed + epoch)
        inputs['x_ano'] = x_ano
        inputs['edge_index_ano'] = edge_index_ano
        inputs['y_pseudo'] = y_pseudo

    @staticmethod
    def _zscore(score):
        """Z-score normalize a score tensor."""
        return (score - score.mean()) / score.std(unbiased=False).clamp(
            min=1e-12)

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
            attr_loss_weight=self.attr_loss_weight,
            struct_loss_weight=self.struct_loss_weight,
            jepa_loss_weight=self.jepa_loss_weight,
            struct_row_all=self.struct_row_all).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        batch_size = num_nodes if self.batch_size == 0 else self.batch_size
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
                out = self.model(
                    inputs['x'], inputs['edge_index'],
                    inputs['x_ano'], inputs['edge_index_ano'],
                    inputs['y_pseudo'],
                    node_indices=node_indices)
                loss = out['loss']
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)
                optimizer.step()
                self.model.update_target_encoder(self.ema_momentum)

                logs = out['logs']
                residual = out['residual']
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
            if self.verbose and trange is not None:
                epoch_iter.set_postfix(
                    loss='{:.4f}'.format(loss_value),
                    attr='{:.4f}'.format(epoch_attr / denom),
                    struct='{:.4f}'.format(epoch_struct / denom),
                    jepa='{:.4f}'.format(epoch_jepa / denom),
                    resid='{:.4f}'.format(epoch_residual / denom),
                    w_attr='{:.4f}'.format(epoch_w_attr / denom),
                    w_struct='{:.4f}'.format(epoch_w_struct / denom),
                    w_jepa='{:.4f}'.format(epoch_w_jepa / denom),
                    feat_online='{:.4f}'.format(
                        epoch_feature_online / denom))
            elif self.verbose:
                print('Epoch {:04d}: loss={:.6f} | attr={:.6f} | '
                      'struct={:.6f} | jepa={:.6f} | residual={:.6f} | '
                      'margin={:.6f} | weighted_attr={:.6f} | '
                      'weighted_struct={:.6f} | weighted_jepa={:.6f} | '
                      'feature_online={:.6f}'
                      .format(
                          epoch + 1, loss_value,
                          epoch_attr / denom,
                          epoch_struct / denom,
                          epoch_jepa / denom,
                          epoch_residual / denom,
                          epoch_margin / denom,
                          epoch_w_attr / denom,
                          epoch_w_struct / denom,
                          epoch_w_jepa / denom,
                          epoch_feature_online / denom))

        self._fit_cache = inputs
        if self.verbose:
            print('CONADJEPA: computing final anomaly scores...')
        self.decision_score_ = self.decision_function(data)
        if self.verbose:
            print('CONADJEPA: final score mean={:.6f} '
                  'std={:.6f} min={:.6f} max={:.6f}'.format(
                      self.decision_score_.mean().item(),
                      self.decision_score_.std(unbiased=False).item(),
                      self.decision_score_.min().item(),
                      self.decision_score_.max().item()))
        self._process_decision_score()
        return self

    def decision_function(self, data, label=None):
        """Return raw anomaly scores for a PyG graph.

        Scores are computed as zscore(JEPA_residual) + zscore(recon_error).
        No pseudo anomaly injection at test time.
        """
        del label
        if self.model is None:
            raise RuntimeError('CONADJEPA must be fitted before scoring.')

        self.process_graph(data)
        x = data.x.detach().float().to(self.device)
        edge_index = data.edge_index.detach().long().to(self.device)
        num_nodes = x.shape[0]

        batch_size = num_nodes if self.batch_size == 0 else self.batch_size
        loader = DataLoader(torch.arange(num_nodes), batch_size=batch_size,
                            shuffle=False)

        residual_all = torch.zeros(num_nodes, device=self.device)
        recon_all = torch.zeros(num_nodes, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for node_indices in loader:
                node_indices = node_indices.to(self.device)
                out = self.model(x, edge_index,
                                 node_indices=node_indices)
                residual_all[node_indices] = out['residual'].detach()
                recon_all[node_indices] = out['recon_error'].detach()

        score = self._zscore(residual_all) + self._zscore(recon_all)

        if self.verbose:
            print('CONADJEPA: residual mean={:.6f} '
                  'std={:.6f} min={:.6f} max={:.6f}'.format(
                      residual_all.mean().item(),
                      residual_all.std(unbiased=False).item(),
                      residual_all.min().item(),
                      residual_all.max().item()))
            print('CONADJEPA: reconstruction mean={:.6f} '
                  'std={:.6f} min={:.6f} max={:.6f}'.format(
                      recon_all.mean().item(),
                      recon_all.std(unbiased=False).item(),
                      recon_all.min().item(),
                      recon_all.max().item()))
        return score.detach().cpu()


__all__ = ['CONADJEPA']
