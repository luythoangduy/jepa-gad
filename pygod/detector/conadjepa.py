# CONAD-JEPA: Contrastive Attributed Network Anomaly Detection
#             with JEPA-style Latent Prediction
# Gap fixes: PPR dual-view target, uncertainty weighting,
#            adaptive margin, z-score score normalization

"""CONAD-JEPA detector."""

import random

import numpy as np
import torch
import torch.nn.functional as F
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
                 target_mode='ppr',
                 ego_hops=1,
                 grad_clip=5.0,
                 fast_batch=True,
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

    def _prepare_inputs(self, data):
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
        elif self.target_mode in ('ego', 'feature'):
            if self.verbose:
                if self.target_mode == 'ego':
                    print('CONADJEPA: using ego-graph target view...')
                else:
                    print('CONADJEPA: using feature-only target view...')
            pi = torch.empty((0, 0), dtype=torch.float32)
            topk_indices = torch.empty((num_nodes, 0), dtype=torch.long)
            topk_values = torch.empty((num_nodes, 0), dtype=torch.float32)
        else:
            raise ValueError("target_mode must be 'ppr', 'ego', "
                             "or 'feature'.")

        x = x_cpu.to(self.device)
        edge_index = edge_index_cpu.to(self.device)
        topk_indices = topk_indices.to(self.device)
        topk_values = topk_values.to(self.device)
        pi_device = pi.to(self.device)
        if self.verbose:
            print('CONADJEPA: injecting pseudo anomalies...')
        x_ano, edge_index_ano, y_pseudo = inject_anomalies(
            x, edge_index, num_nodes, anomaly_ratio=self.anomaly_ratio,
            seed=self.seed)

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

    @staticmethod
    def _zscore(score):
        return (score - score.mean()) / score.std(unbiased=False).clamp(
            min=1e-12)

    def _score_from_outputs(self, residual, a_hat, x_hat, x, edge_index):
        num_nodes = x.shape[0]
        adj = torch.zeros((num_nodes, num_nodes), dtype=a_hat.dtype,
                          device=a_hat.device)
        if edge_index.numel() > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        struct_err = F.binary_cross_entropy(
            a_hat.clamp(1e-6, 1.0 - 1e-6), adj, reduction='none').mean(dim=1)
        attr_err = F.mse_loss(x_hat, x, reduction='none').mean(dim=1)
        recon_err = self.alpha * struct_err + (1.0 - self.alpha) * attr_err
        score = self._zscore(residual.detach()) + self._zscore(
            recon_err.detach())
        return score.detach().cpu()

    def fit(self, data, label=None):
        """Fit CONAD-JEPA on a PyG graph."""
        del label
        self._set_seed()
        inputs = self._prepare_inputs(data)
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
            fast_batch=self.fast_batch).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.batch_size == 0 and self.fast_batch and \
                self.target_mode in ('ppr', 'ego'):
            batch_size = min(512, num_nodes)
        elif self.batch_size == 0:
            batch_size = num_nodes
        else:
            batch_size = self.batch_size
        loader = DataLoader(torch.arange(num_nodes), batch_size=batch_size,
                            shuffle=True)

        self.model.train()
        if self.verbose and trange is not None:
            epoch_iter = trange(self.epoch, desc='CONADJEPA training')
        else:
            epoch_iter = range(self.epoch)
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for node_indices in loader:
                node_indices = node_indices.to(self.device)
                optimizer.zero_grad()
                loss, _, _, _, _ = self.model(
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
                epoch_loss += loss.item() * node_indices.numel()

            loss_value = epoch_loss / num_nodes
            if self.verbose and trange is not None:
                epoch_iter.set_postfix(loss='{:.6f}'.format(loss_value))
            elif self.verbose:
                print('Epoch {:04d}: loss={:.6f}'.format(
                    epoch + 1, loss_value))

        self._fit_cache = inputs
        if self.verbose:
            print('CONADJEPA: computing final anomaly scores...')
        self.decision_score_ = self.decision_function(data)
        self._process_decision_score()
        return self

    def decision_function(self, data, label=None):
        """Return raw anomaly scores for a PyG graph."""
        del label
        if self.model is None:
            raise RuntimeError('CONADJEPA must be fitted before scoring.')

        inputs = self._prepare_inputs(data)
        self.model.eval()
        with torch.no_grad():
            _, residual, a_hat, x_hat, _ = self.model(
                inputs['x'], inputs['edge_index'],
                inputs['x_ano'], inputs['edge_index_ano'],
                inputs['Pi'], inputs['topk_indices'],
                inputs['topk_values'], inputs['y_pseudo'])
            score = self._score_from_outputs(
                residual, a_hat, x_hat, inputs['x'], inputs['edge_index'])
        return score


__all__ = ['CONADJEPA']
