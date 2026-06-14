import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GADJEPABase(nn.Module):
    """
    JEPA/BYOL-style graph anomaly detector base module.

    The online encoder predicts target-encoder node representations from a
    masked graph view. The target encoder is updated by EMA and acts as the
    slowly moving semantic space. A running normal center provides an
    asynchronous projection target for anomaly scoring.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 mask_rate=0.3,
                 target_momentum=0.99,
                 normal_momentum=0.99,
                 contrast_mode='linear',
                 contrast_weight=0.1,
                 normal_weight=0.5,
                 temperature=0.2,
                 **kwargs):
        super().__init__()

        self.online_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=num_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = MLP(hid_dim, hid_dim, hid_dim)
        self.projector = MLP(hid_dim, hid_dim, hid_dim)
        self.normal_projector = MLP(hid_dim, hid_dim, hid_dim)
        self.mask_token = nn.Parameter(torch.zeros(in_dim))

        self.mask_rate = mask_rate
        self.target_momentum = target_momentum
        self.normal_momentum = normal_momentum
        self.contrast_mode = contrast_mode
        self.contrast_weight = contrast_weight
        self.normal_weight = normal_weight
        self.temperature = temperature

        self.register_buffer('normal_center', torch.zeros(hid_dim))
        self.register_buffer('normal_ready', torch.tensor(False))
        self.emb = None

    @torch.no_grad()
    def update_target_encoder(self):
        for online, target in zip(self.online_encoder.parameters(),
                                  self.target_encoder.parameters()):
            target.data.mul_(self.target_momentum)
            target.data.add_(online.data, alpha=1 - self.target_momentum)

    @torch.no_grad()
    def update_normal_center(self, z, score):
        keep = max(1, int(0.8 * z.shape[0]))
        idx = torch.argsort(score.detach())[:keep]
        batch_center = z.detach()[idx].mean(dim=0)
        if not bool(self.normal_ready):
            self.normal_center.copy_(batch_center)
            self.normal_ready.fill_(True)
        else:
            self.normal_center.mul_(self.normal_momentum)
            self.normal_center.add_(batch_center, alpha=1 - self.normal_momentum)

    def mask_features(self, x):
        num_nodes = x.shape[0]
        mask = torch.rand(num_nodes, device=x.device) < self.mask_rate
        if not torch.any(mask):
            mask[torch.randint(0, num_nodes, (1,), device=x.device)] = True

        x_masked = x.clone()
        x_masked[mask] = self.mask_token
        return x_masked, mask

    def _jepa_loss(self, pred, target, mask):
        pred = F.normalize(pred[mask], dim=-1)
        target = F.normalize(target[mask].detach(), dim=-1)
        return 2 - 2 * (pred * target).sum(dim=-1).mean()

    def _linear_contrastive_loss(self, z_online, z_target, batch_size):
        if self.contrast_mode == 'none' or self.contrast_weight == 0:
            return z_online.new_tensor(0.)

        q = F.normalize(self.projector(z_online[:batch_size]), dim=-1)
        k = F.normalize(z_target[:batch_size].detach(), dim=-1)

        if self.contrast_mode == 'linear':
            return 2 - 2 * (q * k).sum(dim=-1).mean()
        if self.contrast_mode == 'infonce':
            logits = q @ k.t() / self.temperature
            labels = torch.arange(q.shape[0], device=q.device)
            return F.cross_entropy(logits, labels)

        raise ValueError("contrast_mode must be one of 'none', 'linear', "
                         "or 'infonce'.")

    def _normal_projection_loss(self, z_online, z_target, base_score,
                                batch_size):
        z_proj = self.normal_projector(z_online[:batch_size])
        z_proj_norm = F.normalize(z_proj, dim=-1)
        z_target_norm = F.normalize(z_target[:batch_size].detach(), dim=-1)
        normal_dist = 1 - (z_proj_norm * z_target_norm).sum(dim=-1)

        keep = max(1, int(0.8 * batch_size))
        normal_idx = torch.argsort(base_score.detach())[:keep]
        normal_loss = normal_dist[normal_idx].mean()
        return normal_loss, normal_dist

    def forward(self, x, edge_index, batch_size=None):
        if batch_size is None:
            batch_size = x.shape[0]

        if self.training:
            self.update_target_encoder()
            x_masked, mask = self.mask_features(x)
        else:
            x_masked = x
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        z_online = self.online_encoder(x_masked, edge_index)
        pred = self.predictor(z_online)
        with torch.no_grad():
            if self.training:
                z_target = self.target_encoder(x, edge_index)
            else:
                z_target = self.online_encoder(x, edge_index)

        self.emb = z_online

        jepa_loss = self._jepa_loss(pred, z_target, mask)
        contrast_loss = self._linear_contrastive_loss(z_online,
                                                      z_target,
                                                      batch_size)

        pred_norm = F.normalize(pred[:batch_size], dim=-1)
        target_norm = F.normalize(z_target[:batch_size].detach(), dim=-1)
        pred_error = 1 - (pred_norm * target_norm).sum(dim=-1)

        normal_loss, projector_dist = self._normal_projection_loss(
            z_online, z_target, pred_error, batch_size)

        proj = F.normalize(self.projector(z_online[:batch_size]), dim=-1)
        center = F.normalize(self.normal_center.detach().view(1, -1), dim=-1)
        center_dist = 1 - (proj * center).sum(dim=-1)
        
        score = pred_error.clone()
        if self.normal_weight > 0:
            score += self.normal_weight * projector_dist
        if self.contrast_mode != 'none' and self.contrast_weight > 0:
            score += self.contrast_weight * center_dist

        if self.training:
            self.update_normal_center(proj, score)

        loss = (jepa_loss +
                self.contrast_weight * contrast_loss +
                self.normal_weight * normal_loss)
        return loss, score
