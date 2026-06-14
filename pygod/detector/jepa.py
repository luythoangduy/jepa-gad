# -*- coding: utf-8 -*-
"""JEPA-style Graph Anomaly Detection."""

import torch
from torch_geometric.nn import GCN

from . import DeepDetector
from ..nn import GADJEPABase


class GADJEPA(DeepDetector):
    """
    Graph anomaly detector with JEPA/BYOL-style masked prediction.

    GADJEPA masks node attributes, predicts EMA target-encoder
    representations, and scores anomalies by prediction mismatch plus
    distance from an asynchronously updated normal projection center.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 mask_rate=0.1,
                 target_momentum=0.99,
                 normal_momentum=0.99,
                 contrast_mode='infonce',
                 contrast_weight=0.1,
                 normal_weight=0.5,
                 temperature=0.2,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):

        super(GADJEPA, self).__init__(hid_dim=hid_dim,
                                      num_layers=num_layers,
                                      dropout=dropout,
                                      weight_decay=weight_decay,
                                      act=act,
                                      backbone=backbone,
                                      contamination=contamination,
                                      lr=lr,
                                      epoch=epoch,
                                      gpu=gpu,
                                      batch_size=batch_size,
                                      num_neigh=num_neigh,
                                      verbose=verbose,
                                      save_emb=save_emb,
                                      compile_model=compile_model,
                                      **kwargs)

        self.mask_rate = mask_rate
        self.target_momentum = target_momentum
        self.normal_momentum = normal_momentum
        self.contrast_mode = contrast_mode
        self.contrast_weight = contrast_weight
        self.normal_weight = normal_weight
        self.temperature = temperature

    def process_graph(self, data):
        if data.x is None:
            raise ValueError('GADJEPA requires node features in data.x.')

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return GADJEPABase(in_dim=self.in_dim,
                           hid_dim=self.hid_dim,
                           num_layers=self.num_layers,
                           dropout=self.dropout,
                           act=self.act,
                           backbone=self.backbone,
                           mask_rate=self.mask_rate,
                           target_momentum=self.target_momentum,
                           normal_momentum=self.normal_momentum,
                           contrast_mode=self.contrast_mode,
                           contrast_weight=self.contrast_weight,
                           normal_weight=self.normal_weight,
                           temperature=self.temperature,
                           **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)

        loss, normality_score = self.model(x, edge_index,
                                           batch_size=batch_size)
        anomaly_score = -normality_score
        return loss, anomaly_score.detach().cpu()
