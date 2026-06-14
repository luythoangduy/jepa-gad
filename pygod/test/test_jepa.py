# -*- coding: utf-8 -*-
import os
import unittest
from numpy.testing import assert_equal
from numpy.testing import assert_raises

import torch
from torch_geometric.nn import GAT
from torch_geometric.seed import seed_everything

from pygod.detector import GADJEPA

seed_everything(717)


class TestGADJEPA(unittest.TestCase):
    def setUp(self):
        self.train_data = torch.load(os.path.join('pygod/test/train_graph.pt'),
                                     weights_only=False)
        self.test_data = torch.load(os.path.join('pygod/test/test_graph.pt'),
                                    weights_only=False)

    def test_full(self):
        detector = GADJEPA(epoch=2, hid_dim=8)
        detector.fit(self.train_data)

        score = detector.predict(return_pred=False, return_score=True)
        assert_equal(score.shape[0], self.train_data.y.shape[0])

        pred, score, conf = detector.predict(self.test_data,
                                             return_pred=True,
                                             return_score=True,
                                             return_conf=True)

        assert_equal(pred.shape[0], self.test_data.y.shape[0])
        assert_equal(score.shape[0], self.test_data.y.shape[0])
        assert_equal(conf.shape[0], self.test_data.y.shape[0])
        assert (conf.min() >= 0)
        assert (conf.max() <= 1)

    def test_sample_and_modes(self):
        detector = GADJEPA(hid_dim=8,
                           num_layers=2,
                           dropout=0.2,
                           backbone=GAT,
                           epoch=1,
                           batch_size=16,
                           num_neigh=1,
                           mask_rate=0.4,
                           contrast_mode='infonce',
                           contrast_weight=0.2,
                           verbose=0,
                           save_emb=True,
                           act_first=True)
        detector.fit(self.train_data)

        pred, score, emb = detector.predict(self.test_data,
                                            return_pred=True,
                                            return_score=True,
                                            return_emb=True)

        assert_equal(pred.shape[0], self.test_data.y.shape[0])
        assert_equal(score.shape[0], self.test_data.y.shape[0])
        assert_equal(emb.shape[1], detector.hid_dim)

    def test_invalid_contrast_mode(self):
        detector = GADJEPA(epoch=1, hid_dim=8, contrast_mode='bad')
        with assert_raises(ValueError):
            detector.fit(self.train_data)
