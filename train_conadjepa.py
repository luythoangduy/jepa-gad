# CONAD-JEPA: Contrastive Attributed Network Anomaly Detection
#             with JEPA-style Latent Prediction
# Gap fixes: PPR dual-view target, uncertainty weighting,
#            adaptive margin, z-score score normalization

"""Train CONAD-JEPA on one PyGOD dataset."""

import argparse
import os

import numpy as np

from pygod.detector import CONADJEPA
from pygod.metric import eval_average_precision, eval_roc_auc
from pygod.utils import load_data


def main():
    """Run CONAD-JEPA training and evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='weibo',
                        choices=['weibo', 'inj_cora', 'inj_amazon',
                                 'reddit'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--target-mode', default='ppr',
                        choices=['ppr', 'ego', 'feature'])
    parser.add_argument('--ego-hops', type=int, default=1)
    parser.add_argument('--ppr-k', type=int, default=32)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    args = parser.parse_args()

    data = load_data(args.dataset)
    model = CONADJEPA(device=args.device,
                      verbose=True,
                      epoch=args.epoch,
                      target_mode=args.target_mode,
                      ego_hops=args.ego_hops,
                      ppr_k=args.ppr_k,
                      grad_clip=args.grad_clip)
    model.fit(data)

    score = model.decision_score_
    auc = eval_roc_auc(data.y.numpy(), score)
    ap = eval_average_precision(data.y.numpy(), score)
    print(f'AUC: {auc:.4f} | AP: {ap:.4f}')

    os.makedirs('results', exist_ok=True)
    np.save(os.path.join('results', f'conadjepa_{args.dataset}.npy'),
            score.numpy())


if __name__ == '__main__':
    main()
