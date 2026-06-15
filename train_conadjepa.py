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
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--target-mode', default='ppr',
                        choices=['ppr', 'ego', 'feature'])
    parser.add_argument('--ego-hops', type=int, default=1)
    parser.add_argument('--ppr-k', type=int, default=32)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--context-mask-rate', type=float, default=1.0)
    parser.add_argument('--refresh-anomaly-every', type=int, default=1,
                        help='Refresh pseudo anomaly view every N epochs. '
                             'Use 0 to keep one fixed view.')
    parser.add_argument('--attr-loss-weight', type=float, default=1.0)
    parser.add_argument('--struct-loss-weight', type=float, default=1.0)
    parser.add_argument('--jepa-loss-weight', type=float, default=1.0)
    parser.add_argument('--struct-batch-only', action='store_true',
                        help='Use CONAD-style structure loss only within '
                             'the active batch instead of batch-to-all.')
    parser.add_argument('--exact-subgraph', action='store_true',
                        help='Use slower per-node subgraph extraction.')
    args = parser.parse_args()

    data = load_data(args.dataset)
    model = CONADJEPA(device=args.device,
                      verbose=True,
                      epoch=args.epoch,
                      batch_size=args.batch_size,
                      target_mode=args.target_mode,
                      ego_hops=args.ego_hops,
                      ppr_k=args.ppr_k,
                      grad_clip=args.grad_clip,
                      context_mask_rate=args.context_mask_rate,
                      refresh_anomaly_every=args.refresh_anomaly_every,
                      attr_loss_weight=args.attr_loss_weight,
                      struct_loss_weight=args.struct_loss_weight,
                      jepa_loss_weight=args.jepa_loss_weight,
                      struct_row_all=not args.struct_batch_only,
                      fast_batch=not args.exact_subgraph)
    model.fit(data)

    score = model.decision_score_
    label = data.y.bool().long().numpy()
    auc = eval_roc_auc(label, score)
    ap = eval_average_precision(label, score)
    print(f'AUC: {auc:.4f} | AP: {ap:.4f}')

    os.makedirs('results', exist_ok=True)
    np.save(os.path.join('results', f'conadjepa_{args.dataset}.npy'),
            score.numpy())


if __name__ == '__main__':
    main()
