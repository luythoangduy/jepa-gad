"""
CONADJEPA Full Training Script — Weibo Dataset
Run on Google Colab with GPU.

Usage:
    python train_conadjepa_full.py
    python train_conadjepa_full.py --seeds 0 1 2 3 4
    python train_conadjepa_full.py --epoch 200 --lr 5e-4
"""

import argparse
import time

import numpy as np
import torch

from pygod.detector import CONADJEPA
from pygod.metric import eval_average_precision, eval_roc_auc
from pygod.utils import load_data


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description='CONADJEPA training with full hyperparameter control')

    # --- Dataset ---
    parser.add_argument('--dataset', type=str, default='weibo',
                        choices=['weibo', 'inj_cora', 'inj_amazon', 'reddit'],
                        help='Dataset name. Default: weibo')

    # --- Seeds ---
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds for trials. Default: [42, 123, 456]')

    # --- Device ---
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda', 'cuda:0', or 'cpu'. "
                             "Default: 'cuda'")

    # --- Architecture ---
    parser.add_argument('--hid-dim', type=int, default=64,
                        help='Hidden dimension for encoder/decoder/predictor. '
                             'Default: 64')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of GCN layers in encoder. Default: 2')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate. Default: 0.0')

    # --- Training ---
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of training epochs. Default: 100')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate. Default: 1e-3')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Mini-batch size. 0 = full-batch. Default: 0')
    parser.add_argument('--grad-clip', type=float, default=5.0,
                        help='Gradient clipping norm. 0 = no clipping. '
                             'Default: 5.0')

    # --- JEPA ---
    parser.add_argument('--ema-momentum', type=float, default=0.99,
                        help='EMA momentum for feature target encoder. '
                             'Default: 0.99')
    parser.add_argument('--jepa-loss-weight', type=float, default=1.0,
                        help='Weight for JEPA loss in uncertainty weighting. '
                             'Default: 1.0')

    # --- Reconstruction ---
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Balance between structure and attribute error '
                             'in scoring. 0 = attr only, 1 = struct only. '
                             'Default: 0.5')
    parser.add_argument('--attr-loss-weight', type=float, default=1.0,
                        help='Weight for attribute recon loss. Default: 1.0')
    parser.add_argument('--struct-loss-weight', type=float, default=1.0,
                        help='Weight for structure recon loss. Default: 1.0')
    parser.add_argument('--struct-batch-only', action='store_true',
                        help='Structure loss within batch only (not batch-to-all). '
                             'Default: False (use batch-to-all)')

    # --- Anomaly injection ---
    parser.add_argument('--anomaly-ratio', type=float, default=0.1,
                        help='Ratio of pseudo anomalies to inject. '
                             'Default: 0.1')
    parser.add_argument('--refresh-anomaly-every', type=int, default=1,
                        help='Re-inject pseudo anomalies every N epochs. '
                             '0 = fixed view. Default: 1')

    # --- Scoring ---
    parser.add_argument('--contamination', type=float, default=0.1,
                        help='Expected proportion of outliers for threshold. '
                             'Default: 0.1')

    # --- Output ---
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print training progress. Default: True')
    parser.add_argument('--save-scores', action='store_true',
                        help='Save per-node anomaly scores to .npy file')

    args = parser.parse_args()

    # --- Check device ---
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('WARNING: CUDA not available, falling back to CPU')
        args.device = 'cpu'
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # --- Print config ---
    print('=' * 70)
    print('CONADJEPA Training Configuration')
    print('=' * 70)
    for key, val in sorted(vars(args).items()):
        print(f'  {key:25s}: {val}')
    print('=' * 70)

    # --- Load data ---
    print(f'\nLoading {args.dataset}...')
    data = load_data(args.dataset)
    labels = data.y.bool().long().numpy()
    print(f'  Nodes:     {data.num_nodes}')
    print(f'  Edges:     {data.num_edges}')
    print(f'  Features:  {data.x.shape[1]}')
    print(f'  Anomalies: {labels.sum()} ({labels.mean()*100:.1f}%)')
    print()

    # --- Run trials ---
    all_aucs = []
    all_aps = []
    all_times = []

    for i, seed in enumerate(args.seeds):
        print(f'--- Trial {i+1}/{len(args.seeds)} (seed={seed}) ---')
        set_seed(seed)

        model = CONADJEPA(
            # Architecture
            hid_dim=args.hid_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            # Training
            lr=args.lr,
            epoch=args.epoch,
            batch_size=args.batch_size,
            grad_clip=args.grad_clip,
            # JEPA
            ema_momentum=args.ema_momentum,
            jepa_loss_weight=args.jepa_loss_weight,
            # Reconstruction
            alpha=args.alpha,
            attr_loss_weight=args.attr_loss_weight,
            struct_loss_weight=args.struct_loss_weight,
            struct_row_all=not args.struct_batch_only,
            # Anomaly injection
            anomaly_ratio=args.anomaly_ratio,
            refresh_anomaly_every=args.refresh_anomaly_every,
            # Scoring
            contamination=args.contamination,
            # System
            device=args.device,
            verbose=args.verbose,
            seed=seed,
        )

        t0 = time.time()
        model.fit(data)
        elapsed = time.time() - t0

        score = model.decision_score_
        auc = eval_roc_auc(labels, score)
        ap = eval_average_precision(labels, score)

        all_aucs.append(auc)
        all_aps.append(ap)
        all_times.append(elapsed)

        print(f'  AUC: {auc:.4f}  |  AP: {ap:.4f}  |  Time: {elapsed:.1f}s')

        if args.save_scores:
            import os
            os.makedirs('results', exist_ok=True)
            fname = f'results/conadjepa_{args.dataset}_seed{seed}.npy'
            np.save(fname, score.numpy())
            print(f'  Scores saved to {fname}')
        print()

    # --- Summary ---
    print('=' * 70)
    print(f'RESULTS: {args.dataset} | {args.epoch} epochs | '
          f'{len(args.seeds)} trials')
    print('=' * 70)
    print(f'  AUC:  {np.mean(all_aucs):.4f} +/- {np.std(all_aucs):.4f}  '
          f'(min={np.min(all_aucs):.4f}, max={np.max(all_aucs):.4f})')
    print(f'  AP:   {np.mean(all_aps):.4f} +/- {np.std(all_aps):.4f}  '
          f'(min={np.min(all_aps):.4f}, max={np.max(all_aps):.4f})')
    print(f'  Time: {np.mean(all_times):.1f}s +/- {np.std(all_times):.1f}s')
    print('=' * 70)


if __name__ == '__main__':
    main()
