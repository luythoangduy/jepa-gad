"""
CONADJEPA Grid Search Script
Run on Google Colab to automatically test multiple combinations.
Data is loaded exactly ONCE to save time.
"""

import argparse
import time
import itertools
import numpy as np
import torch
from tqdm.auto import tqdm

from pygod.detector import CONADJEPA
from pygod.metric import eval_average_precision, eval_roc_auc
from pygod.utils import load_data


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='CONADJEPA Grid Search')
    
    # --- Data & Device ---
    parser.add_argument('--dataset', type=str, default='weibo')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--contamination', type=float, default=0.04)
    
    # --- Fixed Hyperparams (Best practices) ---
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hid-dim', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    
    # --- Grid Search Params (Pass lists of values) ---
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.2, 0.5, 0.8],
                        help='Grid values for alpha')
    parser.add_argument('--jepa-weights', type=float, nargs='+', default=[0.1, 0.5, 1.0, 2.0],
                        help='Grid values for jepa-loss-weight')
    parser.add_argument('--anomaly-ratios', type=float, nargs='+', default=[0.05, 0.1],
                        help='Grid values for anomaly-ratio')

    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'

    # --- Load Data Once ---
    print(f"Loading {args.dataset}...")
    data = load_data(args.dataset)
    labels = data.y.bool().long().numpy()
    
    # Generate Grid
    keys = ['alpha', 'jepa_weight', 'ano_ratio']
    grid = list(itertools.product(args.alphas, args.jepa_weights, args.anomaly_ratios))
    
    print("=" * 60)
    print(f"Starting Grid Search: {len(grid)} combinations x {len(args.seeds)} seeds")
    print(f"Fixed: Epoch={args.epoch}, LR={args.lr}, Hid={args.hid_dim}")
    print("=" * 60)

    # Store results: dict of { tuple_of_params: {'auc_mean': x, 'ap_mean': y, 'auc_std': x, 'ap_std': y} }
    results = {}

    for idx, (alpha, jepa_w, ano_r) in enumerate(grid):
        print(f"\n[{idx+1}/{len(grid)}] Testing alpha={alpha}, jepa_w={jepa_w}, ano_r={ano_r}")
        
        all_aucs = []
        all_aps = []
        
        pbar = tqdm(args.seeds, desc=f"Seeds")
        for seed in pbar:
            set_seed(seed)

            model = CONADJEPA(
                hid_dim=args.hid_dim, num_layers=args.num_layers, dropout=args.dropout,
                lr=args.lr, epoch=args.epoch, batch_size=args.batch_size, grad_clip=args.grad_clip,
                # Fixed values for now
                ema_momentum=0.99, attr_loss_weight=1.0, struct_loss_weight=1.0, refresh_anomaly_every=1,
                # Grid values
                alpha=alpha, jepa_loss_weight=jepa_w, anomaly_ratio=ano_r,
                # System
                contamination=args.contamination, device=args.device, verbose=False, seed=seed,
            )

            model.fit(data)
            score = model.decision_score_
            auc = eval_roc_auc(labels, score)
            ap = eval_average_precision(labels, score)
            
            all_aucs.append(auc)
            all_aps.append(ap)
            pbar.set_postfix({'AUC': f'{auc:.4f}', 'AP': f'{ap:.4f}'})
        
        results[(alpha, jepa_w, ano_r)] = {
            'auc_mean': np.mean(all_aucs), 'auc_std': np.std(all_aucs),
            'ap_mean': np.mean(all_aps), 'ap_std': np.std(all_aps)
        }

    # --- Print Final Grid Search Table ---
    print("\n" + "=" * 80)
    print(f" GRID SEARCH RESULTS | {args.dataset}")
    print("=" * 80)
    print(f"| {'Alpha':^7} | {'JEPA_w':^8} | {'Ano_rat':^9} || {'AUC (Mean ± Std)':^19} | {'AP (Mean ± Std)':^19} |")
    print("-" * 80)
    
    # Sort by AP descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ap_mean'], reverse=True)
    
    for (alpha, jepa_w, ano_r), metrics in sorted_results:
        auc_str = f"{metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}"
        ap_str = f"{metrics['ap_mean']:.4f} ± {metrics['ap_std']:.4f}"
        print(f"| {alpha:^7.2f} | {jepa_w:^8.2f} | {ano_r:^9.2f} || {auc_str:^19} | {ap_str:^19} |")
    print("=" * 80)

    best_params = sorted_results[0][0]
    best_metrics = sorted_results[0][1]
    print(f"\n🌟 BEST PARAMS (by AP): alpha={best_params[0]}, jepa_weight={best_params[1]}, anomaly_ratio={best_params[2]}")
    print(f"   => AUC: {best_metrics['auc_mean']:.4f} | AP: {best_metrics['ap_mean']:.4f}")

if __name__ == '__main__':
    main()
