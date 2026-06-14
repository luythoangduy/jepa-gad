import os
import torch
import numpy as np
import random
from pygod.detector import GADJEPA
from pygod.utils import load_data
from torch_geometric.nn import GAT, GCN

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_ranks(score):
    # argsort(-score) gives the indices of nodes from highest to lowest score
    # argsort of that gives the rank of each node
    return np.argsort(np.argsort(-score)) + 1

def main():
    print("Loading reddit dataset...")
    data = load_data('reddit')
    y = data.y.bool().cpu().numpy()
    k = int(y.sum())
    print(f"Total nodes: {len(y)}, Anomalies: {k}")

    settings = {
        'A (Best GAT)': {'backbone': GAT, 'mask_rate': 0.1, 'contrast_mode': 'none', 'normal_weight': 0.0},
        'B (GAT + Normal)': {'backbone': GAT, 'mask_rate': 0.1, 'contrast_mode': 'none', 'normal_weight': 0.5},
        'C (GAT + InfoNCE)': {'backbone': GAT, 'mask_rate': 0.1, 'contrast_mode': 'infonce', 'normal_weight': 0.5},
        'D (High Mask)': {'backbone': GAT, 'mask_rate': 0.3, 'contrast_mode': 'none', 'normal_weight': 0.0},
        'E (GCN Best)': {'backbone': GCN, 'mask_rate': 0.1, 'contrast_mode': 'infonce', 'normal_weight': 0.5}
    }

    # Step 1: Run Baseline
    print("\nRunning Setting A (Baseline/Best GAT)...")
    set_seed(42)
    model = GADJEPA(hid_dim=64, lr=0.01, epoch=100, gpu=-1, verbose=0, **settings['A (Best GAT)'])
    model.fit(data)
    score_A = model.decision_score_.numpy()
    ranks_A = get_ranks(score_A)

    # Find Top 10 FP: y=0, highest scores (lowest rank number)
    fp_mask = (y == 0)
    fp_scores = score_A.copy()
    fp_scores[~fp_mask] = -np.inf
    top_10_fp = np.argsort(-fp_scores)[:10]

    # Find Top 10 FN: y=1, lowest scores (highest rank number)
    fn_mask = (y == 1)
    fn_scores = score_A.copy()
    fn_scores[~fn_mask] = np.inf
    top_10_fn = np.argsort(fn_scores)[:10]

    print(f"Top 10 FPs: {top_10_fp.tolist()}")
    print(f"Top 10 FNs: {top_10_fn.tolist()}")

    # Log results
    results_fp = {name: [] for name in settings.keys()}
    results_fn = {name: [] for name in settings.keys()}

    results_fp['A (Best GAT)'] = ranks_A[top_10_fp].tolist()
    results_fn['A (Best GAT)'] = ranks_A[top_10_fn].tolist()

    for name, config in list(settings.items())[1:]:
        print(f"Running Setting {name}...")
        set_seed(42)
        model = GADJEPA(hid_dim=64, lr=0.01, epoch=100, gpu=-1, verbose=0, **config)
        
        try:
            model.fit(data)
            score = model.decision_score_.numpy()
            ranks = get_ranks(score)
            results_fp[name] = ranks[top_10_fp].tolist()
            results_fn[name] = ranks[top_10_fn].tolist()
        except Exception as e:
            print(f"Failed on {name}: {e}")
            results_fp[name] = [np.nan] * 10
            results_fn[name] = [np.nan] * 10

    def print_table(title, node_ids, results_dict):
        print(f"\n================ {title} ================")
        header = f"{'Node ID':>10}"
        for name in settings.keys():
            header += f" | {name[:12]:>14}"
        print(header)
        for i, node_id in enumerate(node_ids):
            row = f"{node_id:>10}"
            for name in settings.keys():
                val = results_dict[name][i]
                row += f" | {str(val):>14}"
            print(row)

    print_table("FALSE POSITIVE RANKS (Lower = Worse, means highly anomalous)", top_10_fp, results_fp)
    print_table("FALSE NEGATIVE RANKS (Higher = Worse, means normal-like)", top_10_fn, results_fn)

if __name__ == '__main__':
    main()
