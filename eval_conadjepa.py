# CONAD-JEPA: Contrastive Attributed Network Anomaly Detection
#             with JEPA-style Latent Prediction
# Gap fixes: PPR dual-view target, uncertainty weighting,
#            adaptive margin, z-score score normalization

"""Evaluate CONAD-JEPA against DOMINANT and CONAD baselines."""

import argparse
import random

import numpy as np
import torch

from pygod.detector import CONAD, CONADJEPA, DOMINANT
from pygod.metric import eval_average_precision, eval_roc_auc
from pygod.utils import load_data

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DATASETS = ['inj_cora', 'inj_amazon', 'weibo', 'reddit']
SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed):
    """Set random seeds for one evaluation run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(method, seed, args):
    """Create a detector for a named method."""
    if method == 'DOMINANT':
        gpu = -1 if args.device == 'cpu' else int(args.device)
        return DOMINANT(gpu=gpu)
    if method == 'CONAD':
        gpu = -1 if args.device == 'cpu' else int(args.device)
        return CONAD(gpu=gpu)
    if method == 'CONAD-JEPA':
        return CONADJEPA(device='cpu' if args.device == 'cpu'
                        else f'cuda:{args.device}',
                        seed=seed,
                        epoch=args.epoch,
                        target_mode=args.conadjepa_target_mode,
                        ego_hops=args.ego_hops,
                        ppr_k=args.ppr_k,
                        grad_clip=args.grad_clip,
                        batch_size=args.batch_size,
                        fast_batch=not args.exact_subgraph,
                        verbose=args.verbose)
    raise ValueError(method)


def evaluate_method(method, dataset, args):
    """Evaluate one method on one dataset over all seeds."""
    aucs = []
    aps = []
    print(f'Loading dataset={dataset} for method={method}...')
    data = load_data(dataset)
    labels = data.y.bool().long().numpy()
    if tqdm is not None:
        seed_iter = tqdm(SEEDS, desc=f'{method} on {dataset}', leave=False)
    else:
        seed_iter = SEEDS
    for seed in seed_iter:
        print(f'Running method={method} dataset={dataset} seed={seed}...')
        set_seed(seed)
        model = build_model(method, seed, args)
        model.fit(data)
        score = model.decision_score_
        auc = eval_roc_auc(labels, score)
        ap = eval_average_precision(labels, score)
        aucs.append(auc)
        aps.append(ap)
        print(f'Finished method={method} dataset={dataset} seed={seed}: '
              f'AUC={auc:.4f} AP={ap:.4f}')
    return np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps)


def format_cell(metrics):
    """Format AUC/AP mean and standard deviation."""
    auc_mean, auc_std, ap_mean, ap_std = metrics
    return f'{auc_mean:.4f} +/- {auc_std:.4f} ({ap_mean:.4f} +/- {ap_std:.4f})'


def main():
    """Run all evaluations and print a formatted table."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu',
                        help="'cpu' or CUDA device index such as '0'.")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--conadjepa-target-mode', default='ppr',
                        choices=['ppr', 'ego', 'feature'])
    parser.add_argument('--ego-hops', type=int, default=1)
    parser.add_argument('--ppr-k', type=int, default=32)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--exact-subgraph', action='store_true',
                        help='Use slower per-node subgraph extraction.')
    args = parser.parse_args()

    methods = ['DOMINANT', 'CONAD', 'CONAD-JEPA']
    results = {}
    total_jobs = len(methods) * len(DATASETS)
    job_id = 0
    for method in methods:
        results[method] = {}
        for dataset in DATASETS:
            job_id += 1
            print(f'[{job_id}/{total_jobs}] Evaluating {method} on '
                  f'{dataset}...')
            results[method][dataset] = evaluate_method(method, dataset, args)

    header = 'Method       | inj_cora | inj_amazon | weibo | reddit'
    sep = '-------------|----------|------------|-------|-------'
    print('Each cell: AUC mean +/- std (AP mean +/- std)')
    print(header)
    print(sep)
    for method in methods:
        row = [method.ljust(12)]
        row.extend(format_cell(results[method][dataset])
                   for dataset in DATASETS)
        print(' | '.join(row))


if __name__ == '__main__':
    main()
