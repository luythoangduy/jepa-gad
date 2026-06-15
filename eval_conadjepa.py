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


DATASETS = ['inj_cora', 'inj_amazon', 'weibo', 'reddit']
SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed):
    """Set random seeds for one evaluation run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(method, seed, device):
    """Create a detector for a named method."""
    if method == 'DOMINANT':
        gpu = -1 if device == 'cpu' else int(device)
        return DOMINANT(gpu=gpu)
    if method == 'CONAD':
        gpu = -1 if device == 'cpu' else int(device)
        return CONAD(gpu=gpu)
    if method == 'CONAD-JEPA':
        return CONADJEPA(device='cpu' if device == 'cpu'
                        else f'cuda:{device}', seed=seed)
    raise ValueError(method)


def evaluate_method(method, dataset, device):
    """Evaluate one method on one dataset over all seeds."""
    aucs = []
    aps = []
    data = load_data(dataset)
    labels = data.y.numpy()
    for seed in SEEDS:
        set_seed(seed)
        model = build_model(method, seed, device)
        model.fit(data)
        score = model.decision_score_
        aucs.append(eval_roc_auc(labels, score))
        aps.append(eval_average_precision(labels, score))
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
    args = parser.parse_args()

    methods = ['DOMINANT', 'CONAD', 'CONAD-JEPA']
    results = {}
    for method in methods:
        results[method] = {}
        for dataset in DATASETS:
            results[method][dataset] = evaluate_method(method, dataset,
                                                       args.device)

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
