"""
CONADJEPA Ablation Study — Validate hypotheses before refactoring.

Tests 4 hypotheses by modifying model behavior minimally:
  H4: Frozen random MLP target vs EMA MLP target
  H5: Feature mode + masking vs no masking (current)
  H6: Cosine loss vs MSE loss (current)
  H7: feature_online_loss = 0 vs current

Usage:
  python ablation_conadjepa.py --dataset inj_cora --epoch 100 --num-trials 3
  python ablation_conadjepa.py --dataset weibo --epoch 100 --num-trials 3
"""

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.seed import seed_everything

from pygod.detector import CONADJEPA
from pygod.metric import eval_average_precision, eval_roc_auc
from pygod.utils import load_data


# ---------------------------------------------------------------------------
# Monkey-patch helpers: each returns a cleanup function
# ---------------------------------------------------------------------------

def _patch_no_ema(model_nn):
    """H4: Disable EMA update for feature_target_encoder → frozen random."""
    original_fn = model_nn.update_target_encoder

    def noop_update(momentum=0.99):
        # Only update context→target GCN (not feature pair)
        for param_c, param_t in zip(model_nn.context_encoder.parameters(),
                                     model_nn.target_encoder.parameters()):
            param_t.data.mul_(momentum)
            param_t.data.add_(param_c.data, alpha=1.0 - momentum)
        # Skip feature_encoder → feature_target_encoder EMA

    model_nn.update_target_encoder = noop_update
    return lambda: setattr(model_nn, 'update_target_encoder', original_fn)


def _patch_mask_centers(model_nn):
    """H5: Add center masking to feature mode training."""
    original_forward = model_nn.forward

    def patched_forward(x, edge_index, x_ano, edge_index_ano, Pi,
                        topk_indices, topk_values, y_pseudo,
                        node_indices=None):
        if node_indices is None:
            node_indices = torch.arange(x.shape[0], device=x.device)

        if model_nn.target_mode == 'feature' and model_nn.training:
            # Mask center nodes before encoding (like fast_batch does)
            x_ctx = x_ano.clone()
            x_ctx[node_indices] = 0.0
            z_context_all = model_nn.context_encoder(x_ctx, edge_index_ano)
            z_c_center = z_context_all[node_indices]
            with torch.no_grad():
                z_t_center = model_nn.feature_target_encoder(
                    x[node_indices])
            z_t_online = model_nn.feature_encoder(x[node_indices])

            # Continue with the rest of the original forward
            pred = model_nn.predictor(z_c_center)
            model_nn.last_z_center = z_c_center
            model_nn.last_z_all = z_context_all

            a_hat_all, x_hat_all, z_struct_all = model_nn.decoder(
                z_context_all, edge_index_ano)
            x_hat = x_hat_all[node_indices]
            model_nn.last_z_struct_all = z_struct_all
            model_nn.last_z_struct_center = z_struct_all[node_indices]
            if model_nn.struct_row_all:
                a_hat = torch.matmul(model_nn.last_z_struct_center,
                                     model_nn.last_z_struct_all.t())
            else:
                a_hat = torch.matmul(model_nn.last_z_struct_center,
                                     model_nn.last_z_struct_center.t())

            pred_norm = F.normalize(pred, dim=-1)
            z_t_norm = F.normalize(z_t_center, dim=-1)
            residual = F.mse_loss(pred_norm, z_t_norm,
                                  reduction='none').sum(dim=-1)
            z_t_online_norm = F.normalize(z_t_online, dim=-1)
            feature_online_loss = F.mse_loss(
                z_t_online_norm, pred_norm.detach(),
                reduction='none').sum(dim=-1).mean()

            y_batch = y_pseudo[node_indices]
            normal_mask = y_batch == 0
            anomaly_mask = y_batch == 1
            if torch.any(normal_mask):
                normal_res = residual[normal_mask].detach()
                m_adaptive = normal_res.mean() + 2.0 * normal_res.std(
                    unbiased=False)
                l_jepa_normal = residual[normal_mask].mean()
            else:
                m_adaptive = residual.detach().mean()
                l_jepa_normal = residual.mean()
            if torch.any(anomaly_mask):
                l_jepa_margin = F.relu(
                    m_adaptive - residual[anomaly_mask]).pow(2).mean()
            else:
                l_jepa_margin = residual.new_tensor(0.0)
            l_jepa = l_jepa_normal + l_jepa_margin + feature_online_loss

            x_target = x[node_indices]
            diff_attr = torch.pow(x_target - x_hat, 2)
            attr_error = torch.sqrt(
                torch.sum(diff_attr, dim=1).clamp(min=1e-12))
            l_attr = attr_error.mean()
            l_struct = model_nn._structure_loss(
                model_nn.last_z_struct_center,
                model_nn.last_z_struct_all, a_hat,
                node_indices, edge_index)

            weighted_attr = model_nn.attr_loss_weight * l_attr
            weighted_struct = model_nn.struct_loss_weight * l_struct
            weighted_jepa = model_nn.jepa_loss_weight * l_jepa
            total_loss = model_nn.uncertainty_weighting([
                weighted_attr, weighted_struct, weighted_jepa])
            logs = {
                'loss_attr': float(l_attr.detach().cpu()),
                'loss_struct': float(l_struct.detach().cpu()),
                'loss_jepa': float(l_jepa.detach().cpu()),
                'loss_feature_online': float(
                    feature_online_loss.detach().cpu()),
                'weighted_attr': float(weighted_attr.detach().cpu()),
                'weighted_struct': float(weighted_struct.detach().cpu()),
                'weighted_jepa': float(weighted_jepa.detach().cpu()),
                'margin': float(m_adaptive.detach().cpu()),
            }
            return total_loss, residual, a_hat, x_hat, logs
        else:
            return original_forward(x, edge_index, x_ano, edge_index_ano,
                                    Pi, topk_indices, topk_values,
                                    y_pseudo, node_indices=node_indices)

    model_nn.forward = patched_forward
    return lambda: setattr(model_nn, 'forward', original_forward)


def _patch_cosine_loss(model_nn):
    """H6: Use cosine loss instead of MSE for feature mode."""
    original_forward = model_nn.forward

    def patched_forward(x, edge_index, x_ano, edge_index_ano, Pi,
                        topk_indices, topk_values, y_pseudo,
                        node_indices=None):
        # Call original to get everything
        result = original_forward(x, edge_index, x_ano, edge_index_ano,
                                  Pi, topk_indices, topk_values,
                                  y_pseudo, node_indices=node_indices)
        if not model_nn.training or model_nn.target_mode != 'feature':
            return result

        # Re-compute residual with cosine instead of MSE
        total_loss, _, a_hat, x_hat, logs = result
        if node_indices is None:
            node_indices = torch.arange(x.shape[0], device=x.device)

        z_c_center = model_nn.last_z_center
        with torch.no_grad():
            z_t_center = model_nn.feature_target_encoder(x[node_indices])

        pred = model_nn.predictor(z_c_center)
        pred_norm = F.normalize(pred, dim=-1)
        z_t_norm = F.normalize(z_t_center, dim=-1)
        # Cosine distance instead of MSE
        residual = 1.0 - (pred_norm * z_t_norm).sum(dim=-1)

        z_t_online = model_nn.feature_encoder(x[node_indices])
        z_t_online_norm = F.normalize(z_t_online, dim=-1)
        feature_online_loss = (
            1.0 - (z_t_online_norm * pred_norm.detach()).sum(dim=-1)
        ).mean()

        y_batch = y_pseudo[node_indices]
        normal_mask = y_batch == 0
        anomaly_mask = y_batch == 1
        if torch.any(normal_mask):
            normal_res = residual[normal_mask].detach()
            m_adaptive = normal_res.mean() + 2.0 * normal_res.std(
                unbiased=False)
            l_jepa_normal = residual[normal_mask].mean()
        else:
            m_adaptive = residual.detach().mean()
            l_jepa_normal = residual.mean()
        if torch.any(anomaly_mask):
            l_jepa_margin = F.relu(
                m_adaptive - residual[anomaly_mask]).pow(2).mean()
        else:
            l_jepa_margin = residual.new_tensor(0.0)
        l_jepa = l_jepa_normal + l_jepa_margin + feature_online_loss

        x_target = x[node_indices]
        diff_attr = torch.pow(x_target - x_hat, 2)
        attr_error = torch.sqrt(
            torch.sum(diff_attr, dim=1).clamp(min=1e-12))
        l_attr = attr_error.mean()
        l_struct = model_nn._structure_loss(
            model_nn.last_z_struct_center,
            model_nn.last_z_struct_all, a_hat,
            node_indices, edge_index)

        weighted_attr = model_nn.attr_loss_weight * l_attr
        weighted_struct = model_nn.struct_loss_weight * l_struct
        weighted_jepa = model_nn.jepa_loss_weight * l_jepa
        total_loss = model_nn.uncertainty_weighting([
            weighted_attr, weighted_struct, weighted_jepa])
        logs['loss_jepa'] = float(l_jepa.detach().cpu())
        logs['loss_feature_online'] = float(
            feature_online_loss.detach().cpu())
        logs['weighted_jepa'] = float(weighted_jepa.detach().cpu())
        logs['margin'] = float(m_adaptive.detach().cpu())
        return total_loss, residual, a_hat, x_hat, logs

    model_nn.forward = patched_forward
    return lambda: setattr(model_nn, 'forward', original_forward)


def _patch_no_online_loss(model_nn):
    """H7: Set feature_online_loss = 0."""
    original_forward = model_nn.forward

    def patched_forward(x, edge_index, x_ano, edge_index_ano, Pi,
                        topk_indices, topk_values, y_pseudo,
                        node_indices=None):
        result = original_forward(x, edge_index, x_ano, edge_index_ano,
                                  Pi, topk_indices, topk_values,
                                  y_pseudo, node_indices=node_indices)
        if not model_nn.training or model_nn.target_mode != 'feature':
            return result

        # Re-compute JEPA loss without feature_online_loss
        total_loss, residual_orig, a_hat, x_hat, logs = result
        if node_indices is None:
            node_indices = torch.arange(x.shape[0], device=x.device)

        z_c_center = model_nn.last_z_center
        with torch.no_grad():
            z_t_center = model_nn.feature_target_encoder(x[node_indices])
        pred = model_nn.predictor(z_c_center)
        pred_norm = F.normalize(pred, dim=-1)
        z_t_norm = F.normalize(z_t_center, dim=-1)
        residual = F.mse_loss(pred_norm, z_t_norm,
                              reduction='none').sum(dim=-1)

        y_batch = y_pseudo[node_indices]
        normal_mask = y_batch == 0
        anomaly_mask = y_batch == 1
        if torch.any(normal_mask):
            normal_res = residual[normal_mask].detach()
            m_adaptive = normal_res.mean() + 2.0 * normal_res.std(
                unbiased=False)
            l_jepa_normal = residual[normal_mask].mean()
        else:
            m_adaptive = residual.detach().mean()
            l_jepa_normal = residual.mean()
        if torch.any(anomaly_mask):
            l_jepa_margin = F.relu(
                m_adaptive - residual[anomaly_mask]).pow(2).mean()
        else:
            l_jepa_margin = residual.new_tensor(0.0)
        # NO feature_online_loss
        l_jepa = l_jepa_normal + l_jepa_margin

        x_target = x[node_indices]
        diff_attr = torch.pow(x_target - x_hat, 2)
        attr_error = torch.sqrt(
            torch.sum(diff_attr, dim=1).clamp(min=1e-12))
        l_attr = attr_error.mean()
        l_struct = model_nn._structure_loss(
            model_nn.last_z_struct_center,
            model_nn.last_z_struct_all, a_hat,
            node_indices, edge_index)

        weighted_attr = model_nn.attr_loss_weight * l_attr
        weighted_struct = model_nn.struct_loss_weight * l_struct
        weighted_jepa = model_nn.jepa_loss_weight * l_jepa
        total_loss = model_nn.uncertainty_weighting([
            weighted_attr, weighted_struct, weighted_jepa])
        logs['loss_jepa'] = float(l_jepa.detach().cpu())
        logs['loss_feature_online'] = 0.0
        logs['weighted_jepa'] = float(weighted_jepa.detach().cpu())
        logs['margin'] = float(m_adaptive.detach().cpu())
        return total_loss, residual, a_hat, x_hat, logs

    model_nn.forward = patched_forward
    return lambda: setattr(model_nn, 'forward', original_forward)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    'baseline': {
        'desc': 'Current feature mode (no changes)',
        'patches': [],
    },
    'H4_frozen_target': {
        'desc': 'H4: Disable EMA for feature_target_encoder (frozen random)',
        'patches': [_patch_no_ema],
    },
    'H5_mask_centers': {
        'desc': 'H5: Add center masking to feature mode train',
        'patches': [_patch_mask_centers],
    },
    'H6_cosine_loss': {
        'desc': 'H6: Cosine loss instead of MSE for feature mode',
        'patches': [_patch_cosine_loss],
    },
    'H7_no_online_loss': {
        'desc': 'H7: Remove feature_online_loss regularization',
        'patches': [_patch_no_online_loss],
    },
}


def run_experiment(exp_name, exp_config, data, args, trial):
    """Run one experiment trial and return metrics."""
    seed = args.seed_base + trial * 100
    seed_everything(seed)

    detector = CONADJEPA(
        device=args.device,
        verbose=False,
        epoch=args.epoch,
        batch_size=args.batch_size,
        target_mode='feature',
        grad_clip=5.0,
        refresh_anomaly_every=1,
        attr_loss_weight=1.0,
        struct_loss_weight=1.0,
        jepa_loss_weight=1.0,
        struct_row_all=True,
        seed=seed,
    )

    # We need to call fit in a modified way:
    # 1. Build model via partial fit
    # 2. Apply patches to nn model
    # 3. Continue training

    # --- Build model (will train with patches) ---
    detector.fit(data)  # This will be patched if needed...

    # Actually, we need to patch DURING fit. Let me restructure.
    # The issue is fit() creates model internally. We need to hook into it.
    # Solution: override fit by monkeypatching the model post-creation.

    # Better approach: use a wrapper that patches after model creation
    return None  # placeholder


def run_experiment_v2(exp_name, exp_config, data, args, trial):
    """Run one experiment, applying patches after model init but before train."""
    seed = args.seed_base + trial * 100
    seed_everything(seed)

    detector = CONADJEPA(
        device=args.device,
        verbose=False,
        epoch=args.epoch,
        batch_size=args.batch_size,
        target_mode='feature',
        grad_clip=5.0,
        refresh_anomaly_every=1,
        attr_loss_weight=1.0,
        struct_loss_weight=1.0,
        jepa_loss_weight=1.0,
        struct_row_all=True,
        seed=seed,
    )

    # Monkey-patch fit to inject patches after model creation
    original_fit = detector.fit

    def patched_fit(data_arg, label=None):
        # Store original so we can intercept
        original_fit.__func__(detector, data_arg, label)

    # We can't easily intercept mid-fit. Instead, let's patch the
    # CONADJEPAModel class temporarily before fit.
    from pygod.nn.conadjepa import CONADJEPAModel
    original_init = CONADJEPAModel.__init__

    cleanups = []

    def hooked_init(self_model, *a, **kw):
        original_init(self_model, *a, **kw)
        # Apply patches after __init__
        for patch_fn in exp_config['patches']:
            cleanup = patch_fn(self_model)
            cleanups.append(cleanup)

    CONADJEPAModel.__init__ = hooked_init
    try:
        t0 = time.time()
        detector.fit(data)
        elapsed = time.time() - t0
    finally:
        CONADJEPAModel.__init__ = original_init
        for cleanup in cleanups:
            cleanup()

    score = detector.decision_score_
    label = data.y.bool().long().numpy()
    auc = eval_roc_auc(label, score)
    ap = eval_average_precision(label, score)

    return {'auc': auc, 'ap': ap, 'time': elapsed}


def main():
    parser = argparse.ArgumentParser(description='CONADJEPA Ablation Study')
    parser.add_argument('--dataset', default='inj_cora',
                        choices=['weibo', 'inj_cora', 'inj_amazon', 'reddit'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--num-trials', type=int, default=3)
    parser.add_argument('--seed-base', type=int, default=42)
    parser.add_argument('--experiments', nargs='+',
                        default=list(EXPERIMENTS.keys()),
                        choices=list(EXPERIMENTS.keys()),
                        help='Which experiments to run')
    args = parser.parse_args()

    print(f'Loading {args.dataset}...')
    data = load_data(args.dataset)
    print(f'  Nodes: {data.num_nodes}, Edges: {data.num_edges}, '
          f'Features: {data.x.shape[1]}, '
          f'Anomalies: {data.y.bool().sum().item()}')
    print()

    results = {}

    for exp_name in args.experiments:
        exp_config = EXPERIMENTS[exp_name]
        print(f'=== {exp_name}: {exp_config["desc"]} ===')

        trial_results = []
        for trial in range(args.num_trials):
            print(f'  Trial {trial + 1}/{args.num_trials}...', end=' ',
                  flush=True)
            metrics = run_experiment_v2(
                exp_name, exp_config, data, args, trial)
            trial_results.append(metrics)
            print(f'AUC={metrics["auc"]:.4f}  AP={metrics["ap"]:.4f}  '
                  f'Time={metrics["time"]:.1f}s')

        aucs = [r['auc'] for r in trial_results]
        aps = [r['ap'] for r in trial_results]
        times = [r['time'] for r in trial_results]

        results[exp_name] = {
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'ap_mean': np.mean(aps),
            'ap_std': np.std(aps),
            'time_mean': np.mean(times),
        }
        print(f'  => AUC: {results[exp_name]["auc_mean"]:.4f} '
              f'+/- {results[exp_name]["auc_std"]:.4f}  |  '
              f'AP: {results[exp_name]["ap_mean"]:.4f} '
              f'+/- {results[exp_name]["ap_std"]:.4f}  |  '
              f'Time: {results[exp_name]["time_mean"]:.1f}s')
        print()

    # --- Summary table ---
    print('=' * 80)
    print(f'ABLATION RESULTS — {args.dataset} — {args.epoch} epochs '
          f'— {args.num_trials} trials')
    print('=' * 80)
    print(f'{"Experiment":<25} {"AUC":>15} {"AP":>15} {"Time":>8}')
    print('-' * 80)

    baseline_auc = results.get('baseline', {}).get('auc_mean', 0)
    baseline_ap = results.get('baseline', {}).get('ap_mean', 0)

    for exp_name in args.experiments:
        r = results[exp_name]
        auc_delta = r['auc_mean'] - baseline_auc
        ap_delta = r['ap_mean'] - baseline_ap
        delta_str = ''
        if exp_name != 'baseline':
            delta_str = f'  (Delta AUC={auc_delta:+.4f}, Delta AP={ap_delta:+.4f})'
        print(f'{exp_name:<25} '
              f'{r["auc_mean"]:.4f}+/-{r["auc_std"]:.4f} '
              f'{r["ap_mean"]:.4f}+/-{r["ap_std"]:.4f} '
              f'{r["time_mean"]:>6.1f}s'
              f'{delta_str}')

    print('=' * 80)
    print()

    # --- Recommendations ---
    print('RECOMMENDATIONS:')
    for exp_name in args.experiments:
        if exp_name == 'baseline':
            continue
        r = results[exp_name]
        b = results.get('baseline', r)
        auc_diff = r['auc_mean'] - b['auc_mean']
        ap_diff = r['ap_mean'] - b['ap_mean']

        if auc_diff < -0.01 or ap_diff < -0.01:
            verdict = '[REJECT] significant degradation'
        elif auc_diff < -0.005 or ap_diff < -0.005:
            verdict = '[CAUTION] minor degradation, needs more trials'
        elif auc_diff > 0.005 or ap_diff > 0.005:
            verdict = '[ACCEPT] improvement!'
        else:
            verdict = '[NEUTRAL] no significant difference'

        print(f'  {exp_name}: {verdict}')


if __name__ == '__main__':
    main()
