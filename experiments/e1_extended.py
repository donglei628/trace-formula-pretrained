"""
E1-Extended: Mapping the Radial Fraction Landscape.

Six experiments to characterize R_F across configurations.
Run individual experiments or all at once.

Usage:
    python experiments/e1_extended.py sanity      # Run sanity checks
    python experiments/e1_extended.py e1_1        # Width scan
    python experiments/e1_extended.py e1_2        # Activation scan
    python experiments/e1_extended.py e1_3        # Per-layer stratification
    python experiments/e1_extended.py e1_4        # Input distribution
    python experiments/e1_extended.py e1_5        # Weight correlation (THE experiment)
    python experiments/e1_extended.py e1_6        # Attention vs MLP
    python experiments/e1_extended.py all         # Run everything
"""

import sys
import os
import json
import math
import time
from datetime import datetime

import numpy as np
import torch

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))
from e1_common import (
    radial_fraction, make_x_hat_torch, make_mlp_torch,
    make_correlated_mlp_torch, run_sanity_checks, ensure_output_dir
)


def save_result(name, data):
    """Save experiment results to JSON."""
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, f'{name}.json')
    # Convert numpy/torch types
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    serializable = json.loads(json.dumps(data, default=convert))
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved to {filepath}")


# ============================================================
# E1.1: Width Scan
# ============================================================

def e1_1_width_scan():
    """Verify R_F = 1/m across widths m in {64, 128, 256, 512, 1024, 2048}."""
    print("\n" + "=" * 70)
    print("E1.1: WIDTH SCAN -- Verify R_F = 1/m across two decades")
    print("=" * 70)

    widths = [64, 128, 256, 512, 1024, 2048]
    n_seeds = 50
    B = 4096

    results = {}
    for m in widths:
        R_values = []
        for seed in range(n_seeds):
            F_fn, _, _ = make_mlp_torch(m, activation='relu', seed=seed)
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
            with torch.no_grad():
                F_out = F_fn(x_hat)
            R_values.append(radial_fraction(x_hat, F_out))
        R_arr = np.array(R_values)
        results[str(m)] = {
            'mean': float(R_arr.mean()),
            'std': float(R_arr.std()),
            'predicted_1_over_m': 1.0 / m,
            'product_m_times_RF': float(R_arr.mean() * m),
            'all_values': R_arr.tolist(),
        }
        print(f"  m={m:5d}: R_F = {R_arr.mean():.5e} +/- {R_arr.std():.2e}, "
              f"1/m = {1/m:.5e}, m*R_F = {R_arr.mean()*m:.4f}")

    # Log-log regression for slope
    log_m = np.log(widths)
    log_RF = np.array([np.log(results[str(m)]['mean']) for m in widths])
    slope, intercept = np.polyfit(log_m, log_RF, 1)
    results['regression'] = {'slope': float(slope), 'intercept': float(intercept)}
    print(f"\n  Log-log slope: {slope:.4f} (expected: -1.00)")
    products = [f'{results[str(m)]["product_m_times_RF"]:.4f}' for m in widths]
    print(f"  m*R_F products: {products}")

    save_result('e1_1_width_scan', results)
    return results


# ============================================================
# E1.2: Activation Scan
# ============================================================

def e1_2_activation_scan():
    """Confirm R_F = 1/m is independent of activation function."""
    print("\n" + "=" * 70)
    print("E1.2: ACTIVATION SCAN -- Confirm universality across activations")
    print("=" * 70)

    m = 512
    n_seeds = 50
    B = 4096
    activations = ['identity', 'relu', 'gelu', 'silu', 'tanh']

    results = {}
    for act in activations:
        R_values = []
        for seed in range(n_seeds):
            F_fn, _, _ = make_mlp_torch(m, activation=act, seed=seed)
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
            with torch.no_grad():
                F_out = F_fn(x_hat)
            R_values.append(radial_fraction(x_hat, F_out))
        R_arr = np.array(R_values)
        results[act] = {
            'mean': float(R_arr.mean()),
            'std': float(R_arr.std()),
            'product_m_RF': float(R_arr.mean() * m),
            'all_values': R_arr.tolist(),
        }
        print(f"  {act:10s}: R_F = {R_arr.mean():.5e} +/- {R_arr.std():.2e}, "
              f"m*R_F = {R_arr.mean()*m:.4f}")

    # Max relative deviation
    means = [results[a]['mean'] for a in activations]
    max_dev = (max(means) - min(means)) / np.mean(means)
    results['max_relative_deviation'] = float(max_dev)
    print(f"\n  Max relative deviation across activations: {max_dev:.3%}")
    print(f"  1/m = {1/m:.6f}")

    save_result('e1_2_activation_scan', results)
    return results


# ============================================================
# E1.3: Per-Layer Depth Stratification
# ============================================================

def e1_3_per_layer():
    """Check if R_F stays at 1/m across stacked Pre-Norm layers."""
    print("\n" + "=" * 70)
    print("E1.3: PER-LAYER DEPTH -- R_F stability across 32 layers")
    print("=" * 70)

    m = 512
    L = 32
    n_seeds = 20
    B = 1024
    sigma_F_sq = 0.04  # moderate update scale

    R_per_layer = np.zeros((n_seeds, L))
    rho_per_layer = np.zeros((n_seeds, L + 1))
    norm_per_layer = np.zeros((n_seeds, L + 1))

    for seed in range(n_seeds):
        # Initialize on sphere
        torch.manual_seed(seed + 5000)
        x = torch.randn(B, m)
        x = x * math.sqrt(m) / x.norm(dim=-1, keepdim=True)
        x_0_hat = x.clone()
        rho_per_layer[seed, 0] = 1.0
        norm_per_layer[seed, 0] = (x ** 2).sum(dim=-1).mean().item()

        for l in range(L):
            # RMSNorm
            x_hat = x * math.sqrt(m) / x.norm(dim=-1, keepdim=True)

            # Generate MLP with standard init
            F_fn, _, _ = make_mlp_torch(m, activation='relu', seed=seed * L + l + 20000)
            with torch.no_grad():
                F_out = F_fn(x_hat)

            # Rescale to target sigma_F^2
            actual_energy = (F_out ** 2).sum(dim=-1).mean().item()
            target_energy = sigma_F_sq * m
            if actual_energy > 0:
                F_out = F_out * math.sqrt(target_energy / actual_energy)

            # Measure R_F
            R_per_layer[seed, l] = radial_fraction(x_hat, F_out)

            # Residual update
            x = x + F_out

            # Measure angular correlation and norm
            x_hat_new = x * math.sqrt(m) / x.norm(dim=-1, keepdim=True)
            rho_l = (x_hat_new * x_0_hat).sum(dim=-1).mean().item() / m
            rho_per_layer[seed, l + 1] = rho_l
            norm_per_layer[seed, l + 1] = (x ** 2).sum(dim=-1).mean().item()

    mean_R = R_per_layer.mean(axis=0)
    std_R = R_per_layer.std(axis=0)
    mean_rho = rho_per_layer.mean(axis=0)
    mean_norm = norm_per_layer.mean(axis=0)

    # Theory predictions
    theory_rho = np.array([(1 + l * sigma_F_sq) ** (-0.5) for l in range(L + 1)])
    theory_norm = np.array([m * (1 + l * sigma_F_sq) for l in range(L + 1)])
    L_star = (math.e ** 2 - 1) / sigma_F_sq

    print(f"  sigma_F^2 = {sigma_F_sq}, L* = {L_star:.1f}")
    print(f"\n  {'Layer':>6} {'R_F':>10} {'R_F_std':>10} {'m*R_F':>8} "
          f"{'rho_emp':>10} {'rho_th':>10} {'||x||2_emp':>12} {'||x||2_th':>12}")
    print(f"  {'-'*85}")
    for l in [0, 4, 8, 16, 24, 31]:
        print(f"  {l:>6} {mean_R[l]:>10.6f} {std_R[l]:>10.6f} {mean_R[l]*m:>8.4f} "
              f"{mean_rho[l+1]:>10.4f} {theory_rho[l+1]:>10.4f} "
              f"{mean_norm[l+1]:>12.1f} {theory_norm[l+1]:>12.1f}")

    # Check for drift
    early_R = mean_R[:8].mean()
    late_R = mean_R[24:].mean()
    drift = (late_R - early_R) / early_R
    print(f"\n  R_F drift (last 8 vs first 8 layers): {drift:+.2%}")
    print(f"  {'STABLE' if abs(drift) < 0.20 else 'DRIFT DETECTED'}")

    results = {
        'R_per_layer_mean': mean_R.tolist(),
        'R_per_layer_std': std_R.tolist(),
        'rho_per_layer_mean': mean_rho.tolist(),
        'norm_per_layer_mean': mean_norm.tolist(),
        'theory_rho': theory_rho.tolist(),
        'theory_norm': theory_norm.tolist(),
        'L_star': L_star,
        'sigma_F_sq': sigma_F_sq,
        'm': m, 'L': L,
        'drift_fraction': float(drift),
    }
    save_result('e1_3_per_layer', results)
    return results


# ============================================================
# E1.4: Input Distribution Sensitivity
# ============================================================

def e1_4_input_distribution():
    """Test R_F robustness under different input distributions."""
    print("\n" + "=" * 70)
    print("E1.4: INPUT DISTRIBUTION -- Delocalization sensitivity")
    print("=" * 70)

    m = 512
    n_seeds = 50
    B = 4096
    distributions = ['gaussian', 'sparse', 'heavy_tail', 'spike']

    results = {}
    for dist in distributions:
        R_values = []
        for seed in range(n_seeds):
            F_fn, _, _ = make_mlp_torch(m, activation='relu', seed=seed)
            x_hat = make_x_hat_torch(B, m, dist, seed=seed + 10000)
            with torch.no_grad():
                F_out = F_fn(x_hat)
            R_values.append(radial_fraction(x_hat, F_out))
        R_arr = np.array(R_values)
        results[dist] = {
            'mean': float(R_arr.mean()),
            'std': float(R_arr.std()),
            'product_m_RF': float(R_arr.mean() * m),
            'all_values': R_arr.tolist(),
        }
        print(f"  {dist:12s}: R_F = {R_arr.mean():.5e} +/- {R_arr.std():.2e}, "
              f"m*R_F = {R_arr.mean()*m:.4f}")

    # Compute kurtosis of each distribution for analysis
    print(f"\n  Reference: 1/m = {1/m:.6f}, m = {m}")
    print(f"\n  Deviation from Gaussian:")
    gauss_mean = results['gaussian']['mean']
    for dist in distributions:
        dev = (results[dist]['mean'] - gauss_mean) / gauss_mean
        print(f"    {dist:12s}: {dev:+.2%}")

    save_result('e1_4_input_distribution', results)
    return results


# ============================================================
# E1.5: Weight Correlation (THE BRIDGE EXPERIMENT)
# ============================================================

def e1_5_weight_correlation():
    """
    THE MOST IMPORTANT EXPERIMENT.
    Measure R_F as a function of weight correlation rho between W1 and W2.
    """
    print("\n" + "=" * 70)
    print("E1.5: WEIGHT CORRELATION -- The Bridge Experiment")
    print("=" * 70)

    m = 512
    n_seeds = 50
    B = 4096
    rhos = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    results = {}
    for rho in rhos:
        R_values = []
        for seed in range(n_seeds):
            F_fn, _, _ = make_correlated_mlp_torch(
                m, rho=rho, activation='relu', seed=seed
            )
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
            with torch.no_grad():
                F_out = F_fn(x_hat)
            R_values.append(radial_fraction(x_hat, F_out))
        R_arr = np.array(R_values)
        results[str(rho)] = {
            'rho': rho,
            'mean': float(R_arr.mean()),
            'std': float(R_arr.std()),
            'ci_95': [float(np.percentile(R_arr, 2.5)),
                      float(np.percentile(R_arr, 97.5))],
            'all_values': R_arr.tolist(),
        }
        print(f"  rho={rho:.2f}: R_F = {R_arr.mean():.6f} +/- {R_arr.std():.4f}")

    # Key reference points
    print(f"\n  Reference values:")
    print(f"    1/m      = {1/m:.6f}")
    print(f"    1/pi     = {1/math.pi:.6f}")
    print(f"    2/pi     = {2/math.pi:.6f}")
    print(f"    1/2      = 0.500000")

    # Also test with other activations at a few key rho values
    print(f"\n  Cross-check with other activations at key rho values:")
    for act in ['gelu', 'silu', 'tanh']:
        for rho_check in [0.0, 0.5, 0.99]:
            R_values = []
            for seed in range(20):
                F_fn, _, _ = make_correlated_mlp_torch(
                    m, rho=rho_check, activation=act, seed=seed
                )
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
                with torch.no_grad():
                    F_out = F_fn(x_hat)
                R_values.append(radial_fraction(x_hat, F_out))
            R_arr = np.array(R_values)
            key = f"{act}_rho{rho_check}"
            results[key] = {
                'activation': act, 'rho': rho_check,
                'mean': float(R_arr.mean()), 'std': float(R_arr.std()),
            }
            print(f"    {act:5s} rho={rho_check:.2f}: R_F = {R_arr.mean():.6f}")

    # Compute theoretical L* for each rho
    print(f"\n  Implied critical depth L* (sigma_F^2=0.05):")
    sigma_test = 0.05
    for rho in rhos:
        R = results[str(rho)]['mean']
        if R < 1.0:
            L_star = (math.exp(2 / (1 - R)) - 1) / sigma_test
        else:
            L_star = float('inf')
        print(f"    rho={rho:.2f}: R_F={R:.4f}, L*={L_star:.1f}")

    save_result('e1_5_weight_correlation', results)
    return results


# ============================================================
# E1.6: Attention vs MLP
# ============================================================

def e1_6_attention_vs_mlp():
    """Compare R_F between MLP, single-head attention, and multi-head attention."""
    print("\n" + "=" * 70)
    print("E1.6: ATTENTION vs MLP -- Component-wise R_F decomposition")
    print("=" * 70)

    m = 512
    n_seeds = 30
    seq_len = 128  # sequence length (number of tokens)
    B = seq_len  # each token is a sample

    results = {}

    # --- MLP baseline ---
    print("\n  MLP baseline:")
    R_mlp = []
    for seed in range(n_seeds):
        F_fn, _, _ = make_mlp_torch(m, activation='relu', seed=seed)
        x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
        with torch.no_grad():
            F_out = F_fn(x_hat)
        R_mlp.append(radial_fraction(x_hat, F_out))
    R_arr = np.array(R_mlp)
    results['mlp'] = {'mean': float(R_arr.mean()), 'std': float(R_arr.std())}
    print(f"    R_F = {R_arr.mean():.6f} +/- {R_arr.std():.4f} (m*R_F = {R_arr.mean()*m:.4f})")

    # --- Single-head attention ---
    print("\n  Single-head attention:")
    R_attn1 = []
    for seed in range(n_seeds):
        torch.manual_seed(seed + 30000)
        W_Q = torch.randn(m, m) / math.sqrt(m)
        W_K = torch.randn(m, m) / math.sqrt(m)
        W_V = torch.randn(m, m) / math.sqrt(m)
        W_O = torch.randn(m, m) / math.sqrt(m)

        x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
        with torch.no_grad():
            Q = x_hat @ W_Q.T  # (B, m)
            K = x_hat @ W_K.T  # (B, m)
            V = x_hat @ W_V.T  # (B, m)
            scores = (Q @ K.T) / math.sqrt(m)  # (B, B)
            attn = torch.softmax(scores, dim=-1)
            attn_out = attn @ V  # (B, m)
            F_out = attn_out @ W_O.T  # (B, m) through output projection

        R_attn1.append(radial_fraction(x_hat, F_out))
    R_arr = np.array(R_attn1)
    results['attn_1head'] = {'mean': float(R_arr.mean()), 'std': float(R_arr.std())}
    print(f"    R_F = {R_arr.mean():.6f} +/- {R_arr.std():.4f} (m*R_F = {R_arr.mean()*m:.4f})")

    # --- Multi-head attention ---
    for h in [4, 8, 16]:
        d_head = m // h
        print(f"\n  Multi-head attention (h={h}, d_head={d_head}):")
        R_mh = []
        for seed in range(n_seeds):
            torch.manual_seed(seed + 40000 + h * 1000)
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)

            all_heads = []
            with torch.no_grad():
                for head_idx in range(h):
                    W_Q = torch.randn(d_head, m) / math.sqrt(m)
                    W_K = torch.randn(d_head, m) / math.sqrt(m)
                    W_V = torch.randn(d_head, m) / math.sqrt(m)

                    Q = x_hat @ W_Q.T  # (B, d_head)
                    K = x_hat @ W_K.T
                    V = x_hat @ W_V.T
                    scores = (Q @ K.T) / math.sqrt(d_head)
                    attn = torch.softmax(scores, dim=-1)
                    all_heads.append(attn @ V)  # (B, d_head)

                concat = torch.cat(all_heads, dim=-1)  # (B, m)
                W_O = torch.randn(m, m) / math.sqrt(m)
                F_out = concat @ W_O.T

            R_mh.append(radial_fraction(x_hat, F_out))
        R_arr = np.array(R_mh)
        results[f'attn_{h}head'] = {'mean': float(R_arr.mean()), 'std': float(R_arr.std())}
        print(f"    R_F = {R_arr.mean():.6f} +/- {R_arr.std():.4f} (m*R_F = {R_arr.mean()*m:.4f})")

    # --- Attention WITHOUT output projection (to isolate softmax effect) ---
    print(f"\n  Single-head attention WITHOUT W_O:")
    R_no_wo = []
    for seed in range(n_seeds):
        torch.manual_seed(seed + 50000)
        W_Q = torch.randn(m, m) / math.sqrt(m)
        W_K = torch.randn(m, m) / math.sqrt(m)
        W_V = torch.randn(m, m) / math.sqrt(m)

        x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
        with torch.no_grad():
            Q = x_hat @ W_Q.T
            K = x_hat @ W_K.T
            V = x_hat @ W_V.T
            scores = (Q @ K.T) / math.sqrt(m)
            attn = torch.softmax(scores, dim=-1)
            F_out = attn @ V  # no W_O

        R_no_wo.append(radial_fraction(x_hat, F_out))
    R_arr = np.array(R_no_wo)
    results['attn_no_wo'] = {'mean': float(R_arr.mean()), 'std': float(R_arr.std())}
    print(f"    R_F = {R_arr.mean():.6f} +/- {R_arr.std():.4f} (m*R_F = {R_arr.mean()*m:.4f})")

    # Summary
    print(f"\n  {'Component':>25} {'R_F':>10} {'m*R_F':>8}")
    print(f"  {'-'*48}")
    for name, vals in results.items():
        print(f"  {name:>25} {vals['mean']:>10.6f} {vals['mean']*m:>8.4f}")
    print(f"  {'1/m (theory)':>25} {1/m:>10.6f} {1.0:>8.4f}")

    save_result('e1_6_attention_vs_mlp', results)
    return results


# ============================================================
# Main
# ============================================================

EXPERIMENTS = {
    'sanity': run_sanity_checks,
    'e1_1': e1_1_width_scan,
    'e1_2': e1_2_activation_scan,
    'e1_3': e1_3_per_layer,
    'e1_4': e1_4_input_distribution,
    'e1_5': e1_5_weight_correlation,
    'e1_6': e1_6_attention_vs_mlp,
}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python e1_extended.py <experiment>")
        print(f"  Available: {', '.join(EXPERIMENTS.keys())}, all")
        sys.exit(1)

    target = sys.argv[1].lower()

    print("=" * 70)
    print(f"E1-EXTENDED: Radial Fraction Landscape")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Target: {target}")
    print("=" * 70)

    if target == 'all':
        # Run sanity first, then all experiments in order
        if not run_sanity_checks():
            print("Sanity checks failed. Aborting.")
            sys.exit(1)
        for name in ['e1_1', 'e1_2', 'e1_3', 'e1_4', 'e1_5', 'e1_6']:
            t0 = time.time()
            EXPERIMENTS[name]()
            dt = time.time() - t0
            print(f"\n  [{name} completed in {dt:.1f}s]")
    elif target in EXPERIMENTS:
        t0 = time.time()
        EXPERIMENTS[target]()
        dt = time.time() - t0
        print(f"\n  [Completed in {dt:.1f}s]")
    else:
        print(f"Unknown experiment: {target}")
        print(f"Available: {', '.join(EXPERIMENTS.keys())}, all")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
