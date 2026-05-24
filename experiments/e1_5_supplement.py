"""
E1.5 Supplement: rho=1.0 exact limit and finer grid.

Tests:
1. rho=1.0 for all activations (relu, gelu, silu, tanh, identity)
2. Finer rho grid near the transition: 0.0 to 1.0 in smaller steps
3. Multiple m values at rho=1.0 to check width dependence
"""

import sys
import os
import json
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import (
    radial_fraction, make_x_hat_torch, make_correlated_mlp_torch,
    ensure_output_dir
)


def run_rho_1_all_activations():
    """Test rho=1.0 exactly for all activations."""
    print("=" * 70)
    print("E1.5 SUPPLEMENT: rho=1.0 exact limit")
    print("=" * 70)

    m = 512
    n_seeds = 50
    B = 4096
    activations = ['identity', 'relu', 'gelu', 'silu', 'tanh']

    results = {}
    for act in activations:
        R_values = []
        for seed in range(n_seeds):
            F_fn, W1, W2 = make_correlated_mlp_torch(
                m, rho=1.0, activation=act, seed=seed
            )
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
            with torch.no_grad():
                F_out = F_fn(x_hat)
            R_values.append(radial_fraction(x_hat, F_out))
        R_arr = np.array(R_values)
        results[act] = {
            'rho': 1.0,
            'mean': float(R_arr.mean()),
            'std': float(R_arr.std()),
            'ci_95': [float(np.percentile(R_arr, 2.5)),
                      float(np.percentile(R_arr, 97.5))],
            'all_values': R_arr.tolist(),
        }
        print(f"  {act:10s}: R_F = {R_arr.mean():.6f} +/- {R_arr.std():.6f}")

    # Reference values
    print(f"\n  Reference values:")
    print(f"    1/m    = {1/m:.6f}")
    print(f"    1/pi   = {1/math.pi:.6f}")
    print(f"    2/pi   = {2/math.pi:.6f}")
    print(f"    1/2    = 0.500000")
    return results


def run_finer_grid():
    """Run ReLU with a finer rho grid."""
    print("\n" + "=" * 70)
    print("E1.5 SUPPLEMENT: Finer rho grid (ReLU)")
    print("=" * 70)

    m = 512
    n_seeds = 50
    B = 4096
    # Complete grid including values between existing points
    rhos = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
            0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
            0.9, 0.92, 0.95, 0.97, 0.99, 0.995, 1.0]

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
            'all_values': R_arr.tolist(),
        }
        print(f"  rho={rho:.3f}: R_F = {R_arr.mean():.6f} +/- {R_arr.std():.6f}")

    return results


def run_width_at_rho1():
    """Check if R_F(rho=1) depends on m."""
    print("\n" + "=" * 70)
    print("E1.5 SUPPLEMENT: Width dependence at rho=1.0 (ReLU)")
    print("=" * 70)

    widths = [64, 128, 256, 512, 1024, 2048]
    n_seeds = 30
    B = 4096

    results = {}
    for m in widths:
        R_values = []
        for seed in range(n_seeds):
            F_fn, _, _ = make_correlated_mlp_torch(
                m, rho=1.0, activation='relu', seed=seed
            )
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
            with torch.no_grad():
                F_out = F_fn(x_hat)
            R_values.append(radial_fraction(x_hat, F_out))
        R_arr = np.array(R_values)
        results[str(m)] = {
            'width': m,
            'mean': float(R_arr.mean()),
            'std': float(R_arr.std()),
        }
        print(f"  m={m:5d}: R_F(rho=1) = {R_arr.mean():.6f} +/- {R_arr.std():.6f}")

    return results


def run_all_activations_finer_grid():
    """Run all activations at key rho values for cross-comparison."""
    print("\n" + "=" * 70)
    print("E1.5 SUPPLEMENT: All activations at key rho values")
    print("=" * 70)

    m = 512
    n_seeds = 50
    B = 4096
    activations = ['relu', 'gelu', 'silu', 'tanh', 'identity']
    rhos = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

    results = {}
    for act in activations:
        act_results = {}
        for rho in rhos:
            R_values = []
            for seed in range(n_seeds):
                F_fn, _, _ = make_correlated_mlp_torch(
                    m, rho=rho, activation=act, seed=seed
                )
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 10000)
                with torch.no_grad():
                    F_out = F_fn(x_hat)
                R_values.append(radial_fraction(x_hat, F_out))
            R_arr = np.array(R_values)
            act_results[str(rho)] = {
                'rho': rho,
                'mean': float(R_arr.mean()),
                'std': float(R_arr.std()),
            }
        results[act] = act_results
        # Print summary for this activation
        vals = [(r, act_results[str(r)]['mean']) for r in rhos]
        print(f"  {act:10s}: " + "  ".join(f"rho={r}:{v:.4f}" for r, v in [(0.0, vals[0][1]), (0.5, vals[4][1]), (1.0, vals[-1][1])]))

    return results


if __name__ == '__main__':
    all_results = {}

    # Part 1: rho=1.0 for all activations
    all_results['rho1_activations'] = run_rho_1_all_activations()

    # Part 2: Finer grid for ReLU
    all_results['finer_grid_relu'] = run_finer_grid()

    # Part 3: Width dependence at rho=1.0
    all_results['width_at_rho1'] = run_width_at_rho1()

    # Part 4: All activations at key rho values
    all_results['all_activations_grid'] = run_all_activations_finer_grid()

    # Save
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'e1_5_supplement.json')
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nAll results saved to {filepath}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Activation':>10} {'rho=0':>10} {'rho=0.5':>10} {'rho=0.99':>10} {'rho=1.0':>10}")
    print("-" * 55)
    for act in ['identity', 'relu', 'gelu', 'silu', 'tanh']:
        r0 = all_results['all_activations_grid'][act]['0.0']['mean']
        r5 = all_results['all_activations_grid'][act]['0.5']['mean']
        r99 = all_results['all_activations_grid'][act]['0.99']['mean']
        r1 = all_results['all_activations_grid'][act]['1.0']['mean']
        print(f"{act:>10} {r0:>10.6f} {r5:>10.6f} {r99:>10.6f} {r1:>10.6f}")
    print(f"\n  1/m = {1/512:.6f}, 1/pi = {1/math.pi:.6f}, 2/pi = {2/math.pi:.6f}, 1/2 = 0.500000")
