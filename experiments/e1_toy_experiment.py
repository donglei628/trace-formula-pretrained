"""
E1 Toy Experiment: Verify critical depth scaling L* for Pre-Norm Transformers.

This experiment simulates a simplified Pre-Norm Transformer with i.i.d. Gaussian
weights to verify the angular correlation decay formula:

    rho_L = (1 + L*sigma_F^2)^{-(1-R_F)/2}

And the critical depth formula:

    L* = (e^{2/(1-R_F)} - 1) / sigma_F^2

For i.i.d. Gaussian W2 (independent of W1, xhat), we expect R_F ~= 1/m ~= 0,
so the simplified prediction is:

    rho_L ~= (1 + L*sigma_F^2)^{-1/2}
    L* ~= (e^2 - 1) / sigma_F^2 ~= 6.39 / sigma_F^2

The experiment measures:
1. Angular correlation rho_L = <xhat_L, xhat_0> / m at each layer
2. Residual norm growth ||x_L||^2
3. Empirical R_F (radial fraction) at each layer
4. Comparison with theoretical predictions

Usage:
    python experiments/e1_toy_experiment.py
"""

import numpy as np
import math
import os
import json
from datetime import datetime

# ============================================================
# Pre-Norm Block Simulation
# ============================================================

def rmsnorm(x):
    """RMSNorm: project onto sphere of radius sqrt(m)."""
    m = x.shape[-1]
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True))
    return x / rms  # This gives ||output|| = sqrt(m)


def simulate_prenorm_mlp(m, n_layers, sigma_F, activation='relu', seed=0):
    """
    Simulate a Pre-Norm Transformer with MLP blocks:
        x_{l+1} = x_l + F_l(RMSNorm(x_l))
    where F_l(xhat) = W2 * phi(W1 * xhat) with fresh i.i.d. Gaussian weights.

    Args:
        m: model width (residual dimension)
        n_layers: number of layers to simulate
        sigma_F: scale of update (per-coordinate std dev, so ||F||^2 ~= sigma_F^2 * m)
        activation: 'relu', 'gelu', 'silu', 'identity'
        seed: random seed

    Returns:
        dict with angular correlations, norms, radial fractions, etc.
    """
    rng = np.random.default_rng(seed)

    # Activation function
    if activation == 'relu':
        phi = lambda z: np.maximum(z, 0)
    elif activation == 'gelu':
        from scipy.stats import norm as normal_dist
        phi = lambda z: z * normal_dist.cdf(z)
    elif activation == 'silu':
        phi = lambda z: z / (1.0 + np.exp(-z))
    elif activation == 'identity':
        phi = lambda z: z
    elif activation == 'tanh':
        phi = lambda z: np.tanh(z)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Initialize x_0 on the sphere (||x_0|| = sqrt(m))
    x = rng.standard_normal(m)
    x = x / np.linalg.norm(x) * np.sqrt(m)

    x0_hat = x / np.linalg.norm(x) * np.sqrt(m)  # = x itself at init

    # Storage
    angular_corrs = [1.0]  # rho_0 = 1
    norms_sq = [np.sum(x**2)]  # ||x_0||^2
    radial_fractions = []
    step_sizes = []  # geodesic step sizes

    # Hidden dim for MLP (typically 4*m in real transformers, we use m for simplicity)
    d_hidden = m

    for l in range(n_layers):
        # RMSNorm
        x_hat = rmsnorm(x)

        # MLP: F(xhat) = W2 * phi(W1 * xhat)
        # W1: d_hidden x m, entries ~ N(0, 1/m)
        # W2: m x d_hidden, entries ~ N(0, 1/d_hidden)
        # This gives ||F||^2 ~= m * E[phi(z)^2] (per standard init)
        # We scale to achieve desired sigma_F^2

        W1 = rng.standard_normal((d_hidden, m)) / np.sqrt(m)
        h = phi(W1 @ x_hat)  # hidden activations

        W2 = rng.standard_normal((m, d_hidden)) / np.sqrt(d_hidden)
        F = W2 @ h  # update vector

        # Scale F to achieve desired sigma_F^2
        # We want E[||F||^2] = sigma_F^2 * m
        # Current scale: ||F||^2 ~= E[phi(z)^2] * m (for standard init)
        # So we scale by sigma_F / sqrt(E[phi(z)^2])
        actual_F_norm_sq = np.sum(F**2)
        # Instead of trying to match exactly, just scale to desired sigma_F
        # The "natural" scale gives sigma_F^2 ~= E[phi(z)^2] ~= 0.5 for ReLU
        # We want a specific sigma_F^2, so rescale
        target_F_norm_sq = sigma_F**2 * m
        scale = np.sqrt(target_F_norm_sq / actual_F_norm_sq) if actual_F_norm_sq > 0 else 0
        F = F * scale

        # Measure radial fraction before update
        radial_component = np.dot(x_hat, F) / np.sqrt(m)  # xhat^T F / sqrt(m)
        radial_energy = radial_component**2 / m  # (xhat^T F)^2 / m
        total_energy = np.sum(F**2)
        R_F = radial_energy / total_energy * m if total_energy > 0 else 0
        radial_fractions.append(R_F)

        # Pre-norm update
        x_new = x + F

        # Measure angular step
        x_hat_new = rmsnorm(x_new)
        cos_step = np.dot(x_hat, x_hat_new) / m
        cos_step = np.clip(cos_step, -1, 1)
        step_sizes.append(np.arccos(cos_step))

        # Measure angular correlation with x_0
        cos_corr = np.dot(x_hat_new, x0_hat) / m
        angular_corrs.append(cos_corr)

        # Measure norm
        norms_sq.append(np.sum(x_new**2))

        x = x_new

    return {
        'angular_corrs': np.array(angular_corrs),
        'norms_sq': np.array(norms_sq),
        'radial_fractions': np.array(radial_fractions),
        'step_sizes': np.array(step_sizes),
    }


# ============================================================
# Theory predictions
# ============================================================

def theory_angular_corr(L, sigma_F_sq, R_F=0.0):
    """
    Theoretical angular correlation at depth L.
    rho_L = (1 + L*sigma_F^2)^{-(1-R_F)/2}
    """
    return (1 + L * sigma_F_sq) ** (-(1 - R_F) / 2)


def theory_norm_sq(L, m, sigma_F_sq):
    """
    Theoretical residual norm squared at depth L.
    E[||x_L||^2] = m + L*sigma_F^2*m = m(1 + L*sigma_F^2)
    """
    return m * (1 + L * sigma_F_sq)


def theory_critical_depth(sigma_F_sq, R_F=0.0):
    """
    Critical depth L* where rho_{L*} = 1/e.
    L* = (e^{2/(1-R_F)} - 1) / sigma_F^2
    """
    return (math.exp(2.0 / (1 - R_F)) - 1) / sigma_F_sq


# ============================================================
# Main experiment
# ============================================================

def run_experiment(m_values, sigma_F_values, n_seeds=20, activation='relu',
                   max_layers=500, verbose=True):
    """
    Run the full E1 experiment across parameter grid.
    """
    results = {}

    for m in m_values:
        for sigma_F_sq in sigma_F_values:
            sigma_F = np.sqrt(sigma_F_sq)

            # Determine number of layers to simulate
            # Go to at least 3x the predicted L* or max_layers
            L_star_pred = theory_critical_depth(sigma_F_sq, R_F=0.0)
            n_layers = min(int(3 * L_star_pred) + 10, max_layers)
            # Ensure at least a reasonable number
            n_layers = max(n_layers, 50)

            if verbose:
                print(f"\n{'='*60}")
                print(f"m={m}, sigma_F^2={sigma_F_sq:.4f}, L*_theory={L_star_pred:.1f}, simulating {n_layers} layers")
                print(f"{'='*60}")

            all_corrs = []
            all_norms = []
            all_RFs = []

            for seed in range(n_seeds):
                res = simulate_prenorm_mlp(
                    m=m, n_layers=n_layers, sigma_F=sigma_F,
                    activation=activation, seed=seed
                )
                all_corrs.append(res['angular_corrs'])
                all_norms.append(res['norms_sq'])
                all_RFs.append(res['radial_fractions'])

            # Average across seeds
            all_corrs = np.array(all_corrs)  # (n_seeds, n_layers+1)
            all_norms = np.array(all_norms)
            all_RFs = np.array(all_RFs)

            mean_corrs = np.mean(all_corrs, axis=0)
            std_corrs = np.std(all_corrs, axis=0)
            mean_norms = np.mean(all_norms, axis=0)
            mean_RFs = np.mean(all_RFs, axis=0)
            mean_RF = np.mean(all_RFs)

            # Find empirical L* (where mean_corrs first drops below 1/e)
            L_star_empirical = None
            for l in range(len(mean_corrs)):
                if mean_corrs[l] < 1/math.e:
                    L_star_empirical = l
                    break

            # Theory predictions at each layer
            layers = np.arange(n_layers + 1)
            theory_corrs_RF0 = np.array([theory_angular_corr(l, sigma_F_sq, R_F=0.0) for l in layers])
            theory_corrs_RFmean = np.array([theory_angular_corr(l, sigma_F_sq, R_F=mean_RF) for l in layers])
            theory_norms = np.array([theory_norm_sq(l, m, sigma_F_sq) for l in layers])

            # Compute fit quality (RMSE on log scale for correlations)
            valid = mean_corrs > 0.01  # avoid log(0)
            if np.any(valid):
                log_empirical = np.log(mean_corrs[valid])
                log_theory = np.log(theory_corrs_RF0[:len(mean_corrs)][valid])
                rmse_log = np.sqrt(np.mean((log_empirical - log_theory)**2))
            else:
                rmse_log = float('inf')

            key = f"m{m}_s{sigma_F_sq}"
            results[key] = {
                'm': m,
                'sigma_F_sq': sigma_F_sq,
                'n_layers': n_layers,
                'n_seeds': n_seeds,
                'activation': activation,
                'L_star_theory_RF0': theory_critical_depth(sigma_F_sq, 0.0),
                'L_star_theory_RFmean': theory_critical_depth(sigma_F_sq, mean_RF),
                'L_star_empirical': L_star_empirical,
                'mean_RF': float(mean_RF),
                'rmse_log': float(rmse_log),
                'mean_corrs': mean_corrs.tolist(),
                'std_corrs': std_corrs.tolist(),
                'mean_norms': mean_norms.tolist(),
                'theory_corrs_RF0': theory_corrs_RF0.tolist(),
                'theory_norms': theory_norms.tolist(),
            }

            if verbose:
                print(f"  Mean empirical R_F = {mean_RF:.6f}")
                print(f"  L* (theory, R_F=0) = {results[key]['L_star_theory_RF0']:.1f}")
                print(f"  L* (theory, R_F={mean_RF:.4f}) = {results[key]['L_star_theory_RFmean']:.1f}")
                print(f"  L* (empirical)     = {L_star_empirical}")
                print(f"  RMSE(log rho)        = {rmse_log:.4f}")

                # Print correlation at a few key layers
                check_layers = [0, 5, 10, 20, 50, 100, int(L_star_pred)]
                check_layers = [l for l in check_layers if l < n_layers + 1]
                print(f"\n  {'Layer':>6} {'rho_emp':>10} {'rho_th(RF=0)':>12} {'||x||2_emp':>12} {'||x||2_th':>12}")
                print(f"  {'-'*55}")
                for l in check_layers:
                    print(f"  {l:>6} {mean_corrs[l]:>10.4f} {theory_corrs_RF0[l]:>12.4f} {mean_norms[l]:>12.1f} {theory_norms[l]:>12.1f}")

    return results


# ============================================================
# Summary and Visualization
# ============================================================

def print_summary(results):
    """Print a summary table of all results."""
    print("\n" + "=" * 90)
    print("E1 EXPERIMENT SUMMARY")
    print("=" * 90)
    print(f"{'m':>6} {'sigma_F^2':>8} {'L*_th(RF=0)':>12} {'L*_empirical':>13} {'ratio':>8} {'mean_RF':>8} {'RMSE(logrho)':>11}")
    print("-" * 90)

    for key, r in sorted(results.items()):
        ratio = r['L_star_empirical'] / r['L_star_theory_RF0'] if r['L_star_empirical'] else float('nan')
        print(f"{r['m']:>6} {r['sigma_F_sq']:>8.4f} {r['L_star_theory_RF0']:>12.1f} "
              f"{str(r['L_star_empirical']):>13} {ratio:>8.3f} {r['mean_RF']:>8.5f} {r['rmse_log']:>11.4f}")

    # Analysis
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    # Check scaling with sigma_F^2
    print("\n1. sigma_F^2 scaling (fixed m): L* should scale as 1/sigma_F^2")
    m_groups = {}
    for key, r in results.items():
        m = r['m']
        if m not in m_groups:
            m_groups[m] = []
        m_groups[m].append(r)

    for m, group in sorted(m_groups.items()):
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r['sigma_F_sq'])
        print(f"\n  m = {m}:")
        for i in range(len(group) - 1):
            r1, r2 = group[i], group[i+1]
            if r1['L_star_empirical'] and r2['L_star_empirical']:
                sigma_ratio = r1['sigma_F_sq'] / r2['sigma_F_sq']
                L_ratio = r2['L_star_empirical'] / r1['L_star_empirical']
                # If L* ~ 1/sigma_F^2, then L2/L1 = sigma1^2/sigma2^2
                expected_ratio = sigma_ratio
                print(f"    sigma_F^2 ratio: {r2['sigma_F_sq']:.4f}/{r1['sigma_F_sq']:.4f} = {1/sigma_ratio:.2f}")
                print(f"    L* ratio:   {r2['L_star_empirical']}/{r1['L_star_empirical']} = {L_ratio:.3f}")
                print(f"    Expected:   {expected_ratio:.3f}  (diff: {abs(L_ratio - expected_ratio)/expected_ratio*100:.1f}%)")

    # Check scaling with m
    print("\n2. m scaling (fixed sigma_F^2): L* should NOT depend on m (in our simplified formula)")
    sigma_groups = {}
    for key, r in results.items():
        s = r['sigma_F_sq']
        if s not in sigma_groups:
            sigma_groups[s] = []
        sigma_groups[s].append(r)

    for s, group in sorted(sigma_groups.items()):
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r['m'])
        print(f"\n  sigma_F^2 = {s}:")
        L_stars = [r['L_star_empirical'] for r in group if r['L_star_empirical']]
        ms = [r['m'] for r in group if r['L_star_empirical']]
        if len(L_stars) >= 2:
            for r in group:
                if r['L_star_empirical']:
                    print(f"    m = {r['m']:>5}: L* = {r['L_star_empirical']}")
            # Check if L* is roughly constant across m
            L_mean = np.mean(L_stars)
            L_std = np.std(L_stars)
            print(f"    Mean L* = {L_mean:.1f}, Std = {L_std:.1f}, CV = {L_std/L_mean*100:.1f}%")

    # Check R_F values
    print("\n3. Empirical R_F values:")
    for key, r in sorted(results.items()):
        print(f"  m={r['m']:>5}, sigma_F^2={r['sigma_F_sq']:.4f}: R_F = {r['mean_RF']:.6f} (~= 1/m = {1/r['m']:.6f})")


def save_results(results, output_dir):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'e1_results.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {}
    for key, val in results.items():
        serializable[key] = {k: convert(v) for k, v in val.items()}

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {filepath}")


# ============================================================
# Entry point
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("E1 TOY EXPERIMENT: Critical Depth Scaling Verification")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()
    print("Theory prediction (R_F ~ 0 for i.i.d. Gaussian W2):")
    print("  rho_L = (1 + L*sigma_F^2)^{-1/2}")
    print(f"  L* = (e^2 - 1) / sigma_F^2 = {(math.e**2 - 1):.4f} / sigma_F^2")
    print()

    # Experiment parameters
    # Use a range of m to check m-independence of L*
    # Use a range of sigma_F^2 to check 1/sigma_F^2 scaling
    m_values = [128, 256, 512, 1024]
    sigma_F_sq_values = [0.01, 0.02, 0.05, 0.1]
    n_seeds = 20

    print(f"Parameters:")
    print(f"  m values: {m_values}")
    print(f"  sigma_F^2 values: {sigma_F_sq_values}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Activation: ReLU")

    results = run_experiment(
        m_values=m_values,
        sigma_F_values=sigma_F_sq_values,
        n_seeds=n_seeds,
        activation='relu',
        max_layers=500,
        verbose=True,
    )

    print_summary(results)

    # Save results
    save_results(results, os.path.join(os.path.dirname(__file__), 'results'))

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
