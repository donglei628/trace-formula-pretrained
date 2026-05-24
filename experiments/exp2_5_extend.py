"""
Extend exp2_5 to TinyLlama-1.1B and Pythia-1B.

Generates REAL per-layer R_F_formula and R_F_linear_empirical data
using the same methodology as the original exp2_5 (Monte Carlo with
10 seeds x 2048 samples per layer).

Output: results/exp2_5_tinyllama_pythia.json
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as torchF
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import radial_fraction, make_x_hat_torch, ensure_output_dir


def compute_trace_formula(M, m):
    """Compute the exact trace formula for R_F(linear) with (m+2) denominator."""
    tr_M = float(torch.trace(M).item())
    M_fro_sq = float((M ** 2).sum().item())
    tr_M2 = float(torch.trace(M @ M).item())
    denom = (m + 2) * M_fro_sq
    R_F = (tr_M ** 2 + M_fro_sq + tr_M2) / denom
    return {
        'R_F_formula': R_F,
        'tr_M': tr_M,
        'M_fro_sq': M_fro_sq,
        'tr_M2': tr_M2,
        'term_trace': tr_M ** 2 / denom,
        'term_baseline': M_fro_sq / denom,
        'term_quadratic': tr_M2 / denom,
    }


def run_trace_verification(model_key, model_name, model_type, B=2048, n_seeds=10):
    """
    Run trace formula verification for a single model.
    Returns dict with per-layer R_F_formula and R_F_linear_empirical.
    """
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"  Loading: {model_key} ({model_name})")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float32, device_map='cpu',
    )
    model.eval()

    config = model.config
    m = config.hidden_size
    num_layers = config.num_hidden_layers

    print(f"  hidden_size={m}, num_layers={num_layers}")
    print(f"  B={B}, n_seeds={n_seeds}")

    # Get decoder layers
    if model_type == 'swiglu':
        decoder_layers = model.model.layers
    elif model_type == 'standard_mlp':
        decoder_layers = model.gpt_neox.layers
    else:
        raise ValueError(f"Unknown type: {model_type}")

    print(f"\n  {'Layer':>5} {'R_F formula':>12} {'R_F lin emp':>12} {'ratio':>8} "
          f"{'R_F full':>10} {'alpha':>8} {'pct_trace':>10}")
    print(f"  {'-'*75}")

    layer_results = []

    for layer_idx in range(num_layers):
        layer = decoder_layers[layer_idx]
        mlp = layer.mlp

        # Extract W1 (up) and W2 (down) weight matrices
        if model_type == 'swiglu':
            W1 = mlp.up_proj.weight.detach().float()
            W2 = mlp.down_proj.weight.detach().float()
        elif model_type == 'standard_mlp':
            W1 = mlp.dense_h_to_4h.weight.detach().float()
            W2 = mlp.dense_4h_to_h.weight.detach().float()

        # Compute M = W2 @ W1
        M = W2 @ W1
        tf = compute_trace_formula(M, m)

        # Empirical R_F (linear): Monte Carlo measurement
        RF_lin_values = []
        for seed in range(n_seeds):
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
            with torch.no_grad():
                F_out = (M @ x_hat.T).T
            RF_lin_values.append(radial_fraction(x_hat, F_out))
        RF_lin_emp = float(np.mean(RF_lin_values))

        # Empirical R_F (full nonlinear): Monte Carlo measurement
        RF_full_values = []
        for seed in range(n_seeds):
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
            with torch.no_grad():
                if model_type == 'swiglu':
                    W_gate = mlp.gate_proj.weight.detach().float()
                    F_out = (W2 @ (torchF.silu(W_gate @ x_hat.T) * (W1 @ x_hat.T))).T
                elif model_type == 'standard_mlp':
                    F_out = (W2 @ torchF.gelu(W1 @ x_hat.T)).T
            RF_full_values.append(radial_fraction(x_hat, F_out))
        RF_full_emp = float(np.mean(RF_full_values))

        ratio = RF_lin_emp / tf['R_F_formula'] if tf['R_F_formula'] > 1e-10 else float('inf')
        alpha = RF_full_emp / tf['R_F_formula'] if tf['R_F_formula'] > 1e-10 else float('inf')
        pct_trace = 100 * tf['term_trace'] / tf['R_F_formula'] if tf['R_F_formula'] > 1e-10 else 0

        print(f"  {layer_idx:>5} {tf['R_F_formula']:>12.6f} {RF_lin_emp:>12.6f} {ratio:>8.4f} "
              f"{RF_full_emp:>10.6f} {alpha:>8.4f} {pct_trace:>9.1f}%")

        layer_results.append({
            'layer': layer_idx,
            'R_F_formula': tf['R_F_formula'],
            'R_F_linear_empirical': RF_lin_emp,
            'R_F_full_empirical': RF_full_emp,
            'ratio_linear': ratio,
            'alpha_suppression': alpha,
            'tr_M': tf['tr_M'],
            'M_fro_sq': tf['M_fro_sq'],
            'tr_M2': tf['tr_M2'],
            'pct_trace': pct_trace,
        })

    # Summary statistics
    ratios = [r['ratio_linear'] for r in layer_results]
    alphas = [r['alpha_suppression'] for r in layer_results]
    rf_formula = [r['R_F_formula'] for r in layer_results]
    rf_full = [r['R_F_full_empirical'] for r in layer_results]

    print(f"\n  Formula accuracy: mean ratio = {np.mean(ratios):.6f}, std = {np.std(ratios):.6f}")
    print(f"  Activation suppression alpha: mean = {np.mean(alphas):.4f}, std = {np.std(alphas):.4f}")

    if len(rf_formula) > 3:
        pr, pp = stats.pearsonr(rf_formula, rf_full)
        sr, sp = stats.spearmanr(rf_formula, rf_full)
        print(f"\n  Correlation R_F(formula) vs R_F(full):")
        print(f"    Pearson r = {pr:.4f} (p = {pp:.2e})")
        print(f"    Spearman rho = {sr:.4f} (p = {sp:.2e})")
    else:
        pr, sr = None, None

    result = {
        'model': model_name,
        'type': model_type,
        'hidden_size': m,
        'num_layers': num_layers,
        'layers': layer_results,
        'summary': {
            'mean_ratio_linear': float(np.mean(ratios)),
            'std_ratio_linear': float(np.std(ratios)),
            'mean_alpha': float(np.mean(alphas)),
            'std_alpha': float(np.std(alphas)),
            'pearson_r': float(pr) if pr is not None else None,
            'spearman_rho': float(sr) if sr is not None else None,
        }
    }

    del model
    import gc; gc.collect()

    return result


def main():
    print("=" * 70)
    print("EXP 2.5 EXTENSION: TRACE FORMULA ON TinyLlama + Pythia-1B")
    print("=" * 70)

    all_results = {}

    # TinyLlama-1.1B (SwiGLU, m=2048, 22 layers)
    all_results['tinyllama'] = run_trace_verification(
        'tinyllama',
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'swiglu',
    )

    # Pythia-1B (Standard MLP + GELU, m=2048, 16 layers)
    all_results['pythia-1b'] = run_trace_verification(
        'pythia-1b',
        'EleutherAI/pythia-1b',
        'standard_mlp',
    )

    # Save results
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'exp2_5_tinyllama_pythia.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {filepath}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, res in all_results.items():
        s = res['summary']
        print(f"  {key}: mean_ratio={s['mean_ratio_linear']:.6f}, "
              f"std={s['std_ratio_linear']:.6f}, "
              f"layers={res['num_layers']}")


if __name__ == '__main__':
    main()
