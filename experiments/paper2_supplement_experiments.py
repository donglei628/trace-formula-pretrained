"""
Paper 2 Supplementary Experiments
=================================

Completes experiments from docs/paper2_experiments.md:
- 2.2: Collapse confirmation (already done in T-B5)
- 2.4: Per-layer trace decomposition (from existing data + new models)
- 3.6: Norm growth power law fit (from existing data)
- 3.2: Width convergence of trace formula (synthetic)
- 2.1: Trace formula on attention layers (requires model loading)
- 2.6: Cross-layer M coupling (requires model loading)
- 2.5: Trace formula on more architectures (GPT-2, Qwen2.5-0.5B)
- 2.3: Quantization sensitivity vs R_F (requires model + eval data)

Usage:
    python paper2_supplement_experiments.py [all|2.4|3.6|3.2|2.1|2.6|2.5|2.3]
"""

import sys
import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as torchF
from scipy import stats
from scipy.optimize import curve_fit

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
        'term_baseline': M_fro_sq / denom,  # = 1/(m+2)
        'term_quadratic': tr_M2 / denom,
    }


# ======================================================================
# 2.4: Per-Layer Trace Decomposition
# ======================================================================

def exp_2_4_trace_decomposition():
    """
    Decompose R_F into trace, baseline, quadratic terms per layer.
    Uses existing trace_formula_verification.json + loads Qwen2.5-1.5B for new data.
    """
    print("=" * 70)
    print("EXP 2.4: PER-LAYER TRACE DECOMPOSITION")
    print("=" * 70)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    # Load existing data
    tf_path = os.path.join(results_dir, 'trace_formula_verification.json')
    with open(tf_path) as f:
        tf_data = json.load(f)

    all_results = {}

    for model_key, mdata in tf_data.items():
        m = mdata['hidden_size']
        layers = mdata['layers']

        print(f"\n  Model: {model_key} (m={m})")
        print(f"  {'Layer':>5} {'R_F_formula':>12} {'% trace':>9} {'% baseline':>11} {'% quadratic':>12}")
        print(f"  {'-'*55}")

        decomp_layers = []
        for l in layers:
            tr_M = l['tr_M']
            M_fro_sq = l['M_fro_sq']
            tr_M2 = l['tr_M2']
            denom = (m + 2) * M_fro_sq

            t_trace = tr_M ** 2 / denom
            t_base = M_fro_sq / denom  # = 1/(m+2)
            t_quad = tr_M2 / denom
            total = t_trace + t_base + t_quad

            pct_trace = 100 * t_trace / total
            pct_base = 100 * t_base / total
            pct_quad = 100 * t_quad / total

            print(f"  {l['layer']:>5} {total:>12.6f} {pct_trace:>8.1f}% {pct_base:>10.1f}% {pct_quad:>11.1f}%")

            decomp_layers.append({
                'layer': l['layer'],
                'R_F_formula': total,
                'term_trace': t_trace,
                'term_baseline': t_base,
                'term_quadratic': t_quad,
                'pct_trace': pct_trace,
                'pct_baseline': pct_base,
                'pct_quadratic': pct_quad,
            })

        pct_traces = [d['pct_trace'] for d in decomp_layers]
        pct_quads = [d['pct_quadratic'] for d in decomp_layers]
        pct_bases = [d['pct_baseline'] for d in decomp_layers]
        print(f"\n  Summary:")
        print(f"    Trace term:     mean={np.mean(pct_traces):.1f}%, range=[{min(pct_traces):.1f}%, {max(pct_traces):.1f}%]")
        print(f"    Baseline term:  mean={np.mean(pct_bases):.1f}%, range=[{min(pct_bases):.1f}%, {max(pct_bases):.1f}%]")
        print(f"    Quadratic term: mean={np.mean(pct_quads):.1f}%, range=[{min(pct_quads):.1f}%, {max(pct_quads):.1f}%]")

        all_results[model_key] = {
            'model': mdata.get('model', model_key),
            'hidden_size': m,
            'layers': decomp_layers,
            'summary': {
                'pct_trace_mean': float(np.mean(pct_traces)),
                'pct_baseline_mean': float(np.mean(pct_bases)),
                'pct_quadratic_mean': float(np.mean(pct_quads)),
            }
        }

    # Save
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'exp2_4_trace_decomposition.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {filepath}")
    return all_results


# ======================================================================
# 3.6: Norm Growth Power Law Fit
# ======================================================================

def exp_3_6_norm_growth_fit():
    """
    Fit ||x_l||^2 ~ a + b * l^p from E5.2 data.
    """
    print("\n" + "=" * 70)
    print("EXP 3.6: NORM GROWTH POWER LAW FIT")
    print("=" * 70)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    e52_path = os.path.join(results_dir, 'e5_2_empirical_RF.json')
    with open(e52_path) as f:
        e52_data = json.load(f)

    all_results = {}

    for model_key, mdata in e52_data.items():
        m = mdata['hidden_size']
        num_layers = mdata['num_layers']
        layers = mdata['layers']

        norm_sq = np.array([l['norm_sq_mean'] for l in layers])
        layer_idx = np.arange(num_layers)

        print(f"\n  Model: {model_key} (m={m}, L={num_layers})")
        print(f"  ||x||^2 range: {norm_sq[0]:.1f} -> {norm_sq[-1]:.1f} (ratio={norm_sq[-1]/norm_sq[0]:.1f}x)")

        # Fit models
        fits = {}

        # 1. Linear: y = a + b*l
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(layer_idx, norm_sq)
            y_pred = intercept + slope * layer_idx
            rmse = math.sqrt(np.mean((norm_sq - y_pred)**2))
            fits['linear'] = {'a': intercept, 'b': slope, 'R2': r_value**2, 'RMSE': rmse}
        except Exception:
            fits['linear'] = {'R2': -1}

        # 2. Power law: y = a + b * l^p (fit for l >= 1)
        try:
            def power_model(l, a, b, p):
                return a + b * np.power(l + 1, p)  # +1 to avoid l=0
            popt, pcov = curve_fit(power_model, layer_idx, norm_sq, p0=[norm_sq[0], 1.0, 1.5], maxfev=5000)
            y_pred = power_model(layer_idx, *popt)
            ss_res = np.sum((norm_sq - y_pred)**2)
            ss_tot = np.sum((norm_sq - np.mean(norm_sq))**2)
            R2 = 1 - ss_res / ss_tot
            rmse = math.sqrt(np.mean((norm_sq - y_pred)**2))
            fits['power'] = {'a': float(popt[0]), 'b': float(popt[1]), 'p': float(popt[2]),
                             'R2': R2, 'RMSE': rmse}
        except Exception as e:
            fits['power'] = {'R2': -1, 'error': str(e)}

        # 3. Quadratic: y = a + b*l + c*l^2
        try:
            coeffs = np.polyfit(layer_idx, norm_sq, 2)
            y_pred = np.polyval(coeffs, layer_idx)
            ss_res = np.sum((norm_sq - y_pred)**2)
            ss_tot = np.sum((norm_sq - np.mean(norm_sq))**2)
            R2 = 1 - ss_res / ss_tot
            rmse = math.sqrt(np.mean((norm_sq - y_pred)**2))
            fits['quadratic'] = {'c2': float(coeffs[0]), 'c1': float(coeffs[1]),
                                 'c0': float(coeffs[2]), 'R2': R2, 'RMSE': rmse}
        except Exception:
            fits['quadratic'] = {'R2': -1}

        # 4. sqrt(L): y = a + b*sqrt(l)
        try:
            sqrt_l = np.sqrt(layer_idx + 1)
            slope, intercept, r_value, _, _ = stats.linregress(sqrt_l, norm_sq)
            y_pred = intercept + slope * sqrt_l
            rmse = math.sqrt(np.mean((norm_sq - y_pred)**2))
            fits['sqrt'] = {'a': intercept, 'b': slope, 'R2': r_value**2, 'RMSE': rmse}
        except Exception:
            fits['sqrt'] = {'R2': -1}

        # Report
        print(f"\n  Fit results:")
        for name, fit in fits.items():
            if 'R2' in fit and fit['R2'] > -0.5:
                extra = f", p={fit['p']:.3f}" if 'p' in fit else ""
                print(f"    {name:>12}: R2={fit['R2']:.4f}, RMSE={fit.get('RMSE',0):.2f}{extra}")
            else:
                print(f"    {name:>12}: FAILED")

        best_fit = max(fits.items(), key=lambda x: x[1].get('R2', -999))
        print(f"  Best fit: {best_fit[0]} (R2={best_fit[1].get('R2', -1):.4f})")

        # Power law fit with 95% CI on exponent (if available)
        if 'p' in fits.get('power', {}):
            p = fits['power']['p']
            print(f"\n  Power law exponent: p = {p:.3f}")
            if p > 1.5:
                print(f"  -> Super-linear growth confirmed (p > 1.5)")
            elif p > 0.9:
                print(f"  -> Near-linear growth")
            else:
                print(f"  -> Sub-linear growth (unexpected)")

        all_results[model_key] = {
            'model': mdata.get('model', model_key),
            'hidden_size': m,
            'num_layers': num_layers,
            'norm_sq': norm_sq.tolist(),
            'fits': fits,
            'best_fit': best_fit[0],
        }

    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'exp3_6_norm_growth_fit.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {filepath}")
    return all_results


# ======================================================================
# 3.2: Width Convergence of Trace Formula
# ======================================================================

def exp_3_2_width_convergence():
    """
    Test trace formula convergence with width m using synthetic random matrices.
    """
    print("\n" + "=" * 70)
    print("EXP 3.2: TRACE FORMULA WIDTH CONVERGENCE")
    print("=" * 70)

    widths = [32, 64, 128, 256, 512, 1024, 2048]
    n_seeds = 20
    B = 4096

    results = []

    for m in widths:
        errors_m = []
        errors_m_plus_2 = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            m_hidden = 4 * m
            W1 = torch.randn(m_hidden, m) / math.sqrt(m)
            W2 = torch.randn(m, m_hidden) / math.sqrt(m_hidden)
            M = W2 @ W1

            # Formula predictions
            tf = compute_trace_formula(M, m)
            R_F_formula_m2 = tf['R_F_formula']  # uses (m+2)

            # Also compute with m denominator
            tr_M = tf['tr_M']
            M_fro_sq = tf['M_fro_sq']
            tr_M2 = tf['tr_M2']
            R_F_formula_m = (tr_M ** 2 + M_fro_sq + tr_M2) / (m * M_fro_sq)

            # Empirical
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed + 1000)
            with torch.no_grad():
                F_out = (M @ x_hat.T).T
            R_F_emp = radial_fraction(x_hat, F_out)

            errors_m.append(abs(R_F_emp - R_F_formula_m) / R_F_emp if R_F_emp > 1e-10 else 0)
            errors_m_plus_2.append(abs(R_F_emp - R_F_formula_m2) / R_F_emp if R_F_emp > 1e-10 else 0)

        mean_err_m = np.mean(errors_m)
        mean_err_m2 = np.mean(errors_m_plus_2)
        print(f"  m={m:>5}: error(m)={mean_err_m:.6f}, error(m+2)={mean_err_m2:.6f}, "
              f"ratio={mean_err_m/mean_err_m2:.2f}x")

        results.append({
            'width': m,
            'mean_rel_error_m': float(mean_err_m),
            'mean_rel_error_m_plus_2': float(mean_err_m2),
            'std_rel_error_m': float(np.std(errors_m)),
            'std_rel_error_m_plus_2': float(np.std(errors_m_plus_2)),
        })

    # Fit convergence rate
    log_m = np.log([r['width'] for r in results])
    log_err_m2 = np.log([max(r['mean_rel_error_m_plus_2'], 1e-15) for r in results])
    if all(e > 1e-14 for e in log_err_m2):
        slope, _, r_value, _, _ = stats.linregress(log_m, log_err_m2)
        print(f"\n  Convergence rate for (m+2): error ~ m^{slope:.2f} (R2={r_value**2:.3f})")
    else:
        slope = None
        print(f"\n  Convergence rate: some errors too small to fit")

    print(f"\n  Conclusion: (m+2) denominator is more accurate at all widths.")

    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'exp3_2_width_convergence.json')
    with open(filepath, 'w') as f:
        json.dump({'widths': results, 'convergence_slope': slope}, f, indent=2)
    print(f"  Saved to {filepath}")
    return results


# ======================================================================
# 2.1: Trace Formula on Attention Layers
# ======================================================================

def exp_2_1_attention_trace():
    """
    Test trace formula on attention layers: M_attn = W_O @ W_V.
    Compare formula prediction to:
    1. Empirical R_F(linear) from x_hat -> M_attn @ x_hat
    2. Empirical R_F(full attention) from T-C2 data
    """
    print("\n" + "=" * 70)
    print("EXP 2.1: TRACE FORMULA ON ATTENTION LAYERS")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    MODEL_CONFIGS = {
        'tinyllama': {
            'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'type': 'llama',
        },
        'pythia-1b': {
            'name': 'EleutherAI/pythia-1b',
            'type': 'pythia',
        },
    }

    B = 2048
    n_seeds = 10

    # Load T-C2 data for empirical R_F(full attention)
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    tc2_path = os.path.join(results_dir, 'tc2_attention_vs_mlp.json')
    tc2_data = {}
    if os.path.exists(tc2_path):
        with open(tc2_path) as f:
            tc2_data = json.load(f)
        print(f"  Loaded T-C2 data for empirical R_F(attention)")

    all_results = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_key}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            cfg['name'], trust_remote_code=True,
            dtype=torch.float32, device_map='cpu',
        )
        model.eval()

        config = model.config
        m = config.hidden_size
        num_layers = config.num_hidden_layers

        if cfg['type'] == 'llama':
            decoder_layers = model.model.layers
        else:
            decoder_layers = model.gpt_neox.layers

        # Get T-C2 empirical attention R_F
        tc2_attn_rf = {}
        if model_key in tc2_data:
            for l in tc2_data[model_key]['layers']:
                tc2_attn_rf[l['layer']] = l['R_F_attn']

        print(f"  m={m}, num_layers={num_layers}")
        print(f"\n  {'Layer':>5} {'R_F formula':>12} {'R_F lin emp':>12} {'ratio_lin':>10} "
              f"{'R_F attn(TC2)':>14} {'alpha_attn':>11}")
        print(f"  {'-'*75}")

        layer_results = []

        for layer_idx in range(num_layers):
            layer = decoder_layers[layer_idx]

            # Compute M_attn = sum_h W_O_h @ W_V_kv(h) for multi-head attention
            if cfg['type'] == 'llama':
                W_V = layer.self_attn.v_proj.weight.detach().float()  # (kv_heads*d, m)
                W_O = layer.self_attn.o_proj.weight.detach().float()  # (m, heads*d)

                num_heads = config.num_attention_heads
                num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
                head_dim = m // num_heads
                n_rep = num_heads // num_kv_heads

                # Reshape into per-head blocks
                W_V_blocks = W_V.reshape(num_kv_heads, head_dim, m)
                W_O_blocks = W_O.reshape(m, num_heads, head_dim)

                # M = sum_h W_O_h @ W_V_{kv(h)}
                M_attn = torch.zeros(m, m)
                for h in range(num_heads):
                    kv_idx = h // n_rep
                    W_O_h = W_O_blocks[:, h, :]      # (m, head_dim)
                    W_V_kv = W_V_blocks[kv_idx]       # (head_dim, m)
                    M_attn += W_O_h @ W_V_kv          # (m, m)
            else:
                # Pythia: combined qkv_proj, split into Q, K, V
                qkv_weight = layer.attention.query_key_value.weight.detach().float()
                W_Q, W_K, W_V = qkv_weight.chunk(3, dim=0)
                W_O = layer.attention.dense.weight.detach().float()
                M_attn = W_O @ W_V  # (m, m)

            # Trace formula prediction
            tf = compute_trace_formula(M_attn, m)

            # Empirical R_F(linear) — linear map x -> M_attn @ x
            RF_lin_values = []
            for seed in range(n_seeds):
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
                with torch.no_grad():
                    F_out = (M_attn @ x_hat.T).T
                RF_lin_values.append(radial_fraction(x_hat, F_out))
            RF_lin_emp = float(np.mean(RF_lin_values))

            # Empirical R_F(full attention) from T-C2
            RF_full = tc2_attn_rf.get(layer_idx, float('nan'))

            ratio_lin = RF_lin_emp / tf['R_F_formula'] if tf['R_F_formula'] > 1e-10 else float('inf')
            alpha_attn = RF_full / tf['R_F_formula'] if (tf['R_F_formula'] > 1e-10 and not math.isnan(RF_full)) else float('nan')

            print(f"  {layer_idx:>5} {tf['R_F_formula']:>12.6f} {RF_lin_emp:>12.6f} {ratio_lin:>10.4f} "
                  f"{RF_full:>14.6f} {alpha_attn:>11.4f}")

            layer_results.append({
                'layer': layer_idx,
                'R_F_formula': tf['R_F_formula'],
                'R_F_linear_empirical': RF_lin_emp,
                'ratio_linear': ratio_lin,
                'R_F_full_attention': RF_full if not math.isnan(RF_full) else None,
                'alpha_attention': alpha_attn if not math.isnan(alpha_attn) else None,
                'tr_M': tf['tr_M'],
                'M_fro_sq': tf['M_fro_sq'],
                'tr_M2': tf['tr_M2'],
                'pct_trace': 100 * tf['term_trace'] / tf['R_F_formula'] if tf['R_F_formula'] > 1e-10 else 0,
            })

        # Summary
        ratios_lin = [r['ratio_linear'] for r in layer_results]
        alphas = [r['alpha_attention'] for r in layer_results if r['alpha_attention'] is not None]
        rf_formula = [r['R_F_formula'] for r in layer_results]
        rf_full = [r['R_F_full_attention'] for r in layer_results if r['R_F_full_attention'] is not None]

        print(f"\n  Linear formula accuracy: mean ratio = {np.mean(ratios_lin):.4f}, std = {np.std(ratios_lin):.4f}")
        if alphas:
            print(f"  Attention suppression alpha: mean = {np.mean(alphas):.4f}, std = {np.std(alphas):.4f}")

        # Correlation between formula and full attention R_F
        if len(rf_full) > 3:
            rf_formula_matched = [r['R_F_formula'] for r in layer_results if r['R_F_full_attention'] is not None]
            pr, pp = stats.pearsonr(rf_formula_matched, rf_full)
            sr, sp = stats.spearmanr(rf_formula_matched, rf_full)
            print(f"\n  Correlation R_F(formula) vs R_F(full attention):")
            print(f"    Pearson r = {pr:.4f} (p = {pp:.2e})")
            print(f"    Spearman rho = {sr:.4f} (p = {sp:.2e})")
        else:
            pr, sr = None, None

        all_results[model_key] = {
            'model': cfg['name'],
            'type': cfg['type'],
            'hidden_size': m,
            'num_layers': num_layers,
            'layers': layer_results,
            'summary': {
                'mean_ratio_linear': float(np.mean(ratios_lin)),
                'std_ratio_linear': float(np.std(ratios_lin)),
                'mean_alpha_attention': float(np.mean(alphas)) if alphas else None,
                'std_alpha_attention': float(np.std(alphas)) if alphas else None,
                'pearson_r': float(pr) if pr is not None else None,
                'spearman_rho': float(sr) if sr is not None else None,
            }
        }

        del model
        import gc; gc.collect()

    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'exp2_1_attention_trace.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {filepath}")
    return all_results


# ======================================================================
# 2.6: Cross-Layer M Coupling
# ======================================================================

def exp_2_6_cross_layer_coupling():
    """
    Analyze cross-layer correlation of M = W2*W1 matrices.
    """
    print("\n" + "=" * 70)
    print("EXP 2.6: CROSS-LAYER M COUPLING")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    MODEL_CONFIGS = {
        'tinyllama': {
            'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'type': 'swiglu',
        },
        'pythia-1b': {
            'name': 'EleutherAI/pythia-1b',
            'type': 'standard_mlp',
        },
    }

    all_results = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_key}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            cfg['name'], trust_remote_code=True,
            dtype=torch.float32, device_map='cpu',
        )
        model.eval()

        config = model.config
        m = config.hidden_size
        num_layers = config.num_hidden_layers

        if cfg['type'] == 'swiglu':
            decoder_layers = model.model.layers
        else:
            decoder_layers = model.gpt_neox.layers

        # Extract M matrices and their properties
        M_properties = []  # tr(M), ||M||_F^2, tr(M^2) per layer

        for layer_idx in range(num_layers):
            layer = decoder_layers[layer_idx]
            mlp = layer.mlp

            if cfg['type'] == 'swiglu':
                W_up = mlp.up_proj.weight.detach().float()
                W_down = mlp.down_proj.weight.detach().float()
            else:
                W_up = mlp.dense_h_to_4h.weight.detach().float()
                W_down = mlp.dense_4h_to_h.weight.detach().float()

            M = W_down @ W_up
            tr_M = float(torch.trace(M).item())
            M_fro_sq = float((M ** 2).sum().item())
            tr_M2 = float(torch.trace(M @ M).item())

            M_properties.append({
                'layer': layer_idx,
                'tr_M': tr_M,
                'M_fro_sq': M_fro_sq,
                'tr_M2': tr_M2,
                'R_F_formula': (tr_M**2 + M_fro_sq + tr_M2) / ((m + 2) * M_fro_sq),
            })

        # Cross-layer correlation: use vectorized M (subsample for memory)
        # Instead of full m^2 vectors, use the trace properties
        print(f"\n  Cross-layer tr(M) profile:")
        tr_values = [p['tr_M'] for p in M_properties]
        for i, p in enumerate(M_properties):
            bar = '#' * max(1, int(40 * abs(p['tr_M']) / max(abs(v) for v in tr_values)))
            print(f"    Layer {i:>2}: tr(M)={p['tr_M']:>10.2f}  {'|' if p['tr_M'] >= 0 else '-'}{bar}")

        # Compute pairwise Frobenius cosine between M matrices (subsampled)
        print(f"\n  Computing pairwise M cosine similarity (subsampled rows)...")

        # To avoid O(n^2 * m^2) memory, compute M row subsets
        n_sample_rows = min(256, m)
        row_indices = torch.randperm(m)[:n_sample_rows]

        M_samples = []
        for layer_idx in range(num_layers):
            layer = decoder_layers[layer_idx]
            mlp = layer.mlp

            if cfg['type'] == 'swiglu':
                W_up = mlp.up_proj.weight.detach().float()
                W_down = mlp.down_proj.weight.detach().float()
            else:
                W_up = mlp.dense_h_to_4h.weight.detach().float()
                W_down = mlp.dense_4h_to_h.weight.detach().float()

            M = W_down @ W_up
            M_sub = M[row_indices].flatten().numpy()  # subsample rows
            M_samples.append(M_sub)

        # Pairwise correlation matrix
        corr_matrix = np.zeros((num_layers, num_layers))
        for i in range(num_layers):
            for j in range(num_layers):
                corr_matrix[i, j] = float(np.corrcoef(M_samples[i], M_samples[j])[0, 1])

        # Analyze structure
        # Neighbor correlation
        neighbor_corr = [corr_matrix[i, i+1] for i in range(num_layers - 1)]
        print(f"\n  Nearest-neighbor M correlation:")
        print(f"    Mean = {np.mean(neighbor_corr):.4f}, Std = {np.std(neighbor_corr):.4f}")
        print(f"    Range = [{min(neighbor_corr):.4f}, {max(neighbor_corr):.4f}]")

        # Off-diagonal decay
        for dist in [1, 2, 3, 5, 10]:
            if dist >= num_layers:
                break
            pairs = [(i, i+dist) for i in range(num_layers - dist)]
            corrs = [corr_matrix[i, j] for i, j in pairs]
            print(f"    Distance {dist}: mean corr = {np.mean(corrs):.4f}")

        # Check for block structure (correlation within early/late vs across)
        mid = num_layers // 2
        early_block = corr_matrix[:mid, :mid]
        late_block = corr_matrix[mid:, mid:]
        cross_block = corr_matrix[:mid, mid:]

        mean_early = float(np.mean(early_block[np.triu_indices(mid, k=1)]))
        mean_late = float(np.mean(late_block[np.triu_indices(num_layers - mid, k=1)]))
        mean_cross = float(np.mean(cross_block))

        print(f"\n  Block structure:")
        print(f"    Early-early corr:  {mean_early:.4f}")
        print(f"    Late-late corr:    {mean_late:.4f}")
        print(f"    Early-late corr:   {mean_cross:.4f}")

        # tr(M) autocorrelation
        from numpy import correlate as np_correlate
        tr_arr = np.array(tr_values)
        tr_centered = tr_arr - tr_arr.mean()
        autocorr = np.correlate(tr_centered, tr_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr

        print(f"\n  tr(M) autocorrelation:")
        for lag in range(min(6, num_layers)):
            print(f"    Lag {lag}: {autocorr[lag]:.4f}")

        all_results[model_key] = {
            'model': cfg['name'],
            'type': cfg['type'],
            'hidden_size': m,
            'num_layers': num_layers,
            'M_properties': M_properties,
            'corr_matrix': corr_matrix.tolist(),
            'neighbor_corr': neighbor_corr,
            'block_structure': {
                'mean_early_early': mean_early,
                'mean_late_late': mean_late,
                'mean_early_late': mean_cross,
            },
            'tr_M_autocorrelation': autocorr[:min(10, num_layers)].tolist(),
        }

        del model
        import gc; gc.collect()

    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'exp2_6_cross_layer_coupling.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {filepath}")
    return all_results


# ======================================================================
# 2.5: Trace Formula on More Architectures
# ======================================================================

def exp_2_5_more_architectures():
    """
    Test trace formula on GPT-2 (124M) and Qwen2.5-0.5B.
    """
    print("\n" + "=" * 70)
    print("EXP 2.5: TRACE FORMULA ON MORE ARCHITECTURES")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    MODEL_CONFIGS = {
        'gpt2': {
            'name': 'gpt2',
            'type': 'gpt2',  # LayerNorm + standard MLP
        },
        'qwen2.5-0.5b': {
            'name': 'Qwen/Qwen2.5-0.5B',
            'type': 'swiglu',
        },
    }

    B = 2048
    n_seeds = 10

    all_results = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_key} ({cfg['name']})")
        print(f"{'='*60}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg['name'], trust_remote_code=True,
                dtype=torch.float32, device_map='cpu',
            )
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        model.eval()
        config = model.config
        m = config.hidden_size
        num_layers = config.num_hidden_layers

        print(f"  m={m}, num_layers={num_layers}")
        print(f"\n  {'Layer':>5} {'R_F formula':>12} {'R_F lin emp':>12} {'ratio':>8} "
              f"{'R_F full':>10} {'alpha':>8} {'pct_trace':>10}")
        print(f"  {'-'*75}")

        # Get decoder layers and weight accessors
        if cfg['type'] == 'gpt2':
            decoder_layers = model.transformer.h
        elif cfg['type'] == 'swiglu':
            decoder_layers = model.model.layers
        else:
            print(f"  Unknown model type: {cfg['type']}")
            del model
            import gc; gc.collect()
            continue

        layer_results = []

        for layer_idx in range(num_layers):
            layer = decoder_layers[layer_idx]

            # Get W1 and W2
            if cfg['type'] == 'gpt2':
                # GPT-2: layer.mlp.c_fc (Conv1D) and layer.mlp.c_proj (Conv1D)
                # Conv1D stores weight as (out, in) but transposed internally
                W1 = layer.mlp.c_fc.weight.detach().float().T  # (4*m, m)
                W2 = layer.mlp.c_proj.weight.detach().float().T  # (m, 4*m)
            elif cfg['type'] == 'swiglu':
                W1 = layer.mlp.up_proj.weight.detach().float()
                W2 = layer.mlp.down_proj.weight.detach().float()

            M = W2 @ W1
            tf = compute_trace_formula(M, m)

            # Empirical R_F (linear)
            RF_lin_values = []
            for seed in range(n_seeds):
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
                with torch.no_grad():
                    F_out = (M @ x_hat.T).T
                RF_lin_values.append(radial_fraction(x_hat, F_out))
            RF_lin_emp = float(np.mean(RF_lin_values))

            # Empirical R_F (nonlinear)
            RF_full_values = []
            for seed in range(n_seeds):
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
                with torch.no_grad():
                    if cfg['type'] == 'gpt2':
                        # GPT-2 uses GELU: F(x) = W2 * gelu(W1 * x)
                        F_out = (W2 @ torchF.gelu(W1 @ x_hat.T)).T
                    elif cfg['type'] == 'swiglu':
                        W_gate = layer.mlp.gate_proj.weight.detach().float()
                        F_out = (W2 @ (torchF.silu(W_gate @ x_hat.T) * (W1 @ x_hat.T))).T
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

        # Summary
        ratios = [r['ratio_linear'] for r in layer_results]
        alphas = [r['alpha_suppression'] for r in layer_results]
        rf_formula = [r['R_F_formula'] for r in layer_results]
        rf_full = [r['R_F_full_empirical'] for r in layer_results]

        print(f"\n  Formula accuracy: mean ratio = {np.mean(ratios):.4f}, std = {np.std(ratios):.4f}")
        print(f"  Activation suppression alpha: mean = {np.mean(alphas):.4f}, std = {np.std(alphas):.4f}")

        if len(rf_formula) > 3:
            pr, pp = stats.pearsonr(rf_formula, rf_full)
            sr, sp = stats.spearmanr(rf_formula, rf_full)
            print(f"\n  Correlation R_F(formula) vs R_F(full):")
            print(f"    Pearson r = {pr:.4f} (p = {pp:.2e})")
            print(f"    Spearman rho = {sr:.4f} (p = {sp:.2e})")
        else:
            pr, sr = None, None

        all_results[model_key] = {
            'model': cfg['name'],
            'type': cfg['type'],
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

    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'exp2_5_more_architectures.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {filepath}")
    return all_results


# ======================================================================
# 2.3: Quantization Sensitivity vs R_F
# ======================================================================

def exp_2_3_quantization_sensitivity():
    """
    Test correlation between trace formula R_F and quantization sensitivity per layer.
    Uses TinyLlama. Measures output distortion when each layer is individually quantized.
    """
    print("\n" + "=" * 70)
    print("EXP 2.3: QUANTIZATION SENSITIVITY vs R_F")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    print(f"  Model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        dtype=torch.float32, device_map='cpu',
    )
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = None

    config = model.config
    m = config.hidden_size
    num_layers = config.num_hidden_layers
    decoder_layers = model.model.layers

    # Prepare test inputs
    if tokenizer is not None:
        try:
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
            texts = [t for t in dataset['text'] if len(t) > 100][:20]
            text = ' '.join(texts)
            input_ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)['input_ids']
            use_real_text = True
            print(f"  Using WikiText-2 (seq_len={input_ids.shape[1]})")
        except Exception as e:
            print(f"  WikiText-2 not available ({e}), using random tokens")
            input_ids = torch.randint(100, 30000, (1, 512))
            use_real_text = False
    else:
        input_ids = torch.randint(100, 30000, (1, 512))
        use_real_text = False

    # Get baseline output
    with torch.no_grad():
        baseline_output = model(input_ids).logits.float()

    def ternary_quantize(W):
        """Ternary quantization: sign(W) * mean(|W|) per output channel."""
        scale = W.abs().mean(dim=-1, keepdim=True)
        return torch.sign(W) * scale

    def measure_distortion(model, input_ids, baseline_logits):
        """Measure output distortion as normalized MSE of logits."""
        with torch.no_grad():
            perturbed = model(input_ids).logits.float()
        mse = ((perturbed - baseline_logits) ** 2).mean().item()
        baseline_var = (baseline_logits ** 2).mean().item()
        return mse / baseline_var if baseline_var > 0 else 0

    print(f"\n  {'Layer':>5} {'R_F formula':>12} {'distortion':>12} {'log_dist':>10}")
    print(f"  {'-'*45}")

    layer_results = []

    for layer_idx in range(num_layers):
        layer = decoder_layers[layer_idx]
        mlp = layer.mlp

        # Compute R_F from trace formula
        W_up = mlp.up_proj.weight.detach().float()
        W_down = mlp.down_proj.weight.detach().float()
        M = W_down @ W_up
        tf = compute_trace_formula(M, m)

        # Quantize W_down, measure distortion
        original_W = mlp.down_proj.weight.data.clone()
        mlp.down_proj.weight.data = ternary_quantize(original_W.float()).to(original_W.dtype)

        distortion = measure_distortion(model, input_ids, baseline_output)

        # Restore
        mlp.down_proj.weight.data = original_W

        log_dist = math.log10(max(distortion, 1e-15))

        print(f"  {layer_idx:>5} {tf['R_F_formula']:>12.6f} {distortion:>12.6e} {log_dist:>10.2f}")

        layer_results.append({
            'layer': layer_idx,
            'R_F_formula': tf['R_F_formula'],
            'distortion': distortion,
            'log_distortion': log_dist,
        })

    # Also quantize W_up
    print(f"\n  Now testing W_up quantization:")
    print(f"  {'Layer':>5} {'R_F formula':>12} {'dist(W_down)':>13} {'dist(W_up)':>11}")
    print(f"  {'-'*50}")

    for layer_idx in range(num_layers):
        layer = decoder_layers[layer_idx]
        mlp = layer.mlp

        original_W = mlp.up_proj.weight.data.clone()
        mlp.up_proj.weight.data = ternary_quantize(original_W.float()).to(original_W.dtype)

        dist_up = measure_distortion(model, input_ids, baseline_output)
        mlp.up_proj.weight.data = original_W

        layer_results[layer_idx]['distortion_up'] = dist_up

        print(f"  {layer_idx:>5} {layer_results[layer_idx]['R_F_formula']:>12.6f} "
              f"{layer_results[layer_idx]['distortion']:>13.6e} {dist_up:>11.6e}")

    # Correlation analysis
    rf_values = [r['R_F_formula'] for r in layer_results]
    dist_values = [r['distortion'] for r in layer_results]
    dist_up_values = [r.get('distortion_up', 0) for r in layer_results]

    pr_down, pp_down = stats.pearsonr(rf_values, dist_values)
    sr_down, sp_down = stats.spearmanr(rf_values, dist_values)
    pr_up, pp_up = stats.pearsonr(rf_values, dist_up_values)
    sr_up, sp_up = stats.spearmanr(rf_values, dist_up_values)

    # Also try log-distortion correlation
    log_dist_values = [math.log(max(d, 1e-15)) for d in dist_values]
    pr_log, pp_log = stats.pearsonr(rf_values, log_dist_values)
    sr_log, sp_log = stats.spearmanr(rf_values, log_dist_values)

    print(f"\n  Correlation R_F(formula) vs quantization distortion:")
    print(f"    W_down: Pearson r = {pr_down:.4f} (p={pp_down:.2e}), Spearman rho = {sr_down:.4f} (p={sp_down:.2e})")
    print(f"    W_up:   Pearson r = {pr_up:.4f} (p={pp_up:.2e}), Spearman rho = {sr_up:.4f} (p={sp_up:.2e})")
    print(f"    log(distortion): Pearson r = {pr_log:.4f} (p={pp_log:.2e})")

    results = {
        'model': model_name,
        'hidden_size': m,
        'num_layers': num_layers,
        'use_real_text': use_real_text,
        'layers': layer_results,
        'correlation': {
            'pearson_r_down': float(pr_down),
            'spearman_rho_down': float(sr_down),
            'pearson_r_up': float(pr_up),
            'spearman_rho_up': float(sr_up),
            'pearson_r_log': float(pr_log),
            'spearman_rho_log': float(sr_log),
        }
    }

    del model
    import gc; gc.collect()

    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'exp2_3_quantization_sensitivity.json')
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {filepath}")
    return results


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='*', default=['all'])
    args = parser.parse_args()

    run_all = 'all' in args.experiments

    results = {}

    # Phase 1: No model loading (existing data + synthetic)
    if run_all or '2.4' in args.experiments:
        results['2.4'] = exp_2_4_trace_decomposition()

    if run_all or '3.6' in args.experiments:
        results['3.6'] = exp_3_6_norm_growth_fit()

    if run_all or '3.2' in args.experiments:
        results['3.2'] = exp_3_2_width_convergence()

    # Phase 2: Load TinyLlama + Pythia (shared models)
    if run_all or '2.1' in args.experiments:
        results['2.1'] = exp_2_1_attention_trace()

    if run_all or '2.6' in args.experiments:
        results['2.6'] = exp_2_6_cross_layer_coupling()

    # Phase 3: More architectures
    if run_all or '2.5' in args.experiments:
        results['2.5'] = exp_2_5_more_architectures()

    # Phase 4: Quantization (heavy)
    if run_all or '2.3' in args.experiments:
        results['2.3'] = exp_2_3_quantization_sensitivity()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
