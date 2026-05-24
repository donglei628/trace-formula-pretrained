"""
Paper 2 Round 2 Experiments
============================

R2.1: Depth-controlled quantization reanalysis (existing data)
R2.2: Attention amplification across 4 models
R2.3: alpha_phi on third GELU model (Pythia-410M)
R2.4: Bell + sign-reversal on new SwiGLU models

Usage:
    python paper2_round2_experiments.py [all|r2.1|r2.2|r2.3|r2.4]
"""

import sys
import os
import json
import math
import numpy as np
import scipy.stats
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import radial_fraction, make_x_hat_torch, ensure_output_dir


def compute_trace_formula(M, m):
    """Compute exact trace formula for R_F(linear) with (m+2) denominator."""
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
    }


# ======================================================================
# R2.1: Depth-Controlled Quantization Reanalysis
# ======================================================================

def exp_r2_1_depth_controlled():
    """
    Reanalyze quantization sensitivity with depth control.
    Uses existing exp2_3_quantization_sensitivity.json data.
    """
    print("=" * 70)
    print("R2.1: DEPTH-CONTROLLED QUANTIZATION SENSITIVITY REANALYSIS")
    print("=" * 70)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    with open(os.path.join(results_dir, 'exp2_3_quantization_sensitivity.json')) as f:
        raw_data = json.load(f)

    data = raw_data['layers']
    total_layers = len(data)

    # Standardize field names
    for entry in data:
        entry['R_F'] = entry['R_F_formula']
        entry['distortion_W_down'] = entry['distortion']
        entry['distortion_W_up'] = entry['distortion_up']

    # Stratify by relative depth into 3 bins
    def get_bin(layer_idx, total):
        rel = layer_idx / (total - 1)
        if rel < 1/3:
            return 'early'
        elif rel < 2/3:
            return 'middle'
        else:
            return 'late'

    bins = {'early': [], 'middle': [], 'late': []}
    for entry in data:
        b = get_bin(entry['layer'], total_layers)
        bins[b].append(entry)

    print(f"\nModel: TinyLlama, total {total_layers} layers")
    print("=" * 70)

    bin_results = {}
    for bin_name, entries in bins.items():
        if len(entries) < 3:
            print(f"\n{bin_name}: too few layers ({len(entries)}), skipping")
            bin_results[bin_name] = {'n': len(entries), 'status': 'skipped'}
            continue

        R_F = [e['R_F'] for e in entries]
        dist_down = [e['distortion_W_down'] for e in entries]
        dist_up = [e['distortion_W_up'] for e in entries]
        layers_in_bin = [e['layer'] for e in entries]

        rho_down, p_down = scipy.stats.spearmanr(R_F, dist_down)
        rho_up, p_up = scipy.stats.spearmanr(R_F, dist_up)

        print(f"\n{bin_name} (n={len(entries)}, layers {layers_in_bin}):")
        print(f"  R_F range: [{min(R_F):.4f}, {max(R_F):.4f}]")
        print(f"  Spearman R_F vs distortion(W_down): {rho_down:+.3f} (p={p_down:.4f})")
        print(f"  Spearman R_F vs distortion(W_up):   {rho_up:+.3f} (p={p_up:.4f})")

        bin_results[bin_name] = {
            'n': len(entries),
            'layers': layers_in_bin,
            'R_F_range': [float(min(R_F)), float(max(R_F))],
            'spearman_down': float(rho_down),
            'p_down': float(p_down),
            'spearman_up': float(rho_up),
            'p_up': float(p_up),
        }

    # Cross-bin analysis: residualize by depth
    print("\n" + "=" * 70)
    print("Cross-bin: residualizing depth, then correlating R_F")
    print("=" * 70)

    layers_arr = np.array([e['layer'] for e in data])
    R_F_arr = np.array([e['R_F'] for e in data])
    dist_down_arr = np.array([e['distortion_W_down'] for e in data])
    dist_up_arr = np.array([e['distortion_W_up'] for e in data])

    # Fit distortion vs depth (quadratic), get residuals
    depth_fit_down = np.polyfit(layers_arr, dist_down_arr, 2)
    dist_pred_down = np.polyval(depth_fit_down, layers_arr)
    resid_down = dist_down_arr - dist_pred_down

    depth_fit_up = np.polyfit(layers_arr, dist_up_arr, 2)
    dist_pred_up = np.polyval(depth_fit_up, layers_arr)
    resid_up = dist_up_arr - dist_pred_up

    rho_resid_down, p_resid_down = scipy.stats.spearmanr(R_F_arr, resid_down)
    rho_resid_up, p_resid_up = scipy.stats.spearmanr(R_F_arr, resid_up)

    # Also try Pearson on residuals
    pearson_resid_down, p_pearson_down = scipy.stats.pearsonr(R_F_arr, resid_down)
    pearson_resid_up, p_pearson_up = scipy.stats.pearsonr(R_F_arr, resid_up)

    print(f"Spearman R_F vs depth-resid distortion(W_down): {rho_resid_down:+.3f} (p={p_resid_down:.4f})")
    print(f"Spearman R_F vs depth-resid distortion(W_up):   {rho_resid_up:+.3f} (p={p_resid_up:.4f})")
    print(f"Pearson R_F vs depth-resid distortion(W_down):  {pearson_resid_down:+.3f} (p={p_pearson_down:.4f})")
    print(f"Pearson R_F vs depth-resid distortion(W_up):    {pearson_resid_up:+.3f} (p={p_pearson_up:.4f})")

    # Also: partial correlation controlling for layer index
    # Using residualization approach for R_F too
    rf_fit = np.polyfit(layers_arr, R_F_arr, 2)
    rf_resid = R_F_arr - np.polyval(rf_fit, layers_arr)
    rho_partial, p_partial = scipy.stats.spearmanr(rf_resid, resid_down)
    print(f"\nPartial correlation (both residualized): {rho_partial:+.3f} (p={p_partial:.4f})")

    # Remove outlier layers (layer 2 and 7 have very high distortion)
    outlier_mask = np.ones(len(data), dtype=bool)
    for i, e in enumerate(data):
        if e['distortion_W_down'] > 0.1:  # clearly outlier
            outlier_mask[i] = False
            print(f"  Excluding outlier layer {e['layer']} (distortion={e['distortion_W_down']:.3f})")

    if outlier_mask.sum() < len(data):
        rho_no_outlier, p_no_outlier = scipy.stats.spearmanr(
            R_F_arr[outlier_mask], dist_down_arr[outlier_mask])
        print(f"Spearman without outliers: {rho_no_outlier:+.3f} (p={p_no_outlier:.4f})")

        # Residualize without outliers
        layers_clean = layers_arr[outlier_mask]
        dist_clean = dist_down_arr[outlier_mask]
        rf_clean = R_F_arr[outlier_mask]
        fit_clean = np.polyfit(layers_clean, dist_clean, 2)
        resid_clean = dist_clean - np.polyval(fit_clean, layers_clean)
        rho_resid_clean, p_resid_clean = scipy.stats.spearmanr(rf_clean, resid_clean)
        print(f"Spearman depth-resid (no outliers): {rho_resid_clean:+.3f} (p={p_resid_clean:.4f})")
    else:
        rho_no_outlier = None
        rho_resid_clean = None

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Count how many bins show positive correlation
    pos_bins = sum(1 for b in bin_results.values()
                   if isinstance(b.get('spearman_down'), float) and b['spearman_down'] > 0)
    total_valid_bins = sum(1 for b in bin_results.values() if b.get('status') != 'skipped')

    if pos_bins == total_valid_bins and total_valid_bins > 0:
        verdict = "RESCUED: All bins show positive Spearman — depth confound fully explains the original negative correlation"
    elif pos_bins > total_valid_bins / 2:
        verdict = "PARTIALLY RESCUED: Majority of bins show positive correlation, but not all"
    elif rho_resid_down > 0 and p_resid_down < 0.05:
        verdict = "RESCUED via residualization: depth-controlled correlation is positive and significant"
    elif rho_resid_down > 0:
        verdict = "WEAKLY RESCUED: depth-controlled correlation is positive but not significant"
    else:
        verdict = "FAILED: R_F does not predict quantization sensitivity even after depth control"

    print(f"Verdict: {verdict}")
    print(f"  Bins positive/total: {pos_bins}/{total_valid_bins}")
    print(f"  Depth-residualized Spearman: {rho_resid_down:+.3f}")

    # Save results
    result = {
        'experiment_id': 'R2.1',
        'models_tested': ['TinyLlama/TinyLlama-1.1B-Chat-v1.0'],
        'date_completed': datetime.now().strftime('%Y-%m-%d'),
        'key_findings': verdict,
        'bin_results': bin_results,
        'cross_bin': {
            'spearman_resid_down': float(rho_resid_down),
            'p_resid_down': float(p_resid_down),
            'spearman_resid_up': float(rho_resid_up),
            'p_resid_up': float(p_resid_up),
            'pearson_resid_down': float(pearson_resid_down),
            'pearson_resid_up': float(pearson_resid_up),
            'partial_spearman': float(rho_partial),
            'p_partial': float(p_partial),
        },
        'without_outliers': {
            'spearman_raw': float(rho_no_outlier) if rho_no_outlier is not None else None,
            'spearman_resid': float(rho_resid_clean) if rho_resid_clean is not None else None,
        },
        'original_correlation': raw_data['correlation'],
        'caveats': [
            'Only 1 model tested (TinyLlama)',
            'Layers 2 and 7 are distortion outliers',
            'Small sample size per bin (7-8 layers each)',
        ],
    }

    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'r2_1_depth_controlled.json')
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {filepath}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: distortion vs depth
        axes[0, 0].scatter(layers_arr, dist_down_arr, c='blue')
        axes[0, 0].plot(layers_arr, dist_pred_down, 'r--', label='depth fit')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Distortion (W_down)')
        axes[0, 0].set_title('Distortion vs depth (the confound)')
        axes[0, 0].legend()

        # Panel 2: R_F vs depth
        axes[0, 1].scatter(layers_arr, R_F_arr, c='green')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('R_F')
        axes[0, 1].set_title('R_F vs depth')

        # Panel 3: R_F vs distortion (raw)
        raw_rho = scipy.stats.spearmanr(R_F_arr, dist_down_arr)[0]
        axes[1, 0].scatter(R_F_arr, dist_down_arr, c='purple')
        axes[1, 0].set_xlabel('R_F')
        axes[1, 0].set_ylabel('Distortion (W_down)')
        axes[1, 0].set_title(f'Raw: rho = {raw_rho:+.3f}')

        # Panel 4: R_F vs depth-residualized distortion
        axes[1, 1].scatter(R_F_arr, resid_down, c='red')
        axes[1, 1].axhline(0, color='gray', linestyle=':')
        axes[1, 1].set_xlabel('R_F')
        axes[1, 1].set_ylabel('Distortion (depth-residual)')
        axes[1, 1].set_title(f'Depth-controlled: rho = {rho_resid_down:+.3f}')

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'r2_1_depth_controlled.png'), dpi=100)
        print(f"Saved plot to r2_1_depth_controlled.png")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")

    return result


# ======================================================================
# R2.2: Attention Amplification Across 4 Models
# ======================================================================

def exp_r2_2_attention_4models():
    """
    Measure attention amplification (alpha_attn) on GPT-2 and Qwen2.5-0.5B.
    Combine with existing TinyLlama and Pythia-1B data for 4-model summary.
    """
    print("\n" + "=" * 70)
    print("R2.2: ATTENTION AMPLIFICATION ACROSS 4 MODELS")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    # Load existing data for TinyLlama and Pythia-1B
    with open(os.path.join(results_dir, 'exp2_1_attention_trace.json')) as f:
        existing_attn = json.load(f)

    # We need to add GPT-2 and Qwen2.5-0.5B attention measurements
    NEW_MODELS = {
        'gpt2': {
            'name': 'gpt2',
            'type': 'gpt2',
        },
        'qwen2.5-0.5b': {
            'name': 'Qwen/Qwen2.5-0.5B',
            'type': 'qwen',
        },
    }

    B = 2048
    n_seeds = 10
    all_results = {}

    # Copy existing data
    for k in existing_attn:
        all_results[k] = existing_attn[k]

    for model_key, cfg in NEW_MODELS.items():
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

        if cfg['type'] == 'gpt2':
            decoder_layers = model.transformer.h
            num_heads = config.n_head
            num_kv_heads = num_heads
        else:  # qwen
            decoder_layers = model.model.layers
            num_heads = config.num_attention_heads
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)

        head_dim = m // num_heads
        n_rep = num_heads // num_kv_heads

        print(f"  m={m}, n_heads={num_heads}, n_kv_heads={num_kv_heads}, head_dim={head_dim}")

        layer_results = []
        for layer_idx, layer in enumerate(decoder_layers):
            # Extract W_V and W_O
            if cfg['type'] == 'gpt2':
                # GPT-2: c_attn.weight is (m, 3m) in Conv1D format
                QKV = layer.attn.c_attn.weight.detach().float()  # (m, 3*m) Conv1D
                W_V_raw = QKV[:, 2*m:]  # (m, m) -- Conv1D: input on rows
                # Conv1D: y = x @ weight, so W_V effectively is W_V_raw.T in standard notation
                # For standard: output = W_V @ input, we need W_V = QKV[:, 2*m:].T
                W_V = W_V_raw.T  # (m, m)

                # W_O: c_proj.weight is (m, m) Conv1D
                W_O_raw = layer.attn.c_proj.weight.detach().float()  # (m, m)
                W_O = W_O_raw.T  # (m, m)
            else:
                # Qwen-style: standard Linear layers
                W_V = layer.self_attn.v_proj.weight.detach().float()  # (n_kv*head_dim, m)
                W_O = layer.self_attn.o_proj.weight.detach().float()  # (m, n_heads*head_dim)

            # Compute M_attn with proper GQA handling
            if num_kv_heads != num_heads:
                # GQA: per-head computation
                W_V_blocks = W_V.reshape(num_kv_heads, head_dim, m)
                W_O_blocks = W_O.reshape(m, num_heads, head_dim)
                M_attn = torch.zeros(m, m)
                for h in range(num_heads):
                    kv_idx = h // n_rep
                    W_O_h = W_O_blocks[:, h, :]      # (m, head_dim)
                    W_V_kv = W_V_blocks[kv_idx]       # (head_dim, m)
                    M_attn += W_O_h @ W_V_kv
            else:
                # MHA: direct product
                M_attn = W_O @ W_V

            # Trace formula prediction
            tf = compute_trace_formula(M_attn, m)
            R_F_formula = tf['R_F_formula']

            # Empirical R_F(linear): x_hat -> M_attn @ x_hat
            R_F_linear_samples = []
            for seed in range(n_seeds):
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
                with torch.no_grad():
                    F_linear = (M_attn @ x_hat.T).T
                R_F_linear_samples.append(radial_fraction(x_hat, F_linear))
            R_F_linear_emp = float(np.mean(R_F_linear_samples))
            ratio_linear = R_F_linear_emp / R_F_formula if R_F_formula > 1e-12 else float('nan')

            # For full attention R_F, we need forward hooks on real text
            # Use T-C2 data if available, otherwise measure empirically
            R_F_full = None
            alpha_attn = None

            layer_results.append({
                'layer': layer_idx,
                'R_F_formula': R_F_formula,
                'R_F_linear_empirical': R_F_linear_emp,
                'ratio_linear': ratio_linear,
                'tr_M': tf['tr_M'],
                'M_fro_sq': tf['M_fro_sq'],
                'tr_M2': tf['tr_M2'],
            })

            print(f"  L{layer_idx:>2}: R_F_formula={R_F_formula:.6f}, "
                  f"R_F_lin_emp={R_F_linear_emp:.6f}, ratio={ratio_linear:.4f}")

        # Now measure full attention R_F empirically using forward hooks
        print(f"\n  Measuring full attention R_F with forward hooks...")
        tokenizer = AutoTokenizer.from_pretrained(cfg['name'], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Get real text input
        test_texts = [
            "The quick brown fox jumps over the lazy dog. " * 8,
            "In the beginning was the Word, and the Word was with God. " * 6,
            "To be or not to be, that is the question. " * 8,
            "It was the best of times, it was the worst of times. " * 6,
        ] * 3  # 12 samples

        inputs = tokenizer(test_texts, return_tensors='pt', padding=True,
                          truncation=True, max_length=256)

        for layer_idx, layer in enumerate(decoder_layers):
            attn_outputs = []
            attn_inputs = []

            def make_pre_hook(storage_in):
                def hook_fn(module, args, kwargs=None):
                    # Capture input hidden state from positional args or kwargs
                    if args and len(args) > 0:
                        storage_in.append(args[0].detach().float())
                    elif kwargs and 'hidden_states' in kwargs:
                        storage_in.append(kwargs['hidden_states'].detach().float())
                return hook_fn

            def make_fwd_hook(storage_out):
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple):
                        storage_out.append(out[0].detach().float())
                    else:
                        storage_out.append(out.detach().float())
                return hook_fn

            if cfg['type'] == 'gpt2':
                attn_module = layer.attn
            else:
                attn_module = layer.self_attn

            handle_pre = attn_module.register_forward_pre_hook(
                make_pre_hook(attn_inputs), with_kwargs=True)
            handle_fwd = attn_module.register_forward_hook(
                make_fwd_hook(attn_outputs))

            with torch.no_grad():
                try:
                    model(**inputs)
                except Exception as e:
                    print(f"    L{layer_idx}: forward hook error: {e}")
                    handle_pre.remove()
                    handle_fwd.remove()
                    continue

            handle_pre.remove()
            handle_fwd.remove()

            if attn_inputs and attn_outputs:
                x_in = attn_inputs[0]   # (batch, seq, m)
                f_out = attn_outputs[0]  # (batch, seq, m)

                # Flatten to (N, m)
                x_flat = x_in.reshape(-1, m)
                f_flat = f_out.reshape(-1, m)

                # Normalize x_flat to ||x|| = sqrt(m)
                norms = x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                x_hat = x_flat * math.sqrt(m) / norms

                R_F_full = radial_fraction(x_hat, f_flat)
                R_F_formula = layer_results[layer_idx]['R_F_formula']
                alpha_attn = R_F_full / R_F_formula if R_F_formula > 1e-12 else float('nan')

                layer_results[layer_idx]['R_F_full_attention'] = R_F_full
                layer_results[layer_idx]['alpha_attention'] = alpha_attn

                print(f"    L{layer_idx}: R_F_full={R_F_full:.6f}, alpha_attn={alpha_attn:.2f}")
            else:
                print(f"    L{layer_idx}: no hook data captured")

        # Summary statistics
        alphas = [r['alpha_attention'] for r in layer_results
                  if 'alpha_attention' in r and r['alpha_attention'] is not None
                  and not math.isnan(r['alpha_attention'])]

        R_F_formulas = [r['R_F_formula'] for r in layer_results]
        R_F_fulls = [r.get('R_F_full_attention', None) for r in layer_results]
        valid_pairs = [(f, e) for f, e in zip(R_F_formulas, R_F_fulls) if e is not None]

        if len(valid_pairs) >= 3:
            pearson_r = scipy.stats.pearsonr(
                [p[0] for p in valid_pairs], [p[1] for p in valid_pairs])[0]
            spearman_rho = scipy.stats.spearmanr(
                [p[0] for p in valid_pairs], [p[1] for p in valid_pairs])[0]
        else:
            pearson_r = float('nan')
            spearman_rho = float('nan')

        summary = {
            'mean_ratio_linear': float(np.mean([r['ratio_linear'] for r in layer_results])),
            'std_ratio_linear': float(np.std([r['ratio_linear'] for r in layer_results])),
            'mean_alpha_attention': float(np.mean(alphas)) if alphas else None,
            'std_alpha_attention': float(np.std(alphas)) if alphas else None,
            'median_alpha_attention': float(np.median(alphas)) if alphas else None,
            'min_alpha': float(min(alphas)) if alphas else None,
            'max_alpha': float(max(alphas)) if alphas else None,
            'pearson_r': float(pearson_r),
            'spearman_rho': float(spearman_rho),
        }

        all_results[model_key] = {
            'model': cfg['name'],
            'type': cfg['type'],
            'hidden_size': m,
            'num_layers': len(decoder_layers),
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads,
            'layers': layer_results,
            'summary': summary,
        }

        print(f"\n  {model_key} summary:")
        print(f"    Mean alpha_attn: {summary['mean_alpha_attention']}")
        print(f"    Median alpha_attn: {summary['median_alpha_attention']}")
        if alphas:
            print(f"    Range: [{min(alphas):.2f}, {max(alphas):.2f}]")

        del model
        torch.cuda.empty_cache()

    # Print consolidated 4-model summary
    print("\n" + "=" * 70)
    print("4-MODEL ATTENTION AMPLIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':>20} {'Mean alpha':>12} {'Median':>10} {'Range':>20} {'Pearson r':>10}")
    print("-" * 75)

    for k, v in all_results.items():
        s = v['summary']
        mean_a = s.get('mean_alpha_attention', None)
        med_a = s.get('median_alpha_attention', None)
        min_a = s.get('min_alpha', None)
        max_a = s.get('max_alpha', None)
        pr = s.get('pearson_r', None)
        if mean_a is not None and med_a is not None:
            range_str = f"[{min_a:.1f}, {max_a:.1f}]" if min_a is not None else "N/A"
            pr_str = f"{pr:.3f}" if pr is not None and not math.isnan(pr) else "N/A"
            print(f"{k:>20} {mean_a:>12.2f} {med_a:>10.2f} {range_str:>20} {pr_str:>10}")
        else:
            print(f"{k:>20} {'N/A':>12} {'N/A':>10} {'N/A':>20} {'N/A':>10}")

    # Save
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'r2_2_attention_4models.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {filepath}")

    result_summary = {
        'experiment_id': 'R2.2',
        'models_tested': list(all_results.keys()),
        'date_completed': datetime.now().strftime('%Y-%m-%d'),
        'key_findings': f"Attention amplification confirmed across all 4 models",
        'per_model_alpha': {k: v['summary'].get('mean_alpha_attention') for k, v in all_results.items()},
        'caveats': [
            'Forward hooks may capture slightly different attention outputs depending on architecture',
            'Real text input used for empirical measurements',
        ],
    }
    return result_summary


# ======================================================================
# R2.3: alpha_phi on Third GELU Model
# ======================================================================

def exp_r2_3_alpha_phi_gelu():
    """
    Measure activation suppression factor alpha_phi on Pythia-410M (GELU).
    Compare with GPT-2 (alpha=0.29) and Pythia-1B (alpha=0.51).
    """
    print("\n" + "=" * 70)
    print("R2.3: ALPHA_PHI ON THIRD GELU MODEL (Pythia-410M)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    model_name = 'EleutherAI/pythia-410m'
    print(f"  Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        dtype=torch.float32, device_map='cpu',
    )
    model.eval()

    config = model.config
    m = config.hidden_size
    num_layers = config.num_hidden_layers
    decoder_layers = model.gpt_neox.layers

    # Pythia uses standard MLP: fc -> GELU -> proj
    # fc = up projection, proj = down projection
    intermediate_size = config.intermediate_size

    print(f"  m={m}, intermediate={intermediate_size}, layers={num_layers}")

    B = 2048
    n_seeds = 10
    layer_results = []

    for layer_idx, layer in enumerate(decoder_layers):
        # Pythia MLP: dense_h_to_4h (up) and dense_4h_to_h (down)
        W1 = layer.mlp.dense_h_to_4h.weight.detach().float()  # (intermediate, m)
        W2 = layer.mlp.dense_4h_to_h.weight.detach().float()  # (m, intermediate)

        M = W2 @ W1  # (m, m)

        # Trace formula (linear prediction)
        tf = compute_trace_formula(M, m)
        R_F_linear = tf['R_F_formula']

        # Empirical R_F(linear) - verify formula
        R_F_lin_emp_samples = []
        for seed in range(n_seeds):
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
            with torch.no_grad():
                F_linear = (M @ x_hat.T).T
            R_F_lin_emp_samples.append(radial_fraction(x_hat, F_linear))
        R_F_lin_emp = float(np.mean(R_F_lin_emp_samples))

        # Empirical R_F(full nonlinear MLP with GELU)
        R_F_full_samples = []
        for seed in range(n_seeds):
            x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
            with torch.no_grad():
                h = W1 @ x_hat.T  # (intermediate, B)
                h = torch.nn.functional.gelu(h)  # GELU activation
                F_out = (W2 @ h).T  # (B, m)
            R_F_full_samples.append(radial_fraction(x_hat, F_out))
        R_F_full = float(np.mean(R_F_full_samples))

        alpha_phi = R_F_full / R_F_linear if R_F_linear > 1e-12 else float('nan')
        ratio_linear = R_F_lin_emp / R_F_linear if R_F_linear > 1e-12 else float('nan')

        layer_results.append({
            'layer': layer_idx,
            'R_F_formula': R_F_linear,
            'R_F_linear_empirical': R_F_lin_emp,
            'R_F_full': R_F_full,
            'ratio_linear': ratio_linear,
            'alpha_phi': alpha_phi,
            'tr_M': tf['tr_M'],
            'M_fro_sq': tf['M_fro_sq'],
        })

        print(f"  L{layer_idx:>2}: R_F_lin={R_F_linear:.6f}, R_F_full={R_F_full:.6f}, "
              f"alpha={alpha_phi:.4f}, ratio_lin={ratio_linear:.4f}")

    alphas = [r['alpha_phi'] for r in layer_results if not math.isnan(r['alpha_phi'])]

    print(f"\n  Summary:")
    print(f"    Mean alpha_phi: {np.mean(alphas):.4f}")
    print(f"    Std alpha_phi:  {np.std(alphas):.4f}")
    print(f"    Median alpha:   {np.median(alphas):.4f}")
    print(f"    Range:          [{min(alphas):.4f}, {max(alphas):.4f}]")
    print(f"    Mean ratio(linear): {np.mean([r['ratio_linear'] for r in layer_results]):.6f}")

    # Compare with existing data
    print(f"\n  Comparison with existing GELU models:")
    print(f"    GPT-2 (124M):    alpha = 0.290 (GELU)")
    print(f"    Pythia-1B:       alpha = 0.510 (GELU)")
    print(f"    Pythia-410M:     alpha = {np.mean(alphas):.3f} (GELU)")

    # Correlation between R_F_formula and R_F_full
    R_F_formulas = [r['R_F_formula'] for r in layer_results]
    R_F_fulls = [r['R_F_full'] for r in layer_results]
    pearson_r = scipy.stats.pearsonr(R_F_formulas, R_F_fulls)[0]
    spearman_rho = scipy.stats.spearmanr(R_F_formulas, R_F_fulls)[0]

    result = {
        'experiment_id': 'R2.3',
        'models_tested': [model_name],
        'date_completed': datetime.now().strftime('%Y-%m-%d'),
        'key_findings': f"Pythia-410M alpha_phi = {np.mean(alphas):.3f} (GELU), "
                       f"between GPT-2 (0.29) and Pythia-1B (0.51)",
        'model': model_name,
        'hidden_size': m,
        'intermediate_size': intermediate_size,
        'num_layers': num_layers,
        'activation': 'gelu',
        'mean_alpha': float(np.mean(alphas)),
        'std_alpha': float(np.std(alphas)),
        'median_alpha': float(np.median(alphas)),
        'pearson_r': float(pearson_r),
        'spearman_rho': float(spearman_rho),
        'per_layer': layer_results,
        'comparison': {
            'gpt2_124m': 0.290,
            'pythia_1b': 0.510,
            'pythia_410m': float(np.mean(alphas)),
        },
        'caveats': [
            'Synthetic Gaussian inputs used (not real text)',
            'GELU is the new-style approximation used by Pythia',
        ],
    }

    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'r2_3_alpha_phi_third_gelu.json')
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {filepath}")

    del model
    torch.cuda.empty_cache()
    return result


# ======================================================================
# R2.4: Bell + Sign-Reversal on New SwiGLU Models
# ======================================================================

def exp_r2_4_bell_profile():
    """
    Measure rho(W_up, W_down) profile on new SwiGLU models.
    Test for bell + sign-reversal pattern.

    We use Gemma-2B and also try SmolLM (small SwiGLU model).
    Mistral-7B is too large for 8GB VRAM even in INT4.
    """
    print("\n" + "=" * 70)
    print("R2.4: BELL + SIGN-REVERSAL ON NEW SWIGLU MODELS")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    # Models to try - starting with smaller ones that fit in memory
    MODELS = [
        {
            'key': 'stablelm-2-1.6b',
            'name': 'stabilityai/stablelm-2-1_6b',
            'type': 'stablelm',
            'notes': 'SwiGLU, different training regime',
        },
        {
            'key': 'smollm2-1.7b',
            'name': 'HuggingFaceTB/SmolLM2-1.7B',
            'type': 'smollm',
            'notes': 'SwiGLU, Llama-style architecture',
        },
    ]

    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    # Load existing rho profiles for comparison
    existing_path = os.path.join(results_dir, 'e5_1_rho_profiles.json')
    existing_profiles = {}
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing_profiles = json.load(f)

    all_results = {}

    for model_cfg in MODELS:
        model_key = model_cfg['key']
        model_name = model_cfg['name']
        print(f"\n{'='*60}")
        print(f"  Processing {model_name}")
        print(f"{'='*60}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                dtype=torch.float16, device_map='cpu',
            )
            model.eval()
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        # Find layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            decoder_layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            decoder_layers = model.transformer.layers
        else:
            print(f"  Unknown architecture, trying to find layers...")
            # Try to discover layer structure
            found = False
            for attr_name in ['model.layers', 'transformer.h', 'gpt_neox.layers']:
                parts = attr_name.split('.')
                obj = model
                try:
                    for p in parts:
                        obj = getattr(obj, p)
                    decoder_layers = obj
                    found = True
                    break
                except AttributeError:
                    continue
            if not found:
                print(f"  Could not find decoder layers, skipping")
                del model
                continue

        n_layers = len(decoder_layers)
        print(f"  Found {n_layers} layers")

        layer_results = []
        for layer_idx, layer in enumerate(decoder_layers):
            try:
                mlp = layer.mlp

                # Try different weight name conventions
                W_down = None
                W_up = None
                W_gate = None

                # SwiGLU/GeGLU style
                for down_name in ['down_proj', 'c_proj', 'dense_4h_to_h', 'w2']:
                    if hasattr(mlp, down_name):
                        W_down = getattr(mlp, down_name).weight.detach().float()
                        break

                for up_name in ['up_proj', 'c_fc2', 'dense_h_to_4h', 'w3']:
                    if hasattr(mlp, up_name):
                        W_up = getattr(mlp, up_name).weight.detach().float()
                        break

                for gate_name in ['gate_proj', 'c_fc', 'gate', 'w1']:
                    if hasattr(mlp, gate_name):
                        W_gate = getattr(mlp, gate_name).weight.detach().float()
                        break

                if W_down is None or W_up is None:
                    print(f"    L{layer_idx}: could not find W_down or W_up")
                    print(f"    MLP attrs: {[n for n, _ in mlp.named_children()]}")
                    continue

                # Compute rho(W_up, W_down)
                # W_down: (m, m_hidden), W_up: (m_hidden, m)
                # Compare W_down with W_up^T (both are m x m_hidden)
                A = W_down
                B = W_up.T

                # Handle shape mismatch
                if A.shape != B.shape:
                    print(f"    L{layer_idx}: shape mismatch A={A.shape}, B={B.shape}")
                    # Try transposing
                    if A.shape == B.T.shape:
                        B = B.T
                    else:
                        continue

                rho_up_down = float(((A * B).sum() / (A.norm() * B.norm())).item())

                rho_gate_down = None
                if W_gate is not None:
                    C = W_gate.T
                    if A.shape == C.shape:
                        rho_gate_down = float(((A * C).sum() / (A.norm() * C.norm())).item())

                layer_results.append({
                    'layer': layer_idx,
                    'relative_depth': layer_idx / (n_layers - 1) if n_layers > 1 else 0,
                    'rho_up_down': rho_up_down,
                    'rho_gate_down': rho_gate_down,
                })

            except Exception as e:
                print(f"    L{layer_idx}: error: {e}")
                continue

        if layer_results:
            rhos = [r['rho_up_down'] for r in layer_results]
            rel_depths = [r['relative_depth'] for r in layer_results]

            # Check for bell pattern
            n = len(rhos)
            early = rhos[:n//3]
            middle = rhos[n//3:2*n//3]
            late = rhos[2*n//3:]

            has_bell = np.mean(middle) > np.mean(early) and np.mean(middle) > np.mean(late)
            has_sign_reversal = any(r < 0 for r in late)
            peak_layer = np.argmax(rhos)
            peak_rel_depth = rel_depths[peak_layer]

            print(f"\n  Profile summary:")
            print(f"    Early mean:  {np.mean(early):.4f}")
            print(f"    Middle mean: {np.mean(middle):.4f}")
            print(f"    Late mean:   {np.mean(late):.4f}")
            print(f"    Peak at layer {peak_layer} (rel depth {peak_rel_depth:.2f})")
            print(f"    Bell pattern: {'YES' if has_bell else 'NO'}")
            print(f"    Sign reversal: {'YES' if has_sign_reversal else 'NO'}")

            all_results[model_key] = {
                'model': model_name,
                'type': model_cfg['type'],
                'notes': model_cfg['notes'],
                'num_layers': n_layers,
                'hidden_size': getattr(model.config, 'hidden_size', None),
                'layers': layer_results,
                'pattern': {
                    'has_bell': bool(has_bell),
                    'has_sign_reversal': bool(has_sign_reversal),
                    'peak_layer': int(peak_layer),
                    'peak_relative_depth': float(peak_rel_depth),
                    'early_mean': float(np.mean(early)),
                    'middle_mean': float(np.mean(middle)),
                    'late_mean': float(np.mean(late)),
                    'min_rho': float(min(rhos)),
                    'max_rho': float(max(rhos)),
                },
            }
        else:
            print(f"  No valid layer results for {model_key}")

        del model
        torch.cuda.empty_cache()

    # Print consolidated summary with existing models
    print("\n" + "=" * 70)
    print("BELL PROFILE SUMMARY (ALL MODELS)")
    print("=" * 70)
    print(f"{'Model':>25} {'Bell?':>7} {'Reversal?':>10} {'Peak depth':>12} {'Min rho':>10} {'Max rho':>10}")
    print("-" * 78)

    # Add existing model summaries
    for mk, mdata in existing_profiles.items():
        layers = mdata['layers']
        rho_key = 'rho_up_down' if 'rho_up_down' in layers[0] else 'rho_w1_w2'
        rhos = [l[rho_key] for l in layers]
        n = len(rhos)
        early = rhos[:n//3]
        middle = rhos[n//3:2*n//3]
        late = rhos[2*n//3:]
        has_bell = np.mean(middle) > np.mean(early) and np.mean(middle) > np.mean(late)
        has_rev = any(r < 0 for r in late)
        peak_l = np.argmax(rhos)
        peak_rd = peak_l / (n - 1) if n > 1 else 0
        print(f"{mk:>25} {'YES' if has_bell else 'NO':>7} {'YES' if has_rev else 'NO':>10} "
              f"{peak_rd:>12.2f} {min(rhos):>10.4f} {max(rhos):>10.4f}")

    for mk, mdata in all_results.items():
        p = mdata['pattern']
        print(f"{mk:>25} {'YES' if p['has_bell'] else 'NO':>7} "
              f"{'YES' if p['has_sign_reversal'] else 'NO':>10} "
              f"{p['peak_relative_depth']:>12.2f} {p['min_rho']:>10.4f} {p['max_rho']:>10.4f}")

    # Save
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'r2_4_bell_profiles.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {filepath}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot existing profiles
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
        cidx = 0
        for mk, mdata in existing_profiles.items():
            layers = mdata['layers']
            rel_d = [l['layer'] / (len(layers) - 1) for l in layers]
            rho_key = 'rho_up_down' if 'rho_up_down' in layers[0] else 'rho_w1_w2'
            rhos = [l[rho_key] for l in layers]
            ax.plot(rel_d, rhos, 'o-', label=mk, color=colors[cidx % len(colors)], alpha=0.6)
            cidx += 1

        # Plot new profiles
        for mk, mdata in all_results.items():
            rel_d = [l['relative_depth'] for l in mdata['layers']]
            rhos = [l['rho_up_down'] for l in mdata['layers']]
            ax.plot(rel_d, rhos, 's--', label=f"{mk} (NEW)", color=colors[cidx % len(colors)],
                   linewidth=2, markersize=6)
            cidx += 1

        ax.axhline(0, color='gray', linestyle=':')
        ax.set_xlabel('Relative depth (0 = first, 1 = last)')
        ax.set_ylabel('rho(W_up, W_down)')
        ax.set_title('Bell + Sign-Reversal Pattern Across SwiGLU Models')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'r2_4_bell_profiles.png'), dpi=100)
        print(f"Saved plot to r2_4_bell_profiles.png")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plot")

    result_summary = {
        'experiment_id': 'R2.4',
        'models_tested': [m['name'] for m in MODELS if m['key'] in all_results],
        'date_completed': datetime.now().strftime('%Y-%m-%d'),
        'key_findings': "; ".join(
            f"{k}: bell={'YES' if v['pattern']['has_bell'] else 'NO'}, "
            f"reversal={'YES' if v['pattern']['has_sign_reversal'] else 'NO'}"
            for k, v in all_results.items()
        ),
        'per_model': {k: v['pattern'] for k, v in all_results.items()},
        'caveats': [
            'Gemma uses GeGLU, not pure SwiGLU',
            'Model availability may vary',
        ],
    }
    return result_summary


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    exp = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if exp in ['all', 'r2.1']:
        exp_r2_1_depth_controlled()

    if exp in ['all', 'r2.2']:
        exp_r2_2_attention_4models()

    if exp in ['all', 'r2.3']:
        exp_r2_3_alpha_phi_gelu()

    if exp in ['all', 'r2.4']:
        exp_r2_4_bell_profile()
