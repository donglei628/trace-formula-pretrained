"""
Paper 2 Round 3 Experiments — Causal Mechanism Tests
=====================================================

R3.1: Controlled quantization-R_F causal test (synthetic layers)
R3.2: Attention amplification input dependence test (TinyLlama, Pythia-1B)

Usage:
    python paper2_round3_experiments.py [all|r3.1|r3.2]
"""

import sys
import os
import json
import math
import numpy as np
import scipy.stats
from datetime import datetime

import torch
import torch.nn.functional as torchF

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import radial_fraction, make_x_hat_torch, ensure_output_dir


# ======================================================================
# R3.1: Controlled Quantization-R_F Causal Test
# ======================================================================

def construct_layer_with_target_RF(m, target_rho, seed):
    """
    Construct (W_1, W_2) with controlled rho via:
    W_2 = rho * kappa * W_1^T + sqrt(1-rho^2) * W_2_indep
    where kappa normalizes so ||W_2|| ~ ||W_2_indep||.
    """
    rng = np.random.default_rng(seed)
    W_1 = torch.tensor(rng.normal(0, 1/np.sqrt(m), (m, m)).astype(np.float32))
    W_2_indep = torch.tensor(rng.normal(0, 1/np.sqrt(m), (m, m)).astype(np.float32))

    norm_indep = torch.norm(W_2_indep)
    norm_W1T = torch.norm(W_1.T)
    kappa = norm_indep / norm_W1T

    W_2 = target_rho * kappa * W_1.T + math.sqrt(1 - target_rho**2) * W_2_indep
    return W_1, W_2


def compute_trace_formula_RF(W_1, W_2):
    """Compute R_F via trace formula."""
    M = W_2 @ W_1
    m = M.shape[0]
    tr_M = torch.trace(M).item()
    tr_M2 = torch.trace(M @ M).item()
    norm_F_sq = (M**2).sum().item()
    if norm_F_sq < 1e-15:
        return 1.0 / m
    return (tr_M**2 + tr_M2 + norm_F_sq) / ((m + 2) * norm_F_sq)


def ternary_quantize(W):
    """Per-row ternary: W_ij -> sign(W_ij) * mean(|W_i.|)."""
    alpha = W.abs().mean(dim=1, keepdim=True)
    return torch.sign(W) * alpha


def apply_activation(z, name):
    if name == 'identity':
        return z
    elif name == 'relu':
        return torch.relu(z)
    elif name == 'gelu':
        return torchF.gelu(z)
    elif name == 'silu':
        return torchF.silu(z)
    else:
        raise ValueError(f"Unknown activation: {name}")


def measure_quantization_distortion(W_1, W_2, activation, n_samples=4000):
    """Measure normalized MSE from ternary quantization."""
    m = W_1.shape[0]
    x = torch.randn(n_samples, m)
    x_hat = x * math.sqrt(m) / x.norm(dim=1, keepdim=True).clamp(min=1e-12)

    # Baseline
    z = x_hat @ W_1.T
    z_phi = apply_activation(z, activation)
    F_baseline = z_phi @ W_2.T

    # Quantize W_2 only
    W_2_q = ternary_quantize(W_2)
    F_quantized_W2 = z_phi @ W_2_q.T

    # Quantize W_1 only
    W_1_q = ternary_quantize(W_1)
    z_q = x_hat @ W_1_q.T
    z_q_phi = apply_activation(z_q, activation)
    F_quantized_W1 = z_q_phi @ W_2.T

    # Quantize both
    F_quantized_both = z_q_phi @ W_2_q.T

    def normalized_mse(F_ref, F_test):
        num = ((F_ref - F_test)**2).sum(1).mean().item()
        den = (F_ref**2).sum(1).mean().item()
        return num / den if den > 1e-15 else 0.0

    return {
        'distortion_W2_only': normalized_mse(F_baseline, F_quantized_W2),
        'distortion_W1_only': normalized_mse(F_baseline, F_quantized_W1),
        'distortion_both': normalized_mse(F_baseline, F_quantized_both),
    }


def exp_r3_1_controlled_quantization():
    """R3.1: Controlled causal test of R_F vs quantization sensitivity."""
    print("=" * 70)
    print("R3.1: CONTROLLED QUANTIZATION-R_F CAUSAL TEST")
    print("=" * 70)

    m = 512
    n_seeds = 10
    target_rhos = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    activations = ['identity', 'relu', 'gelu', 'silu']

    results = []

    for target_rho in target_rhos:
        for seed in range(n_seeds):
            W_1, W_2 = construct_layer_with_target_RF(m, target_rho, seed)
            actual_R_F = compute_trace_formula_RF(W_1, W_2)

            W_1_norm = torch.norm(W_1).item()
            W_2_norm = torch.norm(W_2).item()
            M = W_2 @ W_1
            M_norm = torch.norm(M).item()
            spectral_radius = torch.linalg.matrix_norm(M, ord=2).item()

            for activation in activations:
                distortion = measure_quantization_distortion(W_1, W_2, activation)
                results.append({
                    'target_rho': target_rho,
                    'actual_R_F': actual_R_F,
                    'seed': seed,
                    'activation': activation,
                    'distortion_W2_only': distortion['distortion_W2_only'],
                    'distortion_W1_only': distortion['distortion_W1_only'],
                    'distortion_both': distortion['distortion_both'],
                    'W_1_norm': W_1_norm,
                    'W_2_norm': W_2_norm,
                    'M_norm': M_norm,
                    'spectral_radius': spectral_radius,
                })

            if seed == 0:
                print(f"  rho={target_rho:.2f}, R_F={actual_R_F:.4f}, "
                      f"||W1||={W_1_norm:.2f}, ||W2||={W_2_norm:.2f}, ||M||={M_norm:.2f}")

    # === Analysis ===
    print("\n" + "=" * 70)
    print("CONTROL VARIABLE STABILITY CHECK")
    print("=" * 70)

    control_data = {}
    for rho in target_rhos:
        matching = [r for r in results if r['target_rho'] == rho and r['activation'] == 'identity']
        w1_norms = [r['W_1_norm'] for r in matching]
        w2_norms = [r['W_2_norm'] for r in matching]
        m_norms = [r['M_norm'] for r in matching]
        rf_vals = [r['actual_R_F'] for r in matching]

        cv_w1 = np.std(w1_norms) / np.mean(w1_norms) * 100
        cv_w2 = np.std(w2_norms) / np.mean(w2_norms) * 100
        cv_m = np.std(m_norms) / np.mean(m_norms) * 100

        print(f"  rho={rho:.2f}: ||W1||={np.mean(w1_norms):.3f} (CV={cv_w1:.1f}%), "
              f"||W2||={np.mean(w2_norms):.3f} (CV={cv_w2:.1f}%), "
              f"||M||={np.mean(m_norms):.3f} (CV={cv_m:.1f}%), "
              f"R_F={np.mean(rf_vals):.4f}")

        control_data[rho] = {
            'mean_W1': float(np.mean(w1_norms)),
            'mean_W2': float(np.mean(w2_norms)),
            'mean_M': float(np.mean(m_norms)),
            'cv_W1': float(cv_w1),
            'cv_W2': float(cv_w2),
            'cv_M': float(cv_m),
            'mean_R_F': float(np.mean(rf_vals)),
        }

    # Check if ||M|| drifts with rho (potential confound)
    rhos_list = sorted(control_data.keys())
    m_norms_list = [control_data[r]['mean_M'] for r in rhos_list]
    m_drift = max(m_norms_list) / min(m_norms_list)
    print(f"\n  ||M|| drift ratio (max/min across rho): {m_drift:.2f}")
    if m_drift > 1.5:
        print(f"  WARNING: ||M|| varies by {m_drift:.1f}x — may confound results")

    # Main result
    print("\n" + "=" * 70)
    print("MAIN RESULT: R_F vs QUANTIZATION DISTORTION")
    print("=" * 70)

    correlation_results = {}
    for activation in activations:
        matching = [r for r in results if r['activation'] == activation]
        R_F = np.array([r['actual_R_F'] for r in matching])

        for dist_key in ['distortion_W2_only', 'distortion_W1_only', 'distortion_both']:
            dist = np.array([r[dist_key] for r in matching])

            rho_s, p_s = scipy.stats.spearmanr(R_F, dist)
            rho_p, p_p = scipy.stats.pearsonr(R_F, dist)

            # Also log-log Pearson
            log_R_F = np.log(np.maximum(R_F, 1e-10))
            log_dist = np.log(np.maximum(dist, 1e-15))
            rho_loglog, p_loglog = scipy.stats.pearsonr(log_R_F, log_dist)

            key = f"{activation}_{dist_key}"
            correlation_results[key] = {
                'spearman': float(rho_s),
                'p_spearman': float(p_s),
                'pearson': float(rho_p),
                'p_pearson': float(p_p),
                'pearson_loglog': float(rho_loglog),
                'p_loglog': float(p_loglog),
            }

            print(f"  {activation:>10s} {dist_key:>20s}: "
                  f"Spearman={rho_s:+.3f} (p={p_s:.1e}), "
                  f"Pearson={rho_p:+.3f}, "
                  f"log-log r={rho_loglog:+.3f}")

    # Also: correlation AFTER controlling for ||M||
    print("\n" + "=" * 70)
    print("||M||-CONTROLLED CORRELATIONS")
    print("=" * 70)

    for activation in activations:
        matching = [r for r in results if r['activation'] == activation]
        R_F = np.array([r['actual_R_F'] for r in matching])
        M_norms = np.array([r['M_norm'] for r in matching])
        dist_w2 = np.array([r['distortion_W2_only'] for r in matching])

        # Residualize both R_F and distortion by ||M||
        rf_fit = np.polyfit(M_norms, R_F, 1)
        rf_resid = R_F - np.polyval(rf_fit, M_norms)
        dist_fit = np.polyfit(M_norms, dist_w2, 1)
        dist_resid = dist_w2 - np.polyval(dist_fit, M_norms)

        rho_controlled, p_controlled = scipy.stats.spearmanr(rf_resid, dist_resid)
        print(f"  {activation:>10s}: Spearman(R_F, distortion | ||M||) = {rho_controlled:+.3f} (p={p_controlled:.1e})")

        correlation_results[f"{activation}_M_controlled"] = {
            'spearman': float(rho_controlled),
            'p_value': float(p_controlled),
        }

    # Determine verdict
    w2_spearman_vals = [correlation_results[f"{a}_distortion_W2_only"]['spearman']
                        for a in activations]
    all_positive = all(s > 0 for s in w2_spearman_vals)
    all_significant = all(
        correlation_results[f"{a}_distortion_W2_only"]['p_spearman'] < 0.01
        for a in activations)
    mean_spearman = np.mean(w2_spearman_vals)

    if all_positive and all_significant and mean_spearman > 0.5:
        verdict = "RESCUED: R_F causally predicts quantization sensitivity (all activations, Spearman > 0.5)"
    elif all_positive and mean_spearman > 0.3:
        verdict = "PARTIALLY RESCUED: Positive correlation in all activations but moderate strength"
    elif any(s > 0.3 for s in w2_spearman_vals):
        verdict = "MIXED: Some activations show correlation, others don't"
    else:
        verdict = "FAILED: No consistent causal relationship between R_F and quantization sensitivity"

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"  W2 Spearman per activation: {dict(zip(activations, [f'{s:+.3f}' for s in w2_spearman_vals]))}")
    print(f"{'='*70}")

    # Generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for col, activation in enumerate(activations):
            matching = [r for r in results if r['activation'] == activation]
            R_F = np.array([r['actual_R_F'] for r in matching])

            for row, dist_key in enumerate(['distortion_W2_only', 'distortion_W1_only']):
                dist = np.array([r[dist_key] for r in matching])
                rho_s = scipy.stats.spearmanr(R_F, dist)[0]

                ax = axes[row, col]
                ax.scatter(R_F, dist, alpha=0.4, s=10)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('R_F (trace formula)')
                label = 'W2' if 'W2' in dist_key else 'W1'
                ax.set_ylabel(f'Distortion ({label})')
                ax.set_title(f'{activation}\nSpearman={rho_s:+.3f}')

        plt.tight_layout()
        outdir = ensure_output_dir('results')
        plt.savefig(os.path.join(outdir, 'r3_1_controlled_quantization.png'), dpi=100)
        print(f"Saved plot to r3_1_controlled_quantization.png")
        plt.close()

        # Control variable plot
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        rhos_arr = np.array(rhos_list)
        axes2[0].plot(rhos_arr, [control_data[r]['mean_W1'] for r in rhos_list], 'bo-')
        axes2[0].set_xlabel('target rho')
        axes2[0].set_ylabel('||W1||')
        axes2[0].set_title('W1 norm stability')

        axes2[1].plot(rhos_arr, [control_data[r]['mean_W2'] for r in rhos_list], 'go-')
        axes2[1].set_xlabel('target rho')
        axes2[1].set_ylabel('||W2||')
        axes2[1].set_title('W2 norm stability')

        axes2[2].plot(rhos_arr, [control_data[r]['mean_M'] for r in rhos_list], 'ro-')
        axes2[2].set_xlabel('target rho')
        axes2[2].set_ylabel('||M||')
        axes2[2].set_title('M norm vs rho')

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'r3_1_control_variables.png'), dpi=100)
        print(f"Saved control variable plot")
        plt.close()
    except ImportError:
        print("matplotlib not available, skipping plots")

    # Save
    outdir = ensure_output_dir('results')
    output = {
        'experiment_id': 'R3.1',
        'date_completed': datetime.now().strftime('%Y-%m-%d'),
        'models_tested': ['synthetic'],
        'parameters': {
            'm': m,
            'n_seeds': n_seeds,
            'target_rhos': target_rhos,
            'activations': activations,
            'n_samples': 4000,
        },
        'key_findings': verdict,
        'control_variables': control_data,
        'M_norm_drift_ratio': float(m_drift),
        'correlations': correlation_results,
        'per_activation_W2_spearman': dict(zip(activations, [float(s) for s in w2_spearman_vals])),
        'raw_data': results,
        'caveats': [
            f'||M|| varies by {m_drift:.1f}x across rho values',
            'Synthetic layers only (m=512)',
            'Ternary quantization may behave differently from INT4/INT8',
        ],
    }
    filepath = os.path.join(outdir, 'r3_1_controlled_quantization.json')
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {filepath}")
    return output


# ======================================================================
# R3.2: Attention Amplification Input Dependence Test
# ======================================================================

def exp_r3_2_attention_input_dependence():
    """
    Test whether alpha_attn depends on input distribution.
    Compare random tokens, natural text, repeated tokens on TinyLlama and Pythia-1B.
    """
    print("\n" + "=" * 70)
    print("R3.2: ATTENTION AMPLIFICATION INPUT DEPENDENCE TEST")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_CONFIGS = [
        {
            'key': 'tinyllama',
            'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'type': 'llama',
        },
        {
            'key': 'pythia-1b',
            'name': 'EleutherAI/pythia-1b',
            'type': 'pythia',
        },
    ]

    input_types = ['random_tokens', 'natural_text', 'repeated_tokens']
    n_batches = 10
    batch_size = 4
    seq_len = 256

    all_results = {}

    for model_cfg in MODEL_CONFIGS:
        model_key = model_cfg['key']
        model_name = model_cfg['name']
        print(f"\n{'='*60}")
        print(f"  Model: {model_key}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            dtype=torch.float16, device_map='cuda',
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = model.config
        m = config.hidden_size

        if model_cfg['type'] == 'llama':
            decoder_layers = model.model.layers
            num_heads = config.num_attention_heads
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
        else:
            decoder_layers = model.gpt_neox.layers
            num_heads = config.num_attention_heads
            num_kv_heads = num_heads

        n_layers = len(decoder_layers)
        head_dim = m // num_heads
        n_rep = num_heads // num_kv_heads if num_kv_heads != num_heads else 1

        # Pre-compute R_F_linear for each layer (architecture-only, doesn't depend on input)
        R_F_linear_per_layer = []
        for layer_idx, layer in enumerate(decoder_layers):
            if model_cfg['type'] == 'llama':
                W_V = layer.self_attn.v_proj.weight.detach().float()
                W_O = layer.self_attn.o_proj.weight.detach().float()
            else:
                # Pythia: combined QKV in query_key_value, separate dense (output proj)
                QKV = layer.attention.query_key_value.weight.detach().float()
                # Pythia QKV shape: (3 * n_heads * head_dim, m) = (3*m, m)
                # Split into Q, K, V
                W_V = QKV[2*m:]  # (m, m) for standard MHA
                W_O = layer.attention.dense.weight.detach().float()  # (m, m)

            # Compute M_attn with GQA handling (on CPU for trace formula)
            W_V_cpu = W_V.cpu().float()
            W_O_cpu = W_O.cpu().float()
            if num_kv_heads != num_heads:
                W_V_blocks = W_V_cpu.reshape(num_kv_heads, head_dim, m)
                W_O_blocks = W_O_cpu.reshape(m, num_heads, head_dim)
                M_attn = torch.zeros(m, m)
                for h in range(num_heads):
                    kv_idx = h // n_rep
                    W_O_h = W_O_blocks[:, h, :]
                    W_V_kv = W_V_blocks[kv_idx]
                    M_attn += W_O_h @ W_V_kv
            else:
                M_attn = W_O_cpu @ W_V_cpu

            tr_M = torch.trace(M_attn).item()
            tr_M2 = torch.trace(M_attn @ M_attn).item()
            norm_F_sq = (M_attn**2).sum().item()
            R_F_lin = (tr_M**2 + tr_M2 + norm_F_sq) / ((m + 2) * norm_F_sq)
            R_F_linear_per_layer.append(R_F_lin)

        # Generate input batches for each type
        def get_input_batches(input_type):
            if input_type == 'random_tokens':
                vocab_size = tokenizer.vocab_size
                return [torch.randint(0, vocab_size, (batch_size, seq_len))
                        for _ in range(n_batches)]

            elif input_type == 'natural_text':
                # Use a variety of natural text
                texts = [
                    "The United States of America is a federal republic composed of 50 states. "
                    "It is the third-largest country by total area and the third-most populous. "
                    "The nation was founded through a revolution against British colonial rule. " * 3,
                    "Machine learning is a subset of artificial intelligence that provides "
                    "systems the ability to automatically learn and improve from experience. "
                    "It focuses on the development of computer programs that can access data. " * 3,
                    "In physics, the theory of relativity encompasses two interrelated theories "
                    "by Albert Einstein: special relativity and general relativity. These theories "
                    "describe how measurements of physical quantities differ. " * 3,
                    "The Renaissance was a cultural movement that profoundly affected European "
                    "intellectual life in the early modern period. Beginning in Italy, it spread "
                    "to the rest of Europe by the 16th century, marking the end. " * 3,
                ] * (n_batches * batch_size // 4 + 1)
                batches = []
                for i in range(n_batches):
                    batch_texts = texts[i*batch_size:(i+1)*batch_size]
                    encoded = tokenizer(batch_texts, return_tensors='pt', padding='max_length',
                                       truncation=True, max_length=seq_len)
                    batches.append(encoded['input_ids'])
                return batches

            elif input_type == 'repeated_tokens':
                the_token = tokenizer.encode('the', add_special_tokens=False)[0]
                return [torch.full((batch_size, seq_len), the_token)
                        for _ in range(n_batches)]

        model_results = {}

        for input_type in input_types:
            print(f"\n  Input type: {input_type}")
            batches = get_input_batches(input_type)

            layer_results = []
            for layer_idx in range(n_layers):
                attn_inputs = []
                attn_outputs = []

                if model_cfg['type'] == 'llama':
                    attn_module = decoder_layers[layer_idx].self_attn
                else:
                    attn_module = decoder_layers[layer_idx].attention

                def make_pre_hook(storage):
                    def hook_fn(module, args, kwargs=None):
                        if args and len(args) > 0:
                            storage.append(args[0].detach().float())
                        elif kwargs and 'hidden_states' in kwargs:
                            storage.append(kwargs['hidden_states'].detach().float())
                    return hook_fn

                def make_fwd_hook(storage):
                    def hook_fn(module, inp, out):
                        if isinstance(out, tuple):
                            storage.append(out[0].detach().float())
                        else:
                            storage.append(out.detach().float())
                    return hook_fn

                handle_pre = attn_module.register_forward_pre_hook(
                    make_pre_hook(attn_inputs), with_kwargs=True)
                handle_fwd = attn_module.register_forward_hook(
                    make_fwd_hook(attn_outputs))

                R_F_batch_values = []
                for batch in batches:
                    batch = batch.to(model.device)
                    with torch.no_grad():
                        try:
                            model(batch)
                        except Exception as e:
                            continue

                    if attn_inputs and attn_outputs:
                        x_in = attn_inputs[-1]
                        f_out = attn_outputs[-1]
                        x_flat = x_in.reshape(-1, m)
                        f_flat = f_out.reshape(-1, m)
                        norms = x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                        x_hat = x_flat * math.sqrt(m) / norms
                        R_F_val = radial_fraction(x_hat, f_flat)
                        R_F_batch_values.append(R_F_val)

                handle_pre.remove()
                handle_fwd.remove()

                if R_F_batch_values:
                    R_F_emp = float(np.mean(R_F_batch_values))
                    R_F_lin = R_F_linear_per_layer[layer_idx]
                    alpha = R_F_emp / R_F_lin if R_F_lin > 1e-12 else float('nan')

                    layer_results.append({
                        'layer': layer_idx,
                        'R_F_linear': R_F_lin,
                        'R_F_empirical': R_F_emp,
                        'R_F_empirical_std': float(np.std(R_F_batch_values)),
                        'alpha_attn': alpha,
                    })

                    if layer_idx % 5 == 0:
                        print(f"    L{layer_idx:>2}: alpha={alpha:.2f} (R_F_emp={R_F_emp:.6f})")
                else:
                    layer_results.append({
                        'layer': layer_idx,
                        'R_F_linear': R_F_linear_per_layer[layer_idx],
                        'R_F_empirical': None,
                        'alpha_attn': None,
                    })
                    if layer_idx % 5 == 0:
                        print(f"    L{layer_idx:>2}: no data")

            model_results[input_type] = layer_results

        # Summary per input type
        print(f"\n  Summary for {model_key}:")
        print(f"  {'Input type':<25s} {'Mean alpha':>12s} {'Median':>10s} {'Std':>10s} {'Range':>20s}")
        for input_type in input_types:
            alphas = [l['alpha_attn'] for l in model_results[input_type]
                      if l['alpha_attn'] is not None and not math.isnan(l['alpha_attn'])]
            if alphas:
                print(f"  {input_type:<25s} {np.mean(alphas):>12.2f} {np.median(alphas):>10.2f} "
                      f"{np.std(alphas):>10.2f} [{min(alphas):.2f}, {max(alphas):.2f}]")

        # Friedman test
        try:
            alpha_arrays = []
            for input_type in input_types:
                alphas = [l['alpha_attn'] for l in model_results[input_type]
                          if l['alpha_attn'] is not None and not math.isnan(l['alpha_attn'])]
                alpha_arrays.append(alphas)

            min_len = min(len(a) for a in alpha_arrays)
            alpha_arrays = [a[:min_len] for a in alpha_arrays]
            stat, p = scipy.stats.friedmanchisquare(*alpha_arrays)
            print(f"  Friedman test: chi2={stat:.2f}, p={p:.4f}")
            friedman = {'chi2': float(stat), 'p': float(p)}
        except Exception as e:
            print(f"  Friedman test failed: {e}")
            friedman = None

        # Natural/Random alpha ratio per layer
        random_alphas = {l['layer']: l['alpha_attn'] for l in model_results['random_tokens']
                         if l['alpha_attn'] is not None}
        natural_alphas = {l['layer']: l['alpha_attn'] for l in model_results['natural_text']
                          if l['alpha_attn'] is not None}
        ratios = []
        for layer in random_alphas:
            if layer in natural_alphas and random_alphas[layer] > 0:
                ratios.append(natural_alphas[layer] / random_alphas[layer])
        if ratios:
            print(f"  Natural/Random ratio: mean={np.mean(ratios):.2f}, "
                  f"median={np.median(ratios):.2f}, range=[{min(ratios):.2f}, {max(ratios):.2f}]")

        all_results[model_key] = {
            'model': model_name,
            'hidden_size': m,
            'num_layers': n_layers,
            'per_input_type': model_results,
            'friedman_test': friedman,
            'natural_random_ratio': {
                'mean': float(np.mean(ratios)) if ratios else None,
                'median': float(np.median(ratios)) if ratios else None,
                'values': [float(r) for r in ratios],
            },
        }

        del model
        torch.cuda.empty_cache()

    # Cross-model comparison
    print("\n" + "=" * 70)
    print("CROSS-MODEL INPUT DEPENDENCE SUMMARY")
    print("=" * 70)

    for model_key, mdata in all_results.items():
        print(f"\n  {model_key}:")
        for input_type in input_types:
            alphas = [l['alpha_attn'] for l in mdata['per_input_type'][input_type]
                      if l['alpha_attn'] is not None and not math.isnan(l['alpha_attn'])]
            if alphas:
                cv = np.std(alphas) / np.mean(alphas) * 100
                print(f"    {input_type:<25s}: mean={np.mean(alphas):.2f}, CV={cv:.0f}%")

    # Determine verdict
    verdicts = []
    for model_key, mdata in all_results.items():
        means = {}
        for input_type in input_types:
            alphas = [l['alpha_attn'] for l in mdata['per_input_type'][input_type]
                      if l['alpha_attn'] is not None and not math.isnan(l['alpha_attn'])]
            means[input_type] = np.mean(alphas) if alphas else 0

        if means:
            vals = list(means.values())
            overall_cv = np.std(vals) / np.mean(vals) * 100 if np.mean(vals) > 0 else 0
            verdicts.append(overall_cv)

    mean_cv = np.mean(verdicts) if verdicts else 0
    if mean_cv < 30:
        verdict = "ARCHITECTURAL: alpha_attn varies little across input types (CV < 30%)"
    elif mean_cv < 50:
        verdict = "WEAKLY INPUT-DEPENDENT: moderate variation across input types (30% < CV < 50%)"
    else:
        verdict = "INPUT-DEPENDENT: alpha_attn varies significantly with input distribution (CV > 50%)"

    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}")

    # Save
    outdir = ensure_output_dir('results')
    output = {
        'experiment_id': 'R3.2',
        'date_completed': datetime.now().strftime('%Y-%m-%d'),
        'models_tested': [cfg['name'] for cfg in MODEL_CONFIGS],
        'input_types': input_types,
        'key_findings': verdict,
        'per_model': all_results,
        'caveats': [
            'Natural text uses fixed paragraphs, not diverse corpus',
            'Repeated tokens may cause degenerate attention patterns',
            'Only 2 models tested (extremes of amplification range)',
        ],
    }
    filepath = os.path.join(outdir, 'r3_2_attention_input_dependence.json')
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {filepath}")
    return output


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    exp = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if exp in ['all', 'r3.1']:
        exp_r3_1_controlled_quantization()

    if exp in ['all', 'r3.2']:
        exp_r3_2_attention_input_dependence()
