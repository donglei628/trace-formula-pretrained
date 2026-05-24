"""
Round 5 Experiment: T-C2 — Attention vs MLP R_F in Trained Models
==================================================================

Separately measure R_F for attention layers and MLP layers in trained models.
This extends E5.2 by splitting the per-layer measurement into attention and MLP components.

Design:
1. Hook on attention and MLP sub-layers separately
2. For each, measure R_F, sigma_F^2
3. Compare: do attention and MLP have different R_F profiles?
4. Test: does attention contribute to outlier layer behavior?

Usage:
    python round5_tc2_attention.py
"""

import sys
import os
import json
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import ensure_output_dir


def run_tc2():
    """T-C2: Attention vs MLP R_F separation."""
    print("=" * 70)
    print("T-C2: ATTENTION vs MLP R_F IN TRAINED MODELS")
    print("=" * 70)

    from transformers import AutoTokenizer, AutoModelForCausalLM

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

    n_batches = 10
    batch_size = 4
    seq_len = 256

    all_results = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_key}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(cfg['name'], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg['name'], trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map='cpu',
        )
        model.eval()

        config = model.config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        m = hidden_size

        # Data structure: for each layer, track attention and MLP separately
        layer_data = {i: {
            'attn_R_F': [], 'attn_sigma_F_sq': [], 'attn_norm_sq': [],
            'mlp_R_F': [], 'mlp_sigma_F_sq': [], 'mlp_norm_sq': [],
            'full_R_F': [], 'full_sigma_F_sq': [], 'full_norm_sq': [],
        } for i in range(num_layers)}

        hooks = []

        # Strategy: We need hooks at 3 points per layer:
        # 1. Before the layer (x_pre)
        # 2. After attention + residual (x_mid = x_pre + attn_output)
        # 3. After MLP + residual (x_post = x_mid + mlp_output)
        #
        # For LLaMA: each layer has self_attn and mlp sub-modules
        # The layer's forward does:
        #   residual = x
        #   x = self_attn(norm(x)) + residual   (attention sublayer)
        #   residual = x
        #   x = mlp(post_attention_norm(x)) + residual  (MLP sublayer)

        # We'll hook on the full layer (input/output) and the attention module
        # to capture the intermediate state.

        # Store intermediate states
        intermediate_states = {}

        def make_attn_hook(layer_idx):
            """Hook on attention output to capture the post-attention state."""
            def hook_fn(module, input, output):
                # For LLaMA: output is (attn_output, attn_weights, past_kv)
                # For Pythia: output is (attn_output, ...)
                with torch.no_grad():
                    attn_out = output[0].detach().float()
                    intermediate_states[layer_idx] = attn_out
            return hook_fn

        def make_layer_hook(layer_idx):
            """Hook on full decoder layer to capture input and output."""
            def hook_fn(module, input, output):
                with torch.no_grad():
                    x_pre = input[0].detach().float()  # (B, S, m)
                    x_post = output[0].detach().float() if isinstance(output, tuple) else output.detach().float()

                    B_actual, S, m_actual = x_pre.shape
                    x_pre_flat = x_pre.reshape(-1, m_actual)
                    x_post_flat = x_post.reshape(-1, m_actual)

                    # Full layer update
                    F_full = x_post_flat - x_pre_flat

                    # Get attention output from intermediate state
                    attn_out = intermediate_states.get(layer_idx, None)
                    if attn_out is not None:
                        # x_mid = x_pre + attn_out (attention sublayer residual)
                        # But wait: the actual forward may include normalization
                        # Let's compute x_mid by looking at x_pre + attn_out
                        attn_out_flat = attn_out.reshape(-1, m_actual)

                        # Attention update: F_attn = attn_out (already the residual addition)
                        F_attn = attn_out_flat  # This is what gets added to residual

                        # MLP update: F_mlp = F_full - F_attn
                        F_mlp = F_full - F_attn

                        # Compute R_F for each component
                        norm_sq = (x_pre_flat ** 2).sum(dim=-1)
                        rms = torch.sqrt(norm_sq / m_actual).clamp(min=1e-8).unsqueeze(-1)
                        x_hat = x_pre_flat / rms

                        # Attention R_F
                        dot_attn = torch.sum(x_hat * F_attn, dim=-1)
                        radial_sq_attn = dot_attn ** 2 / m_actual
                        F_attn_norm_sq = (F_attn ** 2).sum(dim=-1)
                        R_F_attn = float((radial_sq_attn.mean() / F_attn_norm_sq.mean()).item()) if F_attn_norm_sq.mean() > 1e-15 else 0
                        sigma_F_sq_attn = float((F_attn_norm_sq / m_actual).mean().item())

                        # MLP R_F (using x_mid as the reference direction for MLP)
                        x_mid = x_pre_flat + F_attn
                        norm_sq_mid = (x_mid ** 2).sum(dim=-1)
                        rms_mid = torch.sqrt(norm_sq_mid / m_actual).clamp(min=1e-8).unsqueeze(-1)
                        x_hat_mid = x_mid / rms_mid

                        dot_mlp = torch.sum(x_hat_mid * F_mlp, dim=-1)
                        radial_sq_mlp = dot_mlp ** 2 / m_actual
                        F_mlp_norm_sq = (F_mlp ** 2).sum(dim=-1)
                        R_F_mlp = float((radial_sq_mlp.mean() / F_mlp_norm_sq.mean()).item()) if F_mlp_norm_sq.mean() > 1e-15 else 0
                        sigma_F_sq_mlp = float((F_mlp_norm_sq / m_actual).mean().item())

                        layer_data[layer_idx]['attn_R_F'].append(R_F_attn)
                        layer_data[layer_idx]['attn_sigma_F_sq'].append(sigma_F_sq_attn)
                        layer_data[layer_idx]['attn_norm_sq'].append(float(norm_sq.mean().item()))
                        layer_data[layer_idx]['mlp_R_F'].append(R_F_mlp)
                        layer_data[layer_idx]['mlp_sigma_F_sq'].append(sigma_F_sq_mlp)
                        layer_data[layer_idx]['mlp_norm_sq'].append(float(norm_sq_mid.mean().item()))

                    # Full layer R_F (same as before)
                    norm_sq = (x_pre_flat ** 2).sum(dim=-1)
                    rms = torch.sqrt(norm_sq / m_actual).clamp(min=1e-8).unsqueeze(-1)
                    x_hat = x_pre_flat / rms

                    dot_full = torch.sum(x_hat * F_full, dim=-1)
                    radial_sq_full = dot_full ** 2 / m_actual
                    F_full_norm_sq = (F_full ** 2).sum(dim=-1)
                    R_F_full = float((radial_sq_full.mean() / F_full_norm_sq.mean()).item()) if F_full_norm_sq.mean() > 1e-15 else 0
                    sigma_F_sq_full = float((F_full_norm_sq / m_actual).mean().item())

                    layer_data[layer_idx]['full_R_F'].append(R_F_full)
                    layer_data[layer_idx]['full_sigma_F_sq'].append(sigma_F_sq_full)
                    layer_data[layer_idx]['full_norm_sq'].append(float(norm_sq.mean().item()))

            return hook_fn

        # Register hooks
        if cfg['type'] == 'llama':
            decoder_layers = model.model.layers
            for i, layer in enumerate(decoder_layers):
                h1 = layer.self_attn.register_forward_hook(make_attn_hook(i))
                h2 = layer.register_forward_hook(make_layer_hook(i))
                hooks.extend([h1, h2])
        elif cfg['type'] == 'pythia':
            decoder_layers = model.gpt_neox.layers
            for i, layer in enumerate(decoder_layers):
                h1 = layer.attention.register_forward_hook(make_attn_hook(i))
                h2 = layer.register_forward_hook(make_layer_hook(i))
                hooks.extend([h1, h2])

        # Run forward passes
        print(f"    Running {n_batches} forward passes...")
        for batch_idx in range(n_batches):
            input_ids = torch.randint(100, 30000, (batch_size, seq_len))
            intermediate_states.clear()
            with torch.no_grad():
                model(input_ids)
            if (batch_idx + 1) % 5 == 0:
                print(f"      Batch {batch_idx + 1}/{n_batches}")

        for h in hooks:
            h.remove()

        # Analyze
        print(f"\n    Per-layer R_F comparison:")
        print(f"    {'Layer':>5} {'R_F attn':>10} {'R_F mlp':>10} {'R_F full':>10} "
              f"{'sig_attn':>10} {'sig_mlp':>10} {'sig_full':>10} {'attn/mlp':>8}")
        print(f"    {'-'*80}")

        layer_results = []
        for i in range(num_layers):
            d = layer_data[i]
            if not d['attn_R_F']:
                continue

            rf_attn = float(np.mean(d['attn_R_F']))
            rf_mlp = float(np.mean(d['mlp_R_F']))
            rf_full = float(np.mean(d['full_R_F']))
            sf_attn = float(np.mean(d['attn_sigma_F_sq']))
            sf_mlp = float(np.mean(d['mlp_sigma_F_sq']))
            sf_full = float(np.mean(d['full_sigma_F_sq']))
            ratio = sf_attn / sf_mlp if sf_mlp > 1e-15 else float('inf')

            print(f"    {i:>5} {rf_attn:>10.6f} {rf_mlp:>10.6f} {rf_full:>10.6f} "
                  f"{sf_attn:>10.6f} {sf_mlp:>10.6f} {sf_full:>10.6f} {ratio:>8.3f}")

            layer_results.append({
                'layer': i,
                'R_F_attn': rf_attn,
                'R_F_mlp': rf_mlp,
                'R_F_full': rf_full,
                'sigma_F_sq_attn': sf_attn,
                'sigma_F_sq_mlp': sf_mlp,
                'sigma_F_sq_full': sf_full,
                'energy_ratio_attn_mlp': ratio,
            })

        # Summary
        if layer_results:
            rf_attns = [r['R_F_attn'] for r in layer_results]
            rf_mlps = [r['R_F_mlp'] for r in layer_results]
            rf_fulls = [r['R_F_full'] for r in layer_results]
            print(f"\n    Summary:")
            print(f"      Mean R_F(attn): {np.mean(rf_attns):.6f}")
            print(f"      Mean R_F(mlp):  {np.mean(rf_mlps):.6f}")
            print(f"      Mean R_F(full): {np.mean(rf_fulls):.6f}")
            print(f"      R_F(attn) > R_F(mlp) in {sum(a>m for a,m in zip(rf_attns, rf_mlps))}/{num_layers} layers")

        all_results[model_key] = {
            'model': cfg['name'],
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'layers': layer_results,
        }

        del model, tokenizer
        import gc
        gc.collect()

    # Save
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'tc2_attention_vs_mlp.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {filepath}")

    return all_results


if __name__ == '__main__':
    run_tc2()
