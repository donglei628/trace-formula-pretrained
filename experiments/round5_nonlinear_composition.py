"""
Round 5 Experiment: Nonlinear Composition Test
================================================

Key question: Where does R_F >> 1/m come from in trained networks?

For each trained MLP layer, compare R_F of:
1. Full nonlinear: F(x) = W_down * phi(W_up * x)       [trained W_up, W_down]
2. Linearized:     F(x) = W_down * W_up * x              [same weights, no activation]
3. Shuffled W_down: F(x) = W_down' * phi(W_up * x)      [random W_down, trained W_up]
4. Shuffled W_up:  F(x) = W_down * phi(W_up' * x)       [trained W_down, random W_up]

If R_F(full) >> R_F(linear) >> R_F(shuffled): activation function creates R_F
If R_F(full) ≈ R_F(linear) >> R_F(shuffled): the W_up-W_down pairing creates R_F linearly
If R_F(full) >> R_F(linear) ≈ R_F(shuffled): it's the nonlinear interaction specifically

Usage:
    python round5_nonlinear_composition.py
"""

import sys
import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as torchF

sys.path.insert(0, os.path.dirname(__file__))
from e1_common import radial_fraction, make_x_hat_torch, ensure_output_dir


def run_nonlinear_test():
    print("=" * 70)
    print("NONLINEAR COMPOSITION TEST")
    print("Where does R_F >> 1/m come from?")
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

    B = 2048  # batch size for measurement
    n_seeds = 10

    all_results = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_key}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            cfg['name'], trust_remote_code=True,
            torch_dtype=torch.float32, device_map='cpu',
        )
        model.eval()

        config = model.config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        m = hidden_size

        print(f"    hidden_size={m}, num_layers={num_layers}")
        print(f"    1/m = {1/m:.6f}")

        if cfg['type'] == 'swiglu':
            decoder_layers = model.model.layers
        else:
            decoder_layers = model.gpt_neox.layers

        layer_results = []

        for layer_idx in range(num_layers):
            layer = decoder_layers[layer_idx]
            mlp = layer.mlp

            if cfg['type'] == 'swiglu':
                # SwiGLU: F(x) = W_down * (silu(W_gate * x) * (W_up * x))
                W_gate = mlp.gate_proj.weight.detach().float()  # (intermediate, hidden)
                W_up = mlp.up_proj.weight.detach().float()
                W_down = mlp.down_proj.weight.detach().float()  # (hidden, intermediate)

                def make_swiglu(wg, wu, wd):
                    def f(x):
                        return (wd @ (torchF.silu(wg @ x.T) * (wu @ x.T))).T
                    return f

                F_full = make_swiglu(W_gate, W_up, W_down)

                # Linearized: F(x) = W_down * (W_gate * x * W_up * x) ≈ not well-defined
                # For SwiGLU linearization, use identity activation:
                # F_lin(x) = W_down * (W_gate * x * W_up * x)  but this is quadratic
                # Better: F_lin(x) = W_down * W_up * x (ignore gate)
                def make_linear_no_gate(wd, wu):
                    def f(x):
                        return (wd @ (wu @ x.T)).T
                    return f

                F_linear = make_linear_no_gate(W_down, W_up)

                # Also test with gate but identity activation: F(x) = W_down * ((W_gate * x) * (W_up * x))
                def make_identity_gate(wg, wu, wd):
                    def f(x):
                        return (wd @ ((wg @ x.T) * (wu @ x.T))).T
                    return f

                F_identity_gate = make_identity_gate(W_gate, W_up, W_down)

            else:
                # Standard MLP: F(x) = W2 * gelu(W1 * x)
                W1 = mlp.dense_h_to_4h.weight.detach().float()  # (4*hidden, hidden)
                W2 = mlp.dense_4h_to_h.weight.detach().float()  # (hidden, 4*hidden)

                def make_mlp(w1, w2, act='gelu'):
                    def f(x):
                        h = w1 @ x.T
                        if act == 'gelu':
                            h = torchF.gelu(h)
                        elif act == 'relu':
                            h = torchF.relu(h)
                        elif act == 'identity':
                            pass
                        return (w2 @ h).T
                    return f

                F_full = make_mlp(W1, W2, 'gelu')
                F_linear = make_mlp(W1, W2, 'identity')

            # Measure R_F for each variant
            RF_values = {
                'full': [],
                'linear': [],
            }

            if cfg['type'] == 'swiglu':
                RF_values['identity_gate'] = []

            for seed in range(n_seeds):
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
                with torch.no_grad():
                    # Full nonlinear
                    F_out = F_full(x_hat)
                    RF_values['full'].append(radial_fraction(x_hat, F_out))

                    # Linearized
                    F_out_lin = F_linear(x_hat)
                    RF_values['linear'].append(radial_fraction(x_hat, F_out_lin))

                    if cfg['type'] == 'swiglu':
                        F_out_ig = F_identity_gate(x_hat)
                        RF_values['identity_gate'].append(radial_fraction(x_hat, F_out_ig))

            # Shuffled W2: random W2 with trained W1
            RF_shuffled_w2 = []
            for seed in range(n_seeds):
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
                torch.manual_seed(seed + 50000)
                if cfg['type'] == 'swiglu':
                    intermediate = W_down.shape[1]
                    W_down_rand = torch.randn_like(W_down) / math.sqrt(intermediate)
                    F_rand = make_swiglu(W_gate, W_up, W_down_rand)
                else:
                    intermediate = W2.shape[1]
                    W2_rand = torch.randn_like(W2) / math.sqrt(intermediate)
                    F_rand = make_mlp(W1, W2_rand, 'gelu')

                with torch.no_grad():
                    F_out = F_rand(x_hat)
                    RF_shuffled_w2.append(radial_fraction(x_hat, F_out))
            RF_values['shuffled_w2'] = RF_shuffled_w2

            # Shuffled W1: trained W2 with random W1
            RF_shuffled_w1 = []
            for seed in range(n_seeds):
                x_hat = make_x_hat_torch(B, m, 'gaussian', seed=seed)
                torch.manual_seed(seed + 60000)
                if cfg['type'] == 'swiglu':
                    W_up_rand = torch.randn_like(W_up) / math.sqrt(m)
                    W_gate_rand = torch.randn_like(W_gate) / math.sqrt(m)
                    F_rand = make_swiglu(W_gate_rand, W_up_rand, W_down)
                else:
                    W1_rand = torch.randn_like(W1) / math.sqrt(m)
                    F_rand = make_mlp(W1_rand, W2, 'gelu')

                with torch.no_grad():
                    F_out = F_rand(x_hat)
                    RF_shuffled_w1.append(radial_fraction(x_hat, F_out))
            RF_values['shuffled_w1'] = RF_shuffled_w1

            # Compute means
            result = {'layer': layer_idx}
            for key, vals in RF_values.items():
                arr = np.array(vals)
                result[f'R_F_{key}'] = float(arr.mean())
                result[f'R_F_{key}_std'] = float(arr.std())

            layer_results.append(result)

            # Print every few layers
            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                parts = [f"L{layer_idx:>2}"]
                for key in ['full', 'linear', 'shuffled_w2', 'shuffled_w1']:
                    if f'R_F_{key}' in result:
                        parts.append(f"{key}={result[f'R_F_{key}']:.6f}")
                if cfg['type'] == 'swiglu' and 'R_F_identity_gate' in result:
                    parts.append(f"id_gate={result['R_F_identity_gate']:.6f}")
                print(f"    {'  '.join(parts)}")

        # Summary table
        print(f"\n    Full comparison table:")
        header = f"    {'Layer':>5} {'R_F full':>10} {'R_F linear':>12} {'R_F shuf_w2':>12} {'R_F shuf_w1':>12}"
        if cfg['type'] == 'swiglu':
            header += f" {'R_F id_gate':>12}"
        header += f" {'full/linear':>12} {'1/m':>8}"
        print(header)
        print(f"    {'-'*(len(header)-4)}")

        for r in layer_results:
            line = f"    {r['layer']:>5} {r['R_F_full']:>10.6f} {r['R_F_linear']:>12.6f} " \
                   f"{r['R_F_shuffled_w2']:>12.6f} {r['R_F_shuffled_w1']:>12.6f}"
            if cfg['type'] == 'swiglu' and 'R_F_identity_gate' in r:
                line += f" {r['R_F_identity_gate']:>12.6f}"
            ratio = r['R_F_full'] / r['R_F_linear'] if r['R_F_linear'] > 1e-10 else float('inf')
            line += f" {ratio:>12.3f} {1/m:>8.6f}"
            print(line)

        # Averages
        print(f"\n    Layer-averaged R_F:")
        for key in ['full', 'linear', 'shuffled_w2', 'shuffled_w1']:
            vals = [r[f'R_F_{key}'] for r in layer_results]
            print(f"      {key:>15}: mean={np.mean(vals):.6f}, median={np.median(vals):.6f}, "
                  f"m*R_F={np.mean(vals)*m:.2f}")
        if cfg['type'] == 'swiglu':
            vals = [r['R_F_identity_gate'] for r in layer_results]
            print(f"      {'identity_gate':>15}: mean={np.mean(vals):.6f}, median={np.median(vals):.6f}, "
                  f"m*R_F={np.mean(vals)*m:.2f}")

        all_results[model_key] = {
            'model': cfg['name'],
            'type': cfg['type'],
            'hidden_size': m,
            'num_layers': num_layers,
            'layers': layer_results,
        }

        del model
        import gc; gc.collect()

    # Save
    outdir = ensure_output_dir('results')
    filepath = os.path.join(outdir, 'nonlinear_composition.json')
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {filepath}")

    return all_results


if __name__ == '__main__':
    run_nonlinear_test()
