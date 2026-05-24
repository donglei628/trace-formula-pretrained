# A Trace Formula for Pre-Norm Transformers: Critical Depth and Quantization Geometry

**Dong Lei** (Independent Researcher, 2026)

Companion code and pre-computed results for reproducing all experiments, tables, and figures in the paper.

## Key Results

| Result | Formula / Value | Verification |
|--------|----------------|--------------|
| **Trace formula** (Thm 4.1) | $\mathcal{R}_F^{\text{linear}} = \frac{\text{tr}(M)^2 + \text{tr}(M^2) + \|M\|_F^2}{(m+2)\|M\|_F^2}$ | 6 architectures, 126 layers, ratio 0.997–1.000 |
| **Critical depth** (Thm 5.5) | $L^* = (e^2-1)/\sigma_F^2 \approx 6.39/\sigma_F^2$ | 16 toy-transformer configs, 1–5% error |
| **ReLU depth boost** (Cor 5.2) | $(e^3-1)/(e^2-1) \approx 2.988$ | Exact (no free parameters) |
| **Quantization robustness** (Finding 8.1) | Spearman $\rho \approx -0.98$ | 12 conditions (4 activations × 3 targets) |
| **Activation suppression** (Prop 6.4) | $\mathcal{R}_F^{\text{full}} \approx \alpha_\phi^{(\text{supp})} \cdot \mathcal{R}_F^{\text{linear}}$ | 4/5 architectures, Pearson $r > 0.95$ |

## Requirements

- Python 3.10+
- GPU with ≥8 GB VRAM (for pretrained model experiments)
- CPU-only experiments: `e1_toy_experiment.py`, `e1_extended.py`, `e1_5_supplement.py`, `verify_theorem5.py`

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── experiments/
│   ├── e1_common.py                      # Core utilities (R_F measurement, input generators)
│   ├── e1_toy_experiment.py              # Critical depth verification (Section 5.4)
│   ├── e1_extended.py                    # Regime A/B unification (Section 5.5)
│   ├── e1_5_supplement.py                # ρ=1 saturation limits (Table 2)
│   ├── exp2_5_extend.py                  # Trace formula on pretrained models (Table 4)
│   ├── round5_trace_formula.py           # Trace formula verification (Table 4)
│   ├── paper2_supplement_experiments.py   # Supplementary experiments (Tables, Fig 2–3 data)
│   ├── paper2_round2_experiments.py       # Attention, α_φ, sign-reversal (Tables 9–11)
│   ├── paper2_round3_experiments.py       # Quantization robustness (Section 8, Tables 7–8)
│   ├── round5_tc2_attention.py           # Attention amplification (Section 7)
│   ├── round5_nonlinear_composition.py    # Activation suppression (Section 6.4, Table 6)
│   ├── verify_theorem5.py                # Activation integral verification (Theorem 5.6)
│   └── results/                          # Pre-computed JSON results (22 files)
├── figures/
│   ├── generate_figures.py               # Regenerate all 4 paper figures from results
│   ├── fig1_sphere_walk.pdf
│   ├── fig2_critical_depth.pdf
│   ├── fig3_trace_formula.pdf
│   └── fig4_quantization.pdf
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Quick Reproduction

### Regenerate all figures from pre-computed results (no GPU needed)

```bash
python figures/generate_figures.py
```

### Core experiments (CPU, ~5 min each)

```bash
# Trace formula: Regime A (i.i.d.) and Regime B (controlled correlation)
python experiments/e1_extended.py all

# Critical depth L* = 6.39/σ_F² verification
python experiments/e1_toy_experiment.py

# Activation integrals: α_ReLU = α_GELU = α_SiLU = 1/2
python experiments/verify_theorem5.py

# ρ=1 saturation limits for all activations
python experiments/e1_5_supplement.py
```

### Pretrained model experiments (GPU, ~10–30 min each)

```bash
# Trace formula on TinyLlama + Pythia-1B (Table 4)
python experiments/exp2_5_extend.py

# Trace formula on GPT-2 + Qwen2.5-0.5B (Table 4)
python experiments/paper2_supplement_experiments.py 2.5

# Activation suppression: full vs linear R_F (Table 6)
python experiments/round5_nonlinear_composition.py

# Attention amplification (Table 9)
python experiments/round5_tc2_attention.py

# Quantization robustness controlled experiment (Tables 7–8, Fig 4)
python experiments/paper2_round3_experiments.py r3.1

# Attention input dependence (Finding 7.2)
python experiments/paper2_round3_experiments.py r3.2

# Sign-reversal profiles, broader models (Tables 10–11)
python experiments/paper2_round2_experiments.py all
```

## Experiment → Paper Mapping

| Paper Section | Table/Figure | Script | Result File |
|--------------|-------------|--------|-------------|
| §5.3 Regime A | Table 1 | `e1_extended.py e1_1`, `e1_2` | `e1_1_width_scan.json`, `e1_2_activation_scan.json` |
| §5.5 Regime B | Tables 2–3 | `e1_extended.py e1_5`, `e1_5_supplement.py` | `e1_5_weight_correlation.json`, `e1_5_supplement.json` |
| §5.4 Critical depth | Table 5, Fig 2 | `e1_toy_experiment.py` | `e1_results.json` |
| §6.1 Trace formula | Table 4, Fig 3 | `exp2_5_extend.py`, `paper2_supplement_experiments.py` | `exp2_5_tinyllama_pythia.json`, `exp2_5_more_architectures.json` |
| §6.4 Suppression | Table 6 | `round5_nonlinear_composition.py` | `nonlinear_composition.json` |
| §7 Attention | Table 9 | `round5_tc2_attention.py`, `paper2_round2_experiments.py` | `tc2_attention_vs_mlp.json`, `r2_2_attention_4models.json` |
| §8 Quantization | Tables 7–8, Fig 4 | `paper2_round3_experiments.py r3.1` | `r3_1_controlled_quantization.json` |
| §9.2 Sign-reversal | Table 10 | `paper2_round2_experiments.py` | `r2_4_bell_profiles.json` |
| §5.6 Activation integrals | Theorem 5.6 | `verify_theorem5.py` | `e5_3_theorem5_validation.json` |
| Fig 1 | Sphere walk | `figures/generate_figures.py` | `e5_1_rho_profiles.json`, `t2_step_variance.json` |
| Fig 2 | Critical depth | `figures/generate_figures.py` | `e1_results.json`, `exp3_2_width_convergence.json` |
| Fig 3 | Trace formula | `figures/generate_figures.py` | `exp2_5_*.json`, `exp2_4_trace_decomposition.json` |
| Fig 4 | Quantization | `figures/generate_figures.py` | `r3_1_controlled_quantization.json` |

## Pre-computed Results

All 22 JSON files in `experiments/results/` contain the raw data used to produce the paper's tables and figures. To verify paper numbers without re-running experiments:

```bash
python figures/generate_figures.py  # regenerates all 4 figures from JSON data
```

## Pretrained Models Used

| Model | HuggingFace ID | License |
|-------|---------------|---------|
| TinyLlama-1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Apache 2.0 |
| Pythia-1B | `EleutherAI/pythia-1b` | Apache 2.0 |
| Pythia-410M | `EleutherAI/pythia-410m` | Apache 2.0 |
| GPT-2 (124M) | `gpt2` | MIT |
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | Apache 2.0 |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | Apache 2.0 |

## Hardware

All experiments were conducted on a single NVIDIA RTX 4070 Laptop GPU (8 GB VRAM). CPU-only experiments run on any modern machine.

## Citation

```bibtex
@article{lei2026trace,
  title={A Trace Formula for Pre-Norm Transformers: Critical Depth and Quantization Geometry},
  author={Lei, Dong},
  year={2026},
  note={arXiv preprint}
}
```

## License

MIT
