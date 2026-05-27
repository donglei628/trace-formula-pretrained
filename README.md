# Radial Fraction in Pre-Norm Transformers: A Trace Formula with Applications to Critical Depth and Quantization

**Dong Lei** (Independent Researcher, 2026)

Companion code and pre-computed results for reproducing all experiments, tables, and figures in the paper.

## Key Results

| Result | Formula / Value | Verification |
|--------|----------------|--------------|
| **Trace formula** (Thm 6.1) | $\mathcal{R}_F^{\text{linear}} = \frac{\text{tr}(M)^2 + \text{tr}(M^2) + \|M\|_F^2}{(m+2)\|M\|_F^2}$ | 6 architectures, 126 layers, ratio 0.997--1.000 |
| **Critical depth** (Thm 5.1) | $L^* = (e^2-1)/\sigma_F^2 \approx 6.389/\sigma_F^2$ | 12 toy-network configs, 0.2--9.4% relative error |
| **ReLU depth boost** (Cor 5.2) | $(e^3-1)/(e^2-1) \approx 2.988$ | Exact from $1/e$ threshold and sphere geometry |
| **Quantization robustness** (Finding 8.1) | Spearman $\rho \approx -0.98$ | 12 conditions (4 activations x 3 targets) |
| **Activation suppression** (Prop 6.5) | $\mathcal{R}_F^{\text{full}} \approx \alpha_\phi^{(\text{supp})} \cdot \mathcal{R}_F^{\text{linear}}$ | 4/5 architectures, Pearson $r > 0.95$ |

## Requirements

- Python 3.10+
- GPU with >= 8 GB VRAM (for pretrained model experiments)
- CPU-only experiments: `e1_toy_experiment.py`, `e1_extended.py`, `e1_5_supplement.py`, `verify_theorem5.py`

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── experiments/
│   ├── e1_common.py                      # Core utilities (R_F measurement, input generators)
│   ├── e1_toy_experiment.py              # Critical depth verification (Section 5.4)
│   ├── e1_extended.py                    # Regime A/B unification (Section 6.3)
│   ├── e1_5_supplement.py                # rho=1 saturation limits (Table 2)
│   ├── exp2_5_extend.py                  # Trace formula on pretrained models (Table 5)
│   ├── round5_trace_formula.py           # Trace formula verification (Table 5)
│   ├── paper2_supplement_experiments.py   # Supplementary experiments (Tables, Fig 2-3 data)
│   ├── paper2_round2_experiments.py       # Attention, alpha_phi, sign-reversal (Tables 8, 13)
│   ├── paper2_round3_experiments.py       # Quantization robustness (Section 8, Tables 11-12)
│   ├── round5_tc2_attention.py           # Attention amplification (Section 7)
│   ├── round5_nonlinear_composition.py    # Activation suppression (Section 6.6, Table 7)
│   ├── verify_theorem5.py                # Activation integral verification (Proposition 6.3)
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

# Critical depth L* = 6.39/sigma_F^2 verification
python experiments/e1_toy_experiment.py

# Activation integrals: alpha_ReLU = alpha_GELU = alpha_SiLU = 1/2
python experiments/verify_theorem5.py

# rho=1 saturation limits for all activations
python experiments/e1_5_supplement.py
```

### Pretrained model experiments (GPU, ~10-30 min each)

```bash
# Trace formula on TinyLlama + Pythia-1B (Table 5)
python experiments/exp2_5_extend.py

# Trace formula on GPT-2 + Qwen2.5-0.5B (Table 5)
python experiments/paper2_supplement_experiments.py 2.5

# Activation suppression: full vs linear R_F (Table 7)
python experiments/round5_nonlinear_composition.py

# Attention amplification (Table 8)
python experiments/round5_tc2_attention.py

# Quantization robustness controlled experiment (Tables 11-12, Fig 4)
python experiments/paper2_round3_experiments.py r3.1

# Attention input dependence (Finding 7.2)
python experiments/paper2_round3_experiments.py r3.2

# Sign-reversal profiles, broader models (Table 13)
python experiments/paper2_round2_experiments.py all
```

## Experiment-to-Paper Mapping

| Paper Section | Table/Figure | Script | Result File |
|--------------|-------------|--------|-------------|
| Section 5.4 Critical depth | Fig 2(a) | `e1_toy_experiment.py` | `e1_results.json` |
| Section 6.3 Regime A | -- | `e1_extended.py e1_1`, `e1_2` | `e1_1_width_scan.json`, `e1_2_activation_scan.json` |
| Section 6.3 Regime B | Table 3 | `e1_extended.py e1_5`, `e1_5_supplement.py` | `e1_5_weight_correlation.json`, `e1_5_supplement.json` |
| Section 5.2-5.3 Depth boost | Tables 1-2 | `e1_5_supplement.py` | `e1_5_supplement.json` |
| Section 6.1 Trace formula | Table 5, Fig 3 | `exp2_5_extend.py`, `paper2_supplement_experiments.py` | `exp2_5_tinyllama_pythia.json`, `exp2_5_more_architectures.json` |
| Section 6.4 Trace decomposition | Table 4 | `paper2_supplement_experiments.py` | `exp2_4_trace_decomposition.json` |
| Section 6.5 Width convergence | Table 6, Fig 2(b) | `paper2_supplement_experiments.py` | `exp3_2_width_convergence.json` |
| Section 6.6 Suppression | Table 7 | `round5_nonlinear_composition.py` | `nonlinear_composition.json` |
| Section 7.1 Attention | Table 8 | `round5_tc2_attention.py`, `paper2_round2_experiments.py` | `tc2_attention_vs_mlp.json`, `r2_2_attention_4models.json` |
| Section 7.2 Input independence | Table 9, Finding 7.2 | `paper2_round3_experiments.py r3.2` | `r3_2_attention_input_dependence.json` |
| Section 8 Quantization | Tables 11-12, Fig 4 | `paper2_round3_experiments.py r3.1` | `r3_1_controlled_quantization.json` |
| Section 9.1 Sign-reversal | Table 13 | `paper2_round2_experiments.py` | `r2_4_bell_profiles.json`, `round4_broader_models.json` |
| Proposition 6.3 Activation integrals | Table 3 | `verify_theorem5.py` | `e5_3_theorem5_validation.json` |
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
  title={Radial Fraction in Pre-Norm Transformers: A Trace Formula with Applications to Critical Depth and Quantization},
  author={Lei, Dong},
  journal={arXiv preprint arXiv:2605.XXXXX},
  year={2026}
}
```

## License

MIT
