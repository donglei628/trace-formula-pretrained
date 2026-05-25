"""
Generate publication-quality figures for Paper 2:
"A Trace Formula for Pre-Norm Transformers"

Figures saved as PDF to paper2/figures/
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Global style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})

BASE = Path(__file__).resolve().parent.parent  # project root
RESULTS = BASE / "experiments" / "results"
FIGDIR = BASE / "figures"
FIGDIR.mkdir(exist_ok=True)

# Color palette for models
COLORS = {
    'TinyLlama': '#1f77b4',
    'Pythia-1B': '#ff7f0e',
    'GPT-2': '#2ca02c',
    'Qwen2.5-0.5B': '#d62728',
    'Qwen2.5-1.5B': '#9467bd',
}

MARKERS = {
    'TinyLlama': 'o',
    'Pythia-1B': 's',
    'GPT-2': '^',
    'Qwen2.5-0.5B': 'D',
    'Qwen2.5-1.5B': 'v',
}


def load_json(name):
    with open(RESULTS / name, 'r') as f:
        return json.load(f)


# ======================================================================
# FIGURE 1: Sphere walk schematic
# ======================================================================
def generate_figure1():
    """
    Left: Schematic of radial-tangential decomposition on sphere
    Center: Weight correlation rho_up_down vs depth for 3 models
    Right: Per-layer step variance sigma_F^2 vs depth
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # --- Left panel: Sphere schematic ---
    ax = axes[0]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(r'(a) Radial-tangential decomposition', fontsize=10, pad=10)

    # Draw sphere (circle)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5)

    # Point on sphere
    angle_pt = np.pi/4
    px, py = np.cos(angle_pt), np.sin(angle_pt)
    ax.plot(px, py, 'ko', markersize=8, zorder=5)
    ax.annotate(r'$\hat{x}_\ell$', (px+0.08, py+0.08), fontsize=12, fontweight='bold')

    # Radial direction (from center through point, outward)
    rad_len = 0.7
    rx, ry = rad_len * np.cos(angle_pt), rad_len * np.sin(angle_pt)
    ax.annotate('', xy=(px + rx, py + ry), xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5))
    ax.annotate(r'$\mathbf{r}$ (radial)', (px + rx + 0.05, py + ry - 0.15),
                fontsize=9, color='#d62728')

    # Tangential direction (perpendicular to radial)
    tang_angle = angle_pt + np.pi/2
    tang_len = 0.7
    tx, ty = tang_len * np.cos(tang_angle), tang_len * np.sin(tang_angle)
    ax.annotate('', xy=(px + tx, py + ty), xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2.5))
    ax.annotate(r'$\mathbf{t}$ (tangent)', (px + tx - 0.55, py + ty + 0.05),
                fontsize=9, color='#1f77b4')

    # Update vector F(x)
    f_angle = angle_pt + np.pi/6  # slightly off-radial
    f_len = 0.55
    fx, fy = f_len * np.cos(f_angle), f_len * np.sin(f_angle)
    ax.annotate('', xy=(px + fx, py + fy), xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2.5))
    ax.annotate(r'$F_\ell(\hat{x}_\ell)$', (px + fx + 0.05, py + fy + 0.05),
                fontsize=10, color='#2ca02c')

    # Dashed projections
    # Radial projection of F
    proj_r = (fx * np.cos(angle_pt) + fy * np.sin(angle_pt))
    prx = proj_r * np.cos(angle_pt)
    pry = proj_r * np.sin(angle_pt)
    ax.plot([px + fx, px + prx], [py + fy, py + pry], 'k--', alpha=0.4, lw=1)
    ax.plot([px + fx, px + fx - prx], [py + fy, py + fy - pry + (pry - fy + pry)],
            'k--', alpha=0.0)  # placeholder

    # R_F annotation
    ax.annotate(r'$\mathcal{R}_F = \frac{\|F^{\mathrm{rad}}\|^2}{\|F\|^2}$',
                xy=(0.0, -1.3), fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

    # Origin
    ax.plot(0, 0, 'k+', markersize=8, markeredgewidth=1.5)

    # --- Center panel: Weight correlation profiles ---
    ax = axes[1]
    rho_data = load_json('e5_1_rho_profiles.json')

    for key, label, color in [('tinyllama', 'TinyLlama', COLORS['TinyLlama']),
                               ('pythia-1b', 'Pythia-1B', COLORS['Pythia-1B']),
                               ('qwen2.5-1.5b', 'Qwen2.5-1.5B', COLORS['Qwen2.5-1.5B'])]:
        model = rho_data[key]
        layers_list = model['layers']
        depths = [l['layer'] for l in layers_list]
        if key == 'pythia-1b':
            rhos = [l.get('rho_w1_w2', 0) for l in layers_list]
        else:
            rhos = [l['rho_up_down'] for l in layers_list]
        ax.plot(depths, rhos, 'o-', color=color, label=label, markersize=3.5, linewidth=1.2)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Layer depth $\\ell$')
    ax.set_ylabel(r'Weight correlation $\rho_{W_1,W_2}$')
    ax.set_title(r'(b) Weight correlation vs depth', fontsize=10, pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # --- Right panel: Step variance ---
    ax = axes[2]
    step_data = load_json('t2_step_variance.json')

    for key, label, color in [('tinyllama', 'TinyLlama', COLORS['TinyLlama']),
                               ('pythia-1b', 'Pythia-1B', COLORS['Pythia-1B'])]:
        model = step_data[key]
        layers_list = model['layers']
        # Skip layer 0 for TinyLlama (has extreme value)
        if key == 'tinyllama':
            layers_list = [l for l in layers_list if l['layer'] >= 1]
        depths = [l['layer'] for l in layers_list]
        sigma_sq = [l['sigma_F_sq'] for l in layers_list]
        ax.semilogy(depths, sigma_sq, 'o-', color=color, label=label, markersize=3.5, linewidth=1.2)

    ax.set_xlabel('Layer depth $\\ell$')
    ax.set_ylabel(r'Step variance $\sigma_F^2$')
    ax.set_title(r'(c) Per-layer step variance', fontsize=10, pad=10)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIGDIR / 'fig1_sphere_walk.pdf', format='pdf')
    plt.close(fig)
    print("Figure 1 saved: fig1_sphere_walk.pdf")


# ======================================================================
# FIGURE 2: Critical depth validation
# ======================================================================
def generate_figure2():
    """
    Left: L* vs 1/sigma_F^2 with theoretical prediction line
    Right: Width convergence table visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left panel: L* vs 1/sigma^2 ---
    ax = axes[0]
    lstar_data = load_json('e5_4_Lstar_predictions.json')

    inv_sigmas = []
    label_map = {'tinyllama': 'TinyLlama', 'pythia-1b': 'Pythia-1B', 'qwen2.5-1.5b': 'Qwen2.5-1.5B'}
    for key in ['tinyllama', 'pythia-1b', 'qwen2.5-1.5b']:
        m = lstar_data[key]
        inv_sigma = 1.0 / m['avg_sigma_F_sq']
        inv_sigmas.append(inv_sigma)
        color = COLORS[label_map[key]]
        marker = MARKERS[label_map[key]]
        ax.scatter(inv_sigma, m['L_star'], color=color, marker=marker, s=100,
                   zorder=5, label=f"{label_map[key]} (L*={m['L_star']:.1f})", edgecolors='black', linewidth=0.8)

    # Theoretical line: L* = 6.389 / sigma^2 = 6.389 * (1/sigma^2)
    x_max = max(inv_sigmas) * 1.1
    x_theory = np.linspace(0, x_max, 100)
    e2_minus_1 = np.e**2 - 1  # 6.389
    ax.plot(x_theory, e2_minus_1 * x_theory, 'k--', linewidth=1.5,
            label=f'$L^* = (e^2-1)/\\sigma_F^2$ (slope={e2_minus_1:.2f})')

    ax.set_xlabel(r'$1/\sigma_F^2$')
    ax.set_ylabel(r'Critical depth $L^*$')
    ax.set_title(r'(a) Critical depth vs inverse step variance', fontsize=10, pad=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # --- Right panel: Width convergence ---
    ax = axes[1]
    width_data = load_json('exp3_2_width_convergence.json')

    widths = []
    err_m = []
    err_m2 = []
    for entry in width_data['widths']:
        widths.append(entry['width'])
        err_m.append(entry['mean_rel_error_m'] * 100)
        err_m2.append(entry['mean_rel_error_m_plus_2'] * 100)

    x_pos = np.arange(len(widths))
    bar_width = 0.35
    bars1 = ax.bar(x_pos - bar_width/2, err_m, bar_width, label=r'Denominator $m$',
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x_pos + bar_width/2, err_m2, bar_width, label=r'Denominator $(m+2)$',
                   color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Width $m$')
    ax.set_ylabel('Mean relative error (%)')
    ax.set_title(r'(b) Width convergence: $m$ vs $(m+2)$ denominator', fontsize=10, pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(w) for w in widths], fontsize=8)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate improvement at small width
    if len(widths) > 0:
        improvement = err_m[0] / err_m2[0] if err_m2[0] > 0 else 0
        ax.annotate(f'{improvement:.1f}x better',
                    xy=(0, err_m[0]), xytext=(1.2, err_m[0] * 0.7),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=8, color='gray')

    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIGDIR / 'fig2_critical_depth.pdf', format='pdf')
    plt.close(fig)
    print("Figure 2 saved: fig2_critical_depth.pdf")


# ======================================================================
# FIGURE 3: Trace formula validation (KEY FIGURE)
# ======================================================================
def generate_figure3():
    """
    Left: Predicted vs measured R_F scatter for all layers across architectures
    Right: Trace-term fraction per layer for TinyLlama and Pythia
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Left panel: Predicted vs Measured scatter ---
    ax = axes[0]

    # Collect all (predicted, measured) pairs
    all_pred = []
    all_meas = []

    # GPT-2 from exp2_5
    exp25 = load_json('exp2_5_more_architectures.json')
    for model_key, label in [('gpt2', 'GPT-2'), ('qwen2.5-0.5b', 'Qwen2.5-0.5B')]:
        if model_key in exp25:
            layers_list = exp25[model_key]['layers']
            pred = [l['R_F_formula'] for l in layers_list]
            meas = [l['R_F_linear_empirical'] for l in layers_list]
            color = COLORS[label]
            marker = MARKERS[label]
            ax.scatter(pred, meas, c=color, marker=marker, s=30, alpha=0.8,
                       label=f'{label} ({len(pred)} layers)', edgecolors='black', linewidth=0.3)
            all_pred.extend(pred)
            all_meas.extend(meas)

    # TinyLlama + Pythia-1B from exp2_5_extend (real per-layer empirical data)
    exp25_ext = load_json('exp2_5_tinyllama_pythia.json')
    decomp = load_json('exp2_4_trace_decomposition.json')  # still needed for right panel
    for model_key, label in [('tinyllama', 'TinyLlama'), ('pythia-1b', 'Pythia-1B')]:
        if model_key in exp25_ext:
            layers_list = exp25_ext[model_key]['layers']
            pred = [l['R_F_formula'] for l in layers_list]
            meas = [l['R_F_linear_empirical'] for l in layers_list]
            color = COLORS[label]
            marker = MARKERS[label]
            ax.scatter(pred, meas, c=color, marker=marker, s=30, alpha=0.8,
                       label=f'{label} ({len(pred)} layers)', edgecolors='black', linewidth=0.3)
            all_pred.extend(pred)
            all_meas.extend(meas)

    # Perfect agreement line
    min_val = min(min(all_pred), min(all_meas)) * 0.8
    max_val = max(max(all_pred), max(all_meas)) * 1.1
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.2, alpha=0.6, label='$y = x$')

    # Compute overall R^2
    pred_arr = np.array(all_pred)
    meas_arr = np.array(all_meas)
    ss_res = np.sum((meas_arr - pred_arr)**2)
    ss_tot = np.sum((meas_arr - np.mean(meas_arr))**2)
    r_squared = 1 - ss_res / ss_tot
    total_layers = len(all_pred)

    ax.set_xlabel(r'$\mathcal{R}_F^{\mathrm{predicted}}$ (trace formula)')
    ax.set_ylabel(r'$\mathcal{R}_F^{\mathrm{measured}}$ (empirical)')
    ax.set_title(f'(a) Trace formula validation ({total_layers} layers, ratio 0.997\u20131.000)', fontsize=10, pad=10)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Inset: zoom on low-R_F region
    axins = ax.inset_axes([0.55, 0.08, 0.4, 0.4])
    for model_key, label in [('gpt2', 'GPT-2'), ('qwen2.5-0.5b', 'Qwen2.5-0.5B')]:
        if model_key in exp25:
            layers_list = exp25[model_key]['layers']
            pred = [l['R_F_formula'] for l in layers_list]
            meas = [l['R_F_linear_empirical'] for l in layers_list]
            color = COLORS[label]
            marker = MARKERS[label]
            low_pred = [p for p, m in zip(pred, meas) if p < 0.06]
            low_meas = [m for p, m in zip(pred, meas) if p < 0.06]
            if low_pred:
                axins.scatter(low_pred, low_meas, c=color, marker=marker, s=15, alpha=0.8,
                              edgecolors='black', linewidth=0.2)

    for model_key, label in [('tinyllama', 'TinyLlama'), ('pythia-1b', 'Pythia-1B')]:
        if model_key in exp25_ext:
            layers_list = exp25_ext[model_key]['layers']
            pred = [l['R_F_formula'] for l in layers_list]
            meas = [l['R_F_linear_empirical'] for l in layers_list]
            color = COLORS[label]
            marker = MARKERS[label]
            low_pred = [p for p, m in zip(pred, meas) if p < 0.06]
            low_meas = [m for p, m in zip(pred, meas) if p < 0.06]
            if low_pred:
                axins.scatter(low_pred, low_meas, c=color, marker=marker, s=15, alpha=0.8,
                              edgecolors='black', linewidth=0.2)

    axins.plot([0, 0.06], [0, 0.06], 'k--', linewidth=0.8, alpha=0.5)
    axins.set_xlim(0, 0.06)
    axins.set_ylim(0, 0.06)
    axins.set_xlabel(r'$\mathcal{R}_F^{\mathrm{pred}}$', fontsize=7)
    axins.set_ylabel(r'$\mathcal{R}_F^{\mathrm{meas}}$', fontsize=7)
    axins.tick_params(labelsize=6)
    axins.grid(True, alpha=0.2)
    ax.indicate_inset_zoom(axins, edgecolor='gray', alpha=0.5)

    # --- Right panel: Trace-term fraction ---
    ax = axes[1]

    for model_key, label, color in [('tinyllama', 'TinyLlama', COLORS['TinyLlama']),
                                     ('pythia-1b', 'Pythia-1B', COLORS['Pythia-1B'])]:
        layers_list = decomp[model_key]['layers']
        depths = [l['layer'] for l in layers_list]
        pct_trace = [l['pct_trace'] for l in layers_list]
        pct_baseline = [l['pct_baseline'] for l in layers_list]
        pct_quad = [l['pct_quadratic'] for l in layers_list]

        ax.plot(depths, pct_trace, 'o-', color=color, markersize=4, linewidth=1.3,
                label=f'{label}: $\\mathrm{{tr}}(M)^2$ term')

    # Add horizontal reference lines
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    # Annotate summary statistics
    tl_mean = decomp['tinyllama']['summary']['pct_trace_mean']
    py_mean = decomp['pythia-1b']['summary']['pct_trace_mean']
    ax.annotate(f'TinyLlama mean: {tl_mean:.1f}%',
                xy=(15, tl_mean), fontsize=8, color=COLORS['TinyLlama'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['TinyLlama'], alpha=0.8))
    ax.annotate(f'Pythia mean: {py_mean:.1f}%',
                xy=(10, py_mean - 5), fontsize=8, color=COLORS['Pythia-1B'],
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['Pythia-1B'], alpha=0.8))

    ax.set_xlabel('Layer depth $\\ell$')
    ax.set_ylabel(r'$\mathrm{tr}(M)^2$ term fraction (%)')
    ax.set_title(r'(b) Trace term dominance by layer', fontsize=10, pad=10)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(FIGDIR / 'fig3_trace_formula.pdf', format='pdf')
    plt.close(fig)
    print("Figure 3 saved: fig3_trace_formula.pdf")


# ======================================================================
# FIGURE 4: Quantization robustness
# ======================================================================
def generate_figure4():
    """
    Left: R_F vs quantization distortion for controlled experiment (4 activations)
    Right: Control variable stability
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    r31 = load_json('r3_1_controlled_quantization.json')
    raw = r31['raw_data']

    act_colors = {
        'identity': '#1f77b4',
        'relu': '#ff7f0e',
        'gelu': '#2ca02c',
        'silu': '#d62728',
    }

    act_labels = {
        'identity': 'Identity',
        'relu': 'ReLU',
        'gelu': 'GELU',
        'silu': 'SiLU',
    }

    # --- Left panel: R_F vs distortion (W2 quantization) ---
    ax = axes[0]

    spearman_vals = r31['per_activation_W2_spearman']
    mean_spearman = np.mean(list(spearman_vals.values()))

    for act in ['identity', 'relu', 'gelu', 'silu']:
        rf_vals = []
        dist_vals = []
        for entry in raw:
            if entry['activation'] == act:
                rf_vals.append(entry['actual_R_F'])
                dist_vals.append(entry['distortion_W2_only'])

        rho = spearman_vals[act]
        ax.scatter(rf_vals, dist_vals, c=act_colors[act], s=12, alpha=0.5,
                   label=f'{act_labels[act]} ($\\rho_s$={rho:.3f})', edgecolors='none')

    ax.set_xlabel(r'Radial fraction $\mathcal{R}_F$')
    ax.set_ylabel(r'Quantization distortion $\|\delta_q \hat{x}\| / \|\hat{x}\|$')
    ax.set_title(f'(a) $\\mathcal{{R}}_F$ vs distortion (mean Spearman $\\rho$={mean_spearman:.3f})', fontsize=10, pad=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, markerscale=2)
    ax.grid(True, alpha=0.3)

    # --- Right panel: Control variable stability ---
    ax = axes[1]

    ctrl = r31['control_variables']
    rhos = sorted(ctrl.keys(), key=float)
    rf_means = [ctrl[r]['mean_R_F'] for r in rhos]
    w1_means = [ctrl[r]['mean_W1'] for r in rhos]
    w2_means = [ctrl[r]['mean_W2'] for r in rhos]

    ax2 = ax.twinx()

    rho_floats = [float(r) for r in rhos]

    # R_F on primary axis (varies 250×)
    line1, = ax.plot(rho_floats, rf_means, 'o-', color='#d62728',
                     linewidth=2, markersize=5, label=r'$\mathcal{R}_F$ (varies 250$\times$)')

    # W norms on secondary axis (stay constant) — use a wide y-range to show flatness
    line2, = ax2.plot(rho_floats, w1_means, 's--', color='#1f77b4',
                      linewidth=1.5, markersize=4, label=r'$\|W_1\|_F$ (CV<0.1%)')
    line3, = ax2.plot(rho_floats, w2_means, '^--', color='#2ca02c',
                      linewidth=1.5, markersize=4, label=r'$\|W_2\|_F$ (CV<0.2%)')

    ax.set_xlabel(r'Target correlation $\rho$')
    ax.set_ylabel(r'Mean $\mathcal{R}_F$', color='#d62728')
    ax2.set_ylabel(r'Frobenius norm $\|W\|_F$', color='#1f77b4')
    ax.tick_params(axis='y', labelcolor='#d62728')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # Fix secondary y-axis range so norms appear visually flat (they ARE flat, CV<0.2%)
    w_center = (np.mean(w1_means) + np.mean(w2_means)) / 2
    w_spread = max(np.std(w1_means), np.std(w2_means)) * 100  # 100-sigma range
    w_margin = max(w_spread, w_center * 0.2)  # at least ±20% of center value
    ax2.set_ylim(w_center - w_margin, w_center + w_margin)

    # Combined legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center left', fontsize=8, framealpha=0.9)

    ax.set_title('(b) Control variable stability', fontsize=10, pad=10)
    ax.grid(True, alpha=0.3)

    # Annotate the key insight
    ax.annotate(r'$\|W\|_F$ held constant' + '\n' + r'while $\mathcal{R}_F$ varies 250$\times$',
                xy=(0.5, 0.25), xycoords='axes fraction', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
                ha='center')

    fig.tight_layout(w_pad=3)
    fig.savefig(FIGDIR / 'fig4_quantization.pdf', format='pdf')
    plt.close(fig)
    print("Figure 4 saved: fig4_quantization.pdf")


# ======================================================================
# Main
# ======================================================================
if __name__ == '__main__':
    print("Generating figures for Paper 2...")
    print(f"Output directory: {FIGDIR}")
    print()

    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()

    print()
    print("All 4 figures generated successfully!")
