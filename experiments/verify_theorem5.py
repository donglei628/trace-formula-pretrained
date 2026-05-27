"""
Activation Integral Verification (Proposition 6.3): Radial Fraction R_F for various activation functions.

Formula:
    R_F^phi = E[phi(z)*z]^2 / (E[phi(z)^2] * E[z^2])

where z ~ N(0,1), so E[z^2] = 1.

We compute each integral numerically via scipy.integrate.quad and also
via Monte Carlo for cross-validation.
"""

import numpy as np
from scipy import integrate
from scipy.stats import norm
import math

# Standard normal PDF
def phi_pdf(z):
    return norm.pdf(z)

# ============================================================
# Activation functions
# ============================================================

def relu(z):
    return np.maximum(z, 0)

def gelu(z):
    """GELU: z * Phi(z) where Phi is the standard normal CDF."""
    return z * norm.cdf(z)

def silu(z):
    """SiLU / Swish: z * sigmoid(z)"""
    return z / (1.0 + np.exp(-z))

def tanh_act(z):
    return np.tanh(z)

def identity(z):
    return z

# ============================================================
# Numerical integration (quad)
# ============================================================

def compute_RF_quad(activation_fn, name=""):
    """Compute R_F via numerical integration."""

    # E[phi(z) * z]
    def integrand_phiz(z):
        return activation_fn(z) * z * phi_pdf(z)

    # E[phi(z)^2]
    def integrand_phi2(z):
        return activation_fn(z)**2 * phi_pdf(z)

    # E[z^2] = 1 for N(0,1)

    E_phiz, err1 = integrate.quad(integrand_phiz, -np.inf, np.inf)
    E_phi2, err2 = integrate.quad(integrand_phi2, -np.inf, np.inf)
    E_z2 = 1.0

    RF = E_phiz**2 / (E_phi2 * E_z2)

    print(f"\n{'='*50}")
    print(f"Activation: {name}")
    print(f"  E[phi(z)*z]   = {E_phiz:.10f}  (err: {err1:.2e})")
    print(f"  E[phi(z)^2]   = {E_phi2:.10f}  (err: {err2:.2e})")
    print(f"  E[z^2]        = {E_z2:.10f}")
    print(f"  R_F = E[phi*z]^2 / (E[phi^2]*E[z^2]) = {RF:.6f}")

    return RF, E_phiz, E_phi2

# ============================================================
# Monte Carlo cross-validation
# ============================================================

def compute_RF_mc(activation_fn, name="", n_samples=10_000_000, seed=42):
    """Compute R_F via Monte Carlo."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_samples)

    phi_z = activation_fn(z)

    E_phiz = np.mean(phi_z * z)
    E_phi2 = np.mean(phi_z**2)
    E_z2 = 1.0

    RF = E_phiz**2 / (E_phi2 * E_z2)

    # Standard errors
    se_phiz = np.std(phi_z * z) / np.sqrt(n_samples)
    se_phi2 = np.std(phi_z**2) / np.sqrt(n_samples)

    print(f"  [MC] E[phi*z]  = {E_phiz:.10f} +/- {se_phiz:.2e}")
    print(f"  [MC] E[phi^2]  = {E_phi2:.10f} +/- {se_phi2:.2e}")
    print(f"  [MC] R_F       = {RF:.6f}")

    return RF

# ============================================================
# Analytical values for comparison
# ============================================================

def analytical_RF_relu():
    """ReLU: E[ReLU(z)*z] = E[z^2 * 1_{z>0}] = 1/2
             E[ReLU(z)^2] = E[z^2 * 1_{z>0}] = 1/2
       R_F = (1/2)^2 / (1/2 * 1) = 1/2 / (1/2) = 1/2 ???

       Wait, let me recompute. For z ~ N(0,1):
       E[max(z,0) * z] = E[z^2 * 1_{z>0}] = 1/2
       E[max(z,0)^2] = E[z^2 * 1_{z>0}] = 1/2
       R_F = (1/2)^2 / (1/2) = 1/2

       But the paper claims R_F^ReLU = 1/pi ≈ 0.318...

       Hmm, that doesn't match. Let me think more carefully.

       The formula in the paper is for F(x̂) = W2 * phi(W1 * x̂).
       The radial fraction R_F is about the correlation between
       x̂^T F(x̂) and ||F(x̂)||.

       For a TWO-LAYER structure W2 * phi(W1 * x̂):
       The radial component is x̂^T W2 phi(W1 x̂).

       With W2 independent of W1, x̂, the radial fraction becomes
       the squared correlation of phi(z) with z, passed through
       the Bussgang-type formula.

       Actually, let me re-read the formula more carefully.

       R_F^phi = E[phi(z)*z]^2 / (E[phi(z)^2] * E[z^2])

       For ReLU:
       E[ReLU(z)*z] = E[z^2 * 1_{z>0}] = 1/2
       E[ReLU(z)^2] = E[z^2 * 1_{z>0}] = 1/2
       E[z^2] = 1

       R_F = (1/2)^2 / (1/2 * 1) = 1/4 / (1/2) = 1/2

       But paper says 1/pi... Let me check if the formula is different.

       Actually, maybe the Bussgang coefficient for ReLU is:
       corr(ReLU(z), z) = E[ReLU(z)*z] / sqrt(E[ReLU(z)^2] * E[z^2])
                        = (1/2) / sqrt(1/2) = 1/sqrt(2)
       corr^2 = 1/2

       So R_F = corr^2 = 1/2, not 1/pi.

       But 1/pi comes from the sign function:
       corr(sign(z), z) = E[sign(z)*z] / sqrt(E[sign(z)^2]*E[z^2])
                        = E[|z|] / 1 = sqrt(2/pi)
       corr^2 = 2/pi

       So 1/pi is NOT the radial fraction of ReLU. There might be
       a different formula at play. Let me just compute numerically
       and report.
    """
    # E[ReLU(z)*z] = E[z^2 * 1_{z>0}] = 1/2
    # E[ReLU(z)^2] = E[z^2 * 1_{z>0}] = 1/2
    # R_F = (1/2)^2 / (1/2 * 1) = 1/2
    return 0.5

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("THEOREM 5 VERIFICATION: Radial Fraction R_F")
    print("Formula: R_F = E[phi(z)*z]^2 / (E[phi(z)^2] * E[z^2])")
    print("where z ~ N(0,1)")
    print("=" * 60)

    activations = [
        (identity, "Identity"),
        (relu, "ReLU"),
        (gelu, "GELU"),
        (silu, "SiLU/Swish"),
        (tanh_act, "tanh"),
    ]

    paper_values = {
        "Identity": 1.0,
        "ReLU": 1/math.pi,  # = 0.31831...
        "GELU": 0.358,
        "SiLU/Swish": 0.371,
        "tanh": 0.405,
    }

    results = {}

    for act_fn, name in activations:
        rf_quad, e_phiz, e_phi2 = compute_RF_quad(act_fn, name)
        rf_mc = compute_RF_mc(act_fn, name)

        paper_val = paper_values.get(name, None)
        if paper_val is not None:
            print(f"  [Paper]  R_F = {paper_val:.6f}")
            print(f"  [Quad]   R_F = {rf_quad:.6f}  (diff from paper: {(rf_quad - paper_val)/paper_val*100:+.2f}%)")

        results[name] = {
            "quad": rf_quad,
            "mc": rf_mc,
            "paper": paper_val,
            "E_phiz": e_phiz,
            "E_phi2": e_phi2,
        }

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Activation':<15} {'Quad R_F':>10} {'MC R_F':>10} {'Paper R_F':>10} {'Match?':>8}")
    print("-" * 60)
    for name, r in results.items():
        match = "YES" if r["paper"] and abs(r["quad"] - r["paper"]) < 0.005 else "NO"
        paper_str = f"{r['paper']:.6f}" if r['paper'] else "N/A"
        print(f"{name:<15} {r['quad']:>10.6f} {r['mc']:>10.6f} {paper_str:>10} {match:>8}")

    print("\n" + "=" * 60)
    print("NOTE ON ReLU:")
    print("The straightforward Bussgang formula gives R_F^ReLU = 1/2,")
    print("NOT 1/pi as claimed in the paper.")
    print(f"  1/pi   = {1/math.pi:.6f}")
    print(f"  1/2    = {0.5:.6f}")
    print("If the paper claims 1/pi, there may be a different formula")
    print("at play (e.g., involving the two-layer W2*ReLU(W1*x) structure")
    print("rather than the single-Bussgang-coefficient formula).")
    print("=" * 60)
