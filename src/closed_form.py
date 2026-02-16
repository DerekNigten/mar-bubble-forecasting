"""
closed_form.py
Implements the Cauchy closed-form predictive density for purely noncausal
MAR(0,1) processes (Equation 5, Hecq & Voisin 2021), and a MAR(1,1) wrapper
that maps y* -> u* via the causal filter before evaluating the density.
"""

import numpy as np
from src.preprocessing import MARParams


def cauchy_predictive_density(
    u_T: float,
    psi: float,
    gamma: float,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Equation 5 (h=1): one-step ahead predictive density of u*_{T+1} given u_T
    for a Cauchy MAR(0,1) process.

    Parameters
    ----------
    u_T   : last observed noncausal component
    psi   : noncausal (lead) coefficient
    gamma : scale of the error εt (NOT the marginal scale of u_t)
    grid  : candidate values for u*_{T+1}  (np.ndarray, shape (N,))

    Returns
    -------
    density : np.ndarray, shape (N,), unnormalised but correctly scaled
    """
    g2 = gamma ** 2

    # Factor 1: normalising constant
    const = 1.0 / (np.pi * gamma)

    # Factor 2: Cauchy transition kernel  g(u_T - ψ·u*)
    transition = 1.0 / (1.0 + (u_T - psi * grid) ** 2 / g2)

    # Factor 3: ratio of stationary densities  l(u_T) / l(u*)
    #   u_t ~ Cauchy(0, γ/(1-ψ))  =>  l(u) ∝ 1 / (γ² + (1-ψ)²·u²)
    #   the γ² factors in numerator/denominator do not cancel because
    #   Equation 5 keeps them explicit; ratio = (γ² + (1-ψ)²·u_T²)
    #                                           ─────────────────────
    #                                           (γ² + (1-ψ)²·u*²)
    one_minus_psi_sq = (1.0 - psi) ** 2
    stationary_ratio = (g2 + one_minus_psi_sq * u_T ** 2) / (
        g2 + one_minus_psi_sq * grid ** 2
    )

    return const * transition * stationary_ratio


def mar11_predictive_density(
    y_T: float,
    y_T_lag: float,
    phi: float,
    psi: float,
    gamma: float,
    grid: np.ndarray,
) -> np.ndarray:
    """
    One-step ahead predictive density of y*_{T+1} for a Cauchy MAR(1,1) process.

    Applies the substitution u_t = y_t - φ·y_{t-1} throughout Equation 5,
    as described in the paragraph below Equation 5 in Hecq & Voisin (2021).

    The transformation y* -> u* = y* - φ·y_T is linear with Jacobian 1,
    so the density over y* equals the density over u* evaluated at u* = y* - φ·y_T.

    Parameters
    ----------
    y_T     : last observed value of y
    y_T_lag : y_{T-1}, one period before y_T
    phi     : causal (lag) coefficient
    psi     : noncausal (lead) coefficient
    gamma   : scale of the error εt
    grid    : candidate values for y*_{T+1}

    Returns
    -------
    density : np.ndarray, shape matching grid
    """
    u_T = y_T - phi * y_T_lag          # observed noncausal component
    u_star = grid - phi * y_T          # candidate noncausal values

    return cauchy_predictive_density(u_T, psi, gamma, u_star)


def make_grid(center: float, gamma: float, psi: float, n_points: int = 2000) -> np.ndarray:
    """
    Convenience function: build a symmetric grid around `center` wide enough
    to capture both modes of the predictive density during a bubble episode.

    Width is set to ±20 × marginal scale of u_t so tails are well covered.
    """
    marginal_scale = gamma / (1.0 - psi)
    half_width = 20.0 * marginal_scale
    return np.linspace(center - half_width, center + half_width, n_points)

# ── Validation log ─────────────────────────────────────────────────────────────
# Run: python -c "from src.closed_form import ..."  (see README or notebook 03)
# Results (2025-02, last observed u_t):
#   - density integrates to 1.0 ✓
#   - grid range: -34059.4 to 38245.5 ✓
#   - max density at: 2614.5 ✓
# Full visual validation (Figure 2 replication) → notebooks/03_forecasting_methods.ipynb