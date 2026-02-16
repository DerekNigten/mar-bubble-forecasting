"""
forecasting_sample.py
Sample-based approximation of the predictive density for MAR(r,1) processes.
Method: Gouriéroux & Jasiak (2016), Section 4.2 of Hecq & Voisin (2021).
"""

import numpy as np
from scipy.stats import t as student_t


def compute_sample_weights(
    u_T: float,
    u_sample: np.ndarray,
    psi: float,
    df: float,
    scale: float,
) -> np.ndarray:
    """
    Compute sample-based weights for the predictive density.
    Approximates the marginal density l(u_T) using all observed sample
    values, as in Equation 11 of Hecq & Voisin (2021).

    Weight for each sample point u_i:
        g(u_T - ψ · u_i)
    where g is the Student-t pdf. Normalised to sum to 1.

    Parameters
    ----------
    u_T      : last observed noncausal component
    u_sample : all observed values of u_t in the sample, shape (T,)
    psi      : noncausal lead coefficient
    df       : degrees of freedom
    scale    : scale parameter

    Returns
    -------
    weights : np.ndarray, shape (T,), sums to 1
    """
    residuals = u_T - psi * u_sample
    log_w = student_t.logpdf(residuals, df=df, scale=scale)
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()

def sample_predictive_density(
    u_T: float,
    u_sample: np.ndarray,
    psi: float,
    df: float,
    scale: float,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Approximate the one-step ahead predictive density of u*_{T+1}
    using sample-based weights (Equation 12, Hecq & Voisin 2021).

    For each candidate value u* on the grid:
        l(u*|u_T) ∝ g(u_T - ψu*) × Σ_i g(u* - ψu_i) / Σ_i g(u_T - ψu_i)

    Parameters
    ----------
    u_T      : last observed noncausal component
    u_sample : all observed u_t values, shape (T,)
    psi      : noncausal lead coefficient
    df       : degrees of freedom
    scale    : scale parameter
    grid     : candidate values for u*_{T+1}

    Returns
    -------
    density : np.ndarray, shape matching grid
    """
    # Denominator: Σ_i g(u_T - ψ·u_i) — scalar, same for all grid points
    denom_residuals = u_T - psi * u_sample
    log_denom = student_t.logpdf(denom_residuals, df=df, scale=scale)
    denom = np.exp(log_denom).sum()

    density = np.zeros(len(grid))
    for k, u_star in enumerate(grid):
        # Transition: g(u_T - ψ·u*)
        transition = np.exp(
            student_t.logpdf(u_T - psi * u_star, df=df, scale=scale)
        )
        # Numerator: Σ_i g(u* - ψ·u_i)
        numer_residuals = u_star - psi * u_sample
        numer = np.exp(
            student_t.logpdf(numer_residuals, df=df, scale=scale)
        ).sum()

        density[k] = transition * numer / denom

    # Normalise so density integrates to 1
    normaliser = np.trapz(density, grid)
    return density / normaliser

def sample_crash_probability(
    u_T: float,
    u_sample: np.ndarray,
    psi: float,
    df: float,
    scale: float,
    threshold: float,
    n_grid: int = 1000,
) -> float:
    """
    Estimate crash probability P(u*_{T+1} <= threshold | u_T)
    using the sample-based predictive density.

    Parameters
    ----------
    u_T       : last observed noncausal component
    u_sample  : all observed u_t values
    psi       : noncausal lead coefficient
    df        : degrees of freedom
    scale     : scale parameter
    threshold : crash threshold
    n_grid    : number of grid points for integration

    Returns
    -------
    prob : float
    """
    # Grid wide enough to capture both modes
    marginal_scale = scale / (1.0 - psi)
    lower = min(0, u_T) - 20 * marginal_scale
    upper = max(threshold, u_T / psi) + 20 * marginal_scale
    grid = np.linspace(lower, upper, n_grid)

    density = sample_predictive_density(u_T, u_sample, psi, df, scale, grid)

    # Integrate density up to threshold
    below_threshold = grid <= threshold
    if not below_threshold.any():
        return 0.0
    return float(np.trapz(density[below_threshold], grid[below_threshold]))