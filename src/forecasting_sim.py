"""
forecasting_sim.py
Simulation-based approximation of the predictive density for MAR(r,1) processes.
Method: Lanne et al. (2012a), Section 4.1 of Hecq & Voisin (2021).
"""

import numpy as np
from scipy.stats import t as student_t
from src.preprocessing import MARParams


def simulate_future_errors(N: int, M: int, df: float, scale: float) -> np.ndarray:
    """
    Draw N independent sequences of M future errors from Student-t(df, scale).

    Returns
    -------
    errors : np.ndarray, shape (N, M)
        errors[j, i] is the i-th future error of the j-th simulated path.
    """
    return student_t.rvs(df=df, scale=scale, size=(N, M))

def compute_future_u(errors: np.ndarray, psi: float, h: int = 1) -> np.ndarray:
    """
    For each simulated error path, compute the implied u*_{T+h} using
    the truncated sum from Equation 6:
        u*_{T+h} ≈ Σ_{i=0}^{M-h} ψ^i · ε*_{T+h+i}

    Parameters
    ----------
    errors : np.ndarray, shape (N, M)
    psi    : noncausal lead coefficient
    h      : forecast horizon (default 1)

    Returns
    -------
    u_future : np.ndarray, shape (N,)
    """
    M = errors.shape[1]
    # Build geometric weights [ψ^0, ψ^1, ..., ψ^(M-h)]
    powers = np.arange(M - h + 1)
    weights = psi ** powers          # shape (M-h+1,)
    # For each path, sum ε*_{T+h}, ε*_{T+h+1}, ... weighted by ψ^i
    return errors[:, h-1:] @ weights  # shape (N,)

def compute_weights(u_T: float, errors: np.ndarray, psi: float,
                    df: float, scale: float) -> np.ndarray:
    """
    Compute importance weights for each simulated path based on how
    consistent it is with the last observed value u_T.

    Weight for path j: g(u_T - Σ_{i=1}^{M} ψ^i · ε*_{T+i})
    where g is the Student-t pdf. Weights are normalised to sum to 1.

    Uses log-weights internally to avoid numerical underflow.

    Parameters
    ----------
    u_T    : last observed noncausal component
    errors : np.ndarray, shape (N, M)
    psi    : noncausal lead coefficient
    df     : degrees of freedom
    scale  : scale parameter

    Returns
    -------
    weights : np.ndarray, shape (N,), sums to 1
    """
    M = errors.shape[1]
    # Build weights [ψ^1, ψ^2, ..., ψ^M]
    powers = np.arange(1, M + 1)
    psi_weights = psi ** powers           # shape (M,)

    # For each path, compute u_T - Σ ψ^i ε*_{T+i}
    weighted_errors = errors @ psi_weights  # shape (N,)
    residuals = u_T - weighted_errors       # shape (N,)

    # Evaluate log Student-t pdf at each residual
    log_w = student_t.logpdf(residuals, df=df, scale=scale)

    # Normalise in log space for numerical stability
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()

def simulate_predictive_cdf(
    u_T: float,
    psi: float,
    df: float,
    scale: float,
    grid: np.ndarray,
    N: int = 100_000,
    M: int = 100,
) -> np.ndarray:
    """
    Approximate the predictive CDF of u*_{T+1} given u_T using
    weighted simulations (Equation 10, Hecq & Voisin 2021).

    Parameters
    ----------
    u_T   : last observed noncausal component
    psi   : noncausal lead coefficient
    df    : degrees of freedom
    scale : scale parameter
    grid  : candidate values for u*_{T+1}
    N     : number of simulated paths
    M     : truncation parameter

    Returns
    -------
    cdf : np.ndarray, shape matching grid, values in [0,1]
    """
    errors   = simulate_future_errors(N, M, df, scale)
    u_future = compute_future_u(errors, psi, h=1)
    weights  = compute_weights(u_T, errors, psi, df, scale)

    # Weighted empirical CDF: for each grid point x,
    # sum weights of paths where u*_{T+1} <= x
    cdf = np.array([
        weights[u_future <= x].sum()
        for x in grid
    ])
    return cdf

def crash_probability(
    u_T: float,
    psi: float,
    df: float,
    scale: float,
    threshold: float,
    N: int = 100_000,
    M: int = 100,
) -> float:
    """
    Estimate the probability that u*_{T+1} falls below a threshold.
    This is P(u*_{T+1} <= threshold | u_T).

    Parameters
    ----------
    u_T       : last observed noncausal component
    psi       : noncausal lead coefficient
    df        : degrees of freedom
    scale     : scale parameter
    threshold : crash threshold (e.g. 0.8 * u_T for a 20% drop)
    N         : number of simulated paths
    M         : truncation parameter

    Returns
    -------
    prob : float, probability of crash
    """
    errors   = simulate_future_errors(N, M, df, scale)
    u_future = compute_future_u(errors, psi, h=1)
    weights  = compute_weights(u_T, errors, psi, df, scale)
    return float(weights[u_future <= threshold].sum())