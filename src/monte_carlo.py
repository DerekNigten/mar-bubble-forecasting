"""
monte_carlo.py
Monte Carlo validation of simulation-based and sample-based forecasting methods.
Reproduces Tables 1 and 2 of Hecq & Voisin (2021).
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from src.forecasting_sim import crash_probability
from src.forecasting_sample import sample_crash_probability
from src.closed_form import cauchy_predictive_density, make_grid


def simulate_mar01(T: int, psi: float, df: float, scale: float) -> np.ndarray:
    """
    Simulate a purely noncausal MAR(0,1) process of length T.
    Uses the forward representation: u_t = ψ·u_{t+1} + ε_t
    Simulated backwards from a terminal value of 0.

    Parameters
    ----------
    T     : number of observations
    psi   : noncausal lead coefficient
    df    : degrees of freedom
    scale : scale parameter

    Returns
    -------
    u : np.ndarray, shape (T,)
    """
    errors = student_t.rvs(df=df, scale=scale, size=T)
    u = np.zeros(T)
    for t in range(T - 2, -1, -1):
        u[t] = psi * u[t + 1] + errors[t]
    return u

def find_bubble_point(u: np.ndarray, quantile: float = 0.995) -> tuple[int, float]:
    """
    Find the last index where the series reaches a given quantile level.
    This represents the peak of a bubble episode.

    Parameters
    ----------
    u        : simulated MAR(0,1) series
    quantile : quantile threshold for bubble definition

    Returns
    -------
    idx     : index of bubble point
    u_T     : value at bubble point
    """
    threshold = np.quantile(u, quantile)
    indices = np.where(u >= threshold)[0]
    if len(indices) == 0:
        return None, None
    # Take the last occurrence to capture bubble peak
    idx = indices[-1]
    return idx, float(u[idx])

def monte_carlo_sim(
    psi: float,
    df: float,
    n_replications: int = 1000,
    T: int = 500,
    quantile: float = 0.995,
    N: int = 100_000,
    M: int = 100,
    crash_pct: float = 0.25,
    scale: float = 1.0,
) -> dict:
    """
    Monte Carlo study for Method 1 (simulation-based).
    Reproduces Table 1 of Hecq & Voisin (2021).

    For each replication:
    1. Simulate MAR(0,1) series
    2. Find bubble point at quantile
    3. Compute crash probability using Method 1
    4. Compare against Cauchy closed-form (if df=1)

    Returns
    -------
    dict with keys:
        mean_prob  : average crash probability across replications
        std_prob   : standard deviation
        cauchy_ref : theoretical Cauchy probability (only if df=1)
    """
    probs = []

    for rep in range(n_replications):
        u = simulate_mar01(T=T, psi=psi, df=df, scale=scale)
        idx, u_T = find_bubble_point(u, quantile)

        if u_T is None:
            continue

        threshold = (1 - crash_pct) * u_T
        prob = crash_probability(
            u_T=u_T,
            psi=psi,
            df=df,
            scale=scale,
            threshold=threshold,
            N=N,
            M=M,
        )
        probs.append(prob)

        if (rep + 1) % 100 == 0:
            print(f"  Replication {rep+1}/{n_replications} done")

    probs = np.array(probs)

    # Cauchy closed-form reference (only valid for df=1)
    cauchy_ref = None
    if abs(df - 1.0) < 0.01:
        grid = make_grid(u_T, scale, psi)
        density = cauchy_predictive_density(u_T, psi, scale, grid)
        cdf = np.cumsum(density) * (grid[1] - grid[0])
        cauchy_ref = float(cdf[grid <= threshold].max())

    return {
        "psi":       psi,
        "df":        df,
        "mean_prob": float(probs.mean()),
        "std_prob":  float(probs.std()),
        "cauchy_ref": cauchy_ref,
        "n_valid":   len(probs),
    }

def monte_carlo_sample(
    psi: float,
    df: float,
    sample_size: int,
    n_replications: int = 1000,
    quantile: float = 0.995,
    crash_pct: float = 0.25,
    scale: float = 1.0,
) -> dict:
    """
    Monte Carlo study for Method 2 (sample-based).
    Reproduces Table 2 of Hecq & Voisin (2021).

    Parameters
    ----------
    psi         : noncausal lead coefficient
    df          : degrees of freedom
    sample_size : length of simulated series (100, 200, 500, 1000)
    n_replications : number of Monte Carlo replications

    Returns
    -------
    dict with keys:
        mean_prob : mean crash probability
        std_prob  : std of crash probabilities
        q1_prob   : first quartile (how much variation across trajectories)
        mode_prob : most frequent probability (upper bound per paper)
    """
    probs = []

    for rep in range(n_replications):
        u = simulate_mar01(T=sample_size, psi=psi, df=df, scale=scale)
        idx, u_T = find_bubble_point(u, quantile)

        if u_T is None:
            continue

        threshold = (1 - crash_pct) * u_T
        prob = sample_crash_probability(
            u_T=u_T,
            u_sample=u,
            psi=psi,
            df=df,
            scale=scale,
            threshold=threshold,
        )
        probs.append(prob)

        if (rep + 1) % 100 == 0:
            print(f"  Replication {rep+1}/{n_replications} done")

    probs = np.array(probs)

    return {
        "psi":        psi,
        "df":         df,
        "sample_size": sample_size,
        "mean_prob":  float(probs.mean()),
        "std_prob":   float(probs.std()),
        "q1_prob":    float(np.percentile(probs, 25)),
        "mode_prob":  float(probs.max()),
        "n_valid":    len(probs),
    }

def run_table1(
    psi_values: list = [0.2, 0.5, 0.8],
    df_values: list = [1.0, 2.0, 3.0],
    n_replications: int = 1000,
    N: int = 1_000_000,
    M: int = 100,
    crash_pct: float = 0.25,
) -> pd.DataFrame:
    """
    Reproduce Table 1 of Hecq & Voisin (2021).
    Simulation-based crash probabilities across ψ and df values.

    Warning: computationally intensive.
    With n_replications=1000 and N=1_000_000 expect several hours.
    For testing use n_replications=50 and N=10_000.
    """
    rows = []
    total = len(psi_values) * len(df_values)
    done = 0

    for psi in psi_values:
        for df in df_values:
            done += 1
            print(f"\n[{done}/{total}] Running MAR(0,1) psi={psi}, df={df}")
            result = monte_carlo_sim(
                psi=psi,
                df=df,
                n_replications=n_replications,
                N=N,
                M=M,
                crash_pct=crash_pct,
            )
            rows.append(result)

    return pd.DataFrame(rows)


def run_table2(
    psi_values: list = [0.2, 0.5, 0.8],
    df_values: list = [1.0, 2.0, 3.0],
    sample_sizes: list = [100, 200, 500, 1000],
    n_replications: int = 1000,
    crash_pct: float = 0.25,
) -> pd.DataFrame:
    """
    Reproduce Table 2 of Hecq & Voisin (2021).
    Sample-based crash probabilities across ψ, df and sample sizes.

    Warning: computationally intensive.
    With n_replications=1000 expect several hours.
    For testing use n_replications=50.
    """
    rows = []
    total = len(psi_values) * len(df_values) * len(sample_sizes)
    done = 0

    for psi in psi_values:
        for df in df_values:
            for T in sample_sizes:
                done += 1
                print(f"\n[{done}/{total}] Running psi={psi}, df={df}, T={T}")
                result = monte_carlo_sample(
                    psi=psi,
                    df=df,
                    sample_size=T,
                    n_replications=n_replications,
                    crash_pct=crash_pct,
                )
                rows.append(result)

    return pd.DataFrame(rows)

