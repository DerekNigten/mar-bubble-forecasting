"""
run_monte_carlo.py
Runs the full Monte Carlo study and saves results to outputs/tables/.
Expected runtime: several hours.
"""

import pandas as pd
from src.monte_carlo import run_table1, run_table2

print("=== Running Table 1 (Method 1 - Simulation-based) ===")
t1 = run_table1(
    psi_values     = [0.2, 0.5, 0.8],
    df_values      = [1.0, 2.0, 3.0],
    n_replications = 1000,
    N              = 1_000_000,
    M              = 100,
    crash_pct      = 0.25,
)
t1.to_csv("outputs/tables/table1_simulation.csv", index=False)
print("\nTable 1 saved to outputs/tables/table1_simulation.csv")
print(t1)

print("\n=== Running Table 2 (Method 2 - Sample-based) ===")
t2, t2_raw = run_table2(
    psi_values     = [0.2, 0.5, 0.8],
    df_values      = [1.0, 2.0, 3.0],
    sample_sizes   = [100, 200, 500, 1000],
    n_replications = 1000,
    crash_pct      = 0.25,
)
t2.to_csv("outputs/tables/table2_sample.csv", index=False)
t2_raw.to_csv("outputs/tables/table2_raw_draws.csv", index=False)
print("\nTable 2 saved to outputs/tables/table2_sample.csv")
print(f"Raw draws saved to outputs/tables/table2_raw_draws.csv  ({len(t2_raw):,} rows)")
print(t2)

print("\n=== Monte Carlo complete ===")