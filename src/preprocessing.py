"""
preprocessing.py
Loads and validates the two outputs from marx_estimation.R:
  - data/processed/mar_parameters.csv
  - data/processed/nickel_filtered.csv
"""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
PARAMS_PATH   = ROOT / "data" / "processed" / "mar_parameters.csv"
FILTERED_PATH = ROOT / "data" / "processed" / "nickel_filtered.csv"


# ── Parameter container ────────────────────────────────────────────────────────
@dataclass
class MARParams:
    """Estimated MAR(r,s) parameters from MARX."""
    phi:   float   # causal AR coefficient
    psi:   float   # noncausal AR coefficient
    df:    float   # Student-t degrees of freedom
    scale: float   # Student-t scale (σ)
    r:     int     # causal lag order
    s:     int     # noncausal lag order

    def __post_init__(self):
        assert self.r >= 0 and self.s >= 0, "Lag orders must be non-negative."
        assert self.df > 0, "Degrees of freedom must be positive."
        assert self.scale > 0, "Scale must be positive."


# ── Date parser ────────────────────────────────────────────────────────────────
def _parse_mar_date(date_str: str) -> pd.Timestamp:
    """Parse '1980M02' → Timestamp('1980-02-01')."""
    year, month = int(date_str[:4]), int(date_str[5:])
    return pd.Timestamp(year=year, month=month, day=1)


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_parameters(path: Path = PARAMS_PATH) -> MARParams:
    """
    Load MAR parameters from CSV.
    Handles both:
      - Wide format: single row with columns phi, psi, df, scale, r, s
      - Long format: columns 'name' and 'value'
    """
    df = pd.read_csv(path)

    if {"name", "value"}.issubset(df.columns):
        df = df.set_index("name")["value"].to_frame().T.reset_index(drop=True)

    row = df.iloc[0]
    return MARParams(
        phi=float(row["phi"]),
        psi=float(row["psi"]),
        df=float(row["df"]),
        scale=float(row["scale"]),
        r=int(row["r"]),
        s=int(row["s"]),
    )


def load_filtered_series(path: Path = FILTERED_PATH) -> pd.DataFrame:
    """
    Load HP-filtered nickel prices.
    Returns DataFrame with DatetimeIndex (freq='MS') and columns [cycle, u_t].
    """
    df = pd.read_csv(path)
    df["Date"] = df["Date"].apply(_parse_mar_date)
    df = df.set_index("Date").sort_index()
    df.index.freq = pd.tseries.frequencies.to_offset("MS")
    cols = [c for c in ["cycle", "u_t"] if c in df.columns]
    if not cols:
        raise ValueError(f"Expected 'cycle' and/or 'u_t'. Got: {df.columns.tolist()}")

    df = df[cols].astype(float)

    if df.isnull().any().any():
        raise ValueError(f"Series contains {df.isnull().sum().sum()} NaN(s).")

    return df


# ── Main entry point ───────────────────────────────────────────────────────────
def load_data(
    params_path:   Path = PARAMS_PATH,
    filtered_path: Path = FILTERED_PATH,
) -> tuple[MARParams, pd.DataFrame]:
    """
    Returns
    -------
    params : MARParams
    series : pd.DataFrame  — columns [cycle, u_t], DatetimeIndex (freq='MS')
    """
    params = load_parameters(params_path)
    series = load_filtered_series(filtered_path)

    print(f"Parameters : MAR({params.r},{params.s}) | "
          f"φ={params.phi}, ψ={params.psi}, df={params.df:.4f}, σ={params.scale:.2f}")
    print(f"Series     : {len(series)} obs "
          f"[{series.index[0].strftime('%Y-%m')} → {series.index[-1].strftime('%Y-%m')}]")

    return params, series


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    params, series = load_data()
    print(series.head(3))
    print(series.describe().round(2))