#!/usr/bin/env python
"""
Run baseline GARCH(1,1) and two-step GARCHX on the aligned dataset.

Expects: data/processed/aligned_returns_features.csv
Columns: timestamp, price, ret, <ETH feature columns...>
"""

from pathlib import Path
import pandas as pd
import numpy as np


from src.models.baseline_garch import fit_garch11, one_step_forecast
from src.models.garchx_two_step import GarchXTwoStep
from src.models.metrics import realized_vol_squared, rmse, qlike

PRO = Path("data/processed")


def load_aligned_df() -> pd.DataFrame:
    """
    Load aligned_returns_features.csv robustly:
    - If 'timestamp' exists as a column, parse it and set as index.
    - Otherwise treat first column as datetime index.
    Drops rows with any NaNs (e.g., first lag row) and sorts by time.
    """
    path = PRO / "aligned_returns_features.csv"
    try:
        # Case 1: timestamp is a named column
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
    except ValueError:
        # Case 2: timestamp was saved as the index (blank header), use first column
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        df.index.name = "timestamp"

    # Drop rows created by lagging (NaNs in features) or any incomplete rows
    df = df.dropna(how="any")
    return df


def main():
    df = load_aligned_df()

    # Target returns (BTC)
    r = df["ret"].astype(float)

    # Exogenous ETH graph features (drop non-features if present)
    X = df.drop(columns=[c for c in ["price", "ret"] if c in df.columns], errors="ignore")
    X = np.log1p(X)
    # ---------- Baseline GARCH(1,1)
    garch_res = fit_garch11(r)  # fits on full sample (quick in-sample comparison)
    sigma2_garch = one_step_forecast(garch_res)  # in-sample one-step variance (unscaled)

    # ---------- Two-step GARCHX
    gx = GarchXTwoStep(lag_k=1, dist="t").fit(r, X)
    sigma2_garchx = gx.predict_sigma2_in_sample(X)

    # Realized variance proxy (r_t^2)
    rv = realized_vol_squared(r)

    # Align for scoring
    rv_g, s2_g = rv.align(sigma2_garch, join="inner")
    rv_x, s2_x = rv.align(sigma2_garchx, join="inner")

    print("\n=== Sample sizes ===")
    print(f"Realized Var: {len(rv)} | GARCH aligned: {len(rv_g)} | GARCHX aligned: {len(rv_x)}")

    print("\n=== Baseline GARCH(1,1) ===")
    print(garch_res.summary())
    print(f"RMSE  (σ^2 vs r^2): {rmse(rv_g, s2_g):.6e}")
    print(f"QLIKE (lower=better): {qlike(rv_g, s2_g):.6f}")

    print("\n=== Two-step GARCHX ===")
    print("Gamma (feature effects):")
    for k, v in (gx.params_["gamma"] or {}).items():
        print(f"  {k}: {v:.6f}")
    print(f"Intercept: {gx.params_['gamma_intercept']:.6f}")
    print(f"RMSE  (σ^2 vs r^2): {rmse(rv_x, s2_x):.6e}")
    print(f"QLIKE (lower=better): {qlike(rv_x, s2_x):.6f}")


if __name__ == "__main__":
    main()
