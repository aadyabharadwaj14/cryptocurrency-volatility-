#!/usr/bin/env python
"""
Rolling (out-of-sample) backtest for GARCH(1,1) and GARCHX
----------------------------------------------------------
Trains on expanding window and forecasts one-step-ahead volatility.
Outputs OOS QLIKE and RMSE for both models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.models.baseline_garch import fit_garch11, one_step_forecast
from src.models.garchx_two_step import GarchXTwoStep
from src.models.metrics import realized_vol_squared, rmse, qlike

PRO = Path("data/processed")

def rolling_backtest(df, start_train_frac=0.7, step=1):
    """
    Perform expanding-window one-step-ahead forecasts.
    start_train_frac: fraction of data to use for initial training
    step: how many days to advance each iteration (1 = daily)
    """
    r = df["ret"].astype(float)
    X = df.drop(columns=[c for c in ["price", "ret"] if c in df.columns], errors="ignore")

    # log1p-scale the features for stability
    X = np.log1p(X)

    n = len(df)
    start_idx = int(n * start_train_frac)

    preds_garch = []
    preds_garchx = []
    realized = []

    dates = df.index

    for i in tqdm(range(start_idx, n - step), desc="Rolling backtest"):
        train_slice = slice(0, i)
        test_slice = i + step - 1

        # --- Train up to i ---
        r_train = r.iloc[train_slice]
        X_train = X.iloc[train_slice]

        # --- Fit GARCH(1,1) ---
        try:
            garch_res = fit_garch11(r_train)
            sigma2_pred = one_step_forecast(garch_res).iloc[-1]  # forecast variance for t+1
        except Exception:
            sigma2_pred = np.nan

        # --- Fit GARCHX (two-step) ---
        try:
            gx = GarchXTwoStep(lag_k=1, dist="t").fit(r_train, X_train)
            sigma2_pred_x = gx.predict_sigma2_in_sample(X_train).iloc[-1]
        except Exception:
            sigma2_pred_x = np.nan

        preds_garch.append(sigma2_pred)
        preds_garchx.append(sigma2_pred_x)
        realized.append(r.iloc[test_slice] ** 2)

    out_df = pd.DataFrame({
        "timestamp": dates[start_idx+1: n],
        "realized_var": realized,
        "garch_var": preds_garch,
        "garchx_var": preds_garchx
    }).set_index("timestamp")

    # Drop missing
    out_df = out_df.dropna()

    return out_df


def main():
    path = PRO / "aligned_returns_features.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.dropna(how="any")

    print(f"Loaded {len(df)} rows from {path}")

    results = rolling_backtest(df, start_train_frac=0.7)
    print(f"\n✅ Produced {len(results)} OOS forecasts")

    # Evaluate OOS performance
    rv = results["realized_var"]
    garch = results["garch_var"]
    garchx = results["garchx_var"]

    print("\n=== Out-of-sample performance ===")
    print(f"GARCH(1,1):  RMSE={rmse(rv, garch):.6e},  QLIKE={qlike(rv, garch):.4f}")
    print(f"GARCHX:      RMSE={rmse(rv, garchx):.6e},  QLIKE={qlike(rv, garchx):.4f}")

    # Save to file
    out_file = PRO / "rolling_backtest_results.csv"
    results.to_csv(out_file)
    print(f"💾 Saved detailed forecasts: {out_file}")


if __name__ == "__main__":
    main()
