#!/usr/bin/env python
"""
Predict today's BTC volatility with the best variant (GARCH + Ridge adj),
auto-update BTC data if possible, and self-validate past forecasts.

Files touched:
- READ  : data/processed/BTC_returns_daily.csv
- READ? : data/processed/aligned_returns_features.csv (optional)
- WRITE : data/processed/volatility_forecasts_log.csv (append/update)
"""

import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime, UTC
from arch import arch_model
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

PRO = Path("data/processed")
RET_CSV = PRO / "BTC_returns_daily.csv"
ALIGNED_CSV = PRO / "aligned_returns_features.csv"
LOG_CSV = PRO / "volatility_forecasts_log.csv"

# ----------------- helpers -----------------
def try_update_btc_with_yfinance():
    """
    Append missing BTC-USD daily rows using yfinance (if installed).
    Ensures we only add rows STRICTLY after the last timestamp in file.
    Prints the fetched date range for debugging.
    """
    try:
        import yfinance as yf
    except Exception:
        return False, "yfinance not installed; skipping update"

    if not RET_CSV.exists():
        return False, f"{RET_CSV} not found; skipping update"

    # Read and clean the CSV first
    df = pd.read_csv(RET_CSV)
    
    # Remove any rows where timestamp can't be parsed (corrupted data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    
    # Remove any rows with invalid price values
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    
    # Keep only the expected columns
    expected_cols = ["timestamp", "price", "ret"]
    df = df[[col for col in expected_cols if col in df.columns]]
    
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    
    if df.empty:
        return False, "BTC CSV has no valid rows after cleaning"

    last_ts = pd.to_datetime(df["timestamp"]).max()            # last timestamp we have
    if pd.isna(last_ts):
        return False, "BTC CSV last timestamp is NaT"

    # We'll ask Yahoo from the last_ts (inclusive) but will FILTER strictly > last_ts
    start_date = last_ts.date().isoformat()

    today_utc = pd.Timestamp.now(UTC).normalize().tz_localize(None)  # naive UTC TS
    # Ask until 'today + 1' to ensure the latest completed candle is included
    end_date = (today_utc + pd.Timedelta(days=1)).date().isoformat()

    # Download
    try:
        data = yf.download(
            "BTC-USD",
            start=start_date,                 # inclusive on Yahoo side
            end=end_date,                     # exclusive on Yahoo side
            interval="1d",
            progress=False,
            auto_adjust=True  # Explicitly set to avoid warning
        )
    except Exception as e:
        return False, f"yfinance download failed: {e}"

    if data is None or data.empty:
        return False, "No new rows from yfinance (UTC timing or market data gap)"

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    add = (
        data.reset_index()[["Date", "Close"]]
        .rename(columns={"Date": "timestamp", "Close": "price"})
    )
    add["timestamp"] = pd.to_datetime(add["timestamp"]).dt.tz_localize(None).dt.normalize()
    
    # Ensure price is float and clean any invalid values
    add["price"] = pd.to_numeric(add["price"], errors="coerce")
    add = add.dropna(subset=["timestamp", "price"])

    # Debug: show what Yahoo returned
    print(f"[DEBUG] Yahoo returned dates: {add['timestamp'].min().date()} to {add['timestamp'].max().date()}")
    
    # Keep ONLY rows strictly newer than last_ts
    before_filter = len(add)
    add = add[add["timestamp"] > last_ts]
    after_filter = len(add)
    
    # Debug: show what we're keeping
    if not add.empty:
        print(f"[DEBUG] After filter, keeping dates: {add['timestamp'].min().date()} to {add['timestamp'].max().date()}")

    # Debug info to understand what's coming from Yahoo
    fetched_min = data.index.min().date() if len(data) else None
    fetched_max = data.index.max().date() if len(data) else None
    dbg = f"Fetched {len(data)} rows from Yahoo ({fetched_min} -> {fetched_max}); " \
          f"candidate new rows before filter: {before_filter}, after filter: {after_filter}"

    if add.empty:
        return False, f"No rows strictly after {last_ts.date()} to add. {dbg}"

    # Merge - keep only timestamp and price columns from original df
    all_df = pd.concat([df[["timestamp", "price"]], add[["timestamp", "price"]]], ignore_index=True)
    all_df = all_df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    
    # Recompute log returns for the entire dataset
    all_df["log_price"] = np.log(all_df["price"].astype(float))
    all_df["ret"] = all_df["log_price"].diff()
    all_df = all_df.drop(columns=["log_price"])
    
    # Debug: check what we're about to save
    print(f"[DEBUG] DataFrame shape before save: {all_df.shape}")
    print(f"[DEBUG] Last 3 dates in DataFrame: {all_df['timestamp'].tail(3).tolist()}")
    
    # Save back to CSV
    all_df.to_csv(RET_CSV, index=False)
    
    # Verify the save worked
    verify_df = pd.read_csv(RET_CSV, parse_dates=["timestamp"])
    print(f"[DEBUG] Last date after re-reading CSV: {verify_df['timestamp'].max().date()}")

    last_added = all_df["timestamp"].max().date()
    added = len(add)  # Count from filtered new rows, not total length difference
    return True, f"Added {added} new row(s). Updated BTC to {last_added}. {dbg}"


def build_lagged_feature_matrix(df_aligned: pd.DataFrame):
    need = ["ret", "n_nodes", "n_edges", "total_volume", "avg_degree", "avg_clustering"]
    for c in need:
        if c not in df_aligned.columns:
            raise KeyError(f"Missing required column: {c}")

    df = df_aligned.copy()
    base_feats = ["n_nodes","n_edges","total_volume","avg_degree","avg_clustering"]

    for c in base_feats:
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_lag1_log1p"] = np.log1p(np.maximum(df[f"{c}_lag1"], 0))

    df["ret_abs"] = df["ret"].abs()
    df["ret_sq"]  = df["ret"] ** 2

    feat_cols = [f"{c}_lag1_log1p" for c in base_feats] + ["ret_abs", "ret_sq"]
    return df, feat_cols

def qlike_piece(rv, pv, eps=1e-12):
    rv = max(rv, eps); pv = max(pv, eps)
    return rv / pv - math.log(rv / pv) - 1.0

# ----------------- main -----------------
def main():
    PRO.mkdir(parents=True, exist_ok=True)

    # 0) Optional data update
    updated, msg = try_update_btc_with_yfinance()
    print(f"[UPDATE] {msg}")

    # 1) Load returns (reload AFTER update so we see new rows)
    if not RET_CSV.exists():
        print(f"ERROR: {RET_CSV} not found.")
        sys.exit(1)

    df_ret = pd.read_csv(RET_CSV, parse_dates=["timestamp"]).dropna(subset=["ret"]).copy()
    df_ret = df_ret.sort_values("timestamp")
    df_ret["ret_pct"] = df_ret["ret"] * 100.0

    last_date = df_ret["timestamp"].iloc[-1]
    next_day = last_date + timedelta(days=1)
    
    print(f"[DATA] Last available date in CSV: {last_date.date()}")
    print(f"[DATA] Forecasting for: {next_day.date()}")

    # 2) Fit baseline GARCH(1,1) on last N days
    N = 1000
    recent = df_ret.tail(N).reset_index(drop=True)
    print(f"[MODEL] Using last {len(recent)} days ending {last_date.date()}")

    am = arch_model(recent["ret_pct"], mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")

    sigma_t_pct = res.conditional_volatility.values
    sigma2_t = (sigma_t_pct ** 2) / (100.0 ** 2)

    fc = res.forecast(horizon=1, reindex=False)
    var_pred_base = fc.variance.values[-1, 0] / (100.0 ** 2)  # in return^2 units

    # 3) Optional Ridge adjustment (best variant)
    use_adjustment = False
    var_pred_adj = None

    if ALIGNED_CSV.exists():
        try:
            df_aligned = pd.read_csv(ALIGNED_CSV, parse_dates=["timestamp"]).sort_values("timestamp")
            start_cut = recent["timestamp"].iloc[0]
            df_a = df_aligned[df_aligned["timestamp"] >= start_cut].copy()
            df_a = df_a.set_index("timestamp").reindex(recent["timestamp"]).reset_index()
            df_a["ret"] = recent["ret"].values

            df_feat, feat_cols = build_lagged_feature_matrix(df_a)
            df_feat["sigma2_base"] = sigma2_t
            df_feat = df_feat.dropna(subset=feat_cols + ["sigma2_base"])

            if len(df_feat) >= 50:
                y = np.log(np.maximum(df_feat["sigma2_base"].values, 1e-12))
                X = df_feat[feat_cols].values
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                ridge = RidgeCV(cv=5).fit(Xs, y)

                log_sigma2_base_t = float(np.log(max(df_feat["sigma2_base"].values[-1], 1e-12)))
                X_last = df_feat[feat_cols].values[-1:].astype(float)
                log_sigma2_pred_t = float(ridge.predict(scaler.transform(X_last))[0])

                adj = float(np.clip(log_sigma2_pred_t - log_sigma2_base_t, -2.0, 2.0))
                var_pred_adj = var_pred_base * math.exp(adj)
                use_adjustment = True
            else:
                print("[RIDGE] Not enough aligned rows; using baseline only.")
        except Exception as e:
            print(f"[RIDGE] Skipped due to error: {e}")
    else:
        print("[RIDGE] aligned_returns_features.csv not found; using baseline only.")

    # 4) Print forecast(s)
    vol_pred_base_pct = 100.0 * math.sqrt(max(var_pred_base, 0.0))
    print(f"\nBaseline GARCH forecast for {next_day.date()}: {vol_pred_base_pct:.4f}%")

    vol_pred_adj_pct = None
    if use_adjustment and var_pred_adj is not None:
        vol_pred_adj_pct = 100.0 * math.sqrt(max(var_pred_adj, 0.0))
        print(f"Ridge-adjusted forecast for {next_day.date()}: {vol_pred_adj_pct:.4f}%  (best variant)")
    else:
        print("Ridge-adjusted forecast unavailable; baseline shown only.")

    # 5) Append to forecast log
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    if LOG_CSV.exists():
        log = pd.read_csv(LOG_CSV, parse_dates=["forecast_date", "generated_at"])
    else:
        log = pd.DataFrame(columns=[
            "forecast_date","generated_at","last_train_date","N_used",
            "baseline_var","baseline_vol_pct",
            "ridge_adj_var","ridge_adj_vol_pct",
            "realized_var","realized_vol_pct",
            "abs_error_pct","rel_error_pct","qlike"
        ])

    new_row = {
        "forecast_date": next_day.normalize(),
        "generated_at": pd.Timestamp.now(UTC).normalize(),
        "last_train_date": last_date.normalize(),
        "N_used": len(recent),
        "baseline_var": var_pred_base,
        "baseline_vol_pct": vol_pred_base_pct,
        "ridge_adj_var": (None if var_pred_adj is None else var_pred_adj),
        "ridge_adj_vol_pct": (None if vol_pred_adj_pct is None else vol_pred_adj_pct),
        "realized_var": None,
        "realized_vol_pct": None,
        "abs_error_pct": None,
        "rel_error_pct": None,
        "qlike": None,
    }
    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)

    # 6) Backfill realized volatility for any past forecast dates that now exist in returns
    df_back = df_ret.copy()
    df_back["date"] = df_back["timestamp"].dt.normalize()
    df_back["realized_var"] = (df_back["ret"] ** 2).astype(float)
    df_back["realized_vol_pct"] = df_back["ret"].abs() * 100.0

    filled = 0
    for i in range(len(log)):
        if pd.isna(log.loc[i, "realized_vol_pct"]):
            fdate = pd.to_datetime(log.loc[i, "forecast_date"]).normalize()
            hit = df_back[df_back["date"] == fdate]
            if not hit.empty:
                rv = float(hit["realized_var"].iloc[0])
                rvol = float(hit["realized_vol_pct"].iloc[0])
                log.loc[i, "realized_var"] = rv
                log.loc[i, "realized_vol_pct"] = rvol

                # prefer ridge error if available, else baseline
                pred_vol = log.loc[i, "ridge_adj_vol_pct"] if not pd.isna(log.loc[i, "ridge_adj_vol_pct"]) else log.loc[i, "baseline_vol_pct"]
                if not pd.isna(pred_vol) and rvol > 0:
                    log.loc[i, "abs_error_pct"] = abs(pred_vol - rvol)
                    log.loc[i, "rel_error_pct"] = 100.0 * abs(pred_vol - rvol) / rvol
                # QLIKE needs variances
                pv = log.loc[i, "ridge_adj_var"] if not pd.isna(log.loc[i, "ridge_adj_var"]) else log.loc[i, "baseline_var"]
                log.loc[i, "qlike"] = qlike_piece(rv, float(pv))
                filled += 1

    log = log.drop_duplicates(subset=["forecast_date"], keep="last").sort_values("forecast_date")
    log.to_csv(LOG_CSV, index=False)

    print(f"\n[LOG] Saved/updated forecast log: {LOG_CSV}")
    if filled:
        print(f"[LOG] Backfilled realized volatility for {filled} day(s).")

if __name__ == "__main__":
    main()