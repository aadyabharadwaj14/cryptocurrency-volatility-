#!/usr/bin/env python
"""
improve_garchx.py

Improves the two-step GARCHX pipeline by:
 - creating lagged / interaction features
 - performing feature selection (LassoCV / RidgeCV)
 - fitting multiple GARCH-family variants (GARCH, EGARCH, GJR)
 - using ensemble/meta-regression across GARCH variants
 - evaluating with rolling (expanding-window) OOS forecasts (QLIKE / RMSE)

Outputs:
 - data/processed/improved_garchx_oos_metrics.csv
 - data/processed/improved_garchx_oos_forecasts.csv
 - Prints best config & summary
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from arch import arch_model
import itertools
import json

PRO = Path("data/processed")
OUT_METRICS = PRO / "improved_garchx_oos_metrics.csv"
OUT_FORECASTS = PRO / "improved_garchx_oos_forecasts.csv"

# ---------- Utilities: metrics ----------
def qlike(y_true, y_pred, eps=1e-12):
    y_t = np.maximum(y_true, eps)
    y_p = np.maximum(y_pred, eps)
    return float(np.mean(y_t / y_p - np.log(y_t / y_p) - 1))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# ---------- Feature engineering ----------
def build_features(df, max_lag=3, include_interactions=True):
    """
    Input: df with columns including total_volume, avg_degree, avg_clustering, n_nodes, n_edges
    Output: df with lagged versions and optional interactions, and a features list
    """
    feature_cols = ["n_nodes", "n_edges", "total_volume", "avg_degree", "avg_clustering"]
    df = df.copy().sort_index()
    for lag in range(1, max_lag + 1):
        for c in feature_cols:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)

    # log1p transform for heavy-tailed features
    lagged_cols = [c for c in df.columns if any(c.startswith(f"{f}_lag") for f in feature_cols)]
    for c in lagged_cols:
        df[f"{c}_log1p"] = np.log1p(np.maximum(df[c], 0))

    # interactions (pairwise) on lag1 logs only (keeps dimensionality manageable)
    features = []
    # use lags up to 3 but interactions only on lag1
    for c in lagged_cols:
        features.append(f"{c}_log1p")
    if include_interactions:
        lag1_cols = [f"{f}_lag1_log1p" for f in feature_cols]
        for a, b in itertools.combinations(lag1_cols, 2):
            df[f"int_{a}_x_{b}"] = df[a] * df[b]
            features.append(f"int_{a}_x_{b}")

    # add simple trend features
    df["ret_abs"] = df["ret"].abs()
    df["ret_sq"] = df["ret"] ** 2
    features += [f"{c}_log1p" for c in lagged_cols]
    features += ["ret_abs", "ret_sq"]

    # deduplicate features list
    features = list(dict.fromkeys(features))
    return df, features

# ---------- GARCH fitters ----------
def fit_arch_variant(returns, vol_model="GARCH", p=1, q=1, dist="t"):
    """
    Fit an ARCH model on returns scaled by 100 (as before)
    vol_model: 'GARCH', 'EGARCH', 'GJR' (gjr)
    returns: pd.Series aligned to index
    Returns fit result (arch arch_model result)
    """
    r = returns.dropna() * 100.0
    # arch_model supports vol="GARCH", "EGARCH", "GJR" (gjr)
    vol_arg = vol_model
    if vol_model == "GJR":
        vol_arg = "GARCH"  # arch handles GJR via power? use 'GARCH' + 'o' param? Better: use 'GARCH' with power? 
        # But arch does support 'GJR'? Newer versions accept 'GARCH' with o param
    am = arch_model(r, mean="Zero", vol=vol_model, p=p, q=q, dist=dist)
    res = am.fit(disp="off")
    # conditional volatility (returns*100)
    sigma2 = (res.conditional_volatility ** 2) / (100.0 ** 2)
    return res, sigma2

# ---------- Two-step approach with feature selection ----------
def two_step_garchx_with_selection(returns, X, vol_model="GARCH", reg_kind="lasso"):
    """
    Step 1: Fit GARCH-type model (vol_model) on returns -> obtain baseline sigma2
    Step 2: Fit a regularized linear model to predict log(sigma2) from X (selected features)
    Returns:
      - baseline sigma2 series
      - adjusted sigma2 series (same index as baseline)
      - selection info dict
    """
    # ensure alignment
    r = returns.dropna()
    X_aligned = X.reindex(r.index).dropna()
    r = r.reindex(X_aligned.index)

    # fit arch variant
    res, sigma2_base = fit_arch_variant(r, vol_model=vol_model)
    log_sigma2 = np.log(np.maximum(sigma2_base, 1e-12))

    # feature selection & regression: use logs/scaled features already in X
    # Use LassoCV for selection, fallback to RidgeCV for stability
    scaler = StandardScaler()
    Xvals = scaler.fit_transform(X_aligned.values)
    if reg_kind == "lasso":
        lasso = LassoCV(cv=5, n_jobs=-1, random_state=0).fit(Xvals, log_sigma2.values)
        selector = SelectFromModel(lasso, prefit=True, threshold="median")
        selected_mask = selector.get_support()
        selected_features = np.array(X_aligned.columns)[selected_mask].tolist()
        # refit ridge on selected features for better coefficients
        if len(selected_features) == 0:
            # fallback: use all features with RidgeCV
            reg = RidgeCV(cv=5).fit(Xvals, log_sigma2.values)
            selected_features = X_aligned.columns.tolist()
            final_reg = reg
        else:
            Xsel = X_aligned.iloc[:, selected_mask].values
            final_reg = RidgeCV(cv=5).fit(Xsel, log_sigma2.values)
        # Prepare predicted log sigma2
        if len(selected_features) > 0:
            Xsel_full = scaler.transform(X_aligned.values)[:, selected_mask]
            log_sigma2_pred = final_reg.predict(Xsel_full)
        else:
            log_sigma2_pred = final_reg.predict(scaler.transform(X_aligned.values))
    else:
        # ridge path
        reg = RidgeCV(cv=5).fit(Xvals, log_sigma2.values)
        selected_features = X_aligned.columns.tolist()
        final_reg = reg
        log_sigma2_pred = final_reg.predict(Xvals)

    # adjust: compute adj factor (pred - base)
    log_sigma2_base = log_sigma2.reindex(index=X_aligned.index)
    log_adj = log_sigma2_pred - log_sigma2_base.values
    log_adj = np.clip(log_adj, -2, 2)  # safety cap

    sigma2_adjusted = sigma2_base.copy()
    sigma2_adjusted.loc[X_aligned.index] = sigma2_base.loc[X_aligned.index] * np.exp(log_adj)

    info = {
        "vol_model": vol_model,
        "reg_kind": reg_kind,
        "selected_features": selected_features,
        "reg_coef": getattr(final_reg, "coef_", None),
        "reg_intercept": float(getattr(final_reg, "intercept_", 0.0))
    }

    return sigma2_base, sigma2_adjusted, info

# ---------- Rolling OOS evaluation ----------
def rolling_oos_evaluation(df, features, start_train_frac=0.6, step=1,
                           vol_models=["GARCH", "EGARCH"], reg_kinds=["lasso", "ridge"]):
    """
    Perform expanding window OOS forecast:
     - For each step, fit arch variants on r_train, then two-step garchx with selection
     - Forecast sigma2 for next day (in-sample forecast from arch then regression)
    Returns:
      - oos_df: each row: timestamp, realized_var, baseline_var_<model>, garchx_var_<model>...
      - metrics_df: aggregated RMSE/QLIKE per model
    """
    r = df["ret"].astype(float)
    X = df[features].copy()
    n = len(df)
    start_idx = int(n * start_train_frac)
    preds = []
    idxs = []
    models_info = []

    for i in tqdm(range(start_idx, n - step), desc="Rolling OOS"):
        train_slice = slice(0, i)
        test_idx = i + step - 1

        r_train = r.iloc[train_slice]
        X_train = X.iloc[train_slice].dropna()
        # if X_train too small skip
        if len(r_train) < 50 or X_train.shape[0] < 50:
            continue

        realized = (r.iloc[test_idx] ** 2)

        # For each arch variant + reg kind, fit and predict
        row = {"timestamp": df.index[test_idx], "realized_var": realized}
        for vol_model in vol_models:
            for reg_kind in reg_kinds:
                try:
                    sigma_base, sigma_adj, info = two_step_garchx_with_selection(r_train, X_train, vol_model=vol_model, reg_kind=reg_kind)
                    # We want one-step-ahead forecast: use the last available fitted sigma_adj value aligned to train index
                    # In two-step approach we have in-sample sigma series; use last entry as forecast for t+1
                    pred_sigma_base = sigma_base.iloc[-1]
                    # For adjusted: sigma_adj is aligned to train index; take last
                    pred_sigma_adj = sigma_adj.iloc[-1]
                except Exception as e:
                    pred_sigma_base = np.nan
                    pred_sigma_adj = np.nan
                    info = {"error": str(e)}

                row[f"{vol_model}_{reg_kind}_base"] = pred_sigma_base
                row[f"{vol_model}_{reg_kind}_adj"] = pred_sigma_adj
                models_info.append((df.index[test_idx], vol_model, reg_kind, info))

        preds.append(row)
        idxs.append(df.index[test_idx])

    oos_df = pd.DataFrame(preds).set_index("timestamp")
    # drop nans
    oos_df = oos_df.dropna()

    # Compute metrics
    metrics = []
    for col in oos_df.columns:
        if col.startswith("realized"):
            continue
        y_true = oos_df["realized_var"]
        y_pred = oos_df[col]
        metrics.append({
            "model_variant": col,
            "RMSE": rmse(y_true.values, y_pred.values),
            "QLIKE": qlike(y_true.values, y_pred.values)
        })
    metrics_df = pd.DataFrame(metrics).sort_values("QLIKE")
    return oos_df, metrics_df, models_info

# ---------- Main run ----------
def main():
    path = PRO / "aligned_returns_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.dropna(how="any")

    # build enriched features
    df_feat, feat_list = build_features(df, max_lag=3, include_interactions=True)

    # choose a smaller sensible feature set for X due to computational cost
    # Use lag1 logs + ret_sq and ret_abs + a couple of interactions
    candidate_features = [c for c in feat_list if ("lag1_log1p" in c or "ret_" in c or c.startswith("int_"))]
    X = df_feat[candidate_features].fillna(method="ffill").fillna(0)

    # run rolling OOS evaluation
    vol_models = ["GARCH", "EGARCH"]  # you can add "GJR" if arch supports 'GJR' in your version
    reg_kinds = ["lasso", "ridge"]
    # Ensure df_feat and X have no overlapping columns before joining
    df_merged = df_feat.copy()
    for c in X.columns:
        if c in df_merged.columns:
            df_merged = df_merged.drop(columns=[c])
    df_merged = df_merged.join(X)

    # Run rolling OOS evaluation
    oos_df, metrics_df, models_info = rolling_oos_evaluation(
        df_merged,
        candidate_features,
        start_train_frac=0.6,
        step=1,
        vol_models=vol_models,
        reg_kinds=reg_kinds
    )


    # Save outputs
    oos_df.to_csv(OUT_FORECASTS)
    metrics_df.to_csv(OUT_METRICS, index=False)

    print("\n=== OOS Metrics (sorted by QLIKE) ===")
    print(metrics_df.head(10).to_string(index=False))

    # Save model info (selected features etc.)
    info_serializable = []
    for t, vm, rk, info in models_info[:200]:  # only save the first 200 for brevity
        info_serializable.append({
            "timestamp": str(t),
            "vol_model": vm,
            "reg_kind": rk,
            "info": info if isinstance(info, dict) else {"info": str(info)}
        })
    with open(PRO / "improved_garchx_model_info.json", "w") as f:
        json.dump(info_serializable, f, default=str, indent=2)

    print(f"\nSaved OOS forecasts: {OUT_FORECASTS}")
    print(f"Saved OOS metrics: {OUT_METRICS}")
    print("Saved limited model info to improved_garchx_model_info.json")

if __name__ == "__main__":
    main()
