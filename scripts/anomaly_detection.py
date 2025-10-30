#!/usr/bin/env python
"""
Anomaly detection on BTC volatility using model residuals & unsupervised methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

PRO = Path("data/processed")

def main():
    # --- Load model outputs & features ---
    vol_df = pd.read_csv(PRO / "rolling_backtest_results.csv", parse_dates=["timestamp"]).set_index("timestamp")
    feats_df = pd.read_csv(PRO / "aligned_returns_features.csv", parse_dates=["timestamp"]).set_index("timestamp")
    df = vol_df.join(feats_df, how="inner").dropna()

    # --- Compute model residuals (realized - predicted GARCHX variance) ---
    df["residual"] = df["realized_var"] - df["garchx_var"]

    # --- Standardize features for anomaly model ---
    X = df[["residual", "realized_var", "total_volume", "avg_degree", "avg_clustering"]]
    X_scaled = StandardScaler().fit_transform(np.log1p(X))

    # --- Isolation Forest (unsupervised anomaly detection) ---
    iso = IsolationForest(contamination=0.03, random_state=42)
    df["anomaly_flag"] = iso.fit_predict(X_scaled)
    df["anomaly"] = df["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

    anomalies = df[df["anomaly"] == 1]

    # --- Visualization: Volatility anomalies over time ---
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["realized_var"], label="Realized Volatility", alpha=0.6)
    plt.scatter(anomalies.index, anomalies["realized_var"], color="red", s=25, label="Anomalies")
    plt.title("BTC Volatility Anomalies (Isolation Forest)")
    plt.xlabel("Date")
    plt.ylabel("Realized Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PRO / "volatility_anomalies_timeline.png", dpi=300)
    plt.close()

    # --- Heatmap of anomalous features ---
    if len(anomalies) > 0:
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            np.log1p(anomalies[["realized_var", "total_volume", "avg_degree", "avg_clustering"]]).corr(),
            annot=True, cmap="coolwarm", fmt=".2f", center=0
        )
        plt.title("Feature Correlations on Anomalous Days")
        plt.tight_layout()
        plt.savefig(PRO / "anomaly_feature_heatmap.png", dpi=300)
        plt.close()

    # --- Save results ---
    df.to_csv(PRO / "volatility_anomalies.csv")
    print(f"\n✅  Saved:")
    print(" - volatility_anomalies.csv")
    print(" - volatility_anomalies_timeline.png")
    print(" - anomaly_feature_heatmap.png (if anomalies found)")
    print(f"\nDetected {len(anomalies)} anomalous trading days.")

if __name__ == "__main__":
    main()
