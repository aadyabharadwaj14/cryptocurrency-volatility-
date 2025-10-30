#!/usr/bin/env python
"""
Final Visualization & Summary Dashboard
---------------------------------------
Combines descriptive stats, clustering, and anomaly detection outputs
into a single multi-panel figure and summary tables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PRO = Path("data/processed")

def main():
    # ---------- Load Data ----------
    desc_df = pd.read_csv(PRO / "aligned_returns_features.csv", parse_dates=["timestamp"]).set_index("timestamp")
    clusters = pd.read_csv(PRO / "clustered_features.csv", parse_dates=["timestamp"]).set_index("timestamp")
    anomalies = pd.read_csv(PRO / "volatility_anomalies.csv", parse_dates=["timestamp"]).set_index("timestamp")
    metrics = pd.read_csv(PRO / "true_garchx_in_sample_results.csv")

    # Merge cluster + anomaly flags
    df = clusters.join(anomalies[["anomaly"]], how="left").fillna({"anomaly": 0})

    # ---------- Figure Layout ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # === Panel 1: BTC Volatility Timeline + Anomalies ===
    axes[0, 0].plot(df.index, df["volatility"], color="steelblue", alpha=0.6, label="Volatility (r²)")
    axes[0, 0].scatter(df[df["anomaly"] == 1].index,
                       df[df["anomaly"] == 1]["volatility"],
                       color="red", s=25, label="Anomalies")
    axes[0, 0].set_title("BTC Volatility Timeline & Anomalies")
    axes[0, 0].set_xlabel("Date"); axes[0, 0].set_ylabel("Realized Variance")
    axes[0, 0].legend()

    # === Panel 2: Volatility–Volume Clusters ===
    sns.scatterplot(ax=axes[0, 1],
                    data=df,
                    x=np.log1p(df["total_volume"]),
                    y=np.log1p(df["volatility"]),
                    hue="cluster",
                    palette="viridis", alpha=0.7)
    axes[0, 1].set_title("Volatility vs Volume (K-Means Clusters)")
    axes[0, 1].set_xlabel("log(1 + Total Volume)")
    axes[0, 1].set_ylabel("log(1 + Volatility)")
    axes[0, 1].legend(title="Cluster")

    # === Panel 3: ETH Network Feature Correlations ===
    feat_cols = ["n_nodes", "n_edges", "total_volume", "avg_degree", "avg_clustering"]
    corr = desc_df[feat_cols + ["ret"]].corr(method="spearman")
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", center=0, ax=axes[1, 0])
    axes[1, 0].set_title("Correlation Matrix: BTC & ETH Network Features")

    # === Panel 4: Model Comparison (RMSE / QLIKE) ===
    metrics_plot = metrics.copy()
    axes[1, 1].bar(metrics_plot["model"], metrics_plot["QLIKE"], color=["#4c72b0", "#dd8452"])
    axes[1, 1].set_title("Model Comparison (QLIKE ↓ = Better)")
    axes[1, 1].set_ylabel("QLIKE Score")
    for i, v in enumerate(metrics_plot["QLIKE"]):
        axes[1, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    plt.suptitle("Trend Analysis in Cryptocurrency Volatility (BTC–ETH 2024–2025)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(PRO / "final_dashboard.png", dpi=300)
    plt.close()

    # ---------- Summary Tables ----------
    cluster_summary = df.groupby("cluster")[["volatility", "total_volume", "avg_degree", "avg_clustering"]].mean()
    anomaly_ratio = df["anomaly"].mean() * 100

    print("\n=== Cluster Summary (Mean Values) ===")
    print(cluster_summary.round(6))
    print(f"\nDetected anomalies: {anomaly_ratio:.2f}% of total trading days")
    print("\n=== GARCH Model Metrics ===")
    print(metrics.to_string(index=False))
    print("\n✅ Dashboard saved at data/processed/final_dashboard.png")

if __name__ == "__main__":
    main()
