#!/usr/bin/env python
"""
K-Means clustering on BTC volatility & ETH network features.
Groups trading days into volatility regimes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

PRO = Path("data/processed")

def main():
    path = PRO / "aligned_returns_features.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.dropna()

    # ----- 1️⃣ Feature construction -----
    df["volatility"] = df["ret"] ** 2   # realized variance
    features = ["volatility", "total_volume", "avg_degree", "avg_clustering"]
    X = np.log1p(df[features])          # stability scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----- 2️⃣ Choose K -----
    inertias = []
    K_range = range(2, 7)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(K_range, inertias, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(PRO / "kmeans_elbow_plot.png", dpi=300)
    plt.close()

    # ----- 3️⃣ Fit final K-Means -----
    k_opt = 3  # adjust after seeing elbow
    kmeans = KMeans(n_clusters=k_opt, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df["cluster"] = labels

    # ----- 4️⃣ Visualization -----
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=np.log1p(df["total_volume"]),
        y=np.log1p(df["volatility"]),
        hue=df["cluster"],
        palette="viridis",
        alpha=0.7
    )
    plt.title("K-Means Clusters: BTC Volatility vs ETH Volume")
    plt.xlabel("log(1 + Total Volume)")
    plt.ylabel("log(1 + BTC Volatility)")
    plt.tight_layout()
    plt.savefig(PRO / "kmeans_volatility_clusters.png", dpi=300)
    plt.close()

    # ----- 5️⃣ Cluster Summary -----
    summary = df.groupby("cluster")[features].mean()
    print("\n=== Cluster Centers (mean of each variable) ===")
    print(summary)

    df.to_csv(PRO / "clustered_features.csv")
    print("\n✅ Saved:")
    print(" - kmeans_elbow_plot.png")
    print(" - kmeans_volatility_clusters.png")
    print(" - clustered_features.csv")

if __name__ == "__main__":
    main()
