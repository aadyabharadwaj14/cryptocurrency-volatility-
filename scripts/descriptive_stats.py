#!/usr/bin/env python
"""
Descriptive statistics and correlation analysis for
BTC returns and ETH graph features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PRO = Path("data/processed")

def main():
    path = PRO / "aligned_returns_features.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()

    # Drop missing values
    df = df.dropna()

    # --- 1️⃣ Summary statistics ---
    desc = df.describe().T
    print("\n=== Summary Statistics ===")
    print(desc)

    # --- 2️⃣ Correlation matrix ---
    corr = df.corr(method="spearman")
    print("\n=== Spearman Correlation Matrix ===")
    print(corr["ret"].sort_values(ascending=False))

    # --- 3️⃣ Visualizations ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df["ret"], bins=50, kde=True)
    plt.title("BTC Daily Returns Distribution (2024–2025)")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(PRO / "btc_returns_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Correlation Heatmap: BTC Returns & ETH Graph Features")
    plt.tight_layout()
    plt.savefig(PRO / "feature_correlation_heatmap.png", dpi=300)
    plt.close()

    # --- 4️⃣ Feature-Return relationships ---
    for col in [c for c in df.columns if c not in ["price", "ret"]]:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df["ret"], alpha=0.6)
        plt.title(f"BTC Return vs {col}")
        plt.xlabel(col)
        plt.ylabel("BTC Daily Return")
        plt.tight_layout()
        plt.savefig(PRO / f"ret_vs_{col}.png", dpi=300)
        plt.close()

    print("\n✅ Saved descriptive plots:")
    print(" - btc_returns_distribution.png")
    print(" - feature_correlation_heatmap.png")
    print(" - ret_vs_<feature>.png (for each ETH feature)")

if __name__ == "__main__":
    main()
