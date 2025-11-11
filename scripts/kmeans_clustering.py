#!/usr/bin/env python
"""
Enhanced K-Means clustering on BTC volatility & ETH network features.
Groups trading days into volatility regimes with improved feature engineering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PRO = Path("data/processed")

def engineer_features(df):
    """Enhanced feature engineering for clustering."""
    # Original volatility
    df["volatility"] = df["ret"] ** 2
    
    # Rolling volatility (7-day and 30-day)
    df["volatility_7d"] = df["volatility"].rolling(window=7).mean()
    df["volatility_30d"] = df["volatility"].rolling(window=30).mean()
    
    # Volatility regime (high/low relative to rolling mean)
    df["vol_regime"] = (df["volatility"] / df["volatility_30d"]).fillna(1)
    
    # Volume momentum
    df["volume_ma_7"] = df["total_volume"].rolling(window=7).mean()
    df["volume_momentum"] = (df["total_volume"] / df["volume_ma_7"]).fillna(1)
    
    # Network activity ratios
    df["degree_density"] = df["avg_degree"] * df["avg_clustering"]
    df["network_activity"] = df["total_volume"] * df["avg_degree"]
    
    return df

def remove_outliers(X, threshold=3):
    """Remove outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(X))
    return (z_scores < threshold).all(axis=1)

def find_optimal_k(X_scaled, k_range=range(2, 8)):
    """Find optimal K using both elbow method and silhouette analysis."""
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    # Create subplots for both methods
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Elbow plot
    ax1.plot(k_range, inertias, marker="o", linewidth=2, markersize=8)
    ax1.set_title("Elbow Method", fontsize=14)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_range, silhouette_scores, marker="s", linewidth=2, markersize=8, color='orange')
    ax2.set_title("Silhouette Analysis", fontsize=14)
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Average Silhouette Score")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PRO / "cluster_optimization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find optimal K (highest silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    max_silhouette = max(silhouette_scores)
    
    print(f"Optimal K based on silhouette score: {optimal_k} (score: {max_silhouette:.3f})")
    return optimal_k, silhouette_scores

def plot_silhouette_analysis(X_scaled, k_optimal):
    """Detailed silhouette analysis for optimal K."""
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette_avg = silhouette_score(X_scaled, labels)
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    
    for i in range(k_optimal):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(float(i) / k_optimal)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'Average Score: {silhouette_avg:.3f}')
    ax.set_xlabel('Silhouette Coefficient Values')
    ax.set_ylabel('Cluster Label')
    ax.set_title(f'Silhouette Analysis for K={k_optimal}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PRO / "silhouette_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return labels, silhouette_avg

def analyze_clusters(df, labels, features):
    """Comprehensive cluster analysis."""
    df_analysis = df.copy()
    df_analysis['cluster'] = labels
    
    # Cluster statistics
    cluster_stats = df_analysis.groupby('cluster')[features].agg(['mean', 'std', 'median'])
    cluster_counts = df_analysis['cluster'].value_counts().sort_index()
    
    print("\n=== Cluster Analysis ===")
    print(f"Cluster sizes: {dict(cluster_counts)}")
    print(f"\nCluster Statistics (mean values):")
    print(cluster_stats.xs('mean', level=1, axis=1).round(4))
    
    return cluster_stats, cluster_counts

def create_enhanced_visualizations(df, labels, features):
    """Create comprehensive visualization suite."""
    df_viz = df.copy()
    df_viz['cluster'] = labels
    
    # 1. PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(df[features]))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA plot
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('PCA Visualization of Clusters')
    plt.colorbar(scatter, ax=ax1)
    
    # Volatility vs Volume
    sns.scatterplot(data=df_viz, x=np.log1p(df_viz["total_volume"]), 
                   y=np.log1p(df_viz["volatility"]), hue="cluster", 
                   palette="viridis", alpha=0.7, ax=ax2)
    ax2.set_title("Volatility vs Volume")
    ax2.set_xlabel("log(1 + Total Volume)")
    ax2.set_ylabel("log(1 + Volatility)")
    
    # Network features
    sns.scatterplot(data=df_viz, x=df_viz["avg_degree"], 
                   y=df_viz["avg_clustering"], hue="cluster", 
                   palette="viridis", alpha=0.7, ax=ax3)
    ax3.set_title("Network Structure")
    ax3.set_xlabel("Average Degree")
    ax3.set_ylabel("Average Clustering")
    
    # Time series of clusters
    df_viz.reset_index(inplace=True)
    ax4.scatter(df_viz.index, df_viz['cluster'], c=df_viz['cluster'], 
               cmap='viridis', alpha=0.6)
    ax4.set_title("Cluster Evolution Over Time")
    ax4.set_xlabel("Time Index")
    ax4.set_ylabel("Cluster")
    
    plt.tight_layout()
    plt.savefig(PRO / "enhanced_cluster_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    path = PRO / "aligned_returns_features.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.dropna()

    # ----- 1️⃣ Enhanced Feature Engineering -----
    print("🔧 Engineering features...")
    df = engineer_features(df)
    
    # Enhanced feature set
    features = [
        "volatility", "volatility_7d", "vol_regime",
        "total_volume", "volume_momentum",
        "avg_degree", "avg_clustering", "degree_density", "network_activity"
    ]
    
    # Clean data and handle outliers
    df_clean = df[features].dropna()
    
    # Use RobustScaler for better outlier handling
    outlier_mask = remove_outliers(df_clean)
    df_clean = df_clean[outlier_mask]
    
    print(f"📊 Data shape after cleaning: {df_clean.shape}")
    
    # Feature scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(np.log1p(df_clean.abs() + 1e-8))

    # ----- 2️⃣ Optimal K Selection -----
    print("🎯 Finding optimal number of clusters...")
    k_optimal, silhouette_scores = find_optimal_k(X_scaled)

    # ----- 3️⃣ Final Clustering with Validation -----
    print(f"🎪 Performing clustering with K={k_optimal}...")
    labels, silhouette_avg = plot_silhouette_analysis(X_scaled, k_optimal)
    
    # Align labels with original dataframe
    df_aligned = df.loc[df_clean.index].copy()
    df_aligned['cluster'] = labels

    # ----- 4️⃣ Comprehensive Analysis -----
    cluster_stats, cluster_counts = analyze_clusters(df_aligned, labels, features)
    
    # ----- 5️⃣ Enhanced Visualizations -----
    print("📈 Creating visualizations...")
    create_enhanced_visualizations(df_clean, labels, features)

    # ----- 6️⃣ Export Results -----
    # Save detailed results
    results_df = df_aligned.copy()
    results_df.to_csv(PRO / "enhanced_clustered_features.csv")
    
    # Save cluster summary
    cluster_summary = pd.DataFrame({
        'cluster_size': cluster_counts,
        'silhouette_score': silhouette_avg
    })
    cluster_summary.to_csv(PRO / "cluster_summary.csv")

    print(f"\n✅ Clustering completed with silhouette score: {silhouette_avg:.3f}")
    print("📁 Files saved:")
    print(" - cluster_optimization.png")
    print(" - silhouette_detailed.png") 
    print(" - enhanced_cluster_analysis.png")
    print(" - enhanced_clustered_features.csv")
    print(" - cluster_summary.csv")

if __name__ == "__main__":
    main()
