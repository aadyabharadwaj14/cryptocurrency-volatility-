import pandas as pd
import networkx as nx
from pathlib import Path
from src.utils.io import RAW, PRO
from src.utils.logging_utils import get_logger

LOGGER = get_logger("graph_features")

WEI_PER_ETH = 10**18  # BigQuery/Kaggle `transactions.value` is in wei

def features_from_edges(df_edges: pd.DataFrame) -> dict:
    """
    df_edges columns: ['from_address','to_address','value_eth'] where value_eth is float ETH.
    """
    if df_edges.empty:
        return {
            "n_nodes": 0.0, "n_edges": 0.0, "total_volume": 0.0,
            "avg_degree": 0.0, "avg_clustering": 0.0
        }

    # Combine parallel edges by summing value
    grouped = (
        df_edges.groupby(["from_address", "to_address"], as_index=False)["value_eth"]
        .sum()
    )

    G = nx.DiGraph()
    for row in grouped.itertuples(index=False):
        G.add_edge(row.from_address, row.to_address, weight=float(row.value_eth))

    n = G.number_of_nodes()
    m = G.number_of_edges()
    total_volume = float(grouped["value_eth"].sum())

    Gu = G.to_undirected()
    avg_degree = (sum(dict(Gu.degree()).values()) / n) if n else 0.0
    avg_clustering = nx.average_clustering(Gu) if n > 1 else 0.0

    return {
        "n_nodes": float(n),
        "n_edges": float(m),
        "total_volume": total_volume,     # in ETH
        "avg_degree": float(avg_degree),
        "avg_clustering": float(avg_clustering),
    }

def main():
    in_file = RAW / "ethereum_transactions_clean.csv"  # or ethereum_transactions.csv if you used that name
    out_file = PRO / "ETH_graph_features_daily.csv"
    PRO.mkdir(parents=True, exist_ok=True)

    # Read robustly; force string for addresses, object for value, parse timestamps later
    usecols = ["from_address", "to_address", "value", "block_timestamp"]
    df = pd.read_csv(
        in_file,
        usecols=usecols,
        dtype={"from_address": "string", "to_address": "string", "value": "object"},
        low_memory=False,
    )

    # Parse timestamp → date bucket (daily)
    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["block_timestamp"])
    df["date"] = df["block_timestamp"].dt.floor("D")

    # Coerce value to numeric (handles strings); convert wei → ETH; drop non-positive/NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["value_eth"] = df["value"] / WEI_PER_ETH
    df = df.drop(columns=["value"])
    df = df.dropna(subset=["from_address", "to_address", "value_eth"])
    df = df[df["value_eth"] > 0]

    rows = []
    for dt, g in df.groupby("date", sort=True):
        feats = features_from_edges(g[["from_address", "to_address", "value_eth"]].rename(columns={"value_eth": "value_eth"}))
        feats["date"] = dt
        rows.append(feats)
        if len(rows) % 10 == 0:
            LOGGER.info(f"Processed {len(rows)} days so far...")

    feat_df = pd.DataFrame(rows).set_index("date").sort_index()
    feat_df.to_csv(out_file)
    LOGGER.info(f"✅ Saved {out_file} with {len(feat_df)} rows and columns {feat_df.columns.tolist()}.")

if __name__ == "__main__":
    main()
