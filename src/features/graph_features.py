import pandas as pd
import networkx as nx

def features_from_edges(df_edges: pd.DataFrame) -> dict:
    # df_edges columns: ["src","dst","value"]
    G = nx.DiGraph()
    for row in df_edges.itertuples(index=False):
        G.add_edge(row.src, row.dst, weight=row.value)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    avg_deg = sum(degs) / n if n else 0.0
    cc = nx.average_clustering(G.to_undirected()) if n > 1 else 0.0
    return {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": avg_deg,
        "avg_clustering": cc,
    }
