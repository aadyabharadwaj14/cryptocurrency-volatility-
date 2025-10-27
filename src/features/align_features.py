import pandas as pd
from src.utils.io import PRO
from src.utils.logging_utils import get_logger

LOGGER = get_logger("align")

def main():
    returns = pd.read_csv(PRO / "BTC_returns_daily.csv", parse_dates=["timestamp"]).set_index("timestamp")
    X = pd.read_csv(PRO / "ETH_graph_features_daily.csv", parse_dates=["date"]).set_index("date")

    X = X.asfreq("D")
    X_lag = X.shift(1)  # use t-1 features to predict t returns (causal)

    df = returns.join(X_lag, how="inner").sort_index()

    # Ensure the date is a proper column named 'timestamp'
    out = PRO / "aligned_returns_features.csv"
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df.to_csv(out, index=False)
    LOGGER.info(f"Wrote {out} with shape {df.shape}")

if __name__ == "__main__":
    main()
