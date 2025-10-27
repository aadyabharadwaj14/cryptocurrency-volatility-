import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.io import RAW, PRO
from src.utils.logging_utils import get_logger

LOGGER = get_logger("dataset")

def compute_log_returns(price_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": price_series})
    df["ret"] = np.log(df["price"]).diff()
    df = df.dropna()
    return df

def main(asset="BTC"):
    price_file = RAW / f"{asset}_prices_coingecko_daily.csv"
    df = pd.read_csv(price_file, parse_dates=["timestamp"], index_col="timestamp")
    out = compute_log_returns(df["price"])
    out.to_csv(PRO / f"{asset}_returns_daily.csv")
    LOGGER.info(f"Wrote {PRO / f'{asset}_returns_daily.csv'} ({len(out)} rows).")

if __name__ == "__main__":
    main()
