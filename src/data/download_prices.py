import time
import requests
import pandas as pd
from pathlib import Path
from src.utils.io import RAW
from src.utils.logging_utils import get_logger

LOGGER = get_logger("prices")

ID_MAP = {"BTC": "bitcoin", "ETH": "ethereum"}

def fetch_coingecko_daily(asset: str = "BTC", vs_currency: str = "usd", days: str = "max") -> pd.DataFrame:
    coin_id = ID_MAP[asset.upper()]
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()["prices"]  # [ [ms, price], ... ]
    df = pd.DataFrame(data, columns=["ts_ms", "price"])
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.drop(columns=["ts_ms"]).set_index("timestamp").sort_index()
    return df

def main(asset="BTC"):
    df = fetch_coingecko_daily(asset)
    out = RAW / f"{asset}_prices_coingecko_daily.csv"
    df.to_csv(out)
    LOGGER.info(f"Saved {out} with {len(df)} rows.")

if __name__ == "__main__":
    main()
