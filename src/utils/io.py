from pathlib import Path

DATA_DIR = Path("data")
RAW = DATA_DIR / "raw"
PRO = DATA_DIR / "processed"
EXT = DATA_DIR / "external"

RAW.mkdir(parents=True, exist_ok=True)
PRO.mkdir(parents=True, exist_ok=True)
EXT.mkdir(parents=True, exist_ok=True)
