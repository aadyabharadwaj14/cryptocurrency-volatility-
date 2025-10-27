#!/usr/bin/env bash
set -e
source .venv/bin/activate 2>/dev/null || true
python -m src.data.download_prices
python -m src.data.make_dataset
# (add on-chain download + features when ready)
