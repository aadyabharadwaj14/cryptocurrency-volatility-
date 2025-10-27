import numpy as np
import pandas as pd

def realized_vol_squared(ret: pd.Series) -> pd.Series:
    """Use r_t^2 as a simple realized volatility proxy."""
    return (ret.astype(float)) ** 2

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = pd.Series(y_true).align(pd.Series(y_pred), join="inner")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def qlike(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    QLIKE for variance forecasts:
      L(σ_t^2, \hat{σ}_t^2) = σ_t^2 / \hat{σ}_t^2 - log(σ_t^2 / \hat{σ}_t^2) - 1
    Lower is better. Require positive predictions.
    """
    y_true, y_pred = pd.Series(y_true).align(pd.Series(y_pred), join="inner")
    eps = 1e-12
    ratio = (y_true.clip(lower=eps)) / (y_pred.clip(lower=eps))
    return float(np.mean(ratio - np.log(ratio) - 1))
