import pandas as pd
from arch import arch_model

def fit_garch11(returns: pd.Series, dist: str = "t"):
    """
    Fit GARCH(1,1) with zero mean. Returns the arch fit result object.
    NOTE: arch is numerically happier if returns are scaled by 100.
    """
    r = returns.dropna() * 100.0
    am = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
    res = am.fit(disp="off")
    return res

def one_step_forecast(res) -> pd.Series:
    """
    Produce in-sample one-step-ahead variance forecasts aligned to the sample index.
    Returns σ_t^2 (variance of returns*100); divide by 100^2 to unscale.
    """
    f = res.forecast(horizon=1, reindex=True)
    var = f.variance.iloc[:, 0]
    # unscale back (since we multiplied returns by 100, variance multiplies by 100^2)
    return var / (100.0 ** 2)
