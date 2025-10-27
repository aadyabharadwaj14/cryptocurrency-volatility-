import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from arch import arch_model

class GarchXTwoStep:
    """
    Step 1: Fit GARCH(1,1) on returns to get σ_t^2 (baseline conditional variance).
    Step 2: Regress log(σ_t^2) on lagged exogenous features X_{t-k}.
    Predict: log(σ_t^2) = log(σ_t^2_GARCH) + γ_0 + γ^T X_{t-k}
    """

    def __init__(self, lag_k: int = 1, dist: str = "t"):
        self.lag_k = lag_k
        self.dist = dist
        self.arch_res_ = None
        self.scaler_ = None
        self.reg_ = None
        self.feature_names_ = None

    def fit(self, returns: pd.Series, X: pd.DataFrame):
        # Align returns and features (X is already lagged by align_features.py)
        r = returns.dropna()
        X = X.reindex(r.index)
        df = pd.concat([r.rename("ret"), X], axis=1).dropna()
        r = df["ret"]
        X = df.drop(columns=["ret"])

        # Step 1: GARCH on returns (scaled by 100)
        am = arch_model(r * 100.0, mean="Zero", vol="GARCH", p=1, q=1, dist=self.dist)
        self.arch_res_ = am.fit(disp="off")
        sigma2_base = (self.arch_res_.conditional_volatility ** 2) / (100.0 ** 2)

        # Step 2: Regress log(σ²) on features
        # Use residuals from log(σ²_GARCH) to avoid extreme adjustments
        log_sigma2 = np.log(sigma2_base.clip(lower=1e-12))
        
        self.feature_names_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X.values)
        Xz = self.scaler_.transform(X.values)
        
        # Use Ridge with small alpha for regularization to prevent extreme coefficients
        self.reg_ = Ridge(alpha=0.1).fit(Xz, log_sigma2.values)
        return self

    def predict_sigma2_in_sample(self, X: pd.DataFrame) -> pd.Series:
        """Return σ^2 adjusted by exogenous features."""
        # X is already lagged, so don't lag again
        # Base σ^2 from the fitted arch model
        sigma2_base = (self.arch_res_.conditional_volatility ** 2) / (100.0 ** 2)
        log_sigma2_base = np.log(sigma2_base.clip(lower=1e-12))
        
        # Align X with the GARCH fitted index
        X_aligned = X.reindex(sigma2_base.index)
        valid = X_aligned.notna().all(axis=1)
        
        if not valid.any():
            return sigma2_base
        
        # Get feature adjustment (this predicts log(σ²) directly)
        Xz = self.scaler_.transform(X_aligned.loc[valid, self.feature_names_].values)
        log_sigma2_pred = self.reg_.predict(Xz)
        
        # Compute adjustment as difference from baseline
        log_adj = log_sigma2_pred - log_sigma2_base.loc[valid].values
        
        # Cap adjustments to prevent explosions: limit to ±2 (exp(2) ≈ 7.4x, exp(-2) ≈ 0.14x)
        log_adj = np.clip(log_adj, -2, 2)
        
        # Apply adjustment
        sigma2_adjusted = sigma2_base.copy()
        sigma2_adjusted.loc[valid] = sigma2_base.loc[valid] * np.exp(log_adj)
        
        # Final safety clip
        sigma2_adjusted = sigma2_adjusted.clip(lower=1e-8, upper=1e2)
        
        return sigma2_adjusted

    @property
    def params_(self):
        return {
            "arch": self.arch_res_.params.to_dict() if self.arch_res_ else None,
            "gamma": dict(zip(self.feature_names_, self.reg_.coef_)) if self.reg_ else None,
            "gamma_intercept": float(self.reg_.intercept_) if self.reg_ else None,
        }