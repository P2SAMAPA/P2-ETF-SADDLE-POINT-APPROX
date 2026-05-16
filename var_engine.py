import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

from config import ROLLING_WINDOW, MIN_OBS_FOR_CGF, VAR_ALPHAS, CGF_RIDGE, SOLVER_TOL, SOLVER_MAX_ITER
from cgf_estimator import CGFEstimator
from saddlepoint import lugannani_rice, var_from_saddlepoint

class VarEngine:
    def __init__(self, portfolio_returns: pd.Series):
        """
        portfolio_returns: daily returns (pandas Series with DatetimeIndex)
        """
        self.returns = portfolio_returns.sort_index()
        self.dates = self.returns.index
    
    def rolling_forecast(self, window: int = ROLLING_WINDOW) -> pd.DataFrame:
        """
        Walk-forward VaR forecasts.
        Returns DataFrame with dates, actual returns, and VaR for each alpha.
        """
        results = []
        n = len(self.returns)
        
        for i in tqdm(range(window, n), desc="Rolling VaR"):
            train_dates = self.dates[i-window:i]
            test_date = self.dates[i]
            train_rets = self.returns[train_dates].dropna().values
            
            if len(train_rets) < MIN_OBS_FOR_CGF:
                # Not enough data: use empirical quantile
                var_forecasts = {}
                for alpha in VAR_ALPHAS:
                    var_forecasts[f"VaR_{int(alpha*100)}"] = -np.percentile(train_rets, alpha*100)
            else:
                cgf = CGFEstimator(train_rets, ridge=CGF_RIDGE)
                var_forecasts = {}
                for alpha in VAR_ALPHAS:
                    var_val = var_from_saddlepoint(
                        cgf, 
                        alpha=alpha,
                        max_iter=SOLVER_MAX_ITER
                    )
                    var_forecasts[f"VaR_{int(alpha*100)}"] = var_val
            
            row = {
                "date": test_date,
                "actual_return": self.returns.iloc[i]
            }
            row.update(var_forecasts)
            results.append(row)
        
        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        
        # Add violation indicators
        for alpha in VAR_ALPHAS:
            col = f"VaR_{int(alpha*100)}"
            df[f"violation_{int(alpha*100)}"] = (df["actual_return"] < -df[col]).astype(int)
        
        return df
    
    def tail_probability_forecast(self, loss_threshold: float, window: int = ROLLING_WINDOW) -> pd.Series:
        """
        For each day, estimate P(return < -loss_threshold) using saddlepoint.
        """
        probs = []
        dates_used = []
        n = len(self.returns)
        x = -loss_threshold   # we want P(R < x)
        
        for i in range(window, n):
            train_rets = self.returns.iloc[i-window:i].dropna().values
            if len(train_rets) < MIN_OBS_FOR_CGF:
                # empirical
                p = np.mean(train_rets <= x)
            else:
                cgf = CGFEstimator(train_rets, ridge=CGF_RIDGE)
                # If x is outside range, return extreme prob
                if x <= cgf.min_ret:
                    p = 0.0
                elif x >= cgf.max_ret:
                    p = 1.0
                else:
                    s = cgf.solve_saddlepoint(x)
                    if np.isnan(s):
                        p = np.mean(train_rets <= x)
                    else:
                        K = cgf.K(s)
                        K2 = cgf.K2(s)
                        tail, _ = lugannani_rice(s, x, K, K2)
                        p = 1 - tail
            probs.append(p)
            dates_used.append(self.returns.index[i])
        
        return pd.Series(probs, index=dates_used, name=f"P_loss_{loss_threshold*100:.1f}pct")
