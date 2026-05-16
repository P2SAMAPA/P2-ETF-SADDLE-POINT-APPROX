import pandas as pd
import numpy as np
from tqdm import tqdm
from config import ROLLING_WINDOW, MIN_OBS_FOR_CGF, VAR_ALPHAS, CGF_RIDGE, SOLVER_MAX_ITER
from cgf_estimator import CGFEstimator
from saddlepoint import var_from_saddlepoint

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
        Returns DataFrame with datetime index, actual returns, and VaR for each alpha.
        """
        results = []
        n = len(self.returns)
        
        for i in tqdm(range(window, n), desc="Rolling VaR"):
            train_rets = self.returns.iloc[i-window:i].dropna().values
            test_date = self.returns.index[i]
            test_ret = self.returns.iloc[i]
            
            if len(train_rets) < MIN_OBS_FOR_CGF:
                var_forecasts = {}
                for alpha in VAR_ALPHAS:
                    var_forecasts[f"VaR_{int(alpha*100)}"] = -np.percentile(train_rets, alpha*100)
            else:
                cgf = CGFEstimator(train_rets, ridge=CGF_RIDGE)
                var_forecasts = {}
                for alpha in VAR_ALPHAS:
                    var_val = var_from_saddlepoint(cgf, alpha=alpha, max_iter=SOLVER_MAX_ITER)
                    var_forecasts[f"VaR_{int(alpha*100)}"] = var_val
            
            row = {"actual_return": test_ret}
            row.update(var_forecasts)
            row["date"] = test_date
            results.append(row)
        
        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        
        for alpha in VAR_ALPHAS:
            col = f"VaR_{int(alpha*100)}"
            df[f"violation_{int(alpha*100)}"] = (df["actual_return"] < -df[col]).astype(int)
        
        return df
