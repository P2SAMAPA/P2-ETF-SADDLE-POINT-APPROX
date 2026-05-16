import pandas as pd
import numpy as np
from scipy.stats import chi2

def kupiec_test(violations: pd.Series, alpha: float) -> dict:
    n = len(violations)
    x = violations.sum()
    if n == 0:
        return {"LR": np.nan, "p_value": np.nan, "reject": False}
    
    p_hat = x / n
    if p_hat == 0 or p_hat == 1:
        LR = np.inf
        p_value = 0.0
        reject = True
    else:
        LR = -2 * (np.log((1-alpha)**(n-x) * alpha**x) - np.log((1-p_hat)**(n-x) * p_hat**x))
        p_value = 1 - chi2.cdf(LR, df=1)
        reject = p_value < 0.05
    
    return {
        "LR_stat": LR,
        "p_value": p_value,
        "reject_H0": reject,
        "actual_coverage": p_hat,
        "violations": x,
        "total_days": n,
        "violation_rate": p_hat
    }

def backtest_summary(var_df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for col in var_df.columns:
        if col.startswith("violation_"):
            alpha = int(col.split("_")[1]) / 100.0
            violations = var_df[col]
            result = kupiec_test(violations, alpha)
            result["VaR_level"] = f"{int(alpha*100)}%"
            summary.append(result)
    return pd.DataFrame(summary)
