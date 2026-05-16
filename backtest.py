import pandas as pd
import numpy as np
from scipy.stats import chi2

def kupiec_test(violations: pd.Series, alpha: float) -> dict:
    """
    Kupiec unconditional coverage test.
    violations: binary series (1 if violation)
    alpha: nominal VaR level (e.g., 0.01 for 99% VaR)
    """
    n = len(violations)
    x = violations.sum()
    if n == 0:
        return {"LR_stat": np.nan, "p_value": np.nan, "reject_H0": False, "actual_coverage": np.nan}
    
    p_hat = x / n
    # Avoid log(0) by using a small epsilon if p_hat is 0 or 1
    if p_hat == 0:
        p_hat_adj = 1e-10
    elif p_hat == 1:
        p_hat_adj = 1 - 1e-10
    else:
        p_hat_adj = p_hat
    
    try:
        LR = -2 * (np.log((1-alpha)**(n-x) * alpha**x) - np.log((1-p_hat_adj)**(n-x) * p_hat_adj**x))
    except:
        LR = np.inf
    
    p_value = 1 - chi2.cdf(LR, df=1)
    reject = p_value < 0.05
    return {"LR_stat": LR, "p_value": p_value, "reject_H0": reject, "actual_coverage": p_hat}

def backtest_summary(var_df: pd.DataFrame) -> pd.DataFrame:
    """Produce backtest statistics for each VaR level."""
    summary = []
    for col in var_df.columns:
        if col.startswith("violation_"):
            alpha_str = col.split("_")[1]
            alpha = int(alpha_str) / 100.0
            violations = var_df[col]
            result = kupiec_test(violations, alpha)
            result["VaR_level"] = f"{int(alpha*100)}%"
            result["violations"] = violations.sum()
            result["total_days"] = len(violations)
            result["violation_rate"] = violations.mean()
            summary.append(result)
    return pd.DataFrame(summary)
