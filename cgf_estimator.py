import numpy as np
from scipy.optimize import root_scalar
from typing import Tuple

class CGFEstimator:
    """
    Empirical cumulant generating function (CGF) of portfolio returns.
    K(t) = log( E[exp(t * R)] )
    with small ridge regularisation on the second cumulant.
    """
    def __init__(self, returns: np.ndarray, ridge: float = 1e-6):
        self.returns = np.asarray(returns).flatten()
        self.n = len(self.returns)
        self.ridge = ridge
        
        # Precompute sorted returns for potential edge cases
        self.returns_sorted = np.sort(self.returns)
        self.min_ret = self.returns_sorted[0]
        self.max_ret = self.returns_sorted[-1]
    
    def K(self, t: float) -> float:
        """CGF: log(mean(exp(t * R)))"""
        if abs(t) < 1e-10:
            return 0.0
        max_exp = np.exp(t * self.returns)
        return np.log(np.mean(max_exp))
    
    def K1(self, t: float) -> float:
        """First derivative: E[R e^{tR}] / E[e^{tR}]"""
        if abs(t) < 1e-10:
            return np.mean(self.returns)
        exp_vals = np.exp(t * self.returns)
        mean_exp = np.mean(exp_vals)
        mean_rexp = np.mean(self.returns * exp_vals)
        return mean_rexp / mean_exp
    
    def K2(self, t: float) -> float:
        """Second derivative: Var[R under exponential tilting] + ridge"""
        if abs(t) < 1e-10:
            return np.var(self.returns) + self.ridge
        exp_vals = np.exp(t * self.returns)
        mean_exp = np.mean(exp_vals)
        mean_rexp = np.mean(self.returns * exp_vals)
        mean_r2exp = np.mean((self.returns**2) * exp_vals)
        var_tilt = (mean_r2exp / mean_exp) - (mean_rexp / mean_exp)**2
        return var_tilt + self.ridge
    
    def solve_saddlepoint(self, x: float, t0: float = 0.0) -> float:
        """
        Solve saddlepoint equation K'(s) = x.
        Returns s (the saddlepoint) or np.nan if fails.
        """
        def equation(s):
            return self.K1(s) - x
        
        # Bracket search (s can be negative or positive depending on x)
        # x must be between min(R) and max(R) for proper solution
        if x <= self.min_ret or x >= self.max_ret:
            return np.nan
        
        # Expand bracket
        s_low = -10.0
        s_high = 10.0
        f_low = equation(s_low)
        f_high = equation(s_high)
        
        # If bracket not valid, try to find where sign changes
        for _ in range(50):
            if f_low * f_high < 0:
                break
            if abs(f_low) < abs(f_high):
                s_low *= 1.5
                f_low = equation(s_low)
            else:
                s_high *= 1.5
                f_high = equation(s_high)
        else:
            # No sign change, try secant from t0
            try:
                sol = root_scalar(equation, x0=t0, method='secant', maxiter=50, rtol=1e-8)
                if sol.converged:
                    return sol.root
                else:
                    return np.nan
            except:
                return np.nan
        
        # Solve with Brent's method
        try:
            sol = root_scalar(equation, bracket=[s_low, s_high], method='brentq', rtol=1e-8, maxiter=50)
            if sol.converged:
                return sol.root
            else:
                return np.nan
        except:
            return np.nan
