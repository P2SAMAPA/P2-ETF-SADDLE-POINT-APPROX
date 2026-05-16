import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional

def lugannani_rice(
    s: float,
    x: float,
    K: float,
    K2: float,
    n: int = 1
) -> Tuple[float, float]:
    """
    Lugannani-Rice formula for tail probability P(R > x) and PDF.
    s: saddlepoint solving K'(s)=x
    x: quantile
    K: K(s)
    K2: K''(s)
    n: sample size (not used if we treat as one observation, but kept for scaling)
    
    Returns:
        (tail_prob, density_approx)
    """
    # w = sign(s) * sqrt(2*(s*x - K))
    if s == 0:
        w = 0.0
    else:
        w = np.sqrt(2 * (s * x - K)) * np.sign(s)
    
    # v = s * sqrt(K2)
    v = s * np.sqrt(K2)
    
    # Normal CDF and PDF
    Phi_w = norm.cdf(w)
    phi_w = norm.pdf(w)
    
    # Lugannani-Rice tail probability
    if abs(v) < 1e-8:
        # Edge case: use first-order expansion
        tail = 1 - Phi_w
    else:
        tail = 1 - Phi_w + phi_w * (1/v - 1/w)
    
    # Density approximation
    density = np.exp(K - s*x) / np.sqrt(2 * np.pi * K2)
    
    # Clip probabilities
    tail = np.clip(tail, 0.0, 1.0)
    return tail, density

def var_from_saddlepoint(
    cgf_estimator,
    alpha: float,
    x0: float = 0.0,
    max_iter: int = 30
) -> float:
    """
    Compute VaR at tail probability alpha (e.g. 0.01 for 99% VaR)
    by solving P(R < -VaR) = alpha  (or equivalently P(R > VaR) = 1-alpha)
    We work with lower tail: P(R <= q) = alpha.
    """
    # We need to find q such that P(R <= q) = alpha.
    # Using Lugannani-Rice: tail = 1 - Phi(w) + phi(w)(1/v - 1/w) gives P(R > x).
    # So P(R <= x) = 1 - tail.
    # We'll solve for x = -VaR.
    
    # Use quantile of normal as initial guess
    q0 = np.percentile(cgf_estimator.returns, alpha * 100)
    
    def objective(q):
        # For given q, compute P(R <= q) using LR
        # If q is outside range, return 0 or 1
        if q <= cgf_estimator.min_ret:
            return alpha - 0.0   # probability is 0
        if q >= cgf_estimator.max_ret:
            return alpha - 1.0   # probability is 1
        
        s = cgf_estimator.solve_saddlepoint(q, t0=0.0)
        if np.isnan(s):
            # Fallback to empirical CDF
            p = np.mean(cgf_estimator.returns <= q)
            return alpha - p
        
        K = cgf_estimator.K(s)
        K2 = cgf_estimator.K2(s)
        tail, _ = lugannani_rice(s, q, K, K2)
        p_less = 1 - tail
        return alpha - p_less
    
    # Simple bisection search on q
    q_low = cgf_estimator.min_ret - 1e-6
    q_high = cgf_estimator.max_ret + 1e-6
    f_low = objective(q_low)
    f_high = objective(q_high)
    
    # Expand if needed
    for _ in range(20):
        if f_low * f_high < 0:
            break
        if abs(f_low) < abs(f_high):
            q_low -= 0.1 * (q_high - q_low)
            f_low = objective(q_low)
        else:
            q_high += 0.1 * (q_high - q_low)
            f_high = objective(q_high)
    else:
        # Fallback to empirical
        return -np.percentile(cgf_estimator.returns, alpha * 100)
    
    # Brent method from scipy?
    # Simple bisection
    for _ in range(max_iter):
        q_mid = (q_low + q_high) / 2
        f_mid = objective(q_mid)
        if abs(f_mid) < 1e-6:
            break
        if f_low * f_mid < 0:
            q_high = q_mid
            f_high = f_mid
        else:
            q_low = q_mid
            f_low = f_mid
    
    var = -q_mid
    return max(var, 0.0)   # VaR is positive number
