import os
from datetime import datetime

# Hugging Face credentials
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Data repository (contains ETF + macro data, 2008–present)
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"

# Results repository (your team's output)
OUTPUT_REPO = "P2SAMAPA/p2-etf-saddle-point-approx-results"

# Asset universe (subset of liquid ETFs from the master data)
ETF_TICKERS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "USO",
    "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLK", "XLB",
    "VXX", "UUP", "FXE", "FXY"
]

# Portfolio weights: None = equal weight, or a list of same length
PORTFOLIO_WEIGHTS = None

# Macro tickers (present in master data, used for conditional analysis if needed)
MACRO_TICKERS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# Date range (full available)
START_DATE = "2008-01-02"
END_DATE = None          # None means latest available date in dataset

# Saddlepoint engine parameters
ROLLING_WINDOW = 504     # days (2 trading years)
MIN_OBS_FOR_CGF = 250    # minimum number of returns to fit CGF
SOLVER_TOL = 1e-8
SOLVER_MAX_ITER = 50

# VaR levels (alpha: tail probability)
VAR_ALPHAS = [0.01, 0.025, 0.05]   # 99%, 97.5%, 95% VaR

# Regularization for CGF estimation (ridge on the empirical cumulants)
CGF_RIDGE = 1e-6

# Backtest settings
REBALANCE_FREQ = "monthly"  # "daily", "weekly", "monthly"

# Upload settings
UPLOAD_TO_HF = True
LOCAL_RESULTS_DIR = "results"
