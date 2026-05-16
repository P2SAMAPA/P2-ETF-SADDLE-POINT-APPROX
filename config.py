"""
Configuration for P2-ETF-SADDLE-POINT-APPROX engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-saddle-point-approx-results"

# --- Universe Definitions (same as your hybrid/VAE engines) ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM", "IWD"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# Which universe to use for the saddlepoint VaR engine
ACTIVE_UNIVERSE = "COMBINED"   # or "FI_COMMODITIES" / "EQUITY_SECTORS"

# --- Macro Features (available from 2008) ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Portfolio weighting scheme ---
PORTFOLIO_WEIGHTS = None   # None = equal weight across selected universe

# --- Saddlepoint Engine Parameters ---
ROLLING_WINDOW = 252          # days (1 trading year)
MIN_OBS_FOR_CGF = 250
SOLVER_TOL = 1e-8
SOLVER_MAX_ITER = 50

# VaR levels (tail probabilities)
VAR_ALPHAS = [0.01, 0.025, 0.05]   # 99%, 97.5%, 95%

# Regularization for CGF
CGF_RIDGE = 1e-6

# --- Date range (set to None to use full dataset) ---
START_DATE = None   # e.g., "2008-01-02"
END_DATE = None

# --- Results ---
UPLOAD_TO_HF = True
LOCAL_RESULTS_DIR = "results"

# --- Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# --- Run ID (timestamp) ---
TODAY = datetime.now().strftime("%Y-%m-%d")
