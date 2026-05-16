import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
import os
from typing import List, Optional, Tuple
from datetime import datetime

from config import HF_TOKEN, DATA_REPO, ETF_TICKERS, MACRO_TICKERS, START_DATE, END_DATE

class DataManager:
    def __init__(self):
        if HF_TOKEN:
            login(HF_TOKEN)
        self.dataset = None
        self._load_data()
    
    def _load_data(self):
        """Load full dataset from Hugging Face."""
        print(f"Loading data from {DATA_REPO}...")
        ds = load_dataset(DATA_REPO, split="train", token=HF_TOKEN if HF_TOKEN else None)
        df = ds.to_pandas()
        
        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Filter date range
        if START_DATE:
            df = df[df["date"] >= START_DATE]
        if END_DATE:
            df = df[df["date"] <= END_DATE]
        
        # Separate ETF returns and macro
        etf_cols = [col for col in ETF_TICKERS if col in df.columns]
        macro_cols = [col for col in MACRO_TICKERS if col in df.columns]
        
        self.etf_returns = df[["date"] + etf_cols].copy()
        self.macro = df[["date"] + macro_cols].copy()
        
        # Fill missing values (forward fill, then back fill)
        self.etf_returns = self.etf_returns.fillna(method="ffill").fillna(method="bfill")
        self.macro = self.macro.fillna(method="ffill").fillna(method="bfill")
        
        # Compute daily returns for ETFs (if not already present)
        for col in etf_cols:
            if col in self.etf_returns.columns and not col.endswith("_ret"):
                self.etf_returns[f"{col}_ret"] = self.etf_returns[col].pct_change()
        
        print(f"Loaded {len(self.etf_returns)} days, {len(etf_cols)} ETFs, {len(macro_cols)} macro series")
        
    def get_portfolio_returns(self, weights: Optional[List[float]] = None) -> pd.Series:
        """Compute daily portfolio returns based on ETF weights."""
        etf_ret_cols = [f"{ticker}_ret" for ticker in ETF_TICKERS if f"{ticker}_ret" in self.etf_returns.columns]
        rets = self.etf_returns[["date"] + etf_ret_cols].copy()
        
        if weights is None:
            # Equal weight
            w = np.ones(len(etf_ret_cols)) / len(etf_ret_cols)
        else:
            w = np.array(weights)
            assert len(w) == len(etf_ret_cols), "Weights length mismatch"
        
        # Align dates and drop NaNs (first day of returns)
        rets = rets.dropna()
        portfolio_ret = rets[etf_ret_cols].dot(w)
        portfolio_ret.index = rets["date"]
        return portfolio_ret
    
    def get_macro_series(self) -> pd.DataFrame:
        """Return macro dataframe with date index."""
        macro_df = self.macro.set_index("date")
        return macro_df
    
    def get_dates(self) -> pd.DatetimeIndex:
        return self.etf_returns["date"]
