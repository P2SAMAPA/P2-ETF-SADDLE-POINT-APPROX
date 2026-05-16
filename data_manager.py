import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
import os
from typing import List, Optional

from config import HF_TOKEN, DATA_REPO, ETF_TICKERS, MACRO_TICKERS, START_DATE, END_DATE

class DataManager:
    def __init__(self):
        if HF_TOKEN:
            login(HF_TOKEN)
        self.dataset = None
        self._load_data()
    
    def _load_data(self):
        print(f"Loading data from {DATA_REPO}...")
        ds = load_dataset(DATA_REPO, split="train", token=HF_TOKEN if HF_TOKEN else None)
        df = ds.to_pandas()
        
        # The dataset has no 'date' column; the index is the datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Create a 'date' column for easier manipulation
        df['date'] = df.index
        
        # Filter by date range
        if START_DATE:
            df = df[df['date'] >= START_DATE]
        if END_DATE:
            df = df[df['date'] <= END_DATE]
        
        # Find ETF and macro columns that actually exist
        etf_cols = [col for col in ETF_TICKERS if col in df.columns]
        macro_cols = [col for col in MACRO_TICKERS if col in df.columns]
        
        missing_etfs = set(ETF_TICKERS) - set(etf_cols)
        if missing_etfs:
            print(f"Warning: ETFs not found: {missing_etfs}")
        missing_macro = set(MACRO_TICKERS) - set(macro_cols)
        if missing_macro:
            print(f"Warning: Macro series not found: {missing_macro}")
        
        self.etf_returns = df[['date'] + etf_cols].copy()
        self.macro = df[['date'] + macro_cols].copy()
        
        # Fill missing values (pandas 2.0+ style)
        self.etf_returns = self.etf_returns.ffill().bfill()
        self.macro = self.macro.ffill().bfill()
        
        # Compute daily returns for each ETF
        for col in etf_cols:
            if col in self.etf_returns.columns and not col.endswith('_ret'):
                self.etf_returns[f'{col}_ret'] = self.etf_returns[col].pct_change()
        
        print(f"Loaded {len(self.etf_returns)} days, {len(etf_cols)} ETFs, {len(macro_cols)} macro series")
    
    def get_portfolio_returns(self, weights: Optional[List[float]] = None) -> pd.Series:
        etf_ret_cols = [f'{ticker}_ret' for ticker in ETF_TICKERS if f'{ticker}_ret' in self.etf_returns.columns]
        rets = self.etf_returns[['date'] + etf_ret_cols].copy()
        
        if weights is None:
            w = np.ones(len(etf_ret_cols)) / len(etf_ret_cols)
        else:
            w = np.array(weights)
            assert len(w) == len(etf_ret_cols), "Weights length mismatch"
        
        rets = rets.dropna()
        port_ret = rets[etf_ret_cols].dot(w)
        port_ret.index = rets['date']
        return port_ret
    
    def get_macro_series(self) -> pd.DataFrame:
        return self.macro.set_index('date')
    
    def get_dates(self) -> pd.DatetimeIndex:
        return self.etf_returns['date']
