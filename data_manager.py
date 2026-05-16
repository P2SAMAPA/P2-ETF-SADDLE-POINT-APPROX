import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN if config.HF_TOKEN else None
    )
    df = pd.read_parquet(path)
    # Ensure datetime index
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    return df

class DataManager:
    def __init__(self):
        self.df = load_master_data()
        
        # Apply date filters if provided
        if config.START_DATE:
            self.df = self.df[self.df.index >= config.START_DATE]
        if config.END_DATE:
            self.df = self.df[self.df.index <= config.END_DATE]
        
        if len(self.df) == 0:
            raise ValueError("No data after filtering. Check START_DATE/END_DATE.")
        
        # Get tickers from the selected universe
        self.tickers = config.UNIVERSES[config.ACTIVE_UNIVERSE]
        # Keep only those that actually exist in the dataset
        self.available_tickers = [t for t in self.tickers if t in self.df.columns]
        
        # Macro columns
        self.macro_tickers = [m for m in config.MACRO_COLS if m in self.df.columns]
        
        print(f"Loaded {len(self.df)} days from {self.df.index[0].date()} to {self.df.index[-1].date()}")
        print(f"Universe: {config.ACTIVE_UNIVERSE} ({len(self.available_tickers)} tickers)")
        print(f"Macro found: {self.macro_tickers}")
    
    def get_portfolio_returns(self, weights=None):
        """Return daily simple returns of the portfolio (pandas Series with datetime index)."""
        prices = self.df[self.available_tickers].copy()
        prices = prices.ffill().bfill()
        rets = prices.pct_change().dropna()
        
        if weights is None:
            w = np.ones(len(self.available_tickers)) / len(self.available_tickers)
        else:
            w = np.array(weights)
            assert len(w) == len(self.available_tickers), "Weights length mismatch"
        
        port_ret = rets.dot(w)
        port_ret.name = "portfolio_return"
        return port_ret
    
    def get_macro_series(self):
        return self.df[self.macro_tickers].copy()
    
    def get_dates(self):
        return self.df.index
