#!/usr/bin/env python
import pandas as pd
from data_manager import DataManager
from var_engine import VarEngine
from backtest import backtest_summary
from results_uploader import upload_results
from config import PORTFOLIO_WEIGHTS, VAR_ALPHAS, ROLLING_WINDOW

def main():
    print("Loading data...")
    dm = DataManager()
    
    print("Computing portfolio returns...")
    port_returns = dm.get_portfolio_returns(weights=PORTFOLIO_WEIGHTS)
    
    print("Running rolling VaR forecasts (saddlepoint)...")
    engine = VarEngine(port_returns)
    var_forecasts = engine.rolling_forecast(window=ROLLING_WINDOW)
    
    print("Backtest diagnostics...")
    stats = backtest_summary(var_forecasts)
    
    print("Uploading results...")
    upload_results({
        "var_forecasts": var_forecasts,
        "backtest_stats": stats
    }, run_id=None)
    
    print("Done.")

if __name__ == "__main__":
    main()
