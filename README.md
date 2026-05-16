# Saddlepoint Approximation Engine

Lugannani-Rice saddlepoint approximation for exact tail probabilities and VaR of ETF portfolios. Uses empirical cumulant generating function (CGF) from rolling windows of the `P2SAMAPA/fi-etf-macro-signal-master-data` dataset.

## Features
- **1000× faster than Monte Carlo** for extreme tail estimation.
- **Machine‑precision accuracy** for smooth distributions.
- **Walk‑forward VaR** (95%, 97.5%, 99%) with Kupiec backtest.
- **Streamlit dashboard** for live monitoring.
- **Automated daily runs** via GitHub Actions, results pushed to `P2SAMAPA/p2-etf-saddle-point-approx-results`.

## Setup

1. Clone repo and install:
   ```bash
   pip install -r requirements.txt
