import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
from scipy.stats import norm

# Import our modules
from data_manager import DataManager
from cgf_estimator import CGFEstimator
from saddlepoint import var_from_saddlepoint
from config import UNIVERSES, ACTIVE_UNIVERSE, VAR_ALPHAS, ROLLING_WINDOW, START_DATE, END_DATE

st.set_page_config(page_title="Saddlepoint VaR Engine", layout="wide")

# Cache expensive operations
@st.cache_resource
def get_data_manager():
    return DataManager()

@st.cache_data(ttl=3600)
def compute_portfolio_returns(universe_key):
    dm = get_data_manager()
    tickers = UNIVERSES[universe_key]
    # Filter available tickers
    available = [t for t in tickers if t in dm.df.columns]
    if not available:
        return None, None
    prices = dm.df[available].ffill().bfill()
    rets = prices.pct_change().dropna()
    # Equal weight portfolio
    weights = np.ones(len(available)) / len(available)
    port_ret = rets.dot(weights)
    return port_ret, available

@st.cache_data(ttl=3600)
def compute_rolling_var(returns, window=252, alphas=[0.01, 0.025, 0.05]):
    """Compute rolling VaR using saddlepoint approximation."""
    dates = returns.index
    n = len(returns)
    results = []
    for i in range(window, n):
        train = returns.iloc[i-window:i].values
        if len(train) < 250:
            var_vals = {f"VaR_{int(a*100)}": -np.percentile(train, a*100) for a in alphas}
        else:
            cgf = CGFEstimator(train, ridge=1e-6)
            var_vals = {}
            for a in alphas:
                var_vals[f"VaR_{int(a*100)}"] = var_from_saddlepoint(cgf, alpha=a, max_iter=50)
        results.append({
            "date": dates[i],
            "actual_return": returns.iloc[i],
            **var_vals
        })
    df = pd.DataFrame(results).set_index("date")
    for a in alphas:
        col = f"VaR_{int(a*100)}"
        df[f"violation_{int(a*100)}"] = (df["actual_return"] < -df[col]).astype(int)
    return df

@st.cache_data(ttl=3600)
def compute_etf_stats(universe_key):
    dm = get_data_manager()
    tickers = UNIVERSES[universe_key]
    available = [t for t in tickers if t in dm.df.columns]
    if not available:
        return pd.DataFrame()
    prices = dm.df[available].ffill().bfill()
    rets = prices.pct_change().dropna()
    stats = []
    for t in available:
        r = rets[t].dropna()
        if len(r) < 2:
            continue
        mean = r.mean() * 252  # annualized
        vol = r.std() * np.sqrt(252)
        sharpe = mean / vol if vol > 0 else 0
        # Empirical 99% VaR
        var99 = -np.percentile(r, 1)
        stats.append({
            "Ticker": t,
            "Annual Return (%)": mean * 100,
            "Annual Volatility (%)": vol * 100,
            "Sharpe Ratio": sharpe,
            "99% Daily VaR (%)": var99 * 100
        })
    df_stats = pd.DataFrame(stats).sort_values("Sharpe Ratio", ascending=False)
    return df_stats

def plot_portfolio_var(var_df, universe_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=var_df.index, y=var_df["actual_return"]*100, mode='lines', name='Portfolio Return (%)'))
    for alpha in VAR_ALPHAS:
        col = f"VaR_{int(alpha*100)}"
        fig.add_trace(go.Scatter(x=var_df.index, y=-var_df[col]*100, mode='lines', name=f'{int(alpha*100)}% VaR (%)'))
    fig.update_layout(
        title=f"Portfolio VaR – {universe_name}",
        xaxis_title="Date",
        yaxis_title="Percent",
        hovermode="x unified"
    )
    return fig

def main():
    st.title("📊 Saddlepoint Approximation Engine")
    st.markdown("Lugannani-Rice saddlepoint VaR for ETF portfolios")

    # Sidebar
    st.sidebar.header("Configuration")
    selected_universe = st.sidebar.selectbox(
        "Select Universe",
        list(UNIVERSES.keys()),
        index=list(UNIVERSES.keys()).index(ACTIVE_UNIVERSE)
    )
    rolling_window = st.sidebar.slider("Rolling Window (days)", 126, 504, ROLLING_WINDOW, step=63)

    # Load data
    with st.spinner("Loading data..."):
        port_ret, tickers = compute_portfolio_returns(selected_universe)
        if port_ret is None:
            st.error(f"No data available for universe {selected_universe}")
            return
        var_df = compute_rolling_var(port_ret, window=rolling_window, alphas=VAR_ALPHAS)
        etf_stats = compute_etf_stats(selected_universe)

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Portfolio VaR", "🏆 Top ETFs", "📋 Full Universe"])

    with tab1:
        st.plotly_chart(plot_portfolio_var(var_df, selected_universe), use_container_width=True)

        # Violation metrics
        cols = st.columns(len(VAR_ALPHAS))
        for i, alpha in enumerate(VAR_ALPHAS):
            col = f"violation_{int(alpha*100)}"
            violations = var_df[col].sum()
            total = len(var_df)
            rate = violations / total
            expected = alpha
            cols[i].metric(
                f"{int(alpha*100)}% VaR",
                f"{violations} violations",
                delta=f"{rate*100:.2f}% (exp. {expected*100:.1f}%)"
            )

    with tab2:
        st.subheader(f"Top 3 ETFs in {selected_universe}")
        top3 = etf_stats.head(3)
        if not top3.empty:
            st.dataframe(top3.style.format({
                "Annual Return (%)": "{:.2f}",
                "Annual Volatility (%)": "{:.2f}",
                "Sharpe Ratio": "{:.3f}",
                "99% Daily VaR (%)": "{:.2f}"
            }), use_container_width=True)

        # Bar chart of Sharpe ratios
        fig = px.bar(etf_stats.head(10), x="Ticker", y="Sharpe Ratio", title="Sharpe Ratio (Top 10)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader(f"All ETFs in {selected_universe} ({len(etf_stats)} assets)")
        st.dataframe(etf_stats.style.format({
            "Annual Return (%)": "{:.2f}",
            "Annual Volatility (%)": "{:.2f}",
            "Sharpe Ratio": "{:.3f}",
            "99% Daily VaR (%)": "{:.2f}"
        }), use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption(f"Data from {START_DATE or '2008'} to {END_DATE or 'latest'} | Rolling window: {rolling_window} days | VaR levels: {VAR_ALPHAS}")

if __name__ == "__main__":
    main()
