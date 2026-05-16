import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
import os
from config import HF_OUTPUT_REPO, UNIVERSES, ACTIVE_UNIVERSE, VAR_ALPHAS, HF_TOKEN

st.set_page_config(layout="wide")
st.title("📈 Saddlepoint Approximation Engine – VaR Dashboard")

@st.cache_data
def load_latest_results():
    """Download most recent parquet files from HF repo."""
    try:
        # We assume the latest run folder is the one with the most recent timestamp
        # For simplicity, try to fetch a known file; if fails, show message.
        var_df = pd.read_parquet(
            hf_hub_download(
                repo_id=HF_OUTPUT_REPO,
                filename="var_forecasts.parquet",
                repo_type="dataset",
                token=HF_TOKEN if HF_TOKEN else None
            )
        )
        stats_df = pd.read_parquet(
            hf_hub_download(
                repo_id=HF_OUTPUT_REPO,
                filename="backtest_stats.parquet",
                repo_type="dataset",
                token=HF_TOKEN if HF_TOKEN else None
            )
        )
        return var_df, stats_df
    except Exception as e:
        st.warning(f"Could not load results from HF: {e}")
        return None, None

def main():
    st.subheader(f"Universe: {ACTIVE_UNIVERSE} – {len(UNIVERSES[ACTIVE_UNIVERSE])} ETFs")
    var_df, stats_df = load_latest_results()
    if var_df is None:
        st.info("No results loaded. Run the backtest script first.")
        return

    st.subheader("VaR Forecasts vs Actual Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=var_df.index, y=var_df["actual_return"], mode='lines', name='Actual Return'))
    for alpha in VAR_ALPHAS:
        col = f"VaR_{int(alpha*100)}"
        if col in var_df.columns:
            fig.add_trace(go.Scatter(x=var_df.index, y=-var_df[col], mode='lines', name=f'{int(alpha*100)}% VaR'))
    fig.update_layout(yaxis_title="Return", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Violations")
    for alpha in VAR_ALPHAS:
        col = f"violation_{int(alpha*100)}"
        if col in var_df.columns:
            vcount = var_df[col].sum()
            rate = vcount / len(var_df) * 100
            st.metric(f"{int(alpha*100)}% VaR violations", vcount, delta=f"{rate:.2f}% rate")

    if stats_df is not None:
        st.subheader("Kupiec Backtest")
        st.dataframe(stats_df)

if __name__ == "__main__":
    main()
