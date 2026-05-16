import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
import os

from config import OUTPUT_REPO, ETF_TICKERS, VAR_ALPHAS

st.set_page_config(layout="wide")
st.title("📈 Saddlepoint Approximation Engine – VaR Dashboard")

@st.cache_data
def load_latest_results():
    """Download most recent parquet files from HF repo."""
    try:
        # List files in the repo (simplified: we assume a 'latest' symlink or just list)
        # For demo, we'll let user upload or we fetch known files
        # In production, use huggingface_hub.list_repo_files
        var_df = pd.read_parquet(hf_hub_download(repo_id=OUTPUT_REPO, filename="var_forecasts.parquet", token=os.environ.get("HF_TOKEN","")))
        stats_df = pd.read_parquet(hf_hub_download(repo_id=OUTPUT_REPO, filename="backtest_stats.parquet", token=os.environ.get("HF_TOKEN","")))
        return var_df, stats_df
    except:
        st.warning("No precomputed results found. Please run backtest first.")
        return None, None

def main():
    var_df, stats_df = load_latest_results()
    if var_df is None:
        st.info("No results loaded. Run the backtest script to generate forecasts.")
        return
    
    st.subheader("VaR Forecasts vs Actual Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=var_df.index, y=var_df["actual_return"], mode='lines', name='Actual Return'))
    for alpha in VAR_ALPHAS:
        col = f"VaR_{int(alpha*100)}"
        fig.add_trace(go.Scatter(x=var_df.index, y=-var_df[col], mode='lines', name=f'{int(alpha*100)}% VaR (negative)'))
    fig.update_layout(yaxis_title="Return", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Violations")
    for alpha in VAR_ALPHAS:
        col = f"violation_{int(alpha*100)}"
        if col in var_df.columns:
            vcount = var_df[col].sum()
            st.metric(f"{int(alpha*100)}% VaR violations", vcount, delta=f"{vcount/len(var_df)*100:.2f}% rate")
    
    if stats_df is not None:
        st.subheader("Kupiec Backtest")
        st.dataframe(stats_df)
    
    st.subheader("Loss Threshold Tail Probability (example: -2%)")
    # Option to compute dynamically would require running engine again. For now, static.

if __name__ == "__main__":
    main()
