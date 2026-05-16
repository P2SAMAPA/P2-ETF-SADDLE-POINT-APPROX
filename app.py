import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import os
import re
from config import HF_OUTPUT_REPO, UNIVERSES, ACTIVE_UNIVERSE, VAR_ALPHAS, HF_TOKEN

st.set_page_config(layout="wide")
st.title("📈 Saddlepoint Approximation Engine – VaR Dashboard")

@st.cache_data(ttl=3600)
def get_latest_run_folder():
    """Get the most recent timestamped run folder from the HF dataset."""
    if not HF_TOKEN:
        return None
    api = HfApi()
    try:
        # List all files in the dataset
        files = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset", token=HF_TOKEN)
        # Extract folder names that look like YYYYMMDD_HHMMSS
        run_folders = set()
        for f in files:
            match = re.match(r"(\d{8}_\d{6})/", f)
            if match:
                run_folders.add(match.group(1))
        if not run_folders:
            return None
        # Sort descending and take the latest
        latest = sorted(run_folders, reverse=True)[0]
        return latest
    except Exception as e:
        st.warning(f"Could not list repo files: {e}")
        return None

@st.cache_data
def load_results(run_folder):
    """Download parquet files from a specific run folder."""
    try:
        var_df = pd.read_parquet(
            hf_hub_download(
                repo_id=HF_OUTPUT_REPO,
                filename=f"{run_folder}/var_forecasts.parquet",
                repo_type="dataset",
                token=HF_TOKEN if HF_TOKEN else None
            )
        )
        stats_df = pd.read_parquet(
            hf_hub_download(
                repo_id=HF_OUTPUT_REPO,
                filename=f"{run_folder}/backtest_stats.parquet",
                repo_type="dataset",
                token=HF_TOKEN if HF_TOKEN else None
            )
        )
        return var_df, stats_df
    except Exception as e:
        st.warning(f"Could not load results from {run_folder}: {e}")
        return None, None

def main():
    st.subheader(f"Universe: {ACTIVE_UNIVERSE} – {len(UNIVERSES[ACTIVE_UNIVERSE])} ETFs")
    
    latest_run = get_latest_run_folder()
    if latest_run is None:
        st.info("No results found in the HF repository. Please run the backtest script first.")
        return
    
    st.write(f"Loading results from run: `{latest_run}`")
    var_df, stats_df = load_results(latest_run)
    
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
