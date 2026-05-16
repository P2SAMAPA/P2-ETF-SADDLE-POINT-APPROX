import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import os
from config import HF_OUTPUT_REPO, UNIVERSES, ACTIVE_UNIVERSE, VAR_ALPHAS, HF_TOKEN

st.set_page_config(layout="wide")
st.title("📈 Saddlepoint Approximation Engine – VaR Dashboard")

@st.cache_data
def get_latest_run_folder():
    """Return the name of the most recent run subfolder in the HF repo."""
    api = HfApi()
    try:
        # List all files in the repo (recursively) to get top-level directories
        files = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset", token=HF_TOKEN if HF_TOKEN else None)
        # Find all top-level directories that match timestamp pattern YYYYMMDD_HHMMSS
        # They will appear as e.g., "20250321_143000/var_forecasts.parquet"
        dirs = set()
        for f in files:
            if '/' in f:
                dir_name = f.split('/')[0]
                # Check if dir_name looks like timestamp
                if len(dir_name) == 15 and dir_name[4] == dir_name[7] == '_' and dir_name[8:10].isdigit():
                    dirs.add(dir_name)
        if not dirs:
            return None
        # Sort descending (latest first)
        latest = sorted(dirs, reverse=True)[0]
        return latest
    except Exception as e:
        st.warning(f"Could not list repo files: {e}")
        return None

@st.cache_data
def load_latest_results():
    """Download the most recent var_forecasts and backtest_stats parquet files."""
    latest_run = get_latest_run_folder()
    if latest_run is None:
        return None, None
    try:
        var_df = pd.read_parquet(
            hf_hub_download(
                repo_id=HF_OUTPUT_REPO,
                filename=f"{latest_run}/var_forecasts.parquet",
                repo_type="dataset",
                token=HF_TOKEN if HF_TOKEN else None
            )
        )
        stats_df = pd.read_parquet(
            hf_hub_download(
                repo_id=HF_OUTPUT_REPO,
                filename=f"{latest_run}/backtest_stats.parquet",
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
