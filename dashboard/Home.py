import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from dashboard.style import apply_shared_style

st.set_page_config(page_title="Quant Research Pipeline", page_icon="chart", layout="wide")
apply_shared_style()

st.title("Quant Research Pipeline")
st.markdown(
    """
    An interactive front end over the full backtesting pipeline: walk-forward
    validated strategies (rule-based and ML), compared head-to-head, and
    combined into real, dollar-allocated portfolios.

    **Use the sidebar to navigate:**
    - **Single Ticker** — pick a ticker and a strategy, see its walk-forward
      results
    - **Strategy Comparison** — run every strategy on one ticker side by side
    - **Portfolio Builder** — combine tickers with real rupee allocations
      into a single portfolio
    """
)