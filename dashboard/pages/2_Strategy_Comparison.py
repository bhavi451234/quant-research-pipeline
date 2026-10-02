import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from dashboard.style import apply_shared_style, POSITIVE, NEGATIVE, metric_color
from src.pipeline import run_walkforward_for_ticker, run_walkforward_ml_for_ticker
from dashboard.pages_shared import STRATEGY_PRESETS, ML_FEATURE_CFG, ML_FEATURE_COLUMNS, build_base_config

st.set_page_config(page_title="Strategy Comparison | Quant Pipeline", page_icon="chart", layout="wide")
apply_shared_style()

TICKERS = ["AAPL", "MSFT", "TSLA", "SNAP"]

ALL_STRATEGIES = list(STRATEGY_PRESETS.keys()) + ["ml_logistic", "ml_gradient_boosting"]


@st.cache_data(show_spinner=False)
def run_one_strategy(ticker, strategy_key, train_size, test_size, initial_capital):
    base = build_base_config(train_size, test_size, initial_capital)
    if strategy_key.startswith("ml_"):
        model_type = strategy_key.replace("ml_", "")
        base["features"] = ML_FEATURE_CFG
        base["ml"] = {"model_type": model_type, "feature_columns": ML_FEATURE_COLUMNS, "threshold": 0.05}
        return run_walkforward_ml_for_ticker(ticker, base)
    else:
        preset = STRATEGY_PRESETS[strategy_key]
        base["features"] = preset["feature_cfg"](preset["params"])
        base["strategy"] = {"name": strategy_key, "params": preset["params"]}
        return run_walkforward_for_ticker(ticker, base)


st.title("Strategy Comparison")
st.caption("Runs every strategy, rule-based and ML, on one ticker, using each strategy's default parameters.")

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", TICKERS)
    train_size = st.number_input("Train size (days)", value=504, step=21)
    test_size = st.number_input("Test size (days)", value=126, step=21)
    initial_capital = st.number_input("Initial capital (Rs)", value=100000, step=10000)
    run_clicked = st.button("Run All Strategies", type="primary", use_container_width=True)

if run_clicked:
    rows = []
    errors = []
    progress = st.progress(0.0, text="Starting...")
    for i, strategy_key in enumerate(ALL_STRATEGIES):
        progress.progress((i) / len(ALL_STRATEGIES), text=f"Running {strategy_key}...")
        try:
            result = run_one_strategy(ticker, strategy_key, train_size, test_size, initial_capital)
            summary = result["summary"]
            rows.append({
                "strategy": strategy_key,
                "mean_sharpe": summary["mean_sharpe"],
                "std_sharpe": summary["std_sharpe"],
                "pct_positive_windows": summary["pct_positive_sharpe_windows"],
                "worst_window": summary["worst_window"],
            })
        except Exception as e:
            errors.append(f"{strategy_key}: {e}")
    progress.progress(1.0, text="Done.")
    st.session_state["comparison_rows"] = rows
    st.session_state["comparison_errors"] = errors
    st.session_state["comparison_ticker"] = ticker

if "comparison_rows" in st.session_state:
    rows = st.session_state["comparison_rows"]
    errors = st.session_state["comparison_errors"]
    ticker = st.session_state["comparison_ticker"]

    if errors:
        with st.expander(f"{len(errors)} strategy(ies) failed to run - click to see why"):
            for e in errors:
                st.text(e)

    if rows:
        df = pd.DataFrame(rows).set_index("strategy").sort_values("mean_sharpe", ascending=False)

        st.subheader(f"{ticker}: Mean Sharpe by Strategy")
        colors = [metric_color(v) for v in df["mean_sharpe"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df["mean_sharpe"], marker_color=colors))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=420, margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Strategy", yaxis_title="Mean Sharpe",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full comparison table")
        st.dataframe(df.round(4), use_container_width=True)
    else:
        st.error("Every strategy failed to run. See the errors above.")
else:
    st.info("Set your options in the sidebar and click Run All Strategies.")