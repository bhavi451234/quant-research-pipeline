import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from dashboard.style import apply_shared_style, ACCENT, metric_color
from dashboard.pages_shared import STRATEGY_PRESETS, ML_FEATURE_CFG, ML_FEATURE_COLUMNS, build_base_config, union_feature_cfg
from src.pipeline import run_portfolio_single_strategy, run_portfolio_best_per_ticker

st.set_page_config(page_title="Portfolio Builder | Quant Pipeline", page_icon="chart", layout="wide")
apply_shared_style()

TICKERS = ["AAPL", "MSFT", "TSLA", "SNAP"]
STRATEGY_OPTIONS = list(STRATEGY_PRESETS.keys()) + ["ml_logistic", "ml_gradient_boosting"]
PORTFOLIO_METRICS = ["sharpe", "sortino", "max_drawdown", "cagr", "total_return"]
PER_TICKER_METRICS = PORTFOLIO_METRICS + ["win_rate", "turnover", "direction_accuracy"]


def strategy_entry(strategy_key, params=None):
    if strategy_key.startswith("ml_"):
        return {"type": "ml", "model_type": strategy_key.replace("ml_", ""),
               "feature_columns": ML_FEATURE_COLUMNS, "threshold": 0.05}
    preset = STRATEGY_PRESETS[strategy_key]
    return {"type": "strategy", "name": strategy_key, "params": params or preset["params"]}


@st.cache_data(show_spinner=False)
def cached_single(config_key, config):
    return run_portfolio_single_strategy(config)


@st.cache_data(show_spinner=False)
def cached_best_per_ticker(config_key, config):
    return run_portfolio_best_per_ticker(config)


st.title("Portfolio Builder")

with st.sidebar:
    st.header("Controls")
    selected_tickers = st.multiselect("Tickers", TICKERS, default=TICKERS)

    st.caption("Allocations (Rs)")
    allocations = {}
    for t in selected_tickers:
        allocations[t] = st.number_input(f"{t}", value=100000, step=10000, key=f"alloc_{t}")

    mode = st.radio("Strategy mode", ["Single strategy for all tickers", "Best strategy per ticker"])

    per_ticker_strategy = {}
    shared_strategy = None
    if mode == "Single strategy for all tickers":
        shared_strategy = st.selectbox("Strategy", STRATEGY_OPTIONS)
    else:
        st.caption("Pick each ticker's strategy")
        for t in selected_tickers:
            per_ticker_strategy[t] = st.selectbox(f"{t} strategy", STRATEGY_OPTIONS, key=f"strat_{t}")

    st.caption("Walk-forward windows")
    train_size = st.number_input("Train size (days)", value=504, step=21)
    test_size = st.number_input("Test size (days)", value=126, step=21)

    run_clicked = st.button("Build Portfolio", type="primary", use_container_width=True)

if run_clicked:
    if not selected_tickers:
        st.error("Select at least one ticker.")
    else:
        base = build_base_config(train_size, test_size, None)
        base["evaluation"]["metrics"] = PORTFOLIO_METRICS
        base["portfolio"] = {"allocations": allocations}

        if mode == "Single strategy for all tickers":
            base["features"] = union_feature_cfg([shared_strategy])
            if shared_strategy.startswith("ml_"):
                base["ml"] = {"model_type": shared_strategy.replace("ml_", ""),
                             "feature_columns": ML_FEATURE_COLUMNS, "threshold": 0.05}
            else:
                preset = STRATEGY_PRESETS[shared_strategy]
                base["strategy"] = {"name": shared_strategy, "params": preset["params"]}
            config_key = ("single", tuple(sorted(allocations.items())), shared_strategy, train_size, test_size)
            with st.spinner("Building portfolio..."):
                result = cached_single(config_key, base)
            st.session_state["portfolio_mode"] = "fair"
        else:
            base["features"] = union_feature_cfg(list(per_ticker_strategy.values()))
            base["portfolio"]["per_ticker_strategy"] = {
                t: strategy_entry(per_ticker_strategy[t]) for t in selected_tickers
            }
            config_key = ("best", tuple(sorted(allocations.items())),
                          tuple(sorted(per_ticker_strategy.items())), train_size, test_size)
            with st.spinner("Building portfolio..."):
                result = cached_best_per_ticker(config_key, base)
            st.session_state["portfolio_mode"] = "hindsight"

        st.session_state["portfolio_result"] = result

if "portfolio_result" in st.session_state:
    result = st.session_state["portfolio_result"]

    if st.session_state.get("portfolio_mode") == "hindsight":
        st.warning(
            "Each ticker's strategy was chosen manually here, if those choices were "
            "based on which strategy already scored best in earlier results, this "
            "portfolio reflects hindsight selection - treat its performance as an "
            "upper bound, not a fair prospective estimate."
        )

    tabs = st.tabs(["Portfolio Overview", "Per-Ticker Breakdown"])

    with tabs[0]:
        summary = result["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Sharpe", f"{summary['mean_sharpe']:.2f}")
        c2.metric("Std Sharpe", f"{summary['std_sharpe']:.2f}")
        c3.metric("% Positive Windows", f"{summary['pct_positive_sharpe_windows']*100:.0f}%")
        c4.metric("Worst Window", f"#{summary['worst_window']}")

        pr = result["portfolio_result"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pr.index, y=pr["equity"], mode="lines",
            line=dict(color=ACCENT, width=2), name="Portfolio Equity",
        ))
        fig.update_layout(
            title="Combined Portfolio Equity Curve", template="plotly_dark",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=420, margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(result["per_window"].round(4), use_container_width=True)

    with tabs[1]:
        from src.evaluation.metrics import walk_forward_evaluate, summarize_windows
        eval_cfg = {"risk_free_rate_annual": 0.05}
        for ticker, bt in result["ticker_backtests"].items():
            st.subheader(ticker)
            per_window = walk_forward_evaluate(bt, PER_TICKER_METRICS, eval_cfg)
            t_summary = summarize_windows(per_window)
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean Sharpe", f"{t_summary['mean_sharpe']:.2f}")
            c2.metric("Win Rate (avg)", f"{per_window['win_rate'].mean()*100:.0f}%")
            c3.metric("Direction Accuracy (avg)", f"{per_window['direction_accuracy'].mean()*100:.0f}%")
            st.dataframe(per_window.round(4), use_container_width=True)
else:
    st.info("Set your options in the sidebar and click Build Portfolio.")