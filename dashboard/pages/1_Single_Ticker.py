import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from dashboard.style import apply_shared_style, ACCENT, metric_color
from dashboard.pages_shared import STRATEGY_PRESETS, ML_FEATURE_CFG, ML_FEATURE_COLUMNS, build_base_config
from src.pipeline import run_walkforward_for_ticker, run_walkforward_ml_for_ticker

st.set_page_config(page_title="Single Ticker | Quant Pipeline", page_icon="chart", layout="wide")
apply_shared_style()

TICKERS = ["AAPL", "MSFT", "TSLA", "SNAP"]


@st.cache_data(show_spinner=False)
def cached_run(ticker, config_key, config):
    if config["kind"] == "rule":
        return run_walkforward_for_ticker(ticker, config["full_config"])
    else:
        return run_walkforward_ml_for_ticker(ticker, config["full_config"])


def build_config(kind, strategy_name, strategy_params, model_type, threshold,
                 train_size, test_size, initial_capital):
    base = build_base_config(train_size, test_size, initial_capital)
    if kind == "rule":
        preset = STRATEGY_PRESETS[strategy_name]
        base["features"] = preset["feature_cfg"](strategy_params)
        base["strategy"] = {"name": strategy_name, "params": strategy_params}
    else:
        base["features"] = ML_FEATURE_CFG
        base["ml"] = {
            "model_type": model_type,
            "feature_columns": ML_FEATURE_COLUMNS,
            "threshold": threshold,
        }
    return base


st.title("Single Ticker Walk-Forward")

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", TICKERS)
    kind_label = st.radio("Strategy type", ["Rule-based", "Machine Learning"])
    kind = "rule" if kind_label == "Rule-based" else "ml"

    strategy_name = None
    strategy_params = {}
    model_type = None
    threshold = 0.05

    if kind == "rule":
        strategy_name = st.selectbox("Strategy", list(STRATEGY_PRESETS.keys()))
        defaults = STRATEGY_PRESETS[strategy_name]["params"]
        st.caption("Parameters")
        for key, default_value in defaults.items():
            if isinstance(default_value, float):
                strategy_params[key] = st.number_input(key, value=default_value, step=0.01)
            else:
                strategy_params[key] = st.number_input(key, value=int(default_value), step=1)
    else:
        model_type = st.selectbox("Model", ["logistic", "gradient_boosting"])
        threshold = st.slider("Confidence threshold", 0.0, 0.3, 0.05, 0.01)
        if model_type == "gradient_boosting":
            st.caption(
                "Note: gradient boosting results can vary slightly between "
                "runs, even with the same settings, due to floating-point "
                "nondeterminism in how trees pick splits. Logistic regression "
                "doesn't have this issue."
            )

    st.caption("Walk-forward windows")
    train_size = st.number_input("Train size (days)", value=504, step=21)
    test_size = st.number_input("Test size (days)", value=126, step=21)

    st.caption("Capital")
    initial_capital = st.number_input("Initial capital (Rs)", value=100000, step=10000)

    run_clicked = st.button("Run", type="primary", use_container_width=True)

if run_clicked:
    config = build_config(kind, strategy_name, strategy_params, model_type, threshold,
                          train_size, test_size, initial_capital)
    config_key = (ticker, kind, str(strategy_params), model_type, threshold, train_size, test_size, initial_capital)

    with st.spinner(f"Running walk-forward for {ticker}..."):
        result = cached_run(ticker, config_key, {"kind": kind, "full_config": config})

    st.session_state["last_result"] = result
    st.session_state["last_ticker"] = ticker

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    ticker = st.session_state["last_ticker"]

    tab_names = ["Overview", "Per-Window Detail"]
    if "coefficients" in result:
        tab_names.append("Feature Importance")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        summary = result["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Sharpe", f"{summary['mean_sharpe']:.2f}")
        c2.metric("Std Sharpe", f"{summary['std_sharpe']:.2f}")
        c3.metric("% Positive Windows", f"{summary['pct_positive_sharpe_windows']*100:.0f}%")
        c4.metric("Worst Window", f"#{summary['worst_window']}")

        bt = result["backtest_result"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bt.index, y=bt["equity"], mode="lines",
            line=dict(color=ACCENT, width=2), name="Equity",
        ))
        fig.update_layout(
            title=f"{ticker} Equity Curve", template="plotly_dark",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=420, margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        per_window = result["per_window"]
        colors = [metric_color(v) for v in per_window["sharpe"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=per_window.index, y=per_window["sharpe"], marker_color=colors))
        fig2.update_layout(
            title="Sharpe by Window", template="plotly_dark",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=320, margin=dict(l=40, r=20, t=50, b=40),
            xaxis_title="Window", yaxis_title="Sharpe",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(per_window.round(4), use_container_width=True)

    if "coefficients" in result:
        with tabs[2]:
            coefs = result["coefficients"]
            mean_coefs = coefs.mean().sort_values()
            colors = [metric_color(v) for v in mean_coefs]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=mean_coefs.values, y=mean_coefs.index, orientation="h", marker_color=colors))
            fig3.update_layout(
                title="Mean Feature Weight Across Windows", template="plotly_dark",
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                height=420, margin=dict(l=160, r=20, t=50, b=40),
            )
            st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Set your options in the sidebar and click Run.")