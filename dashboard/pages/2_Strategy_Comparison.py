import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from dashboard.style import apply_shared_style, POSITIVE, NEGATIVE, metric_color
from src.pipeline import run_walkforward_for_ticker, run_walkforward_ml_for_ticker
from dashboard.pages_shared import (
    STRATEGY_PRESETS, ML_FEATURE_CFG, ML_FEATURE_COLUMNS,
    build_base_config, evaluation_summary, buy_and_hold,
)

st.set_page_config(page_title="Strategy Comparison | Quant Pipeline", page_icon="chart", layout="wide")
apply_shared_style()

TICKERS = ["AAPL", "MSFT", "TSLA", "SNAP", "JPM", "XOM", "JNJ"]

ALL_STRATEGIES = list(STRATEGY_PRESETS.keys()) + ["ml_logistic", "ml_gradient_boosting"]

CHART_METRICS = {
    "Total return (%)": "total_return_pct",
    "CAGR (%)": "cagr_pct",
    "Mean Sharpe": "mean_sharpe",
}


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
    n_windows = None
    progress = st.progress(0.0, text="Starting...")
    for i, strategy_key in enumerate(ALL_STRATEGIES):
        progress.progress((i) / len(ALL_STRATEGIES), text=f"Running {strategy_key}...")
        try:
            result = run_one_strategy(ticker, strategy_key, train_size, test_size, initial_capital)
            summary = result["summary"]
            overall = evaluation_summary(result, test_size)
            n_windows = len(result["per_window"])
            rows.append({
                "strategy": strategy_key,
                "total_return_pct": overall["total_return"] * 100,
                "cagr_pct": overall["cagr"] * 100,
                "worst_drawdown_pct": overall["worst_drawdown"] * 100,
                "mean_sharpe": summary["mean_sharpe"],
                "std_sharpe": summary["std_sharpe"],
                "pct_positive_windows": summary["pct_positive_sharpe_windows"],
                "worst_window": summary["worst_window"],
            })
        except Exception as e:
            errors.append(f"{strategy_key}: {e}")

    benchmarks = {}
    if n_windows:
        for name in (ticker, "SPY"):
            benchmarks[name] = buy_and_hold(name, n_windows, train_size, test_size)
    progress.progress(1.0, text="Done.")
    st.session_state["comparison_rows"] = rows
    st.session_state["comparison_errors"] = errors
    st.session_state["comparison_ticker"] = ticker
    st.session_state["comparison_benchmarks"] = benchmarks

if "comparison_rows" in st.session_state:
    rows = st.session_state["comparison_rows"]
    errors = st.session_state["comparison_errors"]
    ticker = st.session_state["comparison_ticker"]
    benchmarks = st.session_state["comparison_benchmarks"]

    if errors:
        with st.expander(f"{len(errors)} strategy(ies) failed to run - click to see why"):
            for e in errors:
                st.text(e)

    if rows:
        df = pd.DataFrame(rows).set_index("strategy")

        st.subheader("What just holding would have returned (same test days)")
        cols = st.columns(len(benchmarks) * 2)
        for i, (name, b) in enumerate(benchmarks.items()):
            label = f"{name} buy & hold"
            cols[i * 2].metric(f"{label}: total return", f"{b['total_return'] * 100:.1f}%")
            cols[i * 2 + 1].metric(f"{label}: CAGR", f"{b['cagr'] * 100:.1f}%")

        chart_label = st.radio("Chart metric", list(CHART_METRICS), horizontal=True)
        metric = CHART_METRICS[chart_label]
        ranked = df.sort_values(metric, ascending=False)

        st.subheader(f"{ticker}: {chart_label} by Strategy")
        colors = [metric_color(v) for v in ranked[metric]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ranked.index, y=ranked[metric], marker_color=colors))
        if metric != "mean_sharpe":
            hold = benchmarks[ticker]["total_return" if metric == "total_return_pct" else "cagr"] * 100
            fig.add_hline(y=hold, line_dash="dash", line_color="#E5E7EB",
                          annotation_text=f"{ticker} buy & hold", annotation_position="top left")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=420, margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Strategy", yaxis_title=chart_label,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full comparison table")
        st.dataframe(ranked.round(4), use_container_width=True)
        st.caption(
            "Total return and CAGR compound all test windows. Worst drawdown is the deepest "
            "drop inside any single window. Sharpe is the average across windows."
        )
    else:
        st.error("Every strategy failed to run. See the errors above.")
else:
    st.info("Set your options in the sidebar and click Run All Strategies.")