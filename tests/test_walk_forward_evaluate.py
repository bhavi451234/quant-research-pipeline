import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.validation.walk_forward import walk_forward_signals
from src.backtest.engine import BacktestEngine
from src.evaluation.metrics import walk_forward_evaluate, summarize_windows

FEATURE_CFG = [
    {"name": "sma", "params": {"window": 20}},
    {"name": "sma", "params": {"window": 50}},
]
PARAMS = {"fast_window": 20, "slow_window": 50}
METRIC_NAMES = ["sharpe", "sortino", "max_drawdown", "win_rate", "turnover", "cagr"]
EVAL_CONFIG = {"risk_free_rate_annual": 0.05}


def make_prices(n=700, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * (1 + rng.normal(0.0005, 0.01, n)).cumprod()
    return pd.DataFrame({"Close": close}, index=dates)


def run_chain(df, train_size=200, test_size=50):
    wf = walk_forward_signals(df, FEATURE_CFG, "sma_crossover", PARAMS, train_size, test_size)
    engine = BacktestEngine(initial_capital=100_000, transaction_cost_bps=5, slippage_bps=2, position_size_pct=1.0)
    bt = engine.run(wf)
    return bt, wf


def test_one_row_per_window():
    df = make_prices()
    bt, wf = run_chain(df)
    per_window = walk_forward_evaluate(bt, METRIC_NAMES, EVAL_CONFIG)

    assert list(per_window.index) == sorted(wf["window"].unique())
    assert list(per_window.columns) == METRIC_NAMES


def test_matches_evaluate_on_a_single_window():
    df = make_prices()
    bt, wf = run_chain(df)
    per_window = walk_forward_evaluate(bt, METRIC_NAMES, EVAL_CONFIG)

    from src.evaluation.metrics import evaluate
    window_0 = bt[bt["window"] == 0]
    expected = evaluate(window_0, METRIC_NAMES, EVAL_CONFIG)

    for name in METRIC_NAMES:
        a, b = per_window.loc[0, name], expected[name]
        if pd.isna(a) and pd.isna(b):
            continue
        assert abs(a - b) < 1e-9, f"{name}: {a} != {b}"


def test_summarize_windows_default_metric():
    df = make_prices()
    bt, _ = run_chain(df)
    per_window = walk_forward_evaluate(bt, METRIC_NAMES, EVAL_CONFIG)
    summary = summarize_windows(per_window)

    assert "mean_sharpe" in summary
    assert "std_sharpe" in summary
    assert "pct_positive_sharpe_windows" in summary
    assert summary["worst_window"] in per_window.index
    assert per_window.loc[summary["worst_window"], "sharpe"] == per_window["sharpe"].min()


def test_summarize_windows_other_metric():
    df = make_prices()
    bt, _ = run_chain(df)
    per_window = walk_forward_evaluate(bt, METRIC_NAMES, EVAL_CONFIG)
    summary = summarize_windows(per_window, metric="cagr")

    assert "mean_cagr" in summary
    assert "std_cagr" in summary


def test_pct_positive_is_a_fraction_between_0_and_1():
    df = make_prices()
    bt, _ = run_chain(df)
    per_window = walk_forward_evaluate(bt, METRIC_NAMES, EVAL_CONFIG)
    summary = summarize_windows(per_window)

    assert 0.0 <= summary["pct_positive_sharpe_windows"] <= 1.0


if __name__ == "__main__":
    test_one_row_per_window()
    test_matches_evaluate_on_a_single_window()
    test_summarize_windows_default_metric()
    test_summarize_windows_other_metric()
    test_pct_positive_is_a_fraction_between_0_and_1()
    print("WALK-FORWARD EVALUATE TESTS PASSED")