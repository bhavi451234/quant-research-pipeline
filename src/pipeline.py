"""
Pipeline orchestrator.

Responsibility: wire together data -> features -> strategy -> backtest
-> evaluation for ONE ticker, entirely driven by a config dict. This
is the only file that's allowed to know about all the other layers -
everything else stays isolated so you can change one layer (say, swap
the strategy) without touching the others.
"""

import pandas as pd

from src.data.loader import load_ticker, train_test_split
from src.features.library import compute_features
from src.strategies.library import compute_signal
from src.backtest.engine import BacktestEngine
from src.evaluation.metrics import evaluate, compare_train_test, walk_forward_evaluate, summarize_windows
from src.validation.walk_forward import walk_forward_signals
from src.evaluation.metrics import evaluate, compare_train_test, walk_forward_evaluate, summarize_windows
from src.validation.walk_forward import walk_forward_signals, walk_forward_ml_signals


def run_pipeline_for_ticker(ticker: str, config: dict) -> dict:
    """
    Runs the full pipeline for a single ticker and returns a dict with:
      - train_backtest, test_backtest: full backtested DataFrames
      - train_metrics, test_metrics: computed metric dicts
      - comparison: the train-vs-test comparison table (the headline result)
    """
    data_cfg = config["data"]
    feature_cfg = config["features"]
    strategy_cfg = config["strategy"]
    backtest_cfg = config["backtest"]
    eval_cfg = config["evaluation"]

    # 1. Load raw data
    raw = load_ticker(ticker, data_cfg["start_date"], data_cfg["end_date"])

    # 2. Compute features on the FULL series first (features like SMA need
    #    history), then split - this avoids artificially truncating the
    #    warm-up window right at the split boundary.
    featured = compute_features(raw, feature_cfg)

    train, test = train_test_split(featured, data_cfg["train_test_split_date"])

    # 3. Compute strategy signal for train and test separately and explicitly.
    #    Nothing about the test set should ever influence the train run.
    train = train.copy()
    test = test.copy()
    train["signal"] = compute_signal(train, strategy_cfg["name"], strategy_cfg["params"])
    test["signal"] = compute_signal(test, strategy_cfg["name"], strategy_cfg["params"])

    # 4. Backtest each split independently with the SAME engine/config -
    #    identical cost assumptions on both sides, or the comparison is meaningless.
    engine = BacktestEngine(
        initial_capital=backtest_cfg["initial_capital"],
        transaction_cost_bps=backtest_cfg["transaction_cost_bps"],
        slippage_bps=backtest_cfg["slippage_bps"],
        position_size_pct=backtest_cfg["position_size_pct"],
    )
    train_bt = engine.run(train)
    test_bt = engine.run(test)

    # 5. Evaluate both
    train_metrics = evaluate(train_bt, eval_cfg["metrics"], eval_cfg)
    test_metrics = evaluate(test_bt, eval_cfg["metrics"], eval_cfg)

    comparison = compare_train_test(train_metrics, test_metrics)

    return {
        "ticker": ticker,
        "train_backtest": train_bt,
        "test_backtest": test_bt,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "comparison": comparison,
    }


def run_pipeline(config: dict) -> dict:
    """Runs the pipeline across every ticker in the config's universe."""
    results = {}
    for ticker in config["data"]["tickers"]:
        results[ticker] = run_pipeline_for_ticker(ticker, config)
    return results


def run_walkforward_for_ticker(ticker: str, config: dict) -> dict:
    """
    Runs walk-forward validation for a single ticker and returns a dict with:
      - backtest_result: the full stitched, backtested DataFrame (all windows)
      - per_window: one row of metrics per window (from walk_forward_evaluate)
      - summary: cross-window summary stats (from summarize_windows)

    Unlike run_pipeline_for_ticker, there is no train/test split date here -
    the config's `walk_forward` section (train_size, test_size) does that job
    instead, sliding across the whole history.
    """
    data_cfg = config["data"]
    feature_cfg = config["features"]
    strategy_cfg = config["strategy"]
    backtest_cfg = config["backtest"]
    eval_cfg = config["evaluation"]
    wf_cfg = config["walk_forward"]

    # 1. Load raw data - same loader as the single-split pipeline.
    raw = load_ticker(ticker, data_cfg["start_date"], data_cfg["end_date"])

    # 2. Walk the whole history: for each window, compute features on that
    #    window's block only, get signals, and keep the out-of-sample rows.
    #    This one call replaces steps 2-3 (features + split) of the
    #    single-split pipeline, since walk_forward_signals does both itself,
    #    once per window.
    signals = walk_forward_signals(
        raw, feature_cfg, strategy_cfg["name"], strategy_cfg["params"],
        train_size=wf_cfg["train_size"], test_size=wf_cfg["test_size"],
    )

    # 3. Backtest the stitched out-of-sample record ONCE, continuously -
    #    same engine, same cost assumptions as everywhere else in the project.
    engine = BacktestEngine(
        initial_capital=backtest_cfg["initial_capital"],
        transaction_cost_bps=backtest_cfg["transaction_cost_bps"],
        slippage_bps=backtest_cfg["slippage_bps"],
        position_size_pct=backtest_cfg["position_size_pct"],
    )
    backtest_result = engine.run(signals)

    # 4. Metrics per window, then a cross-window summary.
    per_window = walk_forward_evaluate(backtest_result, eval_cfg["metrics"], eval_cfg)
    summary = summarize_windows(per_window)

    return {
        "ticker": ticker,
        "backtest_result": backtest_result,
        "per_window": per_window,
        "summary": summary,
    }


def run_walkforward(config: dict) -> dict:
    """Runs walk-forward validation across every ticker in the config's universe."""
    results = {}
    for ticker in config["data"]["tickers"]:
        results[ticker] = run_walkforward_for_ticker(ticker, config)
    return results

def run_walkforward_ml_for_ticker(ticker: str, config: dict) -> dict:
    """
    Runs the ML walk-forward strategy for a single ticker: fits a fresh
    logistic regression at each window and predicts on that window's test
    rows, then backtests and evaluates the stitched out-of-sample record.
    Same output shape as run_walkforward_for_ticker, so results are directly
    comparable to the rule-based strategies.
    """
    data_cfg = config["data"]
    feature_cfg = config["features"]
    ml_cfg = config["ml"]
    backtest_cfg = config["backtest"]
    eval_cfg = config["evaluation"]
    wf_cfg = config["walk_forward"]

    raw = load_ticker(ticker, data_cfg["start_date"], data_cfg["end_date"])

    signals = walk_forward_ml_signals(
        raw, feature_cfg, ml_cfg["feature_columns"],
        train_size=wf_cfg["train_size"], test_size=wf_cfg["test_size"],
        threshold=ml_cfg.get("threshold", 0.05),
    )

    engine = BacktestEngine(
        initial_capital=backtest_cfg["initial_capital"],
        transaction_cost_bps=backtest_cfg["transaction_cost_bps"],
        slippage_bps=backtest_cfg["slippage_bps"],
        position_size_pct=backtest_cfg["position_size_pct"],
    )
    backtest_result = engine.run(signals)

    per_window = walk_forward_evaluate(backtest_result, eval_cfg["metrics"], eval_cfg)
    summary = summarize_windows(per_window)

    return {
        "ticker": ticker,
        "backtest_result": backtest_result,
        "per_window": per_window,
        "summary": summary,
    }


def run_walkforward_ml(config: dict) -> dict:
    """Runs the ML walk-forward strategy across every ticker in the config's universe."""
    results = {}
    for ticker in config["data"]["tickers"]:
        results[ticker] = run_walkforward_ml_for_ticker(ticker, config)
    return results



