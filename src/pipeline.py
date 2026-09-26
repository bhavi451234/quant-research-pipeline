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
from src.evaluation.metrics import evaluate, compare_train_test


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
