"""
Quick smoke test using synthetic price data, so the pipeline logic
(features -> strategy -> backtest -> evaluation) can be verified
without needing live network access to a data provider.
Run with: python tests/smoke_test.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features.library import compute_features
from src.strategies.library import compute_signal
from src.backtest.engine import BacktestEngine
from src.evaluation.metrics import evaluate, compare_train_test
from src.data.loader import train_test_split


def make_synthetic_ohlcv(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    returns = rng.normal(0.0004, 0.015, size=n)
    close = 100 * (1 + returns).cumprod()
    df = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, n)),
        "High": close * (1 + np.abs(rng.normal(0, 0.004, n))),
        "Low": close * (1 - np.abs(rng.normal(0, 0.004, n))),
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=dates)
    df.index.name = "date"
    return df


def main():
    raw = make_synthetic_ohlcv()

    feature_cfg = [
        {"name": "sma", "params": {"window": 20}},
        {"name": "sma", "params": {"window": 50}},
        {"name": "rsi", "params": {"window": 14}},
        {"name": "volatility", "params": {"window": 20}},
    ]
    featured = compute_features(raw, feature_cfg)
    print(f"[ok] features computed, columns: {list(featured.columns)}")

    split_date = featured.index[int(len(featured) * 0.7)]
    train, test = train_test_split(featured, str(split_date.date()))
    print(f"[ok] split at {split_date.date()}: train={len(train)} rows, test={len(test)} rows")

    train = train.copy()
    test = test.copy()
    train["signal"] = compute_signal(train, "sma_crossover", {"fast_window": 20, "slow_window": 50})
    test["signal"] = compute_signal(test, "sma_crossover", {"fast_window": 20, "slow_window": 50})
    print(f"[ok] signals computed, train signal distribution:\n{train['signal'].value_counts()}")

    engine = BacktestEngine(initial_capital=100_000, transaction_cost_bps=5, slippage_bps=2, position_size_pct=1.0)
    train_bt = engine.run(train)
    test_bt = engine.run(test)
    print(f"[ok] backtest run, train final equity: {train_bt['equity'].iloc[-1]:.2f}")
    print(f"[ok] backtest run, test final equity: {test_bt['equity'].iloc[-1]:.2f}")

    metric_names = ["sharpe", "sortino", "max_drawdown", "win_rate", "turnover", "cagr"]
    train_metrics = evaluate(train_bt, metric_names, {"risk_free_rate_annual": 0.05})
    test_metrics = evaluate(test_bt, metric_names, {"risk_free_rate_annual": 0.05})

    comparison = compare_train_test(train_metrics, test_metrics)
    print("\n=== Train vs Test comparison (synthetic data) ===")
    print(comparison.round(4))

    print("\nSMOKE TEST PASSED: full pipeline (features -> strategy -> backtest -> evaluation) runs end-to-end.")


if __name__ == "__main__":
    main()
