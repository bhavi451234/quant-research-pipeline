import numpy as np
import pandas as pd

from src.evaluation.metrics import alpha_beta


def _benchmark(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.01, n))


def test_strategy_equal_to_double_benchmark_has_beta_two_alpha_zero():
    bench = _benchmark()
    result = alpha_beta(2 * bench, bench, risk_free_rate_annual=0.0)
    assert abs(result["beta"] - 2.0) < 1e-9
    assert abs(result["alpha_annual"]) < 1e-9


def test_constant_daily_edge_shows_up_as_alpha():
    bench = _benchmark()
    strategy = 0.001 + 0.5 * bench
    result = alpha_beta(strategy, bench, risk_free_rate_annual=0.0)
    assert abs(result["beta"] - 0.5) < 1e-9
    assert abs(result["alpha_annual"] - 0.001 * 252) < 1e-9


def test_too_few_days_returns_nan():
    bench = _benchmark(n=10)
    result = alpha_beta(bench, bench)
    assert np.isnan(result["alpha_annual"]) and np.isnan(result["beta"])