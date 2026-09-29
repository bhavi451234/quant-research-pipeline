import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features.library import compute_features
from src.strategies.library import compute_signal
from src.validation.walk_forward import rolling_windows, walk_forward_signals

FEATURE_CFG = [
    {"name": "sma", "params": {"window": 20}},
    {"name": "sma", "params": {"window": 50}},
]
PARAMS = {"fast_window": 20, "slow_window": 50}


def make_prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * (1 + rng.normal(0.0005, 0.01, n)).cumprod()
    return pd.DataFrame({"Close": close}, index=dates)


def test_matches_computing_features_once():
    df = make_prices()
    result = walk_forward_signals(df, FEATURE_CFG, "sma_crossover", PARAMS, train_size=100, test_size=50)

    featured_all = compute_features(df, FEATURE_CFG)
    expected = compute_signal(featured_all, "sma_crossover", PARAMS).loc[result.index]

    pd.testing.assert_series_equal(result["signal"], expected, check_names=False, check_freq=False)


def test_output_shape_and_coverage():
    df = make_prices()
    n_windows = len(rolling_windows(len(df), 100, 50))
    result = walk_forward_signals(df, FEATURE_CFG, "sma_crossover", PARAMS, train_size=100, test_size=50)

    assert len(result) == n_windows * 50
    assert list(result.index) == list(df.index[100:100 + n_windows * 50])
    assert result["window"].tolist() == [w for w in range(n_windows) for _ in range(50)]


def test_future_data_does_not_change_past_signals():
    df = make_prices()
    full = walk_forward_signals(df, FEATURE_CFG, "sma_crossover", PARAMS, 100, 50)
    short = walk_forward_signals(df.iloc[:350], FEATURE_CFG, "sma_crossover", PARAMS, 100, 50)

    pd.testing.assert_series_equal(full.loc[short.index, "signal"], short["signal"], check_freq=False)


def test_too_short_train_window_is_rejected():
    df = make_prices()
    try:
        walk_forward_signals(df, FEATURE_CFG, "sma_crossover", PARAMS, train_size=10, test_size=50)
    except ValueError:
        return
    raise AssertionError("a train window shorter than the longest lookback should raise ValueError")


def test_overlapping_step_is_rejected():
    df = make_prices()
    try:
        walk_forward_signals(df, FEATURE_CFG, "sma_crossover", PARAMS, 100, 50, step_size=10)
    except ValueError:
        return
    raise AssertionError("step_size != test_size should raise ValueError")


def test_data_too_short_for_one_window_is_rejected():
    df = make_prices(n=100)
    try:
        walk_forward_signals(df, FEATURE_CFG, "sma_crossover", PARAMS, train_size=100, test_size=50)
    except ValueError:
        return
    raise AssertionError("too little data for even one window should raise ValueError")


if __name__ == "__main__":
    test_matches_computing_features_once()
    test_output_shape_and_coverage()
    test_future_data_does_not_change_past_signals()
    test_too_short_train_window_is_rejected()
    test_overlapping_step_is_rejected()
    test_data_too_short_for_one_window_is_rejected()
    print("WALK-FORWARD SIGNAL TESTS PASSED")