import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.validation.walk_forward import rolling_windows, walk_forward_ml_signals

FEATURE_CFG = [
    {"name": "momentum", "params": {"window": 5}},
    {"name": "rsi", "params": {"window": 14}},
]
FEATURE_COLUMNS = ["momentum_window5", "rsi_window14"]


def make_random_prices(n=700, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * (1 + rng.normal(0.0005, 0.01, n)).cumprod()
    return pd.DataFrame({"Close": close}, index=dates)


def make_predictable_prices(n=700, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = np.zeros(n)
    for i in range(5, n):
        direction = 1 if returns[i-5:i].sum() > 0 else -1
        returns[i] = direction * 0.01 + rng.normal(0, 0.004)
    close = 100 * (1 + pd.Series(returns)).cumprod()
    return pd.DataFrame({"Close": close.values}, index=dates)


def test_output_shape_and_coverage():
    df = make_random_prices()
    n_windows = len(rolling_windows(len(df), 200, 100))
    result = walk_forward_ml_signals(df, FEATURE_CFG, FEATURE_COLUMNS, train_size=200, test_size=100)

    assert len(result) == n_windows * 100
    assert list(result.index) == list(df.index[200:200 + n_windows * 100])
    assert result["window"].tolist() == [w for w in range(n_windows) for _ in range(100)]
    assert set(result["signal"].unique()) <= {-1, 0, 1}


def test_future_data_does_not_change_past_signals():
    df = make_random_prices()
    full = walk_forward_ml_signals(df, FEATURE_CFG, FEATURE_COLUMNS, 200, 100)
    short = walk_forward_ml_signals(df.iloc[:500], FEATURE_CFG, FEATURE_COLUMNS, 200, 100)

    pd.testing.assert_series_equal(full.loc[short.index, "signal"], short["signal"], check_freq=False)


def test_model_does_better_than_random_on_learnable_data():
    df = make_predictable_prices()
    result = walk_forward_ml_signals(df, FEATURE_CFG, FEATURE_COLUMNS, train_size=200, test_size=100)

    from src.validation.ml_signals import make_next_day_direction_labels
    from src.features.library import compute_features

    featured_all = compute_features(df, FEATURE_CFG).dropna()
    actual = make_next_day_direction_labels(featured_all).reindex(result.index)
    predicted = result["signal"].map({1: 1.0, -1: 0.0, 0: None}).dropna()

    common = predicted.index.intersection(actual.dropna().index)
    accuracy = (predicted.loc[common] == actual.loc[common]).mean()
    n_traded = len(common)

    assert n_traded > 50, f"model traded on too few days to judge ({n_traded})"
    assert accuracy > 0.55, f"expected better-than-random accuracy on traded days, got {accuracy:.2f} over {n_traded} days"


def test_too_short_train_window_is_rejected():
    df = make_random_prices()
    try:
        walk_forward_ml_signals(df, FEATURE_CFG, FEATURE_COLUMNS, train_size=10, test_size=100)
    except ValueError:
        return
    raise AssertionError("a train window shorter than the longest lookback should raise ValueError")


def test_overlapping_step_is_rejected():
    df = make_random_prices()
    try:
        walk_forward_ml_signals(df, FEATURE_CFG, FEATURE_COLUMNS, 200, 100, step_size=50)
    except ValueError:
        return
    raise AssertionError("step_size != test_size should raise ValueError")


def test_data_too_short_for_one_window_is_rejected():
    df = make_random_prices(n=100)
    try:
        walk_forward_ml_signals(df, FEATURE_CFG, FEATURE_COLUMNS, train_size=200, test_size=100)
    except ValueError:
        return
    raise AssertionError("too little data for even one window should raise ValueError")


def test_threshold_reduces_trading():
    df = make_random_prices()
    loose = walk_forward_ml_signals(df, FEATURE_CFG, FEATURE_COLUMNS, 200, 100, threshold=0.0)
    strict = walk_forward_ml_signals(df, FEATURE_CFG, FEATURE_COLUMNS, 200, 100, threshold=0.2)

    loose_trades = (loose["signal"] != 0).sum()
    strict_trades = (strict["signal"] != 0).sum()
    assert strict_trades <= loose_trades, (
        f"a bigger threshold should trade less often or equal, got strict={strict_trades} > loose={loose_trades}"
    )


if __name__ == "__main__":
    test_output_shape_and_coverage()
    test_future_data_does_not_change_past_signals()
    test_model_does_better_than_random_on_learnable_data()
    test_too_short_train_window_is_rejected()
    test_overlapping_step_is_rejected()
    test_data_too_short_for_one_window_is_rejected()
    test_threshold_reduces_trading()
    print("WALK-FORWARD ML SIGNAL TESTS PASSED")