import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.features.library import FEATURE_REGISTRY


def make_prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * (1 + rng.normal(0.0005, 0.01, n)).cumprod()
    return pd.DataFrame({"Close": close}, index=dates)


FEATURE_PARAMS = {
    "sma": {"window": 20},
    "ema": {"span": 20},
    "rsi": {"window": 14},
    "volatility": {"window": 20},
    "momentum": {"window": 20},
    "bollinger_middle": {"window": 20},
    "bollinger_upper": {"window": 20, "num_std": 2.0},
    "bollinger_lower": {"window": 20, "num_std": 2.0},
}


def test_features_only_use_past_data():
    df = make_prices()
    cut = 200
    for name, params in FEATURE_PARAMS.items():
        func = FEATURE_REGISTRY[name]
        full = func(df, **params)
        truncated = func(df.iloc[:cut], **params)
        pd.testing.assert_series_equal(
            full.iloc[:cut], truncated, check_freq=False, obj=f"feature '{name}'"
        )


def test_detector_catches_a_leaky_feature():
    df = make_prices()
    cut = 200
    leaky = lambda d: d["Close"].shift(-1)
    full = leaky(df)
    truncated = leaky(df.iloc[:cut])
    try:
        pd.testing.assert_series_equal(full.iloc[:cut], truncated, check_freq=False)
    except AssertionError:
        return
    raise AssertionError("the leak detector failed to notice a leaky feature")


if __name__ == "__main__":
    test_features_only_use_past_data()
    test_detector_catches_a_leaky_feature()
    print("NO-LOOKAHEAD TESTS PASSED")