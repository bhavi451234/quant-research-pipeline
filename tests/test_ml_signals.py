import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.validation.ml_signals import make_next_day_direction_labels, fit_and_predict_logistic


def test_labels_shift_correctly():
    df = pd.DataFrame({"Close": [10, 12, 11, 15, 14]})
    labels = make_next_day_direction_labels(df)

    # day0->day1: 10->12, up   -> 1
    # day1->day2: 12->11, down -> 0
    # day2->day3: 11->15, up   -> 1
    # day3->day4: 15->14, down -> 0
    # day4: no next day        -> NaN
    expected = pd.Series([1.0, 0.0, 1.0, 0.0, np.nan])
    pd.testing.assert_series_equal(labels, expected, check_names=False)


def make_predictable_prices(n=300, seed=0):
    """
    Prices with a genuine, learnable pattern: whether the last 5 days'
    returns summed positive or negative determines tomorrow's direction,
    plus a bit of noise. This gives the model something real to find, so
    we can tell "the pipeline works" apart from "there's nothing to learn."
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = np.zeros(n)
    for i in range(5, n):
        direction = 1 if returns[i-5:i].sum() > 0 else -1
        returns[i] = direction * 0.01 + rng.normal(0, 0.005)
    close = 100 * (1 + pd.Series(returns)).cumprod()
    return pd.DataFrame({"Close": close.values}, index=dates)


def test_model_beats_random_on_learnable_data():
    from src.features.library import compute_features

    df = make_predictable_prices()
    feature_cfg = [{"name": "momentum", "params": {"window": 5}}]
    feature_columns = ["momentum_window5"]

    featured = compute_features(df, feature_cfg).dropna()
    split = 200
    train_featured = featured.iloc[:split]
    test_featured = featured.iloc[split:]

    probabilities, _ = fit_and_predict_logistic(train_featured, test_featured, feature_columns)
    predicted_direction = (probabilities > 0.5).astype(float)

    actual_labels = make_next_day_direction_labels(test_featured)
    valid = actual_labels.notna()

    accuracy = (predicted_direction[valid] == actual_labels[valid]).mean()
    assert accuracy > 0.65, f"expected clearly-better-than-random accuracy, got {accuracy:.2f}"


def test_predictions_are_valid_probabilities():
    from src.features.library import compute_features

    df = make_predictable_prices()
    feature_cfg = [{"name": "momentum", "params": {"window": 5}}]
    feature_columns = ["momentum_window5"]

    featured = compute_features(df, feature_cfg).dropna()
    train_featured = featured.iloc[:200]
    test_featured = featured.iloc[200:]

    probabilities, _ = fit_and_predict_logistic(train_featured, test_featured, feature_columns)

    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    assert list(probabilities.index) == list(test_featured.index)


if __name__ == "__main__":
    test_labels_shift_correctly()
    test_model_beats_random_on_learnable_data()
    test_predictions_are_valid_probabilities()
    print("ML SIGNAL TESTS PASSED")