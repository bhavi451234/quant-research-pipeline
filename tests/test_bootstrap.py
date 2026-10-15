import numpy as np

from src.evaluation.metrics import bootstrap_mean_ci, paired_bootstrap_difference

# Nine made-up "window" values, standing in for nine walk-forward windows.
CLEARLY_POSITIVE = [1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.3, 0.7, 1.0]
BALANCED_AROUND_ZERO = [1.0, -1.0, 0.5, -0.5, 0.8, -0.8, 0.3, -0.3, 0.0]


def test_clear_advantage_is_significant():
    zeros = [0.0] * 9
    result = paired_bootstrap_difference(CLEARLY_POSITIVE, zeros)
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.05
    assert abs(result["mean_diff"] - np.mean(CLEARLY_POSITIVE)) < 1e-12


def test_identical_strategies_are_not_different():
    result = paired_bootstrap_difference(CLEARLY_POSITIVE, CLEARLY_POSITIVE)
    assert result["mean_diff"] == 0
    assert result["ci_low"] == 0 and result["ci_high"] == 0
    assert result["p_value"] == 1.0


def test_differences_balanced_around_zero_are_not_significant():
    zeros = [0.0] * 9
    result = paired_bootstrap_difference(BALANCED_AROUND_ZERO, zeros)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["p_value"] > 0.05


def test_swapping_a_and_b_flips_the_sign():
    zeros = [0.0] * 9
    forward = paired_bootstrap_difference(CLEARLY_POSITIVE, zeros)
    backward = paired_bootstrap_difference(zeros, CLEARLY_POSITIVE)
    assert abs(forward["mean_diff"] + backward["mean_diff"]) < 1e-12
    assert backward["ci_high"] < 0


def test_nan_windows_are_dropped_from_both_strategies():
    a = [1.0, np.nan, 0.8, 1.1, 0.9, 1.0, 1.3, 0.7, 1.0]
    b = [0.0] * 9
    result = paired_bootstrap_difference(a, b)
    assert result["n_windows"] == 8


def test_too_few_windows_returns_nan():
    result = paired_bootstrap_difference([1.0, 2.0], [0.0, 0.0])
    assert np.isnan(result["mean_diff"]) and np.isnan(result["p_value"])
    assert result["n_windows"] == 2


def test_same_seed_gives_same_answer():
    zeros = [0.0] * 9
    first = paired_bootstrap_difference(BALANCED_AROUND_ZERO, zeros, seed=7)
    second = paired_bootstrap_difference(BALANCED_AROUND_ZERO, zeros, seed=7)
    assert first == second


def test_bootstrap_mean_ci_tests_a_single_strategy_against_zero():
    result = bootstrap_mean_ci(CLEARLY_POSITIVE)
    assert result["ci_low"] > 0
    assert abs(result["mean_diff"] - np.mean(CLEARLY_POSITIVE)) < 1e-12