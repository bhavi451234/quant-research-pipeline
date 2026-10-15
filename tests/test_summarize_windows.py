import numpy as np
import pandas as pd

from src.evaluation.metrics import summarize_windows


def test_all_nan_windows_do_not_crash():
    per_window = pd.DataFrame({"sharpe": [np.nan, np.nan]}, index=[0, 1])
    summary = summarize_windows(per_window)
    assert np.isnan(summary["mean_sharpe"])
    assert np.isnan(summary["worst_window"])
    assert summary["pct_positive_sharpe_windows"] == 0


def test_worst_window_is_still_found_when_some_values_exist():
    per_window = pd.DataFrame({"sharpe": [0.5, np.nan, -1.0]}, index=[0, 1, 2])
    summary = summarize_windows(per_window)
    assert summary["worst_window"] == 2