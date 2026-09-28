import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.validation.walk_forward import rolling_windows, walk_forward_splits


def make_df(n):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": range(n)}, index=dates)


def test_rolling_windows():
    assert rolling_windows(10, 4, 2) == [(0, 4, 4, 6), (2, 6, 6, 8), (4, 8, 8, 10)]


def test_sizes_and_order():
    df = make_df(10)
    splits = walk_forward_splits(df, train_size=4, test_size=2)

    assert len(splits) == 3
    for train, test in splits:
        assert len(train) == 4
        assert len(test) == 2
        assert train.index.max() < test.index.min()
        gap = df.index.get_loc(test.index[0]) - df.index.get_loc(train.index[-1])
        assert gap == 1


def test_test_windows_do_not_overlap():
    df = make_df(10)
    splits = walk_forward_splits(df, 4, 2)
    seen = []
    for _, test in splits:
        seen.extend(test.index.tolist())
    assert len(seen) == len(set(seen))


def test_step_size():
    assert rolling_windows(10, 4, 2, step_size=1) == [
        (0, 4, 4, 6), (1, 5, 5, 7), (2, 6, 6, 8), (3, 7, 7, 9), (4, 8, 8, 10)
    ]
    assert rolling_windows(10, 4, 2, step_size=2) == rolling_windows(10, 4, 2)


def test_invalid_step_size():
    for bad in (0, 3):
        try:
            rolling_windows(10, 4, 2, step_size=bad)
        except ValueError:
            continue
        raise AssertionError(f"step_size={bad} should have raised ValueError")


if __name__ == "__main__":
    test_rolling_windows()
    test_sizes_and_order()
    test_test_windows_do_not_overlap()
    test_step_size()
    test_invalid_step_size()
    print("ALL WALK-FORWARD TESTS PASSED")