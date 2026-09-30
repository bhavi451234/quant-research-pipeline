import pandas as pd

from src.features.library import compute_features
from src.strategies.library import compute_signal
from src.validation.ml_signals import fit_and_predict_logistic


def rolling_windows(total_length: int, train_size: int, test_size: int, step_size: int = None) -> list:
    """
    Generate rolling windows for walk-forward validation.

    Args:
        total_length (int): Total number of rows in the time series.
        train_size (int): Number of rows in each training window.
        test_size (int): Number of rows in each testing window.
        step_size (int, optional): How far the window slides each time.
            Defaults to test_size if not provided.

    Returns:
        list: A list of (train_start, train_end, test_start, test_end) tuples of row
        positions. Each end position is excluded, like Python slicing.
    """
    if step_size is None:
        step_size = test_size

    if step_size < 1 or step_size > test_size:
        raise ValueError(
            f"step_size must be between 1 and test_size ({test_size}), got {step_size}. "
            f"A step larger than test_size would leave gaps in the out-of-sample record."
        )

    windows = []
    start = 0
    while start + train_size + test_size <= total_length:
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = test_start + test_size
        windows.append((train_start, train_end, test_start, test_end))
        start += step_size

    return windows


def walk_forward_splits(df: pd.DataFrame, train_size: int, test_size: int, step_size: int = None) -> list:
    """
    Cut a DataFrame into (train, test) chunks, one pair per rolling window.

    Args:
        df (pd.DataFrame): The input time series.
        train_size (int): Number of rows in each training chunk.
        test_size (int): Number of rows in each testing chunk.
        step_size (int, optional): How far the window slides each time.
            Defaults to test_size if not provided.

    Returns:
        list: A list of (train_df, test_df) tuples, one per window.
    """
    windows = rolling_windows(len(df), train_size, test_size, step_size)

    splits = []
    for train_start, train_end, test_start, test_end in windows:
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        splits.append((train_df, test_df))

    return splits


def walk_forward_signals(df: pd.DataFrame, feature_cfg: list, strategy_name: str,
                         strategy_params: dict, train_size: int, test_size: int,
                         step_size: int = None) -> pd.DataFrame:
    """
    Generate strategy signals for out-of-sample days only, one window at a time.

    For each window, the train and test rows are joined into one block, features
    are computed on that block alone (so nothing after the window can leak in),
    the strategy produces signals, and only the test rows are kept. The kept
    pieces are stacked into one continuous record.

    Args:
        df (pd.DataFrame): Raw price table (no features yet) with a Close column.
        feature_cfg (list): The feature settings from the config's `features` section.
        strategy_name (str): Name of a strategy in STRATEGY_REGISTRY.
        strategy_params (dict): Settings for that strategy.
        train_size (int): Rows in each train chunk. Must cover the longest feature lookback.
        test_size (int): Rows in each test chunk.
        step_size (int, optional): Must equal test_size (or be left out), because
            overlapping test windows would give one day several signals.

    Returns:
        pd.DataFrame: Test rows only, with Close, signal and window columns.
    """
    if step_size is not None and step_size != test_size:
        raise ValueError(
            f"walk_forward_signals needs step_size equal to test_size ({test_size}), got {step_size}. "
            f"A smaller step would make test windows overlap, so one day would get several signals."
        )

    splits = walk_forward_splits(df, train_size, test_size)
    if len(splits) == 0:
        raise ValueError(
            f"No complete window fits: {len(df)} rows is fewer than "
            f"train_size + test_size = {train_size + test_size}."
        )

    pieces = []
    for window_number, (train, test) in enumerate(splits):
        window_df = pd.concat([train, test])
        featured = compute_features(window_df, feature_cfg)

        feature_columns = [c for c in featured.columns if c not in window_df.columns]
        if featured.loc[test.index, feature_columns].isna().any().any():
            raise ValueError(
                f"Window {window_number}: some features are still NaN on test rows. "
                f"train_size={train_size} is shorter than the longest feature lookback."
            )

        signal = compute_signal(featured, strategy_name, strategy_params)

        piece = pd.DataFrame({
            "Close": featured.loc[test.index, "Close"],
            "signal": signal.loc[test.index],
        })
        piece["window"] = window_number
        pieces.append(piece)

    return pd.concat(pieces)


def walk_forward_ml_signals(df: pd.DataFrame, feature_cfg: list, feature_columns: list,
                            train_size: int, test_size: int, step_size: int = None,
                            threshold: float = 0.05) -> pd.DataFrame:
    """
    ML counterpart to walk_forward_signals. Instead of a fixed rule, fits a
    fresh logistic regression on each window's train rows and predicts on
    that window's test rows, so the model is retrained at every step rather
    than fit once on the full history.

    Args:
        df: raw price table (no features yet), with a Close column.
        feature_cfg: feature settings, as in a config's `features` section -
            determines which columns compute_features will produce.
        feature_columns: which of those produced columns the model should
            actually use as inputs (a subset of what feature_cfg produces).
        train_size, test_size, step_size: same meaning as walk_forward_signals.
        threshold: how far from 0.5 the model's predicted probability must be
            to trigger a signal. Long above 0.5+threshold, short below
            0.5-threshold, flat in between.

    Returns:
        pd.DataFrame: test rows only, with Close, signal and window columns -
        identical shape to walk_forward_signals's output.
    """
    if step_size is not None and step_size != test_size:
        raise ValueError(
            f"walk_forward_ml_signals needs step_size equal to test_size ({test_size}), got {step_size}. "
            f"A smaller step would make test windows overlap, so one day would get several signals."
        )

    splits = walk_forward_splits(df, train_size, test_size)
    if len(splits) == 0:
        raise ValueError(
            f"No complete window fits: {len(df)} rows is fewer than "
            f"train_size + test_size = {train_size + test_size}."
        )

    pieces = []
    for window_number, (train, test) in enumerate(splits):
        window_df = pd.concat([train, test])
        featured = compute_features(window_df, feature_cfg)

        if featured.loc[test.index, feature_columns].isna().any().any():
            raise ValueError(
                f"Window {window_number}: some model features are still NaN on test rows. "
                f"train_size={train_size} is shorter than the longest feature lookback."
            )

        train_featured = featured.loc[train.index]
        test_featured = featured.loc[test.index]

        probabilities = fit_and_predict_logistic(train_featured, test_featured, feature_columns)

        signal = pd.Series(0, index=test.index)
        signal[probabilities > 0.5 + threshold] = 1
        signal[probabilities < 0.5 - threshold] = -1

        piece = pd.DataFrame({
            "Close": featured.loc[test.index, "Close"],
            "signal": signal,
        })
        piece["window"] = window_number
        pieces.append(piece)

    return pd.concat(pieces)