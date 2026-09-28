import pandas as pd

from src.features.library import compute_features
from src.strategies.library import compute_signal



def rolling_windows(total_length: int, train_size: int, test_size: int, step_size: int = None) -> list:
    """
    Generate rolling windows for walk-forward validation.

    Args:
        total_length (int): Total length of the time series data.
        train_size (int): Size of the training window.
        test_size (int): Size of the testing window.
        step_size (int, optional): Step size for rolling windows. Defaults to test_size if not provided.

    Returns:
        list: A list of tuples, each containing the start and end indices for training and testing windows.
    """
    if step_size is None:
        step_size = test_size

    if step_size < 1 or step_size > test_size:
         raise ValueError(
             f"step_size must be between 1 and test_size ({test_size}), got {step_size}."
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
    Generate walk-forward splits for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing time series data.
        train_size (int): Size of the training window.
        test_size (int): Size of the testing window.
        step_size (int, optional): Step size for rolling windows. Defaults to test_size if not provided.

    Returns:
        list: A list of tuples, each containing the training and testing DataFrames for each split.
    """
    total_length = len(df)
    windows = rolling_windows(total_length, train_size, test_size, step_size)
    
    splits = []
    for train_start, train_end, test_start, test_end in windows:
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        splits.append((train_df, test_df))
    
    return splits


def walk_forward_signals(df: pd.DataFrame, feature_cfg: list, strategy_name: str,
                         strategy_params: dict, train_size: int, test_size: int,
                         step_size: int = None) -> pd.DataFrame:
       if step_size is None:
             step_size = test_size
       if step_size < 1 or step_size > test_size :
             raise ValueError(
                  f"step_size must be between 1 and test_size"
             )
       
    
       splits = walk_forward_splits
       if splits is [] : 
          raise ValueError(
                f"No splits possible for the given conditions"
          )
       
       pieces = [] 
       for window_number, (train, test) in enumerate(splits):
         a. window_df = pd.concat([train, test])
        b. featured = compute_features(window_df, feature_cfg)
         c. the NaN guard (code below)
         d. signal = compute_signal(featured, strategy_name, strategy_params)
         e. build a table of Close and signal for the TEST rows only,
             then add a "window" column holding window_number
         f. pieces.append(that table)
        return pd.concat(pieces)