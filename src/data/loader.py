
"""
Data ingestion layer.

Responsibility: turn a config's `data` section into clean, aligned
OHLCV DataFrames — and nothing else. No feature computation, no
strategy logic here. Keeping this layer "dumb" is what lets you swap
data sources later (a different API, a local CSV dump, a database)
without touching anything downstream.

Deliberately no caching: every call fetches fresh from yfinance. Given
how fast the pipeline runs end-to-end, the simplicity of "always live
data" outweighs the speed benefit of a cache for now.

Deliberately strict: if a ticker fails to download, this raises
immediately rather than silently skipping it. A pipeline that quietly
drops bad data is more dangerous than one that stops and tells you.
"""

import pandas as pd
import yfinance as yf


def load_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load daily OHLCV data for a single ticker between start_date and end_date.
    Raises ValueError if no data comes back - callers should let this
    propagate rather than catching it, per the "strict" design decision.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start_date} and {end_date}")

    # yfinance sometimes returns MultiIndex columns for a single ticker; flatten if so.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "date"

    return df


def load_universe(tickers: list, start_date: str, end_date: str) -> dict:
    """
    Load OHLCV data for a list of tickers. Returns {ticker: DataFrame}.
    Kept as a dict rather than one big merged frame - merging is a
    features/backtest concern, not a loading concern.

    The ticker list itself is entirely open - this function has no
    knowledge of or limit on which tickers exist. Whatever list the
    config gives it, it loads.
    """
    return {
        ticker: load_ticker(ticker, start_date, end_date)
        for ticker in tickers
    }


def train_test_split(df: pd.DataFrame, split_date: str) -> tuple:
    """
    Split a single ticker's DataFrame into (train, test) by date.
    Strict: the split date itself belongs to test, so there's no leakage
    of the boundary bar into training.
    """
    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()
    return train, test
