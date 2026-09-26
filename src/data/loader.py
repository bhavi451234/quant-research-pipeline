"""
Data ingestion layer.

Responsibility: turn a config's `data` section into clean, aligned
OHLCV DataFrames — and nothing else. No feature computation, no
strategy logic here. Keeping this layer "dumb" is what lets you swap
data sources later (a different API, a local CSV dump, a database)
without touching anything downstream.
"""

import os
from pathlib import Path
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{ticker}_{start}_{end}.csv"


def load_ticker(ticker: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Load daily OHLCV data for a single ticker between start_date and end_date.
    Caches to disk so re-running an experiment doesn't re-hit the network
    every time — important once you're iterating quickly.
    """
    cache_file = _cache_path(ticker, start_date, end_date)

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start_date} and {end_date}")

    # yfinance sometimes returns MultiIndex columns for a single ticker; flatten if so.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "date"

    if use_cache:
        df.to_csv(cache_file)

    return df


def load_universe(tickers: list, start_date: str, end_date: str, use_cache: bool = True) -> dict:
    """
    Load OHLCV data for a list of tickers. Returns {ticker: DataFrame}.
    Kept as a dict rather than one big merged frame — merging is a
    features/backtest concern, not a loading concern.
    """
    return {
        ticker: load_ticker(ticker, start_date, end_date, use_cache=use_cache)
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
