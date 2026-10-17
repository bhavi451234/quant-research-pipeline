"""
Data ingestion layer.

Responsibility: turn a config's `data` section into clean, aligned
OHLCV DataFrames - and nothing else. No feature computation, no
strategy logic here. Keeping this layer "dumb" is what lets you swap
data sources later (a different API, a local CSV dump, a database)
without touching anything downstream.

Pinned snapshots: the first time a (ticker, start, end) request is made,
the data is downloaded from yfinance and saved to data/cache/. Every
later request reads that file instead of calling yfinance. This makes
runs reproducible: the same config always sees the same prices, even if
Yahoo later revises adjusted prices. Pass refresh=True (or delete the
file) to download fresh data on purpose.

Deliberately strict: if a ticker fails to download, this raises
immediately rather than silently skipping it. A pipeline that quietly
drops bad data is more dangerous than one that stops and tells you.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    """File that holds the snapshot for one (ticker, start, end) request."""
    safe = ticker.replace("^", "").replace("/", "-")
    return CACHE_DIR / f"{safe}_{start_date}_{end_date}.csv"


def _download(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch from yfinance and clean into the standard OHLCV shape."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start_date} and {end_date}")

    # yfinance sometimes returns MultiIndex columns for a single ticker; flatten if so.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "date"
    return df


def load_ticker(ticker: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """
    Load daily OHLCV data for a single ticker between start_date and end_date.

    Reads the saved snapshot if one exists; otherwise downloads, saves it,
    and returns it. refresh=True forces a new download and overwrites the
    snapshot. Raises ValueError if no data comes back - callers should let
    this propagate rather than catching it, per the "strict" design decision.
    """
    path = _cache_path(ticker, start_date, end_date)

    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="date", parse_dates=True)

    df = _download(ticker, start_date, end_date)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_universe(tickers: list, start_date: str, end_date: str, refresh: bool = False) -> dict:
    """
    Load OHLCV data for a list of tickers. Returns {ticker: DataFrame}.
    Kept as a dict rather than one big merged frame - merging is a
    features/backtest concern, not a loading concern.

    The ticker list itself is entirely open - this function has no
    knowledge of or limit on which tickers exist. Whatever list the
    config gives it, it loads.
    """
    return {
        ticker: load_ticker(ticker, start_date, end_date, refresh=refresh)
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