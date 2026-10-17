"""
Data ingestion layer.

Responsibility: turn a config's `data` section into clean, aligned
OHLCV DataFrames - and nothing else. No feature computation, no
strategy logic here. Keeping this layer "dumb" is what lets you swap
data sources later (a different API, a local CSV dump, a database)
without touching anything downstream.

Pinned snapshots: each ticker is downloaded ONCE over a fixed wide range
(FULL_START to FULL_END) and saved as a single file in data/cache/. Every
request is then cut out of that file by date. So two runs with different
date ranges still see exactly the same price for the same day, and nothing
is downloaded again unless you pass refresh=True.

Deliberately strict: if a ticker fails to download, or a request falls
outside the pinned range, this raises immediately rather than silently
returning less data. A pipeline that quietly drops bad data is more
dangerous than one that stops and tells you.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# The pinned range every ticker is downloaded over. FULL_END is exclusive,
# like yfinance's `end`, so the last day included is 2025-12-31.
FULL_START = "2013-01-01"
FULL_END = "2026-01-01"


def _cache_path(ticker: str) -> Path:
    """File that holds the pinned snapshot for one ticker."""
    safe = ticker.replace("^", "").replace("/", "-")
    return CACHE_DIR / f"{safe}.csv"


def _download(ticker: str) -> pd.DataFrame:
    """Fetch the full pinned range from yfinance and clean it into OHLCV shape."""
    df = yf.download(ticker, start=FULL_START, end=FULL_END, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {FULL_START} and {FULL_END}")

    # yfinance sometimes returns MultiIndex columns for a single ticker; flatten if so.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "date"
    return df


def _load_full(ticker: str, refresh: bool) -> pd.DataFrame:
    """Read the ticker's snapshot, downloading and saving it first if needed."""
    path = _cache_path(ticker)

    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="date", parse_dates=True)

    df = _download(ticker)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_ticker(ticker: str, start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """
    Load daily OHLCV data for a single ticker between start_date (included)
    and end_date (excluded). The dates must lie inside the pinned range
    FULL_START to FULL_END. Raises ValueError otherwise, or if no rows remain.
    """
    start_date, end_date = str(start_date), str(end_date)

    if start_date < FULL_START or end_date > FULL_END:
        raise ValueError(
            f"Requested {start_date} to {end_date} is outside the pinned range "
            f"{FULL_START} to {FULL_END}"
        )

    full = _load_full(ticker, refresh)
    df = full[(full.index >= start_date) & (full.index < end_date)]

    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start_date} and {end_date}")

    return df.copy()


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