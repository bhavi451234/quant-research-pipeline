"""
Feature library.

Responsibility: pure functions that take price data in and return a
named feature (a pandas Series) out. No lookahead — every function
here must only use data up to and including the current row. That
constraint is what makes this library backtest-safe.

Each feature is registered in FEATURE_REGISTRY so the pipeline can look
features up by name from a config, rather than you hardcoding which
features get computed inside the pipeline itself.
"""

import numpy as np
import pandas as pd


def sma(df: pd.DataFrame, window: int) -> pd.Series:
    """Simple moving average of the close price."""
    return df["Close"].rolling(window=window, min_periods=window).mean()


def ema(df: pd.DataFrame, span: int) -> pd.Series:
    """Exponential moving average of the close price."""
    return df["Close"].ewm(span=span, adjust=False).mean()


def rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Relative Strength Index. Standard Wilder smoothing."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling annualized volatility of daily returns."""
    daily_returns = df["Close"].pct_change()
    return daily_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)


def momentum(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Simple price momentum: % change over the window."""
    return df["Close"].pct_change(periods=window)

def bollinger_middle(df: pd.DataFrame, window: int = 20) ->pd.Series:
    """Middle band of Bollinger Bands: simple moving average."""
    return df["Close"].rolling(window=window, min_periods=window).mean()

def bollinger_upper(df: pd.DataFrame, window: int = 20, num_std: float = 2) ->pd.Series:
    """Upper band of Bollinger Bands: SMA + num_std * rolling std."""
    sma = bollinger_middle(df, window)
    rolling_std = df["Close"].rolling(window=window, min_periods=window).std()
    return sma + (num_std * rolling_std)

def bollinger_lower(df: pd.DataFrame, window: int = 20, num_std: float = 2) ->pd.Series:
    """Lower band of Bollinger Bands: SMA - num_std * rolling std."""
    sma = bollinger_middle(df, window)
    rolling_std = df["Close"].rolling(window=window, min_periods=window).std()
    return sma - (num_std * rolling_std)

def price_to_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Scale-free version of SMA: how far price sits from its average, as a
    fraction of the average. Unlike raw SMA, this is comparable across time
    even as the stock's price level changes (e.g. AAPL at $15 vs $220).
    """
    baseline = sma(df, window)
    return (df["Close"] - baseline) / baseline


def price_to_ema(df: pd.DataFrame, span: int = 20) -> pd.Series:
    """Scale-free version of EMA, same idea as price_to_sma."""
    baseline = ema(df, span)
    return (df["Close"] - baseline) / baseline


def bollinger_position(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Where price sits within the Bollinger Bands, as a 0-to-1 position
    (0 = touching the lower band, 1 = touching the upper band, 0.5 = at
    the middle). Scale-free, unlike the raw band values.
    """
    upper = bollinger_upper(df, window, num_std)
    lower = bollinger_lower(df, window, num_std)
    return (df["Close"] - lower) / (upper - lower)


FEATURE_REGISTRY = {
    "sma": sma,
    "ema": ema,
    "rsi": rsi,
    "volatility": volatility,
    "momentum": momentum,
    "bollinger_middle": bollinger_middle,
    "bollinger_upper": bollinger_upper,
    "bollinger_lower": bollinger_lower,
    "price_to_sma": price_to_sma,
    "price_to_ema": price_to_ema,
    "bollinger_position": bollinger_position,
}

def compute_features(df: pd.DataFrame, feature_configs: list) -> pd.DataFrame:
    """
    Given a DataFrame and a list of feature configs (as they appear in
    the YAML config, e.g. {"name": "sma", "params": {"window": 20}}),
    compute each feature and attach it as a new, clearly-named column.
    """
    out = df.copy()

    for fc in feature_configs:
        name = fc["name"]
        params = fc.get("params", {})

        if name not in FEATURE_REGISTRY:
            raise ValueError(f"Unknown feature '{name}'. Available: {list(FEATURE_REGISTRY.keys())}")

        # column name encodes the params so sma(20) and sma(50) don't collide
        param_str = "_".join(f"{k}{v}" for k, v in params.items())
        col_name = f"{name}_{param_str}" if param_str else name

        out[col_name] = FEATURE_REGISTRY[name](df, **params)

    return out



