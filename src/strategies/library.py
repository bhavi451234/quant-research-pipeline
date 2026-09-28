"""
Strategy library.

Responsibility: given a DataFrame that already has features computed,
produce a `signal` column (1 = long, -1 = short, 0 = flat). Nothing
here should know about costs, position sizing, or equity curves —
that's the backtest engine's job. This separation is what lets you
plug any strategy into the same, already-hardened backtester.
"""

import pandas as pd



def sma_crossover(df: pd.DataFrame, fast_window: int, slow_window: int) -> pd.Series:
    """
    Classic trend-following signal: long when the fast SMA is above the
    slow SMA, flat otherwise. Requires sma_window{fast_window} and
    sma_window{slow_window} to already exist as columns (i.e. they must
    be listed in the config's `features` section with matching windows).
    """
    fast_col = f"sma_window{fast_window}"
    slow_col = f"sma_window{slow_window}"

    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(
            f"sma_crossover needs '{fast_col}' and '{slow_col}' as features. "
            f"Add them to the config's features list."
        )

    signal = (df[fast_col] > df[slow_col]).astype(int)
    return signal


def rsi_mean_reversion(df: pd.DataFrame, rsi_window: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """
    Mean-reversion signal: long when RSI signals oversold, short when
    overbought, flat otherwise. Requires rsi_window{rsi_window} as a feature.
    """
    rsi_col = f"rsi_window{rsi_window}"
    if rsi_col not in df.columns:
        raise ValueError(f"rsi_mean_reversion needs '{rsi_col}' as a feature.")

    signal = pd.Series(0, index=df.index)
    signal[df[rsi_col] < oversold] = 1
    signal[df[rsi_col] > overbought] = -1
    return signal

def momentum_strategy(df: pd.DataFrame, momentum_window: int = 20, threshold: float = 0.02) -> pd.Series:
    """
    Momentum strategy: long when momentum is positive, short when negative.
    Requires momentum_window{momentum_window} as a feature.
    """
    momentum_col = f"momentum_window{momentum_window}"
    if momentum_col not in df.columns:
        raise ValueError(f"momentum_strategy needs '{momentum_col}' as a feature.")

    signal = pd.Series(0, index=df.index)
    signal[df[momentum_col] > threshold] = 1
    signal[df[momentum_col] < -threshold] = -1
    return signal

def ema_crossover(df: pd.DataFrame, fast_span: int, slow_span: int) -> pd.Series:
    """
    Trend-following signal: long when the fast EMA is above the
    slow EMA, flat otherwise. Requires ema_span{fast_span} and
    ema_span{slow_span} to already exist as columns (i.e. they must
    be listed in the config's `features` section with matching spans).
    """
    fast_col = f"ema_span{fast_span}"
    slow_col = f"ema_span{slow_span}"

    if fast_col not in df.columns or slow_col not in df.columns:
        raise ValueError(
            f"ema_crossover needs '{fast_col}' and '{slow_col}' as features. "
            f"Add them to the config's features list."
        )

    signal = (df[fast_col] > df[slow_col]).astype(int)
    return signal

def bollinger_reversion(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Mean-reversion signal based on Bollinger Bands: long when price is below
    the lower band, short when above the upper band, flat otherwise. Requires
    bollinger_lower_window{window}_num_std{num_std} and
    bollinger_upper_window{window}_num_std{num_std} to already exist as columns.
    """
    lower_col = f"bollinger_lower_window{window}_num_std{num_std}"
    upper_col = f"bollinger_upper_window{window}_num_std{num_std}"

    if lower_col not in df.columns or upper_col not in df.columns:
        raise ValueError(
            f"bollinger_reversion needs '{lower_col}' and '{upper_col}' as features. "
            f"Add them to the config's features list."
        )

    signal = pd.Series(0, index=df.index)
    signal[df["Close"] < df[lower_col]] = 1
    signal[df["Close"] > df[upper_col]] = -1
    return signal


STRATEGY_REGISTRY = {
    "sma_crossover": sma_crossover,
    "rsi_mean_reversion": rsi_mean_reversion,
    "ema_crossover": ema_crossover,
    "momentum_strategy": momentum_strategy,
    "bollinger_reversion": bollinger_reversion,
    
}


def compute_signal(df: pd.DataFrame, strategy_name: str, params: dict) -> pd.Series:
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[strategy_name](df, **params)
