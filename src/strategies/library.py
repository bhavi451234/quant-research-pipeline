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


STRATEGY_REGISTRY = {
    "sma_crossover": sma_crossover,
    "rsi_mean_reversion": rsi_mean_reversion,
}


def compute_signal(df: pd.DataFrame, strategy_name: str, params: dict) -> pd.Series:
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[strategy_name](df, **params)
