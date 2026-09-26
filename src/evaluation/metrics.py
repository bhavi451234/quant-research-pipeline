"""
Evaluation layer.

Responsibility: turn a backtested equity curve into honest performance
numbers, and make it trivial to compare train vs test performance side
by side. That comparison is the single most important output of this
whole pipeline — a strategy that looks great in-sample and falls apart
out-of-sample is the default outcome of most backtesting, not the
exception, and this layer exists to catch it rather than hide it.
"""

import numpy as np
import pandas as pd


def _annualize_return(strategy_returns: pd.Series) -> float:
    n_days = len(strategy_returns)
    if n_days == 0:
        return np.nan
    total_return = (1 + strategy_returns).prod() - 1
    years = n_days / 252
    if years <= 0:
        return np.nan
    return (1 + total_return) ** (1 / years) - 1


def sharpe_ratio(strategy_returns: pd.Series, risk_free_rate_annual: float = 0.05) -> float:
    excess = strategy_returns - (risk_free_rate_annual / 252)
    if excess.std() == 0 or excess.isna().all():
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(252)


def sortino_ratio(strategy_returns: pd.Series, risk_free_rate_annual: float = 0.05) -> float:
    excess = strategy_returns - (risk_free_rate_annual / 252)
    downside = excess[excess < 0]
    if downside.std() == 0 or downside.empty:
        return np.nan
    return (excess.mean() / downside.std()) * np.sqrt(252)


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def win_rate(strategy_returns: pd.Series) -> float:
    traded = strategy_returns[strategy_returns != 0]
    if traded.empty:
        return np.nan
    return (traded > 0).mean()


def turnover(trades: pd.Series) -> float:
    """Average absolute position change per bar - a proxy for how much you're trading."""
    return trades.abs().mean()


def cagr(strategy_returns: pd.Series) -> float:
    return _annualize_return(strategy_returns)


METRIC_REGISTRY = {
    "sharpe": lambda bt, cfg: sharpe_ratio(bt["strategy_return"], cfg.get("risk_free_rate_annual", 0.05)),
    "sortino": lambda bt, cfg: sortino_ratio(bt["strategy_return"], cfg.get("risk_free_rate_annual", 0.05)),
    "max_drawdown": lambda bt, cfg: max_drawdown(bt["equity"]),
    "win_rate": lambda bt, cfg: win_rate(bt["strategy_return"]),
    "turnover": lambda bt, cfg: turnover(bt["trade"]),
    "cagr": lambda bt, cfg: cagr(bt["strategy_return"]),
}


def evaluate(backtest_result: pd.DataFrame, metric_names: list, eval_config: dict) -> dict:
    """Compute the requested metrics for a single backtest result."""
    results = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}")
        results[name] = METRIC_REGISTRY[name](backtest_result, eval_config)
    return results


def compare_train_test(train_metrics: dict, test_metrics: dict) -> pd.DataFrame:
    """
    The headline output of the whole pipeline: a side-by-side table of
    in-sample vs out-of-sample metrics, plus the delta. A strategy whose
    Sharpe collapses from train to test is telling you it was overfit -
    this table is what makes that impossible to miss or quietly omit.
    """
    rows = []
    for metric in train_metrics:
        train_val = train_metrics[metric]
        test_val = test_metrics.get(metric, np.nan)
        delta = test_val - train_val if pd.notna(train_val) and pd.notna(test_val) else np.nan
        rows.append({"metric": metric, "train": train_val, "test": test_val, "delta": delta})
    return pd.DataFrame(rows).set_index("metric")
