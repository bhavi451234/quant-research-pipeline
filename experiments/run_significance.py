"""
Significance tests (Phase 9).

Reads the per-window results saved by run_benchmark.py and asks, with a paired
bootstrap over the nine walk-forward windows, which of the headline
comparisons are distinguishable from noise.

    python experiments/run_significance.py

A positive mean difference always means the first-named strategy (A) was
better: higher Sharpe, higher total return, or a shallower (less negative)
max drawdown.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import bootstrap_mean_ci, paired_bootstrap_difference

STOCKS = ["AAPL", "MSFT", "TSLA", "SNAP"]
BUY_HOLD = "buy_and_hold"
LOGISTIC = "logistic_regression_walkforward"
GRADIENT = "gradient_boosting_walkforward"
RULE_BASED = [
    "sma_crossover_walkforward",
    "ema_crossover_walkforward",
    "momentum_walkforward",
    "rsi_mean_reversion_walkforward",
    "bollinger_reversion_walkforward",
]


def window_values(df: pd.DataFrame, strategy: str, ticker: str, metric: str) -> np.ndarray:
    """One value per window, in window order, for a single strategy on a single ticker."""
    sub = df[(df["strategy"] == strategy) & (df["ticker"] == ticker)].sort_values("window")
    return sub[metric].to_numpy(dtype=float)


def pooled_values(df: pd.DataFrame, strategy: str, metric: str) -> np.ndarray:
    """One value per window: the average across the four stocks in that window."""
    sub = df[(df["strategy"] == strategy) & (df["ticker"].isin(STOCKS))]
    return sub.groupby("window")[metric].mean().sort_index().to_numpy(dtype=float)


def verdict(result: dict) -> str:
    if np.isnan(result["ci_low"]):
        return "n/a"
    if result["ci_low"] > 0:
        return "A better"
    if result["ci_high"] < 0:
        return "A worse"
    return "inconclusive"


def make_row(comparison: str, metric: str, ticker: str, a_name: str, b_name: str, result: dict) -> dict:
    row = {"comparison": comparison, "metric": metric, "ticker": ticker,
           "a": a_name, "b": b_name, **result}
    row["verdict"] = verdict(result)
    return row


def cell(row) -> str:
    if np.isnan(row["mean_diff"]):
        return "n/a"
    mark = "" if row["verdict"] == "inconclusive" else " *"
    return f"{row['mean_diff']:+.2f} [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]{mark}"


def show(results: pd.DataFrame, comparison: str, metric: str) -> None:
    sub = results[(results["comparison"] == comparison) & (results["metric"] == metric)].copy()
    print(f"\n=== {comparison} | metric: {metric} ===")
    if sub["b"].nunique() > 1:
        print(sub[["ticker", "a", "b", "mean_diff", "ci_low", "ci_high", "p_value", "verdict"]]
              .round(3).to_string(index=False))
        return
    sub["cell"] = sub.apply(cell, axis=1)
    table = sub.pivot(index="a", columns="ticker", values="cell")
    columns = [c for c in STOCKS + ["ALL"] if c in table.columns]
    print(table[columns].to_string())


def main():
    parser = argparse.ArgumentParser(description="Bootstrap significance tests on per-window results.")
    parser.add_argument("--windows", type=str, default="results/benchmark_windows.csv")
    parser.add_argument("--out", type=str, default="results/significance.csv")
    parser.add_argument("--metrics", nargs="+", default=["sharpe", "max_drawdown", "total_return"])
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    df = pd.read_csv(args.windows)
    strategies = [s for s in df["strategy"].unique() if s != BUY_HOLD]
    rows = []

    # 1. Every strategy against buying and holding the same stock.
    for metric in args.metrics:
        for strategy in strategies:
            for ticker in STOCKS:
                a = window_values(df, strategy, ticker, metric)
                b = window_values(df, BUY_HOLD, ticker, metric)
                rows.append(make_row("strategy vs buy-and-hold", metric, ticker, strategy, BUY_HOLD,
                                     paired_bootstrap_difference(a, b, n_boot=args.n_boot)))
            a = pooled_values(df, strategy, metric)
            b = pooled_values(df, BUY_HOLD, metric)
            rows.append(make_row("strategy vs buy-and-hold", metric, "ALL", strategy, BUY_HOLD,
                                 paired_bootstrap_difference(a, b, n_boot=args.n_boot)))

    # 2. Logistic regression against gradient boosting.
    for ticker in STOCKS:
        a = window_values(df, LOGISTIC, ticker, "sharpe")
        b = window_values(df, GRADIENT, ticker, "sharpe")
        rows.append(make_row("logistic vs gradient boosting", "sharpe", ticker, LOGISTIC, GRADIENT,
                             paired_bootstrap_difference(a, b, n_boot=args.n_boot)))
    rows.append(make_row("logistic vs gradient boosting", "sharpe", "ALL", LOGISTIC, GRADIENT,
                         paired_bootstrap_difference(pooled_values(df, LOGISTIC, "sharpe"),
                                                     pooled_values(df, GRADIENT, "sharpe"),
                                                     n_boot=args.n_boot)))

    # 3. Logistic regression against the best rule-based strategy on each ticker.
    #    "Best" is picked after the fact, which makes this a tougher test for logistic.
    for ticker in STOCKS:
        best = max(RULE_BASED, key=lambda s: np.nanmean(window_values(df, s, ticker, "sharpe")))
        a = window_values(df, LOGISTIC, ticker, "sharpe")
        b = window_values(df, best, ticker, "sharpe")
        rows.append(make_row("logistic vs best rule-based", "sharpe", ticker, LOGISTIC, best,
                             paired_bootstrap_difference(a, b, n_boot=args.n_boot)))

    # 4. Is any strategy's mean Sharpe distinguishable from zero?
    for strategy in strategies + [BUY_HOLD]:
        for ticker in STOCKS:
            rows.append(make_row("mean Sharpe vs zero", "sharpe", ticker, strategy, "zero",
                                 bootstrap_mean_ci(window_values(df, strategy, ticker, "sharpe"),
                                                   n_boot=args.n_boot)))
        rows.append(make_row("mean Sharpe vs zero", "sharpe", "ALL", strategy, "zero",
                             bootstrap_mean_ci(pooled_values(df, strategy, "sharpe"), n_boot=args.n_boot)))

    results = pd.DataFrame(rows)

    shown = results[["comparison", "metric"]].drop_duplicates()
    for _, r in shown.iterrows():
        show(results, r["comparison"], r["metric"])

    print("\nCells show: mean difference [95% interval]. * = the interval excludes 0.")
    print("Positive = first-named strategy (A) better. With this many comparisons,")
    print("a few starred results are expected by chance alone.")

    results.to_csv(args.out, index=False)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()