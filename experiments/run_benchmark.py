"""
Benchmark comparison (Phase 8).

Scores every walk-forward strategy against (a) the S&P 500 (SPY) and
(b) simply holding the same stock, on the SAME nine out-of-sample windows
and with the SAME costs, then reports each strategy's alpha and beta
against SPY.

    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --configs config/walkforward_sma.yaml config/walkforward_ml.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import BacktestEngine
from src.data.loader import load_ticker
from src.evaluation.metrics import alpha_beta, summarize_windows, walk_forward_evaluate
from src.pipeline import run_walkforward, run_walkforward_ml
from src.validation.walk_forward import buy_and_hold_signals

DEFAULT_CONFIGS = [
    "config/walkforward_sma.yaml",
    "config/walkforward_ema.yaml",
    "config/walkforward_momentum.yaml",
    "config/walkforward_rsi.yaml",
    "config/walkforward_bollinger.yaml",
    "config/walkforward_ml.yaml",
    "config/walkforward_ml_gb.yaml",
]


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def buy_and_hold_backtest(ticker: str, base_config: dict) -> pd.DataFrame:
    """Hold `ticker` long through every out-of-sample window, same engine and costs."""
    data_cfg = base_config["data"]
    wf_cfg = base_config["walk_forward"]
    bt_cfg = base_config["backtest"]

    raw = load_ticker(ticker, data_cfg["start_date"], data_cfg["end_date"])
    signals = buy_and_hold_signals(raw, wf_cfg["train_size"], wf_cfg["test_size"])

    engine = BacktestEngine(
        initial_capital=bt_cfg["initial_capital"],
        transaction_cost_bps=bt_cfg["transaction_cost_bps"],
        slippage_bps=bt_cfg["slippage_bps"],
        position_size_pct=bt_cfg["position_size_pct"],
    )
    return engine.run(signals)


def mean_sharpe(backtest: pd.DataFrame, eval_cfg: dict) -> float:
    per_window = walk_forward_evaluate(backtest, ["sharpe"], eval_cfg)
    return summarize_windows(per_window)["mean_sharpe"]


def main():
    parser = argparse.ArgumentParser(description="Compare strategies against SPY and buy-and-hold.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--out", type=str, default="results/benchmark.csv")
    args = parser.parse_args()

    configs = [load_config(p) for p in args.configs]
    base = configs[0]
    eval_cfg = base["evaluation"]
    rf = eval_cfg.get("risk_free_rate_annual", 0.05)
    tickers = base["data"]["tickers"]

    # SPY: the market benchmark every strategy is measured against.
    spy_bt = buy_and_hold_backtest("SPY", base)
    spy_returns = spy_bt["Close"].pct_change()
    spy_sharpe = mean_sharpe(spy_bt, eval_cfg)
    print(f"\nSPY buy-and-hold, mean Sharpe across windows: {spy_sharpe:.3f}\n")

    rows = []

    # Sanity row: SPY against itself must give beta ~1 and alpha ~0.
    ab = alpha_beta(spy_bt["strategy_return"], spy_returns, rf)
    rows.append({"strategy": "buy_and_hold", "ticker": "SPY", "mean_sharpe": spy_sharpe,
                 "alpha_annual": ab["alpha_annual"], "beta": ab["beta"]})

    # Buy-and-hold each stock itself.
    own_sharpe = {}
    for ticker in tickers:
        bt = buy_and_hold_backtest(ticker, base)
        own_sharpe[ticker] = mean_sharpe(bt, eval_cfg)
        ab = alpha_beta(bt["strategy_return"], spy_returns, rf)
        rows.append({"strategy": "buy_and_hold", "ticker": ticker,
                     "mean_sharpe": own_sharpe[ticker],
                     "alpha_annual": ab["alpha_annual"], "beta": ab["beta"]})

    # Every walk-forward strategy.
    for path, cfg in zip(args.configs, configs):
        name = cfg["experiment_name"]
        results = run_walkforward_ml(cfg) if "ml" in cfg else run_walkforward(cfg)
        for ticker, res in results.items():
            ab = alpha_beta(res["backtest_result"]["strategy_return"], spy_returns, rf)
            rows.append({"strategy": name, "ticker": ticker,
                         "mean_sharpe": res["summary"]["mean_sharpe"],
                         "alpha_annual": ab["alpha_annual"], "beta": ab["beta"]})
        print(f"done: {name}")

    df = pd.DataFrame(rows)
    df["vs_spy_sharpe"] = df["mean_sharpe"] - spy_sharpe
    df["vs_own_buy_hold_sharpe"] = df.apply(
        lambda r: r["mean_sharpe"] - own_sharpe[r["ticker"]] if r["ticker"] in own_sharpe else float("nan"),
        axis=1,
    )

    pd.set_option("display.width", 200)
    print("\n=== Mean Sharpe across windows ===")
    print(df.pivot(index="strategy", columns="ticker", values="mean_sharpe").round(2))
    print("\n=== Mean Sharpe minus SPY's mean Sharpe (positive = beat the market) ===")
    print(df.pivot(index="strategy", columns="ticker", values="vs_spy_sharpe").round(2))
    print("\n=== Annualized alpha vs SPY, % ===")
    print((df.pivot(index="strategy", columns="ticker", values="alpha_annual") * 100).round(1))
    print("\n=== Beta vs SPY ===")
    print(df.pivot(index="strategy", columns="ticker", values="beta").round(2))

    df.to_csv(args.out, index=False)
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()