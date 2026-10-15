"""
Transaction-cost sensitivity sweep (Phase 12).

For every walk-forward config, builds each ticker's out-of-sample signals ONCE
(signals do not depend on costs), then re-runs only the backtest at several
cost levels. Reports mean Sharpe vs cost, and each strategy's breakeven cost
(the cost at which mean Sharpe reaches zero).

    python experiments/run_cost_sweep.py
    python experiments/run_cost_sweep.py --configs config/walkforward_sma.yaml config/walkforward_ml.yaml
    python experiments/run_cost_sweep.py --costs 0 5 10 20 40

`cost` here is the TOTAL one-way cost in bps (transaction cost + slippage
combined; the engine is given slippage_bps=0). A round trip costs twice this.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_COSTS = [0, 5, 7, 10, 20]


def breakeven_cost(costs, sharpes):
    """
    Cost (bps) at which Sharpe crosses from positive to <= 0, by linear
    interpolation between the swept points.
      np.nan -> Sharpe is already <= 0 at the lowest cost (never positive)
      np.inf -> Sharpe is still positive at the highest cost swept
    """
    pairs = [(c, s) for c, s in zip(costs, sharpes) if not np.isnan(s)]
    if len(pairs) < 2:
        return np.nan
    pairs.sort()
    c = [p[0] for p in pairs]
    s = [p[1] for p in pairs]
    if s[0] <= 0:
        return np.nan
    for i in range(len(c) - 1):
        if s[i] > 0 and s[i + 1] <= 0:
            return c[i] + s[i] * (c[i + 1] - c[i]) / (s[i] - s[i + 1])
    return np.inf


def format_breakeven(value, max_cost):
    if np.isnan(value):
        return "never > 0"
    if np.isinf(value):
        return f"> {max_cost}"
    return f"{value:.1f}"


def sweep_config(config, costs):
    from src.data.loader import load_ticker
    from src.validation.walk_forward import walk_forward_signals, walk_forward_ml_signals
    from src.backtest.engine import BacktestEngine
    from src.evaluation.metrics import walk_forward_evaluate

    data_cfg = config["data"]
    feature_cfg = config["features"]
    wf_cfg = config["walk_forward"]
    bt_cfg = config["backtest"]
    eval_cfg = config["evaluation"]
    is_ml = "ml" in config
    name = config["experiment_name"]

    rows = []
    for ticker in data_cfg["tickers"]:
        raw = load_ticker(ticker, data_cfg["start_date"], data_cfg["end_date"])

        if is_ml:
            ml_cfg = config["ml"]
            signals, _ = walk_forward_ml_signals(
                raw, feature_cfg, ml_cfg["feature_columns"],
                train_size=wf_cfg["train_size"], test_size=wf_cfg["test_size"],
                threshold=ml_cfg.get("threshold", 0.05),
                model_type=ml_cfg.get("model_type", "logistic"),
            )
        else:
            strat = config["strategy"]
            signals = walk_forward_signals(
                raw, feature_cfg, strat["name"], strat["params"],
                train_size=wf_cfg["train_size"], test_size=wf_cfg["test_size"],
            )

        for cost in costs:
            engine = BacktestEngine(
                initial_capital=bt_cfg["initial_capital"],
                transaction_cost_bps=cost,
                slippage_bps=0,
                position_size_pct=bt_cfg["position_size_pct"],
            )
            result = engine.run(signals)
            per_window = walk_forward_evaluate(result, eval_cfg["metrics"], eval_cfg)

            rows.append({
                "strategy": name,
                "ticker": ticker,
                "cost_bps": cost,
                "mean_sharpe": per_window["sharpe"].mean(),
                "mean_turnover": per_window["turnover"].mean() if "turnover" in per_window else np.nan,
                "total_return": result["equity"].iloc[-1] / bt_cfg["initial_capital"] - 1,
            })
        print(f"  done: {name} / {ticker}")
    return rows


def load_configs(paths):
    configs = []
    for p in paths:
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        if "walk_forward" not in cfg or "portfolio" in cfg:
            print(f"  skipping {p} (not a single-strategy walk-forward config)")
            continue
        configs.append(cfg)
    return configs


def main():
    sys.path.append(str(Path(__file__).resolve().parents[1]))

    parser = argparse.ArgumentParser(description="Transaction-cost sensitivity sweep.")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Config files (default: config/walkforward_*.yaml)")
    parser.add_argument("--costs", nargs="*", type=float, default=DEFAULT_COSTS,
                        help="Total one-way cost levels in bps")
    parser.add_argument("--out", type=str, default="results/cost_sweep.csv")
    args = parser.parse_args()

    paths = args.configs or sorted(glob.glob("config/walkforward_*.yaml"))
    costs = sorted(args.costs)
    print(f"\nCost levels (bps): {costs}\nConfigs: {paths}\n")

    all_rows = []
    for cfg in load_configs(paths):
        print(f"=== {cfg['experiment_name']} ===")
        all_rows.extend(sweep_config(cfg, costs))

    df = pd.DataFrame(all_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)

    print("\n=== Mean Sharpe vs cost (average over tickers) ===")
    pivot = df.pivot_table(index="strategy", columns="cost_bps",
                           values="mean_sharpe", aggfunc="mean")
    print(pivot.round(3))

    print("\n=== Mean turnover per window (at lowest cost) ===")
    low = df[df["cost_bps"] == costs[0]]
    print(low.groupby("strategy")["mean_turnover"].mean().round(3))

    print("\n=== Breakeven cost, bps (Sharpe reaches 0) ===")
    max_cost = costs[-1]
    be_rows = []
    for strategy, g in df.groupby("strategy"):
        avg_curve = g.groupby("cost_bps")["mean_sharpe"].mean()
        row = {"strategy": strategy,
               "ALL (avg)": format_breakeven(breakeven_cost(avg_curve.index, avg_curve.values), max_cost)}
        for ticker, gt in g.groupby("ticker"):
            curve = gt.sort_values("cost_bps")
            row[ticker] = format_breakeven(breakeven_cost(curve["cost_bps"], curve["mean_sharpe"]), max_cost)
        be_rows.append(row)
    print(pd.DataFrame(be_rows).set_index("strategy"))

    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()