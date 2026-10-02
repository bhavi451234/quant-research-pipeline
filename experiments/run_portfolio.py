"""
Portfolio experiment runner.

Runs one shared strategy across a portfolio of tickers, each with its own
dollar allocation, and reports the combined portfolio's performance.

    python experiments/run_portfolio.py --config config/portfolio_single_strategy.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_portfolio_single_strategy, run_portfolio_best_per_ticker


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run a single-strategy portfolio backtest.")
    parser.add_argument(
        "--config", type=str, default="config/portfolio_single_strategy.yaml",
        help="Path to a portfolio experiment config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"\n=== Portfolio run: {config['experiment_name']} ===\n")

    if config["experiment_name"] == "portfolio_best_per_ticker":
        result = run_portfolio_best_per_ticker(config)
    else:
        result = run_portfolio_single_strategy(config)

    print("Allocations:")
    for ticker, amount in config["portfolio"]["allocations"].items():
        print(f"  {ticker}: {amount}")
    print()

    print("Portfolio per-window metrics:")
    print(result["per_window"].round(4))
    print()

    print("Summary across windows:")
    for key, value in result["summary"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    from src.evaluation.metrics import walk_forward_evaluate, summarize_windows

    eval_cfg = config["evaluation"]
    per_ticker_metric_names = eval_cfg.get("per_ticker_metrics", eval_cfg["metrics"])

    print()
    print("=== Per-ticker metrics (not combinable at the portfolio level) ===")
    for ticker, bt in result["ticker_backtests"].items():
        per_window = walk_forward_evaluate(bt, per_ticker_metric_names, eval_cfg)
        summary = summarize_windows(per_window)
        print(f"\n--- {ticker} ---")
        print(per_window.round(4))
        print("Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print()
    print("Each ticker's own individual backtest is also available in")
    print("result['ticker_backtests'] if you want to inspect them separately.")

    return result


if __name__ == "__main__":
    main()