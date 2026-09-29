"""
Walk-forward experiment runner.

The walk-forward counterpart to run_experiment.py: point it at a config,
get out a per-window metrics table and a cross-window summary for every
ticker, instead of a single train/test comparison.

    python experiments/run_walkforward.py --config config/walkforward_sma.yaml

Same rule as run_experiment.py: to try a new idea (different strategy,
different train/test window sizes, different tickers), edit the config,
never this file or the pipeline code underneath it.
"""

import argparse
import sys
from pathlib import Path

import yaml

# Make `src` importable when running this script directly.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_walkforward


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run a walk-forward validated experiment.")
    parser.add_argument(
        "--config", type=str, default="config/walkforward_sma.yaml",
        help="Path to a walk-forward experiment config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"\n=== Walk-forward run: {config['experiment_name']} ===\n")

    results = run_walkforward(config)

    for ticker, result in results.items():
        print(f"--- {ticker} ---")
        print(result["per_window"].round(4))
        print()
        print("Summary across windows:")
        for key, value in result["summary"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()

    print("Done. Each ticker's full stitched backtest DataFrame, per-window table,")
    print("and summary are available in the returned results dict.")

    return results


if __name__ == "__main__":
    main()