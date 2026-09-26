"""
Experiment runner.

This is the ONLY file you should need to touch to try a new idea:
point it at a config, get results out. Want to test a new feature, a
new strategy, or a different ticker universe? Copy default.yaml to a
new file (e.g. config/rsi_experiment.yaml), change what you want to
change, and run:

    python experiments/run_experiment.py --config config/rsi_experiment.yaml

That's the "feedback loop" piece of the pipeline: iterating on an idea
should never require touching pipeline code, only the config.
"""

import argparse
import sys
from pathlib import Path

import yaml

# Make `src` importable when running this script directly.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_pipeline


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run a quant research pipeline experiment.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to experiment config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"\n=== Running experiment: {config['experiment_name']} ===\n")

    results = run_pipeline(config)

    for ticker, result in results.items():
        print(f"--- {ticker} ---")
        print(result["comparison"].round(4))
        print()

    print("Done. Each ticker's full backtest DataFrames are available in the")
    print("returned results dict if you want to plot equity curves or dig deeper.")

    return results


if __name__ == "__main__":
    main()
