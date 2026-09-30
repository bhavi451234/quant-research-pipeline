"""
ML walk-forward experiment runner.

Same idea as run_walkforward.py, but for the ML strategy: a fresh model
gets fit at each window rather than a fixed rule being applied.

    python experiments/run_walkforward_ml.py --config config/walkforward_ml.yaml

Same rule as everywhere else: to try a different feature set, model
threshold, or ticker universe, edit the config, never this file.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_walkforward_ml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run the ML walk-forward strategy.")
    parser.add_argument(
        "--config", type=str, default="config/walkforward_ml.yaml",
        help="Path to an ML walk-forward experiment config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"\n=== ML walk-forward run: {config['experiment_name']} ===\n")

    results = run_walkforward_ml(config)

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

        print("Mean coefficient per feature, across windows:")
        print(result["coefficients"].mean().round(4).sort_values())
        print()

    print("Done. Each ticker's full stitched backtest DataFrame, per-window table,")
    print("and summary are available in the returned results dict.")
   

    return results


if __name__ == "__main__":
    main()