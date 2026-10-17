"""
Hyperparameter tuning (Phase 10).

Picks settings using ONLY data from before the evaluation period, so the
nine evaluation windows used everywhere else in this project never influence
the choice. The chosen values are then frozen and evaluated on those
untouched windows.

    python experiments/run_tuning.py
"""

import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_ticker
from src.pipeline import run_walkforward, run_walkforward_ml

TUNE_START = "2014-01-01"
TUNE_END = "2020-01-01"  # yfinance treats the end date as exclusive: data runs to 31 Dec 2019
MIN_WINDOWS = 29  # a setting needs at least this many valid (ticker, window) Sharpes to be eligible

ML_THRESHOLDS = [0.02, 0.05, 0.08, 0.10, 0.15]
ML_BASE_CONFIG = "config/walkforward_ml.yaml"

MOMENTUM_WINDOWS = [10, 20, 40]
MOMENTUM_THRESHOLDS = [0.0, 0.02, 0.04]
MOMENTUM_BASE_CONFIG = "config/walkforward_momentum.yaml"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def check_no_overlap(eval_cfg: dict) -> pd.Timestamp:
    """Refuse to run if the tuning data could reach into the evaluation period."""
    ticker = eval_cfg["data"]["tickers"][0]
    raw = load_ticker(ticker, eval_cfg["data"]["start_date"], eval_cfg["data"]["end_date"])
    first_test_day = raw.index[eval_cfg["walk_forward"]["train_size"]]
    if pd.Timestamp(TUNE_END) > first_test_day:
        raise ValueError(
            f"Tuning data ends {TUNE_END}, which is after the first evaluation test day "
            f"{first_test_day.date()}. Tuning must use only earlier data."
        )
    return first_test_day


def tuning_config(base_cfg: dict) -> dict:
    """A copy of the config pointed at the tuning period instead of the evaluation period."""
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["start_date"] = TUNE_START
    cfg["data"]["end_date"] = TUNE_END
    return cfg


def pooled_mean_sharpe(results: dict) -> tuple:
    """Mean Sharpe over every (ticker, window) pair, and how many pairs that was."""
    values = []
    for res in results.values():
        values.extend(res["per_window"]["sharpe"].dropna().tolist())
    if not values:
        return float("nan"), 0
    return sum(values) / len(values), len(values)


def grid_row(results: dict) -> dict:
    """Per-ticker mean Sharpe plus the pooled figure, for one setting."""
    row = {ticker: res["summary"]["mean_sharpe"] for ticker, res in results.items()}
    row["pooled_mean_sharpe"], row["n_windows"] = pooled_mean_sharpe(results)
    return row


def tune_ml(base: dict) -> pd.DataFrame:
    rows = []
    for threshold in ML_THRESHOLDS:
        cfg = tuning_config(base)
        cfg["ml"]["threshold"] = threshold
        rows.append({"threshold": threshold, **grid_row(run_walkforward_ml(cfg))})
        print(f"done: logistic threshold {threshold}")
    return pd.DataFrame(rows)


def tune_momentum() -> pd.DataFrame:
    base = load_config(MOMENTUM_BASE_CONFIG)
    rows = []
    for window in MOMENTUM_WINDOWS:
        for threshold in MOMENTUM_THRESHOLDS:
            cfg = tuning_config(base)
            cfg["features"] = [{"name": "momentum", "params": {"window": window}}]
            cfg["strategy"]["params"]["momentum_window"] = window
            cfg["strategy"]["params"]["threshold"] = threshold
            rows.append({"window": window, "threshold": threshold, **grid_row(run_walkforward(cfg))})
            print(f"done: momentum window {window}, threshold {threshold}")
    return pd.DataFrame(rows)


def report(title: str, table: pd.DataFrame, out_path: str, setting_columns: list) -> None:
    pd.set_option("display.width", 200)
    print(f"\n=== {title}: mean Sharpe on the tuning period ===")
    print(table.round(3).to_string(index=False))

    eligible = table[table["n_windows"] >= MIN_WINDOWS]
    if eligible.empty:
        print(f"\nNo setting had at least {MIN_WINDOWS} valid windows.")
    else:
        best = eligible.loc[eligible["pooled_mean_sharpe"].idxmax()]
        chosen = ", ".join(f"{c} = {best[c]}" for c in setting_columns)
        print(f"\nBest setting (needs at least {MIN_WINDOWS} valid windows): {chosen}")

    table.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


def main():
    ml_base = load_config(ML_BASE_CONFIG)
    first_test_day = check_no_overlap(ml_base)
    print(f"\nTuning on {TUNE_START} to 2019-12-31; evaluation starts {first_test_day.date()}.\n")

    report("Logistic regression threshold", tune_ml(ml_base), "results/tuning_ml.csv", ["threshold"])
    report("Momentum", tune_momentum(), "results/tuning_momentum.csv", ["window", "threshold"])


if __name__ == "__main__":
    main()