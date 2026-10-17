#!/usr/bin/env bash
# Regenerates every result in ANALYSIS.md from the committed price snapshot.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs

run() {
  name=$1; shift
  echo ">> $name"
  python "$@" | tee "results/logs/$name.txt" > /dev/null
}

run benchmark              experiments/run_benchmark.py
run significance           experiments/run_significance.py
run tuning                 experiments/run_tuning.py
run cost_sweep             experiments/run_cost_sweep.py
run selection              experiments/run_selection.py
run portfolio_single       experiments/run_portfolio.py --config config/portfolio_single_strategy.yaml
run portfolio_best_per_ticker experiments/run_portfolio.py --config config/portfolio_best_per_ticker.yaml

echo "Done. Outputs are in results/logs/"