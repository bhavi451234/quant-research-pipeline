# Quant Research Pipeline

[![tests](https://github.com/bhavi451234/quant-research-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/bhavi451234/quant-research-pipeline/actions/workflows/tests.yml)

An end-to-end research pipeline for testing trading strategies honestly:
data → features → strategy → backtest → walk-forward evaluation → portfolio,
all driven by config files. Seven strategies (five rule-based, two ML) were
tested on seven stocks over eleven out-of-sample windows (2020-2025), with
transaction costs, significance tests and a market benchmark.

## What I found

**No strategy beat simply holding the stock, and none beat the S&P 500.**
What the strategies did buy was a smaller worst drop.

Mean Sharpe ratio, averaged over AAPL, MSFT, TSLA, SNAP, JPM, XOM and JNJ,
out of sample, after 7 bps of costs per trade:

| Strategy | Mean Sharpe | Worst drop vs buy-and-hold |
|---|---|---|
| SPY buy-and-hold (benchmark) | 0.93 | n/a |
| Stock buy-and-hold (avg of the 7) | 0.61 | n/a |
| Logistic regression | 0.16 | 12 points shallower * |
| SMA crossover | 0.12 | 8 points shallower * |
| EMA crossover | -0.01 | 7 points shallower * |
| Momentum | -0.23 | no difference |
| Gradient boosting | -0.26 | 5 points shallower * |
| RSI reversion | -0.62 | 14 points shallower * |
| Bollinger reversion | -0.76 | 16 points shallower * |

\* 95% bootstrap interval excludes zero. Part of the shallower drawdown is just
less market exposure, since the strategies are often in cash.

| Portfolio (Rs 5.1 lakh over 7 stocks) | Total return | CAGR | Mean Sharpe |
|---|---|---|---|
| SPY buy-and-hold | 109.0% | 14.3% | 0.93 |
| One strategy (logistic) on all 7 | 50.4% | 7.7% | 0.407 |
| Best strategy per stock, chosen on 2014-2019 data only | 57.1% | 8.6% | 0.399 |

Other results, all tested rather than assumed:

- **A four-stock result did not generalize.** On four tech stocks the logistic
  regression looked good (0.44). Adding a bank, an energy company and a
  healthcare company cut it to 0.16.
- **Picking winners from the past did not help.** Choosing each stock's
  strategy from 2014-2019 data scored 0.399, against 0.407 for one strategy
  everywhere.
- **Tuning made things worse.** Settings tuned on 2014-2019 did worse than the
  original settings on 2020-2025, for both the logistic threshold and momentum.
- **Gradient boosting is not viable.** It trades five times as often as the
  logistic regression, and its Sharpe reaches zero at about 1 bp of cost.
- **Four bugs were found and fixed**, including transaction costs charged at
  roughly 1/500th of their true size, which had made every earlier result
  effectively cost-free.

The full write-up, with every table, the methodology and the limitations, is in
[ANALYSIS.md](ANALYSIS.md).

## How it avoids fooling itself

- **Walk-forward validation.** Each strategy is refit on 504 trading days and
  tested on the next 126, rolling forward. There is no single train/test split.
- **No lookahead.** Signals are shifted one bar before they can be traded, and
  features only use past data. `tests/test_no_lookahead.py` checks this.
- **Real costs.** Every trade pays transaction cost plus slippage (7 bps
  total), and a cost sensitivity sweep shows where each strategy breaks even.
- **Settings are never chosen on the evaluation period.** Tuning and strategy
  selection use 2014-2019 data only, and the code refuses to run if that data
  reaches the evaluation period.
- **Pinned data.** Prices are committed under `data/cache/`, so every run uses
  identical data and gives identical numbers.
- **A benchmark and significance tests.** Everything is compared with SPY and
  with buy-and-hold, and a paired bootstrap says which gaps are real and which
  are noise.

## Reproduce the results

```bash
git clone https://github.com/bhavi451234/quant-research-pipeline.git
cd quant-research-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./reproduce.sh
```

`reproduce.sh` reruns the benchmark, significance tests, tuning, strategy
selection, cost sweep and both portfolios from the committed price snapshot
in under a minute, and writes the output to `results/logs/`.

To run a single walk-forward experiment on its own:

```bash
python experiments/run_walkforward.py --config config/walkforward_sma.yaml
python experiments/run_walkforward_ml.py --config config/walkforward_ml.yaml
python experiments/run_portfolio.py --config config/portfolio_single_strategy.yaml
```

## Dashboard

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/Home.py
```

The Strategy Comparison page ranks all strategies on one stock by total return,
CAGR or Sharpe, next to what holding the stock and holding SPY would have
returned over the same days. Single Ticker shows one strategy's walk-forward
results, and Portfolio Builder combines stocks with real rupee allocations.

## Project layout

```
config/            one YAML file per experiment: data, features, strategy, costs
data/cache/        pinned price snapshot, one CSV per ticker (2013-2025)
src/data/          loader: reads the pinned snapshot, slices by date
src/features/      pure functions: price data in, named feature out
src/strategies/    pure functions: features in, signal (+1 / -1 / 0) out
src/backtest/      signal in, equity curve out, applies costs and slippage
src/evaluation/    equity curve in, Sharpe / drawdown / alpha / beta out
src/pipeline.py    runs all of the above for one config (walk-forward, ML, portfolio)
experiments/       command-line scripts: benchmark, significance, tuning,
                   selection, cost sweep, portfolios
dashboard/         Streamlit app
tests/             unit tests, run automatically on every push
results/           saved outputs behind every table in ANALYSIS.md
```

Each layer only knows about the layer directly below it, so you can swap a data
source, add a feature or try a new strategy without touching the rest.

## Try a new idea

1. Copy a walk-forward config, for example `config/walkforward_sma.yaml`, to
   `config/my_idea.yaml`.
2. Change what you want: the tickers, the strategy parameters, the costs.
3. Run `python experiments/run_walkforward.py --config config/my_idea.yaml`.

To add a feature, write a function in `src/features/library.py` and register it
in `FEATURE_REGISTRY`. To add a strategy, write a function in
`src/strategies/library.py` and register it in `STRATEGY_REGISTRY`. Then
reference either by name in a config. New dates must lie inside the pinned
range (2013-01-01 to 2025-12-31), or the loader will raise an error rather
than silently download new data.

## Strategies

Rule-based: `sma_crossover`, `ema_crossover`, `momentum_strategy`,
`rsi_mean_reversion`, `bollinger_reversion`. ML: logistic regression and
gradient boosting, retrained at every walk-forward window on 13 engineered
features.

## Tests

```bash
python -m pytest -q
```

The suite covers the metrics, the walk-forward splitting, the no-lookahead
guarantee, transaction-cost accounting (a round trip costs exactly two trades'
worth of basis points) and the bootstrap, and runs in GitHub Actions on every
push. `tests/smoke_test.py` runs the whole pipeline on synthetic prices, with no
network access.

## Limitations

- Seven stocks and eleven windows give low statistical power. Most gaps between
  strategies are inconclusive, and "inconclusive" means "cannot tell", not
  "equal".
- Costs are a flat per-trade charge, with no market impact.
- The tuning period (2014-2019) is short, and every strategy scored negatively
  on average in it, so the tuning signal is weak.
- The stocks are large, liquid US companies that mostly rose over the test
  period, which is hard for any trading strategy to beat.
- Results are tied to the committed price snapshot. Refreshing the data changes
  the numbers slightly.