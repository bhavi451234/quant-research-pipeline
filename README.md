# Quant Research Pipeline

An end-to-end research pipeline: data → features → strategy → backtest → evaluation, all driven by a single config file. Built so that testing a new idea never requires touching pipeline code — only the config.

## Why this exists

Most backtests reported by students (and plenty of professionals) are quietly lying: they use data with lookahead bias, ignore transaction costs, and never check whether performance holds up out-of-sample. This pipeline is built specifically to make those failure modes hard to hide:

- **No lookahead** — signals are shifted one bar before being tradable; features only ever use past data.
- **Real costs** — every trade pays transaction cost + slippage in basis points.
- **Honest evaluation** — every experiment reports train (in-sample) vs test (out-of-sample) metrics side by side, with the delta. A strategy whose Sharpe collapses out-of-sample is telling you it was overfit — the comparison table makes that unmissable instead of letting you quietly report only the best number.

## Architecture

```
config/default.yaml      <- defines an entire experiment: data, features, strategy, costs, metrics
src/data/loader.py        <- pulls + caches OHLCV data, does train/test split. Knows nothing about features or strategies.
src/features/library.py   <- pure functions: price data in, named feature out. No lookahead by construction.
src/strategies/library.py <- pure functions: featured data in, signal (+1/-1/0) out. Knows nothing about costs.
src/backtest/engine.py    <- signal in, equity curve out. Applies costs + slippage. Knows nothing about SMA/RSI/etc.
src/evaluation/metrics.py <- equity curve in, Sharpe/Sortino/drawdown/etc out. Produces the train-vs-test comparison.
src/pipeline.py           <- orchestrates all of the above for one config, one ticker (or a universe of tickers).
experiments/run_experiment.py <- CLI entry point. The only file you touch to try a new idea.
```

Each layer only knows about the layer directly below it. This is deliberate: it's what lets you swap a data source, add a feature, or try a new strategy without ripping up the rest of the system — which is the actual point of calling this a "pipeline" rather than a script.

## Running an experiment

```bash
pip install -r requirements.txt
python experiments/run_experiment.py --config config/default.yaml
```

This prints a train-vs-test metrics table for every ticker in the config's universe.

## Trying a new idea

1. Copy `config/default.yaml` to `config/my_idea.yaml`
2. Change whatever you want — add a feature, swap the strategy, change the ticker universe, change the cost assumptions
3. Run: `python experiments/run_experiment.py --config config/my_idea.yaml`
4. Compare the train-vs-test table against your baseline run

No code changes required for any of the above.

## Adding a new feature

Add a function to `src/features/library.py`, register it in `FEATURE_REGISTRY`, then reference it by name in any config's `features` list.

## Adding a new strategy

Add a function to `src/strategies/library.py` that reads whatever feature columns it needs and returns a signal Series, register it in `STRATEGY_REGISTRY`, then reference it by name in any config's `strategy` section.

## Current strategies

- `sma_crossover` — long when fast SMA > slow SMA, flat otherwise
- `rsi_mean_reversion` — long when RSI < oversold threshold, short when RSI > overbought threshold, flat otherwise

## Testing without network access

`tests/smoke_test.py` runs the entire pipeline (features → strategy → backtest → evaluation) against synthetic price data, so the pipeline's correctness can be verified without hitting a live data provider. Run with:

```bash
python tests/smoke_test.py
```

## Status / what's next

- [x] Data ingestion + caching
- [x] Feature library (SMA, EMA, RSI, volatility, momentum)
- [x] Cost-aware backtest engine
- [x] Train/test evaluation with honest comparison table
- [x] Config-driven experiment runner
- [ ] Plug in the strategies already built in the finance-terminal project (EMA/PSI-based)
- [ ] Add walk-forward validation (rolling train/test windows, not just one split)
- [ ] Equity curve + drawdown plotting
- [ ] Multi-ticker portfolio-level backtest (currently runs per-ticker independently)
