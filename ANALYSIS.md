# Quant Research Pipeline: Analysis & Findings

This document summarizes the methodology, bugs found and fixed, and the
substantive findings produced by this project, a config-driven backtesting
pipeline supporting five rule-based strategies and two ML model types,
validated with walk-forward testing and extended to portfolio-level
backtesting.

## 1. Methodology

Every result in this document comes from **walk-forward validation**: the
price history is split into nine rolling windows (504 trading days of
training, 126 days of out-of-sample testing, non-overlapping), and every
strategy, rule-based or ML, is re-evaluated fresh on each window's test
period using only information available up to that point. No result here
comes from a single train/test split or from fitting once on the full
history.

Four tickers were used throughout: **AAPL, MSFT, TSLA, SNAP**, chosen to
span a large, low-volatility mega-cap (AAPL), a large stable tech name
(MSFT), a high-volatility, narrative-driven stock (TSLA), and a smaller,
more erratic name (SNAP).

## 2. Bugs found and fixed along the way

Three real bugs were discovered during development, each is worth noting
because each would have silently produced misleading results if left
unfixed:

- **`win_rate` miscounted at the per-day level.** The original
  implementation counted every day inside a multi-day trade separately,
  double- and triple-counting the same position. Fixed to identify
  individual trades (consecutive days holding the same position) and
  compute win rate per trade, not per day.
- **Sharpe and Sortino ratios became numerically unstable on windows with
  very few actual trades.** A window with only 2-4 real trading days could
  produce a wildly inflated or deflated ratio (one case: -23,514), driven
  by a near-zero but nonzero variance in the denominator. Fixed by adding
  a minimum-sample floor (`n_nonzero < 10` → `NaN`), the same philosophy
  applied consistently wherever a metric could be computed from too few
  observations to be meaningful.
- **A label-construction bug in the ML pipeline.** `NaN > threshold`
  evaluates to `False` in Python rather than `NaN`, which silently
  mislabeled the last, unknowable row of training data as "down" instead
  of leaving it correctly unlabeled. Fixed with an explicit `.isna()`
  check.

## 3. Rule-based strategies (Phase 3a)

Five strategies (SMA crossover, EMA crossover, momentum, RSI mean
reversion, Bollinger reversion) were walk-forward tested across all four
tickers. Mean Sharpe per strategy per ticker:

| Strategy | AAPL | MSFT | TSLA | SNAP |
|---|---|---|---|---|
| SMA crossover | 0.38 | 0.07 | 0.30 | 0.30 |
| EMA crossover | 0.46 | 0.27 | 0.31 | 0.62 |
| Momentum | 0.63 | -0.52 | 0.85 | -0.07 |
| RSI reversion | -0.05 | 0.13 | -0.70 | -0.12 |
| Bollinger reversion | -0.62 | 0.11 | -1.34 | 0.20 |

**Key finding: no single strategy dominates.** Momentum is TSLA's clear
best performer but is actively harmful on MSFT and SNAP. Bollinger
reversion, which bets on prices reverting to a mean, performs worst on
AAPL and TSLA specifically, both tickers that trended strongly over the
test period, a sensible result: mean-reversion strategies lose money when
the underlying assumption (reversion) doesn't hold.

Standard deviation of Sharpe across windows was, in every case, larger
than the mean Sharpe itself, meaning performance swung between strongly
positive and strongly negative windows depending on market regime. This
is the clearest evidence in the project that a single backtest number is
not sufficient: consistency across time matters as much as average
performance.

## 4. Machine learning strategies (Phase 3b)

A logistic regression model, retrained fresh at every walk-forward window
using 13 engineered features (price-relative-to-moving-average, RSI,
volatility, momentum at multiple horizons, Bollinger band position), was
compared against the five rule-based strategies.

| Ticker | Best rule-based (mean Sharpe) | Logistic regression (mean Sharpe) |
|---|---|---|
| AAPL | 0.63 (momentum) | **0.74** |
| MSFT | 0.27 (EMA) | **0.35** |
| TSLA | 0.85 (momentum) | 0.13 |
| SNAP | 0.62 (EMA) | **0.79** |

Logistic regression outperformed every rule-based strategy on AAPL, MSFT,
and SNAP, and was also the most *consistent* on AAPL specifically (std
Sharpe 0.77, versus 0.79-1.92 for every rule-based strategy on that
ticker).

### The accuracy-versus-profitability finding

A second metric, **direction accuracy** (how often the model correctly
called next-day direction on days it actually traded), was added
specifically to test whether strong Sharpe results implied strong
predictive accuracy. They did not: AAPL and SNAP's direction accuracy
averaged **~52-53% across windows, barely better than a coin flip**,
despite being the model's two best-performing tickers by Sharpe. This
indicates the model's edge came not from consistently correct direction
calls, but from a combination of selective trading (a confidence
threshold filters out low-conviction predictions) and favorable
risk/reward asymmetry on the trades it did make, a more sophisticated and
more honest explanation than "the model is good at predicting direction."

### Feature importance

Per-window logistic regression coefficients revealed two patterns
consistent with documented market phenomena:
- **Volatility aversion** (negative weight on recent volatility,
  predicting down-moves after volatility spikes) was the dominant signal
  on AAPL and MSFT.
- **Short-term momentum reversal** (negative weight on recent momentum,
  predicting a pullback after a run) was dominant on TSLA and SNAP.

RSI-14 carried negligible weight on every ticker (coefficients between
-0.011 and +0.014), suggesting it added little information once momentum
and volatility features were already included.

### Gradient boosting comparison

A second model type, gradient boosted trees, was built and compared under
identical conditions. Logistic regression won decisively on 3 of 4
tickers, both on mean Sharpe and on consistency; TSLA was a near-tie.

**A reproducibility finding emerged during this comparison**: re-running
the identical gradient boosting configuration twice in a row produced
different results (0.63 vs 0.57 mean Sharpe on AAPL), traced to
floating-point nondeterminism in how trees select splits when candidate
splits are nearly tied, a property `random_state` cannot control, since it
only governs scikit-learn's own intentional sampling, not low-level
summation order. Logistic regression showed no such variation. This adds
a second, independent reason to prefer the simpler model: it is not only
more accurate on average, it is also fully reproducible.

## 5. Portfolio-level backtesting (Phase 4)

Individual-ticker results were combined into a single portfolio, with
real, independent rupee allocations per ticker (summed at the equity
level, since price and position have no single meaningful combined value
across different assets, only money does).

| Portfolio | Mean Sharpe | Std Sharpe | % Positive Windows |
|---|---|---|---|
| Single strategy (logistic, all 4 tickers) | 0.73 | 1.38 | 67% |
| Best strategy per ticker (logistic ×3, momentum for TSLA) | 0.86 | 0.90 | 78% |

The best-per-ticker portfolio outperforms on every axis, but this
comparison carries an important caveat: each ticker's strategy was
selected *after* seeing which one performed best in earlier results. This
is a form of hindsight selection, and the resulting performance should be
read as an **upper bound**, not a forecast of what a prospectively-chosen
portfolio would have achieved. The single-strategy result, where one
approach was committed to in advance across the whole portfolio, is the
methodologically fairer estimate.

Three metrics (`win_rate`, `turnover`, `direction_accuracy`) could not be
computed at the combined portfolio level, since they depend on knowing a
single position or trade, which has no meaningful definition for a
portfolio that may be simultaneously long one asset and short another.
These remain available, correctly, at the per-ticker level.

## 6. Interactive dashboard (Phase 5)

A Streamlit dashboard provides three interactive views over this same
pipeline: single-ticker walk-forward results (any strategy, any ticker),
a side-by-side strategy comparison, and a portfolio builder supporting
both the single-strategy and best-per-ticker modes described above, with
the hindsight-selection caveat surfaced directly in the UI when relevant.

## 7. Summary

This project's most interesting findings were not simply "which strategy
had the highest Sharpe," but the structural insights that emerged from
testing rigorously: that consistency across market regimes matters as
much as average return, that a profitable model is not necessarily an
accurate one, that model complexity (gradient boosting) can underperform
a simpler, more stable alternative (logistic regression) both on average
performance and on reproducibility, and that portfolio-level comparisons
require explicit honesty about hindsight bias to be trustworthy.