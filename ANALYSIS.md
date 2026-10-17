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
history. The first out-of-sample day is 2020-01-03.

Four tickers were used throughout: **AAPL, MSFT, TSLA, SNAP**, chosen to
span a large, low-volatility mega-cap (AAPL), a large stable tech name
(MSFT), a high-volatility, narrative-driven stock (TSLA), and a smaller,
more erratic name (SNAP).

**Data.** Prices come from Yahoo Finance through `yfinance`, adjusted for
splits and dividends, and are pinned: the first download of each ticker and
date range is saved under `data/cache/` and committed to the repository, and
every run reads that snapshot instead of downloading again. Earlier versions
downloaded fresh data on every run, which made results drift from run to run
(Section 12). Every number in this document was generated from the snapshot.

## 2. Bugs found and fixed along the way

Four real bugs were discovered during development. Each is worth noting
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
- **Transaction costs were charged at a fraction of their true size.** The
  backtest engine computed cost as `|trade| × share price × bps`, but
  `trade` is a fraction of capital, so cost must scale with capital, not
  with share price. For a $200 stock with $100,000 of capital the charge
  was about 500× too small (and far more for lower-priced stocks like
  SNAP), so every result in earlier versions of this document was
  effectively cost-free. Fixed by charging cost on traded capital, and
  pinned with regression tests: a round trip costs exactly two trades'
  worth of basis points, a long-to-short flip costs double, and cost does
  not depend on share price. All results below were regenerated after the
  fix.

## 3. Rule-based strategies (Phase 3a)

Five strategies (SMA crossover, EMA crossover, momentum, RSI mean
reversion, Bollinger reversion) were walk-forward tested across all four
tickers. These use default settings chosen without validation; Section 11
re-tests the two most influential ones on earlier data. Mean Sharpe per
strategy per ticker, after transaction costs (7 bps: 5 bps transaction cost
plus 2 bps slippage):

| Strategy | AAPL | MSFT | TSLA | SNAP |
|---|---|---|---|---|
| SMA crossover | 0.36 | 0.05 | 0.29 | 0.49 |
| EMA crossover | 0.45 | 0.25 | 0.30 | 0.61 |
| Momentum | 0.53 | -0.67 | 0.80 | -0.13 |
| RSI reversion | -0.30 | -0.08 | -0.74 | -0.58 |
| Bollinger reversion | -0.90 | -0.15 | -1.43 | 0.10 |

**Key finding: no single strategy dominates.** Momentum is TSLA's clear
best performer but is actively harmful on MSFT and SNAP. Bollinger
reversion, which bets on prices reverting to a mean, performs worst on
AAPL and TSLA specifically, both tickers that trended strongly over the
test period, a sensible result: mean-reversion strategies lose money when
the underlying assumption (reversion) doesn't hold.

The standard deviation of Sharpe across windows exceeded the absolute mean
Sharpe in 19 of 20 strategy-ticker combinations, meaning performance
swung between strongly positive and strongly negative windows depending on
market regime. The one exception is Bollinger reversion on TSLA (mean
-1.43, std 1.21), which lost money consistently rather than swinging. This
is the clearest evidence in the project that a single backtest number is
not sufficient: consistency across time matters as much as average
performance.

## 4. Machine learning strategies (Phase 3b)

A logistic regression model, retrained fresh at every walk-forward window
using 13 engineered features (price-relative-to-moving-average, RSI,
volatility, momentum at multiple horizons, Bollinger band position), was
compared against the five rule-based strategies.

| Ticker | Best rule-based (mean Sharpe, after costs) | Logistic regression (mean Sharpe, after costs) |
|---|---|---|
| AAPL | 0.53 (momentum) | **0.67** |
| MSFT | 0.25 (EMA) | **0.28** |
| TSLA | 0.80 (momentum) | 0.07 |
| SNAP | 0.61 (EMA) | **0.74** |

Logistic regression outperformed every rule-based strategy on AAPL and
SNAP, and narrowly on MSFT (0.28 vs 0.25). On AAPL it was also the most
consistent of the strategies with a positive mean Sharpe (std 0.81,
versus 1.01 for momentum and 1.90-1.91 for EMA and SMA). Only RSI
reversion had a lower std on that ticker (0.78), but with a negative mean
Sharpe (-0.30).

### The accuracy-versus-profitability finding

A second metric, **direction accuracy** (how often the model correctly
called next-day direction on days it actually traded), was added
specifically to test whether strong Sharpe results implied strong
predictive accuracy. They did not: averaged across windows (skipping
windows with too few trades to compute it), direction accuracy was about
**53% on AAPL and 50% on SNAP, no better than a coin flip or barely better**,
despite these being the model's two best-performing tickers by Sharpe. This
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
identical conditions. Logistic regression had the higher mean Sharpe on
all four tickers (decisively on AAPL, MSFT and SNAP; narrowly on TSLA,
0.07 vs -0.02). On consistency the picture is mixed: logistic regression
had the lower window-to-window standard deviation on AAPL (0.81 vs 1.30)
and MSFT (1.18 vs 1.47), while gradient boosting had the lower figure on
TSLA (0.99 vs 1.21) and SNAP (1.22 vs 1.34), though with a mean Sharpe at
or below zero on both. After costs the gap widens further, because
gradient boosting trades more than four times as often (Section 8).
None of these Sharpe gaps is statistically significant at nine windows
(Section 10); the case against gradient boosting rests on its costs
(Section 8), not on its raw Sharpe.

### A reproducibility finding, and a wrong explanation

Early in the project, identical configurations gave different results on
different runs: gradient boosting on AAPL scored 0.63 and then 0.57, and
later 0.35 and then 0.25 in the cost sweep, and logistic regression on AAPL
moved from 0.66 to 0.64. The gradient boosting variation was first attributed
to floating-point nondeterminism in how trees choose between nearly tied
splits. That explanation was wrong. Once the input prices were pinned to a
saved snapshot (Section 12), repeated full runs agreed on every mean Sharpe
for both models, to every printed digit. The variation came from the input
data, which was re-downloaded from Yahoo Finance on every run, and not from
the models. Both models are reproducible, so reproducibility is not a reason
to prefer one over the other; the remaining reasons are Sharpe, simplicity and
transaction costs.

## 5. Portfolio-level backtesting (Phase 4)

Individual-ticker results were combined into a single portfolio, with
real, independent rupee allocations per ticker (summed at the equity
level, since price and position have no single meaningful combined value
across different assets, only money does).

| Portfolio | Mean Sharpe | Std Sharpe | % Positive Windows |
|---|---|---|---|
| Single strategy (logistic, all 4 tickers) | 0.643 | 1.378 | 67% |
| Best strategy per ticker (logistic ×3, momentum for TSLA) | 0.785 | 1.459 | 67% |

After the cost fix the single-strategy portfolio fell from 0.73 to 0.64, and
the best-per-ticker portfolio fell as well, as expected once costs are
charged properly.

The best-per-ticker portfolio has the higher mean Sharpe, but it is not
better on every axis: its window-to-window standard deviation is also higher
(1.459 vs 1.378) and the share of positive windows is the same (67%). The
comparison also carries an important caveat: each ticker's strategy was
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
a simpler alternative (logistic regression) both on average performance
and after transaction costs, and that portfolio-level comparisons
require explicit honesty about hindsight bias to be trustworthy.

Costs added a further lesson. Sensitivity to transaction costs is driven
by turnover: slow crossover strategies and logistic regression survive
realistic costs, while gradient boosting, which trades over four times as
often as logistic regression, does not. The mean-reversion strategies
(RSI, Bollinger) fail even at zero cost, so no cost assumption rescues
them. Finding that the original cost model undercharged by a factor of
hundreds is itself a reminder to test the plumbing, not just the
strategies.

The later phases tested these findings harder, and the honest picture is more
modest. No strategy beat the S&P 500 (SPY) on Sharpe on any of the four
stocks (Section 9). Only a few differences are statistically distinguishable
from noise at nine windows; the most robust positive result is that logistic
regression reduces the worst drawdown relative to buy-and-hold (Section 10).
Tuning a setting on earlier data did not help out of sample (Section 11). And
all results are tied to a pinned data snapshot so they can be reproduced
exactly (Section 12).

## 8. Transaction-cost sensitivity (Phase 12)

Because earlier versions undercharged costs (Section 2), this section
measures how much costs actually matter. For each strategy and ticker, the
out-of-sample signals were generated once, then the same signals were
re-backtested at total one-way costs of 0, 5, 7, 10 and 20 basis points (7
is the project default: 5 bps transaction cost plus 2 bps slippage). A
round trip costs twice the one-way figure.

**Mean Sharpe vs total one-way cost (average over AAPL, MSFT, TSLA, SNAP)**

| Strategy | 0 bps | 5 bps | 7 bps | 10 bps | 20 bps |
|---|---|---|---|---|---|
| Logistic regression | 0.517 | 0.458 | 0.439 | 0.410 | 0.314 |
| EMA crossover | 0.413 | 0.404 | 0.401 | 0.395 | 0.378 |
| SMA crossover | 0.312 | 0.302 | 0.297 | 0.291 | 0.270 |
| Momentum | 0.223 | 0.159 | 0.133 | 0.094 | -0.035 |
| Gradient boosting | 0.247 | 0.073 | 0.004 | -0.099 | -0.443 |
| Momentum (tuned, Section 11) | -0.184 | -0.232 | -0.251 | -0.279 | -0.374 |
| RSI reversion | -0.581 | -0.387 | -0.423 | -0.478 | -0.661 |
| Bollinger reversion | -0.385 | -0.541 | -0.593 | -0.671 | -0.926 |

**Breakeven cost (bps at which mean Sharpe reaches zero) and turnover**

| Strategy | All (avg) | AAPL | MSFT | SNAP | TSLA | Mean turnover |
|---|---|---|---|---|---|---|
| Logistic regression | >20 | >20 | >20 | >20 | 16.1 | 0.094 |
| EMA crossover | >20 | >20 | >20 | >20 | >20 | 0.017 |
| SMA crossover | >20 | >20 | >20 | >20 | >20 | 0.022 |
| Momentum | 17.3 | >20 | never | never | >20 | 0.180 |
| Momentum (tuned) | never | never | never | never | >20 | 0.139 |
| Gradient boosting | 7.1 | 14.0 | 2.8 | 3.3 | 6.1 | 0.425 |
| RSI reversion | never | never | never | never | never | 0.073 |
| Bollinger reversion | never | never | 3.7 | 14.2 | never | 0.118 |

("never" means Sharpe is already at or below zero with no costs; ">20"
means still positive at the highest cost tested.)

**Findings.**

- Cost damage scales with turnover. Gradient boosting trades the most
  (turnover 0.425) and loses almost all its Sharpe by the 7 bps default;
  the crossover strategies trade rarely and barely notice costs.
- Logistic regression remains the best strategy up to about 10 bps, but
  EMA crossover overtakes it at 20 bps (0.378 vs 0.314).
- Gradient boosting is not viable at realistic costs: its breakeven is
  about 7 bps overall and under 4 bps on MSFT and SNAP. This widens the
  logistic-vs-gradient-boosting gap found in Section 4.
- The two mean-reversion strategies (RSI, Bollinger) have negative
  Sharpe even at zero cost, so costs are not their problem; the signals
  do not work on these four stocks over this period.
- The tuned momentum setting (Section 11) is worse than the original at
  every cost level, for example -0.251 vs 0.133 at 7 bps.

Section 10 tests these gaps for statistical significance.

**Limitations.** Costs are modelled as a flat per-trade charge, with no
market impact or size dependence. With four tickers, the gaps between the
top strategies at low costs are small relative to window-to-window
variation (Section 10 tests this). At 0 bps the RSI row is lower than at
5 bps, which should be impossible since costs can only reduce returns; this
appears to be an artifact of the Sharpe minimum-sample rule at exactly zero
cost, not a real effect.

## 9. Market benchmark (Phase 8)

A strategy has to beat the market, not just earn a positive Sharpe. The
benchmark is buy-and-hold of SPY (an S&P 500 ETF) over the same nine test
windows, which has a mean Sharpe of **1.01**. Each stock's own buy-and-hold is
shown too. Alpha and beta come from regressing the strategy's daily excess
returns (over a 5% annual risk-free rate) on SPY's daily excess returns. Alpha
is annualized, and beta is measured on the strategy's return stream, not on the
stock's.

**Mean Sharpe / annualized alpha (%) / beta vs SPY**

| Strategy | AAPL | MSFT | TSLA | SNAP |
|---|---|---|---|---|
| Buy and hold | 0.96 / 12.6 / 1.21 | 1.01 / 11.7 / 1.20 | 0.88 / 48.0 / 1.55 | 0.61 / 11.8 / 1.50 |
| SMA crossover | 0.36 / 7.1 / 0.44 | 0.05 / -0.1 / 0.61 | 0.29 / 21.3 / 0.91 | 0.49 / 16.2 / 0.38 |
| EMA crossover | 0.45 / 9.7 / 0.52 | 0.25 / 6.1 / 0.56 | 0.30 / 21.7 / 0.90 | 0.61 / 18.4 / 0.35 |
| Momentum | 0.53 / 20.4 / -0.47 | -0.67 / -12.3 / -0.40 | 0.80 / 70.6 / -0.39 | -0.13 / -8.1 / -0.56 |
| RSI reversion | -0.30 / -3.6 / -0.03 | -0.08 / 0.6 / -0.04 | -0.74 / -40.4 / 0.02 | -0.58 / -3.0 / 0.28 |
| Bollinger reversion | -0.90 / -4.8 / 0.10 | -0.15 / 3.9 / 0.08 | -1.43 / -38.9 / -0.01 | 0.10 / 9.0 / 0.04 |
| Logistic regression | 0.67 / 8.4 / 0.77 | 0.28 / -4.5 / 0.52 | 0.07 / 1.4 / 0.19 | 0.74 / 29.2 / 0.07 |
| Gradient boosting | 0.32 / 9.4 / 0.04 | -0.19 / -10.3 / 0.58 | -0.02 / 3.5 / 0.50 | -0.09 / -8.8 / 0.23 |

**Findings.**

- No strategy beat SPY's Sharpe of 1.01 on any of the four stocks. Even
  holding each stock fell short (by 0.00 on MSFT up to 0.40 on SNAP). The
  closest strategies were momentum on TSLA (0.21 below SPY) and logistic
  regression on SNAP (0.27 below).
- Buy-and-hold alpha is positive on all four stocks (11.7% to 48.0% a year)
  because these stocks outperformed SPY over the period. Strategy alphas are
  mixed, and the large ones (momentum on TSLA at +70.6%) come with very
  volatile window-to-window results.
- The strategies carry much less market exposure than the stocks they trade
  (betas of roughly 0 to 0.9, against 1.2 to 1.55 for buy-and-hold), because
  they are often flat. This is also why they have shallower drawdowns
  (Section 10).
- Momentum has a negative beta on all four stocks (-0.39 to -0.56). This was
  not investigated; one hypothesis is its timing around large market moves.
- Logistic regression's beta varies widely by stock (0.77 on AAPL, 0.07 on
  SNAP) even though the model is the same, because beta measures the
  strategy's returns, which depend on when it was invested.

**Limitations.** Alpha and beta come from about four and a half years of daily
returns, and none of the alpha estimates were tested for statistical
significance. The risk-free rate is a flat 5%.

## 10. Statistical significance (Phase 9)

Every comparison so far rests on the mean of nine window-level Sharpe
ratios, which vary a lot from window to window. This section asks which gaps
are distinguishable from noise, using a paired bootstrap over the nine
windows: for each pair of strategies, the nine per-window differences are
resampled with replacement 10,000 times, and the middle 95% of the resampled
means forms a confidence interval. A gap is called significant only if that
interval excludes zero. "Pooled" averages the four stocks within each window
first, so each strategy contributes nine values.

**Strategy minus buy-and-hold (pooled across AAPL, MSFT, TSLA, SNAP)**

| Strategy | Sharpe difference [95% interval] | Max drawdown difference [95% interval] |
|---|---|---|
| Logistic regression | -0.43 [-1.29, +0.36] | +0.14 [+0.10, +0.19] * |
| EMA crossover | -0.60 [-0.90, -0.29] * | +0.09 [+0.02, +0.17] * |
| SMA crossover | -0.65 [-1.02, -0.29] * | +0.09 [+0.03, +0.16] * |
| Momentum | -0.73 [-1.32, -0.12] * | +0.00 [-0.03, +0.04] |
| Gradient boosting | -0.86 [-1.70, -0.09] * | +0.07 [+0.03, +0.11] * |
| RSI reversion | -1.12 [-2.21, +0.25] | +0.17 [+0.10, +0.25] * |
| Bollinger reversion | -1.46 [-2.41, -0.42] * | +0.20 [+0.14, +0.28] * |

(* = interval excludes zero. For drawdown a positive difference means a
shallower worst drop than holding the stock.)

**Findings.**

- On risk-adjusted return, five of the seven strategies are significantly
  worse than buying and holding. Logistic regression and RSI are inconclusive.
  Logistic regression is not significantly worse, but it is not better either.
- On max drawdown the strategies do better: logistic regression has a
  significantly shallower worst drop than buy-and-hold on all four stocks.
  Part of this is lower market exposure, since the strategies are often flat.
- Logistic regression versus gradient boosting is not significant on any
  stock (pooled +0.43, interval [-0.36, +1.17]); the case against gradient
  boosting rests on its transaction costs (Section 8), not on its Sharpe.
- Logistic regression versus the best rule-based strategy on each stock is
  inconclusive everywhere.
- Almost no strategy's mean Sharpe is distinguishable from zero. The one
  exception is logistic regression on AAPL (+0.67, [+0.18, +1.21]); even
  buy-and-hold on AAPL is not significant (+0.96, [-0.04, +2.02]). Pooled
  across the four stocks, logistic regression's mean Sharpe is +0.44 with an
  interval of [-0.04, +0.94], which only just includes zero.

**Limitations.** Nine windows give low power, so wide intervals are expected
and "inconclusive" means "cannot tell", not "equal". Dozens of comparisons
were run, so a few starred results are expected by chance. Windows are not
fully independent, since market regimes persist. The best rule-based
strategy per stock was chosen after the fact, which makes that comparison
conservative for logistic regression.

## 11. Hyperparameter tuning (Phase 10)

Settings such as the logistic confidence threshold (0.05) and the momentum
window and threshold (20 days, 0.02) were originally chosen by reasonable
guess. This section tests them properly: settings are chosen on data from
2014 to 2019 only, using the same walk-forward machinery (504 training days,
126 test days) and the same 7 bps costs, then frozen and evaluated on the
untouched 2020-2024 windows used everywhere else. The script refuses to run
if the tuning data reaches into the evaluation period. The tuning data and the
training data of the first evaluation windows share 2018-2019, but no
evaluation test day was used in tuning. The grids were kept deliberately
small, one setting for all four stocks, to limit overfitting to the tuning
data, and a setting was only eligible to win if at least 15 of the 22 possible
(stock, window) Sharpe ratios were valid.

**Logistic regression threshold (mean Sharpe on the tuning period)**

| Threshold | Pooled mean Sharpe | Valid windows |
|---|---|---|
| 0.02 | 0.010 | 22 |
| **0.05** | **0.152** | 17 |
| 0.08 | -0.489 | 7 |
| 0.10 | -0.724 | 6 |
| 0.15 | -1.198 | 2 |

Higher thresholds make the model trade so rarely that most windows fall
below the Sharpe minimum-sample rule, so their averages rest on few windows
and are not comparable. The best eligible threshold was 0.05, the value
already in use, so the logistic results are unchanged.

**Momentum (pooled mean Sharpe on the tuning period)**

| Window \ threshold | 0.00 | 0.02 | 0.04 |
|---|---|---|---|
| 10 days | -0.279 | -0.447 | -0.651 |
| 20 days | -0.064 | -0.034 | -0.455 |
| 40 days | **0.123** | -0.111 | -0.367 |

The tuning period favoured a 40-day window with threshold 0 over the
original 20 days and 0.02. The margin was small and partly driven by a single
SNAP window (SNAP listed in 2017 and has only one tuning window).

**Evaluation of the tuned setting (mean Sharpe, 2020-2024 windows)**

| Momentum | AAPL | MSFT | TSLA | SNAP |
|---|---|---|---|---|
| Original (20 days, 0.02) | 0.53 | -0.67 | 0.80 | -0.13 |
| Tuned (40 days, 0.0) | -0.46 | -0.44 | 0.14 | -0.24 |

**Findings.**

- The logistic threshold was validated: 0.05 was the best eligible value on
  earlier data, though the margin over 0.02 is small relative to the
  window-to-window variation.
- The tuned momentum setting did worse than the original on three of the four
  stocks in the evaluation period. A setting that looked best on 2014-2019
  data did not transfer to 2020-2024, consistent with momentum's performance
  depending heavily on the market regime.
- The original momentum result on TSLA (0.80), the best single result earlier
  in this document, should therefore be read as partly a product of an
  unvalidated setting, not as a robust finding. The choice between the two
  momentum settings is not made on evaluation-period performance; both are
  reported.

**Limitations.** The tuning period is short and has few windows per stock
(SNAP has one), so the tuning signal is weak. Only the logistic threshold and
momentum were tuned; the other strategies' settings were not, and gradient
boosting was left out because it is not viable after costs. The
best-per-ticker portfolio in Section 5 still uses the original momentum
setting for TSLA.

## 12. Reproducibility (Phase 13)

Results in earlier versions of this project changed slightly between
identical runs (Section 4), because prices were downloaded from Yahoo Finance
on every run. Now the first download of each ticker and date range is saved to
`data/cache/` and committed, and every later run reads the saved file. To test
this, the full benchmark was run twice on the snapshot: every mean Sharpe, for
every strategy and stock, matched to every printed digit, including both ML
models. The only differences were in the 15th to 16th decimal place of alpha
and beta, which is floating-point rounding.

**Limitations.** Results are reproducible only against the committed
snapshot. Refreshing the data (the loader's `refresh` option) will change the
numbers slightly, and all tables in this document would need regenerating.
Reading from the snapshot also means the pipeline does not notice if Yahoo
later corrects its data.