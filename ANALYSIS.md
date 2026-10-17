# Quant Research Pipeline: Analysis & Findings

This document summarizes the methodology, bugs found and fixed, and the
substantive findings produced by this project, a config-driven backtesting
pipeline supporting five rule-based strategies and two ML model types,
validated with walk-forward testing and extended to portfolio-level
backtesting.

## Results at a glance

Seven stocks, eleven out-of-sample windows (2020-01-03 to the end of 2025),
costs of 7 bps per one-way trade. Every number is out-of-sample.

**The headline: no strategy beat simply holding the stock, and none beat the
S&P 500.** What the strategies did buy was a smaller worst drop.

| Strategy | Mean Sharpe (avg of 7 stocks) | Worst-window drawdown vs buy-and-hold |
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

(* = the 95% bootstrap interval excludes zero, Section 10. Drawdown figures
are percentage points of capital, averaged over windows. Part of the
shallower drawdown is simply less market exposure, since strategies are often
in cash.)

| Portfolio (7 stocks, Rs 5.1 lakh) | Total return | CAGR | Worst-window drawdown | Mean Sharpe |
|---|---|---|---|---|
| SPY buy-and-hold | 109.0% | 14.3% | n/a | 0.93 |
| Single strategy (logistic on all 7) | 50.4% | 7.7% | -18.8% | 0.407 |
| Best strategy per stock, chosen without hindsight | 57.1% | 8.6% | -12.7% | 0.399 |

(Portfolio returns compound the 11 window returns; the SPY figures come from
the dashboard's buy-and-hold calculation over the same days. Both portfolios
sit in cash much of the time, which is why they trail the market's raw return.)

Three further findings, each tested rather than assumed:

- Choosing each stock's strategy from 2014-2019 data did not beat using one
  strategy everywhere (0.399 vs 0.407), so past winners did not reliably stay
  winners (Section 5).
- Tuning a setting on 2014-2019 data made both the logistic regression and
  the momentum strategy worse out of sample, not better (Section 11).
- Results are exactly reproducible from a committed data snapshot. An
  earlier, wrong explanation of why results used to vary is documented
  openly (Sections 4 and 12).

## 1. Methodology

Every result in this document comes from **walk-forward validation**: the
price history is split into eleven rolling windows (504 trading days of
training, 126 days of out-of-sample testing, non-overlapping), and every
strategy, rule-based or ML, is re-evaluated fresh on each window's test
period using only information available up to that point. No result here
comes from a single train/test split or from fitting once on the full
history. Data runs from 2018-01-01 to the end of 2025, and the first
out-of-sample day is 2020-01-03.

Seven tickers were used throughout: **AAPL, MSFT, TSLA, SNAP, JPM, XOM, JNJ**.
They were chosen to span a large, low-volatility mega-cap (AAPL), a large
stable tech name (MSFT), a high-volatility, narrative-driven stock (TSLA), a
smaller, more erratic name (SNAP), and three names from other sectors that
behave differently from tech: a bank (JPM), an energy company (XOM) and a
defensive healthcare company (JNJ). The first version of this project used
only the first four, all technology; adding the other three is what exposed
how much results depended on that choice (Section 4).

**Three roles for data.** *Training* data is the 504 days before each test
block and is used to fit a model. *Tuning* data (2014-2019) is used only to
choose settings and strategies, and it ends before the first evaluation test
day. *Evaluation* test days (2020 onward) are never used to choose anything.
Tuning and training data may overlap, but tuning data never includes an
evaluation test day. This is enforced in code.

**Data.** Prices come from Yahoo Finance through `yfinance`, adjusted for
splits and dividends, and are pinned: the first download of each ticker is
saved under `data/cache/` (2013-01-01 to the end of 2025) and committed to the
repository, and every run reads that snapshot instead of downloading again.
Earlier versions downloaded fresh data on every run, which made results drift
from run to run (Section 12). Every number in this document was generated
from the snapshot.

**Metrics.** Mean Sharpe is the average of the per-window Sharpe ratios.
Windows with fewer than 10 non-zero trading days give no Sharpe ratio (it
would be numerically meaningless), so they are skipped when averaging. The
"percent positive windows" figures count such windows as not positive, so the
denominator is always all eleven.

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
reversion, Bollinger reversion) were walk-forward tested across all seven
tickers. These use default settings chosen without validation; Section 11
re-tests the two most influential ones on earlier data. Mean Sharpe per
strategy per ticker, after transaction costs (7 bps: 5 bps transaction cost
plus 2 bps slippage), next to each stock's own buy-and-hold:

| Strategy | AAPL | MSFT | TSLA | SNAP | JPM | XOM | JNJ |
|---|---|---|---|---|---|---|---|
| Buy and hold | 0.78 | 0.84 | 0.83 | 0.38 | 0.87 | 0.55 | 0.03 |
| SMA crossover | 0.03 | 0.03 | 0.37 | 0.33 | 0.63 | 0.06 | -0.59 |
| EMA crossover | 0.13 | -0.07 | 0.31 | 0.01 | 0.47 | 0.02 | -0.89 |
| Momentum | 0.18 | -0.82 | 0.59 | -0.28 | -0.33 | -0.28 | -0.67 |
| RSI reversion | -0.42 | -0.09 | -0.78 | -0.62 | -0.68 | -1.26 | -0.48 |
| Bollinger reversion | -0.80 | -0.16 | -1.50 | 0.09 | -0.85 | -0.72 | -1.36 |

**Key finding: no single strategy dominates, and none beats holding.** Momentum
is TSLA's best strategy (0.59) but is actively harmful on MSFT. SMA is best on
JPM (0.63) and the only rule-based strategy with a positive Sharpe on six of
the seven stocks. The two mean-reversion strategies (RSI and Bollinger) are
negative on almost every stock, a sensible result: they lose money when the
underlying assumption (that prices revert) does not hold, and most of these
stocks trended over the test period. In no case does a rule-based strategy
beat the same stock's buy-and-hold Sharpe.

Performance also swung between strongly positive and strongly negative
windows depending on the market regime. For example the single-strategy
portfolio's window Sharpe ranged from -1.19 to +2.41, with a standard
deviation (1.29) more than three times its mean (0.41). This is the clearest
evidence in the project that a single backtest number is not sufficient:
consistency across time matters as much as average performance.

## 4. Machine learning strategies (Phase 3b)

A logistic regression model, retrained fresh at every walk-forward window
using 13 engineered features (price-relative-to-moving-average, RSI,
volatility, momentum at multiple horizons, Bollinger band position), was
compared against the five rule-based strategies.

| Ticker | Best rule-based (mean Sharpe, after costs) | Logistic regression (mean Sharpe, after costs) |
|---|---|---|
| AAPL | 0.18 (momentum) | **0.50** |
| MSFT | 0.03 (SMA) | **0.24** |
| TSLA | **0.59** (momentum) | 0.12 |
| SNAP | 0.33 (SMA) | **0.55** |
| JPM | **0.63** (SMA) | 0.58 |
| XOM | **0.06** (SMA) | -0.72 |
| JNJ | -0.48 (RSI) | **-0.11** |

Logistic regression had the higher Sharpe than the best rule-based strategy on
four of the seven stocks (AAPL, MSFT, SNAP, JNJ), and was close on JPM (0.58 vs
0.63). It was clearly worse on TSLA and XOM. None of these gaps is
statistically distinguishable from noise (Section 10).

**The four-stock result did not generalize.** On the original four tech stocks
the logistic regression averaged a mean Sharpe of 0.44 at 7 bps. Across all
seven it averages 0.16, because XOM (-0.72) and JNJ (-0.11) drag it down. A finding built only on
tech stocks would have overstated the model's value by almost a factor of
three, which is the reason the universe was widened.

### The accuracy-versus-profitability finding

A second metric, **direction accuracy** (how often the model correctly
called next-day direction on days it actually traded), was added
specifically to test whether strong Sharpe results implied strong
predictive accuracy. They did not: averaged across windows (skipping
windows with too few trades to compute it), direction accuracy was about
**53% on AAPL and 51% on SNAP, no better than a coin flip or barely better**,
despite these being two of the model's best stocks by Sharpe. This indicates
the model's edge came not from consistently correct direction calls, but
from a combination of selective trading (a confidence threshold filters out
low-conviction predictions) and favorable risk/reward asymmetry on the trades
it did make, a more sophisticated and more honest explanation than "the model
is good at predicting direction."

### Feature importance

Per-window logistic regression coefficients were inspected in the original
four-stock run and revealed two patterns consistent with documented market
phenomena (these were not regenerated for the seven-stock run):

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
identical conditions. Logistic regression had the higher mean Sharpe on six of
the seven stocks (the exception is XOM, -0.72 vs -0.61). The pooled difference
is +0.48 with a 95% interval of [-0.08, +1.08], so it just includes zero
(Section 10). After costs the gap widens further, because gradient boosting
trades about five times as often (turnover 0.394 vs 0.078 per window,
Section 8). The case against gradient boosting rests on its costs, not on its
raw Sharpe.

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
transaction costs. (The figures in this paragraph are from the early
four-stock runs.)

## 5. Portfolio-level backtesting (Phase 4)

Individual-ticker results were combined into a single portfolio, with real,
independent rupee allocations per ticker (summed at the equity level, since
price and position have no single meaningful combined value across different
assets, only money does). Allocations: AAPL 100,000; MSFT 80,000; TSLA 50,000;
SNAP 70,000; JPM 80,000; XOM 60,000; JNJ 70,000.

| Portfolio | Mean Sharpe | Std Sharpe | % Positive Windows | Total return | CAGR |
|---|---|---|---|---|---|
| Single strategy (logistic, all 7 stocks) | 0.407 | 1.289 | 64% (7 of 11) | 50.4% | 7.7% |
| Best strategy per stock, chosen without hindsight | 0.399 | 1.184 | 64% (7 of 11) | 57.1% | 8.6% |

Both portfolios trade far below the market's raw return (SPY: 109.0%, 14.3% a
year over the same days), because the strategies hold cash while flat.

Three metrics (`win_rate`, `turnover`, `direction_accuracy`) could not be
computed at the combined portfolio level, since they depend on knowing a
single position or trade, which has no meaningful definition for a
portfolio that may be simultaneously long one asset and short another.
These remain available, correctly, at the per-ticker level.

### 5.1 Choosing each stock's strategy without hindsight (Phase 11)

An earlier version of the best-per-ticker portfolio picked each stock's
strategy by looking at which one had done best in the evaluation results. That
is hindsight selection, and its result can only be read as an upper bound. It
was replaced with a selection rule that cannot see the evaluation period:

1. Run every strategy (default settings) on the 2014-2019 tuning period.
2. Give each stock the strategy with the highest mean Sharpe there, counting
   only strategies with at least 5 valid tuning windows on that stock.
3. If no strategy is eligible (SNAP has only one tuning window, because it
   listed in 2017), fall back to the logistic regression. This rule was fixed
   before any results were seen.

**Mean Sharpe on the tuning period (2014-2019); picks are in bold**

| Strategy | AAPL | MSFT | TSLA | SNAP | JPM | XOM | JNJ |
|---|---|---|---|---|---|---|---|
| SMA crossover | 0.24 | 0.98 | -0.75 | 1.19 | **0.48** | **-0.49** | -0.12 |
| EMA crossover | 0.07 | 1.03 | -0.55 | 1.84 | 0.20 | -0.61 | **-0.00** |
| Momentum | **1.02** | -0.64 | -0.54 | 0.39 | -0.75 | -0.85 | -0.97 |
| RSI reversion | -0.92 | -1.07 | -1.04 | 2.69 | -1.20 | -1.12 | -2.18 |
| Bollinger reversion | -1.36 | -0.59 | -1.11 | -0.02 | -1.05 | -1.42 | -2.66 |
| Logistic regression | -1.36 | **1.34** | **0.58** | **-1.76** | -1.15 | -1.53 | -1.45 |
| Gradient boosting | -1.39 | 0.66 | -0.28 | -1.89 | -0.31 | -0.97 | -1.09 |

(SNAP's numbers each rest on a single window, so none was eligible and its
pick is the fallback. Logistic regression had only 3 valid windows on JNJ and
3 on XOM, so it was ineligible there. Each other stock takes the highest mean
among eligible strategies.)

Resulting picks: AAPL momentum, MSFT logistic, TSLA logistic, SNAP logistic
(fallback), JPM SMA, XOM SMA, JNJ EMA.

**Result.** The portfolio built from these picks has a mean Sharpe of 0.399,
against 0.407 for simply using logistic everywhere. Selection did not help.
Per stock, the picked strategy's evaluation Sharpe was AAPL 0.18, MSFT 0.24,
TSLA 0.12, SNAP 0.55, JPM 0.63, XOM 0.06 and JNJ -0.89. The worst result is
JNJ, where EMA was chosen from a tuning-period Sharpe of about zero and went on
to lose money in most windows (positive in only 2 of 11).

The clearest lesson is what the hindsight version hid: picking the best
strategy after the fact would have given TSLA momentum (0.59 rather than the
selected 0.12) and AAPL logistic (0.50 rather than 0.18). Those picks looked
obvious only in retrospect. The tuning-period Sharpes are large and noisy,
and the differences between strategies on a stock are mostly within that
noise, so the selection step mostly chose between near-equals.

The selection script is `experiments/run_selection.py`; its scores are saved
in `results/selection.csv`, and the portfolio config it writes is generated, not
hand-edited.

## 6. Interactive dashboard (Phase 5)

A Streamlit dashboard provides three interactive views over this same
pipeline: single-ticker walk-forward results (any strategy, any ticker),
a side-by-side strategy comparison, and a portfolio builder supporting
both the single-strategy and best-per-ticker modes described above, with
the hindsight-selection caveat surfaced directly in the UI when relevant.

The strategy comparison page leads with what simply holding the stock and
holding SPY would have returned over the same test days, then ranks the
strategies on total return, CAGR or Sharpe (a switch above the chart), with
the benchmarks drawn as separate bars. It has start and end date pickers
within the pinned data range, and the walk-forward window sizes are under an
advanced section. The other two pages still use fixed dates (2018-2025).

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
realistic costs, while gradient boosting, which trades about five times as
often as logistic regression, does not. The mean-reversion strategies
(RSI, Bollinger) fail even at zero cost, so no cost assumption rescues
them. Finding that the original cost model undercharged by a factor of
hundreds is itself a reminder to test the plumbing, not just the
strategies.

The later phases tested these findings harder, and the honest picture is more
modest. No strategy beat the S&P 500 (SPY) on Sharpe on any of the seven
stocks, and none beat holding the same stock (Sections 9 and 10). Widening the
universe from four tech stocks to seven stocks across four sectors cut the
logistic regression's average Sharpe from 0.44 to 0.16: the original result
partly reflected the choice of stocks. The most robust positive result is
that logistic regression reduces the worst drawdown relative to buy-and-hold
on all seven stocks (Section 10). Neither tuning a setting on earlier data
(Section 11) nor choosing each stock's strategy on earlier data (Section 5.1)
improved out-of-sample results. And all results are tied to a pinned data
snapshot so they can be reproduced exactly (Section 12).

## 8. Transaction-cost sensitivity (Phase 12)

Because earlier versions undercharged costs (Section 2), this section
measures how much costs actually matter. For each strategy and ticker, the
out-of-sample signals were generated once, then the same signals were
re-backtested at total one-way costs of 0, 5, 7, 10 and 20 basis points (7
is the project default: 5 bps transaction cost plus 2 bps slippage). A
round trip costs twice the one-way figure.

**Mean Sharpe vs total one-way cost (average over the 7 stocks)**

| Strategy | 0 bps | 5 bps | 7 bps | 10 bps | 20 bps |
|---|---|---|---|---|---|
| Logistic regression | 0.259 | 0.191 | 0.164 | 0.123 | -0.013 |
| SMA crossover | 0.145 | 0.129 | 0.123 | 0.114 | 0.083 |
| Logistic regression (tuned, Section 11) | 0.200 | 0.098 | 0.074 | 0.037 | -0.084 |
| EMA crossover | 0.014 | 0.000 | -0.005 | -0.013 | -0.039 |
| Gradient boosting | 0.043 | -0.171 | -0.257 | -0.385 | -0.809 |
| Momentum | -0.105 | -0.193 | -0.229 | -0.282 | -0.459 |
| Momentum (tuned, Section 11) | -0.192 | -0.253 | -0.277 | -0.313 | -0.433 |
| RSI reversion | -0.641 | -0.578 | -0.619 | -0.679 | -0.880 |
| Bollinger reversion | -0.553 | -0.698 | -0.759 | -0.849 | -1.138 |

**Breakeven cost (bps at which mean Sharpe reaches zero) and turnover**

| Strategy | All (avg) | AAPL | MSFT | TSLA | SNAP | JPM | XOM | JNJ | Mean turnover |
|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 19.1 | >20 | >20 | >20 | >20 | >20 | never | 3.5 | 0.078 |
| SMA crossover | >20 | 17.0 | 14.8 | >20 | >20 | >20 | >20 | never | 0.023 |
| Logistic (tuned) | 13.1 | >20 | >20 | >20 | >20 | never | never | 17.9 | 0.114 |
| EMA crossover | 5.2 | >20 | never | >20 | 12.1 | >20 | 14.9 | never | 0.019 |
| Gradient boosting | 1.0 | 7.7 | 5.9 | 9.5 | 0.5 | never | never | 1.2 | 0.394 |
| Momentum | never | 18.5 | never | >20 | never | never | never | never | 0.186 |
| Momentum (tuned) | never | never | never | >20 | never | >20 | 1.2 | never | 0.139 |
| RSI reversion | never | never | never | never | never | never | never | 2.3 | 0.068 |
| Bollinger reversion | never | never | 3.3 | never | 12.9 | never | never | never | 0.115 |

("never" means Sharpe is already at or below zero with no costs; ">20"
means still positive at the highest cost tested.)

**Findings.**

- Cost damage scales with turnover. Gradient boosting trades the most
  (turnover 0.394) and is already negative at the 5 bps level on average; the
  crossover strategies trade rarely and barely notice costs.
- Logistic regression remains the best strategy up to 10 bps (0.123 vs 0.114
  for SMA), but SMA overtakes it at 20 bps (0.083 vs -0.013).
- Gradient boosting is not viable at realistic costs: its breakeven is
  about 1 bp overall. This widens the logistic-vs-gradient-boosting gap found
  in Section 4.
- The two mean-reversion strategies (RSI, Bollinger) have negative
  Sharpe even at zero cost, so costs are not their problem; the signals
  do not work on these stocks over this period.
- The tuned settings from Section 11 are worse than the originals at every
  cost level: logistic 0.074 vs 0.164 at 7 bps, momentum -0.277 vs -0.229.
  The tuned logistic threshold (0.02) also trades more (turnover 0.114 vs
  0.078), which is why its breakeven cost falls from 19.1 to 13.1 bps.

Section 10 tests these gaps for statistical significance.

**Limitations.** Costs are modelled as a flat per-trade charge, with no
market impact or size dependence. With seven stocks, the gaps between the
top strategies at low costs are small relative to window-to-window
variation (Section 10 tests this). At 0 bps the RSI row is lower than at
5 bps, which should be impossible since costs can only reduce returns; this
appears to be an artifact of the Sharpe minimum-sample rule at exactly zero
cost, not a real effect.

## 9. Market benchmark (Phase 8)

A strategy has to beat the market, not just earn a positive Sharpe. The
benchmark is buy-and-hold of SPY (an S&P 500 ETF) over the same eleven test
windows, which has a mean Sharpe of **0.933**. Each stock's own buy-and-hold is
shown too. Alpha and beta come from regressing the strategy's daily excess
returns (over a 5% annual risk-free rate) on SPY's daily excess returns. Alpha
is annualized, and beta is measured on the strategy's return stream, not on
the stock's.

**Mean Sharpe / annualized alpha (%) / beta vs SPY**

| Strategy | AAPL | MSFT | TSLA | SNAP |
|---|---|---|---|---|
| Buy and hold | 0.78 / 6.9 / 1.21 | 0.84 / 8.9 / 1.15 | 0.83 / 42.6 / 1.69 | 0.38 / -1.2 / 1.55 |
| SMA crossover | 0.03 / 1.3 / 0.42 | 0.03 / -1.9 / 0.54 | 0.37 / 24.2 / 0.90 | 0.33 / 11.6 / 0.35 |
| EMA crossover | 0.13 / 4.6 / 0.48 | -0.07 / 0.3 / 0.50 | 0.31 / 23.1 / 0.90 | 0.01 / 5.5 / 0.31 |
| Momentum | 0.18 / 13.6 / -0.51 | -0.82 / -14.1 / -0.40 | 0.59 / 54.2 / -0.40 | -0.28 / -15.0 / -0.66 |
| RSI reversion | -0.42 / -3.4 / 0.04 | -0.09 / 0.0 / 0.00 | -0.78 / -38.7 / 0.03 | -0.62 / -4.7 / 0.37 |
| Bollinger reversion | -0.80 / -6.9 / 0.16 | -0.16 / 3.2 / 0.11 | -1.50 / -39.5 / -0.02 | 0.09 / 8.4 / 0.15 |
| Logistic regression | 0.50 / 1.8 / 0.80 | 0.24 / -4.6 / 0.56 | 0.12 / 2.6 / 0.22 | 0.55 / 22.6 / 0.01 |
| Gradient boosting | 0.03 / 4.1 / 0.17 | -0.05 / -7.2 / 0.60 | 0.06 / 6.2 / 0.55 | -0.15 / -11.2 / 0.24 |

| Strategy | JPM | XOM | JNJ |
|---|---|---|---|
| Buy and hold | 0.87 / 4.7 / 1.08 | 0.55 / 5.6 / 0.81 | 0.03 / -3.3 / 0.43 |
| SMA crossover | 0.63 / 8.0 / 0.30 | 0.06 / 1.0 / 0.32 | -0.59 / -7.1 / 0.12 |
| EMA crossover | 0.47 / 2.4 / 0.30 | 0.02 / 2.7 / 0.29 | -0.89 / -11.7 / 0.13 |
| Momentum | -0.33 / -6.7 / -0.47 | -0.28 / 2.0 / -0.24 | -0.67 / -6.7 / -0.26 |
| RSI reversion | -0.68 / -7.5 / 0.18 | -1.26 / -14.0 / 0.30 | -0.48 / -2.2 / 0.01 |
| Bollinger reversion | -0.85 / -5.6 / 0.15 | -0.72 / -5.2 / 0.07 | -1.36 / -7.1 / 0.11 |
| Logistic regression | 0.58 / -0.8 / 0.63 | -0.72 / -12.4 / 0.36 | -0.11 / -6.1 / 0.23 |
| Gradient boosting | -0.62 / -15.7 / 0.47 | -0.61 / -14.4 / 0.36 | -0.45 / -10.1 / 0.17 |

**Findings.**

- No strategy beat SPY's Sharpe of 0.933 on any of the seven stocks. Even
  holding each stock fell short, by 0.06 on JPM up to 0.90 on JNJ. The closest
  strategies were SMA on JPM (0.30 below SPY), momentum on TSLA (0.34 below)
  and logistic regression on JPM (0.36 below).
- Buy-and-hold alpha is positive on five of the seven stocks (4.7% to 42.6% a
  year; TSLA by far the largest) and slightly negative on SNAP and JNJ. Strategy
  alphas are mixed, and the large ones (momentum on TSLA at +54.2%) come with
  very volatile window-to-window results.
- The strategies carry much less market exposure than the stocks they trade
  (betas mostly 0 to 0.9, apart from momentum's negative betas, against 0.4 to
  1.7 for buy-and-hold), because
  they are often flat. This is also why they have shallower drawdowns
  (Section 10).
- Momentum has a negative beta on all seven stocks (-0.24 to -0.66). This was
  not investigated; one hypothesis is its timing around large market moves.
- Logistic regression's beta varies widely by stock (0.80 on AAPL, 0.01 on
  SNAP) even though the model is the same, because beta measures the
  strategy's returns, which depend on when it was invested.

**Limitations.** Alpha and beta come from about five and a half years of daily
returns, and none of the alpha estimates were tested for statistical
significance. The risk-free rate is a flat 5%.

## 10. Statistical significance (Phase 9)

Every comparison so far rests on the mean of eleven window-level Sharpe
ratios, which vary a lot from window to window. This section asks which gaps
are distinguishable from noise, using a paired bootstrap over the eleven
windows: for each pair of strategies, the eleven per-window differences are
resampled with replacement 10,000 times, and the middle 95% of the resampled
means forms a confidence interval. A gap is called significant only if that
interval excludes zero. "Pooled" averages the seven stocks within each window
first, so each strategy contributes eleven values.

**Strategy minus buy-and-hold (pooled across the 7 stocks)**

| Strategy | Sharpe difference [95% interval] | Max drawdown difference [95% interval] | Total return difference per window [95% interval] |
|---|---|---|---|
| Logistic regression | -0.38 [-0.80, +0.02] | +0.12 [+0.09, +0.15] * | -0.10 [-0.21, +0.01] |
| SMA crossover | -0.54 [-0.77, -0.33] * | +0.08 [+0.04, +0.12] * | -0.07 [-0.13, -0.00] * |
| EMA crossover | -0.67 [-0.88, -0.45] * | +0.07 [+0.03, +0.11] * | -0.07 [-0.13, -0.00] * |
| Momentum | -0.84 [-1.17, -0.49] * | -0.00 [-0.03, +0.02] | -0.10 [-0.20, -0.00] * |
| Gradient boosting | -0.87 [-1.29, -0.44] * | +0.05 [+0.02, +0.07] * | -0.14 [-0.25, -0.04] * |
| RSI reversion | -1.14 [-1.73, -0.42] * | +0.14 [+0.11, +0.18] * | -0.15 [-0.30, -0.00] * |
| Bollinger reversion | -1.37 [-1.92, -0.76] * | +0.16 [+0.12, +0.21] * | -0.14 [-0.27, -0.02] * |

(* = interval excludes zero. For drawdown a positive difference means a
shallower worst drop than holding the stock. Total return is per 126-day
window, so -0.10 means 10 percentage points less than buy-and-hold per window.)

**Findings.**

- On risk-adjusted return, six of the seven strategies are significantly
  worse than buying and holding. Only logistic regression is inconclusive, and
  only barely (its interval runs from -0.80 to +0.02). It is not significantly
  worse, but it is not better either.
- On max drawdown the strategies do better: logistic regression has a
  significantly shallower worst drop than buy-and-hold on all seven stocks.
  Part of this is lower market exposure, since the strategies are often flat.
  Momentum is the exception, with no difference.
- Logistic regression versus gradient boosting is not significant on any
  stock (pooled +0.48, interval [-0.08, +1.08]); the case against gradient
  boosting rests on its transaction costs (Section 8), not on its Sharpe.
- Logistic regression versus the best rule-based strategy on each stock is
  inconclusive everywhere.
- Few strategies' mean Sharpe are distinguishable from zero. Pooled across the
  seven stocks, buy-and-hold is positive (+0.61, [+0.22, +1.00]), while
  logistic regression is not (+0.23, [-0.09, +0.60]). Bollinger (-0.76) and RSI
  (-0.53) are significantly negative. Logistic regression on AAPL, which was
  the one significant positive result with four stocks, is no longer
  significant (+0.50, [-0.03, +1.05]); on XOM it is significantly negative
  (-0.72, [-1.51, -0.06]).

**Limitations.** Eleven windows give low power, so wide intervals are expected
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
untouched 2020-2025 windows used everywhere else. The script refuses to run
if the tuning data reaches into the evaluation period. The tuning data and the
training data of the first evaluation windows share 2018-2019, but no
evaluation test day was used in tuning. The grids were kept deliberately
small, one setting for all seven stocks, to limit overfitting to the tuning
data. A setting was only eligible to win if at least 29 of the 43 possible
(stock, window) Sharpe ratios were valid (the earlier four-stock rule of 15 of
22, scaled to the larger universe).

Tuning only reports; it never changes a config automatically. The tuned
settings are kept as separate configs and both versions are reported, and the
choice between them is not made on evaluation-period performance.

**Logistic regression threshold (mean Sharpe on the tuning period)**

| Threshold | Pooled mean Sharpe | Valid windows |
|---|---|---|
| **0.02** | **-0.413** | 42 |
| 0.05 | -0.440 | 28 |
| 0.08 | -0.735 | 12 |
| 0.10 | -0.944 | 8 |
| 0.15 | -1.100 | 2 |

Higher thresholds make the model trade so rarely that most windows fall
below the Sharpe minimum-sample rule, so their averages rest on few windows
and are not comparable. The best eligible threshold was 0.02. The original
0.05 had 28 valid windows, one short of the eligibility minimum, and was only
0.027 behind on mean Sharpe, a margin far smaller than the window-to-window
variation. Note also that every pooled logistic Sharpe on the tuning period is
negative: in 2014-2019 the model did not make money on average at any setting.

**Momentum (pooled mean Sharpe on the tuning period)**

| Window \ threshold | 0.00 | 0.02 | 0.04 |
|---|---|---|---|
| 10 days | -0.525 | -0.609 | -0.845 |
| 20 days | -0.447 | -0.435 | -0.598 |
| 40 days | **-0.094** | -0.178 | -0.519 |

The tuning period favoured a 40-day window with threshold 0 over the original
20 days and 0.02. It is the best of an all-negative set, so it was the
least-bad rather than a profitable setting, and the result is partly driven by
SNAP (+1.70 at that setting; SNAP has a single tuning window).

**Evaluation of the tuned settings (mean Sharpe over the 7 stocks at 7 bps,
2020-2025 windows)**

| Strategy | Original | Tuned |
|---|---|---|
| Logistic regression (threshold 0.05 vs 0.02) | 0.164 | 0.074 |
| Momentum (20 days / 0.02 vs 40 days / 0.0) | -0.229 | -0.277 |

**Findings.**

- Neither tuned setting beat the original out of sample. The setting that
  looked best on 2014-2019 data did not transfer to 2020-2025, consistent with
  strategy performance depending heavily on the market regime.
- The lower logistic threshold (0.02) makes the model trade more, which costs
  more: its breakeven cost falls from 19.1 to 13.1 bps (Section 8).
- The original momentum result on TSLA (0.59), the best single rule-based
  result in this document, should be read as partly a product of an
  unvalidated setting, not as a robust finding. Both momentum settings are
  reported.

**Limitations.** The tuning period is short and has few windows per stock
(SNAP has one), and all the strategies scored negatively on average in it, so
the tuning signal is weak. Only the logistic threshold and momentum were
tuned; the other strategies' settings were not, and gradient boosting was left
out because it is not viable after costs. A per-stock table of tuned versus
original Sharpe was not produced; only the averages above are reported. The
best-per-ticker portfolio uses the original settings, because tuned variants
are deliberately not candidates in the selection step: their settings were
chosen on the same 2014-2019 data the selection scores on, so a comparison
there would favour them by construction.

## 12. Reproducibility (Phase 13)

Results in earlier versions of this project changed slightly between
identical runs (Section 4), because prices were downloaded from Yahoo Finance
on every run. Now the first download of each ticker is saved to
`data/cache/` and committed, and every later run reads the saved file. To test
this, the full benchmark was run twice on the snapshot: every mean Sharpe, for
every strategy and stock, matched to every printed digit, including both ML
models. The only differences were in the 15th to 16th decimal place of alpha
and beta, which is floating-point rounding. The test suite runs automatically
on every push through GitHub Actions.

**Limitations.** Results are reproducible only against the committed
snapshot. Refreshing the data (the loader's `refresh` option) will change the
numbers slightly, and all tables in this document would need regenerating.
Reading from the snapshot also means the pipeline does not notice if Yahoo
later corrects its data.
