---
myst:
  html_meta:
    description: >-
      Rolling portfolio backtests in OptimalPortfolios: causal estimation, decision and
      execution dates, drifted holdings, transaction costs, and executable offline examples.
---

# Rolling portfolio backtests

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/OptimalPortfolios/commit/fb8848d327c0585eaf0933dba6137ec6b8338bbf)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Holdings simulation and analytics use [qis](https://github.com/ArturSepp/QuantInvestStrats);
see the [qis citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A rolling portfolio backtest repeatedly estimates inputs, constructs target allocations and
simulates their implementation through time. A target weight is a decision; a realised holding
is the result of executing that decision and carrying units as prices change.

## Overview

The workflow has three layers: a single-date solver, a rolling construction wrapper, and
`qis.backtest_model_portfolio`. OptimalPortfolios owns the first two; qis converts dated targets
into units, applies execution timing and costs, and returns portfolio data for analytics.
Validate one matrix and one solution before extending that contract through time.
See the [rolling dispatcher source](../src/optimalportfolios/optimization/wrapper_rolling_portfolios.py)
and the [qis backtester](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py).

Use a rolling workflow to study a specified allocation and implementation policy. The result
depends on information availability, eligible assets, trading assumptions and the price panel.
A successful optimisation alone does not establish an implementable historical strategy.

## Inputs, notation, and assumptions

### Inputs and conventions

| Input or symbol | Meaning and units |
|---|---|
| `prices`, $P_{i,t}$ | Floating-point total-return price levels, one column per asset, on an ordered `DatetimeIndex`. |
| `covar_dict`, $\Sigma_{d_k}$ | Asset-labelled covariance matrices keyed by decision date $d_k$. |
| $w_{i,k}$ | Target weight: fraction of portfolio NAV assigned to asset $i$ at decision $k$. |
| $\tau_k$, $L$ | Implementation date and nonnegative lag in observations of the price index. |
| $u_{i,t}$, $V_t$ | Units held in asset $i$ and total portfolio net asset value. |
| $r_{i;a,b}$ | Simple holding-period return $P_{i,b}/P_{i,a}-1$ used to drift holdings. |

Supply covariance dates in chronological insertion order: construction loops over the dictionary
as supplied. Asset labels must be consistent across prices, matrices and constraint vectors.
Define any cash or financing convention explicitly when weights need not sum to one.

The covariance estimator defines its return convention, frequency and annualisation.
`EwmaCovarEstimator` samples log returns at `returns_freq` and returns annualised covariances.
Its `span` is an EWMA span measured in those observations, not a half-life or a finite window.
The estimator's `rebalancing_freq` selects decision dates on the sampled-return grid.
It does not change return frequency. See [covariance estimators](covariance_estimators.md).

Keep estimation history before the requested decision period. The EWMA rolling method filters
its output dates, but does not impose a user-selected minimum history simply because a span
was supplied. The caller must provide and check the required warmup.

## Methodology

### Information and implementation clocks

Every covariance, expected return, risk budget and eligibility decision dated $d_k$ must use
information available by $d_k$. Avoid full-sample demeaning and future-filled inputs.
An implementation lag cannot repair an estimator that already uses future information.

For a dated weight DataFrame, qis finds the first price observation at or after the decision
date, then moves forward by $L$ observations. Thus `weight_implementation_lag=1` means the next
business observation on a business-day panel, but the next month on a monthly panel.
Prices and instrument returns themselves are unchanged. `None` means zero lag.
A decision between two price observations maps forward before the lag is added.

The previous units earn the return into the execution observation. The new units are sized
at that observation's prices and earn subsequent returns. A fully invested, cost-free trade is

$$
u_{i,\tau_k}=\frac{V_{\tau_k}^{-}w_{i,k}}{P_{i,\tau_k}},
$$

where $V_{\tau_k}^{-}$ is NAV immediately before trading. With no trades, financing, fees or
additional cash flows, the units remain fixed and the unchanged cash balance $B$ gives

$$
V_t=B+\sum_i u_{i,\tau_k}P_{i,t}.
$$

For proportional costs, qis charges the absolute units traded times their execution prices
and the fractional cost rate, including entry. It sizes targets using pre-cost NAV and then
deducts costs from cash. Consequently post-cost realised weights need not exactly equal the
target fractions. See [turnover and transaction costs](turnover_and_transaction_costs.md)
and the [qis implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py).

### Point-in-time and drift rules

Before a new solve, `apply_drift_to_weights_0` converts the prior target into an estimated
current baseline. For prior weights $w_i$, simple returns $r_{i;a,b}$ between its price anchors,
and a residual cash balance with no accrual, the identity is

$$
\widetilde{w}_i=
\frac{w_i(1+r_{i;a,b})}{1+\sum_j w_j r_{j;a,b}}.
$$

The denominator is portfolio NAV growth. Replacing it with the sum of risky positions would
incorrectly discard cash or change the convention for long-short and variable-exposure
portfolios. The [drift helper](../src/optimalportfolios/utils/weights_drift.py) uses the NAV-growth
form and returns the original weights when the drift is undefined.

The rolling solvers normally anchor this calculation at the **previous decision date and the
current decision date**. They do not replay qis' realised holdings or execution cash ledger.
With delayed implementation, costs, fees or differing trade eligibility, this construction
baseline can differ from the actual holdings. A turnover constraint applied to the baseline
therefore does not certify a bound on subsequently realised trading. Inspect both.

`OptimiserConfig.use_drifted_weights_0=True` enables this baseline by default.
Setting it to `False` reuses the previous target. On the first solve there is no previous
rolling target; any current holdings supplied through constraints need their own valid anchor.
Do not interpret the toggle as changing qis' unit-holding backtest.

## Worked example

### Decision-date drift versus executed holdings

This synthetic, cost-free example has two assets and four business observations. Asset A rises
10% each observation and B is flat. The target is 50/50 on 2 and 4 January, with lag one.
It intentionally supplies targets directly to qis so the timing can be checked independently
of an optimiser.

```python
import pandas as pd
import qis
import optimalportfolios as opt

toy_dates = pd.bdate_range("2024-01-02", periods=4)
toy_prices = pd.DataFrame(
    {"A": [100.0, 110.0, 121.0, 133.1], "B": [100.0, 100.0, 100.0, 100.0]},
    index=toy_dates,
)
toy_targets = pd.DataFrame(
    [[0.5, 0.5], [0.5, 0.5]], index=toy_dates[[0, 2]], columns=toy_prices.columns
)
toy_portfolio = qis.backtest_model_portfolio(
    prices=toy_prices, weights=toy_targets, initial_nav=100.0,
    weight_implementation_lag=1, rebalancing_costs=0.0,
)
decision_baseline = opt.apply_drift_to_weights_0(
    weights_0=toy_targets.iloc[0], prices=toy_prices,
    prev_date=toy_dates[0], date=toy_dates[2],
)
```

| Observation | Event | NAV |
|---|---|---|
| 2024-01-02 | First target decided; still in cash | 100.0 |
| 2024-01-03 | First target traded at A = 110 and B = 100 | 100.0 |
| 2024-01-04 | Previous units held; second target decided | 105.0 |
| 2024-01-05 | Previous units earn the intervening return; second target traded | 110.5 |

The first trade buys $50/110$ units of A and $50/100$ units of B. On 4 January these
positions are worth 55 and 50, so the realised A weight is $55/105$.
The decision-date baseline instead assumes the first target was established when A was 100;
its A weight is $60.5/110.5$.

| Weight on 2024-01-04 | A | B |
|---|---|---|
| Decision-date drift baseline | 0.547511 | 0.452489 |
| Realised holdings with lag one | 0.523810 | 0.476190 |

The first 10% rise in A is absent from the portfolio return because entry occurs after it.
A constant 50/50 weighted-return calculation would also miss the subsequent drift of holdings.

## Implementation in optimalportfolios

### Minimal offline example

The following preserves the article's fixed monthly example: 24 synthetic total-return
observations, the same repeating return pattern, span six, and four quarterly decisions.
Input prices compound **simple monthly returns**; the estimator subsequently calculates
monthly **log returns** and annualises covariance by 12. The decision period starts after
almost one year of supplied history. No data download is needed.

```python
import numpy as np
import pandas as pd
import qis
import optimalportfolios as opt

dates = pd.date_range("2020-01-31", periods=24, freq="ME")
monthly_returns = np.array([
    [0.010, 0.004, 0.002], [0.015, -0.003, 0.003],
    [-0.008, 0.006, 0.002], [0.012, 0.001, -0.001],
] * 6)
prices = pd.DataFrame(
    100.0 * np.cumprod(1.0 + monthly_returns, axis=0),
    index=dates,
    columns=["Equity", "Bonds", "Diversifier"],
)
estimator = opt.EwmaCovarEstimator(
    returns_freq="ME", span=6, rebalancing_freq="QE"
)
covar_dict = estimator.fit_rolling_covars(
    prices=prices,
    time_period=qis.TimePeriod("31Dec2020", "30Sep2021"),
)
constraints = opt.Constraints(
    is_long_only=True,
    max_weights=pd.Series(0.80, index=prices.columns),
)
weights = opt.compute_rolling_optimal_weights(
    prices=prices,
    constraints=constraints,
    covar_dict=covar_dict,
    portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
)
portfolio = opt.backtest_rolling_optimal_portfolio(
    prices=prices,
    constraints=constraints,
    covar_dict=covar_dict,
    portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
    rebalancing_costs=0.0003,  # 3 bp of traded notional
    weight_implementation_lag=1,
    ticker="Minimum variance",
)
```

`weights` has one row per covariance/decision date and one column per asset.
`portfolio` is a `qis.PortfolioData` with NAV, realised weights, units, turnover and costs.
Computing weights separately and then calling the convenience backtester solves twice.
When targets are already available, call `qis.backtest_model_portfolio` directly to use them.

| Decision date | Implementation date with monthly lag one |
|---|---|
| 2020-12-31 | 2021-01-31 |
| 2021-03-31 | 2021-04-30 |
| 2021-06-30 | 2021-07-31 |
| 2021-09-30 | 2021-10-31 |

The backtest begins at the first retained target date with NAV 100 in cash. Entry occurs at
the January price observation and costs 0.03, leaving NAV **99.97**. The target does not earn
the December-to-January asset return. The price panel continues through December 2021,
so the September decision can be implemented in October and carried afterwards.

### Entry points and scope

| Public entry point | Input and result contract |
|---|---|
| `EwmaCovarEstimator.fit_rolling_covars` | Prices plus a decision period produce dated annualised matrices. |
| `compute_rolling_optimal_weights` | Objective, constraints and inputs produce targets; no realised holdings or NAV. |
| `backtest_rolling_optimal_portfolio` | Computes targets, filters them if requested, then delegates holdings simulation to qis. |
| `apply_drift_to_weights_0` | A prior target and two price anchors produce a construction baseline. |

For minimum variance, the dates come from `covar_dict`. The dispatcher's `returns_freq` and
`span` concern mean estimation for mean-dependent objectives; they do not recompute an already
supplied covariance. Maximum-CARA-mixture constructs its own estimates and schedule instead
of consuming these covariance matrices. Objective-specific requirements remain in the
[optimisation guide](optimization_module_readme.md).

`perf_time_period` in the convenience backtester filters **target rows before simulation**.
It is not a crop of a portfolio already carrying holdings from earlier dates: the backtest
restarts at the first retained target. To view a later period while preserving prior holdings,
simulate the full intended path and select the reporting period afterwards.

Rolling construction returns weights without a per-date `OptimizationOutcome` table.
For a review requiring structured solve diagnostics, inspect the lower-level wrapper outcomes
or use an audited producer such as the [analytics pipeline](documentation_standard.md#analytical-conventions-and-figures).
Weight sums alone cannot establish solver acceptance or absence of fallback.

### Reproduction and verification context

The two Python blocks are the canonical offline examples for this article.
[Their executable tests](../src/optimalportfolios/tests/rolling_backtests_documentation_test.py)
check the displayed dates and values, actual solve outcomes, independent holdings calculations,
and future-input perturbations:

```console
python -m pytest src/optimalportfolios/tests/rolling_backtests_documentation_test.py -q
```

On Windows, use the external interpreter and generated-state setup described in
[AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md).
Local verification on 2026-09-13 used an export of the OptimalPortfolios 7.6.0 working tree,
installed qis 5.26.0, CVXPY 1.9.2 and CLARABEL 0.11.1. This is a working-source verification,
not certification of a published artifact or the current locked CI environment.

## Interpretation and limitations

### Failure modes and missing data

- **Unusable construction inputs.** Some wrappers exclude assets with missing covariance
  diagonals, clamp small variances or return fallback holdings on rejection. Filtering and
  rescaling depend on the objective and configuration. Review eligibility, the effective
  constraints and outcomes; an all-zero or carried portfolio can still form a weight table.
- **Drift fallbacks.** The helper uses only history at or before each anchor and forward-fills
  within that history. An asset with no usable anchor is treated as having zero return;
  other positions still affect portfolio NAV growth. Missing prefixes, absent previous targets,
  near-zero targets or nonpositive NAV growth can leave the entire prior target unchanged.
- **Unavailable execution prices.** qis cannot trade a target at a missing price. Interior
  holes in a held asset's price history can also remove its value from that observation's NAV.
  The drift helper's defensive policy does not validate a panel for performance measurement.
  See [incomplete histories](incomplete_histories.md).
- **Execution-grid boundaries.** Targets whose implementation would be beyond the price panel
  are dropped with a warning; no executable target raises an error. Two decision dates mapping
  to the same execution observation also raise an error. Check the first and last actual trades.
- **Interpretation of synthetic results.** These short examples establish arithmetic and timing,
  not statistical reliability or expected investment performance. Specify rates, fees, return
  conventions and annualisation before extending them into performance comparisons.

## See also

- [Turnover and transaction costs](turnover_and_transaction_costs.md)
- [Incomplete histories](incomplete_histories.md) and [mixed-frequency data](mixed_frequency_data.md)
- [Minimum tracking error](minimum_tracking_error.md) and [risk budgeting](risk_budgeting.md)
- [API reference](api.rst) for the verified public entry points
- [Offline multi-asset rolling example](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/backtests/multiasset_saa.py)
- [Rendered article](https://optimalportfolios.readthedocs.io/en/latest/rolling_backtests.html)
  for viewers without mathematical rendering

## References

- OptimalPortfolios. [Rolling portfolio dispatcher](../src/optimalportfolios/optimization/wrapper_rolling_portfolios.py):
  objective dispatch, target filtering and delegation to qis.
- OptimalPortfolios. [Weight-drift implementation](../src/optimalportfolios/utils/weights_drift.py):
  NAV-growth normalisation, causal price anchors and fallback rules.
- QuantInvestStrats. [Portfolio backtester](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py):
  units, execution lag, cash and transaction costs.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff)
  and [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
