---
myst:
  html_meta:
    description: >-
      Eligibility, frozen target weights, covariance filtering and missing-price
      behavior in OptimalPortfolios rolling workflows and the qis backtester.
---

# Incomplete histories and frozen positions

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/OptimalPortfolios/commit/fb8848d327c0585eaf0933dba6137ec6b8338bbf)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

An incomplete history is a price panel in which an instrument is not observed on every date.
Eligibility determines whether an instrument may enter a solve; freezing restricts changes to
an existing position. These policies require separate inputs.

## Overview

A late starter, an isolated missing price and a position with a dealing restriction describe
different economic states. A finite covariance or a completed backtest does not establish that
the intended investment universe and trading policy were represented.

OptimalPortfolios owns eligibility, constraint alignment and rolling optimization.
[qis](https://github.com/ArturSepp/QuantInvestStrats) owns the EWMA recursion and the simulation
of units, NAV and trading costs. Its
[software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
identifies that delegated calculation layer.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Simple price returns in the drift example; supplied per-observation returns in the EWMA kernel example; `EwmaCovarEstimator` itself samples log returns |
| Estimation grid | Two quarter-end decision dates for the policy panels and the drift; four business days for the holdings paths; three observations for the EWMA kernel |
| Rebalancing grid | Explicit target dates: one opening trade, or trades on the first two days in the rebalance-on-gap path; implementation lag zero |
| Covariance units | Annual return-squared in the filtering example; per observation, not annualised, in the EWMA kernel example |
| Expected returns | None |
| Weight state | Baseline `weights_0` drifted to the decision date by `apply_drift_to_weights_0`; executed units and cash in the qis holdings paths |
| Solver | None; the page aligns constraints and simulates holdings without a solve |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Input or symbol | Meaning and units |
|---|---|
| `prices` | Floating-point prices, with an ordered `DatetimeIndex` and unique ticker columns. |
| `inclusion_indicators` | Binary eligibility, for wrappers that accept it: one includes, zero excludes. |
| `rebalancing_indicators` | Binary trading policy for supported wrappers: one permits rebalancing, zero requests a freeze. |
| `weights_0`, $w_i^0$ | Baseline weight of asset $i$, as a fraction of NAV. |
| $r_i$ | Simple price return between the previous and current decision dates. |
| $g$ | Portfolio NAV growth factor over that interval, with residual cash earning zero. |
| `pd_covar` | Labeled square covariance matrix; the examples below use annual return-squared units. |

Construct complete indicator panels on the covariance decision dates, with every intended
ticker and explicit zeros or ones. Missing rows, missing labels and explicit `NaN` cells are
different inputs; wrappers do not apply one shared filling policy. Observation availability
alone does not identify listing status, liquidation proceeds or permission to trade.

The drift example uses simple returns between two dates and no annualization.
EWMA estimation normally constructs log returns at `returns_freq`, optionally removes an EWMA
mean and annualizes the covariance. See [covariance estimators](covariance_estimators.md)
for those choices. The holdings examples have no funding, carry, fees or transaction costs,
use NAV-based sizing and explicitly set a zero implementation lag.

## Methodology

### Missing observations

Leading missing prices record a period before the first observation; trailing missing prices
record a period after the last. Neither establishes the economic reason for the boundary.
Define an eligibility and valuation policy rather than inferring one from `NaN`.

An interior gap falls between valid observations. Between rebalancings, qis retains units
through such a gap but excludes the unpriced leg from that date's NAV. The resulting drop and
recovery need not be investment returns. A gap on a rebalance date has a different outcome,
described under [price gaps at implementation](#price-gaps-at-implementation).

`EwmaCovarEstimator` passes `NanBackfill.ZERO_FILL` to qis. For the ordinary covariance tensor,
a non-finite covariance update resets that entry of the recursive state to zero. This is
different from replacing a missing input return with zero, which would retain a decaying
contribution from the previous state. A missing source price may already have been treated by
return sampling before the covariance kernel sees it. See the
[estimator source](../src/optimalportfolios/covar_estimation/ewma_covar_estimator.py) and
[qis EWMA implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ewm.py).

`filter_covar_and_vectors_for_nans` removes assets with zero, negative or `NaN` diagonal
variance. Small positive variances survive unchanged unless `variance_floor` is supplied.
The floor applies only after filtering; it does not rescue a zero-variance asset.
The helper alone does not certify a finite or positive-semidefinite matrix: positive infinity
on a diagonal and non-finite off-diagonal entries can survive. Validation or factorization in
the selected solver path remains necessary. In particular, a tiny finite warm-up variance is
not evidence that a late starter has enough history to enter.

### Eligibility versus freezing

Exclusion normally removes an asset from the solve and produces a zero target in wrappers
that restore the full universe. A zero risk budget is a separate, risk-budgeting-specific
route to exclusion; it is not a universal trading flag. An asset that must remain owned should
not simply be removed from the solver universe.

For `Constraints.update_with_valid_tickers`, provide both box sides and a resolved baseline.
For a retained asset whose rebalancing indicator is zero, the intended pin is:

$$
w_i^{\min} = w_i^{\max} = w_i^0.
$$

The method replaces each box side only if that side already exists. Both `min_weights` and
`max_weights` are therefore needed for an exact pin. Long-only frozen weights are clipped at
zero. A supplied `weights_0` overrides the baseline stored on `Constraints`; otherwise the
stored baseline is used. If neither exists, there is no position to pin.

First-date behavior is wrapper-specific. The minimum-variance/return-floor,
maximum-return/volatility-target and alpha-over-tracking-error wrappers pass the first
indicator row to constraint alignment, which can use a stored baseline. `rolling_risk_budgeting`
instead suppresses freezing until a previous rolling allocation exists; its single-date
wrapper uses an explicit `weights_0` for fixed positions. See
[risk-budgeting qualifications](risk_budgeting.md) and the
[authoritative constraints contract](constraints.md#frozen-positions).

### Missing drift prices

`apply_drift_to_weights_0` uses the latest available price at or before each decision anchor.
An unavailable ratio, including an asset with no usable starting price, is treated as a flat
simple return. In the ordinary, finite-growth case:

$$
g = 1 + \sum_i w_i^0 r_i,
\qquad
w_i^{\mathrm{drift}} = \frac{w_i^0(1+r_i)}{g}.
$$

A flat leg can still change weight because it shares the portfolio denominator.
This helper approximates decision-date holdings; it does not reconstruct lagged executions,
funding, fees or cash charges. Whole-vector failure gates can return the original baseline.
See [rolling backtests](rolling_backtests.md#decision-date-drift-versus-executed-holdings)
for the timing distinction and [the drift source](../src/optimalportfolios/utils/weights_drift.py)
for the full fallback contract.

### Price gaps at implementation

The following describes the qis implementation of the locked environment, which the canonical
script of this page checks. A missing opening price yields zero units for that leg; its intended allocation remains cash and is not redistributed.
When prices later appear, entry waits for another scheduled rebalance.

For an already-held leg, the outcome depends on whether a rebalance occurs:

| Missing-price date | Units and valuation |
|---|---|
| Between rebalancings | Existing units remain; their unpriced value is omitted from NAV. |
| On a rebalance | The kernel recomputes target units, replaces the missing-price result with zero, and excludes that leg from the cash-transfer calculation. Previously held units can therefore be cleared without liquidation proceeds. |

The second case is an implementation limitation, not a model of a suspended instrument or a
delisting recovery. Inspect and resolve the price/trading policy before interpreting such a
path. An interior-gap warning does not distinguish the two cases. See the
[qis backtester source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py)
and the executable example below.

A frozen decision-date target is also not an execution-level unit lock. The qis backtester
does not receive the optimizer's per-asset `rebalancing_indicators`. Implementation lag, changed
prices or costs can make executing the same target weight trade units.

## Worked example

All inputs here are synthetic teaching cases. The six Python blocks run in order; they are
excerpts of the canonical script
[`examples/docs/incomplete_histories.py`](../examples/docs/incomplete_histories.py), which runs
them and asserts every number on this page against an exact ledger of units and cash:

```console
python -m examples.docs.incomplete_histories
```

### Separate policy panels

These are independent illustrations: the first panel concerns a late starter, and the second
concerns a locked fund. They are not two masks for the same universe.

```python
import pandas as pd

decision_dates = pd.to_datetime(["2024-03-28", "2024-06-28"])
eligibility = pd.DataFrame(
    {"Liquid": [1.0, 1.0], "Late Starter": [0.0, 1.0]},
    index=decision_dates,
)
can_rebalance = pd.DataFrame(
    {"Liquid": [1.0, 1.0], "Locked Fund": [0.0, 0.0]},
    index=decision_dates,
)
```

### Pin an existing position

This example supplies a stored baseline, so the first alignment can already freeze the fund.

```python
import numpy as np
import optimalportfolios as op

assets = ["Liquid", "Locked Fund"]
spec = op.Constraints(
    min_weights=pd.Series(0.0, index=assets),
    max_weights=pd.Series(1.0, index=assets),
    weights_0=pd.Series([0.60, 0.40], index=assets),
)
aligned = spec.update_with_valid_tickers(
    valid_tickers=assets,
    rebalancing_indicators=can_rebalance.iloc[0],
)
```

The fund's lower and upper bounds are both **0.40**. The liquid asset retains bounds
**0.00 to 1.00**; full-investment and other configured constraints still apply.

### A flat leg changes weight

```python
drift_dates = pd.to_datetime(["2024-03-28", "2024-06-28"])
drift_prices = pd.DataFrame(
    {"Liquid": [100.0, 120.0], "Locked Fund": [np.nan, np.nan]},
    index=drift_dates,
)
drifted = op.apply_drift_to_weights_0(
    weights_0=spec.weights_0,
    prices=drift_prices,
    prev_date=drift_dates[0],
    date=drift_dates[1],
)
```

Starting from NAV 100, the liquid position grows from 60 to 72 and the flat fallback position
remains valued at 40 in this approximation. Total NAV is 112. The resulting weights are
**0.642857** and **0.357143**, respectively. The missing ratio does not preserve a 40% weight.

> **Insight.** A leg whose price is missing is carried at a flat price, not at a fixed weight.
> Its weight still falls when the rest of the portfolio grows, because every weight shares the
> NAV denominator.

### Missing prices in three holdings paths

```python
import warnings
import qis

days = pd.date_range("2024-01-02", periods=4, freq="B")
gap_prices = pd.DataFrame(
    {"Liquid": [100.0, 110.0, 120.0, 130.0],
     "Gapped": [100.0, np.nan, 110.0, 120.0]},
    index=days,
)
opening_targets = pd.DataFrame([[0.60, 0.40]], index=days[:1], columns=gap_prices.columns)
retrade_targets = pd.DataFrame(
    [[0.60, 0.40], [0.60, 0.40]], index=days[:2], columns=gap_prices.columns,
)
late_prices = gap_prices.assign(Gapped=[np.nan, 100.0, 110.0, 120.0])

# Retain the warnings for inspection; these paths intentionally contain missing prices.
with warnings.catch_warnings(record=True) as captured_warnings:
    warnings.simplefilter("always", UserWarning)
    held_gap = qis.backtest_model_portfolio(
        prices=gap_prices, weights=opening_targets, initial_nav=100.0,
        rebalancing_costs=None, weight_implementation_lag=0,
    )
    traded_gap = qis.backtest_model_portfolio(
        prices=gap_prices, weights=retrade_targets, initial_nav=100.0,
        rebalancing_costs=None, weight_implementation_lag=0,
    )
    late_entry = qis.backtest_model_portfolio(
        prices=late_prices, weights=opening_targets, initial_nav=100.0,
        rebalancing_costs=None, weight_implementation_lag=0,
    )
warning_messages = [str(item.message) for item in captured_warnings]
```

| Date | Hold through gap: NAV | Rebalance on gap: NAV | Missing opening price: NAV |
|---|---:|---:|---:|
| 2024-01-02 | 100.00 | 100.00 | 100.00 |
| 2024-01-03 | 66.00 | 66.00 | 106.00 |
| 2024-01-04 | 116.00 | 69.60 | 112.00 |
| 2024-01-05 | 126.00 | 73.20 | 118.00 |

The held-gap path owns 0.6 liquid units and 0.4 gapped units throughout. In the rebalance-on-gap
path, 3 January changes those holdings to 0.36 and zero, with cash 26.40. The original gapped
units do not return when their price returns. The late-entry path owns 0.6 liquid units,
zero gapped units and cash 40 throughout. Its missing allocation is never redistributed.

![Left: NAV of the three paths over four business days; the hold-through-gap and
rebalance-on-gap paths both fall to 66 on 3 January, when the gapped price is missing, after
which the held path recovers to 126 and the rebalanced path stays near 70; the missing-opening-price
path rises from 100 to 118. Right: in the rebalance-on-gap path, the gapped holding is replaced
by cash of 26.40 from 3 January and never returns.](images/incomplete_histories_price_gaps.png)

*Figure: the three holdings paths of the example, and the holdings of the rebalance-on-gap path.
Drawn by the `exhibit` function of the canonical script; the
[analytics gallery](analytics_gallery.md) lists its provenance.*

> **Pitfall.** A missing price on a rebalance date clears a held position without liquidation
> proceeds: the gapped units become zero and do not return when the price does. An optimizer
> freeze does not prevent this, because qis never receives it. Resolve the missing price, or
> pass no target on that date, as in the hold-through-gap path.

### Filtering is separate from history eligibility

```python
names = ["Liquid", "Zero", "Negative", "Missing", "Warmup"]
covariance = pd.DataFrame(
    np.diag([0.04, 0.0, -0.01, np.nan, 1e-12]), index=names, columns=names,
)
filtered, _ = op.filter_covar_and_vectors_for_nans(covariance)
floored, _ = op.filter_covar_and_vectors_for_nans(covariance, variance_floor=1e-6)
eligible, _ = op.filter_covar_and_vectors_for_nans(
    covariance, inclusion_indicators=pd.Series([1, 0, 0, 0, 0], index=names),
)
```

The first two outputs retain `Liquid` and `Warmup`. The warm-up variance is **1e-12** without
a floor and **1e-6** with the explicit floor. The eligibility mask retains only `Liquid`.

### A covariance-state reset

This final block isolates the ordinary qis kernel using supplied, undemeaned return observations
at an arbitrary observation frequency; no price sampling or annualization is performed.

```python
observations = np.array([[0.01, 0.02], [np.nan, 0.03], [0.01, 0.04]])
covariance_states = qis.compute_ewm_covar_tensor(
    a=observations, span=3, nan_backfill=qis.NanBackfill.ZERO_FILL,
)
```

The first asset's variance is **0.00005**, then **0**, then **0.00005**.
Feeding a zero return at the middle observation instead would produce **0.000025**
at that observation. This illustrates why the backfill enum should not be described
as a generic promise to replace missing source returns with zeros.

## Implementation in optimalportfolios

| Entry point | Role and qualification |
|---|---|
| `filter_covar_and_vectors_for_nans` | Filters the solver universe and aligns companion vectors; flooring and finite-vector screening are opt-in. |
| `Constraints.update_with_valid_tickers` | Aligns the retained universe, resolves baseline weights and applies available frozen box sides. |
| `apply_drift_to_weights_0` | Advances a baseline to a decision date using price ratios and explicit fallbacks. |
| `rolling_min_variance_target_return`, `rolling_max_return_target_vol`, `rolling_maximise_alpha_over_tre` | Accept a per-date freezing panel and pass it to alignment. |
| `rolling_risk_budgeting` | Uses a separate fixed-position allocation path; consult its own first-date and risk-share limitations. |
| `qis.backtest_model_portfolio` | Converts target weights to units; it does not consume a per-asset optimizer freeze panel. |

The three constraint-alignment rolling wrappers reindex freezing rows to covariance dates and
fill missing cells with zero. At single-date constraint alignment, a newly inserted ticker label
instead defaults to one. Use complete binary panels so this distinction cannot silently change
the policy. The alignment method treats values not close to one as frozen; do not use that
tolerance as a substitute for input validation.

### Reproduction and verification context

The [canonical script](../examples/docs/incomplete_histories.py) checks the six blocks against
independent position-value and covariance-state references, including the displayed table. The
test suite runs it, and so does the offline examples lane of CI. Its checks characterise the
imported implementation of qis, so a change in qis's gap handling fails them.

## Interpretation and limitations

### Operational checks

Before a rolling run, inspect first/last observations, interior gaps, warm-up requirements,
eligibility and tradability on every decision and implementation date. Identify whether a
missing price represents stale valuation, absent data or an unavailable trade. Forward-filling
may implement an explicit valuation convention; it does not create a tradable quote.

After the run, reconcile target weights, executed units, realized weights, cash, warnings and
solver diagnostics. The absence of an interior-gap warning does not make leading or trailing
missing prices economically harmless.

Frozen positions can breach group limits. Constraint alignment can grant logged one-period
waivers for mismatches introduced by freezing; `relax_frozen_group_bounds=False` disables them.
`max_relaxation_tol` escalates logging rather than capping the waiver. Review the
[aligned bounds and waiver records](constraints.md#frozen-group-bound-waivers) against the mandate.
Risk budgeting has additional limitations when fixed positions alter risk contributions.

These examples characterize data and implementation behavior. Their NAV paths are not
investment-performance evidence, and the flat-price fallback is not a validated valuation model.

## See also

- [Rolling backtests](rolling_backtests.md) and [turnover and transaction costs](turnover_and_transaction_costs.md).
- [Minimum tracking error](minimum_tracking_error.md) for explicit inclusion indicators.
- [Risk budgeting](risk_budgeting.md) for zero budgets and fixed positions.
- [Constraints](constraints.md#universe-alignment-and-rebalancing-policy) for exact alignment and waiver rules.
- [API reference](api.rst) for the public helpers.

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- OptimalPortfolios: [covariance filtering](../src/optimalportfolios/utils/filter_nans.py),
  [constraint alignment](../src/optimalportfolios/optimization/constraints/alignment.py) and
  [rolling minimum variance](../src/optimalportfolios/optimization/saa/min_variance_target_return.py).
- qis: [EWMA implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ewm.py)
  and [holdings simulation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py).
