---
myst:
  html_meta:
    description: >-
      Turnover constraints and transaction costs in OptimalPortfolios: full L1 target changes,
      the turnover penalty's trade-off against tracking error, executed notional, NAV
      denominators, opening trades, and reproducible qis examples.
---

# Turnover and transaction costs

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/OptimalPortfolios/commit/fb8848d327c0585eaf0933dba6137ec6b8338bbf)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Executed holdings, turnover and costs use [qis](https://github.com/ArturSepp/QuantInvestStrats);
see the [qis citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Portfolio turnover measures the amount traded or proposed to trade. Transaction costs measure
the resources consumed by those trades. OptimalPortfolios uses turnover limits or penalties
during construction; qis simulates execution and deducts proportional costs from cash.
The construction budget and realised cost are different quantities.

## Overview

A useful audit reports the proposed target change, the executed trade and the resulting cost
separately. A turnover constraint can alter a target without charging any cash. Conversely,
a backtest can charge costs without imposing a turnover limit on the supplied targets.

This article uses full, two-sided turnover: purchases and sales both count, with no factor
of one half. Its executed examples concern cash securities with floating-point total-return
prices, an explicit implementation lag and no funding, management fees or additional carry.
See the [complete constraints contract](constraints.md) for solver and group-policy details.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | No return series enters the optimisation examples; the backtest values held units at the supplied floating-point total-return prices, so NAV moves with simple price ratios |
| Estimation grid | None; both covariances are fixed synthetic inputs, not estimates |
| Rebalancing grid | Single-date solves for the optimisation examples; in the backtest, targets dated 2 and 3 January 2024 on a three-business-day price grid execute one observation later (`weight_implementation_lag=1`), on 3 and 4 January |
| Covariance units | Annualized fractional return-squared, as supplied; no solver rescales it, so tracking error is annual volatility. Turnover budgets, penalties and cash costs are per decision or price observation and never annualized |
| Expected returns | None; the budget example minimises variance (`PortfolioObjective.MIN_VARIANCE`) and the penalty example passes `alphas=None` |
| Weight state | `weights_0`, stored on `Constraints` or passed to the wrapper, which takes precedence, is the turnover baseline; without it turnover rows and penalties are skipped. The backtest holds units between trades and resizes them to the target at execution prices |
| Solver | CVXPY with CLARABEL, the `OptimiserConfig` default, on an eigendecomposed covariance (`factorize_covar=True`); qis simulates execution and costs without an optimiser |

The notation follows the [conventions page](conventions.md#notation). The table below adds the
symbols and inputs specific to turnover and costs.

| Input or symbol | Meaning |
|---|---|
| $w_i$, $w_{0,i}$ | Proposed and baseline weights as fractions of portfolio NAV. |
| $w^{\mathrm{bm}}$, $d$ | Benchmark weights and active weights $d=w-w^{\mathrm{bm}}$. |
| $\tau$, `turnover_constraint` | Hard budget on the selected full L1 expression. |
| $a_i$, `turnover_costs` | Per-asset multipliers used by the optimiser's turnover expression. |
| $\kappa_{\mathrm{TE}}$, `tre_utility_weight` | Weight of active variance in a utility objective. |
| $\kappa_{\mathrm{TO}}$, `turnover_utility_weight` | Weight of the full L1 turnover in a utility objective; not a cash cost. |
| $u_{i,t}$, $P_{i,t}$ | Executed units and their current price. |
| $c_{i,t}$, `rebalancing_costs` | Backtest cost per unit of traded notional, in fractional units. |
| $V_t$ | Portfolio NAV after trading and costs at observation $t$. |

Provide aligned asset labels and finite, nonnegative cost inputs. Multipliers $a_i$ and cash-cost
rates $c_{i,t}$ need not be the same. If multipliers are cost fractions, their weighted budget
also has cost-fraction units; a limit calibrated for unit multipliers cannot be reused blindly.

Turnover here is measured per decision or price observation. Neither the L1 budget nor the
cash charge is annualised. A rolling or resampled turnover statistic needs its own period label.
Price returns and estimation conventions belong to the upstream
[rolling workflow](rolling_backtests.md), not to the cost-rate units.

## Methodology

### Target turnover in optimisation

The unweighted hard constraint is

$$
\sum_i \lvert w_i-w_{0,i}\rvert \leq \tau.
$$

A five-percentage-point sale and a five-point purchase use `0.10` of budget. Moving from
60/40 to 50/50 uses `0.20`. Both legs count; net weight change would be zero.

With `turnover_costs`, the common CVXPY constraint uses

$$
\sum_i \lvert a_i(w_i-w_{0,i})\rvert \leq \tau.
$$

For a change from 60/40 to 55/45, unit multipliers give `0.10`, while multipliers `[2, 1]`
give `0.15`. This weighted amount changes constraint or objective units; it is not a cash
deduction. The [constraint compiler](../src/optimalportfolios/optimization/constraints/backends.py)
and [total-turnover contract](constraints.md#total-turnover) define the calculation.

A baseline is required. Without resolved `weights_0`, the common CVXPY compiler skips total
and group turnover constraints; it does not infer zero initial holdings. The first solve
**can** be constrained if current holdings are supplied through `Constraints.weights_0`
or the wrapper's `weights_0` argument. In the quadratic wrapper, a supplied argument takes
precedence over the stored constraint baseline.

Rolling solvers normally drift prior targets between decision dates. That construction baseline
can differ from lagged, cost-bearing executed holdings; see
[decision-date drift versus executed holdings](rolling_backtests.md#point-in-time-and-drift-rules).
A construction limit is therefore not a guarantee about subsequently realised turnover.

### Hard limits and utility penalties

`turnover_utility_weight` controls a penalty only in an optimisation path that uses that
utility formulation. A configured penalty field alone does not switch every solver into it:
every `Constraints` carries `turnover_utility_weight=0.40` and `tre_utility_weight=1.0` by
default, and the forced-constraint solve of the budget example below reads neither. With
`ConstraintEnforcementType.UTILITY_CONSTRAINTS`, `wrapper_maximise_alpha_over_tre`
([tactical allocation](alpha_over_tracking_error.md)) maximises

$$
\alpha^{\top}d-\kappa_{\mathrm{TE}}d^{\top}\Sigma d
-\kappa_{\mathrm{TO}}\sum_i \lvert a_i(w_i-w_{0,i})\rvert
$$

over the hard rows that remain, such as the budget and the weight bounds; with `alphas=None`
the first term is dropped. The utility branches of `wrapper_max_return_target_vol` and
`wrapper_min_variance_target_return` add the same turnover term to their own objectives, and
the soft-tracking-error path of `wrapper_maximise_alpha_with_target_return` keeps it only when
no hard turnover cap is set. Without `weights_0` the penalty is skipped, like the hard budget.
Penalty strength, hard-budget size and the backtest cash-cost rate are separate inputs.

Two properties follow for a fixed $\kappa_{\mathrm{TE}}$. First, raising $\kappa_{\mathrm{TO}}$
cannot raise the penalised turnover of the optimum: adding the optimality inequalities of two
penalty weights shows that the larger weight never has the larger turnover, and with no alpha
the tracking error never falls. Second, the absolute value stops trading at a finite weight.
With no alpha, unit multipliers, baseline weights strictly inside their bounds and no hard row
other than the budget, the baseline is optimal exactly when

$$
\kappa_{\mathrm{TO}}\geq\kappa_{\mathrm{TE}}\left(\max_i(\Sigma d_0)_i-\min_i(\Sigma d_0)_i\right),
\qquad d_0=w_0-w^{\mathrm{bm}}.
$$

The vector $2\kappa_{\mathrm{TE}}\Sigma d_0$ is the marginal cost of active variance at the
baseline; the budget row lets a common shift centre it, and the L1 subgradient absorbs up to
$\kappa_{\mathrm{TO}}$ on each asset.

Group turnover applies group loadings inside the absolute-value expression. It does not also
apply the portfolio-level `turnover_costs` multipliers. In forced constraint mode, group and
total caps are additive; in the generic utility builder, a group-turnover object takes
precedence over the total-turnover penalty. Consult
[group turnover](constraints.md#group-turnover) and
[utility group precedence](constraints.md#group-precedence) before combining them.

### Realised turnover and costs in the backtest

qis converts each implemented target to units at execution prices and holds those units
between trades. For cash securities, its per-asset proportional cost is

$$
C_{i,t}=c_{i,t}P_{i,t}\lvert u_{i,t}-u_{i,t-1}\rvert.
$$

For opening-trade costs, pre-entry units are zero, including when entry is on the first
price observation. Rates are read on the actual execution date after applying the lag.
The charge is deducted from cash after sizing the trade using pre-cost NAV.
See the [qis backtester](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py).

With `TurnoverComputationType.EXECUTED_NOTIONAL_NAV`, qis reports two-sided executed turnover as

$$
T_t^{\mathrm{NAV}}=
\frac{\sum_i P_{i,t}\lvert u_{i,t}-u_{i,t-1}\rvert}{V_t}.
$$

This is the default convention of a newly constructed `qis.PortfolioData` in qis 5.31.0, the
minimum version this package requires. The denominator is same-observation, post-cost NAV.
Entry of notional 100 with cost 0.10 therefore gives `100 / 99.90 = 1.001001`, slightly more
than 100%.

For derivatives, `turnover_unit_notional` can represent full contract value including
multipliers and currency conversion; a return-price series alone need not supply that value.
This article's cash-security example uses prices as unit notionals.
The [qis turnover implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/turnover.py)
owns the reporting conventions.

The first turnover row is normally missing because `units.diff()` has no preceding holding.
It does not assume zero pre-entry units for that statistic. Opening costs can still be recorded
on that row. With delayed entry, a prior in-sample zero-unit row makes the later opening turnover
observable. Keep this distinction when reconciling turnover totals and charged costs.

## Worked example

All four Python blocks of this page are excerpts of the canonical script
[`examples/docs/turnover_and_transaction_costs.py`](../examples/docs/turnover_and_transaction_costs.py),
which runs them in order and asserts every number and property on this page against a
reference computed a different way:

```console
python -m examples.docs.turnover_and_transaction_costs
```

### A hard target budget with supplied starting holdings

This preserves the original constraint setup: current weights are 60/40 and the full L1
budget is `0.10` with unit multipliers.

```python
import pandas as pd
import optimalportfolios as opt

current = pd.Series({"A": 0.60, "B": 0.40})
constraints = opt.Constraints(
    is_long_only=True,
    weights_0=current,
    turnover_constraint=0.10,
    turnover_costs=pd.Series({"A": 1.0, "B": 1.0}),
)
```

A fixed, synthetic annual covariance with variances `0.04` and `0.01` and zero covariance
gives a minimum-variance target. The stored starting holdings apply even though this is
the first solve.

```python
covar = pd.DataFrame(
    [[0.04, 0.0], [0.0, 0.01]], index=current.index, columns=current.index
)
optimal_weights, outcome = opt.wrapper_quadratic_optimisation(
    pd_covar=covar, constraints=constraints,
    portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
)
if not (outcome.accepted and outcome.compliant and outcome.fallback_source is None):
    raise RuntimeError(f"Unusable solve: {outcome.status}")
```

| Asset | Baseline weight | Constrained target | Absolute change |
|---|---|---|---|
| A | 0.60 | 0.55 | 0.05 |
| B | 0.40 | 0.45 | 0.05 |

Without the turnover budget, the minimum-variance allocation is 20/80.
The budget restricts A to at least 55%, so the constrained solution stops at 55/45.
Its full L1 change is **0.10**. No cash cost is charged by this solve.

### Trading turnover against tracking error with a penalty

A penalty puts a price on trading instead of a limit on it. A four-asset example isolates the
trade-off: equity, credit, government bonds and gold with annual volatilities of 16%, 8%, 5%
and 15%, a benchmark of 50/20/25/5, and current holdings of 40/25/20/15, a full L1 gap of 0.30.
The canonical script calls `wrapper_maximise_alpha_over_tre` with `alphas=None` and the holdings
as `weights_0`, under `ConstraintEnforcementType.UTILITY_CONSTRAINTS` with
`tre_utility_weight=100.0` and `turnover_utility_weight` from 0 to 0.5 in steps of 0.01. It
computes each turnover and tracking error with NumPy, checks that `compute_tre_turnover_stats`
reports the same two numbers, and solves every weight again exactly by enumerating the
first-order conditions.

At weight 0 the solve holds the benchmark: no tracking error, for the whole 0.30 of turnover.
Credit and government bonds, whose deviations cost least risk, stop trading first, at weights
of about 0.014 and 0.039. The equity-to-gold trade continues: at 0.20 the solve trades 0.0855
of NAV and runs 1.02% tracking error. That trade stops at the threshold
$100\times0.003851=0.3851$, where the solve keeps the holdings and their 1.88% tracking error.
Along the whole path turnover never rises and tracking error never falls.

![Left: full L1 turnover against ex-ante tracking error for each turnover penalty weight, falling
from 30% turnover with zero tracking error at weight 0, the benchmark, through 17.8% at 0.02,
13.2% at 0.1 and 8.5% at 0.2 to zero turnover with 1.88% tracking error at 0.4, the current
holdings. Right: turnover against the penalty weight, dropping steeply below 0.05 as credit and
government bonds stop trading, then falling linearly to zero at the no-trade threshold of
0.3851.](images/turnover_penalty_tradeoff.png)

*Figure: the turnover and ex-ante tracking error of the four-asset example as the turnover
penalty weight grows, with the tracking-error weight fixed at 100. Drawn by the `exhibit`
function of the canonical script; the [analytics gallery](analytics_gallery.md) lists its
provenance.*

> **Insight.** An L1 turnover penalty has a finite no-trade point. With no alpha, the current
> holdings are optimal once `turnover_utility_weight` reaches `tre_utility_weight` times the
> spread of $\Sigma d_0$, the marginal active risk of the current deviations: 0.3851 in the
> example. Below that point the penalty keeps the deviations that cost least risk, here credit
> and government bonds, and spends turnover on the equity-to-gold trade, the pair with the
> largest spread.

### Executed trades from the original backtest example

The following example is separate from both optimisations: it deliberately supplies targets
60/40 and 50/50 to qis. Their target change is `0.20` and they are not claimed to satisfy
the `0.10` constraint above. The original prices, targets, 10 bp rate and lag-one call
are preserved.

```python
import pandas as pd
import qis

prices = pd.DataFrame(
    {"A": [100.0, 102.0, 101.0], "B": [100.0, 99.0, 101.0]},
    index=pd.date_range("2024-01-02", periods=3, freq="B"),
)
targets = pd.DataFrame(
    {"A": [0.60, 0.50], "B": [0.40, 0.50]},
    index=prices.index[:2],
)
portfolio = qis.backtest_model_portfolio(
    prices=prices,
    weights=targets,
    rebalancing_costs=0.0010,
    weight_implementation_lag=1,
    ticker="Cost-aware backtest",
)
```

The 2 January target enters on 3 January at prices 102 and 99. It buys $60/102$ units of A
and $40/99$ units of B, costing 0.06 and 0.04 respectively. The next target trades on 4 January
after the existing units have earned the intervening price returns.

| Trade date | Traded notional | Cash cost | Post-cost NAV |
|---|---|---|---|
| 2024-01-03 | 100.000000 | 0.100000 | 99.900000 |
| 2024-01-04 | 18.603684 | 0.018604 | 100.101242 |

On the second trade, the pre-cost NAV is approximately 100.119846. Target units are sized
from that amount, then costs are deducted. The traded notional is not `0.20` times NAV:
price drift, implementation dates and the existing cost debit affect the actual trade.

## Implementation in optimalportfolios

`compute_tre_turnover_stats(covar, benchmark_weights, weights, weights_0, alphas=None)`
summarises one target. It returns `(te_vol, turnover, port_alpha, port_vol, benchmark_vol)`:
`turnover` is the full L1 change `nansum(abs(weights - weights_0))`, with no half factor, and
`te_vol` is the tracking error in the units of `covar`. The covariance is a NumPy array without
labels, so pass it in the order of the weight index. The weight differences align by label, and
a label missing from `weights_0` becomes a NaN change that the turnover sum drops.

### Cost inputs and timing

| `rebalancing_costs` input | Interpretation |
|---|---|
| Scalar | One fractional rate for every instrument and trade date. |
| Ticker-indexed Series | A separate constant rate for each price column. |
| Date-by-ticker DataFrame | Time-varying rates, forward-filled onto the price grid and read at execution. |

A date-indexed Series is rejected as ambiguous. A cost DataFrame must contain every price
column. In qis 5.31.0, dates before the first schedule row are costless, and missing aligned
DataFrame values become zero. This is an explicit missing-cost policy, not an estimate of
unavailable costs. Supply a complete schedule when zero is unintended.

Setting `turnover_costs` or a turnover utility weight on `Constraints` does not configure
`rebalancing_costs`. The examples invoke the two layers separately.

### Explicit turnover and cost reporting

Use `roll_period=None` for observation-level values. The default reporting window is 260
observations and can give all-missing turnover on a short example.

```python
executed_turnover = portfolio.get_turnover(
    is_agg=True, roll_period=None,
    turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_NAV,
)
target_turnover = portfolio.get_turnover(
    is_agg=True, roll_period=None,
    turnover_computation_type=qis.TurnoverComputationType.TARGET_WEIGHTS,
)
cash_costs = portfolio.realized_costs.sum(axis=1)
cost_fractions = portfolio.get_costs(is_agg=True, roll_period=None)
```

The reporting modes have different numerators or denominators:

| Mode or output | Meaning |
|---|---|
| `EXECUTED_NOTIONAL_NAV` | Absolute units traded at current unit notional, divided by same-date NAV. |
| `EXECUTED_NOTIONAL_GROSS` | The same traded notional divided by current gross exposure. |
| `TARGET_WEIGHTS` | Absolute changes between input target rows, on decision dates; an allocation proxy. |
| `realized_costs` | Per-asset charges in portfolio currency units. |
| `get_costs` with its default normalisation | Charges divided by same-date NAV, then aggregated as requested. |

On 3 and 4 January, executed turnover is **1.001001** and **0.185849**.
The target proxy records **0.20 on 3 January**, the second decision date, not its execution date.
At a constant scalar rate, observed cost fractions equal that rate times observed executed
NAV turnover wherever the latter is defined. This does not supply an opening turnover where
the first row is missing.

For `get_turnover`, the deprecated boolean `is_unit_based_traded_volume=True` selects gross-exposure
turnover, not NAV turnover. Use the enum explicitly. This is separate from the similarly named
`get_costs` option, whose default normalises costs by NAV; `False` returns currency charges.

If `freq` is supplied, reporting first sums by that frequency, then applies `roll_period`.
The rolling count therefore refers to the resampled observations. Summing cost fractions is
not a compounded return penalty or necessarily the difference between independently simulated
gross and net NAV paths. See the
[PortfolioData implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/portfolio_data.py).

### Reproduction and verification context

The [canonical script](../examples/docs/turnover_and_transaction_costs.py) runs all four blocks.
It checks the displayed allocation against a one-dimensional optimality argument, the penalty
path against an exact solution of its first-order conditions and the closed-form no-trade
threshold, and the trade table and reported turnover against an exact rational currency
ledger. It also checks the baseline, cost-schedule, missing-rate, reporting-window and
deprecated-selector statements of this page. The test suite runs it, and so does the offline
examples lane of CI.

## Interpretation and limitations

<a id="interpretation-and-failure-modes"></a>

- A target constraint does not certify an executed turnover limit. Check decision-date
  baselines, trade dates, costs and eligible instruments together.
- Solver support, filtering, constraint rescaling and fallback policies vary by objective.
  Review the resolved constraint and `OptimizationOutcome` instead of relying only on
  an apparent weight sum. See [constraint filtering](constraints.md#universe-alignment-and-rebalancing-policy).
- Unpriced instruments cannot be traded normally, and missing prices inside a held asset's
  history can distort NAV. Read [incomplete histories](incomplete_histories.md); do not infer
  missing-data behavior from the complete-price examples.
- NAV and gross-exposure normalisations differ with cash, leverage, shorts and cost debits.
  Zero denominators make the corresponding turnover undefined rather than zero.
- A missing first turnover row is not evidence of a free opening trade. Reconcile actual
  currency charges before comparing aggregate measures.
- This proportional model does not estimate market impact, order-size capacity, bid/ask
  dynamics or tax effects. The short synthetic examples demonstrate conventions, not
  expected strategy performance.

> **Pitfall.** The default penalty weights do not suit an annualized covariance. Every
> `Constraints` carries `tre_utility_weight=1.0` and `turnover_utility_weight=0.40`. With those
> defaults the four-asset example's no-trade threshold is 0.003851, which 0.40 exceeds more than
> 100 times, so the utility solve returns the current holdings unchanged. Set both weights
> explicitly, in the units of the objective.

## See also

- [Rolling backtests](rolling_backtests.md) for holdings drift and implementation timing
- [Incomplete histories](incomplete_histories.md) for missing and frozen positions
- [Constraints](constraints.md) for full L1, group policies and backend support
- [API reference](api.rst) for `Constraints` and `apply_drift_to_weights_0`
- [Rendered article](https://optimalportfolios.readthedocs.io/en/latest/turnover_and_transaction_costs.html)
  for Markdown viewers without mathematical rendering

## References

- OptimalPortfolios. [Constraint backend compiler](../src/optimalportfolios/optimization/constraints/backends.py)
  and [constraints methodology](constraints.md): target budgets and utility penalties.
- QuantInvestStrats. [Portfolio backtester](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py):
  executed units, proportional costs and execution-date schedules.
- QuantInvestStrats. [Turnover computations](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/turnover.py)
  and [PortfolioData reporting](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/portfolio_data.py):
  explicit conventions, first-row handling and aggregation.
- [OptimalPortfolios citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff)
  and [qis citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
