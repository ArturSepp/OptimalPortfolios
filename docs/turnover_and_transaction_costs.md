---
myst:
  html_meta:
    description: >-
      Turnover constraints and transaction costs in OptimalPortfolios: full L1 target changes,
      executed notional, NAV denominators, opening trades, and reproducible qis examples.
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

| Input or symbol | Meaning |
|---|---|
| $w_i$, $w_{0,i}$ | Proposed and baseline weights as fractions of portfolio NAV. |
| $\tau$, `turnover_constraint` | Hard budget on the selected full L1 expression. |
| $a_i$, `turnover_costs` | Per-asset multipliers used by the optimiser's turnover expression. |
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
utility formulation. A configured penalty field alone does not switch every solver into it.
Penalty strength, hard-budget size and the backtest cash-cost rate are separate inputs.

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

This is the default convention of a newly constructed `qis.PortfolioData` in the verified
environment. The denominator is same-observation, post-cost NAV. Entry of notional 100 with
cost 0.10 therefore gives `100 / 99.90 = 1.001001`, slightly more than 100%.

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

### Executed trades from the original backtest example

The following example is separate from that optimisation: it deliberately supplies targets
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

### Cost inputs and timing

| `rebalancing_costs` input | Interpretation |
|---|---|
| Scalar | One fractional rate for every instrument and trade date. |
| Ticker-indexed Series | A separate constant rate for each price column. |
| Date-by-ticker DataFrame | Time-varying rates, forward-filled onto the price grid and read at execution. |

A date-indexed Series is rejected as ambiguous. A cost DataFrame must contain every price
column. In the verified qis implementation, dates before its first schedule row are costless,
and missing aligned DataFrame values become zero. This is an explicit missing-cost policy,
not an estimate of unavailable costs. Supply a complete schedule when zero is unintended.

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

The four Python blocks are canonical sequential examples.
[Their executable tests](../src/optimalportfolios/tests/turnover_and_transaction_costs_documentation_test.py)
check the displayed allocation and trade tables against independent constrained-allocation
and currency-ledger calculations:

```console
python -m pytest src/optimalportfolios/tests/turnover_and_transaction_costs_documentation_test.py -q
```

Use the external interpreter and generated-state setup in
[AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md) on Windows.
Local verification on 2026-09-13 used an export of the OptimalPortfolios 7.6.0 working tree,
installed qis 5.26.0, CVXPY 1.9.2 and CLARABEL 0.11.1. This identifies the tested working
source and environment, not a published artifact or the current locked CI environment.

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
