---
myst:
  html_meta:
    description: >-
      Minimum tracking error in OptimalPortfolios: benchmark-relative construction,
      covariance units, solver diagnostics, and offline analytics through qis.RiskModel.
---

# Minimum tracking error

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/OptimalPortfolios/commit/fb8848d327c0585eaf0933dba6137ec6b8338bbf)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Ex-ante risk analytics use [qis](https://github.com/ArturSepp/QuantInvestStrats);
see its [software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Minimum tracking error constructs a feasible portfolio with the smallest modeled
volatility of returns relative to a supplied benchmark. The benchmark defines the
reference allocation; mandate constraints determine which departures are necessary.

## Overview

Use this method when a benchmark is the natural starting point but asset caps,
exposure limits, or other constraints prevent holding it directly. Covariance determines
the cost of each deviation, including the effect of correlations.

The objective contains no expected-return forecast. A feasible benchmark achieves zero
tracking error. With a positive-definite covariance it is the unique optimum; with a
singular covariance, other allocations can have the same modeled risk.
The convex quadratic formulation follows the framework in
[Boyd and Vandenberghe (2004)](#references).

<a id="inputs-units-and-alignment"></a>

## Inputs, notation, and assumptions

| Symbol or input | Meaning and contract |
|---|---|
| $n$ | Number of assets in the optimization universe. |
| $w$, $w_b$ | Portfolio and benchmark weight vectors of length $n$, in fractions of NAV. A weight of 0.35 means 35%. |
| $d$ | Active weights, defined as $w-w_b$. |
| $\Sigma$ | Symmetric $n \times n$ covariance of fractional asset returns, in a stated variance scale. |
| $\mathcal{F}$ | Feasible set defined by the applicable hard constraints. |
| `pd_covar` | Covariance DataFrame with unique asset labels; index and columns must match in the same order. |
| `benchmark_weights` | Finite Series for one date or a static rolling benchmark; a DataFrame supplies dated benchmark observations to the rolling function. |
| `weights_0` | Current holdings weights for turnover constraints and fallback selection in a single-date call. |
| `prices` | Total-return price panel for rolling alignment and drift; columns should match the covariance universe. |

Supply a positive-semidefinite covariance and explicitly identify whether it is per
observation or annualized. Neither the optimizer nor `qis.RiskModel` estimates, resamples,
or annualizes the supplied matrix. State simple versus log returns and estimation frequency
when constructing it from data. The fixed example below assumes annualized covariance
of fractional simple returns; it has no historical sample or estimated return series.

Provide complete, finite benchmark weights for the intended universe and align them
explicitly before solving. The single-date and rolling interfaces do not enforce missing
labels identically; see [input filtering](#input-filtering-and-benchmark-coverage).

For rolling use, supply covariance dates in increasing order: the optimizer iterates the
dictionary in insertion order. Use unique, increasing dates for prices, benchmarks, and
eligibility panels. Each covariance and benchmark observation must be available at its
decision date; the solver does not validate how the estimate was produced.

## Methodology

### Active risk and the optimization problem

Active variance and its square root, ex-ante tracking-error volatility, are

$$
d = w-w_b,\qquad
\operatorname{TE}(w;\Sigma)
= \sqrt{d^{\mathsf{T}}\Sigma d}.
$$

The optimization minimizes active variance over the feasible set:

$$
w^\star \in \operatorname*{arg\,min}_{w\in\mathcal{F}}
(w-w_b)^{\mathsf{T}}\Sigma(w-w_b).
$$

Minimizing the nonnegative variance also minimizes its square root. This is a covariance
distance: two deviations of the same weight can carry different risk. Volatility and
cross-asset covariance both affect where displaced benchmark weight is allocated.

For the long-only, fully invested example, the feasible set is

$$
\mathcal{F}
= \left\{w:\ \sum_{i=1}^{n}w_i=1,\quad 0\leq w_i\leq u_i\right\},
$$

where $u_i$ is the cap for asset $i$. Other exposure policies are available through
[Constraints](constraints.md). Linear constraints give a convex quadratic program;
compatible convex risk constraints extend the feasible set specification.

### Covariance scale

For a positive scalar $c$ and the same weights,

$$
\operatorname{TE}(w;c\Sigma)=\sqrt{c}\operatorname{TE}(w;\Sigma).
$$

Scaling the objective covariance alone preserves the mathematical optimum when the
feasible set is unchanged. Risk-based constraint limits must be converted consistently
if their covariance scale changes. Numerical floors and solver tolerances also limit
exact scaling invariance in software.

An annualized fractional covariance produces annualized fractional volatility:
0.0307 is about 3.07%, not 0.0307%. Multiplying monthly covariance by 12 is an
annualization assumption about return dependence; the optimizer does not make that
assumption for the caller.

<a id="minimal-offline-example"></a>

## Worked example

These are synthetic teaching inputs. The covariance, benchmark, and bounds preserve the
original three-asset example. A starts at a benchmark weight of 50% but has a 35% cap.

The following block runs offline on a core installation. It validates benchmark coverage,
requires an accepted and compliant solve, and delegates tracking-error measurement to
`qis.RiskModel` through `optimalportfolios.build_risk_model`.

```python
import numpy as np
import pandas as pd
import optimalportfolios as opt

assets = ["A", "B", "C"]
covar = pd.DataFrame(
    [[0.040, 0.006, 0.002],
     [0.006, 0.022, 0.003],
     [0.002, 0.003, 0.012]],
    index=assets,
    columns=assets,
)
benchmark = pd.Series([0.50, 0.30, 0.20], index=assets)
benchmark = benchmark.reindex(covar.columns)
if not np.isfinite(benchmark.to_numpy()).all():
    raise ValueError("The benchmark must cover every covariance asset.")

constraints = opt.Constraints(
    is_long_only=True,
    min_weights=pd.Series(0.0, index=assets),
    max_weights=pd.Series([0.35, 0.80, 0.80], index=assets),
)
weights, outcome = opt.wrapper_minimise_tracking_error(
    pd_covar=covar,
    benchmark_weights=benchmark,
    constraints=constraints,
    weights_0=benchmark,
)
if not (outcome.accepted and outcome.compliant):
    raise RuntimeError(
        f"Unusable solve: {outcome.status}; {outcome.reason}; "
        f"fallback={outcome.fallback_source}"
    )

date = pd.Timestamp("2024-01-31")  # Synthetic snapshot label, not a data cutoff.
risk_model = opt.build_risk_model({date: covar})
tracking_error = risk_model.compute_tre_at_date(
    benchmark_weights=benchmark,
    portfolio_weights=weights,
    date=date,
)
result = pd.DataFrame({
    "Benchmark": benchmark,
    "Portfolio": weights,
    "Active": weights - benchmark,
})
print(result.round(6))
print(f"Annualized tracking error: {tracking_error:.6%}")
```

The rounded weights are fractions of NAV:

| Asset | Benchmark | Portfolio | Active |
|---|---|---|---|
| A | 0.500000 | 0.350000 | -0.150000 |
| B | 0.300000 | 0.369643 | 0.069643 |
| C | 0.200000 | 0.280357 | 0.080357 |

Annualized tracking error: **3.072781%** in the verified run. Small differences in the
last digits are solver-dependent. A's cap binds, and the other assets absorb its
15-percentage-point reduction. Neither B nor C reaches its cap.

This covariance needs no eigenvalue flooring, so the original and solver covariance
agree to floating-point precision. `RiskModel` uses an exact covariance-grid date and
does not infer annualization from that date.

### Independent allocation check

Fix $w_A=0.35$ and write $w_B=0.30+x$, $w_C=0.35-x$. Then
$d=(-0.15,x,0.15-x)^{\mathsf{T}}$. Setting the directional derivative for a transfer
between B and C to zero gives

$$
(\Sigma d)_B-(\Sigma d)_C
=0.028x-0.00195=0,
\qquad x=\frac{0.00195}{0.028}\approx0.069642857.
$$

The gradient components for B and C agree. The component for A is smaller, so reducing
A from its cap and moving that weight to B or C increases the objective. Together with
positive definiteness and feasibility, these first-order conditions certify the
displayed allocation independently of CVXPY. The executable checks use this certificate;
tracking-error analytics remain in qis.

## Implementation in optimalportfolios

| Public entry point | Inputs and result |
|---|---|
| `wrapper_minimise_tracking_error` | Labeled covariance, benchmark, constraints and optional current weights; returns `(weights, outcome)`. |
| `cvx_minimise_tracking_error` | Array covariance and already aligned `Constraints` carrying benchmark weights; returns `OptimizationOutcome`. Prefer the wrapper for labeled data. |
| `rolling_minimise_tracking_error` | Price panel, covariance dictionary and static or dated benchmark; returns target weights on covariance dates. |
| `build_risk_model` | Dated asset covariance matrices or supported factorlasso covariance containers; returns `qis.RiskModel`. |

The wrapper returns weights in the original covariance order, with excluded assets
filled at zero. The outcome describes the retained universe and aligned constraints.
`accepted` records whether solver weights were used; `compliant` records whether
the returned weights pass the reported hard-constraint residuals.
Inspect both, along with `status`, `reason`, `fallback_source`, and `residuals_frame()`.

Source: [minimum-tracking-error solver](https://github.com/ArturSepp/OptimalPortfolios/blob/main/src/optimalportfolios/optimization/general/minimum_tracking_error.py)
and [risk-model adapter](https://github.com/ArturSepp/OptimalPortfolios/blob/main/src/optimalportfolios/covar_estimation/risk_model_adapter.py).
The [API reference](api.rst) provides generated signatures.

<a id="single-date-versus-rolling-use"></a>

### Single-date versus rolling use

A Series benchmark is reused unchanged at every date. A DataFrame benchmark is
forward-filled onto covariance dates and must have a complete observation no later
than the first decision. No later benchmark observation is borrowed to fill an earlier
date. Explicit NaNs at a selected observation are rejected.

Run this second block after the worked example. Constant synthetic prices isolate
benchmark selection from drift; all three benchmark allocations are feasible under
the new long-only, fully invested constraints.

```python
dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
prices = pd.DataFrame(100.0, index=dates, columns=assets)
benchmarks = pd.DataFrame(
    [[0.50, 0.30, 0.20], [0.20, 0.50, 0.30]],
    index=dates[[0, 2]],
    columns=assets,
)
rolling_weights = opt.rolling_minimise_tracking_error(
    prices=prices,
    constraints=opt.Constraints(is_long_only=True),
    benchmark_weights=benchmarks,
    covar_dict={snapshot: covar for snapshot in dates},
)
print(rolling_weights.round(6))
```

| Decision date | A | B | C |
|---|---|---|---|
| 2024-01-31 | 0.500000 | 0.300000 | 0.200000 |
| 2024-02-29 | 0.500000 | 0.300000 | 0.200000 |
| 2024-03-31 | 0.200000 | 0.500000 | 0.300000 |

February uses January's benchmark; March's observation first applies in March.
With changing prices, prior targets drift into current holdings by default before
the next solve. Missing price history can leave the previous target unchanged, while
missing individual price ratios are treated as flat; see the
[drift helper](https://github.com/ArturSepp/OptimalPortfolios/blob/main/src/optimalportfolios/utils/weights_drift.py).

The rolling function returns weights and discards the per-date outcome objects.
Use single-date calls when an explicit outcome per rebalance is required; a returned
rolling row alone does not certify solver acceptance. There is no prior portfolio
from an earlier solve on its first date. Turnover then has no baseline unless
current weights were already supplied on `Constraints.weights_0`; those stored
weights are not drifted before that first solve.

These functions construct targets. Use [rolling backtests](rolling_backtests.md)
and `qis.backtest_model_portfolio` for holdings simulation, with an explicit
implementation lag and transaction-cost convention.

### Reproduction and verification context

The Python blocks above are the canonical offline examples for this article.
[Their executable tests](../src/optimalportfolios/tests/minimum_tracking_error_documentation_test.py)
check the displayed results and an independent allocation reference:

```console
python -m pytest src/optimalportfolios/tests/minimum_tracking_error_documentation_test.py -q
```

On Windows, use the repository's external interpreter and generated-state setup in
[AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md).
The examples were verified on 2026-09-13 against a source export of the local
OptimalPortfolios 7.6.0 working tree, with installed qis 5.26.0, CVXPY 1.9.2 and
CLARABEL 0.11.1. This identifies the local verification environment rather than
certifying a published artifact or locked CI run.

The existing [one-step and rolling example](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/solvers/minimum_tracking_error.py)
demonstrates a larger workflow and downloads market data; it is separate from the
offline examples above.

<a id="constraints-and-failure-modes"></a>

## Interpretation and limitations

### Input filtering and benchmark coverage

The wrapper removes assets whose covariance diagonal is NaN, zero, or negative.
If supplied, an inclusion indicator must be close to one to retain an asset.
After filtering, the retained covariance must be finite and symmetric. An empty
retained universe cannot be solved.

Filtering also removes the corresponding benchmark components without renormalizing
them. The objective then uses the retained covariance and benchmark; cross-covariances
to excluded benchmark holdings no longer enter the solve. Consequently, this can
differ from minimizing full-universe tracking error with excluded portfolio weights
fixed at zero. Assess the final portfolio against the intended full benchmark using
a complete risk model.

The current single-date wrapper fills a missing benchmark **label** with zero during
alignment; an explicit NaN on a retained asset raises an error. The rolling interface
instead rejects missing benchmark columns relative to the price panel. These are
different input behaviors, not interchangeable validation guarantees. Align and
validate coverage explicitly, as the worked example does.

A true zero-variance cash instrument is removed by this wrapper. Treat that as an input
policy limitation when modeling cash, not a reason to assign an arbitrary variance.

### Stabilization, acceptance, and fallback

By default, `OptimiserConfig.factorize_covar=True` uses a controlled eigendecomposition.
The current absolute eigenvalue floor is `1e-10` in the input variance units.
Materially negative eigenvalues raise an error; tiny negative or positive eigenvalues
below the floor are raised. Inspect `outcome.covar_factorization` for adjustments.
Its `covar` is the matrix used for optimization and constraint validation; use that
matrix with the aligned asset labels when auditing stabilized risk through `RiskModel`.
Changing covariance units can change which eigenvalues reach an absolute floor.

A solver error, infeasibility, or failed acceptance check can return fallback weights.
The finite-candidate order is current weights, benchmark, then zeros; availability
depends on the aligned inputs. A fallback is not guaranteed to satisfy the mandate,
and a feasible fallback is not proof of optimality. The worked example stops when
either acceptance or compliance fails. The complete tolerance and constraint policies
remain in [Constraints](constraints.md).

Minimum modeled tracking error does not establish realized tracking performance,
expected outperformance, or robustness to covariance estimation error. Ex-ante risk
depends on the chosen benchmark, universe, and covariance; realized tracking error
requires a separate return-history analysis.

## See also

- [Constraints and solver contracts](constraints.md)
- [Rolling backtests](rolling_backtests.md)
- [Covariance estimators](covariance_estimators.md)
- [Risk budgeting](risk_budgeting.md)
- [API reference](api.rst)
- [qis RiskModel implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/risk_model.py)

## References

- Boyd, S., and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
  [Authors' book page](https://web.stanford.edu/~boyd/cvxbook/). Chapters 4–5 cover
  convex quadratic optimization and optimality conditions.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
