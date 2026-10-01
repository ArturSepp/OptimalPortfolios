---
myst:
  html_meta:
    description: >-
      Fixed-core overlay allocation with a named linear floor: Sharpe scaling, exposure
      budgets, the Bear-regime coverage floor, reproducible examples, QIS risk checks and
      solver limitations.
---

# Overlay optimisation with a fixed core and linear side constraints

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-12](https://github.com/ArturSepp/OptimalPortfolios/commit/a5d635895e0e05cfa64966e1c5f77ff9ab255afe)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

A fixed-core overlay allocation holds the core exposure constant while choosing an additional
sleeve under a common objective and constraints. A linear floor imposes a minimum value on a
supplied weighted characteristic, such as a scenario-return contribution. This article describes
the fixed-exposure maximum-Sharpe pattern implemented through `Constraints` and a named
`LinearConstraints` row, with risk measurement delegated to
[QIS](https://github.com/ArturSepp/QuantInvestStrats). With the core's Bear-regime loss as the
characteristic, it is the [coverage-floor allocation](#the-coverage-floor) of Sepp and
Kastenholz (2026).

## Overview

### The problem

The core has weight 1.0 relative to portfolio capital; long-only overlays sum to a specified
budget $W$. With $W=1$, total model exposure is 2.0. This is the original "return stacking"
example: a core plus an additional overlay sleeve. Normalising those weights to sum to one
would change both the core mandate and the floor.

The objective uses supplied expected excess returns and covariance. The side condition is
linear in weights. A coefficient constructed from volatility times a bear-regime score is a
**proxy supplied by the caller**: its weighted sum is not automatically portfolio expected
shortfall, a return quantile, or a guarantee against losses.

The [constraint guide](constraints.md) owns general constraint semantics. Here the scope is
fixed core, fixed exposure, asset bounds, compatible sleeve budgets and one linear floor.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Arithmetic excess-return exposures on one capital base, linear in the weights; the solver converts no log returns |
| Estimation grid | None; `create_synthetic_inputs` supplies fixed annual moments, with no sample, window or regime selection |
| Rebalancing grid | None; one static solve per floor, and the risk-model date 2024-12-31 is a synthetic key |
| Covariance units | Annual decimal-return variance of a one-factor model whose factor is the core (volatility 0.10), with an idiosyncratic variance floor of $10^{-6}$ |
| Expected returns | Supplied annual expected excess returns `means`, Sharpe ratio times volatility, define the objective; the floor coefficients enter separately as `asset_returns` |
| Weight state | Exposures per unit of capital: core fixed at 1.0, long-only overlays summing to 1.0, total 2.0; `weights_0` appears only as the prior of the infeasibility probe |
| Solver | CVXPY with CLARABEL on the fixed-exposure Charnes–Cooper path of `cvx_maximize_portfolio_sharpe`; an exposure band routes to SciPy SLSQP |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning and convention |
|---|---|
| $w$ | $n$ dimensionless exposures in a common capital unit. |
| $c$ | Index of the core; $w_c=1$. |
| $W$ | Nonnegative overlay budget; the example uses 1.0. |
| $E$ | Fixed total exposure $1+W$, strictly positive. |
| $e$ | Vector of $n$ ones. |
| $\mu$ | Supplied expected excess returns, annual decimal units in the example. |
| $\Sigma$ | Symmetric positive-definite covariance, annual decimal-return variance here. |
| $a$ | Fixed linear coefficients, with a common horizon and unit across assets. |
| $b_0$ | Floor in the same units as $a^\top w$; distinct from expected return $\mu^\top w$. |
| $\theta$ | Coverage floor: the fraction of the core's Bear-regime loss $-a_c$ that the overlays must offset. |
| $y,k$ | Transformed exposures and positive scale; recovered weights are $w=y/k$. |

For arithmetic excess-return exposures measured on the same capital base, the one-period model is

$$
r_p^{e}=w^\top r^{e}.
$$

Financing and derivative-payoff conventions must be reflected in the supplied return exposures.
The solver does not create a funding leg, deduct a cash rate, infer margin requirements, or convert
log returns into arithmetic returns. A sum of 2.0 is an exposure budget, not proof that a trade is
funded. All covariance, expected-return and side-condition inputs must refer to the same ordered
universe. The numerical entry point does not align pandas labels or clean missing inputs.

The example uses fixed annual moments and a synthetic characteristic; it estimates no sample,
selects no bear regime, and needs no annualisation factor. For estimated inputs, specify the
return convention, observation frequency, estimation window, annualisation and regime-selection
rule. Inputs used for a decision at $t$ must be available at $t$; this single-date calculation
does not implement a backtest or establish point-in-time data availability.

## Methodology

The ratio formulation and its positive-return normalization follow standard portfolio
optimization; see Cornuéjols and Tütüncü's
[author-hosted draft, section 8.2](https://www.andrew.cmu.edu/user/gc0v/webpub/OptFinFirstEdition2006.pdf).
The original Charnes–Cooper paper gives the related
[homogeneous-variable transformation](https://iiif.library.cmu.edu/file/Cooper_box00010_fld00009_bdl0001_doc0001/Cooper_box00010_fld00009_bdl0001_doc0001.pdf)
for linear-fractional programs, and Schaible (1974) extends it to the quadratic-fractional
programs to which a Sharpe ratio belongs. The floor encoding below follows directly from fixed
exposure.

The target problem, with optional per-asset bounds added as needed, is

$$
\begin{aligned}
\max_w\quad & \frac{\mu^\top w}{\sqrt{w^\top\Sigma w}}\\
\text{subject to}\quad & w_c=1,\quad w_i\geq0,\quad \sum_{i\ne c}w_i=W,\\
& a^\top w\geq b_0.
\end{aligned}
$$

A positive attainable numerator and nonzero risk are required for the positive-scale
reformulation used here. The synthetic inputs satisfy both. If every feasible expected excess
return is nonpositive, the normalized positive-return problem need not represent the best
negative Sharpe ratio.

### The encoding

The fixed-exposure implementation chooses the positive normalization constant $E$. Its
transformed objective and recovery are

$$
\min_{y,k} y^\top\Sigma y,\qquad
\mu^\top y=E,\qquad y=kw,\qquad w=\frac{y}{k}.
$$

The source imposes $k\geq0$. In this bounded long-only example, $k=0$ would force $y=0$,
contradicting $\mu^\top y=E \gt 0$, so a feasible transformed solution has $k \gt 0$.

**Fixed core and sleeve budget.** Core minimum and maximum weights both equal 1.0; total
minimum and maximum exposures both equal $E$. The backend scales these rows correctly:

$$
y_c=k,\qquad e^\top y=kE,\qquad
k\ell_i\leq y_i\leq k u_i.
$$

Here $\ell_i,u_i$ are asset lower and upper bounds. Multiple sleeves can use
`group_lower_upper_constraints`, a `GroupLowerUpperConstraints` with disjoint membership columns
and equal group lower/upper budgets. Keep total exposure fixed too: the current Sharpe entry
point dispatches by equality of `min_exposure` and `max_exposure`, not by inferring an equality
from group rows.

**Linear floor.** Supply the floor as a named row of
[`LinearConstraints`](constraints.md#named-signed-linear-rows) with loadings $a$ and lower bound
$b_0$. The CVXPY compiler multiplies the bound by the scale, and for a positive scale the
transformed row is the original one:

$$
a^\top y\geq k b_0
\iff a^\top w\geq b_0
\quad\text{when }k \gt 0.
$$

`asset_returns=a` with `target_return=b0` compiles the same row. When exposure is a band
rather than fixed, SciPy enforces the row directly on weights.

**Homogeneous cross-check.** With fixed total exposure the floor also has an exact encoding with
a zero right side, which needs no scaled bound. Results computed before the return-floor
correction recorded in the
[changelog](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CHANGELOG.md) used it, and
the canonical script keeps it as an independent reference:

$$
\begin{aligned}
\widetilde a &= a-\frac{b_0}{E}e,\\
a^\top w-b_0 &= \widetilde a^\top w
\quad\text{when }e^\top w=E,\\
a^\top w\geq b_0
&\iff \widetilde a^\top y\geq0
\quad\text{when }k>0.
\end{aligned}
$$

For this encoding, set `asset_returns=a - b0 / E` and `target_return=0.0`. Subtract from
**all** coefficients, including the core. A zero floor is the special case $\widetilde a=a$.
Changing $E$ requires recomputing the coefficients.

The nonzero floor has not disappeared economically: it is encoded in $\widetilde a$.
The expected-return vector passed as `means` still defines the objective; `asset_returns`
is a separate constraint vector in this pattern.

### The coverage floor

Sepp and Kastenholz (2026), Section II, allocate overlays to a fixed core with program (11):
they maximise the Sharpe ratio of the core plus a long-only overlay sleeve subject to a floor on
the portfolio's Bear-regime contribution. The Bear regime is the set of periods in which the
core's own return lies below its 16% quantile (their Definition 1). Let $a_i$ be the annual
Bear-regime return contribution of asset $i$, its volatility times its Bear-regime Sharpe
contribution on the core's regimes, and let $a_c \lt 0$ be the core's own Bear-regime loss. With
the core held at one, the floor of program (11) is

$$
a_c+\sum_{i\ne c}a_iw_i\geq(1-\theta) a_c ,
$$

the linear floor $a^\top w\geq b_0$ with $b_0=(1-\theta) a_c$. A coverage $\theta=0$ lets the
overlays add nothing to the core's Bear-regime loss, and $\theta=1$ requires them to offset it
in full. An allocation covers $1-a^\top w/a_c$ of the loss, and the reachability bound under
[Interpretation and limitations](#reachability-and-fallback) gives the largest coverage,
$\theta_{\max}=-W\max_{i\ne c}a_i/a_c$, when no other constraint binds.

Because the regimes are fixed by the core, the contributions add across overlays, their
Proposition 4, which keeps the floor linear in the weights. Estimating $a$, the regime
classification, the Bear-regime Sharpe contributions and the regime-mixture covariance of their
equation (10), belongs to
[qis](https://quantinveststrats.readthedocs.io/en/latest/convexity_premium.html); this page
consumes the coefficients. The paper's
[synthetic companion](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/smart_diversification_joim_2026)
runs the whole pipeline, the
[smart diversification case study](app_smart_diversification_overlays.md) applies this floor to
a simulated core and four overlays, and the [research papers](research_papers.md) page records
the paper.

## Worked example

The Python blocks on this page run in
order and need no market data or random draw. They are excerpts of the canonical script
[`examples/docs/overlay_tail_floor.py`](../examples/docs/overlay_tail_floor.py), which runs them
and asserts every number and property on this page against a reference computed a different
way. The blocks reuse the input factory and helper of
[the original synthetic example](../examples/solvers/overlay_tail_floor.py), with one core and
four overlays, a one-factor covariance and fixed characteristic coefficients, so they run from
a source checkout:

```console
python -m examples.docs.overlay_tail_floor
```

```python
from dataclasses import replace
import numpy as np
import pandas as pd
import optimalportfolios as opt
from examples.solvers.overlay_tail_floor import (
    create_synthetic_inputs, solve_overlay_tail_floor,
)

means, covar, a = create_synthetic_inputs()
tickers = covar.index
assert tickers.equals(covar.columns)
assert means.index.equals(tickers) and a.index.equals(tickers)
assert np.isfinite(covar.to_numpy()).all()
assert np.isfinite(means).all() and np.isfinite(a).all()
assert np.linalg.eigvalsh(covar).min() > 0.0
inputs = pd.DataFrame({"Expected excess return": means, "Linear coefficient": a})
print(inputs.round(4))
```

| Asset | Expected excess return | Linear coefficient |
|---|---|---|
| Core | 0.0600 | -0.0800 |
| Defensive A | 0.0525 | 0.0900 |
| Defensive B | 0.0360 | 0.0420 |
| Carry C | 0.0900 | -0.0450 |
| Carry D | 0.0800 | -0.0240 |

The factory constructs the characteristic from its nominal volatilities and synthetic bear
scores. Its covariance includes an idiosyncratic variance floor of $10^{-6}$; the core covariance
diagonal is therefore 0.010001 rather than exactly 0.010000. Use the returned covariance for risk.

The following calls preserve the original no-floor and zero-floor cases and add a positive
floor of 0.005 in the characteristic's units, each as a named row `floor`. Every case retains
its complete outcome:

```python
overlay_budget = 1.0
total_exposure = 1.0 + overlay_budget
min_weights = pd.Series(0.0, index=tickers)
min_weights["Core"] = 1.0
max_weights = pd.Series(overlay_budget, index=tickers)
max_weights["Core"] = 1.0
base = opt.Constraints(
    is_long_only=True, min_weights=min_weights, max_weights=max_weights,
    min_exposure=total_exposure, max_exposure=total_exposure,
)

floors = {"No floor": None, "Zero floor": 0.0, "Floor 0.005": 0.005}
specifications = {}
outcomes = {}
for label, floor in floors.items():
    spec = base if floor is None else replace(
        base, linear_constraints=opt.LinearConstraints(
            loadings=a.to_frame("floor"), lower=pd.Series({"floor": floor}),
        ),
    )
    specifications[label] = spec
    outcomes[label] = opt.cvx_maximize_portfolio_sharpe(
        covar=covar.to_numpy(), means=means.to_numpy(), constraints=spec,
        context=f"overlay article: {label}",
    )
    assert outcomes[label].accepted and outcomes[label].compliant

allocation = pd.DataFrame(
    {label: outcome.weights for label, outcome in outcomes.items()}, index=tickers,
)
print(allocation.round(6))
```

| Asset | No floor | Zero floor | Floor 0.005 |
|---|---|---|---|
| Core | 1.000000 | 1.000000 | 1.000000 |
| Defensive A | 0.283145 | 0.791667 | 0.895833 |
| Defensive B | 0.213616 | 0.208333 | 0.104167 |
| Carry C | 0.128974 | 0.000000 | 0.000000 |
| Carry D | 0.374264 | 0.000000 | 0.000000 |

Values are rounded from the computed weights. Tiny solver-scale exposures display as zero.
The positive floor concentrates more exposure in Defensive A. This is a consequence of these
synthetic coefficients and covariance, not a finding about investable strategies. The
homogeneous cross-check reproduces both floored allocations.

For risk, build the canonical `qis.RiskModel` through `optimalportfolios.build_risk_model`.
Its `compute_tre_at_date` tracking error against a zero exposure vector equals portfolio
volatility. The reported model excess Sharpe is the supplied expected excess return divided by
this risk; it is not an ex-post QIS performance statistic.

```python
risk_date = pd.Timestamp("2024-12-31")  # Synthetic risk-model key, not a data cutoff.
risk_model = opt.build_risk_model({risk_date: covar})
zero_benchmark = pd.Series(0.0, index=tickers)
metrics = {}
for label in floors:
    w = allocation[label]
    volatility = risk_model.compute_tre_at_date(
        benchmark_weights=zero_benchmark, portfolio_weights=w, date=risk_date,
    )
    expected_excess = float(means @ w)
    metrics[label] = {
        "Expected excess": expected_excess,
        "Volatility": volatility,
        "Model excess Sharpe": expected_excess / volatility,
        "Linear contribution": float(a @ w),
    }
summary = pd.DataFrame.from_dict(metrics, orient="index")
print(summary.round(6))
```

| Case | Expected excess | Volatility | Model excess Sharpe | Linear contribution |
|---|---|---|---|---|
| No floor | 0.124104 | 0.123441 | 1.005372 | -0.060331 |
| Zero floor | 0.109063 | 0.139076 | 0.784193 | 0.000000 |
| Floor 0.005 | 0.110781 | 0.150114 | 0.737981 | 0.005000 |

The two constrained solutions satisfy the floor at equality within numerical tolerance.
For these inputs only, Carry C and Carry D are zero and the overlay budget is split between
the defensive assets. The floor then gives
$w_{\mathrm{Defensive\ A}}=(b_0+0.038)/0.048$, which independently reproduces their displayed
weights. Feasibility alone does not prove optimality; the canonical script also verifies
first-order conditions against the unused overlays.

Solving the same problem for 90 floors, 0.001 apart from −0.08 to 0.009, traces the path
between these cases. Volatility dips slightly and then rises from 12.3% to 16.0%; the expected
excess return falls while carry leaves the sleeve and recovers as Defensive A replaces
Defensive B.

![Left: stacked overlay weights of Defensive A, Defensive B, Carry C and Carry D against the
floor from −0.08 to 0.009; the sleeve always sums to 1 and is unchanged until the floor passes
the no-floor contribution of −0.060, after which Carry C shrinks to zero, then Carry D, while
Defensive A grows to almost the whole sleeve. Right: volatility and expected excess return in
annual percent; both are flat while the floor is loose, volatility then dips and rises to 16%,
and expected excess return falls to about 10.7% before recovering to about 11.2%.](images/overlay_floor_allocation.png)

*Figure: the overlay sleeve and the portfolio's model risk and return as the floor tightens, for
the inputs of the worked example. Drawn by the `exhibit` function of the canonical script; the
[analytics gallery](analytics_gallery.md) lists its provenance.*

> **Insight.** A floor changes nothing until it exceeds the no-floor allocation's own
> contribution, −0.0603 here; every tighter floor binds at equality. Carry C leaves the sleeve
> first, then Carry D, and at the reachable maximum of 0.01 the whole sleeve must sit in
> Defensive A, the overlay with the largest coefficient.

Read as Bear-regime contributions, the synthetic characteristic gives the core a Bear-regime
loss of −0.08. The no-floor allocation then covers 24.6% of that loss, the zero floor
100% and the floor 0.005 106.25%; the reachable maximum 0.01 is a coverage of 112.5%. A
[coverage floor](#the-coverage-floor) of 40% is the floor $b_0=0.6\times(-0.08)=-0.048$:

```python
coverage = 0.40
bear_coverage = opt.LinearConstraints(
    loadings=a.to_frame("bear_coverage"),
    lower=pd.Series({"bear_coverage": (1.0 - coverage) * a["Core"]}),
)
covered = opt.cvx_maximize_portfolio_sharpe(
    covar=covar.to_numpy(), means=means.to_numpy(),
    constraints=replace(base, linear_constraints=bear_coverage),
    context="overlay article: 40% coverage",
)
assert covered.accepted and covered.compliant
covered_weights = pd.Series(covered.weights, index=tickers)
realised_coverage = 1.0 - float(a @ covered_weights) / a["Core"]
maximum_coverage = -overlay_budget * a.drop("Core").max() / a["Core"]
print(covered_weights.round(6))
print(round(realised_coverage, 6), round(maximum_coverage, 6))
# 0.4 1.125
```

| Asset | 40% coverage |
|---|---|
| Core | 1.000000 |
| Defensive A | 0.350787 |
| Defensive B | 0.260699 |
| Carry C | 0.056942 |
| Carry D | 0.331573 |

The floor binds, so the allocation covers exactly 40%. It keeps all four overlays, moves weight
from the carry overlays to the defensive ones, and is the allocation of the figure's sweep at
−0.048. Its model volatility, 11.98%, is below the no-floor 12.34%, while its model excess
Sharpe ratio falls from 1.005 to 0.997.

## Implementation in optimalportfolios

### Verification

The [Sharpe implementation](../src/optimalportfolios/optimization/general/max_sharpe.py)
returns `OptimizationOutcome` from `cvx_maximize_portfolio_sharpe`. Check `accepted`, then the
stored hard residuals and the floor in its own units:

```python
selected = outcomes["Floor 0.005"]
selected_weights = allocation["Floor 0.005"]
assert float(a @ selected_weights) >= 0.005 - 1e-6
assert abs(selected_weights["Core"] - 1.0) < 1e-6
assert abs(selected_weights.drop("Core").sum() - overlay_budget) < 1e-6
residuals = selected.residuals_frame()
floor_row = residuals[residuals["constraint_type"] == "linear"].iloc[0]
assert floor_row["name"] == "floor" and floor_row["lower"] == 0.005
assert abs(floor_row["actual"] - float(a @ selected_weights)) < 1e-12
hard_breaches = [r for r in selected.constraint_residuals if r.hard and not r.passed]
assert not hard_breaches
```

`compliant` evaluates stored hard residuals with their tolerances. The named row is stored as a
`linear` residual called `floor`, whose `actual` is $a^\top w$ and whose `lower` is $b_0$, both
in the characteristic's units. With the homogeneous cross-check the `target_return` residual
instead measures $\widetilde a^\top w\geq0$, which equals the original margin only when the
exposure equality holds; check the budget and the floor together there. Compliance covers the
encoded specification and does not establish the quality of the proxy.

The labelled wrapper `wrapper_maximize_portfolio_sharpe` returns a weight Series and an
outcome. With this complete input panel and an explicit `OptimiserConfig` it agrees with the
raw call and the original example helper:

```python
config = opt.OptimiserConfig(apply_total_to_good_ratio=False)
labelled_weights, labelled_outcome = opt.wrapper_maximize_portfolio_sharpe(
    pd_covar=covar, means=means, constraints=specifications["Zero floor"],
    optimiser_config=config, context="overlay article: labelled call",
)
assert labelled_outcome.accepted and labelled_outcome.compliant
legacy_weights = solve_overlay_tail_floor(
    means=means, covar=covar, bear_contributions=a, floor_b0=0.0,
)
assert np.allclose(labelled_weights, allocation["Zero floor"], atol=1e-6)
assert np.allclose(legacy_weights, allocation["Zero floor"], atol=1e-6)
```

The original `solve_overlay_tail_floor` helper remains available in the repository example;
it extracts only `.weights` and returns a Series. Use a direct outcome-returning call when
acceptance and fallback diagnostics must be retained. `examples/` is not part of the installed
wheel, so the factory and helper imports above require a checkout.

The raw call is positional internally: covariance rows/columns, means, coefficient vectors
and bounds must have identical asset order. The labelled wrapper filters unusable assets and
aligns supported fields; it can change the effective universe. Reject missing mandatory-core
inputs before solving. Reassess the floor and budgets after filtering rather than assuming the
same mandate survived. The wrapper's default configuration enables automatic universe-ratio
bound rescaling; the example disables it explicitly with `apply_total_to_good_ratio=False`.

Source owners: [constraint compilation](../src/optimalportfolios/optimization/constraints/backends.py),
[residual evaluation](../src/optimalportfolios/optimization/constraints/analytics.py),
[solver outcomes](../src/optimalportfolios/optimization/solver_diagnostics.py), and
[risk-model adapter](../src/optimalportfolios/covar_estimation/risk_model_adapter.py).

The [canonical script](../examples/docs/overlay_tail_floor.py) runs the eight blocks with
sockets blocked. It checks the allocations against a linear-algebra solution, a first-order
optimality certificate and the homogeneous cross-check, the coverage case against the figure's
sweep, the risk table against the quadratic form, the compiled CVXPY and SciPy rows at known
points, several sleeves as group rows, the fallback and the equivalent floor formulations. The
test suite runs it, and so does the offline examples lane of CI.

## Interpretation and limitations

### Reachability and fallback

Without further sleeve caps or other constraints, the largest linear contribution for a
long-only sleeve with budget $W$ is

$$
a_c+W\max_{i\ne c}a_i.
$$

Thus $b_0\leq a_c+W\max_{i\ne c}a_i$ is necessary. It is sufficient for the **linear floor alone**
when all the budget can be assigned to the maximizing overlay. Extra caps, group requirements
and objective-domain conditions can prevent a feasible solve even below that bound.

Here the bound is $-0.08+1(0.09)=0.01$, a coverage of 112.5%. A floor of 0.02 is impossible:

```python
maximum_linear = float(a["Core"] + overlay_budget * a.drop("Core").max())
impossible_floor = 0.02
assert impossible_floor > maximum_linear
impossible = replace(
    base, linear_constraints=opt.LinearConstraints(
        loadings=a.to_frame("floor"), lower=pd.Series({"floor": impossible_floor}),
    ),
    weights_0=allocation["No floor"],
)
rejected = opt.cvx_maximize_portfolio_sharpe(
    covar=covar.to_numpy(), means=means.to_numpy(), constraints=impossible,
    context="overlay article: deliberate infeasibility",
)
print(rejected.status, rejected.accepted, rejected.fallback_source, rejected.compliant)
# infeasible False weights_0 False
```

The constructor accepts this row. Its box check lets every overlay reach its own cap of 1.0,
where the row could reach $-0.08+0.09+0.042=0.052$; only the sleeve budget makes 0.02
unreachable, and only the solver detects it.

The fallback keeps the prior no-floor allocation and violates the impossible floor. The shared
fallback preference is finite prior weights, then finite benchmark weights, then zeros.
It does not project a fallback into the feasible set. Zero weights would also violate the fixed
core and total-exposure mandates. Solver status, acceptance, fallback source and compliance are
separate facts; an application must decide whether to trade, retry or skip.

### Backend and modeling limits

The fixed-exposure path does not automatically homogenize every possible side condition.
Do not assume that a turnover, volatility or benchmark-relative row has been transformed
correctly merely because it belongs to `Constraints`. This article verifies only the stated
asset, exposure, sleeve and linear-floor pattern.

An exposure band routes the entry point to SLSQP. Its compiler includes exposure,
asset bounds, group allocations, return floors and named signed linear rows. It optimises
the ratio directly, so solver convergence does not certify a global optimum.

The single-row `target_return` form and the homogeneous cross-check reproduce the named floor
of 0.005, and SLSQP enforces the named zero floor under an exposure band:

```python
direct_floor = replace(base, asset_returns=a, target_return=0.005)
shifted_floor = replace(base, asset_returns=a - 0.005 / total_exposure, target_return=0.0)
banded = replace(specifications["Zero floor"], min_exposure=1.8)
cross_checks = {}
for label, spec in {"Direct floor": direct_floor, "Shifted floor": shifted_floor,
                    "Exposure band": banded}.items():
    candidate = opt.cvx_maximize_portfolio_sharpe(
        covar=covar.to_numpy(), means=means.to_numpy(), constraints=spec,
        context=f"overlay article: floor check {label}",
    )
    cross_checks[label] = candidate
    print(label, candidate.solver, candidate.status, candidate.accepted)
    assert candidate.accepted and candidate.compliant
# Direct floor CLARABEL optimal True
# Shifted floor CLARABEL optimal True
# Exposure band SLSQP optimal True
for label in ("Direct floor", "Shifted floor"):
    np.testing.assert_allclose(cross_checks[label].weights, allocation["Floor 0.005"],
                               atol=1e-8)
```

Other risk and trading rows remain subject to the transformation limitations already stated.

> **Pitfall.** Before the return-floor correction recorded in the
> [changelog](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CHANGELOG.md), the
> compiler did not scale the return-floor bound on this path, and SLSQP omitted it. For a
> negative bound and a scale greater than one, the transformed floor was too tight: a feasible
> but suboptimal allocation could still report accepted and compliant. Post-solve feasibility
> checks cannot certify that the intended optimisation problem was formulated. Compare
> objective values and allocations against an independent formulation.

A linear characteristic floor does not control the full loss distribution or nonlinear option
payoffs. Use a common scenario/regime definition to make coefficients additive. If the tail
sample is chosen separately for each asset, their individual tail statistics generally do not
aggregate into a portfolio tail statistic.

## See also

- [Optimization module](optimization_module_readme.md): objectives, configuration and outcomes.
- [Portfolio constraints](constraints.md): canonical compiler, alignment and residual contracts.
- [Rolling backtests](rolling_backtests.md): decision and execution timing.
- [Turnover and transaction costs](turnover_and_transaction_costs.md): trades, budgets and costs.
- [Minimum tracking error](minimum_tracking_error.md): benchmark-relative risk and QIS integration.
- [Smart diversification with portfolio overlays](app_smart_diversification_overlays.md): the
  coverage floor applied to a simulated 60/40 core and four overlays.
- [Research papers](research_papers.md): the convexity-premium paper and its synthetic companion.
- [qis: the convexity premium and smart diversification](https://quantinveststrats.readthedocs.io/en/latest/convexity_premium.html):
  the regime contributions that supply a coverage floor's coefficients.

## References

- Charnes, A. and Cooper, W. W. (1962). *Programming with Linear Fractional Functionals*.
  Naval Research Logistics Quarterly, 9(3–4), 181–186.
  [Primary paper in the CMU archive](https://iiif.library.cmu.edu/file/Cooper_box00010_fld00009_bdl0001_doc0001/Cooper_box00010_fld00009_bdl0001_doc0001.pdf).
- Cornuéjols, G. and Tütüncü, R. (2007). *Optimization Methods in Finance*.
  Cambridge University Press.
  [Publisher record](https://www.cambridge.org/core/books/optimization-methods-in-finance/FAE3FDF1D69C6B0704EEC81B617B706A).
  [Author-hosted January 2006 draft](https://www.andrew.cmu.edu/user/gc0v/webpub/OptFinFirstEdition2006.pdf),
  section 8.2, "Maximizing the Sharpe Ratio."
- Schaible, S. (1974). *Parameter-free convex equivalent and dual programs of fractional
  programming problems*. Zeitschrift für Operations Research, 18(5), 187–196.
  [DOI 10.1007/BF02026600](https://doi.org/10.1007/BF02026600).
- Sepp, A. and Kastenholz, M. (2026). *The Convexity Premium of Portfolio Overlays*. Journal of
  Investment Management, forthcoming. Section II, program (11): the coverage floor; Proposition 4:
  the aggregation of Bear-regime contributions.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
