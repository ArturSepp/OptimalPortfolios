---
myst:
  html_meta:
    description: >-
      Fixed-core overlay allocation with a homogeneous linear floor: Sharpe scaling,
      exposure budgets, reproducible examples, QIS risk checks and solver limitations.
---

# Overlay optimisation with a fixed core and linear side constraints

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-12](https://github.com/ArturSepp/OptimalPortfolios/commit/a5d635895e0e05cfa64966e1c5f77ff9ab255afe)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

A fixed-core overlay allocation holds the core exposure constant while choosing an additional
sleeve under a common objective and constraints. A linear floor imposes a minimum value on a
supplied weighted characteristic, such as a scenario-return contribution. This article describes
the fixed-exposure maximum-Sharpe pattern implemented through `Constraints`, with risk
measurement delegated to [QIS](https://github.com/ArturSepp/QuantInvestStrats).

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

| Symbol | Meaning and convention |
|---|---|
| $w$ | $n$ dimensionless exposures in a common capital unit. |
| $c$ | Index of the core; $w_c=1$. |
| $W$ | Nonnegative overlay budget; the example uses 1.0. |
| $E$ | Fixed total exposure $1+W$, strictly positive. |
| $e$ | Length-$n$ vector of ones. |
| $\mu$ | Supplied expected excess returns, annual decimal units in the example. |
| $\Sigma$ | Symmetric positive-definite covariance, annual decimal-return variance here. |
| $a$ | Fixed linear coefficients, with a common horizon and unit across assets. |
| $b_0$ | Floor in the same units as $a^\top w$; distinct from expected return $\mu^\top w$. |
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
for linear-fractional programs. The floor encoding below follows directly from fixed exposure.

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
\min_{y,k}\;y^\top\Sigma y,\qquad
\mu^\top y=E,\qquad y=kw,\qquad w=\frac{y}{k}.
$$

The source imposes $k\geq0$. In this bounded long-only example, $k=0$ would force $y=0$,
contradicting $\mu^\top y=E>0$, so a feasible transformed solution has $k>0$.

**Fixed core and sleeve budget.** Core minimum and maximum weights both equal 1.0; total
minimum and maximum exposures both equal $E$. The backend scales these rows correctly:

$$
y_c=k,\qquad e^\top y=kE,\qquad
k\,\ell_i\leq y_i\leq k\,u_i.
$$

Here $\ell_i,u_i$ are asset lower and upper bounds. Multiple sleeves can use
`group_lower_upper_constraints` with disjoint membership columns and equal group lower/upper
budgets. Keep total exposure fixed too: the current Sharpe entry point dispatches by equality
of `min_exposure` and `max_exposure`, not by inferring an equality from group rows.

**Linear floor.** The current CVXPY return-row compiler emits
`asset_returns @ y >= target_return` without multiplying the right side by $k$.
A direct nonzero `target_return=b0` therefore generally represents the wrong transformed
constraint. Fixed total exposure gives an exact homogeneous encoding:

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

Set `asset_returns=a - b0 / E` and `target_return=0.0`. Subtract from **all** coefficients,
including the core. A zero floor is the special case $\widetilde a=a$.
Changing $E$ requires recomputing the coefficients.

The nonzero floor has not disappeared economically: it is encoded in $\widetilde a$.
The expected-return vector passed as `means` still defines the objective; `asset_returns`
is a separate constraint vector in this pattern.

## Worked example

Run these Python blocks in order from a source checkout. They reuse the unchanged input factory
in [the original synthetic example](../examples/solvers/overlay_tail_floor.py), with one core
and four overlays, a one-factor covariance and fixed characteristic coefficients. No market
data or random draws are used.

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
floor of 0.005 in the characteristic's units. Every case retains its complete outcome:

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
        base, asset_returns=a - floor / total_exposure, target_return=0.0,
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
synthetic coefficients and covariance, not a finding about investable strategies.

For risk, build the canonical `qis.RiskModel` through `optimalportfolios.build_risk_model`.
Tracking error against a zero exposure vector equals portfolio volatility. The reported
model excess Sharpe is the supplied expected excess return divided by this risk; it is not
an ex-post QIS performance statistic.

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
weights. Feasibility alone does not prove optimality; the article tests also verify first-order
conditions against the unused overlays.

## Implementation in optimalportfolios

### Verification

The [Sharpe implementation](../src/optimalportfolios/optimization/general/max_sharpe.py)
returns `OptimizationOutcome` from `cvx_maximize_portfolio_sharpe`. Check `accepted`, then the
stored hard residuals and the **original** unshifted floor in its own units:

```python
selected = outcomes["Floor 0.005"]
selected_weights = allocation["Floor 0.005"]
original_margin = float(a @ selected_weights - 0.005)
encoded_margin = float(specifications["Floor 0.005"].asset_returns @ selected_weights)
assert abs(original_margin - encoded_margin) < 1e-8
assert original_margin >= -1e-6
assert abs(selected_weights["Core"] - 1.0) < 1e-6
assert abs(selected_weights.drop("Core").sum() - overlay_budget) < 1e-6
residuals = selected.residuals_frame()
hard_breaches = [r for r in selected.constraint_residuals if r.hard and not r.passed]
assert not hard_breaches
```

`compliant` evaluates stored hard residuals with their tolerances. For the homogeneous encoding,
the `target_return` residual measures $\widetilde a^\top w\geq0$. It equals the original margin
only when the exposure equality is satisfied; check the budget and the floor together.
Compliance covers the encoded specification and does not establish the quality of the proxy.

The labelled wrapper returns a weight Series and an outcome. With this complete input panel
and explicit configuration it agrees with the raw call and the original example helper:

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
same mandate survived. The example disables automatic universe-ratio bound rescaling explicitly.

Source owners: [constraint compilation](../src/optimalportfolios/optimization/constraints/backends.py),
[residual evaluation](../src/optimalportfolios/optimization/constraints/analytics.py),
[solver outcomes](../src/optimalportfolios/optimization/solver_diagnostics.py), and
[risk-model adapter](../src/optimalportfolios/covar_estimation/risk_model_adapter.py).

From a C-local source export, using the external interpreter and setup described in
[AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md):

```text
python examples/solvers/overlay_tail_floor.py
python tools/check_docs.py --files docs/overlay_tail_floor.md
python -m pytest src/optimalportfolios/tests/overlay_tail_floor_documentation_test.py
```

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

Here the bound is $-0.08+1(0.09)=0.01$. A floor of 0.02 is impossible:

```python
maximum_linear = float(a["Core"] + overlay_budget * a.drop("Core").max())
impossible_floor = 0.02
assert impossible_floor > maximum_linear
impossible = replace(
    base, asset_returns=a - impossible_floor / total_exposure, target_return=0.0,
    weights_0=allocation["No floor"],
)
rejected = opt.cvx_maximize_portfolio_sharpe(
    covar=covar.to_numpy(), means=means.to_numpy(), constraints=impossible,
    context="overlay article: deliberate infeasibility",
)
print(rejected.status, rejected.accepted, rejected.fallback_source, rejected.compliant)
# infeasible False weights_0 False
```

The fallback keeps the prior no-floor allocation and violates the impossible floor. The shared
fallback preference is finite prior weights, then finite benchmark weights, then zeros.
It does not project a fallback into the feasible set. Zero weights would also violate the fixed
core and total-exposure mandates. Solver status, acceptance, fallback source and compliance are
separate facts; an application must decide whether to trade, retry or skip.

### Backend and modeling limits

The fixed-exposure path does not automatically homogenize every possible side condition.
Do not assume that a turnover, volatility or benchmark-relative row has been transformed
correctly merely because it belongs to `Constraints`. This article verifies only the stated
asset, exposure, sleeve and homogeneous-floor pattern.

An exposure band routes the current entry point to SLSQP. Its compiler includes exposure,
asset bounds and group allocations, but **does not add `asset_returns/target_return`** to the
optimization problem. Post-solve validation can reject a returned candidate for violating
that floor; it does not solve the missing constrained problem. Variable exposure is therefore
not an alternative encoding for this article's floor. This is the implementation's routing
and support boundary, not a general impossibility of fractional reformulation.

These two deliberate limitation probes both reach solver status `optimal` on the synthetic
inputs, then fail acceptance because the original return floor was violated:

```python
direct_nonzero = replace(base, asset_returns=a, target_return=0.005)
banded = replace(base, min_exposure=1.8, asset_returns=a, target_return=0.0)
unsupported_outcomes = {}
for label, spec in {"Unscaled floor": direct_nonzero, "Exposure band": banded}.items():
    candidate = opt.cvx_maximize_portfolio_sharpe(
        covar=covar.to_numpy(), means=means.to_numpy(), constraints=spec,
        context=f"overlay article: limitation probe {label}",
    )
    unsupported_outcomes[label] = candidate
    print(label, candidate.solver, candidate.status, candidate.accepted)
    assert not candidate.accepted
# Unscaled floor CLARABEL optimal False
# Exposure band SLSQP optimal False
```

The first call illustrates the unscaled nonzero CVXPY right-hand side; the second illustrates
the missing SLSQP return row. Their rejected fallback weights must not be interpreted as
optimized overlays.

A linear characteristic floor does not control the full loss distribution or nonlinear option
payoffs. Use a common scenario/regime definition to make coefficients additive. If the tail
sample is chosen separately for each asset, their individual tail statistics generally do not
aggregate into a portfolio tail statistic.

Local verification on 2026-09-14 used OptimalPortfolios 7.6.0 working source, QIS 5.26.0,
FactorLasso 0.18.0, NumPy 2.5.2, pandas 3.0.5, CVXPY 1.9.2 and CLARABEL 0.11.1.
This does not certify the existing lockfile's QIS 5.22.3 environment. No numerical implementation
or original example was changed. Sphinx, GitHub and VS Code previews require separate review.

## See also

- [Optimization module](optimization_module_readme.md): objectives, configuration and outcomes.
- [Portfolio constraints](constraints.md): canonical compiler, alignment and residual contracts.
- [Rolling backtests](rolling_backtests.md): decision and execution timing.
- [Turnover and transaction costs](turnover_and_transaction_costs.md): trades, budgets and costs.
- [Minimum tracking error](minimum_tracking_error.md): benchmark-relative risk and QIS integration.

## References

- Charnes, A. and Cooper, W. W. (1962). *Programming with Linear Fractional Functionals*.
  Naval Research Logistics Quarterly, 9(3–4), 181–186.
  [Primary paper in the CMU archive](https://iiif.library.cmu.edu/file/Cooper_box00010_fld00009_bdl0001_doc0001/Cooper_box00010_fld00009_bdl0001_doc0001.pdf).
- Cornuéjols, G. and Tütüncü, R. (2007). *Optimization Methods in Finance*.
  Cambridge University Press.
  [Publisher record](https://www.cambridge.org/core/books/optimization-methods-in-finance/FAE3FDF1D69C6B0704EEC81B617B706A).
  [Author-hosted January 2006 draft](https://www.andrew.cmu.edu/user/gc0v/webpub/OptFinFirstEdition2006.pdf),
  section 8.2, "Maximizing the Sharpe Ratio."
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
