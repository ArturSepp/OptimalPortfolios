---
myst:
  html_meta:
    description: >-
      Strategic asset allocation in Python with optimalportfolios: minimum variance at a target
      return and maximum expected return at a target volatility, their hard and utility forms,
      proofs that both solvers and the variance-penalty form trace one efficient frontier, what
      the code returns for infeasible or slack targets, and a verified offline example on a
      19-asset multi-asset universe.
---

# Strategic allocation: target return and target volatility

*Author: [Artur Sepp](https://github.com/ArturSepp)*

The strategic allocation solvers are implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

A strategic asset allocation (SAA) turns capital market assumptions, the expected returns of the
asset classes, and a covariance matrix into a long-run portfolio. The package solves the two
classical forms of the mean-variance problem of Markowitz (1952): the minimum-variance portfolio
whose expected return reaches a target, and the portfolio with the highest expected return whose
volatility stays within a target. Each has a rolling function, a single-date wrapper and two
CVXPY solvers: a hard form that imposes the target as a constraint, and a utility (soft) form,
selected by `constraint_enforcement_type`, that prices risk and trading in the objective instead.

This page proves that the two hard problems are one efficient frontier read from two ends: when
the targets bind, each solver returns the other's portfolio. The utility form of the volatility
solver, which subtracts a variance penalty, traces the same frontier as its weight varies and
meets the hard solution at one weight, the shadow price of the volatility target. The page also
states what the code returns when a target cannot be met or does not bind, and how the rolling
functions align expected returns and targets with the rebalancing dates. The worked example checks
each statement on the 19-instrument multi-asset fixture that ships with the package.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | The solvers sample no returns. The example's covariance is estimated from monthly log returns, the `EwmaCovarEstimator` default; the rolling functions drift the previous weights with simple price ratios |
| Estimation grid | None in the solvers. The example estimates on monthly returns (`'ME'`) with span 36, the settings of [`examples/backtests/multiasset_saa.py`](../examples/backtests/multiasset_saa.py) |
| Rebalancing grid | One solve per single-date call; the rolling functions solve at the keys of `covar_dict`, the 21 year ends from 2005 to 2025 in the example |
| Covariance units | Annual, in fractional return squared, as the estimator returns it; `target_vol` is in the square-root units of the covariance and is never annualised |
| Expected returns | `expected_returns`, the caller's capital market assumptions, in the units of `target_return`; the example uses the stylised rule $\mu_i = 0.02 + 0.30 \sigma_i$ |
| Weight state | Target weights on rebalancing dates; the rolling functions pass the previous targets, drifted to the date, as `weights_0` |
| Solver | CVXPY with CLARABEL (`OptimiserConfig.solver`) on the floored covariance factor; CVXPY on raw arrays for the independent references of the example |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $r^{\ast}$ | Target return, `target_return`: a floor on the expected return $\mu^{\top} w$ |
| $\sigma^{\ast}$ | Target volatility, `target_vol`: a ceiling on $\sigma(w)$, or on $\mathrm{TE}(w)$ with a benchmark |
| $\mathcal{C}$ | Portfolios that meet every other hard row of the `Constraints` |
| $\mathcal{R}(r^{\ast})$, $\mathcal{V}(\sigma^{\ast})$ | Minimum variance at a target return; maximum expected return at a target volatility |
| $w^{\mathrm{mv}}$, $r^{\mathrm{mv}}$, $\sigma^{\mathrm{mv}}$ | Minimum-variance portfolio on $\mathcal{C}$, its expected return and its volatility |
| $r^{\max}$, $\sigma^{\mathrm{top}}$ | Highest expected return on $\mathcal{C}$; the lowest volatility among the portfolios that attain it |
| $\mathcal{U}$ | Portfolios that meet the rows kept hard in utility mode |
| $\phi$, $\phi^{\ast}$ | Variance penalty weight, `tre_utility_weight`; the shadow price of the variance row of $\mathcal{V}(\sigma^{\ast})$ |
| $w_{\phi}$ | Solution of the utility form with weight $\phi$ and no turnover term |
| $\kappa$, $c$ | Turnover penalty weight, `turnover_utility_weight`, and per-asset multipliers, `turnover_costs` |

The propositions assume that $\Sigma$ is positive definite and that $\mathcal{C}$ is closed,
convex and bounded; it is bounded for a long-only, fully invested portfolio. The solvers use the
covariance after an eigenvalue floor of $10^{-10}$, described in
[solver numerics and outcomes](solver_numerics_and_outcomes.md), so the matrix they see is
positive definite; the smallest eigenvalue of the example's covariance is above $10^{-5}$, and the
floor leaves it unchanged. Expected returns and targets share one horizon and scaling, annual
decimals in the example.

## Methodology

### Two problems

The hard solvers solve

$$
\mathcal{R}(r^{\ast}): \quad \min_{w \in \mathcal{C}} w^{\top} \Sigma w \quad \text{subject to} \quad \mu^{\top} w \geq r^{\ast} ,
$$

$$
\mathcal{V}(\sigma^{\ast}): \quad \max_{w \in \mathcal{C}} \mu^{\top} w \quad \text{subject to} \quad w^{\top} \Sigma w \leq \sigma^{\ast 2} .
$$

The return row is a floor and the variance row a ceiling. $\mathcal{C}$ holds every other row that
`Constraints.set_cvx_all_constraints` compiles: full investment, long-only, boxes, group bands and
any configured turnover, tracking-error, deviation or beta limit. A `max_target_portfolio_vol_an`
configured on the constraints belongs to $\mathcal{C}$ in $\mathcal{R}$; in $\mathcal{V}$
without a benchmark, the wrapper replaces it with $\sigma^{\ast}$. By default the variance row is compiled as the
second-order cone $\lVert B^{\top} w \rVert_2 \leq \sigma^{\ast}$, with $B$ the factor of the
floored covariance, which is the same set; with `factorize_covar=False` it is a quadratic row. The
constraints page states the two rows under
[minimum target return](constraints.md#minimum-target-return) and
[maximum portfolio volatility](constraints.md#maximum-portfolio-volatility).

With benchmark weights, both problems move to the active weights $d = w - w^{\mathrm{bm}}$:
$\mathcal{R}$ minimises $d^{\top} \Sigma d$ under the same floor on $\mu^{\top} w$, and
$\mathcal{V}$ maximises $\mu^{\top} d$ under $\mathrm{TE}(w) \leq \sigma^{\ast}$, the wrapper
setting `tracking_err_vol_constraint` to $\sigma^{\ast}$. The constant $\mu^{\top} w^{\mathrm{bm}}$
does not move the maximiser, and the results below hold with $\mathrm{TE}(w)$ in place of
$\sigma(w)$.

The targets and the expected returns reach the solvers as follows:

| Entry points | Return target | Volatility target | Expected returns |
|---|---|---|---|
| `rolling_min_variance_target_return`, `rolling_max_return_target_vol` | `target_returns`, a Series forward-filled to the keys of `covar_dict` | `target_vols`, likewise | `expected_returns`, a date-by-asset table forward-filled to the keys, then filled with zeros |
| `wrapper_min_variance_target_return`, `wrapper_max_return_target_vol` | `target_return`, a number | `target_vol`, a number | `expected_returns` at the date; a missing value becomes zero with a warning |
| `cvx_min_variance_target_return`, `cvx_min_variance_target_return_utility` | the `Constraints` fields `target_return` and `asset_returns` | none | `asset_returns` |
| `cvx_max_return_target_vol`, `cvx_max_return_target_vol_utility` | none | `max_target_portfolio_vol_an`, or `tracking_err_vol_constraint` with a benchmark | the `alphas` argument |

The return-floor wrapper writes `expected_returns` into `asset_returns` and the target into
`target_return`, replacing any values configured on the constraints. In the rolling functions, the
row of capital market assumptions in force at a date is therefore the coefficient vector of the
floor in the return solver and the objective in the volatility solver.

### Frontier duality

Let $w^{\mathrm{mv}}$ be the unique minimiser of $w^{\top} \Sigma w$ on $\mathcal{C}$, with return
$r^{\mathrm{mv}}$ and volatility $\sigma^{\mathrm{mv}}$; let $r^{\max}$ be the largest
$\mu^{\top} w$ on $\mathcal{C}$ and $\sigma^{\mathrm{top}}$ the smallest volatility among the
portfolios of $\mathcal{C}$ that earn $r^{\max}$. The efficient frontier is the curve of
$(\sigma(w), \mu^{\top} w)$ traced by the solutions of $\mathcal{R}(r^{\ast})$ as $r^{\ast}$ runs
from $r^{\mathrm{mv}}$ to $r^{\max}$.

**Proposition 1 (frontier duality).** Let $\Sigma$ be positive definite and $\mathcal{C}$
closed, convex and bounded.

1. If $r^{\ast} \leq r^{\max}$, $\mathcal{R}(r^{\ast})$ has a unique solution $w^{\ast}$, and
   $\mathcal{V}(\sigma(w^{\ast}))$ has the unique solution $w^{\ast}$. The floor binds,
   $\mu^{\top} w^{\ast} = r^{\ast}$, when $r^{\ast} \geq r^{\mathrm{mv}}$; otherwise
   $w^{\ast} = w^{\mathrm{mv}}$, whose return exceeds the target.
2. If $\sigma^{\mathrm{mv}} \leq \sigma^{\ast} \lt \sigma^{\mathrm{top}}$,
   $\mathcal{V}(\sigma^{\ast})$ has a unique solution $w^{\ast}$, the variance row binds,
   $\sigma(w^{\ast}) = \sigma^{\ast}$, and $\mathcal{R}(\mu^{\top} w^{\ast})$ has the unique
   solution $w^{\ast}$.
3. $\mathcal{R}(r^{\ast})$ is infeasible when $r^{\ast} \gt r^{\max}$, and
   $\mathcal{V}(\sigma^{\ast})$ when $\sigma^{\ast} \lt \sigma^{\mathrm{mv}}$. When
   $\sigma^{\ast} \geq \sigma^{\mathrm{top}}$, $\mathcal{V}(\sigma^{\ast})$ earns $r^{\max}$ and
   its variance row need not bind.

**Proof.** (1) The variance is strictly convex, so a feasible $\mathcal{R}(r^{\ast})$ has one
minimiser $w^{\ast}$ on its compact feasible set. Because $w^{\ast}$ is feasible for
$\mathcal{V}(\sigma(w^{\ast}))$, any solution $v$ of that problem has
$\mu^{\top} v \geq \mu^{\top} w^{\ast} \geq r^{\ast}$ and $\sigma(v) \leq \sigma(w^{\ast})$: it
is feasible for $\mathcal{R}(r^{\ast})$ and no worse than its minimiser, so $v = w^{\ast}$. If
$r^{\ast} \leq r^{\mathrm{mv}}$, the global minimiser $w^{\mathrm{mv}}$ meets the floor and
$w^{\ast} = w^{\mathrm{mv}}$. If $r^{\ast} \gt r^{\mathrm{mv}}$ and the floor were slack at
$w^{\ast}$, a short step from $w^{\ast}$ towards $w^{\mathrm{mv}}$ would keep the floor and, by
strict convexity, lower the variance, contradicting optimality.

(2) $w^{\mathrm{mv}}$ is feasible, so a solution $v$ exists. If $\sigma(v) \lt \sigma^{\ast}$, the
variance row is slack near $v$, so $v$ is a local maximiser of the linear objective on the convex
set $\mathcal{C}$ and therefore a global one: $\mu^{\top} v = r^{\max}$ with
$\sigma(v) \lt \sigma^{\mathrm{top}}$, which contradicts the definition of $\sigma^{\mathrm{top}}$.
So every solution has $\sigma(v) = \sigma^{\ast}$. Two different solutions would have a midpoint
with the same return and, by strict convexity, a volatility below $\sigma^{\ast}$, which was just
excluded, so the solution $w^{\ast}$ is unique. The solution $u$ of
$\mathcal{R}(\mu^{\top} w^{\ast})$ has $\sigma(u) \leq \sigma(w^{\ast}) = \sigma^{\ast}$ and
$\mu^{\top} u \geq \mu^{\top} w^{\ast}$, so it solves $\mathcal{V}(\sigma^{\ast})$ and equals
$w^{\ast}$.

(3) follows from the definitions of $r^{\max}$, $\sigma^{\mathrm{mv}}$ and
$\sigma^{\mathrm{top}}$. $\square$

In the code, an infeasible target makes CVXPY report `infeasible`, and the solve is rejected and
replaced by the fallback: the pre-trade weights $w_0$, else the benchmark, else zeros, as
[solver numerics and outcomes](solver_numerics_and_outcomes.md#acceptance-and-fallback) describes.
A single-date call without `weights_0` or a benchmark therefore returns zeros. Before solving,
`wrapper_min_variance_target_return` lowers a target above the largest single expected return,
$\max_i \mu_i$, to that value with a warning. Caps and bands usually keep $r^{\max}$ below
$\max_i \mu_i$, so the lowered target can still be infeasible. `wrapper_max_return_target_vol`
does not adjust its target. A slack target is solved as part 1 or part 3 states: a return target
below $r^{\mathrm{mv}}$ returns $w^{\mathrm{mv}}$, and a volatility target at or above
$\sigma^{\mathrm{top}}$ returns a highest-return portfolio, whose volatility can be below the
target.

### Hard and utility forms

The wrappers choose the solver from `constraint_enforcement_type`:
`ConstraintEnforcementType.UTILITY_CONSTRAINTS` selects the `_utility` function, and any other
value, including the default `FORCED_CONSTRAINTS`, the hard one. The utility forms solve on
$\mathcal{U}$, the rows that stay hard in utility mode: exposure, long-only, boxes, the return
floor, group allocation, deviations and beta. Volatility, tracking-error and turnover caps are
dropped, so $\mathcal{U} = \mathcal{C}$ when none is configured, as in the example. The
constraints page describes the split under
[hard and utility enforcement](constraints.md#hard-and-utility-enforcement) and each solver's
penalty terms under [solver-specific utility paths](constraints.md#solver-specific-utility-paths).

`cvx_min_variance_target_return_utility` solves

$$
\min_{w \in \mathcal{U}} w^{\top} \Sigma w + \kappa \lVert c \odot (w - w_0) \rVert_1 \quad \text{subject to} \quad \mu^{\top} w \geq r^{\ast} ,
$$

with $\odot$ the elementwise product and the active variance $d^{\top} \Sigma d$ in place of
$w^{\top} \Sigma w$ when a benchmark is given. The return floor stays hard. The turnover term is
added only when `weights_0` is supplied and $\kappa$, `turnover_utility_weight`, 0.40 by default,
is not `None`; $c$ is `turnover_costs`, or ones. `tre_utility_weight` is not used, so without
`weights_0` the utility form solves $\mathcal{R}(r^{\ast})$ on $\mathcal{U}$.

`cvx_max_return_target_vol_utility` solves, without a benchmark,

$$
\max_{w \in \mathcal{U}} \mu^{\top} w - \phi w^{\top} \Sigma w - \kappa \lVert c \odot (w - w_0) \rVert_1 ,
$$

with $\phi$ = `tre_utility_weight`, 1.0 by default, and the same turnover term; `None` removes
either penalty. The target $\sigma^{\ast}$ does not enter. Without the turnover term, this is the
quadratic utility $\mu^{\top} w - \gamma w^{\top} \Sigma w / 2$ of
[mean-variance objectives](mean_variance_objectives.md) at the risk aversion $\gamma = 2 \phi$; the
script checks that `wrapper_quadratic_optimisation` with `carra` set to $2 \phi^{\ast}$ returns
the same portfolio as the utility form at $\phi^{\ast}$. With a benchmark the solver calls the
generic utility builder of the constraints page, which maximises
$\mu^{\top} d - \phi d^{\top} \Sigma d$ less the turnover penalty, and in which configured group
tracking-error and group turnover penalties replace the total ones. In utility mode the
volatility, tracking-error and turnover residuals are soft and never reject a solve.

**Proposition 2 (the utility form traces the frontier).** Let $\Sigma$ be positive definite,
$\mathcal{U}$ closed, convex and bounded, and let $\mathcal{V}$, $r^{\mathrm{mv}}$,
$\sigma^{\mathrm{mv}}$, $r^{\max}$ and $\sigma^{\mathrm{top}}$ refer to $\mathcal{U}$. For
$\phi \gt 0$:

1. $w_{\phi}$ is the unique solution of $\mathcal{V}(\sigma(w_{\phi}))$, so every utility
   solution lies on the frontier.
2. $\sigma(w_{\phi})$ and $\mu^{\top} w_{\phi}$ do not increase as $\phi$ grows.
3. If $\sigma^{\mathrm{mv}} \lt \sigma^{\ast} \lt \sigma^{\mathrm{top}}$, the variance row of
   $\mathcal{V}(\sigma^{\ast})$ has a shadow price $\phi^{\ast} \gt 0$, and
   $w_{\phi^{\ast}}$ is the solution of $\mathcal{V}(\sigma^{\ast})$.
4. The tracking error of $w_{\phi}$ against $w^{\mathrm{mv}}$ is at most
   $\sqrt{(r^{\max} - r^{\mathrm{mv}}) / \phi}$, so $w_{\phi}$ tends to $w^{\mathrm{mv}}$ as
   $\phi$ grows.

**Proof.** (1) The utility is strictly concave, so $w_{\phi}$ is unique. For any
$w \in \mathcal{U}$ with $\sigma(w) \leq \sigma(w_{\phi})$, optimality of $w_{\phi}$ gives

$$
\mu^{\top} w \leq \mu^{\top} w_{\phi} - \phi \left( w_{\phi}^{\top} \Sigma w_{\phi} - w^{\top} \Sigma w \right) \leq \mu^{\top} w_{\phi} ,
$$

with equality only if $w$ attains the utility of $w_{\phi}$, that is, $w = w_{\phi}$.

(2) Take $\phi_a \lt \phi_b$, with solutions $w_a$, $w_b$ and variances $v_a$, $v_b$. Each is
optimal at its own weight:
$\mu^{\top} w_a - \phi_a v_a \geq \mu^{\top} w_b - \phi_a v_b$ and
$\mu^{\top} w_b - \phi_b v_b \geq \mu^{\top} w_a - \phi_b v_a$. Adding the two gives
$(\phi_b - \phi_a)(v_a - v_b) \geq 0$, so $v_a \geq v_b$, and the first inequality then gives
$\mu^{\top} w_a - \mu^{\top} w_b \geq \phi_a (v_a - v_b) \geq 0$.

(3) Near $w^{\mathrm{mv}}$ there are points of the relative interior of $\mathcal{U}$ with
volatility below $\sigma^{\ast}$, so Slater's condition holds, and Lagrange duality for the convex
problem $\mathcal{V}(\sigma^{\ast})$ gives a multiplier $\phi^{\ast} \geq 0$ of the variance row
such that its solution $w^{\ast}$ maximises $\mu^{\top} w - \phi^{\ast} (w^{\top} \Sigma w - \sigma^{\ast 2})$
on $\mathcal{U}$ (Boyd and Vandenberghe 2004, chapter 5). If $\phi^{\ast} = 0$, $w^{\ast}$ would
maximise $\mu^{\top} w$ on $\mathcal{U}$ with $\sigma(w^{\ast}) \lt \sigma^{\mathrm{top}}$, which
is impossible, so $\phi^{\ast} \gt 0$. The Lagrangian differs from the utility at
$\phi^{\ast}$ by the constant $\phi^{\ast} \sigma^{\ast 2}$, so $w_{\phi^{\ast}} = w^{\ast}$.

(4) Optimality of $w_{\phi}$ against $w^{\mathrm{mv}}$ gives
$\phi (\sigma(w_{\phi})^2 - \sigma(w^{\mathrm{mv}})^2) \leq \mu^{\top} w_{\phi} - r^{\mathrm{mv}} \leq r^{\max} - r^{\mathrm{mv}}$.
The first-order condition of $w^{\mathrm{mv}}$,
$(\Sigma w^{\mathrm{mv}})^{\top} (w - w^{\mathrm{mv}}) \geq 0$ for every $w \in \mathcal{U}$,
gives
$\sigma(w)^2 \geq \sigma(w^{\mathrm{mv}})^2 + (w - w^{\mathrm{mv}})^{\top} \Sigma (w - w^{\mathrm{mv}})$,
and the two combine to the bound. $\square$

Parts 2 and 3 describe how the utility form approaches a hard volatility target
$\sigma^{\mathrm{mv}} \lt \sigma^{\ast} \lt \sigma^{\mathrm{top}}$. Below $\phi^{\ast}$, its
solutions carry at least the target volatility and earn at least the hard return. As $\phi$
grows, the excess volatility shrinks and vanishes at $\phi^{\ast}$, where the utility and hard
solutions coincide; a larger weight carries the portfolio below the target, towards
$w^{\mathrm{mv}}$. The shadow price depends on the target, the expected returns and the
covariance, so one fixed weight does not hold a volatility target from date to date.

**Remark (a turnover penalty that holds the portfolio).** Let $w_0 \in \mathcal{U}$ meet the
return floor, take $c$ equal to ones and write $g = 2 \Sigma w_0$ for the gradient of the variance
at $w_0$. If $\kappa \gt \max_i \lvert g_i \rvert$, the minimum-variance utility form without a
benchmark returns $w_0$. Indeed, by convexity, every $w \in \mathcal{U}$ has
$w^{\top} \Sigma w \geq w_0^{\top} \Sigma w_0 + g^{\top} (w - w_0) \geq w_0^{\top} \Sigma w_0 - \max_i \lvert g_i \rvert \lVert w - w_0 \rVert_1$,
so any other feasible portfolio has a larger objective. With annual variances, $g$ is of the order
of a variance, far below the default $\kappa$ of 0.40.

## Worked example

The canonical script of this page,
[`examples/docs/strategic_allocation_targets.py`](../examples/docs/strategic_allocation_targets.py),
runs offline and asserts every number quoted here:

```console
python -m examples.docs.strategic_allocation_targets
```

The universe and settings are those of the offline
[multi-asset example](../examples/backtests/multiasset_saa.py): the 19 instruments of the packaged
monthly fixture, in fixed income, equity, alternatives and cash, and a long-only portfolio with at
most 25% in any instrument, at least 20% in fixed income, and at most 60% in equities and 40% in
alternatives, which the script's `saa_constraints` builds from the fixture's asset-class labels
with `GroupLowerUpperConstraints`. The covariance is the EWMA estimate of monthly returns with
span 36 at 31 December 2025, the last of 21 year-end estimates. The capital market assumptions are
stylised: each instrument earns a cash rate of 2% plus 0.30 times its volatility, the same Sharpe
ratio for all, which gives expected returns from 2.15% for cash to 6.27% for Asia ex-Japan
equities. They are a teaching input, not a forecast:

```python
data = load_multiasset_data()
estimator = op.EwmaCovarEstimator(returns_freq=RETURNS_FREQ, span=EWMA_SPAN,
                                  rebalancing_freq=REBALANCING_FREQ)
covar_dict = estimator.fit_rolling_covars(
    prices=data.prices,
    time_period=qis.TimePeriod(start=BACKTEST_START, end=data.prices.index[-1]))
covar = covar_dict[pd.Timestamp(DATE)]
cmas = equal_sharpe_cmas(covar)
constraints = saa_constraints(data.group_data)
assert covar.shape == (19, 19) and len(covar_dict) == 21
assert np.linalg.eigvalsh(covar.to_numpy()).min() > 1e-5
assert round(cmas['Cash'], 4) == 0.0215 and round(cmas.max(), 4) == 0.0627
assert cmas.idxmax() == 'Asia Ex-Japan'
```

The frontier runs from the minimum-variance portfolio, with an expected return of 2.90% and a
volatility of 1.86%, which holds cash and global bonds at their 25% caps, to the highest attainable
return of 5.5%, whose lowest-volatility portfolio has 9.65% volatility. An independent CVXPY solve
on raw arrays gives the same minimum-variance weights. The caps and bands keep the highest return
below the best single expected return of 6.27%:

```python
w_mv, _ = op.wrapper_min_variance_target_return(
    pd_covar=covar, expected_returns=cmas, target_return=0.0, constraints=constraints)
r_mv, vol_mv = cmas @ w_mv, volatility(w_mv, covar)
ends = frontier_ends(covar, cmas, data.group_data)
assert np.abs(w_mv.to_numpy() - ends['w_mv']).max() < WEIGHT_TOL
assert round(r_mv, 4) == 0.0290 and round(vol_mv, 4) == 0.0186
assert abs(w_mv['Cash'] - MAX_WEIGHT) < 1e-6 and abs(w_mv['Global Bonds'] - MAX_WEIGHT) < 1e-6
r_max, vol_top = ends['r_max'], volatility(ends['w_top'], covar)
assert round(r_max, 3) == 0.055 and round(vol_top, 4) == 0.0965
```

A target return of 4.5% gives a portfolio with 4.32% volatility. Given that volatility as its
target, the volatility solver returns the same weights, to $10^{-3}$ in every instrument, and the
same expected return, as part 1 of Proposition 1 states:

```python
w_return, outcome = op.wrapper_min_variance_target_return(
    pd_covar=covar, expected_returns=cmas, target_return=TARGET_RETURN,
    constraints=constraints)
vol_return = volatility(w_return, covar)
w_back, _ = op.wrapper_max_return_target_vol(
    pd_covar=covar, expected_returns=cmas, target_vol=vol_return, constraints=constraints)
assert outcome.accepted and abs(cmas @ w_return - TARGET_RETURN) < MOMENT_TOL
assert np.abs(w_back - w_return).max() < WEIGHT_TOL
assert abs(cmas @ w_back - TARGET_RETURN) < MOMENT_TOL and round(vol_return, 4) == 0.0432
```

The reverse trip, part 2, starts from a target volatility of 5%. The volatility solver earns 4.81%
with alternatives at their 40% band, and the return solver, given that return as its target,
returns the same portfolio at 5% volatility:

```python
w_vol, _ = op.wrapper_max_return_target_vol(
    pd_covar=covar, expected_returns=cmas, target_vol=TARGET_VOL, constraints=constraints)
return_vol = cmas @ w_vol
w_forth, _ = op.wrapper_min_variance_target_return(
    pd_covar=covar, expected_returns=cmas, target_return=return_vol,
    constraints=constraints)
assert abs(volatility(w_vol, covar) - TARGET_VOL) < MOMENT_TOL
assert np.abs(w_forth - w_vol).max() < WEIGHT_TOL
assert abs(volatility(w_forth, covar) - TARGET_VOL) < MOMENT_TOL
assert round(return_vol, 4) == 0.0481
assert abs(w_vol.groupby(data.group_data).sum()['Alternatives'] - 0.40) < 1e-6
```

The script repeats both trips at the six return targets and eight volatility targets of the
figure below. For the utility form, an independent CVXPY solve of $\mathcal{V}(0.05)$ with the
variance written as a quadratic row gives the dual value of that row, $\phi^{\ast} = 4.46$.
With `tre_utility_weight` set to it, the utility form returns the hard 5% portfolio, as part 3 of
Proposition 2 states:

```python
phi_star = shadow_price(covar, cmas, data.group_data, TARGET_VOL)
soft = constraints.copy(
    constraint_enforcement_type=op.ConstraintEnforcementType.UTILITY_CONSTRAINTS,
    tre_utility_weight=phi_star)
w_soft, soft_outcome = op.wrapper_max_return_target_vol(
    pd_covar=covar, expected_returns=cmas, target_vol=TARGET_VOL, constraints=soft)
assert soft_outcome.accepted and np.abs(w_soft - w_vol).max() < WEIGHT_TOL
assert round(phi_star, 2) == 4.46
```

> **Insight.** A target return, a target volatility and a variance penalty are three names for one
> point of the frontier. At the end of 2025, a 5% volatility target, a return target equal to that
> portfolio's 4.81% and a penalty weight of 4.46 all return the same portfolio.

Along a grid of weights from 0.5 to 50, each utility solution equals the hard solution at its own
volatility, and the volatility falls as the weight grows: 7.07%, 5.99% and 4.60% at weights of 1,
2 and 5. Below $\phi^{\ast}$ the utility solutions exceed the 5% target, and above it they stay
below. The tracking error against the minimum-variance portfolio respects the bound of part 4:

```python
soft_vols, soft_returns = [], []
for phi in PENALTY_WEIGHTS:
    w_phi, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=TARGET_VOL,
        constraints=soft.copy(tre_utility_weight=phi))
    w_hard, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=volatility(w_phi, covar),
        constraints=constraints)
    assert np.abs(w_hard - w_phi).max() < WEIGHT_TOL
    tracking_error = volatility(w_phi - w_mv, covar)
    assert tracking_error <= np.sqrt((r_max - r_mv) / phi)
    soft_vols.append(volatility(w_phi, covar))
    soft_returns.append(cmas @ w_phi)
assert np.all(np.diff(soft_vols) < 0.0) and np.all(np.diff(soft_returns) < 0.0)
assert all((vol > TARGET_VOL) == (phi < phi_star)
           for phi, vol in zip(PENALTY_WEIGHTS, soft_vols))
assert [round(vol, 4) for vol in soft_vols[1:4]] == [0.0707, 0.0599, 0.0460]
```

![Left: expected return against volatility at the end of 2025. Six minimum-variance solutions at
target returns from 3% to 5.5%, eight maximum-return solutions at target volatilities from 2% to
9%, and seven utility solutions at penalty weights from 0.5 to 50 all lie on one frontier, which
runs from the minimum-variance portfolio at 1.86% volatility and 2.90% return to the highest return
of 5.5% at 9.65% volatility. Right: the volatility of the utility solution against the penalty
weight on a log scale, falling from 8% at a weight of 0.3 towards the minimum-variance level of
1.86% and crossing the 5% target at the shadow price 4.46, where the hard solution
sits.](images/saa_target_duality.png)

*Figure: the target-return, target-volatility and utility solvers on the frontier of the 19-asset
example, and the path of the utility solution as its penalty weight grows. Drawn by the `exhibit`
function of the canonical script; the [analytics gallery](analytics_gallery.md) lists its
provenance.*

The default utility form does not use its target at all. With `tre_utility_weight` left at 1.0,
targets of 3% and 7% return the same portfolio, the solution of the utility with $\phi = 1$,
which an independent solve confirms, at 7.07% volatility:

```python
utility = constraints.copy(
    constraint_enforcement_type=op.ConstraintEnforcementType.UTILITY_CONSTRAINTS)
w_3, _ = op.wrapper_max_return_target_vol(
    pd_covar=covar, expected_returns=cmas, target_vol=0.03, constraints=utility)
w_7, _ = op.wrapper_max_return_target_vol(
    pd_covar=covar, expected_returns=cmas, target_vol=0.07, constraints=utility)
assert np.abs(w_3 - w_7).max() < 1e-9 and utility.tre_utility_weight == 1.0
reference = utility_reference(covar, cmas, data.group_data, phi=1.0)
assert np.abs(w_3.to_numpy() - reference).max() < WEIGHT_TOL
assert round(volatility(w_3, covar), 4) == 0.0707
```

> **Pitfall.** In utility mode, `wrapper_max_return_target_vol` ignores `target_vol`: it
> maximises $\mu^{\top} w - \phi w^{\top} \Sigma w$ with $\phi$ = `tre_utility_weight`, 1.0 by
> default. In the example, targets of 3% and 7% both return a portfolio with 7.07% volatility.
> Keep the hard form to hold a volatility target, or set the weight to the target's shadow price.

The Remark applies to the return-floor utility form. Started from the 5% portfolio as `weights_0`,
which earns 4.81% and so meets a 4.5% floor, it returns that portfolio unchanged: the largest
entry of $2 \Sigma w_0$ is 0.0089, below the default $\kappa$ of 0.40. Without `weights_0` it
returns the hard 4.5% solution.

Targets outside the frontier are rejected, as part 3 of Proposition 1 predicts. A return target of
5.6%, above the highest attainable 5.5%, and a volatility target of 1.5%, below the minimum of
1.86%, are both infeasible, and with no `weights_0` and no benchmark the fallback is zeros. A
return target of 7% is first lowered to 6.27% with a warning and is still infeasible. A slack
return target of 2% returns the minimum-variance portfolio, earning 2.90%, and a volatility target
of 12% returns the highest-return portfolio at 9.65%:

```python
w_high, high = op.wrapper_min_variance_target_return(
    pd_covar=covar, expected_returns=cmas, target_return=0.056, constraints=constraints)
w_low, low = op.wrapper_max_return_target_vol(
    pd_covar=covar, expected_returns=cmas, target_vol=0.015, constraints=constraints)
for weights, result in ((w_high, high), (w_low, low)):
    assert not result.accepted and 'infeasible' in result.status
    assert result.fallback_source == 'zeros' and (weights == 0.0).all()
```

The script also checks the benchmark-relative forms: against equal weights, the tracking error of
the 4.5% solution, given back as the target, returns the same portfolio.

In the rolling functions, the capital market assumptions are a date-by-asset table and the target
a Series; both are forward-filled to the keys of `covar_dict`. With one row of assumptions per
year end and a single 4.5% target dated at the first key, the return solver earns exactly 4.5% at
all 21 year ends from 2005 to 2025. Given the resulting volatilities as its targets, the volatility
solver returns the same weights at every date:

```python
cma_table = pd.DataFrame(
    {date: equal_sharpe_cmas(estimate) for date, estimate in covar_dict.items()}).T
dates = list(covar_dict)
saa_return = op.rolling_min_variance_target_return(
    prices=data.prices, expected_returns=cma_table,
    target_returns=pd.Series(TARGET_RETURN, index=dates[:1]),
    constraints=constraints, benchmark_weights=None, covar_dict=covar_dict)
path_vols = pd.Series(
    {date: volatility(saa_return.loc[date], covar_dict[date]) for date in dates})
saa_vol = op.rolling_max_return_target_vol(
    prices=data.prices, expected_returns=cma_table, target_vols=path_vols,
    constraints=constraints, benchmark_weights=None, covar_dict=covar_dict)
assert np.allclose((saa_return * cma_table).sum(axis=1), TARGET_RETURN, atol=MOMENT_TOL)
assert (saa_vol - saa_return).abs().max().max() < WEIGHT_TOL
assert dates[0] == pd.Timestamp('2005-12-31') and dates[-1] == pd.Timestamp(DATE)
```

## Implementation in optimalportfolios

Both families live in
[`optimization/saa/`](../src/optimalportfolios/optimization/saa/__init__.py) and are importable
from `optimalportfolios`. Each has three layers, as described in
[choosing an objective](optimization_module_readme.md):

- `rolling_min_variance_target_return(prices, expected_returns, target_returns, constraints,
  benchmark_weights, covar_dict, rebalancing_indicators=None, optimiser_config=OptimiserConfig())`
  and `rolling_max_return_target_vol(prices, expected_returns, target_vols, constraints,
  benchmark_weights, covar_dict, rebalancing_indicators=None, optimiser_config=OptimiserConfig())`
  solve at each key of `covar_dict` and return a date-by-asset table of target weights on the
  columns of `prices`. `benchmark_weights` has no default: pass `None`, a Series held constant or
  a table forward-filled to the keys. `expected_returns` is forward-filled and then zero-filled,
  the targets only forward-filled. Each solve receives the previous weights, drifted to the date
  when `OptimiserConfig.use_drifted_weights_0` is true, as `weights_0`; after an all-zero result
  the next solve starts without them.
- `wrapper_min_variance_target_return(pd_covar, expected_returns, target_return, constraints,
  benchmark_weights=None, weights_0=None, rebalancing_indicators=None,
  optimiser_config=OptimiserConfig(), context='')` and `wrapper_max_return_target_vol(pd_covar,
  expected_returns, target_vol, constraints, benchmark_weights=None, weights_0=None,
  rebalancing_indicators=None, optimiser_config=OptimiserConfig(), context='')` remove the assets
  with a missing or non-positive variance, align the constraints to the rest, route on
  `constraint_enforcement_type`, and return the weights on the original labels, with zeros for
  removed assets, together with the `OptimizationOutcome`. Their default `OptimiserConfig()` has
  `apply_total_to_good_ratio=False`.
- `cvx_min_variance_target_return(covar, constraints, has_benchmark=False, solver='CLARABEL',
  verbose=False, context='', factorize_covar=True)` and
  `cvx_min_variance_target_return_utility` with the same arguments read the floor from
  `constraints`; `cvx_max_return_target_vol(covar, alphas, constraints, has_benchmark=False,
  solver='CLARABEL', verbose=False, context='', factorize_covar=True)` and
  `cvx_max_return_target_vol_utility` with the same arguments take the expected returns as
  `alphas`. All four return an `OptimizationOutcome`, accepted or with its fallback, as
  [solver numerics and outcomes](solver_numerics_and_outcomes.md) describes.

The rolling dispatcher `compute_rolling_optimal_weights` has no route to these solvers; call them
directly.

## Interpretation and limitations

- The package takes the expected returns and the covariance as given. It adds no shrinkage,
  resampling or robust counterpart for their estimation error, and the example's equal-Sharpe rule
  is a teaching input. The volatility target is ex ante, under the supplied covariance; the
  realised volatility of the allocation will differ.
- Markowitz (1952) leaves the choice among efficient portfolios to the investor's preference
  between expected return and variance; the package asks for that choice as a target or a
  penalty weight. The frontier here is that of one date, under the mandate's caps and bands.
- The ROSAA article of Sepp, Ossa and Kastenholz (2026) builds its strategic layer differently:
  its section *Optimization of the SAA Portfolio* solves constrained risk budgeting (Equation 19)
  to avoid relying on capital market assumptions, under the budget, bounds and asset-class rows of
  Equations 20 to 22, which are the rows of $\mathcal{C}$ in the example. Its tactical problem
  (Equations 32 and 33) maximises alpha under a quadratic tracking-error limit, and its soft
  version (Equation 38) penalises group tracking-error variance and group turnover instead. The
  article recommends the hard rows for production and the soft form for rolling backtests,
  because the hard problem can fail when its limits cannot be met. The utility forms here keep that
  distinction, but penalise total variance and total turnover in the absolute case.
- A rejected rolling solve falls back to the drifted previous weights, which are not re-optimised
  and can breach the mandate; check the outcome of the single-date wrappers where it matters.
- Dates before the first row of `expected_returns` receive zero expected returns. The return
  solver then lowers its target to zero with a warning and returns the minimum-variance portfolio;
  the volatility solver has a zero objective, and every feasible portfolio solves it. Dates before
  the first entry of the target Series receive no target, and CVXPY raises a `ValueError` for the
  missing value.
- A fixed penalty weight does not hold a volatility target over time: at $\phi = 4.46$, the
  shadow price of the end-2025 target, the rolling utility solutions range from 4.1% to 5.3%
  volatility across the 21 year ends.
- With the default utility settings, the rolling return solver keeps the drifted portfolio
  unchanged whenever it still meets the floor, the caps and the bands, as the Remark predicts. In
  the example it holds in 8 of the 20 years after the first and trades only in the other 12.

## See also

- [Choosing an objective](optimization_module_readme.md)
- [Minimum variance, quadratic utility and maximum Sharpe](mean_variance_objectives.md)
- [CARA utility under Gaussian mixtures](cara_gaussian_mixture.md)
- [Portfolio constraints](constraints.md)
- [Solver numerics and outcomes](solver_numerics_and_outcomes.md)
- [Strategic and tactical allocation with HCGL covariance (ROSAA)](app_rosaa_multi_asset_allocation.md)
- [Rolling backtests](rolling_backtests.md)
- [Conventions, notation and glossary](conventions.md)

## References

- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
  [DOI 10.1111/j.1540-6261.1952.tb01525.x](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).
- Sepp, A., Ossa, I. and Kastenholz, M. (2026). *Robust Optimization of Strategic and Tactical
  Asset Allocation for Multi-Asset Portfolios*. The Journal of Portfolio Management, 52(4),
  86–120. [DOI 10.3905/jpm.2025.1.806](https://doi.org/10.3905/jpm.2025.1.806). The sections
  *Optimization of the SAA Portfolio* (Equations 17 to 22) and *TAA Optimization Problem*
  (Equations 32 to 38).
- Boyd, S. and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
  [DOI 10.1017/CBO9780511804441](https://doi.org/10.1017/CBO9780511804441). Chapter 5, Lagrange
  duality and Slater's condition.
- Diamond, S. and Boyd, S. (2016). *CVXPY: A Python-Embedded Modeling Language for Convex
  Optimization*. Journal of Machine Learning Research, 17(83), 1–5.
  [JMLR](https://www.jmlr.org/papers/v17/15-408.html).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
