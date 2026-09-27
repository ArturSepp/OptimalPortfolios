---
myst:
  html_meta:
    description: >-
      Tactical asset allocation in Python with optimalportfolios: maximise expected active return
      under an ex-ante tracking-error budget against a strategic benchmark, the closed form of the
      active weights and the information ratio, the utility form with tracking-error and turnover
      penalties, group limits, and a portfolio yield floor, with a verified offline example.
---

# Tactical allocation: alpha over tracking error and yield targets

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Tactical allocation against a benchmark is implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Ex-ante tracking error is measured with [qis](https://github.com/ArturSepp/QuantInvestStrats);
see its [software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

## Overview

A tactical allocation (TAA) takes active positions against a benchmark, usually the strategic
allocation, in the direction of alpha signals. The solvers on this page maximise the expected
active return $\alpha^{\top}(w - w^{\mathrm{bm}})$ under a budget on the ex-ante tracking error
and the usual mandate rows. Two variants follow the same pattern: a utility form that prices
tracking error and turnover with penalties instead of limiting them, and a yield-target form that
adds a floor on the portfolio's yield.

Despite its name, "alpha over tracking error" maximises expected active return, not a ratio. This
page proves that the two coincide when only the budget and full investment constrain the solve:
the active weights are then proportional to $\Sigma^{-1}\tilde\alpha$, for the alphas
$\tilde\alpha$ net of the alpha of the minimum-variance portfolio, the information ratio is the
same at every budget, and the budget only scales the position. A second result gives the solution under a binding yield
floor. The worked example checks both against the package's solver, measures the tracking error
with `qis.RiskModel`, and shows how the information ratio falls once long-only bounds bind.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | None: no return series is sampled. Covariance, alphas and yields are fixed synthetic inputs, and the rolling checks of the script use constant prices, so previous targets do not drift |
| Estimation grid | None; the covariance is a fixed annual matrix, not an estimate |
| Rebalancing grid | Single-date solves; the rolling functions solve at the keys of `covar_dict`, three 2024 quarter ends in the script's rolling checks, and forward-fill alphas, benchmarks, yields and targets onto them |
| Covariance units | Annual fractional return squared, as supplied; `tracking_err_vol_constraint` is in its square-root units, so 0.02 is a 2% annual tracking error. Neither the solvers nor `qis.RiskModel` rescale it |
| Expected returns | Alphas are annual expected active returns in decimals; the forced solve depends only on their direction, the utility weights on their scale. Yields and `target_return` are annual decimals |
| Weight state | Target weights; `weights_0` is the turnover baseline and the first fallback, and the rolling functions set it to the previous target drifted with simple price ratios (`use_drifted_weights_0=True`) |
| Solver | CVXPY with CLARABEL (`OptimiserConfig.solver`) on the eigen-factorised covariance (`factorize_covar=True`); the tracking-error limit is the cone $\lVert B^{\top} d \rVert_2 \leq \tau$ |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $\tau$ | Tracking-error budget, `Constraints.tracking_err_vol_constraint` |
| $\mathbf{1}$ | Vector of ones |
| $w^{\mathrm{mv}}$ | Fully invested minimum-variance portfolio, $\Sigma^{-1}\mathbf{1} / (\mathbf{1}^{\top}\Sigma^{-1}\mathbf{1})$ |
| $\eta$ | Alpha of the minimum-variance portfolio, $\alpha^{\top} w^{\mathrm{mv}}$; the multiplier of the budget row |
| $\tilde\alpha$ | Budget-projected alphas, $\alpha - \eta \mathbf{1}$ |
| $\nu$ | Multiplier of the tracking-error row, written as a variance limit |
| $\mathrm{IR}(w)$ | Information ratio, $\alpha^{\top} d / \mathrm{TE}(w)$ |
| $\mathrm{IR}^{\star}$ | $\sqrt{\tilde\alpha^{\top}\Sigma^{-1}\tilde\alpha}$ |
| $\lambda_{\mathrm{TE}}$, $\lambda_{\mathrm{TO}}$ | Utility weights `tre_utility_weight` and `turnover_utility_weight`; not the EWMA decay of the conventions page |
| $L_g$, $d_g$ | Loadings of group $g$ and its masked active weights $L_g \odot d$ |
| $\tau_g$, $\lambda_g$ | Group limit from `group_tre_vols` and group penalty weight from `group_tre_utility_weights` |
| $c$ | Turnover multipliers, `turnover_costs`; ones when absent |
| $y$, $y_{\min}$ | Asset yields, the argument `yields`, and the portfolio yield floor, `target_return` |

The propositions assume a positive-definite covariance, a fully invested portfolio
(`min_exposure = max_exposure = 1`, the default) and a benchmark whose weights sum to one, so that
active weights sum to zero, $\mathbf{1}^{\top} d = 0$. The single-date wrappers remove the assets
with a zero or missing variance, or a missing alpha, before the solve; see
[solver numerics and outcomes](solver_numerics_and_outcomes.md#filtering-the-universe).

## Methodology

### The forced problem

With `ConstraintEnforcementType.FORCED_CONSTRAINTS`, the default, the tactical solver maximises
expected active return under a hard tracking-error budget:

$$
\max_{w} \alpha^{\top}(w - w^{\mathrm{bm}}) \quad \text{subject to} \quad (w - w^{\mathrm{bm}})^{\top}\Sigma(w - w^{\mathrm{bm}}) \leq \tau^{2}, \quad \mathbf{1}^{\top} w = 1, \quad w \in \mathcal{C} .
$$

Here $\mathcal{C}$ collects every other row that `Constraints.set_cvx_all_constraints` compiles:
long-only and box bounds, group allocations, turnover caps, group tracking-error caps, a
volatility cap, a return floor, sector and style deviations and benchmark beta. The budget $\tau$ is
`Constraints.tracking_err_vol_constraint`, in the square-root units of the covariance: annual when
the covariance is annual, as on the [conventions page](conventions.md#units). The benchmark enters
twice, in the objective and in the tracking-error row, and it comes from the wrapper's
`benchmark_weights` argument, which replaces any benchmark stored on the constraints. With the
factor $B$ of the stabilised covariance, which equals $\Sigma$ when no eigenvalue is floored, the
row is compiled as the second-order cone $\lVert B^{\top} d \rVert_2 \leq \tau$; see
[how a solve uses the factor](solver_numerics_and_outcomes.md#how-a-solve-uses-the-factor).

Two invariances follow from the budget. Adding a constant $k$ to every alpha changes the objective
by $k \mathbf{1}^{\top} d = 0$, and a positive multiple of the alphas has the same maximiser. The
forced solve therefore uses only the direction of the alphas, whatever their units, and alpha
scores can be passed as they are. Without a tracking-error limit the problem is a linear program,
and its solution sits at a vertex of $\mathcal{C}$.

### The closed form

**Proposition 1 (closed form).** Let $\tau \gt 0$ and $\tilde\alpha \neq 0$, and suppose that
$w^{\mathrm{bm}} + d^{\star}$, with $d^{\star}$ below, meets every row of $\mathcal{C}$. The
forced problem has the unique solution

$$
d^{\star} = \frac{\tau}{\mathrm{IR}^{\star}} \Sigma^{-1}\tilde\alpha, \qquad \tilde\alpha = \alpha - \eta \mathbf{1}, \qquad \eta = \frac{\mathbf{1}^{\top}\Sigma^{-1}\alpha}{\mathbf{1}^{\top}\Sigma^{-1}\mathbf{1}} = \alpha^{\top} w^{\mathrm{mv}}, \qquad \mathrm{IR}^{\star} = \sqrt{\tilde\alpha^{\top}\Sigma^{-1}\tilde\alpha} .
$$

It spends the whole budget, $\mathrm{TE}(w^{\mathrm{bm}} + d^{\star}) = \tau$, and earns
$\alpha^{\top} d^{\star} = \mathrm{IR}^{\star} \tau$, so its information ratio is
$\mathrm{IR}^{\star}$ at every budget. No budget-neutral active portfolio has a higher ratio.

**Proof.** Full investment and a benchmark that sums to one give $\mathbf{1}^{\top} d = 0$. Drop
$\mathcal{C}$ first: the problem becomes the maximisation of $\alpha^{\top} d$ subject to
$\mathbf{1}^{\top} d = 0$ and $d^{\top}\Sigma d \leq \tau^{2}$, with the Lagrangian

$$
\mathcal{L}(d, \eta, \nu) = \alpha^{\top} d - \eta \mathbf{1}^{\top} d - \nu (d^{\top}\Sigma d - \tau^{2}), \qquad \nu \geq 0 .
$$

Stationarity gives $\alpha - \eta \mathbf{1} = 2 \nu \Sigma d$. With $\nu = 0$ it would force
$\alpha = \eta \mathbf{1}$, which $\tilde\alpha \neq 0$ excludes, so $\nu \gt 0$ and
$d = \Sigma^{-1}(\alpha - \eta \mathbf{1}) / (2\nu)$. The budget row
$\mathbf{1}^{\top} d = 0$ fixes $\eta = \mathbf{1}^{\top}\Sigma^{-1}\alpha / (\mathbf{1}^{\top}\Sigma^{-1}\mathbf{1})$,
which is $\alpha^{\top} w^{\mathrm{mv}}$. Since $\nu \gt 0$, complementary slackness makes the
tracking-error row bind: $\tilde\alpha^{\top}\Sigma^{-1}\tilde\alpha / (4\nu^{2}) = \tau^{2}$, so
$\nu = \mathrm{IR}^{\star} / (2\tau)$. The multiplier scales the direction $\Sigma^{-1}\tilde\alpha$
to the budget and gives $d^{\star}$. The problem is convex and $d = 0$ is strictly feasible, so
these conditions are sufficient. For every budget-neutral $d$, the Cauchy–Schwarz inequality in the
inner product of $\Sigma$ gives
$\alpha^{\top} d = \tilde\alpha^{\top} d \leq \mathrm{IR}^{\star} \sqrt{d^{\top}\Sigma d}$, with
equality only along $\Sigma^{-1}\tilde\alpha$. Hence $\alpha^{\top} d \leq \mathrm{IR}^{\star}\tau$
on the feasible set, $d^{\star}$ is its unique maximiser, and no budget-neutral $d$ has a ratio
above $\mathrm{IR}^{\star}$. Adding $\mathcal{C}$ back shrinks the feasible set, and
$w^{\mathrm{bm}} + d^{\star}$ remains in it, so it remains the unique maximiser. $\square$

Three consequences follow. The budget scales the position and leaves its direction and its
information ratio unchanged, so the expected active return grows in proportion to $\tau$. The
alphas matter only through $\tilde\alpha$, their difference from the alpha of the minimum-variance
portfolio: a common level cannot be captured by a budget-neutral position. And $d^{\star}$ does not
depend on the benchmark, as Roll (1992) observes for portfolios that minimise tracking error for a
given expected active return; the benchmark decides only when a bound binds. A long-only bound
binds first at the budget $\min_{i} w^{\mathrm{bm}}_i \tau / \lvert d^{\star}_i \rvert$ over the
assets with $d^{\star}_i \lt 0$. Beyond it $w^{\mathrm{bm}} + d^{\star}$ breaks the bound,
Proposition 1 no longer applies, and in the example the information ratio falls.

Grinold and Kahn (2000) derive the same structure for an unconstrained active manager: holdings
proportional to the inverse covariance times the alphas, and an information ratio that does not
depend on the level of active risk.

### The utility form

With `ConstraintEnforcementType.UTILITY_CONSTRAINTS`, `wrapper_maximise_alpha_over_tre` routes to
`cvx_maximise_tre_utility`, which maximises

$$
\alpha^{\top} d - \lambda_{\mathrm{TE}} d^{\top}\Sigma d - \lambda_{\mathrm{TO}} \sum_i \lvert c_i (w_i - w_{0,i}) \rvert
$$

over the rows that stay hard in utility mode: long-only and box bounds, net exposure, a return
floor, group allocations, sector and style deviations and benchmark beta. The tracking-error
and turnover limits and the volatility cap are not rows of this problem. The first term is dropped
when `alphas=None`, which makes the solve pure tracking; the second when `tre_utility_weight` is
None; the third when `turnover_utility_weight` is None or `weights_0` is absent. The defaults are
$\lambda_{\mathrm{TE}} = 1.0$ and $\lambda_{\mathrm{TO}} = 0.40$. A
`group_tracking_error_constraint` or `group_turnover_constraint` replaces the corresponding total
penalty with its per-group penalties, as set out under
[group precedence](constraints.md#group-precedence). Tracking error is penalised as a variance,
not as a volatility.

**Corollary (the equivalent utility weight).** Under the assumptions of Proposition 1 and without
a turnover term, the utility form with $\lambda_{\mathrm{TE}} = \mathrm{IR}^{\star} / (2\tau)$ has
the solution $d^{\star}$. For any weight $\lambda_{\mathrm{TE}} \gt 0$ whose solution meets every
row of $\mathcal{C}$, the tracking error is $\mathrm{IR}^{\star} / (2\lambda_{\mathrm{TE}})$.

**Proof.** Without $\mathcal{C}$, the utility is strictly concave on the budget-neutral $d$, and
its maximiser satisfies $\alpha - \eta \mathbf{1} = 2 \lambda_{\mathrm{TE}} \Sigma d$, where the
budget row again fixes the multiplier at $\eta$: the stationarity condition of the proof above
with $\nu$ replaced by $\lambda_{\mathrm{TE}}$. Hence
$d = \Sigma^{-1}\tilde\alpha / (2\lambda_{\mathrm{TE}})$, whose tracking error is
$\mathrm{IR}^{\star} / (2\lambda_{\mathrm{TE}})$, and which is $d^{\star}$ when
$\lambda_{\mathrm{TE}} = \nu$. $\square$

The weight that reproduces a budget is the multiplier of that budget, and it scales with the
alphas: alphas in percent need a weight 100 times larger. The
[constraints page](constraints.md#what-changes-when-a-limit-becomes-a-penalty) compares a hard
limit with a range of penalty weights.

### Group tracking-error limits

A `GroupTrackingErrorConstraint` limits the tracking error of each group's masked active weights
$d_g = L_g \odot d$; its definition is on the
[constraints page](constraints.md#group-tracking-error). In the forced problem each group with a
non-zero loading adds the hard cone $\lVert B^{\top} d_g \rVert_2 \leq \tau_g$ from
`group_tre_vols`, in addition to the total row: both are compiled, and either can bind. In the
utility form the group penalties $\lambda_g d_g^{\top}\Sigma d_g$ from `group_tre_utility_weights`
replace the total penalty. The ROSAA article sets its limits this way, per core asset class
(Sepp, Ossa and Kastenholz 2026, Equation 34). Proposition 1 does not hold once a group row binds.

### The yield-target variant

`cvx_maximise_alpha_with_target_return` adds a floor on the portfolio's yield,
$y^{\top} w \geq y_{\min}$. The floor applies to the absolute yield of the portfolio, not to the
active yield $y^{\top} d$, and `target_return` must be in the units of `yields`: annual decimals
here. The function has two paths.

- **Hard path**, `soft_tracking_error=False`, the default. It maximises
  $\alpha^{\top}(w - w^{\mathrm{bm}})$ under the yield floor and every row of
  `set_cvx_all_constraints`, including the tracking-error limit when one is set. Without a
  benchmark it maximises $\alpha^{\top} w$, and a tracking-error limit raises `ValueError`.
- **Soft path**, `soft_tracking_error=True` with a benchmark. It maximises the utility above under
  the yield floor and the utility-mode hard rows, so the floor takes priority over tracking error
  and the solve cannot fail on a tight budget. A populated `tracking_err_vol_constraint` is ignored
  in the solve and in the validation. Since 7.8.0 the turnover rule is: when neither
  `turnover_constraint` nor `group_turnover_constraint` is set, the turnover penalty
  $\lambda_{\mathrm{TO}}$ stays in the objective; when either is set, the configured caps are hard
  rows and the penalty is dropped. The [constraints page](constraints.md#solver-specific-utility-paths)
  and [turnover and transaction costs](turnover_and_transaction_costs.md#hard-limits-and-utility-penalties)
  state the same rule. Without a benchmark the flag has no effect and the hard path runs.

**Proposition 2 (a binding yield floor).** Let $\tau \gt 0$ and $\tilde\alpha \neq 0$, and suppose
that the solution $d^{\star}$ of Proposition 1 misses the floor,
$y^{\top}(w^{\mathrm{bm}} + d^{\star}) \lt y_{\min}$. Let $A = [\mathbf{1}, y]$ be the matrix of
size $N \times 2$, $q = (0, y_{\min} - y^{\top} w^{\mathrm{bm}})^{\top}$, and suppose that
$\bar\alpha \neq 0$, $\tau \geq \tau_y$ and that $w^{\mathrm{bm}} + d^{\star\star}$ meets every row
of $\mathcal{C}$, with the quantities below. The hard path's solution is

$$
d^{\star\star} = d^{y} + \sqrt{\tau^{2} - \tau_y^{2}} \frac{\Sigma^{-1}\bar\alpha}{\sqrt{\bar\alpha^{\top}\Sigma^{-1}\bar\alpha}}, \qquad d^{y} = \Sigma^{-1} A (A^{\top}\Sigma^{-1} A)^{-1} q, \qquad \bar\alpha = \alpha - A (A^{\top}\Sigma^{-1} A)^{-1} A^{\top}\Sigma^{-1}\alpha,
$$

with $\tau_y = \sqrt{(d^{y})^{\top}\Sigma d^{y}}$. When the benchmark itself misses the floor,
$\tau_y$ is the smallest tracking error that meets it, and for $\tau \lt \tau_y$ the problem is
infeasible.

**Proof.** Drop $\mathcal{C}$ as before. Because $d^{\star}$ is the unique optimum without the
floor and misses it, the floor binds at the optimum of the convex problem with it: an optimum with
a slack floor would also solve the problem without the floor. The optimum therefore maximises
$\alpha^{\top} d$ within the budget over the $d$ with $A^{\top} d = q$. Write $d = d^{y} + e$ with
$A^{\top} e = 0$. Then
$(d^{y})^{\top}\Sigma e = q^{\top} (A^{\top}\Sigma^{-1} A)^{-1} A^{\top} e = 0$, so
$d^{\top}\Sigma d = \tau_y^{2} + e^{\top}\Sigma e$, and $(\alpha - \bar\alpha)^{\top} e = 0$, so
$\alpha^{\top} d = \alpha^{\top} d^{y} + \bar\alpha^{\top} e$. The Cauchy–Schwarz inequality
bounds $\bar\alpha^{\top} e$ by $\sqrt{\bar\alpha^{\top}\Sigma^{-1}\bar\alpha}\sqrt{\tau^{2} - \tau_y^{2}}$,
with equality at the stated $e$, which satisfies $A^{\top} e = 0$ because
$A^{\top}\Sigma^{-1}\bar\alpha = 0$. Adding $\mathcal{C}$ back keeps this maximiser, as in
Proposition 1. When the yield gap $q_2$ is positive, every $d$ that meets the floor has
$\mathbf{1}^{\top} d = 0$ and $y^{\top} d \geq q_2$. The smallest $d^{\top}\Sigma d$ on that set
is attained at $y^{\top} d = q_2$, since scaling a candidate with $y^{\top} d \gt q_2$ towards zero
lowers its variance, and on $A^{\top} d = q$ it is attained by $d^{y}$, whose variance is
$\tau_y^{2}$. A budget below $\tau_y$ therefore leaves no feasible $d$. $\square$

The floor costs expected active return in two ways: the part $\tau_y$ of the budget buys yield
instead of alpha, and the rest follows $\bar\alpha$, the alphas net of what the budget and yield
rows absorb.

## Worked example

The canonical script of this page,
[`examples/docs/alpha_over_tracking_error.py`](../examples/docs/alpha_over_tracking_error.py),
runs offline and asserts every number and property stated here against an independent
computation: the closed forms by NumPy linear solves, tracking error as an explicit quadratic
form, and the soft yield-target solves as CVXPY problems written from raw arrays. It runs the
blocks below in order:

```console
python -m examples.docs.alpha_over_tracking_error
```

The six assets are government bonds, credit, three equity markets and gold, with annual
volatilities from 5% to 21%. The benchmark, the strategic allocation, holds 30% in government
bonds, 25% in US equity and 15%, 10%, 10% and 10% in the others. The alphas are annual expected
active returns, one tenth of each asset's volatility times a score between -1 and 1: 2.1% for EM
equity, 0.7% for credit, -1.6% for US equity, -0.85% for European equity, -0.25% for government
bonds and zero for gold. The script's `closed_form` computes Proposition 1 with NumPy.

### The forced solve and the closed form

A long-only solve with a 2% budget:

```python
covar = covariance()
benchmark = pd.Series(BENCHMARK, index=TICKERS)
alphas = pd.Series(ALPHAS, index=TICKERS)
constraints = op.Constraints(is_long_only=True, tracking_err_vol_constraint=TE_BUDGET)
weights, outcome = op.wrapper_maximise_alpha_over_tre(
    pd_covar=covar, alphas=alphas, benchmark_weights=benchmark, constraints=constraints)
active = weights - benchmark
reference, ir = closed_form(covar.to_numpy(), alphas.to_numpy(), TE_BUDGET)
assert outcome.accepted and outcome.compliant
assert np.abs(active.to_numpy() - reference).max() < WEIGHT_TOL
print((100 * pd.DataFrame({'solver': active, 'closed form': reference})).round(1))
```

The solver's active weights equal the closed form within $10^{-4}$ of net asset value. In
percentage points:

| Asset | Active weight |
|---|---:|
| Govt bonds | -14.3 |
| Credit | 23.1 |
| US equity | -13.1 |
| Europe equity | -5.1 |
| EM equity | 10.6 |
| Gold | -1.2 |

No long-only bound binds: the largest underweight, 14.3 points in government bonds, is below their
30% benchmark weight. The alpha of the minimum-variance portfolio, $\eta$, is -0.53%, and every
alpha counts from that level: gold, with zero alpha, sits 0.53% above it and is still
underweighted, through its correlations with the other assets. The script also checks that alphas
in percent, or shifted by 1%, give the same weights, that another benchmark gives the same active
weights, and that a monthly covariance with the budget divided by $\sqrt{12}$ gives the same solve.
Without a tracking-error limit the solve is a linear program and holds only EM equity.

`qis.RiskModel`, built by `build_risk_model` as on the
[risk analytics page](portfolio_risk_analytics.md#tracking-error-and-exposures-in-qis), measures the
ex-ante tracking error of the solve:

```python
date = pd.Timestamp('2024-12-31')  # a snapshot label for the risk model
risk_model = op.build_risk_model({date: covar})
tracking_error = risk_model.compute_tre_at_date(
    benchmark_weights=benchmark, portfolio_weights=weights, date=date)
active_return = float(alphas @ active)
print(f'{tracking_error:.3%} {active_return:.3%} {active_return / tracking_error:.3f}')
assert abs(tracking_error - TE_BUDGET) < TE_TOL
```

It prints `2.000% 0.673% 0.336`: the solve spends the budget, within $10^{-6}$, earns 0.67% of
expected active return, and its information ratio is $\mathrm{IR}^{\star} = 0.336$. The risk
model's number equals the explicit quadratic form.

### The information ratio and the budget

The script's `information_ratios` solves the forced problem at each budget, requires each solve to
be accepted and to spend its budget, and returns the expected active return over the tracking
error. With and without the long-only bounds:

```python
budgets = [0.01, 0.02, 0.03, 0.04, 0.06, 0.07]
ratios = pd.DataFrame({
    'long-only': information_ratios(covar, benchmark, alphas, budgets, long_only=True),
    'no bounds': information_ratios(covar, benchmark, alphas, budgets, long_only=False)},
    index=budgets)
print(ratios.round(3))
```

| Budget | Long-only | No bounds |
|---:|---:|---:|
| 1% | 0.336 | 0.336 |
| 2% | 0.336 | 0.336 |
| 3% | 0.336 | 0.336 |
| 4% | 0.335 | 0.336 |
| 6% | 0.272 | 0.336 |
| 7% | 0.247 | 0.336 |

Without bounds every solve is Proposition 1. With them, the first bound to bind is US equity's,
whose underweight reaches its 25% benchmark weight at a budget of 3.82%; at 6% the long-only solve
holds no US equity.

![Left: active weights of the long-only solve at a 2% tracking-error budget as bars, with the
closed-form weights as diamonds on the bars: government bonds -14.3, credit +23.1, US equity -13.1,
European equity -5.1, EM equity +10.6 and gold -1.2 percentage points. Right: information ratio
against the tracking-error budget from 0.25% to 8%. Without weight bounds it stays at 0.336; with
long-only bounds it stays there until US equity reaches zero at a 3.82% budget, then falls
steadily.](images/alpha_over_te_active_weights.png)

*Figure: the long-only solve against Proposition 1 at a 2% budget, and the information ratio of
the forced solve as the budget grows, with and without long-only bounds. Drawn by the `exhibit`
function of the canonical script; the [analytics gallery](analytics_gallery.md) lists its
provenance.*

> **Insight.** Until a bound binds, the tracking-error budget only scales the active weights. The
> information ratio is 0.336 at every budget up to 3.82%, so the expected active return is 0.336
> times the budget. Beyond it the long-only bound on US equity binds, and a larger budget buys
> expected active return at a falling ratio: 0.272 at 6%.

### The equivalent utility weight

The corollary's weight, $\mathrm{IR}^{\star} / (2\tau) = 8.41$ for the 2% budget, reproduces the
forced solve in utility mode:

```python
utility = op.Constraints(
    is_long_only=True, tre_utility_weight=ir / (2.0 * TE_BUDGET),
    constraint_enforcement_type=op.ConstraintEnforcementType.UTILITY_CONSTRAINTS)
utility_weights, utility_outcome = op.wrapper_maximise_alpha_over_tre(
    pd_covar=covar, alphas=alphas, benchmark_weights=benchmark, constraints=utility)
assert np.abs(utility_weights - weights).max() < WEIGHT_TOL
```

The script also checks that twice the weight halves the tracking error, that alphas in percent
need 100 times the weight for the same solve, that a tracking-error limit or a volatility cap on
the utility constraints changes nothing, and that `alphas=None` holds the benchmark. In the forced
form `alphas=None` raises `AttributeError`. A group object that puts every asset in one group with
the same weight gives the same solve whatever `tre_utility_weight` says. And a hard cap of 1% on
the tracking error of the three equity markets binds together with the 2% total budget and lowers
the expected active return.

### A yield floor

The annual yields are 3.5% for government bonds, 5% for credit, 1.5%, 3% and 2.5% for the US,
European and EM equity markets, and zero for gold. The benchmark yields 2.725% and the 2% solve
above yields 3.29%, so a floor of 3.5% binds:

```python
yields = pd.Series(YIELDS, index=TICKERS)
floored, floored_outcome = op.wrapper_maximise_alpha_with_target_return(
    pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
    constraints=constraints, benchmark_weights=benchmark)
floored_active = floored - benchmark
print(f'{yields @ benchmark:.3%} {yields @ weights:.2%} {yields @ floored:.2%}',
      f'{alphas @ floored_active:.2%}')
```

It prints `2.725% 3.29% 3.50% 0.64%`. The solve meets the floor, spends the whole 2% budget and
equals Proposition 2 within $10^{-4}$. Reaching the floor takes 1.61% of tracking error, the
$\tau_y$ of Proposition 2, and the expected active return falls by 3 basis points, from 0.67% to
0.64%. With a 1.5% budget the hard path is infeasible and falls back to `weights_0`; a floor of
3%, which the solve already meets, changes nothing. Without a benchmark the hard path maximises $\alpha^{\top} w$: it
holds 60% in EM equity and 40% in credit, the mix that yields exactly 3.5%, and it raises
`ValueError` when the constraints carry a tracking-error limit. A missing yield counts as zero,
with a warning.

The soft path keeps the floor hard and prices tracking error with the equivalent weight. From
holdings at the benchmark, once with the default turnover penalty and once without it:

```python
soft = op.Constraints(is_long_only=True, tre_utility_weight=ir / (2.0 * TE_BUDGET))
penalised, penalised_outcome = op.wrapper_maximise_alpha_with_target_return(
    pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
    constraints=soft, benchmark_weights=benchmark, soft_tracking_error=True,
    weights_0=benchmark)
unpenalised, _ = op.wrapper_maximise_alpha_with_target_return(
    pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
    constraints=soft.copy(turnover_utility_weight=None), benchmark_weights=benchmark,
    soft_tracking_error=True, weights_0=benchmark)
for portfolio in (penalised, unpenalised):
    print(f'{(portfolio - benchmark).abs().sum():.0%} {alphas @ (portfolio - benchmark):.2%}')
```

It prints `36% 0.25%` and `81% 0.76%`: turnover and expected active return. Both solves equal
independent CVXPY solves and meet the floor. With the penalty, the solve is the trade of least
turnover that adds the 0.775% of yield the floor needs: it buys credit and sells US equity and
gold. Without it, tracking error rises to 2.28%, above the 2% that the same weight gives without
the floor. Without `weights_0` the penalty has no baseline and is skipped. A hard turnover cap of
50% replaces the penalty, as the 7.8.0 rule states, and the solve then uses the whole cap.

> **Pitfall.** `Constraints` carries `turnover_utility_weight=0.40` by default, and both utility
> paths apply it whenever `weights_0` is known, which in a rolling run is every date after the
> first. Against alphas in annual decimals the penalty dominates: from holdings at the benchmark,
> the soft yield-target solve trades only the 36% of turnover that the 3.5% floor forces and earns
> 0.25% of expected active return, against 81% and 0.76% with `turnover_utility_weight=None`, and
> the alpha-over-tracking-error utility solve does not trade at all. Scale the penalty to the
> alphas, or set it to None.

## Implementation in optimalportfolios

The seven functions are in
[`maximise_alpha_over_tre.py`](../src/optimalportfolios/optimization/taa/maximise_alpha_over_tre.py)
and
[`maximise_alpha_with_target_yield.py`](../src/optimalportfolios/optimization/taa/maximise_alpha_with_target_yield.py)
of `optimalportfolios.optimization.taa`, and all are importable from `optimalportfolios`. Each
family has the three layers described in [choosing an objective](optimization_module_readme.md).

| Entry point | Arguments, then keyword defaults | What it does and returns |
|---|---|---|
| `cvx_maximise_alpha_over_tre` | `covar`, `alphas`, `constraints`; `solver='CLARABEL'`, `verbose=False`, `context=''`, `diagnose=False`, `factorize_covar=True` | Solves the forced problem on NumPy inputs and constraints already aligned with a benchmark; it needs alphas. Returns an `OptimizationOutcome` |
| `cvx_maximise_tre_utility` | `covar`, `constraints`, `alphas=None`; the same keyword arguments | Solves the utility form. Returns an `OptimizationOutcome` |
| `wrapper_maximise_alpha_over_tre` | `pd_covar`, `alphas`, `benchmark_weights`, `constraints`; `weights_0=None`, `rebalancing_indicators=None`, `optimiser_config=OptimiserConfig()`, `context=''` | Removes the assets with a zero or missing variance or a missing alpha, aligns the constraints with `update_with_valid_tickers`, injecting the benchmark and the current weights, and routes `UTILITY_CONSTRAINTS` to `cvx_maximise_tre_utility` and any other `constraint_enforcement_type` to `cvx_maximise_alpha_over_tre`. Returns the weights on the original labels, zero for removed assets, and the outcome |
| `rolling_maximise_alpha_over_tre` | `prices`, `alphas`, `constraints`, `benchmark_weights`, `covar_dict`; `rebalancing_indicators=None`, `optimiser_config=OptimiserConfig()`, `benchmark_beta_loadings=None` | Solves at each key of `covar_dict`, in insertion order. Forward-fills the alphas and sets a missing alpha to zero, so the asset stays in the solve without a view; uses a Series benchmark at every date and forward-fills a DataFrame; passes the previous weights, drifted to the date, as `weights_0`; requires per-date `benchmark_beta_loadings` with a `benchmark_beta_constraint`. Returns a table of weights with one column per price, without the outcomes |
| `cvx_maximise_alpha_with_target_return` | `covar`, `alphas`, `constraints`; `soft_tracking_error=False`, `verbose=False`, `solver='CLARABEL'`, `context=''`, `factorize_covar=True` | Solves the hard or the soft path on aligned constraints that carry `asset_returns` and `target_return`. Returns an `OptimizationOutcome` |
| `wrapper_maximise_alpha_with_target_return` | `pd_covar`, `alphas`, `yields`, `target_return`, `constraints`; `benchmark_weights=None`, `soft_tracking_error=False`, `weights_0=None`, `optimiser_config=OptimiserConfig(apply_total_to_good_ratio=True)`, `context=''` | Filters as above, passes `yields` as `asset_returns` and sets a missing yield to zero with a warning. Returns the weights and the outcome |
| `rolling_maximise_alpha_with_target_return` | `prices`, `alphas`, `yields`, `target_returns`, `constraints`, `covar_dict`; `benchmark_weights=None`, `soft_tracking_error=False`, `optimiser_config=OptimiserConfig(apply_total_to_good_ratio=True)` | Forward-fills alphas, yields, targets and the benchmark onto the keys of `covar_dict`. Returns the weights, with one outcome record per date in `attrs['optimization_outcomes']` |

The two wrapper families differ in their default `apply_total_to_good_ratio`, `False` for alpha
over tracking error and `True` for the yield target; the
[conventions page](conventions.md#solvers-and-outcomes) lists the defaults of every entry point.

### Validation, acceptance and fallback

Filtering, the covariance factorisation, the acceptance checks and the fallback order are shared
with every CVXPY solver and are described in
[solver numerics and outcomes](solver_numerics_and_outcomes.md). What is specific to these
wrappers:

- Only `wrapper_maximise_alpha_over_tre` runs the pre-solve input contract of
  `OptimiserConfig.validate_inputs`, the elastic diagnosis of `diagnose_infeasibility` after a
  rejected solve, and the logging threshold `max_constraint_relaxation`, and only it and its rolling
  function accept `rebalancing_indicators` to freeze positions. The yield-target wrapper ignores
  the three fields and cannot freeze.
- The tracking-error row alone never makes the problem infeasible, because the benchmark has zero
  tracking error. Infeasibility comes from the rows the benchmark itself does not meet: a yield
  floor out of reach of the budget, as in the example, a turnover cap from holdings far from the
  benchmark, or boxes and groups that exclude it.
- A rejected solve returns `weights_0`, else the benchmark. `wrapper_maximise_alpha_over_tre`
  always injects a benchmark, so without current weights its fallback is the benchmark itself,
  with zero tracking error.

## Interpretation and limitations

- The objective is expected active return, not a ratio. Proposition 1 shows that the solve also
  maximises the information ratio while no bound or group row binds; beyond that, as the table
  shows, a larger budget buys expected active return at a falling ratio.
- The closed form trades spreads between correlated assets, because they cost little tracking
  error: at a 2% budget it moves 37 points between credit and government bonds. Box bounds, group
  tracking-error limits and turnover limits keep such spreads within a mandate, at the price of
  Proposition 1.
- The forced solve depends only on the direction of the alphas, so alpha scores can be used as
  they are. The utility weights depend on their scale: the weight that reproduces a budget is
  $\mathrm{IR}^{\star} / (2\tau)$ in the units of the alphas, and the default penalties
  `tre_utility_weight=1.0` and `turnover_utility_weight=0.40` are not calibrated to any of them.
- The tracking error is ex ante, under the supplied covariance; it is not a forecast of realised
  tracking error. A covariance assembled with `residual_var_weight` below one understates it, as
  the [ROSAA case study](app_rosaa_multi_asset_allocation.md) warns.
- Roll (1992) shows that portfolios chosen for expected active return and tracking error are not,
  in general, efficient in total return and can carry more total risk than their benchmark. The
  budget does not control total volatility: the example's tactical allocation has 8.64% against
  the benchmark's 8.20%. `max_target_portfolio_vol_an` adds a hard volatility cap in the forced
  problem; the utility form has none.
- In `rolling_maximise_alpha_over_tre`, a dated benchmark that starts after the first key of
  `covar_dict` is all zeros at the earlier dates. The tracking-error row then caps total
  volatility, which no fully invested portfolio of the example can bring to 2%, and the fallback
  returns the zero benchmark: the script's rolling check returns zero weights at that date. A
  missing benchmark column is likewise a zero weight, without renormalisation. Start the benchmark
  at or before the first decision and give it every asset.
- The implementation does not inherit everything from its sources. Unlike Grinold and Kahn (2000),
  it controls total active risk, including active beta, rather than residual risk, takes alphas as
  given without refining them, and has no transaction-cost model beyond the L1 penalty. Unlike the
  ROSAA article, which measures tracking error against the strategic allocation with the
  covariance of the joint strategic and tactical universes (Equation 33), it needs the benchmark
  on the covariance's assets, and it gives each group its own penalty weight where the article's
  soft form (Equation 38) uses one. The article's objective $\alpha^{\top} w$ (Equation 32) has
  the same maximiser as $\alpha^{\top} d$ under full investment.

## See also

- [Minimum tracking error](minimum_tracking_error.md)
- [Portfolio constraints](constraints.md)
- [Turnover and transaction costs](turnover_and_transaction_costs.md)
- [Ex-ante risk contributions and betas](portfolio_risk_analytics.md)
- [Strategic and tactical allocation with HCGL covariance (ROSAA)](app_rosaa_multi_asset_allocation.md)
- [Alpha signals](alphas_module_readme.md)
- [Solver numerics and outcomes](solver_numerics_and_outcomes.md)
- [Choosing an objective](optimization_module_readme.md)

## References

- Grinold, R. C. and Kahn, R. N. (2000). *Active Portfolio Management: A Quantitative Approach for
  Producing Superior Returns and Controlling Risk*, 2nd edition. McGraw-Hill. ISBN 0-07-024882-6.
  The information ratio, active risk and the optimal active holdings of an unconstrained manager.
- Roll, R. (1992). *A Mean/Variance Analysis of Tracking Error*. The Journal of Portfolio
  Management, 18(4), 13–22. [DOI 10.3905/jpm.1992.701922](https://doi.org/10.3905/jpm.1992.701922).
- Sepp, A., Ossa, I. and Kastenholz, M. (2026). *Robust Optimization of Strategic and Tactical
  Asset Allocation for Multi-Asset Portfolios*. The Journal of Portfolio Management, 52(4),
  86–120. [DOI 10.3905/jpm.2025.1.806](https://doi.org/10.3905/jpm.2025.1.806);
  [author-shared copy](https://eprints.pm-research.com/17511/143431/index.html). The section
  "Optimization of TAA Portfolio", Equations 32 to 38, states the tactical problem: expected alpha
  under a tracking-error limit against the strategic allocation, per asset class, with group
  turnover limits, bounds and exposure rows, and a soft form with penalties.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
