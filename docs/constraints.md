# Portfolio constraints

This guide documents the complete constraint contract used by
`optimalportfolios.optimization.constraints`: the mathematical rows, required inputs,
hard-versus-utility behavior, backend coverage, universe alignment, and post-solve analytics.
The examples are synthetic, deterministic, and require no market-data connection.

The central object is the immutable `Constraints` dataclass. It describes policy; a backend
compiler translates the supported parts of that policy into CVXPY, SciPy, or risk-budgeting
solver rows. This separation matters: a field existing on `Constraints` does not imply that every
backend enforces it.

```python
from optimalportfolios.optimization.constraints import (
    BenchmarkBetaConstraint,
    BenchmarkDeviationConstraints,
    ConstraintEnforcementType,
    Constraints,
    GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
    evaluate_constraint_residuals,
)
```

## Notation and units

The formulas below use:

| Symbol | Meaning |
|---|---|
| {math}`w` | portfolio weights in solver order |
| {math}`b` | benchmark weights (`benchmark_weights`) |
| {math}`w_0` | current weights before the proposed trade (`weights_0`) |
| {math}`\mu` | expected asset returns (`asset_returns`) |
| {math}`\Sigma` | covariance matrix |
| {math}`L` | an assets-by-groups or assets-by-factors loading matrix |
| {math}`L_g` | the loading vector for one group or factor |
| {math}`\odot` | elementwise multiplication |

Weights, exposure, and turnover are fractions of NAV. `0.05` therefore means 5%. Exposure is
**signed net exposure** {math}`\sum_i w_i`, not gross leverage {math}`\sum_i \lvert w_i\rvert`. For
example, `[1.20, -0.20]` has net exposure `1.00` and gross leverage `1.40`.

The constraint layer does no resampling or annualisation. Keep the following units consistent:

- `asset_returns` and `target_return` must use the same horizon and scaling;
- covariance entries are variance in the caller's chosen units;
- volatility and tracking-error limits use the square-root units of that covariance;
- utility coefficients depend on the units of both alpha and the penalized quantity.

Despite its historical `_an` suffix, `max_target_portfolio_vol_an` does not annualise either the
covariance or the limit. If the covariance is annualized, supply an annualized limit; if it is
monthly, supply a monthly limit.

## The constraint map

Some `Constraints` fields are policy limits and some are supporting state required to evaluate a
limit.

| Field | Mathematical contract | Required state |
|---|---|---|
| `is_long_only` | {math}`w_i \ge 0` | none |
| `min_weights`, `max_weights` | {math}`w_i^{min} \le w_i \le w_i^{max}` | aligned asset index |
| `min_exposure`, `max_exposure` | {math}`E_{min} \le \sum_i w_i \le E_{max}` | none |
| `target_return` | {math}`\mu^\top w \ge r_{target}` | `asset_returns` |
| `max_target_portfolio_vol_an` | {math}`\sqrt{w^\top\Sigma w} \le \sigma_{max}` | covariance or factorization |
| `tracking_err_vol_constraint` | {math}`\sqrt{(w-b)^\top\Sigma(w-b)} \le \tau` | benchmark and covariance |
| `turnover_constraint` | {math}`\sum_i \lvert c_i(w_i-w_{0,i})\rvert \le T` | `weights_0`; costs {math}`c_i` are optional |
| `group_lower_upper_constraints` | {math}`l_g \le L_g^\top w \le u_g` | group loadings |
| `group_tracking_error_constraint` | {math}`\sqrt{a_g^\top\Sigma a_g} \le \tau_g`, {math}`a_g=L_g\odot(w-b)` | benchmark and covariance |
| `group_turnover_constraint` | {math}`\sum_i \lvert L_{ig}(w_i-w_{0,i})\rvert \le T_g` | `weights_0` |
| `sector_deviation_constraints` | {math}`\lvert L_g^\top(w-b)\rvert \le d_g` | benchmark |
| `style_deviation_constraints` | {math}`\lvert L_g^\top(w-b)\rvert \le d_g` | benchmark |
| `benchmark_beta_constraint` | {math}`\beta_{min} \le c^\top w \le \beta_{max}` | per-date beta loadings {math}`c` |

The supporting fields `benchmark_weights`, `weights_0`, `asset_returns`, and `turnover_costs` do
nothing by themselves. They supply the reference vectors needed by another configured limit or
utility term.

`Constraints()` defaults to a long-only, fully invested portfolio: {math}`w_i\ge0` and
{math}`\sum_iw_i=1`. There are no explicit per-name boxes unless they are supplied.

## Exposure, long-only, and instrument boxes

CVXPY compiles three independent pieces:

```{math}
w_i \ge 0 \quad \text{when is_long_only is true},
```

```{math}
E_{min} \le \sum_i w_i \le E_{max},
```

```{math}
w_i^{min} \le w_i \le w_i^{max}.
```

Exactly equal stored exposure limits produce one equality. Distinct values, even if numerically
close, remain an exposure band. Thus `min_exposure=1.0, max_exposure=1.000005` is not treated as
an equality.

```python
import pandas as pd

assets = pd.Index(["A", "B"])
box = Constraints(
    is_long_only=True,
    min_exposure=0.80,
    max_exposure=1.00,
    min_weights=pd.Series([0.10, 0.00], index=assets),
    max_weights=pd.Series([0.70, 0.80], index=assets),
)
```

The candidate `[0.60, 0.30]` passes: total exposure is `0.90`, A is in `[0.10, 0.70]`, and B is in
`[0.00, 0.80]`.

For Charnes--Cooper transformations, `exposure_scaler=k` scales exposure, per-name boxes, and
group-allocation rows. Other rows are not automatically homogenized; the scaler is an internal
solver feature, not a general way to lever every constraint family.

### Constructor validation

`Constraints` rejects:

- `min_weights > max_weights + 1e-10` when the two Series have exactly equal indexes;
- a long-only specification with `min_weights < -1e-10`;
- several loading-weighted group/box contradictions described under
  [Feasibility and diagnostics](#feasibility-and-diagnostics).

Always align both the labels and their order before calling a low-level compiler. Several rows
ultimately use NumPy arrays, which cannot recover a misplaced pandas label.

## Target return and portfolio volatility

### Minimum target return

The target-return floor is linear and remains hard in both enforcement modes:

```{math}
\mu^\top w \ge r_{target}.
```

```python
returns = pd.Series({"A": 0.08, "B": 0.02})
return_floor = Constraints(asset_returns=returns, target_return=0.05)
```

`w=[0.50, 0.50]` produces `0.05` and passes. `w=[0.40, 0.60]` produces `0.044` and fails. A
configured `target_return` without `asset_returns` raises `ValueError` during CVXPY compilation.

### Maximum portfolio volatility

Without a covariance factorization, the hard row is:

```{math}
w^\top\Sigma w \le \sigma_{max}^2.
```

With a factor {math}`B` satisfying {math}`BB^\top=\Sigma_{stabilized}`, the same policy is
compiled as a second-order cone:

```{math}
\|B^\top w\|_2 \le \sigma_{max}.
```

For {math}`\Sigma=\operatorname{diag}(0.04,0.01)` and `w=[0.5,0.5]`, volatility is
`sqrt(0.0125) = 0.1118`; a `0.12` ceiling passes.

The factorization takes precedence if both `covar` and `covar_factorization` are supplied. This is
also true in the residual analytics, so the audit measures the exact stabilized geometry used by
the solver.

`set_cvx_all_constraints()` always emits a configured volatility cap, regardless of the enum
stored on the object. The generic utility compiler omits the cap. Utility SAA solvers use their
own variance objective or penalty; they do not convert `max_target_portfolio_vol_an` into a
penalty coefficient.

## Benchmark-relative risk

Let active weights be {math}`a=w-b`.

### Total tracking error

The hard limit is:

```{math}
\operatorname{TE}(w,b)=\sqrt{a^\top\Sigma a}\le\tau.
```

`benchmark_weights` and a covariance or factorization are required. In factorized form the hard
row is {math}`\lVert B^\top a\rVert_2\le\tau`.

### Group tracking error

`GroupTrackingErrorConstraint` masks the active vector before computing risk:

```{math}
a_g=L_g\odot(w-b), \qquad
\operatorname{TE}_g=\sqrt{a_g^\top\Sigma a_g}\le\tau_g.
```

This is not the same as multiplying total TE by a group's portfolio weight. Cross-covariances
inside the masked vector remain part of the calculation.

```python
group_tre = GroupTrackingErrorConstraint(
    group_loadings=pd.DataFrame(
        {
            "Growth": [1.0, 0.0, 0.0],
            "Defensive": [0.0, 1.0, 1.0],
        },
        index=["Equity", "Bond", "Gold"],
    ),
    group_tre_vols=pd.Series({"Growth": 0.05, "Defensive": 0.05}),
)
```

In forced mode, total and group TE caps are additive policy controls: configuring both emits both.
In the generic utility builder, the presence of a group-TE object gives group penalties precedence
over the total TE penalty. See [Hard and utility enforcement](#hard-and-utility-enforcement).

At least one of `group_tre_vols` and `group_tre_utility_weights` is required. Supply finite,
non-negative values and exact coverage for every loaded group used by the selected mode. Missing
coverage warns during construction but can raise on `.loc[group]` during compilation. A `NaN`
utility coefficient skips that penalty; a `NaN` hard cap is not an intentional skip.

## Group allocation and benchmark deviations

All loading matrices are oriented **assets by groups/factors**: asset labels on rows, group names
on columns.

Use explicit zero for “not loaded.” Existing `NaN` loading values are generally unsafe: nested
alignment selects rows but does not universally replace missing cells before expressions are
compiled.

### Absolute group allocation

`GroupLowerUpperConstraints` applies linear floors and ceilings:

```{math}
l_g \le L_g^\top w \le u_g.
```

```python
group_allocation = GroupLowerUpperConstraints(
    group_loadings=pd.DataFrame(
        {
            "Growth": [1.0, 0.0, 0.0],
            "Defensive": [0.0, 1.0, 1.0],
        },
        index=["Equity", "Bond", "Gold"],
    ),
    group_min_allocation=pd.Series({"Growth": 0.30, "Defensive": 0.45}),
    group_max_allocation=pd.Series({"Growth": 0.55, "Defensive": 0.70}),
)
```

For `w=[0.45,0.40,0.15]`, Growth is `0.45` and Defensive is `0.55`; both pass.

Loadings need not be binary. Fractional loadings describe partial exposure, and signed loadings
describe a linear spread. A signed column still compiles as {math}`L_g^\top w`; it is not
silently converted to membership.

Columns that are exactly all zero or all missing are dropped at construction. A column that is
only numerically close to zero remains stored but emits no solver row. Missing allocation bounds
are reindexed to `NaN`, warned about, and skipped; this permits a group to have only a lower or
only an upper bound.

Use finite bounds. The constructor checks several box-versus-group reachability conditions, but it
does not directly reject `group_min_allocation > group_max_allocation` when no other check exposes
the contradiction.

`merge_group_lower_upper_constraints(first, second)` combines two loading systems. Overlapping
group names receive `_1` and `_2` suffixes, missing asset loadings are filled with zero, and a
missing side remains `NaN` rather than inventing a bound.

### Sector and style deviations

`BenchmarkDeviationConstraints` implements the same formula for both fields:

```{math}
|L_g^\top(w-b)|\le d_g.
```

The difference is interpretation:

- sector loadings are usually binary, so the result is active sector weight;
- style loadings are usually continuous, so the result is an active factor exposure in the
  loading's own scale.

```python
sector_deviation = BenchmarkDeviationConstraints(
    factor_loading_mat=pd.DataFrame(
        {"Risk assets": [1.0, 0.0, 1.0]},
        index=["Equity", "Bond", "Gold"],
    ),
    factor_max_deviation=pd.Series({"Risk assets": 0.08}),
)

style_deviation = BenchmarkDeviationConstraints(
    factor_loading_mat=pd.DataFrame(
        {"Inflation": [0.5, -0.5, 1.0]},
        index=["Equity", "Bond", "Gold"],
    ),
    factor_max_deviation=pd.Series({"Inflation": 0.12}),
)
```

With benchmark `[0.45,0.40,0.15]` and portfolio `[0.50,0.35,0.15]`, active Risk-assets exposure
is `0.05` and passes the `0.08` limit. These deviations remain hard in utility mode.

Supply every bound label as a loading column and use finite limits. Missing labels warn at
construction but can fail later during compilation; unlike group-allocation bounds, a `NaN`
deviation limit is not an intentional skip contract.

## Turnover and trading constraints

### Total turnover

Without cost multipliers, the hard budget is full L1 turnover:

```{math}
\sum_i|w_i-w_{0,i}|\le T.
```

There is no factor of one half. Moving 10% from A to B has turnover `0.20`:

```text
w0 = [0.60, 0.40]
w  = [0.50, 0.50]
L1 = |-0.10| + |+0.10| = 0.20
```

With `turnover_costs=c`, the quantity becomes:

```{math}
\sum_i|c_i(w_i-w_{0,i})|.
```

For costs `[2,1]`, the same trade has weighted turnover `0.30`. These values are multipliers in
the constraint or utility penalty; this layer does not subtract transaction costs from portfolio
NAV.

If `weights_0` is absent, total and group turnover rows are skipped with a debug log. That is not
equivalent to assuming zero starting weights.

### Group turnover

For each group:

```{math}
\sum_i|L_{ig}(w_i-w_{0,i})|\le T_g.
```

```python
group_turnover = GroupTurnoverConstraint(
    group_loadings=pd.DataFrame(
        {
            "Growth": [1.0, 0.0, 0.0],
            "Defensive": [0.0, 1.0, 1.0],
        },
        index=["Equity", "Bond", "Gold"],
    ),
    group_max_turnover=pd.Series({"Growth": 0.15, "Defensive": 0.15}),
)
```

Group turnover does not apply the portfolio-level `turnover_costs`. Loadings themselves provide
the group multipliers. In forced mode, group and total turnover caps are both emitted when both
are configured.

Loading signs disappear inside the absolute value, buys and sells do not net, and overlapping
groups can count one asset's trade more than once. At least one of `group_max_turnover` and
`group_turnover_utility_weights` is required. As with group TE, configure finite, non-negative
values with complete group coverage; missing hard labels can fail at compilation, while a `NaN`
utility coefficient skips that group's penalty.

## Benchmark beta

`BenchmarkBetaConstraint` is linear after per-asset beta loadings {math}`c` have been computed:

```{math}
\beta(w)=c^\top w, \qquad
\beta_{min}\le c^\top w\le\beta_{max}.
```

At least one side of the range is required. A rolling optimizer should keep static bounds and
inject the current date's loadings with `.with_loadings(...)` before compilation.

When portfolio assets and benchmark constituents are in one joint covariance, loadings are:

```{math}
c=\frac{\Sigma_{assets,constituents}b}
        {b^\top\Sigma_{constituents,constituents}b}.
```

```python
from optimalportfolios.optimization.constraints import (
    compute_benchmark_beta_loadings_from_covar,
)

assets = pd.Index(["Equity", "Bond", "Gold"])
covar = pd.DataFrame(
    [
        [0.0324, 0.0018, 0.0024],
        [0.0018, 0.0064, 0.0006],
        [0.0024, 0.0006, 0.0144],
    ],
    index=assets,
    columns=assets,
)
benchmark = pd.Series([0.45, 0.40, 0.15], index=assets)

beta_loadings = compute_benchmark_beta_loadings_from_covar(
    covar=covar,
    benchmark_weights=benchmark,
    asset_tickers=assets.tolist(),
)
beta_constraint = BenchmarkBetaConstraint(
    beta_min=0.85,
    beta_max=1.15,
).with_loadings(beta_loadings)
```

Because the benchmark is constructed from the same constituents, `beta_loadings @ benchmark` is
one, up to floating-point precision. The helper validates that benchmark variance is positive and
finite. `compute_benchmark_beta_loadings(...)` provides the analogous calculation for a factor
model, including optional benchmark idiosyncratic variance.

Beta remains hard in utility mode. Compiling before loadings are injected raises `ValueError`.
The constrained quantity is absolute portfolio beta, not active beta relative to the benchmark.

(hard-and-utility-enforcement)=
## Hard and utility enforcement

`ConstraintEnforcementType` supports two policy interpretations in the CVXPY SAA/TAA paths.

### Forced constraints

`FORCED_CONSTRAINTS` uses limit values as feasibility rows. In particular:

- total and group TE caps are both hard and additive;
- total and group turnover caps are both hard and additive;
- portfolio volatility is a hard cap;
- exposure, boxes, target return, group allocation, deviations, and beta are hard.

### Utility constraints

The generic utility builder keeps mandate rows hard but turns tracking error and turnover into
objective trade-offs. With total penalties its maximization objective has the form:

```{math}
\alpha^\top(w-b)
-\lambda_{TE}(w-b)^\top\Sigma(w-b)
-\lambda_{TO}\sum_i|c_i(w_i-w_{0,i})|.
```

Tracking error is penalized as **variance**, not volatility. Group TE similarly contributes
{math}`-\lambda_g a_g^\top\Sigma a_g`. Turnover is penalized as full L1 turnover.

The hard rows retained by the generic utility builder are:

- long-only, exposure, and per-name boxes;
- target return;
- group allocation;
- sector and style deviations;
- benchmark beta.

The generic builder does not retain a configured maximum-volatility cap. Individual utility
solvers may add absolute or active variance in their own objective; consult the selected solver's
docstring rather than interpreting a target-volatility field as a lambda.

### Group precedence

Utility precedence is based on object presence:

- if `group_tracking_error_constraint` exists, its group penalties are used and the total
  `tre_utility_weight` penalty is not added;
- if `group_turnover_constraint` exists, its group penalties are used and the total
  `turnover_utility_weight` penalty is not added.

This differs intentionally from forced mode, where group and total caps are additive. A group
object containing only hard caps is not reinterpreted in utility mode: it must also carry
`group_tre_utility_weights` or `group_turnover_utility_weights`. Conversely, a utility-only group
object cannot be sent to the forced compiler. There is no fallback between the two meanings.

Hard cap magnitudes do not calibrate utility strength. These are separate inputs:

| Hard/reporting value | Utility coefficient |
|---|---|
| `tracking_err_vol_constraint` | `tre_utility_weight` |
| `group_tre_vols` | `group_tre_utility_weights` |
| `turnover_constraint` | `turnover_utility_weight` |
| `group_max_turnover` | `group_turnover_utility_weights` |

Finally, the enum is descriptive state used by wrappers and analytics; it does not dispatch a
low-level method. `set_cvx_all_constraints()` always builds hard rows. For utility behavior, call
`set_cvx_utility_objective_constraints()` or a public utility solver.

### Solver-specific utility paths

The rules above describe the shared generic builder. Public solvers that already own a risk
objective use narrower paths:

| Utility solver/path | Risk term | Trading term | Important detail |
|---|---|---|---|
| alpha over tracking error | shared total/group TE utility | shared total/group turnover utility | follows group-over-total precedence exactly |
| max return at target volatility, benchmark-relative | active variance penalty | shared utility path | `target_vol` is not converted into lambda or retained as a hard cap |
| max return at target volatility, absolute | portfolio variance penalty | total turnover penalty | manual path; group utility precedence is not used |
| min variance at target return | portfolio or active variance is the objective | total turnover penalty | return floor stays hard; `tre_utility_weight` is not added |
| alpha with target return and `soft_tracking_error=True` | total TE utility | total and group turnover are explicitly re-added as hard caps | the flag is separate from the enforcement enum and is active only with a benchmark |

This is why the backend table says “generic utility” and why a target-volatility argument should
not be read as a universal penalty calibration.

## Backend capability matrix

| Constraint family | CVXPY forced | CVXPY generic utility | SciPy | Risk budgeting / PyRB |
|---|---|---|---|---|
| Long-only and boxes | Hard | Hard | Bounds/callback | Bounds |
| Min/max net exposure | Hard | Hard | Two callbacks | Full investment is solver policy; arbitrary bands are not compiled |
| Target return | Hard | Hard | Unsupported | Unsupported |
| Portfolio volatility | Hard cap | No generic cap; solver-specific risk objective | Unsupported | Unsupported |
| Total and group TE | Both hard | Soft; group object takes precedence | Unsupported | Unsupported |
| Total and group turnover | Both hard | Soft; group object takes precedence | Unsupported | Unsupported |
| Group allocation | Hard | Hard | Callbacks | Matrix rows |
| Sector/style deviation | Hard | Hard | Unsupported | Unsupported |
| Benchmark beta | Hard | Hard | Unsupported | Unsupported |

“Unsupported” means the backend compiler emits no row for that field. It does not mean the
backend approximates the policy, and a post-solve report cannot retroactively make the solve
constrained.

SciPy has two box defaults worth knowing:

- with no explicit box side, long-only uses `[0,1]` per asset and long/short uses no bounds;
- if either side is supplied, a missing lower side becomes `0` for long-only or `-inf` for
  long/short, and a missing upper side becomes `1`.

SciPy represents an exact exposure target as two opposite inequalities. PyRB receives box bounds
and group rows {math}`-L_g^\top w\le-l_g` and {math}`L_g^\top w\le u_g`; its risk-budgeting
solver and validator own the full-investment contract.

## A complete forced-constraint example

The following problem intentionally includes every constraint family. The covariance and all
risk limits use annualized units in this example; changing to monthly units would require changing
both together.

```python
import cvxpy as cvx
import numpy as np
import pandas as pd

from optimalportfolios.optimization.constraints import (
    BenchmarkBetaConstraint,
    BenchmarkDeviationConstraints,
    Constraints,
    GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
    compute_benchmark_beta_loadings_from_covar,
    evaluate_constraint_residuals,
)

assets = pd.Index(["Equity", "Bond", "Gold"], name="asset")
covar = pd.DataFrame(
    [
        [0.0324, 0.0018, 0.0024],
        [0.0018, 0.0064, 0.0006],
        [0.0024, 0.0006, 0.0144],
    ],
    index=assets,
    columns=assets,
)
benchmark = pd.Series([0.45, 0.40, 0.15], index=assets)
current = pd.Series([0.40, 0.45, 0.15], index=assets)
expected_returns = pd.Series([0.070, 0.035, 0.040], index=assets)

groups = pd.DataFrame(
    {
        "Growth": [1.0, 0.0, 0.0],
        "Defensive": [0.0, 1.0, 1.0],
    },
    index=assets,
)
beta_loadings = compute_benchmark_beta_loadings_from_covar(
    covar=covar,
    benchmark_weights=benchmark,
    asset_tickers=assets.tolist(),
)

constraints = Constraints(
    min_weights=pd.Series([0.20, 0.20, 0.05], index=assets),
    max_weights=pd.Series([0.60, 0.65, 0.25], index=assets),
    min_exposure=1.0,
    max_exposure=1.0,
    benchmark_weights=benchmark,
    tracking_err_vol_constraint=0.06,
    weights_0=current,
    turnover_constraint=0.25,
    turnover_costs=pd.Series([1.0, 0.5, 2.0], index=assets),
    target_return=0.045,
    asset_returns=expected_returns,
    max_target_portfolio_vol_an=0.13,
    group_lower_upper_constraints=GroupLowerUpperConstraints(
        group_loadings=groups,
        group_min_allocation=pd.Series({"Growth": 0.30, "Defensive": 0.45}),
        group_max_allocation=pd.Series({"Growth": 0.55, "Defensive": 0.70}),
    ),
    group_tracking_error_constraint=GroupTrackingErrorConstraint(
        group_loadings=groups,
        group_tre_vols=pd.Series({"Growth": 0.05, "Defensive": 0.05}),
        group_tre_utility_weights=pd.Series({"Growth": 5.0, "Defensive": 5.0}),
    ),
    group_turnover_constraint=GroupTurnoverConstraint(
        group_loadings=groups,
        group_max_turnover=pd.Series({"Growth": 0.15, "Defensive": 0.15}),
        group_turnover_utility_weights=pd.Series({"Growth": 0.02, "Defensive": 0.02}),
    ),
    sector_deviation_constraints=BenchmarkDeviationConstraints(
        factor_loading_mat=pd.DataFrame(
            {"Risk assets": [1.0, 0.0, 1.0]}, index=assets
        ),
        factor_max_deviation=pd.Series({"Risk assets": 0.08}),
    ),
    style_deviation_constraints=BenchmarkDeviationConstraints(
        factor_loading_mat=pd.DataFrame(
            {"Inflation": [0.5, -0.5, 1.0]}, index=assets
        ),
        factor_max_deviation=pd.Series({"Inflation": 0.12}),
    ),
    benchmark_beta_constraint=BenchmarkBetaConstraint(
        beta_min=0.85,
        beta_max=1.15,
        beta_loadings=beta_loadings,
    ),
)

w = cvx.Variable(len(assets))
rows = constraints.set_cvx_all_constraints(
    w=w,
    covar=cvx.psd_wrap(covar.to_numpy()),
)
problem = cvx.Problem(cvx.Maximize(expected_returns.to_numpy() @ w), rows)
problem.solve(solver="CLARABEL")

solution = pd.Series(w.value, index=assets)
print(solution.round(6))
```

The solution is approximately:

```text
asset
Equity    0.548333
Bond      0.320000
Gold      0.131667
dtype: float64
```

The code below independently checks every applicable policy row:

```python
residuals = evaluate_constraint_residuals(
    solution.to_numpy(),
    constraints,
    covar=covar.to_numpy(),
)
hard_breaches = [r for r in residuals if r.hard and not r.passed]
assert hard_breaches == []
```

## Converting the example to utility mode

`Constraints.copy()` deep-copies the contained pandas objects and accepts field overrides. The
dataclass is frozen, so copying is the normal way to change policy:

```python
utility_constraints = constraints.copy(
    constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
)

w_utility = cvx.Variable(len(assets))
utility, hard_rows = utility_constraints.set_cvx_utility_objective_constraints(
    w=w_utility,
    alphas=np.array([0.030, 0.005, 0.010]),
    covar=cvx.psd_wrap(covar.to_numpy()),
)
problem = cvx.Problem(cvx.Maximize(utility), hard_rows)
problem.solve(solver="CLARABEL")

utility_solution = pd.Series(w_utility.value, index=assets)
print(utility_solution.round(6))
```

This produces approximately `Equity=0.411341`, `Bond=0.438660`, and `Gold=0.150000`. Because the
group TE and group-turnover objects exist and carry utility weights, their penalties replace the
total penalties in this generic builder. The hard group caps stored on the same objects do not
constrain this utility solve.

A lambda is a trade-off, not a feasibility guarantee. Analytics therefore retain breached soft
limits as visible records while marking them non-binding:

```python
soft_spec = Constraints(
    min_weights=pd.Series([0.20, 0.20, 0.05], index=assets),
    max_weights=pd.Series([0.60, 0.65, 0.25], index=assets),
    benchmark_weights=benchmark,
    tracking_err_vol_constraint=0.01,
    weights_0=current,
    turnover_constraint=0.05,
    constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
)
candidate = np.array([0.35, 0.40, 0.25])
records = evaluate_constraint_residuals(
    candidate, soft_spec, covar=covar.to_numpy()
)
soft_violations = [r for r in records if not r.hard and r.violation > 0]
[(r.constraint_type, round(r.violation, 6), r.passed) for r in soft_violations]
```

The result is:

```text
[("turnover", 0.15, True), ("tracking_error", 0.010494, True)]
```

`passed=True` here means “does not determine hard compliance,” not “is below the displayed soft
reference limit.” The positive `violation` preserves the magnitude for reporting.

## Universe alignment and rebalancing policy

### Use the production alignment method

`update_with_valid_tickers(...)` is the full production path. It aligns flat Series and all nested
loading blocks to one ordered solver universe. The simpler `update(valid_tickers, **kwargs)` only
aligns nested blocks and should not be described as a complete rebalance update.

Missing labels inserted by `update_with_valid_tickers` receive:

| Series | Fill value for a missing label |
|---|---:|
| `min_weights`, `max_weights` | `0.0` |
| `weights_0` | `0.0` |
| `asset_returns` | `0.0` |
| `benchmark_weights` | `0.0` |
| `turnover_costs` | `1.0` |
| `rebalancing_indicators` | `1.0` (tradable) |

These are reindex fill values. An existing explicit `NaN` is generally preserved, so clean the
source data rather than relying on reindexing to replace it.

When `total_to_good_ratio` is supplied, only two policies scale:

- `turnover_constraint` is multiplied by the ratio;
- per-name maxima are multiplied by the ratio, except values numerically close to `1.0`, which
  remain `1.0`.

Minimum weights, exposure limits, group bounds, target return, and risk limits do not scale.

### Current-to-model eligibility corridor

`compute_eligible_rebalancing_bounds` projects candidate boxes into the interval between current
and model weights. It permits holding or moving toward the model but not overshooting it.

```python
from optimalportfolios.optimization.constraints import (
    compute_eligible_rebalancing_bounds,
)

assets = ["a", "b", "c", "d"]
current = pd.Series([0.5, 0.3, 0.0, 0.0], index=assets)
model = pd.Series([0.2, 0.3, 0.5, 0.0], index=assets)
lower, upper, indicators = compute_eligible_rebalancing_bounds(
    current_weights=current,
    model_weights=model,
    current_min_weights=pd.Series(0.0, index=assets),
    current_max_weights=pd.Series(1.0, index=assets),
)

print(lower.tolist())      # [0.2, 0.3, 0.0, 0.0]
print(upper.tolist())      # [0.5, 0.3, 0.5, 0.0]
print(indicators.tolist()) # [1, 1, 1, 0]
```

An indicator is one when either current or model absolute weight is strictly greater than `1e-8`.
An asset absent from both is ineligible. A nonzero asset already at model weight still has an
indicator of one, but its corridor is a single point.

### Frozen positions

When both `weights_0` and `rebalancing_indicators` are supplied, a value not numerically close to
one freezes an asset. The method replaces each box side that already exists with the current
weight. An exact pin therefore requires both `min_weights` and `max_weights` to be configured.

For long-only books, a tiny negative frozen weight is clipped to zero to absorb solver-scale
numerical noise. Long/short books retain the signed weight.

### Frozen group-bound waivers

A frozen live position can already exceed a group ceiling, leaving the new solve infeasible even
though the optimizer cannot trade that name. With `relax_frozen_group_bounds=True`, the aligned
object grants a logged one-period waiver using positive loadings as membership:

```python
assets = pd.Index(["Alternatives", "Liquid"])
spec = Constraints(
    min_weights=pd.Series(0.0, index=assets),
    max_weights=pd.Series(1.0, index=assets),
    group_lower_upper_constraints=GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Illiquid": [1.0, 0.0]}, index=assets),
        group_min_allocation=None,
        group_max_allocation=pd.Series({"Illiquid": 0.20}),
    ),
)
aligned = spec.update_with_valid_tickers(
    valid_tickers=assets.tolist(),
    weights_0=pd.Series([0.25, 0.75], index=assets),
    rebalancing_indicators=pd.Series([0, 1], index=assets),
)
aligned.group_lower_upper_constraints.group_max_allocation["Illiquid"]
# 0.2501
```

The `1e-4` increment preserves a small numerical feasibility tolerance. The symmetric rule lowers
a group floor to the loading-weighted frozen maximum minus `1e-4` when necessary.

`max_relaxation_tol` controls log escalation only; it does not cap, reject, or undo a waiver.
`relax_frozen_group_bounds=False` disables the waiver so the infeasible selected trade set remains
visible. Negative-only signed loading groups remain valid solver rows but do not create a
frozen-membership waiver.

(feasibility-and-diagnostics)=
## Feasibility and diagnostics

### Before solving

The `Constraints` constructor catches three common group/box contradictions using a `1e-4`
tolerance and positive group loadings:

1. loading-weighted asset caps cannot reach a group floor;
2. loading-weighted asset floors already exceed a group ceiling;
3. one asset's loading-weighted floor exceeds a group ceiling.

The solver input validator also checks covariance shape/finiteness, global box-versus-budget
reachability, group reachability, and whether a benchmark is compatible with the configured box
and exposure policy. These are cheap structural checks, not a proof that every quadratic,
turnover, deviation, and beta row is jointly feasible.

### After solving

`evaluate_constraint_residuals` evaluates a candidate against the aligned policy. Each immutable
`ConstraintResidual` contains:

| Field | Meaning |
|---|---|
| `constraint_type`, `name` | stable row identity |
| `actual` | realized exposure, return, volatility, turnover, or loading value |
| `lower`, `upper` | applicable policy sides |
| `violation` | distance outside the allowed interval, otherwise zero |
| `tolerance` | acceptance tolerance |
| `hard` | whether the row determines compliance |
| `passed` | hard acceptance result; always true for a soft row |

The default aggregate tolerance is `1e-4`; long-only and instrument boxes use `1e-6`. Risk
residuals are emitted only when covariance or factorization is supplied. Benchmark-relative risk
also needs benchmark weights. Missing analytical state means no corresponding residual is emitted,
not that the policy was implicitly verified.

```python
candidate = np.array([0.62, 0.28, 0.10])
records = evaluate_constraint_residuals(
    candidate,
    constraints,
    covar=covar.to_numpy(),
)
frame = pd.DataFrame([vars(record) for record in records])
breaches = frame.loc[
    frame["hard"] & ~frame["passed"],
    ["constraint_type", "name", "actual", "lower", "upper", "violation"],
]
```

This deliberately invalid candidate reports breaches in the Equity cap, total and group turnover,
both group allocations, Risk-assets deviation, and benchmark beta. The structured frame is the
supported analytics interface; it is more reliable than parsing formatted diagnostic text.

Public solver wrappers return an `OptimizationOutcome` carrying the exact aligned constraints,
covariance factorization, and residual tuple used for acceptance. Use:

```python
outcome.compliant
outcome.residuals_frame()
hard_breaches = [
    residual
    for residual in outcome.constraint_residuals
    if residual.hard and not residual.passed
]
```

`accepted` and `compliant` answer different questions. `accepted` says whether the solver vector
was used instead of a fallback. `compliant` says whether all emitted hard residuals pass. A
fallback is not assumed to satisfy the mandate.

## Configuration checklist

Before running a constrained optimizer:

1. Put every Series and loading matrix in one identical ordered asset universe.
2. Keep returns, covariance, volatility, and tracking-error limits in consistent units.
3. Choose a backend that actually compiles every required family.
4. Supply `benchmark_weights` for TE and deviation constraints.
5. Supply `weights_0` for turnover constraints; absence skips those rows.
6. Supply `asset_returns` with `target_return`.
7. Inject current beta loadings before compiling beta bounds.
8. Give group objects the hard series for forced mode or utility series for utility mode.
9. Remember that total and group TE/turnover are additive when hard, but group penalties take
   precedence in the generic utility builder.
10. Audit the returned `OptimizationOutcome`; do not infer compliance from solver status alone.
