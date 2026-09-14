---
myst:
  html_meta:
    description: >-
      Contributor guide to OptimalPortfolios constraints: module ownership, ordered inputs,
      backend compilation, feasibility residuals, and development checks.
---

# Optimization Constraints

*[author / affiliation / date — placeholder]*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

This package owns the portfolio-constraint system used by `optimalportfolios`. It separates four
concerns that must remain independently testable:

1. frozen policy specifications;
2. point-in-time universe alignment and rebalancing policy;
3. translation into supported solver backends;
4. solver-independent feasibility and residual analytics.

Import public constraint types from the package facade:

```python
from optimalportfolios.optimization.constraints import (
    ConstraintEnforcementType,
    ConstraintResidual,
    Constraints,
    GroupLowerUpperConstraints,
    evaluate_constraint_residuals,
)
```

The five names above are also re-exported by `optimalportfolios`. The constraint facade
exports additional types, including `BenchmarkBetaConstraint` and `RelaxationRecord`, that
are not package-root exports. Use the [facade](./__init__.py) and its
[compatibility tests](./tests/constraint_api_compatibility_test.py) to check an import.

For every formula, backend capability, and full forced/utility example, see the
[portfolio constraint guide](../../../../docs/constraints.md). That article is the calculation
contract; this README explains where contributors maintain it. Generic portfolio risk analytics
and reporting belong to [QIS](https://github.com/ArturSepp/QuantInvestStrats)
([software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)).
Constraint residuals remain here because they describe the policy that a particular solve used.

## Folder map

```text
constraints/
├── README.md                     # this ownership and analytics guide
├── __init__.py                   # stable public facade and compatibility exports
├── core.py                       # Constraints aggregate and enforcement enum
├── alignment.py                  # valid-universe alignment, freezes, and waivers
├── analytics.py                  # pure feasibility and residual calculations
├── backends.py                   # CVXPY, SciPy, and PyRB translations
├── benchmarks.py                 # deviation and benchmark-beta specifications
├── expressions.py                # shared CVXPY variance/objective expressions
├── groups.py                     # allocation, group-TE, and group-turnover specs
├── run_local/
│   └── constraints_run.py        # manual formatted constraint diagnostics
└── tests/
    ├── backend_compilation_test.py
    ├── constraint_api_compatibility_test.py
    ├── constraint_translation_contract_test.py
    ├── constraints_branches_test.py
    ├── constraints_test.py
    ├── exposure_policy_test.py
    ├── frozen_overshoot_relaxation_test.py
    ├── rebalancing_constraints_test.py
    ├── scipy_group_validation_test.py
    ├── specialised_constraints_test.py
    ├── tracking_error_policy_test.py
    └── utility_mandate_policy_test.py
```

### Module ownership

These modules are implementation owners; import supported names through the facade.

| Module | Owns | Does not own |
|---|---|---|
| [`core.py`](./core.py) | the frozen `Constraints` aggregate, enforcement enum, construction-time validation, and delegation methods | solver expressions or rolling state |
| [`alignment.py`](./alignment.py) | ticker alignment, current-to-model eligibility corridors, frozen positions, and logged group-bound waivers | mathematical residuals or solver objects |
| [`analytics.py`](./analytics.py) | pure reachability calculations and candidate-weight residual records | compilation, solving, mutation, or logging |
| [`backends.py`](./backends.py) | compiler functions for CVXPY, SciPy, and PyRB | policy alignment or post-solve acceptance |
| [`benchmarks.py`](./benchmarks.py) | benchmark-deviation and beta range dataclasses | rolling beta estimation |
| [`expressions.py`](./expressions.py) | reusable CVXPY covariance-risk and objective-expression leaves | constraint policy |
| [`groups.py`](./groups.py) | group allocation, group tracking-error, group turnover, merge behavior, and dropped-group records | whole-portfolio risk/trading policy |
| [`__init__.py`](./__init__.py) | the supported import surface | implementation logic |

Benchmark-beta loading calculations remain in
[`optimalportfolios.utils.benchmark_beta`](../../utils/benchmark_beta.py); the
constraint facade re-exports the two loading helpers used when configuring a solve.
The [risk-budgeting solver](../risk_allocation/risk_budgeting_solver.py) owns the solve and its
full-investment validation. The name `set_pyrb_constraints` identifies a compatibility
matrix format; it does not imply that this module solves a risk-budgeting problem.

`Constraints` is a frozen dataclass, but its contained pandas objects are mutable.
Treat those objects as policy inputs. `copy(**overrides)` deep-copies existing state and
returns a replacement; it does not align a new universe. See the
[copy example](../../../../docs/constraints.md#converting-the-example-to-utility-mode).


## Constraint analytics

`analytics.py` is intentionally solver-independent. It operates on aligned weights and a
`Constraints` specification, preserves the caller's units, and returns data rather than printing
or logging. This makes the same calculations usable for solver acceptance, reporting, and an
independently supplied portfolio.

### `ConstraintResidual`

One `ConstraintResidual` describes one evaluated policy row:

| Field | Meaning |
|---|---|
| `constraint_type` | stable family identifier such as `exposure`, `turnover`, or `group_weight` |
| `name` | row identifier such as a ticker, group name, or `total` |
| `actual` | realized value in the constraint's units |
| `lower`, `upper` | configured sides; either can be `None` |
| `violation` | non-negative distance beyond the allowed interval |
| `tolerance` | absolute acceptance tolerance for that row |
| `hard` | whether the row determines mandate compliance |
| `passed` | hard-row acceptance result; always `True` for a soft row |

Soft records deliberately retain a positive `violation`. Their `passed=True` means “does not
determine hard compliance,” not “is below the displayed soft reference limit.”

### `evaluate_constraint_residuals`

Use `evaluate_constraint_residuals` to audit a candidate without invoking a solver.
This complete offline example retains the original three-asset inputs. Weights are fractions
of portfolio capital. The diagonal covariance represents annual variance, so the tracking-error
limit `0.04` means 4% annual volatility. The evaluator does not resample or annualize inputs.
Turnover is the full L1 weight change over one trade, with no cost weights or half-turnover factor
in this example. No dates or market-data observations are involved:

```python
import numpy as np
import pandas as pd

from optimalportfolios.optimization.constraints import (
    Constraints,
    evaluate_constraint_residuals,
)

assets = pd.Index(["Equity", "Bond", "Gold"])
covar = np.diag([0.0324, 0.0064, 0.0144])
spec = Constraints(
    min_weights=pd.Series([0.20, 0.20, 0.05], index=assets),
    max_weights=pd.Series([0.55, 0.65, 0.25], index=assets),
    benchmark_weights=pd.Series([0.45, 0.40, 0.15], index=assets),
    tracking_err_vol_constraint=0.04,
    weights_0=pd.Series([0.40, 0.45, 0.15], index=assets),
    turnover_constraint=0.20,
)

candidate = np.array([0.60, 0.25, 0.15])
records = evaluate_constraint_residuals(candidate, spec, covar=covar)
frame = pd.DataFrame([vars(record) for record in records])
hard_breaches = frame.loc[frame["hard"] & ~frame["passed"]]
print(hard_breaches["constraint_type"].tolist())
print(hard_breaches["violation"].round(2).tolist())
```

Expected output:

```text
['instrument_weight', 'turnover']
[0.05, 0.2]
```

The Equity weight exceeds its 55% cap by 5 percentage points. Moving from the current
weights to the candidate trades 40% of capital on the full L1 convention, exceeding the
20% cap by 20 percentage points. Annual tracking error is about 2.955%, below its 4% limit.
The example finds violations; it does not optimize or repair the candidate.

The evaluator emits applicable records in a deterministic order:

1. total exposure and long-only;
2. instrument minima and maxima;
3. target return and portfolio volatility;
4. total and group turnover;
5. total and group tracking error;
6. group allocation;
7. sector and style deviation;
8. benchmark beta.

The default tolerance is `1e-4` for aggregate rows. Long-only and individual boxes use `1e-6`.
These tolerances describe post-solve acceptance; they do not change the rows sent to the solver.

### Required analytical state

Only rows with enough state to evaluate are emitted:

- portfolio volatility needs `covar` or `covar_factorization`;
- tracking error additionally needs `benchmark_weights`;
- turnover needs `weights_0`;
- target return needs `asset_returns`;
- sector/style deviations need their loading block and `benchmark_weights`;
- beta needs injected `beta_loadings`.

The candidate is positional: even a pandas Series is converted to a NumPy vector without
using its labels to reorder weights. Align the specification, both covariance axes and the
candidate before evaluation. See the continuation below.

Omitting required analytical state omits that residual. It is not evidence that the omitted policy
passed. Production wrappers avoid this ambiguity by carrying the exact aligned specification and
solver covariance into validation.

When a `CovarianceFactorization` is supplied, its stabilized covariance takes precedence over a
separate `covar` argument. Residual analytics therefore audit the same risk geometry the
factorized solver enforced.

### Hard and utility interpretation

Under `UTILITY_CONSTRAINTS`, the evaluator marks these limit families soft:

- maximum portfolio volatility;
- total and group tracking error;
- total and group turnover.

Exposure, long-only, boxes, target return, group allocation, sector/style deviations, and beta
remain hard. The evaluator reports hard/soft policy; it does not reconstruct the solver objective
or claim that an unsupported backend enforced a field. In particular, generic utility compilation
does not add a maximum-volatility cap, and group risk/trading penalties take precedence over
their total counterparts. The diagnostic can still report both configured limits. Read the
[utility contract](../../../../docs/constraints.md#utility-constraints) alongside the
[backend capability matrix](../../../../docs/constraints.md#backend-capability-matrix).

### Shared analytical kernels

The underscore-prefixed functions in `analytics.py` are internal implementation contracts, not
public imports:

- `_resolve_asset_index` finds the canonical ordered universe from the first indexed constraint
  field, falling back to a numeric range only when necessary.
- `_exposure_facts` preserves the literal exposure equality rule: only exactly equal stored limits
  are an equality.
- `_budget_box_residuals` computes exposure, long-only, and per-name telemetry once for both
  validation and reporting.
- `_iter_finite_group_bounds` normalizes stated group-allocation sides and skips absent/`NaN`
  sides.
- `_construction_group_reachability_errors` supplies the early group-versus-box checks used by
  `Constraints.__post_init__`.
- `_static_reachability_findings` supplies pre-solve box, group, and benchmark findings to solver
  diagnostics.
- `_group_allocation_residuals` evaluates loaded group rows in compiler order.

Keeping these calculations in one pure module prevents the compiler, constructor, and diagnostic
layer from developing different definitions of exposure, group reachability, or violation size.

## Ordered inputs and backend compilation

Use `update_with_valid_tickers(...)` for the full alignment and rebalancing path.
It aligns flat Series and nested loading blocks to one ordered universe; the shorter
`update(valid_tickers, **kwargs)` only aligns nested blocks. Neither method dates the input data
or establishes that a covariance estimate is point-in-time.

Continue the candidate example to reorder the universe and freeze Gold at its current weight:

```python
solver_assets = ["Gold", "Equity", "Bond"]
aligned = spec.update_with_valid_tickers(
    valid_tickers=solver_assets,
    weights_0=spec.weights_0,
    rebalancing_indicators=pd.Series([0, 1, 1], index=solver_assets),
    relax_frozen_group_bounds=False,
)
covar_frame = pd.DataFrame(covar, index=assets, columns=assets)
aligned_covar = covar_frame.loc[solver_assets, solver_assets].to_numpy()
aligned_candidate = pd.Series(candidate, index=assets).reindex(solver_assets).to_numpy()
aligned_records = evaluate_constraint_residuals(
    aligned_candidate, aligned, covar=aligned_covar,
)
print(aligned.min_weights.index.tolist())
print([float(aligned.min_weights["Gold"]), float(aligned.max_weights["Gold"])])
```

Expected output is `['Gold', 'Equity', 'Bond']` followed by `[0.15, 0.15]`.
Both box sides exist, so Gold is pinned at 15%; the other two breaches remain. The original
`spec` retains its original order and bounds.

Missing labels receive field-specific defaults; an explicit `NaN` generally survives reindexing.
A missing maximum-weight label becomes zero, which can remove an asset from the feasible
universe. Freezing only replaces box sides already configured, so both sides are needed for
an exact pin. Consult the complete
[alignment and freezing rules](../../../../docs/constraints.md#universe-alignment-and-rebalancing-policy).

Frozen group-bound waivers are enabled by default. They modify the aligned policy and are
logged; `max_relaxation_tol` controls log escalation, not a cap on the waiver. The example
disables waivers explicitly and has no group constraints. See
[frozen group-bound waivers](../../../../docs/constraints.md#frozen-group-bound-waivers).

### Compiler entry points

Compilation returns solver inputs; it does not solve or assess a returned allocation.

| `Constraints` method | Returns | Integration rule |
|---|---|---|
| `set_cvx_all_constraints(w, covar, ...)` | CVXPY constraint list | Combine with a caller-owned objective for forced enforcement. |
| `set_cvx_utility_objective_constraints(w, alphas, covar, ...)` | Utility expression, hard constraint list | Maximize the expression or combine it with the chosen solver's objective. |
| `set_scipy_constraints(covar)` | Callback list, bounds array or `None` | Callbacks use nonnegative feasibility values; only supported families are compiled. |
| `set_pyrb_constraints(covar)` | Bounds, group matrix, group right-hand side | The group pair can be `None`. Full investment belongs to the risk-budgeting solver. |

A low-level call to `set_cvx_all_constraints` still compiles hard rows if the specification's
enum is `UTILITY_CONSTRAINTS`. Select the compiler and enforcement policy together;
changing the enum alone does not change that method's output.

SciPy compiles boxes, net exposure and group allocation. The risk-budgeting matrix helper
compiles boxes and group allocation. Fields outside those capabilities are not enforced.
Use the [backend matrix](../../../../docs/constraints.md#backend-capability-matrix) and
[full forced/utility examples](../../../../docs/constraints.md#worked-example) for the complete
contract rather than inferring capabilities from fields on the shared dataclass.

## Solver-outcome integration

The normal lifecycle is:

```text
Constraints specification
        │
        ├── alignment.py ──> one ordered, point-in-time universe
        │
        ├── backends.py  ──> solver rows and bounds
        │
        └── analytics.py <── candidate or returned weights
                              │
                              └── ConstraintResidual tuple
                                      │
                                      └── OptimizationOutcome
```

[`validate_solution`](../solver_diagnostics.py) stores residuals, the exact aligned constraints,
and any supplied covariance factorization on `OptimizationOutcome`. The outcome is exported
at the package root; the validator lives in `optimalportfolios.optimization.solver_diagnostics`.
Two outcome attributes answer different questions:

- `outcome.accepted` says whether the solver vector was used instead of a fallback;
- `outcome.compliant` says whether every emitted hard residual passed.

Use `outcome.residuals_frame()` for a report-ready table. A fallback is not presumed compliant,
and solver status alone is never proof that the returned vector satisfies every mandate row.
`compliant` checks only emitted hard rows; it also returns `True` for an empty residual tuple.
Retain the input state and verify coverage before treating that flag as a mandate audit.

## Extending the subsystem

A new constraint family normally requires coordinated changes in this order:

1. add its immutable specification to the owning module and aggregate it in `core.py`;
2. register ticker-indexed state in `alignment.py`;
3. add only the backend translations that genuinely support it;
4. add an analytical residual with the same formula and units;
5. expose public names through `__init__.py` when they are part of the supported API;
6. add construction, translation, residual, backend-capability, and compatibility tests.

Do not treat residual reporting as a substitute for backend compilation. An unsupported field can
be measured after a solve, but it did not constrain that solve.

## Verification

Use the external interpreter and C-local setup required by
[AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md).
Run checks from a C-local source export; the commands below assume that setup.

| Work area | Existing checks |
|---|---|
| Public imports, signatures and serialization | [API compatibility](./tests/constraint_api_compatibility_test.py). |
| Compiler rows and backend differences | [Compilation](./tests/backend_compilation_test.py), [translation contracts](./tests/constraint_translation_contract_test.py) and [SciPy group validation](./tests/scipy_group_validation_test.py). |
| Universe alignment and frozen policy | [Rebalancing](./tests/rebalancing_constraints_test.py) and [frozen overshoots](./tests/frozen_overshoot_relaxation_test.py). |
| Hard/soft mandates and residuals | [Utility policy](./tests/utility_mandate_policy_test.py), [tracking error](./tests/tracking_error_policy_test.py) and [solver diagnostics](../tests/solver_diagnostics_test.py). |
| Public methodology and examples | [Constraint article tests](../../tests/constraints_documentation_test.py). |

Run all constraint-owned contracts:

```powershell
python -m pytest src/optimalportfolios/optimization/constraints/tests -q
```

For a focused core run:

```powershell
python -m pytest src/optimalportfolios/optimization/constraints/tests/constraints_test.py -v
```

Check this guide's source and the authoritative article's executable contracts:

```text
python tools/check_docs.py --files src/optimalportfolios/optimization/constraints/README.md
python -m pytest src/optimalportfolios/tests/constraints_documentation_test.py
```

Manual formatting and inspection belong in
[`run_local/constraints_run.py`](./run_local/constraints_run.py). It uses a fixed synthetic
ten-asset universe and SCS through CVXPY, prints diagnostics, and defaults to
`Locals.GROUP_ALLOCATION`. No data download is needed. Run it after the same setup:

```text
python -m optimalportfolios.optimization.constraints.run_local.constraints_run
```

The runner follows `Locals` / `run_local(local=...)` and is excluded from distributions.
Production modules and public `__init__.py` files must not import `run_local`.
Preserve numerical defaults, fixtures and seeds; proposed numerical corrections need independent
references. Stale source docstrings, including alignment-scaling/waiver and PyRB wording,
remain a separate reconciliation task.

## References

- [Portfolio constraints: complete methodology](../../../../docs/constraints.md).
- [Software design and package boundaries](../../../../docs/software_design.md).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
