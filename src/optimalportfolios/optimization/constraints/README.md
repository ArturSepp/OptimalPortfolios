# Optimization Constraints

This package owns the portfolio-constraint system used by `optimalportfolios`. It separates four
concerns that must remain independently testable:

1. immutable policy specifications;
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

For every formula, backend capability, and full forced/utility example, see the
[portfolio constraint guide](../../../../docs/constraints.md).

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

| Module | Owns | Does not own |
|---|---|---|
| `core.py` | the frozen `Constraints` aggregate, enforcement enum, construction-time validation, and delegation methods | solver expressions or rolling state |
| `alignment.py` | ticker alignment, current-to-model eligibility corridors, frozen positions, and logged group-bound waivers | mathematical residuals or solver objects |
| `analytics.py` | pure reachability calculations and candidate-weight residual records | compilation, solving, mutation, or logging |
| `backends.py` | compiler functions for CVXPY, SciPy, and PyRB | policy alignment or post-solve acceptance |
| `benchmarks.py` | benchmark-deviation and beta range dataclasses | rolling beta estimation |
| `expressions.py` | reusable CVXPY covariance-risk and objective-expression leaves | constraint policy |
| `groups.py` | group allocation, group tracking-error, group turnover, merge behavior, and dropped-group records | whole-portfolio risk/trading policy |
| `__init__.py` | the supported import surface | implementation logic |

Benchmark-beta loading calculations remain in `optimalportfolios.utils.benchmark_beta`; the
constraint facade re-exports the two loading helpers used when configuring a solve.

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

Use `evaluate_constraint_residuals` to audit a candidate without invoking a solver:

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
```

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
- beta needs injected `beta_loadings`.

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
or claim that an unsupported backend enforced a field.

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

`validate_solution` stores residuals, the exact aligned constraints, and the covariance
factorization on `OptimizationOutcome`. Two outcome properties answer different questions:

- `outcome.accepted` says whether the solver vector was used instead of a fallback;
- `outcome.compliant` says whether every emitted hard residual passed.

Use `outcome.residuals_frame()` for a report-ready table. A fallback is not presumed compliant,
and solver status alone is never proof that the returned vector satisfies every mandate row.

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

Run all constraint-owned contracts from the repository root:

```powershell
uv run --no-sync pytest src/optimalportfolios/optimization/constraints/tests -q
```

For a focused core run:

```powershell
uv run --no-sync pytest src/optimalportfolios/optimization/constraints/tests/constraints_test.py -v
```

Manual formatting and inspection belong in `run_local/constraints_run.py`; production modules and
public `__init__.py` files must not import `run_local`.
