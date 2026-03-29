# Optimization Module

Developer documentation for the portfolio optimisation solvers in
`optimalportfolios`. For user-facing overview and quick start, see the
[project README](../README.md).

## Architecture

```
optimization/
├── config.py                       # OptimiserConfig dataclass
├── constraints.py                  # Constraint specification and CVXPY/scipy/pyrb translation
├── portfolio_result.py             # Result container
├── wrapper_rolling_portfolios.py   # Dispatcher: PortfolioObjective → solver routing
├── general/                        # Objective-driven solvers, no benchmark semantics
│   ├── quadratic.py                #   MIN_VARIANCE, QUADRATIC_UTILITY
│   ├── max_sharpe.py               #   MAXIMUM_SHARPE_RATIO (Charnes-Cooper)
│   ├── max_diversification.py      #   MAX_DIVERSIFICATION
│   ├── risk_budgeting.py           #   EQUAL_RISK_CONTRIBUTION
│   └── carra_mixture.py            #   MAX_CARA_MIXTURE
├── saa/                            # Strategic solvers: CMAs + return/vol targets
│   ├── min_variance_target_return.py
│   └── max_return_target_vol.py
├── taa/                            # Tactical solvers: alphas + benchmark + TE
│   ├── maximise_alpha_over_tre.py
│   └── maximise_alpha_with_target_yield.py
└── tests/                          # One test file per solver
    ├── quadratic_test.py
    ├── max_sharpe_test.py
    ├── max_diversification_test.py
    ├── risk_budgeting_test.py
    ├── carra_mixture_test.py
    ├── min_variance_target_return_test.py
    ├── max_return_target_vol_test.py
    ├── maximise_alpha_over_tre_test.py
    ├── maximise_alpha_with_target_yield_test.py
    ├── constraints_test.py
    ├── test_constraints.py
    └── constraints_dev_tests.py
```

### Submodule roles

**general/** — solvers that take covariance (and optionally returns) as
input and produce standalone allocations. No benchmark, no active overlay
semantics. Used as building blocks or for single-objective portfolio
construction.

**saa/** — strategic asset allocation solvers. Take CMA inputs (expected
returns), return floors or volatility budgets, and optionally a benchmark
for tracking-error-based risk minimisation. Produce the strategic anchor
allocation. Both hard constraint and utility penalty formulations are
supported via `ConstraintEnforcementType`.

**taa/** — tactical asset allocation solvers. Take alpha signals, a
benchmark (SAA weights), and tracking error or turnover budgets. Produce
active tilts over the SAA anchor. The separation between SAA and TAA is
central to the ROSAA framework.

### Dispatch flow

```
CovarEstimator.fit_rolling_covars()
        │
        ▼
  covar_dict: Dict[Timestamp, DataFrame]
        │
        ▼
compute_rolling_optimal_weights(covar_dict, portfolio_objective, ...)
        │
        ├── EQUAL_RISK_CONTRIBUTION  → general/risk_budgeting.py
        ├── MAX_DIVERSIFICATION      → general/max_diversification.py
        ├── MIN_VARIANCE             → general/quadratic.py
        ├── QUADRATIC_UTILITY        → general/quadratic.py
        ├── MAXIMUM_SHARPE_RATIO     → general/max_sharpe.py
        └── MAX_CARA_MIXTURE         → general/carra_mixture.py
```

SAA and TAA solvers are called directly (not through the dispatcher) since
they require additional inputs (benchmarks, alphas, return targets) that
don't fit the generic `PortfolioObjective` enum.

## Three-layer solver pattern

Every solver file follows the same three-layer structure:

| Layer | Function prefix | Input | Output | Responsibility |
|-------|----------------|-------|--------|---------------|
| **Rolling** | `rolling_*` | `prices`, `covar_dict`, `optimiser_config` | `pd.DataFrame` (weights) | Loop over rebalancing dates, forward-fill signals, warm-start |
| **Wrapper** | `wrapper_*` | `pd.DataFrame` (covar), `optimiser_config` | `pd.Series` (weights) | NaN/zero-variance filtering, constraint update, reindex to full universe |
| **Solver** | `cvx_*` / `opt_*` | `np.ndarray` (covar), `solver`, `verbose` | `np.ndarray` (weights) | Pure numerical optimisation via CVXPY, scipy, or pyrb |

The rolling and wrapper layers accept `OptimiserConfig`; the lowest-level
solver functions take raw `solver: str` and `verbose: bool` parameters,
keeping them framework-agnostic.

```
Layer 3: Rolling          Layer 2: Wrapper           Layer 1: Solver
┌──────────────────┐     ┌───────────────────────┐  ┌──────────────────┐
│ rolling_xxx()    │────>│ wrapper_xxx()         │─>│ cvx_xxx()        │
│                  │     │                       │  │ opt_xxx()        │
│ • slice prices   │     │ • filter NaN assets   │  │                  │
│ • estimate covar │     │ • update constraints  │  │ • solve QP/SOCP  │
│ • for each date: │     │   with valid_tickers  │  │ • return weights │
│   call wrapper   │     │ • call solver         │  │                  │
│ • output weight  │     │ • zero-fill missing   │  │                  │
│   time series    │     │   asset weights       │  │                  │
└──────────────────┘     └───────────────────────┘  └──────────────────┘
```

Adding a new solver means implementing these three functions, placing the
file in `general/`, `saa/`, or `taa/`, and adding exports to the
submodule `__init__.py`.

## OptimiserConfig

Solver configuration shared across all solvers, defined in `config.py`:

```python
@dataclass(frozen=True)
class OptimiserConfig:
    solver: str = 'CLARABEL'            # CVXPY solver name (ignored by scipy/pyrb)
    verbose: bool = False               # print solver diagnostics
    apply_total_to_good_ratio: bool = False  # rescale constraints for excluded assets
```

All `rolling_*` and `wrapper_*` functions accept
`optimiser_config: OptimiserConfig = OptimiserConfig()` as an optional
argument, ensuring backward compatibility.

## Solver reference

| Objective | Module | File | Backend | Inputs beyond Σ | Convexity |
|-----------|--------|------|---------|-----------------|-----------|
| `MIN_VARIANCE` | general | `quadratic.py` | CVXPY | — | Convex QP |
| `QUADRATIC_UTILITY` | general | `quadratic.py` | CVXPY | μ, γ | Convex QP |
| `MAXIMUM_SHARPE_RATIO` | general | `max_sharpe.py` | CVXPY | μ | SOCP (Charnes-Cooper) |
| `MAX_DIVERSIFICATION` | general | `max_diversification.py` | scipy SLSQP | — | Non-convex (ratio) |
| `EQUAL_RISK_CONTRIBUTION` | general | `risk_budgeting.py` | pyrb (ADMM) | b (risk budgets) | Convex (Spinu) |
| `MAX_CARA_MIXTURE` | general | `carra_mixture.py` | scipy SLSQP | GMM params, γ | Non-convex |
| Min var + return floor | saa | `min_variance_target_return.py` | CVXPY | μ, r_target, [w_b] | Convex QP |
| Max return + vol budget | saa | `max_return_target_vol.py` | CVXPY | μ, σ_max, [w_b] | SOCP |
| Alpha over TE | taa | `maximise_alpha_over_tre.py` | CVXPY | α, w_b, TE_max | SOCP |
| Alpha + target yield | taa | `maximise_alpha_with_target_yield.py` | CVXPY | α, y, r_target | SOCP / LP |


## Constraint system

### Why constraints are shared but objectives are not

Portfolio optimisation has two components: an **objective function** (what
to optimise) and **constraints** (what is feasible). Constraints are
almost always the same regardless of the objective, while the objective
function changes per solver.

Consider three different portfolio problems:

| Problem | Objective | Constraints |
|---------|-----------|-------------|
| Min variance | min w'Σw | long-only, sum=1, weight bounds, group limits |
| Max Sharpe | max μ'w / √(w'Σw) | long-only, sum=1, weight bounds, group limits |
| Risk budgeting | min Σᵢ(wᵢσᵢ/σₚ - bᵢ)² | long-only, sum=1, weight bounds, group limits |

The constraints column is identical. This reflects how institutional
portfolios work: the investment policy statement (IPS) defines what the
portfolio *may* hold (asset bounds, group allocations, tracking error
budgets, turnover limits). The PM then chooses *how* to allocate within
those bounds. The IPS doesn't change when the PM switches from
min-variance to max-Sharpe.

```
Constraints (shared)              Solvers (objective-specific)
┌───────────────────────┐         ┌─────────────────────────────┐
│ is_long_only          │         │ max_diversification.py      │
│ min_weights           │    ┌───>│   obj: max Σσᵢwᵢ/√(w'Σw)  │
│ max_weights           │    │    └─────────────────────────────┘
│ max/min_exposure      │    │    ┌─────────────────────────────┐
│ benchmark_weights     │────┼───>│ quadratic.py                │
│ tracking_err_vol      │    │    │   obj: max μ'w - γw'Σw      │
│ weights_0             │    │    └─────────────────────────────┘
│ turnover_constraint   │    │    ┌─────────────────────────────┐
│ group_lower_upper     │    ├───>│ risk_budgeting.py           │
│ group_tracking_error  │    │    │   obj: min risk budget gap   │
│ group_turnover        │    │    └─────────────────────────────┘
│ sector_deviation      │    │    ┌─────────────────────────────┐
│ style_deviation       │    └───>│ tracking_error.py           │
│ target_return         │         │   obj: max α'w s.t. TE ≤ σ  │
│ asset_returns         │         └─────────────────────────────┘
└───────────────────────┘
```

Each solver calls `constraints.set_cvx_all_constraints(w, covar)` (or the
SciPy/PyRB equivalent) to get the constraint set, then constructs its own
objective function. This means:

- Adding a new constraint type (e.g. `BenchmarkDeviationConstraints`)
  automatically applies to **all** solvers
- Adding a new solver only requires writing the objective — constraints
  are inherited for free
- The constraint object can be inspected, printed, and validated
  independently of any solver


### Solver backends

The `Constraints` class generates constraints for three backends:

```python
constraints.set_cvx_all_constraints(w, covar)     # → list of cvxpy constraints
constraints.set_scipy_constraints(covar)           # → (list of dicts, bounds) for scipy
constraints.set_pyrb_constraints(covar)            # → (bounds, C, d) for pyrb
```

The CVXPY backend supports the full constraint set. SciPy and PyRB support
exposure bounds and group allocation constraints (no tracking error or
turnover at the solver level — these are handled in the wrapper layer).


### `Constraints` — the main container

The central dataclass that holds all portfolio constraints. Immutable
(`frozen=True`); all mutation methods return new instances.

**Individual asset constraints:**

- `is_long_only` — no short positions (w ≥ 0)
- `min_weights` / `max_weights` — per-asset weight bounds
- `max_exposure` / `min_exposure` — total portfolio exposure (sum of weights)

**Benchmark-relative constraints:**

- `benchmark_weights` — reference portfolio for tracking error
- `tracking_err_vol_constraint` — max annualised tracking error vol
- `sector_deviation_constraints` — max active sector deviation vs benchmark
- `style_deviation_constraints` — max active style deviation vs benchmark

**Turnover constraints:**

- `weights_0` — current portfolio weights (for turnover calculation)
- `turnover_constraint` — max L1 turnover (Σ|wᵢ - wᵢ₀|)
- `turnover_costs` — per-asset transaction costs (scales turnover)

**Return/volatility targets:**

- `target_return` / `asset_returns` — minimum portfolio return constraint
- `max_target_portfolio_vol_an` — maximum annualised portfolio volatility

**Group-level constraints:**

- `group_lower_upper_constraints` — group allocation bounds
- `group_tracking_error_constraint` — per-group tracking error limits
- `group_turnover_constraint` — per-group turnover limits

**Enforcement mode:**

- `constraint_enforcement_type` — hard constraints vs utility penalties
- `tre_utility_weight` / `turnover_utility_weight` — penalty weights for
  soft enforcement


### Constraint enforcement types

SAA and TAA solvers support two modes via `ConstraintEnforcementType`:

- **`FORCED_CONSTRAINTS`**: TE, turnover, and vol budgets are hard CVXPY
  constraints. The objective is purely linear or quadratic.

- **`UTILITY_CONSTRAINTS`**: TE and turnover are penalised in the objective
  with configurable weights (λ_TE, λ_TO). The return floor remains hard.
  Always feasible, smoother weight transitions.


### Supported constraints summary

| Constraint | Parameter | Used by |
|-----------|-----------|---------|
| Long-only | `is_long_only` | All solvers |
| Weight bounds | `min_weights`, `max_weights` | All solvers |
| Full investment | Automatic (1'w = 1) | All solvers |
| Group exposure | `group_lower_upper_constraints` | CVXPY, scipy, pyrb |
| Sector deviation | `sector_deviation_constraints` | CVXPY solvers |
| Style deviation | `style_deviation_constraints` | CVXPY solvers |
| Tracking error | `tracking_err_vol_constraint` | SAA, TAA solvers |
| Group TE | `group_tracking_error_constraint` | SAA, TAA solvers |
| Turnover | `turnover_constraint` | SAA, TAA solvers |
| Group turnover | `group_turnover_constraint` | SAA, TAA solvers |
| Return target | `target_return` + `asset_returns` | SAA, TAA solvers |
| Vol budget | `max_target_portfolio_vol_an` | SAA solvers |
| Risk budget | Passed directly to solver | `risk_budgeting.py` |


### Constraint classes

#### `GroupLowerUpperConstraints`

Constrains aggregate allocation to groups of assets:

```
group_min ≤ group_loading' @ w ≤ group_max
```

Where `group_loading` is a column of the loading matrix (binary for simple
sector/region groups, fractional for factor exposures).

```python
gluc = GroupLowerUpperConstraints(
    group_loadings=pd.DataFrame({
        "Equities":  [1, 1, 0, 0, 0],
        "Bonds":     [0, 0, 1, 1, 0],
        "Gold":      [0, 0, 0, 0, 1],
    }, index=tickers, dtype=float),
    group_min_allocation=pd.Series({"Equities": 0.30, "Bonds": 0.20, "Gold": 0.05}),
    group_max_allocation=pd.Series({"Equities": 0.60, "Bonds": 0.50, "Gold": 0.20}),
)
```

Validation: `__post_init__` drops groups with all-zero loadings, reindexes
allocation series, and warns on missing entries.

Merge: `merge_group_lower_upper_constraints()` combines two constraint
objects, handling overlapping group names with `_1`/`_2` suffixes.


#### `BenchmarkDeviationConstraints`

Constrains the active deviation of each factor group relative to a
benchmark:

```
|factor_loading' @ (w - w_bm)| ≤ max_deviation
```

Useful for sector tilts (e.g. "Tech allocation may deviate at most 5%
from benchmark") and style constraints (e.g. "Growth vs Value tilt within
±3%").

```python
bdc = BenchmarkDeviationConstraints(
    factor_loading_mat=pd.DataFrame({
        "Tech":    [1, 1, 0, 0, 0],
        "Finance": [0, 0, 1, 1, 0],
        "Energy":  [0, 0, 0, 0, 1],
    }, index=tickers, dtype=float),
    factor_max_deviation=pd.Series({"Tech": 0.05, "Finance": 0.05, "Energy": 0.03}),
)
```

Key difference from `GroupLowerUpperConstraints`: deviation constraints are
relative to a benchmark (symmetric around benchmark weight), while group
bounds are absolute allocation limits.


#### `GroupTrackingErrorConstraint`

Per-group quadratic tracking error constraints:

```
(group_loading ⊙ (w - w_bm))' Σ (group_loading ⊙ (w - w_bm)) ≤ σ²
```

Can be enforced as hard constraints (`group_tre_vols`) or as utility
penalties in the objective function (`group_tre_utility_weights`).


#### `GroupTurnoverConstraint`

Per-group L1 turnover constraints:

```
||group_loading ⊙ (w - w₀)||₁ ≤ max_turnover
```

Useful when different asset classes have different liquidity profiles
(e.g. equities can trade 10% per quarter, alternatives only 3%).


### Feasibility validation

`Constraints.__post_init__` runs three checks when group constraints are
present:

1. **Can the group minimum be reached?** Sum of loading-weighted asset
   max_weights must be ≥ group_min_allocation
2. **Can the group maximum be respected?** Sum of loading-weighted asset
   min_weights must be ≤ group_max_allocation
3. **Single-asset dominance:** No single asset's loading-weighted minimum
   may exceed the group maximum

These catch common configuration errors before the solver is invoked,
producing clear error messages with specific remediation suggestions.


### NaN handling and universe filtering

The wrapper layer calls `filter_covar_and_vectors_for_nans()` to remove
assets with NaN or zero-variance entries. The `Constraints` object is
updated via `update_with_valid_tickers()`, which subsets weight bounds,
rescales group exposures by `total_to_good_ratio`, injects benchmark
weights, and carries forward `weights_0` for warm-start. After solving,
weights are reindexed to the full ticker set with excluded assets at zero.


### Debug utilities

Two methods on `Constraints` for inspecting the CVXPY constraint stack:

```python
# before solving: print each constraint's type, shape, and string form
constraints.print_constraints(constraint_list)

# after solving: check which constraints are binding or violated
constraints.check_constraints_violation(constraint_list)
```


## Test pattern

All test files in `tests/` follow a consistent structure:

```python
class LocalTests(Enum):
    SIMPLE_CASE = 1          # 2-4 asset synthetic covariance
    WITH_BOUNDS = 2          # weight caps and group constraints
    WRAPPER_WITH_NANS = 3    # NaN filtering and edge cases
    FRONTIER = 4             # sweep over constraint parameter

def run_local_test(local_test: LocalTests):
    ...

if __name__ == '__main__':
    run_local_test(local_test=LocalTests.SIMPLE_CASE)
```

When adding a new solver, create a matching test file with at least
SIMPLE_CASE and WRAPPER_WITH_NANS cases.

### Constraint test files

| File | Tests | Purpose |
|------|-------|---------|
| `constraints_test.py` | 5 | Original feasibility validation tests (enum-driven) |
| `test_constraints.py` | 63 | Comprehensive automated suite covering all constraint classes, all backends, update/copy/merge logic. Run: `python test_constraints.py` or `python test_constraints.py <section>` (sections 1–6) |
| `constraints_dev_tests.py` | 20 | Interactive development tests with detailed output. Run: `python constraints_dev_tests.py` or `python constraints_dev_tests.py SECTOR_DEVIATION` |


## References

Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation
for Multi-Asset Portfolios",
*The Journal of Portfolio Management*, 52(4), 86-120.

Sepp A., Hansen E., and Kastenholz M. (2026),
"Capital Market Assumptions and Strategic Asset Allocation Using
Multi-Asset Tradable Factors",
*Working Paper*.

Sepp A. (2023),
"Optimal Allocation to Cryptocurrencies in Diversified Portfolios",
*Risk Magazine*, pp. 1-6, October 2023.
Available at https://ssrn.com/abstract=4217841