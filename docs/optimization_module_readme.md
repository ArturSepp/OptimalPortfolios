# Optimization Module

Developer documentation for the portfolio optimisation solvers in
`optimalportfolios`. For the repository's worked examples, see the
[examples guide](examples_readme.md).

## Architecture

```
optimization/
├── config.py                       # OptimiserConfig dataclass
├── constraints/                    # Canonical public facade and constraint owners
│   ├── core.py                     # Constraints aggregate and enforcement enum
│   ├── alignment.py                # Universe alignment and frozen-bound relaxation
│   ├── analytics.py                # Pure residual and feasibility analytics
│   ├── backends.py                 # CVXPY, SciPy and risk-budgeting translations
│   ├── benchmarks.py               # Benchmark-deviation and beta containers
│   ├── expressions.py              # Shared CVXPY risk and objective expressions
│   ├── groups.py                   # Group allocation, TRE and turnover constraints
│   ├── run_local/                  # Manual constraint diagnostics
│   └── tests/                      # Constraint-owned unit/API/translation contracts
├── portfolio_result.py             # Result container
├── wrapper_rolling_portfolios.py   # Dispatcher: PortfolioObjective → solver routing
├── general/                        # Objective-driven solvers, no benchmark semantics
│   ├── quadratic.py                #   MIN_VARIANCE, QUADRATIC_UTILITY
│   ├── max_sharpe.py               #   MAXIMUM_SHARPE_RATIO (Charnes-Cooper)
│   ├── max_diversification.py      #   MAX_DIVERSIFICATION
│   ├── risk_budgeting.py           #   EQUAL_RISK_CONTRIBUTION
│   ├── carra_mixture.py            #   MAX_CARA_MIXTURE
│   └── run_local/                  #   <solver>_run.py development runners
├── saa/                            # Strategic solvers: CMAs + return/vol targets
│   ├── min_variance_target_return.py
│   ├── max_return_target_vol.py
│   └── run_local/
├── taa/                            # Tactical solvers: alphas + benchmark + TE
│   ├── maximise_alpha_over_tre.py
│   ├── maximise_alpha_with_target_yield.py
│   └── run_local/
├── risk_allocation/
│   └── run_local/
└── tests/                          # Offline pytest modules only
    └── <cross-solver-contract>_test.py
```

Run automated contracts with pytest, for example
`pytest src/optimalportfolios/optimization/constraints/tests/constraints_test.py -v`. Run a manual solver
diagnostic explicitly from the repository root, for example
`python -m optimalportfolios.optimization.general.run_local.quadratic_run`; these diagnostics may
plot or use local data and are excluded from pytest and built distributions.

### Submodule roles

**constraints/** — the canonical public facade and owner package for the `Constraints` aggregate.
The owner modules separate core specifications, ticker alignment, pure analytics, backend
translation, benchmark-relative constraints, shared expressions and group constraints. Pure
benchmark-beta calculations live in `optimalportfolios.utils.benchmark_beta`; the facade retains
the established constraint-module aliases.

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
        ├── EQUAL_RISK_CONTRIBUTION  → risk_allocation/risk_budgeting.py
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
| **Solver** | `cvx_*` / `opt_*` | `np.ndarray` (covar), `solver`, `verbose` | `np.ndarray` (weights) | Pure numerical optimisation via CVXPY, scipy, or the internal CCD/ADMM solver |

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
    solver: str = 'CLARABEL'            # CVXPY solver name (ignored by scipy and risk-budgeting solvers)
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
| `EQUAL_RISK_CONTRIBUTION` | general | `risk_budgeting.py` | internal CCD/ADMM | b (risk budgets) | Convex (Spinu) |
| `MAX_CARA_MIXTURE` | general | `carra_mixture.py` | scipy SLSQP | GMM params, γ | Non-convex |
| Min var + return floor | saa | `min_variance_target_return.py` | CVXPY | μ, r_target, [w_b] | Convex QP |
| Max return + vol budget | saa | `max_return_target_vol.py` | CVXPY | μ, σ_max, [w_b] | SOCP |
| Alpha over TE | taa | `maximise_alpha_over_tre.py` | CVXPY | α, w_b, TE_max | SOCP |
| Alpha + target yield | taa | `maximise_alpha_with_target_yield.py` | CVXPY | α, y, r_target | SOCP / LP |


## Constraint system

The formula-by-formula reference, backend matrix, alignment policy, residual analytics, and
complete forced/utility examples live in the dedicated
[Portfolio constraint guide](constraints.md). This section gives only the architectural summary.

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

Each compatible solver calls a CVXPY, SciPy, or PyRB compiler for the subset it supports, then
constructs its own objective function. This means:

- Adding a new constraint type centralises its policy and translations, but each backend and
  solver path must explicitly support it
- Adding a new solver can reuse the shared compiler for its compatible constraint subset
- The constraint object can be inspected, printed, and validated
  independently of any solver


### Solver backends

The `Constraints` class generates constraints for three backends:

```python
constraints.set_cvx_all_constraints(w, covar)     # → list of cvxpy constraints
constraints.set_scipy_constraints(covar)           # → (list of dicts, bounds) for scipy
constraints.set_pyrb_constraints(covar)            # → (bounds, C, d) for the risk-budgeting solver
```

The forced CVXPY backend supports the full constraint set. SciPy compiles long-only, net-exposure,
instrument-box, and group-allocation rows. PyRB compiles instrument boxes and group allocations;
the risk-budgeting solver owns full investment. Neither backend compiles return, volatility,
tracking-error, turnover, deviation, or beta fields.


### `Constraints` — the main container

The central dataclass that holds all portfolio constraints. Immutable
(`frozen=True`); all mutation methods return new instances.

**Individual asset constraints:**

- `is_long_only` — no short positions (w ≥ 0)
- `min_weights` / `max_weights` — per-asset weight bounds
- `max_exposure` / `min_exposure` — total portfolio exposure (sum of weights)

**Benchmark-relative constraints:**

- `benchmark_weights` — reference portfolio for tracking error
- `tracking_err_vol_constraint` — max tracking-error volatility in the covariance's units
- `sector_deviation_constraints` — max active sector deviation vs benchmark
- `style_deviation_constraints` — max active style deviation vs benchmark
- `benchmark_beta_constraint` — range for absolute ex-ante benchmark beta

**Turnover constraints:**

- `weights_0` — current portfolio weights (for turnover calculation)
- `turnover_constraint` — max L1 turnover (Σ|wᵢ - wᵢ₀|)
- `turnover_costs` — per-asset transaction costs (scales turnover)

**Return/volatility targets:**

- `target_return` / `asset_returns` — minimum portfolio return constraint
- `max_target_portfolio_vol_an` — maximum portfolio volatility in the covariance's units; the
  historical name does not annualise inputs

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
  with configurable weights (λ_TE, λ_TO). Exposure/box, return, group-allocation,
  sector/style-deviation and beta mandate rows remain hard. Utility penalties do
  not guarantee feasibility when these mandate rows conflict.


### Backend and enforcement capabilities

| Constraint family | CVXPY forced | CVXPY utility | SciPy | Risk budgeting |
|---|---|---|---|---|
| Exposure, long-only, box | Hard | Hard | Hard | Box bounds; full investment is solver policy |
| Target return | Hard | Hard | Unsupported | Unsupported |
| Portfolio volatility | Hard cap | No generic cap; solver-specific risk objective | Unsupported | Unsupported |
| Total + group tracking error | Both hard | Soft; group penalty takes precedence | Unsupported | Unsupported |
| Total + group turnover | Both hard | Soft; group penalty takes precedence | Unsupported | Unsupported |
| Group allocation | Hard | Hard | Hard | Hard matrix rows |
| Sector/style deviation | Hard | Hard | Unsupported | Unsupported |
| Benchmark beta | Hard | Hard | Unsupported | Unsupported |

“Unsupported” means the backend does not compile that field; it must not be inferred from a
post-solve report. The utility mode deliberately softens only tracking error and turnover. A
configured maximum-volatility field is a hard cap in forced mode; utility SAA formulations use a
risk penalty and do not silently reinterpret that field as a cap.


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

Loading membership is deliberately operation-specific. Construction removes a column only when
every loading is exactly zero (or every value is missing). Solver compilation treats a column as
active when at least one loading is not numerically close to zero, so signed factor loadings remain
valid. Frozen-position bound relaxation uses strictly positive loadings as membership; negative
factor exposures therefore do not create a frozen-holding overhang waiver.

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

The wrapper layer calls `filter_covar_and_vectors_for_nans()` to remove assets with NaN or
zero-variance entries. `update_with_valid_tickers()` aligns flat Series and every nested loading
block, injects benchmark/current state, and can scale per-name maxima and the total turnover cap by
`total_to_good_ratio`; it does not scale group bounds. After solving, weights are reindexed to the
full ticker set with excluded assets at zero.


### Structured constraint inspection

Solver outcomes carry the exact aligned specification and report-ready residual records:

```python
outcome.residuals_frame()
hard_breaches = [
    residual
    for residual in outcome.constraint_residuals
    if residual.hard and not residual.passed
]
```

For an independently supplied candidate, call
`evaluate_constraint_residuals(weights, constraints, covar=...)`. The legacy print helpers remain
available for compatibility, but structured residuals are the supported analytical interface.


## Test pattern

All Python test modules in `tests/` end in `_test.py` and expose ordinary pytest cases. Target a
module, node, or keyword expression through pytest rather than adding an executable runner:

```bash
pytest src/optimalportfolios/optimization/constraints/tests/constraints_test.py -v
pytest src/optimalportfolios/optimization/constraints/tests/constraints_test.py -k group -v
```

When adding a solver, add deterministic offline contracts in a matching `_test.py` module. Put
plots, verbose solver demonstrations, and local-data workflows in the nearest `run_local/`
directory as `<solver>_run.py`, where a `Locals` enum selects scenarios through `run_local()`.

### Constraint test files

| File | Purpose |
|------|---------|
| `constraints_test.py` | Core deterministic feasibility contracts for constraint classes and translations |
| `constraints_branches_test.py` | Validation, warning, update, and branch coverage |
| `specialised_constraints_test.py` | Group tracking-error, turnover, and deviation contracts |
| `constraints/run_local/constraints_run.py` | Manual formatted constraint printing and visual inspection |


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
