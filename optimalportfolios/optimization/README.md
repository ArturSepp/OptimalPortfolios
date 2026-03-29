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
    └── constraints_test.py
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

The `Constraints` dataclass is the single specification point for all
portfolio constraints. It translates to three solver backends:

```python
constraints.set_cvx_all_constraints(w, covar)     # → list of cvxpy constraints
constraints.set_scipy_constraints(covar)           # → (list of dicts, bounds) for scipy
constraints.set_pyrb_constraints(covar)            # → (bounds, C, d) for pyrb
```

### Supported constraints

| Constraint | Parameter | Used by |
|-----------|-----------|---------|
| Long-only | `is_long_only` | All solvers |
| Weight bounds | `min_weights`, `max_weights` | All solvers |
| Full investment | Automatic (1'w = 1) | All solvers |
| Group exposure | `group_lower_upper_constraints` | CVXPY, scipy, pyrb |
| Tracking error | `tracking_err_vol_constraint` | SAA, TAA solvers |
| Turnover | `turnover_constraint` | SAA, TAA solvers |
| Return target | `target_return` + `asset_returns` | SAA, TAA solvers |
| Vol budget | `max_target_portfolio_vol_an` | SAA solvers |
| Risk budget | Passed directly to solver | `risk_budgeting.py` |

### Constraint enforcement types

SAA and TAA solvers support two modes via `ConstraintEnforcementType`:

- **`FORCED_CONSTRAINTS`**: TE, turnover, and vol budgets are hard CVXPY
  constraints. The objective is purely linear or quadratic.

- **`UTILITY_CONSTRAINTS`**: TE and turnover are penalised in the objective
  with configurable weights (λ_TE, λ_TO). The return floor remains hard.
  Always feasible, smoother weight transitions.

### NaN handling and universe filtering

The wrapper layer calls `filter_covar_and_vectors_for_nans()` to remove
assets with NaN or zero-variance entries. The `Constraints` object is
updated via `update_with_valid_tickers()`, which subsets weight bounds,
rescales group exposures by `total_to_good_ratio`, injects benchmark
weights, and carries forward `weights_0` for warm-start. After solving,
weights are reindexed to the full ticker set with excluded assets at zero.

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