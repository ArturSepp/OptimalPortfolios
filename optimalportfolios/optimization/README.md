# Optimization Module

Portfolio optimisation solvers for the `optimalportfolios` package.

This module implements six portfolio construction objectives, a unified
constraint system, and a dispatcher that routes any `PortfolioObjective`
to the appropriate solver. All solvers accept pre-computed covariance
matrices from any `CovarEstimator`, decoupling estimation from optimisation.

## Architecture

```
optimization/
├── wrapper_rolling_portfolios.py   # Dispatcher: objective → solver routing
├── constraints.py                  # Constraint specification and CVXPY/scipy/pyrb translation
├── config.py                       # PortfolioObjective enum, OptimiserConfig
├── portfolio_result.py             # Result container
├── solvers/
│   ├── quadratic.py                # MIN_VARIANCE, QUADRATIC_UTILITY
│   ├── max_sharpe.py               # MAXIMUM_SHARPE_RATIO
│   ├── max_diversification.py      # MAX_DIVERSIFICATION
│   ├── risk_budgeting.py           # EQUAL_RISK_CONTRIBUTION
│   ├── carra_mixture.py            # MAX_CARA_MIXTURE
│   ├── target_return.py            # Alpha maximisation with return constraint
│   ├── tracking_error.py           # Alpha maximisation with TE constraint
│   └── tests/                      # One test file per solver
└── tests/
    └── constraints_test.py
```

### Dispatch flow

Every portfolio construction call follows the same path:

```
CovarEstimator.fit_rolling_covars()
        │
        ▼
  covar_dict: Dict[Timestamp, DataFrame]
        │
        ▼
compute_rolling_optimal_weights(covar_dict, portfolio_objective, ...)
        │
        ├── EQUAL_RISK_CONTRIBUTION  → rolling_risk_budgeting()
        ├── MAX_DIVERSIFICATION      → rolling_maximise_diversification()
        ├── MIN_VARIANCE             → rolling_quadratic_optimisation()
        ├── QUADRATIC_UTILITY        → rolling_quadratic_optimisation()
        ├── MAXIMUM_SHARPE_RATIO     → rolling_maximize_portfolio_sharpe()
        └── MAX_CARA_MIXTURE         → rolling_maximize_cara_mixture()
```

The `backtest_rolling_optimal_portfolio()` convenience function chains
weight computation with `qis.backtest_model_portfolio()` for end-to-end
strategy evaluation.

### Three-layer solver pattern

Every solver file follows the same three-layer structure:

| Layer | Function prefix | Input | Output | Responsibility |
|-------|----------------|-------|--------|---------------|
| **Rolling** | `rolling_*` | `prices`, `covar_dict` | `pd.DataFrame` (weights) | Loop over rebalancing dates, warm-start |
| **Wrapper** | `wrapper_*` | `pd.DataFrame` (covar) | `pd.Series` (weights) | NaN/zero-variance filtering, constraint update, reindex to full universe |
| **Solver** | `cvx_*` / `opt_*` | `np.ndarray` (covar) | `np.ndarray` (weights) | Pure numerical optimisation via CVXPY, scipy, or pyrb |

Adding a new solver means implementing these three functions and adding
a routing case in `compute_rolling_optimal_weights()`.

## Solver reference

| Objective | File | Backend | Inputs beyond Σ | Convexity |
|-----------|------|---------|-----------------|-----------|
| `MIN_VARIANCE` | `quadratic.py` | CVXPY | — | Convex QP |
| `QUADRATIC_UTILITY` | `quadratic.py` | CVXPY | μ (means), γ (CARA) | Convex QP |
| `MAXIMUM_SHARPE_RATIO` | `max_sharpe.py` | CVXPY | μ (EWMA means) | SOCP (rescaling trick) |
| `MAX_DIVERSIFICATION` | `max_diversification.py` | scipy SLSQP | — | Non-convex (ratio) |
| `EQUAL_RISK_CONTRIBUTION` | `risk_budgeting.py` | pyrb (ADMM) | b (risk budgets) | Convex (Spinu reformulation) |
| `MAX_CARA_MIXTURE` | `carra_mixture.py` | scipy SLSQP | GMM(μ_k, Σ_k, p_k), γ | Non-convex (mixture exponential) |
| Alpha + target return | `target_return.py` | CVXPY | α (alphas), y (yields), r_target | SOCP / LP |
| Alpha over TE | `tracking_error.py` | CVXPY | α (alphas), w_b (benchmark), TE_max | SOCP |

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
| Full investment | Automatic (Σw = 1) | All solvers |
| Group exposure | `group_exposures` | CVXPY, scipy |
| Tracking error | `tracking_err_vol_constraint` | `tracking_error.py` |
| Turnover | `turnover_constraint` | `tracking_error.py` |
| Return target | `target_return` (via `update_with_valid_tickers`) | `target_return.py` |
| Vol budget | `vol_constraint` | CVXPY solvers |
| Risk budget | Passed directly to solver | `risk_budgeting.py` |

### Constraint enforcement types

For the tracking error solver, two modes are supported:

- **`HARD_CONSTRAINTS`**: TE and turnover are explicit CVXPY constraints.
  The objective is purely linear (maximise active alpha). Use when the TE
  budget is a strict mandate.

- **`UTILITY_CONSTRAINTS`**: TE and turnover are penalised in the objective
  with configurable weights λ_TE and λ_TO. Always feasible, smoother weight
  transitions, but requires calibrating penalty weights.

### NaN handling and universe filtering

The wrapper layer calls `filter_covar_and_vectors_for_nans()` to remove
assets with NaN or zero-variance entries from the covariance matrix. The
`Constraints` object is then updated via `update_with_valid_tickers()`,
which:

1. Subsets weight bounds to valid tickers
2. Rescales group exposures by `total_to_good_ratio` (N_total / N_valid)
3. Injects benchmark weights and rebalancing indicators if provided
4. Carries forward `weights_0` for warm-start and turnover control

After solving on the reduced universe, weights are reindexed to the full
ticker set with excluded assets receiving zero weight.

## Test pattern

All test files in `solvers/tests/` follow a consistent structure:

```python
class LocalTests(Enum):
    SIMPLE_CASE = 1          # 2-3 asset synthetic covariance
    WITH_BOUNDS = 2          # realistic universe with weight caps
    WRAPPER_WITH_NANS = 3    # test NaN filtering and edge cases
    ROLLING_BACKTEST = 4     # full rolling backtest with EwmaCovarEstimator

def run_local_test(local_test: LocalTests):
    ...

if __name__ == '__main__':
    run_local_test(local_test=LocalTests.SIMPLE_CASE)
```

Each test prints weights, portfolio vol, and objective-specific diagnostics
(risk contributions, diversification ratio, Sharpe, tracking error, etc.).
When adding a new solver, create a matching test file with at least the
SIMPLE_CASE and WRAPPER_WITH_NANS cases.

## Usage examples

### Minimum variance with EWMA covariance

```python
import optimalportfolios as opt
import qis

prices = ...  # pd.DataFrame of asset prices
time_period = qis.TimePeriod('31Dec2004', '30Jun2025')

# 1. estimate covariance
estimator = opt.EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)

# 2. optimise
weights = opt.compute_rolling_optimal_weights(
    prices=prices,
    constraints=opt.Constraints(is_long_only=True),
    covar_dict=covar_dict,
    portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
)
```

### Risk budgeting with factor covariance

```python
# 1. estimate factor covariance
lasso_model = opt.LassoModel(model_type=opt.LassoModelType.GROUP_LASSO_CLUSTERS,
                              reg_lambda=1e-5, span=36, warmup_period=12)
factor_estimator = opt.FactorCovarEstimator(
    lasso_model=lasso_model, factor_returns_freq='ME', rebalancing_freq='QE')

asset_returns_dict = qis.compute_asset_returns_dict(
    prices=prices, is_log_returns=True, returns_freqs='ME')
covar_dict = factor_estimator.fit_rolling_covars(
    risk_factor_prices=factor_prices,
    asset_returns_dict=asset_returns_dict,
    time_period=time_period)

# 2. optimise with equal risk budgets
risk_budget = {col: 1.0 / len(prices.columns) for col in prices.columns}
weights = opt.rolling_risk_budgeting(
    prices=prices,
    constraints=opt.Constraints(is_long_only=True),
    covar_dict=covar_dict,
    risk_budget=risk_budget,
)
```

### End-to-end backtest

```python
portfolio = opt.backtest_rolling_optimal_portfolio(
    prices=prices,
    constraints=opt.Constraints(is_long_only=True),
    covar_dict=covar_dict,
    portfolio_objective=opt.PortfolioObjective.MAX_DIVERSIFICATION,
    ticker='Max Div Portfolio',
    rebalancing_costs=0.0010,
    weight_implementation_lag=1,
)
# portfolio is a qis.PortfolioData ready for factsheet generation
```

## References

Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation
for Multi-Asset Portfolios",
*The Journal of Portfolio Management*, 52(4), 86-120.


Sepp A., Hansen E., and Kastenholz M. (2026),
"Capital Market Assumptions and Strategic Asset Allocation Using
Multi-Asset Tradable Factors",
*Working Paper.


Sepp A. (2023),
"Optimal Allocation to Cryptocurrencies in Diversified Portfolios",
*Risk Magazine*, pp. 1-6, October 2023.
Available at https://ssrn.com/abstract=4217841