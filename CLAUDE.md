# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OptimalPortfolios** (v4.1.1) is a Python library for constructing and backtesting multi-asset portfolios. It provides the full production pipeline: alpha signal computation → covariance estimation (EWMA or HCGL factor model) → portfolio optimisation (risk budgeting, max diversification, max Sharpe, alpha-over-tracking-error, etc.) → rolling backtest with NaN-aware data handling.

The package is the reference implementation for the ROSAA framework published in *The Journal of Portfolio Management* (Sepp, Ossa, Kastenholz, 2026).

## Commands

### Install (editable with dev dependencies)
```bash
pip install -e ".[dev]"
```

### Run all tests
```bash
pytest optimalportfolios/
```

### Run a single test file
```bash
pytest optimalportfolios/alphas/tests/signals_test.py -v
pytest optimalportfolios/optimization/tests/constraints_test.py -v
```

### Formatting and linting
```bash
black optimalportfolios/       # Format (line-length=100)
isort optimalportfolios/       # Sort imports (profile=black)
flake8 optimalportfolios/      # Lint
mypy optimalportfolios/        # Type-check
```

Test files follow `*_test.py` naming; pytest markers include `slow`, `integration`, `unit`, `optimization`, `backtesting`.

## Architecture

### Package Structure
```
optimalportfolios/
├── alphas/                    # Alpha signal computation (NEW in v4.1.1)
│   ├── signals/
│   │   ├── momentum.py        # compute_momentum_alpha()
│   │   ├── low_beta.py        # compute_low_beta_alpha()
│   │   └── managers_alpha.py  # compute_managers_alpha()
│   ├── alpha_data.py          # AlphasData container
│   ├── backtest_alphas.py     # Signal backtesting tool
│   └── tests/
├── covar_estimation/          # Covariance matrix estimation
│   ├── covar_estimator.py     # CovarEstimator ABC
│   ├── ewma_covar_estimator.py    # EwmaCovarEstimator
│   ├── factor_covar_estimator.py  # FactorCovarEstimator (HCGL)
│   ├── rolling_covar.py       # RollingFactorCovarData, CurrentFactorCovarData
│   └── covar_reporting.py     # Rolling covariance diagnostics
├── lasso/                     # HCGL factor model
│   └── lasso_model_estimator.py
├── optimization/              # Portfolio optimisation
│   ├── constraints.py         # Constraints, GroupLowerUpperConstraints
│   ├── wrapper_rolling_portfolios.py  # compute_rolling_optimal_weights()
│   └── solvers/               # One module per solver, each with 3 layers
├── utils/                     # Auxiliary analytics
├── reports/                   # Performance reporting
└── examples/                  # Worked examples and paper reproductions
```

### Module Dependency Order
```
alphas/  →  optimization/  →  covar_estimation/  →  lasso/  →  utils/  →  reports/
```
`alphas/` depends only on `qis` and standard libraries. `optimization/` depends on `covar_estimation/` only for type hints — covariance estimation is separated from optimisation (covar_dict is passed, not estimated internally).

### Key Design Principle: Estimation/Optimisation Separation

Covariance estimation is separated from portfolio optimisation. Estimate first, then pass `covar_dict` to any solver:

```python
# estimate once
estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)

# reuse across multiple solvers
weights_rb = rolling_risk_budgeting(prices=prices, covar_dict=covar_dict, ...)
weights_md = rolling_maximise_diversification(prices=prices, covar_dict=covar_dict, ...)
```

Rolling solvers do NOT estimate covariance internally — `covar_dict` is a required parameter.

### Three-Layer Solver Pattern

Every portfolio solver is implemented in three layers. This pattern MUST be followed when adding new solvers:

| Layer | Prefix | Input | Output |
|-------|--------|-------|--------|
| 1 – Math | `opt_*` / `cvx_*` | Clean `np.ndarray`, no NaNs | `np.ndarray` weights |
| 2 – Wrapper | `wrapper_*` | `pd.DataFrame` (may have NaNs) | `pd.Series` weights |
| 3 – Rolling | `rolling_*` / `backtest_*` | `pd.DataFrame` prices + `covar_dict` | `pd.DataFrame` weight time series |

The wrapper layer filters NaN assets via `filter_covar_and_vectors_for_nans()`, calls `constraints.update_with_valid_tickers()`, then delegates to layer 1. The rolling layer iterates over rebalancing dates in `covar_dict` and calls the wrapper.

### Alpha Signals Module (v4.1.1)

Three standalone signal functions with a consistent interface:

```python
def compute_*_alpha(
    prices: pd.DataFrame,
    returns_freq: Union[str, pd.Series],  # single or mixed frequency
    group_data: Optional[pd.Series],      # for within-group scoring
    **signal_params,
) -> Tuple[pd.DataFrame, pd.DataFrame]:   # (score, raw_signal)
```

Each function handles both single-frequency (`returns_freq='ME'`) and mixed-frequency (`returns_freq=pd.Series(...)`) internally. No separate `_different_freqs` variants.

Naming convention: raw signal → score → alpha.

`AlphasData` is the output container holding `alpha_scores` (portfolio-ready) plus all intermediate components for diagnostics.

**Important**: The alpha aggregation logic (routing assets to signals, combination rules, CDF mapping) is NOT in this package. It belongs in the private `rosaa` package (`rosaa.alphas.AlphaAggregator`). This package provides only the individual signal building blocks.

### Main Entry Points

- **`compute_rolling_optimal_weights()`** (`optimization/wrapper_rolling_portfolios.py`) — unified dispatcher for all solvers via `PortfolioObjective` enum.
- **`EwmaCovarEstimator`** (`covar_estimation/ewma_covar_estimator.py`) — EWMA covariance. Use `.fit_rolling_covars(prices=...)` for backtesting, `.fit_current_covar(prices=...)` for live.
- **`FactorCovarEstimator`** (`covar_estimation/factor_covar_estimator.py`) — HCGL factor model. Use `.fit_rolling_factor_covars(risk_factor_prices=..., asset_returns_dict=...)`. Takes `asset_returns_dict` (not `prices`) for mixed-frequency support.
- **`compute_momentum_alpha()`, `compute_low_beta_alpha()`, `compute_managers_alpha()`** (`alphas/signals/`) — individual alpha signal functions.

### Key Classes

**`Constraints`** (`optimization/constraints.py`) — Dataclass specifying all portfolio constraints (long-only, weight bounds, tracking error, turnover, group constraints). Backend-agnostic; convert with `.set_cvx_all_constraints()`, `.set_scipy_constraints()`, or `.set_pyrb_constraints()`. Use `.update_with_valid_tickers()` to subset to valid assets.

**`EwmaCovarEstimator`** (`covar_estimation/ewma_covar_estimator.py`) — EWMA covariance estimator inheriting from `CovarEstimator` ABC. Parameters: `returns_freq`, `span`, `rebalancing_freq`.

**`FactorCovarEstimator`** (`covar_estimation/factor_covar_estimator.py`) — HCGL factor model covariance estimator inheriting from `CovarEstimator` ABC. Takes `LassoModel` for beta estimation, produces `RollingFactorCovarData` with `.get_y_covars()` and `.get_y_betas()` accessors.

**`AlphasData`** (`alphas/alpha_data.py`) — Container for alpha computation results. Primary output: `alpha_scores` (T × N DataFrame). Also holds intermediate components: `momentum_score`, `beta_score`, `managers_scores`, and raw signals.

**`PortfolioObjective`** (`config.py`) — Enum selecting the solver: `MIN_VARIANCE`, `MAX_DIVERSIFICATION`, `EQUAL_RISK_CONTRIBUTION`, `QUADRATIC_UTILITY`, `MAXIMUM_SHARPE_RATIO`, `MAX_CARA_MIXTURE`.

### Conventions

- All DataFrames use DatetimeIndex for rows and ticker strings for columns/index. Prices, returns, covariance matrices, and weights all follow this convention (aligned with the `qis` library).
- `span` and `roll_window` are in periods, not days. Adjust when changing `returns_freq` (e.g., `span=52` with `'W-WED'` ≈ 1 year; `span=12` with `'ME'` ≈ 1 year).
- `ConstraintEnforcementType.FORCED_CONSTRAINTS` = hard constraint; `UTILITY_CONSTRAINTS` = soft penalty added to objective.
- Solver fallback messages use `warnings.warn()`, not `print()`.
- Solvers live in `optimization/solvers/`; each module implements all three layers.
- Examples in `examples/` serve as integration tests and usage documentation.

### Adding a New Solver

1. Create `optimization/solvers/my_solver.py` implementing all three layers.
2. Add enum entry to `PortfolioObjective` in `config.py`.
3. Add dispatch branch in `wrapper_rolling_portfolios.py`.
4. Export from `optimization/solvers/__init__.py`.
5. Add tests in `optimization/solvers/tests/`.

### Adding a New Alpha Signal

1. Create `alphas/signals/new_signal.py` with `compute_new_signal_alpha()` following the standard `(prices, returns_freq, group_data, **params) → (score, raw)` interface.
2. Handle both single-freq and mixed-freq via `isinstance(returns_freq, pd.Series)` check.
3. Export from `alphas/signals/__init__.py` and `alphas/__init__.py`.
4. Add tests in `alphas/tests/`.
5. The signal is ready for use by any aggregator (the routing logic lives outside this package).

### Deleted in v4.1.1

- `utils/factor_alphas.py` — all functions migrated to `alphas/signals/`. Do NOT recreate.
- `utils/manager_alphas.py` — `AlphasData` moved to `alphas/alpha_data.py`, `compute_joint_alphas()` moved to private `rosaa.alphas.AlphaAggregator`. Do NOT recreate.
- `reports/backtest_alphas.py` — moved to `alphas/backtest_alphas.py`.

### Key External Dependencies

- **`qis`** (QuantInvestStrats) — data loading, `TimePeriod`, performance analytics, backtesting utilities. Most data structures align with `qis` conventions.
- **`cvxpy`** — primary convex optimization backend (default solver: `CLARABEL`).
- **`scipy`** — secondary optimization backend for some solvers (SLSQP).
- **`pyrb`** — risk budgeting solver backend (forked within this repo). Uses Spinu (2013) convex reformulation via ADMM — preferred over scipy SLSQP for risk budgeting.

### References

- Sepp A., Ossa I., Kastenholz M. (2026), "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios", *JPM* 52(4), 86-120.
- Sepp A., Hansen E., Kastenholz M. (2026), "Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors", *Working Paper*.
- Sepp A. (2023), "Optimal Allocation to Cryptocurrencies in Diversified Portfolios", *Risk Magazine*, October 2023, 1-6.