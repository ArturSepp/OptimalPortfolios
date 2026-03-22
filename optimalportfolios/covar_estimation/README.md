# Covariance Estimation Module

Covariance matrix estimation framework for the `optimalportfolios` package.

This module implements multiple covariance estimators sharing a common
abstract interface (`CovarEstimator`), rolling estimation with
mixed-frequency support, and diagnostic reporting. The factor model data
containers (`CurrentFactorCovarData`, `RollingFactorCovarData`,
`VarianceColumns`) are provided by the
[`factorlasso`](https://github.com/ArturSepp/factorlasso) package and
re-exported here. All estimators produce
`Dict[pd.Timestamp, pd.DataFrame]` for direct use by the optimization module.

## Architecture

```
covar_estimation/
├── covar_estimator.py          # ABC: shared interface for all estimators
├── ewma_covar_estimator.py     # EWMA covariance (no factor model)
├── factor_covar_estimator.py   # LASSO-based factor covariance (uses factorlasso)
├── covar_reporting.py          # Diagnostic plots (clusters, betas, R²)
├── utils.py                    # Shared utilities (returns computation)
└── tests/
    └── factor_covar_estimator_test.py

# External (from factorlasso package):
#   CurrentFactorCovarData      — single-date factor covariance decomposition
#   RollingFactorCovarData      — time-indexed collection of CurrentFactorCovarData
#   VarianceColumns             — column name enum for y_variances DataFrame
#   LassoModel                  — sparse factor model estimator (fit/predict/score)
```

### Estimator hierarchy

```
CovarEstimator (ABC)
├── fit_current_covar(**kwargs) → pd.DataFrame
├── fit_rolling_covars(**kwargs) → Dict[Timestamp, DataFrame]
└── rebalancing_freq: str
        │
        ├── EwmaCovarEstimator
        │     Input: prices
        │     No decomposition, single-pass O(T) tensor
        │
        └── FactorCovarEstimator
              Input: risk_factor_prices + asset_returns_dict
              Decomposition: Σ_y = β Σ_x β' + D
              Additional: fit_current_factor_covars() → CurrentFactorCovarData
                          fit_rolling_factor_covars() → RollingFactorCovarData
```

Both estimators share the output contract
`fit_rolling_covars() → Dict[Timestamp, DataFrame]`, so the optimization
module is agnostic to the estimation method. `FactorCovarEstimator`
additionally exposes the full decomposition (betas, residuals, clusters,
R²) via the factor-specific methods.

### Why factorlasso is a separate package

The LASSO/Group LASSO/HCGL estimator was originally part of
`optimalportfolios` (the `lasso/` module). It was extracted into
[`factorlasso`](https://github.com/ArturSepp/factorlasso) for three reasons:

1. **The solver is domain-agnostic.** Sign-constrained sparse regression
   with prior-centered regularisation applies to genomics, macro-econometrics,
   signal processing — not just finance. Bundling it inside a portfolio
   package limits its audience and makes it invisible to non-finance users.

2. **Independent release and citation.** `factorlasso` targets a JMLR
   MLOSS (Machine Learning Open Source Software) publication. JMLR requires
   a standalone package with its own test suite, CI, and documentation —
   not a submodule of a larger framework.

3. **Dependency isolation.** `factorlasso` depends only on numpy, pandas,
   scipy, and cvxpy. It does not require `qis`, `matplotlib`, `numba`,
   or any of the 13 other dependencies that `optimalportfolios` pulls in.
   Users who only need the LASSO solver install a 4-dependency package
   instead of a 13-dependency one.

### Separation of concerns: factorlasso vs covar_estimation

[`factorlasso`](https://github.com/ArturSepp/factorlasso) is the
**domain-agnostic** sparse factor model solver. It provides:
- `LassoModel` — estimates β in Y_t = α + β X_t + ε_t with sign
  constraints, prior-centered regularisation, and HCGL clustering
- `CurrentFactorCovarData` — assembles Σ_y = β Σ_x β' + D, computes
  model vols, systematic/residual decomposition, save/load to Excel
- `RollingFactorCovarData` — time-indexed panel accessors (R², betas,
  vols, alphas over time)

`factorlasso` knows nothing about finance, asset prices, frequencies,
annualisation, or rebalancing schedules. It works with raw numpy/pandas
DataFrames of any domain (finance, genomics, macro-econometrics).

This module adds the **finance-specific** layers:

**`estimate_lasso_factor_covar_data()`** — the bridge function:
- Computes factor returns from prices at the specified frequency
- Estimates annualised factor covariance Σ_x via EWMA
- Calls `factorlasso.LassoModel.fit()` separately per frequency for
  mixed-frequency universes (e.g., monthly equities + quarterly alternatives)
- Annualises residual variances, R², and alphas across frequencies
- Merges multi-frequency betas into a single (N × M) loading matrix
- Returns a `factorlasso.CurrentFactorCovarData` with the full decomposition

**`FactorCovarEstimator`** — the rolling wrapper:
- Generates rebalancing schedules via `qis.TimePeriod`
- Calls `estimate_lasso_factor_covar_data()` at each date with
  expanding-window data
- Exposes two APIs: plain covar dict for solvers, full decomposition
  for diagnostics and reporting

## Mathematical Framework

### Factor Model Covariance Decomposition

The factor model decomposes the asset covariance matrix as:

```
Σ_y = β Σ_x β' + D
```

where:
- `β` is the factor loadings matrix (N × M), estimated via LASSO / Group LASSO / HCGL
- `Σ_x` is the factor covariance matrix (M × M), estimated via EWMA
- `D = diag(σ²_ε)` is the diagonal idiosyncratic variance matrix (N × N)
- `Σ_y` is the resulting asset covariance matrix (N × N)

The loadings β are sparse: the LASSO penalty shrinks irrelevant factor
exposures to exactly zero, producing interpretable and stable estimates
even when N > M or factors are correlated.

### Variance Decomposition per Asset

For each asset i:

```
Var(y_i) = β_i' Σ_x β_i  +  σ²_ε,i
           ───────────────    ─────────
           systematic var     idiosyncratic var
```

The R² diagnostic measures the fraction explained by factors:

```
R²_i = 1 - σ²_ε,i / Var(y_i)
```

### Mixed-Frequency Annualisation

When factor covariance and asset returns are estimated at different
frequencies, each component is annualised independently:

```
Σ_y(annual) = β' (a_f · Σ_x) β  +  a_ε · D
```

where `a_f` is the annualisation factor for the factor frequency and
`a_ε` is the annualisation factor for each asset's return frequency.

This is the key enabler for mixed-frequency estimation: equity betas
can be estimated from daily returns while bond betas use monthly returns,
and both are combined into a single annualised covariance matrix.

### LASSO Estimation Methods

Three sparse regression methods are supported, all implemented via CVXPY
in the `factorlasso` package:

| Method | `LassoModelType` | Penalty | Groups |
|--------|------------------|---------|--------|
| Standard LASSO | `LASSO` | λ ‖β - β₀‖₁ | Independent per element |
| Group LASSO | `GROUP_LASSO` | Σ_g λ √(\|g\|/G) ‖β_g - β₀_g‖₂ | User-defined asset groups |
| HCGL | `GROUP_LASSO_CLUSTERS` | Same as Group LASSO | Hierarchical clustering from correlation |

All methods support:
- **Sign constraints** on factor loadings (non-negative, non-positive, zero, free)
- **Prior-centered regularisation**: penalty ‖β - β₀‖ instead of ‖β‖
- **EWMA weighting** of observations for time-varying loadings
- **NaN-aware estimation** via validity masking (assets with different history lengths)

### EWMA Covariance

The EWMA estimator computes the exponentially weighted covariance tensor
in a single O(T) pass, then extracts slices at rebalancing dates:

```
Σ_t = (1 - λ) r_t r_t' + λ Σ_{t-1}
```

where λ = 1 - 2/(span+1). Optional features:
- **Vol-normalised returns** (DCC-like): normalise by rolling vol before estimation, then rescale

## Data Containers (from factorlasso)

### `CurrentFactorCovarData`

Dataclass holding the factor model decomposition at a single date:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `x_covar` | (M × M) | Factor covariance Σ_x |
| `y_betas` | (N × M) | Factor loadings β (index=assets, columns=factors) |
| `y_variances` | (N × 4) | Per-asset diagnostics: ewma_var, residual_var, insample_alpha, r2 |
| `residuals` | (T × N) | In-sample residuals ε_t = y_t - x_t β', annualised |
| `clusters` | Dict[str, Series] | HCGL cluster assignments per frequency |
| `linkages` | Dict[str, ndarray] | Dendrograms per frequency |
| `cutoffs` | Dict[str, float] | Dendrogram cut thresholds |

Key methods:

```python
data.y_covar                                    # Σ_y = β Σ_x β' + D (property)
data.get_y_covar(residual_var_weight=0.5)       # Σ_y with scaled D
data.get_snapshot()                             # betas + R² + vols + alpha (DataFrame)
data.get_model_vols()                           # total, systematic, residual vols
data.estimate_alpha(alpha_span=120)             # rolling alpha from residuals
data.filter_on_tickers(['SPY', 'AGG'])          # subset to selected assets
data.save('covar_data.xlsx')                    # serialise to Excel
CurrentFactorCovarData.load('covar_data.xlsx')  # load from Excel
```

### `RollingFactorCovarData`

Container for `Dict[Timestamp, CurrentFactorCovarData]` with panel accessors:

```python
rolling.get_y_covars()              # Dict[Timestamp, DataFrame] — for optimisation
rolling.get_r2()                    # DataFrame (dates × assets)
rolling.get_systematic_vars()       # DataFrame (dates × assets) — diag(β Σ_x β')
rolling.get_residual_vars()         # DataFrame (dates × assets)
rolling.get_total_vols()            # DataFrame (dates × assets) — √(sys + resid)
rolling.get_residual_vols()         # DataFrame (dates × assets)
rolling.get_alphas(alpha_span=120)  # DataFrame (dates × assets)
rolling.get_beta(factor='SPY')      # DataFrame (dates × assets) for one factor
rolling.get_snapshot()              # Dict[Timestamp, DataFrame] — full diagnostics
```

### `VarianceColumns` Enum

Column names in `y_variances` follow a consistent naming convention:

| Column | Enum | Source |
|--------|------|--------|
| `ewma_var` | `EWMA_VARIANCE` | EWMA-weighted total variance of y |
| `residual_var` | `RESIDUAL_VARS` | Residual variance of y - βx |
| `insample_alpha` | `INSAMPLE_ALPHA` | EWMA mean of residuals (in-sample) |
| `r2` | `R2` | 1 - residual_var / ewma_var |
| `stat_alpha` | `ALPHA` | EWM of residuals at alpha_span (derived) |
| `total_vol` | `TOTAL_VOL` | √(systematic_var + residual_var) (derived) |
| `sys_vol` | `SYST_VOL` | √(β' Σ_x β) (derived) |
| `resid_vol` | `RESID_VOL` | √(residual_var) (derived) |

The first four are stored in `y_variances`; the rest are computed on demand
by `get_model_vols()` and `get_snapshot()`.

## Usage Examples

### EWMA covariance (simplest case)

```python
import optimalportfolios as opt
import qis

estimator = opt.EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')

# single date
current_covar = estimator.fit_current_covar(prices=prices)

# rolling
covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)
```

### Factor LASSO covariance

```python
lasso_model = opt.LassoModel(
    model_type=opt.LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-5, span=36, warmup_period=12)

estimator = opt.FactorCovarEstimator(
    lasso_model=lasso_model,
    factor_returns_freq='ME',
    rebalancing_freq='QE')

asset_returns_dict = qis.compute_asset_returns_dict(
    prices=asset_prices, is_log_returns=True, returns_freqs='ME')

# shared interface — plain covar dict
covar_dict = estimator.fit_rolling_covars(
    risk_factor_prices=factor_prices,
    asset_returns_dict=asset_returns_dict,
    time_period=time_period)

# factor-specific — full decomposition
rolling_data = estimator.fit_rolling_factor_covars(
    risk_factor_prices=factor_prices,
    asset_returns_dict=asset_returns_dict,
    time_period=time_period)

# access diagnostics
r2_panel = rolling_data.get_r2()
sys_vars = rolling_data.get_systematic_vars()
total_vols = rolling_data.get_total_vols()
```

### Mixed-frequency asset returns

```python
# equities at daily, bonds at weekly, alternatives at monthly
returns_freqs = pd.Series({
    'SPY': 'B', 'EZU': 'B', 'EEM': 'B',
    'TLT': 'W-WED', 'HYG': 'W-WED',
    'GLD': 'ME'})

asset_returns_dict = qis.compute_asset_returns_dict(
    prices=asset_prices, is_log_returns=True, returns_freqs=returns_freqs)

# factor covariance estimated at weekly frequency
estimator = opt.FactorCovarEstimator(
    lasso_model=lasso_model,
    factor_returns_freq='W-WED',
    rebalancing_freq='QE')

# betas estimated per-frequency, annualised independently, merged
covar_dict = estimator.fit_rolling_covars(
    risk_factor_prices=factor_prices,
    asset_returns_dict=asset_returns_dict,
    time_period=time_period)
```

### Diagnostic reporting

```python
from optimalportfolios.covar_estimation.covar_reporting import (
    plot_current_covar_data, run_rolling_covar_report)

# single-date diagnostics
figs = plot_current_covar_data(covar_data=rolling_data.get_latest())

# rolling diagnostics with cluster plots
figs, dfs = run_rolling_covar_report(
    risk_factor_prices=factor_prices,
    prices=asset_prices,
    covar_estimator=estimator,
    time_period=time_period,
    asset_returns_dict=asset_returns_dict)
```

## References

Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation
for Multi-Asset Portfolios",
*The Journal of Portfolio Management*, 52(4), 86-120.
[Paper link](https://eprints.pm-research.com/17511/143431/index.html)

Sepp A., Hansen E., and Kastenholz M. (2026),
"Capital Market Assumptions and Strategic Asset Allocation
Using Multi-Asset Tradable Factors",
*Under revision at the Journal of Portfolio Management*.