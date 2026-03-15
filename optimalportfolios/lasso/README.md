# LASSO Module

Sparse factor model estimation for the `optimalportfolios` package.

This module implements LASSO and Group LASSO regression via CVXPY for
estimating sparse factor loadings in the model Y = Xβ' + ε. It is the
core estimation engine used by `FactorCovarEstimator` to produce the
factor covariance decomposition Σ_y = β Σ_x β' + D.

## Architecture

```
lasso/
├── lasso_estimator.py      # LassoModel dataclass, solver functions, clustering
└── tests/
    └── lasso_estimator_test.py   # Synthetic data tests for all model types
```

### Key components

| Component | Description |
|-----------|-------------|
| `LassoModel` | Configurable dataclass estimator. Holds parameters, calls solver, stores results |
| `LassoModelType` | Enum: `LASSO`, `GROUP_LASSO`, `GROUP_LASSO_CLUSTERS` (HCGL) |
| `LassoEstimationResult` | Output container: beta (N×M), alpha, ss_total, ss_res, r2 |
| `solve_lasso_cvx_problem` | Numpy-level L1 LASSO solver via CVXPY |
| `solve_group_lasso_cvx_problem` | Numpy-level Group LASSO solver via CVXPY |
| `get_x_y_np` | Prepares numpy arrays with NaN masking and EWMA demeaning |
| `compute_clusters_from_corr_matrix` | Ward's hierarchical clustering for HCGL |

### Three-layer pattern

The module follows the same layered design as the optimization solvers:

| Layer | Function | Input | Output |
|-------|----------|-------|--------|
| **Model** | `LassoModel.fit()` | `pd.DataFrame` (x, y) | `LassoModel` (self, with estimated_betas) |
| **Preparation** | `get_x_y_np()` | `pd.DataFrame` | `np.ndarray` (x, y, valid_mask) |
| **Solver** | `solve_lasso_cvx_problem()` / `solve_group_lasso_cvx_problem()` | `np.ndarray` | `LassoEstimationResult` |

`LassoModel.fit()` orchestrates the full pipeline: prepare data → select
solver based on `model_type` → apply warmup filtering → store results.

## Mathematical Framework

### Factor Model Convention

The factor model follows the paper convention:

```
Y_t = β X_t + ε_t
```

where Y_t is (N×1), X_t is (M×1), β is (N×M), and ε_t is (N×1).
In matrix form for T observations:

```
Y = X β' + E
```

where Y is (T×N), X is (T×M), β is (N×M), E is (T×N).

The `estimated_betas` DataFrame has `index=assets` (N rows) and
`columns=factors` (M columns), matching the paper convention.

### LASSO (L1 Regularisation)

Standard element-wise sparsity:

```
min_β  (1/T) ‖W ⊙ (Xβ' - Y)‖²_F  +  λ ‖β - β₀‖₁
```

where W is the observation weight matrix (EWMA × valid_mask), λ is
the regularisation strength, and β₀ is the optional prior.

The L1 penalty shrinks individual beta elements to zero independently,
producing sparse loadings where each asset loads on only a few factors.

### Group LASSO (L2/L1 Regularisation)

Group-level sparsity over assets:

```
min_β  (1/T) ‖W ⊙ (Xβ' - Y)‖²_F  +  Σ_g λ √(|g|/G) ‖β_g - β₀_g‖₂
```

where g indexes groups of assets (rows of β), |g| is the group size,
and G is the total number of groups. The L2 norm within each group
encourages entire groups of assets to share the same sparsity pattern
(all load on a factor or none do).

Groups can be:
- **Predefined** (`GROUP_LASSO`): user-specified via `group_data` (e.g., asset class labels)
- **Data-driven** (`GROUP_LASSO_CLUSTERS` / HCGL): discovered via hierarchical clustering of the asset correlation matrix

### HCGL (Hierarchical Clustering Group LASSO)

The HCGL method automates group discovery:

1. Compute EWMA correlation matrix of asset returns
2. Convert to distance: d = 1 - ρ
3. Apply Ward's agglomerative clustering
4. Cut dendrogram at 50% of maximum pairwise distance
5. Use resulting clusters as groups for Group LASSO

This produces data-adaptive groups that reflect the current correlation
structure, automatically capturing regime changes in asset relationships.

### EWMA Observation Weighting

The solver objective weights observations exponentially:

```
W_t = √(λ)^(T-t) × valid_mask_t
```

where λ = 1 - 2/(span+1) is the EWMA decay parameter. The square root
is necessary because the objective squares the weighted residuals:

```
‖W ⊙ (Xβ' - Y)‖²_F = Σ_t Σ_i W²_ti (x_t β_i - y_ti)²
```

So W_t = √(λ)^(T-t) produces decay λ^(T-t) on the squared residuals.

### NaN Handling via Validity Masking

Instead of removing rows with NaN (which discards valid observations
for other assets), a binary valid_mask is applied element-wise:

```
valid_mask_ti = 1  if y_ti is observed
                0  if y_ti is NaN
```

This is multiplied into the observation weights W, zeroing out the
contribution of missing observations in the objective function while
preserving all valid data for each asset independently.

NaN values in y are filled with 0.0 before solving (their contribution
is zeroed out by the mask, so the fill value is irrelevant).

### Prior-Centered Regularisation

When `factors_beta_prior` (β₀) is provided, the penalty becomes:

```
λ ‖β - β₀‖  instead of  λ ‖β‖
```

This shrinks estimates toward the prior rather than toward zero.
NaN entries in β₀ are treated as zero (standard shrinkage for those
elements). The prior is subtracted directly inside the penalty norm —
no variable substitution is needed.

Use cases:
- Warm-starting from a previous estimation period
- Incorporating economic priors (e.g., equity assets should load positively on equity factors)
- Stabilising estimates when data is scarce

### Sign Constraints

The `factors_beta_loading_signs` matrix (N×M) constrains individual
beta elements:

| Value | Constraint |
|-------|-----------|
| `1.0` | β_ij ≥ 0 (non-negative) |
| `-1.0` | β_ij ≤ 0 (non-positive) |
| `0.0` | β_ij = 0 (forced zero) |
| `NaN` | β_ij free (unconstrained) |

Sign constraints are implemented as CVXPY element-wise inequality
constraints using mask multiplication. They are orthogonal to the
LASSO penalty — a beta can be sign-constrained and regularised
simultaneously.

### In-Sample Diagnostics

After solving, `_compute_solver_diagnostics` produces per-asset metrics
using the same EWMA weights as the solver:

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| `alpha` | Σ_t w_t ε_ti | Weighted mean residual (model bias) |
| `ss_total` | Σ_t w_t (y_ti - ȳ_i)² | Weighted total variance |
| `ss_res` | Σ_t w_t ε²_ti | Weighted residual variance |
| `r2` | 1 - ss_res / ss_total | Fraction of variance explained |

Weights w_t are the solver weights normalised per-column (no additional
squaring, since `np.square` is applied explicitly).

## Usage

### Basic LASSO

```python
from optimalportfolios import LassoModel, LassoModelType

model = LassoModel(
    model_type=LassoModelType.LASSO,
    reg_lambda=1e-5,
    span=36,
    warmup_period=12,
)
model.fit(x=factor_returns, y=asset_returns)

# estimated_betas: DataFrame (N×M), index=assets, columns=factors
print(model.estimated_betas)

# diagnostics
print(f"R²: {model.estimation_result_.r2}")
```

### HCGL (Group LASSO with automatic clustering)

```python
model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-5,
    span=36,
    warmup_period=12,
)
model.fit(x=factor_returns, y=asset_returns)

# discovered clusters
print(model.clusters)
print(f"Cutoff: {model.cutoff}")
```

### Group LASSO with predefined groups

```python
import pandas as pd

group_data = pd.Series({
    'SPY': 'Equity', 'EZU': 'Equity', 'EEM': 'Equity',
    'TLT': 'Bonds', 'HYG': 'Credit', 'GLD': 'Commodity',
})

model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO,
    group_data=group_data,
    reg_lambda=1e-5,
    span=36,
)
model.fit(x=factor_returns, y=asset_returns)
```

### Sign constraints

```python
import numpy as np

# equity assets must load non-negatively on equity factor,
# bond assets forced to zero on equity factor
signs = pd.DataFrame(
    [[1.0, np.nan],    # SPY: equity>=0, bond free
     [1.0, np.nan],    # EZU: equity>=0, bond free
     [0.0, np.nan],    # TLT: equity=0, bond free
     [0.0, np.nan]],   # AGG: equity=0, bond free
    index=['SPY', 'EZU', 'TLT', 'AGG'],
    columns=['Equity_Factor', 'Bond_Factor'],
)

model = LassoModel(
    model_type=LassoModelType.LASSO,
    reg_lambda=1e-5,
    span=36,
    factors_beta_loading_signs=signs,
)
model.fit(x=factor_returns, y=asset_returns)
```

### Prior-centered regularisation

```python
# shrink toward a prior from previous estimation
prior_betas = previous_model.estimated_betas

model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-5,
    span=36,
    factors_beta_prior=prior_betas,
)
model.fit(x=factor_returns, y=asset_returns)
```

### Mixed-frequency span

```python
# different EWMA spans for different return frequencies
model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
    reg_lambda=1e-5,
    span=36,                           # default span
    span_freq_dict={'ME': 36, 'QE': 12},  # frequency-specific overrides
    warmup_period=12,
)

# FactorCovarEstimator passes freq key when calling fit()
```

## Test Coverage

The test file `lasso_estimator_test.py` covers seven scenarios using
synthetic data (y = x @ true_betas.T + noise):

| Test | What it verifies |
|------|-----------------|
| `LASSO_BASIC` | Sparse recovery, DataFrame orientation, diagnostics shapes |
| `LASSO_WITH_NANS` | NaN masking for assets with different history lengths |
| `GROUP_LASSO_PREDEFINED` | Group penalty with user-defined asset groups |
| `GROUP_LASSO_CLUSTERS_HCGL` | Automatic clustering, dendrogram cutoff |
| `SIGN_CONSTRAINTS` | Per-element sign constraints (≥0, ≤0, =0, free) |
| `GET_X_Y_NP_STANDALONE` | Data preparation, EWMA demeaning, index assertions |
| `SOLVER_STANDALONE_WITH_NANS` | Raw numpy solver with internal vs explicit valid_mask |

Each test generates data with known true betas and verifies that the
estimated betas recover the true sparsity pattern within tolerance.

## Integration with FactorCovarEstimator

`LassoModel` is a parameter of `FactorCovarEstimator`. The estimator
calls `model.fit(x, y, span=span_f)` at each frequency in the
`asset_returns_dict`, then assembles the results into
`CurrentFactorCovarData`:

```
FactorCovarEstimator.fit_current_factor_covars()
    │
    ├── for each freq in asset_returns_dict:
    │       LassoModel.fit(x=factor_returns, y=asset_returns[freq])
    │       → estimated_betas (N_freq × M)
    │       → estimation_result_ (alpha, ss_total, ss_res, r2)
    │       → clusters, linkage, cutoff (HCGL only)
    │
    └── merge across frequencies → CurrentFactorCovarData
            y_betas: (N × M)
            y_variances: (N × 4) [ewma_var, residual_var, insample_alpha, r2]
            x_covar: (M × M)
            residuals, clusters, linkages, cutoffs
```

The `span_freq_dict` parameter on `LassoModel` allows different EWMA
spans for different return frequencies (e.g., span=36 for monthly,
span=12 for quarterly), which `FactorCovarEstimator` selects automatically
based on the frequency key.

## References

Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation
for Multi-Asset Portfolios",
*The Journal of Portfolio Management*, 52(4), 86-120.
Available at https://www.pm-research.com/content/iijpormgmt/52/4/86

Sepp A., Hansen E., and Kastenholz M. (2026),
"Capital Market Assumptions and Strategic Asset Allocation
Using Multi-Asset Tradable Factors",
*Working Paper*.