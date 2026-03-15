# Robust Optimization of Strategic Asset Allocation

Example implementation of the Hierarchical Clustering Group Lasso (HCGL)
covariance estimation method and risk-budgeted Strategic Asset Allocation (SAA)
as introduced in:

**Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
*The Journal of Portfolio Management*, 52(4), 86-120.**

Published version: https://www.pm-research.com/content/iijpormgmt/52/4/86


## What this example demonstrates

The script `hcgl_covar_for_rolling_backtest.py` implements a minimal end-to-end
pipeline from the paper:

1. **HCGL covariance estimation** — rolling factor model where asset betas
   are estimated via Group LASSO with hierarchical clustering of assets into
   groups. The LASSO penalty encourages sparsity (most assets load on a
   small number of factors), while the group structure ensures assets within
   the same cluster share factor exposure patterns.

2. **Risk-budgeted SAA** — constrained risk budgeting where each asset's
   contribution to portfolio risk matches a prescribed budget. Uses the
   `pyrb` package (forked within `optimalportfolios`) for the ADMM-based
   solver with linear inequality constraints.

3. **Rolling backtest** — covariance matrices are estimated on an expanding
   window (no look-ahead), the SAA portfolio is rebalanced quarterly, and
   the resulting weights are backtested with realistic transaction costs and
   implementation lag.


## HCGL method overview

The paper introduces a three-step covariance estimation procedure:

**Step 1: Factor model estimation.** For each asset *i* and factor *j*,
estimate the EWMA regression beta β_{ij} using Group LASSO:

    min_β  Σ_t λ^{T-t} ||r_t - B f_t||² + λ Σ_g ||β_g||₂

where *g* indexes asset groups (determined by hierarchical clustering of
the asset correlation matrix) and the group norm penalty encourages entire
groups of betas to be zero simultaneously.

**Step 2: Factor covariance.** Estimate the factor covariance matrix Σ_f
using EWMA on factor returns.

**Step 3: Asset covariance reconstruction.**

    Σ_y = B Σ_f B' + D

where D is the diagonal residual variance matrix. The residual variance
weight parameter controls the blend between the pure factor model (D=0,
for TAA tracking error) and the full model (D included, for SAA risk budgeting).


## Running the example

```bash
pip install optimalportfolios yfinance qis
python hcgl_covar_for_rolling_backtest.py
```

The script uses `yfinance` to download price data for a 6-asset multi-asset
universe: SPY (US equities), EZU (Europe equities), EEM (EM equities),
TLT (US Treasuries), HYG (High Yield), GLD (Gold).


## Code walkthrough

```python
from optimalportfolios import (Constraints, LassoModelType, LassoModel,
                               FactorCovarEstimator, rolling_risk_budgeting)

# 1. configure the LASSO factor model
lasso_model = LassoModel(
    model_type=LassoModelType.GROUP_LASSO_CLUSTERS,  # hierarchical clustering groups
    reg_lambda=1e-5,   # LASSO regularisation strength
    span=36,           # EWMA half-life in months for beta estimation
    warmup_period=12)  # minimum months before first beta estimate

# 2. configure the factor covariance estimator
covar_estimator = FactorCovarEstimator(
    lasso_model=lasso_model,
    factor_returns_freq='ME',   # monthly factor returns
    rebalancing_freq='QE',      # quarterly covariance updates
    factor_covar_span=36)       # EWMA span for factor covariance

# 3. estimate rolling covariance matrices
asset_returns_dict = qis.compute_asset_returns_dict(
    prices=universe_prices, is_log_returns=True, returns_freqs='ME')
rolling_covar_data = covar_estimator.fit_rolling_factor_covars(
    risk_factor_prices=risk_factor_prices,
    asset_returns_dict=asset_returns_dict,
    time_period=time_period)

# 4. solve risk budgeting with the estimated covariance
risk_budget = {asset: 1.0 / n_assets for asset in assets}
saa_weights = rolling_risk_budgeting(
    prices=universe_prices,
    covar_dict=rolling_covar_data.y_covars,
    risk_budget=risk_budget,
    constraints=Constraints(is_long_only=True))
```

The key insight from the paper is that the HCGL covariance estimator
produces more stable portfolio weights than sample EWMA covariance because
the factor model structure reduces estimation noise — particularly important
for multi-asset universes with heterogeneous asset classes, mixed return
frequencies, and illiquid alternatives.


## Key parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `reg_lambda` | 1e-5 | LASSO penalty. Higher → sparser betas, more factor model shrinkage |
| `span` | 36 | EWMA half-life in months for beta estimation. 36 = 3 years |
| `factor_returns_freq` | 'ME' | Monthly factor returns for beta regression |
| `rebalancing_freq` | 'QE' | Quarterly SAA rebalancing |
| `residual_var_weight` | 1.0 (SAA) / 0.0 (TAA) | Include residual variance for SAA risk budgeting; exclude for TAA tracking error |


## Connection to the full ROSAA framework

This example demonstrates the SAA component of the ROSAA (Robust Optimization
of Strategic and Tactical Asset Allocation) framework. The full production
implementation additionally includes:

- **Tactical Asset Allocation (TAA)** with alpha-over-tracking-error
  optimisation using momentum, low-beta, and managers alpha signals
- **Mixed-frequency covariance estimation** for universes combining
  monthly-rebalanced equities with quarterly-rebalanced alternatives
- **Group constraints** for asset class allocation bounds
- **Turnover management** via utility penalties or hard constraints

These production components are implemented in the private `rosaa` package,
while the building blocks (covariance estimation, alpha signals, optimisation
solvers) are available in the public `optimalportfolios` package.


## References

Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
*The Journal of Portfolio Management*, 52(4), 86-120.

Sepp A., Hansen E., and Kastenholz M. (2026),
"Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors",
*Working Paper*.

Spinu F. (2013),
"An Algorithm for Computing Risk Parity Weights",
*SSRN Working Paper*.