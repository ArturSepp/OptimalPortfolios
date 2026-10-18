---
myst:
  html_meta:
    description: >-
      Retained EWMA covariance interfaces in OptimalPortfolios and the completed
      factor-model handoff from TradableFactors to optimisation and QIS risk analytics.
---

# Covariance inputs for portfolio optimisation

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2025-08-10](https://github.com/ArturSepp/OptimalPortfolios/commit/2057085d2fc04fe8b6876f05ad513a358797f2c3)*

Contributor documentation for [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Portfolio optimisers consume labelled, annual asset covariance matrices. This module
retains direct EWMA estimation using QIS. The complete factor-model pipeline now
belongs to [TradableFactors](https://github.com/ArturSepp/tradablefactors).

## Retained interfaces

| Module | Responsibility |
|---|---|
| [Exports](./__init__.py) | EWMA estimator and return preparation helpers. |
| [covar_estimator.py](./covar_estimator.py) | Abstract current/rolling covariance interface. |
| [ewma_covar_estimator.py](./ewma_covar_estimator.py) | Direct EWMA orchestration using QIS kernels. |
| [utils.py](./utils.py) | Return preparation through QIS. |

`EwmaCovarEstimator.fit_current_covar` returns one annual covariance DataFrame;
`fit_rolling_covars` returns a date-keyed dictionary of those matrices. Asset order,
return convention, frequency and estimation cutoff must match the optimisation inputs.
The [EWMA methodology](../../../docs/covariance_estimators.md) describes the retained
numerical contract. The estimator has not changed in this migration.

## Factor-model handoff

Use `tradablefactors.FactorCovarEstimator` with an explicit FactorLasso `LassoModel`.
`fit_current(..., cutoff=...)` returns a completed `FactorModelSnapshot`;
`fit_rolling(..., time_period=...)` returns `RollingFactorModel`.
Pass `.asset_covariance` or `.covariances` to the relevant OP optimisation workflow.
Keep the TF result when factor loadings, residual histories or diagnostics are needed.
Export its QIS risk view with `.to_risk_model()` for portfolio analytics and stress reports.
For a plain covariance dictionary, use `qis.RiskModel(covar=...)` directly.

TF owns covariance preparation, factor fitting, residual-risk assembly and
method-specific covariance reports. FactorLasso owns the statistical estimator.
OP owns optimisation for supplied expected returns and covariance.

The removed OP estimator, GLS/CMA, covariance-reporting and risk-adapter modules
have no forwarding implementations. See the [direct migration guide](https://github.com/ArturSepp/tradablefactors/blob/main/docs/migration.md).
Historical reproduction uses its frozen prior source/dependency cohort.

## Development checkpoint

Stages 0–3 move numerical ownership. The retained OP result and opportunity code
still use FactorLasso/TF within the composed development environment. Stage 5
completes the supplied-input boundary before a release. Older documentation exhibits
retain their historical receipts; they are not evidence of a refreshed TF release.
