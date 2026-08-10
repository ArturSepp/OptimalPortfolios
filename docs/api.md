---
icon: material/code-tags
---

# API Reference

**Every name on this page is importable from the package root**, whatever heading it
appears under:

```python
from optimalportfolios import AlphasData, Constraints, EwmaCovarEstimator
```

The grouping follows where each name is defined, because that is where the root namespace
gets it: `optimalportfolios/__init__.py` re-exports from `config`, `utils`,
`covar_estimation`, `optimization`, `universe`, `reports` and `alphas`, plus nine names
from `factorlasso` and one from `qis`.

Submodules reachable through the root namespace (`optimalportfolios.optimization` and
friends) are import plumbing rather than API, and are not listed.

## Configuration

The objective enum every optimiser dispatches on. Extend this rather than passing raw
strings.

::: optimalportfolios.config.PortfolioObjective

## Alpha signals

Signal construction and diagnostics. The naming convention is `compute_<signal>_alpha`
for a cross-sectional score and `compute_<signal>_cluster_alpha` for the within-cluster
variant; `profile_<signal>` builds the diagnostic profile for one signal family.

::: optimalportfolios.alphas.AlphasData

::: optimalportfolios.alphas.ProfileSignal

::: optimalportfolios.alphas.align_rolling_clusters

::: optimalportfolios.alphas.backtest_alpha_rank_portfolio

::: optimalportfolios.alphas.compare_signal_diagnostics

::: optimalportfolios.alphas.compute_alpha_rank_analysis_table

::: optimalportfolios.alphas.compute_low_beta_alpha

::: optimalportfolios.alphas.compute_low_beta_cluster_alpha

::: optimalportfolios.alphas.compute_managers_alpha

::: optimalportfolios.alphas.compute_momentum_alpha

::: optimalportfolios.alphas.compute_momentum_cluster_alpha

::: optimalportfolios.alphas.compute_ra_carry_alphas

::: optimalportfolios.alphas.compute_residual_momentum_alpha

::: optimalportfolios.alphas.compute_residual_momentum_cluster_alpha

::: optimalportfolios.alphas.compute_residual_reversal_alpha

::: optimalportfolios.alphas.compute_residual_reversal_cluster_alpha

::: optimalportfolios.alphas.compute_top_quantile_equal_weights

::: optimalportfolios.alphas.estimate_rolling_ewma_means

::: optimalportfolios.alphas.extract_rolling_clusters

::: optimalportfolios.alphas.generate_alpha_profile_report

::: optimalportfolios.alphas.profile_alpha_signals

::: optimalportfolios.alphas.profile_carry

::: optimalportfolios.alphas.profile_low_beta

::: optimalportfolios.alphas.profile_momentum

::: optimalportfolios.alphas.profile_residual_momentum

::: optimalportfolios.alphas.run_signal_diagnostics

::: optimalportfolios.alphas.run_signal_diagnostics_per_component

::: optimalportfolios.alphas.score_within_clusters

::: optimalportfolios.alphas.signal_diagnostics_panel

## Covariance estimation

EWMA estimators and the HCGL sparse factor model. Both estimator classes expose
`fit_rolling_covars`, so an optimiser call is unchanged by the choice between them.

::: optimalportfolios.covar_estimation.EwmaCovarEstimator

::: optimalportfolios.covar_estimation.FactorCovarEstimator

::: optimalportfolios.covar_estimation.build_risk_model

::: optimalportfolios.covar_estimation.compute_returns_from_prices

::: optimalportfolios.covar_estimation.estimate_current_ewma_covar

::: optimalportfolios.covar_estimation.estimate_lasso_factor_covar_data

::: optimalportfolios.covar_estimation.plot_current_covar_data

::: optimalportfolios.covar_estimation.plot_hcgl_covar_data

::: optimalportfolios.covar_estimation.run_rolling_covar_report

## Optimisation

Constraints and solvers. Three layers, distinguishable by prefix: `rolling_*` runs a
solver across rebalancing dates, `wrapper_*` adapts one single-date solve to the common
signature, and `cvx_*` / `opt_*` are the single-date problems themselves.

::: optimalportfolios.optimization.ConstraintEnforcementType

::: optimalportfolios.optimization.ConstraintResidual

::: optimalportfolios.optimization.Constraints

::: optimalportfolios.optimization.CovarianceFactorization

::: optimalportfolios.optimization.GroupLowerUpperConstraints

::: optimalportfolios.optimization.GroupTrackingErrorConstraint

::: optimalportfolios.optimization.GroupTurnoverConstraint

::: optimalportfolios.optimization.OptimiserConfig

::: optimalportfolios.optimization.OptimizationOutcome

::: optimalportfolios.optimization.PortfolioOptimisationResult

::: optimalportfolios.optimization.backtest_rolling_optimal_portfolio

::: optimalportfolios.optimization.compute_eligible_rebalancing_bounds

::: optimalportfolios.optimization.compute_rolling_optimal_weights

::: optimalportfolios.optimization.cvx_max_return_target_vol

::: optimalportfolios.optimization.cvx_max_return_target_vol_utility

::: optimalportfolios.optimization.cvx_maximise_alpha_over_tre

::: optimalportfolios.optimization.cvx_maximise_alpha_with_target_return

::: optimalportfolios.optimization.cvx_maximise_tre_utility

::: optimalportfolios.optimization.cvx_maximize_portfolio_sharpe

::: optimalportfolios.optimization.cvx_min_variance_target_return

::: optimalportfolios.optimization.cvx_min_variance_target_return_utility

::: optimalportfolios.optimization.cvx_minimise_tracking_error

::: optimalportfolios.optimization.cvx_quadratic_optimisation

::: optimalportfolios.optimization.evaluate_constraint_residuals

::: optimalportfolios.optimization.factorize_covariance

::: optimalportfolios.optimization.merge_group_lower_upper_constraints

::: optimalportfolios.optimization.opt_maximise_diversification

::: optimalportfolios.optimization.opt_maximize_cara

::: optimalportfolios.optimization.opt_maximize_cara_mixture

::: optimalportfolios.optimization.opt_risk_budgeting

::: optimalportfolios.optimization.rolling_max_return_target_vol

::: optimalportfolios.optimization.rolling_maximise_alpha_over_tre

::: optimalportfolios.optimization.rolling_maximise_alpha_with_target_return

::: optimalportfolios.optimization.rolling_maximise_diversification

::: optimalportfolios.optimization.rolling_maximize_cara_mixture

::: optimalportfolios.optimization.rolling_maximize_portfolio_sharpe

::: optimalportfolios.optimization.rolling_min_variance_target_return

::: optimalportfolios.optimization.rolling_minimise_tracking_error

::: optimalportfolios.optimization.rolling_quadratic_optimisation

::: optimalportfolios.optimization.rolling_risk_budgeting

::: optimalportfolios.optimization.solve_analytic_log_opt

::: optimalportfolios.optimization.solve_for_risk_budgets_from_given_weights

::: optimalportfolios.optimization.wrapper_max_return_target_vol

::: optimalportfolios.optimization.wrapper_maximise_alpha_over_tre

::: optimalportfolios.optimization.wrapper_maximise_alpha_with_target_return

::: optimalportfolios.optimization.wrapper_maximise_diversification

::: optimalportfolios.optimization.wrapper_maximize_cara_mixture

::: optimalportfolios.optimization.wrapper_maximize_portfolio_sharpe

::: optimalportfolios.optimization.wrapper_min_variance_target_return

::: optimalportfolios.optimization.wrapper_minimise_tracking_error

::: optimalportfolios.optimization.wrapper_quadratic_optimisation

::: optimalportfolios.optimization.wrapper_risk_budgeting

## Universe

The instrument universe container and its metadata schema.

::: optimalportfolios.universe.MetadataField

::: optimalportfolios.universe.UniverseData

::: optimalportfolios.universe.copy_universe_data_with_unsmoothed_prices

## Reports

Reporting helpers. Factsheet generation itself goes through
[`qis`](https://github.com/ArturSepp/QuantInvestStrats); this package adds only what is
specific to portfolio construction.

::: optimalportfolios.reports.plot_efficient_frontier

## Utilities

Portfolio arithmetic shared across the pipeline — risk contributions, diversification
ratio, NaN-aware covariance filtering, and the self-financing weight drift applied
between rebalancing dates.

::: optimalportfolios.utils.apply_drift_to_weights_0

::: optimalportfolios.utils.calculate_diversification_ratio

::: optimalportfolios.utils.compute_portfolio_risk_contribution_outputs

::: optimalportfolios.utils.compute_portfolio_variance

::: optimalportfolios.utils.compute_portfolio_vol

::: optimalportfolios.utils.compute_risk_contributions

::: optimalportfolios.utils.compute_tre_turnover_stats

::: optimalportfolios.utils.filter_covar_and_vectors

::: optimalportfolios.utils.filter_covar_and_vectors_for_nans

::: optimalportfolios.utils.fit_gaussian_mixture

::: optimalportfolios.utils.round_weights_to_pct

## Re-exported from factorlasso

Owned by [`factorlasso`](https://github.com/ArturSepp/factorlasso) and re-exported here
for backward compatibility, so `optimalportfolios.LassoModel is factorlasso.LassoModel`
holds. Import them from either package.

::: factorlasso.CurrentFactorCovarData

::: factorlasso.DependenceMeasure

::: factorlasso.DistanceTransform

::: factorlasso.LassoModel

::: factorlasso.LassoModelType

::: factorlasso.RollingFactorCovarData

::: factorlasso.VarianceColumns

::: factorlasso.compute_dependence_matrix

::: factorlasso.compute_gerber_matrix

## Re-exported from qis

Owned by [`qis`](https://github.com/ArturSepp/QuantInvestStrats). This package used to
carry an independent reimplementation; the re-export keeps the import path working.

::: qis.models.linear.corr_cov_matrix.estimate_rolling_ewma_covar
