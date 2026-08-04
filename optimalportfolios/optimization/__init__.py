"""Portfolio optimisation module.

Re-exports public symbols from constraints, config, covariance factorization,
structured solver outcomes, solver submodules (general, saa, taa), the rolling
portfolio dispatcher, and portfolio result.

Compatible CVXPY solvers factor covariance once per solve by default through
``OptimiserConfig.factorize_covar``. Their single-date wrappers return weights
and an ``OptimizationOutcome`` with report-ready ``ConstraintResidual`` records.

Submodule structure:
    general/    — objective-driven solvers (min-var, max Sharpe, max div, CARA, risk budgeting)
    saa/        — strategic solvers with CMA inputs, return/vol targets
    taa/        — tactical solvers with alpha signals, TE constraints, benchmarks
"""
# config
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.covar_factorization import (
    CovarianceFactorization as CovarianceFactorization,
    factorize_covariance as factorize_covariance,
)
from optimalportfolios.optimization.solver_diagnostics import (
    ConstraintResidual,
    OptimizationOutcome,
    evaluate_constraint_residuals,
)

# constraints
from optimalportfolios.optimization.constraints import (
    ConstraintEnforcementType,
    Constraints,
    GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
    merge_group_lower_upper_constraints,
)

# general solvers
from optimalportfolios.optimization.general import (
    rolling_quadratic_optimisation,
    wrapper_quadratic_optimisation,
    cvx_quadratic_optimisation,
    solve_analytic_log_opt,
    rolling_maximize_portfolio_sharpe,
    wrapper_maximize_portfolio_sharpe,
    cvx_maximize_portfolio_sharpe,
    rolling_maximise_diversification,
    wrapper_maximise_diversification,
    opt_maximise_diversification,
    rolling_maximize_cara_mixture,
    wrapper_maximize_cara_mixture,
    opt_maximize_cara_mixture,
    opt_maximize_cara,
    rolling_risk_budgeting,
    wrapper_risk_budgeting,
    opt_risk_budgeting,
    solve_for_risk_budgets_from_given_weights,
)

# saa solvers
from optimalportfolios.optimization.saa import (
    rolling_min_variance_target_return,
    wrapper_min_variance_target_return,
    cvx_min_variance_target_return,
    cvx_min_variance_target_return_utility,
    rolling_max_return_target_vol,
    wrapper_max_return_target_vol,
    cvx_max_return_target_vol,
    cvx_max_return_target_vol_utility,
)

# taa solvers
from optimalportfolios.optimization.taa import (
    rolling_maximise_alpha_over_tre,
    wrapper_maximise_alpha_over_tre,
    cvx_maximise_alpha_over_tre,
    cvx_maximise_tre_utility,
    rolling_maximise_alpha_with_target_return,
    wrapper_maximise_alpha_with_target_return,
    cvx_maximise_alpha_with_target_return,
)

# dispatcher
from optimalportfolios.optimization.wrapper_rolling_portfolios import (
    compute_rolling_optimal_weights,
    backtest_rolling_optimal_portfolio,
)

# result container
from optimalportfolios.optimization.portfolio_result import PortfolioOptimisationResult
