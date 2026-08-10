"""General-purpose portfolio optimisation solvers.

Objective-driven solvers used as building blocks or for standalone portfolio
construction, including covariance tracking-error minimization.

Inputs retain their caller-supplied return and covariance units, while portfolio weights are
dimensionless. Boundary: this namespace re-exports solvers but owns no estimation or reporting.
"""
from optimalportfolios.optimization.general.quadratic import (
    rolling_quadratic_optimisation,
    wrapper_quadratic_optimisation,
    cvx_quadratic_optimisation,
    solve_analytic_log_opt,
)

from optimalportfolios.optimization.general.max_sharpe import (
    rolling_maximize_portfolio_sharpe,
    wrapper_maximize_portfolio_sharpe,
    cvx_maximize_portfolio_sharpe,
)

from optimalportfolios.optimization.general.max_diversification import (
    rolling_maximise_diversification,
    wrapper_maximise_diversification,
    opt_maximise_diversification,
)

from optimalportfolios.optimization.general.carra_mixture import (
    rolling_maximize_cara_mixture,
    wrapper_maximize_cara_mixture,
    opt_maximize_cara_mixture,
    opt_maximize_cara,
)

from optimalportfolios.optimization.general.risk_budgeting import (
    rolling_risk_budgeting,
    wrapper_risk_budgeting,
    opt_risk_budgeting,
    solve_for_risk_budgets_from_given_weights,
)

from optimalportfolios.optimization.general.minimum_tracking_error import (
    rolling_minimise_tracking_error,
    wrapper_minimise_tracking_error,
    cvx_minimise_tracking_error,
)


# The export surface of this subpackage. `from ... import *` — including the star
# imports in the top-level __init__ — re-exports exactly this list.
__all__ = [
    'cvx_maximize_portfolio_sharpe',
    'cvx_minimise_tracking_error',
    'cvx_quadratic_optimisation',
    'opt_maximise_diversification',
    'opt_maximize_cara',
    'opt_maximize_cara_mixture',
    'opt_risk_budgeting',
    'rolling_maximise_diversification',
    'rolling_maximize_cara_mixture',
    'rolling_maximize_portfolio_sharpe',
    'rolling_minimise_tracking_error',
    'rolling_quadratic_optimisation',
    'rolling_risk_budgeting',
    'solve_analytic_log_opt',
    'solve_for_risk_budgets_from_given_weights',
    'wrapper_maximise_diversification',
    'wrapper_maximize_cara_mixture',
    'wrapper_maximize_portfolio_sharpe',
    'wrapper_minimise_tracking_error',
    'wrapper_quadratic_optimisation',
    'wrapper_risk_budgeting',
]
