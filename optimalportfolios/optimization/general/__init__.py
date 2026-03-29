"""General-purpose portfolio optimisation solvers.

Objective-driven solvers with no benchmark or active overlay semantics.
Used as building blocks or for standalone portfolio construction.
"""
from optimalportfolios.optimization.general.quadratic import (
    rolling_quadratic_optimisation,
    wrapper_quadratic_optimisation,
    cvx_quadratic_optimisation,
    max_qp_portfolio_vol_target,
    solve_analytic_log_opt,
    print_portfolio_outputs,
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