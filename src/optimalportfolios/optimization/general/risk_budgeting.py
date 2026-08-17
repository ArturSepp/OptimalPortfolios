"""Compatibility imports for risk budgeting's canonical risk-allocation home.

Risk-budgeting implementations moved to
``optimalportfolios.optimization.risk_allocation.risk_budgeting`` in 6.21.0.
This module preserves existing direct imports from ``optimization.general``.
"""

from optimalportfolios.optimization.risk_allocation.risk_budgeting import (
    opt_risk_budgeting as opt_risk_budgeting,
    opt_risk_budgeting_scipy as opt_risk_budgeting_scipy,
    risk_budget_objective as risk_budget_objective,
    rolling_risk_budgeting as rolling_risk_budgeting,
    solve_for_risk_budgets_from_given_weights,  # noqa: F401 -- compatibility export
    wrapper_risk_budgeting as wrapper_risk_budgeting,
)
