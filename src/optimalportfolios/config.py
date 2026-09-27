"""Portfolio objectives implemented by the rolling optimisation engine."""

from enum import Enum


class PortfolioObjective(Enum):
    """
    Objective selected by ``compute_rolling_optimal_weights``.

    The dispatcher in ``optimization/wrapper_rolling_portfolios.py`` routes each member to one
    rolling solver: ``EQUAL_RISK_CONTRIBUTION`` to ``rolling_risk_budgeting``,
    ``MAX_DIVERSIFICATION`` (its default) to ``rolling_maximise_diversification``,
    ``MIN_VARIANCE`` and ``QUADRATIC_UTILITY`` to ``rolling_quadratic_optimisation``,
    ``MAXIMUM_SHARPE_RATIO`` to ``rolling_maximize_portfolio_sharpe`` and ``MAX_CARA_MIXTURE``
    to ``rolling_maximize_cara_mixture``; any other value raises ``NotImplementedError``.
    Minimum tracking error and the SAA and TAA solvers have no member and are called directly.
    The value of ``MAX_CARA_MIXTURE`` keeps the spelling ``'MaxCarraMixture'``.
    """
    # risk-based:
    MAX_DIVERSIFICATION = 'MaxDiversification'  # maximum diversification measure
    EQUAL_RISK_CONTRIBUTION = 'EqualRisk'  # constrained risk budgeting in risk_allocation
    MIN_VARIANCE = 'MinVariance'  # min w^t @ covar @ w
    # return-risk based
    QUADRATIC_UTILITY = 'QuadraticUtil'  # max means^t*w- 0.5*gamma*w^t*covar*w
    MAXIMUM_SHARPE_RATIO = 'MaximumSharpe'  # max means^t*w / sqrt(w^t*covar*w)
    # return-skeweness based
    MAX_CARA_MIXTURE = 'MaxCarraMixture'  # carra for mixture distributions


