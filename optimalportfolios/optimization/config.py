from enum import Enum


class PortfolioObjective(Enum):
    MIN_VAR = 1  # min w^t @ covar @ w
    QUADRATIC_UTILITY = 2  # max means^t*w- 0.5*gamma*w^t*covar*w
    EQUAL_RISK_CONTRIBUTION = 3  # implementation in risk_parity
    RISK_PARITY_ALT = 4  # alternative implementation of risk_parity
    MAX_DIVERSIFICATION = 5
    MAXIMUM_SHARPE_RATIO = 6
    CARA_MIXURE = 7