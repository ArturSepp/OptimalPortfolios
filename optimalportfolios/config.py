from enum import Enum


class PortfolioObjective(Enum):
    """
    implemented portfolios in rolling_engine
    """
    # risk-based:
    MAX_DIVERSIFICATION = 'MaxDiversification'  # maximum diversification measure
    EQUAL_RISK_CONTRIBUTION = 'EqualRisk'  # implementation in risk_parity
    MIN_VARIANCE = 'MinVariance'  # min w^t @ covar @ w
    # return-risk based
    QUADRATIC_UTILITY = 'QuadraticUtil'  # max means^t*w- 0.5*gamma*w^t*covar*w
    MAXIMUM_SHARPE_RATIO = 'MaximumSharpe'  # max means^t*w / sqrt(*w^t*covar*w)
    # return-skeweness based
    MAX_CARA_MIXTURE = 'MaxCarraMixture'  # carra for mixture distributions


