"""OP's retained EWMA covariance convenience functions.

Factor estimation, diagnostics and covariance reporting live in TradableFactors.
"""

from optimalportfolios.covar_estimation.ewma_covar_estimator import (EwmaCovarEstimator,
                                                                     estimate_current_ewma_covar,
                                                                     estimate_rolling_ewma_covar)

from optimalportfolios.covar_estimation.utils import compute_returns_from_prices
