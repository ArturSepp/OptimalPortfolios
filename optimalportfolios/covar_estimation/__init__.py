"""Public API of the covariance layer: the EWMA and factor (HCGL) estimators and
their reporting helpers.
"""

from optimalportfolios.covar_estimation.factor_covar_estimator import (FactorCovarEstimator,
                                                                       estimate_lasso_factor_covar_data)

from optimalportfolios.covar_estimation.ewma_covar_estimator import (EwmaCovarEstimator,
                                                                     estimate_current_ewma_covar,
                                                                     estimate_rolling_ewma_covar)

from optimalportfolios.covar_estimation.covar_reporting import (plot_current_covar_data,
                                                                plot_hcgl_covar_data,
                                                                run_rolling_covar_report)

from optimalportfolios.covar_estimation.utils import compute_returns_from_prices

from optimalportfolios.covar_estimation.risk_model_adapter import (
    build_risk_model as build_risk_model,
)


# The export surface of this subpackage. `from ... import *` — including the star
# imports in the top-level __init__ — re-exports exactly this list.
__all__ = [
    'EwmaCovarEstimator',
    'FactorCovarEstimator',
    'build_risk_model',
    'compute_returns_from_prices',
    'estimate_current_ewma_covar',
    'estimate_lasso_factor_covar_data',
    'estimate_rolling_ewma_covar',
    'plot_current_covar_data',
    'plot_hcgl_covar_data',
    'run_rolling_covar_report',
]
