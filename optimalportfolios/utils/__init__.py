"""Public API of the shared utilities: NaN filtering, portfolio statistics, weight
rounding and drift.
"""

from optimalportfolios.utils.filter_nans import (filter_covar_and_vectors,
                                                 filter_covar_and_vectors_for_nans)

from optimalportfolios.utils.portfolio_funcs import (compute_portfolio_vol,
                                                     compute_tre_turnover_stats)

from optimalportfolios.utils.portfolio_funcs import (compute_portfolio_variance,
                                                     calculate_diversification_ratio,
                                                     compute_portfolio_risk_contribution_outputs,
                                                     round_weights_to_pct,
                                                     compute_risk_contributions)

from optimalportfolios.utils.gaussian_mixture import fit_gaussian_mixture

from optimalportfolios.utils.weights_drift import apply_drift_to_weights_0


# The export surface of this subpackage. `from ... import *` — including the star
# imports in the top-level __init__ — re-exports exactly this list.
__all__ = [
    'apply_drift_to_weights_0',
    'calculate_diversification_ratio',
    'compute_portfolio_risk_contribution_outputs',
    'compute_portfolio_variance',
    'compute_portfolio_vol',
    'compute_risk_contributions',
    'compute_tre_turnover_stats',
    'filter_covar_and_vectors',
    'filter_covar_and_vectors_for_nans',
    'fit_gaussian_mixture',
    'round_weights_to_pct',
]
