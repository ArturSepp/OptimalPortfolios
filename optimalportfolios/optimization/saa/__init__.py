"""Strategic asset allocation (SAA) solvers.

Solvers that take CMA inputs, return floors, and volatility budgets to
produce the strategic anchor allocation.
"""
from optimalportfolios.optimization.saa.min_variance_target_return import (
    rolling_min_variance_target_return,
    wrapper_min_variance_target_return,
    cvx_min_variance_target_return,
    cvx_min_variance_target_return_utility,
)

from optimalportfolios.optimization.saa.max_return_target_vol import (
    rolling_max_return_target_vol,
    wrapper_max_return_target_vol,
    cvx_max_return_target_vol,
    cvx_max_return_target_vol_utility,
)


# The export surface of this subpackage. `from ... import *` — including the star
# imports in the top-level __init__ — re-exports exactly this list.
__all__ = [
    'cvx_max_return_target_vol',
    'cvx_max_return_target_vol_utility',
    'cvx_min_variance_target_return',
    'cvx_min_variance_target_return_utility',
    'rolling_max_return_target_vol',
    'rolling_min_variance_target_return',
    'wrapper_max_return_target_vol',
    'wrapper_min_variance_target_return',
]
