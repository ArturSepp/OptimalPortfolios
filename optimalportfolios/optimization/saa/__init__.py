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