"""Public API of the alpha signal layer, one constructor pair per signal."""

from optimalportfolios.alphas.alpha_data import AlphasData
from optimalportfolios.alphas.signals.momentum import (
    compute_momentum_alpha,
    compute_momentum_cluster_alpha,
)
from optimalportfolios.alphas.signals.low_beta import (
    compute_low_beta_alpha,
    compute_low_beta_cluster_alpha,
)
from optimalportfolios.alphas.signals.carry import (
    compute_ra_carry_alpha,
    compute_ra_carry_cluster_alpha,
    compute_ra_carry_alphas,
)
from optimalportfolios.alphas.signals.managers_alpha import compute_managers_alpha
from optimalportfolios.alphas.signals.residual_momentum import (
    compute_residual_momentum_alpha,
    compute_residual_momentum_cluster_alpha,
)
from optimalportfolios.alphas.signals.residual_reversal import (
    compute_residual_reversal_alpha,
    compute_residual_reversal_cluster_alpha,
)
from optimalportfolios.alphas.signals.rolling_ewma_mean import estimate_rolling_ewma_means
from optimalportfolios.alphas.signals.utils import (
    extract_rolling_clusters,
    score_within_clusters,
)


# The export surface of this subpackage. `from ... import *` — including the star
# imports in the top-level __init__ — re-exports exactly this list.
__all__ = [
    'AlphasData',
    'compute_low_beta_alpha',
    'compute_low_beta_cluster_alpha',
    'compute_managers_alpha',
    'compute_momentum_alpha',
    'compute_momentum_cluster_alpha',
    'compute_ra_carry_alpha',
    'compute_ra_carry_alphas',
    'compute_ra_carry_cluster_alpha',
    'compute_residual_momentum_alpha',
    'compute_residual_momentum_cluster_alpha',
    'compute_residual_reversal_alpha',
    'compute_residual_reversal_cluster_alpha',
    'estimate_rolling_ewma_means',
    'extract_rolling_clusters',
    'score_within_clusters',
]
