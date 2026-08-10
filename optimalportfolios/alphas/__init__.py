"""Public API of the alpha layer: signals, profiling and alpha backtests."""

from optimalportfolios.alphas.alpha_data import AlphasData
from optimalportfolios.alphas.profile import (
    backtest_alpha_rank_portfolio,
    compute_top_quantile_equal_weights,
    compute_alpha_rank_analysis_table,
    generate_alpha_profile_report,
    ProfileSignal,
    profile_momentum,
    profile_low_beta,
    profile_residual_momentum,
    profile_carry,
    profile_alpha_signals,
)
from optimalportfolios.alphas.signals.momentum import (
    compute_momentum_alpha,
    compute_momentum_cluster_alpha,
)
from optimalportfolios.alphas.signals.low_beta import (
    compute_low_beta_alpha,
    compute_low_beta_cluster_alpha,
)
from optimalportfolios.alphas.signals.carry import compute_ra_carry_alphas
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
    align_rolling_clusters,
    extract_rolling_clusters,
    score_within_clusters,
)

from optimalportfolios.alphas.signal_diagnostics import (
    signal_diagnostics_panel,
    run_signal_diagnostics,
    run_signal_diagnostics_per_component,
    compare_signal_diagnostics,
)


# The export surface of this subpackage. `from ... import *` — including the star
# imports in the top-level __init__ — re-exports exactly this list.
__all__ = [
    'AlphasData',
    'ProfileSignal',
    'align_rolling_clusters',
    'backtest_alpha_rank_portfolio',
    'compare_signal_diagnostics',
    'compute_alpha_rank_analysis_table',
    'compute_low_beta_alpha',
    'compute_low_beta_cluster_alpha',
    'compute_managers_alpha',
    'compute_momentum_alpha',
    'compute_momentum_cluster_alpha',
    'compute_ra_carry_alphas',
    'compute_residual_momentum_alpha',
    'compute_residual_momentum_cluster_alpha',
    'compute_residual_reversal_alpha',
    'compute_residual_reversal_cluster_alpha',
    'compute_top_quantile_equal_weights',
    'estimate_rolling_ewma_means',
    'extract_rolling_clusters',
    'generate_alpha_profile_report',
    'profile_alpha_signals',
    'profile_carry',
    'profile_low_beta',
    'profile_momentum',
    'profile_residual_momentum',
    'run_signal_diagnostics',
    'run_signal_diagnostics_per_component',
    'score_within_clusters',
    'signal_diagnostics_panel',
]
