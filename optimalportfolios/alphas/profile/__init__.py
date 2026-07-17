"""
alpha profiling submodule.

core is the single backtester/profiler for alphas (rank-based, top-quantile long-only). The
per-signal wrappers in signal_profilers compute a named signal's panel and call the core.
"""
from optimalportfolios.alphas.profile.core import (
    backtest_alpha_rank_portfolio,
    compute_top_quantile_equal_weights,
    compute_alpha_rank_analysis_table,
    generate_alpha_profile_report,
)
from optimalportfolios.alphas.profile.signal_profilers import (
    ProfileSignal,
    profile_momentum,
    profile_low_beta,
    profile_residual_momentum,
    profile_carry,
    profile_alpha_signals,
)