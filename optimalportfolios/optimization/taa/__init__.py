"""Tactical asset allocation (TAA) solvers.

Solvers that take alpha signals, TE constraints, and benchmark-relative
objectives to produce active tilts over the SAA anchor.
"""
from optimalportfolios.optimization.taa.maximise_alpha_over_tre import (
    rolling_maximise_alpha_over_tre,
    wrapper_maximise_alpha_over_tre,
    cvx_maximise_alpha_over_tre,
    cvx_maximise_tre_utility,
)

from optimalportfolios.optimization.taa.maximise_alpha_with_target_yield import (
    rolling_maximise_alpha_with_target_return,
    wrapper_maximise_alpha_with_target_return,
    cvx_maximise_alpha_with_target_return,
)