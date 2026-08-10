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


# The export surface of this subpackage. `from ... import *` — including the star
# imports in the top-level __init__ — re-exports exactly this list.
__all__ = [
    'cvx_maximise_alpha_over_tre',
    'cvx_maximise_alpha_with_target_return',
    'cvx_maximise_tre_utility',
    'rolling_maximise_alpha_over_tre',
    'rolling_maximise_alpha_with_target_return',
    'wrapper_maximise_alpha_over_tre',
    'wrapper_maximise_alpha_with_target_return',
]
