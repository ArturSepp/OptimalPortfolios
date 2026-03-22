from typing import NamedTuple


class OptimiserConfig(NamedTuple):
    """Configuration for alpha over tracking error optimization."""
    solver: str = 'CLARABEL'  # CVXPY solver choice
    # this specify weights of utility function for ConstraintEnforcementType.UTILITY_CONSTRAINTS
    apply_total_to_good_ratio: bool = False  # adjust constraints for non-investable assets
