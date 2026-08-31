"""Public portfolio-constraint API and solver-independent data models.

The submodules separate immutable specifications, universe alignment, solver
compilation, and analytical evaluation.  This facade preserves the historical
``optimalportfolios.optimization.constraints`` import surface.
"""
import logging

from optimalportfolios.optimization.constraints.alignment import (
    RelaxationRecord,
    compute_eligible_rebalancing_bounds,
)
from optimalportfolios.optimization.constraints.analytics import (
    ConstraintResidual,
    evaluate_constraint_residuals,
)
from optimalportfolios.optimization.constraints.backends import (
    long_only_constraint,
    make_max_constraint,
    make_min_constraint,
    total_weight_constraint,
)
from optimalportfolios.optimization.constraints.benchmarks import (
    BenchmarkBetaConstraint,
    BenchmarkDeviationConstraints,
)
from optimalportfolios.optimization.constraints.core import (
    ConstraintEnforcementType,
    Constraints,
)
from optimalportfolios.optimization.constraints.expressions import (
    add_term_to_objective_function,
    cvx_covar_variance,
)
from optimalportfolios.optimization.constraints.groups import (
    DroppedGroupRecord,
    GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
    merge_group_lower_upper_constraints,
)
from optimalportfolios.utils.benchmark_beta import (
    compute_benchmark_beta_loadings,
    compute_benchmark_beta_loadings_from_covar,
)


logger = logging.getLogger(__name__)


__all__ = [
    "BenchmarkBetaConstraint",
    "BenchmarkDeviationConstraints",
    "ConstraintEnforcementType",
    "ConstraintResidual",
    "Constraints",
    "DroppedGroupRecord",
    "GroupLowerUpperConstraints",
    "GroupTrackingErrorConstraint",
    "GroupTurnoverConstraint",
    "RelaxationRecord",
    "add_term_to_objective_function",
    "compute_benchmark_beta_loadings",
    "compute_benchmark_beta_loadings_from_covar",
    "compute_eligible_rebalancing_bounds",
    "cvx_covar_variance",
    "evaluate_constraint_residuals",
    "long_only_constraint",
    "make_max_constraint",
    "make_min_constraint",
    "merge_group_lower_upper_constraints",
    "total_weight_constraint",
]
