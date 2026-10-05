"""Cash-funded ranked execution over consumer-resolved portfolio inputs."""
from optimalportfolios.execution.types import ExecutionRankingConfig, ResolvedExecutionProblem
from optimalportfolios.execution.ranking import score_execution_trades, resolve_effective_max_trades
from optimalportfolios.execution.feasibility import compute_group_bound_bridges
from optimalportfolios.execution.solver import (
    ExecutionOptimizationResult,
    ExecutionSolverInfeasibility,
    build_selected_execution_constraints,
    solve_selected_execution_portfolio,
    solve_feasible_execution_portfolio,
)
from optimalportfolios.execution.api import solve_ranked_execution
from optimalportfolios.execution.improvement import (
    ExecutionScoreMethod,
    ExecutionSearchConfig,
    ExecutionSearchResult,
    ExecutionCorridorDiagnostic,
    score_execution_candidates,
    improve_ranked_execution,
    diagnose_execution_corridors,
)
from optimalportfolios.execution.guarded import ExecutionGuardConfig, solve_guarded_execution

from optimalportfolios.execution.branches import (
    ExecutionBranchConfig, ExecutionBranchSearchResult, solve_branched_execution,
)
