"""Prepared-input V2 execution with an explicit production policy adapter."""

from optimalportfolios.execution.v2.types import (
    ExecutionProblem,
    ExecutionRules,
    RankedConfig,
    ExecutionResult,
)
from optimalportfolios.execution.v2.domain import audit_execution, compile_states
from optimalportfolios.execution.v2.ranked import wrapper_ranked_execution
from optimalportfolios.execution.v2.exact import wrapper_exact_execution
from optimalportfolios.execution.v2.bounds import compute_perspective_lower_bound
from optimalportfolios.execution.v2.adapter import from_resolved_problem
from optimalportfolios.execution.v2.preparation import prepare_common_execution
from optimalportfolios.execution.v2.production import (
    V2ExecutionPolicyConfig,
    resolve_production_problem,
    solve_v2_execution_portfolio,
)
from optimalportfolios.execution.v2.primary import (
    MatchedExecutionPolicyConfig,
    wrapper_matched_execution,
    solve_matched_execution_portfolio,
)
