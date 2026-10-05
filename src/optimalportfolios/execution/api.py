"""End-to-end numerical execution from consumer-resolved inputs."""
from dataclasses import replace

from optimalportfolios.execution.ranking import score_execution_trades
from optimalportfolios.execution.solver import (
    ExecutionOptimizationResult,
    solve_feasible_execution_portfolio,
)
from optimalportfolios.execution.types import ResolvedExecutionProblem


def solve_ranked_execution(problem: ResolvedExecutionProblem) -> ExecutionOptimizationResult:
    """Rank, project and audit one resolved legacy execution decision.

    Args:
        problem: Copied numerical inputs and effective policy settings.

    Returns:
        Weights, enriched trade table and OP solver outcome. Check both accepted
        and compliant before using the weights. Structural corridor failures raise
        ExecutionSolverInfeasibility with the last trade table and bridge diagnostics.

    This seam does not resolve lifecycle, desk instructions, cash bootstrap or
    retry target construction with a different group mechanism. Consumers retain
    that orchestration until separate end-to-end parity is established.
    """
    snapshot = replace(problem)
    scored = score_execution_trades(
        target=snapshot.target,
        alphas=snapshot.alphas,
        covariance=snapshot.covariance,
        asset_classes=snapshot.asset_classes,
        config=snapshot.ranking_config,
        asset_class_tre_weights=snapshot.asset_class_tre_weights,
    )
    return solve_feasible_execution_portfolio(
        trade_table=scored,
        base_constraints=snapshot.constraints,
        covariance=snapshot.covariance,
        optimiser_config=snapshot.optimiser_config,
        context=snapshot.context,
        expand_for_feasibility=snapshot.expand_for_feasibility,
        sign_directed_rescue=snapshot.sign_directed_rescue,
        allow_corridor_relaxation=snapshot.allow_corridor_relaxation,
        partition_groups=snapshot.partition_groups,
    )
