"""Same-date target retry under unchanged product constraints."""
from dataclasses import replace
from typing import Optional
import pandas as pd
from optimalportfolios.execution.v1.solver import ExecutionOptimizationResult
from optimalportfolios.execution.v1.policy import CUTOFF_RESIDUAL_BRIDGE

GROUP_MECHANISM_ABANDONED = 'group_mechanism_abandoned'
GROUP_MECHANISM_RESIDUAL_BRIDGE = 'group_mechanism_residual_bridge'
GROUP_MECHANISM_RESIDUAL_BRIDGE_BP = 'group_mechanism_residual_bridge_bp'
GROUP_MECHANISM_FAILURE_DIAGNOSTIC = 'group_mechanism_failure_diagnostic'
GROUP_MECHANISM_FALLBACK_DIAGNOSTIC = 'group_mechanism_fallback_diagnostic'


class ExecutionGroupFallbackFailure(RuntimeError):
    """Both the group-aware and same-date mechanism-disabled solve failed."""

    def __init__(
            self,
            enabled_diagnostic: str,
            disabled_diagnostic: str,
            trade_table: Optional[pd.DataFrame] = None,
            bridges: Optional[pd.DataFrame] = None,
    ) -> None:
        """Retain both failed attempts and their diagnostics."""
        super().__init__(
            'group-aware execution and same-date mechanism-disabled fallback '
            f'failed; enabled={enabled_diagnostic}; disabled={disabled_diagnostic}'
        )
        self.enabled_diagnostic = enabled_diagnostic
        self.disabled_diagnostic = disabled_diagnostic
        self.trade_table = trade_table
        self.bridges = pd.DataFrame() if bridges is None else bridges.copy()


def solve_with_group_retry(build_table, solve_table) -> ExecutionOptimizationResult:
    """Retry target preparation once, preserving the original hard constraints."""
    def attempt(
            group_mechanism_enabled: bool,
    ) -> tuple[
        Optional[ExecutionOptimizationResult], str,
        Optional[pd.DataFrame], pd.DataFrame,
    ]:
        """Return one audited target/solve attempt and any failure evidence."""
        table: Optional[pd.DataFrame] = None
        try:
            table = build_table(group_mechanism_enabled)
            solved = solve_table(table)
            if solved.accepted and solved.compliant:
                return solved, '', solved.trade_table, pd.DataFrame()
            diagnostic = (
                f'status={solved.outcome.status}, '
                f'reason={solved.outcome.reason}'
            )
            return solved, diagnostic, solved.trade_table, pd.DataFrame()
        except (ValueError, RuntimeError) as error:
            audited = getattr(error, 'trade_table', None)
            if audited is None:
                audited = table
            return (
                None,
                f'{type(error).__name__}: {error}',
                audited,
                getattr(error, 'bridges', pd.DataFrame()),
            )

    enabled, enabled_diagnostic, enabled_table, enabled_bridges = attempt(True)
    if enabled is not None and enabled.accepted and enabled.compliant:
        audited = enabled.trade_table.copy()
        audited[GROUP_MECHANISM_ABANDONED] = False
        audited[GROUP_MECHANISM_RESIDUAL_BRIDGE] = 0.0
        audited[GROUP_MECHANISM_RESIDUAL_BRIDGE_BP] = 0.0
        audited[GROUP_MECHANISM_FAILURE_DIAGNOSTIC] = ''
        audited[GROUP_MECHANISM_FALLBACK_DIAGNOSTIC] = ''
        return replace(enabled, trade_table=audited)

    fallback, fallback_diagnostic, fallback_table, fallback_bridges = attempt(False)
    bridge_cost = 0.0
    if not enabled_bridges.empty:
        group_bridges = enabled_bridges.drop(index='cash', errors='ignore')
        if not group_bridges.empty:
            bridge_cost = float(
                group_bridges[['bridge_min', 'bridge_max']].max(axis=1).max()
            )
    if bridge_cost == 0.0 and enabled_table is not None:
        if CUTOFF_RESIDUAL_BRIDGE in enabled_table:
            bridge_cost = float(enabled_table[CUTOFF_RESIDUAL_BRIDGE].max())
    if fallback is not None and fallback.accepted and fallback.compliant:
        audited = fallback.trade_table.copy()
        audited[GROUP_MECHANISM_ABANDONED] = True
        audited[GROUP_MECHANISM_RESIDUAL_BRIDGE] = bridge_cost
        audited[GROUP_MECHANISM_RESIDUAL_BRIDGE_BP] = 10_000.0 * bridge_cost
        audited[GROUP_MECHANISM_FAILURE_DIAGNOSTIC] = enabled_diagnostic
        audited[GROUP_MECHANISM_FALLBACK_DIAGNOSTIC] = ''
        return replace(fallback, trade_table=audited)

    raise ExecutionGroupFallbackFailure(
        enabled_diagnostic=enabled_diagnostic,
        disabled_diagnostic=fallback_diagnostic,
        trade_table=fallback_table if fallback_table is not None else enabled_table,
        bridges=(
            fallback_bridges if not fallback_bridges.empty else enabled_bridges
        ),
    )
