"""Matched CORRIDOR MIQP primary with audited V2 fallback on one prepared domain."""

from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
import numpy as np

from .domain import audit_execution, result
from .exact import wrapper_exact_execution
from .preparation import prepare_common_execution
from .production import V2ExecutionPolicyConfig, _production_result, resolve_production_problem
from .ranked import wrapper_ranked_execution


@dataclass(frozen=True, kw_only=True)
class MatchedExecutionPolicyConfig(V2ExecutionPolicyConfig):
    """Use matched MIQP first, retaining V2's domain and mandatory-repair semantics.

    The fallback is always V2 on the same prepared problem. A V1 escape would
    change that contract and is rejected. Native time-limited incumbents remain
    eligible only with a native audited witness and a consistent reported bound.
    """

    allow_v1_fallback: bool = False
    primary_time_limit_s: float = 10.
    primary_threads: int = 1
    primary_gap_rel: float = 1e-4

    @property
    def execution_method(self):
        """Keep matched MIQP distinct from the operational Exact implementation."""
        return 'matched_miqp_v2'

    def __post_init__(self):
        """Validate finite native budgets without changing feasibility tolerances."""
        super().__post_init__()
        if self.allow_v1_fallback:
            raise ValueError('Matched MIQP permits only audited V2 fallback')
        if (not np.isfinite(self.primary_time_limit_s) or self.primary_time_limit_s <= 0
                or not np.isfinite(self.primary_gap_rel) or self.primary_gap_rel < 0
                or isinstance(self.primary_threads, bool)
                or not isinstance(self.primary_threads, (int, np.integer))
                or self.primary_threads < 1):
            raise ValueError('Invalid matched MIQP budget or solver controls')


def wrapper_matched_execution(problem, config):
    """Solve one immutable prepared problem, auditing every possible delivery.

    Native failure, unavailability, callback errors, absent native witnesses,
    invalid weights and inconsistent bounds trigger V2 lazily. A solver status
    or the common incumbent alone cannot establish a successful primary solve.
    Neither solver may weaken the original domain or audit tolerances. The
    prepared problem owns all constraints; config supplies only search controls.
    """
    started = perf_counter()
    diagnostics = {}
    reason, solved = None, None
    tick = perf_counter()
    try:
        candidate = wrapper_exact_execution(
            problem, 'CORRIDOR', time_limit_s=config.primary_time_limit_s,
            threads=config.primary_threads, gap_rel=config.primary_gap_rel)
        audit = audit_execution(problem, candidate.weights)
        native = candidate.diagnostics
        witness = any(
            event.get('audit', {}).get('passed') and
            (event.get('source', '').startswith('native')
             or event.get('source') == 'boundary_refinement')
            for event in native.get('incumbent_events', []))
        diagnostics['primary'] = dict(
            status=candidate.status, diagnostics=native, audit=audit,
            seconds=perf_counter()-tick)
        if not audit['passed']:
            reason = 'primary_independent_audit_failed'
        elif native.get('native_status') in (
                'solver_error', 'solver_unavailable', 'not_run', 'preparation_deadline'):
            reason = 'native_unavailable_or_failed'
        elif native.get('callback_errors'):
            reason = 'native_callback_error'
        elif not witness:
            reason = 'no_audited_native_candidate'
        elif (native.get('native_bound_check') or {}).get('consistent_with_delivery') is False:
            reason = 'native_bound_inconsistent'
        else:
            solved = candidate
    except (ValueError, RuntimeError, cp.error.SolverError) as error:
        diagnostics['primary'] = dict(error=repr(error), seconds=perf_counter()-tick)
        reason = 'primary_exception'
    engine = 'matched_miqp'
    if solved is None:
        tick = perf_counter()
        solved = wrapper_ranked_execution(problem, config.ranked_config)
        audit = audit_execution(problem, solved.weights)
        diagnostics['v2'] = dict(status=solved.status, diagnostics=solved.diagnostics,
                                 audit=audit, seconds=perf_counter()-tick)
        if not audit['passed']:
            error = RuntimeError('V2 fallback rejected by unchanged common audit')
            error.diagnostics = dict(diagnostics, fallback_reason=reason)
            raise error
        engine = 'v2_audited_fallback'
    diagnostics.update(engine=engine, fallback_reason=reason,
                       total_search_seconds=perf_counter()-started)
    return result(problem, solved.weights, 'audited_incumbent', diagnostics)


def solve_matched_execution_portfolio(trade_table, base_constraints, covariance, config, *,
                                      context=''):
    """Resolve and prepare once, then apply matched primary and audited V2 fallback."""
    diagnostics = {}
    try:
        raw, constraints = resolve_production_problem(
            trade_table, base_constraints, covariance, config)
        problem, preparation = prepare_common_execution(
            raw, time_limit_s=config.preparation_time_limit_s, solver=config.preparation_solver)
        diagnostics['preparation'] = preparation
        if problem is None:
            raise RuntimeError('Matched MIQP common preparation failed')
        solved = wrapper_matched_execution(problem, config)
        diagnostics.update(solved.diagnostics)
        return _production_result(
            problem, solved.weights, trade_table, constraints, solved.diagnostics['engine'],
            diagnostics, context)
    except (ValueError, RuntimeError, cp.error.SolverError) as error:
        diagnostics.update(getattr(error, 'diagnostics', {}))
        error.trade_table = trade_table.copy()
        error.diagnostics = diagnostics
        raise
