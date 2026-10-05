"""Independent branch, original-anchor, checkpoint and projection-budget contracts."""
from types import SimpleNamespace

import pandas as pd
import pytest

from optimalportfolios.execution import branches as b, guarded as g, schema as s, solver
from optimalportfolios.execution._budget import (
    _ProjectionBudget, _ProjectionBudgetExhausted, _consume_projection, _projection_scope,
)
from optimalportfolios.execution.tests.improvement_test import _problem


def _values(te=10., tickets=3, sized=2, turnover=100.):
    """Supply scalar references independent of the branch qualification implementation."""
    return dict(te_bp=te, tickets=tickets, sized_tickets=sized, gross_turnover_bp=turnover)


def _scripted(monkeypatch, values, replies=None, seed_cost=1):
    """Replace projections with counted, pre-audited outcomes and independent metrics."""
    problem = _problem(max_trades=3)
    baseline = b._solve(problem, b.score_execution_candidates(problem), rescue=True)
    original = b._solve
    metrics, response = iter(values), iter(replies) if replies is not None else None
    calls = []

    def solve(p, table, rescue):
        """Consume baseline-independent attempts, including a simulated rescue sequence."""
        calls.append((rescue, table[s.SELECTED_TRADE].copy()))
        if len(calls) == 1:
            return baseline
        for _ in range(seed_cost if rescue else 1):
            _consume_projection()
        value = next(response) if response is not None else baseline
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(b, '_solve', solve)
    monkeypatch.setattr(b, '_metrics', lambda *args: next(metrics))
    return problem, baseline, calls, original


def test_unqualified_seed_can_compress_to_best_eligible_return(monkeypatch):
    """A size/turnover-ineligible seed remains exploratory until its compressed state passes."""
    p, baseline, _, _ = _scripted(monkeypatch, [
        _values(sized=1), _values(te=8, sized=3, turnover=120),
        _values(te=10.5, tickets=2, sized=1, turnover=95),
        _values(te=8.5, tickets=1, sized=1, turnover=90)])
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(
        methods=('partial_risk',), max_projection_calls=3))
    trace = result.attempts
    assert trace.branch.tolist() == ['partial_risk', 'legacy', 'partial_risk']
    assert trace.branch_retained.tolist() == [True, True, True]
    assert trace.return_retained.tolist() == [False, True, True]
    assert trace.parent_checkpoint.tolist() == ['baseline', 'baseline', 'checkpoint_00001']
    assert result.summary['selected_checkpoint'] == 'checkpoint_00003'
    assert result.summary['final'] == _values(te=8.5, tickets=1, sized=1, turnover=90)
    assert result.checkpoints['baseline'] is baseline and len(result.checkpoints) == 4


def test_each_branch_uses_original_risk_allowance(monkeypatch):
    """A 0.8 bp seed does not permit a further 0.8 bp loss during deletion."""
    p, _, _, _ = _scripted(monkeypatch, [_values(), _values(te=10.8, tickets=2),
        _values(te=11.6, tickets=1), _values(te=11.6, tickets=1)])
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(
        methods=('partial_risk',), max_projection_calls=3))
    assert result.attempts.branch_retained.tolist() == [True, False, False]
    assert result.summary['final']['te_bp'] == 10.8
    assert result.summary['baseline']['te_bp'] == 10.


def test_shared_budget_includes_proposal_rescues(monkeypatch):
    """A rescue sequence cannot exceed the ceiling or consume a separate branch allowance."""
    p, baseline, _, _ = _scripted(monkeypatch, [_values()], seed_cost=3)
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(max_projection_calls=2))
    assert result.result is baseline
    assert result.summary['search_projection_calls'] == 2
    assert result.summary['stop_reason'] == 'search_projection_budget'
    assert result.attempts.iloc[0]['status'] == 'projection_budget'
    assert len(result.checkpoints) == 1


def test_nonqualifying_seed_is_never_returned_on_budget_stop(monkeypatch):
    """Returning early keeps the original incumbent, not the last exploratory state."""
    p, baseline, _, _ = _scripted(monkeypatch, [_values(), _values(te=5, turnover=200)])
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(max_projection_calls=1))
    assert result.result is baseline and result.summary['selected_checkpoint'] == 'baseline'
    assert len(result.checkpoints) == 2
    assert result.attempts.iloc[0].return_reason == 'more_turnover'


def test_return_preference_compares_all_branches(monkeypatch):
    """A later qualifying branch does not replace an already better eligible portfolio."""
    p, _, _, _ = _scripted(monkeypatch, [_values(), _values(te=8), _values(te=9)])
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(max_removal_trials=0))
    assert result.attempts.return_retained.tolist() == [True, False]
    assert result.attempts.iloc[-1].return_reason == 'incumbent_preferred'
    assert result.summary['selected_method'] == 'partial_risk'


@pytest.mark.parametrize('reply,reason', [
    (b.ExecutionSolverInfeasibility('conflict'), 'conflict'),
    (SimpleNamespace(accepted=False, compliant=False, outcome=SimpleNamespace(status='rejected')),
     'candidate_not_accepted'),
    (SimpleNamespace(accepted=True, compliant=True, outcome=SimpleNamespace(status='optimal'),
     trade_table=pd.DataFrame({b.RELAXED_CORRIDOR_SOLVE: [True]})), 'candidate_relaxed_corridors'),
])
def test_failed_seed_does_not_create_branch(monkeypatch, reply, reason):
    """Infeasible, rejected and relaxed seeds cannot become exploratory strict branches."""
    p, baseline, _, _ = _scripted(monkeypatch, [_values()], [reply])
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(
        methods=('partial_risk',), max_removal_trials=0))
    assert result.result is baseline and result.summary['branches'] == {'legacy': 'baseline'}
    assert result.attempts.iloc[0].branch_reason == reason


@pytest.mark.parametrize('accepted,relaxed,reason', [
    (False, False, 'baseline_not_accepted'), (True, True, 'baseline_relaxed_corridors')])
def test_unusable_baseline_stops_before_branch_search(monkeypatch, accepted, relaxed, reason):
    """Baseline rejection or explicit relaxation is not relabeled as a strict result."""
    baseline = SimpleNamespace(accepted=accepted, compliant=accepted,
        trade_table=pd.DataFrame({b.RELAXED_CORRIDOR_SOLVE: [relaxed]}))
    monkeypatch.setattr(b, '_solve', lambda *args, **kwargs: baseline)
    result = b.solve_branched_execution(_problem())
    assert result.result is baseline and result.summary['stop_reason'] == reason
    assert result.attempts.empty and result.checkpoints == {'baseline': baseline}


@pytest.mark.parametrize('limit', [None, True, -1, 1.5])
def test_branch_projection_ceiling_must_be_finite_integer(limit):
    """Each branch policy requires an explicit nonnegative work ceiling."""
    with pytest.raises(ValueError):
        b.ExecutionBranchConfig(max_projection_calls=limit)


def test_guard_config_without_ceiling_is_not_branch_config():
    """An accidental legacy control object cannot remove the branch work ceiling."""
    with pytest.raises(ValueError, match='requires max_projection_calls'):
        b.solve_branched_execution(_problem(), g.ExecutionGuardConfig())


@pytest.mark.parametrize('controls,reason', [
    ({'max_projection_calls': 0}, 'search_projection_budget'),
    ({'max_search_seconds': 0}, 'search_time_budget'),
])
def test_zero_budget_returns_exact_baseline(controls, reason):
    """Neither seed construction nor compression runs after an exhausted search budget."""
    p = _problem()
    baseline = b._solve(p, b.score_execution_candidates(p), rescue=True)
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(**controls))
    pd.testing.assert_series_equal(result.result.weights, baseline.weights)
    assert result.attempts.empty and result.summary['stop_reason'] == reason


def test_real_projection_count_and_all_saved_checkpoints(monkeypatch):
    """Independent instrumentation confirms the shared ceiling and hard-audited checkpoints."""
    p = _problem(max_trades=3)
    original = solver._solve_selected_execution_portfolio_once
    calls = []

    def counted(*args, **kwargs):
        """Count only projections actually admitted to the existing implementation."""
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, '_solve_selected_execution_portfolio_once', counted)
    before = p.target.copy(deep=True)
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(max_projection_calls=5,
        canonical_ties=False, max_sized_ticket_increase=None, max_turnover_increase_bp=None))
    assert len(calls) == 1+result.summary['search_projection_calls'] <= 6
    assert result.attempts.projection_calls.sum() == result.summary['search_projection_calls']
    for checkpoint in result.checkpoints.values():
        assert checkpoint.accepted and checkpoint.compliant
        residuals = checkpoint.outcome.residuals_frame()
        assert residuals.loc[residuals.hard, 'passed'].all()
    assert result.result is result.checkpoints[result.summary['selected_checkpoint']]
    pd.testing.assert_frame_equal(before, p.target)


def test_empty_branch_does_not_starve_active_branch(monkeypatch):
    """The round-robin scheduler skips exhausted legacy paths and continues proposals."""
    p, _, _, _ = _scripted(monkeypatch, [_values(), _values(te=9), _values(te=9, tickets=2)])
    pending = iter([[], ['A'], []])
    monkeypatch.setattr(b, '_removable', lambda *args: next(pending))
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(
        methods=('partial_risk',), max_projection_calls=5, max_removal_trials=1))
    assert result.attempts.branch.tolist() == ['partial_risk', 'partial_risk']
    assert result.summary['removal_budget_exhausted']


def test_zero_optional_domain_has_no_deletion_attempts():
    """Mandatory-only decisions preserve protected coordinates in every branch."""
    p = _problem(max_trades=3)
    p.target[s.MANDATORY_TRADE] = ~p.target[s.SETTLEMENT_CASH]
    p.target[s.RULE4_TRADE] = False
    p.target[s.TRADE_CANDIDATE] = False
    result = b.solve_branched_execution(p)
    assert result.summary['removal_trials'] == 0 and result.result.accepted


def test_unexpected_failure_propagates_and_clears_budget_scope(monkeypatch):
    """Programming faults are visible and cannot leak exhausted state into later work."""
    p, _, _, original = _scripted(monkeypatch, [_values()], [ValueError('broken')])
    with pytest.raises(ValueError, match='broken'):
        b.solve_branched_execution(p, b.ExecutionBranchConfig(max_projection_calls=1))
    assert original(p, b.score_execution_candidates(p), rescue=True).accepted


def test_nested_budget_checks_are_atomic_and_context_is_restored():
    """Nested limits count the same admitted work and reject before partial reservations."""
    outer, inner = _ProjectionBudget(2), _ProjectionBudget(1)
    with _projection_scope(outer):
        with _projection_scope(inner):
            _consume_projection()
            with pytest.raises(_ProjectionBudgetExhausted):
                _consume_projection()
            assert outer.used == inner.used == 1
        _consume_projection()
        assert outer.used == 2
    _consume_projection()
    assert outer.used == 2


def test_existing_guard_can_use_same_projection_ceiling():
    """The select-then-compress comparator receives the same post-baseline work ceiling."""
    result = g.solve_guarded_execution(_problem(), g.ExecutionGuardConfig(max_projection_calls=1))
    assert result.summary['search_projection_calls'] == 1
    assert result.summary['stop_reason'] == 'search_projection_budget'
    assert result.attempts.projection_calls.sum() == 1


def test_existing_guard_handles_exhaustion_inside_rescue(monkeypatch):
    """An interrupted proposal rescue retains the current audited incumbent."""
    p = _problem()
    original = g._solve
    baseline = original(p, g.score_execution_candidates(p), rescue=True)
    calls = []

    def exhausted(*args, **kwargs):
        """Consume two rescue attempts before trying to exceed their shared allowance."""
        calls.append(1)
        if len(calls) == 1:
            return baseline
        for _ in range(3):
            _consume_projection()
        raise AssertionError('budget should have interrupted rescue')

    monkeypatch.setattr(g, '_solve', exhausted)
    result = g.solve_guarded_execution(p, g.ExecutionGuardConfig(max_projection_calls=2))
    assert result.result is baseline and result.summary['search_projection_calls'] == 2
    assert result.attempts.iloc[0].reason == 'search_projection_budget'


def test_budget_stops_mid_round_before_next_branch():
    """Two proposal attempts and one legacy deletion consume the entire shared quota."""
    result = b.solve_branched_execution(_problem(max_trades=3),
                                       b.ExecutionBranchConfig(max_projection_calls=3))
    assert result.summary['search_projection_calls'] == 3
    assert result.attempts.branch.tolist() == ['partial_risk', 'sequential_full_risk', 'legacy']
    assert result.summary['removal_trials'] == 1
