"""Independent acceptance boundaries, tie controls and guarded-search regressions."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import guarded as g, schema as s
from optimalportfolios.execution.improvement import ExecutionScoreMethod, score_execution_candidates
from optimalportfolios.execution.tests.improvement_test import _problem


@pytest.mark.parametrize('changes', [
    {'methods': ('legacy',)}, {'methods': ('partial_risk', 'partial_risk')},
    {'canonical_ties': 'yes'}, {'max_removal_trials': True}, {'max_removal_trials': -1},
    {'max_sized_ticket_increase': .5}, {'te_allowance_bp': np.nan},
    {'min_te_improvement_bp': 0}, {'ticket_size_bp': 0}, {'max_turnover_increase_bp': -1},
    {'max_search_seconds': np.inf}, {'max_search_seconds': True},
])
def test_bad_controls_fail_before_solving(changes):
    """Malformed budgets cannot silently disable or widen the guard."""
    with pytest.raises(ValueError):
        g.ExecutionGuardConfig(**changes)


def test_disabled_bank_and_compression_preserve_exact_legacy_result():
    """An opt-in wrapper with no work has exact legacy numerical parity."""
    p = _problem()
    expected = g._solve(p, score_execution_candidates(p), rescue=True)
    actual = g.solve_guarded_execution(p, g.ExecutionGuardConfig(methods=(), max_removal_trials=0))
    pd.testing.assert_series_equal(actual.result.weights, expected.weights)
    pd.testing.assert_frame_equal(actual.result.trade_table, expected.trade_table)
    assert actual.attempts.empty
    assert actual.summary['baseline'] == actual.summary['final']


def test_real_guarded_search_respects_original_operational_bounds():
    """Actual joint projections retain audit compliance and the original input basis."""
    p = _problem(max_trades=3)
    before = p.target.copy(deep=True)
    result = g.solve_guarded_execution(p)
    base, final = result.summary['baseline'], result.summary['final']
    assert result.result.accepted and result.result.compliant
    assert final['te_bp'] <= base['te_bp']+1
    assert final['tickets'] <= base['tickets']
    assert final['sized_tickets'] <= base['sized_tickets']
    assert final['gross_turnover_bp'] <= base['gross_turnover_bp']+1e-8
    pd.testing.assert_frame_equal(before, p.target)


@pytest.mark.parametrize('method', [ExecutionScoreMethod.FULL_RISK,
    ExecutionScoreMethod.PARTIAL_RISK, ExecutionScoreMethod.SEQUENTIAL_FULL_RISK])
def test_exact_ties_change_only_selection_not_numerical_order(method):
    """Equal funded moves select the supplied priority without permuting the covariance."""
    p = _problem()
    p.target[s.CURRENT_WEIGHT] = .25
    p.target[s.BASE_WEIGHT] = .25
    p.target[s.RAW_MODEL_WEIGHT] = [.375, .375, .25, 0.]
    p.target[s.EFFECTIVE_MODEL_WEIGHT] = p.target[s.RAW_MODEL_WEIGHT]
    p.target[s.DESIRED_REWEIGHT] = [.125, .125, 0., 0.]
    p.target.loc['C', [s.TRADE_CANDIDATE, s.MATERIAL_TRADE]] = False
    p.covariance.iloc[:, :] = np.diag([.01, .01, .01, 0.])
    default = score_execution_candidates(p, method)
    priority = pd.Series([1, 0, 2, 3], index=p.target.index)
    alternative = score_execution_candidates(p, method, tie_priority=priority)
    assert list(default.index[default[s.SELECTED_TRADE]]) == ['A']
    assert list(alternative.index[alternative[s.SELECTED_TRADE]]) == ['B']
    pd.testing.assert_series_equal(default[s.TRADE_SCORE], alternative[s.TRADE_SCORE])
    assert alternative.index.equals(p.target.index)


@pytest.mark.parametrize('priority,method,message', [
    (pd.Series([0, 1, 2, 3], index=['A', 'B', 'C', 'Cash']), 'legacy', 'legacy'),
    (pd.Series([0, 1], index=['A', 'A']), 'partial_risk', 'cover exactly'),
    (pd.Series([0], index=['A']), 'partial_risk', 'cover exactly'),
    (pd.Series([0, 0, 2, 3], index=['A', 'B', 'C', 'Cash']), 'partial_risk', 'finite and unique'),
    (pd.Series([0, np.nan, 2, 3], index=['A', 'B', 'C', 'Cash']),
     'partial_risk', 'finite and unique'),
])
def test_invalid_tie_overrides_are_explicit(priority, method, message):
    """Missing, ambiguous or nonfinite priorities cannot silently affect rankings."""
    with pytest.raises(ValueError, match=message):
        score_execution_candidates(_problem(), method, tie_priority=priority)


def _metrics(te=10., tickets=3, sized=2, turnover=100.):
    """Supply independently controlled metrics to test acceptance inequalities."""
    return dict(te_bp=te, tickets=tickets, sized_tickets=sized, gross_turnover_bp=turnover)


@pytest.mark.parametrize('candidate,reason', [
    (_metrics(te=9, tickets=4), 'more_registered_tickets'),
    (_metrics(te=9, sized=3), 'more_sized_tickets'),
    (_metrics(te=9, turnover=101), 'more_turnover'),
    (_metrics(te=9.9998), 'tracking_improvement'),
    (_metrics(te=11, tickets=2), 'ticket_saving_within_allowance'),
    (_metrics(te=11.00001, tickets=2), 'no_qualifying_improvement'),
    (_metrics(te=10.5), 'no_qualifying_improvement'),
    (_metrics(), 'no_qualifying_improvement'),
])
def test_independent_guard_boundaries(candidate, reason):
    """Protect both economic thresholds and the original registered-ticket contract."""
    assert g._qualification(_metrics(), candidate, g.ExecutionGuardConfig()) == reason


def test_optional_operational_guards_and_priority_order():
    """Disabled extra controls reproduce the original registered-count eligibility rule."""
    c = g.ExecutionGuardConfig(max_sized_ticket_increase=None, max_turnover_increase_bp=None)
    assert (g._qualification(_metrics(), _metrics(te=9, sized=3, turnover=200), c)
            == 'tracking_improvement')
    assert g._preference(_metrics(te=9, tickets=3)) > g._preference(_metrics(te=11, tickets=2))
    assert g._canonical_priority(pd.Index(['B', 'A'])).tolist() == [1, 0]
    with pytest.raises(ValueError, match='distinct'):
        g._canonical_priority(pd.Index([1, '1']))


def _scripted(monkeypatch, values, results=None):
    """Inject audited trial outputs and independent scalar references in solve order."""
    p = _problem(max_trades=3)
    real = g._solve(p, score_execution_candidates(p), rescue=True)
    measured = iter(values)
    monkeypatch.setattr(g, '_metrics', lambda *args: next(measured))
    replies = iter(results) if results is not None else None

    def solve(*args, **kwargs):
        """Return the scripted solver outcome or raise its structural failure."""
        response = next(replies) if replies is not None else real
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(g, '_solve', solve)
    return p, real


def test_compression_cannot_reset_allowance_to_winning_proposal(monkeypatch):
    """A 0.8 bp proposal followed by another 0.8 bp removal exceeds the original 1 bp."""
    p, _ = _scripted(monkeypatch, [_metrics(), _metrics(te=10.8, tickets=2),
                                  _metrics(te=11.6, tickets=1)])
    result = g.solve_guarded_execution(p, g.ExecutionGuardConfig(
        methods=(ExecutionScoreMethod.PARTIAL_RISK,), max_removal_trials=1))
    assert result.attempts.retained.tolist() == [True, False]
    assert result.summary['final']['te_bp'] == 10.8
    assert result.summary['removal_budget_exhausted']


def test_operational_caps_cannot_reset_to_a_proposal(monkeypatch):
    """Permitted turnover increase is total, not repeated at each compression step."""
    p, _ = _scripted(monkeypatch, [_metrics(), _metrics(te=9, tickets=2, turnover=110),
                                  _metrics(te=9, tickets=1, turnover=120)])
    result = g.solve_guarded_execution(p, g.ExecutionGuardConfig(
        methods=('partial_risk',), max_removal_trials=1, max_turnover_increase_bp=10))
    assert result.attempts.retained.tolist() == [True, False]
    assert result.attempts.iloc[-1].reason == 'more_turnover'


def test_proposal_ordering_and_compression_require_actual_ticket_saving(monkeypatch):
    """A lower-ranked proposal or a removal with only TE gain cannot replace the winner."""
    p, _ = _scripted(monkeypatch, [_metrics(), _metrics(te=9), _metrics(te=9.5),
                                  _metrics(te=8)])
    result = g.solve_guarded_execution(p, g.ExecutionGuardConfig(max_removal_trials=1))
    assert result.attempts.retained.tolist() == [True, False, False]
    assert result.attempts.reason.tolist()[-2:] == ['incumbent_preferred', 'no_ticket_removed']


def test_multiple_compressions_keep_one_anchor(monkeypatch):
    """Successful removals update the incumbent but leave the baseline reference intact."""
    p, _ = _scripted(monkeypatch, [_metrics(), _metrics(te=10.5, tickets=2),
                                  _metrics(te=10.9, tickets=1)])
    result = g.solve_guarded_execution(p, g.ExecutionGuardConfig(methods=(), max_removal_trials=2))
    assert result.attempts.retained.all()
    assert result.summary['baseline']['te_bp'] == 10
    assert result.summary['final']['tickets'] == 1


def test_zero_time_budget_keeps_audited_baseline(monkeypatch):
    """An exhausted search budget never starts proposals or removal projections."""
    p, baseline = _scripted(monkeypatch, [_metrics()])
    result = g.solve_guarded_execution(p, g.ExecutionGuardConfig(max_search_seconds=0))
    assert result.result is baseline
    assert result.attempts.empty
    assert result.summary['stop_reason'] == 'search_time_budget'


@pytest.mark.parametrize('accepted,relaxed,reason', [
    (False, False, 'baseline_not_accepted'), (True, True, 'baseline_relaxed_corridors')])
def test_unusable_baseline_launches_no_search(monkeypatch, accepted, relaxed, reason):
    """A failed or relaxed legacy result cannot be relabeled as a strict improvement."""
    baseline = SimpleNamespace(accepted=accepted, compliant=accepted,
        trade_table=pd.DataFrame({g.RELAXED_CORRIDOR_SOLVE: [relaxed]}))
    monkeypatch.setattr(g, '_solve', lambda *args, **kwargs: baseline)
    result = g.solve_guarded_execution(_problem())
    assert result.result is baseline and result.summary['stop_reason'] == reason


@pytest.mark.parametrize('candidate,reason', [
    (g.ExecutionSolverInfeasibility('trial conflict'), 'trial conflict'),
    (SimpleNamespace(accepted=False, compliant=False, outcome=SimpleNamespace(status='rejected')),
     'candidate_not_accepted'),
    (SimpleNamespace(accepted=True, compliant=True, outcome=SimpleNamespace(status='optimal'),
     trade_table=pd.DataFrame({g.RELAXED_CORRIDOR_SOLVE: [True]})), 'candidate_relaxed_corridors'),
])
def test_failed_candidates_retain_legacy(monkeypatch, candidate, reason):
    """Structural, audited and relaxed-corridor candidate failures preserve the incumbent."""
    p = _problem()
    baseline = g._solve(p, score_execution_candidates(p), rescue=True)
    replies = iter([baseline, candidate])

    def solve(*args, **kwargs):
        """Return the controlled baseline and candidate outcomes."""
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(g, '_solve', solve)
    result = g.solve_guarded_execution(p, g.ExecutionGuardConfig(
        methods=('partial_risk',), max_removal_trials=0))
    assert result.result is baseline and result.attempts.iloc[0].reason == reason


@pytest.mark.parametrize(
    'defect,message', [('order', 'order'), ('nan', 'finite'), ('risk', 'tracking')])
def test_invalid_metrics_cannot_be_accepted(defect, message):
    """Incorrect alignment or missing numerical values stop a claimed comparison."""
    p = _problem()
    weights = p.target[s.CURRENT_WEIGHT].copy()
    if defect == 'order':
        weights = weights.iloc[::-1]
    elif defect == 'nan':
        weights.iloc[0] = np.nan
    risk = SimpleNamespace(compute_tre_at_date=lambda *args: np.nan if defect == 'risk' else .1)
    with pytest.raises(ValueError, match=message):
        g._metrics(p, SimpleNamespace(weights=weights), risk, 1.)


def test_cash_is_excluded_and_current_not_base_defines_ticket_sizes():
    """Independent scalar recount includes mandatory changes and ignores funding cash."""
    p = _problem()
    weights = p.target[s.CURRENT_WEIGHT]+pd.Series([.001, 1e-6, 0., -.001001], index=p.target.index)
    p.target[s.BASE_WEIGHT] = weights
    risk = SimpleNamespace(compute_tre_at_date=lambda *args: .002)
    actual = g._metrics(p, SimpleNamespace(weights=weights), risk, 1.)
    assert actual['te_bp'] == 20
    assert actual['tickets'] == 2 and actual['sized_tickets'] == 1
    assert actual['gross_turnover_bp'] == pytest.approx(10.01)


def test_protected_rows_never_enter_removal_trials():
    """Mandatory and cadence-pinned coordinates remain protected by the saved domain."""
    p = _problem(max_trades=3)
    p.target.loc['A', s.MANDATORY_TRADE] = True
    p.target.loc[['A', 'B'], s.TRADE_CANDIDATE] = False
    p.target.loc['B', s.REBALANCE_CADENCE_ELIGIBLE] = False
    result = g.solve_guarded_execution(p, g.ExecutionGuardConfig(methods=(), max_removal_trials=3))
    assert not result.attempts.removed.isin(['A', 'B', 'Cash']).any()


@pytest.mark.parametrize('method', [ExecutionScoreMethod.FULL_RISK,
    ExecutionScoreMethod.PARTIAL_RISK, ExecutionScoreMethod.SEQUENTIAL_FULL_RISK])
def test_tie_priority_cannot_expand_an_empty_candidate_domain(method):
    """Pandas assignment to an empty frame must not introduce protected instruments."""
    p = _problem()
    p.target[s.TRADE_CANDIDATE] = False
    p.target[s.RULE4_TRADE] = False
    priority = g._canonical_priority(p.target.index)
    result = score_execution_candidates(p, method, tie_priority=priority)
    assert not result[s.SELECTED_TRADE].any()
    assert result[s.TRADE_RANK].isna().all()
    assert result.index.equals(p.target.index)


def test_empty_optional_domain_retains_legacy_through_the_bank():
    """A valid hold decision remains available when every optional coordinate is pinned."""
    p = _problem()
    p.target[s.TRADE_CANDIDATE] = False
    p.target[s.RULE4_TRADE] = False
    expected = g._solve(p, score_execution_candidates(p), rescue=True)
    result = g.solve_guarded_execution(p)
    assert result.result.accepted and result.result.compliant
    pd.testing.assert_series_equal(result.result.weights, expected.weights)
    assert not result.attempts.retained.any()
