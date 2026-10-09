"""Eligible-incumbent delivery is independent of exploratory search checkpoints."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import branches as b, schema as s
from optimalportfolios.execution.tests.branches_test import _scripted, _values
from optimalportfolios.execution.tests.improvement_test import _problem


def test_only_original_guard_eligible_replacements_are_delivered(monkeypatch):
    """An attractive but turnover-ineligible seed must never escape as an incumbent."""
    problem, _, _, _ = _scripted(monkeypatch, [
        _values(), _values(te=8, turnover=120), _values(te=9)])
    events = []
    result = b.solve_branched_execution(problem, b.ExecutionBranchConfig(max_removal_trials=0),
        on_incumbent=lambda key, value: events.append((key, value)))
    assert [key for key, _ in events] == ['baseline', 'checkpoint_00002']
    assert result.summary['selected_checkpoint'] == events[-1][0]
    assert result.attempts.return_retained.tolist() == [False, True]


@pytest.mark.parametrize('reuse', [False, True])
def test_callback_copies_cannot_mutate_search_and_final_decision_is_identical(reuse):
    """Even a mutating consumer cannot corrupt weights, tables or numerical outcomes."""
    problem = _problem(max_trades=3)
    controls = b.ExecutionBranchConfig(max_projection_calls=8,
        allow_support_plateaus=True, preserve_guarded_path=True, reuse_projection_solves=reuse)
    expected = b.solve_branched_execution(problem, controls)
    events = []

    def consume(key, result):
        """Keep a clean delivered vector, then deliberately corrupt the detached result."""
        events.append((key, result.weights.copy()))
        result.weights.iloc[:] = np.nan
        result.trade_table[s.SELECTED_TRADE] = False
        result.outcome.weights[:] = np.nan

    actual = b.solve_branched_execution(problem, controls, on_incumbent=consume)
    pd.testing.assert_series_equal(actual.result.weights, expected.result.weights)
    pd.testing.assert_frame_equal(actual.result.trade_table, expected.result.trade_table)
    pd.testing.assert_frame_equal(actual.attempts.drop(columns='seconds'),
                                  expected.attempts.drop(columns='seconds'))
    assert events[0][0] == 'baseline'
    assert events[-1][0] == actual.summary['selected_checkpoint']
    pd.testing.assert_series_equal(events[-1][1], actual.result.weights)


@pytest.mark.parametrize('accepted,relaxed', [(False, False), (True, True)])
def test_unusable_baseline_is_not_delivered(monkeypatch, accepted, relaxed):
    """The callback makes no promise for a rejected or relaxed baseline."""
    baseline = SimpleNamespace(accepted=accepted, compliant=accepted,
        trade_table=pd.DataFrame({b.RELAXED_CORRIDOR_SOLVE: [relaxed]}))
    monkeypatch.setattr(b, '_solve', lambda *args, **kwargs: baseline)
    events = []
    result = b.solve_branched_execution(_problem(),
        on_incumbent=lambda *event: events.append(event))
    assert not events and result.result is baseline


def test_zero_search_budget_still_delivers_baseline():
    """Saving a fallback does not consume the post-baseline projection allowance."""
    events = []
    result = b.solve_branched_execution(_problem(), b.ExecutionBranchConfig(max_projection_calls=0),
        on_incumbent=lambda *event: events.append(event))
    assert len(events) == 1 and events[0][0] == 'baseline'
    pd.testing.assert_series_equal(events[0][1].weights, result.result.weights)
    assert result.summary['search_projection_calls'] == 0


@pytest.mark.parametrize('key', ['baseline', 'checkpoint_00001'])
def test_consumer_error_propagates_without_silent_recovery(monkeypatch, key):
    """Persistence failure is an error, not a successfully delivered checkpoint."""
    problem, _, _, _ = _scripted(monkeypatch, [_values(), _values(te=9)])

    def consume(checkpoint, result):
        """Fail at the chosen delivery boundary."""
        if checkpoint == key:
            raise OSError('checkpoint storage failed')

    with pytest.raises(OSError, match='checkpoint storage failed'):
        b.solve_branched_execution(problem,
            b.ExecutionBranchConfig(methods=('partial_risk',), max_removal_trials=0),
            on_incumbent=consume)


def test_completed_incumbent_survives_later_projection_error(monkeypatch):
    """The caller already owns the last delivered result when later work fails."""
    problem = _problem(max_trades=3)
    baseline = b._solve(problem, b.score_execution_candidates(problem), rescue=True)
    # Supply the first proposal without evaluating another numerical problem.
    proposal = replace(baseline, weights=baseline.weights.copy())
    problem, _, _, _ = _scripted(monkeypatch,
        [_values(), _values(te=9)], replies=[proposal, RuntimeError('solver crashed')])
    events = []
    with pytest.raises(RuntimeError, match='solver crashed'):
        b.solve_branched_execution(problem, b.ExecutionBranchConfig(max_removal_trials=0),
            on_incumbent=lambda *event: events.append(event))
    assert [key for key, _ in events] == ['baseline', 'checkpoint_00001']


def test_noncallable_observer_is_rejected_before_numerical_work(monkeypatch):
    """Invalid callbacks cannot start a costly baseline solve."""
    calls = []
    monkeypatch.setattr(b, '_solve', lambda *args, **kwargs: calls.append(1))
    with pytest.raises(TypeError, match='on_incumbent must be callable'):
        b.solve_branched_execution(_problem(), on_incumbent=1)
    assert not calls
