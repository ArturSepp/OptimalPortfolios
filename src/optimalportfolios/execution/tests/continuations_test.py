"""Support-plateau progress and simultaneous guarded/exploratory continuations."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import branches as b, schema as s, solver
from optimalportfolios.execution._budget import _consume_projection
from optimalportfolios.execution.tests.branches_test import _scripted, _values
from optimalportfolios.execution.tests.improvement_test import _problem


def _support_oracle(monkeypatch, outcome):
    """Provide a finite support graph with metrics independent of search decisions."""
    problem = _problem(max_trades=3)
    baseline = b._solve(problem, b.score_execution_candidates(problem), rescue=True)
    names = b._removable(problem, baseline)
    values = {id(baseline): _values()}
    calls = []

    def solve(p, table, rescue):
        """Retain requested supports while counting every admitted projection."""
        if not calls:
            calls.append('baseline')
            return baseline
        _consume_projection()
        selected = frozenset(table.index[table[s.SELECTED_TRADE]])
        calls.append(selected)
        result = replace(baseline, trade_table=table.copy(deep=True))
        values[id(result)] = outcome(selected, names)
        return result

    monkeypatch.setattr(b, '_solve', solve)
    monkeypatch.setattr(b, '_metrics', lambda p, result, *args: values[id(result)])
    return problem, baseline, names, calls


def test_flat_counts_cross_support_plateau_without_resetting_risk(monkeypatch):
    """A flat-count, worse-TE deletion enables a later saving under the original anchor."""
    def outcome(selected, names):
        """Require the first deletion before permitting a two-ticket continuation."""
        if names[0] in selected:
            return _values(te=12)
        if len(selected) == 2:
            return _values(te=10.8)
        if len(selected) == 1:
            return _values(te=10.9, tickets=2, sized=1)
        return _values(te=11.6, tickets=1, sized=1)

    problem, baseline, _, _ = _support_oracle(monkeypatch, outcome)
    result = b.solve_branched_execution(problem, b.ExecutionBranchConfig(
        methods=(), allow_support_plateaus=True))
    assert result.attempts.branch_retained.tolist() == [True, True, False]
    assert result.attempts.branch_reason.iloc[0] == 'support_plateau'
    assert result.attempts.return_retained.tolist() == [False, True, False]
    assert result.summary['final'] == _values(te=10.9, tickets=2, sized=1)
    assert result.checkpoints['baseline'] is baseline
    supports = [set(r.trade_table.index[r.trade_table[s.SELECTED_TRADE]])
                for r in result.checkpoints.values()]
    assert supports[2] < supports[1] < supports[0]


def test_disabled_plateau_keeps_previous_search_policy(monkeypatch):
    """The new controls do not silently change legacy branch advancement."""
    problem, baseline, _, _ = _support_oracle(monkeypatch, lambda *args: _values(te=10.8))
    result = b.solve_branched_execution(problem, b.ExecutionBranchConfig(methods=()))
    assert result.result is baseline
    assert not result.attempts.branch_retained.any()
    assert 'guarded_seed_checkpoint' not in result.summary


@pytest.mark.parametrize('defect', ['registered', 'sized', 'same_support', 'extra_deletion',
                                   'reordered', 'protected', 'missing', 'unselected'])
def test_plateau_requires_exact_eligible_deletion(defect):
    """Flat counts cannot hide growth, protected removal or a different support operation."""
    p = _problem(max_trades=3)
    parent = b._solve(p, b.score_execution_candidates(p), rescue=True)
    name = b._removable(p, parent)[0]
    table = parent.trade_table.copy(deep=True)
    table.at[name, s.SELECTED_TRADE] = False
    values = _values()
    if defect == 'registered':
        values['tickets'] += 1
    elif defect == 'sized':
        values['sized_tickets'] += 1
    elif defect == 'same_support':
        table.at[name, s.SELECTED_TRADE] = True
    elif defect == 'extra_deletion':
        table.at[b._removable(p, parent)[1], s.SELECTED_TRADE] = False
    elif defect == 'reordered':
        table = table.iloc[::-1]
    elif defect == 'protected':
        parent.trade_table.at[name, s.MANDATORY_TRADE] = True
    elif defect == 'missing':
        name = 'absent'
    else:
        parent.trade_table.at[name, s.SELECTED_TRADE] = False
    branch = b._Branch('legacy', parent, _values(), 'baseline', [])
    assert not b._support_plateau(branch, replace(parent, trade_table=table), values, name)


def test_guarded_continuation_skips_ineligible_free_move(monkeypatch):
    """Guarded search retains a feasible alternative after free search takes another route."""
    def outcome(selected, names):
        """The first free deletion blocks the good continuation available via the second."""
        if names[0] not in selected:
            return _values(te=9, tickets=2, sized=1, turnover=120)
        if names[1] not in selected:
            return _values(te=10.8, tickets=1, sized=1, turnover=90)
        return _values(te=12)

    problem, _, _, calls = _support_oracle(monkeypatch, outcome)
    result = b.solve_branched_execution(problem, b.ExecutionBranchConfig(
        methods=(), preserve_guarded_path=True))
    trace = result.attempts
    assert trace.branch.iloc[:3].tolist() == ['guarded', 'legacy', 'guarded']
    assert trace.branch_retained.iloc[:3].tolist() == [False, True, True]
    assert trace.branch_reason.iloc[0] == 'guarded_return_rejected'
    assert result.summary['selected_method'] == 'guarded'
    assert result.summary['final'] == _values(te=10.8, tickets=1, sized=1, turnover=90)
    assert len(calls)-1 == trace.projection_calls.sum() == result.summary['search_projection_calls']
    assert result.summary['guarded_seed_checkpoint'] == 'baseline'


def test_guarded_path_preserves_seed_rehabilitation(monkeypatch):
    """The additional path cannot suppress creation or compression of an ineligible seed."""
    p, _, _, _ = _scripted(monkeypatch, [_values(sized=1),
        _values(te=8, sized=3, turnover=120), _values(te=12), _values(te=12),
        _values(te=8.5, tickets=1, sized=1, turnover=90)])
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(
        methods=('partial_risk',), preserve_guarded_path=True, max_projection_calls=4))
    assert result.attempts.branch.tolist() == ['partial_risk', 'guarded', 'legacy', 'partial_risk']
    assert result.attempts.return_retained.tolist() == [False, False, False, True]
    assert result.summary['selected_method'] == 'partial_risk'
    assert result.summary['search_projection_calls'] == 4


def test_guarded_path_starts_at_best_eligible_seed(monkeypatch):
    """It follows proposal-bank selection, not the first seed or last exploratory seed."""
    p, _, _, _ = _scripted(monkeypatch, [_values(), _values(te=8), _values(te=7, turnover=120)])
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(
        preserve_guarded_path=True, max_removal_trials=0))
    assert result.summary['guarded_seed_checkpoint'] == 'checkpoint_00001'
    assert result.summary['branches']['guarded'] == 'checkpoint_00001'


@pytest.mark.parametrize('turnover,retained', [(100., True), (100.1, False)])
def test_guarded_plateau_keeps_caps_even_without_qualifying_improvement(
        monkeypatch, turnover, retained):
    """A guard-respecting flat step need not itself replace the returned incumbent."""
    p, baseline, _, _ = _support_oracle(monkeypatch,
        lambda *args: _values(te=10.8, turnover=turnover))
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(methods=(),
        preserve_guarded_path=True, allow_support_plateaus=True, max_projection_calls=1))
    assert bool(result.attempts.branch_retained.iloc[0]) is retained
    assert result.result is baseline
    assert result.summary['search_projection_calls'] == 1


@pytest.mark.parametrize('field', ['allow_support_plateaus', 'preserve_guarded_path'])
@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_continuation_controls_require_booleans(field, value):
    """Reject truthy coercion of configuration before numerical work."""
    with pytest.raises(ValueError, match='must be boolean'):
        b.ExecutionBranchConfig(**{field: value})


def test_real_combined_projection_budget_and_protected_support(monkeypatch):
    """Count real solves and inspect exact support deletion for every retained checkpoint."""
    p = _problem(max_trades=3)
    original, calls = solver._solve_selected_execution_portfolio_once, []

    def counted(*args, **kwargs):
        """Count admitted numerical calls independently of the branch trace."""
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, '_solve_selected_execution_portfolio_once', counted)
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(
        allow_support_plateaus=np.bool_(True), preserve_guarded_path=True, max_projection_calls=8))
    assert len(calls) == 1+result.summary['search_projection_calls'] <= 9
    for row in result.attempts[result.attempts.branch_retained].itertuples():
        child = result.checkpoints[row.checkpoint]
        assert child.accepted and child.compliant
        hard = child.outcome.residuals_frame().query('hard')
        assert hard.passed.all()
        if row.stage == 'removal':
            parent = result.checkpoints[row.parent_checkpoint].trade_table
            expected = parent[s.SELECTED_TRADE].copy()
            expected.at[row.removed] = False
            pd.testing.assert_series_equal(child.trade_table[s.SELECTED_TRADE], expected)
    assert result.result is result.checkpoints[result.summary['selected_checkpoint']]
