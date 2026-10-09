"""Exact projection identity, isolation, budget and independent work regressions."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import branches as b, schema as s, solver
from optimalportfolios.execution._budget import _ProjectionBudget, _projection_scope
from optimalportfolios.execution._projection_cache import (
    _ProjectionCache, _project, _projection_reuse_scope,
)
from optimalportfolios.execution.tests.improvement_test import _problem


def _inputs():
    """Build the actual resolved wrapper inputs without running its optimiser."""
    p = _problem(max_trades=3)
    table = b.score_execution_candidates(p)
    return dict(pd_covar=p.covariance, benchmark_weights=table[s.RAW_MODEL_WEIGHT],
                weights_0=table[s.CURRENT_WEIGHT], optimiser_config=p.optimiser_config,
                constraints=solver.build_selected_execution_constraints(p.constraints, table),
                context='one decision')


def _stub(**inputs):
    """Supply detached finite output for identity tests, not a feasibility oracle."""
    weights = inputs['benchmark_weights'].copy(deep=True)
    return weights, SimpleNamespace(accepted=True, compliant=True, status='optimal',
        weights=weights.to_numpy(copy=True), constraints=deepcopy(inputs['constraints']))


@pytest.mark.parametrize('change', ['covariance', 'benchmark', 'holdings', 'bounds',
                                   'constraint_field', 'solver', 'factorization',
                                   'order', 'context'])
def test_complete_resolved_identity_rejects_each_changed_input(change):
    """Equal support alone cannot reuse a different numerical projection."""
    inputs = _inputs()
    cache = _ProjectionCache()
    cache.project(_stub, inputs)
    changed = deepcopy(inputs)
    if change == 'covariance':
        changed['pd_covar'].iloc[0, 0] += 0.1
    elif change == 'benchmark':
        changed['benchmark_weights'].iloc[0] += 0.01
    elif change == 'holdings':
        changed['weights_0'].iloc[0] += 0.01
    elif change == 'bounds':
        changed['constraints'].max_weights.iloc[0] -= 0.01
    elif change == 'constraint_field':
        changed['constraints'] = replace(changed['constraints'], max_exposure=1.1)
    elif change == 'solver':
        changed['optimiser_config'] = replace(changed['optimiser_config'], solver='SCS')
    elif change == 'factorization':
        changed['optimiser_config'] = replace(changed['optimiser_config'], factorize_covar=False)
    elif change == 'order':
        changed['pd_covar'] = changed['pd_covar'].iloc[::-1, ::-1]
    else:
        changed['context'] = 'different decision'
    cache.project(_stub, changed)
    assert cache.numerical_calls == 2 and cache.hits == 0
    cache.project(_stub, deepcopy(inputs))
    assert cache.numerical_calls == 2 and cache.hits == 1


def test_hits_are_detached_from_fresh_outputs_and_other_hits():
    """Mutable weight, outcome and constraint objects cannot poison cached results."""
    inputs, cache = _inputs(), _ProjectionCache()
    first = cache.project(_stub, inputs)
    expected = deepcopy(first)
    first[0].iloc[0] = -10
    first[1].weights[1] = -10
    first[1].constraints.max_weights.iloc[0] = -10
    second = cache.project(_stub, inputs)
    pd.testing.assert_series_equal(second[0], expected[0], check_exact=True)
    np.testing.assert_array_equal(second[1].weights, expected[1].weights)
    pd.testing.assert_series_equal(second[1].constraints.max_weights,
                                   expected[1].constraints.max_weights, check_exact=True)
    second[0].iloc[0] = -20
    third = cache.project(_stub, inputs)
    pd.testing.assert_series_equal(third[0], expected[0], check_exact=True)
    assert cache.numerical_calls == 1 and cache.hits == 2


@pytest.mark.parametrize('defect', ['rejected', 'noncompliant', 'inaccurate',
                                   'nonfinite_series', 'nonfinite_outcome', 'exception'])
def test_failures_and_inaccurate_or_nonfinite_results_are_never_reused(defect):
    """Caching must not turn a later retry into an earlier unresolved outcome."""
    cache, calls = _ProjectionCache(), []

    def projection(**inputs):
        """Return the chosen first-call failure and then a successful result."""
        result = _stub(**inputs)
        calls.append(True)
        if len(calls) == 1:
            if defect == 'exception':
                raise RuntimeError('solver error')
            if defect == 'rejected':
                result[1].accepted = False
            elif defect == 'noncompliant':
                result[1].compliant = False
            elif defect == 'inaccurate':
                result[1].status = 'optimal_inaccurate'
            elif defect == 'nonfinite_series':
                result[0].iloc[0] = np.nan
            else:
                result[1].weights[0] = np.inf
        return result

    if defect == 'exception':
        with pytest.raises(RuntimeError, match='solver error'):
            cache.project(projection, _inputs())
    else:
        cache.project(projection, _inputs())
    assert not cache.entries
    cache.project(projection, _inputs())
    cache.project(projection, _inputs())
    assert len(calls) == cache.numerical_calls == 2 and cache.hits == 1


def test_unserializable_metadata_falls_back_to_fresh_projection():
    """Optional reuse cannot make previously valid inputs require serialization."""
    inputs, cache = _inputs(), _ProjectionCache()
    inputs['pd_covar'].attrs['annotation'] = lambda: None
    cache.project(_stub, inputs)
    cache.project(_stub, inputs)
    assert cache.numerical_calls == 2 and cache.hits == 0 and not cache.entries


def test_scopes_restore_after_exceptions_and_never_cross_decisions():
    """An inner scope, explicit bypass or exception cannot leak cached state."""
    inputs, outer, inner = _inputs(), _ProjectionCache(), _ProjectionCache()
    with _projection_reuse_scope(outer):
        _project(_stub, **inputs)
        with pytest.raises(RuntimeError), _projection_reuse_scope(inner):
            _project(_stub, **inputs)
            raise RuntimeError('leave scope')
        _project(_stub, **inputs)
        with _projection_reuse_scope(None):
            _project(_stub, **inputs)
    _project(_stub, **inputs)
    assert (outer.numerical_calls, outer.hits) == (1, 1)
    assert (inner.numerical_calls, inner.hits) == (1, 0)


@pytest.mark.parametrize('limit', [1, 5, 52])
def test_reuse_preserves_real_solver_paths_and_logical_budget(monkeypatch, limit):
    """An independent numerical-wrapper counter proves saved work and exact decisions."""
    p = _problem(max_trades=3)
    config = b.ExecutionBranchConfig(allow_support_plateaus=True, preserve_guarded_path=True,
        max_projection_calls=limit, max_sized_ticket_increase=0, max_turnover_increase_bp=0)
    calls = []
    original = solver.wrapper_minimise_tracking_error

    def counted(**inputs):
        """Count actual numerical invocations independently of cache telemetry."""
        calls.append(True)
        return original(**inputs)

    monkeypatch.setattr(solver, 'wrapper_minimise_tracking_error', counted)
    expected = b.solve_branched_execution(p, config)
    uncached_calls = len(calls)
    calls.clear()
    outer_budget = _ProjectionBudget(None)
    with _projection_scope(outer_budget):
        actual = b.solve_branched_execution(p, replace(config, reuse_projection_solves=True))
    assert actual.summary['search_projection_calls'] == expected.summary['search_projection_calls']
    assert outer_budget.used == 1+actual.summary['search_projection_calls']
    assert list(actual.checkpoints) == list(expected.checkpoints)
    for name, result in actual.checkpoints.items():
        reference = expected.checkpoints[name]
        pd.testing.assert_series_equal(result.weights, reference.weights, check_exact=True)
        pd.testing.assert_frame_equal(result.trade_table, reference.trade_table, check_exact=True)
        pd.testing.assert_frame_equal(result.outcome.residuals_frame(),
                                      reference.outcome.residuals_frame(), check_exact=True)
    columns = expected.attempts.columns.drop('seconds')
    pd.testing.assert_frame_equal(actual.attempts[columns], expected.attempts[columns],
                                  check_exact=True)
    for name, value in expected.summary.items():
        if name != 'seconds':
            assert actual.summary[name] == value
    assert len(calls)-1 == actual.summary['search_numerical_projection_calls']
    assert actual.attempts.numerical_projection_calls.sum() == len(calls)-1
    assert (actual.attempts.projection_cache_hits.sum()
            == actual.summary['search_projection_cache_hits'])
    assert len(calls) <= uncached_calls
    if limit == 52:
        assert len(calls) < uncached_calls and actual.summary['search_projection_cache_hits'] > 0


def test_projection_callable_identity_is_part_of_cache_key():
    """Replacing the numerical implementation cannot hit a previous callable's entry."""
    cache, inputs = _ProjectionCache(), _inputs()
    cache.project(_stub, inputs)
    cache.project(lambda **kwargs: _stub(**kwargs), inputs)
    assert cache.numerical_calls == 2 and cache.hits == 0


@pytest.mark.parametrize('invalid', [1, 'yes', None])
def test_reuse_option_requires_boolean(invalid):
    """Reject implicit truthiness for the opt-in numerical reuse policy."""
    with pytest.raises(ValueError, match='reuse_projection_solves must be boolean'):
        b.ExecutionBranchConfig(reuse_projection_solves=invalid)


def test_reuse_rejects_wall_clock_stopping():
    """Faster solves must not silently admit additional trials before a time limit."""
    with pytest.raises(ValueError, match='max_search_seconds=None'):
        b.ExecutionBranchConfig(reuse_projection_solves=True, max_search_seconds=1)


def test_reused_projection_rebuilds_each_requests_trade_table():
    """Numerical identity does not reuse branch-specific ranking or audit metadata."""
    p = _problem(max_trades=3)
    first = b.score_execution_candidates(p)
    first['audit_tag'] = 'first branch'
    second = first.copy(deep=True)
    second['audit_tag'] = 'second branch'
    second[s.TRADE_SCORE] += 1.0
    cache, budget = _ProjectionCache(), _ProjectionBudget(2)
    with _projection_scope(budget), _projection_reuse_scope(cache):
        a = solver.solve_selected_execution_portfolio(first, p.constraints, p.covariance,
                                                       p.optimiser_config)
        c = solver.solve_selected_execution_portfolio(second, p.constraints, p.covariance,
                                                       p.optimiser_config)
    assert cache.numerical_calls == cache.hits == 1 and budget.used == 2
    assert a.trade_table.audit_tag.eq('first branch').all()
    assert c.trade_table.audit_tag.eq('second branch').all()
    pd.testing.assert_series_equal(c.trade_table[s.TRADE_SCORE], second[s.TRADE_SCORE])
    pd.testing.assert_series_equal(a.weights, c.weights, check_exact=True)


def test_dust_cleanup_is_recomputed_after_reusing_the_raw_projection():
    """A changed execution dust threshold acts on copied raw weights after a hit."""
    from optimalportfolios.execution.tests.solver_test import _trade_table
    from optimalportfolios.optimization.config import OptimiserConfig
    from optimalportfolios.optimization.constraints import Constraints

    table = _trade_table(current=[0.5, 0.001, 0.499], model=[0.5, 0.001, 0.499],
                         selected=[False, True, False])
    table[s.CUTOFF_SELL_DOWN] = table.index == 'B'
    table[s.SELL_DOWN_DUST_THRESHOLD] = 0.0
    constraints = Constraints(min_weights=pd.Series(0., index=table.index),
                              max_weights=pd.Series(1., index=table.index))
    covariance = pd.DataFrame(np.diag([0.01, 0.01, 0.01]),
                               index=table.index, columns=table.index)
    config, cache = OptimiserConfig(solver='CLARABEL'), _ProjectionCache()
    with _projection_reuse_scope(cache):
        retained = solver.solve_selected_execution_portfolio(table, constraints, covariance, config)
        table[s.SELL_DOWN_DUST_THRESHOLD] = 0.0025
        cleaned = solver.solve_selected_execution_portfolio(table, constraints, covariance, config)
    fresh = solver.solve_selected_execution_portfolio(table, constraints, covariance, config)
    assert retained.weights['B'] > 0.0005 and cleaned.weights['B'] == 0.0
    assert cache.numerical_calls == cache.hits == 1
    pd.testing.assert_series_equal(cleaned.weights, fresh.weights, check_exact=True)
    pd.testing.assert_frame_equal(cleaned.trade_table, fresh.trade_table, check_exact=True)
    assert cleaned.accepted and cleaned.compliant
