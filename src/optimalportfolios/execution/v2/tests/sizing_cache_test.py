"""Verify exact conditional-QP reuse without removing search trials."""

from dataclasses import replace
from itertools import product

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from optimalportfolios.execution.v2 import audit_execution, compile_states, wrapper_ranked_execution
from optimalportfolios.execution.v2._sizing import _CachedSizingModel, _FixedStateSizer, _same_data
from optimalportfolios.execution.v2.domain import variance
from optimalportfolios.execution.v2.model import ConicModel, build_model, solve_qp
from optimalportfolios.execution.v2.tests.formulation_test import example, independent_kkt
from optimalportfolios.execution.v2.tests.fixed_rows_test import one_trade
from optimalportfolios.execution.v2.tests.group_audit_test import pinned_group
from optimalportfolios.execution.v2.tests.scarce_search_test import replacement_problem


def test_search_reuses_compilation_without_removing_sizing_trials(monkeypatch):
    """The public synthetic pair case must keep its optimum with fewer graph builds."""
    import optimalportfolios.execution.v2.ranked as module
    original = module.build_model
    fixed_builds = []

    def counted(problem, **kwargs):
        """Count expensive fixed-state graph construction without changing the solver."""
        if kwargs.get('fixed_states') is not None:
            fixed_builds.append(tuple(sorted(kwargs['fixed_states'].items())))
        return original(problem, **kwargs)

    monkeypatch.setattr(module, 'build_model', counted)
    problem = replacement_problem()
    result = wrapper_ranked_execution(problem)
    assert audit_execution(problem, result.weights)['passed']
    np.testing.assert_allclose(result.weights, [.1, .1, .16, .04, .6], atol=2e-6, rtol=0.)
    trials = len(result.diagnostics['trials'])
    assert trials >= 8
    assert len(fixed_builds) < trials / 2


@pytest.mark.parametrize('factor', [False, True])
@pytest.mark.parametrize('groups', ['signed', 'none'])
@pytest.mark.parametrize('cap', [None, 2])
def test_enumerated_canonical_data_and_native_weights_match_reference(factor, groups, cap):
    """Enumerate all tiny supports against a separately built CVXPY graph."""
    problem = example(cap=cap, coupled=False)
    if factor:
        loadings = np.random.default_rng(1103).normal(size=(5, 2)) * .12
        problem = replace(problem, covariance=None, factor_loadings=loadings,
                          factor_covariance=np.eye(2),
                          residual_variance=np.array([.02, .015, .01, .025, .004]))
    if groups == 'signed':
        rows = np.array([[.3, -.7, .1, .2, .4], [-1., -1., 0., 0., 0.]])
        exposure = rows @ problem.base
        problem = replace(problem, groups=rows, group_lower=exposure-.04,
                          group_upper=exposure+.04)
    problem = replace(problem, rules=replace(problem.rules, discretionary_allowance=cap))
    states = compile_states(problem)
    sizer = _FixedStateSizer(problem, states, build_model, 'CLARABEL')
    for choices in product(*(range(len(row)) for row in states[:-1])):
        fixed = dict(enumerate(choices))
        expected = build_model(problem, fixed_states=fixed)
        actual = sizer(fixed)
        if isinstance(actual, _CachedSizingModel):
            data = expected.optimisation.get_problem_data('CLARABEL', solver_opts={})[0]
            for name in ('P', 'A', 'b', 'c'):
                left, right = data[name], actual.data[name]
                if sp.issparse(left):
                    for field in ('indptr', 'indices', 'data'):
                        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
                else:
                    np.testing.assert_array_equal(left, right)
        first, second = solve_qp(expected), solve_qp(actual)
        assert expected.optimisation.status == actual.optimisation.status
        assert (first is None) == (second is None)
        if first is not None:
            np.testing.assert_array_equal(first, second)
    assert sizer.diagnostics['reuses'] > 0


@pytest.mark.parametrize('side', ['lower', 'upper'])
@pytest.mark.parametrize('gap', [.99e-7, 1.01e-7])
def test_cached_constant_rows_keep_audit_boundary_and_genuine_breach(side, gap):
    """Reuse must retain both admitted residuals and genuinely contradictory rows."""
    problem = pinned_group(side, gap)
    fixed, states = one_trade(problem)
    sizer = _FixedStateSizer(problem, states, build_model, 'CLARABEL')
    for _ in range(2):
        model = sizer(fixed)
        weights = solve_qp(model)
        if gap > problem.rules.audit_tolerance:
            assert weights is None and model.optimisation.status == cp.INFEASIBLE
        else:
            assert audit_execution(problem, weights)['passed']
            reference = replace(problem, group_lower=np.array([-1.]), group_upper=np.array([1.]))
            optimum = independent_kkt(reference, {
                i: (states[i][j].lower, states[i][j].upper) for i, j in fixed.items()})
            assert abs(variance(problem, weights) - optimum) < 1e-12


def test_layout_guard_falls_back_without_changing_sizing(monkeypatch):
    """A future incompatible CVXPY layout must keep using the original compiler."""
    import optimalportfolios.execution.v2._sizing as module
    problem = example()
    states = compile_states(problem)
    fixed = {i: next(j for j, s in enumerate(row) if s.ticket == (i < 2))
             for i, row in enumerate(states[:-1])}
    expected = solve_qp(build_model(problem, fixed_states=fixed))
    monkeypatch.setattr(module, '_same_data', lambda *args: False)
    sizer = _FixedStateSizer(problem, states, build_model, 'CLARABEL')
    for _ in range(2):
        model = sizer(fixed)
        assert isinstance(model, ConicModel)
        np.testing.assert_array_equal(solve_qp(model), expected)
    assert sizer.diagnostics == dict(layouts=1, reuses=0, fallbacks=2)


def test_bounded_layout_eviction_and_other_solver_keep_reference_results(monkeypatch):
    """Eviction changes compilation cost only; other solvers retain their old path."""
    import optimalportfolios.execution.v2._sizing as module
    monkeypatch.setattr(module, '_LAYOUT_LIMIT', 1)
    problem = example()
    states = compile_states(problem)
    sizer = _FixedStateSizer(problem, states, build_model, 'CLARABEL')
    other = _FixedStateSizer(problem, states, build_model, 'SCS')
    for count in (1, 2, 1):
        fixed = {i: next(j for j, s in enumerate(row) if s.ticket == (i < count))
                 for i, row in enumerate(states[:-1])}
        original = other(fixed)
        assert isinstance(original, ConicModel)
        np.testing.assert_array_equal(solve_qp(sizer(fixed)), solve_qp(original))
        assert len(sizer.layouts) == 1
    assert sizer.diagnostics['layouts'] == 3
    with pytest.raises(ValueError, match='CLARABEL'):
        sizer(fixed).solve('SCS')


@pytest.mark.parametrize('error', ['missing', 'shape', 'value', 'vector'])
def test_layout_guard_rejects_every_coefficient_mismatch(error):
    """Neither sparse layout changes nor numeric differences permit approximate reuse."""
    supplied = dict(P=sp.csc_array(np.eye(2)), c=np.ones(2))
    compiled = dict(supplied)
    if error == 'missing':
        compiled.pop('P')
    elif error == 'shape':
        compiled['P'] = sp.csc_array(np.eye(3))
    elif error == 'value':
        compiled['P'] = sp.csc_array(np.eye(2) * 2)
    else:
        compiled['c'] = np.zeros(2)
    assert not _same_data(compiled, supplied)


def test_cached_solve_retains_bounded_fresh_solver_retry(monkeypatch):
    """The existing retry settings apply to reused data and reset on each trial."""
    problem = example()
    states = compile_states(problem)
    fixed = {i: next(j for j, s in enumerate(row) if s.ticket == (i < 2))
             for i, row in enumerate(states[:-1])}
    model = _FixedStateSizer(problem, states, build_model, 'CLARABEL')(fixed)
    chain = model.template[2]
    original = chain.solve_via_data
    calls = []

    def fail_first(*args, **kwargs):
        """Inject one native failure and then execute the real equivalent retry."""
        calls.append(kwargs)
        if len(calls) == 1:
            raise cp.SolverError('numerical failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(chain, 'solve_via_data', fail_first)
    assert audit_execution(problem, solve_qp(model))['passed']
    assert len(calls) == 2
    assert all(not call['warm_start'] for call in calls)
    assert calls[0]['solver_opts'] == dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    assert calls[1]['solver_opts'] == dict(calls[0]['solver_opts'], equilibrate_enable=False)
    assert model.qp_diagnostics['attempts'] == 2
