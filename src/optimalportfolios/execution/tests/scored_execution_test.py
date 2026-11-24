"""Independent analytic references for costs, scores, state identity and deadlines."""

from dataclasses import replace
from itertools import product

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution import (
    ExecutionProblem, ExecutionRules, RankedConfig, audit_execution, compile_states,
    compute_perspective_lower_bound, wrapper_exact_execution, wrapper_ranked_execution,
)
from optimalportfolios.execution.domain import (
    TradeState, _cost_states, objective, score_terms, state_side, variance,
)
from optimalportfolios.execution.model import SCALE, build_model, solve_qp
from optimalportfolios.execution._sizing import _FixedStateSizer
from optimalportfolios.execution.ranked_execution import _state_adjustment, state_gain
from optimalportfolios.execution.tests.formulation_test import example


def scalar_problem():
    """Supply a riskless-cash example with an interior single-name target."""
    current = np.array([.2, .8])
    return ExecutionProblem(
        names=("Asset", "Cash"), cash=1, current=current, base=current,
        target=np.array([.4, .6]), lower=np.zeros(2), upper=np.ones(2),
        full_lower=np.zeros(2), full_upper=np.ones(2),
        groups=np.zeros((0, 2)), group_lower=np.zeros(0), group_upper=np.zeros(0),
        mandatory=np.zeros(2, bool), categories=("", ""), eligible=np.ones(2, bool),
        frozen=np.zeros(2, bool), rules=ExecutionRules(holding_floor=0.),
        covariance=np.diag([.01, 0.]),
    )


@pytest.mark.parametrize("cost,expected", [(0., .4), (.001, .3), (.002, .2), (.003, .2)])
def test_scalar_cost_threshold_and_gain(cost, expected):
    """Recover the closed-form soft threshold by joint sizing and Ranked selection."""
    p = replace(scalar_problem(), trade_cost=np.array([cost, 0.]))
    w = solve_qp(build_model(p, continuous=True))
    np.testing.assert_allclose(w, [expected, 1-expected], atol=1e-6, rtol=0.)
    gain = max(.002-cost, 0.) ** 2 / .01
    assert abs(objective(p, p.base)-objective(p, w)-gain) < 1e-10
    ranked = wrapper_ranked_execution(p)
    np.testing.assert_allclose(ranked.weights, w, atol=1e-6, rtol=0.)
    score, paid = score_terms(p, ranked.weights)
    assert ranked.diagnostics['scored_objective'] == variance(p, ranked.weights)-2*score+2*paid


def test_doubling_and_negative_objective_bound():
    """The budget-only scored minimiser is 2m-b, with a valid negative lower bound."""
    p = replace(scalar_problem(), weight_score=np.array([.001, 0.]))
    w = solve_qp(build_model(p, continuous=True))
    np.testing.assert_allclose(w, [.5, .5], atol=1e-8, rtol=0.)
    value = objective(p, w)
    assert value < 0
    for result in compute_perspective_lower_bound(p).values():
        assert result['lower_bound'] <= value+1e-10
        assert abs(result['primal']-value) < 1e-9
    pinned = replace(p, lower=w, upper=w, full_lower=w, full_upper=w)
    for result in compute_perspective_lower_bound(pinned).values():
        assert result['lower_bound'] <= value


@pytest.mark.parametrize('field,value', [
    ('trade_cost', [-.1, 0.]), ('trade_cost', [.1, .1]),
    ('trade_cost', [float('nan'), 0.]), ('weight_score', [float('inf'), 0.]),
    ('weight_score', [0.]),
])
def test_invalid_score_vectors(field, value):
    """Reject invalid units or shapes at the supplied-input boundary."""
    with pytest.raises(ValueError):
        replace(scalar_problem(), **{field:np.array(value)})


def test_absent_and_explicit_zero_scores_preserve_graph_and_weights():
    """Zero vectors retain the exact unscored compiler and search path."""
    p = example()
    q = replace(p, weight_score=np.zeros(5), trade_cost=np.zeros(5))
    assert q.trade_cost is None and q.weight_score is None
    assert compile_states(p) == compile_states(q)
    np.testing.assert_array_equal(wrapper_ranked_execution(p).weights,
                                  wrapper_ranked_execution(q).weights)


def test_state_splitting_and_absolute_cost_identity():
    """Test sign splitting independently without relaxing compulsory input guards."""
    p = scalar_problem()
    crossing = TradeState(.1, .4, True)
    states = _cost_states([crossing], p.current[0])
    assert len(states) == 2 and all(s.ticket for s in states)
    rng = np.random.default_rng(61117)
    for state in states:
        for weight in rng.uniform(state.lower, state.upper, 25):
            assert abs(weight-p.current[0]) == pytest.approx(
                state_side(state, p.current[0])*(weight-p.current[0]))
    with pytest.raises(ValueError, match='crosses'):
        state_side(crossing, p.current[0])
    with pytest.raises(ValueError, match='Mandatory interval'):
        replace(p, base=np.array([.25, .75]), mandatory=np.array([True, False]),
                categories=('instruction_min_buy', ''),
                rules=replace(p.rules, exemption_mask=np.array([True, False])))


def test_scored_reversal_and_removal_priorities_include_constant():
    """Compare the whole scored difference to a reversed or removed trade."""
    p = replace(example(coupled=False), trade_cost=np.array([.002, .003, 0., 0., 0.]),
                weight_score=np.array([.001, -.001, 0., 0., .0005]))
    point = p.base.copy()
    point[1] -= .01
    point[-1] += .01
    active = p.matvec(point-p.target)
    direction = np.array([0., 1., 0., 0., -1.])
    curvature = float(direction @ p.matvec(direction))
    for state in compile_states(p)[1]:
        slope, constant = _state_adjustment(p, point, 1, state)
        gain, step = state_gain(active[1]-active[-1]+slope, curvature,
                                state.lower-point[1], state.upper-point[1])
        assert objective(p, point)-objective(p, point+step*direction) == pytest.approx(
            gain-constant, abs=1e-14)


def test_cached_scored_sizing_matches_independent_absolute_cost_qp():
    """A separately written absolute-cost graph checks each admissible support."""
    p = replace(example(cap=2, coupled=False),
                trade_cost=np.array([.0001, .0003, .0002, .0001, 0.]),
                weight_score=np.array([.0001, -.0002, .0003, 0., .0001]))
    states = compile_states(p)
    sizer = _FixedStateSizer(p, states, build_model, 'CLARABEL')
    for selected in product(*(range(len(row)) for row in states[:-1])):
        if sum(states[i][j].ticket for i, j in enumerate(selected)) > 2:
            continue
        fixed = dict(enumerate(selected))
        w = solve_qp(sizer(fixed))
        x = cp.Variable(5)
        constraints = [cp.sum(x)==p.investment, x[-1]>=p.lower[-1], x[-1]<=p.upper[-1]]
        constraints += [x[i]>=states[i][j].lower for i,j in fixed.items()]
        constraints += [x[i]<=states[i][j].upper for i,j in fixed.items()]
        reference = cp.Problem(cp.Minimize(SCALE*(cp.quad_form(x-p.target,p.covariance)
            -2*p.weight_score@(x-p.current)+2*p.trade_cost@cp.abs(x-p.current))),constraints)
        reference.solve(solver='CLARABEL',tol_feas=1e-10,tol_gap_abs=1e-10,tol_gap_rel=1e-10)
        assert audit_execution(p,w)['passed']
        assert abs(objective(p,w)-reference.value/SCALE)<1e-10
    assert sizer.diagnostics['reuses']>0


@pytest.mark.parametrize('seconds',[0., -1., float('inf'), float('nan'), True])
def test_invalid_deadline(seconds):
    """Require an explicit positive finite optional time allowance."""
    with pytest.raises(ValueError,match='time limit'):
        RankedConfig(time_limit_s=seconds)


def test_deadline_retains_base_before_new_solve():
    """An exhausted initial allowance returns the audited prepared book."""
    p = example()
    result = wrapper_ranked_execution(p,RankedConfig(time_limit_s=1e-12))
    assert result.status=='time_limit' and result.diagnostics['time_limit_reached']
    np.testing.assert_array_equal(result.weights,p.base)
    assert audit_execution(p,result.weights)['passed']


def test_deadline_returns_completed_sizing_and_restores_context(monkeypatch):
    """Keep a new audited incumbent when the solve itself consumes the remaining budget."""
    from optimalportfolios.execution import ranked_execution as ranked
    from optimalportfolios.execution import _deadline as deadline
    p = scalar_problem()
    clock = [0.]
    monkeypatch.setattr(deadline,'perf_counter',lambda:clock[0])
    solve = ranked.solve_qp

    def finish_after_deadline(*args):
        """Advance only the method clock after an ordinary sizing solve returns."""
        w=solve(*args)
        clock[0]=2.
        return w

    monkeypatch.setattr(ranked,'solve_qp',finish_after_deadline)
    result=wrapper_ranked_execution(p,RankedConfig(time_limit_s=1.))
    assert result.status=='time_limit'
    assert objective(p,result.weights)<objective(p,p.base)-1e-6
    assert audit_execution(p,result.weights)['passed']
    assert deadline._active.get() is None


def six_name_problem(cost):
    """Supply independent assets whose scalar gains determine the best two tickets."""
    current = np.array([.08, .07, .06, .05, .04, .70])
    target = np.array([.12, .10, .08, .04, .06, .60])
    return ExecutionProblem(
        names=('a', 'b', 'c', 'd', 'e', 'cash'), cash=5,
        current=current, base=current, target=target,
        lower=np.r_[np.minimum(current, target)[:-1], 0.],
        upper=np.r_[np.maximum(current, target)[:-1], 1.],
        full_lower=np.zeros(6), full_upper=np.ones(6),
        groups=np.zeros((0, 6)), group_lower=np.zeros(0), group_upper=np.zeros(0),
        mandatory=np.zeros(6, bool), categories=('',)*6,
        eligible=np.ones(6, bool), frozen=np.zeros(6, bool),
        rules=ExecutionRules(ticket_cap=2, holding_floor=0.),
        covariance=np.diag([.04, .03, .02, .01, .06, 0.]),
        trade_cost=np.array([.001, .0004, .0001, .0002, .0002, 0.])*cost,
    )


def enumerate_reference(problem):
    """Enumerate supports using an independent absolute-cost convex graph."""
    states = compile_states(problem)
    best, weights = np.inf, None
    for selected in product(*(range(len(row)) for row in states[:-1])):
        if sum(states[i][j].ticket for i, j in enumerate(selected)) > 2:
            continue
        x = cp.Variable(len(problem.names))
        rows = [cp.sum(x) == 1., x[-1] >= 0., x[-1] <= 1.]
        rows += [x[i] >= states[i][j].lower for i, j in enumerate(selected)]
        rows += [x[i] <= states[i][j].upper for i, j in enumerate(selected)]
        value = cp.quad_form(x-problem.target, problem.covariance)
        if problem.trade_cost is not None:
            value += 2*problem.trade_cost @ cp.abs(x-problem.current)
        graph = cp.Problem(cp.Minimize(SCALE*value), rows)
        graph.solve(solver='CLARABEL', tol_feas=1e-10, tol_gap_abs=1e-10, tol_gap_rel=1e-10)
        if graph.value/SCALE < best:
            best, weights = graph.value/SCALE, x.value.copy()
    return best, weights


@pytest.mark.parametrize('cost', [0., 1.])
def test_separable_ranked_support_and_enumeration(cost):
    """Greedy support matches independent exhaustive sizing under separability."""
    from optimalportfolios.execution.ranked_execution import ranked_selection

    p = six_name_problem(cost)
    reference, weights = enumerate_reference(p)
    support, _, _, _ = ranked_selection(p, compile_states(p), 2)
    expected = set(np.flatnonzero(np.abs(weights[:-1]-p.current[:-1]) > 1e-5))
    assert set(support) == expected
    solved = wrapper_ranked_execution(p)
    assert audit_execution(p, solved.weights)['passed']
    assert objective(p, solved.weights) == pytest.approx(reference, abs=1e-10)


@pytest.mark.skipif(
    'MOSEK' not in cp.installed_solvers(), reason='Optional native adapter unavailable'
)
@pytest.mark.parametrize('cost', [0., 1.])
def test_six_name_native_exact_matches_independent_enumeration(cost):
    """The integer affine-cost formulation recovers the enumerated six-name optimum."""
    p = six_name_problem(cost)
    reference, _ = enumerate_reference(p)
    solved = wrapper_exact_execution(p, 'CORRIDOR', threads=1, time_limit_s=10.)
    if 'license' in solved.diagnostics.get('error', '').lower():
        pytest.skip('Optional native licence unavailable')
    assert audit_execution(p, solved.weights)['passed']
    assert solved.diagnostics['native_status'] == 'solsta.integer_optimal'
    assert objective(p, solved.weights) == pytest.approx(reference, abs=1e-10)
    assert solved.diagnostics['native_bound_check']['consistent_with_delivery']
    assert solved.diagnostics['native_gap_absolute'] >= 0.


@pytest.mark.parametrize('cost,step', [
    (0., .1), (.0009, .1), (.0014, .05), (.00185, .005),
    (.00187, .005), (.001875, 0.), (.00188, 0.),
])
def test_correlated_two_asset_cost_example(cost, step):
    """Check target clamp, minimum-size clamp, indifference and closed trade."""
    current = np.array([.2, .2, .6])
    p = replace(scalar_problem(), names=('a', 'b', 'cash'), cash=2,
        current=current, base=current, target=np.array([.3, .3, .4]),
        lower=np.array([.2, .2, 0.]), upper=np.array([.3, .2, 1.]),
        full_lower=np.array([0., .2, 0.]), full_upper=np.array([1., .2, 1.]),
        groups=np.zeros((0, 3)), mandatory=np.zeros(3, bool), categories=('',)*3,
        eligible=np.ones(3, bool), frozen=np.zeros(3, bool),
        rules=ExecutionRules(minimum_ticket=.005, holding_floor=0., ticket_cap=1),
        covariance=.01*np.array([[1., .9, 0.], [.9, 1., 0.], [0., 0., 0.]]),
        trade_cost=np.array([cost, 0., 0.]))
    solved = wrapper_ranked_execution(p)
    assert solved.weights[0]-current[0] == pytest.approx(step, abs=2e-6)
    if cost == 0:
        assert np.sqrt(variance(p, solved.weights))*1e4 == pytest.approx(100., abs=1e-3)
        full = solve_qp(build_model(p, 'FULL', continuous=True))
        assert np.sqrt(variance(p, full))*1e4 == pytest.approx(43.5889894, abs=1e-4)


def test_continuous_scalar_gain_without_minimum_size():
    """The analytical interval permits every magnitude including a zero movement."""
    for cost in (0., .001, .002, .003):
        gain, step = state_gain(-.002+cost, .01, 0., 1.)
        assert step == pytest.approx(max(.002-cost, 0.)/.01)
        assert gain == pytest.approx(max(.002-cost, 0.)**2/.01)


def test_deadline_interrupts_bound_and_retry(monkeypatch):
    """An expired budget starts neither a guidance certificate nor a numerical retry."""
    from optimalportfolios.execution import _deadline as deadline

    p = scalar_problem()
    clock = [0.]
    monkeypatch.setattr(deadline, 'perf_counter', lambda: clock[0])
    with deadline._SearchDeadline(1., 1e-12):
        clock[0] = 2.
        with pytest.raises(deadline._DeadlineReached):
            compute_perspective_lower_bound(p)
    clock[0] = 0.

    def fail_after_budget(self, **kwargs):
        """Spend the allowance during the first native sizing attempt."""
        clock[0] = 2.
        raise cp.SolverError('injected failure')

    model = build_model(p, continuous=True)
    monkeypatch.setattr(cp.Problem, 'solve', fail_after_budget)
    with deadline._SearchDeadline(1., 1e-12):
        with pytest.raises(deadline._DeadlineReached):
            solve_qp(model)
    assert deadline._active.get() is None


def test_finite_unexpired_deadline_matches_unlimited():
    """A completed finite search keeps the same result as its unlimited counterpart."""
    p = scalar_problem()
    np.testing.assert_array_equal(wrapper_ranked_execution(p).weights,
        wrapper_ranked_execution(p, RankedConfig(time_limit_s=60.)).weights)


@pytest.mark.skipif(
    'MOSEK' not in cp.installed_solvers(), reason='Optional native adapter unavailable'
)
def test_native_negative_scored_objective_and_original_gap_units():
    """Native signed objectives preserve their constant, bound and stopping-gap units."""
    p = replace(scalar_problem(), weight_score=np.array([.001, 0.]))
    solved = wrapper_exact_execution(p, 'CORRIDOR', time_limit_s=10., threads=1)
    if 'license' in solved.diagnostics.get('error', '').lower():
        pytest.skip('Optional native licence unavailable')
    assert audit_execution(p, solved.weights)['passed']
    # The continuous doubling test checks weights to 1e-8. Integer stopping is
    # objective-based; check its analytic optimum without a new weight tolerance.
    assert objective(p, solved.weights) == pytest.approx(-.0005, abs=1e-10)
    assert solved.diagnostics['native_bound_check']['consistent_with_delivery']
    assert solved.diagnostics['native_objective_origin'] == 'scored_objective'
    assert solved.diagnostics['native_gap_absolute'] >= 0.
