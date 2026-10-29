"""Scarce-ticket regressions for coupled replacements and incumbent restarts."""
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution.v2 import (
    ExecutionProblem, ExecutionRules, RankedConfig, audit_execution, compile_states,
    wrapper_ranked_execution,
)
from optimalportfolios.execution.v2.ranked import _constrained_pair_gain, _coordinated_supports


def replacement_problem():
    """Two disjoint funded pairs compete for the same two-ticket capacity."""
    current = np.array([.1, .1, .1, .1, .6])
    target = np.array([.12, .08, .16, .04, .6])
    return ExecutionProblem(
        names=('small_buy', 'small_sell', 'large_buy', 'large_sell', 'cash'), cash=4,
        current=current, base=current, target=target,
        lower=np.minimum(current, target), upper=np.maximum(current, target),
        full_lower=np.zeros(5), full_upper=np.ones(5),
        groups=np.array([[1., 1., 0., 0., 0.], [0., 0., 1., 1., 0.]]),
        group_lower=np.array([.2, .2]), group_upper=np.array([.2, .2]),
        mandatory=np.zeros(5, bool), categories=('',)*5,
        eligible=np.ones(5, bool), frozen=np.zeros(5, bool),
        rules=ExecutionRules(ticket_cap=2, minimum_ticket=.01),
        covariance=np.diag([1., 1., 1., 1., 0.]),
    )


def test_two_for_two_replacement_is_reachable_without_spare_tickets():
    """A useful funded pair must be reachable when both slots are already occupied."""
    problem = replacement_problem()
    states = compile_states(problem)
    selected = {i: next(j for j, s in enumerate(states[i]) if s.ticket) for i in (0, 1)}
    desired = tuple((i, next(j for j, s in enumerate(states[i]) if s.ticket)) for i in (2, 3))
    point = problem.current.copy()
    point[:2] = problem.target[:2]
    proposals = list(_coordinated_supports(problem, states, selected, point, 2, set()))
    assert desired in {key for _, _, key in proposals}
    assert all(len(key) <= 2 for _, _, key in proposals)


def test_pair_replacement_recovers_from_inferior_funded_pair(monkeypatch):
    """The unchanged sizing QP, not a relaxed group row, must accept the better pair."""
    import optimalportfolios.execution.v2.ranked as module
    problem = replacement_problem()
    states = compile_states(problem)
    selected = {i: next(j for j, s in enumerate(states[i]) if s.ticket) for i in (0, 1)}
    point = problem.current.copy()
    point[:2] = problem.target[:2]
    monkeypatch.setattr(module, 'ranked_selection', lambda *args: (selected, point, [], 0.))
    result = wrapper_ranked_execution(problem)
    assert audit_execution(problem, result.weights)['passed']
    # Exact independent reference: the larger pair removes 2*.06**2 risk,
    # leaving only the two .02 residuals. Every cross-pair violates a group row.
    assert result.diagnostics['variance'] < .00080001
    assert result.diagnostics['audit']['tickets'] == 2


def scarce_problem():
    """Reproduce a seeded public synthetic case with six slots and tight group bands."""
    rng = np.random.default_rng(41038)
    n, cash = 39, 38
    current = np.full(n, .94/38)
    current[cash] = .06
    base = current.copy()
    base[:2] = 0.
    base[cash] += current[:2].sum()
    target = np.zeros(n)
    desired = np.exp(rng.normal(0., .85, 36))
    target[2:cash] = .98*desired/desired.sum()
    target[cash] = .02
    full_lower, full_upper = np.zeros(n), np.full(n, .35)
    full_upper[:2] = 0.
    full_lower[cash], full_upper[cash] = .005, .5
    lower, upper = np.minimum(base, target), np.minimum(np.maximum(base, target), full_upper)
    lower[cash], upper[cash] = full_lower[cash], full_upper[cash]
    groups = np.zeros((3, n))
    for i in range(n-1):
        groups[i % 3, i] = 1.
    exposure = groups@base
    factors = rng.normal(0., .08, (n, 4))
    factors[cash] = 0.
    covariance = factors@factors.T + np.diag(np.r_[np.full(n-1, .08**2), .001**2])
    mandatory = np.zeros(n, bool)
    mandatory[:2] = True
    return ExecutionProblem(
        names=tuple([f'Asset{i:03}' for i in range(n-1)]+['Cash']), cash=cash,
        current=current, base=base, target=target, lower=lower, upper=upper,
        full_lower=full_lower, full_upper=full_upper, groups=groups,
        group_lower=exposure-.005, group_upper=exposure+.005,
        mandatory=mandatory, categories=('model_exit', 'model_exit')+('',)*(n-2),
        eligible=np.ones(n, bool), frozen=np.zeros(n, bool),
        rules=ExecutionRules(minimum_ticket=.005, holding_floor=.0025,
                             ticket_cap=8, discretionary_allowance=6), covariance=covariance,
    )


def test_scarce_search_finds_the_missed_funded_combination():
    """A public development counterexample must improve without using an Exact incumbent."""
    problem = scarce_problem()
    result = wrapper_ranked_execution(problem)
    assert audit_execution(problem, result.weights)['passed']
    assert result.diagnostics['annual_te_bp'] < 143.
    assert result.diagnostics['audit']['discretionary_tickets'] <= 6
    assert sum(p['trials'] for p in result.diagnostics['coordinated_passes']) <= 24
    assert (sum(p['trials'] for p in result.diagnostics['scarce_passes'])
            + result.diagnostics['coordinated_seed_trials']) <= 24


def test_disabled_search_keeps_the_existing_initial_policy():
    """The existing exchange switch still disables all coordinated improvements."""
    problem = scarce_problem()
    config = RankedConfig(exchange=False, rescue_trials=0)
    first = wrapper_ranked_execution(problem, config)
    second = wrapper_ranked_execution(problem, replace(config, improvement_passes=0))
    np.testing.assert_array_equal(first.weights, second.weights)


@pytest.mark.parametrize('gradient', [[-.02, .01], [.04, -.01], [0., 0.]])
def test_group_aware_priority_matches_independent_convex_solver(gradient):
    """Check the analytic polygon priority against a separately built two-variable QP."""
    problem = replacement_problem()
    x = cp.Variable(2)
    hessian = np.array([[1., .3], [.3, 2.]])
    delta = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.], [-1., -1.]])
    weights = problem.current + delta @ x
    tol = problem.rules.audit_tolerance
    qp = cp.Problem(cp.Minimize(cp.quad_form(x, hessian) + 2*np.array(gradient)@x),
                    [x >= -.04, x <= .04,
                     weights[problem.cash] >= problem.lower[problem.cash],
                     weights[problem.cash] <= problem.upper[problem.cash],
                     problem.groups@weights >= problem.group_lower-tol,
                     problem.groups@weights <= problem.group_upper+tol])
    qp.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    gain = _constrained_pair_gain(problem, problem.current, 0, 1, gradient, (1., 2.), .3,
                                  (-.04, .04, -.04, .04))
    assert gain == pytest.approx(-qp.value, abs=1e-9)


@pytest.mark.parametrize('failure', ['exception', 'nonfinite', 'infeasible'])
def test_guidance_failure_cannot_replace_an_audited_incumbent(monkeypatch, failure):
    """Search guidance never bypasses fixed-state sizing or changes acceptance."""
    import optimalportfolios.execution.v2.ranked as module
    original = module.solve_qp

    def solve(model, solver):
        """Fail only the continuous guidance model, leaving real sizing intact."""
        if model.selectors:
            if failure == 'exception':
                raise cp.SolverError('guidance failure')
            return None if failure == 'infeasible' else np.full(5, np.nan)
        return original(model, solver)

    monkeypatch.setattr(module, 'solve_qp', solve)
    problem = replacement_problem()
    result = wrapper_ranked_execution(problem)
    assert audit_execution(problem, result.weights)['passed']
    assert result.diagnostics['variance'] <= .00080001
    assert result.diagnostics['qp_counts']['guidance'] == 1


def test_expanded_search_preserves_genuine_group_breach():
    """A real target breach cannot be accepted as a useful coordinated combination."""
    problem = replacement_problem()
    problem = replace(problem, group_lower=np.array([.2, .19]),
                      group_upper=np.array([.2, .19]))
    assert not audit_execution(problem, problem.target)['passed']
    result = wrapper_ranked_execution(problem)
    assert result.weights is None or audit_execution(problem, result.weights)['passed']


def test_multiple_states_of_one_name_never_use_two_slots():
    """A replacement pair must consist of distinct instruments, even in a wide domain."""
    problem = replacement_problem()
    problem = replace(problem, lower=np.zeros(5), upper=np.ones(5))
    states = compile_states(problem)
    selected = {i: next(j for j, s in enumerate(states[i]) if s.ticket) for i in (0, 1)}
    proposals = list(_coordinated_supports(problem, states, selected, problem.base, 2, set()))
    assert proposals
    assert all(len(dict(key)) == len(key) <= 2 for _, _, key in proposals)


def test_nonfinite_selector_is_not_a_guided_support(monkeypatch):
    """A native selector fault must not produce a rounded support from invalid priorities."""
    import optimalportfolios.execution.v2.ranked as module
    original = module.solve_qp

    def corrupt(model, solver):
        """Inject nonfinite selector data after an otherwise successful relaxation."""
        weights = original(model, solver)
        next(iter(model.selectors.values()))._value[:] = np.nan
        return weights

    monkeypatch.setattr(module, 'solve_qp', corrupt)
    problem = replacement_problem()
    support, status = module._guided_support(problem, compile_states(problem), 2, 'CLARABEL')
    assert support is None and status == 'nonfinite_occupancy'


@pytest.mark.parametrize('seed', [{0: 1, 2: 1}, {0: 1, 1: 0}])
def test_inferior_or_infeasible_guided_branch_retains_original_search(monkeypatch, seed):
    """A failed or inferior alternate start cannot erase the completed original incumbent."""
    import optimalportfolios.execution.v2.ranked as module
    problem = replacement_problem()
    monkeypatch.setattr(module, '_guided_support', lambda *args: (seed, 'test_seed'))
    result = wrapper_ranked_execution(problem, RankedConfig(rescue_trials=1))
    assert audit_execution(problem, result.weights)['passed']
    assert result.diagnostics['variance'] < .00080001
