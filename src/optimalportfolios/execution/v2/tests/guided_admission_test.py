"""Synthetic support-coverage regressions from the second broader-domain cohort."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from optimalportfolios.execution.v2 import (
    RankedConfig, audit_execution, compile_states, wrapper_ranked_execution,
)
from optimalportfolios.execution.v2.ranked import _support_possible
from optimalportfolios.execution.v2.tests.combination_search_test import broad_problem
from optimalportfolios.execution.v2.tests.scarce_search_test import replacement_problem


@pytest.mark.parametrize('optional,budget,seed,disjoint,reference', [
    (24, 2, 81002, False, 191.08197029063922),
    (72, 2, 81002, False, 148.1912744923786),
    (72, 4, 81001, True, 180.08882780926223),
    (72, 4, 81001, False, 180.94561279903095),
    (72, 4, 81002, False, 143.27348617889112),
])
def test_unvisited_combinations_recover_without_reference_starts(
    optional, budget, seed, disjoint, reference
):
    """Default V2 must recover the archived reference with unchanged rules."""
    problem = broad_problem(optional, budget, seed)
    if disjoint:
        problem = replace(problem, groups=problem.groups[:3],
                          group_lower=problem.group_lower[:3], group_upper=problem.group_upper[:3])
    result = wrapper_ranked_execution(problem, RankedConfig())
    assert audit_execution(problem, result.weights)['passed']
    assert result.diagnostics['annual_te_bp'] <= reference + 1e-4
    assert result.diagnostics['audit']['discretionary_tickets'] <= budget
    assert sum(r['trials'] for r in result.diagnostics['completion_passes']) <= 24 ** 2
    assert len(result.diagnostics['completion_passes']) <= 3
    assert result.diagnostics['completion_screened'] > 0


@pytest.mark.parametrize('sign', [-1., 1.])
def test_interval_screen_keeps_all_feasible_corners_with_signed_rows(sign):
    """Independent corner construction verifies the outer screen and input immutability."""
    problem = replacement_problem()
    problem = replace(problem, groups=sign * problem.groups,
                      group_lower=sign * problem.group_upper,
                      group_upper=sign * problem.group_lower)
    states = compile_states(problem)
    before = {name: getattr(problem, name).copy() for name in
              ('base', 'lower', 'upper', 'groups', 'group_lower', 'group_upper')}
    feasible = 0
    rejected = 0
    for choices in product(*(range(len(row)) for row in states[:-1])):
        support = {i: j for i, j in enumerate(choices) if states[i][j].ticket}
        possible = _support_possible(problem, states, support)
        rejected += not possible
        for endpoints in product((0, 1), repeat=4):
            weights = np.array([
                (states[i][j].lower, states[i][j].upper)[endpoints[i]]
                for i, j in enumerate(choices)
            ] + [0.])
            weights[-1] = problem.investment - weights[:-1].sum()
            if audit_execution(problem, weights)['passed']:
                feasible += 1
                assert possible
    assert feasible and rejected
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(problem, name), value)


def test_interval_screen_rejects_unfundable_support_without_group_rows():
    """An unbalanced minimum trade cannot fit the unchanged fixed cash interval."""
    problem = replacement_problem()
    problem = replace(problem, groups=np.empty((0, 5)), group_lower=np.array([]),
                      group_upper=np.array([]))
    states = compile_states(problem)
    buy = next(j for j, state in enumerate(states[0]) if state.ticket)
    assert not _support_possible(problem, states, {0: buy})
    assert _support_possible(problem, states, {})


def test_completion_retains_prior_incumbent_when_all_proposals_are_screened(monkeypatch):
    """The new phase cannot remove an earlier audited solution when it adds no trials."""
    import optimalportfolios.execution.v2.ranked as module
    problem = replacement_problem()
    original = wrapper_ranked_execution(problem)
    monkeypatch.setattr(module, '_support_possible', lambda *args: False)
    screened = wrapper_ranked_execution(problem)
    assert audit_execution(problem, screened.weights)['passed']
    np.testing.assert_allclose(screened.weights, original.weights, atol=1e-10, rtol=0.)
    assert all(r['trials'] == 0 for r in screened.diagnostics['completion_passes'])
