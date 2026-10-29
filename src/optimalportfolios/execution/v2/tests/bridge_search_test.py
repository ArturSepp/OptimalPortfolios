"""Regression for the third broader-cohort three-position replacement miss."""

from dataclasses import replace

import numpy as np

from optimalportfolios.execution.v2 import audit_execution, wrapper_ranked_execution
from optimalportfolios.execution.v2.tests.combination_search_test import broad_problem


def test_three_position_replacement_recovers_without_reference_start():
    """Default search must recover the archived reference through unchanged sizing."""
    problem = broad_problem(72, 4, 91001)
    before = {name: getattr(problem, name).copy() for name in
              ('current', 'base', 'target', 'lower', 'upper', 'groups',
               'group_lower', 'group_upper', 'covariance')}
    result = wrapper_ranked_execution(problem)
    assert audit_execution(problem, result.weights)['passed']
    assert result.diagnostics['annual_te_bp'] <= 171.44064703748734 + 1e-4
    assert result.diagnostics['audit']['discretionary_tickets'] <= 4
    bridges = result.diagnostics['bridge_restarts']
    assert 0 < len(bridges) <= 24
    assert all(len(r['passes']) <= 3 for r in bridges)
    assert bridges[-1]['improved']
    assert bridges[-1]['seed_value'] > bridges[-1]['incumbent_value']
    assert bridges[-1]['completed_value'] < bridges[-1]['incumbent_value']
    completion = result.diagnostics['bridge_completion_passes']
    assert 0 < len(completion) <= 3
    assert sum(r['trials'] for r in completion) <= 24 ** 2
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(problem, name), value)


def test_nonimproving_cached_branches_retain_known_feasible_optimum():
    """A known zero-risk target remains unchanged when alternative restarts fail."""
    from optimalportfolios.execution.v2.tests.scarce_search_test import replacement_problem
    problem = replacement_problem()
    problem = replace(problem, target=problem.base.copy())
    result = wrapper_ranked_execution(problem)
    assert audit_execution(problem, result.weights)['passed']
    np.testing.assert_array_equal(result.weights, problem.base)
    assert all(not r['improved'] for r in result.diagnostics['bridge_restarts'])
    assert not result.diagnostics['bridge_completion_passes']
