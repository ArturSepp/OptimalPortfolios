"""Public seeded counterexamples requiring coordinated support changes and recourse."""

from dataclasses import replace

import numpy as np
import pytest

from optimalportfolios.execution.v2 import audit_execution, wrapper_ranked_execution
from optimalportfolios.execution.v2.tests.scarce_search_test import scarce_problem


def recourse_problem(seed, budget, width, narrow):
    """Generate the public synthetic family independently of the paper runner."""
    template = scarce_problem()
    rng = np.random.default_rng(seed + 36)
    target = np.zeros(39)
    desired = np.exp(rng.normal(0., .85, 36))
    target[2:38] = .98 * desired / desired.sum()
    target[38] = .02
    factors = rng.normal(0., .08, (39, 4))
    factors[38] = 0.
    covariance = factors @ factors.T + np.diag(np.r_[np.full(38, .08**2), .001**2])
    full_lower, full_upper = template.full_lower.copy(), template.full_upper.copy()
    if narrow:
        full_lower[38], full_upper[38] = template.base[38]-.002, template.base[38]+.002
    lower, upper = np.minimum(template.base, target), np.maximum(template.base, target)
    upper = np.minimum(upper, full_upper)
    lower[38], upper[38] = full_lower[38], full_upper[38]
    exposure = template.groups @ template.base
    return replace(template, target=target, covariance=covariance, lower=lower, upper=upper,
                   full_lower=full_lower, full_upper=full_upper,
                   group_lower=np.maximum(exposure-width, 0.), group_upper=exposure+width,
                   rules=replace(template.rules, ticket_cap=budget+2,
                                 discretionary_allowance=budget))


@pytest.mark.parametrize('seed,budget,width,narrow,reference', [
    (41002, 6, .005, False, 142.47615604066596),
    (51001, 3, .005, False, 284.84860700510745),
    (51002, 6, .005, True, 170.5322444062589),
    (51002, 6, .005, False, 160.35737019090524),
    (51002, 6, .08, True, 143.30967372746883),
])
def test_reported_support_misses_are_recovered(seed, budget, width, narrow, reference):
    """Recover archived audited objectives without passing any reference support to V2."""
    problem = recourse_problem(seed, budget, width, narrow)
    result = wrapper_ranked_execution(problem)
    assert audit_execution(problem, result.weights)['passed']
    assert result.diagnostics['annual_te_bp'] <= reference + 1e-4
    assert result.diagnostics['audit']['discretionary_tickets'] <= budget
    assert result.diagnostics['qp_counts']['restart_guidance'] <= budget
    assert len(result.diagnostics['restarts']) <= budget + 2
    assert all(len(r.get('passes', [])) <= 3 for r in result.diagnostics['restarts'])
    assert result.diagnostics['sizing_cache_hits'] > 0


def test_guidance_restrictions_never_reach_sizing_or_mutate_inputs(monkeypatch):
    """Only continuous guidance sees a pin; every executable QP sees the original arrays."""
    import optimalportfolios.execution.v2.ranked as module
    from optimalportfolios.execution.v2.tests.scarce_search_test import replacement_problem
    problem = replacement_problem()
    original = module.build_model
    calls = []
    before = {name: getattr(problem, name).copy() for name in
              ('current', 'base', 'target', 'lower', 'upper', 'groups', 'covariance')}

    def observe(candidate, **kwargs):
        """Check the domain used by each native graph, not merely final feasibility."""
        if kwargs.get('fixed_states') is not None:
            assert candidate is problem
        if kwargs.get('continuous') and candidate is not problem:
            assert np.all(candidate.lower >= problem.lower)
            assert np.all(candidate.upper <= problem.upper)
            assert candidate.rules == problem.rules
            np.testing.assert_array_equal(candidate.groups, problem.groups)
            np.testing.assert_array_equal(candidate.covariance, problem.covariance)
            calls.append(candidate)
        return original(candidate, **kwargs)

    monkeypatch.setattr(module, 'build_model', observe)
    result = wrapper_ranked_execution(problem)
    assert calls and audit_execution(problem, result.weights)['passed']
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(problem, name), value)


def test_restricted_guidance_failure_retains_the_completed_incumbent(monkeypatch):
    """Failure of every diversification model cannot erase earlier audited work."""
    import cvxpy as cp
    import optimalportfolios.execution.v2.ranked as module
    from optimalportfolios.execution.v2.tests.scarce_search_test import replacement_problem
    problem = replacement_problem()
    original = module._guided_support

    def fail_restricted(candidate, *args):
        """Leave the original guidance intact and fail only strictly restricted variants."""
        if candidate is not problem:
            raise cp.SolverError('injected restricted guidance failure')
        return original(candidate, *args)

    monkeypatch.setattr(module, '_guided_support', fail_restricted)
    result = wrapper_ranked_execution(problem)
    assert result.diagnostics['variance'] < .00080001
    assert audit_execution(problem, result.weights)['passed']
    assert any(r.get('skipped') == 'failed' for r in result.diagnostics['restarts'])
