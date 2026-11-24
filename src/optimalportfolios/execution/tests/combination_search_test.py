"""Public counterexamples with tight overlapping groups and coordinated admissions."""

from dataclasses import replace

import numpy as np
import pytest

from optimalportfolios.execution import (
    ExecutionProblem, ExecutionRules, RankedConfig, audit_execution, wrapper_ranked_execution,
)


def broad_problem(optional, budget, seed):
    """Reconstruct the public synthetic domain without importing paper or private code."""
    n, cash = optional + 3, optional + 2
    rng = np.random.default_rng(np.random.SeedSequence([seed, optional]))
    relative = np.exp(rng.normal(0., .45, n - 1))
    current = np.r_[.006 + (.94 - .006 * (n - 1)) * relative / relative.sum(), .06]
    base = current.copy()
    base[cash] += base[:2].sum()
    base[:2] = 0.
    desired = current[2:cash] * np.exp(rng.normal(0., 1., optional))
    target = np.r_[0., 0., .98 * desired / desired.sum(), .02]
    sector = np.zeros((3, n))
    sector[np.arange(n - 1) % 3, np.arange(n - 1)] = 1.
    baskets = np.c_[rng.uniform(.05, .95, (3, n - 1)), np.zeros(3)]
    groups = np.vstack([sector, baskets])
    # Known synthetic PSD construction, not a replacement covariance estimator.
    loadings = np.c_[rng.uniform(.04, .12, n), .10 * sector.T,
                     rng.normal(0., .04, (n, 2))]
    loadings[cash] = 0.
    residual = np.r_[rng.uniform(.04, .09, n - 1)**2, .001**2]
    covariance = loadings @ loadings.T + np.diag(residual)
    full_lower, full_upper = np.zeros(n), np.full(n, .35)
    full_upper[:2] = 0.
    full_lower[cash], full_upper[cash] = base[cash] - .0005, base[cash] + .0005
    lower, upper = np.minimum(base, target), np.minimum(np.maximum(base, target), full_upper)
    lower[cash], upper[cash] = full_lower[cash], full_upper[cash]
    exposure = groups @ base
    return ExecutionProblem(
        names=tuple([f'Asset{i:03}' for i in range(n - 1)] + ['Cash']), cash=cash,
        current=current, base=base, target=target, lower=lower, upper=upper,
        full_lower=full_lower, full_upper=full_upper, groups=groups,
        group_lower=np.maximum(0., exposure - .001), group_upper=exposure + .001,
        mandatory=np.arange(n) < 2, categories=('model_exit', 'model_exit') + ('',) * (n - 2),
        eligible=np.ones(n, bool), frozen=np.zeros(n, bool), covariance=covariance,
        rules=ExecutionRules(minimum_ticket=.005, holding_floor=.0025,
                             ticket_cap=budget + 2, discretionary_allowance=budget),
    )


@pytest.mark.parametrize('optional,budget,seed,reference', [
    (24, 2, 71001, 412.7798783083),
    (24, 4, 71002, 270.4664230955),
    (72, 2, 71002, 129.7731737213),
    (72, 4, 71001, 81.96769705464),
])
@pytest.mark.parametrize('passes', [1, 3])
def test_shared_missed_combinations_recover_without_reference_starts(
    optional, budget, seed, reference, passes
):
    """Both fixed search variants must find the archived reference quality unaided."""
    problem = broad_problem(optional, budget, seed)
    result = wrapper_ranked_execution(problem, replace(RankedConfig(), improvement_passes=passes))
    assert audit_execution(problem, result.weights)['passed']
    assert result.diagnostics['annual_te_bp'] <= reference + 1e-4
    assert result.diagnostics['audit']['discretionary_tickets'] <= budget
    assert result.diagnostics['qp_counts']['conditional_guidance'] <= 24
    assert len(result.diagnostics['conditional_restarts']) <= 25
    assert all(len(r['passes']) <= passes for r in result.diagnostics['conditional_restarts'])


@pytest.mark.parametrize('included', [True, False])
def test_conditional_state_rounding_respects_the_decision_without_mutating_domain(included):
    """A forced or excluded state affects guidance only, with immutable input arrays."""
    from optimalportfolios.execution import compile_states
    from optimalportfolios.execution.ranked_execution import _guided_support
    from optimalportfolios.execution.tests.scarce_search_test import replacement_problem
    problem = replacement_problem()
    before = {name: getattr(problem, name).copy() for name in
              ('lower', 'upper', 'groups', 'group_lower', 'group_upper', 'covariance')}
    states = compile_states(problem)
    j = next(j for j, state in enumerate(states[2]) if state.ticket)
    seed, status = _guided_support(problem, states, 2, 'CLARABEL', (2, j, included))
    assert status == 'optimal' and seed is not None
    assert ((2, j) in seed.items()) == included
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(problem, name), value)


@pytest.mark.parametrize('failure', ['exception', 'no_result'])
def test_failed_conditional_guidance_preserves_prior_audited_search(monkeypatch, failure):
    """Failure of the new phase cannot invalidate or discard completed search work."""
    import cvxpy as cp
    import optimalportfolios.execution.ranked_execution as module
    from optimalportfolios.execution.tests.scarce_search_test import replacement_problem
    original = module._guided_support

    def fail_only_conditional(problem, states, limit, solver, decision=None):
        """Leave earlier search unchanged and fail only the additional state decisions."""
        if decision is not None:
            if failure == 'exception':
                raise cp.SolverError('conditional guidance failure')
            return None, 'injected_no_result'
        return original(problem, states, limit, solver)

    monkeypatch.setattr(module, '_guided_support', fail_only_conditional)
    problem = replacement_problem()
    result = wrapper_ranked_execution(problem)
    assert audit_execution(problem, result.weights)['passed']
    assert result.diagnostics['variance'] < .00080001
    assert result.diagnostics['conditional_guidance']
    assert all(r['skipped'] == 'failed' for r in result.diagnostics['conditional_guidance'])
    assert not result.diagnostics['conditional_restarts']
