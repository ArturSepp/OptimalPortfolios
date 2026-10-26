"""Independently known repairs survive a contradictory native discrete tie-break."""
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution.v2 import audit_execution
from optimalportfolios.execution.v2 import preparation as prep
from optimalportfolios.execution.v2.tests.preparation_stability_test import tied_problem


def two_group_repairs(full=False, cap=10):
    """Each disjoint group needs one 1% top-up; identities A and C are canonical."""
    p = tied_problem()
    hi = np.array([.06, .06, .06, .06, .2])
    corridor_hi = p.base.copy() if full else hi
    corridor_hi[-1] = .2
    return replace(p, names=('b', 'a', 'd', 'c', 'cash'), upper=corridor_hi,
                   full_upper=hi, groups=np.array([[1., 1., 0., 0., 0.],
                                                  [0., 0., 1., 1., 0.]]),
                   group_lower=np.array([.09, .09]), group_upper=np.array([.12, .12]),
                   rules=replace(p.rules, ticket_cap=cap))


def inject_identity_contradiction(monkeypatch, *, recovery_fault=None):
    """Simulate the observed infeasible tie-break after genuine primary optimisations."""
    original = prep._solve
    calls = []

    def solve(objective, rows, solver, deadline, log, label):
        """Fail the normal identity stage; let recovery solve the real unchanged graph."""
        calls.append((solver, deadline, label))
        if solver == 'SCIPY' and ':repair_identity:' in label:
            log.append(dict(stage=label, status=cp.INFEASIBLE, solver=solver))
            return False
        if solver == 'SCIPY_REPAIR' and recovery_fault == 'timeout':
            log.append(dict(stage=label, status='time_limit', solver='SCIPY'))
            return False
        success = original(objective, rows, solver, deadline, log, label)
        if solver == 'SCIPY_REPAIR' and recovery_fault == 'unproven':
            log[-1]['status'] = cp.OPTIMAL_INACCURATE
        return success

    monkeypatch.setattr(prep, '_solve', solve)
    return calls


@pytest.mark.parametrize('full', [False, True])
def test_discrete_retry_preserves_known_optimum_and_all_hard_rows(monkeypatch, full):
    """A numerical tie-break contradiction must not lose a hand-verifiable repair."""
    p = two_group_repairs(full)
    calls = inject_identity_contradiction(monkeypatch)
    result, record = prep.prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert {r['name'] for r in record['repairs']} == {'a', 'c'}
    expected = p.base.copy()
    expected[[1, 3]] = .05-p.rules.audit_tolerance
    expected[-1] = p.investment-expected[:-1].sum()
    np.testing.assert_allclose(result.base, expected, atol=2e-10, rtol=0)
    np.testing.assert_array_equal(result.group_lower, p.group_lower)
    np.testing.assert_array_equal(result.group_upper, p.group_upper)
    np.testing.assert_array_equal(result.current, p.current)
    assert result.rules.audit_tolerance == p.rules.audit_tolerance
    assert result.rules.zero_tolerance == p.rules.zero_tolerance
    assert result.rules.minimum_ticket == p.rules.minimum_ticket
    assert result.rules.ticket_cap == p.rules.ticket_cap
    assert audit_execution(result, result.base)['passed']
    retry = next(r for r in record['stages'] if r['status'] == 'discrete_retry')
    for name, cap in retry['primary_caps'].items():
        primary = next(r for r in record['stages'] if r['stage'] == f'{record["mode"]}:{name}')
        assert cap == primary['objective']+prep._OBJECTIVE_GUARD_BP
    assert set(retry['primary_caps']) == ({'excursion', 'turnover'} if full else {'turnover'})
    assert len({deadline for _, deadline, _ in calls}) == 1


@pytest.mark.parametrize('fault', ['timeout', 'unproven'])
def test_retry_without_a_proved_count_cannot_supply_an_incumbent(monkeypatch, fault):
    """Recovery remains bounded and cannot substitute an unproved integer optimum."""
    calls = inject_identity_contradiction(monkeypatch, recovery_fault=fault)
    result, record = prep.prepare_common_execution(two_group_repairs(), solver='SCIPY')
    assert result is None, record
    assert any(solver == 'SCIPY_REPAIR' for solver, _, _ in calls)
    assert len({deadline for _, deadline, _ in calls}) == 1


def test_recovered_preparation_still_rejects_a_genuine_total_ticket_breach(monkeypatch):
    """Re-proving the repair support never expands a fixed one-ticket policy cap."""
    inject_identity_contradiction(monkeypatch)
    result, record = prep.prepare_common_execution(two_group_repairs(cap=1), solver='SCIPY')
    assert result is None
    assert record['status'] == 'prepared_base_rejected'
    assert 'total_ticket_cap' in record['prepared_audit']['violations']


def test_strict_recovery_keeps_the_existing_native_deadline_and_tightens_precision(monkeypatch):
    """The independent numerical retry disables presolve without widening tolerances."""
    captured = {}

    def solve(self, **kwargs):
        """Inspect native options without depending on a commercial solver licence."""
        captured.update(kwargs)
        self._status, self._value = cp.OPTIMAL, 0.

    monkeypatch.setattr(cp.Problem, 'solve', solve)
    assert prep._solve(cp.Constant(0.), [], 'SCIPY_REPAIR',
                       prep.perf_counter()+3., [], 'retry')
    assert captured['solver'] == 'SCIPY'
    options = captured['scipy_options']
    assert options['presolve'] is False
    assert 0 < options['time_limit'] <= 3.
    assert options['mip_rel_gap'] == 0.
    assert options['mip_feasibility_tolerance'] == 1e-10
    assert options['primal_feasibility_tolerance'] == 1e-10
    assert options['dual_feasibility_tolerance'] == 1e-10
