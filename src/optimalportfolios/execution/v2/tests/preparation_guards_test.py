"""Preparation deadlines, numerical faults and genuine holding-domain conflicts."""
from dataclasses import replace
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution.v2 import preparation as prep
from optimalportfolios.execution.v2 import audit_execution
from optimalportfolios.execution.v2.tests.formulation_test import example


def floor_repair_problem():
    """A held A needs a one-percentage-point top-up, funded entirely by cash."""
    p = example(coupled=False)
    floor = np.array([.05, .005, .005, .005, 0.])
    return replace(p, rules=replace(p.rules, holding_floor=floor))


def test_zero_holding_branch_is_selected_when_exit_is_the_smallest_repair():
    """A 20 bp holding can be exited for less turnover than its allowed 50 bp top-up."""
    p = example(coupled=False)
    current = np.array([.002, 0., 0., 0., .198])
    lo, hi = np.zeros(5), np.array([.1, 0., 0., 0., .2])
    p = replace(p, current=current, base=current, lower=lo, upper=hi,
                full_lower=lo, full_upper=hi)
    prepared, record = prep.prepare_common_execution(p, solver='SCIPY')
    assert prepared is not None, record
    assert prepared.base[0] == 0.
    assert prepared.mandatory_count == 1
    assert audit_execution(prepared, prepared.base)['passed']


def test_unavailable_holding_state_is_not_replaced_with_an_illegal_exit():
    """A positive hard upper bound below its floor can make a pinned holding impossible."""
    p = example(coupled=False)
    lo, hi = p.lower.copy(), p.upper.copy()
    lo[0] = hi[0] = .002
    p = replace(p, lower=lo, upper=hi, full_lower=lo, full_upper=hi)
    result, record = prep.prepare_common_execution(p, solver='SCIPY')
    assert result is None
    assert [s['status'] for s in record['stages']] == ['empty_domain', 'empty_domain']


@pytest.mark.parametrize('offset, expected', [(5e-6, True), (1., False)])
def test_native_endpoint_fault_requires_a_new_valid_solve(monkeypatch, offset, expected):
    """A tiny floor undershoot is re-solved; a large endpoint error remains rejected."""
    original = prep._solve
    injections = []

    def faulty_endpoint(objective, rows, solver, deadline, log, label):
        """Inject a faulty native free coordinate after an otherwise real solve."""
        success = original(objective, rows, solver, deadline, log, label)
        if success and label.endswith('fixed_sizing:0'):
            variable = objective.variables()[0]
            value = variable.value.copy()
            value[0] = -offset
            variable.save_value(value)
            injections.append(label)
        return success

    monkeypatch.setattr(prep, '_solve', faulty_endpoint)
    result, record = prep.prepare_common_execution(floor_repair_problem(), solver='SCIPY')
    assert injections
    assert (result is not None) == expected, record
    if expected:
        assert result.base[0] == .05
        assert any(row['status'] == 'endpoint_resolve' for row in record['stages'])
        assert audit_execution(result, result.base)['passed']


def test_nonintegral_native_support_is_rejected(monkeypatch):
    """A nominally successful native result cannot silently round a fractional repair flag."""
    original = prep._solve

    def fractional_support(objective, rows, solver, deadline, log, label):
        """Preserve native optimisation while corrupting its claimed integer result."""
        success = original(objective, rows, solver, deadline, log, label)
        if success and label.endswith('repair_count'):
            variable = objective.variables()[0]
            variable.save_value(variable.value+.1)
        return success

    monkeypatch.setattr(prep, '_solve', fractional_support)
    result, record = prep.prepare_common_execution(floor_repair_problem(), solver='SCIPY')
    assert result is None
    assert any(row['status'] == 'invalid_integrality' for row in record['stages'])


@pytest.mark.parametrize('clock, status', [([2.], 'time_limit'),
                                         ([0., 2.], 'construction_time_limit')])
def test_deadline_includes_construction(monkeypatch, clock, status):
    """No native solve starts after construction has consumed the shared deadline."""
    values = iter(clock)
    monkeypatch.setattr(prep, 'perf_counter', lambda: next(values))
    log = []
    assert not prep._solve(cp.Constant(0.), [], 'SCIPY', 1., log, 'deadline')
    assert log == [dict(stage='deadline', status=status)]


def test_native_failure_retains_its_diagnosis():
    """An unavailable backend is a recorded failure, never a feasible incumbent."""
    log = []
    assert not prep._solve(cp.Constant(0.), [], 'NOT_AN_INSTALLED_SOLVER',
                           prep.perf_counter()+10., log, 'backend')
    assert log[-1]['status'] == 'solver_error'


def test_mosek_receives_remaining_budget_and_reproducible_controls(monkeypatch):
    """The native MIP cannot acquire a fresh full allowance at each lexicographic stage."""
    received = {}

    def capture(self, **kwargs):
        """Observe the native boundary without requiring a licence in base CI."""
        received.update(kwargs)
        self._status, self._value = cp.OPTIMAL, 0.

    monkeypatch.setattr(cp.Problem, 'solve', capture)
    assert prep._solve(cp.Constant(0.), [], 'MOSEK', prep.perf_counter()+3., [], 'native')
    options = received['mosek_params']
    assert 0 < options['MSK_DPAR_MIO_MAX_TIME'] <= 3.
    assert options['MSK_IPAR_NUM_THREADS'] == 1
    assert options['MSK_IPAR_MIO_SEED'] == 0
    assert options['MSK_DPAR_MIO_TOL_FEAS'] == 1e-9


def test_declared_mandatory_movement_and_numerical_pins_are_distinct():
    """A sub-zero-threshold optional adjustment gets no extra ticket or exemption."""
    p = example()
    mandatory = np.array([True, False, False, False, False])
    current = p.current.copy()
    current[0] -= .01
    current[4] += .01
    full_lo, full_hi = p.full_lower.copy(), p.full_upper.copy()
    full_lo[0], full_hi[0] = p.lower[0], p.upper[0]
    p = replace(p, current=current, mandatory=mandatory,
                categories=('product_min_buy', '', '', '', ''),
                full_lower=full_lo, full_upper=full_hi,
                rules=replace(p.rules, exemption_mask=mandatory))
    w = p.base.copy()
    w[0] += .001
    w[1] += 1e-10
    w[4] -= .001+1e-10
    result, declaration = prep._declare(p, w, 'CORRIDOR')
    assert result.base[0] == w[0]
    assert declaration['repairs'] == []
    assert len(declaration['numerical_pins']) == 1
    assert result.mandatory_count == p.mandatory_count
    assert not result.rules.exemption_mask[1]


def test_failed_fixed_state_sizing_cannot_deliver_the_mip_endpoint(monkeypatch):
    """Successful support selection still needs its independently sized endpoint."""
    original = prep._solve

    def fail_sizing(objective, rows, solver, deadline, log, label):
        """Simulate an unavailable continuous backend after successful support selection."""
        if 'fixed_sizing' in label:
            return False
        return original(objective, rows, solver, deadline, log, label)

    monkeypatch.setattr(prep, '_solve', fail_sizing)
    result, record = prep.prepare_common_execution(floor_repair_problem(), solver='SCIPY')
    assert result is None
    assert record['status'] == 'preparation_failed'


def test_repeated_endpoint_faults_exhaust_the_fixed_sizing_limit(monkeypatch):
    """A second different boundary fault cannot silently acquire a third sizing attempt."""
    p = example(coupled=False)
    current = p.current.copy()
    current[0] -= .01
    current[1] -= .02
    current[-1] += .03
    upper, full_lo, full_hi = p.upper.copy(), p.full_lower.copy(), p.full_upper.copy()
    upper[1] = .08
    full_lo[:2], full_hi[:2] = p.lower[:2], upper[:2]
    mandatory = np.array([True, True, False, False, False])
    p = replace(p, current=current, upper=upper, full_lower=full_lo, full_upper=full_hi,
                mandatory=mandatory, categories=('product_min_buy', 'product_min_buy', '', '', ''),
                rules=replace(p.rules, holding_floor=np.array([.05, .04, .005, .005, 0.]),
                              exemption_mask=mandatory))
    original = prep._solve
    calls = []

    def fail_next_boundary(objective, rows, solver, deadline, log, label):
        """Corrupt the first remaining free native coordinate on each fixed solve."""
        success = original(objective, rows, solver, deadline, log, label)
        if success and 'fixed_sizing' in label:
            variable = objective.variables()[0]
            value = variable.value.copy()
            value[0] = -5e-6
            variable.save_value(value)
            calls.append(label)
        return success

    monkeypatch.setattr(prep, '_solve', fail_next_boundary)
    result, record = prep.prepare_common_execution(p, solver='SCIPY')
    assert result is None, record
    assert calls == [f'{mode}:fixed_sizing:{i}' for mode in ('CORRIDOR', 'FULL') for i in (0, 1)]


def test_out_of_domain_repair_is_rejected_at_declaration(monkeypatch):
    """A backend cannot manufacture admissibility by declaring an outside endpoint fixed."""
    p = floor_repair_problem()
    bad = p.base.copy()
    bad[1] = .5
    monkeypatch.setattr(prep, '_repair', lambda *a, **k: bad)
    result, record = prep.prepare_common_execution(p, solver='SCIPY')
    assert result is None
    assert all(row['status'] == 'declaration_rejected' for row in record['stages'])


def test_contradictory_fixed_cash_bounds_fail_closed():
    """Even a malformed internal hard cash interval cannot enter fixed-state sizing."""
    p = example()
    hi = p.upper.copy()
    hi[-1] = -.1
    damaged = SimpleNamespace(**vars(p))
    damaged.upper = hi
    with pytest.raises(ValueError, match='contradicts a holding state'):
        prep._graph(damaged, 'CORRIDOR', dict(held=[], repair=[1, 1, 1, 1]))
