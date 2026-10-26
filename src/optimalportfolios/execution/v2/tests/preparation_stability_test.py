"""Identity-stable risk-neutral repair selection under tied primary objectives."""
from dataclasses import replace

import numpy as np
import pytest

from optimalportfolios.execution.v2 import audit_execution
from optimalportfolios.execution.v2.preparation import prepare_common_execution
from optimalportfolios.execution.v2.tests.formulation_test import example


def tied_problem(order=(0, 1, 2, 3, 4), perturbation=0.):
    """Two interchangeable group repairs have an independently known minimum."""
    p = example()
    current = np.full(5, .04)
    current[0] += perturbation
    current[-1] -= perturbation
    lo, hi = current.copy(), np.array([.08, .08, .04, .04, .2])
    lo[-1] = 0.
    p = replace(p, names=('b', 'a', 'c', 'd', 'cash'), current=current, base=current,
                target=hi, lower=lo, upper=hi, full_lower=lo, full_upper=hi,
                group_lower=np.array([.09]), group_upper=np.array([.12]))
    idx = np.array(order)
    arrays = {key: getattr(p, key)[idx] for key in
              ('current', 'base', 'target', 'lower', 'upper', 'full_lower', 'full_upper',
               'mandatory', 'eligible', 'frozen')}
    return replace(p, **arrays, names=tuple(p.names[i] for i in idx),
                   categories=tuple(p.categories[i] for i in idx),
                   cash=int(np.flatnonzero(idx == p.cash)[0]), groups=p.groups[:, idx],
                   covariance=p.covariance[np.ix_(idx, idx)])


@pytest.mark.parametrize('order', [(0, 1, 2, 3, 4), (4, 3, 1, 2, 0)])
@pytest.mark.parametrize('perturbation', [0., -5e-10, 5e-10])
def test_tied_repair_uses_identity_not_native_column_order(order, perturbation):
    """Canonical A supplies the same minimum group top-up after tiny perturbations."""
    p = tied_problem(order, perturbation)
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert [r['name'] for r in record['repairs']] == ['a']
    assert audit_execution(result, result.base)['passed']
    a = p.names.index('a')
    expected = .05 - perturbation - p.rules.audit_tolerance
    assert abs(result.base[a]-expected) < 2e-10
    np.testing.assert_array_equal(result.group_lower, p.group_lower)
    assert result.rules.audit_tolerance == p.rules.audit_tolerance


def test_canonical_preference_cannot_override_a_real_capacity_shortfall():
    """An earlier identity with insufficient capacity cannot replace a feasible repair."""
    p = tied_problem()
    hi = p.upper.copy()
    hi[1] = .045
    p = replace(p, upper=hi, full_upper=hi)
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert [r['name'] for r in record['repairs']] == ['b']
    assert audit_execution(result, result.base)['passed']


def test_two_repairs_use_lexicographic_identities_within_each_required_group():
    """Two disjoint deficits require one repair each; choose A then C uniquely."""
    p = tied_problem()
    hi = np.array([.06, .06, .06, .06, .2])
    p = replace(p, names=('b', 'a', 'd', 'c', 'cash'), upper=hi, full_upper=hi,
                groups=np.array([[1., 1., 0., 0., 0.], [0., 0., 1., 1., 0.]]),
                group_lower=np.array([.09, .09]), group_upper=np.array([.12, .12]))
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert {r['name'] for r in record['repairs']} == {'a', 'c'}
    assert sum(abs(result.base[:-1]-p.base[:-1])) == pytest.approx(.02, abs=3e-7)
    assert audit_execution(result, result.base)['passed']


def test_lower_turnover_precedes_canonical_identity():
    """A double-loaded B repairs a deficit with half A's turnover and must win."""
    p = tied_problem()
    p = replace(p, groups=np.array([[2., 1., 0., 0., 0.]]),
                group_lower=np.array([.14]), group_upper=np.array([.20]))
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert [r['name'] for r in record['repairs']] == ['b']
    assert audit_execution(result, result.base)['passed']


@pytest.mark.parametrize('full', [False, True])
def test_numerical_bound_pin_does_not_open_a_material_repair_interval(full):
    """A sub-zero-convention bound correction is fixed and does not count as a repair."""
    p = tied_problem()
    current = p.current.copy()
    current[0] += 5e-10
    current[-1] -= 5e-10
    lo, hi = np.zeros(5), np.array([.04, .04, .08, .08, .2])
    full_hi = hi.copy()
    if full:
        hi[2:4] = .04
        full_hi[0] = .08
    p = replace(p, current=current, base=current, lower=lo, upper=hi,
                full_lower=lo, full_upper=full_hi, groups=np.array([[0., 0., 1., 1., 0.]]))
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    count = next(r['objective'] for r in record['stages'] if r['stage'].endswith('repair_count'))
    assert abs(count-1) < 1e-7
    assert [r['name'] for r in record['repairs']] == ['c']
    assert result.base[0] == .04
    assert any(r['name'] == 'b' for r in record['numerical_pins'])
    assert audit_execution(result, result.base)['passed']


def test_numerical_corridor_movements_do_not_inflate_material_repair_count():
    """A funding deficit needs one sale, not a ticket for each tiny corridor movement."""
    p = tied_problem()
    lo = p.base.copy()-5e-9
    hi = p.base.copy()
    lo[-1], hi[-1] = .06, .2
    full_lo = np.zeros(5)
    full_lo[-1] = .06
    p = replace(p, lower=lo, upper=hi, full_lower=full_lo,
                full_upper=np.full(5, .2), groups=np.zeros((0, 5)),
                group_lower=np.zeros(0), group_upper=np.zeros(0))
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    count = next(r['objective'] for r in record['stages']
                 if r['stage'] == 'FULL:repair_count')
    assert abs(count-1) < 1e-7
    assert [r['name'] for r in record['repairs']] == ['a']
    assert audit_execution(result, result.base)['passed']


def tied_holding_problem():
    """Exiting 2.5% and topping up to a 5% floor have equal primary objectives."""
    p = example(coupled=False)
    current = np.array([.025, 0., 0., 0., .175])
    lo, hi = np.zeros(5), np.array([.1, 0., 0., 0., .2])
    return replace(p, current=current, base=current, lower=lo, upper=hi,
                   full_lower=lo, full_upper=hi,
                   rules=replace(p.rules, holding_floor=np.array([.05, 0., 0., 0., 0.])))


def test_equal_exit_or_top_up_preserves_the_base_holding_state():
    """The discrete floor tie prefers retaining the positive base holding."""
    p = tied_holding_problem()
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert result.base[0] == pytest.approx(.05, abs=1e-10)
    assert audit_execution(result, result.base)['passed']


@pytest.mark.parametrize('stage', ['repair_identity', 'holding_identity'])
@pytest.mark.parametrize('fault', ['deadline', 'unproven', 'fractional'])
def test_unproved_or_corrupt_tie_break_fails_closed(monkeypatch, stage, fault):
    """A timed-out, unproved or fractional tie-break cannot choose an arbitrary support."""
    import cvxpy as cp
    from optimalportfolios.execution.v2 import preparation as prep
    original = prep._solve
    calls = []

    def faulty_tie_break(objective, rows, solver, deadline, log, label):
        """Inject only the specified native tie-break failure after real primary solves."""
        if stage not in label:
            return original(objective, rows, solver, deadline, log, label)
        calls.append(label)
        if fault == 'deadline':
            log.append(dict(stage=label, status='time_limit'))
            return False
        success = original(objective, rows, solver, deadline, log, label)
        assert success
        if fault == 'unproven':
            log[-1]['status'] = cp.OPTIMAL_INACCURATE
        else:
            variable = objective.variables()[0]
            variable.save_value(variable.value+.1)
        return True

    monkeypatch.setattr(prep, '_solve', faulty_tie_break)
    result, record = prep.prepare_common_execution(tied_holding_problem(), solver='SCIPY')
    assert calls
    assert result is None, record
    assert record['status'] == 'preparation_failed'
