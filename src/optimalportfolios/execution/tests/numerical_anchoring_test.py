"""Numerical base anchoring must retain legal discretionary trade intervals."""
from dataclasses import replace

import numpy as np
import pytest

from optimalportfolios.execution import audit_execution, compile_states
from optimalportfolios.execution.preparation import _declare, prepare_common_execution
from optimalportfolios.execution.tests.formulation_test import example


def anchoring_problem(anchor=0., residual=-5e-15):
    """Combine a numerical anchor with an independently necessary 1% group repair."""
    p = example()
    current = p.current.copy()
    current[0] = anchor + residual
    current[2] = .04
    current[-1] = p.investment - current[:-1].sum()
    lo, hi = np.zeros(5), np.full(5, .1)
    hi[0] = .08
    return replace(
        p, current=current, base=current, lower=lo, upper=hi,
        full_lower=lo, full_upper=hi, groups=np.array([[0., 1., 0., 0., 0.]]),
        group_lower=np.array([.04]), group_upper=np.array([.07]),
        rules=replace(p.rules, discretionary_allowance=2, ticket_cap=2),
    )


@pytest.mark.parametrize('anchor', [0., .005, .08])
@pytest.mark.parametrize('residual', [-5e-15, 5e-15])
def test_numerical_anchor_retains_optional_states_and_compulsory_pin(anchor, residual):
    """Roundoff at zero, a floor or a cap cannot erase a buy or sell opportunity."""
    p = anchoring_problem(anchor, residual)
    weights = p.base.copy()
    weights[0], weights[1] = anchor, .04
    weights[-1] = p.investment - weights[:-1].sum()
    result, record = _declare(p, weights, 'CORRIDOR')
    assert result.base[0] == anchor
    for field in ('lower', 'upper', 'full_lower', 'full_upper'):
        assert getattr(result, field)[0] == getattr(p, field)[0]
    assert any(state.ticket for state in compile_states(result)[0])
    assert record['numerical_pins'][0]['bounds_preserved']
    assert result.base[1] == result.lower[1] == result.full_upper[1] == .04
    assert result.mandatory.tolist() == [False, True, False, False, False]
    assert result.categories[1] == 'compulsory_saa_funding_repair'
    assert result.rules.ticket_cap == 3
    assert result.rules.zero_tolerance == p.rules.zero_tolerance
    assert result.rules.audit_tolerance == p.rules.audit_tolerance
    np.testing.assert_array_equal(result.current, p.current)
    np.testing.assert_array_equal(result.group_lower, p.group_lower)
    np.testing.assert_array_equal(result.group_upper, p.group_upper)
    assert audit_execution(result, result.base)['passed']
    # A separately constructed 1%-NAV buy is still admissible after preparation.
    trial = result.base.copy()
    change = -.01 if anchor == .08 else .01
    trial[0] += change
    trial[-1] -= change
    assert audit_execution(result, trial)['passed']


@pytest.mark.parametrize('residual', [-5e-15, 0., 5e-15])
def test_real_preparation_keeps_zero_residual_buy_interval(residual):
    """A genuine group deficit cannot turn harmless zero dust into a frozen asset."""
    p = anchoring_problem(residual=residual)
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert result.lower[0] == 0. and result.upper[0] == .08
    assert any(state.ticket for state in compile_states(result)[0])
    assert [row['name'] for row in record['repairs']] == ['b']
    assert result.lower[1] == result.upper[1]
    assert result.mandatory_count == 1
    assert audit_execution(result, result.base)['passed']


def test_move_beyond_zero_convention_remains_a_pinned_compulsory_repair():
    """The anchoring exception cannot hide a real intervention or its ticket."""
    p = anchoring_problem(anchor=.005, residual=0.)
    weights = p.base.copy()
    weights[0] += 2*p.rules.zero_tolerance
    weights[1] = .04
    weights[-1] = p.investment - weights[:-1].sum()
    result, record = _declare(p, weights, 'CORRIDOR')
    assert result.lower[0] == result.upper[0] == weights[0]
    assert result.full_lower[0] == result.full_upper[0] == weights[0]
    assert result.mandatory_count == 2
    assert result.rules.ticket_cap == 4
    assert {row['name'] for row in record['repairs']} == {'a', 'b'}
    assert not record['numerical_pins']


def test_numerical_full_domain_exception_does_not_open_free_corridor_capacity():
    """An anchor outside the supplied corridor retains its declared fixed exception."""
    p = anchoring_problem(anchor=.005, residual=-5e-15)
    hi = p.upper.copy()
    hi[0] = p.current[0]
    p = replace(p, upper=hi)
    weights = p.base.copy()
    weights[0], weights[1] = .005, .04
    weights[-1] = p.investment - weights[:-1].sum()
    result, record = _declare(p, weights, 'FULL')
    assert result.lower[0] == result.upper[0] == .005
    assert result.full_lower[0] == result.full_upper[0] == .005
    assert record['corridor_exceptions'][0]['name'] == 'a'
    assert not record['numerical_pins'][0]['bounds_preserved']
    assert not result.mandatory[0]


@pytest.mark.parametrize('residual', [5e-15, 1e-8])
def test_positive_numerical_holding_remains_inside_original_corridor(residual):
    """An audit-zero positive base can be held without widening its positive lower bound."""
    p = anchoring_problem(residual=residual)
    lo = p.lower.copy()
    lo[0] = residual
    p = replace(p, lower=lo)
    assert any(not state.ticket for state in compile_states(p)[0])
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert result.base[0] == residual
    assert result.lower[0] == residual and result.upper[0] == .08
    assert [row['name'] for row in record['repairs']] == ['b']
    assert result.mandatory_count == 1
    assert audit_execution(result, result.base)['passed']


def test_positive_holding_above_zero_convention_still_requires_floor_repair():
    """A real positive holding cannot use the numerical-zero hold state."""
    p = anchoring_problem(residual=2e-8)
    lo = p.lower.copy()
    lo[0] = p.current[0]
    p = replace(p, lower=lo)
    assert not any(not state.ticket for state in compile_states(p)[0])
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert result.base[0] >= p.rules.holding_floor
    assert {row['name'] for row in record['repairs']} == {'a', 'b'}
    assert result.mandatory_count == 2
    assert result.lower[0] == result.upper[0] == result.base[0]


def test_valid_holding_below_zero_convention_keeps_its_actual_floor_branch():
    """A user-supplied microscopic floor must not make a valid retained hold infeasible."""
    p = anchoring_problem(residual=5e-9)
    floor = np.full(5, .005)
    floor[0] = 1e-9
    p = replace(p, rules=replace(p.rules, holding_floor=floor))
    result, record = prepare_common_execution(p, solver='SCIPY')
    assert result is not None, record
    assert result.base[0] == p.base[0]
    assert [row['name'] for row in record['repairs']] == ['b']
    assert audit_execution(result, result.base)['passed']
