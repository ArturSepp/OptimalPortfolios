"""Independent contracts for production-ready common execution preparation."""
from dataclasses import replace

import numpy as np
import pytest

from optimalportfolios.execution.v2 import audit_execution
from optimalportfolios.execution.v2.tests.formulation_test import example


def prepare(problem, **kwargs):
    """Import the implementation inside tests so a missing API is a test failure."""
    from optimalportfolios.execution.v2.preparation import prepare_common_execution
    return prepare_common_execution(problem, solver='SCIPY', **kwargs)


def test_already_audited_base_is_preserved_exactly():
    """A valid common book needs no solver, extra exemptions or changed domain."""
    p = example()
    prepared, diagnostics = prepare(p)
    assert prepared is p
    assert diagnostics['status'] == 'already_audited'
    assert not diagnostics['stages']


def test_compulsory_group_repair_is_risk_neutral_and_counted():
    """A hand-solvable class deficit needs one minimum-size repair ticket."""
    p = example()
    # Current A+B=.07; their corridors can provide another .01 through A.
    p = replace(p, group_lower=np.array([.08]), group_upper=np.array([.10]),
                rules=replace(p.rules, discretionary_allowance=2, ticket_cap=2))
    prepared, record = prepare(p)
    assert prepared is not None, record
    assert audit_execution(prepared, prepared.base)['passed']
    assert prepared.mandatory_count == 1
    assert prepared.rules.ticket_cap == 3
    assert prepared.rules.discretionary_allowance == 2
    assert abs(prepared.base[0] - .05) < 2e-7
    assert prepared.base[0] == prepared.lower[0] == prepared.full_upper[0]
    np.testing.assert_array_equal(prepared.base[1:4], p.base[1:4])
    np.testing.assert_array_equal(prepared.lower[1:4], p.lower[1:4])
    assert not record['numerical_pins']
    assert np.array_equal(prepared.current, p.current)
    assert np.array_equal(prepared.group_lower, p.group_lower)
    other, _ = prepare(replace(p, covariance=p.covariance*100., target=p.current))
    np.testing.assert_allclose(other.base, prepared.base, atol=1e-9, rtol=0)


def test_genuine_hard_group_breach_is_not_relaxed():
    """Insufficient eligible hard capacity remains a recorded preparation failure."""
    p = example()
    p = replace(p, group_lower=np.array([.25]), group_upper=np.array([.30]))
    prepared, record = prepare(p)
    assert prepared is None
    assert record['status'] == 'preparation_failed'
    assert p.group_lower[0] == .25


def test_full_domain_exception_is_declared_and_pinned():
    """A repair outside a target corridor is explicit and cannot open free capacity."""
    p = example()
    p = replace(p, group_lower=np.array([.10]), group_upper=np.array([.12]))
    prepared, record = prepare(p)
    assert prepared is not None, record
    assert record['mode'] == 'FULL'
    assert record['corridor_exceptions']
    for index in np.flatnonzero(prepared.mandatory):
        assert prepared.lower[index] == prepared.upper[index]
    assert audit_execution(prepared, prepared.base)['passed']


def test_preparation_cannot_exceed_a_fixed_total_cap():
    """Compulsory additions increase M+b only when that convention was declared."""
    p = example(cap=0)
    p = replace(p, group_lower=np.array([.08]), group_upper=np.array([.10]))
    prepared, record = prepare(p)
    assert prepared is None
    assert record['status'] == 'prepared_base_rejected'


@pytest.mark.parametrize('seconds', [0., -1., float('nan')])
def test_bad_preparation_budget_is_rejected(seconds):
    """Malformed deadlines cannot accidentally become unlimited solves."""
    with pytest.raises(ValueError, match='positive'):
        prepare(example(), time_limit_s=seconds)
