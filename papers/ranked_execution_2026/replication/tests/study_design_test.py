"""Independent contracts for sparse scaling inputs; no benchmark solver is invoked."""
from dataclasses import replace

import numpy as np
import pytest

from papers.ranked_execution_2026.replication.study_design import (
    generated_problem,structural_summary,identity,save_problem,load_problem,original_audit,
)


def design(**changes):
    """Return a small declared configuration with valid unchanged trade thresholds."""
    return dict(dict(universe=300,held=96,additions=100,mandatory=2,budget=30,
        groups='overlap',regime='tight',pattern='exit_rich',seed=2026102951),**changes)


@pytest.mark.parametrize('n',[300,1000,3000])
@pytest.mark.parametrize('pattern',['diffuse','exit_rich'])
def test_sparse_portfolios_remain_valid_without_lower_floors(n,pattern):
    """Larger universes cannot imply overweight current books or smaller thresholds."""
    p=generated_problem(design(universe=n,pattern=pattern)); s=structural_summary(p)
    assert s['current_holdings']==96 and s['optional_candidates']>=100
    assert s['optional_candidates']<300
    assert np.isclose(p.current.sum(),1.) and np.isclose(p.target.sum(),1.)
    assert np.min(p.current[:96])>=.006
    assert p.rules.minimum_ticket==.005 and p.rules.holding_floor==.0025
    assert p.rules.audit_tolerance==1e-7 and p.rules.zero_tolerance==1e-8
    assert s['base_audit']['passed'] and s['base_audit']['tickets']==2
    assert np.all(p.current[96:n]==0.)


@pytest.mark.parametrize('changes',[{'held':300,'universe':1000},{'universe':100},
    {'mandatory':96},{'budget':-1},{'additions':200},{'pattern':'invalid'}])
def test_impossible_designs_are_rejected(changes):
    """Invalid resource/economic declarations cannot silently generate bad portfolios."""
    with pytest.raises(ValueError): generated_problem(design(**changes))


def test_representation_changes_neither_risk_nor_constraints():
    """Dense and exact structured inputs apply the same covariance to reference vectors."""
    p=generated_problem(design(universe=36,held=16,additions=16,mandatory=2,budget=10))
    dense=replace(p,covariance=p.dense_covariance(),factor_loadings=None,factor_covariance=None,residual_variance=None)
    for v in (p.base,p.target,np.arange(len(p.names),dtype=float)/100):
        np.testing.assert_allclose(p.matvec(v),dense.matvec(v),atol=1e-14,rtol=1e-12)
    for key in ('current','base','target','lower','upper','full_lower','full_upper','groups','group_lower','group_upper'):
        np.testing.assert_array_equal(getattr(p,key),getattr(dense,key))


def test_typed_roundtrip_and_input_tampering(tmp_path):
    """Both covariance forms round trip; numerical inputs and identities are immutable."""
    p=generated_problem(design()); row=save_problem(tmp_path/'input.npz',p,{'id':'test'})
    q=load_problem(tmp_path,row)
    assert identity(p)==identity(q)
    with pytest.raises(ValueError): q.current[0]=0.
    (tmp_path/'input.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Unverified'): load_problem(tmp_path,row)


def test_audit_preserves_genuine_breaches():
    """A feasible witness does not authorize enlarging any original audit envelope."""
    p=generated_problem(design())
    w=p.base.copy(); w[p.cash]+=.0001
    assert not original_audit(p,w)['passed']
    w=p.base.copy(); w[100]=.001; w[p.cash]-=.001
    audit=original_audit(p,w)
    assert not audit['passed'] and 'minimum_ticket' in audit['violations']
    with pytest.raises(ValueError): original_audit(p,p.base,'invalid')


def test_freshness_and_constraints_are_explicit():
    """Regime and target changes are declared, while thresholds stay fixed."""
    a=generated_problem(design()); b=generated_problem(design(pattern='diffuse'))
    for key in ('current','base','full_lower','full_upper','groups','group_lower','group_upper',
                'factor_loadings','factor_covariance','residual_variance'):
        np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    assert identity(a)!=identity(b)
    assert a.rules.minimum_ticket==b.rules.minimum_ticket
