"""Independent analytical and public-format contracts for additional inputs."""
import numpy as np
import pandas as pd
import pytest

from papers.ranked_execution_2026.replication import study_inputs as inputs
from papers.ranked_execution_2026.replication.benchmark import sha256,tracking_error_bp
from papers.ranked_execution_2026.replication.study_design import original_audit


def test_corridor_counterexample_has_analytic_full_and_corridor_minima():
    """One-ticket cases reduce to two identical scalar quadratics and unchanged cash."""
    from optimalportfolios.execution.v2 import wrapper_exact_execution
    from optimalportfolios.execution.v2.domain import variance
    p=inputs.corridor_counterexample()
    corridor=np.array([.3,.2,.5]); full=np.array([.39,.2,.41])
    assert original_audit(p,corridor)['passed']
    assert original_audit(p,full,'FULL')['passed']
    assert not original_audit(p,full)['passed']
    # With x = w_A - .3 and w_B fixed at .2, risk is
    # .01*(x*x - .18*x + .01), minimized at x=.09. The corridor
    # restricts x to [-.1,0], where the minimum is at zero. Symmetry
    # covers trading B; the no-trade alternative is strictly worse.
    assert variance(p,corridor)==pytest.approx(.0001,abs=1e-15)
    assert variance(p,full)==pytest.approx(.000019,abs=1e-15)
    assert tracking_error_bp(p,full)==pytest.approx(43.5889894354,abs=1e-8)
    for domain,expected in [('CORRIDOR',.0001),('FULL',.000019)]:
        result=wrapper_exact_execution(p,domain,time_limit_s=10,threads=1,gap_rel=1e-4)
        assert original_audit(p,result.weights,domain)['passed']
        assert variance(p,result.weights)==pytest.approx(expected,abs=3e-10)


def test_external_parser_checks_shape_and_positive_prices(tmp_path):
    """The source format cannot silently lose columns or accept malformed prices."""
    path=tmp_path/'source.txt'
    path.write_text('2 2\n1 2 3\n4 5 6\n7 8 9\n')
    np.testing.assert_array_equal(inputs.read_or_library(path),[[1,4,7],[2,5,8],[3,6,9]])
    for text in ('2 2 1 2','2.5 2 1 2','2 2 1 2 3 4 5 6 7 8 0','2 2 nan'):
        path.write_text(text)
        with pytest.raises(ValueError): inputs.read_or_library(path)


def test_external_risk_uses_only_the_declared_history(tmp_path,monkeypatch):
    """The owning estimator sees no later observations and annualisation is explicit."""
    path=tmp_path/'source.txt'; prices=np.exp(np.arange(70)[:,None]*np.array([[.001,.002,.003]]))
    path.write_text('2 69\n'+' '.join(str(x) for x in prices.T.ravel()))
    calls=[]
    def estimate(frame,**kwargs):
        """Record the estimator contract and return a known period covariance."""
        calls.append((frame.copy(),kwargs))
        return {frame.index[-1]:pd.DataFrame([[.001,.0002],[.0002,.002]])}
    monkeypatch.setattr(inputs.qis,'estimate_rolling_ewma_covar',estimate)
    case=dict(universe=2,source_file=path.name,source_sha256=sha256(path),observations=60)
    # This test isolates external risk preparation from generator economic size.
    monkeypatch.setattr(inputs,'generated_problem',lambda case: inputs.corridor_counterexample())
    p=inputs.external_problem(case,tmp_path)
    assert len(calls[0][0])==61
    np.testing.assert_array_equal(calls[0][0].to_numpy(),prices[:61,1:])
    assert calls[0][1]['apply_an_factor'] is False and calls[0][1]['demean'] is True
    np.testing.assert_allclose(p.covariance[:2,:2],52*np.array([[.001,.0002],[.0002,.002]]))
    assert p.covariance[-1,-1]==1e-6
    path.write_text('modified')
    with pytest.raises(ValueError,match='Unverified'): inputs.external_problem(case,tmp_path)
