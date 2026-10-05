"""Closed-form two-asset risk and frontier references for correlated residuals."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from factorlasso import ResidualCorrelationData
from optimalportfolios.matf.opportunity import (
    OpportunityModel, portfolio_opportunity_metrics, solve_opportunity,
)


def test_correlated_risk_and_frontier_match_two_asset_formula():
    """Equal beta active risk is 2 d (1-rho) times the squared transfer."""
    ids = pd.Index(['a', 'b'])
    date = pd.Timestamp('2026-06-30')
    corr = ResidualCorrelationData(pd.DataFrame([[1., .8], [.8, 1.]], index=ids, columns=ids),
        pd.DataFrame([[1., 1.]], index=[date], columns=ids), pd.DataFrame(index=ids),
        'QE', 40., date, date)
    baseline = OpportunityModel(pd.DataFrame(1., index=ids, columns=['market']),
        pd.Series(.04, index=ids), pd.DataFrame(.09, index=['market'], columns=['market']),
        pd.Series([0., .04], index=ids), date)
    model = replace(baseline, residual_correlation=corr, residual_corr_weight=.5)
    reference = pd.Series(.5, index=ids)
    w = pd.Series([.4, .6], index=ids)
    metrics, _, _ = portfolio_opportunity_metrics(model, w, reference)
    assert metrics['total_te']**2 == pytest.approx(2*.04*(1-.4)*.1**2)
    assert metrics['residual_te'] == pytest.approx(metrics['total_te'])
    assert metrics['factor_te'] == pytest.approx(0., abs=1e-12)
    assert metrics['portfolio_vol']**2 == pytest.approx(.09+.04*(.4**2+.6**2+2*.4*.4*.6))
    result = solve_opportunity(model, reference, total_te=.02, factor_te=0.)
    assert result.accepted
    assert result.metrics['alpha_increment'] == pytest.approx(.04*.02/np.sqrt(2*.04*.6), abs=2e-8)
    zero = replace(baseline, residual_correlation=corr, residual_corr_weight=0.)
    old = portfolio_opportunity_metrics(baseline, w, reference)[0]
    assert portfolio_opportunity_metrics(zero, w, reference)[0] == old
    with pytest.raises(ValueError):
        replace(baseline, residual_corr_weight=.5)
    with pytest.raises(ValueError):
        replace(baseline, residual_correlation=corr, residual_corr_weight=2.)


def test_singular_empirical_risk_allows_a_zero_risk_transfer():
    """Perfectly correlated equal-beta assets have zero active risk on the simplex."""
    ids = pd.Index(['a', 'b'])
    date = pd.Timestamp('2026-06-30')
    corr = ResidualCorrelationData(pd.DataFrame(1., index=ids, columns=ids),
        pd.DataFrame([[1., 1.]], index=[date], columns=ids), pd.DataFrame(index=ids),
        'QE', 40., date, date)
    model = OpportunityModel(pd.DataFrame(1., index=ids, columns=['market']),
        pd.Series(.04, index=ids), pd.DataFrame(.09, index=['market'], columns=['market']),
        pd.Series([0., .04], index=ids), date, residual_correlation=corr)
    reference = pd.Series(.5, index=ids)
    result = solve_opportunity(model, reference, total_te=0., factor_te=0.)
    assert result.accepted
    assert result.weights['b'] == pytest.approx(1., abs=1e-7)
    assert result.metrics['alpha_increment'] == pytest.approx(.02, abs=1e-8)
    assert result.metrics['total_te'] == pytest.approx(0., abs=1e-8)
    assert all(np.isfinite(value) for value in result.metrics.values())
    assert result.metrics['residual_te'] == pytest.approx(0., abs=1e-8)
    assert result.metrics['gls_neutral_book_vol'] == pytest.approx(0., abs=1e-8)
