"""Closed-form frontier references using supplied full residual covariance."""
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.matf.opportunity import (
    OpportunityModel, portfolio_opportunity_metrics, solve_opportunity,
)


def test_correlated_risk_and_frontier_match_two_asset_formula():
    """Equal-beta active risk is 2 d (1-rho) times the squared transfer."""
    ids = pd.Index(['a', 'b'])
    b = pd.DataFrame(1., index=ids, columns=['market'])
    f = pd.DataFrame(.09, index=b.columns, columns=b.columns)
    residual = pd.DataFrame(.04*np.array([[1., .4], [.4, 1.]]), index=ids, columns=ids)
    model = OpportunityModel(b @ f @ b.T+residual,
        pd.Series([0., .04], index=ids), b, f, residual, date=pd.Timestamp('2026-06-30'))
    reference = pd.Series(.5, index=ids)
    weights = pd.Series([.4, .6], index=ids)
    metrics, _, _ = portfolio_opportunity_metrics(model, weights, reference)
    assert metrics['total_te']**2 == pytest.approx(2*.04*(1-.4)*.1**2)
    assert metrics['residual_te'] == pytest.approx(metrics['total_te'])
    assert metrics['factor_te'] == pytest.approx(0., abs=1e-12)
    assert metrics['portfolio_vol']**2 == pytest.approx(.09+.04*(.4**2+.6**2+2*.4*.4*.6))
    result = solve_opportunity(model, reference, total_te=.02, factor_te=0.)
    assert result.accepted
    assert result.metrics['alpha_increment'] == pytest.approx(
        .04*.02/np.sqrt(2*.04*.6), abs=2e-8)


def test_singular_supplied_risk_allows_a_zero_risk_transfer():
    """Perfectly correlated equal-beta assets have zero active risk on the simplex."""
    ids = pd.Index(['a', 'b'])
    b = pd.DataFrame(1., index=ids, columns=['market'])
    f = pd.DataFrame(.09, index=b.columns, columns=b.columns)
    residual = pd.DataFrame(.04, index=ids, columns=ids)
    model = OpportunityModel(b @ f @ b.T+residual,
        pd.Series([0., .04], index=ids), b, f, residual)
    reference = pd.Series(.5, index=ids)
    result = solve_opportunity(model, reference, total_te=0., factor_te=0.)
    assert result.accepted
    assert result.weights['b'] == pytest.approx(1., abs=1e-7)
    assert result.metrics['alpha_increment'] == pytest.approx(.02, abs=1e-8)
    assert result.metrics['total_te'] == pytest.approx(0., abs=1e-8)
    assert all(np.isfinite(value) for value in result.metrics.values())
    assert result.metrics['residual_te'] == pytest.approx(0., abs=1e-8)
