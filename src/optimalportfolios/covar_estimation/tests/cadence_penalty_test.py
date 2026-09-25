"""Fixed cadence penalties agree with independently configured individual fits."""
import numpy as np
import pandas as pd
import pytest
import qis

from factorlasso import LassoModel, LassoModelType
from optimalportfolios.covar_estimation.factor_covar_estimator import (
    FactorCovarEstimator, estimate_lasso_factor_covar_data, _fit_lasso_frequency,
)


def _inputs():
    """Build deterministic monthly and quarterly log returns on one price panel."""
    rng = np.random.default_rng(734)
    index = pd.date_range('2000-01-31', periods=120, freq='ME')
    x = pd.DataFrame(rng.normal(0, .04, (120, 2)), index=index, columns=['F1', 'F2'])
    prices = np.exp(x.cumsum())
    monthly = pd.DataFrame({'M': .7*x.F1 + rng.normal(0, .01, 120)}, index=index)
    quarterly = np.log(prices.resample('QE').last()).diff()
    quarterly = pd.DataFrame({'Q': 1.2*quarterly.F2})
    return prices, {'ME': monthly, 'QE': quarterly}


def _model():
    """Use the portable public solver without requiring unreleased FactorLasso APIs."""
    return LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5,
                      span_freq_dict={'ME': 60, 'QE': 40}, warmup_period=0)


def test_cadence_penalties_match_independent_fits_and_preserve_configuration():
    """One mixed fit equals two scalar fits, with no lingering lambda mutation."""
    prices, panels = _inputs()
    penalties = {'ME': .000104264890398061, 'QE': .0000519664260036678}
    model = _model()
    estimator = FactorCovarEstimator(lasso_model=model, reg_lambda_freq_dict=penalties)
    result = estimator.fit_current_factor_covars(prices, panels)
    for freq, panel in panels.items():
        reference = _model()
        reference.reg_lambda = penalties[freq]
        separate = estimate_lasso_factor_covar_data(prices, {freq: panel}, reference)
        np.testing.assert_allclose(result.y_betas.loc[panel.columns], separate.y_betas,
                                   rtol=1e-7, atol=1e-9)
    assert model.reg_lambda == 1e-5
    assert model.estimated_betas is not None
    assert estimator.copy().reg_lambda_freq_dict == penalties
    assert estimator.to_dict()['reg_lambda_freq_dict'] == penalties


@pytest.mark.parametrize('penalties', [{'ME': -1.}, {'ME': np.nan}, {'ME': np.inf}])
def test_invalid_penalties_rejected(penalties):
    """Invalid fixed penalties cannot reach a solver."""
    prices, panels = _inputs()
    with pytest.raises(ValueError, match='finite and nonnegative'):
        estimate_lasso_factor_covar_data(prices, panels, _model(),
                                        reg_lambda_freq_dict=penalties)


def test_missing_cadence_rejected():
    """A configured map must cover every fitted cadence rather than fall back silently."""
    prices, panels = _inputs()
    with pytest.raises(KeyError, match='QE'):
        estimate_lasso_factor_covar_data(prices, panels, _model(),
                                        reg_lambda_freq_dict={'ME': .0001})


def test_solver_failure_restores_scalar_penalty(monkeypatch):
    """A failed solve must not poison a model reused at another cadence."""
    prices, panels = _inputs()
    model = _model()

    def fail(**kwargs):
        """Observe the override at the solver boundary before raising."""
        assert model.reg_lambda == .0001
        raise RuntimeError('solver failed')

    monkeypatch.setattr(model, 'fit', fail)
    with pytest.raises(RuntimeError, match='solver failed'):
        _fit_lasso_frequency(freq='ME', asset_returns=panels['ME'],
                             risk_factor_prices=prices, lasso_model=model,
                             verbose=False, reg_lambda=.0001)
    assert model.reg_lambda == 1e-5


def test_rolling_fits_keep_fixed_cadence_penalties_on_each_history():
    """Expanding histories use the same calibrated penalties at every endpoint."""
    prices, panels = _inputs()
    penalties = {'ME': .000104264890398061, 'QE': .0000519664260036678}
    estimator = FactorCovarEstimator(lasso_model=_model(), reg_lambda_freq_dict=penalties)
    rolling = estimator.fit_rolling_factor_covars(
        prices, panels, qis.TimePeriod('30Jun2009', '31Dec2009'), rebalancing_freq='QE')
    assert len(rolling.data) >= 2
    for date, result in rolling.data.items():
        for freq, panel in panels.items():
            model = _model()
            model.reg_lambda = penalties[freq]
            reference = estimate_lasso_factor_covar_data(
                prices.loc[:date], {freq: panel.loc[:date]}, model)
            np.testing.assert_allclose(result.y_betas.loc[panel.columns], reference.y_betas,
                                       rtol=1e-7, atol=1e-9)
