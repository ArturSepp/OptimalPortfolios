"""Offline numerical contracts for the example's monthly FM/FL covariance stage."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from examples.oss_portfolio_pipeline.config import PipelineConfig, all_tickers
from examples.oss_portfolio_pipeline.covariance import (
    estimate_covariances, production_estimator, save_covariances,
)
from examples.oss_portfolio_pipeline.pipeline import build_inputs, save_inputs

pytestmark = pytest.mark.factor_integration


@pytest.fixture(scope='module')
def prepared():
    """Build complete random ETF histories, including a weekend February month-end."""
    config = PipelineConfig(end='2020-03-01')
    dates = pd.bdate_range('2014-12-01', '2020-02-28')
    rng = np.random.default_rng(914)
    common = rng.normal(0., .003, (len(dates), 4))
    loadings = rng.normal(0., .6, (4, len(all_tickers())))
    returns = .0002 + common @ loadings + rng.normal(0., .003, (len(dates), len(all_tickers())))
    prices = pd.DataFrame(100 * np.exp(returns.cumsum(axis=0)),
                          index=dates, columns=all_tickers())
    prices['BIL'] = 100 * np.exp(.00008 * np.arange(len(dates)))
    return build_inputs(prices, config), config


@pytest.fixture(scope='module')
def fitted(prepared):
    """Execute three real FM/FL fits using the explicitly selected open-source solver."""
    inputs, config = prepared
    return estimate_covariances(inputs, config, solver='CLARABEL')


def test_production_parameters():
    """Freeze the material TAA production controls independently of library defaults."""
    estimator = production_estimator()
    model = estimator.lasso_model
    expected = {
        'span': 36, 'span_freq_dict': {'ME': 36, 'WOM-2FRI': 36, 'QE': 12},
        'reg_lambda': 1e-5, 'l1_weight': 0., 'loss_normalization': 'weight_sum',
        'demean': True, 'warmup_period': 12, 'solver': 'MOSEK', 'solver_fallbacks': None,
        'cutoff_fraction': .60, 'linkage_method': 'ward',
        'auto_sign_constraints': True, 'auto_sign_adaptive_weights': True,
        'auto_sign_threshold_t': 1., 'auto_sign_adaptive_floor': .5,
        'auto_sign_use_fit_span': True, 'auto_sign_variance': 'date',
        'apply_ols_prior': True, 'prior_selection_type': 'highest_r2',
        'expert_prior_bound_n_std': 1., 'expert_prior_hac_lags': 3,
        'expert_prior_hac_lags_freq_dict': {'ME': 3, 'WOM-2FRI': 3, 'QE': 1},
        'expert_prior_bound_centre': 'prior', 'named_set_max_vif': None,
    }
    for name, value in expected.items():
        assert getattr(model, name) == value
    assert model.model_type.name == 'FACTOR_CLUSTER_GROUP_LASSO'
    assert model.distance_transform.value == 'one_minus_rho'
    assert model.dependence_measure.value == 'pearson'
    assert model.cluster_smoother_type.name == 'NONE'
    assert estimator.factor_returns_freq == 'W-WED'
    assert estimator.factor_covar_span == 52
    assert estimator.rebalancing_freq == 'ME'
    assert estimator.residual_type == 'empirical'
    assert estimator.residual_corr_weight == estimator.residual_var_weight == 1.
    assert not estimator.is_apply_vol_normalised_returns


def test_monthly_excess_returns_and_calendar(prepared, fitted):
    """Check cash subtraction, endpoint dates and all 36 output assets without resampling APIs."""
    inputs, _ = prepared
    assert fitted.schedule.index.tolist() == list(pd.to_datetime(
        ['2019-12-31', '2020-01-31', '2020-02-29']))
    assert fitted.schedule.observation_date.tolist() == list(pd.to_datetime(
        ['2019-12-31', '2020-01-31', '2020-02-28']))
    first = inputs.prices.loc['2020-01-31']
    last = inputs.prices.loc['2020-02-28']
    expected = np.log(last / first) - np.log(last['BIL'] / first['BIL'])
    np.testing.assert_allclose(fitted.asset_excess_returns.loc['2020-02-29'], expected,
                               rtol=1e-12, atol=1e-14)
    assert fitted.asset_excess_returns.iloc[0].isna().all()
    assert fitted.asset_excess_returns.BIL.iloc[1:].eq(0.).all()
    assert list(fitted.covariances) == list(fitted.schedule.observation_date)
    assert fitted.model.snapshots[pd.Timestamp('2020-02-29')].provenance[
        'factor_observation_date'] == pd.Timestamp('2020-02-28')
    assert all(c.shape == (36, 36) for c in fitted.covariances.values())


def test_factor_covariance_independent_weighted_moment(prepared, fitted):
    """Recover FM's 52-week demeaned covariance using pandas means and a weighted Gram matrix."""
    inputs, _ = prepared
    for date, snapshot in fitted.model.snapshots.items():
        history = inputs.factor_navs.loc[:date]
        wednesdays = pd.date_range(history.index[0], history.index[-1], freq='W-WED')
        weekly = history.reindex(wednesdays, method='ffill')
        returns = np.log(weekly).diff().iloc[1:]
        centered = (returns - returns.ewm(span=52, adjust=False).mean()).iloc[1:]
        weights = (2 / 53) * (51 / 53) ** np.arange(len(centered) - 1, -1, -1)
        expected = 52 * (centered.to_numpy().T * weights) @ centered.to_numpy()
        np.testing.assert_allclose(snapshot.factor_covariance, expected, rtol=1e-11, atol=1e-13)
        b, f, residual = snapshot.loadings, snapshot.factor_covariance, snapshot.residual_covariance
        np.testing.assert_allclose(snapshot.asset_covariance, b @ f @ b.T + residual,
                                   rtol=1e-10, atol=1e-13)
        assert np.linalg.eigvalsh(snapshot.asset_covariance).min() >= -1e-10
        assert np.abs(residual.to_numpy() - np.diag(np.diag(residual))).max() > 1e-8
        # BIL is the cash numeraire: its excess risk is exactly zero, not ridge-filled.
        np.testing.assert_allclose(snapshot.asset_covariance.loc['BIL'], 0., atol=1e-12)


def test_future_prices_do_not_change_past_risk(prepared, fitted):
    """Refit perturbed future prices and a truncated history to detect forward information."""
    inputs, config = prepared
    cutoff = pd.Timestamp('2019-12-31')
    asset_prices, factor_navs = inputs.prices.copy(), inputs.factor_navs.copy()
    asset_prices.loc[asset_prices.index > cutoff] *= 1.12
    factor_navs.loc[factor_navs.index > cutoff] *= 1.08
    changed = estimate_covariances(
        replace(inputs, prices=asset_prices, factor_navs=factor_navs), config, 'CLARABEL')
    prefix = estimate_covariances(
        replace(inputs, prices=inputs.prices.loc[:cutoff],
                factor_navs=inputs.factor_navs.loc[:cutoff]),
        replace(config, end='2020-01-01'), 'CLARABEL')
    for other in (changed, prefix):
        for field in ('factor_covariance', 'loadings', 'residual_covariance', 'asset_covariance'):
            pd.testing.assert_frame_equal(getattr(fitted.model.snapshots[cutoff], field),
                                          getattr(other.model.snapshots[cutoff], field))


def test_covariance_bundle(prepared, fitted, tmp_path):
    """Persist all risk matrices with a hash link to their input snapshot and solver settings."""
    inputs, config = prepared
    save_inputs(inputs, config, tmp_path)
    save_covariances(fitted, tmp_path)
    manifest = json.loads((tmp_path / 'covariance_manifest.json').read_text())
    assert manifest['input_manifest_sha256'] == hashlib.sha256(
        (tmp_path / 'manifest.json').read_bytes()).hexdigest()
    assert manifest['settings']['lasso_model']['solver'] == 'CLARABEL'
    for name, checksum in manifest['file_sha256'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == checksum
    table = pd.read_csv(tmp_path / 'factor_covariance.csv', index_col=[0, 1])
    assert table.shape == (3 * 11, 11)


def test_solver_and_start_validation(prepared, monkeypatch):
    """Reject an unavailable requested solver and an incomplete first monthly update."""
    import examples.oss_portfolio_pipeline.covariance as covariance
    inputs, config = prepared
    with pytest.raises(ValueError, match='solver must'):
        production_estimator('invalid')
    monkeypatch.setattr(covariance.cp, 'installed_solvers', lambda: ['CLARABEL'])
    with pytest.raises(RuntimeError, match='MOSEK is unavailable'):
        estimate_covariances(inputs, config)
    with pytest.raises(ValueError, match='completed calendar month-end'):
        estimate_covariances(inputs, replace(config, backtest_start='2019-12-30'), 'CLARABEL')
