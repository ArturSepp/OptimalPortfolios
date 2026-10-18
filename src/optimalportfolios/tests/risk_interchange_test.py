"""Provider-neutral risk boundary, including full residual and native coordinates.

These required base tests contain no fitted-estimator dependency. Numerical oracles
use direct supplied-array calculations independently of QIS result methods.
"""

import numpy as np
import pandas as pd
import pytest
from qis import RiskModel
from qis.perfstats.config import ReturnTypes

from optimalportfolios.optimization.portfolio_result import PortfolioOptimisationResult

DATE = pd.Timestamp('2026-07-31')
ASSETS = pd.Index(['equity', 'bond', 'private'])
FACTORS = pd.Index(['growth', 'yield_change'])
WEIGHTS = pd.Series([0.5, 0.3, 0.2], index=ASSETS, name='candidate')
BENCHMARK = pd.Series([0.3, 0.5, 0.2], index=ASSETS, name='benchmark')
BETAS = pd.DataFrame([[1.0, -0.02], [0.1, -0.08], [0.6, -0.01]],
                     index=ASSETS, columns=FACTORS)
FACTOR_COVARIANCE = pd.DataFrame([[0.04, -0.01], [-0.01, 0.36]],
                                 index=FACTORS, columns=FACTORS)
RESIDUAL = pd.DataFrame([[0.01, 0.003, 0.002], [0.003, 0.008, 0.001],
                         [0.002, 0.001, 0.016]], index=ASSETS, columns=ASSETS)
COVARIANCE = BETAS @ FACTOR_COVARIANCE @ BETAS.T + RESIDUAL


def _result(**kwargs) -> PortfolioOptimisationResult:
    """Supply stable weights and an explicit decision date for each boundary test."""
    return PortfolioOptimisationResult(weights=WEIGHTS, benchmark_weights=BENCHMARK,
                                       optimisation_date=DATE, **kwargs)


def _factor_risk(include_diagonal: bool = False) -> RiskModel:
    """Build independent supplied QIS data, retaining native factor coordinates."""
    kwargs = {}
    if include_diagonal:
        kwargs = {'factor_covar': {DATE: FACTOR_COVARIANCE},
                  'residual_vars': {DATE: pd.Series(np.diag(RESIDUAL), index=ASSETS)}}
    return RiskModel(covar={DATE: COVARIANCE}, factor_loadings={DATE: BETAS},
                     factor_return_types={'growth': ReturnTypes.LOG,
                                          'yield_change': ReturnTypes.DIFFERENCE}, **kwargs)


def test_covariance_only_result_uses_supplied_total_risk() -> None:
    """A base optimiser result needs no factor model to report portfolio and active risk."""
    result = _result(covariance=COVARIANCE)
    active = WEIGHTS - BENCHMARK
    assert result.compute_portfolio_variance() == pytest.approx(WEIGHTS @ COVARIANCE @ WEIGHTS)
    assert result.compute_tracking_error() ** 2 == pytest.approx(active @ COVARIANCE @ active)
    assert set(result.compute_risk_summary()['portfolio'].columns) == {
        'total_vol', 'tracking_error'}
    assert result.compute_factor_exposures_summary() == {}
    assert 'total_vol' in result.report()['asset_snapshot']


def test_full_residual_and_native_metadata_survive_interchange() -> None:
    """Correlated residual covariance drives total risk; native loadings stay unscaled."""
    model = _factor_risk()
    result = _result(risk_model=model)
    exposure = WEIGHTS @ BETAS
    systematic = float(exposure @ FACTOR_COVARIANCE @ exposure)
    full_residual = float(WEIGHTS @ RESIDUAL @ WEIGHTS)
    diagonal_residual = float(WEIGHTS.pow(2) @ np.diag(RESIDUAL))
    assert full_residual != pytest.approx(diagonal_residual)
    assert result.compute_portfolio_variance() == pytest.approx(systematic + full_residual)
    pd.testing.assert_series_equal(result._compute_factor_exposures_for_weights(WEIGHTS),
                                   exposure, check_names=False)
    assert result.risk_model.factor_return_types == model.factor_return_types
    assert 'factor_vol' not in result.compute_risk_summary()['portfolio']
    assert 'residual_te' not in result.compute_active_risk_attribution().index
    assert result.compute_factor_exposures_summary()['risk_pct_portfolio'].empty


def test_diagonal_reference_is_explicit_and_does_not_replace_total_covariance() -> None:
    """A deliberate reference decomposition is labelled and may differ from full risk."""
    with pytest.raises(ValueError, match='declare decomposition_mode'):
        _result(risk_model=_factor_risk(include_diagonal=True))
    result = _result(risk_model=_factor_risk(include_diagonal=True),
                     decomposition_mode='diagonal_reference')
    active = WEIGHTS - BENCHMARK
    risk = result.compute_risk_summary()['portfolio'].loc['candidate']
    reference = BETAS @ FACTOR_COVARIANCE @ BETAS.T + np.diag(np.diag(RESIDUAL))
    assert risk['total_vol'] ** 2 == pytest.approx(WEIGHTS @ COVARIANCE @ WEIGHTS)
    assert risk['tracking_error'] ** 2 == pytest.approx(active @ COVARIANCE @ active)
    assert risk['diagonal_reference_tracking_error'] ** 2 == pytest.approx(
        active @ reference @ active)
    assert risk['residual_reference_vol'] ** 2 == pytest.approx(
        WEIGHTS.pow(2) @ np.diag(RESIDUAL))
    assert 'factor_vol' not in risk and 'residual_te' not in risk
    attribution = result.compute_active_risk_attribution()['candidate_active']
    assert attribution['factor_reference_pct'] + attribution['residual_reference_pct'] == (
        pytest.approx(1.0))


def test_weights_only_result_is_a_complete_optimisation_output() -> None:
    """Weights, turnover and prepared tables remain usable without any risk object."""
    result = _result(current_weights=BENCHMARK,
                     asset_diagnostics=pd.DataFrame({'score': [1., 2., 3.]}, index=ASSETS))
    assert result.compute_turnover() == pytest.approx(0.2)
    assert list(result.get_asset_betas_table().columns) == ['score']
    assert result.report()['risk_summary'] == {}
    assert 'tracking_error' not in result.summary()
    assert 'vol=' not in repr(result)
    assert _result().get_asset_betas_table().empty
    with pytest.raises(ValueError, match='risk analytics require'):
        result.compute_tracking_error()
    with pytest.raises(ValueError, match='risk analytics require'):
        result.compute_portfolio_variance()


def test_date_selection_is_asof_and_explicit_dates_are_exact() -> None:
    """Later covariance cannot leak into the selected decision-date context."""
    before, after = DATE - pd.Timedelta(days=1), DATE + pd.Timedelta(days=1)
    model = RiskModel(covar={before: COVARIANCE, after: 7 * COVARIANCE})
    result = _result(risk_model=model)
    assert result.risk_model_date == before
    pd.testing.assert_frame_equal(result.covariance, COVARIANCE)
    with pytest.raises(ValueError, match='exact covariance date'):
        _result(risk_model=model, risk_model_date=DATE)
    with pytest.raises(ValueError, match='after optimisation_date'):
        _result(risk_model=model, risk_model_date=after)
    with pytest.raises(ValueError, match='no risk model is available'):
        _result(risk_model=RiskModel(covar={after: COVARIANCE}))


def test_asset_order_is_aligned_and_missing_assets_are_rejected() -> None:
    """Labelled risk and weights align explicitly rather than relying on array order."""
    reversed_assets = ASSETS[::-1]
    result = PortfolioOptimisationResult(
        weights=WEIGHTS, benchmark_weights=BENCHMARK.loc[reversed_assets],
        covariance=COVARIANCE.loc[reversed_assets, reversed_assets], optimisation_date=DATE)
    assert result.compute_tracking_error() == pytest.approx(
        _result(covariance=COVARIANCE).compute_tracking_error())
    with pytest.raises(ValueError, match='missing assets'):
        _result(covariance=COVARIANCE.iloc[:2, :2])
    with pytest.raises(ValueError, match='asset labels must match'):
        PortfolioOptimisationResult(weights=WEIGHTS, benchmark_weights=BENCHMARK.iloc[:2])


def test_result_owns_supplied_context_and_prepared_diagnostics() -> None:
    """Later edits to the provider's frames cannot mutate an already constructed result."""
    model = _factor_risk()
    diagnostics = pd.DataFrame({'score': [1., 2., 3.]}, index=ASSETS[::-1])
    result = _result(risk_model=model, asset_diagnostics=diagnostics)
    expected = result.covariance.copy()
    model.covar[DATE] = 10 * COVARIANCE
    diagnostics.iloc[0, 0] = -99
    pd.testing.assert_frame_equal(result.covariance, expected)
    assert list(result.asset_diagnostics.index) == list(ASSETS)
    assert result.asset_diagnostics.loc['private', 'score'] == 1.0


def test_report_weight_vectors_cannot_expand_the_result_universe() -> None:
    """A larger provider universe cannot make arbitrary report vectors change scope."""
    universe = ASSETS.append(pd.Index(['outside']))
    covariance = COVARIANCE.reindex(index=universe, columns=universe, fill_value=0.)
    covariance.loc['outside', 'outside'] = .2
    loadings = BETAS.reindex(universe, fill_value=0.)
    risk = RiskModel(covar={DATE: covariance}, factor_loadings={DATE: loadings})
    result = _result(risk_model=risk)
    assert result.compute_portfolio_variance() == pytest.approx(WEIGHTS @ COVARIANCE @ WEIGHTS)
    expanded = WEIGHTS.reindex(universe, fill_value=.1)
    for method in (result.compute_portfolio_variance,
                   result._compute_factor_exposures_for_weights,
                   result._compute_factor_risk_contribution):
        with pytest.raises(ValueError, match='result asset universe'):
            method(expanded)
    with pytest.raises(ValueError, match='must be finite'):
        result.compute_portfolio_variance(WEIGHTS * np.nan)


@pytest.mark.parametrize('weights,benchmark,message', [
    (WEIGHTS.iloc[:0], BENCHMARK.iloc[:0], 'nonempty, unique'),
    (pd.Series([.5, .5], index=['A', 'A']),
     pd.Series([.5, .5], index=['A', 'A']), 'nonempty, unique'),
    (WEIGHTS, BENCHMARK * np.nan, 'finite weights'),
])
def test_initial_weights_must_be_unique_and_finite(weights, benchmark, message) -> None:
    """An invalid holdings object cannot produce apparently complete portfolio reports."""
    with pytest.raises(ValueError, match=message):
        PortfolioOptimisationResult(weights=weights, benchmark_weights=benchmark)


def test_undated_covariance_needs_an_explicit_date() -> None:
    """A covariance without either a model date or decision date has no as-of meaning."""
    with pytest.raises(ValueError, match='requires a valid'):
        PortfolioOptimisationResult(weights=WEIGHTS, benchmark_weights=BENCHMARK,
                                    covariance=COVARIANCE, optimisation_date=None)


@pytest.mark.parametrize('kwargs,match', [
    ({'decomposition_mode': 'implicit'}, 'decomposition_mode'),
    ({'covariance': COVARIANCE, 'risk_model': RiskModel(covar={DATE: COVARIANCE})},
     'not both'),
    ({'risk_model_date': DATE}, 'requires risk context'),
    ({'risk_model': object()}, 'must be a qis.RiskModel'),
    ({'asset_diagnostics': pd.DataFrame({'score': [1.]}, index=['equity'])}, 'cover every'),
])
def test_invalid_context_is_rejected_before_reporting(kwargs: dict, match: str) -> None:
    """Invalid context cannot become a misleading apparently complete report."""
    with pytest.raises((ValueError, TypeError), match=match):
        _result(**kwargs)
