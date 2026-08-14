"""Characterisation tests for PortfolioOptimisationResult risk delegation."""

import numpy as np
import pandas as pd
from factorlasso import CurrentFactorCovarData, VarianceColumns

from optimalportfolios import PortfolioOptimisationResult

SEED = 20260809
RTOL = 1e-12
ATOL = 1e-16


def _build_result() -> PortfolioOptimisationResult:
    """Build a seeded two-portfolio result with a shared benchmark and current weights."""
    rng = np.random.default_rng(SEED)
    assets = pd.Index([f'asset_{idx}' for idx in range(5)])
    factors = pd.Index(['growth', 'inflation'])
    betas = rng.normal(scale=0.35, size=(5, 2))
    factor_root = rng.normal(scale=0.16, size=(2, 2))
    factor_covar = factor_root @ factor_root.T
    residual_vars = rng.uniform(0.003, 0.018, size=5)
    covar_data = CurrentFactorCovarData(
        x_covar=pd.DataFrame(factor_covar, index=factors, columns=factors),
        y_betas=pd.DataFrame(betas, index=assets, columns=factors),
        y_variances=pd.DataFrame(
            {VarianceColumns.RESIDUAL_VARS.value: residual_vars}, index=assets),
        estimation_date=pd.Timestamp('2026-07-31'),
    )
    weights = pd.DataFrame({
        'balanced': rng.dirichlet(np.ones(5)),
        'defensive': rng.dirichlet(np.ones(5)),
    }, index=assets)
    benchmark = pd.Series(
        rng.dirichlet(np.ones(5)), index=assets, name='benchmark')
    current = pd.Series(rng.dirichlet(np.ones(5)), index=assets, name='current')
    expected_return = pd.Series(
        rng.normal(loc=0.06, scale=0.02, size=5), index=assets)
    return PortfolioOptimisationResult(
        weights=weights,
        benchmark_weights=benchmark,
        covar_data=covar_data,
        group_attributions={},
        current_weights=current,
        expected_return=expected_return,
        optimisation_date=pd.Timestamp('2026-07-31'),
    )


def test_portfolio_result_risk_outputs_are_characterised() -> None:
    """Pin every facade output that T2b delegates to qis.RiskModel."""
    result = _build_result()
    expected_tracking_errors = {
        'balanced': 0.08961496807543852,
        'defensive': 0.1216953920489923,
    }
    for name, expected in expected_tracking_errors.items():
        np.testing.assert_allclose(
            result.compute_tracking_error(name), expected, rtol=RTOL, atol=ATOL)

    expected_snapshot = pd.DataFrame(
        [
            [0.03931525717683616, 0.0551712300042752,
             0.03445498085303398, 0.06361293204928387],
            [0.10245694506643982, 0.13751120842181713,
             0.06112255476612994, 0.145397070996579],
            [0.08416540666121644, 0.12466151499592229,
             0.026459750008613562, 0.13024217565356808],
            [0.05842610644138056, 0.058041701564908225,
             0.0550985329262062, 0.06463191112298437],
            [0.6748145644025308, 0.8218425144859448,
             0.1873995210677228, 0.8024019382714992],
            [0.3251854355974691, 0.17815748551405516,
             0.8126004789322773, 0.19759806172850083],
            [0.3837246674819512, 0.4012126039576123,
             0.5637032186378218, 0.4375117848885731],
            [0.08961496807543852, 0.12169539204899232, np.nan, np.nan],
            [0.06999968350718923, 0.10855635952379843, np.nan, np.nan],
            [0.05595432791174608, 0.055002593147030446, np.nan, np.nan],
        ],
        index=[
            'exp_return', 'total_vol', 'factor_vol', 'residual_vol',
            'factor_pct', 'residual_pct', 'sharpe_ratio', 'tracking_error',
            'factor_te', 'residual_te',
        ],
        columns=['balanced', 'defensive', 'benchmark', 'current'],
    )
    pd.testing.assert_frame_equal(
        result.compute_returns_risk_snapshot(), expected_snapshot,
        check_exact=False, rtol=RTOL, atol=ATOL)

    expected_attribution = pd.DataFrame(
        [
            [0.08961496807543852, 0.12169539204899232],
            [0.06999968350718923, 0.10855635952379843],
            [0.05595432791174608, 0.055002593147030446],
            [0.610142172403141, 0.7957236628015303],
            [0.38985782759685905, 0.20427633719846963],
        ],
        index=['tracking_error', 'factor_te', 'residual_te',
               'factor_pct', 'residual_pct'],
        columns=['balanced_active', 'defensive_active'],
    )
    actual_attribution = result.compute_active_risk_attribution()
    pd.testing.assert_frame_equal(
        actual_attribution, expected_attribution,
        check_exact=False, rtol=RTOL, atol=ATOL)

    expected_exposures = pd.Series(
        [-0.08515517231322406, -0.2192202115684679],
        index=['growth', 'inflation'],
    )
    pd.testing.assert_series_equal(
        result._compute_factor_exposures_for_weights(result.get_weights('balanced')),
        expected_exposures, check_exact=False, rtol=RTOL, atol=ATOL)

    covar = result.covar_data.y_covar.to_numpy()
    for name in result.portfolio_names:
        active = result.get_active_weights(name).to_numpy()
        reference_te = np.sqrt(np.einsum('i,ij,j->', active, covar, active))
        np.testing.assert_allclose(
            result.compute_tracking_error(name), reference_te, rtol=RTOL, atol=ATOL)
        active_column = f'{name}_active'
        np.testing.assert_allclose(
            actual_attribution.loc['tracking_error', active_column] ** 2,
            actual_attribution.loc['factor_te', active_column] ** 2
            + actual_attribution.loc['residual_te', active_column] ** 2,
            rtol=RTOL, atol=ATOL,
        )
