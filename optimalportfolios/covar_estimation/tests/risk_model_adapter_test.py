"""Numerical and structural tests for the qis risk-model adapter."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from factorlasso import CurrentFactorCovarData, RollingFactorCovarData, VarianceColumns

from optimalportfolios import build_risk_model

SEED = 20260808
N_DATES = 3
N_ASSETS = 6
N_FACTORS = 2
RTOL = 1e-12
ATOL = 1e-16


def _factor_fixture() -> Tuple[
        RollingFactorCovarData,
        Dict[pd.Timestamp, Tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """Construct three direct factor snapshots from seeded raw arrays."""
    rng = np.random.default_rng(SEED)
    assets = pd.Index([f'asset_{idx}' for idx in range(N_ASSETS)])
    factors = pd.Index([f'factor_{idx}' for idx in range(N_FACTORS)])
    dates = pd.date_range('2024-01-31', periods=N_DATES, freq='ME')
    snapshots = {}
    raw_parts = {}

    for date in dates:
        betas = rng.normal(scale=0.25, size=(N_ASSETS, N_FACTORS))
        factor_root = rng.normal(scale=0.12, size=(N_FACTORS, N_FACTORS))
        factor_covar = factor_root @ factor_root.T
        residual_vars = rng.uniform(0.0025, 0.0125, size=N_ASSETS)
        snapshots[date] = CurrentFactorCovarData(
            x_covar=pd.DataFrame(factor_covar, index=factors, columns=factors),
            y_betas=pd.DataFrame(betas, index=assets, columns=factors),
            y_variances=pd.DataFrame(
                {VarianceColumns.RESIDUAL_VARS.value: residual_vars}, index=assets),
            estimation_date=date,
        )
        raw_parts[date] = betas, factor_covar, residual_vars

    return RollingFactorCovarData(data=snapshots), raw_parts


def test_build_risk_model_from_factor_containers_matches_raw_arrays() -> None:
    """Both factor-container paths preserve orientation and raw covariance arithmetic."""
    rolling, raw_parts = _factor_fixture()
    portfolio = pd.Series([0.24, 0.18, 0.20, 0.15, 0.13, 0.10],
                          index=rolling.get_latest().y_betas.index)
    benchmark = pd.Series([0.20, 0.20, 0.15, 0.15, 0.15, 0.15], index=portfolio.index)
    active = portfolio.to_numpy() - benchmark.to_numpy()

    for model in (build_risk_model(rolling), build_risk_model(rolling.data)):
        assert len(model.dates) == N_DATES
        for date, (betas, factor_covar, residual_vars) in raw_parts.items():
            assert model.factor_loadings[date].shape == (N_ASSETS, N_FACTORS)
            assert len(model.residual_vars[date]) == N_ASSETS
            raw_covar = betas @ factor_covar @ betas.T + np.diag(residual_vars)
            expected_te = np.sqrt(np.einsum('i,ij,j->', active, raw_covar, active))
            actual_te = model.compute_tre_at_date(benchmark, portfolio, date)
            decomposition = model.compute_tre_decomposition_at_date(
                benchmark, portfolio, date)

            np.testing.assert_allclose(model.covar[date].to_numpy(), raw_covar,
                                       rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(actual_te, expected_te, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(decomposition['tracking_error'], actual_te,
                                       rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(
                model.compute_exposures_at_date(portfolio, date).to_numpy(),
                betas.T @ portfolio.to_numpy(), rtol=RTOL, atol=ATOL)


def test_build_covariance_only_model_and_reject_unsupported_input() -> None:
    """Plain covariance dictionaries omit factors and other inputs fail clearly."""
    rolling, _ = _factor_fixture()
    covars = rolling.get_y_covars(residual_var_weight=1.0)
    model = build_risk_model(covars)
    date = rolling.dates[0]
    portfolio = pd.Series([0.24, 0.18, 0.20, 0.15, 0.13, 0.10],
                          index=covars[date].index)
    benchmark = pd.Series([0.20, 0.20, 0.15, 0.15, 0.15, 0.15], index=portfolio.index)
    active = portfolio.to_numpy() - benchmark.to_numpy()
    expected_te = np.sqrt(np.einsum('i,ij,j->', active, covars[date].to_numpy(), active))

    np.testing.assert_allclose(
        model.compute_tre_at_date(benchmark, portfolio, date), expected_te,
        rtol=RTOL, atol=ATOL)
    with pytest.raises(ValueError, match='factor_loadings'):
        model.compute_tre_decomposition_at_date(benchmark, portfolio, date)
    with pytest.raises(ValueError, match='list'):
        build_risk_model([])
