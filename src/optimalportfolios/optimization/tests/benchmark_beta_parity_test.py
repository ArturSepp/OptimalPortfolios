"""Cross-package parity for the retained benchmark-beta constraint helper."""

import numpy as np
import pandas as pd
import qis

from optimalportfolios.optimization.constraints import (
    compute_benchmark_beta_loadings_from_covar,
)


def test_joint_covariance_benchmark_beta_matches_public_risk_model() -> None:
    """Match loadings and linear portfolio beta on a seeded joint covariance."""
    # Seed 20260811; benchmark constituents are a strict subset of the joint universe.
    rng = np.random.default_rng(20260811)
    universe = pd.Index([f'asset_{idx}' for idx in range(8)])
    root = rng.normal(scale=0.15, size=(len(universe), len(universe)))
    covar_values = root @ root.T + np.diag(rng.uniform(0.002, 0.006, len(universe)))
    covar = pd.DataFrame(covar_values, index=universe, columns=universe)
    benchmark_weights = pd.Series(
        [0.55, 0.30, 0.15], index=['asset_1', 'asset_4', 'asset_6'])
    assert len(benchmark_weights) < len(universe)
    asset_tickers = ['asset_0', 'asset_2', 'asset_3', 'asset_5', 'asset_7']
    portfolio_weights = pd.Series(
        [0.80, -0.25, 0.35, -0.15, 0.25], index=asset_tickers)
    date = pd.Timestamp('2026-06-30')

    constraint_loadings = compute_benchmark_beta_loadings_from_covar(
        covar=covar,
        benchmark_weights=benchmark_weights,
        asset_tickers=asset_tickers,
    )
    full_benchmark = pd.Series(0.0, index=universe)
    full_benchmark.loc[benchmark_weights.index] = benchmark_weights
    full_portfolio = pd.Series(0.0, index=universe)
    full_portfolio.loc[portfolio_weights.index] = portfolio_weights
    risk_model = qis.RiskModel(covar={date: covar})
    risk_model_loadings = risk_model.compute_benchmark_beta_loadings_at_date(
        benchmark_weights=full_benchmark, date=date).loc[asset_tickers]

    np.testing.assert_allclose(
        constraint_loadings.to_numpy(), risk_model_loadings.to_numpy(),
        rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(
        float(constraint_loadings @ portfolio_weights),
        risk_model.compute_benchmark_beta_at_date(
            benchmark_weights=full_benchmark,
            portfolio_weights=full_portfolio,
            date=date,
        ),
        rtol=1e-12,
        atol=1e-16,
    )
