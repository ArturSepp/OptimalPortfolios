"""Offline tests for the paper-local August 2024 parity engine."""

from __future__ import annotations

import os
from dataclasses import replace

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

import qis
from papers.crypto_allocation_risk_2023.replication.parity_2024 import (
    GOLDEN_HEADLINE_MEDIAN,
    GOLDEN_MEDIAN_WEIGHT_ROWS,
    GOLDEN_PANEL_PATH,
    GOLDEN_WORKBOOK_PATH,
    MixtureParameters,
    Parity2024Config,
    ParityDependencyError,
    ParityMethod,
    dependency_report,
    estimate_ewm_covariances_2024,
    estimate_max_sharpe_inputs_2024,
    golden_headline_from_constants,
    golden_runtime_available,
    marginal_weights_2024,
    monthly_log_returns_2024,
    require_parity_dependencies,
    verify_golden_workbook,
)


def _synthetic_prices() -> pd.DataFrame:
    """Return a deterministic monthly panel with positive, non-collinear returns."""

    index = pd.date_range("2008-01-31", periods=100, freq="ME")
    step = np.arange(len(index), dtype=float)
    log_returns = np.column_stack(
        [
            0.0040 + 0.0060 * np.sin(step / 5.0),
            0.0030 + 0.0045 * np.cos(step / 7.0),
            0.0025 + 0.0035 * np.sin(step / 9.0 + 0.7),
            0.0090 + 0.0180 * np.cos(step / 4.0 + 0.2),
        ]
    )
    levels = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(
        levels,
        index=index,
        columns=["60/40", "HFs", "Gold", "BTC"],
    )


def _synthetic_config(**overrides: object) -> Parity2024Config:
    """Return a short reporting interval while retaining 60 observations of history."""

    config = Parity2024Config(
        estimation_start="2013-03-31",
        reporting_start="2013-03-31",
        end_date="2016-04-30",
        max_sharpe_solver="CLARABEL",
    )
    return replace(config, **overrides)


def _deterministic_mixture(
    values: np.ndarray,
    n_components: int,
    annualisation: float,
) -> MixtureParameters:
    """Supply stable mixture inputs so orchestration tests do not require sklearn."""

    assert n_components == 3
    mean = annualisation * np.mean(values, axis=0)
    covariance = annualisation * np.cov(values, rowvar=False, bias=True)
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-6
    return MixtureParameters(
        means=(mean - 0.01, mean, mean + 0.01),
        covariances=(0.8 * covariance, covariance, 1.3 * covariance),
        probabilities=np.array([0.2, 0.6, 0.2]),
    )


def test_max_sharpe_uses_exact_60_month_window_and_carried_raw_seed() -> None:
    """The estimator matches an independent point-in-time reconstruction."""

    prices = _synthetic_prices()
    config = _synthetic_config(estimation_start=None, reporting_start=None)
    inputs = estimate_max_sharpe_inputs_2024(prices, config)
    dates = list(inputs.covariances)
    assert len(dates) >= 2

    returns = monthly_log_returns_2024(prices, config)
    annualisation = qis.get_annualization_factor(freq=config.returns_freq)
    first_date, second_date = dates[:2]
    first_end = returns.index.get_loc(first_date)
    second_end = returns.index.get_loc(second_date)
    first_window = returns.iloc[first_end - 59 : first_end + 1]
    second_window = returns.iloc[second_end - 59 : second_end + 1]

    assert len(first_window) == config.roll_window == 60
    assert inputs.window_bounds.loc[first_date, "observations"] == 60
    assert inputs.window_bounds.loc[first_date, "first_observation"] == first_window.index[0]
    assert inputs.window_bounds.loc[first_date, "last_observation"] == first_date
    np.testing.assert_allclose(
        inputs.expected_returns.loc[first_date],
        annualisation * first_window.mean(axis=0),
        rtol=0.0,
        atol=1e-14,
    )

    raw_first = qis.compute_ewm_covar(
        a=first_window.to_numpy(),
        span=config.span,
        covar0=np.zeros((prices.shape[1], prices.shape[1])),
    )
    raw_second = qis.compute_ewm_covar(
        a=second_window.to_numpy(),
        span=config.span,
        covar0=raw_first,
    )
    np.testing.assert_allclose(
        inputs.raw_covariances[first_date], raw_first, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        inputs.raw_covariances[second_date], raw_second, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        inputs.covariances[first_date],
        annualisation * qis.matrix_regularization(raw_first, cut=1e-5),
        rtol=0.0,
        atol=1e-15,
    )


def test_max_sharpe_inputs_do_not_look_beyond_the_rebalance_date() -> None:
    """Changing future prices cannot change already-formed estimator inputs."""

    prices = _synthetic_prices()
    config = _synthetic_config(estimation_start=None, reporting_start=None)
    baseline = estimate_max_sharpe_inputs_2024(prices, config)
    decision_date = next(iter(baseline.covariances))

    changed = prices.copy()
    future = changed.index > decision_date
    multipliers = np.exp(np.linspace(0.0, 1.0, future.sum()))
    changed.loc[future, "BTC"] *= multipliers
    replay = estimate_max_sharpe_inputs_2024(changed, config)

    pd.testing.assert_series_equal(
        baseline.expected_returns.loc[decision_date],
        replay.expected_returns.loc[decision_date],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        baseline.raw_covariances[decision_date],
        replay.raw_covariances[decision_date],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        baseline.covariances[decision_date],
        replay.covariances[decision_date],
        check_exact=True,
    )


def test_with_and_without_covariances_are_estimated_on_their_own_labels() -> None:
    """The regression guard excludes crypto from the without-crypto covariance."""

    prices = _synthetic_prices()
    config = _synthetic_config()
    with_crypto = estimate_ewm_covariances_2024(prices, config)
    without_crypto = estimate_ewm_covariances_2024(prices.drop(columns="BTC"), config)
    assert with_crypto.keys() == without_crypto.keys()
    for covariance in with_crypto.values():
        assert covariance.index.equals(prices.columns)
        assert covariance.columns.equals(prices.columns)
    for covariance in without_crypto.values():
        assert covariance.index.equals(prices.columns.drop("BTC"))
        assert covariance.columns.equals(prices.columns.drop("BTC"))
        assert "BTC" not in covariance


@pytest.mark.parametrize("method", [ParityMethod.ERC, ParityMethod.MAX_DIV])
def test_expanding_covar_methods_skip_empty_initial_estimates(method: ParityMethod) -> None:
    """A panel beginning after the nominal start waits for an eligible covariance."""

    prices = _synthetic_prices()
    prices.index = pd.date_range("2018-02-28", periods=len(prices), freq="ME")
    config = Parity2024Config(
        estimation_start="2010-07-19",
        reporting_start="2016-03-31",
        end_date="2026-09-04",
    )
    first_covariance = next(iter(estimate_ewm_covariances_2024(prices, config).values()))
    np.testing.assert_allclose(np.diag(first_covariance), 0.0, rtol=0.0, atol=0.0)

    result = marginal_weights_2024(
        prices=prices,
        marginal_asset="BTC",
        method=method,
        is_alternatives=True,
        config=config,
    )

    assert not result.without_asset.empty
    assert not result.with_asset.empty
    assert result.without_asset.index[0] == pd.Timestamp("2018-06-30")
    assert result.with_asset.index[0] == pd.Timestamp("2018-06-30")
    assert result.without_asset.columns.equals(prices.columns.drop("BTC"))
    assert result.with_asset.columns.equals(prices.columns)
    np.testing.assert_allclose(result.without_asset.sum(axis=1), 1.0, rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(result.with_asset.sum(axis=1), 1.0, rtol=0.0, atol=2e-5)


@pytest.mark.parametrize("method", list(ParityMethod))
@pytest.mark.parametrize("is_alternatives", [True, False])
def test_all_methods_are_long_only_fully_invested_and_label_aligned(
    method: ParityMethod,
    is_alternatives: bool,
) -> None:
    """Synthetic replays preserve universe labels and full-investment constraints."""

    prices = _synthetic_prices()
    config = _synthetic_config()
    fitter = _deterministic_mixture if method is ParityMethod.CARA else None
    result = marginal_weights_2024(
        prices=prices,
        marginal_asset="BTC",
        method=method,
        is_alternatives=is_alternatives,
        config=config,
        mixture_fitter=fitter,
    )

    assert result.without_asset.columns.equals(prices.columns.drop("BTC"))
    assert result.with_asset.columns.equals(prices.columns)
    for weights in (result.without_asset, result.with_asset):
        assert not weights.empty
        assert np.isfinite(weights.to_numpy()).all()
        assert weights.min().min() >= -2e-5
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=2e-5)

    if not is_alternatives and method is not ParityMethod.ERC:
        np.testing.assert_allclose(
            result.without_asset.iloc[:, 0],
            config.first_asset_target_weight,
            rtol=0.0,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            result.with_asset.iloc[:, 0],
            config.first_asset_target_weight,
            rtol=0.0,
            atol=2e-5,
        )


def test_dependency_report_is_explicit_and_never_substitutes_a_solver() -> None:
    """Runtime metadata states the unpinned-history limitation and fails closed."""

    report = dependency_report((ParityMethod.MAX_SHARPE, ParityMethod.CARA))
    manifest = report.to_dict()
    assert manifest["reference_commit"].startswith("6038fba")
    assert manifest["exact_historical_versions_known"] is False
    assert set(("numpy", "pandas", "scipy", "qis", "cvxpy", "scikit-learn")) <= set(
        manifest["packages"]
    )

    if report.ready:
        assert require_parity_dependencies(
            (ParityMethod.MAX_SHARPE, ParityMethod.CARA)
        ).ready
    else:
        with pytest.raises(ParityDependencyError, match="not ready"):
            require_parity_dependencies((ParityMethod.MAX_SHARPE, ParityMethod.CARA))

    bad_config = _synthetic_config(max_sharpe_solver="NOT_A_REAL_SOLVER")
    with pytest.raises(ParityDependencyError, match="NOT_A_REAL_SOLVER"):
        marginal_weights_2024(
            prices=_synthetic_prices(),
            marginal_asset="BTC",
            method=ParityMethod.MAX_SHARPE,
            is_alternatives=True,
            config=bad_config,
        )


def test_golden_constants_reconstruct_the_archived_headline() -> None:
    """The four workbook F3 cells reproduce the reported 3.4% allocation."""

    assert len(GOLDEN_MEDIAN_WEIGHT_ROWS) == 4
    assert golden_headline_from_constants() == pytest.approx(
        GOLDEN_HEADLINE_MEDIAN, abs=5e-9
    )


@pytest.mark.skipif(
    not GOLDEN_WORKBOOK_PATH.is_file(),
    reason="The private archived 2024 workbook is not available on this host",
)
def test_private_archived_workbook_matches_the_golden_constants() -> None:
    """The private workbook checksum and exact B3:F3 cells have not changed."""

    observed = verify_golden_workbook()
    assert observed.keys() == GOLDEN_MEDIAN_WEIGHT_ROWS.keys()


_GOLDEN_RUNTIME_READY, _GOLDEN_RUNTIME_REASON = golden_runtime_available()
_RUN_GOLDEN_ENGINE = os.environ.get("RUN_CRYPTO_PARITY_GOLDEN") == "1"


@pytest.mark.skipif(
    not (_GOLDEN_RUNTIME_READY and _RUN_GOLDEN_ENGINE),
    reason=(
        _GOLDEN_RUNTIME_REASON
        if not _GOLDEN_RUNTIME_READY
        else "set RUN_CRYPTO_PARITY_GOLDEN=1 to run the private archived-panel replay"
    ),
)
def test_private_archived_panel_reproduces_all_golden_weight_medians() -> None:
    """Run all sixteen private panel/method combinations against the workbook oracle."""

    assert "ECOS_BB" in cvx.installed_solvers()
    panel = pd.read_csv(GOLDEN_PANEL_PATH, index_col=0, parse_dates=True)
    config = Parity2024Config(end_date="2024-08-16")
    scenarios = (
        ("BTC", True, "weight_(A) 100% Alts with BTC"),
        ("BTC", False, "weight_(C) 75%25% BalAlts with BTC"),
        ("ETH", True, "weight_(B) 100% Alts with ETH"),
        ("ETH", False, "weight_(D) 75%25% BalAlts with ETH"),
    )
    method_order = (
        ParityMethod.ERC,
        ParityMethod.MAX_DIV,
        ParityMethod.MAX_SHARPE,
        ParityMethod.CARA,
    )
    for crypto, is_alternatives, sheet in scenarios:
        other_crypto = "ETH" if crypto == "BTC" else "BTC"
        prices = panel.drop(columns=other_crypto).dropna()
        if is_alternatives:
            prices = prices.drop(columns="60/40")
        expected = GOLDEN_MEDIAN_WEIGHT_ROWS[sheet]
        observed: list[float] = []
        for method in method_order:
            result = marginal_weights_2024(
                prices=prices,
                marginal_asset=crypto,
                method=method,
                is_alternatives=is_alternatives,
                config=config,
            )
            observed.append(float(result.with_asset[crypto].median()))
        observed.append(float(np.median(observed)))
        # The archived environment was not locked.  ECOS_BB/CVXPY patch-level
        # drift moves the MaxSharpe median by about 3e-6 under the recorded
        # current runtime, while leaving the economic result unchanged.
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=3.1e-6)
