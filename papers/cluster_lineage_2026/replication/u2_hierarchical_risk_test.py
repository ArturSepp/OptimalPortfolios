"""Focused checks for the U2 Rolling-Ward risk-allocation experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlasso.cluster_smoothing import _iter_correlation_inputs

import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_hierarchical_risk as run


def _correlation(covar: pd.DataFrame) -> pd.DataFrame:
    """Normalize one positive-diagonal covariance matrix to correlation."""
    diagonal = np.diag(covar.to_numpy(dtype=float))
    inverse = 1.0 / np.sqrt(diagonal)
    values = covar.to_numpy(dtype=float) * np.outer(inverse, inverse)
    return pd.DataFrame(values, index=covar.index, columns=covar.columns)


def test_rolling_covariance_is_the_exact_factorlasso_correlation_input() -> None:
    """The retained covariance must normalize to FactorLasso's frozen input exactly."""
    index = pd.date_range("2020-01-02", periods=24, freq="W-THU")
    values = np.column_stack(
        [
            np.linspace(-0.02, 0.03, len(index)),
            np.sin(np.arange(len(index)) / 3.0) / 50.0,
            np.cos(np.arange(len(index)) / 5.0) / 40.0,
        ]
    )
    values[:4, 2] = np.nan
    values[11, 1] = np.nan
    returns = pd.DataFrame(values, index=index, columns=["a", "b", "c"])
    dates = pd.DatetimeIndex(index[[8, 14, 23]])
    model = funds._model(24, "W-THU")

    covariances = dict(run._iter_covariance_inputs(returns, dates, model))
    correlations = dict(_iter_correlation_inputs(returns, list(dates), model))
    for date in dates:
        np.testing.assert_allclose(
            _correlation(covariances[date]), correlations[date], atol=0.0, rtol=0.0
        )


def test_u2_paper_method_set_excludes_herc() -> None:
    """The owner-excluded HERC variant must not be computed or reported for U2."""
    assert "ward_herc" not in run.METHODS
    assert tuple(run.PAPER_LONG_ONLY_METHODS) == (
        "flat_erc",
        "single_hrp",
        "ward_hrp",
    )


def test_u2_conditioning_caps_only_tiny_eigenvalues() -> None:
    """The common risk matrix must obey the frozen condition-number cap."""
    covar = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-12]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    conditioned, diagnostics = run._condition_covariance(covar)
    assert diagnostics["conditioned_condition_number"] <= run.MAX_CONDITION_NUMBER * (1.0 + 1e-12)
    assert diagnostics["conditioned_eigenvalues_floored"] == 1
    np.testing.assert_allclose(conditioned.loc[["a", "b"], ["a", "b"]], [[1.0, 0.0], [0.0, 1e-4]])
