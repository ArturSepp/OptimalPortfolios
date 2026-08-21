"""Unit checks for the paper's cluster-aware risk allocation methods."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from papers.cluster_lineage_2026.replication.hierarchical_risk_allocations import (
    cluster_risk_budget,
    herc_volatility_weights,
    hrp_weights,
    risk_contribution_summary,
)


ASSETS = pd.Index(["a", "b", "c", "d", "e", "f"])
VOLS = np.array([0.10, 0.12, 0.18, 0.20, 0.25, 0.30])
CORR = np.array(
    [
        [1.00, 0.80, 0.10, 0.05, 0.00, 0.00],
        [0.80, 1.00, 0.10, 0.05, 0.00, 0.00],
        [0.10, 0.10, 1.00, 0.70, 0.05, 0.00],
        [0.05, 0.05, 0.70, 1.00, 0.05, 0.00],
        [0.00, 0.00, 0.05, 0.05, 1.00, 0.60],
        [0.00, 0.00, 0.00, 0.00, 0.60, 1.00],
    ]
)
COVAR = pd.DataFrame(np.outer(VOLS, VOLS) * CORR, index=ASSETS, columns=ASSETS)
DISTANCE = np.sqrt(np.clip(2.0 * (1.0 - CORR), 0.0, None))
LINKAGE = linkage(squareform(DISTANCE, checks=False), method="ward")
CLUSTERS = pd.Series(fcluster(LINKAGE, 3, criterion="maxclust"), index=ASSETS)


def test_cluster_budget_exponent_one_is_equal_asset_budget() -> None:
    """Alpha one must be the exact flat equal-asset risk-budget control."""
    actual = cluster_risk_budget(CLUSTERS, cluster_size_exponent=1.0)
    np.testing.assert_allclose(actual.to_numpy(), np.full(len(ASSETS), 1.0 / len(ASSETS)))


@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0])
def test_cluster_budget_matches_the_stated_group_formula(exponent: float) -> None:
    """Asset budgets must aggregate to n_g**alpha divided by its cross-group sum."""
    actual = cluster_risk_budget(CLUSTERS, cluster_size_exponent=exponent)
    sizes = CLUSTERS.value_counts().sort_index().astype(float)
    expected = sizes.pow(exponent) / sizes.pow(exponent).sum()
    grouped = actual.groupby(CLUSTERS).sum().sort_index()
    pd.testing.assert_series_equal(grouped, expected, check_names=False, atol=1e-15)
    assert actual.sum() == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("allocator", [hrp_weights, herc_volatility_weights])
def test_hierarchical_allocators_are_long_only_and_fully_invested(allocator) -> None:
    """Both literature allocators must return deterministic long-only unit weights."""
    kwargs = {"clusters": CLUSTERS} if allocator is herc_volatility_weights else {}
    first = allocator(COVAR, LINKAGE, **kwargs)
    second = allocator(COVAR, LINKAGE, **kwargs)
    pd.testing.assert_series_equal(first, second, check_exact=True)
    assert first.sum() == pytest.approx(1.0, abs=1e-15)
    assert first.min() >= 0.0
    assert first.index.equals(ASSETS)


def test_risk_summary_reconciles_euler_contributions() -> None:
    """Reported asset and cluster contributions must add back to portfolio volatility."""
    weights = hrp_weights(COVAR, LINKAGE)
    summary = risk_contribution_summary(weights, COVAR, CLUSTERS)
    assert summary["risk_contribution"].sum() == pytest.approx(
        summary["portfolio_volatility"].iloc[0], abs=1e-14
    )
    assert summary["risk_share"].sum() == pytest.approx(1.0, abs=1e-14)
    assert summary["absolute_risk_share"].sum() == pytest.approx(1.0, abs=1e-14)

