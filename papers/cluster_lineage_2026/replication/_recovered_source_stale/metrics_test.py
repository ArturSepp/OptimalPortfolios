"""Regression tests for the stage-E0 empirical metric library."""
from __future__ import annotations

import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication import metrics
from papers.cluster_lineage_2026.replication.sp500_baseline import (
    estimate_rolling,
    extract_partitions,
    load_inputs,
)


def test_partition_distance_identities() -> None:
    """Verify identity and crossed-partition ARI/VI formulas."""
    labels = pd.Series([0, 0, 1, 1])
    crossed = pd.Series([0, 1, 0, 1])
    assert metrics.adjusted_rand_index(labels, labels) == 1.0
    assert metrics.variation_of_information(labels, labels) == 0.0
    assert np.isclose(metrics.variation_of_information(labels, crossed), 2.0 * np.log(2.0))


def test_frozen_sp500_regression_and_determinism() -> None:
    """Reproduce the six frozen S&P metrics and stable metric-suite bytes."""
    covar_data = estimate_rolling("baseline")
    partitions = extract_partitions(covar_data)
    metadata = load_inputs()["metadata"]
    lineage_first, _ = metrics.lineage_metrics(covar_data)
    lineage_second, _ = metrics.lineage_metrics(covar_data)
    ari_first, _ = metrics.ari_metrics(partitions, metadata)
    ari_second, _ = metrics.ari_metrics(partitions, metadata)
    first = {**lineage_first, **ari_first}
    second = {**lineage_second, **ari_second}

    assert abs(first["lineage_churn_panel"] - 3.2115) <= 0.0001
    assert first["n_derived_tracks"] == 216
    assert abs(first["matcher_attributable_churn"] - 0.486) <= 0.005
    assert abs(first["ari_sector"] - 0.202967) <= 0.005
    assert abs(first["ari_industry_group"] - 0.297012) <= 0.005
    assert abs(first["ari_industry"] - 0.331935) <= 0.005
    assert metrics.deterministic_metric_bytes(first) == metrics.deterministic_metric_bytes(second)
