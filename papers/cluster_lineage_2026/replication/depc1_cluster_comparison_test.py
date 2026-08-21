"""Focused tests for the isolated de-PC1 partition experiment."""

import numpy as np
import pandas as pd
from factorlasso import LassoModel, LassoModelType

import papers.cluster_lineage_2026.replication.run_depc1_cluster_comparison as run


def test_partition_comparators_are_label_invariant():
    """ARI, Rand, and exact partition checks must ignore arbitrary label names."""
    left = pd.Series([1, 1, 2, 2], index=list("abcd"))
    right = pd.Series([9, 9, 4, 4], index=list("abcd"))
    assert run._same_partition(left, right)
    assert run._adjusted_rand(left, right) == 1.0
    assert run._pairwise_rand(left, right) == 1.0


def test_partition_comparators_require_identical_assets_for_exact_match():
    """The exact comparator must reject a silently missing asset."""
    left = pd.Series([1, 1, 2], index=list("abc"))
    right = pd.Series([1, 1], index=list("ab"))
    assert not run._same_partition(left, right)


def test_offdiagonal_uses_one_finite_triangle():
    """Correlation summaries must neither double-count nor retain missing pairs."""
    values = np.array([[1.0, 0.2, np.nan], [0.2, 1.0, -0.4], [np.nan, -0.4, 1.0]])
    np.testing.assert_array_equal(run._offdiagonal(values), [0.2, -0.4])


def test_required_isolated_cache_shape(tmp_path, monkeypatch):
    """Cache paths must follow dePC1/universe/arm/config/YYYYMMDD.pkl."""
    monkeypatch.setenv("CLUSTER_LINEAGE_OUTPUT_DIR", str(tmp_path))
    inputs = run.UniverseInputs(
        universe="synthetic",
        returns=pd.DataFrame(),
        dates=pd.DatetimeIndex(["2026-06-30"]),
        eligibility=pd.DataFrame(),
        model=object(),
        taxonomy={},
        frozen_panel=pd.DataFrame(),
        config_id="ME_span_036",
        input_paths=(),
    )
    path = run._cache_path(inputs, "depc1", pd.Timestamp("2026-06-30"))
    assert path.relative_to(tmp_path).as_posix() == (
        "depc1/synthetic/depc1/ME_span_036/20260630.pkl"
    )


def test_external_injection_validation_uses_estimator_preparation_path():
    """The injected/fitted gate must exercise FactorLasso's preparation path."""
    model = LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        span=12,
    )
    clusters = pd.Series([0, 0, 1, 1], index=list("abcd"))
    linkage = np.zeros((len(clusters) - 1, 4))
    assert run._injected_partition_matches(model, clusters, linkage, 0.5)
