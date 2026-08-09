"""Numerical and behavioural tests for persistent risk-cluster labelling."""

from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment

from factorlasso import CurrentFactorCovarData, RollingFactorCovarData

from optimalportfolios.covar_estimation.risk_labelling import (
    RiskClusterReport,
    TaxonomyConfig,
    _Fingerprint,
    _overlap,
    _psd_clip,
    _qualifies,
    _snapshot_fingerprints,
    analyze_risk_clusters,
    run_risk_label_report,
)


FACTORS = ("Equity", "Rates", "Credit")
SIGMA = pd.DataFrame(
    np.diag([0.04, 0.01, 0.0225]),
    index=FACTORS,
    columns=FACTORS,
)
BETAS = pd.DataFrame(
    {
        "Equity": [1.00, 0.90, 0.05, 0.10, 0.15, 0.20],
        "Rates": [0.05, 0.10, 1.00, 0.90, 0.15, 0.10],
        "Credit": [0.10, 0.05, 0.10, 0.15, 1.00, 0.90],
    },
    index=list("ABCDEF"),
)


def _make_panel(assignments: list[dict[str, object]], *, permute_assets: bool = False,
                seed: int = 7) -> RollingFactorCovarData:
    """Build seeded factor-covariance snapshots from asset-to-cluster assignments."""
    rng = np.random.default_rng(seed)
    data = {}
    for i, assignment in enumerate(assignments):
        date = pd.Timestamp("2024-01-31") + pd.offsets.MonthEnd(i)
        assets = list(assignment)
        if permute_assets:
            assets = list(rng.permutation(assets))
        betas = BETAS.loc[assets].copy()
        variances = pd.DataFrame({"residual_var": 0.0025}, index=assets)
        clusters = pd.Series({asset: assignment[asset] for asset in assets}, name="cluster")
        data[date] = CurrentFactorCovarData(
            x_covar=SIGMA.copy(),
            y_betas=betas,
            y_variances=variances,
            estimation_date=date,
            clusters=clusters,
        )
    return RollingFactorCovarData(data=data)


@pytest.fixture
def persistent_panel() -> RollingFactorCovarData:
    """Return four dates with three uniquely persistent two-asset clusters."""
    return _make_panel(
        [
            {"A": 10, "B": 10, "C": 20, "D": 20, "E": 30, "F": 30},
            {"A": 3, "B": 3, "C": 1, "D": 1, "E": 2, "F": 2},
            {"A": 8, "B": 8, "C": 9, "D": 9, "E": 7, "F": 7},
            {"A": 2, "B": 2, "C": 3, "D": 3, "E": 1, "F": 1},
        ]
    )


def _hungarian(panel: RollingFactorCovarData) -> RiskClusterReport:
    """Run the dependency-free matcher with strict, uniquely identifying overlap links."""
    return analyze_risk_clusters(
        panel,
        method="hungarian",
        overlap_band=(0.20, 0.60),
        spread_vol_cut=0.001,
        bridge_window=1,
    )


def test_hungarian_end_to_end_and_report_views(
        persistent_panel: RollingFactorCovarData) -> None:
    """Hungarian tracking produces stable panels, labels, tables, and figures."""
    report = _hungarian(persistent_panel)

    assert report.relabel.shape == (12, 3)
    assert report.relabel["derived_id"].nunique() == 3
    assert set(report.classification["persistence"]) == {"Core"}
    assert set(report.lineage["event"]) == {"birth", "continue"}
    assert report.transitions.empty

    membership = report.to_membership_panel()
    assert membership.shape == (4, 6)
    assert membership.nunique().eq(1).all()
    factor_labels = report.factor_labels()
    assert factor_labels.str.contains("vol").all()
    assert report.to_label_panel().notna().all().all()

    metadata = pd.DataFrame(
        {
            "Sub Asset Class": ["Equities", "Equities", "Rates", "Rates", "Credit", "Credit"],
            "Asset Class": ["Equity", "Equity", "Rates", "Rates", "Credit", "Credit"],
        },
        index=list("ABCDEF"),
    )
    meta_labels = report.label_tracks(metadata)
    assert meta_labels.index.equals(factor_labels.index)
    assert report.to_label_panel("meta", metadata).notna().all().all()
    assert report._labels("id", None).to_dict() == {track: track for track in report.tracks}
    with pytest.raises(ValueError, match="requires asset_meta"):
        report._labels("meta", None)
    with pytest.raises(ValueError, match="unknown label_kind"):
        report._labels("unknown", None)

    first_date = membership.index[0]
    assert report.labels_at(first_date).notna().all()
    assert report.labels_at(first_date + pd.Timedelta(days=5)).equals(
        report.labels_at(first_date)
    )
    assert report.labels_at(first_date - pd.Timedelta(days=5)).empty
    assert set(report.to_tables()) == {
        "classification", "membership_panel", "lineage", "transitions", "relabel"
    }

    figures = report.to_figures()
    assert len(figures) == 2
    for figure in figures:
        plt.close(figure)


def test_run_report_wrapper_returns_figures_and_tables(
        persistent_panel: RollingFactorCovarData) -> None:
    """The public convenience wrapper returns the same two report artifact groups."""
    figures, tables = run_risk_label_report(persistent_panel, method="hungarian")
    assert len(figures) == 2
    assert "classification" in tables
    for figure in figures:
        plt.close(figure)


def test_mcf_and_hungarian_agree_for_unique_matching(
        persistent_panel: RollingFactorCovarData) -> None:
    """The global and sequential matchers agree when every optimal link is unique."""
    pytest.importorskip("networkx")

    hungarian = _hungarian(persistent_panel)
    mcf = analyze_risk_clusters(
        persistent_panel,
        method="mcf",
        overlap_band=(0.20, 0.60),
        spread_vol_cut=0.001,
        bridge_window=1,
    )

    pd.testing.assert_frame_equal(mcf.to_membership_panel(), hungarian.to_membership_panel())
    pd.testing.assert_series_equal(mcf.factor_labels(), hungarian.factor_labels())


def test_assignment_matches_scipy_and_brute_force_references() -> None:
    """Match SciPy linear_sum_assignment and an independent n<=6 brute-force optimum."""
    panel = _make_panel(
        [
            {"A": 0, "B": 1, "C": 2},
            {"A": 1, "B": 2, "C": 0},
        ]
    )
    report = _hungarian(panel)
    dates = list(panel.dates)
    raw_members = []
    for date in dates:
        clusters = panel[date].clusters
        raw_members.append(
            {label: set(group.index) for label, group in clusters.groupby(clusters)}
        )
    left = sorted(raw_members[0])
    right = sorted(raw_members[1])
    cost = np.array(
        [
            [1.0 - len(raw_members[0][a] & raw_members[1][b]) for b in right]
            for a in left
        ]
    )

    scipy_rows, scipy_cols = linear_sum_assignment(cost)
    scipy_mapping = {left[row]: right[col] for row, col in zip(scipy_rows, scipy_cols)}
    candidates = [
        (sum(cost[row, col] for row, col in enumerate(order)), order)
        for order in permutations(range(len(right)))
    ]
    candidates.sort(key=lambda item: item[0])
    assert candidates[0][0] < candidates[1][0]
    brute_mapping = {left[row]: right[col] for row, col in enumerate(candidates[0][1])}

    first = report.relabel.loc[report.relabel["date"] == dates[0]].set_index("derived_id")
    second = report.relabel.loc[report.relabel["date"] == dates[1]].set_index("derived_id")
    produced = {
        first.loc[track, "raw_label"]: second.loc[track, "raw_label"]
        for track in first.index
    }
    assert produced == scipy_mapping == brute_mapping == {0: 1, 1: 2, 2: 0}


def test_permutation_invariance_and_fixed_seed_determinism() -> None:
    """Asset-row permutations preserve outputs, and a fixed seed is deterministic."""
    assignments = [
        {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2, "F": 2},
        {"A": 2, "B": 2, "C": 0, "D": 0, "E": 1, "F": 1},
        {"A": 1, "B": 1, "C": 2, "D": 2, "E": 0, "F": 0},
    ]
    ordered = _hungarian(_make_panel(assignments))
    permuted_one = _hungarian(_make_panel(assignments, permute_assets=True, seed=36))
    permuted_two = _hungarian(_make_panel(assignments, permute_assets=True, seed=36))

    pd.testing.assert_frame_equal(
        ordered.to_membership_panel(), permuted_one.to_membership_panel()
    )
    pd.testing.assert_frame_equal(permuted_one.relabel, permuted_two.relabel)
    pd.testing.assert_frame_equal(permuted_one.lineage, permuted_two.lineage)
    pd.testing.assert_frame_equal(permuted_one.classification, permuted_two.classification)


def test_degenerate_single_cluster_has_documented_track_result() -> None:
    """A single raw cluster remains one fully covered derived track."""
    panel = _make_panel(
        [
            {"A": "only", "B": "only"},
            {"A": "renamed", "B": "renamed"},
            {"A": "last", "B": "last"},
        ]
    )
    report = _hungarian(panel)

    assert report.relabel["derived_id"].nunique() == 1
    assert report.classification.iloc[0]["coverage"] == 1.0
    assert report.classification.iloc[0]["lifetime"] == 3


def test_disjoint_asset_sets_produce_explicit_death_and_birth() -> None:
    """Disjoint dates with distant betas retire one track and birth another."""
    panel = _make_panel([{"A": 0}, {"D": 0}])
    report = _hungarian(panel)

    assert report.relabel["derived_id"].nunique() == 2
    assert set(report.lineage["event"]) == {"birth", "death"}
    assert report.to_membership_panel().isna().sum().sum() == 2


def test_fingerprint_weighting_affinity_and_psd_helpers() -> None:
    """Helper branches cover inverse-vol weights, overlap metrics, gates, and PSD clipping."""
    panel = _make_panel([{"A": 0, "B": 0, "C": 1}])
    snapshot = panel[panel.dates[0]]
    fingerprints, factors = _snapshot_fingerprints(snapshot, weighting="inv_vol")

    assert factors == list(FACTORS)
    assert set(fingerprints) == {0, 1}
    assert all(fingerprint.total_var > 0.0 for fingerprint in fingerprints.values())
    assert _overlap(("A", "B"), ("B", "C"), {"A", "B", "C"}, "jaccard") == 1 / 3
    assert _overlap(("A", "B"), ("B", "C"), {"A", "B", "C"}, "overlap") == 0.5
    assert _overlap(("A",), ("B",), set(), "overlap") == 0.0

    clipped = _psd_clip(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert np.linalg.eigvalsh(clipped).min() >= -1e-12
    fp_a = _Fingerprint(("A",), np.array([1.0, 0.0]), 0.04, 0.01, 0.05, 0.8, "Equity")
    fp_b = _Fingerprint(("A",), np.array([0.9, 0.0]), 0.03, 0.01, 0.04, 0.75, "Equity")
    qualifies, weight = _qualifies(
        fp_a,
        fp_b,
        {"A"},
        np.eye(2) * 0.01,
        overlap_metric="overlap",
        combine="blend",
        overlap_band=(0.20, 0.60),
        spread_vol_cut=0.025,
        w_overlap=0.6,
    )
    assert qualifies and weight >= 0.5
    qualifies, weight = _qualifies(
        fp_a,
        _Fingerprint(("B",), -fp_a.beta, 0.04, 0.01, 0.05, 0.8, "Equity"),
        {"A", "B"},
        np.eye(2),
        overlap_metric="overlap",
        combine="gated",
        overlap_band=(0.20, 0.60),
        spread_vol_cut=0.025,
        w_overlap=0.6,
    )
    assert not qualifies and weight == 0.0


def test_custom_taxonomy_exercises_classification_thresholds(
        persistent_panel: RollingFactorCovarData) -> None:
    """Explicit taxonomy thresholds are retained and applied by the public API."""
    taxonomy = TaxonomyConfig(core_coverage=0.75, vol_low=0.01, vol_high=0.05)
    report = analyze_risk_clusters(
        persistent_panel,
        method="hungarian",
        taxonomy=taxonomy,
        weighting="equal",
    )

    assert report.params["method"] == "hungarian"
    assert set(report.classification["persistence"]) == {"Core"}
    assert set(report.classification["vol_regime"]) <= {"Mid", "High"}
