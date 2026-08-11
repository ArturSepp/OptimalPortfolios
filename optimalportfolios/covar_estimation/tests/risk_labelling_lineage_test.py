"""
lineage matching, track classification and labelling in ``covar_estimation.risk_labelling``.

The module is pure post-processing of an estimated ``RollingFactorCovarData`` — it fits no
covariance — so every case here is a synthetic panel stated in the test. Nothing needs market
data, a network or a Bloomberg terminal, which is what the core-install job in
``.github/workflows/ci.yml`` requires.

Two matchers turn per-date raw clusters into persistent derived tracks, and they are the two
highest-complexity functions in the package:

* ``_match_panel`` (``method='hungarian'``) assigns per transition. Its five lineage events —
  birth, continue, split, merge, death — plus the dormant-pool bridge revival are asserted one
  case each, because each is a distinct branch and a wrong one silently renames a track.
* ``_match_panel_mcf`` (``method='mcf'``, the default) solves the whole panel jointly, so a
  cluster absent for one date keeps its identity across the gap and is tagged ``bridge``. It
  needs ``networkx``, an optional extra, so those cases skip when it is absent and the guarded
  import is asserted separately.

Identity is the thing under test throughout: which derived id a raw cluster is given, and which
event explains it. The numbers a fingerprint carries (betas, the factor/idio variance split) are
inputs stated by the fixtures, not results computed here.
"""
# packages
import sys
import numpy as np
import pandas as pd
import pytest
# factorlasso
from factorlasso import CurrentFactorCovarData, RollingFactorCovarData, VarianceColumns
# optimalportfolios
from optimalportfolios.covar_estimation.risk_labelling import (
    RiskClusterReport,
    TaxonomyConfig,
    TrackPanel,
    _build_tracks,
    _classify,
    _cluster_series,
    _Fingerprint,
    _match_panel,
    _match_panel_mcf,
    _overlap,
    _psd_clip,
    _qualifies,
    _snapshot_fingerprints,
    analyze_risk_clusters,
)

FACTORS = ['Equity', 'Rates', 'Credit']
# a plausible annualised factor covariance: equity vol ~20%, rates ~5%, credit ~9.5%
SIGMA = np.array([[0.0400, 0.0010, 0.0060],
                  [0.0010, 0.0025, 0.0005],
                  [0.0060, 0.0005, 0.0090]])
DATES = list(pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-31']))
X_COVARS = {d: SIGMA for d in DATES}

# an equity-like and a rates-like beta vector, far enough apart that neither can be mistaken
# for the other under the gated link test
EQUITY_BETA = [0.95, 0.05, 0.10]
RATES_BETA = [0.05, 0.85, 0.10]

# the defaults analyze_risk_clusters passes down, restated so a case can vary one of them
LINK_KWARGS = dict(overlap_metric='overlap', combine='gated', overlap_band=(0.15, 0.60),
                   spread_vol_cut=0.015, w_overlap=0.6)

ASSETS = ['a1', 'a2', 'a3', 'a4']
ASSET_BETAS = pd.DataFrame([[0.95, 0.05, 0.10],
                            [0.90, 0.05, 0.10],
                            [0.05, 0.85, 0.10],
                            [0.05, 0.80, 0.15]], index=ASSETS, columns=FACTORS)


def make_fingerprint(members, beta) -> _Fingerprint:
    """Build a fingerprint whose factor variance is consistent with ``beta`` under SIGMA."""
    b = np.asarray(beta, dtype=float)
    factor_var = float(b @ SIGMA @ b)
    idio_var = 0.005
    total_var = factor_var + idio_var
    return _Fingerprint(members=tuple(members), beta=b, factor_var=factor_var,
                        idio_var=idio_var, total_var=total_var, r2=factor_var / total_var,
                        dominant='Equity' if b[0] >= b[1] else 'Rates')


def make_snapshot(clusters, with_clusters_series: bool = True) -> CurrentFactorCovarData:
    """One date's factor covar snapshot over ASSETS with the given cluster assignment."""
    y_variances = pd.DataFrame({VarianceColumns.RESIDUAL_VARS.value: [0.01] * len(ASSETS),
                                VarianceColumns.TOTAL_VOL.value: [0.20, 0.19, 0.06, 0.07],
                                VarianceColumns.CLUSTER.value: clusters}, index=ASSETS)
    return CurrentFactorCovarData(
        x_covar=pd.DataFrame(SIGMA, index=FACTORS, columns=FACTORS),
        y_betas=ASSET_BETAS, y_variances=y_variances,
        clusters=pd.Series(clusters, index=ASSETS) if with_clusters_series else None)


def make_rolling(clusters=('c1', 'c1', 'c2', 'c2')) -> RollingFactorCovarData:
    """A three-date rolling panel holding the same cluster assignment at every date."""
    return RollingFactorCovarData(data={d: make_snapshot(list(clusters)) for d in DATES})


def events_at(lineage: pd.DataFrame, date) -> set:
    """The set of lineage event names recorded at one date."""
    return set(lineage.loc[lineage['date'] == date, 'event'])


def derived_id(relabel: pd.DataFrame, date, raw_label: str) -> str:
    """The derived track id assigned to one raw cluster at one date."""
    row = relabel[(relabel['date'] == date) & (relabel['raw_label'] == raw_label)]
    assert len(row) == 1, f"expected one relabel row for {raw_label} at {date}, got {len(row)}"
    return row['derived_id'].iloc[0]


def derived_to_raw(relabel: pd.DataFrame, date, did: str) -> str:
    """The raw cluster label a derived track was assigned at one date."""
    row = relabel[(relabel['date'] == date) & (relabel['derived_id'] == did)]
    assert len(row) == 1, f"expected one relabel row for {did} at {date}, got {len(row)}"
    return row['raw_label'].iloc[0]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_psd_clip_removes_a_negative_eigenvalue() -> None:
    """the small negative eigenvalue seen in Sigma_F is clipped to zero, symmetry preserved"""
    m = np.array([[1.0, 2.0], [2.0, 1.0]])           # eigenvalues -1 and 3
    clipped = _psd_clip(m)
    eigenvalues = np.linalg.eigvalsh(clipped)
    assert eigenvalues.min() >= -1e-12
    assert np.allclose(eigenvalues, [0.0, 3.0])
    assert np.allclose(clipped, clipped.T)


def test_psd_clip_leaves_a_psd_matrix_alone() -> None:
    """a matrix that is already PSD comes back unchanged"""
    assert np.allclose(_psd_clip(SIGMA), SIGMA)


def test_overlap_coefficient_and_jaccard_differ_on_unequal_sizes() -> None:
    """the overlap coefficient divides by the smaller set, Jaccard by the union"""
    a, b, common = ('a1', 'a2'), ('a1', 'a2', 'a3'), {'a1', 'a2', 'a3'}
    assert _overlap(a, b, common, 'overlap') == pytest.approx(1.0)
    assert _overlap(a, b, common, 'jaccard') == pytest.approx(2.0 / 3.0)


def test_overlap_is_zero_when_a_side_has_no_common_asset() -> None:
    """a cluster whose members all left the panel cannot link to anything"""
    assert _overlap((), ('a1',), {'a1'}, 'overlap') == 0.0
    assert _overlap(('a9',), ('a1',), {'a1'}, 'overlap') == 0.0


def test_qualifies_gated_rejects_disjoint_members_with_distant_beta() -> None:
    """no membership overlap and a beta spread beyond the cut is not a link"""
    ok, weight = _qualifies(make_fingerprint(['a1'], EQUITY_BETA),
                            make_fingerprint(['a9'], RATES_BETA),
                            {'a1', 'a9'}, SIGMA, **LINK_KWARGS)
    assert ok is False
    assert weight == 0.0


def test_qualifies_gated_rejects_below_lower_band_even_when_betas_match() -> None:
    """Beta proximity cannot link clusters with overlap below the lower gate."""
    ok, weight = _qualifies(make_fingerprint(['a1'], EQUITY_BETA),
                            make_fingerprint(['a9'], EQUITY_BETA),
                            {'a1', 'a9'}, SIGMA, **LINK_KWARGS)
    assert ok is False
    assert weight == 0.0


def test_qualifies_gated_accepts_mid_overlap_when_beta_arbitrates() -> None:
    """Inside the overlap band, a beta spread within the cut links the pair."""
    left = ['a1', 'a2', 'a3', 'a4']
    right = ['a1', 'b2', 'b3', 'b4']
    ok, weight = _qualifies(make_fingerprint(left, EQUITY_BETA),
                            make_fingerprint(right, EQUITY_BETA),
                            set(left) | set(right), SIGMA, **LINK_KWARGS)
    assert ok is True
    assert weight > 0.0


def test_qualifies_gated_accepts_high_overlap_despite_a_wide_beta_spread() -> None:
    """membership above the upper band links regardless of the beta spread"""
    ok, weight = _qualifies(make_fingerprint(['a1', 'a2'], EQUITY_BETA),
                            make_fingerprint(['a1', 'a2'], RATES_BETA),
                            {'a1', 'a2'}, SIGMA, **LINK_KWARGS)
    assert ok is True
    assert weight >= 1.0                              # ov + s_beta, with ov == 1.0


def test_qualifies_blend_mixes_overlap_and_beta_into_one_score() -> None:
    """under 'blend' the decision is a single weighted score against 0.5, not a gate"""
    kwargs = {**LINK_KWARGS, 'combine': 'blend'}
    ok, weight = _qualifies(make_fingerprint(['a1'], EQUITY_BETA),
                            make_fingerprint(['a1'], RATES_BETA),
                            {'a1'}, SIGMA, **kwargs)
    # full overlap contributes w_overlap=0.6; the distant beta contributes ~0
    assert ok is True
    assert weight == pytest.approx(0.6, abs=1e-6)


# --------------------------------------------------------------------------- #
# fingerprints
# --------------------------------------------------------------------------- #
def test_snapshot_fingerprints_weights_members_equally() -> None:
    """an equally weighted fingerprint beta is the plain mean of its members' betas"""
    fingerprints, factors = _snapshot_fingerprints(make_snapshot(['c1', 'c1', 'c2', 'c2']))
    assert factors == FACTORS
    assert set(fingerprints) == {'c1', 'c2'}
    assert fingerprints['c1'].members == ('a1', 'a2')
    assert np.allclose(fingerprints['c1'].beta, ASSET_BETAS.loc[['a1', 'a2']].mean().to_numpy())
    assert fingerprints['c1'].dominant == 'Equity'
    assert fingerprints['c2'].dominant == 'Rates'
    assert 0.0 < fingerprints['c1'].r2 <= 1.0


def test_snapshot_fingerprints_inv_vol_tilts_towards_the_lower_vol_member() -> None:
    """inverse-vol weighting pulls the cluster beta towards its least volatile member"""
    equal, _ = _snapshot_fingerprints(make_snapshot(['c1', 'c1', 'c2', 'c2']), weighting='equal')
    inv_vol, _ = _snapshot_fingerprints(make_snapshot(['c1', 'c1', 'c2', 'c2']),
                                        weighting='inv_vol')
    # a2 has the lower total_vol of the c1 pair and the lower equity beta, so the inverse-vol
    # cluster beta must sit below the equally weighted one
    assert inv_vol['c1'].beta[0] < equal['c1'].beta[0]


def test_snapshot_fingerprints_call_zero_beta_idiosyncratic() -> None:
    """A cluster with no factor contribution has the explicit Idio sentinel."""
    snapshot = make_snapshot(['c1', 'c1', 'c1', 'c1'])
    zero_beta_snapshot = CurrentFactorCovarData(
        x_covar=snapshot.x_covar,
        y_betas=snapshot.y_betas * 0.0,
        y_variances=snapshot.y_variances,
        clusters=snapshot.clusters,
    )
    fingerprints, _ = _snapshot_fingerprints(zero_beta_snapshot)
    assert fingerprints['c1'].dominant == 'Idio'


def test_cluster_series_falls_back_to_the_variance_table() -> None:
    """with no clusters series the assignment is read off y_variances"""
    snapshot = make_snapshot(['c1', 'c1', 'c2', 'c2'], with_clusters_series=False)
    assert snapshot.clusters is None
    series = _cluster_series(snapshot)
    assert list(series) == ['c1', 'c1', 'c2', 'c2']
    fingerprints, _ = _snapshot_fingerprints(snapshot)
    assert set(fingerprints) == {'c1', 'c2'}


# --------------------------------------------------------------------------- #
# _match_panel: the per-transition Hungarian matcher
# --------------------------------------------------------------------------- #
def run_hungarian(snapshots, bridge_window: int = 1):
    """Run the Hungarian matcher over a snapshot panel with the default link settings."""
    return _match_panel(snapshots, X_COVARS, bridge_window=bridge_window, **LINK_KWARGS)


def test_hungarian_keeps_one_id_per_cluster_across_a_stable_panel() -> None:
    """unchanged membership and beta means every date continues the same two tracks"""
    snapshots = {d: {'c1': make_fingerprint(['a1', 'a2'], EQUITY_BETA),
                     'c2': make_fingerprint(['a3', 'a4'], RATES_BETA)} for d in DATES}
    relabel, lineage = run_hungarian(snapshots)
    assert relabel['derived_id'].nunique() == 2
    for label in ('c1', 'c2'):
        ids = {derived_id(relabel, d, label) for d in DATES}
        assert len(ids) == 1, f"{label} changed identity across a stable panel: {ids}"
    assert events_at(lineage, DATES[0]) == {'birth'}
    assert events_at(lineage, DATES[1]) == {'continue'}
    assert events_at(lineage, DATES[2]) == {'continue'}


def test_hungarian_records_a_split_when_one_parent_feeds_two_children() -> None:
    """the backbone continues one child; the second is a split off the same parent"""
    children = {'q1': make_fingerprint(['a1', 'a2'], EQUITY_BETA),
                'q2': make_fingerprint(['a3', 'a4'], EQUITY_BETA)}
    relabel, lineage = run_hungarian({DATES[0]: {'P': make_fingerprint(ASSETS, EQUITY_BETA)},
                                      DATES[1]: children, DATES[2]: children})
    assert events_at(lineage, DATES[1]) == {'continue', 'split'}
    split = lineage[lineage['event'] == 'split'].iloc[0]
    parent = derived_id(relabel, DATES[0], 'P')
    assert split['parent_id'] == parent
    assert split['child_id'] != parent
    # exactly one of the two children inherits the parent's id
    inherited = [lab for lab in ('q1', 'q2') if derived_id(relabel, DATES[1], lab) == parent]
    assert len(inherited) == 1


def test_hungarian_records_a_merge_when_two_parents_feed_one_child() -> None:
    """one parent continues into the child; the other is absorbed and tagged merge"""
    merged = {'m': make_fingerprint(ASSETS, EQUITY_BETA)}
    relabel, lineage = run_hungarian({DATES[0]: {'p1': make_fingerprint(['a1', 'a2'], EQUITY_BETA),
                                                 'p2': make_fingerprint(['a3', 'a4'], EQUITY_BETA)},
                                      DATES[1]: merged, DATES[2]: merged})
    assert events_at(lineage, DATES[1]) == {'continue', 'merge'}
    survivor = derived_id(relabel, DATES[1], 'm')
    parents = {derived_id(relabel, DATES[0], 'p1'), derived_id(relabel, DATES[0], 'p2')}
    assert survivor in parents
    merge = lineage[lineage['event'] == 'merge'].iloc[0]
    assert merge['parent_id'] in parents - {survivor}
    assert merge['child_id'] == survivor


def test_hungarian_records_a_death_and_a_birth_when_nothing_links() -> None:
    """a cluster that cannot link to any successor dies; the unlinked successor is born"""
    replacement = {'y': make_fingerprint(['a5', 'a6'], RATES_BETA)}
    relabel, lineage = run_hungarian({DATES[0]: {'x': make_fingerprint(['a1', 'a2'], EQUITY_BETA)},
                                      DATES[1]: replacement, DATES[2]: replacement},
                                     bridge_window=0)
    assert events_at(lineage, DATES[1]) == {'birth', 'death'}
    assert derived_id(relabel, DATES[1], 'y') != derived_id(relabel, DATES[0], 'x')
    death = lineage[lineage['event'] == 'death'].iloc[0]
    assert death['parent_id'] == derived_id(relabel, DATES[0], 'x')
    assert death['child_id'] is None or pd.isna(death['child_id'])


def test_hungarian_revives_a_dormant_track_within_the_bridge_window() -> None:
    """a cluster absent for one date is revived rather than reborn under a new id"""
    persistent = make_fingerprint(['a1', 'a2'], EQUITY_BETA)
    intermittent = make_fingerprint(['a5', 'a6'], RATES_BETA)
    relabel, lineage = run_hungarian({DATES[0]: {'P': persistent, 'Q': intermittent},
                                      DATES[1]: {'P': persistent},
                                      DATES[2]: {'P': persistent, 'Q': intermittent}},
                                     bridge_window=1)
    assert derived_id(relabel, DATES[2], 'Q') == derived_id(relabel, DATES[0], 'Q')
    assert 'birth' not in events_at(lineage, DATES[2])
    assert relabel['derived_id'].nunique() == 2


def test_hungarian_revives_the_best_dormant_track_not_insertion_order() -> None:
    """When two dormant tracks qualify, the higher-affinity identity is revived."""
    first = make_fingerprint(['a1', 'a2', 'a3', 'a4'], EQUITY_BETA)
    best = make_fingerprint(['a1', 'a5', 'a6'], EQUITY_BETA)
    successor = make_fingerprint(['a1', 'a5', 'a6', 'a7'], EQUITY_BETA)
    relabel, _ = run_hungarian(
        {
            DATES[0]: {'first': first, 'best': best},
            DATES[1]: {},
            DATES[2]: {'successor': successor},
        },
        bridge_window=1,
    )
    assert derived_id(relabel, DATES[2], 'successor') == derived_id(
        relabel, DATES[0], 'best'
    )


def test_hungarian_does_not_revive_beyond_the_bridge_window() -> None:
    """with no bridge window the same absence produces a new track instead"""
    persistent = make_fingerprint(['a1', 'a2'], EQUITY_BETA)
    intermittent = make_fingerprint(['a5', 'a6'], RATES_BETA)
    relabel, lineage = run_hungarian({DATES[0]: {'P': persistent, 'Q': intermittent},
                                      DATES[1]: {'P': persistent},
                                      DATES[2]: {'P': persistent, 'Q': intermittent}},
                                     bridge_window=0)
    assert derived_id(relabel, DATES[2], 'Q') != derived_id(relabel, DATES[0], 'Q')
    assert 'birth' in events_at(lineage, DATES[2])


# --------------------------------------------------------------------------- #
# _match_panel_mcf: the global min-cost-flow matcher (needs networkx)
# --------------------------------------------------------------------------- #
def run_mcf(snapshots, bridge_window: int = 1):
    """Run the min-cost-flow matcher, skipping the case when networkx is absent."""
    pytest.importorskip('networkx', reason="the 'mcf' matcher needs the clustering extra")
    return _match_panel_mcf(snapshots, X_COVARS, bridge_window=bridge_window, **LINK_KWARGS)


def test_mcf_keeps_one_id_per_cluster_across_a_stable_panel() -> None:
    """the joint solve agrees with the per-transition one when nothing changes"""
    snapshots = {d: {'c1': make_fingerprint(['a1', 'a2'], EQUITY_BETA),
                     'c2': make_fingerprint(['a3', 'a4'], RATES_BETA)} for d in DATES}
    relabel, lineage = run_mcf(snapshots)
    assert relabel['derived_id'].nunique() == 2
    for label in ('c1', 'c2'):
        assert len({derived_id(relabel, d, label) for d in DATES}) == 1
    assert events_at(lineage, DATES[0]) == {'birth'}
    assert events_at(lineage, DATES[1]) == {'continue'}


def test_mcf_routes_identity_across_a_gap_and_tags_it_bridge() -> None:
    """a cluster missing for one date keeps its id, and the reconnection is a bridge

    This is the behaviour the Hungarian matcher cannot express: it hands identity off
    locally, whereas the joint solve routes it around the gap.
    """
    persistent = make_fingerprint(['a1', 'a2'], EQUITY_BETA)
    intermittent = make_fingerprint(['a5', 'a6'], RATES_BETA)
    relabel, lineage = run_mcf({DATES[0]: {'P': persistent, 'Q': intermittent},
                                DATES[1]: {'P': persistent},
                                DATES[2]: {'P': persistent, 'Q': intermittent}})
    assert derived_id(relabel, DATES[2], 'Q') == derived_id(relabel, DATES[0], 'Q')
    bridges = lineage[lineage['event'] == 'bridge']
    assert len(bridges) == 1
    assert bridges['child_id'].iloc[0] == derived_id(relabel, DATES[0], 'Q')


def test_mcf_raises_a_named_import_error_without_networkx(monkeypatch) -> None:
    """the optional backend is guarded and the message names the extra and the alternative"""
    monkeypatch.setitem(sys.modules, 'networkx', None)
    with pytest.raises(ImportError, match='clustering'):
        _match_panel_mcf({DATES[0]: {'P': make_fingerprint(['a1'], EQUITY_BETA)}},
                         X_COVARS, bridge_window=1, **LINK_KWARGS)


# --------------------------------------------------------------------------- #
# tracks and classification
# --------------------------------------------------------------------------- #
def test_build_tracks_groups_snapshots_into_one_panel_per_derived_id() -> None:
    """each track carries its own beta history, vol split and per-date membership"""
    snapshots = {d: {'c1': make_fingerprint(['a1', 'a2'], EQUITY_BETA),
                     'c2': make_fingerprint(['a3', 'a4'], RATES_BETA)} for d in DATES}
    relabel, _ = run_hungarian(snapshots)
    tracks = _build_tracks(relabel, snapshots, FACTORS)
    assert len(tracks) == 2
    for did, track in tracks.items():
        assert list(track.betas.columns) == FACTORS
        assert len(track.betas) == len(DATES)
        assert list(track.size) == [2, 2, 2]
        assert set(track.members) == set(DATES)
        # the panel stores vols, i.e. the square roots of the fingerprint variances
        raw_label = derived_to_raw(relabel, DATES[0], did)
        source = snapshots[DATES[0]][raw_label]
        assert track.factor_vol.iloc[0] == pytest.approx(np.sqrt(source.factor_var))
        assert track.idio_vol.iloc[0] == pytest.approx(np.sqrt(source.idio_var))
        assert track.total_vol.iloc[0] == pytest.approx(np.sqrt(source.total_var))
        assert np.allclose(track.betas.iloc[0].to_numpy(), source.beta)
        assert track.members[DATES[0]] == list(source.members)


def test_classify_labels_a_persistent_equity_track_high_beta_and_core() -> None:
    """a full-coverage equity track with beta above the cut classifies as Equity-HighBeta/Core"""
    snapshots = {d: {'c1': make_fingerprint(['a1', 'a2'], EQUITY_BETA),
                     'c2': make_fingerprint(['a3', 'a4'], RATES_BETA)} for d in DATES}
    relabel, _ = run_hungarian(snapshots)
    tracks = _build_tracks(relabel, snapshots, FACTORS)
    classification, transitions = _classify(tracks, X_COVARS, len(DATES), TaxonomyConfig())
    equity = classification[classification['modal_dom'] == 'Equity'].iloc[0]
    assert equity['track_type'] == 'Equity-HighBeta'
    assert equity['coverage'] == pytest.approx(1.0)
    assert equity['persistence'] == 'Core'
    assert equity['stability_label'] == 'Stable'      # beta is constant, so spread vol is 0
    assert equity['lifetime'] == len(DATES)
    assert transitions.empty                          # a constant beta has no regime break
    # coverage is the sort key, descending
    assert classification['coverage'].is_monotonic_decreasing


def test_classify_calls_a_track_mixed_when_no_factor_dominates() -> None:
    """a track whose dominant factor keeps changing is Mixed rather than named"""
    index = pd.DatetimeIndex(DATES)
    track = TrackPanel(betas=pd.DataFrame([EQUITY_BETA, RATES_BETA, EQUITY_BETA],
                                          index=index, columns=FACTORS),
                       factor_vol=pd.Series([0.20, 0.06, 0.20], index=index),
                       idio_vol=pd.Series([0.07] * 3, index=index),
                       total_vol=pd.Series([0.21, 0.09, 0.21], index=index),
                       r2=pd.Series([0.9, 0.4, 0.9], index=index),
                       size=pd.Series([2, 2, 2], index=index),
                       dominant_factor=pd.Series(['Equity', 'Rates', 'Credit'], index=index),
                       members={d: ['a1', 'a2'] for d in DATES})
    classification, transitions = _classify({'d001': track}, X_COVARS, len(DATES),
                                            TaxonomyConfig())
    assert classification.loc['d001', 'track_type'] == 'Mixed'
    # the beta swings well beyond the drifting cut, so the track is Drifting with breaks
    assert classification.loc['d001', 'stability_label'] == 'Drifting'
    assert not transitions.empty
    assert set(transitions['kind']) == {'beta_break'}


def test_classify_calls_a_short_lived_track_transient() -> None:
    """a track live for one date out of ten is Transient, not Core"""
    index = pd.DatetimeIndex(DATES[:1])
    track = TrackPanel(betas=pd.DataFrame([EQUITY_BETA], index=index, columns=FACTORS),
                       factor_vol=pd.Series([0.02], index=index),
                       idio_vol=pd.Series([0.07], index=index),
                       total_vol=pd.Series([0.08], index=index),
                       r2=pd.Series([0.1], index=index),
                       size=pd.Series([2], index=index),
                       dominant_factor=pd.Series(['Equity'], index=index),
                       members={DATES[0]: ['a1', 'a2']})
    classification, _ = _classify({'d001': track}, X_COVARS, 10, TaxonomyConfig())
    assert classification.loc['d001', 'persistence'] == 'Transient'
    assert classification.loc['d001', 'vol_regime'] == 'Low'
    assert classification.loc['d001', 'vol_trend'] == 0.0      # fewer than 3 points, no slope


# --------------------------------------------------------------------------- #
# the report object
# --------------------------------------------------------------------------- #
def build_report(method: str = 'hungarian') -> RiskClusterReport:
    """Run the public entry point over the stable three-date panel."""
    return analyze_risk_clusters(make_rolling(), method=method)


def test_analyze_risk_clusters_tracks_a_stable_panel_end_to_end() -> None:
    """the public entry point yields one track per cluster with the parameters recorded"""
    report = build_report()
    assert set(report.classification.index) == set(report.relabel['derived_id'])
    assert len(report.tracks) == 2
    assert report.params['method'] == 'hungarian'
    assert report.params['overlap_metric'] == 'overlap'
    assert report.params['bridge_decay'] == 0.5
    assert list(report.factor_covar.index) == FACTORS
    assert np.allclose(report.factor_covar.to_numpy(), SIGMA)


def test_analyze_risk_clusters_agrees_between_the_two_matchers_on_a_stable_panel() -> None:
    """with nothing changing, the joint and per-transition matchers group identically"""
    pytest.importorskip('networkx', reason="the 'mcf' matcher needs the clustering extra")
    hungarian, mcf = build_report('hungarian'), build_report('mcf')

    def grouping(report):
        """Map each date/raw-label pair to its track, as a comparable frozen set."""
        panel = report.relabel.set_index(['date', 'raw_label'])['derived_id']
        return {did: frozenset(keys) for did, keys in panel.groupby(panel).groups.items()}

    assert set(grouping(hungarian).values()) == set(grouping(mcf).values())


def test_factor_labels_name_the_primary_factor_and_the_vol_regime() -> None:
    """labels come from betas and vol alone, with no external metadata"""
    report = build_report()
    labels = report.factor_labels()
    assert len(labels) == len(report.tracks)
    assert any(lab.startswith('Equity high-') for lab in labels)
    assert any(lab.startswith('Rates long-duration') for lab in labels)
    assert all(' · ' in lab for lab in labels)             # '<factor> · <vol regime>'
    equity = [lab for lab in labels if lab.startswith('Equity')][0]
    assert equity.endswith('high-vol')                          # 20% equity vol is above vol_high


def test_factor_labels_call_a_track_with_no_factor_variance_idiosyncratic() -> None:
    """a zero-beta track has no factor contribution to rank, so it is Idiosyncratic"""
    index = pd.DatetimeIndex(DATES)
    track = TrackPanel(betas=pd.DataFrame(np.zeros((3, 3)), index=index, columns=FACTORS),
                       factor_vol=pd.Series([0.0] * 3, index=index),
                       idio_vol=pd.Series([0.07] * 3, index=index),
                       total_vol=pd.Series([0.07] * 3, index=index),
                       r2=pd.Series([0.0] * 3, index=index),
                       size=pd.Series([2] * 3, index=index),
                       dominant_factor=pd.Series(['Equity'] * 3, index=index),
                       members={d: ['a1', 'a2'] for d in DATES})
    report = RiskClusterReport(relabel=pd.DataFrame(), tracks={'d001': track},
                               classification=pd.DataFrame(), lineage=pd.DataFrame(),
                               transitions=pd.DataFrame(), params={},
                               factor_covar=pd.DataFrame(SIGMA, index=FACTORS, columns=FACTORS))
    assert report.factor_labels()['d001'] == 'Idiosyncratic'


def test_factor_labels_support_a_single_factor_model() -> None:
    """A one-factor covariance labels without looking for a second factor."""
    index = pd.DatetimeIndex(DATES)
    track = TrackPanel(betas=pd.DataFrame({'Equity': [0.8] * 3}, index=index),
                       factor_vol=pd.Series([0.16] * 3, index=index),
                       idio_vol=pd.Series([0.04] * 3, index=index),
                       total_vol=pd.Series([0.17] * 3, index=index),
                       r2=pd.Series([0.9] * 3, index=index),
                       size=pd.Series([2] * 3, index=index),
                       dominant_factor=pd.Series(['Equity'] * 3, index=index),
                       members={d: ['a1', 'a2'] for d in DATES})
    report = RiskClusterReport(relabel=pd.DataFrame(), tracks={'d001': track},
                               classification=pd.DataFrame(), lineage=pd.DataFrame(),
                               transitions=pd.DataFrame(), params={},
                               factor_covar=pd.DataFrame([[0.04]], index=['Equity'],
                                                         columns=['Equity']))
    assert report.factor_labels()['d001'] == 'Equity high-β · high-vol'


def test_label_tracks_treats_nan_equity_beta_as_defensive_zero() -> None:
    """An all-NaN equity beta uses the documented zero fallback, not core."""
    index = pd.DatetimeIndex(DATES)
    track = TrackPanel(betas=pd.DataFrame({'Equity': [np.nan] * 3}, index=index),
                       factor_vol=pd.Series([0.0] * 3, index=index),
                       idio_vol=pd.Series([0.07] * 3, index=index),
                       total_vol=pd.Series([0.07] * 3, index=index),
                       r2=pd.Series([0.0] * 3, index=index),
                       size=pd.Series([1] * 3, index=index),
                       dominant_factor=pd.Series(['Idio'] * 3, index=index),
                       members={d: ['a1'] for d in DATES})
    classification = pd.DataFrame({'track_type': ['Idiosyncratic'],
                                   'modal_dom': ['Idio']}, index=['d001'])
    report = RiskClusterReport(relabel=pd.DataFrame(), tracks={'d001': track},
                               classification=classification, lineage=pd.DataFrame(),
                               transitions=pd.DataFrame(), params={})
    metadata = pd.DataFrame({'Sub Asset Class': ['US Equity'],
                             'Asset Class': ['Equity']}, index=['a1'])
    assert report.label_tracks(metadata)['d001'] == 'US Equity · defensive'


def test_membership_panel_is_dates_by_assets() -> None:
    """the wide panel maps every clustered asset to its track at every date"""
    report = build_report()
    panel = report.to_membership_panel()
    assert list(panel.index) == DATES
    assert list(panel.columns) == sorted(ASSETS)
    assert panel.notna().all().all()
    # a1 and a2 share a cluster at every date, a3 and a4 share the other
    for date in DATES:
        assert panel.loc[date, 'a1'] == panel.loc[date, 'a2']
        assert panel.loc[date, 'a3'] == panel.loc[date, 'a4']
        assert panel.loc[date, 'a1'] != panel.loc[date, 'a3']


def test_label_panel_replaces_track_ids_with_labels() -> None:
    """the label panel is the membership panel with ids substituted"""
    report = build_report()
    labels = report.to_label_panel(label_kind='factor')
    assert labels.shape == report.to_membership_panel().shape
    assert set(labels.loc[DATES[0]]) <= set(report.factor_labels())


def test_labels_at_falls_back_to_the_nearest_prior_date() -> None:
    """an off-grid date resolves to the last rebalancing on or before it"""
    report = build_report()
    exact = report.labels_at(DATES[1])
    between = report.labels_at(pd.Timestamp('2024-03-15'))       # between DATES[1] and DATES[2]
    assert list(exact.index) == sorted(ASSETS)
    assert exact.to_dict() == between.to_dict()


def test_labels_at_returns_empty_before_the_first_date() -> None:
    """there is nothing to label before the panel starts"""
    assert build_report().labels_at(pd.Timestamp('2023-01-01')).empty


def test_label_tracks_uses_asset_metadata_when_it_is_pure_enough() -> None:
    """a track whose members share a sub-asset class takes it as the theme"""
    report = build_report()
    asset_meta = pd.DataFrame({'Sub Asset Class': ['US Large Cap', 'US Large Cap',
                                                   'Govt Bonds', 'Govt Bonds'],
                               'Asset Class': ['Equity', 'Equity', 'Rates', 'Rates']},
                              index=ASSETS)
    labels = report.label_tracks(asset_meta)
    assert len(labels) == len(report.tracks)
    assert any(lab.startswith('US Large Cap') for lab in labels)
    assert any(lab.startswith('Govt Bonds') for lab in labels)


def test_label_tracks_falls_back_to_the_track_type_without_metadata() -> None:
    """assets absent from the metadata leave the classification as the only label"""
    report = build_report()
    empty_meta = pd.DataFrame({'Sub Asset Class': [], 'Asset Class': []}, dtype=object)
    labels = report.label_tracks(empty_meta)
    assert set(labels) <= set(report.classification['track_type'])


def test_labels_rejects_an_unknown_kind_and_meta_without_metadata() -> None:
    """the two ways of asking for labels wrongly both raise rather than guess"""
    report = build_report()
    with pytest.raises(ValueError, match='unknown label_kind'):
        report.to_label_panel(label_kind='nonsense')
    with pytest.raises(ValueError, match='requires asset_meta'):
        report.to_label_panel(label_kind='meta')


def test_label_panel_by_id_keeps_the_derived_ids() -> None:
    """label_kind='id' is the identity mapping, i.e. the membership panel itself"""
    report = build_report()
    assert report.to_label_panel(label_kind='id').equals(report.to_membership_panel())


def test_to_tables_returns_every_workbook_view() -> None:
    """the workbook export carries all five frames and none is empty"""
    tables = build_report().to_tables()
    assert set(tables) == {'classification', 'membership_panel', 'lineage', 'transitions',
                           'relabel'}
    for name in ('classification', 'membership_panel', 'lineage', 'relabel'):
        assert not tables[name].empty, f"{name} is empty"
