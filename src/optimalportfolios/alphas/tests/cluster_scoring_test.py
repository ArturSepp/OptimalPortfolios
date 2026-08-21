"""
within-cluster signal scoring and the alpha-signal dispatcher.

``score_within_clusters`` is where a raw signal becomes a cross-sectional score, and it is the
one place in the alpha stack whose *fallbacks* matter more than its main path. A z-score taken
over two observations is noise, so small clusters are scored against the full cross-section
instead; dates before the first cluster estimation get 0.0 rather than a look-ahead
assignment; a degenerate partition falls back to global scoring. Each of those branches
changes the number an asset is ranked on, and none of them raises when it is wrong.

So the cases below are built around cluster *sizes* and *dates* rather than around plausible
market data: a partition with one large and one small cluster, a signal panel that starts
before the clusters do, a single-cluster partition. The values are stated in the test, so the
expected score is arithmetic rather than a recorded output.

``resolve_span`` is covered alongside it: it is the guard that stops one scalar span meaning a
year on monthly data and three years on quarterly, and every branch of it is an error path.
"""
# packages
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import qis
# optimalportfolios
import optimalportfolios.alphas.backtest_alphas as backtest_module
from optimalportfolios.alphas.backtest_alphas import AlphaSignal, compute_signal_scores
from optimalportfolios.alphas.signals.utils import (
    _global_zscore,
    _split_cluster_label,
    extract_rolling_clusters,
    resolve_span,
    score_within_clusters,
)

SEED = 20260810
TICKERS = [f't{i}' for i in range(8)]
SIGNAL_DATES = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-31'])


def make_raw_signal(dates=SIGNAL_DATES) -> pd.DataFrame:
    """A raw signal panel with a distinct, monotone spread across tickers."""
    values = np.arange(len(TICKERS), dtype=float)
    return pd.DataFrame([values + offset for offset in range(len(dates))],
                        index=dates, columns=TICKERS)


# --------------------------------------------------------------------------- #
# resolve_span
# --------------------------------------------------------------------------- #
def test_resolve_span_passes_a_scalar_through_unchanged() -> None:
    """a scalar span applies at every cadence, so existing callers are unaffected"""
    assert resolve_span(12, freq='ME') == 12
    assert resolve_span(12, freq='QE') == 12


def test_resolve_span_picks_the_entry_for_the_cadence() -> None:
    """a mapping gives each cadence its own horizon in periods"""
    assert resolve_span({'ME': 12, 'QE': 4}, freq='ME') == 12
    assert resolve_span({'ME': 12, 'QE': 4}, freq='QE') == 4


def test_resolve_span_keeps_none_optional() -> None:
    """an optional span stays optional rather than becoming a default"""
    assert resolve_span(None, freq='ME') is None


def test_resolve_span_refuses_to_let_a_cadence_inherit_another_horizon() -> None:
    """an uncovered cadence is an error, because inheriting silently changes the horizon"""
    with pytest.raises(ValueError, match='add the cadence rather than letting it inherit'):
        resolve_span({'ME': 12}, freq='QE')


def test_resolve_span_rejects_a_non_integer_span() -> None:
    """a span is a number of periods, so a float or a string is a mistake not a cast"""
    with pytest.raises(ValueError, match='must be an int number of periods'):
        resolve_span(12.5, freq='ME')
    with pytest.raises(ValueError, match='must be an int number of periods'):
        resolve_span('12', freq='ME', name='long_span')


def test_resolve_span_rejects_a_bool_masquerading_as_an_int() -> None:
    """bool is an int in Python; a span of True is never intended"""
    with pytest.raises(ValueError, match='must be an int number of periods'):
        resolve_span(True, freq='ME')


def test_resolve_span_rejects_a_non_positive_span() -> None:
    """a zero or negative EWMA span has no meaning"""
    with pytest.raises(ValueError, match='must be > 0'):
        resolve_span(0, freq='ME')
    with pytest.raises(ValueError, match='must be > 0'):
        resolve_span(-4, freq='ME')


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_split_cluster_label_separates_the_frequency_prefix() -> None:
    """cluster ids carry the estimating cadence, and it must survive alignment"""
    assert _split_cluster_label('QE:4') == ('QE', '4')
    assert _split_cluster_label('ME:12') == ('ME', '12')


def test_split_cluster_label_handles_a_bare_id() -> None:
    """an unprefixed label keeps an empty prefix rather than raising"""
    assert _split_cluster_label('4') == ('', '4')


def test_global_zscore_standardises_the_named_columns() -> None:
    """the fallback scorer is a plain z-score over the columns it is given"""
    row = pd.Series([1.0, 2.0, 3.0, 4.0], index=TICKERS[:4])
    scored = _global_zscore(row, list(row.index))
    assert scored.mean() == pytest.approx(0.0, abs=1e-12)
    assert scored.std() == pytest.approx(1.0)


def test_global_zscore_returns_zeros_when_there_is_nothing_to_standardise() -> None:
    """one observation, or a constant row, has no spread and scores flat"""
    row = pd.Series([1.0, 1.0, 1.0], index=TICKERS[:3])
    assert (_global_zscore(row, list(row.index)) == 0.0).all()
    assert (_global_zscore(row, TICKERS[:1]) == 0.0).all()


# --------------------------------------------------------------------------- #
# extract_rolling_clusters
# --------------------------------------------------------------------------- #
def make_rolling_covar_data(clusters_by_date: dict) -> SimpleNamespace:
    """The shape ``extract_rolling_clusters`` reads: ``.data[date].clusters``.

    ``RollingFactorCovarData`` comes out of a factorlasso estimation over real prices, which
    this suite deliberately does not run. Only the two attributes the function touches are
    stood up here, so the branches are exercised on stated assignments rather than on whatever
    an estimator happened to produce.
    """
    return SimpleNamespace(
        data={date: SimpleNamespace(clusters=clusters)
              for date, clusters in clusters_by_date.items()})


def test_extract_rolling_clusters_keys_the_assignments_by_estimation_date() -> None:
    """the flat per-date Series is passed through, keyed by the date it was estimated on"""
    assignment = pd.Series(['QE:1'] * 4 + ['QE:2'] * 4, index=TICKERS)
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: assignment,
                                 SIGNAL_DATES[1]: assignment}))
    assert list(rolling.keys()) == [SIGNAL_DATES[0], SIGNAL_DATES[1]]
    pd.testing.assert_series_equal(rolling[SIGNAL_DATES[0]], assignment)


def test_extract_rolling_clusters_skips_dates_with_no_assignment() -> None:
    """a date the estimator produced nothing for is absent, not present and empty

    An empty Series left in the dict would be scored as a partition of zero assets, which
    ``score_within_clusters`` reads as a degenerate cluster rather than as missing data.
    """
    assignment = pd.Series(['QE:1'] * 8, index=TICKERS)
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: None,
                                 SIGNAL_DATES[1]: pd.Series(dtype=object),
                                 SIGNAL_DATES[2]: assignment}))
    assert list(rolling.keys()) == [SIGNAL_DATES[2]]


def test_extract_rolling_clusters_keeps_the_last_label_of_a_duplicated_ticker() -> None:
    """a ticker carrying both an ME and a QE label would otherwise form two clusters"""
    assignment = pd.Series(['ME:1', 'QE:3'], index=[TICKERS[0], TICKERS[0]])
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: assignment}))
    assert rolling[SIGNAL_DATES[0]].to_dict() == {TICKERS[0]: 'QE:3'}


def test_extract_rolling_clusters_filters_to_the_requested_universe() -> None:
    """the covariance universe is wider than the signal universe, so it is narrowed here"""
    assignment = pd.Series(['QE:1'] * 4 + ['QE:2'] * 4, index=TICKERS)
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: assignment}), assets=TICKERS[:3])
    assert list(rolling[SIGNAL_DATES[0]].index) == TICKERS[:3]


def test_extract_rolling_clusters_drops_a_date_that_covers_none_of_the_universe() -> None:
    """filtering can empty a date, and an emptied date is dropped like an absent one"""
    assignment = pd.Series(['QE:1'] * 4, index=TICKERS[4:])
    rolling = extract_rolling_clusters(
        make_rolling_covar_data({SIGNAL_DATES[0]: assignment}), assets=TICKERS[:3])
    assert rolling == {}


# --------------------------------------------------------------------------- #
# score_within_clusters
# --------------------------------------------------------------------------- #
def test_scoring_without_clusters_falls_back_to_the_full_cross_section() -> None:
    """with no cluster data at all the signal is scored globally, not zeroed"""
    raw = make_raw_signal()
    scored = score_within_clusters(raw_signal=raw, rolling_clusters={})
    assert scored.shape == raw.shape
    assert not np.allclose(scored.to_numpy(), 0.0)


def test_dates_before_the_first_cluster_estimation_score_zero() -> None:
    """a date with no assignment yet must not borrow a later one — that is look-ahead"""
    raw = make_raw_signal()
    clusters = {SIGNAL_DATES[1]: pd.Series(['a'] * 4 + ['b'] * 4, index=TICKERS)}
    scored = score_within_clusters(raw_signal=raw, rolling_clusters=clusters)
    assert np.allclose(scored.loc[SIGNAL_DATES[0]].to_numpy(), 0.0)
    assert not np.allclose(scored.loc[SIGNAL_DATES[1]].to_numpy(), 0.0)


def test_a_large_cluster_is_scored_against_its_own_members() -> None:
    """within-cluster scoring standardises against the cluster, not the universe"""
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    # one cluster of 5 (above min_cluster_size) and one of 3
    clusters = {SIGNAL_DATES[0]: pd.Series(['big'] * 5 + ['small'] * 3, index=TICKERS)}
    scored = score_within_clusters(raw_signal=raw, rolling_clusters=clusters,
                                   min_cluster_size=3)
    big_cols = TICKERS[:5]
    row = raw.loc[SIGNAL_DATES[0], big_cols]
    expected = (row - row.mean()) / row.std()
    np.testing.assert_allclose(scored.loc[SIGNAL_DATES[0], big_cols].to_numpy(),
                               expected.to_numpy(), atol=1e-12)


def test_a_small_cluster_is_scored_against_the_whole_cross_section() -> None:
    """a z-score over three points is noise, so small clusters use global statistics"""
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    clusters = {SIGNAL_DATES[0]: pd.Series(['big'] * 5 + ['small'] * 3, index=TICKERS)}
    scored = score_within_clusters(raw_signal=raw, rolling_clusters=clusters,
                                   min_cluster_size=3)
    small_cols = TICKERS[5:]
    whole_row = raw.loc[SIGNAL_DATES[0]]
    expected = (whole_row[small_cols] - whole_row.mean()) / whole_row.std()
    np.testing.assert_allclose(scored.loc[SIGNAL_DATES[0], small_cols].to_numpy(),
                               expected.to_numpy(), atol=1e-12)


def test_min_cluster_size_moves_the_boundary() -> None:
    """raising the threshold pushes a cluster from within-cluster to global scoring"""
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    clusters = {SIGNAL_DATES[0]: pd.Series(['big'] * 5 + ['small'] * 3, index=TICKERS)}
    within = score_within_clusters(raw_signal=raw, rolling_clusters=clusters,
                                   min_cluster_size=3)
    globally = score_within_clusters(raw_signal=raw, rolling_clusters=clusters,
                                     min_cluster_size=6)
    assert not np.allclose(within.loc[SIGNAL_DATES[0], TICKERS[:5]].to_numpy(),
                           globally.loc[SIGNAL_DATES[0], TICKERS[:5]].to_numpy())


def test_a_single_cluster_partition_is_scored_globally() -> None:
    """one cluster is not a partition, so the degenerate branch scores the universe"""
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    clusters = {SIGNAL_DATES[0]: pd.Series(['only'] * len(TICKERS), index=TICKERS)}
    scored = score_within_clusters(raw_signal=raw, rolling_clusters=clusters)
    whole_row = raw.loc[SIGNAL_DATES[0]]
    expected = (whole_row - whole_row.mean()) / whole_row.std()
    np.testing.assert_allclose(scored.loc[SIGNAL_DATES[0]].to_numpy(),
                               expected.to_numpy(), atol=1e-12)


def test_the_most_recent_assignment_is_carried_forward() -> None:
    """a signal date between estimations uses the last assignment, never a future one"""
    dates = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-31'])
    raw = make_raw_signal(dates=dates)
    clusters = {dates[0]: pd.Series(['a'] * 5 + ['b'] * 3, index=TICKERS),
                dates[2]: pd.Series(['b'] * 3 + ['a'] * 5, index=TICKERS)}
    scored = score_within_clusters(raw_signal=raw, rolling_clusters=clusters,
                                   min_cluster_size=3)
    # the February row is scored under January's partition: the first five are the big cluster
    row = raw.loc[dates[1], TICKERS[:5]]
    expected = (row - row.mean()) / row.std()
    np.testing.assert_allclose(scored.loc[dates[1], TICKERS[:5]].to_numpy(),
                               expected.to_numpy(), atol=1e-12)


def test_assets_absent_from_the_partition_score_zero() -> None:
    """an unclustered asset gets no score rather than an arbitrary one"""
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    clusters = {SIGNAL_DATES[0]: pd.Series(['big'] * 5, index=TICKERS[:5])}
    scored = score_within_clusters(raw_signal=raw, rolling_clusters=clusters,
                                   min_cluster_size=3)
    assert np.allclose(scored.loc[SIGNAL_DATES[0], TICKERS[5:]].to_numpy(), 0.0)


def test_a_row_with_almost_everything_missing_scores_on_the_raw_values() -> None:
    """with fewer than two observations there is no dispersion to standardise against

    Estimating a mean and a standard deviation from one point gives 0 and NaN, and dividing by
    NaN would silently blank the whole row. The fallback states (0, 1) instead, so the one
    asset that did report keeps its raw value and stays comparable to the zeros around it.
    """
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    raw.loc[SIGNAL_DATES[0], TICKERS[1:]] = np.nan  # only t0 reported this date
    # two clusters over three assets, so the degenerate single-cluster branch is not taken
    clusters = {SIGNAL_DATES[0]: pd.Series(['a', 'a', 'b'], index=TICKERS[:3])}
    scored = score_within_clusters(raw_signal=raw, rolling_clusters=clusters,
                                   min_cluster_size=3)
    assert scored.loc[SIGNAL_DATES[0], TICKERS[0]] == raw.loc[SIGNAL_DATES[0], TICKERS[0]]


def test_a_lookup_that_returns_nothing_scores_the_row_zero(monkeypatch) -> None:
    """the None backstop keeps a build whose as-of lookup returns None from raising

    ``find_upto_date_from_datetime_index`` returns None below the first index date in the qis
    build pinned here; the guard on ``date < first_cluster_date`` normally means the call is
    never made in that state. The backstop exists for builds where it is, and is pinned here
    by forcing the lookup to return None on a date that does have an assignment.
    """
    monkeypatch.setattr(qis, 'find_upto_date_from_datetime_index',
                        lambda index, date: None)
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    clusters = {SIGNAL_DATES[0]: pd.Series(['a'] * 5 + ['b'] * 3, index=TICKERS)}
    scored = score_within_clusters(raw_signal=raw, rolling_clusters=clusters)
    assert np.allclose(scored.loc[SIGNAL_DATES[0]].to_numpy(), 0.0)


def test_a_lookup_that_raises_scores_the_row_zero(monkeypatch) -> None:
    """the sibling backstop: other qis builds raise where this one returns None

    Scoring the row 0.0 is the same answer as the None path — a date with no usable
    assignment contributes nothing, rather than aborting the whole backtest.
    """
    def _raise(index, date):
        """Stand in for a qis build that raises instead of returning None."""
        raise KeyError(date)

    monkeypatch.setattr(qis, 'find_upto_date_from_datetime_index', _raise)
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    clusters = {SIGNAL_DATES[0]: pd.Series(['a'] * 5 + ['b'] * 3, index=TICKERS)}
    scored = score_within_clusters(raw_signal=raw, rolling_clusters=clusters)
    assert np.allclose(scored.loc[SIGNAL_DATES[0]].to_numpy(), 0.0)


def test_nan_cluster_assignments_are_dropped() -> None:
    """a NaN assignment must not create a cluster of its own"""
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    assignment = pd.Series(['big'] * 5 + ['small'] * 2 + [np.nan], index=TICKERS)
    scored = score_within_clusters(raw_signal=raw, rolling_clusters={SIGNAL_DATES[0]:
                                                                     assignment})
    assert scored.shape == raw.shape
    assert scored.loc[SIGNAL_DATES[0], TICKERS[-1]] == 0.0


def test_stability_pooling_off_is_byte_identical_on_a_production_shaped_panel() -> None:
    """V0 must preserve every output byte of the existing scorer."""
    from factorlasso import StabilityPoolingType, score_with_stability_pooled_clusters

    raw = make_raw_signal()
    clusters = {
        SIGNAL_DATES[0]: pd.Series(['ME:1'] * 5 + ['QE:2'] * 3, index=TICKERS),
        SIGNAL_DATES[2]: pd.Series(['ME:1'] * 4 + ['QE:2'] * 4, index=TICKERS),
    }
    weights = pd.DataFrame(0.25, index=SIGNAL_DATES, columns=TICKERS)
    baseline = score_within_clusters(raw, clusters, min_cluster_size=3)
    explicit_v0 = score_within_clusters(
        raw,
        clusters,
        min_cluster_size=3,
        stability_pooling_type=StabilityPoolingType.NONE,
        stability_weights=weights,
    )
    factorlasso_v0 = score_with_stability_pooled_clusters(
        raw,
        clusters,
        weights,
        min_cluster_size=3,
        pooling_type=StabilityPoolingType.NONE,
    )

    pd.testing.assert_frame_equal(explicit_v0, baseline, check_exact=True)
    pd.testing.assert_frame_equal(factorlasso_v0, baseline, check_exact=True)
    assert explicit_v0.to_numpy().tobytes() == baseline.to_numpy().tobytes()
    assert factorlasso_v0.to_numpy().tobytes() == baseline.to_numpy().tobytes()


@pytest.mark.parametrize(
    'pooling_name',
    [
        'CLUSTER_VARIANCE',
        'ASSET_VARIANCE',
    ],
)
def test_unit_stability_is_exactly_the_existing_cluster_score(pooling_name) -> None:
    """Every pooled variant at w=1 must reduce bit-for-bit to V0."""
    from factorlasso import StabilityPoolingType

    pooling_type = StabilityPoolingType[pooling_name]
    raw = make_raw_signal(dates=SIGNAL_DATES[:1])
    clusters = {SIGNAL_DATES[0]: pd.Series(['big'] * 5 + ['small'] * 3, index=TICKERS)}
    weights = pd.DataFrame(1.0, index=SIGNAL_DATES[:1], columns=TICKERS)
    baseline = score_within_clusters(raw, clusters, min_cluster_size=3)
    actual = score_within_clusters(
        raw,
        clusters,
        min_cluster_size=3,
        stability_pooling_type=pooling_type,
        stability_weights=weights,
    )

    pd.testing.assert_frame_equal(actual, baseline, check_exact=True)
    assert actual.to_numpy().tobytes() == baseline.to_numpy().tobytes()


# --------------------------------------------------------------------------- #
# the signal dispatcher
# --------------------------------------------------------------------------- #
def make_prices(n_months: int = 90) -> pd.DataFrame:
    """A seeded monthly price panel long enough for a 24-period beta span."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range('2016-01-31', periods=n_months, freq='ME')
    returns = rng.normal(0.005, 0.04, size=(n_months, 4))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=['a', 'b', 'c', 'd'])


@pytest.mark.parametrize('alpha_signal', list(AlphaSignal))
def test_every_alpha_signal_produces_a_score_panel(alpha_signal) -> None:
    """each enum member routes to its signal and comes back shaped like the prices"""
    prices = make_prices()
    scores = compute_signal_scores(prices=prices, alpha_signal=alpha_signal,
                                   returns_freq='ME', mom_long_span=12, beta_span=24,
                                   momentum_span=12)
    assert list(scores.columns) == list(prices.columns)
    assert len(scores) > 0
    assert scores.notna().any().any(), f"{alpha_signal} produced an all-NaN panel"


def test_compute_signal_scores_accepts_group_data() -> None:
    """within-group scoring is an option on every signal, not a separate entry point"""
    prices = make_prices()
    groups = pd.Series(['x', 'x', 'y', 'y'], index=prices.columns)
    scores = compute_signal_scores(prices=prices, alpha_signal=AlphaSignal.MOMENTUM,
                                   group_data=groups, returns_freq='ME')
    assert list(scores.columns) == list(prices.columns)


def test_classic_dispatch_forwards_lookback_and_skip(monkeypatch) -> None:
    """Classic dispatcher parameters must reach the fixed-window constructor unchanged."""
    captured = {}
    panel = make_prices()

    def fake_classic(**kwargs):
        """Record the dispatch keywords and return a correctly shaped placeholder."""
        captured.update(kwargs)
        values = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
        return values, values

    monkeypatch.setattr(
        backtest_module, 'compute_classic_momentum_alpha', fake_classic
    )
    compute_signal_scores(
        prices=panel,
        alpha_signal=AlphaSignal.CLASSIC_MOMENTUM,
        classic_lookback_periods=9,
        classic_skip_periods=2,
    )

    assert captured['lookback_periods'] == 9
    assert captured['skip_periods'] == 2


def test_compute_signal_scores_rejects_an_unrouted_signal() -> None:
    """a value that is not an AlphaSignal raises rather than returning an empty panel"""
    with pytest.raises(NotImplementedError, match='alpha_signal'):
        compute_signal_scores(prices=make_prices(), alpha_signal='Momentum')


def test_composite_signals_combine_their_two_legs() -> None:
    """a composite is the scaled sum of its parts, so it differs from either alone"""
    prices = make_prices()
    momentum = compute_signal_scores(prices=prices, alpha_signal=AlphaSignal.MOMENTUM,
                                     returns_freq='ME')
    composite = compute_signal_scores(prices=prices,
                                      alpha_signal=AlphaSignal.MOMENTUM_AND_BETA,
                                      returns_freq='ME')
    common = momentum.index.intersection(composite.index)
    assert len(common) > 0
    assert not np.allclose(momentum.loc[common].fillna(0.0).to_numpy(),
                           composite.loc[common].fillna(0.0).to_numpy())
