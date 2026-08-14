"""
the reporting, grouping and summary surface of ``PortfolioOptimisationResult``.

``portfolio_result_delegation_test.py`` pins the *numbers* the risk facade delegates to
``qis.RiskModel``. This file covers the rest of the container: the group and factor
attribution tables, the weight summaries, the report bundle and the frontier data — the
parts a factsheet consumes, which no other test reaches.

The fixture deliberately uses the *harder* shape of every option, because the easy shape is
already exercised elsewhere and the branches differ:

* **per-portfolio benchmarks** (a DataFrame), not one shared Series, so the deduplication in
  ``_get_all_labelled_weight_vectors`` is live rather than short-circuited;
* **two group attributions**, so ``compute_group_attribution`` iterates;
* **``ac_bounds`` present**, so the bounds block of that table is built;
* **current weights present**, so turnover and the trade columns exist.

A second, deliberately degenerate fixture covers the skip branches: a self-benchmarked
portfolio contributes no active vector and no benchmark vector, which is the case that
silently produced duplicate rows before the dedup existed.

Weights here are arbitrary but seeded — the assertions are about table *shape, labelling and
internal consistency* (a weight column that sums to the portfolio total, risk percentages that
sum to one), not about reproducing a number the optimiser produced.
"""
# packages
import numpy as np
import pandas as pd
import pytest
# factorlasso
from factorlasso import CurrentFactorCovarData, VarianceColumns
# optimalportfolios
from optimalportfolios import PortfolioOptimisationResult

SEED = 20260810
ASSETS = pd.Index([f'asset_{i}' for i in range(6)])
FACTORS = pd.Index(['Equity', 'Rates'])
GROUPS = ['Equities', 'Equities', 'Bonds', 'Bonds', 'Alternatives', 'Alternatives']
REGIONS = ['US', 'EU', 'US', 'EU', 'US', 'EU']


def make_covar_data(seed: int = SEED) -> CurrentFactorCovarData:
    """A seeded two-factor snapshot over ASSETS with a complete variance table."""
    rng = np.random.default_rng(seed)
    betas = rng.normal(scale=0.40, size=(len(ASSETS), len(FACTORS)))
    root = rng.normal(scale=0.15, size=(len(FACTORS), len(FACTORS)))
    # every VarianceColumns entry is populated: get_snapshot() reads more of them than
    # the risk maths does, and a partial table raises a KeyError inside factorlasso
    y_variances = pd.DataFrame(
        {column.value: rng.uniform(0.01, 0.20, len(ASSETS)) for column in VarianceColumns
         if column is not VarianceColumns.CLUSTER}, index=ASSETS)
    y_variances[VarianceColumns.RESIDUAL_VARS.value] = rng.uniform(0.004, 0.020, len(ASSETS))
    y_variances[VarianceColumns.CLUSTER.value] = GROUPS
    return CurrentFactorCovarData(
        x_covar=pd.DataFrame(root @ root.T, index=FACTORS, columns=FACTORS),
        y_betas=pd.DataFrame(betas, index=ASSETS, columns=FACTORS),
        y_variances=y_variances,
        estimation_date=pd.Timestamp('2026-07-31'))


def make_result(shared_benchmark: bool = False,
                with_current: bool = True,
                with_groups: bool = True,
                with_bounds: bool = True,
                self_benchmarked: bool = False,
                current_as_frame: bool = False,
                cross_benchmarked: bool = False,
                with_expected_return: bool = True) -> PortfolioOptimisationResult:
    """Build a two-portfolio result, varying the options that select different branches."""
    rng = np.random.default_rng(SEED)
    weights = pd.DataFrame({'balanced': rng.dirichlet(np.ones(len(ASSETS))),
                            'defensive': rng.dirichlet(np.ones(len(ASSETS)))}, index=ASSETS)
    if self_benchmarked:
        benchmark = weights.copy()
    elif cross_benchmarked:
        # each portfolio is benchmarked against the *other* one, so both benchmark vectors
        # already appear in the portfolio section and both must be deduplicated away
        benchmark = pd.DataFrame({'balanced': weights['defensive'].to_numpy(),
                                  'defensive': weights['balanced'].to_numpy()}, index=ASSETS)
    elif shared_benchmark:
        benchmark = pd.Series(rng.dirichlet(np.ones(len(ASSETS))), index=ASSETS, name='benchmark')
    else:
        benchmark = pd.DataFrame({'balanced': rng.dirichlet(np.ones(len(ASSETS))),
                                  'defensive': rng.dirichlet(np.ones(len(ASSETS)))}, index=ASSETS)
    if not with_current:
        current = None
    elif current_as_frame:
        current = pd.DataFrame({'balanced': rng.dirichlet(np.ones(len(ASSETS))),
                                'defensive': rng.dirichlet(np.ones(len(ASSETS)))}, index=ASSETS)
    else:
        current = pd.Series(rng.dirichlet(np.ones(len(ASSETS))), index=ASSETS, name='current')
    group_attributions = ({'asset_class': pd.Series(GROUPS, index=ASSETS),
                           'region': pd.Series(REGIONS, index=ASSETS)} if with_groups else {})
    ac_bounds = (pd.DataFrame({'min': [0.0, 0.1, 0.0], 'max': [0.6, 0.5, 0.3]},
                              index=['Equities', 'Bonds', 'Alternatives'])
                 if with_bounds else None)
    return PortfolioOptimisationResult(
        weights=weights, benchmark_weights=benchmark, covar_data=make_covar_data(),
        group_attributions=group_attributions, current_weights=current,
        metadata=pd.DataFrame({'Asset Class': GROUPS, 'Region': REGIONS}, index=ASSETS),
        expected_return=(pd.Series(rng.normal(0.06, 0.02, len(ASSETS)), index=ASSETS)
                         if with_expected_return else None),
        ac_bounds=ac_bounds, reference_ccy='USD', portfolio_id='test',
        optimisation_date=pd.Timestamp('2026-07-31'))


# --------------------------------------------------------------------------- #
# construction and validation
# --------------------------------------------------------------------------- #
def test_properties_describe_the_universe() -> None:
    """the container reports its portfolios and assets straight off the weight frame"""
    result = make_result()
    assert result.portfolio_names == ['balanced', 'defensive']
    assert result.n_portfolios == 2
    assert result.n_assets == len(ASSETS)
    assert list(result.tickers) == list(ASSETS)
    assert result.has_current_weights is True


def test_a_series_of_weights_is_normalised_to_a_single_portfolio() -> None:
    """the single-portfolio shape is the DataFrame shape with one named column"""
    weights = pd.Series(np.full(len(ASSETS), 1.0 / len(ASSETS)), index=ASSETS, name='solo')
    result = PortfolioOptimisationResult(
        weights=weights, benchmark_weights=weights.rename('bench'),
        covar_data=make_covar_data(), group_attributions={})
    assert result.portfolio_names == ['solo']
    assert result.n_portfolios == 1
    assert result.has_current_weights is False
    assert result.current_active_weights_df is None
    assert result.trade_weights_df is None


def test_benchmark_columns_must_match_the_portfolio_columns() -> None:
    """a per-portfolio benchmark naming a portfolio that does not exist is rejected"""
    weights = pd.DataFrame({'a': [0.5, 0.5] + [0.0] * 4, 'b': [0.0] * 4 + [0.5, 0.5]},
                           index=ASSETS)
    benchmark = weights.rename(columns={'b': 'typo'})
    with pytest.raises(ValueError, match='Portfolio names mismatch'):
        PortfolioOptimisationResult(weights=weights, benchmark_weights=benchmark,
                                    covar_data=make_covar_data(), group_attributions={})


def test_current_weight_columns_must_match_the_portfolio_columns() -> None:
    """the same check guards current weights, which are optional but must align"""
    weights = pd.DataFrame({'a': [0.5, 0.5] + [0.0] * 4, 'b': [0.0] * 4 + [0.5, 0.5]},
                           index=ASSETS)
    with pytest.raises(ValueError, match='Portfolio names mismatch'):
        PortfolioOptimisationResult(weights=weights, benchmark_weights=weights,
                                    covar_data=make_covar_data(), group_attributions={},
                                    current_weights=weights.rename(columns={'b': 'typo'}))


def test_group_attributions_must_cover_every_ticker() -> None:
    """a grouping missing an asset would silently drop it from every attribution table"""
    weights = pd.DataFrame({'a': np.full(len(ASSETS), 1.0 / len(ASSETS))}, index=ASSETS)
    partial = pd.Series(GROUPS[:-1], index=ASSETS[:-1])
    with pytest.raises(ValueError, match='weights and group index mismatch'):
        PortfolioOptimisationResult(weights=weights, benchmark_weights=weights,
                                    covar_data=make_covar_data(),
                                    group_attributions={'asset_class': partial})


def test_current_weights_are_required_for_the_current_accessor() -> None:
    """asking for current weights that were never supplied raises rather than returning zeros"""
    result = make_result(with_current=False)
    assert result.has_current_weights is False
    with pytest.raises(ValueError, match='current_weights not provided'):
        result.get_current('balanced')


# --------------------------------------------------------------------------- #
# weight algebra
# --------------------------------------------------------------------------- #
def test_active_and_trade_weights_are_the_stated_differences() -> None:
    """active is weights minus benchmark, trade is weights minus current"""
    result = make_result()
    for name in result.portfolio_names:
        expected_active = result.get_weights(name) - result.get_benchmark(name)
        pd.testing.assert_series_equal(result.get_active_weights(name), expected_active)
        expected_trade = result.get_weights(name) - result.get_current(name)
        pd.testing.assert_series_equal(result.get_trade_weights(name), expected_trade)
    pd.testing.assert_frame_equal(result.active_weights_df,
                                  result._weights_df - result._benchmark_df)
    pd.testing.assert_frame_equal(result.trade_weights_df,
                                  result._weights_df - result._current_df)
    pd.testing.assert_frame_equal(result.current_active_weights_df,
                                  result._current_df - result._benchmark_df)


def test_turnover_is_half_the_absolute_trade_and_the_analysis_agrees() -> None:
    """one-way turnover halves the two-way total, and the breakdown reconciles to it"""
    result = make_result()
    two_way = result.compute_turnover(name='balanced', one_way=False)
    one_way = result.compute_turnover(name='balanced', one_way=True)
    assert one_way == pytest.approx(two_way / 2.0)
    assert two_way == pytest.approx(result.get_trade_weights('balanced').abs().sum())

    analysis = result.compute_turnover_analysis(name='balanced')
    assert set(analysis.index) == {'turnover', 'buys', 'sells', 'n_trades',
                                   'avg_trade_size', 'max_trade_size'}
    # buys and sells are both reported positive and together make the two-way total
    assert analysis['buys'] + analysis['sells'] == pytest.approx(two_way)
    assert analysis['turnover'] == pytest.approx(one_way)
    assert analysis['max_trade_size'] >= analysis['avg_trade_size']


def test_portfolio_vol_is_the_square_root_of_the_quadratic_form() -> None:
    """vol and variance stay consistent, and the benchmark-relative vol is the TE"""
    result = make_result()
    weights = result.get_weights('balanced')
    variance = result.compute_portfolio_variance(weights)
    assert result.compute_portfolio_vol(weights) == pytest.approx(np.sqrt(variance))
    # computed the other way, straight off the covariance
    covar = result.covar_data.y_covar.loc[ASSETS, ASSETS].to_numpy()
    assert variance == pytest.approx(float(weights.to_numpy() @ covar @ weights.to_numpy()))
    # with no argument the first portfolio is used
    assert result.compute_portfolio_variance() == pytest.approx(variance)


# --------------------------------------------------------------------------- #
# group attribution
# --------------------------------------------------------------------------- #
def test_group_attribution_returns_one_block_per_group() -> None:
    """each grouping yields weight, risk-contribution and percentage tables plus bounds"""
    result = make_result()
    attribution = result.compute_group_attribution()
    assert set(attribution) == {'asset_class', 'region'}
    for group_name, block in attribution.items():
        assert {'weight', 'rc', 'rc_pct'} <= set(block)
        assert 'bounds' in block if group_name == 'asset_class' else True
        weight_table = block['weight']
        # rows are the labelled weight vectors, columns the group levels
        assert 'balanced' in weight_table.index
        assert 'defensive' in weight_table.index
        # every portfolio's weights sum to that portfolio's total across the group levels
        for name in result.portfolio_names:
            assert weight_table.loc[name].sum() == pytest.approx(
                result.get_weights(name).sum())
        # risk percentages are shares of one
        for row in block['rc_pct'].index:
            assert block['rc_pct'].loc[row].sum() == pytest.approx(1.0)


def test_group_attribution_labels_cover_portfolios_benchmarks_and_actives() -> None:
    """the labelled vector list names every series the factsheet shows"""
    result = make_result()
    labels = [label for label, _ in result._get_all_labelled_weight_vectors()]
    assert labels[:2] == ['balanced', 'defensive']
    assert 'current' in labels
    assert 'balanced_active' in labels and 'defensive_active' in labels
    assert any(label.endswith('_benchmark') for label in labels)
    assert len(labels) == len(set(labels)), f"duplicate labels: {labels}"


def test_self_benchmarked_portfolios_contribute_no_active_or_benchmark_vector() -> None:
    """a portfolio benchmarked against itself has a zero active weight and is skipped"""
    result = make_result(self_benchmarked=True)
    labels = [label for label, _ in result._get_all_labelled_weight_vectors()]
    assert not any(label.endswith('_active') and not label.startswith('current')
                   for label in labels)
    assert not any(label.endswith('_benchmark') for label in labels)
    assert labels[:2] == ['balanced', 'defensive']


def test_shared_benchmark_is_listed_once_under_its_own_name() -> None:
    """one Series benchmark broadcast to every portfolio appears as a single vector"""
    result = make_result(shared_benchmark=True)
    labels = [label for label, _ in result._get_all_labelled_weight_vectors()]
    assert labels.count('benchmark') == 1


def test_a_benchmark_that_is_another_portfolio_is_listed_once() -> None:
    """cross-benchmarked portfolios must not restate each other's weights as benchmarks

    With ``balanced`` benchmarked against ``defensive`` and vice versa, both benchmark
    vectors are already in the portfolio section. Listing them again would put the same
    weights in the attribution tables twice under two names — the duplication the key-based
    skip exists to prevent, and one that reads as a real second position.
    """
    result = make_result(cross_benchmarked=True)
    labels = [label for label, _ in result._get_all_labelled_weight_vectors()]
    assert not any(label.endswith('_benchmark') for label in labels)
    assert len(labels) == len(set(labels)), f"duplicate labels: {labels}"
    # the active vectors are still there: the portfolios genuinely differ from each other
    assert {'balanced_active', 'defensive_active'} <= set(labels)


def test_per_portfolio_current_weights_are_labelled_per_portfolio() -> None:
    """a current-weights frame means each mandate has its own starting portfolio

    The Series form is one live book shared by every candidate portfolio; the frame form is
    one book per mandate. Collapsing the second into the first would compute every mandate's
    trade against somebody else's holdings.
    """
    result = make_result(current_as_frame=True)
    labels = [label for label, _ in result._get_all_labelled_weight_vectors()]
    assert {'balanced_current', 'defensive_current'} <= set(labels)
    assert {'balanced_current_active', 'defensive_current_active'} <= set(labels)
    assert 'current' not in labels
    # and the accessor returns that portfolio's own column
    pd.testing.assert_series_equal(result.get_current('balanced'),
                                   result.current_weights['balanced'], check_names=False)


def test_per_portfolio_current_weights_are_reordered_onto_the_weight_columns() -> None:
    """the frame is aligned to the weight columns, not trusted to arrive in their order"""
    result = make_result(current_as_frame=True)
    swapped = result.current_weights[['defensive', 'balanced']]
    realigned = PortfolioOptimisationResult(
        weights=result.weights, benchmark_weights=result.benchmark_weights,
        covar_data=result.covar_data, group_attributions={}, current_weights=swapped)
    pd.testing.assert_series_equal(realigned.get_current('balanced'),
                                   result.get_current('balanced'))


def test_group_allocation_aggregates_one_portfolio_over_group_levels() -> None:
    """the allocation of a portfolio sums its weights within each group level"""
    result = make_result()
    allocation = result.compute_group_allocation('asset_class', name='balanced')
    assert set(allocation.index) == {'Equities', 'Bonds', 'Alternatives'}
    assert allocation.sum() == pytest.approx(result.get_weights('balanced').sum())
    rounded = result.compute_group_allocation('asset_class', name='balanced',
                                              weights_to_pct=True)
    assert rounded.sum() == pytest.approx(100.0, abs=1.0)


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def test_weight_summary_carries_every_weight_column() -> None:
    """with current weights present the summary gains the current and trade columns"""
    result = make_result()
    summary = result.compute_weight_summary(name='balanced')
    assert list(summary.columns) == ['new', 'benchmark', 'active', 'current', 'trade']
    assert np.allclose(summary['active'], summary['new'] - summary['benchmark'])
    assert np.allclose(summary['trade'], summary['new'] - summary['current'])


def test_weight_summary_in_percent_keeps_the_differences_consistent() -> None:
    """rounding to percent recomputes active and trade so the columns still reconcile"""
    summary = make_result().compute_weight_summary(name='balanced', weights_to_pct=True)
    assert np.allclose(summary['active'], summary['new'] - summary['benchmark'])
    assert np.allclose(summary['trade'], summary['new'] - summary['current'])
    assert summary['new'].sum() == pytest.approx(100.0, abs=1.0)


def test_weight_summary_without_current_weights_stops_at_active() -> None:
    """the current and trade columns exist only when current weights were supplied"""
    summary = make_result(with_current=False).compute_weight_summary()
    assert list(summary.columns) == ['new', 'benchmark', 'active']


def test_all_weights_summary_is_tickers_by_portfolio() -> None:
    """the cross-portfolio view transposes the per-portfolio summary"""
    result = make_result()
    tables = result.compute_all_weights_summary()
    assert set(tables) == {'portfolio', 'benchmark', 'active', 'current', 'trade'}
    for table in tables.values():
        assert list(table.index) == list(ASSETS)
        assert list(table.columns) == result.portfolio_names
    assert np.allclose(tables['active'], tables['portfolio'] - tables['benchmark'])
    assert set(make_result(with_current=False).compute_all_weights_summary()) == {
        'portfolio', 'benchmark', 'active'}


def test_all_weights_summary_can_report_whole_percentage_points() -> None:
    """the percent form rounds every column to integers that still sum to the total

    A naive round-to-percent leaves the column summing to 99 or 101, which shows up on a
    factsheet as a portfolio that is not fully invested. ``round_weights_to_pct`` corrects
    for that, and applying it column by column is what keeps each portfolio's total intact.
    """
    result = make_result()
    tables = result.compute_all_weights_summary(weights_to_pct=True)
    portfolio = tables['portfolio']
    for name in result.portfolio_names:
        assert portfolio[name].sum() == pytest.approx(100.0, abs=1.0)
    # the raw form is untouched, so the rounding is a view rather than a mutation
    assert result.compute_all_weights_summary()['portfolio'].max().max() < 1.0


def test_factor_exposures_summary_splits_and_aggregates() -> None:
    """six tables by default, two when aggregated, over the same factor columns"""
    result = make_result()
    split = result.compute_factor_exposures_summary(aggregate=False)
    assert set(split) == {'exposure_portfolio', 'exposure_benchmark', 'exposure_active',
                          'risk_pct_portfolio', 'risk_pct_benchmark', 'risk_pct_active'}
    aggregated = result.compute_factor_exposures_summary(aggregate=True)
    assert set(aggregated) == {'factor_exposures', 'factor_risk_pct'}
    # the aggregate view holds the same rows, merged rather than recomputed
    assert len(aggregated['factor_exposures']) >= len(split['exposure_portfolio'])


def test_factor_exposures_summary_lists_a_shared_benchmark_once() -> None:
    """one benchmark for every portfolio is one row, under the benchmark's own name"""
    result = make_result(shared_benchmark=True)
    split = result.compute_factor_exposures_summary(aggregate=False)
    assert list(split['exposure_benchmark'].index) == ['benchmark']
    assert list(split['exposure_benchmark'].columns) == list(FACTORS)


def test_factor_exposures_summary_skips_a_self_benchmarked_portfolio() -> None:
    """with zero active weights there is no active exposure to report, and no benchmark row

    The benchmark rows would repeat the portfolio rows exactly, and the active rows would be
    a table of zeros — both are noise on a factsheet, and the zero row in particular reads as
    a measured result rather than as an absent one.
    """
    result = make_result(self_benchmarked=True)
    split = result.compute_factor_exposures_summary(aggregate=False)
    assert split['exposure_benchmark'].empty
    assert split['exposure_active'].empty
    assert list(split['exposure_portfolio'].index) == result.portfolio_names


def test_factor_exposures_summary_deduplicates_a_cross_benchmark() -> None:
    """a benchmark already present as a portfolio is not restated in the benchmark block"""
    split = make_result(cross_benchmarked=True).compute_factor_exposures_summary()
    assert split['exposure_benchmark'].empty
    assert not split['exposure_active'].empty


def test_risk_summary_splits_and_aggregates() -> None:
    """the risk view offers the same split/aggregate choice as the exposures view"""
    result = make_result()
    assert isinstance(result.compute_risk_summary(aggregate=False), dict)
    assert isinstance(result.compute_risk_summary(aggregate=True), dict)


def test_risk_summary_lists_a_shared_benchmark_once() -> None:
    """the benchmark block collapses to the one benchmark every portfolio shares"""
    split = make_result(shared_benchmark=True).compute_risk_summary(aggregate=False)
    assert list(split['benchmark'].index) == ['benchmark']


def test_risk_summary_skips_self_and_cross_benchmarked_vectors() -> None:
    """the same two skips as the exposures view, so the two tables stay row-comparable

    A factsheet puts these side by side. If one deduplicates and the other does not, the
    reader is left comparing a four-row exposure table with a six-row risk table and no
    indication which rows correspond.
    """
    self_split = make_result(self_benchmarked=True).compute_risk_summary(aggregate=False)
    assert self_split['benchmark'].empty
    cross_split = make_result(cross_benchmarked=True).compute_risk_summary(aggregate=False)
    assert cross_split['benchmark'].empty


def test_the_returns_risk_snapshot_reports_a_benchmark_per_portfolio() -> None:
    """with per-portfolio benchmarks each gets its own column, unlike the shared case

    Tracking error is only on the portfolio rows: a benchmark has no tracking error against
    itself, so the benchmark columns carry the risk decomposition alone.
    """
    result = make_result()
    snapshot = result.compute_returns_risk_snapshot()
    assert {'balanced', 'defensive', 'balanced_benchmark', 'defensive_benchmark'} <= set(
        snapshot.columns)
    assert snapshot.loc['tracking_error', 'balanced'] > 0.0
    assert pd.isna(snapshot.loc['tracking_error', 'balanced_benchmark'])
    # the shared case names the benchmark once instead
    shared = make_result(shared_benchmark=True).compute_returns_risk_snapshot()
    assert 'benchmark' in shared.columns
    assert 'balanced_benchmark' not in shared.columns


def test_summary_reports_one_line_of_metrics_per_portfolio() -> None:
    """the one-line summary names the portfolio and includes turnover when known"""
    result = make_result()
    summary = result.summary(name='defensive')
    assert summary.name == 'defensive_summary'
    assert summary['n_assets'] == len(ASSETS)
    assert summary['total_vol'] > 0.0
    assert summary['turnover'] == pytest.approx(result.compute_turnover(name='defensive'))
    assert 'n_trades' in summary
    assert 'turnover' not in make_result(with_current=False).summary()


def test_to_weights_df_matches_the_weight_summary_columns() -> None:
    """the simple comparison frame is the summary without the percent handling"""
    result = make_result()
    frame = result.to_weights_df(name='balanced')
    assert list(frame.columns) == ['new', 'benchmark', 'active', 'current', 'trade']
    pd.testing.assert_series_equal(frame['new'], result.get_weights('balanced').rename('new'))
    assert list(make_result(with_current=False).to_weights_df().columns) == [
        'new', 'benchmark', 'active']


# --------------------------------------------------------------------------- #
# asset tables and the report bundle
# --------------------------------------------------------------------------- #
def test_assets_metadata_appends_the_expected_return() -> None:
    """the expected return joins the metadata under the requested column name"""
    result = make_result()
    table = result.get_assets_metadata(return_name='CMA')
    assert 'CMA' in table.columns
    assert 'Asset Class' in table.columns
    # passing None leaves the metadata untouched
    assert 'CMA' not in result.get_assets_metadata(return_name=None).columns


def test_asset_betas_table_leads_with_the_expected_return() -> None:
    """the covariance snapshot per asset is prefixed by the return column"""
    table = make_result().get_asset_betas_table(return_name='CMA')
    assert table.columns[0] == 'CMA'
    assert list(table.index) == list(ASSETS)


def test_combined_asset_weight_table_joins_metadata_to_weights() -> None:
    """one row per asset carrying both its metadata and its weight columns"""
    table = make_result().get_combined_asset_weight_table()
    assert list(table.index) == list(ASSETS)
    assert {'Asset Class', 'new', 'benchmark', 'active'} <= set(table.columns)


def test_report_bundles_every_section() -> None:
    """the report carries weights, risk, asset detail, turnover and the group blocks"""
    result = make_result()
    report = result.report(name='balanced')
    assert {'weights', 'risk_summary', 'asset_snapshot', 'turnover'} <= set(report)
    assert {'asset_class', 'region'} <= set(report)
    # add_asset_details prepends the metadata columns to the weights table
    assert 'Asset Class' in report['weights'].columns
    assert 'Asset Class' not in result.report(name='balanced',
                                              add_asset_details=False)['weights'].columns
    snapshot = report['asset_snapshot']
    assert {'CMA', 'weight', 'active_weight', 'current_weight', 'trade'} <= set(snapshot.columns)


def test_report_without_current_weights_omits_turnover() -> None:
    """turnover needs a starting portfolio, so the section is absent without one"""
    assert 'turnover' not in make_result(with_current=False).report()


def test_report_without_expected_returns_drops_the_cma_column_only() -> None:
    """the asset snapshot is built on the tickers when there is no return forecast

    ``expected_return`` is optional — a risk-only mandate has none — and the snapshot must
    still carry the weights. Seeding the frame from an absent forecast would give an empty
    index and drop every asset row silently.
    """
    snapshot = make_result(with_expected_return=False).report(name='balanced')[
        'asset_snapshot']
    assert list(snapshot.index) == list(ASSETS)
    assert 'CMA' not in snapshot.columns
    assert {'weight', 'active_weight', 'current_weight', 'trade'} <= set(snapshot.columns)


def test_efficient_frontier_data_defaults_to_one_profile() -> None:
    """with no profiles given, every portfolio lands in a single 'all' profile"""
    result = make_result()
    frame, blocks = result.compute_efficient_frontier_data()
    assert set(blocks) == {'all - portfolio', 'all - benchmark'}
    assert list(blocks['all - portfolio'].index) == result.portfolio_names
    assert {'exp_return', 'total_vol'} <= set(blocks['all - portfolio'].columns)
    assert {'mandate', 'hue'} <= set(frame.columns)
    assert set(frame['hue']) == set(blocks)
    assert (blocks['all - portfolio']['total_vol'] > 0).all()


def test_efficient_frontier_data_splits_by_named_profile() -> None:
    """named profiles partition the portfolios into their own blocks"""
    result = make_result()
    _, blocks = result.compute_efficient_frontier_data(
        profiles={'core': ['balanced'], 'cautious': ['defensive']})
    assert set(blocks) == {'core - portfolio', 'core - benchmark',
                           'cautious - portfolio', 'cautious - benchmark'}
    assert list(blocks['core - portfolio'].index) == ['balanced']


def test_repr_states_vol_tracking_error_and_turnover() -> None:
    """the one-line repr is the object's headline and mentions turnover when known"""
    text = repr(make_result())
    assert text.startswith('PortfolioOptimisationResult(')
    assert 'n_portfolios=2' in text
    assert f'n_assets={len(ASSETS)}' in text
    assert 'vol=' in text and 'TE=' in text and 'turnover=' in text
    assert 'turnover=' not in repr(make_result(with_current=False))
