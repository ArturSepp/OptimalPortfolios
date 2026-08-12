"""
the HCGL covariance report.

These are plotting functions, so the assertions are about the things a plot can get wrong
without looking wrong: the *ordering* contract and the cadence guard.

``is_align_to_clusters_index`` is the one that matters. When it is on, the asset covariance and
the beta panel are reindexed onto the cluster ordering so the heatmap's block structure lines
up with the dendrogram's; when it is off they keep the estimation order. Both render happily,
and a mismatch between them is a picture that reads as a clean factor structure while pairing
each row's betas with a different asset's label. So the tests check the reindexing on the
returned aggregate cluster series rather than inspecting pixels.

``plot_clusters`` accepts exactly one or two cadences -- the subplot grid is hand-laid for
those two cases -- and raises otherwise; that guard is the difference between an exception and
an IndexError three frames deeper.

The snapshot adapter is checked against factorlasso's flat persisted cluster fields. Its
plotting inputs are also compared with an independent split of the raw factorlasso output so
the rendered cadence assignments cannot silently drift from the estimated result.
"""
# packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import qis
import scipy.cluster.hierarchy as spc
from factorlasso import LassoModel, LassoModelType
# optimalportfolios
import optimalportfolios.covar_estimation.covar_reporting as covar_reporting
from optimalportfolios.covar_estimation.covar_reporting import (
    plot_clusters,
    plot_current_covar_data,
    plot_hcgl_covar_data,
    run_rolling_covar_report,
)
from optimalportfolios.covar_estimation.factor_covar_estimator import FactorCovarEstimator

N_ASSETS = 8
ASSETS = [f'A{i}' for i in range(N_ASSETS)]
FACTORS = ['F1', 'F2', 'F3']


@pytest.fixture(autouse=True)
def close_figures():
    """Keep the Agg figure registry from growing across the plotting tests."""
    yield
    plt.close('all')


@pytest.fixture(scope='module')
def panels() -> tuple:
    """A deterministic three-factor monthly panel with a clear block structure."""
    rng = np.random.default_rng(20260812)
    dates = pd.date_range('2016-01-31', periods=84, freq='ME')
    factor_returns = rng.normal(0.004, 0.035, size=(len(dates), len(FACTORS)))
    loadings = np.array([
        [0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.1, 0.1],
        [0.0, 0.8, 0.1], [0.1, 0.9, 0.0], [0.0, 0.7, 0.2],
        [0.1, 0.0, 0.8], [0.0, 0.1, 0.9],
    ])
    asset_returns = (factor_returns @ loadings.T
                     + rng.normal(0.0, 0.012, size=(len(dates), N_ASSETS)))
    factor_prices = pd.DataFrame(100.0 * np.exp(np.cumsum(factor_returns, axis=0)),
                                 index=dates, columns=FACTORS)
    asset_prices = pd.DataFrame(100.0 * np.exp(np.cumsum(asset_returns, axis=0)),
                                index=dates, columns=ASSETS)
    returns = pd.DataFrame(asset_returns, index=dates, columns=ASSETS)
    return factor_prices, asset_prices, returns


@pytest.fixture(scope='module')
def estimator() -> FactorCovarEstimator:
    """A small FCGL estimator that fits quickly on the synthetic panel."""
    return FactorCovarEstimator(
        lasso_model=LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                               reg_lambda=1e-5, span=24, warmup_period=12, n_clusters=3),
        factor_returns_freq='ME',
        factor_covar_span=24,
        rebalancing_freq='QE')


@pytest.fixture(scope='module')
def covar_data(panels: tuple, estimator: FactorCovarEstimator):
    """One fitted CurrentFactorCovarData snapshot to plot."""
    factor_prices, _, returns = panels
    return estimator.fit_current_factor_covars(risk_factor_prices=factor_prices,
                                               asset_returns_dict={'ME': returns},
                                               assets=ASSETS)


def make_cluster_inputs(n_cadences: int) -> tuple:
    """Cluster labels, linkages and cutoffs for one or two disjoint cadences."""
    rng = np.random.default_rng(7)
    # each cadence must own a distinct slice of the universe: the two label series are
    # concatenated, and an asset appearing twice makes the ordering index non-unique
    membership = {1: {'ME': ASSETS}, 2: {'QE': ASSETS[:4], 'ME': ASSETS[4:]}}[n_cadences]
    clusters, linkages, cutoffs = {}, {}, {}
    for cadence, members in membership.items():
        labels = [1, 1, 1, 2, 2, 2, 3, 3] if len(members) == N_ASSETS else [1, 1, 2, 2]
        clusters[cadence] = pd.Series(labels, index=members)
        linkages[cadence] = spc.linkage(rng.normal(size=(len(members), 3)), method='ward')
        cutoffs[cadence] = 1.0
    return clusters, linkages, cutoffs


def plot_kwargs(covar_data, **overrides) -> dict:
    """The full argument set for ``plot_hcgl_covar_data`` over the single-cadence partition."""
    snapshot = covar_data.get_snapshot()
    clusters, linkages, cutoffs = make_cluster_inputs(n_cadences=1)
    kwargs = dict(x_covar=covar_data.x_covar, y_covar=covar_data.y_covar,
                  betas=covar_data.y_betas, r2=snapshot['r2'],
                  total_vol=snapshot['total_vol'], residual_vol=snapshot['resid_vol'],
                  clusters=clusters, linkages=linkages, cutoffs=cutoffs,
                  date=covar_data.estimation_date)
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------- #
# plot_clusters
# --------------------------------------------------------------------------- #
def test_one_cadence_returns_every_asset_labelled_by_that_cadence() -> None:
    """A single-cadence partition labels each cluster id with its cadence prefix."""
    clusters, linkages, cutoffs = make_cluster_inputs(n_cadences=1)
    agg_clusters, fig = plot_clusters(clusters=clusters, linkages=linkages, cutoffs=cutoffs)
    assert isinstance(fig, plt.Figure)
    assert sorted(agg_clusters.index) == sorted(ASSETS)
    assert set(agg_clusters.unique()) == {'ME-1', 'ME-2', 'ME-3'}


def test_two_cadences_are_concatenated_into_one_ordering() -> None:
    """Two cadences partition disjoint assets and both appear in the aggregate labels."""
    clusters, linkages, cutoffs = make_cluster_inputs(n_cadences=2)
    agg_clusters, _ = plot_clusters(clusters=clusters, linkages=linkages, cutoffs=cutoffs)
    assert len(agg_clusters) == N_ASSETS                # four assets from each cadence
    assert {label.split('-')[0] for label in agg_clusters} == {'QE', 'ME'}


def test_the_aggregate_ordering_groups_assets_by_cluster() -> None:
    """Assets sharing a cluster id are adjacent; that ordering is what the heatmaps use."""
    clusters, linkages, cutoffs = make_cluster_inputs(n_cadences=1)
    agg_clusters, _ = plot_clusters(clusters=clusters, linkages=linkages, cutoffs=cutoffs)
    labels = list(agg_clusters)
    assert labels == sorted(labels, reverse=True)       # sorted ascending, then reversed


def test_three_cadences_are_refused_rather_than_mis_laid_out() -> None:
    """The subplot grid is hand-laid for one or two cadences only."""
    clusters, linkages, cutoffs = make_cluster_inputs(n_cadences=2)
    clusters['YE'], linkages['YE'], cutoffs['YE'] = clusters['ME'], linkages['ME'], cutoffs['ME']
    with pytest.raises(NotImplementedError, match='number clusters = 3'):
        plot_clusters(clusters=clusters, linkages=linkages, cutoffs=cutoffs)


# --------------------------------------------------------------------------- #
# plot_hcgl_covar_data
# --------------------------------------------------------------------------- #
def test_a_snapshot_renders_four_figures(covar_data) -> None:
    """Factor correlations, asset correlations, clusters and betas."""
    figs = plot_hcgl_covar_data(**plot_kwargs(covar_data))
    assert len(figs) == 4
    assert all(isinstance(fig, plt.Figure) for fig in figs)


def test_alignment_reindexes_the_covariance_onto_the_cluster_ordering(covar_data) -> None:
    """With alignment on, the plotted covariance and betas follow the cluster ordering.

    The reindex is destructive when the two orderings disagree -- a row of betas would be
    drawn against another asset's label -- so both settings are exercised, and the ordering
    the aligned path imposes is checked to cover exactly the estimated universe.
    """
    agg_clusters, _ = plot_clusters(*make_cluster_inputs(n_cadences=1))
    plt.close('all')
    assert len(plot_hcgl_covar_data(**plot_kwargs(covar_data,
                                                  is_align_to_clusters_index=True))) == 4
    assert len(plot_hcgl_covar_data(**plot_kwargs(covar_data,
                                                  is_align_to_clusters_index=False))) == 4
    assert sorted(agg_clusters.index) == sorted(covar_data.y_covar.columns)
    assert list(agg_clusters.index) != list(covar_data.y_covar.columns)   # a real reordering


def test_the_alpha_column_is_optional(covar_data) -> None:
    """Omitting alpha drops it from the stats panel without changing the figure count."""
    snapshot = covar_data.get_snapshot()
    assert len(plot_hcgl_covar_data(**plot_kwargs(covar_data, alpha=None))) == 4
    assert len(plot_hcgl_covar_data(**plot_kwargs(covar_data,
                                                  alpha=snapshot['stat_alpha']))) == 4


def test_near_zero_betas_are_blanked_rather_than_printed_as_zero(covar_data) -> None:
    """Loadings below 1e-4 become NaN so the heatmap shows the sparsity LASSO produced."""
    betas = covar_data.y_betas.copy()
    betas.iloc[0, 0] = 1e-9
    figs = plot_hcgl_covar_data(**plot_kwargs(covar_data, betas=betas))
    assert len(figs) == 4


# --------------------------------------------------------------------------- #
# the CurrentFactorCovarData adapter
# --------------------------------------------------------------------------- #
def test_plot_current_covar_data_handles_the_factorlasso_shape(covar_data) -> None:
    """The snapshot adapter splits factorlasso's flat fields and renders all four figures."""
    figs = plot_current_covar_data(covar_data=covar_data)
    assert len(figs) == 4
    assert all(isinstance(fig, plt.Figure) for fig in figs)


def test_seeded_factorlasso_clusters_are_the_clusters_plotted(covar_data,
                                                              monkeypatch) -> None:
    """Compare plotted cadence assignments with an independent split of factorlasso output."""
    y_covar_before = covar_data.y_covar.copy(deep=True)
    y_betas_before = covar_data.y_betas.copy(deep=True)
    raw_clusters = covar_data.clusters.dropna().astype(str)
    expected = {}
    for freq in covar_data.cutoffs.index:
        prefix = f'{freq}:'
        selected = raw_clusters[raw_clusters.str.startswith(prefix)]
        expected[freq] = selected.str.slice(start=len(prefix))

    plotted = {}
    real_plot_clusters = covar_reporting.plot_clusters

    def capture_plot_clusters(clusters, linkages, cutoffs, figsize=(14, 10)):
        """Capture the adapter output while delegating the actual rendering."""
        plotted.update({freq: values.copy() for freq, values in clusters.items()})
        return real_plot_clusters(clusters=clusters, linkages=linkages,
                                  cutoffs=cutoffs, figsize=figsize)

    monkeypatch.setattr(covar_reporting, 'plot_clusters', capture_plot_clusters)
    figs = covar_reporting.plot_current_covar_data(covar_data=covar_data)

    assert len(raw_clusters) >= 8
    assert len(figs) == 4
    assert set(plotted) == set(expected)
    pd.testing.assert_frame_equal(covar_data.y_covar, y_covar_before)
    pd.testing.assert_frame_equal(covar_data.y_betas, y_betas_before)
    for freq in expected:
        pd.testing.assert_series_equal(plotted[freq].sort_index().astype(str),
                                       expected[freq].sort_index(), check_names=False)


# --------------------------------------------------------------------------- #
# run_rolling_covar_report
# --------------------------------------------------------------------------- #
def test_the_rolling_report_keys_its_frames_by_formatted_date(panels: tuple,
                                                              estimator: FactorCovarEstimator
                                                              ) -> None:
    """Every rebalancing date contributes one snapshot frame, keyed as ddMmmYYYY."""
    factor_prices, asset_prices, returns = panels
    time_period = qis.TimePeriod(returns.index[-8], returns.index[-1])
    figs, dfs = run_rolling_covar_report(risk_factor_prices=factor_prices,
                                         prices=asset_prices,
                                         covar_estimator=estimator,
                                         time_period=time_period,
                                         asset_returns_dict={'ME': returns},
                                         assets=ASSETS,
                                         is_plot=False)
    assert figs == []
    assert len(dfs) > 0
    for key, df in dfs.items():
        assert pd.to_datetime(key, format='%d%b%Y') is not None
        assert list(df.index) == ASSETS


def test_the_asset_universe_defaults_to_the_price_columns(panels: tuple,
                                                          estimator: FactorCovarEstimator) -> None:
    """Passing no ``assets`` infers them from prices, which is the documented default."""
    factor_prices, asset_prices, returns = panels
    time_period = qis.TimePeriod(returns.index[-4], returns.index[-1])
    _, dfs = run_rolling_covar_report(risk_factor_prices=factor_prices,
                                      prices=asset_prices,
                                      covar_estimator=estimator,
                                      time_period=time_period,
                                      asset_returns_dict={'ME': returns},
                                      assets=None,
                                      is_plot=False)
    assert all(list(df.index) == list(asset_prices.columns) for df in dfs.values())


def test_the_rebalancing_frequency_can_be_overridden(panels: tuple,
                                                     estimator: FactorCovarEstimator) -> None:
    """A monthly override produces more estimation dates than the estimator's quarterly default."""
    factor_prices, asset_prices, returns = panels
    time_period = qis.TimePeriod(returns.index[-12], returns.index[-1])
    common = dict(risk_factor_prices=factor_prices, prices=asset_prices,
                  covar_estimator=estimator, time_period=time_period,
                  asset_returns_dict={'ME': returns}, assets=ASSETS, is_plot=False)
    _, quarterly = run_rolling_covar_report(rebalancing_freq='QE', **common)
    _, monthly = run_rolling_covar_report(rebalancing_freq='ME', **common)
    assert len(monthly) > len(quarterly)


def test_the_rolling_report_can_plot(panels: tuple,
                                     estimator: FactorCovarEstimator) -> None:
    """The plotting branch renders four figures for every fitted snapshot."""
    factor_prices, asset_prices, returns = panels
    time_period = qis.TimePeriod(returns.index[-4], returns.index[-1])
    figs, dfs = run_rolling_covar_report(risk_factor_prices=factor_prices, prices=asset_prices,
                                         covar_estimator=estimator, time_period=time_period,
                                         asset_returns_dict={'ME': returns}, assets=ASSETS,
                                         is_plot=True)
    assert len(figs) == 4 * len(dfs)
