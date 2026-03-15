"""
Report clustering of covar matrix.

Uses FactorCovarEstimator.fit_rolling_factor_covars() for rolling estimation
and CurrentFactorCovarData for per-date diagnostics.

Reference:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
    Available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5250221
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as spc
import qis as qis
from matplotlib.colors import ListedColormap
from typing import List, Dict, Tuple, Optional, Union

from optimalportfolios.covar_estimation.factor_covar_estimator import FactorCovarEstimator
from optimalportfolios.covar_estimation.factor_covar_data import CurrentFactorCovarData


def plot_current_covar_data(covar_data: CurrentFactorCovarData,
                            **kwargs
                            ) -> List[plt.Figure]:

    df = covar_data.get_snapshot()
    figs = plot_hcgl_covar_data(x_covar=covar_data.x_covar,
                                y_covar=covar_data.y_covar,
                                betas=covar_data.y_betas,
                                r2=df['r2'],
                                total_vol=df['total_vol'],
                                residual_vol=df['resid_vol'],
                                alpha=df['stat_alpha'],
                                clusters=covar_data.clusters,
                                linkages=covar_data.linkages,
                                cutoffs=covar_data.cutoffs,
                                date=covar_data.estimation_date,
                                **kwargs)
    return figs


def plot_hcgl_covar_data(x_covar: pd.DataFrame,
                         y_covar: pd.DataFrame,
                         betas: pd.DataFrame,
                         r2: pd.Series,
                         total_vol: pd.Series,
                         residual_vol: pd.Series,
                         clusters: Dict[str, pd.Series],
                         linkages: Dict[str, np.ndarray],
                         cutoffs: Dict[str, float],
                         date: pd.Timestamp,
                         alpha: pd.Series = None,
                         figsize: Tuple[float, float] = (14, 10),
                         is_align_to_clusters_index: bool = True,
                         **kwargs
                         ) -> List[plt.Figure]:
    """
    Plot covariance analysis for a single date.

    Returns:
        List of figures with factor/asset correlations, clusters, and betas.
    """
    figs = []

    # Prepare betas and stats
    betas = betas.where(np.abs(betas) > 1e-4, other=np.nan)
    if alpha is not None:
        df = pd.concat([r2.clip(0.0, None), total_vol, residual_vol, alpha], axis=1)
    else:
        df = pd.concat([r2.clip(0.0, None), total_vol, residual_vol], axis=1)

    agg_clusters, fig_clusters = plot_clusters(
        clusters=clusters,
        linkages=linkages,
        cutoffs=cutoffs,
        figsize=figsize
    )

    if is_align_to_clusters_index:
        y_covar = y_covar.loc[agg_clusters.index, agg_clusters.index]

    # Plot factor correlations
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    qis.set_suptitle(fig, title=f"{date.strftime('%d%b%Y')}: Factor correlations / vols")
    figs.append(fig)
    qis.plot_corr_matrix_from_covar(covar=x_covar,
                                    corr_format='{:.2f}',
                                    ax=ax,
                                    **kwargs)

    # Plot asset correlations
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    qis.set_suptitle(fig, title=f"{date.strftime('%d%b%Y')}: Asset correlations / vols")
    figs.append(fig)
    qis.plot_corr_matrix_from_covar(covar=y_covar,
                                    corr_format='{:.1f}',
                                    ax=ax,
                                    **kwargs)

    # Add clusters figure
    qis.set_suptitle(fig_clusters, title=f"{date.strftime('%d%b%Y')}: Clusters")
    figs.append(fig_clusters)

    # Plot betas and R²
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=4, wspace=0.0, hspace=0.0)
    axs = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])]
    qis.set_suptitle(fig, title=f"{date.strftime('%d%b%Y')}: Universe betas")
    figs.append(fig)

    hline_rows = qis.get_table_lines_for_group_data(agg_clusters)
    if is_align_to_clusters_index:
        betas = betas.loc[agg_clusters.index, :]
        df = df.loc[agg_clusters.index, :]

    qis.plot_heatmap(df=betas, var_format='{:.2f}', hline_rows=hline_rows, ax=axs[0])

    # Create custom colormap
    colors = ['white']
    custom_cmap = ListedColormap(colors)
    qis.plot_heatmap(df=df, cmap=custom_cmap, var_format='{:.1%}',
                     hline_rows=hline_rows, ax=axs[1])

    return figs


def plot_clusters(clusters: Dict[str, pd.Series],
                  linkages: Dict[str, np.ndarray],
                  cutoffs: Dict[str, float],
                  figsize: Tuple[float, float] = (14, 10)
                  ) -> Tuple[pd.Series, plt.Figure]:
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    if len(clusters.keys()) == 1:
        gs = fig.add_gridspec(nrows=1, ncols=3, wspace=0.0, hspace=0.0)
        axs = [fig.add_subplot(gs[0, :2])]
        titles = ['Monthly']
    elif len(clusters.keys()) == 2:
        gs = fig.add_gridspec(nrows=4, ncols=3, wspace=0.0, hspace=0.0)
        axs = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[1:, :2])]
        titles = [f"(A) Quarterly", f"(B) Monthly"]
    else:
        raise NotImplementedError(f"number clusters = {len(clusters.keys())}")

    # reverse
    linkages = dict(reversed(linkages.items()))
    agg_clusters = []
    for idx, (freq, linkage) in enumerate(linkages.items()):
        ax = axs[idx]
        spc.dendrogram(linkage, labels=clusters[freq].index.to_list(), orientation="right",
                       color_threshold=cutoffs[freq],
                       ax=ax)
        qis.set_title(ax, title=titles[idx])
        ax.axvline(cutoffs[freq], color='k')
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', which='major', labelsize=10)

        cluster_ = clusters[freq]
        agg_clusters.append(cluster_.apply(lambda x: f"{freq}-{x}"))
    agg_clusters = pd.concat(agg_clusters).sort_values()
    # index by inverse of agg clusters
    agg_clusters = agg_clusters.reindex(index=agg_clusters.index[::-1])

    # plot clusters
    ax = fig.add_subplot(gs[:, 2])
    qis.plot_df_table(df=agg_clusters.to_frame(name='Cluster ID'),
                      index_column_name='Instrument',
                      fontsize=10,
                      title='(C) Cluster IDs',
                      rows_edge_lines=qis.get_table_lines_for_group_data(agg_clusters),
                      ax=ax)

    return agg_clusters, fig


def run_rolling_covar_report(risk_factor_prices: pd.DataFrame,
                             prices: pd.DataFrame,
                             covar_estimator: FactorCovarEstimator,
                             time_period: qis.TimePeriod,
                             asset_returns_dict: Dict[str, pd.DataFrame],
                             assets: Union[List[str], pd.Index] = None,
                             rebalancing_freq: Optional[str] = None,
                             figsize: Tuple[float, float] = (14, 10),
                             is_plot: bool = True,
                             is_align_to_clusters_index: bool = True
                             ) -> Tuple[List[plt.Figure], Dict[str, pd.DataFrame]]:
    """
    Run rolling covariance analysis and generate reports.

    Uses FactorCovarEstimator.fit_rolling_factor_covars() to produce
    rolling CurrentFactorCovarData at each rebalancing date, then
    generates per-date diagnostic plots.

    Args:
        risk_factor_prices: Factor price panel. Index=dates, columns=factors.
        prices: Asset price panel (used only for asset universe inference if assets is None).
        covar_estimator: Configured FactorCovarEstimator instance.
        time_period: Period over which to generate the rebalancing schedule.
        asset_returns_dict: Dict[freq_str, DataFrame] of asset returns at different
            frequencies, computed via qis.compute_asset_returns_dict().
        assets: Asset universe to estimate. If None, inferred from prices.columns.
        rebalancing_freq: Override rebalancing frequency. If None, uses estimator default.
        figsize: Figure size for plots.
        is_plot: If True, generate matplotlib figures.
        is_align_to_clusters_index: If True, reorder covar matrices by cluster membership.

    Returns:
        Tuple of (list of figures, dict of per-date beta+diagnostics DataFrames).
    """
    if assets is None:
        assets = prices.columns

    # fit rolling factor covars using the new API
    rolling_covar_data = covar_estimator.fit_rolling_factor_covars(
        risk_factor_prices=risk_factor_prices,
        asset_returns_dict=asset_returns_dict,
        assets=assets,
        time_period=time_period,
        rebalancing_freq=rebalancing_freq,
    )

    figs = []
    dfs = {}

    for date, covar_data in rolling_covar_data.data.items():
        # get_snapshot returns betas + r2, alpha, insample_alpha, vols in one DataFrame
        df = covar_data.get_snapshot()

        if is_plot:
            date_figs = plot_current_covar_data(
                covar_data=covar_data,
                figsize=figsize,
                is_align_to_clusters_index=is_align_to_clusters_index
            )
            figs.extend(date_figs)
            plt.close('all')

        dfs[date.strftime('%d%b%Y')] = df

    return figs, dfs