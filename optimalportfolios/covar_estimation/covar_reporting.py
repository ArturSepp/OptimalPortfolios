"""
report clustering of covar matrix
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as spc
import qis as qis
from matplotlib.colors import ListedColormap
from typing import List, Dict, Tuple

from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator


def run_rolling_covar_report(risk_factor_prices: pd.DataFrame,
                             prices: pd.DataFrame,
                             covar_estimator: CovarEstimator,
                             time_period: qis.TimePeriod,
                             factors_beta_loading_signs: pd.DataFrame = None,
                             figsize: Tuple[float, float] = (14, 10),
                             is_plot: bool = True,
                             is_align_to_clusters_index: bool = True
                             ) -> Tuple[List[plt.Figure], Dict[str, pd.DataFrame]]:
    """

    """
    # 1. estimate rolling covar for taa and betas
    # we need betas time series to estiate alphas
    rolling_covar_data = covar_estimator.fit_rolling_covars(risk_factor_prices=risk_factor_prices,
                                                            prices=prices,
                                                            time_period=time_period,
                                                            factors_beta_loading_signs=factors_beta_loading_signs)

    figs = []
    dfs = {}
    for date in rolling_covar_data.y_covars.keys():

        betas = rolling_covar_data.asset_last_betas_t[date].T
        betas = betas.where(np.abs(betas) > 1e-4, other=np.nan)

        # r2-s
        r2 = rolling_covar_data.r2_pd.loc[date, :].rename('R^2')
        total_var = np.sqrt(rolling_covar_data.total_vars_pd.loc[date, :]).rename('TotalVol')
        residual_var = np.sqrt(rolling_covar_data.residual_vars_pd.loc[date, :]).rename('ResidualVol')
        df = pd.concat([r2, total_var, residual_var], axis=1)
        dfs[date.strftime('%d%b%Y')] = pd.concat([betas, df], axis=1)

        if is_plot:
            # clusters
            clusters = {}
            linkages = {}
            cutoffs = {}
            for freq, cluster_data in rolling_covar_data.cluster_data.items():
                clusters[freq] = cluster_data.clusters[date]
                linkages[freq] = cluster_data.linkages[date]
                cutoffs[freq] = cluster_data.cutoffs[date]
            agg_clusters, fig_clusters = plot_monthly_quarterly_clusters(clusters=clusters, linkages=linkages, cutoffs=cutoffs,
                                                                figsize=figsize)

            # x covar
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            qis.set_suptitle(fig, title=f"{date.strftime('%d%b%Y')}: Factor correlations / vols")
            figs.append(fig)
            qis.plot_corr_matrix_from_covar(covar=rolling_covar_data.x_covars[date], ax=ax)

            # y covar
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            qis.set_suptitle(fig, title=f"{date.strftime('%d%b%Y')}: Asset correlations / vols")
            figs.append(fig)
            covar = rolling_covar_data.y_covars[date]
            if is_align_to_clusters_index:
                covar = covar.loc[agg_clusters.index, agg_clusters.index]

            qis.plot_corr_matrix_from_covar(covar=covar,
                                            corr_format='{:.1f}',ax=ax)

            # add fig_clusters
            qis.set_suptitle(fig_clusters, title=f"{date.strftime('%d%b%Y')}: Clusters")
            figs.append(fig_clusters)

            # betas and r^2
            fig = plt.figure(figsize=figsize, constrained_layout=True)
            gs = fig.add_gridspec(nrows=1, ncols=4, wspace=0.0, hspace=0.0)
            axs = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])]
            #fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            qis.set_suptitle(fig, title=f"{date.strftime('%d%b%Y')}: Universe betas")
            figs.append(fig)

            hline_rows = qis.get_table_lines_for_group_data(agg_clusters)
            if is_align_to_clusters_index:
                betas = betas.loc[agg_clusters.index, :]
            qis.plot_heatmap(df=betas,
                             var_format='{:.2f}',
                             hline_rows=hline_rows,
                             ax=axs[0])


            # Create custom colormap with different colors
            colors = ['white']  # One color per column
            custom_cmap = ListedColormap(colors)
            #qis.plot_df_table(df=df, var_format='{:.1%}', ax=axs[1])
            if is_align_to_clusters_index:
                df = df.loc[agg_clusters.index, :]
            qis.plot_heatmap(df=df,
                             cmap=custom_cmap,
                             var_format='{:.1%}',
                             hline_rows=hline_rows,
                             ax=axs[1])

            plt.close('all')
    return figs, dfs


def plot_monthly_quarterly_clusters(clusters: Dict[str, pd.Series],
                                    linkages: Dict[str, np.ndarray],
                                    cutoffs: Dict[str, float],
                                    figsize: Tuple[float, float] = (14, 10)
                                    ) -> Tuple[pd.Series, plt.Figure]:
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=4, ncols=3, wspace=0.0, hspace=0.0)
    axs = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[1:, :2])]
    titles = [f"(A) Quarterly", f"(B) Monthly"]
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

