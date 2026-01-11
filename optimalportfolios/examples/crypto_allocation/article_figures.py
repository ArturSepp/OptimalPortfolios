# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from enum import Enum

# qis
import qis
from qis import TimePeriod, PerfParams, BenchmarkReturnsQuantilesRegime, PerfStat

import optimalportfolios.utils.gaussian_mixture as gm
from optimalportfolios.examples.crypto_allocation.load_prices import Assets, load_prices, load_risk_free_rate

PERF_PARAMS = PerfParams(freq_vol='ME', freq_reg='ME', freq_drawdown='ME', rates_data=load_risk_free_rate())
REGIME_CLASSIFIER = BenchmarkReturnsQuantilesRegime(freq='QE')

FIGSIZE = (14, 6)

PERF_COLUMNS0 = [#PerfStat.END_DATE,
    PerfStat.TOTAL_RETURN,
    PerfStat.PA_RETURN,
    PerfStat.AN_LOG_RETURN,
    PerfStat.VOL,
    PerfStat.SHARPE_LOG_EXCESS,
    PerfStat.MAX_DD,
    PerfStat.SKEWNESS,
    PerfStat.ALPHA,
    PerfStat.BETA,
    PerfStat.R2]


PERF_COLUMNS = [#PerfStat.END_DATE,
                PerfStat.TOTAL_RETURN,
                PerfStat.PA_RETURN,
                PerfStat.VOL,
                PerfStat.SHARPE_LOG_EXCESS,
                PerfStat.MAX_DD,
                PerfStat.SKEWNESS,
                PerfStat.ALPHA,
                PerfStat.BETA,
                PerfStat.R2]

PERF_COLUMNS_LONG = [PerfStat.START_DATE,
                     PerfStat.END_DATE,
                     PerfStat.TOTAL_RETURN,
                     PerfStat.PA_RETURN,
                     PerfStat.AN_LOG_RETURN,
                     PerfStat.VOL,
                     PerfStat.SHARPE_RF0,
                     PerfStat.SHARPE_LOG_AN,
                     PerfStat.SHARPE_LOG_EXCESS,
                     PerfStat.MAX_DD,
                     PerfStat.MAX_DD_VOL,
                     PerfStat.SKEWNESS,
                     PerfStat.KURTOSIS,
                     PerfStat.ALPHA,
                     PerfStat.BETA,
                     PerfStat.R2]


def plot_performance_table(prices: pd.DataFrame,
                           benchmark: str,
                           time_period_dict: Dict[str, TimePeriod],
                           **kwargs
                           ) -> Tuple[plt.Figure, Dict]:
    """
    plot performances over different periods
    """
    kwargs = qis.update_kwargs(kwargs, dict(first_column_width=2.5,
                                            #first_row_height=0.2,
                                            rotation_for_columns_headers=0,
                                            #heatmap_columns=[4],
                                            fontsize=9,
                                            transpose=False,
                                            special_rows_colors=[(1, 'lightblue')],
                                            cmap='Greys'))
    fig, axs = plt.subplots(len(time_period_dict.keys()), 1, figsize=(12, 3.0), constrained_layout=True)
    dfs_out = {}
    for idx, (key, time_period) in enumerate(time_period_dict.items()):
        qis.plot_ra_perf_table_benchmark(prices=time_period.locate(prices),
                                         benchmark=benchmark,
                                         perf_params=PERF_PARAMS,
                                         perf_columns=PERF_COLUMNS0,
                                         title=f"{key}",
                                         is_fig_out=True,
                                         ax=axs[idx],
                                         **kwargs)
        dfs_out[key] = qis.plot_ra_perf_table_benchmark(prices=time_period.locate(prices),
                                                        benchmark=benchmark,
                                                        perf_params=PERF_PARAMS,
                                                        perf_columns=PERF_COLUMNS0,
                                                        title=f"{key}",
                                                        is_fig_out=False,
                                                        ax=axs[idx],
                                                        **kwargs)
    return fig, dfs_out


def plot_performance_tables(benchmark: str,
                            time_period_dict: Dict[str, Tuple[TimePeriod, pd.DataFrame]],
                            special_rows_colorss: List[List] = None,
                            **kwargs
                            ) -> plt.Figure:
    """
    plot performances over different periods
    """
    kwargs = qis.update_kwargs(kwargs, dict(first_column_width=2.5,
                                            #first_row_height=0.2,
                                            rotation_for_columns_headers=0,
                                            #heatmap_columns=[4],
                                            fontsize=9,
                                            transpose=False,
                                            cmap='Greys'))

    fig, axs = plt.subplots(len(time_period_dict.keys()), 1, figsize=(11, 5), constrained_layout=True)
    for idx, (label, data) in enumerate(time_period_dict.items()):
        if special_rows_colorss is not None:
            special_rows_colors = special_rows_colorss[idx]
            kwargs = qis.update_kwargs(kwargs, dict(special_rows_colors=special_rows_colors))
        qis.plot_ra_perf_table_benchmark(prices=data[0].locate(df=data[1]),
                                         benchmark=benchmark,
                                         perf_params=PERF_PARAMS,
                                         perf_columns=PERF_COLUMNS,
                                         title=f"{label}",
                                         ax=axs[idx],
                                         **kwargs)
    return fig


def plot_annual_tables(price: pd.Series,
                       perf_params: PerfParams,
                       date_format: str = '%b%Y'
                       ) -> Tuple[plt.Figure, Dict]:

    kwargs = dict(fontsize=9, date_format=date_format, cmap='YlGn')
    dfs_out = {}
    with sns.axes_style("white"):
        fig, axs = plt.subplots(2, 1, figsize=FIGSIZE)
        qis.plot_ra_perf_annual_matrix(price=price,
                                       perf_column=PerfStat.SHARPE_LOG_AN,
                                       perf_params=perf_params,
                                       ax=axs[0],
                                       title='(A) Sharpe Ratio',
                                       is_fig_out=True,
                                       **kwargs)
        dfs_out['(A) Sharpe Ratio'] = qis.plot_ra_perf_annual_matrix(price=price,
                                                                     perf_column=PerfStat.SHARPE_LOG_AN,
                                                                     perf_params=perf_params,
                                                                     ax=axs[0],
                                                                     title='(A) Sharpe Ratio',
                                                                     is_fig_out=False,
                                                                     **kwargs)
        qis.plot_ra_perf_annual_matrix(price=price,
                                       perf_column=PerfStat.SKEWNESS,
                                       perf_params=perf_params,
                                       title='(B) Skewness of monthly returns',
                                       ax=axs[1],
                                       is_fig_out=True,
                                       **kwargs)

        dfs_out['(B) Skewness of monthly returns'] = qis.plot_ra_perf_annual_matrix(price=price,
                                                                                    perf_column=PerfStat.SKEWNESS,
                                                                                    perf_params=perf_params,
                                                                                    title='(B) Skewness of monthly returns',
                                                                                    ax=axs[1],
                                                                                    is_fig_out=False,
                                                                                    **kwargs)

    return fig, dfs_out


def plot_corr_tables(prices: pd.DataFrame,
                     time_period: List[TimePeriod]
                     ) -> Tuple[plt.Figure, Dict]:
    kwargs = dict(square=True, x_rotation=90, fontsize=8, cmap='PiYG')
    titles = ['(A)', '(B)', '(C)']
    dfs_out = {}
    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(1, len(time_period), figsize=(16, 4.5), constrained_layout=True)
        for time_period, ax, title in zip(time_period, axs, titles):
            prices1 = time_period.locate(prices)
            qis.plot_returns_corr_table(prices=prices1,
                                        freq='ME',
                                        ax=ax,
                                        is_fig_out=True,
                                        title=f"{title} {time_period.to_str()}",
                                        **kwargs)

            dfs_out[f"{title} {time_period.to_str()}"] = qis.plot_returns_corr_table(prices=prices1,
                                                                                     freq='ME',
                                                                                     ax=ax,
                                                                                     is_fig_out=False,
                                                                                     title=f"{title} {time_period.to_str()}",
                                                                                     **kwargs)
    return fig, dfs_out


def plot_mixures(prices: pd.DataFrame,
                 start_end_date_full: TimePeriod,
                 time_period: TimePeriod
                 ) -> Tuple[plt.Figure, plt.Figure, Dict]:
    rets = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq='ME')
    n_components = 3

    kwargs = dict(fontsize=12, digits_to_show=1, sharpe_digits=2)

    with sns.axes_style('white'):
        fig1, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
        params = gm.fit_gaussian_mixture(x=rets.to_numpy(), n_components=n_components, idx=1)
        params_btc_1 = params.get_params(idx=1)
        gm.plot_mixure2(x=rets.to_numpy(),
                        n_components=n_components,
                        columns=prices.columns,
                        title=f"(A) Returns and ellipsoids of Gaussian clusters for period {start_end_date_full.to_str()}",
                        ax=axs[0],
                        **kwargs)
        params = gm.fit_gaussian_mixture(x=time_period.locate(rets).to_numpy(),
                                         n_components=n_components,
                                         idx=1)
        params_btc_2 = params.get_params(idx=1)
        gm.plot_mixure2(x=time_period.locate(rets).to_numpy(),
                        n_components=n_components,
                        columns=prices.columns,
                        title=f"(B) Returns and ellipsoids of Gaussian clusters for period {time_period.to_str()}",
                        ax=axs[1],
                        **kwargs)

        fig2, axs = plt.subplots(1, 2, figsize=(15, 1.5), constrained_layout=True)
        params = [params_btc_1, params_btc_2]
        titles = [f"(C) Cluster parameters of Bitcoin for {start_end_date_full.to_str()}",
                  f"(D) Cluster parameters of Bitcoin for {time_period.to_str()}"]
        dfs_out = {}
        for idx, param in enumerate(params):
            df = qis.df_to_str(param, var_format='{:.0%}')
            qis.plot_df_table(df=df,
                              add_index_as_column=True,
                              index_column_name='Cluster',
                              ax=axs[idx],
                              # heatmap_columns=[2],
                              title=titles[idx],
                              **kwargs)
            dfs_out[titles[idx]] = df
        return fig1, fig2, dfs_out


class LocalTests(Enum):
    PERF_TABLES_CRYPTO = 1
    PERF_TABLES_ALL = 2
    ANNUAL_ROLLING_TABLES = 3
    CORR_TABLE = 4
    CORR_TIME_SERIES = 5
    SCATTER = 6
    PDF_PLOT = 7
    PLOT_MIXURE = 8
    PERFORMANCE_CHECK = 9


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    FIGURE_SAVE_PATH = "C://Users//artur//OneDrive//My Papers//Published Papers//CryptoAllocation. Zurich. Jan 2022//UpdatedFigures//"
    SAVE_FIGS = True

    end_date = '30Jun2023'

    if local_test == LocalTests.PERF_TABLES_CRYPTO:

        prices = load_prices(assets=[Assets.BAL, Assets.BTC, Assets.ETH])
        prices = prices.loc['19Jul2010':]  # since bitcoin inception
        prices.loc[:'06Aug2015', 'ETH'] = np.nan  # set eth backfill to nan

        time_period_dict = {f'(A) Since Inception-{end_date}': TimePeriod(start='19Jul2010', end=end_date),
                            '(B) 31Mar2016-31Dec2019': TimePeriod(start='31Mar2016', end='31Dec2019'),
                            f'(C) 31Dec2019-{end_date}': TimePeriod(start='31Dec2019', end=end_date)}

        time_period_dict = {f'(A) Since Inception-{end_date}': TimePeriod(start='19Jul2010', end=end_date),
                            f'(B) 31Mar2016-{end_date}': TimePeriod(start='31Mar2016', end=end_date),
                            f'(C) 31Dec2019-{end_date}': TimePeriod(start='31Dec2019', end=end_date)}

        fig, dfs_out = plot_performance_table(prices=prices,
                                              benchmark=Assets.BAL.value,
                                              time_period_dict=time_period_dict)
        if SAVE_FIGS:
            qis.save_fig(fig, file_name='performance_table_crypto', local_path=FIGURE_SAVE_PATH)
            qis.save_df_to_excel(dfs_out, file_name='performance_table_crypto', local_path=FIGURE_SAVE_PATH)

    elif local_test == LocalTests.PERF_TABLES_ALL:

        prices1 = load_prices(crypto_asset='BTC').dropna()
        prices2 = load_prices(crypto_asset=None).dropna()
        time_period_dict = {'(A) 19Jul2010-18Dec2017': (TimePeriod(start='19Jul2010', end='18Dec2017'), prices1),
                            '(B) 07Aug2015-18Dec2017': (TimePeriod(start='07Aug2015', end='18Dec2017'), prices2),
                            '(C) 18Dec2017-31Mar2031': (TimePeriod(start='18Dec2017', end=end_date), prices2)}
        time_period_dict = {'(A) 19Jul2010-31Dec2015': (TimePeriod(start='19Jul2010', end='31Dec2015'), prices1),
                            '(B) 31Dec2015-31Mar2031': (TimePeriod(start='31Dec2015', end='31Mar2023'), prices2)}
        special_rows_colorss = [[(1, 'lightblue'), (2, 'goldenrod')],
                               [(1, 'lightblue'), (2, 'goldenrod'), (3, 'goldenrod')],
                               [(1, 'lightblue'), (2, 'goldenrod'), (3, 'goldenrod')]]

        fig = plot_performance_tables(time_period_dict=time_period_dict,
                                      benchmark=Assets.BAL.value,
                                      special_rows_colorss=special_rows_colorss)
        if SAVE_FIGS:
            qis.save_fig(fig, file_name='performance_table', local_path=FIGURE_SAVE_PATH)

    elif local_test == LocalTests.ANNUAL_ROLLING_TABLES:
        price = load_prices(assets=[Assets.BTC], is_updated=True).dropna().iloc[:, 0]#.loc[:end_date]
        fig, dfs_out = plot_annual_tables(price=price, perf_params=PERF_PARAMS)
        if SAVE_FIGS:
            qis.save_fig(fig, file_name='rolling_annual_table', local_path=FIGURE_SAVE_PATH)
            qis.save_df_to_excel(dfs_out, file_name='rolling_annual_table', local_path=FIGURE_SAVE_PATH)

    elif local_test == LocalTests.CORR_TABLE:
        time_period = [TimePeriod('19Jul2010', '31Dec2015'),
                       TimePeriod('31Dec2015', '31Dec2019'),
                       TimePeriod('31Dec2019', end_date)]
        time_period = [TimePeriod('19Jul2010', '31Dec2017'),
                       TimePeriod('31Dec2017', end_date),
                       TimePeriod('19Jul2010', end_date)]
        time_period = [TimePeriod('19Jul2010', '31Dec2017'),
                       TimePeriod('31Dec2017', end_date)]
        time_period = [TimePeriod('19Jul2010', '31Mar2016'),
                       TimePeriod('31Mar2016', end_date),
                       TimePeriod('31Dec2019', end_date)]
        prices2 = load_prices(crypto_asset=None).dropna()
        fig, dfs_out = plot_corr_tables(prices=prices2, time_period=time_period)
        if SAVE_FIGS:
            qis.save_fig(fig, file_name='corr_table', local_path=FIGURE_SAVE_PATH)
            qis.save_df_to_excel(dfs_out, file_name='corr_table', local_path=FIGURE_SAVE_PATH)

    elif local_test == LocalTests.CORR_TIME_SERIES:

        is_crypto_bal = True
        if is_crypto_bal:
            prices = load_prices(assets=[Assets.BAL, Assets.BTC, Assets.ETH]).dropna()
        else:
            prices = load_prices().dropna()
            cols = [prices.columns[1]] + [prices.columns[0]] + list(prices.columns[2:])
            prices = prices[cols]

        time_period = TimePeriod('30Jun2016', end_date)
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, constrained_layout=True)
            qis.plot_returns_corr_matrix_time_series(prices=prices,
                                                     time_period=time_period,
                                                     corr_matrix_output=qis.CorrMatrixOutput.TOP_ROW,
                                                     init_type=qis.InitType.ZERO,
                                                     legend_stats=qis.LegendStats.AVG_LAST,
                                                     trend_line=qis.TrendLine.NONE,
                                                     ewm_lambda=1.0-2.0/(24.0+1.0),
                                                     title='EWMA Correlations of monthly returns',
                                                     freq='ME',
                                                     ax=ax,
                                                     **{'framealpha': 0.90})

    elif local_test == LocalTests.SCATTER:
        prices = load_prices().dropna()
        prices1 = prices[[Assets.BAL, Assets.BTC]].dropna()
        kwargs = dict(alpha_format='{0:+0.0%}',
                      beta_format='{:0.1f}',
                      is_vol_norm=False,
                      framealpha=0.9)
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
        qis.plot_returns_scatter(prices=prices1,
                                 benchmark=Assets.BAL.value,
                                 xlabel=f"{Assets.BAL.value} returns",
                                 ylabel=f"{Assets.BTC.value} returns",
                                 freq='ME',
                                 order=1,
                                 ci=95,
                                 ax=axs[0],
                                 **kwargs)
        qis.plot_returns_scatter(prices=prices,
                                 benchmark=Assets.BAL.value,
                                 xlabel=f"{Assets.BAL.value} returns",
                                 ylabel=f"Assets returns",
                                 freq='ME',
                                 order=1,
                                 ci=95,
                                 ax=axs[1],
                                 **kwargs)

    elif local_test == LocalTests.PDF_PLOT:
        prices = load_prices().dropna()
        prices = prices[Assets.BTC]
        time_period = TimePeriod('18Dec2017', end_date)

        rets = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq='ME')
        rets1 = time_period.locate(rets)

        with sns.axes_style('white'):
            fig, axs = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)
            gm.plot_mixure1(x=rets.to_numpy().reshape(-1, 1), ax=axs[0])
            gm.plot_mixure1(x=rets1.to_numpy().reshape(-1, 1), ax=axs[1])

    elif local_test == LocalTests.PLOT_MIXURE:
        prices = load_prices().dropna()
        start_end_date_full = TimePeriod('19Jul2010', end_date)
        time_period = TimePeriod('18Dec2017', end_date)
        start_end_date_full = TimePeriod('19Jul2010', '31Dec2017')
        time_period = TimePeriod('31Dec2017', end_date)
        fig1, fig2, dfs_out = plot_mixures(prices=prices, start_end_date_full=start_end_date_full, time_period=time_period)
        if SAVE_FIGS:
            qis.save_fig(fig1, file_name='clusters', local_path=FIGURE_SAVE_PATH)
            qis.save_fig(fig2, file_name='params', local_path=FIGURE_SAVE_PATH)
            qis.save_df_to_excel(dfs_out, file_name='clusters_params', local_path=FIGURE_SAVE_PATH)

    elif local_test == LocalTests.PERFORMANCE_CHECK:
        time_period_from_last = qis.TimePeriod('30Jun2023', '16Aug2024')
        prices = load_prices(crypto_asset=None, is_updated=True)
        prices1 = time_period_from_last.locate(prices)
        qis.plot_ra_perf_table_benchmark(prices=prices1,
                                         benchmark=Assets.BAL.value,
                                         perf_params=PERF_PARAMS,
                                         perf_columns=PERF_COLUMNS_LONG)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ANNUAL_ROLLING_TABLES)
