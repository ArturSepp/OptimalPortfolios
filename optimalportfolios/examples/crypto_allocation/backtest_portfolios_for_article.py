# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from enum import Enum
import pybloqs as p
import qis as qis
from qis import TimePeriod, PerfParams, BenchmarkReturnsQuantileRegimeSpecs, PerfStat, PortfolioData

from optimalportfolios.examples.crypto_allocation.load_prices import Assets, load_prices, load_risk_free_rate
from optimalportfolios.reports.marginal_backtest import OptimisationParams, OptimisationType, backtest_marginal_optimal_portfolios
from optimalportfolios.reports.config import KWARGS_SUPTITLE, KWARGS_TITLE, KWARGS_FIG, KWARGS_TEXT

PERF_PARAMS = PerfParams(freq_vol='ME', freq_reg='ME', freq_drawdown='ME', rates_data=load_risk_free_rate())
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='QE')

LOCAL_PATH = "C://Users//artur//OneDrive//analytics//outputs//"
FIGURE_SAVE_PATH = "C://Users//artur//OneDrive//My Papers//Published Papers//CryptoAllocation. Zurich. Jan 2022//UpdatedFigures//"

SAVE_FIGS = False


OPTIMISATION_PARAMS = OptimisationParams(first_asset_target_weight=0.75,  # first asset is the benchmark
                                         rebalancing_freq='QE',  # when portfolio weigths are aupdate
                                         roll_window=60,  # number of monthly returns for solvers estimation of mv returns and mixure = 5 years
                                         returns_freq='ME',  # frequency of returns
                                         span=30,  # for window of ewma covariance for monthly return = 2.5 years
                                         carra=0.5,  # carra parameter
                                         n_mixures=3,
                                         rebalancing_costs=0.0050,  # rebalancing costs for portfolios
                                         weight_implementation_lag=1,
                                         marginal_asset_ew_weight=0.02)


PERF_COLUMNS = [PerfStat.TOTAL_RETURN,
                PerfStat.PA_RETURN,
                #PerfStat.AN_LOG_RETURN,
                PerfStat.VOL,
                PerfStat.SHARPE_LOG_EXCESS,
                #PerfStat.SHARPE_EXCESS,
                #PerfStat.SHARPE_LOG_AN,
                PerfStat.MAX_DD,
                PerfStat.SKEWNESS,
                PerfStat.ALPHA,
                PerfStat.BETA,
                PerfStat.R2]


FIG_KWARGS = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                  alpha_format='{0:+0.0%}',
                  beta_format='{:0.1f}',
                  x_date_freq='QE',
                  perf_params=PERF_PARAMS,
                  regime_params=REGIME_PARAMS,
                  perf_columns=PERF_COLUMNS,
                  alpha_an_factor=12  # to annualise alpha in the regression
                  )


def run_joint_backtest(prices_unconstrained: pd.DataFrame,
                       prices_all: pd.DataFrame,
                       marginal_asset: str,
                       time_period: TimePeriod,
                       perf_time_period: TimePeriod,
                       optimisation_type: OptimisationType = OptimisationType.MAX_DIV,
                       ) -> Tuple[PortfolioData, PortfolioData, PortfolioData, PortfolioData]:
    """
    report backtest for alts wo and with BTC and for balanced wo and with BTC
    """
    # run backtest
    alts_wo, alts_crypto = backtest_marginal_optimal_portfolios(prices=prices_unconstrained,
                                                                marginal_asset=marginal_asset,
                                                                optimisation_type=optimisation_type,
                                                                is_alternatives=True,
                                                                time_period=time_period,
                                                                perf_time_period=perf_time_period,
                                                                **OPTIMISATION_PARAMS.to_dict())

    bal_wo, bal_crypto = backtest_marginal_optimal_portfolios(prices=prices_all,
                                                              marginal_asset=marginal_asset,
                                                              optimisation_type=optimisation_type,
                                                              is_alternatives=False,
                                                              time_period=time_period,
                                                              perf_time_period=perf_time_period,
                                                              **OPTIMISATION_PARAMS.to_dict())
    return alts_wo, alts_crypto, bal_wo, bal_crypto


def produce_article_backtests(time_period: TimePeriod,
                              perf_time_period: TimePeriod,
                              perf_attrib_time_period_dict: Dict[str, Dict[str, TimePeriod]],
                              optimisation_types: List[OptimisationType] = (OptimisationType.ERC,
                                                                            OptimisationType.MAX_DIV),
                              benchmark_name: str = '100% Balanced',
                              is_updated: bool = True
                              ) -> None:
    """
    for optimization type:
        for marginal_asset = ['BTC', 'ETH']
            generate ra table (a+b)
        generate weight boxplot (c)
    plot weights time series
    plot performance attribution
    """
    crypto_assets = ['BTC', 'ETH']
    prices_alls = {}
    for crypto_asset in crypto_assets:
        prices_alls[crypto_asset] = load_prices(crypto_asset=crypto_asset, is_updated=is_updated).dropna()
    benchmark_price = prices_alls[crypto_assets[0]][Assets.BAL.value].rename(benchmark_name)

    alts_crypto_weight = {'BTC': [], 'ETH': []}
    bal_crypto_weight = {'BTC': [], 'ETH': []}
    perf_alts_crypto = {'BTC': [], 'ETH': []}  # dict by time periods, only take cryptos
    perf_bal_crypto = {'BTC': [], 'ETH': []}  # dict by time periods, only take cryptos
    alts_cryptos = {'BTC': {}, 'ETH': {}}
    bal_cryptos = {'BTC': {}, 'ETH': {}}
    alts_prices = {}
    balanced_prices = {}
    for optimisation_type in optimisation_types:
        # first add benchmark, then add performances of backtests
        alts_prices_ = {benchmark_name: benchmark_price}
        balanced_prices_ = {benchmark_name: benchmark_price}
        for crypto_asset in crypto_assets:
            prices_unconstrained = prices_alls[crypto_asset].drop(Assets.BAL.value, axis=1)
            alts_wo, alts_crypto, bal_wo, bal_crypto = run_joint_backtest(prices_unconstrained=prices_unconstrained,
                                                                          prices_all=prices_alls[crypto_asset],
                                                                          marginal_asset=crypto_asset,
                                                                          time_period=time_period,
                                                                          perf_time_period=perf_time_period,
                                                                          optimisation_type=optimisation_type)
            alts_cryptos[crypto_asset][optimisation_type] = alts_crypto
            bal_cryptos[crypto_asset][optimisation_type] = bal_crypto
            if optimisation_type == OptimisationType.EW:
                marginal_asset_ew_weight = OPTIMISATION_PARAMS.to_dict()['marginal_asset_ew_weight']
                alts_prices_[f"100% Alts\nw/o crypto"] = alts_wo.nav
                alts_prices_[f"100% Alts\nwith {marginal_asset_ew_weight:0.0%} {crypto_asset}"] = alts_crypto.nav
                balanced_prices_[f"75%/25% Bal/Alts\nw/o crypto"] = bal_wo.nav
                balanced_prices_[f"75%/25% Bal/Alts\nwith {marginal_asset_ew_weight:0.0%} {crypto_asset}"] = bal_crypto.nav
            else:
                alts_prices_[f"100% Alts\nw/o crypto"] = alts_wo.nav
                alts_prices_[f"100% Alts\nwith {crypto_asset}"] = alts_crypto.nav
                balanced_prices_[f"75%/25% Bal/Alts\nw/o crypto"] = bal_wo.nav
                balanced_prices_[f"75%/25% Bal/Alts\nwith {crypto_asset}"] = bal_crypto.nav

            # get weights
            alts_crypto_weight_ = alts_crypto.get_input_weights()
            if not isinstance(alts_crypto_weight_, pd.DataFrame):
                alts_crypto_weight_ = alts_crypto.get_weights()
            alts_crypto_weight[crypto_asset].append(alts_crypto_weight_[crypto_asset].rename(optimisation_type.value))

            bal_crypto_weight_ = bal_crypto.get_input_weights()
            if not isinstance(bal_crypto_weight_, pd.DataFrame):
                bal_crypto_weight_ = bal_crypto.get_weights()
            bal_crypto_weight[crypto_asset].append(bal_crypto_weight_[crypto_asset].rename(optimisation_type.value))

            # perf attribution for crypto_asset
            perf_alts_crypto_, perf_bal_crypto_ = {}, {}
            for label, time_period_ in perf_attrib_time_period_dict[crypto_asset].items():
                perf_alts_crypto_[label] = pd.Series(alts_crypto.get_performance_attribution(time_period=time_period_)[crypto_asset],
                                                     index=[optimisation_type.value])
                perf_bal_crypto_[label] = pd.Series(bal_crypto.get_performance_attribution(time_period=time_period_)[crypto_asset],
                                                    index=[optimisation_type.value])
            perf_alts_crypto_ = pd.DataFrame.from_dict(perf_alts_crypto_, orient='index')
            perf_bal_crypto_ = pd.DataFrame.from_dict(perf_bal_crypto_, orient='index')
            perf_alts_crypto[crypto_asset].append(perf_alts_crypto_)
            perf_bal_crypto[crypto_asset].append(perf_bal_crypto_)
        alts_prices[optimisation_type.value] = pd.DataFrame.from_dict(alts_prices_, orient='columns')
        balanced_prices[optimisation_type.value] = pd.DataFrame.from_dict(balanced_prices_, orient='columns')

    # concat weights to df
    alts_crypto_weights, bal_crypto_weights = {}, {}
    perf_alts_cryptos, perf_bal_cryptos = {}, {}
    for crypto_asset in crypto_assets:
        alts_crypto_weights[crypto_asset] = pd.concat(alts_crypto_weight[crypto_asset], axis=1)
        bal_crypto_weights[crypto_asset] = pd.concat(bal_crypto_weight[crypto_asset], axis=1)
        perf_alts_cryptos[crypto_asset] = pd.concat(perf_alts_crypto[crypto_asset], axis=1).T
        perf_bal_cryptos[crypto_asset] = pd.concat(perf_bal_crypto[crypto_asset], axis=1).T

    # add delta-1 to perf
    for crypto_asset in crypto_assets:
        delta1_total_perf = {}
        for label, time_period_ in perf_attrib_time_period_dict[crypto_asset].items():
            crypto_price = prices_alls[crypto_asset][crypto_asset]
            delta1_total_perf[label] = qis.compute_total_return(prices=time_period_.locate(crypto_price))
        delta1_total_perf = pd.Series(delta1_total_perf).rename(crypto_asset).to_frame().T
        perf_alts_cryptos[crypto_asset] = pd.concat([delta1_total_perf, perf_alts_cryptos[crypto_asset]], axis=0)
        perf_bal_cryptos[crypto_asset] = pd.concat([delta1_total_perf, perf_bal_cryptos[crypto_asset]], axis=0)

    # start report
    vblocks = []

    # add ra tables with descriptive weights
    dfs_out = {}
    for optimisation_type in optimisation_types:
        blocks = []
        ra_tables, axs = plt.subplots(1, 2, figsize=(10, 3.6), tight_layout=True)
        ra_tables_weights, axs1 = plt.subplots(1, 2, figsize=(10, 4.75), tight_layout=True)
        ts_navs, ts_axs = plt.subplots(2, 2, figsize=(16, 8), tight_layout=True)
        ts_weights_alt, weights_axs_alt = plt.subplots(2, 1, figsize=(16, 20), tight_layout=True)
        ts_weights_bal, weights_axs_bal = plt.subplots(2, 1, figsize=(16, 20), tight_layout=True)

        price_dict = {'Alts': alts_prices[optimisation_type.value],
                      'Balanced': balanced_prices[optimisation_type.value] }
        # match names of nav columns
        alt_weight_btc = alts_crypto_weights['BTC'][optimisation_type.value].rename(f"100% Alts\nwith BTC")
        bal_weight_btc = bal_crypto_weights['BTC'][optimisation_type.value].rename(f"75%/25% Bal/Alts\nwith BTC")
        alt_weight_eth = alts_crypto_weights['ETH'][optimisation_type.value].rename(f"100% Alts\nwith ETH")
        bal_weight_eth = bal_crypto_weights['ETH'][optimisation_type.value].rename(f"75%/25% Bal/Alts\nwith ETH")

        weights_dict = {'Alts': pd.concat([alt_weight_btc, alt_weight_eth], axis=1),
                        'Balanced': pd.concat([bal_weight_btc, bal_weight_eth], axis=1)}
        pretitles = ['(A)', '(B)']
        for idx, (key, df) in enumerate(price_dict.items()):
            prices = perf_time_period.locate(df)
            df_desc_weights = qis.compute_df_desc_data(df=weights_dict[key],
                                                       funcs={#'Avg crypto weight': np.nanmean,
                                                           'Min crypto weight': np.nanmin,
                                                           'Median crypto weight': np.nanmedian,
                                                           'Max crypto meight': np.nanmax,
                                                           'Last crypto meight': qis.last_row},
                                                       axis=0)
            # to str
            df_desc_weights = qis.df_to_str(df_desc_weights, var_format='{:.2%}')
            # names are in index
            ra_perf_table = qis.get_ra_perf_benchmark_columns(prices=prices,
                                                              benchmark=benchmark_name,
                                                              **FIG_KWARGS)
            dfs_out[f"ratable_{optimisation_type} {pretitles[idx]} {key}".replace("/", "")] = ra_perf_table.T
            special_rows_colors = [(1, 'aliceblue')]
            qis.plot_df_table(df=ra_perf_table.T,
                              title=f"{pretitles[idx]} {key}",
                              special_rows_colors=special_rows_colors,
                              ax=axs[idx],
                              **qis.update_kwargs(FIG_KWARGS, dict(first_column_width=5.5,
                                                                   first_row_height=0.375,
                                                                   rotation_for_columns_headers=90,
                                                                   fontsize=10,
                                                                   # heatmap_rows=[3],
                                                                   cmap='Greys')))

            # add column of weights
            df_table = pd.concat([ra_perf_table, df_desc_weights.T], axis=1).fillna(' ')
            df_table = df_table.loc[ra_perf_table.index, :].T  # align by original index and transpose to columns
            # highlight weights
            special_rows_colors = special_rows_colors + [(len(df_table.index) - n, 'lightblue') for n in range(len(df_desc_weights.index))]
            qis.plot_df_table(df=df_table,
                              title=f"{pretitles[idx]} {key}",
                              special_rows_colors=special_rows_colors,
                              ax=axs1[idx],
                              **qis.update_kwargs(FIG_KWARGS, dict(first_column_width=5.5,
                                                                   first_row_height=0.375,
                                                                   rotation_for_columns_headers=90,
                                                                   fontsize=10,
                                                                   heatmap_rows=[3],
                                                                   cmap='Greys')))

            prices.columns = [x.replace('\n', ' ') for x in prices.columns]
            perf_stats_labels = [PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_LOG_EXCESS]
            qis.plot_prices_with_dd(prices=prices.drop([benchmark_name], axis=1),
                                    pivot_prices=prices[benchmark_name],
                                    title=f"{pretitles[idx]} {key} cumulative performance",
                                    perf_stats_labels=perf_stats_labels,
                                    axs=ts_axs[:, idx],
                                    **qis.update_kwargs(FIG_KWARGS, dict(fontsize=10)))

        if SAVE_FIGS:
            qis.save_fig(ts_navs, file_name=f"ts_navs_{optimisation_type}", local_path=FIGURE_SAVE_PATH)
            qis.save_fig(ra_tables_weights, file_name=f"ra_tables_weights_{optimisation_type}", local_path=FIGURE_SAVE_PATH)

        # weights
        kwargs_w = qis.update_kwargs(FIG_KWARGS, dict(ncol=3, bbox_to_anchor=(0.5, 1.10),
                                                      framealpha=0.8,
                                                      pad=60,
                                                      freq='ME'))

        for idx, crypto_asset in enumerate(crypto_assets):
            ts_weights_by_crypto, weights_axs_by_crypto = plt.subplots(2, 1, figsize=(16, 20), tight_layout=True)

            alts_cryptos[crypto_asset][optimisation_type].plot_weights(title=f"{optimisation_type} with {crypto_asset} for Alts",
                                                                       ax=weights_axs_alt[idx],
                                                                       **kwargs_w)
            alts_cryptos[crypto_asset][optimisation_type].plot_weights(title=f"(A) {optimisation_type} with {crypto_asset} for Alts",
                                                                       ax=weights_axs_by_crypto[0],
                                                                       **kwargs_w)
            columns0 = bal_cryptos[crypto_asset][optimisation_type].prices.columns
            columns = list(columns0[1:]) + [columns0[0]]
            bal_cryptos[crypto_asset][optimisation_type].plot_weights(title=f"{optimisation_type} with {crypto_asset} for Balanced",
                                                                      columns=columns,
                                                                      ax=weights_axs_bal[idx],
                                                                      **kwargs_w)
            bal_cryptos[crypto_asset][optimisation_type].plot_weights(title=f"(B) {optimisation_type} with {crypto_asset} for Balanced",
                                                                      columns=columns,
                                                                      ax=weights_axs_by_crypto[1],
                                                                      **kwargs_w)

            if SAVE_FIGS:
                qis.save_fig(ts_weights_by_crypto, file_name=f"ts_weights_by_crypto {crypto_asset}_{optimisation_type}", local_path=FIGURE_SAVE_PATH)

        blocks.append(p.Paragraph(f"{optimisation_type}", **KWARGS_SUPTITLE))
        blocks.append(p.Block([
            p.Paragraph(f"{optimisation_type} Risk-adjusted Performance Table with Weights", **KWARGS_TITLE),
            p.Block(ra_tables_weights, **KWARGS_FIG)],
            **KWARGS_TEXT))
        blocks.append(p.Block([
            p.Paragraph(f"{optimisation_type} Time Series", **KWARGS_TITLE),
            p.Block(ts_navs, **KWARGS_FIG)],
            **KWARGS_TEXT))

        vblocks.append(p.VStack(blocks, styles={"page-break-after": "always"}))

        w_block = p.Block([p.Paragraph(f"{optimisation_type} Weights for Alts", **KWARGS_TITLE),
                           p.Block(ts_weights_alt, **KWARGS_FIG)],
                          **KWARGS_TEXT)
        vblocks.append(p.VStack([w_block], styles={"page-break-after": "always"}))

        w_block = p.Block([p.Paragraph(f"{optimisation_type} Weights for Balanced", **KWARGS_TITLE),
                           p.Block(ts_weights_bal, **KWARGS_FIG)],
                          **KWARGS_TEXT)
        vblocks.append(p.VStack([w_block], styles={"page-break-after": "always"}))

        if SAVE_FIGS:
            qis.save_fig(ts_weights_alt, file_name=f"ts_weights_alt {optimisation_type}", local_path=FIGURE_SAVE_PATH)
            qis.save_fig(ts_weights_bal, file_name=f"ts_weights_bal {optimisation_type}", local_path=FIGURE_SAVE_PATH)

    # global weight report
    alt_weight_btc, bal_weight_btc, alt_weight_eth, bal_weight_eth = [], [], [], []
    for optimisation_type in optimisation_types:
        # match names of nav columns
        alt_weight_btc.append(alts_crypto_weights['BTC'][optimisation_type.value])
        bal_weight_btc.append(bal_crypto_weights['BTC'][optimisation_type.value])
        alt_weight_eth.append(alts_crypto_weights['ETH'][optimisation_type.value])
        bal_weight_eth.append(bal_crypto_weights['ETH'][optimisation_type.value])

    weights_dict = {f"(A) 100% Alts with BTC": pd.concat(alt_weight_btc, axis=1),
                    f"(C) 75%/25% Bal/Alts with BTC": pd.concat(bal_weight_btc, axis=1),
                    f"(B) 100% Alts with ETH": pd.concat(alt_weight_eth, axis=1),
                    f"(D) 75%/25% Bal/Alts with ETH": pd.concat(bal_weight_eth, axis=1)}

    with sns.axes_style("darkgrid"):
        fig_weights, axs = plt.subplots(2, 2, figsize=(14, 4))
        for idx, (key, weights_df) in enumerate(weights_dict.items()):
            weights_df = qis.compute_df_desc_data(df=weights_df,
                                                  funcs={  # 'Avg crypto weight': np.nanmean,
                                                      'Min crypto weight': np.nanmin,
                                                      'Median crypto weight': np.nanmedian,
                                                      'Max crypto meight': np.nanmax,
                                                      'Last crypto meight': qis.last_row},
                                                  axis=0)
            weights_df['Median'] = np.median(weights_df, axis=1)
            qis.plot_df_table(df=qis.df_to_str(weights_df, var_format='{:.2%}'),
                              title=f"{key}",
                              ax=axs[idx%2][idx//2],
                              **qis.update_kwargs(FIG_KWARGS, dict(first_column_width=5.5,
                                                                   #first_row_height=0.375,
                                                                   #rotation_for_columns_headers=90,
                                                                   fontsize=10,
                                                                   #heatmap_rows=[3],
                                                                   cmap='Greys'))
                              )
            dfs_out[f"weight_{key}".replace("/", "")] = weights_df

    weights_dict = {f"(A) 100% Alts with BTC": pd.concat(alt_weight_btc, axis=1),
                    f"(C) 75%/25% Bal/Alts with BTC": pd.concat(bal_weight_btc, axis=1),
                    f"(B) 100% Alts with ETH": pd.concat(alt_weight_eth, axis=1),
                    f"(D) 75%/25% Bal/Alts with ETH": pd.concat(bal_weight_eth, axis=1)}

    # contercanate time series
    all_alts = pd.concat([weights_dict[f"(A) 100% Alts with BTC"], weights_dict[f"(B) 100% Alts with ETH"]], axis=0)
    all_balanced = pd.concat([weights_dict[f"(C) 75%/25% Bal/Alts with BTC"], weights_dict[f"(D) 75%/25% Bal/Alts with ETH"]], axis=0)
    alls = {'Alts': all_alts, 'Bal': all_balanced}
    for key, df in alls.items():
        all_alts_weights = qis.compute_df_desc_data(df=df,
                                                    funcs={  # 'Avg crypto weight': np.nanmean,
                                                        'Min crypto weight': np.nanmin,
                                                        'Median crypto weight': np.nanmedian,
                                                        'Max crypto meight': np.nanmax,
                                                        'Last crypto meight': qis.last_row},
                                                    axis=0)
        all_alts_weights['Median'] = np.median(all_alts_weights, axis=1)

    w_block = p.Block([p.Paragraph(f"Weights Table", **KWARGS_TITLE),
                       p.Block(fig_weights, **KWARGS_FIG)],
                      **KWARGS_TEXT)
    vblocks.append(p.VStack([w_block], styles={"page-break-after": "always"}))
    if SAVE_FIGS:
        qis.save_fig(fig_weights, file_name=f"combined_weights_table", local_path=FIGURE_SAVE_PATH)

    # add weights boxplot
    with sns.axes_style("darkgrid"):
        fig_box, axs = plt.subplots(1, 2, figsize=(8, 6))
        for idx, crypto_asset in enumerate(crypto_assets):
            dfs = {'Alts': alts_crypto_weights[crypto_asset], 'Balanced': bal_crypto_weights[crypto_asset]}
            qis.df_dict_boxplot_by_columns(dfs=dfs,
                                           hue_var_name='instruments',
                                           y_var_name='weights',
                                           ylabel='weights',
                                           legend_loc='upper center',
                                           showmedians=True,
                                           ncol=2,
                                           ax=axs[idx])
        b_fig_weights = p.Block([p.Paragraph(
            f"Weights", **KWARGS_TITLE),
            p.Block(fig_box, **KWARGS_FIG)],
            **KWARGS_TEXT)
        vblocks.append(b_fig_weights)

    # add weights time series
    kwargs = qis.update_kwargs(FIG_KWARGS, dict(legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                                var_format='{:,.2%}',
                                                y_limits=(0.0, None),
                                                framealpha=0.9))
    with sns.axes_style('darkgrid'):
        fig_weights, axs = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
        for idx, crypto_asset in enumerate(crypto_assets):
            qis.plot_time_series(df=alts_crypto_weights[crypto_asset],
                                 title=f"{crypto_asset} for Alternatives portfolio",
                                 ax=axs[idx][0],
                                 **kwargs)
            qis.plot_time_series(df=bal_crypto_weights[crypto_asset],
                                 title=f"{crypto_asset} for Balanced portfolio",
                                 ax=axs[idx][1],
                                 **kwargs)
        b_fig_weights = p.Block([p.Paragraph(
            f"Allocation weights", **KWARGS_TITLE),
            p.Block(fig_weights, **KWARGS_FIG)],
            **KWARGS_TEXT)
        vblocks.append(b_fig_weights)

    if SAVE_FIGS:
        qis.save_fig(fig_weights, file_name=f"fig_weights", local_path=FIGURE_SAVE_PATH)

    # add performance attributions
    kwargs1 = qis.update_kwargs(kwargs, dict(fontsize=9,
                                             transpose=False,
                                             special_rows_colors=[(1, 'lightblue')]))

    perf_dict = {f"(A) 100% Alts with BTC": perf_alts_cryptos['BTC'],
                 f"(C) 75%/25% Bal/Alts with BTC": perf_bal_cryptos['BTC'],
                 f"(B) 100% Alts with ETH": perf_alts_cryptos['ETH'],
                 f"(D) 75%/25% Bal/Alts with ETH": perf_bal_cryptos['ETH']}
    fig_periods, axs = plt.subplots(2, 2, figsize=(8, 3), constrained_layout=True)
    for idx, (key, df) in enumerate(perf_dict.items()):
        qis.plot_df_table(df=qis.df_to_str(df, var_format='{:.2%}'),
                          title=f"{key}",
                          ax=axs[idx%2][idx//2],
                          **kwargs1)
        dfs_out[f"attrib_{key}".replace("/", "")] = df

    b_fig_periods = p.Block([p.Paragraph(
        f"Performance attribution", **KWARGS_TITLE),
        p.Block(fig_periods, **KWARGS_FIG)],
        **KWARGS_TEXT)
    vblocks.append(b_fig_periods)
    if SAVE_FIGS:
        qis.save_fig(fig_periods, file_name=f"perf_atrib_all", local_path=FIGURE_SAVE_PATH)

    b_report = p.VStack(vblocks)
    filename = f"{LOCAL_PATH}Crypto_portfolios_backtests_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
    b_report.save(filename)
    print(f"saved article report to {filename}")
    qis.save_df_to_excel(dfs_out, file_name=filename)
    print(f"saved article tables to {filename}.xls")
    plt.close('all')


class UnitTests(Enum):
    BACKTEST_ARTICLE_FIGURES = 1


def run_unit_test(unit_test: UnitTests):

    optimisation_types = [OptimisationType.ERC,
                          OptimisationType.MAX_DIV,
                          OptimisationType.MAX_SHARPE,
                          OptimisationType.MIXTURE]

    # optimisation_types = [OptimisationType.EW]

    end_date = '16Aug2024'
    time_period = TimePeriod('19Jul2010', end_date)  # for weight calculations
    # perf_time_period = TimePeriod('31Mar2016', end_date)  # for reporting
    perf_time_period = TimePeriod('31Mar2016', end_date)  # for reporting

    if unit_test == UnitTests.BACKTEST_ARTICLE_FIGURES:
        time_period_dict_btc = {'16Q2-23Q2': TimePeriod(start='31Mar2016', end=end_date),
                                '21Q1-23Q1': TimePeriod(start='31Dec2020', end='31Mar2023')}
        time_period_dict_eth = {'16Q2-23Q2': TimePeriod(start='31Mar2016', end=end_date),
                                '21Q2-23Q1': TimePeriod(start='31Mar2021', end='31Mar2023')}
        perf_attrib_time_period_dict = {'BTC': time_period_dict_btc, 'ETH': time_period_dict_eth}

        produce_article_backtests(time_period=time_period,
                                  perf_time_period=perf_time_period,
                                  optimisation_types=optimisation_types,
                                  perf_attrib_time_period_dict=perf_attrib_time_period_dict,
                                  is_updated=True)


if __name__ == '__main__':

    unit_test = UnitTests.BACKTEST_ARTICLE_FIGURES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
