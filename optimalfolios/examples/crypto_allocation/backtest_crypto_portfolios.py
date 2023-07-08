"""
backtesting report for BTC portfolios
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from enum import Enum
import pybloqs as p
import qis
from qis import TimePeriod, PerfParams, BenchmarkReturnsQuantileRegimeSpecs, PerfStat, PortfolioData

from optimalfolios.examples.crypto_allocation.load_prices import Assets, load_prices, load_risk_free_rate
from optimalfolios.reports.marginal_backtest import OptimisationParams, OptimisationType, backtest_marginal_optimal_portfolios
from optimalfolios.reports.config import KWARGS_SUPTITLE, KWARGS_TITLE, KWARGS_FIG, KWARGS_TEXT

PERF_PARAMS = PerfParams(freq_vol='M', freq_reg='M', freq_drawdown='M', rates_data=load_risk_free_rate())
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='Q')

LOCAL_PATH = "C://Users//artur//OneDrive//analytics//outputs//"
FIGURE_SAVE_PATH = "C://Users//artur//OneDrive//My Papers//Working Papers//CryptoAllocation. Zurich. Jan 2022//figs1//"
SAVE_FIGS = True

OPTIMISATION_PARAMS = OptimisationParams(first_asset_target_weight=0.75,  # first asset is the benchmark
                                         recalib_freq='Q',  # when portfolio weigths are aupdate
                                         roll_window=23,  # how many quarters are used for rolling estimation of mv returns and mixure = 5.5 years
                                         returns_freq='M',  # frequency of returns
                                         span=34,  # for ewma window for monthly return = 3 years
                                         is_log_returns=True,
                                         carra=0.5,  # carra parameter
                                         n_mixures=3,
                                         rebalancing_costs=0.0050)

PERF_COLUMNS = (PerfStat.TOTAL_RETURN,
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
                PerfStat.R2)

PERF_COLUMNS_LONG = (PerfStat.START_DATE,
                     PerfStat.END_DATE,
                     PerfStat.TOTAL_RETURN,
                     PerfStat.PA_RETURN,
                     PerfStat.AN_LOG_RETURN,
                     PerfStat.VOL,
                     PerfStat.SHARPE,
                     PerfStat.SHARPE_EXCESS,
                     PerfStat.SHARPE_LOG_AN,
                     PerfStat.MAX_DD,
                     PerfStat.MAX_DD_VOL,
                     PerfStat.SKEWNESS,
                     PerfStat.KURTOSIS,
                     PerfStat.ALPHA,
                     PerfStat.BETA,
                     PerfStat.R2)


FIG_KWARGS = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                  alpha_format='{0:+0.0%}',
                  beta_format='{:0.1f}',
                  alpha_an_factor=12,
                  x_date_freq='Q',
                  perf_params=PERF_PARAMS,
                  regime_params=REGIME_PARAMS,
                  perf_columns=PERF_COLUMNS)


def run_joint_backtest(prices_unconstrained: pd.DataFrame,
                       prices_all: pd.DataFrame,
                       marginal_asset: str,
                       time_period: TimePeriod,
                       optimisation_type: OptimisationType = OptimisationType.MAX_DIV
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
                                                                **OPTIMISATION_PARAMS.to_dict())

    bal_wo, bal_crypto = backtest_marginal_optimal_portfolios(prices=prices_all,
                                                              marginal_asset=marginal_asset,
                                                              optimisation_type=optimisation_type,
                                                              is_alternatives=False,
                                                              time_period=time_period,
                                                              **OPTIMISATION_PARAMS.to_dict())
    return alts_wo, alts_crypto, bal_wo, bal_crypto


def report_backtest_all_optimisation_types(time_period: TimePeriod,
                                           crypto_asset: str = 'BTC',
                                           optimisation_types: List[OptimisationType] = (OptimisationType.ERC,
                                                                                         OptimisationType.MAX_DIV),
                                           time_period_dict: Dict[str, TimePeriod] = None,
                                           ) -> None:
    """
    create pdf reports for optimisation types
    """
    prices_all = load_prices(crypto_asset=crypto_asset).dropna()
    benchmark_price = prices_all[Assets.BAL.value].rename('100 Bal')
    prices_unconstrained = prices_all.drop(Assets.BAL.value, axis=1)

    b_reports = []
    for optimisation_type in optimisation_types:
        report = run_backtest_pdf_report(prices_unconstrained=prices_unconstrained,
                                         prices_all=prices_all,
                                         benchmark_price=benchmark_price,
                                         marginal_asset=crypto_asset,
                                         time_period=time_period,
                                         time_period_dict=time_period_dict,
                                         optimisation_type=optimisation_type,
                                         **FIG_KWARGS)
        b_reports.append(p.Block(report))
    b_reports = p.VStack(b_reports, styles={"page-break-after": "always"})
    filename = f"{LOCAL_PATH}{crypto_asset}_backtests_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
    b_reports.save(filename)
    print(f"saved optimisation report to {filename}")


def create_performance_attrib_table(time_period_dict: Dict[str, TimePeriod],
                                    optimisation_types: List[OptimisationType] = (OptimisationType.ERC,
                                                                                  OptimisationType.MAX_DIV)
                                    ) -> None:
    """
    create performance attrib tables
    """
    prices_all = load_prices().dropna()
    benchmark_price = prices_all[Assets.BAL.value].rename('100 Bal')
    marginal_asset = str(Assets.BTC.value)
    prices_unconstrained = prices_all.drop(Assets.BAL.value, axis=1)
    btc_price = prices_all[str(Assets.BTC.value)]

    period_outs = {}
    btc_total_perf = {}
    for period, time_period in time_period_dict.items():
        outs = {}
        btc_total_perf[period] = qis.compute_total_return(prices=time_period.locate(btc_price))
        for optimisation_type in optimisation_types:
            alts_wo, alts_crypto, bal_wo, bal_crypto = run_joint_backtest(prices_unconstrained=prices_unconstrained,
                                                                          prices_all=prices_all,
                                                                          marginal_asset=marginal_asset,
                                                                          time_period=time_period,
                                                                          optimisation_type=optimisation_type)
            # perf attribution
            perf_alts_wo = alts_wo.get_performance_attribution(time_period=time_period).rename('W/O')
            perf_alts_crypto = alts_crypto.get_performance_attribution(time_period=time_period).rename('With')
            alts_perf = pd.concat([perf_alts_crypto, perf_alts_wo], axis=1)
            perf_bal_wo = bal_wo.get_performance_attribution(time_period=time_period).rename('W/O')
            perf_bal_crypto = bal_crypto.get_performance_attribution(time_period=time_period).rename('With')
            bal_perf = pd.concat([perf_bal_crypto, perf_bal_wo], axis=1)
            outs[optimisation_type.value] = pd.Series(dict(Alts=alts_perf.loc['BTC', 'With'],
                                                           Balanced=bal_perf.loc['BTC', 'With']))
        outs = pd.DataFrame.from_dict(outs, orient='index')
        period_outs[period] = outs
    print(f"btc_total_perf={btc_total_perf}")

    fig_periods, axs = plt.subplots(1, len(time_period_dict.keys()), figsize=(8, 2), constrained_layout=True)
    for idx, (period, df) in enumerate(period_outs.items()):
        qis.plot_df_table(df=qis.df_to_str(df, var_format='{:.2%}'),
                          title=period,
                          ax=axs[idx])


def plot_weights_timeseries(time_period: TimePeriod,
                            optimisation_types: List[OptimisationType] = (OptimisationType.ERC,
                                                                          OptimisationType.MAX_DIV),
                            crypto_asset: str = 'BTC'
                            ) -> None:
    """
    create performance attrib tables
    """
    prices_all = load_prices(crypto_asset=crypto_asset).dropna()
    benchmark_price = prices_all[Assets.BAL.value].rename('100 Bal')
    prices_unconstrained = prices_all.drop(Assets.BAL.value, axis=1)

    alts_crypto_weight = {}
    bal_crypto_weight = {}
    for optimisation_type in optimisation_types:
        alts_wo, alts_crypto, bal_wo, bal_crypto = run_joint_backtest(prices_unconstrained=prices_unconstrained,
                                                                      prices_all=prices_all,
                                                                      marginal_asset=crypto_asset,
                                                                      time_period=time_period,
                                                                      optimisation_type=optimisation_type)
        # get weights
        alts_crypto_weight[optimisation_type.value] = alts_crypto.get_weights()[crypto_asset]
        bal_crypto_weight[optimisation_type.value] = bal_crypto.get_weights()[crypto_asset]

    alts_crypto_weight = pd.DataFrame.from_dict(alts_crypto_weight, orient='columns')
    bal_crypto_weight = pd.DataFrame.from_dict(bal_crypto_weight, orient='columns')

    kwargs = qis.update_kwargs(FIG_KWARGS, dict(legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                                var_format='{:,.2%}',
                                                y_limits=(0.0, None)))
    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(2, 1, figsize=(10, 16), tight_layout=True)
        qis.plot_time_series(df=alts_crypto_weight,
                             title='alts_crypto_weight',
                             ax=axs[0],
                             **kwargs)
        qis.plot_time_series(df=bal_crypto_weight,
                             title='bal_crypto_weight',
                             ax=axs[1],
                             **kwargs)


def run_backtest_pdf_report(prices_unconstrained: pd.DataFrame,
                            prices_all: pd.DataFrame,
                            benchmark_price: pd.Series,
                            marginal_asset: str,
                            time_period: TimePeriod,
                            time_period_dict: Optional[Dict[str, TimePeriod]],
                            first_asset_target_weight: float = 0.75,  # first asset is the benchmark
                            optimisation_type: OptimisationType = OptimisationType.MAX_DIV,
                            figsize: Tuple[float, float] = (14, 6),
                            perf_params: PerfParams = PERF_PARAMS,
                            perf_columns: List[PerfStat] = qis.LN_BENCHMARK_TABLE_COLUMNS,
                            **kwargs
                            ) -> p.VStack:
    """
    report backtest and performance attribution to BTC
    """
    # run backtest
    alts_wo, alts_crypto, bal_wo, bal_crypto = run_joint_backtest(prices_unconstrained=prices_unconstrained,
                                                                  prices_all=prices_all,
                                                                  marginal_asset=marginal_asset,
                                                                  time_period=time_period,
                                                                  optimisation_type=optimisation_type)
    alts_navs = pd.concat([benchmark_price, alts_wo.nav, alts_crypto.nav], axis=1).dropna()
    bal_navs = pd.concat([benchmark_price, bal_wo.nav, bal_crypto.nav], axis=1).dropna()

    # start blocks
    blocks = []
    b_title = p.Block(title=f"Backtest using optimisation_type: {optimisation_type} for {marginal_asset}", **KWARGS_SUPTITLE)
    blocks.append(b_title)

    # ra tables
    fig_table_alts, ax = plt.subplots(1, 1, figsize=(figsize[0], 2), constrained_layout=True)
    qis.plot_ra_perf_table_benchmark(prices=time_period.locate(alts_navs),
                                     benchmark=str(benchmark_price.name),
                                     perf_params=perf_params,
                                     perf_columns=perf_columns,
                                     ax=ax,
                                     **qis.update_kwargs(kwargs, dict(is_log_returns=True)))
    b_fig_table_alts = p.Block([p.Paragraph(f"(A) 100% Alts Portfolio", **KWARGS_TITLE),
                                p.Block(fig_table_alts, **KWARGS_FIG)],
                               **KWARGS_TEXT)
    blocks.append(b_fig_table_alts)

    fig_table_bal, ax = plt.subplots(1, 1, figsize=(figsize[0], 2), constrained_layout=True)
    qis.plot_ra_perf_table_benchmark(prices=time_period.locate(bal_navs),
                                     benchmark=str(benchmark_price.name),
                                     perf_params=perf_params,
                                     perf_columns=perf_columns,
                                     ax=ax,
                                     **qis.update_kwargs(kwargs, dict(is_log_returns=True)))
    b_fig_table_bal = p.Block([p.Paragraph(f"(B) {first_asset_target_weight:0.0%}/{1.0-first_asset_target_weight:0.0%}"
                                           f" Balanced/Alts Portfolio", **KWARGS_TITLE),
                               p.Block(fig_table_bal, **KWARGS_FIG)],
                              **KWARGS_TEXT)
    blocks.append(b_fig_table_bal)

    # weights
    with sns.axes_style('darkgrid'):
        kwargs_w = qis.update_kwargs(kwargs, dict(ncol=2, bbox_to_anchor=(0.5, 1.20)))

        fig_weights_alt, ax = plt.subplots(1, 1, figsize=(figsize[0], 6), constrained_layout=True)
        alts_crypto.plot_weights(ax=ax, freq='M', **kwargs_w)
        b_fig_weights_alt = p.Block([p.Paragraph("", **KWARGS_TITLE),
                                     p.Paragraph(f"(A) Alternatives Portfolio Weights", **KWARGS_TITLE),
                                     p.Block(fig_weights_alt, **KWARGS_FIG)],
                                    **KWARGS_TEXT)
        blocks.append(b_fig_weights_alt)

        fig_weights_bal, ax = plt.subplots(1, 1, figsize=(figsize[0], 6), constrained_layout=True)
        # move bal to the end
        columns = list(bal_crypto.weights.columns[1:]) + [bal_crypto.weights.columns[0]]
        bal_crypto.plot_weights(columns=columns, freq='M', ax=ax, **kwargs_w)
        b_fig_weights_bal = p.Block([p.Paragraph("", **KWARGS_TITLE),
                                     p.Paragraph(f"(E) {first_asset_target_weight:0.0%}/{1.0 - first_asset_target_weight:0.0%} "
                                                 f"Balanced/Alts Portfolio weights", **KWARGS_TITLE),
                                     p.Block(fig_weights_bal, **KWARGS_FIG)],
                                    **KWARGS_TEXT)
        blocks.append(b_fig_weights_bal)

    # weights - 2
    with sns.axes_style('darkgrid'):
        kwargs_w = qis.update_kwargs(kwargs, dict(yvar_format='{:.2%}',
                                                  showmedians=True,
                                                  add_y_med_labels=True,
                                                  y_limits=(0.0, None)))

        fig_weights_alt, ax = plt.subplots(1, 1, figsize=(figsize[0], 6), constrained_layout=True)

        weights1 = alts_crypto.get_weights(freq=None)
        qis.df_boxplot_by_columns(df=weights1,
                                  hue_var_name='instruments',
                                  y_var_name='weights',
                                  ylabel='weights',
                                  ax=ax,
                                  **kwargs_w)
        b_fig_weights_alt = p.Block([p.Paragraph("", **KWARGS_TITLE),
                                     p.Paragraph(f"(A) 100% Alts Portfolio Weights", **KWARGS_TITLE),
                                     p.Block(fig_weights_alt, **KWARGS_FIG)],
                                    **KWARGS_TEXT)
        blocks.append(b_fig_weights_alt)

        fig_weights_bal, ax = plt.subplots(1, 1, figsize=(figsize[0], 6), constrained_layout=True)
        # move bal to the end
        columns = list(bal_crypto.weights.columns[1:]) + [bal_crypto.weights.columns[0]]
        weights1 = bal_crypto.get_weights(freq=None)[columns]
        qis.df_boxplot_by_columns(df=weights1,
                                  hue_var_name='instruments',
                                  y_var_name='weights',
                                  ylabel='weights',
                                  ax=ax,
                                  **kwargs_w)

        b_fig_weights_bal = p.Block([p.Paragraph("", **KWARGS_TITLE),
                                     p.Paragraph(
                                         f"(E) {first_asset_target_weight:0.0%}/{1.0 - first_asset_target_weight:0.0%} "
                                         f"Balanced/Alts Portfolio weights", **KWARGS_TITLE),
                                     p.Block(fig_weights_bal, **KWARGS_FIG)],
                                    **KWARGS_TEXT)
        blocks.append(b_fig_weights_bal)

    if time_period_dict is not None:
        fig_alts, ax = plt.subplots(1, 1, figsize=(6.5, 2), constrained_layout=True)
        qis.plot_ra_perf_by_dates(prices=alts_navs,
                                  time_period_dict=time_period_dict,
                                  perf_column=PerfStat.SHARPE_LOG_EXCESS,
                                  perf_params=PERF_PARAMS,
                                  heatmap_columns=[1, 2, 3, 4, 5],
                                  ax=ax,
                                  **kwargs)

        fig_bal, ax = plt.subplots(1, 1, figsize=(6.5, 2), constrained_layout=True)
        qis.plot_ra_perf_by_dates(prices=bal_navs,
                                  time_period_dict=time_period_dict,
                                  perf_column=PerfStat.SHARPE_LOG_EXCESS,
                                  perf_params=PERF_PARAMS,
                                  heatmap_columns=[1, 2, 3, 4, 5],
                                  ax=ax,
                                  **kwargs)
        b_fig_ra1 = p.Block([p.Paragraph(
            f"Alts" + u"\u002A", **KWARGS_TITLE),
                             p.Block(fig_alts, **KWARGS_FIG)],
                            **KWARGS_TEXT)
        b_fig_ra2 = p.Block([p.Paragraph(
            f"Bal" + u"\u002A",
            **KWARGS_TITLE),
                             p.Block(fig_bal, **KWARGS_FIG)],
                            **KWARGS_TEXT)

        b_fig_ra = p.Block([p.Paragraph(f"Performances for past years", **KWARGS_TITLE),
                            p.HStack([b_fig_ra1, b_fig_ra2], styles=KWARGS_TEXT)],
                **KWARGS_TEXT)
        blocks.append(b_fig_ra)
    report = p.VStack(blocks, cascade_cfg=False)
    return report


def produce_article_figures(time_period: TimePeriod,
                            time_period_dict: Dict[str, Dict[str, TimePeriod]],
                            optimisation_types: List[OptimisationType] = (OptimisationType.ERC,
                                                                          OptimisationType.MAX_DIV),
                            benchmark_name: str = '100% Balanced'
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
        prices_alls[crypto_asset] = load_prices(crypto_asset=crypto_asset).dropna()
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
                                                                          optimisation_type=optimisation_type)
            alts_cryptos[crypto_asset][optimisation_type] = alts_crypto
            bal_cryptos[crypto_asset][optimisation_type] = bal_crypto
            alts_prices_[f"100% Alts\nw/o crypto"] = alts_wo.nav
            alts_prices_[f"100% Alts\nwith {crypto_asset}"] = alts_crypto.nav
            balanced_prices_[f"75%/25% Bal/Alts\nw/o crypto"] = bal_wo.nav
            balanced_prices_[f"75%/25% Bal/Alts\nwith {crypto_asset}"] = bal_crypto.nav

            # get weights
            alts_crypto_weight[crypto_asset].append(alts_crypto.get_weights(freq=None)[crypto_asset].rename(optimisation_type.value))
            bal_crypto_weight[crypto_asset].append(bal_crypto.get_weights(freq=None)[crypto_asset].rename(optimisation_type.value))

            # perf attribution for crypto_asset
            perf_alts_crypto_, perf_bal_crypto_ = {}, {}
            for label, time_period_ in time_period_dict[crypto_asset].items():
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
        for label, time_period_ in time_period_dict[crypto_asset].items():
            crypto_price = prices_alls[crypto_asset][crypto_asset]
            delta1_total_perf[label] = qis.compute_total_return(prices=time_period_.locate(crypto_price))
        delta1_total_perf = pd.Series(delta1_total_perf).rename(crypto_asset).to_frame().T
        perf_alts_cryptos[crypto_asset] = pd.concat([delta1_total_perf, perf_alts_cryptos[crypto_asset]], axis=0)
        perf_bal_cryptos[crypto_asset] = pd.concat([delta1_total_perf, perf_bal_cryptos[crypto_asset]], axis=0)

    # start report
    vblocks = []

    # add ra tables with descriptive weights
    for optimisation_type in optimisation_types:
        blocks = []
        ra_tables, axs = plt.subplots(1, 2, figsize=(10, 3.6), tight_layout=True)
        ra_tables_weights, axs1 = plt.subplots(1, 2, figsize=(10, 4.75), tight_layout=True)
        ts_navs, ts_axs = plt.subplots(2, 2, figsize=(16, 8), tight_layout=True)
        ts_weights, weights_axs = plt.subplots(4, 1, figsize=(16, 20), tight_layout=True)

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
            prices = time_period.locate(df)
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
                                                              is_log_returns=True,
                                                              **FIG_KWARGS)

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
            qis.plot_prices_with_dd(prices=prices.drop([benchmark_name], axis=1),
                                    pivot_prices=prices[benchmark_name],
                                    performance_label=qis.PerformanceLabel.DETAILED,
                                    axs=ts_axs[:, idx],
                                    **qis.update_kwargs(FIG_KWARGS, dict(fontsize=10)))

        if SAVE_FIGS:
            qis.save_fig(ra_tables, file_name=f"ratable_{optimisation_type}", local_path=FIGURE_SAVE_PATH)

        # weights
        kwargs_w = qis.update_kwargs(FIG_KWARGS, dict(ncol=3, bbox_to_anchor=(0.5, 1.20), freq='M'))

        for idx, crypto_asset in enumerate(crypto_assets):
            alts_cryptos[crypto_asset][optimisation_type].plot_weights(title=f"{optimisation_type} with {crypto_asset} for Alts",
                                                                       ax=weights_axs[2*idx],
                                                                       **kwargs_w)
            columns0 = bal_cryptos[crypto_asset][optimisation_type].input_weights.columns
            columns = list(columns0[1:]) + [columns0[0]]
            bal_cryptos[crypto_asset][optimisation_type].plot_weights(title=f"{optimisation_type} with {crypto_asset} for Balanced",
                                                                      columns=columns,
                                                                      ax=weights_axs[2*idx+1],
                                                                      **kwargs_w)

        blocks.append(p.Paragraph(f"{optimisation_type}", **KWARGS_SUPTITLE))
        blocks.append(p.Block([
            p.Paragraph(f"{optimisation_type} Risk-adjusted Performance Table", **KWARGS_TITLE),
            p.Block(ra_tables, **KWARGS_FIG)],
            **KWARGS_TEXT))
        blocks.append(p.Block([
            p.Paragraph(f"{optimisation_type} Risk-adjusted Performance Table with Weights", **KWARGS_TITLE),
            p.Block(ra_tables_weights, **KWARGS_FIG)],
            **KWARGS_TEXT))
        blocks.append(p.Block([
            p.Paragraph(f"{optimisation_type} Time Series", **KWARGS_TITLE),
            p.Block(ts_navs, **KWARGS_FIG)],
            **KWARGS_TEXT))

        vblocks.append(p.VStack(blocks, styles={"page-break-after": "always"}))

        w_block = p.Block([p.Paragraph(f"{optimisation_type} Weights", **KWARGS_TITLE),
                           p.Block(ts_weights, **KWARGS_FIG)],
                          **KWARGS_TEXT)
        vblocks.append(p.VStack([w_block], styles={"page-break-after": "always"}))

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
        print(key)
        print(all_alts_weights)

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
                                                y_limits=(0.0, None)))
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
    vblocks.append(b_fig_weights)
    if SAVE_FIGS:
        qis.save_fig(fig_periods, file_name=f"perf_atrib_all", local_path=FIGURE_SAVE_PATH)

    b_report = p.VStack(vblocks)
    filename = f"{LOCAL_PATH}Crypto_portfolios_backtests_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
    b_report.save(filename)
    print(f"saved article report to {filename}")
    plt.close('all')


def backtest_constant_weight_portfolios(crypto_asset: str = 'BTC',
                                        rebalance_freq: str = 'Q',
                                        is_alternatives: bool = False,
                                        time_period: TimePeriod = None,
                                        time_period_dict: Dict[str, TimePeriod] = None
                                        ):
    """
    backtest using constant weights
    """
    crypto_weights = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    benchmark_str = Assets.BAL.value
    prices = load_prices(crypto_asset=crypto_asset).dropna()
    if not is_alternatives:
        prices = prices[[benchmark_str, crypto_asset]]

    if time_period is not None:
        prices = time_period.locate(prices)

    if is_alternatives:
        benchmark_name = f"EW Alts {1.0:0.0%}, {crypto_asset} {0.0:0.0%}"
    else:
        benchmark_name = f"60/40 {1.0:0.0%}, {crypto_asset} {0.0:0.0%}"

    navs = [prices[benchmark_str].rename(benchmark_name)]
    for crypto_weight in crypto_weights:
        if is_alternatives:
            ew_weight = (1.0 - crypto_weight) / (len(prices.columns) - 1)
            weights = ew_weight * np.ones(len(prices.columns))
            weights[0] = crypto_weight
            ticker = f"EW Alts {1.0 - crypto_weight:0.0%}, {crypto_asset} {crypto_weight:0.0%}"
            title = f"Alternatives Portfolio with {crypto_asset}"
        else:
            if np.isclose(crypto_weight, 0.0):  # skip zero weight
                continue
            weights = np.array([1.0 - crypto_weight, crypto_weight])
            ticker = f"60/40 {1.0 - crypto_weight:0.0%}, {crypto_asset} {crypto_weight:0.0%}"
            title = f"60/40 Portfolio Portfolio with {crypto_asset}"
            prices = prices[[benchmark_str, crypto_asset]]

        portfolio_wo = qis.backtest_model_portfolio(prices=prices,
                                                    weights=weights,
                                                    rebalance_freq=rebalance_freq,
                                                    is_rebalanced_at_first_date=True,
                                                    ticker=ticker,
                                                    is_output_portfolio_data=True)
        navs.append(portfolio_wo.nav)
    navs = pd.concat(navs, axis=1)

    figs = []
    kwargs = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                  alpha_format='{0:+0.0%}',
                  beta_format='{:0.1f}',
                  alpha_an_factor=12)
    # time series
    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8), tight_layout=True)
        figs.append(fig)
        qis.plot_prices(prices=navs,
                        title=f"Simulation of 60/40 portfolio with {crypto_asset} overlay",
                        performance_label=qis.PerformanceLabel.TOTAL_DETAILED,
                        perf_params=PERF_PARAMS,
                        ax=ax,
                        **kwargs)

        fig, axs = plt.subplots(2, 1, figsize=(10, 12), tight_layout=True)
        figs.append(fig)
        qis.plot_prices_with_dd(prices=navs,
                                performance_label=qis.PerformanceLabel.TOTAL_DETAILED,
                                perf_params=PERF_PARAMS,
                                axs=axs,
                                **kwargs)
    # ra table
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), tight_layout=True)
    figs.append(fig)
    qis.plot_ra_perf_table_benchmark(prices=navs,
                                     benchmark=benchmark_name,
                                     perf_params=PERF_PARAMS,
                                     perf_columns=PERF_COLUMNS,
                                     is_log_returns=True,
                                     title=title,
                                     ax=ax,
                                     **kwargs)
    if time_period_dict is not None:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 2), constrained_layout=True)
        figs.append(fig)
        qis.plot_ra_perf_by_dates(prices=navs,
                                  time_period_dict=time_period_dict,
                                  perf_column=PerfStat.SHARPE_LOG_EXCESS,
                                  perf_params=PERF_PARAMS,
                                  heatmap_columns=[1, 2],
                                  ax=ax,
                                  **kwargs)
    filename = f"{LOCAL_PATH}{crypto_asset}_constant_weight_backtests.pdf"
    qis.save_figs_to_pdf(figs, file_name=filename)


class UnitTests(Enum):
    ALL_OPTIMISATION_TYPES = 1
    PERFORMANCE_ATTRIB_TABLE = 2
    WEIGHTS_FIGURE = 3
    CONSTANT_WEIGHT_PORTFOLIOS = 4
    ARTICLE_FIGURES = 5


def run_unit_test(unit_test: UnitTests):

    optimisation_types = [OptimisationType.ERC,
                          OptimisationType.MAX_DIV,
                          OptimisationType.MAX_SHARPE,
                          OptimisationType.MIXTURE]

    # optimisation_types = [OptimisationType.ERC, OptimisationType.MAX_DIV]
    # optimisation_types = [OptimisationType.MAX_DIV]

    # time_period = TimePeriod('31Dec2015', '31Mar2023')
    time_period = TimePeriod('31Mar2016', '31Mar2023')

    if unit_test == UnitTests.ALL_OPTIMISATION_TYPES:

        time_period_dict = {'1y': TimePeriod(start='31Mar2022', end=time_period.end),
                            '2y': TimePeriod(start='31Mar2021', end=time_period.end),
                            '3y': TimePeriod(start='31Mar2020', end=time_period.end),
                            '5y': TimePeriod(start='31Mar2018', end=time_period.end),
                            '7y': TimePeriod(start='31Mar2016', end=time_period.end)}

        report_backtest_all_optimisation_types(crypto_asset='BTC',
                                               optimisation_types=optimisation_types,
                                               time_period=time_period,
                                               time_period_dict=time_period_dict)

    elif unit_test == UnitTests.PERFORMANCE_ATTRIB_TABLE:
        time_period_dict = {'2016Q1-now': TimePeriod(start='31Dec2015', end='31Mar2023'),
                            '2021Q1-now': TimePeriod(start='31Dec2020', end='31Mar2023')}
        create_performance_attrib_table(optimisation_types=optimisation_types,
                                        time_period_dict=time_period_dict)

    elif unit_test == UnitTests.WEIGHTS_FIGURE:
        plot_weights_timeseries(time_period=time_period,
                                optimisation_types=optimisation_types)

    elif unit_test == UnitTests.CONSTANT_WEIGHT_PORTFOLIOS:
        time_period_dict = {'2016Q1-now': TimePeriod(start='31Dec2015', end='31Mar2023'),
                            '2021Q1-now': TimePeriod(start='31Dec2020', end='31Mar2023')}
        time_period = TimePeriod('31Dec2015', '30May2023')
        backtest_constant_weight_portfolios(
                                        crypto_asset='ETH',
                                        rebalance_freq='Q',
                                        is_alternatives=False,
                                        time_period=time_period,
                                        time_period_dict=time_period_dict)

    elif unit_test == UnitTests.ARTICLE_FIGURES:
        time_period_dict_btc = {'16Q2-23Q1': TimePeriod(start='31Mar2016', end='31Mar2023'),
                                '21Q1-23Q1': TimePeriod(start='31Dec2020', end='31Mar2023')}
        time_period_dict_eth = {'16Q2-23Q1': TimePeriod(start='31Mar2016', end='31Mar2023'),
                                '21Q2-23Q1': TimePeriod(start='31Mar2021', end='31Mar2023')}
        time_period_dict = {'BTC': time_period_dict_btc, 'ETH': time_period_dict_eth}
        produce_article_figures(time_period=time_period,
                                optimisation_types=optimisation_types,
                                time_period_dict=time_period_dict)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.ARTICLE_FIGURES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
