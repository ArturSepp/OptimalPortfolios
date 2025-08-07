"""
backtesting report for BTC portfolios
to use pybloqs for pandas > 2.x
locate file "...\Lib\site-packages\pybloqs\jinja\table.html"
change line 44 from:
{% for col_name, cell in row.iteritems() %}
to:
{% for col_name, cell in row.items() %}
"""

# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from enum import Enum
import pybloqs as p
import qis as qis
from qis import TimePeriod, PerfParams, BenchmarkReturnsQuantileRegimeSpecs, PerfStat

from optimalportfolios.examples.crypto_allocation.load_prices import Assets, load_prices, load_risk_free_rate
from optimalportfolios.reports.marginal_backtest import OptimisationType
from optimalportfolios.reports.config import KWARGS_SUPTITLE, KWARGS_TITLE, KWARGS_FIG, KWARGS_TEXT
from optimalportfolios.examples.crypto_allocation.backtest_portfolios_for_article import run_joint_backtest

PERF_PARAMS = PerfParams(freq_vol='ME', freq_reg='ME', freq_drawdown='ME', rates_data=load_risk_free_rate())
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='QE')

LOCAL_PATH = "C://Users//artur//OneDrive//analytics//outputs//"
FIGURE_SAVE_PATH = "C://Users//artur//OneDrive//My Papers//Working Papers//CryptoAllocation. Zurich. Jan 2022//figs1//"
SAVE_FIGS = False


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
                  perf_columns=PERF_COLUMNS)


def report_backtest_all_optimisation_types(time_period: TimePeriod,
                                           perf_time_period: qis.TimePeriod,
                                           crypto_asset: str = 'BTC',
                                           optimisation_types: List[OptimisationType] = (OptimisationType.ERC,
                                                                                         OptimisationType.MAX_DIV),
                                           time_period_dict: Dict[str, TimePeriod] = None,
                                           ) -> None:
    """
    create pdf reports for optimisation types
    """
    prices_all = load_prices(crypto_asset=crypto_asset, is_updated=True)
    benchmark_price = prices_all[Assets.BAL.value].rename('100 Bal')
    prices_unconstrained = prices_all.drop(Assets.BAL.value, axis=1)

    b_reports = []
    for optimisation_type in optimisation_types:
        report = run_backtest_pdf_report(prices_unconstrained=prices_unconstrained,
                                         prices_all=prices_all,
                                         benchmark_price=benchmark_price,
                                         marginal_asset=crypto_asset,
                                         time_period=time_period,
                                         perf_time_period=perf_time_period,
                                         time_period_dict=time_period_dict,
                                         optimisation_type=optimisation_type,
                                         **FIG_KWARGS)
        b_reports.append(p.Block(report))
    b_reports = p.VStack(b_reports, styles={"page-break-after": "always"})
    filename = f"{LOCAL_PATH}{crypto_asset}_backtests_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
    b_reports.save(filename)
    print(f"saved optimisation report to {filename}")


def create_performance_attrib_table(time_period_dict: Dict[str, TimePeriod],
                                    time_period: TimePeriod,
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
    for period, perf_time_period in time_period_dict.items():
        outs = {}
        btc_total_perf[period] = qis.compute_total_return(prices=perf_time_period.locate(btc_price))
        for optimisation_type in optimisation_types:
            alts_wo, alts_crypto, bal_wo, bal_crypto = run_joint_backtest(prices_unconstrained=prices_unconstrained,
                                                                          prices_all=prices_all,
                                                                          marginal_asset=marginal_asset,
                                                                          perf_time_period=perf_time_period,
                                                                          time_period=time_period,
                                                                          optimisation_type=optimisation_type)
            # perf attribution
            perf_alts_wo = alts_wo.get_performance_attribution(time_period=perf_time_period).rename('W/O')
            perf_alts_crypto = alts_crypto.get_performance_attribution(time_period=perf_time_period).rename('With')
            alts_perf = pd.concat([perf_alts_crypto, perf_alts_wo], axis=1)
            perf_bal_wo = bal_wo.get_performance_attribution(time_period=perf_time_period).rename('W/O')
            perf_bal_crypto = bal_crypto.get_performance_attribution(time_period=perf_time_period).rename('With')
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
                            perf_time_period: TimePeriod,
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
                                                                      perf_time_period=perf_time_period,
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
                            perf_time_period: TimePeriod,
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
                                                                  perf_time_period=perf_time_period,
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
                                     **kwargs)
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
                                     **kwargs)
    b_fig_table_bal = p.Block([p.Paragraph(f"(B) {first_asset_target_weight:0.0%}/{1.0-first_asset_target_weight:0.0%}"
                                           f" Balanced/Alts Portfolio", **KWARGS_TITLE),
                               p.Block(fig_table_bal, **KWARGS_FIG)],
                              **KWARGS_TEXT)
    blocks.append(b_fig_table_bal)

    # weights
    with sns.axes_style('darkgrid'):
        kwargs_w = qis.update_kwargs(kwargs, dict(ncol=2, bbox_to_anchor=(0.5, 1.20)))

        fig_weights_alt, ax = plt.subplots(1, 1, figsize=(figsize[0], 6), constrained_layout=True)
        alts_crypto.plot_weights(ax=ax, freq='ME', **kwargs_w)
        b_fig_weights_alt = p.Block([p.Paragraph("", **KWARGS_TITLE),
                                     p.Paragraph(f"(A) Alternatives Portfolio Weights", **KWARGS_TITLE),
                                     p.Block(fig_weights_alt, **KWARGS_FIG)],
                                    **KWARGS_TEXT)
        blocks.append(b_fig_weights_alt)

        fig_weights_bal, ax = plt.subplots(1, 1, figsize=(figsize[0], 6), constrained_layout=True)
        # move bal to the end
        columns = list(bal_crypto.weights.columns[1:]) + [bal_crypto.weights.columns[0]]
        bal_crypto.plot_weights(columns=columns, freq='ME', ax=ax, **kwargs_w)
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


def backtest_constant_weight_portfolios(crypto_asset: str = 'BTC',
                                        rebalancing_freq: str = 'QE',
                                        is_alternatives: bool = False,
                                        perf_time_period: TimePeriod = None,
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

    if perf_time_period is not None:
        prices = perf_time_period.locate(prices)

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
                                                    rebalancing_freq=rebalancing_freq,
                                                    is_rebalanced_at_first_date=True,
                                                    ticker=ticker)
        navs.append(portfolio_wo.nav)
    navs = pd.concat(navs, axis=1)

    figs = []
    kwargs = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                  alpha_format='{0:+0.0%}',
                  beta_format='{:0.1f}')
    # time series
    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8), tight_layout=True)
        figs.append(fig)
        qis.plot_prices(prices=navs,
                        title=f"Simulation of 60/40 portfolio with {crypto_asset} overlay",
                        perf_stats_labels=qis.PerfStatsLabels.DETAILED_EXCESS_SHARPE.value,
                        perf_params=PERF_PARAMS,
                        ax=ax,
                        **kwargs)

        fig, axs = plt.subplots(2, 1, figsize=(10, 12), tight_layout=True)
        figs.append(fig)
        qis.plot_prices_with_dd(prices=navs,
                                perf_stats_labels=qis.PerfStatsLabels.DETAILED_EXCESS_SHARPE.value,
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


class LocalTests(Enum):
    ALL_OPTIMISATION_TYPES = 1
    PERFORMANCE_ATTRIB_TABLE = 2
    WEIGHTS_FIGURE = 3
    CONSTANT_WEIGHT_PORTFOLIOS = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    optimisation_types = [OptimisationType.ERC,
                          OptimisationType.MAX_DIV,
                          OptimisationType.MAX_SHARPE,
                          OptimisationType.MIXTURE]

    end_date = '16Aug2024'
    time_period = TimePeriod('19Jul2010', end_date)  # for weight calculations
    perf_time_period = TimePeriod('31Mar2016', end_date)  # for reporting

    if local_test == LocalTests.ALL_OPTIMISATION_TYPES:

        time_period_dict = {'1y': TimePeriod(start='31Mar2022', end=perf_time_period.end),
                            '2y': TimePeriod(start='31Mar2021', end=perf_time_period.end),
                            '3y': TimePeriod(start='31Mar2020', end=perf_time_period.end),
                            '5y': TimePeriod(start='31Mar2018', end=perf_time_period.end),
                            '7y': TimePeriod(start='31Mar2016', end=perf_time_period.end)}

        report_backtest_all_optimisation_types(crypto_asset='ETH',
                                               optimisation_types=optimisation_types,
                                               time_period=time_period,
                                               perf_time_period=perf_time_period,
                                               time_period_dict=time_period_dict)

    elif local_test == LocalTests.PERFORMANCE_ATTRIB_TABLE:
        time_period_dict = {'2016Q1-now': TimePeriod(start='31Dec2015', end=end_date),
                            '2021Q1-now': TimePeriod(start='31Dec2020', end=end_date)}
        create_performance_attrib_table(optimisation_types=optimisation_types,
                                        time_period_dict=time_period_dict,
                                        time_period=time_period)

    elif local_test == LocalTests.WEIGHTS_FIGURE:
        plot_weights_timeseries(time_period=time_period,
                                perf_time_period=perf_time_period,
                                optimisation_types=optimisation_types)

    elif local_test == LocalTests.CONSTANT_WEIGHT_PORTFOLIOS:
        time_period_dict = {'2016Q2-now': TimePeriod(start='31Mar2016', end=end_date),
                            '2021Q1-now': TimePeriod(start='31Dec2020', end=end_date)}
        backtest_constant_weight_portfolios(
                                        crypto_asset='ETH',
                                        rebalancing_freq='QE',
                                        is_alternatives=False,
                                        perf_time_period=perf_time_period,
                                        time_period_dict=time_period_dict)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ALL_OPTIMISATION_TYPES)
