"""
backtest alphas
can be used for specific universes to check sensetivity of alphas for params
"""
import pandas as pd
import qis as qis
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Optional
from optimalportfolios.utils.factor_alphas import AlphaSignal, compute_alpha_long_only_weights


def backtest_alpha_signas(prices: pd.DataFrame,
                          group_data: pd.Series,
                          group_order: List[str],
                          rebalancing_costs: pd.Series,
                          benchmark_prices: pd.DataFrame,
                          time_period: qis.TimePeriod,
                          alpha_signal: AlphaSignal = AlphaSignal.MOMENTUM,
                          mom_long_span: int = 12,
                          mom_short_span: Optional[int] = 1,
                          beta_span: int = 24,
                          rebalancing_freq: str = 'QE'
                          ) -> List[plt.Figure]:
    """
    create portfolios based on alpha signals
    """
    signal_weights = compute_alpha_long_only_weights(prices=prices,
                                                     group_data=group_data,
                                                     group_order=group_order,
                                                     alpha_signal=alpha_signal,
                                                     mom_long_span=mom_long_span,
                                                     mom_short_span=mom_short_span,
                                                     beta_span=beta_span)

    equal_weights = qis.df_to_equal_weight_allocation(df=prices)
    weights = {alpha_signal.value: signal_weights, 'Equal Weight': equal_weights}
    portfolio_datas = []
    for ticker, weight in weights.items():
        weight = time_period.locate(weight).asfreq(rebalancing_freq, method='ffill')
        mandate_portfolio = qis.backtest_model_portfolio(prices=prices.loc[weight.index[0]:, :],
                                                         weights=weight,
                                                         rebalance_freq=rebalancing_freq,
                                                         rebalancing_costs=rebalancing_costs,
                                                         ticker=f"{ticker}")
        mandate_portfolio.set_group_data(group_data=group_data, group_order=group_order)
        portfolio_datas.append(mandate_portfolio)

    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
    kwargs = qis.fetch_default_report_kwargs(time_period=time_period, reporting_frequency=qis.ReportingFrequency.MONTHLY,
                                             add_rates_data=False)
    figs = qis.generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=multi_portfolio_data,
                                                         backtest_name=f"{alpha_signal.value} portfolios",
                                                         time_period=time_period,
                                                         add_brinson_attribution=True,
                                                         add_exposures_pnl_attribution=True,
                                                         add_strategy_factsheet=True,  # for strategy factsheet
                                                         add_grouped_exposures=False,  # for strategy factsheet
                                                         add_grouped_cum_pnl=False,  # for strategy factsheet
                                                         **kwargs)
    return figs


def multi_backtest_alpha_signas(prices: pd.DataFrame,
                                group_data: pd.Series,
                                group_order: List[str],
                                rebalancing_costs: pd.Series,
                                benchmark_prices: pd.DataFrame,
                                time_period: qis.TimePeriod,
                                mom_long_span: int = 12,
                                mom_short_span: Optional[int] = 1,
                                beta_span: int = 24,
                                rebalancing_freq: str = 'QE',
                                returns_freq: Optional[str] = None
                                ) -> List[plt.Figure]:
    """
    create portfolios based on alpha signals
    """
    weights = dict()
    weights['Equal Weight'] = qis.df_to_equal_weight_allocation(df=prices)
    for alpha_signal in list(AlphaSignal):
        weights[alpha_signal.value] = compute_alpha_long_only_weights(prices=prices,
                                                                      group_data=group_data,
                                                                      group_order=group_order,
                                                                      alpha_signal=alpha_signal,
                                                                      mom_long_span=mom_long_span,
                                                                      mom_short_span=mom_short_span,
                                                                      beta_span=beta_span,
                                                                      returns_freq=returns_freq)

    portfolio_datas = []
    for ticker, weight in weights.items():
        weight = time_period.locate(weight).asfreq(rebalancing_freq, method='ffill')
        mandate_portfolio = qis.backtest_model_portfolio(prices=prices.loc[weight.index[0]:, :],
                                                         weights=weight,
                                                         rebalance_freq=rebalancing_freq,
                                                         rebalancing_costs=rebalancing_costs,
                                                         ticker=f"{ticker}")
        mandate_portfolio.set_group_data(group_data=group_data, group_order=group_order)
        portfolio_datas.append(mandate_portfolio)

    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
    kwargs = qis.fetch_default_report_kwargs(time_period=time_period, reporting_frequency=qis.ReportingFrequency.MONTHLY,
                                             add_rates_data=False)
    figs = qis.generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                  backtest_name=f"Alpha portfolios",
                                                  time_period=time_period,
                                                  add_strategy_factsheets=False,
                                                  add_brinson_attribution=True,
                                                  add_exposures_pnl_attribution=True,
                                                  add_strategy_factsheet=True,  # for strategy factsheet
                                                  add_grouped_exposures=False,  # for strategy factsheet
                                                  add_grouped_cum_pnl=False,  # for strategy factsheet
                                                  **kwargs)
    return figs


class CrossBacktestParam(Enum):
    MOM_SPAN = 'MOM_SPAN'
    BETA_SPAN = 'BETA_SPAN'
    MOM_BETA_SPAN = 'MOM_BETA_SPAN'


def cross_backtest_alpha_signas(prices: pd.DataFrame,
                                group_data: pd.Series,
                                group_order: List[str],
                                rebalancing_costs: pd.Series,
                                benchmark_prices: pd.DataFrame,
                                time_period: qis.TimePeriod,
                                cross_backtest_param: CrossBacktestParam = CrossBacktestParam.MOM_SPAN,
                                mom_long_span: int = 12,
                                mom_short_span: Optional[int] = 1,
                                beta_span: int = 24,
                                rebalancing_freq: str = 'ME',
                                returns_freq: Optional[str] = None
                                ) -> List[plt.Figure]:
    """
    create portfolios based on alpha signals
    """
    if cross_backtest_param == CrossBacktestParam.MOM_SPAN:
        alpha_signal = AlphaSignal.MOMENTUM
        mom_long_spans = [3, 6, 12, 18, 24, 36, 60]
        beta_spans = [beta_span for _ in mom_long_spans]
        backtest_names = [f"long_span={mom_long_span:0.0f}" for mom_long_span in mom_long_spans]

    elif cross_backtest_param == CrossBacktestParam.BETA_SPAN:
        alpha_signal = AlphaSignal.LOW_BETA
        beta_spans = [3, 6, 12, 18, 24, 36, 60]
        mom_long_spans = [mom_long_span for _ in beta_spans]
        backtest_names = [f"beta_span={beta_span:0.0f}" for beta_span in beta_spans]

    elif cross_backtest_param == CrossBacktestParam.MOM_BETA_SPAN:
        alpha_signal = AlphaSignal.MOMENTUM_AND_BETA
        beta_spans = [3, 6, 12, 18, 24, 36, 60]
        mom_long_spans = beta_spans
        backtest_names = [f"spans={beta_span:0.0f}" for beta_span in beta_spans]
    else:
        raise NotImplementedError(f"{cross_backtest_param}")

    weights = dict()
    weights['Equal Weight'] = qis.df_to_equal_weight_allocation(df=prices)
    for mom_long_span, beta_span, backtest_name in zip(mom_long_spans, beta_spans, backtest_names):
        weights[backtest_name] = compute_alpha_long_only_weights(prices=prices,
                                                                 group_data=group_data,
                                                                 group_order=group_order,
                                                                 alpha_signal=alpha_signal,
                                                                 mom_long_span=mom_long_span,
                                                                 mom_short_span=mom_short_span,
                                                                 beta_span=beta_span,
                                                                 returns_freq=returns_freq)

    portfolio_datas = []
    for ticker, weight in weights.items():
        weight = time_period.locate(weight).asfreq(rebalancing_freq, method='ffill')
        mandate_portfolio = qis.backtest_model_portfolio(prices=prices.loc[weight.index[0]:, :],
                                                         weights=weight,
                                                         rebalance_freq=rebalancing_freq,
                                                         rebalancing_costs=rebalancing_costs,
                                                         ticker=f"{ticker}")
        mandate_portfolio.set_group_data(group_data=group_data, group_order=group_order)
        portfolio_datas.append(mandate_portfolio)

    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
    kwargs = qis.fetch_default_report_kwargs(time_period=time_period, reporting_frequency=qis.ReportingFrequency.MONTHLY,
                                             add_rates_data=False)
    figs = qis.generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                  backtest_name=f"{alpha_signal.value} cross portfolios",
                                                  time_period=time_period,
                                                  add_strategy_factsheets=False,
                                                  add_brinson_attribution=True,
                                                  add_exposures_pnl_attribution=True,
                                                  add_strategy_factsheet=True,  # for strategy factsheet
                                                  add_grouped_exposures=False,  # for strategy factsheet
                                                  add_grouped_cum_pnl=False,  # for strategy factsheet
                                                  **kwargs)
    return figs
