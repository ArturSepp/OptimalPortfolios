"""
Backtest alpha signals.

Standalone research tool for evaluating individual alpha signals
and their combinations via long-only portfolio backtests.

Uses the public signal functions from optimalportfolios.alphas.signals
directly, without the AlphaAggregator (which is for production TAA).
"""
import pandas as pd
import numpy as np
import qis as qis
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Optional

from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.low_beta import compute_low_beta_alpha


class AlphaSignal(Enum):
    """Enumeration of alpha signals available for backtesting."""
    MOMENTUM = 'Momentum'
    LOW_BETA = 'LowBeta'
    MOMENTUM_AND_BETA = 'MomentumAndBeta'


def compute_signal_scores(prices: pd.DataFrame,
                          alpha_signal: AlphaSignal,
                          group_data: pd.Series = None,
                          benchmark_price: pd.Series = None,
                          returns_freq: Optional[str] = None,
                          mom_long_span: int = 12,
                          mom_short_span: Optional[int] = 1,
                          beta_span: int = 24,
                          vol_span: Optional[int] = 13,
                          ) -> pd.DataFrame:
    """
    Compute cross-sectional alpha scores for a given signal type.

    Dispatches to the appropriate signal function(s) and combines
    if MOMENTUM_AND_BETA is selected.

    Args:
        prices: Asset price panel.
        alpha_signal: Which signal to compute.
        group_data: Optional group labels for within-group scoring.
        benchmark_price: Benchmark for excess return / beta computation.
        returns_freq: Return frequency.
        mom_long_span: Momentum long EWMA span.
        mom_short_span: Momentum short EWMA span (reversal).
        beta_span: Beta estimation EWMA span.
        vol_span: Vol normalisation span for momentum.

    Returns:
        Cross-sectional scores (T × N).
    """
    if alpha_signal == AlphaSignal.MOMENTUM:
        scores, _ = compute_momentum_alpha(
            prices=prices,
            benchmark_price=benchmark_price,
            returns_freq=returns_freq or 'ME',
            group_data=group_data,
            long_span=mom_long_span,
            short_span=mom_short_span,
            vol_span=vol_span)

    elif alpha_signal == AlphaSignal.LOW_BETA:
        scores, _ = compute_low_beta_alpha(
            prices=prices,
            benchmark_price=benchmark_price,
            returns_freq=returns_freq or 'ME',
            group_data=group_data,
            beta_span=beta_span)

    elif alpha_signal == AlphaSignal.MOMENTUM_AND_BETA:
        mom_scores, _ = compute_momentum_alpha(
            prices=prices,
            benchmark_price=benchmark_price,
            returns_freq=returns_freq or 'ME',
            group_data=group_data,
            long_span=mom_long_span,
            short_span=mom_short_span,
            vol_span=vol_span)

        beta_scores, _ = compute_low_beta_alpha(
            prices=prices,
            benchmark_price=benchmark_price,
            returns_freq=returns_freq or 'ME',
            group_data=group_data,
            beta_span=beta_span)

        beta_scores = beta_scores.reindex(index=mom_scores.index, method='ffill')
        scores = (mom_scores + beta_scores) / np.sqrt(2.0)

    else:
        raise NotImplementedError(f"alpha_signal={alpha_signal}")

    return scores


def backtest_alpha_signals(prices: pd.DataFrame,
                           group_data: pd.Series,
                           group_order: List[str],
                           rebalancing_costs: pd.Series,
                           benchmark_prices: pd.DataFrame,
                           time_period: qis.TimePeriod,
                           alpha_signal: AlphaSignal = AlphaSignal.MOMENTUM,
                           mom_long_span: int = 12,
                           mom_short_span: Optional[int] = 1,
                           beta_span: int = 24,
                           rebalancing_freq: str = 'QE',
                           returns_freq: Optional[str] = None
                           ) -> List[plt.Figure]:
    """
    Backtest a single alpha signal vs equal weight.

    Computes within-group alpha scores, converts to long-only weights,
    and generates a strategy-benchmark factsheet.

    Args:
        prices: Asset price panel.
        group_data: Group labels for within-group scoring and reporting.
        group_order: Display order for groups.
        rebalancing_costs: Per-asset rebalancing cost.
        benchmark_prices: Benchmark prices for performance attribution.
        time_period: Backtest reporting period.
        alpha_signal: Which signal to backtest.
        mom_long_span: Momentum long span.
        mom_short_span: Momentum short span.
        beta_span: Beta estimation span.
        rebalancing_freq: Portfolio rebalancing frequency.
        returns_freq: Return frequency for signal computation.

    Returns:
        List of factsheet figures.
    """
    scores = compute_signal_scores(
        prices=prices, alpha_signal=alpha_signal,
        group_data=group_data, returns_freq=returns_freq,
        mom_long_span=mom_long_span, mom_short_span=mom_short_span,
        beta_span=beta_span)
    signal_weights = qis.df_to_long_only_allocation_sum1(df=scores)

    equal_weights = qis.df_to_equal_weight_allocation(df=prices)
    weights = {alpha_signal.value: signal_weights, 'Equal Weight': equal_weights}

    portfolio_datas = []
    for ticker, weight in weights.items():
        weight = time_period.locate(weight).asfreq(rebalancing_freq, method='ffill')
        mandate_portfolio = qis.backtest_model_portfolio(
            prices=prices.loc[weight.index[0]:, :],
            weights=weight,
            rebalancing_freq=rebalancing_freq,
            rebalancing_costs=rebalancing_costs,
            ticker=f"{ticker}")
        mandate_portfolio.set_group_data(group_data=group_data, group_order=group_order)
        portfolio_datas.append(mandate_portfolio)

    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
    kwargs = qis.fetch_default_report_kwargs(
        time_period=time_period,
        reporting_frequency=qis.ReportingFrequency.MONTHLY,
        add_rates_data=False)
    figs = qis.generate_strategy_benchmark_factsheet_plt(
        multi_portfolio_data=multi_portfolio_data,
        backtest_name=f"{alpha_signal.value} portfolios",
        time_period=time_period,
        add_brinson_attribution=True,
        add_exposures_pnl_attribution=True,
        add_strategy_factsheet=True,
        add_grouped_exposures=False,
        add_grouped_cum_pnl=False,
        **kwargs)
    return figs


def multi_backtest_alpha_signals(prices: pd.DataFrame,
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
    Backtest all alpha signals side-by-side vs equal weight.

    Runs each AlphaSignal variant and generates a multi-portfolio
    comparison factsheet.

    Args:
        prices: Asset price panel.
        group_data: Group labels.
        group_order: Display order.
        rebalancing_costs: Per-asset rebalancing cost.
        benchmark_prices: Benchmark prices.
        time_period: Reporting period.
        mom_long_span: Momentum long span.
        mom_short_span: Momentum short span.
        beta_span: Beta estimation span.
        rebalancing_freq: Portfolio rebalancing frequency.
        returns_freq: Return frequency for signal computation.

    Returns:
        List of factsheet figures.
    """
    weights = dict()
    weights['Equal Weight'] = qis.df_to_equal_weight_allocation(df=prices)

    for alpha_signal in list(AlphaSignal):
        scores = compute_signal_scores(
            prices=prices, alpha_signal=alpha_signal,
            group_data=group_data, returns_freq=returns_freq,
            mom_long_span=mom_long_span, mom_short_span=mom_short_span,
            beta_span=beta_span)
        weights[alpha_signal.value] = qis.df_to_long_only_allocation_sum1(df=scores)

    portfolio_datas = []
    for ticker, weight in weights.items():
        weight = time_period.locate(weight).asfreq(rebalancing_freq, method='ffill')
        mandate_portfolio = qis.backtest_model_portfolio(
            prices=prices.loc[weight.index[0]:, :],
            weights=weight,
            rebalancing_freq=rebalancing_freq,
            rebalancing_costs=rebalancing_costs,
            ticker=f"{ticker}")
        mandate_portfolio.set_group_data(group_data=group_data, group_order=group_order)
        portfolio_datas.append(mandate_portfolio)

    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
    kwargs = qis.fetch_default_report_kwargs(
        time_period=time_period,
        reporting_frequency=qis.ReportingFrequency.MONTHLY,
        add_rates_data=False)
    figs = qis.generate_multi_portfolio_factsheet(
        multi_portfolio_data=multi_portfolio_data,
        backtest_name=f"Alpha portfolios",
        time_period=time_period,
        add_strategy_factsheets=False,
        add_brinson_attribution=True,
        add_exposures_pnl_attribution=True,
        add_strategy_factsheet=True,
        add_grouped_exposures=False,
        add_grouped_cum_pnl=False,
        **kwargs)
    return figs


class CrossBacktestParam(Enum):
    MOM_SPAN = 'MOM_SPAN'
    BETA_SPAN = 'BETA_SPAN'
    MOM_BETA_SPAN = 'MOM_BETA_SPAN'


def cross_backtest_alpha_signals(prices: pd.DataFrame,
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
    Parameter sensitivity sweep for alpha signals.

    Sweeps a single parameter (momentum span, beta span, or both)
    across a range of values and compares resulting portfolios.

    Args:
        prices: Asset price panel.
        group_data: Group labels.
        group_order: Display order.
        rebalancing_costs: Per-asset rebalancing cost.
        benchmark_prices: Benchmark prices.
        time_period: Reporting period.
        cross_backtest_param: Which parameter to sweep.
        mom_long_span: Default momentum span (used when not sweeping).
        mom_short_span: Momentum short span.
        beta_span: Default beta span (used when not sweeping).
        rebalancing_freq: Portfolio rebalancing frequency.
        returns_freq: Return frequency for signal computation.

    Returns:
        List of factsheet figures.
    """
    span_values = [3, 6, 12, 18, 24, 36, 60]

    if cross_backtest_param == CrossBacktestParam.MOM_SPAN:
        alpha_signal = AlphaSignal.MOMENTUM
        configs = [(s, beta_span, f"long_span={s:0.0f}") for s in span_values]

    elif cross_backtest_param == CrossBacktestParam.BETA_SPAN:
        alpha_signal = AlphaSignal.LOW_BETA
        configs = [(mom_long_span, s, f"beta_span={s:0.0f}") for s in span_values]

    elif cross_backtest_param == CrossBacktestParam.MOM_BETA_SPAN:
        alpha_signal = AlphaSignal.MOMENTUM_AND_BETA
        configs = [(s, s, f"spans={s:0.0f}") for s in span_values]

    else:
        raise NotImplementedError(f"{cross_backtest_param}")

    weights = dict()
    weights['Equal Weight'] = qis.df_to_equal_weight_allocation(df=prices)

    for mom_span, b_span, label in configs:
        scores = compute_signal_scores(
            prices=prices, alpha_signal=alpha_signal,
            group_data=group_data, returns_freq=returns_freq,
            mom_long_span=mom_span, mom_short_span=mom_short_span,
            beta_span=b_span)
        weights[label] = qis.df_to_long_only_allocation_sum1(df=scores)

    portfolio_datas = []
    for ticker, weight in weights.items():
        weight = time_period.locate(weight).asfreq(rebalancing_freq, method='ffill')
        mandate_portfolio = qis.backtest_model_portfolio(
            prices=prices.loc[weight.index[0]:, :],
            weights=weight,
            rebalancing_freq=rebalancing_freq,
            rebalancing_costs=rebalancing_costs,
            ticker=f"{ticker}")
        mandate_portfolio.set_group_data(group_data=group_data, group_order=group_order)
        portfolio_datas.append(mandate_portfolio)

    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
    kwargs = qis.fetch_default_report_kwargs(
        time_period=time_period,
        reporting_frequency=qis.ReportingFrequency.MONTHLY,
        add_rates_data=False)
    figs = qis.generate_multi_portfolio_factsheet(
        multi_portfolio_data=multi_portfolio_data,
        backtest_name=f"{alpha_signal.value} cross portfolios",
        time_period=time_period,
        add_strategy_factsheets=False,
        add_brinson_attribution=True,
        add_exposures_pnl_attribution=True,
        add_strategy_factsheet=True,
        add_grouped_exposures=False,
        add_grouped_cum_pnl=False,
        **kwargs)
    return figs