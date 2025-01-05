"""
example of maximization of alpha with target return
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
import yfinance as yf
from enum import Enum
from typing import Tuple

from optimalportfolios import (Constraints, compute_portfolio_vol,
                               wrapper_maximise_alpha_with_target_return,
                               rolling_maximise_alpha_with_target_return)


def run_bonds_etf_optimal_portfolio(prices: pd.DataFrame,
                                    yields: pd.DataFrame,
                                    target_returns: pd.Series,
                                    time_period: qis.TimePeriod = qis.TimePeriod('31Jan2008', '19Jul2024')
                                    ) -> pd.DataFrame:
    """
    run the optimal portfolio
    """
    momentum = qis.compute_ewm_long_short_filtered_ra_returns(returns=qis.to_returns(prices, freq='W-WED'), vol_span=13,
                                                              long_span=13, short_span=None, weight_lag=0)
    # momentum = qis.map_signal_to_weight(signals=momentum, loc=0.0, slope_right=0.5, slope_left=0.5, tail_level=3.0)
    alphas = qis.df_to_cross_sectional_score(df=momentum)

    constraints0 = Constraints(is_long_only=True,
                               min_weights=pd.Series(0.0, index=prices.columns),
                               max_weights=pd.Series(0.2, index=prices.columns),
                               max_target_portfolio_vol_an=0.065,
                               max_exposure=1.0,
                               min_exposure=0.5,
                               turnover_constraint=0.30  # 25% per month
                               )

    weights = rolling_maximise_alpha_with_target_return(prices=prices,
                                                        alphas=alphas,
                                                        yields=yields,
                                                        target_returns=target_returns,
                                                        constraints0=constraints0,
                                                        time_period=time_period,
                                                        span=52,
                                                        rebalancing_freq='ME',
                                                        verbose=True)
    return weights


def compute_dividend_rolling_1y(dividend: pd.Series):
    rolling_1y = 4.0 * dividend.asfreq('ME', method='ffill').fillna(0.0).rolling(3).sum()
    return rolling_1y


def fetch_benchmark_universe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    define custom universe with asset class grouping
    """
    universe_data = dict(
        TLT='Tresuries',
        IEF='Tresuries',
        SHY='Tresuries',
        TFLO='Tresuries',  # floating ust
        TIP='TIPS',
        STIP='TIPS',  # 0-5 y tips
        LQD='IG',
        IGSB='IG',  # 1-5y corporates
        FLOT='IG',  # corporate float
        MUB='Munies/MBS',
        MBB='Munies/MBS',
        HYG='HighYield',
        SHYG='HighYield',  # 0-5y HY
        FALN='HighYield',  # fallen angels
        EMB='EM',
        EMHY='EM',  # em hy
        ICVT='Hybrid'  # converts
        # PFF='Hybrid'  # preferds
    )
    yield_deflators = dict(TIP=0.75, STIP=0.75)  # tips pay income distribution

    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)  # for portfolio reporting
    prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close'][tickers]
    prices = prices.asfreq('B', method='ffill')

    dividends = {}
    yields = {}  # assume that div is paid monthly and extrapolate last 3 m to 1 year, yields are defined on monthly schedule
    for ticker in tickers:
        dividend = yf.Ticker(ticker).dividends
        dividend.index = dividend.index.tz_localize(None)  # remove hours and tz
        dividends[ticker] = dividend
        rolling_1y = compute_dividend_rolling_1y(dividend=dividend)
        if ticker in yield_deflators.keys():
            rolling_1y = yield_deflators[ticker] * rolling_1y
        yields[ticker] = rolling_1y.divide(prices[ticker].reindex(index=rolling_1y.index, method='ffill'))
    dividends = pd.DataFrame.from_dict(dividends, orient='columns')
    yields = pd.DataFrame.from_dict(yields, orient='columns')

    benchmarks = ['AGG']
    benchmark_prices = yf.download(tickers=benchmarks, start=None, end=None, ignore_tz=True)['Adj Close'].to_frame('AGG')#[benchmarks]
    target_returns = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0
    target_returns = target_returns.reindex(index=prices.index).ffill().rename('Target return')
    return prices, benchmark_prices, dividends, yields, target_returns, group_data


class UnitTests(Enum):
    ILLUSTRATE_INPUT_DATA = 1
    ONE_STEP_OPTIMISATION = 2
    ROLLING_OPTIMISATION = 3


def run_unit_test(unit_test: UnitTests):

    import optimalportfolios.local_path as lp

    prices, benchmark_prices, dividends, yields, target_returns, group_data, ac_benchmark_prices = fetch_benchmark_universe_data()

    if unit_test == UnitTests.ILLUSTRATE_INPUT_DATA:
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)
            qis.plot_prices_with_dd(prices=prices, axs=axs)

            fig, axs = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)
            qis.plot_time_series(df=dividends, title='Dividends', ax=axs[0])
            yields = pd.concat([target_returns.reindex(index=yields.index, method='ffill'), yields], axis=1)
            qis.plot_time_series(df=yields, title='Yields', var_format='{:,.2%}', ax=axs[1])
        plt.show()

    elif unit_test == UnitTests.ONE_STEP_OPTIMISATION:
        # optimise using last available data as inputs
        returns = qis.to_returns(prices, freq='W-WED', is_log_returns=True)
        pd_covar = pd.DataFrame(52.0 * qis.compute_masked_covar_corr(data=returns, is_covar=True),
                                index=prices.columns, columns=prices.columns)
        print(f"pd_covar=\n{pd_covar}")
        last_yields = yields.iloc[-1, :]
        print(f"last_yields=\n{last_yields}")
        target_return = target_returns.iloc[-1] + 0.01
        print(f"target_return=\n{target_return}")
        momenum_1y = prices.divide(prices.shift(260)) - 1.0
        alphas = qis.df_to_cross_sectional_score(df=momenum_1y.iloc[-1, :])
        print(f"alphas=\n{alphas}")
        constraints0 = Constraints(is_long_only=True,
                                   min_weights=pd.Series(0.0, index=prices.columns),
                                   max_weights=pd.Series(0.15, index=prices.columns),
                                   max_target_portfolio_vol_an=0.065,
                                   max_exposure=1.0,
                                   min_exposure=0.5
                                   )

        weights = wrapper_maximise_alpha_with_target_return(pd_covar=pd_covar, alphas=alphas, yields=last_yields,
                                                            target_return=target_return,
                                                            constraints0=constraints0)
        print(f"weights={weights}")
        print(f"exposure = {np.sum(weights)}")
        print(f"portfolio_vol = {compute_portfolio_vol(covar=pd_covar, weights=weights)}")
        print(f"portfolio_yield = {np.nansum(weights.multiply(last_yields))}")
        qis.plot_heatmap(df=pd.DataFrame(qis.compute_masked_covar_corr(data=returns, is_covar=False),
                                         index=prices.columns, columns=prices.columns))
        qis.plot_bars(df=weights)
        plt.show()

    elif unit_test == UnitTests.ROLLING_OPTIMISATION:
        # optimise using last available data as inputs
        time_period = qis.TimePeriod('31Jan2008', '19Jul2024')
        time_period = qis.TimePeriod('31Dec2012', '19Jun2024')
        weights = run_bonds_etf_optimal_portfolio(prices=prices,
                                                  yields=yields,
                                                  target_returns=target_returns + 0.005,
                                                  time_period=time_period)
        print(f"weights={weights}")
        print(f"exposure={weights.sum(1)}")
        portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                      weights=weights,
                                                      rebalancing_costs=0.0000,
                                                      weight_implementation_lag=1,
                                                      ticker=f"Optimal Portfolio")
        portfolio_data.set_group_data(group_data=group_data)
        kwargs = qis.fetch_default_report_kwargs(time_period=time_period, add_rates_data=True)
        figs = qis.generate_strategy_factsheet(portfolio_data=portfolio_data,
                                               benchmark_prices=benchmark_prices,
                                               time_period=time_period,
                                               add_grouped_exposures=False,
                                               add_grouped_cum_pnl=False,
                                               **kwargs)
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"target return portfolio", orientation='landscape',
                             local_path=lp.get_output_path())


if __name__ == '__main__':

    unit_test = UnitTests.ROLLING_OPTIMISATION

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
