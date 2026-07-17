"""
example of profiling alpha signals with the rank-based profiler.

The profiler evaluates a signal by holding the top-quantile of assets ranked by the signal,
equal-weighted, against an equal-weight-all benchmark -- no optimiser, no covariance -- so it
isolates the selection power of the signal. This example profiles carry, low-beta and momentum on a
bond-ETF universe, jointly and singly, and prints a performance-plus-turnover table. The carry panel
is a real trailing-12m dividend yield built from yfinance dividend history.
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
import yfinance as yf
from enum import Enum

# optimalportfolios
from optimalportfolios.alphas import (profile_carry,
                                      profile_low_beta,
                                      profile_alpha_signals,
                                      ProfileSignal,
                                      compute_alpha_rank_analysis_table,
                                      generate_alpha_profile_report)


def compute_trailing_dividend_yield(tickers: list) -> pd.DataFrame:
    """build a trailing-12m dividend-yield panel from yfinance dividend history.

    For each ticker, sum the dividends paid over a trailing 365-day window and divide by the closing
    price, giving a daily distribution-yield series. Non-distributing days carry the last trailing-year
    figure forward. Returns a T x N panel aligned across tickers.
    """
    yields = {}
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(period='max', auto_adjust=True)
        if hist.empty or 'Dividends' not in hist.columns:
            continue
        rolling_1y_dividends = hist['Dividends'].rolling(window='365D').sum()
        rolling_yield = rolling_1y_dividends / hist['Close']
        rolling_yield.index = rolling_yield.index.tz_localize(None)
        yields[ticker] = rolling_yield
    return pd.DataFrame(yields)


def fetch_bond_etf_universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """fetch a bond-ETF price panel, a trailing dividend-yield carry panel, a benchmark and groups."""
    universe_data = dict(TLT='Treasuries', IEF='Treasuries', SHY='Treasuries',
                         TIP='TIPS', STIP='TIPS',
                         LQD='IG', IGSB='IG',
                         HYG='HighYield', SHYG='HighYield', FALN='HighYield',
                         EMB='EM', EMHY='EM')
    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)
    prices = yf.download(tickers=tickers, start="2007-12-31", end=None,
                         ignore_tz=True, auto_adjust=True)['Close'][tickers].asfreq('B', method='ffill')

    # carry = trailing-12m distribution yield per ETF, reindexed onto the price grid
    carry = compute_trailing_dividend_yield(tickers)
    carry = carry.reindex(index=prices.index, method='ffill')[tickers]

    benchmark_price = yf.download(tickers='AGG', start="2007-12-31", end=None,
                                  ignore_tz=True, auto_adjust=True)['Close'].asfreq('B', method='ffill')
    if isinstance(benchmark_price, pd.DataFrame):
        benchmark_price = benchmark_price.iloc[:, 0]
    benchmark_price = benchmark_price.reindex(index=prices.index, method='ffill')
    return prices, carry, benchmark_price, group_data


class LocalTests(Enum):
    JOINT_PROFILE = 1        # profile carry + low_beta + momentum together
    SINGLE_CARRY = 2         # profile carry alone
    QUANTILE_SWEEP = 3       # carry at several top-quantiles


def run_local_test(local_test: LocalTests = LocalTests.JOINT_PROFILE) -> None:
    prices, carry, benchmark_price, group_data = fetch_bond_etf_universe()
    time_period = qis.TimePeriod('31Dec2015', prices.index[-1])
    perf_params = qis.PerfParams(freq='ME')

    if local_test == LocalTests.JOINT_PROFILE:
        # rank each signal's top third, equal-weighted, vs equal-weight-all, rebalanced quarterly
        multi_portfolio_data = profile_alpha_signals(
            prices=prices,
            signals=[ProfileSignal.CARRY, ProfileSignal.LOW_BETA, ProfileSignal.MOMENTUM],
            benchmark_price=benchmark_price,
            carry=carry,
            returns_freq='ME',
            quantile=1.0 / 3.0,
            rebalancing_freq='QE',
            time_period=time_period)
        table = compute_alpha_rank_analysis_table(
            multi_portfolio_data, time_period=time_period, perf_params=perf_params)
        print(table.to_string())
        # generate and save the multi-strategy factsheet PDF
        generate_alpha_profile_report(multi_portfolio_data=multi_portfolio_data,
                                      time_period=time_period,
                                      group_data=group_data,
                                      backtest_name='Bond ETF Alpha Signal Profile',
                                      file_name='alpha_signal_profile')
        plt.show()

    elif local_test == LocalTests.SINGLE_CARRY:
        multi_portfolio_data = profile_carry(
            prices=prices, carry=carry, returns_freq='ME', vol_span=13,
            quantile=1.0 / 3.0, rebalancing_freq='QE', time_period=time_period)
        table = compute_alpha_rank_analysis_table(
            multi_portfolio_data, time_period=time_period, perf_params=perf_params)
        print(table.to_string())

    elif local_test == LocalTests.QUANTILE_SWEEP:
        # how does carry's edge change with basket concentration?
        for quantile in [0.25, 1.0 / 3.0, 0.5]:
            multi_portfolio_data = profile_carry(
                prices=prices, carry=carry, returns_freq='ME', vol_span=13,
                quantile=quantile, rebalancing_freq='QE', time_period=time_period)
            table = compute_alpha_rank_analysis_table(
                multi_portfolio_data, time_period=time_period, perf_params=perf_params)
            carry_sharpe = table.loc['carry', 'Sharpe']
            carry_turnover = table.loc['carry', 'Turnover p.a.']
            print(f"quantile={quantile:.2f}: carry Sharpe={carry_sharpe:.2f}, "
                  f"turnover={carry_turnover:.0%}")


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.JOINT_PROFILE)