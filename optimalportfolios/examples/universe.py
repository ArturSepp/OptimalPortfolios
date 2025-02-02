"""
fetch an universe of bond etfs for testing optimisations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
import yfinance as yf
from typing import Tuple
from enum import Enum


def fetch_benchmark_universe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    fetch a universe of etfs
    define custom universe with asset class grouping
    5 asset groups with 3 etfs in each
    """
    universe_data = dict(SPY='Equities',
                         QQQ='Equities',
                         EEM='Equities',
                         TLT='Bonds',
                         IEF='Bonds',
                         TIP='Bonds',
                         IGSB='IG',
                         LQD='IG',
                         EMB='IG',
                         HYG='HighYield',
                         SHYG='HighYield',
                         FALN='HighYield',
                         GLD='Commodts',
                         GSG='Commodts',
                         COMT='Commodts')
    group_data = pd.Series(universe_data)  # for portfolio reporting
    equal_weight = 1.0 / len(universe_data.keys())
    benchmark_weights = {x: equal_weight for x in universe_data.keys()}

    # asset class loadings
    ac_loadings = qis.set_group_loadings(group_data=group_data)

    tickers = list(universe_data.keys())
    benchmark_weights = pd.Series(benchmark_weights)
    prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Close'][tickers]
    prices = prices.asfreq('B').ffill()
    # for group lass
    ac_benchmark_prices = prices[['SPY', 'TLT', 'LQD', 'HYG', 'GSG']].rename(dict(SPY='Equities', TLT='Bonds', IG='LQD', HYG='HighYield', GLD='Commodts'))

    # select asset class benchmarks from universe
    benchmark_prices = prices[['SPY', 'TLT']]

    return prices, benchmark_prices, ac_loadings, benchmark_weights, group_data, ac_benchmark_prices


class UnitTests(Enum):
    ILLUSTRATE_INPUT_DATA = 1


def run_unit_test(unit_test: UnitTests):

    prices, benchmark_prices, ac_loadings, benchmark_weights, group_data, ac_benchmark_prices = fetch_benchmark_universe_data()

    if unit_test == UnitTests.ILLUSTRATE_INPUT_DATA:
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)
            qis.plot_prices_with_dd(prices=prices, axs=axs)

        plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.ILLUSTRATE_INPUT_DATA

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
