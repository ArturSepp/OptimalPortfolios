# imports
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Tuple
from enum import Enum
import qis as qis

from optimalportfolios import compute_rolling_ewma_risk_based_weights
from optimalportfolios import PortfolioObjective, backtest_rolling_optimal_portfolio


def fetch_universe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    fetch universe data for the portfolio construction:
    1. dividend and split adjusted end of day prices: price data may start / end at different dates
    2. benchmark prices which is used for portfolio reporting and benchmarking
    3. universe group data for portfolio reporting and risk attribution for large universes
    this function is using yfinance to fetch the price data
    """
    universe_data = dict(SPY='Equities',
                         QQQ='Equities',
                         EEM='Equities',
                         TLT='Bonds',
                         IEF='Bonds',
                         LQD='Credit',
                         HYG='HighYield',
                         GLD='Gold')
    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)
    prices = yf.download(tickers, start=None, end=None, ignore_tz=True)['Adj Close']
    prices = prices[tickers]  # arrange as given
    prices = prices.asfreq('B', method='ffill')  # refill at B frequency
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, group_data


prices, benchmark_prices, group_data  = fetch_universe_data()

weights = compute_rolling_ewma_risk_based_weights(prices=prices,
                                                  portfolio_objective=PortfolioObjective.MAX_DIVERSIFICATION,
                                                  min_weights=None,
                                                  max_weights=None,
                                                  fixed_weights=None,
                                                  rebalancing_freq='Q',
                                                  returns_freq='W-WED',
                                                  span=52)

