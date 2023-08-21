"""
minimal example of using the backtester
"""

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from typing import Tuple
import qis as qis
from optimalportfolios.optimization.config import PortfolioObjective
from optimalportfolios.optimization.rolling_portfolios import compute_rolling_optimal_weights_ewm_covar


# 1. we define the investment universe and allocation by asset classes
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
                         SHY='Bonds',
                         LQD='Credit',
                         HYG='HighYield',
                         GLD='Gold')
    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)
    prices = yf.download(tickers, start=None, end=None, ignore_tz=True)['Adj Close'].dropna()
    prices = prices[tickers]  # arrange as given
    prices = prices.asfreq('B', method='ffill')
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, group_data


# 2. get universe data
prices, benchmark_prices, group_data = fetch_universe_data()

# 3.a. define optimisation setup
portfolio_objective = PortfolioObjective.MAX_DIVERSIFICATION  # define portfolio objective
weight_mins = np.zeros(len(prices.columns))  # all weights >= 0
weight_maxs = np.ones(len(prices.columns))  # all weights <= 1
rebalancing_freq = 'Q'  # weights rebalancing frequency
returns_freq = None  # use data implied frequency = B
span = 72  # span of number of returns for covariance estimation = 3 months
is_gross_notional_one = True # sum of weights = 1.0
is_long_only = True  # all weights >= 0

# 3.b. compute rolling portfolio weights rebalanced every quarter
weights = compute_rolling_optimal_weights_ewm_covar(prices=prices,
                                                    portfolio_objective=portfolio_objective,
                                                    weight_mins=weight_mins,
                                                    weight_maxs=weight_maxs,
                                                    rebalancing_freq=rebalancing_freq,
                                                    is_gross_notional_one=is_gross_notional_one,
                                                    is_long_only=is_long_only,
                                                    span=span)

# 4. given portfolio weights, construct the performance of the portfolio
funding_rate = None  # on positive / negative cash balances
rebalancing_costs = 0.0010  # rebalancing costs per volume = 10bp
portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                              weights=weights,
                                              is_rebalanced_at_first_date=True,
                                              ticker='MaxDiversification',
                                              funding_rate=funding_rate,
                                              rebalancing_costs=rebalancing_costs,
                                              is_output_portfolio_data=True)


# 5. using portfolio_data run the reporting with strategy factsheet
# for group-based reporting set_group_data
portfolio_data.set_group_data(group_data=group_data, group_order=list(group_data.unique()))
# set time period for portfolio reporting
time_period = qis.TimePeriod('31Dec2005', '17Aug2023')
fig = qis.generate_strategy_factsheet(portfolio_data=portfolio_data,
                                      benchmark_prices=benchmark_prices,
                                      time_period=time_period,
                                      **qis.fetch_default_report_kwargs(time_period=time_period))
# save report to pdf and png
qis.save_figs_to_pdf(figs=[fig],
                     file_name=f"{portfolio_data.nav.name}_portfolio_factsheet",
                     orientation='landscape',
                     local_path="C://Users//Artur//OneDrive//analytics//outputs")
qis.save_fig(fig=fig, file_name=f"example_portfolio_factsheet", local_path=f"figures/")


# 6. can create customised reporting using portfolio_data custom reporting
def run_customised_reporting(portfolio_data) -> plt.Figure:
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), tight_layout=True)
    kwargs = dict(x_date_freq='A', framealpha=0.8)
    portfolio_data.plot_nav(ax=axs[0], **kwargs)
    portfolio_data.plot_weights(ncol=len(prices.columns)//3,
                                legend_stats=qis.LegendStats.AVG_LAST,
                                title='Portfolio weights',
                                bbox_to_anchor=None,
                                ax=axs[1],
                                **kwargs)
    portfolio_data.plot_returns_scatter(benchmark_price=benchmark_prices.iloc[:, 0],
                                        ax=axs[2],
                                        **kwargs)
    return fig


# run customised report
fig = run_customised_reporting(portfolio_data)
# save png
qis.save_fig(fig=fig, file_name=f"example_customised_report", local_path=f"figures/")

plt.show()
