"""
minimal example of using Hierarchical Clustering Group Lasso (HCGL) method
for rolling estimation of covariance matrix and for solving Strategic Asset Allocation
using risk-budgeted optimisation
Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios
"""
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Tuple
import qis as qis

# package
from optimalportfolios import (Constraints, LassoModelType,
                               LassoModel, CovarEstimator, CovarEstimatorType,
                               rolling_risk_budgeting)


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
                         EZU='Equities',
                         EEM='Equities',
                         TLT='Bonds',
                         HYG='HighYield',
                         GLD='Gold')
    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)
    prices = yf.download(tickers, start=None, end=None, ignore_tz=True)['Close']
    prices = prices[tickers].loc['2000':, :]  # arrange as given
    prices = prices.asfreq('B', method='ffill')  # refill at B frequency
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, group_data


# 1. get universe data including prices, risk factor prices and groups
universe_prices, risk_factor_prices, group_data = fetch_universe_data()
# 2. set lasso model
lasso_model = LassoModel(model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
                         reg_lambda=1e-5,  # lambda
                         span=36,  # ewma span in months
                         warmup_period=12)  # at least 12 months of returns to estimate first beta
# 3. set covar estimator
covar_estimator = CovarEstimator(covar_estimator_type=CovarEstimatorType.LASSO,
                                 lasso_model=lasso_model,
                                 factor_returns_freq='ME',  # factor returns
                                 rebalancing_freq='QE',  # saa rebalancing
                                 returns_freqs='ME',  # instrument return frequency
                                 span=lasso_model.span)  # for ewma of factors
# 4. set time period for backtest and fit rolling covars
time_period = qis.TimePeriod('31Dec2004', '30Jun2025')   # period for computing weights backtest
rolling_covar_data = covar_estimator.fit_rolling_covars(prices=universe_prices,
                                                        risk_factor_prices=risk_factor_prices,
                                                        time_period=time_period)
# 5. set equal risk-budgets and compute rolling weights of risk budgets
risk_budget = {asset: 1.0 / len(universe_prices.columns) for asset in universe_prices.columns}
saa_rolling_weights = rolling_risk_budgeting(prices=universe_prices,
                                             time_period=time_period,
                                             covar_dict=rolling_covar_data.y_covars,
                                             risk_budget=risk_budget,
                                             constraints0=Constraints(is_long_only=True))  # trivial constraints
# 6. run backtest using portfolio weights using qis package
saa_portfolio_data = qis.backtest_model_portfolio(prices=universe_prices,
                                                  weights=saa_rolling_weights,
                                                  ticker='Risk Budget SAA',
                                                  weight_implementation_lag=1,  # next day after weights computation
                                                  rebalancing_costs=0.0010)  # rebalancing costs per volume = 10bp
# 7. compute equal weight portfolio as benchmark
benchmark_portfolio_data = qis.backtest_model_portfolio(prices=universe_prices,
                                                        weights=risk_budget,
                                                        ticker='Equal Weights SAA',
                                                        weight_implementation_lag=1,  # next day after weights computation
                                                        rebalancing_costs=0.0010)  # rebalancing costs per volume = 10bp
# 8. generate backtest reporting with strategy benchmark factsheet
multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas=[saa_portfolio_data, benchmark_portfolio_data],
                                              benchmark_prices=risk_factor_prices,
                                              covar_dict=rolling_covar_data.y_covars)
# set groups for backtest reporting
[x.set_group_data(group_data=group_data) for x in multi_portfolio_data.portfolio_datas]
figs = qis.generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=multi_portfolio_data,
                                                     add_strategy_factsheet=True,
                                                     add_brinson_attribution=True,
                                                     time_period=time_period,
                                                     **qis.fetch_default_report_kwargs(reporting_frequency=qis.ReportingFrequency.MONTHLY,
                                                                                       add_rates_data=False))
# 9. save report to pdf and png
qis.save_figs_to_pdf(figs=figs,
                     file_name=f"saa_risk_budget_portfolio_factsheet",
                     orientation='landscape',
                     local_path="C://Users//Artur//OneDrive//analytics//outputs")

plt.show()
