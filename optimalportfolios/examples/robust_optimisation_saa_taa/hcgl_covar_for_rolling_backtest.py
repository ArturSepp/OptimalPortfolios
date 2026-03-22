"""
Minimal example of using Hierarchical Clustering Group Lasso (HCGL) method
for rolling estimation of covariance matrix and for solving Strategic Asset Allocation
using risk-budgeted optimisation as introduced in:

Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
The Journal of Portfolio Management, 52(4), 86-120.

Uses a universe of ETFs for computing and backtesting of rolling SAA portfolio.
"""
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Tuple
import qis as qis

# package
from optimalportfolios import (Constraints, LassoModelType,
                               LassoModel, FactorCovarEstimator,
                               rolling_risk_budgeting)


# 1. define the investment universe and allocation by asset classes
def fetch_universe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Fetch universe data for portfolio construction:
    1. dividend and split adjusted end of day prices
    2. benchmark prices for portfolio reporting and benchmarking
    3. group data for portfolio reporting and risk attribution
    """
    universe_data = dict(SPY='Equities',
                         EZU='Equities',
                         EEM='Equities',
                         TLT='Bonds',
                         HYG='HighYield',
                         GLD='Gold')
    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)
    prices = yf.download(tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close']
    prices = prices[tickers].loc['2000':, :]
    prices = prices.asfreq('B', method='ffill')
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, group_data


# 1. get universe data
universe_prices, risk_factor_prices, group_data = fetch_universe_data()

# 2. set lasso model
lasso_model = LassoModel(model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
                         reg_lambda=1e-5,
                         span=36,
                         warmup_period=12)

# 3. set covar estimator
covar_estimator = FactorCovarEstimator(lasso_model=lasso_model,
                                       factor_returns_freq='ME',
                                       rebalancing_freq='QE')

# 4. compute asset returns dict at monthly frequency (matching factor_returns_freq)
asset_returns_dict = qis.compute_asset_returns_dict(
    prices=universe_prices, is_log_returns=True, returns_freqs='ME',
)

# 5. set time period for backtest and fit rolling covars
time_period = qis.TimePeriod('31Dec2004', '30Jun2025')
rolling_covar_data = covar_estimator.fit_rolling_factor_covars(
    risk_factor_prices=risk_factor_prices,
    asset_returns_dict=asset_returns_dict,
    time_period=time_period,
)

# 6. set equal risk-budgets and compute rolling weights
risk_budget = pd.Series({asset: 1.0 / len(universe_prices.columns) for asset in universe_prices.columns})
saa_rolling_weights = rolling_risk_budgeting(prices=universe_prices,
                                             covar_dict=rolling_covar_data.get_y_covars(),
                                             risk_budget=risk_budget,
                                             constraints=Constraints(is_long_only=True))

# 7. run backtest
saa_portfolio_data = qis.backtest_model_portfolio(prices=universe_prices,
                                                  weights=saa_rolling_weights,
                                                  ticker='Risk Budget SAA',
                                                  weight_implementation_lag=1,
                                                  rebalancing_costs=0.0010)

# 8. compute equal weight benchmark
benchmark_portfolio_data = qis.backtest_model_portfolio(prices=universe_prices,
                                                        weights=risk_budget,
                                                        ticker='Equal Weights SAA',
                                                        weight_implementation_lag=1,
                                                        rebalancing_costs=0.0010)

# 9. generate factsheet
multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas=[saa_portfolio_data, benchmark_portfolio_data],
                                              benchmark_prices=risk_factor_prices,
                                              covar_dict=rolling_covar_data.get_y_covars())
[x.set_group_data(group_data=group_data) for x in multi_portfolio_data.portfolio_datas]
figs = qis.generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=multi_portfolio_data,
                                                     add_strategy_factsheet=True,
                                                     add_brinson_attribution=True,
                                                     time_period=time_period,
                                                     **qis.fetch_default_report_kwargs(reporting_frequency=qis.ReportingFrequency.MONTHLY,
                                                                                       add_rates_data=False))

# 10. save report
qis.save_figs_to_pdf(figs=figs,
                     file_name=f"saa_risk_budget_portfolio_factsheet",
                     orientation='landscape',
                     local_path="C://Users//Artur//OneDrive//analytics//outputs")

plt.show()