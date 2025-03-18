"""
create 60/40 portfolio with static weights
run 90/10 risk budget portfolio
show weights/risk contributions for both
"""
import qis as qis
import matplotlib.pyplot as plt
import yfinance as yf
from optimalportfolios import estimate_rolling_ewma_covar, rolling_risk_budgeting, Constraints
from optimalportfolios import local_path as lp
from qis.portfolio.reports.strategy_benchmark_factsheet import weights_tracking_error_report_by_ac_subac

# specify rebalancings
rebalancing_freq = 'QE'  # for portfolio rebalancing
returns_freq = 'ME'  # for covariance computation
span = 104
time_period = qis.TimePeriod('31Dec2004', '07Mar2025')  # for portfolio computations

static_portfolio_weights = {'SPY': 0.6, 'IEF': 0.4}
risk_budgets = {'SPY': 0.98, 'IEF': 0.020}
prices = yf.download(tickers=list(static_portfolio_weights.keys()), start=None, end=None, ignore_tz=True)['Close']
prices = prices[list(static_portfolio_weights.keys())].asfreq('ME', method='ffill')

# static portfolio rebalanced quarterly
balanced_60_40 = qis.backtest_model_portfolio(prices=prices, weights=static_portfolio_weights, rebalancing_freq='QE',
                                              ticker='60/40 static portfolio')

# compute covar matrix using 1y span
covar_dict = estimate_rolling_ewma_covar(prices=prices, time_period=time_period,
                                          rebalancing_freq=rebalancing_freq,
                                          returns_freq=returns_freq,
                                          span=span)
# portfolio with equal risk contribution
risk_budget_weights = rolling_risk_budgeting(prices=prices,
                                             time_period=time_period,
                                             covar_dict=covar_dict,
                                             risk_budget=risk_budgets,
                                             constraints0=Constraints(is_long_only=True))
risk_budget_portfolio = qis.backtest_model_portfolio(prices=prices, weights=risk_budget_weights,
                                                     ticker='Risk-budgeted portfolio')

multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas=[risk_budget_portfolio, balanced_60_40],
                                              benchmark_prices=prices.iloc[:, 0],
                                              covar_dict=covar_dict)

report_kwargs = qis.fetch_default_report_kwargs(time_period=time_period,
                                                reporting_frequency=qis.ReportingFrequency.MONTHLY,
                                                add_rates_data=False)

figs1 = qis.generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=multi_portfolio_data,
                                                      strategy_idx=0,
                                                      benchmark_idx=1,
                                                      add_benchmarks_to_navs=False,
                                                      add_exposures_comp=True,
                                                      add_strategy_factsheet=True,
                                                      time_period=time_period,
                                                      **report_kwargs)
qis.save_figs_to_pdf(figs1, file_name='risk_portfolio', local_path=lp.get_output_path())

figs2, dfs = weights_tracking_error_report_by_ac_subac(multi_portfolio_data=multi_portfolio_data, time_period=time_period,
                                                       **report_kwargs)

qis.save_figs_to_pdf(figs2, file_name='risk_portfolio2', local_path=lp.get_output_path())

all_navs = multi_portfolio_data.get_navs(add_benchmarks_to_navs=True)

fig = qis.generate_multi_asset_factsheet(prices=all_navs, benchmark='SPY', time_period=time_period,
                                         **report_kwargs)
qis.save_figs_to_pdf([fig], file_name='risk_portfolio3', local_path=lp.get_output_path())

plt.close('all')

