# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import qis as qis

import optimalportfolios.local_path
from optimalportfolios import (LassoModel, LassoModelType,
                               Constraints,
                               rolling_risk_budgeting,
                               estimate_rolling_lasso_covar_different_freq)

# select multi asset ETFs
instrument_data = dict(IEFA='Equity',
                       IEMG='Equity',
                       DVY='Equity',
                       AGG='Bonds',
                       IUSB='Bonds',
                       GVI='Bonds',
                       AOR='Mixed',  # growth
                       AOA='Mixed',  # aggressive
                       AOM='Mixed')  # moderate

group_data = pd.Series(instrument_data)
ac_group_order = ['Equity', 'Bonds', 'Bonds']
# select different observation periods
sampling_freqs = group_data.map({'Equity': 'ME', 'Bonds': 'ME', 'Mixed': 'QE'})

# set
asset_tickers = group_data.index.to_list()
risk_factor_tickers = ['SPY', 'IEF', 'LQD', 'USO', 'GLD', 'UUP']

prices = yf.download(asset_tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][asset_tickers].asfreq('B', method='ffill')
risk_factor_prices = yf.download(risk_factor_tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][risk_factor_tickers].reindex(
    index=prices.index, method='ffill')


# estimate asset betas and covar matrix
time_period=qis.TimePeriod('31Dec2015', '13Dec2024')
lasso_params = dict(reg_lambda=1e-5, span=120, demean=False, solver='ECOS_BB')
lasso_model = LassoModel(model_type=LassoModelType.GROUP_LASSO_CLUSTERS, **lasso_params)
covar_data = estimate_rolling_lasso_covar_different_freq(risk_factor_prices=risk_factor_prices,
                                                         prices=prices,
                                                         returns_freqs=sampling_freqs,
                                                         time_period=time_period,
                                                         factor_returns_freq='W-WED',
                                                         rebalancing_freq='QE',
                                                         lasso_model=lasso_model
                                                         )
# print betas
for date, beta in covar_data.asset_last_betas_t.items():
    print(date)
    print(beta)

# build equal risk budget portfolio
risk_budget = pd.Series(1.0 / len(prices.columns), index=prices.columns)
risk_budget_weights = rolling_risk_budgeting(prices=prices,
                                             constraints0=Constraints(),
                                             time_period=time_period,
                                             covar_dict=covar_data.y_covars,
                                             risk_budget=risk_budget)
erb_portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                  weights=risk_budget_weights,
                                                  weight_implementation_lag=1,
                                                  ticker='EqualRisk')

# build simple equal weight portfolio
ew_portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                 weights=risk_budget,
                                                 weight_implementation_lag=1,
                                                 ticker='EqualWeight')

multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas=[erb_portfolio_data, ew_portfolio_data],
                                              covar_dict=covar_data.y_covars,
                                              benchmark_prices=risk_factor_prices['SPY'].to_frame())


# get linear model
risk_model = covar_data.get_linear_factor_model(x_factors=risk_factor_prices, y_assets=prices)

portfolio_factor_betas = risk_model.compute_agg_factor_exposures(weights=erb_portfolio_data.get_weights())

# portfolio returns at portfolio_factor_betas.index
portfolio_returns = qis.to_returns(prices=erb_portfolio_data.get_portfolio_nav().reindex(index=portfolio_factor_betas.index).ffill(), is_first_zero=True)
attributions = qis.compute_benchmarks_beta_attribution_from_returns(portfolio_returns=portfolio_returns,
                                                                    benchmark_returns=risk_model.x,
                                                                    portfolio_benchmark_betas=portfolio_factor_betas,
                                                                    total_name='Total')

# factor risk
factor_rcs_ratios, factor_risk_contrib_idio, factor_risk_contrib, portfolio_var \
    = risk_model.compute_factor_risk_contribution(weights=erb_portfolio_data.get_weights())

with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), tight_layout=True)
    qis.plot_time_series(df=portfolio_factor_betas,
                         title=f"Portfolio Factor Beta Exposures",
                         var_format='{:,.2f}',
                         ax=axs[0])
    qis.plot_time_series(df=attributions.cumsum(axis=0),
                         title=f"Portfolio Factor Attribution",
                         var_format='{:,.2%}',
                         ax=axs[1])
    qis.plot_time_series(df=factor_risk_contrib,
                         title=f"Portfolio Factor Risk Attribution",
                         var_format='{:,.2%}',
                         ax=axs[2])

    portfolio_var['total'] = np.sum(portfolio_var, axis=1)
    portfolio_var = np.sqrt(portfolio_var)
    print(portfolio_var)

    qis.plot_time_series(df=portfolio_var,
                         title=f"Portfolio sqrt(Vars)",
                         var_format='{:,.2%}',
                         ax=axs[3])

    fig, ax = plt.subplots(1, 1, figsize=(12, 12), tight_layout=True)
    qis.plot_stack(df=factor_rcs_ratios,
                   use_bar_plot=True,
                   title=f"Relative",
                   var_format='{:,.2%}',
                   ax=ax)

"""
# comprehensive report
report_kwargs = qis.fetch_default_report_kwargs(time_period=time_period,
                                                reporting_frequency=qis.ReportingFrequency.MONTHLY,
                                                add_rates_data=False)
figs, dfs = qis.weights_tracking_error_report_by_ac_subac(multi_portfolio_data=multi_portfolio_data,
                                                          time_period=time_period,
                                                          ac_group_data=group_data,
                                                          ac_group_order=ac_group_order,
                                                          sub_ac_group_data=pd.Series(group_data.index, index=group_data.index),
                                                          sub_ac_group_order=None,
                                                          turnover_groups=group_data,
                                                          turnover_order=ac_group_order,
                                                          risk_model=risk_model,
                                                          **report_kwargs)
qis.save_figs_to_pdf(figs, file_name='lasso_risk_report', local_path=optimalportfolios.local_path.get_output_path())
"""
plt.show()
