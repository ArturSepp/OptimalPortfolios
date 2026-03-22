"""
illustrate estimation of covar at different frequencies
"""

# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import qis as qis

from optimalportfolios import LassoModel, LassoModelType
from optimalportfolios import FactorCovarEstimator
from optimalportfolios.examples.covar_estimation.simulate_factor_returns import simulate_factor_model_returns


simulation_results = simulate_factor_model_returns(n_assets=9, n_periods=20*260, seed=40)

asset_prices = qis.returns_to_nav(returns=simulation_results['asset_returns'])
risk_factor_prices = qis.returns_to_nav(returns=simulation_results['factor_returns'])
assets = asset_prices.columns

# select multi asset ETFs
instrument_data = dict(Asset_1='Equity',
                       Asset_2='Equity',
                       Asset_3='Equity',
                       Asset_4='Bonds',
                       Asset_5='Bonds',
                       Asset_6='Bonds',
                       Asset_7='Mixed',
                       Asset_8='Mixed',
                       Asset_9='Mixed')

group_data = pd.Series(instrument_data)
ac_group_order = ['Equity', 'Bonds', 'Bonds']

# set lasso model
lasso_params = dict(reg_lambda=1e-5, span=120, demean=False, solver='CLARABEL', warmup_period=50)
lasso_model = LassoModel(model_type=LassoModelType.GROUP_LASSO_CLUSTERS, **lasso_params)


estimator_daily = FactorCovarEstimator(lasso_model=LassoModel(),
                                 factor_returns_freq='B',
                                 factor_covar_span=252)

estimator_weekly = FactorCovarEstimator(lasso_model=LassoModel(),
                                        factor_returns_freq='W-WED',
                                        factor_covar_span=52)

estimator_monthly = FactorCovarEstimator(lasso_model=LassoModel(),
                                        factor_returns_freq='ME',
                                        factor_covar_span=12)

covar_data_all_daily = estimator_daily.fit_current_factor_covars(
    risk_factor_prices=risk_factor_prices,
    asset_returns_dict=qis.compute_asset_returns_dict(prices=asset_prices, is_log_returns=True, returns_freqs='B'),
    assets=assets)

covar_data_factor_daily_asset_mixed = estimator_daily.fit_current_factor_covars(
    risk_factor_prices=risk_factor_prices,
    asset_returns_dict=qis.compute_asset_returns_dict(prices=asset_prices, is_log_returns=True,
                                                      returns_freqs=group_data.map({'Equity': 'B', 'Bonds': 'W-WED', 'Mixed': 'ME'})),
    assets=assets)


covar_data_factor_weekly_all = estimator_weekly.fit_current_factor_covars(
    risk_factor_prices=risk_factor_prices,
    asset_returns_dict=qis.compute_asset_returns_dict(prices=asset_prices, is_log_returns=True, returns_freqs='W-WED'),
    assets=assets)

covar_data_factor_weekly_asset_mixed = estimator_weekly.fit_current_factor_covars(
    risk_factor_prices=risk_factor_prices,
    asset_returns_dict=qis.compute_asset_returns_dict(prices=asset_prices, is_log_returns=True,
                                                      returns_freqs=group_data.map({'Equity': 'W-WED', 'Bonds': 'W-WED', 'Mixed': 'ME'})),
    assets=assets)

covar_data_factor_monthy_all = estimator_monthly.fit_current_factor_covars(
    risk_factor_prices=risk_factor_prices,
    asset_returns_dict=qis.compute_asset_returns_dict(prices=asset_prices, is_log_returns=True, returns_freqs='ME'),
    assets=assets)

covar_data_factor_monthy_asset_mixed = estimator_monthly.fit_current_factor_covars(
    risk_factor_prices=risk_factor_prices,
    asset_returns_dict=qis.compute_asset_returns_dict(prices=asset_prices, is_log_returns=True,
                                                      returns_freqs=group_data.map({'Equity': 'W-WED', 'Bonds': 'W-WED', 'Mixed': 'ME'})),
    assets=assets)

fig, axs = plt.subplots(2, 4, figsize=(14, 12), constrained_layout=True)
theoretical_asset_covar = pd.DataFrame(260*simulation_results['theoretical_asset_covar'],
                                       index=asset_prices.columns,
                                       columns=asset_prices.columns)
sample_covar = pd.DataFrame(260*np.cov(simulation_results['asset_returns'], rowvar=False, bias=True),
                            index=asset_prices.columns,
                            columns=asset_prices.columns)

qis.plot_heatmap(df=theoretical_asset_covar, title='theoretical_covar', ax=axs[0, 0])
qis.plot_heatmap(df=sample_covar, title='sample_covar', ax=axs[1, 0])
qis.plot_heatmap(df=covar_data_all_daily.y_covar, title='all_daily', ax=axs[0, 1])
qis.plot_heatmap(df=covar_data_factor_daily_asset_mixed.y_covar, title='factor_daily_asset_mixed', ax=axs[1, 1])
qis.plot_heatmap(df=covar_data_factor_weekly_all.y_covar, title='weekly_all', ax=axs[0, 2])
qis.plot_heatmap(df=covar_data_factor_weekly_asset_mixed.y_covar, title='factor_weekly_asset_mixed', ax=axs[1, 2])
qis.plot_heatmap(df=covar_data_factor_monthy_all.y_covar, title='monthy_all', ax=axs[0, 3])
qis.plot_heatmap(df=covar_data_factor_monthy_asset_mixed.y_covar, title='factor_monthy_asset_mixed', ax=axs[1, 3])

plt.show()
