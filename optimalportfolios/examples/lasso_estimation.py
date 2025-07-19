# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso

import yfinance as yf
import qis as qis

from optimalportfolios.lasso.lasso_model_estimator import (solve_lasso_cvx_problem, solve_group_lasso_cvx_problem)

# select multi asset ETFs
instrument_data = dict(SPY='SPY',
                       IEFA='Equity',
                       IEMG='Equity',
                       ITOT='Equity',
                       DVY='Equity',
                       AGG='Bonds',
                       IUSB='Bonds',
                       GVI='Bonds',
                       GBF='Bonds',
                       AOR='Mixed',   # growth
                       AOA='Mixed',   # aggressive
                       AOM='Mixed',  # moderate
                       AOK='Mixed',  # conservatives
                       GSG='Commodts',
                       COMT='Commodts',
                       PDBC='Commodts',
                       FTGC='Commodts')
instrument_data = pd.Series(instrument_data)
asset_tickers = instrument_data.index.to_list()
benchmark_tickers = ['SPY', 'IEF', 'LQD', 'USO', 'GLD', 'UUP']
asset_group_loadings = qis.set_group_loadings(group_data=instrument_data)
print(asset_group_loadings)

asset_prices = yf.download(asset_tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][asset_tickers].asfreq('B', method='ffill')
benchmark_prices = yf.download(benchmark_tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][benchmark_tickers].reindex(index=asset_prices.index, method='ffill')

y = qis.to_returns(asset_prices, freq='ME', drop_first=True)
x = qis.to_returns(benchmark_prices, freq='ME', drop_first=True)
y = y.to_numpy() - np.nanmean(y, axis=0)
x = x.to_numpy() - np.nanmean(x, axis=0)

params = dict(reg_lambda=1e-5, span=24, nonneg=False)


#beta2 = solve_lasso_problem_2d(x=x, y=y, **params, apply_independent_nan_filter=True)
#beta2 = pd.DataFrame(beta2, index=benchmark_tickers, columns=asset_tickers)
#beta2 = beta2.where(np.abs(beta2) > 1e-4, other=np.nan)

beta3, _, _ = solve_lasso_cvx_problem(x=x, y=y, **params, apply_independent_nan_filter=False)
beta3 = pd.DataFrame(beta3, index=benchmark_tickers, columns=asset_tickers)
beta3 = beta3.where(np.abs(beta3) > 1e-4, other=np.nan)
print(beta3)

beta4 = solve_group_lasso_cvx_problem(x=x, y=y, group_loadings=asset_group_loadings.to_numpy(), **params)
beta4 = pd.DataFrame(beta4, index=benchmark_tickers, columns=asset_tickers)
beta4 = beta4.where(np.abs(beta4) > 1e-4, other=np.nan)
print(beta4)

model = MultiTaskLasso(alpha=1e-3, fit_intercept=False)

x, y = qis.select_non_nan_x_y(x=x, y=y)
model.fit(X=x, y=y)
params = pd.DataFrame(model.coef_.T, index=benchmark_tickers, columns=asset_tickers)
params = params.where(np.abs(params) > 1e-4, other=np.nan)
print(params)

fig, axs = plt.subplots(3, 1, figsize=(12, 8), tight_layout=True)
qis.plot_heatmap(df=beta3, title='independent betas same nonnan basis', var_format='{:.2f}', ax=axs[0])
qis.plot_heatmap(df=beta4, title='group betas same nonnan basis', var_format='{:.2f}', ax=axs[1])
qis.plot_heatmap(df=params, title='multi Lasso', var_format='{:.2f}', ax=axs[2])


plt.show()




