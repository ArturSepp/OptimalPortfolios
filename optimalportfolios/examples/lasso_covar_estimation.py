# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso
from enum import Enum

import yfinance as yf
import qis as qis

from optimalportfolios import (LassoModel, LassoModelType,
                               estimate_lasso_covar,
                               estimate_rolling_lasso_covar_different_freq)

# select multi asset ETFs
instrument_data = dict(IEFA='Equity',
                       IEMG='Equity',
                       ITOT='Equity',
                       DVY='Equity',
                       AGG='Bonds',
                       IUSB='Bonds',
                       GVI='Bonds',
                       GBF='Bonds',
                       AOR='Mixed',  # growth
                       AOA='Mixed',  # aggressive
                       AOM='Mixed',  # moderate
                       AOK='Mixed')
group_data = pd.Series(instrument_data)
sampling_freqs = group_data.map({'Equity': 'ME', 'Bonds': 'ME', 'Mixed': 'QE'})

asset_tickers = group_data.index.to_list()
benchmark_tickers = ['SPY', 'IEF', 'LQD', 'USO', 'GLD', 'UUP']
asset_group_loadings = qis.set_group_loadings(group_data=group_data)
print(asset_group_loadings)

asset_prices = yf.download(asset_tickers, start=None, end=None)['Close'][asset_tickers].asfreq('B', method='ffill')
benchmark_prices = yf.download(benchmark_tickers, start=None, end=None)['Close'][benchmark_tickers].reindex(
    index=asset_prices.index, method='ffill')


class UnitTests(Enum):
    LASSO_BETAS = 1
    LASSO_COVAR = 2
    LASSO_COVAR_DIFFERENT_FREQUENCIES = 3


def run_unit_test(unit_test: UnitTests):

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # set lasso model, x and y are demeaned
    lasso_params = dict(group_data=group_data, reg_lambda=1e-5, span=120, demean=False, solver='ECOS_BB')

    # set x and y
    y = qis.to_returns(asset_prices, freq='ME', drop_first=True)
    x = qis.to_returns(benchmark_prices, freq='ME', drop_first=True)
    # demean
    y = y - np.nanmean(y, axis=0)
    x = x - np.nanmean(x, axis=0)

    if unit_test == UnitTests.LASSO_BETAS:
        # full regression
        lasso_model_full = LassoModel(model_type=LassoModelType.LASSO, **qis.update_kwargs(lasso_params, dict(reg_lambda=0.0)))
        betas0, total_vars, residual_vars, r2_t = lasso_model_full.fit(x=x, y=y).compute_residual_alpha_r2()
        betas0 = betas0.where(np.abs(betas0) > 1e-5, other=np.nan)

        # independent Lasso
        lasso_model = LassoModel(model_type=LassoModelType.LASSO, **lasso_params)
        betas_lasso, total_vars, residual_vars, r2_t = lasso_model.fit(x=x, y=y).compute_residual_alpha_r2()
        betas_lasso = betas_lasso.where(np.abs(betas_lasso) > 1e-5, other=np.nan)

        # group Lasso
        group_lasso_model = LassoModel(model_type=LassoModelType.GROUP_LASSO, **lasso_params)
        betas_group_lasso, total_vars, residual_vars, r2_t = group_lasso_model.fit(x=x, y=y).compute_residual_alpha_r2()
        betas_group_lasso = betas_group_lasso.where(np.abs(betas_group_lasso) > 1e-5, other=np.nan)

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), tight_layout=True)
        qis.plot_heatmap(df=betas0, title='(A) Multivariate Regression Betas', var_format='{:.2f}', ax=axs[0])
        qis.plot_heatmap(df=betas_lasso, title='(A) Independent Lasso Betas', var_format='{:.2f}', ax=axs[1])
        qis.plot_heatmap(df=betas_group_lasso, title='(B) Group Lasso Betas', var_format='{:.2f}', ax=axs[2])

    elif unit_test == UnitTests.LASSO_COVAR:
        # ewma covar
        covar_tensor_txy = qis.compute_ewm_covar_tensor(a=y.to_numpy(), span=lasso_params['span'], nan_backfill=qis.NanBackfill.ZERO_FILL)
        covar0 = pd.DataFrame(covar_tensor_txy[-1], index=y.columns, columns=y.columns)

        lasso_model_full = LassoModel(model_type=LassoModelType.LASSO, **qis.update_kwargs(lasso_params, dict(reg_lambda=0.0)))
        covar0 = estimate_lasso_covar(x=x, y=y, lasso_model=lasso_model_full)

        # independent Lasso
        lasso_model = LassoModel(model_type=LassoModelType.LASSO, **lasso_params)
        covar_lasso = estimate_lasso_covar(x=x, y=y, lasso_model=lasso_model)

        # group Lasso
        group_lasso_model = LassoModel(model_type=LassoModelType.GROUP_LASSO, **lasso_params)
        covar_group_lasso = estimate_lasso_covar(x=x, y=y, lasso_model=group_lasso_model)

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), tight_layout=True)
        an = 12.0
        qis.plot_heatmap(df=an*covar0, title='(A) Ewma Covar', var_format='{:.3f}', ax=axs[0])
        qis.plot_heatmap(df=an*covar_lasso, title='(A) Independent Lasso Covar', var_format='{:.3f}', ax=axs[1])
        qis.plot_heatmap(df=an*covar_group_lasso, title='(B) Group Lasso Covar', var_format='{:.3f}', ax=axs[2])

    elif unit_test == UnitTests.LASSO_COVAR_DIFFERENT_FREQUENCIES:
        lasso_model = LassoModel(model_type=LassoModelType.GROUP_LASSO, **lasso_params)
        y_covars = estimate_rolling_lasso_covar_different_freq(risk_factor_prices=benchmark_prices,
                                                               prices=asset_prices,
                                                               rebalancing_freqs=sampling_freqs,
                                                               time_period=qis.TimePeriod('31Dec2019', '13Dec2024'),
                                                               rebalancing_freq='ME',
                                                               lasso_model=lasso_model,
                                                               is_apply_vol_normalised_returns=False
                                                               ).y_covars
        for date, covar in y_covars.items():
            print(date)
            print(covar)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.LASSO_COVAR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
