"""
Test script for LASSO / Group LASSO factor model estimator.

Adapted to the new API:
    - LassoModel.fit() populates estimated_betas (DataFrame) and estimation_result_
    - FactorCovarEstimator.fit_current_factor_covars() replaces
    - qis.compute_asset_returns_dict() handles per-asset return frequencies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import yfinance as yf
import qis as qis

from optimalportfolios import LassoModel, LassoModelType
from optimalportfolios import CovarEstimatorType, FactorCovarEstimator


# ─────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────
instrument_data = dict(
    IEFA='Equity', IEMG='Equity', ITOT='Equity', DVY='Equity',
    AGG='Bonds', IUSB='Bonds', GVI='Bonds', GBF='Bonds',
    AOR='Mixed', AOA='Mixed', AOM='Mixed', AOK='Mixed',
)
group_data = pd.Series(instrument_data)

asset_tickers = group_data.index.to_list()
benchmark_tickers = ['SPY', 'IEF', 'LQD', 'USO', 'GLD', 'UUP']

# per-asset return frequencies: equities monthly, bonds monthly, mixed quarterly
returns_freqs = group_data.map({'Equity': 'ME', 'Bonds': 'ME', 'Mixed': 'QE'})

# ─────────────────────────────────────────────
# Download prices
# ─────────────────────────────────────────────
asset_prices = (
    yf.download(asset_tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)
    ['Close'][asset_tickers]
    .asfreq('B', method='ffill')
)
benchmark_prices = (
    yf.download(benchmark_tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)
    ['Close'][benchmark_tickers]
    .reindex(index=asset_prices.index, method='ffill')
)
assets = asset_prices.columns


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def print_beta_and_r2(label: str, model: LassoModel) -> pd.DataFrame:
    """Print estimated betas (filtered) and per-asset R² from a fitted LassoModel."""
    betas = model.estimated_betas.where(np.abs(model.estimated_betas) > 1e-5, other=np.nan)
    r2 = pd.Series(model.estimation_result_.r2, index=model.estimated_betas.index, name='R2')
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(betas.to_string(float_format='{:.4f}'.format))
    print(f"\nR²:\n{r2.to_string(float_format='{:.4f}'.format)}")
    return betas


class LocalTests(Enum):
    LASSO_BETAS = 1
    LASSO_COVAR_DIFFERENT_FREQUENCIES = 2


def run_local_test(local_test: LocalTests):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    lasso_params = dict(
        group_data=group_data,
        reg_lambda=1e-6,
        span=52,
        demean=False,
        solver='ECOS_BB',
    )

    if local_test == LocalTests.LASSO_BETAS:
        # prepare returns (monthly, demeaned externally since demean=False)
        y = qis.to_returns(asset_prices, freq='W-WED', drop_first=True)
        x = qis.to_returns(benchmark_prices, freq='W-WED', drop_first=True)
        y = y - np.nanmean(y, axis=0)
        x = x - np.nanmean(x, axis=0)

        # ── 1. Unrestricted regression (λ ≈ 0) ──
        model_full = LassoModel(
            model_type=LassoModelType.LASSO,
            **qis.update_kwargs(lasso_params, dict(reg_lambda=0.0)),
        ).fit(x=x, y=y)
        betas_full = print_beta_and_r2('Multivariate Regression (λ=0)', model_full)

        # ── 2. Independent LASSO ──
        model_lasso = LassoModel(
            model_type=LassoModelType.LASSO,
            **lasso_params,
        ).fit(x=x, y=y)
        betas_lasso = print_beta_and_r2('Independent LASSO', model_lasso)

        # ── 3. Group LASSO ──
        model_group = LassoModel(
            model_type=LassoModelType.GROUP_LASSO,
            **lasso_params,
        ).fit(x=x, y=y)
        betas_group = print_beta_and_r2('Group LASSO', model_group)

        # ── 4. HCGL ──
        model_hcgl = LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            **lasso_params,
        ).fit(x=x, y=y)
        betas_hcgl = print_beta_and_r2('HCGL (Group LASSO + Clusters)', model_hcgl)

        # ── Heatmap comparison ──
        fig, axs = plt.subplots(4, 1, figsize=(14, 14), tight_layout=True)
        for ax, (title, betas) in zip(axs, [
            ('(A) Multivariate Regression (λ=0)', betas_full),
            ('(B) Independent LASSO', betas_lasso),
            ('(C) Group LASSO', betas_group),
            ('(D) HCGL (Group LASSO + Clusters)', betas_hcgl),
        ]):
            qis.plot_heatmap(df=betas, title=title, var_format='{:.2f}', ax=ax)

    elif local_test == LocalTests.LASSO_COVAR_DIFFERENT_FREQUENCIES:
        # ── Factor covariance estimation at different frequencies ──
        # using FactorCovarEstimator with mixed-frequency asset returns

        lasso_model = LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            **lasso_params,
        )

        # monthly factor returns, monthly factor covar span
        estimator_monthly = FactorCovarEstimator(
            covar_estimator_type=CovarEstimatorType.LASSO,
            lasso_model=lasso_model,
            factor_returns_freq='ME',
            factor_covar_span=12,
        )

        # (a) all assets at monthly frequency
        asset_returns_dict_all_monthly = qis.compute_asset_returns_dict(
            prices=asset_prices, is_log_returns=True, returns_freqs='ME',
        )
        covar_all_monthly = estimator_monthly.fit_current_factor_covars(
            risk_factor_prices=benchmark_prices,
            asset_returns_dict=asset_returns_dict_all_monthly,
            assets=assets,
        )

        # (b) mixed frequencies: equities monthly, bonds monthly, mixed quarterly
        asset_returns_dict_mixed = qis.compute_asset_returns_dict(
            prices=asset_prices, is_log_returns=True, returns_freqs=returns_freqs,
        )
        covar_mixed_freq = estimator_monthly.fit_current_factor_covars(
            risk_factor_prices=benchmark_prices,
            asset_returns_dict=asset_returns_dict_mixed,
            assets=assets,
        )

        # (c) weekly factor returns with weekly and mixed asset returns
        estimator_weekly = FactorCovarEstimator(
            covar_estimator_type=CovarEstimatorType.LASSO,
            lasso_model=lasso_model,
            factor_returns_freq='W-WED',
            factor_covar_span=52,
        )

        asset_returns_dict_all_weekly = qis.compute_asset_returns_dict(
            prices=asset_prices, is_log_returns=True, returns_freqs='W-WED',
        )
        covar_all_weekly = estimator_weekly.fit_current_factor_covars(
            risk_factor_prices=benchmark_prices,
            asset_returns_dict=asset_returns_dict_all_weekly,
            assets=assets,
        )

        returns_freqs_weekly_mixed = group_data.map(
            {'Equity': 'W-WED', 'Bonds': 'W-WED', 'Mixed': 'ME'}
        )
        asset_returns_dict_weekly_mixed = qis.compute_asset_returns_dict(
            prices=asset_prices, is_log_returns=True, returns_freqs=returns_freqs_weekly_mixed,
        )
        covar_weekly_mixed = estimator_weekly.fit_current_factor_covars(
            risk_factor_prices=benchmark_prices,
            asset_returns_dict=asset_returns_dict_weekly_mixed,
            assets=assets,
        )

        # ── Print covariance matrices ──
        print("\n── All Monthly ──")
        print(covar_all_monthly.y_covar)
        print("\n── Mixed Frequency (ME / ME / QE) ──")
        print(covar_mixed_freq.y_covar)
        print("\n── All Weekly ──")
        print(covar_all_weekly.y_covar)
        print("\n── Weekly factors, mixed asset freq ──")
        print(covar_weekly_mixed.y_covar)

        # ── Heatmap comparison ──
        fig, axs = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
        qis.plot_heatmap(df=covar_all_monthly.y_covar, title='All Monthly', ax=axs[0, 0])
        qis.plot_heatmap(df=covar_mixed_freq.y_covar, title='Mixed Freq (ME/ME/QE)', ax=axs[0, 1])
        qis.plot_heatmap(df=covar_all_weekly.y_covar, title='All Weekly', ax=axs[1, 0])
        qis.plot_heatmap(df=covar_weekly_mixed.y_covar, title='Weekly + Mixed Asset', ax=axs[1, 1])

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.LASSO_BETAS)