"""
Tests for individual alpha signal functions (optimalportfolios.alphas.signals).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import qis as qis

from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.low_beta import compute_low_beta_alpha
from optimalportfolios.alphas.signals.managers_alpha import compute_managers_alpha
from optimalportfolios import LassoModel, LassoModelType, FactorCovarEstimator


class LocalTests(Enum):
    MOMENTUM_SINGLE_FREQ = 1
    MOMENTUM_MIXED_FREQ = 2
    MOMENTUM_GROUPED = 3
    LOW_BETA_SINGLE_FREQ = 4
    LOW_BETA_MIXED_FREQ = 5
    MANAGERS_ALPHA = 6


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes."""

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from optimalportfolios.test_data import load_test_data
    prices = load_test_data()
    prices = prices.loc['2005':, :]
    tickers = prices.columns.to_list()
    benchmark_price = prices.iloc[:, 0]

    if local_test == LocalTests.MOMENTUM_SINGLE_FREQ:
        score, raw = compute_momentum_alpha(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq='ME', long_span=12)

        print(f"── Momentum (monthly, span=12) ──")
        print(f"\nScores (last 5):\n{score.tail().to_string(float_format='{:.3f}'.format)}")
        print(f"\nRaw momentum (last 5):\n{raw.tail().to_string(float_format='{:.4f}'.format)}")

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
        qis.plot_time_series(df=score, title='Momentum Scores',
                             var_format='{:.1f}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST, ax=axs[0])
        qis.plot_time_series(df=raw, title='Raw Momentum',
                             var_format='{:.2f}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST, ax=axs[1])

    elif local_test == LocalTests.MOMENTUM_MIXED_FREQ:
        mid = len(tickers) // 2
        returns_freq = pd.Series(
            ['ME'] * mid + ['QE'] * (len(tickers) - mid), index=tickers)

        score, raw = compute_momentum_alpha(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, long_span=12)

        print(f"── Momentum mixed freq ──")
        print(f"Frequencies: {returns_freq.to_dict()}")
        print(f"\nScores (last 5):\n{score.tail().to_string(float_format='{:.3f}'.format)}")

    elif local_test == LocalTests.MOMENTUM_GROUPED:
        n = len(tickers)
        group_data = pd.Series(
            ['GroupA'] * (n // 2) + ['GroupB'] * (n - n // 2), index=tickers)

        score_grouped, _ = compute_momentum_alpha(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq='ME', group_data=group_data, long_span=12)
        score_global, _ = compute_momentum_alpha(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq='ME', group_data=None, long_span=12)

        print(f"── Grouped vs Global scoring ──")
        print(f"Groups: {group_data.to_dict()}")
        print(f"\nGrouped (last 3):\n{score_grouped.tail(3).to_string(float_format='{:.3f}'.format)}")
        print(f"\nGlobal (last 3):\n{score_global.tail(3).to_string(float_format='{:.3f}'.format)}")
        print(f"\nDifference:\n{(score_grouped - score_global).tail(3).to_string(float_format='{:.3f}'.format)}")

    elif local_test == LocalTests.LOW_BETA_SINGLE_FREQ:
        score, raw_beta = compute_low_beta_alpha(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq='ME', beta_span=12)

        print(f"── Low Beta (monthly, span=12) ──")
        print(f"\nScores (last 5):\n{score.tail().to_string(float_format='{:.3f}'.format)}")
        print(f"\nRaw betas (last 5):\n{raw_beta.tail().to_string(float_format='{:.3f}'.format)}")

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
        qis.plot_time_series(df=score, title='Low Beta Scores',
                             var_format='{:.1f}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST, ax=axs[0])
        qis.plot_time_series(df=raw_beta, title='Raw EWMA Beta',
                             var_format='{:.2f}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST, ax=axs[1])

    elif local_test == LocalTests.LOW_BETA_MIXED_FREQ:
        mid = len(tickers) // 2
        returns_freq = pd.Series(
            ['ME'] * mid + ['QE'] * (len(tickers) - mid), index=tickers)

        score, raw_beta = compute_low_beta_alpha(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, beta_span=12)

        print(f"── Low Beta mixed freq ──")
        print(f"\nScores (last 5):\n{score.tail().to_string(float_format='{:.3f}'.format)}")

    elif local_test == LocalTests.MANAGERS_ALPHA:
        risk_factor_prices = prices.iloc[:, :2]
        asset_prices = prices.iloc[:, 2:]

        lasso_model = LassoModel(
            model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
            reg_lambda=1e-5, span=36, warmup_period=12)
        estimator = FactorCovarEstimator(
            lasso_model=lasso_model,
            factor_returns_freq='ME', factor_covar_span=12, rebalancing_freq='QE')

        asset_returns_dict = {'ME': qis.to_returns(asset_prices, freq='ME', is_log_returns=True)}
        time_period = qis.TimePeriod('31Dec2009', asset_prices.index[-1])
        rolling_data = estimator.fit_rolling_factor_covars(
            risk_factor_prices=risk_factor_prices,
            asset_returns_dict=asset_returns_dict,
            time_period=time_period)

        score, raw_alpha = compute_managers_alpha(
            prices=asset_prices,
            risk_factor_prices=risk_factor_prices,
            estimated_betas=rolling_data.get_y_betas(),
            returns_freq='ME', alpha_span=12)

        print(f"── Managers Alpha (monthly, span=12) ──")
        print(f"\nScores (last 5):\n{score.tail().to_string(float_format='{:.3f}'.format)}")
        print(f"\nRaw alpha (last 5):\n{raw_alpha.tail().to_string(float_format='{:.4f}'.format)}")

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
        qis.plot_time_series(df=score, title='Managers Alpha Scores',
                             var_format='{:.1f}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST, ax=axs[0])
        qis.plot_time_series(df=raw_alpha, title='Raw Managers Alpha (annualised)',
                             var_format='{:.2%}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST, ax=axs[1])

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.MOMENTUM_SINGLE_FREQ)