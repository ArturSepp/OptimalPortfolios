"""
Tests for EWMA covariance matrix estimator.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import qis as qis

from optimalportfolios.covar_estimation.ewma_covar_estimator import (
    EwmaCovarEstimator,
    estimate_rolling_ewma_covar,
)


class LocalTests(Enum):
    CURRENT_COVAR = 1
    CURRENT_COVAR_SPAN_SENSITIVITY = 2
    CURRENT_COVAR_VOL_NORM = 3
    ROLLING_COVARS = 4
    ROLLING_VS_STANDALONE = 5


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes."""

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from optimalportfolios.test_data import load_test_data
    prices = load_test_data()
    prices = prices.loc['2000':, :]
    tickers = prices.columns.to_list()

    if local_test == LocalTests.CURRENT_COVAR:
        # basic single-date EWMA covariance
        estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52)
        covar = estimator.fit_current_covar(prices=prices)

        vols = pd.Series(np.sqrt(np.diag(covar.values)), index=covar.columns, name='Vol')
        corr = covar / np.outer(vols, vols)

        print(f"── EWMA covariance (weekly, span=52) ──")
        print(f"\nAnnualised vols:\n{vols.to_string(float_format='{:.2%}'.format)}")
        print(f"\nCorrelation matrix:\n{corr.to_string(float_format='{:.3f}'.format)}")
        print(f"\nCovariance matrix:\n{covar.to_string(float_format='{:.6f}'.format)}")

        # verify PSD
        eigenvalues = np.linalg.eigvalsh(covar.values)
        print(f"\nMin eigenvalue: {eigenvalues.min():.2e} (should be >= 0)")

    elif local_test == LocalTests.CURRENT_COVAR_SPAN_SENSITIVITY:
        # compare vols across different spans and frequencies
        configs = [
            ('Daily, span=60',    'B',     60),
            ('Daily, span=120',   'B',     120),
            ('Weekly, span=26',   'W-WED', 26),
            ('Weekly, span=52',   'W-WED', 52),
            ('Weekly, span=104',  'W-WED', 104),
            ('Monthly, span=12',  'ME',    12),
            ('Monthly, span=36',  'ME',    36),
        ]

        vol_table = {}
        for label, freq, span in configs:
            estimator = EwmaCovarEstimator(returns_freq=freq, span=span)
            covar = estimator.fit_current_covar(prices=prices)
            vol_table[label] = pd.Series(np.sqrt(np.diag(covar.values)), index=covar.columns)

        vol_df = pd.DataFrame(vol_table).T
        print(f"── Vol sensitivity to span and frequency ──")
        print(vol_df.to_string(float_format='{:.2%}'.format))

    elif local_test == LocalTests.CURRENT_COVAR_VOL_NORM:
        # compare plain vs vol-normalised EWMA
        estimator_plain = EwmaCovarEstimator(returns_freq='W-WED', span=52,
                                              is_apply_vol_normalised_returns=False)
        estimator_norm = EwmaCovarEstimator(returns_freq='W-WED', span=52,
                                             is_apply_vol_normalised_returns=True)

        covar_plain = estimator_plain.fit_current_covar(prices=prices)
        covar_norm = estimator_norm.fit_current_covar(prices=prices)

        vols_plain = pd.Series(np.sqrt(np.diag(covar_plain.values)), index=tickers)
        vols_norm = pd.Series(np.sqrt(np.diag(covar_norm.values)), index=tickers)

        corr_plain = covar_plain / np.outer(vols_plain, vols_plain)
        corr_norm = covar_norm / np.outer(vols_norm, vols_norm)

        comparison = pd.concat([vols_plain.rename('Plain'),
                                vols_norm.rename('VolNorm'),
                                (vols_norm / vols_plain).rename('Ratio')], axis=1)

        print(f"── Plain vs Vol-normalised EWMA ──")
        print(f"\nVols:\n{comparison.to_string(float_format='{:.4%}'.format)}")
        print(f"\nCorrelation difference (VolNorm - Plain):")
        print((corr_norm - corr_plain).to_string(float_format='{:.4f}'.format))

    elif local_test == LocalTests.ROLLING_COVARS:
        # rolling estimation with vol time series
        estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
        time_period = qis.TimePeriod('31Dec2004', prices.index[-1])
        covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)

        print(f"── Rolling EWMA: {len(covar_dict)} matrices ──")

        # extract rolling vols
        rolling_vols = {}
        for date, covar in covar_dict.items():
            rolling_vols[date] = pd.Series(np.sqrt(np.diag(covar.values)), index=covar.columns)
        rolling_vols = pd.DataFrame.from_dict(rolling_vols, orient='index')

        print(f"\nRolling vols (last 5 dates):")
        print(rolling_vols.tail().to_string(float_format='{:.2%}'.format))

        print(f"\nAverage vols:")
        print(rolling_vols.mean().to_string(float_format='{:.2%}'.format))

        # plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 5), tight_layout=True)
        qis.plot_time_series(df=rolling_vols,
                             var_format='{:.0%}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             title='Rolling EWMA Vols (weekly, span=52, quarterly rebal)',
                             ax=ax)

    elif local_test == LocalTests.ROLLING_VS_STANDALONE:
        # verify that EwmaCovarEstimator.fit_rolling_covars matches
        # the standalone estimate_rolling_ewma_covar function
        time_period = qis.TimePeriod('31Dec2014', prices.index[-1])
        params = dict(returns_freq='W-WED', span=52)

        # class-based
        estimator = EwmaCovarEstimator(rebalancing_freq='QE', **params)
        covar_dict_class = estimator.fit_rolling_covars(prices=prices, time_period=time_period)

        # standalone function
        covar_dict_func = estimate_rolling_ewma_covar(prices=prices, time_period=time_period,
                                                       rebalancing_freq='QE', **params)

        # compare dates
        dates_class = sorted(covar_dict_class.keys())
        dates_func = sorted(covar_dict_func.keys())
        dates_match = dates_class == dates_func

        # compare matrices at each date
        max_diff = 0.0
        for date in dates_class:
            diff = np.abs(covar_dict_class[date].values - covar_dict_func[date].values).max()
            max_diff = max(max_diff, diff)

        print(f"── Class vs standalone function ──")
        print(f"Dates match:     {dates_match}")
        print(f"Num dates:       {len(dates_class)}")
        print(f"Max abs diff:    {max_diff:.2e}")
        print(f"Match:           {'YES' if max_diff < 1e-12 else 'NO'}")

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.CURRENT_COVAR)