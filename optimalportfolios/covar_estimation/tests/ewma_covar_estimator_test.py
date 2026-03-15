"""
Tests for EwmaCovarEstimator.

Two test cases verifying:
1. Internal consistency: fit_current_covar matches the last matrix from fit_rolling_covars
2. Rolling output properties: PSD, shape, annualised vol range, rebalancing schedule
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import qis as qis

from optimalportfolios.covar_estimation.ewma_covar_estimator import EwmaCovarEstimator


class LocalTests(Enum):
    CURRENT_VS_ROLLING_LAST = 1
    ROLLING_COVAR_PROPERTIES = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes."""

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from optimalportfolios.test_data import load_test_data
    prices = load_test_data()
    prices = prices.loc['2000':, :]
    tickers = prices.columns.to_list()
    n = len(tickers)

    if local_test == LocalTests.CURRENT_VS_ROLLING_LAST:
        """
        Verify that fit_current_covar produces the same matrix as the
        last entry from fit_rolling_covars when both see the same data.

        This tests internal consistency: the rolling estimator computes the
        full EWMA tensor and extracts slices at rebalancing dates, while
        fit_current_covar computes only the final slice. The two should
        agree exactly (up to floating-point tolerance) at the last
        rebalancing date that coincides with the last available price.
        """
        # use monthly rebalancing to maximise chance the last rebalancing
        # date aligns closely with the last price date
        params = dict(returns_freq='W-WED', span=52, rebalancing_freq='ME')
        estimator = EwmaCovarEstimator(**params)

        # current covariance: single matrix at last date
        current_covar = estimator.fit_current_covar(prices=prices)

        # rolling covariance: extract the last matrix
        time_period = qis.TimePeriod('31Dec2004', prices.index[-1])
        rolling_covars = estimator.fit_rolling_covars(prices=prices, time_period=time_period)
        last_date = max(rolling_covars.keys())
        last_rolling_covar = rolling_covars[last_date]

        # compare
        diff = np.abs(current_covar.values - last_rolling_covar.values)
        max_diff = diff.max()
        mean_diff = diff.mean()

        # extract vols for readable comparison
        vols_current = pd.Series(np.sqrt(np.diag(current_covar.values)), index=tickers, name='Current')
        vols_rolling = pd.Series(np.sqrt(np.diag(last_rolling_covar.values)), index=tickers, name='Rolling Last')
        vol_comparison = pd.concat([vols_current, vols_rolling,
                                    (vols_current - vols_rolling).rename('Diff')], axis=1)

        print(f"── Current vs Rolling Last ──")
        print(f"Last rolling date:  {last_date.strftime('%d%b%Y')}")
        print(f"Price data ends:    {prices.index[-1].strftime('%d%b%Y')}")
        print(f"\nVol comparison:")
        print(vol_comparison.to_string(float_format='{:.6%}'.format))
        print(f"\nMax abs covar diff:  {max_diff:.2e}")
        print(f"Mean abs covar diff: {mean_diff:.2e}")

        # the matrices should match exactly when the last rebalancing date
        # falls on the last available return date; otherwise there's a small
        # difference because fit_current_covar uses all data up to the end
        # while fit_rolling_covars extracts at the rebalancing date
        if max_diff < 1e-10:
            print(f"\nRESULT: EXACT MATCH")
        elif max_diff < 1e-4:
            print(f"\nRESULT: CLOSE MATCH (last rebalancing date likely "
                  f"precedes last price date by a few days)")
        else:
            print(f"\nRESULT: MISMATCH — investigate")

    elif local_test == LocalTests.ROLLING_COVAR_PROPERTIES:
        """
        Verify structural properties of the rolling covariance output:
        1. All matrices are symmetric positive semi-definite (PSD)
        2. All matrices have correct shape (N x N) with correct column/index labels
        3. Annualised vols fall in a reasonable range (1% - 100%)
        4. Rebalancing dates are quarterly and within the time period
        5. No NaN or Inf values in any matrix
        """
        estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
        time_period = qis.TimePeriod('31Dec2004', prices.index[-1])
        covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)

        n_dates = len(covar_dict)
        dates = sorted(covar_dict.keys())

        print(f"── Rolling Covar Properties ──")
        print(f"Estimator:    EWMA (weekly returns, span=52, quarterly rebal)")
        print(f"Universe:     {n} assets: {tickers}")
        print(f"Period:       {dates[0].strftime('%d%b%Y')} – {dates[-1].strftime('%d%b%Y')}")
        print(f"Num matrices: {n_dates}")

        # collect diagnostics across all dates
        all_min_eig = []
        all_max_vol = []
        all_min_vol = []
        n_nan = 0
        n_inf = 0
        n_asymmetric = 0
        n_wrong_shape = 0
        n_wrong_labels = 0

        for date, covar in covar_dict.items():
            mat = covar.values

            # shape check
            if mat.shape != (n, n):
                n_wrong_shape += 1

            # label check
            if list(covar.columns) != tickers or list(covar.index) != tickers:
                n_wrong_labels += 1

            # NaN / Inf check
            if np.any(np.isnan(mat)):
                n_nan += 1
            if np.any(np.isinf(mat)):
                n_inf += 1

            # symmetry check
            if not np.allclose(mat, mat.T, atol=1e-12):
                n_asymmetric += 1

            # PSD check via eigenvalues
            eigenvalues = np.linalg.eigvalsh(mat)
            all_min_eig.append(eigenvalues.min())

            # vol range check (annualised)
            vols = np.sqrt(np.diag(mat))
            all_max_vol.append(vols.max())
            all_min_vol.append(vols[vols > 0].min() if np.any(vols > 0) else 0.0)

        min_eig_series = pd.Series(all_min_eig, index=dates)
        max_vol_series = pd.Series(all_max_vol, index=dates)
        min_vol_series = pd.Series(all_min_vol, index=dates)

        # report
        print(f"\n── Shape and labels ──")
        print(f"Wrong shape:  {n_wrong_shape} / {n_dates}")
        print(f"Wrong labels: {n_wrong_labels} / {n_dates}")

        print(f"\n── Data quality ──")
        print(f"Contains NaN: {n_nan} / {n_dates}")
        print(f"Contains Inf: {n_inf} / {n_dates}")
        print(f"Asymmetric:   {n_asymmetric} / {n_dates}")

        print(f"\n── Positive semi-definiteness ──")
        print(f"Min eigenvalue across all dates: {min_eig_series.min():.6e}")
        print(f"Max eigenvalue min:              {min_eig_series.max():.6e}")
        n_non_psd = (min_eig_series < -1e-10).sum()
        print(f"Non-PSD matrices:                {n_non_psd} / {n_dates}")

        print(f"\n── Annualised vol range ──")
        print(f"Global min vol: {min_vol_series.min():.2%}")
        print(f"Global max vol: {max_vol_series.max():.2%}")
        vol_reasonable = (min_vol_series.min() > 0.01) and (max_vol_series.max() < 1.0)
        print(f"All vols in [1%, 100%]: {'YES' if vol_reasonable else 'NO'}")

        # check rebalancing schedule is quarterly
        date_diffs = pd.Series(dates).diff().dropna()
        min_gap = date_diffs.min().days
        max_gap = date_diffs.max().days
        print(f"\n── Rebalancing schedule ──")
        print(f"Min gap between dates: {min_gap} days")
        print(f"Max gap between dates: {max_gap} days")
        quarterly_ok = (min_gap >= 80) and (max_gap <= 100)
        print(f"Consistent quarterly:  {'YES' if quarterly_ok else 'CHECK'}")

        # summary
        all_pass = (n_wrong_shape == 0 and n_wrong_labels == 0 and
                    n_nan == 0 and n_inf == 0 and n_asymmetric == 0 and
                    n_non_psd == 0 and vol_reasonable)
        print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

        # plot rolling min eigenvalue and max vol for visual inspection
        fig, axs = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True)
        qis.plot_time_series(df=min_eig_series.to_frame('Min Eigenvalue'),
                             title='Min eigenvalue across rolling dates (should be >= 0)',
                             var_format='{:.2e}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             ax=axs[0])
        qis.plot_time_series(df=pd.concat([min_vol_series.rename('Min Vol'),
                                           max_vol_series.rename('Max Vol')], axis=1),
                             title='Rolling annualised vol range',
                             var_format='{:.0%}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             ax=axs[1])

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ROLLING_COVAR_PROPERTIES)