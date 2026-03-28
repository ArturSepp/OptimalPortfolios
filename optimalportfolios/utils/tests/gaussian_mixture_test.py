# packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bernoulli
from enum import Enum
import qis as qis
from optimalportfolios.utils.gaussian_mixture import (fit_gaussian_mixture,
                                                      plot_mixure1,
                                                      plot_mixure2,
                                                      estimate_rolling_mixture)


class LocalTests(Enum):
    FIT1 = 1
    FIT2 = 2
    ROLLING_FIT = 3
    PLOT_MIXURE = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real universe and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.FIT1:
        size = 1000
        p1 = 0.8
        mu1, sigma1 = 0.0, 0.2
        mu2, sigma2 = -1, 0.3
        w = bernoulli.rvs(p1, size=size)

        x1 = np.random.normal(mu1, sigma1, size=size)
        x2 = np.random.normal(mu2, sigma2, size=size)
        x = np.zeros(size)
        for n_ in range(size):
            x[n_] = x1[n_] if w[n_] == 1 else x2[n_]
        x = x.reshape(-1, 1)

        print(np.square(np.std(x1, axis=0)))
        print(np.square(np.std(x2, axis=0)))

        params = fit_gaussian_mixture(x=x)
        print(params)
        plot_mixure1(x=x)

    elif local_test == LocalTests.FIT2:
        size = 1000
        p1 = 0.8
        mu1, sigma1 = np.array([0, 0]), np.array([[0.2, 0.0], [0.0, 0.2]])
        mu2, sigma2 = np.array([-1.0, 1.00]), np.array([[0.1, 0.0], [0.0, 0.1]])
        w = bernoulli.rvs(p1, size=size)

        x1 = np.random.multivariate_normal(mu1, sigma1, size=size)
        x2 = np.random.multivariate_normal(mu2, sigma2, size=size)
        x = np.zeros((size, 2))
        for n_ in range(size):
            x[n_] = x1[n_] if w[n_] == 1 else x2[n_]

        print(np.square(np.std(x1, axis=0)))
        print(np.square(np.std(x2, axis=0)))

        params = fit_gaussian_mixture(x=x)
        print(params)
        plot_mixure2(x)

    elif local_test == LocalTests.ROLLING_FIT:
        from optimalportfolios.test_data import load_test_data
        prices = load_test_data()
        prices = prices.loc['2000':, :]  # have at least 3 assets
        prices = prices['SPY'].dropna()
        means, sigmas, probs = estimate_rolling_mixture(prices=prices)
        print(means)

    elif local_test == LocalTests.PLOT_MIXURE:
        import yfinance as yf
        prices = yf.download(tickers=['SPY', 'TLT'], start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].dropna()
        perf_params = qis.PerfParams(freq='W-WED')
        kwargs = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                      alpha_format='{0:+0.0%}',
                      beta_format='{:0.1f}')

        time_periods = [qis.TimePeriod('31Aug2002', '31Dec2019'), qis.TimePeriod('31Dec2019', '16Dec2022')]

        n_components = 3

        with sns.axes_style('white'):
            fig1, axs = plt.subplots(1, len(time_periods), figsize=(15, 5), constrained_layout=True)

        for idx, time_period in enumerate(time_periods):
            prices_ = time_period.locate(prices)
            rets = qis.to_returns(prices=prices_, is_log_returns=True, drop_first=True, freq=perf_params.freq)
            params = fit_gaussian_mixture(x=rets.to_numpy(), n_components=n_components, idx=1)
            plot_mixure2(x=rets.to_numpy(),
                            n_components=n_components,
                            columns=prices.columns,
                            title=f"({idx+1}) Returns and ellipsoids of Gaussian clusters for period {time_period.to_str()}",
                            ax=axs[idx],
                            **kwargs)

            means, vols, corrs = params.get_all_params(columns=prices.columns, vol_scaler=12.0)
            print(f"means=\n{means}")
            print(f"vols=\n{vols}")
            print(f"corrs=\n{corrs}")

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PLOT_MIXURE)
