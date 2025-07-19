# packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as ss
from sklearn.mixture import GaussianMixture
from scipy.stats import bernoulli
from dataclasses import dataclass
from enum import Enum
from matplotlib.patches import Ellipse
from typing import List, Tuple, Union, Dict
import qis

RANDOM_STATE = 3


@dataclass
class Params:
    means: List[np.ndarray]
    covars: List[np.ndarray]
    probs: np.ndarray

    def print(self):
        print(f"probs=\n{self.probs}")
        print(f"mus=\n{self.means}")
        print(f"sigmas=\n{self.covars}")

    def get_params_pd(self, tickers: List[str],
                      all_tickers: List[str] = None,
                      scaler: float = 1
                      ) -> Tuple[List[pd.DataFrame], List[pd.Series], np.ndarray]:
        """
        returns params as dfs indexed by tickers or all_tickers
        """
        covar_dict = []
        for covar in self.covars:
            covar_pd = pd.DataFrame(scaler*covar, index=tickers, columns=tickers)
            if all_tickers is not None:
                covar_pd = covar_pd.reindex(index=all_tickers).reindex(columns=all_tickers)
            covar_dict.append(covar_pd)

        pd_means = []
        for mean in self.means:
            mean_pd = pd.Series(scaler*mean, tickers)
            if all_tickers is not None:
                mean_pd = mean_pd.reindex(all_tickers)
            pd_means.append(mean_pd)

        return covar_dict, pd_means, self.probs

    def get_params(self, idx: int = 0) -> pd.DataFrame:
        means = np.array([mean[idx] for mean in self.means])
        std = np.array([np.sqrt(covar[idx][idx]) for covar in self.covars])
        probs = pd.Series(self.probs, name='Prob')
        means = pd.Series(means, name='Mean')
        std = pd.Series(std, name='Std')
        return pd.concat([probs, means, std], axis=1)

    def get_all_params(self, columns: List[str], vol_scaler: float = 1.0
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, Union[pd.Series, Dict[str, pd.DataFrame]]]:
        probs = pd.Series(self.probs, name='Prob')
        means = [probs]
        vols = []
        for idx, column in enumerate(columns):
            means.append(pd.Series([vol_scaler*mean[idx] for mean in self.means], name=column))
            vols.append(pd.Series([np.sqrt(vol_scaler)*np.sqrt(covar[idx][idx]) for covar in self.covars], name=column))
        means = pd.concat(means, axis=1)
        means.index.name = 'cluster'
        vols = pd.concat(vols, axis=1)
        vols.index.name = 'cluster'
        if len(columns) == 2:
            corrs = pd.Series([covar[0][1] / np.sqrt(covar[0][0]*covar[1][1]) for covar in self.covars])
        else:
            corrs = {}
            for idx, covar in enumerate(self.covars):
                corrs[f"{idx} cluster"] = pd.DataFrame(qis.covar_to_corr(covar), index=columns, columns=columns)

        return means, vols, corrs


def fit_gaussian_mixture(x: np.ndarray,
                         n_components: int = 2,
                         scaler: float = 1.0,
                         idx: int = None
                         ):
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=RANDOM_STATE)
    gmm.fit(x)
    if idx is not None:
        order = gmm.means_.argsort(axis=0)[:, idx]
        gmm.means_ = gmm.means_[order]
        gmm.covariances_ = gmm.covariances_[order]
        gmm.weights_ = gmm.weights_[order]
        gmm.precisions_ = gmm.precisions_[order]
        gmm.precisions_cholesky_ = gmm.precisions_cholesky_[order]

    return Params(means=[scaler*x for x in gmm.means_],  # convert to lists
                  covars=[scaler*x for x in gmm.covariances_],
                  probs=gmm.weights_)


def draw_ellipse(position, covariance,
                 ax: plt.Subplot,
                 color: str = 'gray',
                 **kwargs) -> None:
    """Draw an ellipse with a given position and covariance"""

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle=angle,
                             color=color,
                             **kwargs))


def plot_mixure1(x: np.ndarray,
                 n_components: int = 2,
                 label=True,
                 columns: List[str] = None,
                 ax=None
                 ) -> None:
    ax = ax or plt.gca()

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=RANDOM_STATE)
    gmm.fit(x)

    x_ = np.linspace(np.min(x), np.max(x), 100)

    # find useful parameters
    fitted_model = gmm.fit(x)
    mean = fitted_model.means_
    covs = fitted_model.covariances_
    weights = fitted_model.weights_

    # create necessary things to plot
    x_axis = np.linspace(1.25*np.min(x), 1.25*np.max(x), 100)
    y_axis0 = ss.norm.pdf(x_axis, float(mean[0][0]), np.sqrt(float(covs[0][0][0]))) * weights[0]  # 1st gaussian
    y_axis1 = ss.norm.pdf(x_axis, float(mean[1][0]), np.sqrt(float(covs[1][0][0]))) * weights[1]  # 2nd gaussian
    ax.hist(x, 10, density=True, color='lightblue')
    ax.plot(x_axis, y_axis0, lw=3, c='C0')
    ax.plot(x_axis, y_axis1, lw=3, c='C1')
    ax.plot(x_axis, y_axis0+y_axis1, lw=3, c='C2', ls='dashed')


def plot_mixure2(x: np.ndarray,
                 n_components: int = 2,
                 label: str = 'Cluster',
                 title: str = None,
                 columns: List[str] = None,
                 ax: plt.Subplot = None,
                 var_format: str = '{:.0%}',
                 idx: int = 1,
                 **kwargs
                 ) -> None:

    if ax is None:
        ax = plt.subplots(1, 1)

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=RANDOM_STATE).fit(x)
    order = gmm.means_.argsort(axis=0)[:, idx]
    gmm.means_ = gmm.means_[order]
    gmm.covariances_ = gmm.covariances_[order]
    gmm.weights_ = gmm.weights_[order]
    gmm.precisions_ = gmm.precisions_[order]
    gmm.precisions_cholesky_ = gmm.precisions_cholesky_[order]

    labels = gmm.predict(x)

    if columns is None:
        columns = [f"X{n+1}" for n in range(x.shape[1])]
    data = pd.DataFrame(x, columns=columns)
    data[label] = labels

    if n_components == 3:
        # colors = ['red', 'orange', 'green']
        colors = ['red', 'slategray', 'green']
    else:
        colors = qis.get_n_colors(n=n_components, last_color_fixed=False)
    sns.scatterplot(data=data,
                    x=columns[0],
                    y=columns[1],
                    hue=label,
                    palette=colors,
                    ax=ax)

    # w_factor = 0.2 / gmm.weights_.max() using for alpha
    for pos, covar, w, color in zip(gmm.means_, gmm.covariances_, gmm.weights_, colors):
        draw_ellipse(pos[:2], covar[:2, :2], ax=ax, alpha=0.1, color=color)

    qis.set_title(ax=ax, title=title, **kwargs)
    qis.set_ax_ticks_format(ax=ax, xvar_format=var_format, yvar_format=var_format, **kwargs)
    qis.set_ax_xy_labels(ax=ax, xlabel=columns[0], ylabel=columns[1], **kwargs)
    for label_, color in zip(ax.legend().get_texts(), colors):
        label_.set_color(color)
        label_.set_size(12)
    ax.get_legend().set_title(label, prop={'size': 12})


def estimate_rolling_mixture(prices: Union[pd.Series, pd.DataFrame],
                             returns_freq: str = 'W-WED',
                             rebalancing_freq: str = 'QE',
                             roll_window: int = 20,
                             n_components: int = 3,
                             annualize: bool = True
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    elif isinstance(prices, pd.DataFrame) and len(prices.columns) > 1:
        raise ValueError(f"supported only 1-d price time series")

    rets = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)

    dates_schedule = qis.generate_dates_schedule(time_period=qis.get_time_period(df=rets),
                                                 freq=rebalancing_freq,
                                                 include_start_date=True,
                                                 include_end_date=True)
    if annualize:
        _, scaler = qis.get_period_days(freq=returns_freq)
    else:
        scaler = 1.0

    means, sigmas, probs = [], [], []
    for idx, end in enumerate(dates_schedule[1:]):
        if idx >= roll_window-1:
            period = qis.TimePeriod(dates_schedule[idx - roll_window+1], end)
            # period.print()
            rets_ = period.locate(rets).to_numpy()
            params = fit_gaussian_mixture(x=rets_, n_components=n_components, scaler=scaler)
            mean = np.stack(params.means, axis=0).T[0]
            std = np.sqrt(np.array([params.covars[0][0], params.covars[1][0]]))
            prob = params.probs
            ranks = mean.argsort().argsort()
            means.append(pd.DataFrame(mean[ranks].reshape(1, -1), index=[end]))
            sigmas.append(pd.DataFrame(std[ranks].reshape(1, -1), index=[end]))
            probs.append(pd.DataFrame(prob[ranks].reshape(1, -1), index=[end]))

    means = pd.concat(means)
    sigmas = pd.concat(sigmas)
    probs = pd.concat(probs)

    return means, sigmas, probs


class UnitTests(Enum):
    FIT1 = 1
    FIT2 = 2
    ROLLING_FIT = 3
    PLOT_MIXURE = 4


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.FIT1:
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

    elif unit_test == UnitTests.FIT2:
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

    elif unit_test == UnitTests.ROLLING_FIT:
        from optimalportfolios.test_data import load_test_data
        prices = load_test_data()
        prices = prices.loc['2000':, :]  # have at least 3 assets
        prices = prices['SPY'].dropna()
        means, sigmas, probs = estimate_rolling_mixture(prices=prices)
        print(means)

    elif unit_test == UnitTests.PLOT_MIXURE:
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

    unit_test = UnitTests.PLOT_MIXURE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
