# packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as ss
from scipy.stats import bernoulli, multivariate_normal
from scipy.cluster.vq import kmeans2
from dataclasses import dataclass
from enum import Enum
from matplotlib.patches import Ellipse
from typing import List, Tuple, Union, Dict, Optional
import qis as qis

RANDOM_STATE = 3


# --- Custom Gaussian Mixture Model (replaces sklearn.mixture.GaussianMixture) ---

@dataclass
class GMMResult:
    """Result container for fitted Gaussian Mixture Model.
    Mirrors sklearn.mixture.GaussianMixture interface for means_, covariances_, weights_, predict().
    """
    means_: np.ndarray       # (n_components, n_features)
    covariances_: np.ndarray  # (n_components, n_features, n_features)
    weights_: np.ndarray      # (n_components,)

    def predict(self, x: np.ndarray) -> np.ndarray:
        resp = _e_step(x, self.means_, self.covariances_, self.weights_)
        return resp.argmax(axis=1)


def _initialize_gmm(x: np.ndarray, n_components: int, random_state: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-means initialization for GMM parameters."""
    rng = np.random.RandomState(random_state)
    centroids, labels = kmeans2(x, n_components, minit='points', seed=rng)

    n_samples, n_features = x.shape
    means = np.zeros((n_components, n_features))
    covariances = np.zeros((n_components, n_features, n_features))
    weights = np.zeros(n_components)

    for k in range(n_components):
        mask = labels == k
        count = mask.sum()
        if count == 0:
            means[k] = centroids[k]
            covariances[k] = np.eye(n_features)
            weights[k] = 1.0 / n_components
        else:
            means[k] = x[mask].mean(axis=0)
            diff = x[mask] - means[k]
            covariances[k] = (diff.T @ diff) / count + 1e-6 * np.eye(n_features)
            weights[k] = count / n_samples

    return means, covariances, weights


def _e_step(x: np.ndarray, means: np.ndarray, covariances: np.ndarray,
            weights: np.ndarray) -> np.ndarray:
    """E-step: compute responsibilities."""
    n_samples = x.shape[0]
    n_components = len(weights)
    resp = np.zeros((n_samples, n_components))

    for k in range(n_components):
        try:
            resp[:, k] = weights[k] * multivariate_normal.pdf(
                x, mean=means[k], cov=covariances[k], allow_singular=True
            )
        except np.linalg.LinAlgError:
            resp[:, k] = 0.0

    resp_sum = resp.sum(axis=1, keepdims=True)
    resp_sum = np.maximum(resp_sum, 1e-300)
    resp /= resp_sum
    return resp


def _m_step(x: np.ndarray, resp: np.ndarray, reg_covar: float = 1e-6
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M-step: update parameters from responsibilities."""
    n_samples, n_features = x.shape
    n_components = resp.shape[1]

    nk = resp.sum(axis=0)
    nk = np.maximum(nk, 1e-10)

    weights = nk / n_samples
    means = (resp.T @ x) / nk[:, np.newaxis]

    covariances = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        diff = x - means[k]
        covariances[k] = (resp[:, k:k + 1] * diff).T @ diff / nk[k]
        covariances[k] += reg_covar * np.eye(n_features)

    return means, covariances, weights


def _compute_log_likelihood(x: np.ndarray, means: np.ndarray,
                            covariances: np.ndarray, weights: np.ndarray) -> float:
    """Compute total log-likelihood of data under the mixture."""
    n_components = len(weights)
    ll = np.zeros((x.shape[0], n_components))
    for k in range(n_components):
        try:
            ll[:, k] = weights[k] * multivariate_normal.pdf(
                x, mean=means[k], cov=covariances[k], allow_singular=True
            )
        except np.linalg.LinAlgError:
            ll[:, k] = 0.0
    return np.log(np.maximum(ll.sum(axis=1), 1e-300)).sum()


def fit_gmm(x: np.ndarray,
            n_components: int = 2,
            random_state: int = RANDOM_STATE,
            max_iter: int = 100,
            tol: float = 1e-6,
            reg_covar: float = 1e-6
            ) -> GMMResult:
    """
    Fit Gaussian Mixture Model via EM algorithm.
    Drop-in replacement for sklearn.mixture.GaussianMixture with covariance_type='full'.
    """
    means, covariances, weights = _initialize_gmm(x, n_components, random_state)

    prev_ll = -np.inf
    for _ in range(max_iter):
        resp = _e_step(x, means, covariances, weights)
        means, covariances, weights = _m_step(x, resp, reg_covar=reg_covar)
        ll = _compute_log_likelihood(x, means, covariances, weights)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return GMMResult(means_=means, covariances_=covariances, weights_=weights)


# --- Original interface (Params, fit_gaussian_mixture, plotting, etc.) ---

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
                         an_factor: float = 1.0,
                         idx: int = None
                         ) -> Params:
    gmm = fit_gmm(x, n_components=n_components, random_state=RANDOM_STATE)

    if idx is not None:
        order = gmm.means_.argsort(axis=0)[:, idx]
        gmm.means_ = gmm.means_[order]
        gmm.covariances_ = gmm.covariances_[order]
        gmm.weights_ = gmm.weights_[order]

    return Params(means=[an_factor * m for m in gmm.means_],
                  covars=[an_factor * c for c in gmm.covariances_],
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

    gmm = fit_gmm(x, n_components=n_components, random_state=RANDOM_STATE)

    mean = gmm.means_
    covs = gmm.covariances_
    weights = gmm.weights_

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
                 idx: Optional[int] = None,
                 x_column: str = None,
                 y_column: str = None,
                 **kwargs
                 ) -> None:

    if ax is None:
        ax = plt.subplots(1, 1)

    gmm = fit_gmm(x, n_components=n_components, random_state=RANDOM_STATE)

    if idx is not None:
        order = gmm.means_.argsort(axis=0)[:, idx]
        gmm.means_ = gmm.means_[order]
        gmm.covariances_ = gmm.covariances_[order]
        gmm.weights_ = gmm.weights_[order]

    labels = gmm.predict(x)

    if columns is None:
        columns = [f"X{n+1}" for n in range(x.shape[1])]
    data = pd.DataFrame(x, columns=columns)
    data[label] = labels

    if n_components == 3:
        colors = ['red', 'slategray', 'green']
    else:
        colors = qis.get_n_colors(n=n_components, last_color_fixed=False)

    x_col = columns[0]
    y_col = columns[1]

    sns.scatterplot(data=data,
                    x=x_col,
                    y=y_col,
                    hue=label,
                    palette=colors,
                    ax=ax)

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
        scaler = qis.get_annualization_factor(freq=returns_freq)
    else:
        scaler = 1.0

    means, sigmas, probs = [], [], []
    for idx, end in enumerate(dates_schedule[1:]):
        if idx >= roll_window-1:
            period = qis.TimePeriod(dates_schedule[idx - roll_window+1], end)
            rets_ = period.locate(rets).to_numpy()
            params = fit_gaussian_mixture(x=rets_, n_components=n_components, an_factor=scaler)
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
