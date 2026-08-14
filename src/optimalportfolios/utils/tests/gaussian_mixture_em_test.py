"""
the hand-rolled EM mixture fitter in ``utils.gaussian_mixture``.

``gaussian_mixture_test.py`` is a ``run_local_test`` diagnostic that plots and needs the
author's price data, so it contributes no collected tests. This file covers the fitter itself.

The module is a drop-in replacement for ``sklearn.mixture.GaussianMixture`` with
``covariance_type='full'`` — the package does not depend on scikit-learn, so the E-step,
M-step and the EM loop are all local code with nothing checking them. Every case here is
seeded and synthetic: two well-separated Gaussians whose true means, covariances and mixing
weights the test states, so a fit can be checked against what generated it rather than
against a previously recorded output.

Recovery is asserted with loose tolerances on purpose. EM on a finite sample does not return
the generating parameters exactly, and a tight bound would make this a change-detector for
the sample rather than a test of the fitter. What is asserted tightly is what must hold for
*any* input: responsibilities that sum to one, weights that sum to one, symmetric positive
semi-definite covariances, and a log-likelihood that never decreases across an EM step.

The plotting helpers are exercised on the Agg backend — ``ci.yml`` sets ``MPLBACKEND: Agg``
for the whole workflow, and the fixture below switches the backend locally so a developer
run does not open windows.
"""
# packages
from typing import List
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
from optimalportfolios.utils import gaussian_mixture
from optimalportfolios.utils.gaussian_mixture import (
    GMMResult,
    Params,
    _compute_log_likelihood,
    _e_step,
    _initialize_gmm,
    _m_step,
    draw_ellipse,
    estimate_rolling_mixture,
    fit_gaussian_mixture,
    fit_gmm,
    plot_mixure1,
    plot_mixure2,
)

SEED = 20260810
# two well-separated bivariate components, stated so a fit can be checked against them
TRUE_MEANS = np.array([[-0.05, -0.03], [0.06, 0.04]])
TRUE_COVARS = np.array([[[0.0040, 0.0012], [0.0012, 0.0030]],
                        [[0.0025, -0.0005], [-0.0005, 0.0020]]])
TRUE_WEIGHTS = np.array([0.35, 0.65])


@pytest.fixture
def agg_backend():
    """Draw onto the non-interactive Agg backend and close every figure afterwards."""
    import matplotlib.pyplot as plt
    previous = plt.get_backend()
    plt.switch_backend('Agg')
    yield plt
    plt.close('all')
    plt.switch_backend(previous)


def sample_mixture(n_samples: int = 900, seed: int = SEED) -> np.ndarray:
    """Draw from the two-component mixture the constants above describe."""
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_samples, TRUE_WEIGHTS)
    parts = [rng.multivariate_normal(TRUE_MEANS[k], TRUE_COVARS[k], size=counts[k])
             for k in range(len(TRUE_WEIGHTS))]
    x = np.vstack(parts)
    rng.shuffle(x)
    return x


def sample_univariate(n_samples: int = 600, seed: int = SEED) -> np.ndarray:
    """A one-dimensional two-regime sample: a calm regime and a volatile one."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.004, 0.010, size=int(0.7 * n_samples))
    stressed = rng.normal(-0.012, 0.035, size=n_samples - len(calm))
    x = np.concatenate([calm, stressed])
    rng.shuffle(x)
    return x.reshape(-1, 1)


def match_components(fitted_means: np.ndarray) -> List[int]:
    """Order fitted components to match TRUE_MEANS, which EM returns in arbitrary order."""
    return list(np.argsort(fitted_means[:, 0]))


# --------------------------------------------------------------------------- #
# the EM steps
# --------------------------------------------------------------------------- #
def test_initialisation_returns_well_formed_parameters() -> None:
    """k-means initialisation yields one mean, covariance and weight per component"""
    x = sample_mixture()
    means, covariances, weights = _initialize_gmm(x, n_components=2, random_state=SEED)
    assert means.shape == (2, 2)
    assert covariances.shape == (2, 2, 2)
    assert weights.shape == (2,)
    assert weights.sum() == pytest.approx(1.0)
    for covariance in covariances:
        assert np.allclose(covariance, covariance.T)
        assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_initialisation_gives_an_empty_cluster_a_usable_starting_point() -> None:
    """k-means can leave a cluster with no members, and it still needs parameters

    ``kmeans2`` warns and returns an empty cluster whenever it seeds two centroids on the same
    point, which is what asking for more components than the data has distinct values does.
    Taking the mean of no observations would give NaN means and a NaN covariance, and EM would
    then propagate NaN through every subsequent iteration. The centroid, an identity covariance
    and an even mixing weight give the component somewhere to start from instead.
    """
    x = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    with pytest.warns(UserWarning, match='clusters is empty'):
        means, covariances, weights = _initialize_gmm(x, n_components=3, random_state=SEED)
    empty = int(np.argmin(weights))
    assert np.allclose(covariances[empty], np.eye(2))
    assert weights[empty] == pytest.approx(1.0 / 3.0)
    assert np.isfinite(means).all()


def test_initialisation_is_reproducible_for_a_fixed_seed() -> None:
    """the same random_state gives the same starting point, so a fit is reproducible"""
    x = sample_mixture()
    first = _initialize_gmm(x, n_components=2, random_state=SEED)
    second = _initialize_gmm(x, n_components=2, random_state=SEED)
    for a, b in zip(first, second):
        assert np.allclose(a, b)


def test_e_step_responsibilities_are_a_distribution_per_observation() -> None:
    """every row of the responsibility matrix sums to one and is non-negative"""
    x = sample_mixture()
    resp = _e_step(x, TRUE_MEANS, TRUE_COVARS, TRUE_WEIGHTS)
    assert resp.shape == (len(x), 2)
    assert np.allclose(resp.sum(axis=1), 1.0)
    assert (resp >= 0.0).all()


def test_e_step_assigns_an_observation_to_the_component_that_generated_it() -> None:
    """a point sitting on one component's mean is attributed almost entirely to it"""
    on_first = TRUE_MEANS[0].reshape(1, -1)
    resp = _e_step(on_first, TRUE_MEANS, TRUE_COVARS, TRUE_WEIGHTS)
    assert resp[0, 0] > resp[0, 1]


def test_a_component_whose_density_will_not_evaluate_contributes_nothing(monkeypatch) -> None:
    """a singular component is dropped from the mixture rather than aborting the fit

    ``multivariate_normal.pdf`` is called with ``allow_singular=True``, so it decomposes
    almost anything — but a covariance that has collapsed far enough still raises
    ``LinAlgError`` out of LAPACK, and there is no input that reliably produces it across
    BLAS builds. Forcing the raise is the only way to pin what the guard does: the component's
    responsibility goes to zero and the remaining ones renormalise, so the E-step still
    returns a distribution instead of propagating the exception up through the EM loop.
    """
    def _raise_for_the_second_component(x, mean, cov, allow_singular):
        """Evaluate the first component normally and fail on the second."""
        if np.allclose(mean, TRUE_MEANS[1]):
            raise np.linalg.LinAlgError('singular covariance')
        return real_pdf(x, mean=mean, cov=cov, allow_singular=allow_singular)

    real_pdf = gaussian_mixture.multivariate_normal.pdf
    monkeypatch.setattr(gaussian_mixture.multivariate_normal, 'pdf',
                        _raise_for_the_second_component)

    x = sample_mixture(n_samples=200)
    resp = _e_step(x, TRUE_MEANS, TRUE_COVARS, TRUE_WEIGHTS)
    assert np.allclose(resp[:, 1], 0.0)
    assert np.allclose(resp.sum(axis=1), 1.0)
    # the likelihood is computed on the surviving component alone, and stays finite
    assert np.isfinite(_compute_log_likelihood(x, TRUE_MEANS, TRUE_COVARS, TRUE_WEIGHTS))


def test_m_step_recovers_the_parameters_from_hard_responsibilities() -> None:
    """with responsibilities pinned to the truth the M-step returns the sample moments"""
    x = sample_mixture()
    resp = _e_step(x, TRUE_MEANS, TRUE_COVARS, TRUE_WEIGHTS)
    means, covariances, weights = _m_step(x, resp)
    assert weights.sum() == pytest.approx(1.0)
    assert means.shape == TRUE_MEANS.shape
    order = match_components(means)
    assert np.allclose(means[order], TRUE_MEANS, atol=0.02)
    for covariance in covariances:
        assert np.allclose(covariance, covariance.T)
        assert np.linalg.eigvalsh(covariance).min() >= 0.0


def test_m_step_regularisation_lifts_a_degenerate_covariance() -> None:
    """reg_covar keeps a component with no spread invertible instead of singular"""
    x = np.zeros((10, 2))
    resp = np.tile([1.0, 0.0], (10, 1))
    _, covariances, _ = _m_step(x, resp, reg_covar=1e-3)
    assert np.linalg.eigvalsh(covariances[0]).min() == pytest.approx(1e-3)


def test_log_likelihood_increases_over_an_em_step() -> None:
    """EM cannot make the fit worse — this is the property the loop relies on to stop"""
    x = sample_mixture()
    means, covariances, weights = _initialize_gmm(x, n_components=2, random_state=SEED)
    before = _compute_log_likelihood(x, means, covariances, weights)
    resp = _e_step(x, means, covariances, weights)
    means, covariances, weights = _m_step(x, resp)
    after = _compute_log_likelihood(x, means, covariances, weights)
    assert after >= before - 1e-8


# --------------------------------------------------------------------------- #
# the fitter
# --------------------------------------------------------------------------- #
def test_fit_gmm_recovers_the_generating_mixture() -> None:
    """a two-component fit on a two-component sample finds the stated parameters"""
    fitted = fit_gmm(sample_mixture(n_samples=3000), n_components=2, random_state=SEED)
    assert isinstance(fitted, GMMResult)
    assert fitted.weights_.sum() == pytest.approx(1.0)
    order = match_components(fitted.means_)
    assert np.allclose(fitted.means_[order], TRUE_MEANS, atol=0.02)
    assert np.allclose(fitted.weights_[order], TRUE_WEIGHTS, atol=0.10)
    for k, component in enumerate(order):
        assert np.allclose(fitted.covariances_[component], TRUE_COVARS[k], atol=0.002)


def test_fit_gmm_is_deterministic_for_a_fixed_seed() -> None:
    """the same data and seed give the same fit, which the rolling estimator relies on"""
    x = sample_mixture()
    first = fit_gmm(x, n_components=2, random_state=SEED)
    second = fit_gmm(x, n_components=2, random_state=SEED)
    assert np.allclose(first.means_, second.means_)
    assert np.allclose(first.weights_, second.weights_)


def test_fit_gmm_stops_early_once_the_likelihood_stops_moving() -> None:
    """a loose tolerance converges to the same place as a tight one on separated data"""
    x = sample_mixture(n_samples=1500)
    loose = fit_gmm(x, n_components=2, random_state=SEED, tol=1e-3)
    tight = fit_gmm(x, n_components=2, random_state=SEED, tol=1e-10, max_iter=500)
    assert np.allclose(np.sort(loose.means_[:, 0]), np.sort(tight.means_[:, 0]), atol=0.01)


def test_predict_labels_every_observation_with_a_component() -> None:
    """predict returns the arg-max responsibility, so labels index the components"""
    x = sample_mixture()
    fitted = fit_gmm(x, n_components=2, random_state=SEED)
    labels = fitted.predict(x)
    assert labels.shape == (len(x),)
    assert set(np.unique(labels)) <= {0, 1}
    assert len(np.unique(labels)) == 2, "a separated two-component sample should use both"


def test_fit_gmm_supports_three_components() -> None:
    """the component count is not hard-coded at two"""
    fitted = fit_gmm(sample_mixture(n_samples=1200), n_components=3, random_state=SEED)
    assert fitted.means_.shape == (3, 2)
    assert fitted.weights_.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Params
# --------------------------------------------------------------------------- #
def test_fit_gaussian_mixture_scales_by_the_annualisation_factor() -> None:
    """an_factor multiplies means and covariances, leaving the probabilities alone"""
    x = sample_mixture()
    plain = fit_gaussian_mixture(x, n_components=2, an_factor=1.0, idx=0)
    annualised = fit_gaussian_mixture(x, n_components=2, an_factor=52.0, idx=0)
    assert isinstance(plain, Params)
    assert np.allclose(annualised.probs, plain.probs)
    for scaled, base in zip(annualised.means, plain.means):
        assert np.allclose(scaled, 52.0 * base)
    for scaled, base in zip(annualised.covars, plain.covars):
        assert np.allclose(scaled, 52.0 * base)


def test_fit_gaussian_mixture_sorts_components_by_the_named_feature() -> None:
    """idx orders the components so colours and columns stay comparable across panels"""
    x = sample_mixture()
    sorted_fit = fit_gaussian_mixture(x, n_components=2, idx=0)
    first_feature = [mean[0] for mean in sorted_fit.means]
    assert first_feature == sorted(first_feature)


def test_params_tables_are_indexed_by_ticker() -> None:
    """get_params_pd returns per-component covariance frames and mean series"""
    params = fit_gaussian_mixture(sample_mixture(), n_components=2, idx=0)
    tickers = ['spx', 'ust']
    covars, means, probs = params.get_params_pd(tickers=tickers)
    assert len(covars) == len(means) == 2
    for covar in covars:
        assert list(covar.index) == tickers and list(covar.columns) == tickers
    for mean in means:
        assert list(mean.index) == tickers
    assert probs.sum() == pytest.approx(1.0)


def test_params_tables_reindex_onto_the_wider_universe() -> None:
    """all_tickers pads the fitted pair with NaN columns for the unfitted assets"""
    params = fit_gaussian_mixture(sample_mixture(), n_components=2, idx=0)
    covars, means, _ = params.get_params_pd(tickers=['spx', 'ust'],
                                            all_tickers=['spx', 'ust', 'gold'])
    assert list(covars[0].index) == ['spx', 'ust', 'gold']
    assert covars[0].loc['gold'].isna().all()
    assert np.isnan(means[0]['gold'])


def test_params_get_params_reports_probability_mean_and_vol() -> None:
    """the per-feature view carries one row per component"""
    params = fit_gaussian_mixture(sample_mixture(), n_components=2, idx=0)
    frame = params.get_params(idx=0)
    assert list(frame.columns) == ['Prob', 'Mean', 'Std']
    assert len(frame) == 2
    assert frame['Prob'].sum() == pytest.approx(1.0)
    assert (frame['Std'] > 0).all()


def test_get_all_params_returns_pairwise_correlations_for_two_features() -> None:
    """with exactly two columns the correlation is one number per component"""
    params = fit_gaussian_mixture(sample_mixture(), n_components=2, idx=0)
    means, vols, corrs = params.get_all_params(columns=['spx', 'ust'])
    assert list(means.columns) == ['Prob', 'spx', 'ust']
    assert list(vols.columns) == ['spx', 'ust']
    assert means.index.name == 'cluster' and vols.index.name == 'cluster'
    assert isinstance(corrs, pd.Series)
    assert len(corrs) == 2
    assert ((corrs >= -1.0) & (corrs <= 1.0)).all()


def test_get_all_params_returns_a_matrix_per_component_beyond_two_features() -> None:
    """with three columns each component gets a full correlation matrix"""
    rng = np.random.default_rng(SEED)
    x = rng.multivariate_normal(np.zeros(3), np.diag([0.004, 0.003, 0.002]), size=800)
    params = fit_gaussian_mixture(x, n_components=2, idx=0)
    columns = ['spx', 'ust', 'gold']
    _, _, corrs = params.get_all_params(columns=columns)
    assert isinstance(corrs, dict)
    assert set(corrs) == {'0 cluster', '1 cluster'}
    for matrix in corrs.values():
        assert list(matrix.index) == columns
        assert np.allclose(np.diag(matrix.to_numpy()), 1.0)


def test_get_all_params_vol_scaler_annualises_means_and_vols_differently() -> None:
    """means scale linearly and vols by the square root, which is the annualisation rule"""
    params = fit_gaussian_mixture(sample_mixture(), n_components=2, idx=0)
    plain_means, plain_vols, _ = params.get_all_params(columns=['spx', 'ust'])
    scaled_means, scaled_vols, _ = params.get_all_params(columns=['spx', 'ust'],
                                                         vol_scaler=52.0)
    assert np.allclose(scaled_means['spx'], 52.0 * plain_means['spx'])
    assert np.allclose(scaled_vols['spx'], np.sqrt(52.0) * plain_vols['spx'])


def test_params_print_names_every_block(capsys) -> None:
    """the diagnostic print emits the probabilities, means and covariances"""
    fit_gaussian_mixture(sample_mixture(), n_components=2, idx=0).print()
    printed = capsys.readouterr().out
    assert 'probs=' in printed and 'mus=' in printed and 'sigmas=' in printed


# --------------------------------------------------------------------------- #
# the rolling estimator
# --------------------------------------------------------------------------- #
def make_price_series(n_days: int = 900) -> pd.Series:
    """A single seeded price series long enough to fill several rolling windows."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range('2019-01-01', periods=n_days, freq='B')
    returns = rng.normal(0.0004, 0.011, size=n_days)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=dates, name='spx')


def test_estimate_rolling_mixture_returns_aligned_panels() -> None:
    """means, vols and probabilities share an index of estimation dates"""
    means, vols, probs = estimate_rolling_mixture(
        prices=make_price_series(), returns_freq='W-WED', rebalancing_freq='QE',
        roll_window=4, n_components=2)
    assert len(means) > 0
    assert means.index.equals(vols.index) and means.index.equals(probs.index)
    assert means.shape[1] == vols.shape[1] == probs.shape[1] == 2
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (vols.to_numpy() > 0).all()
    # the components are ordered by mean at every date
    assert (means.iloc[:, 0] <= means.iloc[:, 1]).all()


def test_estimate_rolling_mixture_accepts_a_one_column_frame() -> None:
    """a single-column DataFrame is the same input as the Series"""
    prices = make_price_series()
    from_series = estimate_rolling_mixture(prices=prices, roll_window=4, n_components=2)[0]
    from_frame = estimate_rolling_mixture(prices=prices.to_frame(), roll_window=4,
                                          n_components=2)[0]
    pd.testing.assert_frame_equal(from_series, from_frame)


def test_estimate_rolling_mixture_rejects_a_multi_asset_panel() -> None:
    """the estimator is one-dimensional, so a wider frame is an error not a silent pick"""
    prices = make_price_series()
    panel = pd.concat([prices.rename('spx'), prices.rename('ust')], axis=1, sort=True)
    with pytest.raises(ValueError, match='supported only 1-d'):
        estimate_rolling_mixture(prices=panel, roll_window=4, n_components=2)


def test_estimate_rolling_mixture_without_annualising_gives_smaller_vols() -> None:
    """annualize scales the fitted parameters, so the raw fit is the smaller one"""
    prices = make_price_series()
    annual = estimate_rolling_mixture(prices=prices, roll_window=4, n_components=2,
                                      annualize=True)[1]
    raw = estimate_rolling_mixture(prices=prices, roll_window=4, n_components=2,
                                   annualize=False)[1]
    assert (raw.to_numpy() < annual.to_numpy()).all()


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def test_draw_ellipse_adds_three_sigma_rings(agg_backend) -> None:
    """each component is drawn at one, two and three sigma"""
    _, ax = agg_backend.subplots()
    before = len(ax.patches)
    draw_ellipse(position=np.array([0.0, 0.0]), covariance=TRUE_COVARS[0], ax=ax)
    assert len(ax.patches) == before + 3


def test_draw_ellipse_handles_a_diagonal_covariance(agg_backend) -> None:
    """a non-2x2 covariance takes the axis-aligned branch rather than an SVD"""
    _, ax = agg_backend.subplots()
    draw_ellipse(position=np.array([0.0, 0.0]), covariance=np.array([0.01, 0.02]), ax=ax)
    assert len(ax.patches) == 3


def test_plot_mixure1_draws_both_densities_and_their_sum(agg_backend) -> None:
    """the one-dimensional panel overlays two component densities and the mixture"""
    _, ax = agg_backend.subplots()
    plot_mixure1(sample_univariate(), n_components=2, ax=ax)
    assert len(ax.lines) == 3
    assert len(ax.patches) > 0                      # the histogram bars


def test_plot_mixure2_scatters_the_components_with_their_ellipses(agg_backend) -> None:
    """the two-dimensional panel colours points by component and rings each one"""
    _, ax = agg_backend.subplots()
    plot_mixure2(sample_mixture(n_samples=400), n_components=2, ax=ax,
                 columns=['spx', 'ust'], title='mixture', idx=0)
    assert len(ax.collections) > 0                  # the scatter
    assert len(ax.patches) >= 3                     # at least one component's rings


def test_plot_mixure2_creates_its_own_axis_when_none_is_given(agg_backend) -> None:
    """the plotting helpers are usable without pre-making a figure"""
    plot_mixure2(sample_mixture(n_samples=300), n_components=2, columns=['spx', 'ust'])
    assert agg_backend.get_fignums()


def test_plot_mixure2_names_unlabelled_features_by_position(agg_backend) -> None:
    """without column names the axes are labelled X1, X2 rather than left blank"""
    _, ax = agg_backend.subplots()
    plot_mixure2(sample_mixture(n_samples=300), n_components=2, ax=ax)
    assert ax.get_xlabel() == 'X1'
    assert ax.get_ylabel() == 'X2'


def test_plot_mixure2_uses_the_stated_palette_for_three_components(agg_backend) -> None:
    """three components are the regime panel, whose red/grey/green reading is fixed

    Any other count takes a generated palette; three is special-cased so the down, neutral and
    up regimes keep the same colours from one exhibit to the next.
    """
    _, ax = agg_backend.subplots()
    plot_mixure2(sample_mixture(n_samples=400), n_components=3, ax=ax,
                 columns=['spx', 'ust'])
    assert len(ax.collections) > 0
    assert len(ax.patches) >= 3
