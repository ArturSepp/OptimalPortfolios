"""Canonical script of docs/cara_gaussian_mixture.md.

The page shows excerpts of this file; every number it quotes is asserted here against a
reference computed a different way. The script runs offline after ``pip install
optimalportfolios`` and needs no data file:

    python -m examples.docs.cara_gaussian_mixture

One step deliberately makes a solve fail and logs its rejection. ``exhibit`` draws the page's
figure; ``tools/docs_analytics/teaching.py`` calls it with the constants below and records their
values.
"""
import inspect
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pandas as pd
import qis

import optimalportfolios as op
from optimalportfolios.optimization.solver_diagnostics import validate_scipy_solution

TICKERS = ['Bonds', 'Equities', 'Crypto']
COMPONENTS = ['Calm', 'Stress', 'Crash']
# Probabilities, annual log-return means and annual volatilities of the three components; one
# correlation matrix serves every component.
PROBS = np.array([0.80, 0.15, 0.05])
MEANS = np.array([
    [0.02, 0.11, 0.50],
    [0.04, -0.06, -0.20],
    [0.06, -0.30, -1.50],
])
VOLS = np.array([
    [0.05, 0.13, 0.60],
    [0.07, 0.20, 0.80],
    [0.08, 0.30, 0.70],
])
CORR = np.array([
    [1.0, -0.2, 0.0],
    [-0.2, 1.0, 0.3],
    [0.0, 0.3, 1.0],
])
GAMMA = 5.0  # risk aversion of the worked example
GAMMA_GRID = [0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
              6.0]
SCALE_GAMMA = 10.0  # risk aversion of the statement on the objective's scale
CRYPTO = TICKERS.index('Crypto')
CRASH = COMPONENTS.index('Crash')
N_WEEKS = 416  # eight years of simulated weekly returns
N_DRAWS = 200_000  # Monte Carlo draws per component
SEED = 11


def component_covariances() -> list:
    """Return the covariance matrix of each component."""
    return [np.outer(vols, vols) * CORR for vols in VOLS]


def mixture_moments(probs, means, covars) -> tuple:
    """Return the mean and covariance of a Gaussian mixture (law of total variance)."""
    means = np.asarray(means)
    mean = probs @ means
    covar = sum(p * (c + np.outer(m - mean, m - mean)) for p, m, c in zip(probs, means, covars))
    return mean, covar


def between_covariance(probs, means) -> np.ndarray:
    """Return the covariance of the component means, sum_k p_k (mu_k - mu)(mu_k - mu)'."""
    deviations = np.asarray(means) - probs @ np.asarray(means)
    return sum(p * np.outer(d, d) for p, d in zip(probs, deviations))


def budget_mean_variance(mean: np.ndarray, covar: np.ndarray, gamma: float) -> np.ndarray:
    """Return the fully invested maximiser of mean'w - gamma/2 w'Cw by a linear solve."""
    ones = np.ones(len(mean))
    inverse_mean, inverse_ones = np.linalg.solve(covar, mean), np.linalg.solve(covar, ones)
    eta = (inverse_mean.sum() - gamma) / inverse_ones.sum()
    return (inverse_mean - eta * inverse_ones) / gamma


def exponents(weights: np.ndarray, gamma: float, scale: float = 1.0) -> np.ndarray:
    """Return q_k(w) = -gamma mu_k'w + gamma^2/2 w'S_k w with the moments divided by scale."""
    return np.array([(-gamma * m @ weights + 0.5 * gamma ** 2 * weights @ c @ weights) / scale
                     for m, c in zip(MEANS, component_covariances())])


def expected_disutility(weights: np.ndarray, gamma: float) -> float:
    """Return E[exp(-gamma w'r)] under the mixture, from the component exponents."""
    return float(PROBS @ np.exp(exponents(weights, gamma)))


def certainty_equivalent(weights: np.ndarray, gamma: float) -> float:
    """Return the certainty equivalent -log(E[exp(-gamma w'r)]) / gamma under the mixture."""
    return -np.log(expected_disutility(weights, gamma)) / gamma


def tilted_probabilities(weights: np.ndarray, gamma: float) -> np.ndarray:
    """Return p_k exp(q_k(w)) normalised: each component weighted by its marginal utility."""
    scaled = PROBS * np.exp(exponents(weights, gamma))
    return scaled / scaled.sum()


def long_only(values: np.ndarray) -> np.ndarray:
    """Set the solver's negligible weights to zero and renormalise."""
    kept = np.where(values > 1e-9, values, 0.0)
    return kept / kept.sum()


def cvxpy_mixture_optimum(gamma: float) -> np.ndarray:
    """Minimise the expected disutility with CVXPY's exponential cone: an independent solve."""
    w = cp.Variable(len(TICKERS))
    terms = [p * cp.exp(-gamma * m @ w + 0.5 * gamma ** 2 * cp.quad_form(w, c))
             for p, m, c in zip(PROBS, MEANS, component_covariances())]
    cp.Problem(cp.Minimize(sum(terms)), [cp.sum(w) == 1, w >= 0]).solve(solver='CLARABEL')
    return long_only(w.value)


def cvxpy_weekly_regimes_optimum(gamma: float) -> np.ndarray:
    """Maximise the annual CE when 52 weeks are independent draws of the mixture scaled by 1/52."""
    w = cp.Variable(len(TICKERS))
    terms = [np.log(p) + (-gamma * m @ w + 0.5 * gamma ** 2 * cp.quad_form(w, c)) / 52.0
             for p, m, c in zip(PROBS, MEANS, component_covariances())]
    cp.Problem(cp.Minimize(cp.log_sum_exp(cp.hstack(terms))),
               [cp.sum(w) == 1, w >= 0]).solve(solver='CLARABEL')
    return long_only(w.value)


def quoted(values, digits: int) -> list:
    """Round each value as the page quotes it."""
    return [round(float(value), digits) for value in np.atleast_1d(values)]


def mixture_draws() -> np.ndarray:
    """Draw annual returns with component counts proportional to PROBS, with Cholesky factors."""
    draws = []
    for k, covar in enumerate(component_covariances()):
        shocks = np.random.default_rng(k).standard_normal((round(PROBS[k] * N_DRAWS), 3))
        draws.append(MEANS[k] + shocks @ np.linalg.cholesky(covar).T)
    return np.concatenate(draws)


def simulated_weekly_returns(seed: int) -> pd.DataFrame:
    """Draw weekly log returns from the mixture scaled to one week, with Cholesky factors."""
    rng = np.random.default_rng(seed)
    regimes = np.searchsorted(np.cumsum(PROBS), rng.random(N_WEEKS), side='right')
    shocks = rng.standard_normal((N_WEEKS, len(TICKERS)))
    returns = np.empty((N_WEEKS, len(TICKERS)))
    for k, covar in enumerate(component_covariances()):
        rows = regimes == k
        factor = np.linalg.cholesky(covar / 52.0)
        returns[rows] = MEANS[k] / 52.0 + shocks[rows] @ factor.T
    dates = pd.date_range('2016-01-13', periods=N_WEEKS, freq='W-WED')
    return pd.DataFrame(returns, index=dates, columns=TICKERS)


def prices_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Compound weekly log returns into prices that start at 100 one week earlier."""
    start = returns.index[0] - pd.Timedelta(days=7)
    levels = pd.concat([pd.DataFrame(0.0, index=[start], columns=returns.columns),
                        returns.cumsum()])
    return 100.0 * np.exp(levels)


def expected_rebalancing_dates(index: pd.DatetimeIndex, window: int) -> list:
    """Return the first return date on or after each quarter end, once a window is complete."""
    quarter_ends = pd.date_range(index[0], index[-1], freq='QE')
    firsts = [index[index >= end][0] for end in quarter_ends if (index >= end).any()]
    return [date for date in firsts if index.get_loc(date) >= window - 1]


def allocations(gammas: list) -> pd.DataFrame:
    """Solve the one- and three-component problems on a grid of risk aversions."""
    mean, covar = mixture_moments(PROBS, MEANS, component_covariances())
    rows = {}
    for gamma in gammas:
        one = op.opt_maximize_cara_mixture(means=[mean], covars=[covar], probs=np.array([1.0]),
                                           constraints=op.Constraints(is_long_only=True),
                                           carra=gamma)
        three = op.opt_maximize_cara_mixture(means=list(MEANS), covars=component_covariances(),
                                             probs=PROBS,
                                             constraints=op.Constraints(is_long_only=True),
                                             carra=gamma)
        closed_form = budget_mean_variance(mean, covar, gamma)
        rows[gamma] = {'one component': one[CRYPTO],
                       'one component, closed form': closed_form[CRYPTO],
                       'closed form is long-only': bool(closed_form.min() >= 0.0),
                       'three components': three[CRYPTO],
                       'three components, CVXPY': cvxpy_mixture_optimum(gamma)[CRYPTO]}
    table = pd.DataFrame(rows).T
    table.index.name = 'gamma'
    return table


def main() -> None:
    """Run the worked example of the page and assert every quoted number."""
    # One component with the mixture's mean and covariance: three routes to the closed form.
    mean, covar = mixture_moments(PROBS, MEANS, component_covariances())
    closed_form = budget_mean_variance(mean, covar, GAMMA)
    quadratic = op.opt_maximize_cara(means=mean, covar=covar, carra=GAMMA)
    exponential = op.opt_maximize_cara(means=mean, covar=covar, carra=GAMMA, is_exp=True)
    one = op.opt_maximize_cara_mixture(means=[mean], covars=[covar], probs=np.array([1.0]),
                                       constraints=op.Constraints(is_long_only=True),
                                       carra=GAMMA)
    for weights in (quadratic, exponential, one):
        assert np.abs(weights - closed_form).max() < 1e-4
    assert quoted(closed_form, 3) == [0.760, 0.168, 0.072]

    # The closed form satisfies the first-order condition; no long-only bound binds.
    assert np.ptp(mean - GAMMA * covar @ closed_form) < 1e-12 and closed_form.min() > 0.05
    # The mixture's mean and volatility, against a Monte Carlo sample of the mixture.
    draws = mixture_draws()
    assert np.abs(draws.mean(axis=0) - mean).max() < 0.01
    assert np.abs(draws.std(axis=0) - np.sqrt(np.diag(covar))).max() < 0.01
    assert quoted(mean, 3) == [0.025, 0.064, 0.295]
    assert quoted(np.sqrt(np.diag(covar)), 3) == [0.056, 0.186, 0.800]

    # At gamma* = 1'S^{-1}mu the budget does not bind: w = S^{-1}mu / gamma*.
    inverse_mean = np.linalg.solve(covar, mean)
    gamma_star = inverse_mean.sum()
    free = op.opt_maximize_cara(means=mean, covar=covar, carra=gamma_star)
    assert np.abs(free - inverse_mean / gamma_star).max() < 1e-4
    assert round(gamma_star, 2) == 12.41
    assert quoted(inverse_mean / gamma_star, 3) == [0.816, 0.161, 0.024]

    # Three components: SLSQP agrees with CVXPY, and the crash lowers the crypto weight.
    three = op.opt_maximize_cara_mixture(
        means=list(MEANS), covars=component_covariances(), probs=PROBS,
        constraints=op.Constraints(is_long_only=True), carra=GAMMA)
    assert np.abs(three - cvxpy_mixture_optimum(GAMMA)).max() < 1e-4
    assert quoted(three, 3) == [0.798, 0.139, 0.063]

    # The figure's grid: the crash lowers the crypto weight at every risk aversion, and the
    # weight falls with risk aversion; only at 0.5 would the closed form short bonds.
    grid = allocations(GAMMA_GRID)
    assert (grid['three components'] < grid['one component'] - 0.005).all()
    assert (np.diff(grid['one component']) < 0).all()
    assert (np.diff(grid['three components']) < 0).all()
    assert (grid['three components'] - grid['three components, CVXPY']).abs().max() < 1e-4
    interior = grid['closed form is long-only'].astype(bool)
    assert list(grid.index[~interior]) == [0.5]
    assert (grid['one component'] - grid['one component, closed form'])[
        interior].abs().max() < 1e-4
    assert quoted(grid.loc[0.5, ['one component', 'three components']], 3) == [0.808, 0.724]
    assert quoted(grid.loc[6.0, ['one component', 'three components']], 2) == [0.06, 0.05]
    lowest = op.opt_maximize_cara_mixture(means=[mean], covars=[covar], probs=np.array([1.0]),
                                          constraints=op.Constraints(is_long_only=True),
                                          carra=0.5)
    assert lowest[0] < 1e-6 and round(lowest[1], 3) == 0.192

    # Proposition 1 by Monte Carlo: the closed form is the mean of exp(-gamma R).
    sampled = np.exp(-GAMMA * draws @ three).mean()
    assert abs(sampled / expected_disutility(three, GAMMA) - 1.0) < 0.01
    # Same mean and variance, different certainty equivalents at the one-component optimum.
    gaussian_ce = closed_form @ mean - 0.5 * GAMMA * closed_form @ covar @ closed_form
    assert quoted([gaussian_ce, certainty_equivalent(closed_form, GAMMA),
                   certainty_equivalent(three, GAMMA)], 4) == [0.0341, 0.0319, 0.0324]

    # First-order condition: the optimum is mean-variance under tilted component weights.
    tilt = tilted_probabilities(three, GAMMA)
    tilted_mean = tilt @ MEANS
    tilted_covar = sum(p * c for p, c in zip(tilt, component_covariances()))
    assert np.abs(three - budget_mean_variance(tilted_mean, tilted_covar, GAMMA)).max() < 1e-4
    assert quoted(tilt, 3)[CRASH] == 0.101

    # The tilt is the marginal-utility weighting of the components, E[exp(-gamma R) | k].
    loss = np.exp(-GAMMA * draws @ three)
    starts = np.cumsum([0] + [round(p * N_DRAWS) for p in PROBS])
    by_component = np.array([loss[a:b].mean() for a, b in zip(starts[:-1], starts[1:])])
    assert np.abs(PROBS * by_component / (PROBS @ by_component) - tilt).max() < 0.01
    assert quoted(tilt, 2)[0] == 0.72 and quoted(tilt, 2)[1] == 0.18

    # The objective's scale at the equal-weight start, and what validation accepts.
    start = np.ones(3) / 3
    optimum = cvxpy_mixture_optimum(SCALE_GAMMA)
    assert round(expected_disutility(start, SCALE_GAMMA)) == 844
    assert round(expected_disutility(optimum, SCALE_GAMMA), 2) == 0.81
    success = SimpleNamespace(success=True, status=0, message='Optimization terminated')
    accepted, valid = validate_scipy_solution(start, success, op.Constraints(is_long_only=True),
                                              n=3)
    assert valid and np.array_equal(accepted, start)
    warm = op.opt_maximize_cara_mixture(
        means=list(MEANS), covars=component_covariances(), probs=PROBS, carra=SCALE_GAMMA,
        constraints=op.Constraints(is_long_only=True, weights_0=pd.Series(
            budget_mean_variance(mean, covar, SCALE_GAMMA), index=TICKERS)))
    assert np.abs(warm - optimum).max() < 1e-4

    # A failed solve returns the fallback: zeros without weights_0 or a benchmark. The caps sum
    # to 0.9, so no fully invested portfolio is feasible; this step logs a rejection.
    capped = op.Constraints(is_long_only=True, max_weights=pd.Series(0.3, index=TICKERS))
    fallback = op.opt_maximize_cara_mixture(means=list(MEANS), covars=component_covariances(),
                                            probs=PROBS, constraints=capped, carra=GAMMA)
    assert np.array_equal(fallback, np.zeros(3))
    labelled = op.wrapper_maximize_cara_mixture(
        means=list(MEANS), covars=component_covariances(), probs=PROBS,
        constraints=op.Constraints(is_long_only=True), tickers=TICKERS, carra=GAMMA)
    assert list(labelled.index) == TICKERS and np.allclose(labelled, three, atol=1e-12)

    # Annualisation by scaling: persistent regimes against independent weekly regimes.
    within = sum(p * c for p, c in zip(PROBS, component_covariances()))
    between = between_covariance(PROBS, MEANS)
    assert np.allclose(covar, within + between, atol=1e-15)
    weekly_iid = within + between / 52.0  # annual covariance of 52 independent weekly draws
    crypto_vols = np.sqrt([covar[CRYPTO, CRYPTO], weekly_iid[CRYPTO, CRYPTO]])
    assert quoted(crypto_vols, 2) == [0.80, 0.64]
    independent = cvxpy_weekly_regimes_optimum(GAMMA)
    assert np.abs(independent - budget_mean_variance(mean, weekly_iid, GAMMA)).max() < 5e-3
    assert round(independent[CRYPTO], 2) == 0.11
    weekly = np.exp(exponents(three, GAMMA, scale=52.0))
    assert PROBS @ weekly ** 52 > (PROBS @ weekly) ** 52  # Jensen: scaling is more averse
    assert np.isclose(PROBS @ weekly ** 52, expected_disutility(three, GAMMA), rtol=1e-12)

    # Fit: the fitted mixture keeps the window's mean; scaling adds 52 x 51 times the
    # covariance of the component means.
    returns = simulated_weekly_returns(SEED)
    fitted = op.fit_gaussian_mixture(x=returns.to_numpy(), n_components=3, an_factor=52.0)
    fitted_mean, fitted_covar = mixture_moments(fitted.probs, fitted.means, fitted.covars)
    sample = returns.to_numpy()
    centred = sample - sample.mean(axis=0)
    persistence = 52.0 * 51.0 * between_covariance(fitted.probs, np.array(fitted.means) / 52.0)
    assert np.allclose(fitted_mean, 52.0 * sample.mean(axis=0), atol=1e-12)
    assert np.allclose(fitted_covar, 52.0 * (centred.T @ centred / len(sample)
                                             + 1e-6 * np.eye(3)) + persistence, atol=1e-10)

    # The size of the persistence term here, and the fit's other properties.
    fitted_vol = np.sqrt(fitted_covar[CRYPTO, CRYPTO])
    sample_vol = np.sqrt(52.0 * centred[:, CRYPTO] @ centred[:, CRYPTO] / len(sample))
    assert 2.85 < fitted_vol < 2.95 and round(sample_vol, 2) == 0.68
    assert np.argmax(np.diag(persistence)) == CRYPTO
    assert abs(fitted.probs.sum() - 1.0) < 1e-12
    again = op.fit_gaussian_mixture(x=returns.to_numpy(), n_components=3, an_factor=52.0)
    assert np.array_equal(fitted.probs, again.probs)
    assert all(np.array_equal(a, b) for a, b in zip(fitted.means, again.means))
    assert all(np.array_equal(a, b) for a, b in zip(fitted.covars, again.covars))
    unscaled = op.fit_gaussian_mixture(x=returns.to_numpy(), n_components=3)
    assert all(np.allclose(52.0 * a, b, rtol=1e-12) for a, b in zip(unscaled.covars,
                                                                     fitted.covars))
    single = op.fit_gaussian_mixture(x=returns.to_numpy(), n_components=1, an_factor=52.0)
    assert np.allclose(single.covars[0], 52.0 * (centred.T @ centred / len(sample)
                                                 + 1e-6 * np.eye(3)), atol=1e-12)

    # Rolling: the dispatcher ignores covar_dict and forwards n_mixures as n_components.
    prices = prices_from_returns(returns)
    rolling = op.rolling_maximize_cara_mixture(
        prices=prices, constraints=op.Constraints(is_long_only=True), time_period=None)
    routed = op.compute_rolling_optimal_weights(
        prices=prices, constraints=op.Constraints(is_long_only=True), covar_dict={},
        portfolio_objective=op.PortfolioObjective.MAX_CARA_MIXTURE, roll_window=312,
        n_mixures=3)
    assert np.allclose(rolling, routed, atol=1e-12)

    # One date of the rolling path, rebuilt from its window.
    weekly_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                    freq='W-WED')
    assert list(rolling.index) == expected_rebalancing_dates(weekly_returns.index, 312)
    first = rolling.index[0]
    window = weekly_returns.loc[:first].iloc[-312:]
    params = op.fit_gaussian_mixture(x=window.to_numpy(), n_components=3, an_factor=52.0)
    direct = op.opt_maximize_cara_mixture(means=params.means, covars=params.covars,
                                          probs=params.probs,
                                          constraints=op.Constraints(is_long_only=True),
                                          carra=0.5)
    assert np.allclose(rolling.loc[first].to_numpy(), direct, atol=1e-10)

    # Dates, budget and the defaults and spellings of the routes.
    assert first == pd.Timestamp('2022-01-05') and len(rolling) == 8
    assert np.allclose(rolling.sum(axis=1), 1.0, atol=1e-6) and (rolling > -1e-6).all().all()
    signature = inspect.signature(op.rolling_maximize_cara_mixture).parameters
    assert (signature['roll_window'].default, signature['returns_freq'].default,
            signature['carra'].default, signature['n_components'].default,
            signature['rebalancing_freq'].default) == (312, 'W-WED', 0.5, 3, 'QE')
    assert signature['time_period'].default is inspect.Parameter.empty
    routing = inspect.signature(op.compute_rolling_optimal_weights).parameters
    assert (routing['roll_window'].default, routing['n_mixures'].default,
            routing['carra'].default) == (20, 3, 0.5)
    assert inspect.signature(op.backtest_rolling_optimal_portfolio).parameters[
        'roll_window'].default == 312
    assert op.PortfolioObjective.MAX_CARA_MIXTURE.value == 'MaxCarraMixture'
    assert inspect.signature(op.fit_gaussian_mixture).parameters['n_components'].default == 2
    short = op.compute_rolling_optimal_weights(
        prices=prices, constraints=op.Constraints(is_long_only=True), covar_dict={},
        portfolio_objective=op.PortfolioObjective.MAX_CARA_MIXTURE)
    assert short.index[0] == expected_rebalancing_dates(weekly_returns.index, 20)[0]
    assert short.index[0] == pd.Timestamp('2016-07-06')

    # A missing return drops the asset from every window that contains it.
    gapped = prices.copy()
    gapped.iloc[250, TICKERS.index('Equities')] = np.nan
    gapped_weights = op.rolling_maximize_cara_mixture(
        prices=gapped, constraints=op.Constraints(is_long_only=True), time_period=None)
    kept = window.drop(columns='Equities')
    kept_params = op.fit_gaussian_mixture(x=kept.to_numpy(), n_components=3, an_factor=52.0)
    kept_weights = op.opt_maximize_cara_mixture(
        means=kept_params.means, covars=kept_params.covars, probs=kept_params.probs,
        constraints=op.Constraints(is_long_only=True), carra=0.5)
    assert gapped_weights.loc[first, 'Equities'] == 0.0
    assert np.allclose(gapped_weights.loc[first, ['Bonds', 'Crypto']], kept_weights, atol=1e-10)
    # Caps of 45% on three assets rise to 45% x 3/2 when one asset drops, whatever the flag.
    capped_policy = op.Constraints(is_long_only=True,
                                   max_weights=pd.Series(0.45, index=TICKERS))
    rescaled = [op.rolling_maximize_cara_mixture(
        prices=gapped, constraints=capped_policy, time_period=None,
        optimiser_config=op.OptimiserConfig(apply_total_to_good_ratio=flag)).loc[first]
        for flag in (True, False)]
    assert np.allclose(rescaled[0], rescaled[1], atol=1e-12)
    assert abs(rescaled[0].sum() - 1.0) < 1e-6 and 0.5 <= rescaled[0].max() <= 0.675 + 1e-6
    aligned = capped_policy.update_with_valid_tickers(valid_tickers=['Bonds', 'Crypto'],
                                                      total_to_good_ratio=3 / 2)
    assert np.allclose(aligned.max_weights, 0.675, atol=1e-12)

    # The same prices fitted at monthly frequency give a different annual mixture.
    monthly = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq='ME')
    monthly_fit = op.fit_gaussian_mixture(x=monthly.to_numpy(), n_components=3, an_factor=12.0)
    monthly_covar = mixture_moments(monthly_fit.probs, monthly_fit.means, monthly_fit.covars)[1]
    assert 1.85 < np.sqrt(monthly_covar[CRYPTO, CRYPTO]) < 1.95

    # CARA of a log return is the power utility of the gross return, with relative risk
    # aversion 1 + gamma; the weighted log return is below the portfolio's log return.
    gross = np.exp(sample[:, CRYPTO])
    assert np.allclose(np.exp(-GAMMA * np.log(gross)), gross ** -GAMMA, rtol=1e-12)
    level, step = 1.05, 1e-4
    utility = [-(level + j * step) ** -GAMMA for j in (-1, 0, 1)]
    slope = (utility[2] - utility[0]) / (2 * step)
    curvature = (utility[2] - 2 * utility[1] + utility[0]) / step ** 2
    assert abs(-level * curvature / slope - (1.0 + GAMMA)) < 1e-5
    portfolio_log = np.log(np.exp(sample) @ three)
    assert (portfolio_log >= sample @ three - 1e-15).all()
    # Free parameters of K full-covariance components in N assets.
    n_assets, n_components = len(TICKERS), len(COMPONENTS)
    free_parameters = n_components * (n_assets + n_assets * (n_assets + 1) // 2) + n_components - 1
    counted = len(fitted.probs) - 1 + sum(m.size for m in fitted.means) + sum(
        np.triu(c).nonzero()[0].size for c in fitted.covars)
    assert free_parameters == counted == 29

    print('cara_gaussian_mixture: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: the crypto weight against risk aversion, and the allocation.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, PercentFormatter

    table = allocations(GAMMA_GRID)
    mean, covar = mixture_moments(PROBS, MEANS, component_covariances())
    one = op.opt_maximize_cara_mixture(means=[mean], covars=[covar], probs=np.array([1.0]),
                                       constraints=op.Constraints(is_long_only=True),
                                       carra=GAMMA)
    three = op.opt_maximize_cara_mixture(
        means=list(MEANS), covars=component_covariances(), probs=PROBS,
        constraints=op.Constraints(is_long_only=True), carra=GAMMA)
    one_series, three_series = table['one component'], table['three components']
    interior = table['closed form is long-only'].astype(bool)
    checks = {
        'one_component_matches_closed_form': bool(
            interior.sum() >= 5 and (one_series - table['one component, closed form'])[
                interior].abs().max() < 1e-4
            and np.abs(one - budget_mean_variance(mean, covar, GAMMA)).max() < 1e-4),
        'three_components_match_cvxpy': bool(
            (three_series - table['three components, CVXPY']).abs().max() < 1e-4
            and np.abs(three - cvxpy_mixture_optimum(GAMMA)).max() < 1e-4),
        'crash_lowers_crypto_weight': bool((three_series < one_series - 0.005).all()),
        'crypto_weight_falls_with_gamma': bool((np.diff(one_series) < 0).all()
                                               and (np.diff(three_series) < 0).all()),
    }

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange = '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface,
                                      gridspec_kw={'width_ratios': [1.35, 1.0]})
    gammas = table.index.to_numpy()
    left.plot(gammas, one_series, color=blue, linewidth=2.2, marker='o', markersize=3.5)
    left.plot(gammas, three_series, color=orange, linewidth=2.2, marker='o', markersize=3.5)
    left.axvline(GAMMA, color=muted, linestyle=':', linewidth=1.2)
    left.text(GAMMA * 0.96, 0.78, f'γ = {GAMMA:g},\nright panel', ha='right', va='top',
              color=ink, fontsize=9)
    left.text(0.62, table.loc[0.6, 'one component'] + 0.04, 'one component', color=ink,
              fontsize=10, ha='left', va='bottom')
    left.text(0.52, 0.30, 'three components', color=ink, fontsize=10, ha='left', va='top')
    left.set_xscale('log')
    left.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3, 4, 5, 6]))
    left.xaxis.set_minor_locator(FixedLocator([]))
    left.set_xticklabels(['0.5', '1', '2', '3', '4', '5', '6'])
    left.set_xlabel('Risk aversion γ')
    left.set_ylabel('Weight of crypto')
    left.set_ylim(0.0, 0.85)
    left.set_title('Crypto weight against risk aversion', loc='left', color=ink)

    x = np.arange(len(TICKERS))
    width = 0.38
    bars_one = right.bar(x - width / 2 - 0.01, one, width, color=blue, label='one component')
    bars_three = right.bar(x + width / 2 + 0.01, three, width, color=orange,
                           label='three components')
    for bars in (bars_one, bars_three):
        for bar in bars:
            right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                       f'{bar.get_height():.1%}', ha='center', va='bottom', color=ink,
                       fontsize=9)
    right.set_xticks(x, TICKERS)
    right.set_ylabel('Weight')
    right.set_ylim(0.0, 0.95)
    right.legend(frameon=False, loc='upper right', fontsize=10, labelcolor=ink)
    right.set_title(f'All weights at γ = {GAMMA:g}', loc='left', color=ink)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
