"""Canonical script of docs/mean_variance_objectives.md.

The page shows excerpts of this file; every number and property it states is asserted here
against a reference computed a different way: the closed forms of minimum variance, the tangency
portfolio and the utility portfolio from explicit linear solves, the frontier variance from its
three scalars and from separate CVXPY solves, a hand-written EWMA recursion and hand-written
CVXPY programs for the estimated-means path. The script runs offline after
``pip install optimalportfolios`` and needs no data file:

    python -m examples.docs.mean_variance_objectives

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qis
from factorlasso import CurrentFactorCovarData

import optimalportfolios as op

TICKERS = ['Govt', 'Credit', 'US eq', 'EM eq', 'Gold']
# Annual expected excess returns, volatilities and correlations of a stylised universe.
MEANS = [0.005, 0.020, 0.045, 0.055, 0.025]
VOLS = [0.05, 0.07, 0.15, 0.19, 0.15]
CORR = [
    [1.00, 0.30, -0.20, -0.10, 0.10],
    [0.30, 1.00, 0.30, 0.25, 0.10],
    [-0.20, 0.30, 1.00, 0.50, 0.00],
    [-0.10, 0.25, 0.50, 1.00, 0.20],
    [0.10, 0.10, 0.00, 0.20, 1.00],
]
GAMMAS = [5.0, 10.0, 20.0, 50.0]  # risk aversions of the utility grid
SHOWN_GAMMA = 5.0  # the utility portfolio whose weights the figure shows
RISK_FREE = 0.02  # a cash rate added to the means to show the total-return mistake
VOL_CAP = 0.12  # a volatility cap twice the tangency volatility
SEED = 5
START, END = '2015-01-01', '2024-12-31'  # simulated business-daily prices
REBALANCING = ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31']
HELD = 1e-4  # a weight above this is a held asset


def covariance(vols, corr, tickers) -> pd.DataFrame:
    """Return the covariance matrix with the given volatilities and correlations."""
    vols = np.asarray(vols, dtype=float)
    return pd.DataFrame(np.outer(vols, vols) * np.asarray(corr, dtype=float),
                        index=tickers, columns=tickers)


def closed_forms(covar: np.ndarray, means: np.ndarray) -> tuple:
    """Return A, B, C and the minimum-variance and tangency portfolios by linear solves."""
    ones = np.ones(len(means))
    inv_ones = np.linalg.solve(covar, ones)
    inv_means = np.linalg.solve(covar, means)
    a, b, c = ones @ inv_ones, ones @ inv_means, means @ inv_means
    return a, b, c, inv_ones / a, inv_means / b


def frontier_variance(m: float, a: float, b: float, c: float) -> float:
    """Minimum variance of a fully invested portfolio with expected return m."""
    return (a * m * m - 2.0 * b * m + c) / (a * c - b * b)


def sharpe(weights, covar: np.ndarray, means: np.ndarray) -> float:
    """Ratio of expected excess return to volatility."""
    w = np.asarray(weights, dtype=float)
    return float(means @ w / np.sqrt(w @ covar @ w))


def simulated_prices(covar: pd.DataFrame, drift: np.ndarray, seed: int) -> pd.DataFrame:
    """Simulate business-daily prices with the given annual log drift and covariance.

    The draws use the Cholesky factor, which is unique, rather than
    ``Generator.multivariate_normal``, whose SVD factor differs between LAPACK builds, so the
    same seed gives the same prices on every platform.
    """
    dates = pd.bdate_range(START, END)
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(covar.to_numpy() / 260.0)
    log_returns = rng.standard_normal((len(dates), len(covar))) @ factor.T + drift / 260.0
    return pd.DataFrame(100.0 * np.exp(np.cumsum(log_returns, axis=0)), index=dates,
                        columns=covar.columns)


def ewma_means_by_hand(prices: pd.DataFrame, dates: list, span: int) -> pd.DataFrame:
    """Annualised EWMA of Wednesday-to-Wednesday log returns, seeded at the first return."""
    weekly = prices.resample('W-WED').last()
    weekly = weekly.loc[weekly.index <= prices.index[-1]]
    log_returns = np.log(weekly / weekly.shift(1)).iloc[1:]
    decay = 1.0 - 2.0 / (span + 1.0)
    state = log_returns.iloc[0].to_numpy()
    states = []
    for row in log_returns.to_numpy():
        state = decay * state + (1.0 - decay) * row
        states.append(state)
    path = pd.DataFrame(52.0 * np.array(states), index=log_returns.index, columns=prices.columns)
    return path.reindex(pd.DatetimeIndex(dates), method='ffill')


def long_only_max_sharpe(covar: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Maximise the Sharpe ratio over long-only budgets: min y'Sy with m'y = 1, then normalise."""
    y = cp.Variable(len(means), nonneg=True)
    cp.Problem(cp.Minimize(cp.quad_form(y, covar)), [means @ y == 1.0]).solve(solver='CLARABEL')
    return y.value / y.value.sum()


def long_only_utility(covar: np.ndarray, means: np.ndarray, gamma: float) -> np.ndarray:
    """Maximise m'w - gamma/2 w'Sw over long-only fully invested weights with CVXPY."""
    w = cp.Variable(len(means), nonneg=True)
    objective = cp.Maximize(means @ w - 0.5 * gamma * cp.quad_form(w, covar))
    cp.Problem(objective, [cp.sum(w) == 1.0]).solve(solver='CLARABEL')
    return w.value


def frontier_by_cvxpy(covar: np.ndarray, means: np.ndarray, m: float) -> float:
    """Minimum variance at expected return m, solved directly with CVXPY."""
    x = cp.Variable(len(means))
    cp.Problem(cp.Minimize(cp.quad_form(x, covar)),
               [cp.sum(x) == 1.0, means @ x == m]).solve(solver='CLARABEL')
    return float(x.value @ covar @ x.value)


def identity_factor_model(covar: pd.DataFrame) -> CurrentFactorCovarData:
    """Wrap a covariance as a factor model with unit loadings and no residual risk."""
    tickers = covar.index
    return CurrentFactorCovarData(
        x_covar=covar,
        y_betas=pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers),
        y_variances=pd.DataFrame({'residual_var': np.zeros(len(tickers))}, index=tickers))


def main() -> None:
    """Run the worked example of the page and assert every quoted number."""
    covar = covariance(VOLS, CORR, TICKERS)
    means = pd.Series(MEANS, index=TICKERS)
    sigma, mu = covar.to_numpy(), means.to_numpy()
    config = op.OptimiserConfig(apply_total_to_good_ratio=False)
    long_only = op.Constraints(is_long_only=True)
    A, B, C, w_mv, w_tan = closed_forms(sigma, mu)
    print(round(A, 1), round(B, 2), round(C, 4), round(np.sqrt(C), 3))

    # The three scalars and the two closed-form portfolios.
    assert round(A, 1) == 559.8 and round(B, 2) == 6.96 and round(C, 4) == 0.1729
    assert round(np.sqrt(C), 3) == 0.416
    assert (w_mv > 0.005).all() and (w_tan > 0.05).all()  # every long-only bound is slack
    assert abs(w_mv @ sigma @ w_mv - 1.0 / A) < 1e-15 and abs(mu @ w_mv - B / A) < 1e-15
    assert abs(mu @ w_tan - C / B) < 1e-15
    assert abs(sharpe(w_tan, sigma, mu) - np.sqrt(C)) < 1e-12
    assert round(np.sqrt(1.0 / A), 4) == 0.0423 and round(B / A, 4) == 0.0124
    assert round(np.sqrt(w_tan @ sigma @ w_tan), 4) == 0.0597 and round(C / B, 4) == 0.0248
    assert round(sharpe(w_mv, sigma, mu), 3) == 0.294
    # Cauchy-Schwarz: no fully invested portfolio, long-only or not, beats the tangency ratio
    # or the minimum variance.
    rng = np.random.default_rng(SEED)
    draws = np.vstack([rng.dirichlet(np.ones(len(TICKERS)), size=2000),
                       rng.standard_normal((2000, len(TICKERS)))])
    draws = draws / draws.sum(axis=1, keepdims=True)
    assert all(sharpe(w, sigma, mu) <= np.sqrt(C) + 1e-12 for w in draws)
    assert all(w @ sigma @ w >= 1.0 / A - 1e-15 for w in draws)

    min_var, min_var_outcome = op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=long_only, optimiser_config=config)
    assert min_var_outcome.accepted and np.abs(min_var.to_numpy() - w_mv).max() < 1e-4
    print(min_var.round(3).tolist())

    # Minimum variance is the default objective of the quadratic wrapper.
    assert min_var.round(3).tolist() == [0.711, 0.13, 0.104, 0.008, 0.048]
    w = min_var.to_numpy()
    assert abs(w @ sigma @ w * A - 1.0) < 1e-6

    fixed, fixed_outcome = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=means, constraints=long_only, optimiser_config=config)
    band = op.Constraints(is_long_only=True, min_exposure=0.8, max_exposure=1.0)
    banded, banded_outcome = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=means, constraints=band, optimiser_config=config)
    print(fixed_outcome.solver, banded_outcome.solver)
    print(fixed.round(3).tolist())
    assert np.abs(fixed.to_numpy() - w_tan).max() < 1e-6
    assert np.abs(banded.to_numpy() / banded.sum() - w_tan).max() < 1e-4

    # Both routes are accepted and reach the ratio of the closed form.
    assert (fixed_outcome.solver, banded_outcome.solver) == ('CLARABEL', 'SLSQP')
    assert fixed_outcome.accepted and banded_outcome.accepted
    assert fixed.round(3).tolist() == [0.285, 0.304, 0.199, 0.102, 0.11]
    assert 0.8 - 1e-6 <= banded.sum() <= 1.0 + 1e-6
    assert abs(sharpe(banded, sigma, mu) - np.sqrt(C)) < 1e-8
    # The ratio does not change when the weights are scaled, so the band fixes no exposure.
    assert abs(sharpe(0.8 * w_tan, sigma, mu) - sharpe(w_tan, sigma, mu)) < 1e-15
    # Weekly means and covariance scale the ratio by 1/sqrt(52) and keep the portfolio.
    weekly, _ = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar / 52.0, means=means / 52.0, constraints=long_only, optimiser_config=config)
    assert np.abs(weekly.to_numpy() - w_tan).max() < 1e-6
    assert abs(sharpe(w_tan, sigma / 52.0, mu / 52.0) * np.sqrt(52.0) - np.sqrt(C)) < 1e-12
    # A fixed exposure of 0.5 keeps the Charnes-Cooper route and scales the tangency portfolio.
    half = op.Constraints(is_long_only=True, min_exposure=0.5, max_exposure=0.5)
    half_weights, half_outcome = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=means, constraints=half, optimiser_config=config)
    assert half_outcome.solver == 'CLARABEL' and half_outcome.accepted
    assert np.abs(half_weights.to_numpy() - 0.5 * w_tan).max() < 1e-6

    utility_weights = {}
    for gamma in GAMMAS:
        utility, outcome = op.wrapper_quadratic_optimisation(
            pd_covar=covar, constraints=long_only,
            portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
            means=means, carra=gamma, optimiser_config=config)
        w, theta = utility.to_numpy(), B / gamma
        assert np.abs(w - ((1.0 - theta) * w_mv + theta * w_tan)).max() < 1e-5
        assert abs(w @ sigma @ w / frontier_variance(mu @ w, A, B, C) - 1.0) < 1e-6
        assert abs(frontier_by_cvxpy(sigma, mu, mu @ w) / (w @ sigma @ w) - 1.0) < 1e-6
        utility_weights[f'gamma {gamma:g}'] = utility
    grid = pd.DataFrame(utility_weights)
    at_b, _ = op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=long_only,
        portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
        means=means, carra=B, optimiser_config=config)
    assert np.abs(at_b.to_numpy() - fixed.to_numpy()).max() < 1e-6

    # The grid stays inside the long-only bounds; the figure shows the first portfolio.
    assert (grid > 0.005).all().all()
    shown = grid[f'gamma {SHOWN_GAMMA:g}'].to_numpy()
    assert round(B / SHOWN_GAMMA, 2) == 1.39
    assert shown.round(3).tolist() == [0.117, 0.373, 0.236, 0.14, 0.134]
    assert round(np.sqrt(shown @ sigma @ shown), 4) == 0.0724 and round(mu @ shown, 4) == 0.0297
    # Units: weekly means and covariance keep gamma; returns in percent need gamma / 100.
    units = ((1.0 / 52.0, 1.0 / 52.0, SHOWN_GAMMA), (100.0, 1e4, SHOWN_GAMMA / 100.0))
    for mean_scale, covar_scale, gamma in units:
        rescaled, _ = op.wrapper_quadratic_optimisation(
            pd_covar=covar * covar_scale, constraints=long_only,
            portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
            means=means * mean_scale, carra=gamma, optimiser_config=config)
        assert np.abs(rescaled.to_numpy() - shown).max() < 1e-4

    ones = np.ones(len(TICKERS))
    budgeted = op.solve_analytic_log_opt(sigma, mu, exposure_budget_eq=(ones, 1.0),
                                         gamma=SHOWN_GAMMA)
    unbudgeted = op.solve_analytic_log_opt(sigma, mu, gamma=SHOWN_GAMMA)
    wide = op.Constraints(is_long_only=False, min_exposure=-10.0, max_exposure=10.0)
    free, _ = op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=wide,
        portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
        means=means, carra=SHOWN_GAMMA, optimiser_config=config)
    print(round(unbudgeted.sum(), 3), round(free.sum(), 3))
    assert np.abs(budgeted - shown).max() < 1e-5
    assert np.abs(unbudgeted - (B / SHOWN_GAMMA) * w_tan).max() < 1e-12

    # The solver agrees with the unbudgeted closed form, which invests B / gamma of capital.
    assert np.abs(free.to_numpy() - unbudgeted).max() < 1e-5
    assert round(unbudgeted.sum(), 3) == 1.393 and round(free.sum(), 3) == 1.393

    total, _ = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=means + RISK_FREE, constraints=long_only, optimiser_config=config)
    print(total.round(3).tolist())

    # Total returns give the frontier portfolio touched by a line from minus the cash rate.
    reference = np.linalg.solve(sigma, mu + RISK_FREE)
    assert np.abs(total.to_numpy() - reference / reference.sum()).max() < 1e-6
    assert total.round(3).tolist() == [0.547, 0.197, 0.14, 0.044, 0.072]
    t = total.to_numpy()
    assert abs(t @ sigma @ t / frontier_variance(mu @ t, A, B, C) - 1.0) < 1e-6
    assert B / A < mu @ t < C / B  # between minimum variance and the tangency
    assert round(sharpe(t, sigma, mu), 2) == 0.38

    capped = op.Constraints(is_long_only=True, max_target_portfolio_vol_an=VOL_CAP)
    cap_weights, cap_outcome = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=means, constraints=capped, optimiser_config=config)
    print(cap_outcome.accepted, cap_outcome.status, cap_outcome.fallback_source)

    # The cap is slack for the tangency portfolio, yet the transformed problem is infeasible.
    assert np.sqrt(w_tan @ sigma @ w_tan) < VOL_CAP / 2.0 + 1e-3
    assert (cap_outcome.accepted, cap_outcome.status) == (False, 'infeasible')
    assert cap_outcome.fallback_source == 'zeros' and (cap_weights == 0.0).all()
    # Negated means: no long-only portfolio has a positive expected return.
    _, negative = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=-means, constraints=long_only, optimiser_config=config)
    assert (negative.accepted, negative.status) == (False, 'infeasible')
    # A long-short book without bounds and B < 0: the transformed program drives k to zero.
    tilted = means.copy()
    tilted['Govt'] = -0.05
    assert np.ones(len(TICKERS)) @ np.linalg.solve(sigma, tilted.to_numpy()) < 0.0
    unbounded, unbounded_outcome = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=tilted, constraints=op.Constraints(is_long_only=False),
        optimiser_config=config)
    assert not unbounded_outcome.accepted or unbounded.abs().max() > 100.0
    # Means without the objective: the quadratic wrapper still solves minimum variance.
    forgot, _ = op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=long_only, means=means, carra=SHOWN_GAMMA,
        optimiser_config=config)
    assert np.abs(forgot.to_numpy() - min_var.to_numpy()).max() < 1e-8
    # ... but a non-finite mean still removes its asset, even for minimum variance.
    gaps = means.where(means.index != 'Gold')
    no_gold, _ = op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=long_only, means=gaps, optimiser_config=config)
    reduced = closed_forms(sigma[:4, :4], mu[:4])[3]
    assert no_gold['Gold'] == 0.0 and np.abs(no_gold.to_numpy()[:4] - reduced).max() < 1e-4
    try:
        op.cvx_quadratic_optimisation(op.PortfolioObjective.MAXIMUM_SHARPE_RATIO, sigma,
                                      long_only, means=mu)
        raise AssertionError('the quadratic solver accepted a Sharpe objective')
    except ValueError:
        pass

    prices = simulated_prices(covar, np.asarray(MEANS), seed=SEED)
    dates = [pd.Timestamp(date) for date in REBALANCING]
    covar_dict = {date: covar for date in dates}
    estimated = op.estimate_rolling_ewma_means(prices=prices, rebalancing_dates=dates)
    print(estimated.round(3))

    # The package's means equal a hand-written recursion on Wednesday log returns.
    assert np.abs(estimated - ewma_means_by_hand(prices, dates, span=52)).max().max() < 1e-12
    december = estimated.loc[pd.Timestamp('2024-12-31')]
    assert [round(december[name], 3) for name in ('Govt', 'US eq', 'EM eq')] == [
        -0.081, 0.310, 0.325]
    for name in ('Govt', 'US eq', 'EM eq'):  # one to two standard errors, each about the vol
        z = abs(december[name] - means[name]) / VOLS[TICKERS.index(name)]
        assert 1.0 < z < 2.0
    early = op.estimate_rolling_ewma_means(prices=prices, rebalancing_dates=['2015-01-05'])
    assert early.isna().all().all()  # a date before the first weekly return is missing
    # Standard error of an annualised EWMA mean: 52 sigma_w sqrt(sum of squared weights).
    impulse = np.zeros(4000)
    impulse[0] = 1.0
    kernel = qis.compute_ewm(np.concatenate([np.zeros(1), impulse]), span=52)[1:]
    assert abs((kernel ** 2).sum() - 1.0 / 52.0) < 1e-12
    assert abs(52.0 * (0.15 / np.sqrt(52.0)) * np.sqrt((kernel ** 2).sum()) - 0.15) < 1e-12

    dispatched = op.compute_rolling_optimal_weights(
        prices=prices, constraints=long_only, covar_dict=covar_dict,
        portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY, optimiser_config=config)
    direct = op.rolling_quadratic_optimisation(
        prices=prices, constraints=long_only, covar_dict=covar_dict,
        portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
        expected_returns=estimated, carra=0.5, optimiser_config=config)
    tangency_path = op.rolling_maximize_portfolio_sharpe(
        prices=prices, expected_returns=estimated, constraints=long_only,
        covar_dict=covar_dict, optimiser_config=config)
    print(dispatched.round(2))
    print(tangency_path.round(2))
    assert np.abs(dispatched.to_numpy() - direct.to_numpy()).max() < 1e-8

    # The dispatcher also routes maximum Sharpe; both paths match hand-written programs.
    routed = op.compute_rolling_optimal_weights(
        prices=prices, constraints=long_only, covar_dict=covar_dict,
        portfolio_objective=op.PortfolioObjective.MAXIMUM_SHARPE_RATIO, optimiser_config=config)
    assert np.abs(routed.to_numpy() - tangency_path.to_numpy()).max() < 1e-8
    for date in dates:
        mu_t = estimated.loc[date].to_numpy()
        assert np.abs(direct.loc[date].to_numpy()
                      - long_only_utility(sigma, mu_t, 0.5)).max() < 1e-5
        assert np.abs(tangency_path.loc[date].to_numpy()
                      - long_only_max_sharpe(sigma, mu_t)).max() < 1e-5
    assert list(dispatched.index) == dates and list(dispatched.columns) == TICKERS
    assert ((dispatched > HELD).sum(axis=1) == 1).all()  # one asset at every date
    assert (dispatched.idxmax(axis=1) == estimated.idxmax(axis=1)).all()
    assert dispatched.round(2).to_numpy().tolist() == [
        [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]
    assert tangency_path.round(2).to_numpy().tolist() == [
        [0.4, 0.0, 0.6, 0.0, 0.0], [0.0, 0.0, 0.8, 0.0, 0.2],
        [0.0, 0.0, 0.79, 0.21, 0.0], [0.0, 0.0, 0.69, 0.31, 0.0]]
    assert (tangency_path['Credit'] < HELD).all() and round(w_tan[1], 3) == 0.304
    try:
        op.rolling_quadratic_optimisation(
            prices=prices, constraints=long_only, covar_dict=covar_dict,
            portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY)
        raise AssertionError('utility without expected returns did not raise')
    except ValueError:
        pass

    result = op.PortfolioOptimisationResult(
        weights=grid, benchmark_weights=min_var.rename('minimum variance'),
        covar_data=identity_factor_model(covar), group_attributions={}, expected_return=means)
    figure = op.plot_efficient_frontier(result, profiles={'utility': list(grid.columns)})
    points, _ = result.compute_efficient_frontier_data(profiles={'utility': list(grid.columns)})

    # The plot draws the given portfolios at their volatility and expected return.
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    drawn = points[points['hue'] == 'utility - portfolio'].set_index('mandate')
    for name in grid.columns:
        w = grid[name].to_numpy()
        assert abs(drawn.loc[name, 'total_vol'] - np.sqrt(w @ sigma @ w)) < 1e-12
        assert abs(drawn.loc[name, 'exp_return'] - mu @ w) < 1e-12
    print('mean_variance_objectives: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: the frontier with the three objectives, and their weights.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')

    covar = covariance(VOLS, CORR, TICKERS)
    means = pd.Series(MEANS, index=TICKERS)
    sigma, mu = covar.to_numpy(), means.to_numpy()
    config = op.OptimiserConfig(apply_total_to_good_ratio=False)
    long_only = op.Constraints(is_long_only=True)
    A, B, C, w_mv, w_tan = closed_forms(sigma, mu)
    min_var, _ = op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=long_only, optimiser_config=config)
    tangency, _ = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=means, constraints=long_only, optimiser_config=config)
    utilities = {}
    for gamma in GAMMAS:
        utilities[gamma], _ = op.wrapper_quadratic_optimisation(
            pd_covar=covar, constraints=long_only,
            portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
            means=means, carra=gamma, optimiser_config=config)

    def point(weights: pd.Series) -> tuple:
        """Volatility and expected excess return of the weights."""
        w = weights.to_numpy()
        return float(np.sqrt(w @ sigma @ w)), float(mu @ w)

    rows = {'minimum variance': point(min_var), 'maximum Sharpe': point(tangency)}
    rows.update({f'utility, gamma = {gamma:g}': point(w) for gamma, w in utilities.items()})
    table = pd.DataFrame(rows, index=['volatility', 'expected_return']).T
    table['frontier_volatility'] = [np.sqrt(frontier_variance(m, A, B, C))
                                    for m in table['expected_return']]
    weights = pd.DataFrame({'minimum variance': min_var, 'maximum Sharpe': tangency,
                            f'utility, gamma = {SHOWN_GAMMA:g}': utilities[SHOWN_GAMMA]})

    ink, muted, grid_colour, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange, aqua, grey = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b3'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid_colour,
                         'axes.labelcolor': muted, 'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)

    returns = np.linspace(0.0035, 0.0335, 400)
    vols = np.sqrt([frontier_variance(m, A, B, C) for m in returns])
    upper = returns >= B / A
    left.plot(100 * vols[upper], 100 * returns[upper], color=muted, linewidth=1.6)
    left.plot(100 * vols[~upper], 100 * returns[~upper], color=grey, linewidth=1.4,
              linestyle='--')
    slope = np.sqrt(C)
    x_line = np.array([3.4, 8.0])
    left.plot(x_line, slope * x_line, color=grey, linewidth=1.1, linestyle=':')
    left.text(3.55, 2.8, f'dotted: line from the origin,\nslope {slope:.3f} = maximum\n'
              'Sharpe ratio', ha='left', va='bottom', color=ink, fontsize=9.5)
    for gamma in GAMMAS:
        vol, ret = rows[f'utility, gamma = {gamma:g}']
        size = 70 if gamma == SHOWN_GAMMA else 38
        left.scatter(100 * vol, 100 * ret, s=size, color=aqua, zorder=3,
                     edgecolor=surface, linewidth=0.8)
        left.text(100 * vol + 0.12, 100 * ret - 0.03, f'γ = {gamma:g}', ha='left', va='top',
                  color=ink, fontsize=9.5)
    for name, colour, offset in (('minimum variance', blue, (0.12, -0.07)),
                                 ('maximum Sharpe', orange, (0.14, -0.13))):
        vol, ret = rows[name]
        left.scatter(100 * vol, 100 * ret, s=70, color=colour, zorder=4, edgecolor=surface,
                     linewidth=0.8)
        left.text(100 * vol + offset[0], 100 * ret + offset[1], name, ha='left',
                  va='center', color=ink, fontsize=10)
    left.text(4.8, 0.75, 'inefficient branch', ha='left', va='center', color=ink, fontsize=9.5)
    left.set_xlim(3.4, 8.0)
    left.set_ylim(0.4, 3.4)
    left.set_xlabel('Volatility, % a year')
    left.set_ylabel('Expected excess return, % a year')
    left.set_title('The frontier and the three objectives', loc='left', color=ink)

    x = np.arange(len(TICKERS))
    width = 0.26
    for k, (name, colour) in enumerate(zip(weights.columns, (blue, orange, aqua))):
        right.bar(x + (k - 1) * (width + 0.01), 100 * weights[name].to_numpy(), width,
                  color=colour, label=name.replace('gamma', 'γ'))
    right.set_xticks(x, TICKERS)
    right.set_ylabel('Weight, %')
    right.set_ylim(0.0, 80.0)
    right.legend(frameon=False, loc='upper right', fontsize=10, labelcolor=ink)
    right.set_title('Weights of the three portfolios', loc='left', color=ink)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.grid(axis='y', color=grid_colour, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    left.grid(axis='x', color=grid_colour, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    shown = utilities[SHOWN_GAMMA].to_numpy()
    checks = {
        'min_variance_matches_closed_form': bool(np.abs(min_var.to_numpy() - w_mv).max() < 1e-4),
        'max_sharpe_matches_tangency': bool(np.abs(tangency.to_numpy() - w_tan).max() < 1e-6),
        'utility_points_on_frontier': bool(np.allclose(table['volatility'],
                                                       table['frontier_volatility'],
                                                       rtol=1e-6)),
        'tangency_ratio_is_slope': bool(abs(sharpe(tangency, sigma, mu) - slope) < 1e-8),
        'shown_utility_is_two_fund_mix': bool(np.abs(
            shown - ((1 - B / SHOWN_GAMMA) * w_mv + B / SHOWN_GAMMA * w_tan)).max() < 1e-5),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
