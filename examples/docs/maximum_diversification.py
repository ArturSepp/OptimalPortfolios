"""Canonical script of docs/maximum_diversification.md.

The page shows excerpts of this file; every number it quotes is asserted here against a
reference computed a different way. The script runs offline after ``pip install
optimalportfolios`` and needs no data file:

    python -m examples.docs.maximum_diversification

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import cvxpy as cp
import numpy as np
import pandas as pd
import qis

import optimalportfolios as op

TICKERS = ['Govt', 'Credit', 'US eq', 'Intl eq', 'EM eq', 'Cmdty']
# Annual volatilities and correlations of a stylised six-asset universe.
VOLS = np.array([0.05, 0.07, 0.16, 0.17, 0.21, 0.24])
CORR = np.array([
    [1.00, 0.50, -0.20, -0.20, -0.15, -0.05],
    [0.50, 1.00, 0.40, 0.40, 0.45, 0.20],
    [-0.20, 0.40, 1.00, 0.85, 0.75, 0.30],
    [-0.20, 0.40, 0.85, 1.00, 0.80, 0.35],
    [-0.15, 0.45, 0.75, 0.80, 1.00, 0.40],
    [-0.05, 0.20, 0.30, 0.35, 0.40, 1.00],
])
HELD = 1e-4  # a weight above this is a held asset
SEED = 7


def covariance(vols: np.ndarray, corr: np.ndarray, tickers: list) -> pd.DataFrame:
    """Return the covariance matrix with the given volatilities and correlations."""
    return pd.DataFrame(np.outer(vols, vols) * corr, index=tickers, columns=tickers)


def correlation_minimum_variance(corr: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """Solve min y'Cy on the simplex with CVXPY and rescale by 1/vol: the identity's route."""
    y = cp.Variable(len(vols))
    cp.Problem(cp.Minimize(cp.quad_form(y, corr)), [cp.sum(y) == 1, y >= 0]).solve(
        solver='CLARABEL')
    weights = np.where(y.value > 1e-9, y.value, 0.0) / vols
    return weights / weights.sum()


def correlations_with_portfolio(covar: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the correlation of each asset with the portfolio."""
    vols = np.sqrt(np.diag(covar))
    return covar @ weights / (vols * np.sqrt(weights @ covar @ weights))


def simulated_prices(covar: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Simulate business-daily prices with the given annual covariance and zero drift."""
    dates = pd.bdate_range('2015-01-01', '2025-12-31')
    rng = np.random.default_rng(seed)
    returns = rng.multivariate_normal(np.zeros(len(covar)), covar.to_numpy() / 260.0,
                                      size=len(dates))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=covar.columns)


def main() -> None:
    """Run the worked example of the page and assert every quoted number."""
    # Two assets: the most diversified portfolio holds them in inverse proportion to volatility.
    two = covariance(np.array([0.05, 0.20]), np.array([[1.0, 0.5], [0.5, 1.0]]), ['A', 'B'])
    weights = op.wrapper_maximise_diversification(
        pd_covar=two, constraints=op.Constraints(is_long_only=True))
    inverse_vol = np.array([1 / 0.05, 1 / 0.20])
    assert np.allclose(weights, inverse_vol / inverse_vol.sum(), atol=1e-4)
    ratio = op.calculate_diversification_ratio(w=weights.to_numpy(), covar=two.to_numpy())
    assert abs(ratio - np.sqrt(2.0 / (1.0 + 0.5))) < 1e-6

    # Six assets: the SLSQP solution equals the minimum-variance portfolio of the correlation
    # matrix, rescaled by 1/vol, which CVXPY computes independently.
    covar = covariance(VOLS, CORR, TICKERS)
    mdp = op.wrapper_maximise_diversification(
        pd_covar=covar, constraints=op.Constraints(is_long_only=True))
    reference = correlation_minimum_variance(CORR, VOLS)
    assert np.abs(mdp.to_numpy() - reference).max() < 1e-4
    assert round(mdp['Govt'], 2) == 0.72 and mdp['Credit'] < HELD

    # Every held asset has correlation 1/DR with the portfolio; an excluded asset, at least that.
    ratio = op.calculate_diversification_ratio(w=mdp.to_numpy(), covar=covar.to_numpy())
    rho = correlations_with_portfolio(covar.to_numpy(), mdp.to_numpy())
    held = mdp.to_numpy() > HELD
    assert np.allclose(rho[held], 1.0 / ratio, atol=1e-4)
    assert (rho[~held] > 1.0 / ratio).all()
    assert round(ratio, 2) == 1.75 and round(1.0 / ratio, 3) == 0.570
    assert round(rho[TICKERS.index('Credit')], 2) == 0.70

    # A 30% cap binds on Govt and brings Credit in; the capped assets no longer share one
    # correlation with the portfolio, so the identity does not hold under the cap.
    caps = pd.Series(0.30, index=TICKERS)
    capped = op.wrapper_maximise_diversification(
        pd_covar=covar, constraints=op.Constraints(is_long_only=True, max_weights=caps))
    assert abs(capped['Govt'] - 0.30) < 1e-4 and capped['Credit'] > 0.05
    rho_capped = correlations_with_portfolio(covar.to_numpy(), capped.to_numpy())
    held_capped = rho_capped[capped.to_numpy() > HELD]
    assert held_capped.max() - held_capped.min() > 0.05

    # Rolling: quarterly EWMA covariances of simulated prices and the dispatcher.
    prices = simulated_prices(covar, seed=SEED)
    covar_dict = op.EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE') \
        .fit_rolling_covars(prices=prices, time_period=qis.TimePeriod('2017-12-31', '2025-12-31'))
    rolling = op.compute_rolling_optimal_weights(
        prices=prices, constraints=op.Constraints(is_long_only=True), covar_dict=covar_dict,
        portfolio_objective=op.PortfolioObjective.MAX_DIVERSIFICATION)
    assert np.allclose(rolling.sum(axis=1), 1.0, atol=1e-6) and (rolling >= 0.0).all().all()
    for date, estimate in covar_dict.items():
        w = rolling.loc[date].to_numpy()
        rho = correlations_with_portfolio(estimate.to_numpy(), w)
        dr = op.calculate_diversification_ratio(w=w, covar=estimate.to_numpy())
        assert np.allclose(rho[w > HELD], 1.0 / dr, atol=2e-3)


def exhibit(path) -> dict:
    """Draw the page's figure: the two routes to the weights, and the correlation property.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    covar = covariance(VOLS, CORR, TICKERS)
    mdp = op.wrapper_maximise_diversification(
        pd_covar=covar, constraints=op.Constraints(is_long_only=True)).to_numpy()
    reference = correlation_minimum_variance(CORR, VOLS)
    ratio = op.calculate_diversification_ratio(w=mdp, covar=covar.to_numpy())
    rho = correlations_with_portfolio(covar.to_numpy(), mdp)
    held = mdp > HELD
    table = pd.DataFrame({'weight_slsqp': mdp, 'weight_correlation_minvar': reference,
                          'correlation_with_mdp': rho, 'held': held}, index=TICKERS)

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange, grey = '#2a78d6', '#eb6834', '#b9b8b3'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    x = np.arange(len(TICKERS))
    width = 0.38
    left.bar(x - width / 2 - 0.01, mdp, width, color=blue, label='SLSQP, the package solver')
    left.bar(x + width / 2 + 0.01, reference, width, color=orange,
             label='CVXPY on C, rescaled by 1/vol')
    left.text(TICKERS.index('Credit'), 0.012, 'not held', ha='center', va='bottom', color=ink,
              fontsize=10)
    left.set_title('Two routes to the same weights', loc='left', color=ink)
    left.set_ylabel('Weight')
    left.set_ylim(0.0, 0.95)
    left.legend(frameon=False, loc='upper right', fontsize=10, labelcolor=ink)
    colours = [blue if h else grey for h in held]
    right.bar(x, rho, 0.6, color=colours)
    right.axhline(1.0 / ratio, color=muted, linestyle='--', linewidth=1.2)
    right.text(len(TICKERS) - 0.5, 1.0 / ratio + 0.015, f'1 / DR = {1.0 / ratio:.3f}',
               ha='right', va='bottom', color=ink, fontsize=10)
    right.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=blue),
                          plt.Rectangle((0, 0), 1, 1, color=grey)],
                 labels=['held by the portfolio', 'excluded'], frameon=False,
                 loc='upper left', fontsize=10, labelcolor=ink)
    right.set_title('Correlation of each asset with the portfolio', loc='left', color=ink)
    right.set_ylim(0.0, 0.9)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.set_xticks(x, TICKERS)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    checks = {
        'routes_agree': bool(np.abs(mdp - reference).max() < 1e-4),
        'held_correlations_equal_inverse_ratio': bool(np.allclose(rho[held], 1.0 / ratio,
                                                                  atol=1e-4)),
        'excluded_correlations_exceed_inverse_ratio': bool((rho[~held] > 1.0 / ratio).all()),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
