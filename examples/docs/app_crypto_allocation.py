"""Canonical script of docs/app_crypto_allocation.md.

The case study quotes the results of Sepp (2023) from the paper; this script does not reproduce
them. The page's Python blocks are excerpts of this file in page order, and ``main`` asserts
every statement of the page in two parts, each against a reference computed a different way:

1. A synthetic monthly panel with a fat-tailed, high-volatility crypto asset runs the study's
   configuration through ``compute_rolling_optimal_weights``. The script checks the risk shares
   by explicit products, the maximum Sharpe weights against a direct SciPy maximisation under
   explicitly summed EWMA means, and the CARA weights against CVXPY on the refitted mixture. A
   fixed two-asset Gaussian mixture, with no simulation, checks how the direction of the crypto
   tail moves the CARA allocation, against a closed form and CVXPY.
2. The same configuration runs with the current API on the ETF-derived columns of the tracked
   2023 panel, ``papers/crypto_allocation_risk_2023/data/crypto_allocation_prices.csv``, and the
   script asserts the numbers the page quotes from it as "current API on the frozen 2023 panel".
   Only the columns in ``FROZEN_COLUMNS`` are read; the hedge-fund and SG index columns of that
   file are never loaded.

The second part reads the tracked file, so the script runs from a source checkout with the core
install. It is offline and runs in under 30 seconds:

    python -m examples.docs.app_crypto_allocation

``exhibit`` draws the page's figure from the frozen panel; ``tools/docs_analytics/teaching.py``
calls it with the constants below and records their values.
"""
import inspect
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import qis
from scipy.optimize import minimize

import optimalportfolios as op

REPO_ROOT = Path(__file__).resolve().parents[2]
# The study's configuration: monthly log returns, quarterly rebalancing, EWMA span 30, CARA
# risk aversion 0.5 with three mixture components fitted on 60 monthly returns.
METHODS = {'ERC': 'EQUAL_RISK_CONTRIBUTION', 'MaxDiv': 'MAX_DIVERSIFICATION',
           'MaxSharpe': 'MAXIMUM_SHARPE_RATIO', 'CARA-3': 'MAX_CARA_MIXTURE'}
SPAN = 30
ROLL_WINDOW = 60
CARRA = 0.5
N_MIXTURES = 3
BALANCED_WEIGHT = 0.75
REPORT_START, REPORT_END = '2016-03-31', '2023-06-30'
HELD = 1e-4  # a weight above this is a held asset
# Synthetic monthly panel: log-return means and volatilities, correlations, and a crypto tail
# component that adds TAIL_JUMP to the crypto return with probability TAIL_PROBABILITY.
SEED = 3
ASSETS = ['Balanced', 'Crypto', 'Private equity', 'Real estate', 'Commodities', 'Gold']
MONTHLY_MEANS = [0.007, 0.025, 0.008, 0.007, 0.005, 0.005]
MONTHLY_VOLS = [0.025, 0.15, 0.05, 0.05, 0.055, 0.04]
CORR = [[1.0, 0.2, 0.7, 0.6, 0.3, 0.1],
        [0.2, 1.0, 0.2, 0.2, 0.1, 0.1],
        [0.7, 0.2, 1.0, 0.6, 0.3, 0.1],
        [0.6, 0.2, 0.6, 1.0, 0.3, 0.2],
        [0.3, 0.1, 0.3, 0.3, 1.0, 0.3],
        [0.1, 0.1, 0.1, 0.2, 0.3, 1.0]]
TAIL_PROBABILITY = 0.05
TAIL_JUMP = 0.40
PANEL_START, PANEL_END = '2010-07-31', '2023-06-30'
# Fixed two-asset mixture of annualised moments, balanced portfolio and crypto: in a 3% component
# the crypto mean is MIXTURE_DEVIATION above its overall mean, and the 97% component offsets it.
MIXTURE_MEANS = [0.06, 0.14]
MIXTURE_VOLS = [0.10, 0.60]
MIXTURE_CORRELATION = 0.3
MIXTURE_TAIL_PROBABILITY = 0.03
MIXTURE_DEVIATION = 3.0
# The ETF-derived columns of the frozen 2023 panel; no other column of the file is read.
FROZEN_PANEL = 'papers/crypto_allocation_risk_2023/data/crypto_allocation_prices.csv'
FROZEN_COLUMNS = ['60/40', 'BTC', 'PE', 'RealEstate', 'Commodities', 'Gold']
TEMPLATES = {'100% Alts with BTC': False, '75%/25% Balanced/Alts with BTC': True}


def simulated_prices(seed: int) -> pd.DataFrame:
    """Simulate month-end prices of the synthetic panel with Cholesky and tail draws."""
    dates = pd.date_range(PANEL_START, PANEL_END, freq='ME')
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((len(dates) - 1, len(ASSETS))) @ np.linalg.cholesky(CORR).T
    tail = (rng.random(len(dates) - 1) < TAIL_PROBABILITY).astype(float)
    returns = np.array(MONTHLY_MEANS) + shocks * np.array(MONTHLY_VOLS)
    returns[:, 1] += TAIL_JUMP * (tail - TAIL_PROBABILITY)
    levels = np.exp(np.vstack([np.zeros((1, len(ASSETS))), np.cumsum(returns, axis=0)]))
    return pd.DataFrame(100.0 * levels, index=dates, columns=ASSETS)


def frozen_panel() -> pd.DataFrame:
    """Read the ETF-derived columns of the tracked 2023 panel up to the end of the study."""
    prices = pd.read_csv(REPO_ROOT / FROZEN_PANEL, index_col=0, parse_dates=True,
                         usecols=lambda name: name in FROZEN_COLUMNS or name == 'Unnamed: 0')
    assert list(prices.columns) == FROZEN_COLUMNS
    return prices.loc[:REPORT_END].dropna()


def covariances(prices: pd.DataFrame) -> dict:
    """Return the quarter-end EWMA covariances of monthly log returns in the report window."""
    return op.EwmaCovarEstimator(returns_freq='ME', span=SPAN, rebalancing_freq='QE') \
        .fit_rolling_covars(prices=prices, time_period=qis.TimePeriod(REPORT_START, REPORT_END))


def allocate(prices: pd.DataFrame, balanced: bool) -> dict:
    """Run the four methods on one template; return a date-by-asset weight table per method."""
    covar_dict = covariances(prices)
    budget, pinned = None, op.Constraints()  # ERC: equal risk budgets, long-only
    if balanced:  # the first column is the balanced portfolio: a 75% risk budget or weight
        budget = pd.Series((1 - BALANCED_WEIGHT) / (prices.shape[1] - 1), index=prices.columns)
        budget.iloc[0] = BALANCED_WEIGHT
        lower, upper = pd.Series(0.0, index=prices.columns), pd.Series(1.0, index=prices.columns)
        lower.iloc[0] = upper.iloc[0] = BALANCED_WEIGHT
        pinned = op.Constraints(min_weights=lower, max_weights=upper)
    return {label: op.compute_rolling_optimal_weights(
        prices=prices, constraints=op.Constraints() if label == 'ERC' else pinned,
        covar_dict=covar_dict, portfolio_objective=op.PortfolioObjective[member],
        time_period=qis.TimePeriod(REPORT_START, REPORT_END), risk_budget=budget,
        returns_freq='ME', rebalancing_freq='QE', span=SPAN, roll_window=ROLL_WINDOW,
        carra=CARRA, n_mixures=N_MIXTURES)
        for label, member in METHODS.items()}


def monthly_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return log returns between the last prices of consecutive months, labelled month end."""
    month_ends = prices.groupby(prices.index.to_period('M')).tail(1)
    month_ends.index = month_ends.index.to_period('M').to_timestamp(how='end').normalize()
    return np.log(month_ends).diff().iloc[1:]


def ewma_mean(returns: np.ndarray, span: int) -> np.ndarray:
    """Return the EWMA of the rows seeded with the first row, as an explicit weighted sum."""
    lam = 1.0 - 2.0 / (span + 1.0)
    weights = (1.0 - lam) * lam ** np.arange(len(returns))[::-1]
    weights[0] = lam ** (len(returns) - 1)
    return weights @ returns


def risk_shares(weights: np.ndarray, covar: np.ndarray) -> np.ndarray:
    """Return each asset's share of portfolio variance."""
    return weights * (covar @ weights) / (weights @ covar @ weights)


def sharpe_ratio(weights: np.ndarray, means: np.ndarray, covar: np.ndarray) -> float:
    """Return the ratio of the portfolio mean to its volatility."""
    return float(means @ weights / np.sqrt(weights @ covar @ weights))


def sharpe_reference(means: np.ndarray, covar: np.ndarray) -> np.ndarray:
    """Maximise the ratio directly with SciPy, long-only and fully invested, from equal weights."""
    n = len(means)
    result = minimize(lambda w: -sharpe_ratio(w, means, covar), np.ones(n) / n, method='SLSQP',
                      bounds=[(0.0, 1.0)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}],
                      options={'ftol': 1e-12, 'maxiter': 1000})
    return result.x


def cvxpy_cara(means: list, covars: list, probs: np.ndarray) -> np.ndarray:
    """Maximise the mixture CARA utility with CVXPY, long-only and fully invested."""
    w = cp.Variable(len(means[0]))
    loss = sum(p * cp.exp(-CARRA * m @ w + 0.5 * CARRA ** 2 * cp.quad_form(w, c))
               for p, m, c in zip(probs, means, covars))
    cp.Problem(cp.Minimize(loss), [cp.sum(w) == 1, w >= 0]).solve(solver='CLARABEL')
    return w.value


def fixed_mixture(sign: float) -> tuple:
    """Return the two-component mixture with the crypto tail up (sign=1) or down (sign=-1)."""
    covar = np.outer(MIXTURE_VOLS, MIXTURE_VOLS) * np.array(
        [[1.0, MIXTURE_CORRELATION], [MIXTURE_CORRELATION, 1.0]])
    p = MIXTURE_TAIL_PROBABILITY
    shift = np.array([0.0, sign * MIXTURE_DEVIATION])
    means = [np.array(MIXTURE_MEANS) - p / (1 - p) * shift, np.array(MIXTURE_MEANS) + shift]
    return means, [covar, covar], np.array([1 - p, p])


def matched_gaussian(means: list, covars: list, probs: np.ndarray) -> tuple:
    """Return the mean vector and covariance matrix of a Gaussian mixture."""
    mean = sum(p * m for p, m in zip(probs, means))
    covar = sum(p * (c + np.outer(m - mean, m - mean)) for p, m, c in zip(probs, means, covars))
    return mean, covar


def btc_weights(results: dict) -> pd.DataFrame:
    """Collect the BTC weight of each method into one date-by-method table."""
    return pd.DataFrame({label: weights['BTC'] for label, weights in results.items()})


def main() -> None:
    """Run the configuration on the synthetic and frozen panels and assert the page."""
    prices = simulated_prices(SEED)
    alts = allocate(prices[ASSETS[1:]], balanced=False)
    blend = allocate(prices, balanced=True)
    for weights in [*alts.values(), *blend.values()]:
        assert weights.index[[0, -1]].strftime('%Y-%m-%d').tolist() == [REPORT_START, REPORT_END]
        assert len(weights) == 30 and np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        assert (weights > -1e-8).all().all()

    for date, covar in covariances(prices).items():
        shares = risk_shares(blend['ERC'].loc[date].to_numpy(), covar.to_numpy())
        assert np.allclose(shares, [BALANCED_WEIGHT] + [0.05] * 5, atol=1e-5)
    for label in ['MaxDiv', 'MaxSharpe', 'CARA-3']:
        assert np.allclose(blend[label]['Balanced'], BALANCED_WEIGHT, atol=1e-6)
    assert abs(blend['ERC']['Balanced'].min() - 0.72) < 0.005
    assert abs(blend['ERC']['Balanced'].max() - 0.81) < 0.005

    assert (alts['ERC'].idxmin(axis=1) == 'Crypto').all()
    assert (blend['MaxDiv']['Crypto'] > blend['ERC']['Crypto']).all()
    assert alts['MaxDiv']['Crypto'].median() > alts['ERC']['Crypto'].median()

    returns = monthly_log_returns(prices[ASSETS[1:]])
    for date, covar in covariances(prices[ASSETS[1:]]).items():
        means = 12.0 * ewma_mean(returns.loc[:date].to_numpy(), SPAN)
        weights = alts['MaxSharpe'].loc[date].to_numpy()
        reference = sharpe_reference(means, covar.to_numpy())
        assert sharpe_ratio(weights, means, covar.to_numpy()) > sharpe_ratio(
            reference, means, covar.to_numpy()) - 1e-6

    window = returns.loc[:REPORT_END].iloc[-ROLL_WINDOW:].to_numpy()
    mixture = op.fit_gaussian_mixture(x=window, n_components=N_MIXTURES, an_factor=12.0)
    reference = cvxpy_cara(mixture.means, mixture.covars, mixture.probs)
    assert np.abs(alts['CARA-3'].iloc[-1].to_numpy() - reference).max() < 1e-3

    up, down = fixed_mixture(1.0), fixed_mixture(-1.0)
    mean, covar = matched_gaussian(*up)
    assert all(np.allclose(a, b) for a, b in zip((mean, covar), matched_gaussian(*down)))
    crypto = {name: op.opt_maximize_cara_mixture(*mix, constraints=op.Constraints(),
                                                 carra=CARRA)[1]
              for name, mix in [('upside tail', up), ('Gaussian', ([mean], [covar], [1.0])),
                                ('downside tail', down)]}
    assert [round(weight, 2) for weight in crypto.values()] == [0.27, 0.25, 0.23]

    gap = covar[0, 0] + covar[1, 1] - 2.0 * covar[0, 1]
    closed_form = ((mean[1] - mean[0]) / CARRA + covar[0, 0] - covar[0, 1]) / gap
    assert abs(crypto['Gaussian'] - closed_form) < 1e-4
    assert abs(crypto['upside tail'] - cvxpy_cara(*up)[1]) < 1e-3
    assert abs(crypto['downside tail'] - cvxpy_cara(*down)[1]) < 1e-3
    third = [sum(p * (m[1] - mean[1]) ** 3 for p, m in zip(mix[2], mix[0])) for mix in (up, down)]
    assert third[0] > 0.0 > third[1]

    cara_route = dict(prices=prices[ASSETS[1:]], constraints=op.Constraints(),
                      portfolio_objective=op.PortfolioObjective.MAX_CARA_MIXTURE,
                      time_period=qis.TimePeriod(REPORT_START, REPORT_END), returns_freq='ME')
    same = op.compute_rolling_optimal_weights(covar_dict={}, roll_window=ROLL_WINDOW, **cara_route)
    assert same.equals(alts['CARA-3'])  # the covariance dictionary is not read
    short = op.compute_rolling_optimal_weights(covar_dict={}, **cara_route)
    defaults = inspect.signature(op.compute_rolling_optimal_weights).parameters
    assert defaults['roll_window'].default == 20 and defaults['returns_freq'].default == 'W-WED'
    assert (short - same).abs().to_numpy().max() > 0.2

    frozen = frozen_panel()
    results = {name: allocate(frozen.iloc[:, 0 if balanced else 1:], balanced)
               for name, balanced in TEMPLATES.items()}
    btc = {name: btc_weights(weights) for name, weights in results.items()}
    for table in btc.values():
        assert table.index[[0, -1]].strftime('%Y-%m-%d').tolist() == [REPORT_START, REPORT_END]
        assert len(table) == 30 and (table[['ERC', 'MaxDiv']] > HELD).all().all()
        assert table.median().idxmax() == 'CARA-3' and 0.17 < table['CARA-3'].median() < 0.27
    alts_btc, blend_btc = btc.values()
    assert np.allclose(alts_btc.median().iloc[:3], [0.053, 0.063, 0.109], atol=1e-3)
    assert np.allclose(blend_btc.median().iloc[:3], [0.013, 0.045, 0.030], atol=1e-3)

    # The remaining statements of the page, beyond its blocks.
    assert frozen.index[0] == pd.Timestamp('2010-07-19')
    assert op.PortfolioObjective.MAX_CARA_MIXTURE.value == 'MaxCarraMixture'
    assert op.EwmaCovarEstimator().demean
    for weights in [w for methods in results.values() for w in methods.values()]:
        assert np.isfinite(weights.to_numpy()).all()
        assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    frozen_blend = results['75%/25% Balanced/Alts with BTC']
    dropped = [table.index[table['MaxSharpe'] < HELD] for table in btc.values()]
    assert [len(dates) for dates in dropped] == [5, 6]
    assert {date.year for dates in dropped for date in dates} == {2020, 2022}
    assert (blend_btc['CARA-3'] > 0.25 - 1e-4).sum() >= 5
    erc_balanced = frozen_blend['ERC']['60/40']
    assert abs(erc_balanced.min() - 0.68) < 0.005 and abs(erc_balanced.max() - 0.83) < 0.005

    # 30 September 2022: no admissible portfolio has a positive EWMA mean, so the maximum Sharpe
    # program is infeasible and the package returns the previous weights drifted to the date.
    date, previous = pd.Timestamp('2022-09-30'), pd.Timestamp('2022-06-30')
    means = ewma_mean(monthly_log_returns(frozen).loc[:date].to_numpy(), SPAN)
    assert BALANCED_WEIGHT * means[0] + (1.0 - BALANCED_WEIGHT) * means[1:].max() < 0.0
    sharpe = frozen_blend['MaxSharpe']
    drifted = sharpe.loc[previous] * frozen.loc[date] / frozen.loc[previous]
    assert np.allclose(sharpe.loc[date], drifted / drifted.sum(), atol=1e-8)
    assert abs(sharpe.loc[date, '60/40'] - 0.765) < 1e-3
    assert np.allclose(sharpe['60/40'].drop(date), BALANCED_WEIGHT, atol=1e-6)
    print('app_crypto_allocation: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: the BTC weight of each method at each quarter end.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frozen = frozen_panel()
    results = {name: allocate(frozen.iloc[:, 0 if balanced else 1:], balanced)
               for name, balanced in TEMPLATES.items()}
    btc = {name: btc_weights(weights) for name, weights in results.items()}
    table = pd.concat(btc, axis=1)
    table.columns = [f'{name} | {label}' for name, label in table.columns]

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = dict(zip(METHODS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    for axis, (name, weights) in zip(axes, btc.items()):
        spread = np.linspace(-0.3, 0.3, len(weights))  # quarters run left to right in a column
        for x, label in enumerate(METHODS):
            values = weights[label].to_numpy()
            axis.scatter(x + spread, values, s=15, color=colours[label], edgecolors=surface,
                         linewidths=0.4, zorder=3)
            axis.hlines(np.median(values), x - 0.38, x + 0.38, color=ink, linewidth=1.6,
                        zorder=4)
        axis.set_xticks(range(len(METHODS)),
                        [f'{label}\n{weights[label].median():.1%}' for label in METHODS],
                        fontsize=10, color=ink)
        axis.set_xlim(-0.5, len(METHODS) - 0.5)
        axis.set_ylim(0.0, None)
        axis.set_ylabel('BTC weight')
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        axis.set_title(name, loc='left', color=ink)
        axis.set_facecolor(surface)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.text(0.01, 0.985, 'Current API on the frozen 2023 panel: BTC weight at 30 quarter ends, '
             'Mar 2016 to Jun 2023 from left to right; bar = median', ha='left', va='top',
             color=ink, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    everything = [weights for methods in results.values() for weights in methods.values()]
    checks = {
        'etf_columns_only': list(frozen.columns) == FROZEN_COLUMNS,
        'report_window': all(len(w) == 30 and w.index[0] == pd.Timestamp(REPORT_START)
                             and w.index[-1] == pd.Timestamp(REPORT_END) for w in everything),
        'weights_finite': all(np.isfinite(w.to_numpy()).all() for w in everything),
        'weights_sum_to_one': all(np.allclose(w.sum(axis=1), 1.0, atol=1e-6) for w in everything),
        'cara_largest_median': all(w.median().idxmax() == 'CARA-3' for w in btc.values()),
        'risk_based_always_hold': all((w[['ERC', 'MaxDiv']] > HELD).all().all()
                                      for w in btc.values()),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
