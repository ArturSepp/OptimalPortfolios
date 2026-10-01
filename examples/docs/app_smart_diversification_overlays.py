"""Canonical script of docs/app_smart_diversification_overlays.md.

The case study follows the overlay allocation of Sepp and Kastenholz (2026), The Convexity
Premium of Portfolio Overlays, Journal of Investment Management, forthcoming. It reproduces none
of the paper's data, numbers or results. It builds the workflow on a simulated monthly panel of a
60/40 core and four stylised overlays, the design of the paper's public synthetic companion:
qis classifies the core's regimes, decomposes each Sharpe ratio and estimates the regime-mixture
covariance, and optimalportfolios solves the fixed-core maximum Sharpe ratio under a coverage
floor on the Bear-regime loss. Every number and property the page states is asserted against a
reference computed a different way: regime masks from raw quantiles, regime contributions from
masked means, the Gaussian null from the normal density, the mixture covariance from its
regime moments and per-regime least squares, each allocation from its first-order conditions and
from the homogeneous floor encoding, and the realised statistics from the simulated returns. The
script runs offline after ``pip install optimalportfolios`` with a fixed seed:

    python -m examples.docs.app_smart_diversification_overlays

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import contextlib
import socket
from unittest.mock import patch

import numpy as np
import pandas as pd

CORE = '60/40 core'
FUNDS = ['Trend', 'Equity L/S', 'Market neutral', 'Tail hedge']
AF = 12  # monthly returns
QUANTILES = [0.0, 0.16, 0.84, 1.0]  # Bear, Normal and Bull at the core's own quantiles
MONTHS = 360
SEED = 17
# The core: 60% equities and 40% bonds, rebalanced monthly, with annual excess-return means,
# volatilities and the correlation of the two generating shocks.
CORE_WEIGHTS = [0.60, 0.40]
CORE_MEANS = [0.065, 0.020]
CORE_VOLS = [0.18, 0.07]
CORE_CORR = -0.15
# Annual excess-return means and volatilities of the four simulated overlays.
FUND_MEANS = [0.045, 0.055, 0.035, -0.040]
FUND_VOLS = [0.10, 0.12, 0.07, 0.12]
OVERLAY_BUDGET = 1.0  # overlay exposure per unit of capital on top of the core's 1.0
# Coverage floors: the fraction of the core's Bear-regime loss the overlays must offset.
COVERAGES = {'No floor': None, '0%': 0.0, '20%': 0.2, '40%': 0.4, '60%': 0.6, '80%': 0.8,
             '100%': 1.0}
OFFLINE = AssertionError('The case study must execute offline')


@contextlib.contextmanager
def offline():
    """Deny socket connections; used as the decorator of ``main``."""
    with (patch.object(socket, 'create_connection', side_effect=OFFLINE),
          patch.object(socket.socket, 'connect', side_effect=OFFLINE)):
        yield


def simulate_panel() -> pd.DataFrame:
    """Simulate the monthly excess returns of the core and the four overlays."""
    rng = np.random.default_rng(SEED)
    shocks = rng.standard_normal((MONTHS, 6))
    equity = CORE_MEANS[0] / AF + CORE_VOLS[0] / np.sqrt(AF) * shocks[:, 0]
    bonds = CORE_MEANS[1] / AF + CORE_VOLS[1] / np.sqrt(AF) * (
        CORE_CORR * shocks[:, 0] + np.sqrt(1.0 - CORE_CORR ** 2) * shocks[:, 1])
    core = CORE_WEIGHTS[0] * equity + CORE_WEIGHTS[1] * bonds
    z = (core - core.mean()) / core.std(ddof=1)
    shapes = np.column_stack([
        0.65 * np.abs(z) + 0.65 * shocks[:, 2],
        0.70 * z + 0.70 * shocks[:, 3],
        0.08 * z + shocks[:, 4],
        -0.75 * z + 0.40 * np.maximum(-z, 0.0) + 0.25 * shocks[:, 5],
    ])
    shapes = (shapes - shapes.mean(axis=0)) / shapes.std(axis=0, ddof=1)
    funds = shapes * np.array(FUND_VOLS) / np.sqrt(AF) + np.array(FUND_MEANS) / AF
    dates = pd.date_range('1996-01-31', periods=MONTHS, freq='ME')
    return pd.DataFrame(np.column_stack([core, funds]), index=dates, columns=[CORE, *FUNDS])


def estimate(panel: pd.DataFrame) -> tuple:
    """Regime statistics, the mixture covariance, expected returns and Bear contributions."""
    import qis.regimes as rg

    sampled = rg.create_sampled_returns_with_regime_id(panel, benchmark=CORE, q=QUANTILES)
    statistics = rg.compute_regime_premium_table(sampled, benchmark=CORE, af=AF, q=QUANTILES)
    betas = rg.compute_regime_betas(sampled, benchmark=CORE, af=AF)
    covar = rg.compute_regime_mixture_covar_from_sample(sampled, benchmark=CORE, af=AF,
                                                        betas=betas)
    names = covar.index
    means = (statistics['sharpe'] * statistics['ann_vol']).reindex(names)
    return statistics, betas, covar, means, statistics['bear_return_pa'].reindex(names)


def allocate(covar: pd.DataFrame, means: pd.Series, bear_contributions: pd.Series,
             coverage) -> object:
    """Solve the fixed-core maximum Sharpe ratio at one coverage floor; None has no floor."""
    from dataclasses import replace
    import optimalportfolios as op

    names = covar.index
    min_weights = pd.Series(0.0, index=names)
    max_weights = pd.Series(OVERLAY_BUDGET, index=names)
    min_weights[CORE] = max_weights[CORE] = 1.0
    base = op.Constraints(is_long_only=True, min_weights=min_weights, max_weights=max_weights,
                          min_exposure=1.0 + OVERLAY_BUDGET, max_exposure=1.0 + OVERLAY_BUDGET)
    floor = None if coverage is None else op.LinearConstraints(
        loadings=bear_contributions.to_frame('bear_coverage'),
        lower=pd.Series({'bear_coverage': (1.0 - coverage) * bear_contributions[CORE]}))
    return op.cvx_maximize_portfolio_sharpe(covar=covar.to_numpy(), means=means.to_numpy(),
                                            constraints=replace(base, linear_constraints=floor))


def realised_statistics(panel: pd.DataFrame, weights: pd.DataFrame,
                        bear_contributions: pd.Series) -> pd.DataFrame:
    """Regime statistics of the core, each core-plus-one-overlay stack and each allocation."""
    import qis.regimes as rg

    single = panel[FUNDS].mul(OVERLAY_BUDGET).add(panel[CORE], axis=0)
    portfolios = pd.concat([panel[CORE], single, panel @ weights], axis=1)
    realised = rg.compute_regime_premium_table(
        rg.create_sampled_returns_with_regime_id(portfolios, benchmark=CORE, q=QUANTILES),
        benchmark=CORE, af=AF, q=QUANTILES)
    realised['coverage'] = 1.0 - realised['bear_return_pa'] / bear_contributions[CORE]
    return realised


def bear_mask(core: pd.Series) -> pd.Series:
    """The Bear months: the core's returns at or below their 16% quantile."""
    return core <= core.quantile(QUANTILES[1])


def regime_masks(core: pd.Series) -> dict:
    """Bear, Normal and Bull months from the core's 16% and 84% quantiles."""
    bear = bear_mask(core)
    bull = core > core.quantile(QUANTILES[2])
    return {'Bear': bear, 'Normal': ~bear & ~bull, 'Bull': bull}


def mixture_covariance_reference(panel: pd.DataFrame) -> tuple:
    """Equation (10) by hand: per-regime least squares, then the law of total covariance.

    Returns:
        The annual covariance, the regime betas and the annual residual volatilities.
    """
    core = panel[CORE].to_numpy()
    masks = regime_masks(panel[CORE])
    loadings = {regime: np.ones(panel.shape[1]) for regime in masks}
    residuals = np.zeros((len(panel), panel.shape[1]))
    for regime, mask in masks.items():
        x = core[mask.to_numpy()]
        for j, column in enumerate(panel.columns[1:], start=1):
            y = panel[column].to_numpy()[mask.to_numpy()]
            slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
            loadings[regime][j] = slope
            residuals[mask.to_numpy(), j] = y - (y.mean() + slope * (x - x.mean()))
    idio_vol = np.sqrt(AF * residuals[:, 1:].var(axis=0, ddof=1))
    second, first = np.zeros((panel.shape[1], panel.shape[1])), np.zeros(panel.shape[1])
    for regime, mask in masks.items():
        x = core[mask.to_numpy()]
        p, m, s = mask.mean(), x.mean(), np.mean(x ** 2)
        second += p * np.outer(loadings[regime], loadings[regime]) * s
        first += p * loadings[regime] * m
    covar = AF * (second - np.outer(first, first)) + np.diag(np.r_[0.0, idio_vol ** 2])
    betas = pd.DataFrame({f'beta_{regime.lower()}': loadings[regime][1:] for regime in masks},
                         index=panel.columns[1:])
    return (pd.DataFrame(covar, index=panel.columns, columns=panel.columns), betas,
            pd.Series(idio_vol, index=panel.columns[1:]))


def kkt_certificate(weights: pd.Series, covar: pd.DataFrame, means: pd.Series,
                    bear_contributions: pd.Series, binding: bool) -> bool:
    """Whether the first-order conditions of the ratio certify the overlays as a maximum.

    On the fixed core and budget, the scaled Sharpe gradient g = mu - SR * Sigma w / sigma
    plus lambda times the floor coefficients must be equal on the held overlays and no larger on
    the overlays at zero, with lambda >= 0 and lambda = 0 when the floor is slack. The Sharpe
    ratio is pseudo-concave on this convex set, so the conditions are sufficient.
    """
    w = weights.to_numpy()
    sigma = np.sqrt(w @ covar.to_numpy() @ w)
    gradient = means.to_numpy() - (means.to_numpy() @ w) / sigma * (covar.to_numpy() @ w) / sigma
    a = bear_contributions.to_numpy()
    overlays = np.arange(1, len(w))
    held = overlays[w[1:] > 1e-6]
    zero = overlays[w[1:] <= 1e-6]
    if binding:
        design = np.column_stack([np.ones(len(held)), -a[held]])
        (level, multiplier), *_ = np.linalg.lstsq(design, gradient[held], rcond=None)
    else:
        level, multiplier = gradient[held].mean(), 0.0
    adjusted = gradient + multiplier * a
    return bool(multiplier >= -1e-9
                and np.allclose(adjusted[held], level, rtol=0, atol=1e-6)
                and (adjusted[zero] <= level + 1e-6).all())


@offline()
def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    from dataclasses import replace
    import numpy as np
    import pandas as pd
    import qis.regimes as rg
    import optimalportfolios as op

    rng = np.random.default_rng(SEED)
    shocks = rng.standard_normal((MONTHS, 6))
    equity = CORE_MEANS[0] / AF + CORE_VOLS[0] / np.sqrt(AF) * shocks[:, 0]
    bonds = CORE_MEANS[1] / AF + CORE_VOLS[1] / np.sqrt(AF) * (
        CORE_CORR * shocks[:, 0] + np.sqrt(1.0 - CORE_CORR ** 2) * shocks[:, 1])
    core = CORE_WEIGHTS[0] * equity + CORE_WEIGHTS[1] * bonds
    z = (core - core.mean()) / core.std(ddof=1)
    shapes = np.column_stack([
        0.65 * np.abs(z) + 0.65 * shocks[:, 2],  # Trend: convex in large core moves
        0.70 * z + 0.70 * shocks[:, 3],  # Equity L/S: linear in the core
        0.08 * z + shocks[:, 4],  # Market neutral: mostly independent
        -0.75 * z + 0.40 * np.maximum(-z, 0.0) + 0.25 * shocks[:, 5],  # Tail hedge
    ])
    shapes = (shapes - shapes.mean(axis=0)) / shapes.std(axis=0, ddof=1)
    funds = shapes * np.array(FUND_VOLS) / np.sqrt(AF) + np.array(FUND_MEANS) / AF
    dates = pd.date_range("1996-01-31", periods=MONTHS, freq="ME")
    panel = pd.DataFrame(np.column_stack([core, funds]), index=dates, columns=[CORE, *FUNDS])

    # The block is the panel helper that the figure uses; the overlays are standardised, so
    # their sample means and volatilities are exactly the design values.
    pd.testing.assert_frame_equal(panel, simulate_panel())
    assert panel.shape == (360, 5) and panel.index[-1] == pd.Timestamp('2025-12-31')
    np.testing.assert_allclose(AF * panel[FUNDS].mean(), FUND_MEANS, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.sqrt(AF) * panel[FUNDS].std(ddof=1), FUND_VOLS,
                               rtol=0, atol=1e-15)
    # The core earns 3.5% a year at a volatility of 10.3% in this sample.
    assert round(AF * panel[CORE].mean(), 3) == 0.035
    assert round(np.sqrt(AF) * panel[CORE].std(ddof=1), 3) == 0.103

    sampled = rg.create_sampled_returns_with_regime_id(panel, benchmark=CORE, q=QUANTILES)
    statistics = rg.compute_regime_premium_table(sampled, benchmark=CORE, af=AF, q=QUANTILES)
    betas = rg.compute_regime_betas(sampled, benchmark=CORE, af=AF)
    covar = rg.compute_regime_mixture_covar_from_sample(sampled, benchmark=CORE, af=AF,
                                                        betas=betas)
    names = covar.index
    means = (statistics["sharpe"] * statistics["ann_vol"]).reindex(names)
    bear_contributions = statistics["bear_return_pa"].reindex(names)

    # The regimes: 58 Bear, 244 Normal and 58 Bull months, the same months as the raw quantile
    # masks of the core.
    masks = regime_masks(panel[CORE])
    labels = sampled.drop(columns=panel.columns).iloc[:, 0].astype(str)
    for regime, mask in masks.items():
        assert (labels == regime).equals(mask)
    assert [int(mask.sum()) for mask in masks.values()] == [58, 244, 58]
    # Proposition 1: each regime contribution is sqrt(af) p_s m_s / sigma with the empirical
    # frequency, and the three add up to the Sharpe ratio exactly.
    sigma = panel.std(ddof=1)
    contributions = pd.DataFrame({
        regime: np.sqrt(AF) * mask.mean() * panel[mask].mean() / sigma
        for regime, mask in masks.items()})
    np.testing.assert_allclose(statistics['bear_sharpe'], contributions['Bear'], atol=1e-12)
    np.testing.assert_allclose(statistics['normal_sharpe'], contributions['Normal'], atol=1e-12)
    np.testing.assert_allclose(statistics['bull_sharpe'], contributions['Bull'], atol=1e-12)
    np.testing.assert_allclose(contributions.sum(axis=1), statistics['sharpe'], atol=1e-12)
    np.testing.assert_allclose(statistics['sharpe'], np.sqrt(AF) * panel.mean() / sigma,
                               atol=1e-12)
    # Definition 2: CP = SR_Bear - (0.16 SR - kappa rho), kappa = sqrt(12) phi(z_0.84) = 0.843.
    from scipy.stats import norm
    kappa = np.sqrt(AF) * norm.pdf(norm.ppf(1.0 - QUANTILES[1]))
    assert round(kappa, 3) == 0.843
    rho = panel.corr()[CORE]
    np.testing.assert_allclose(statistics['rho'], rho, atol=1e-12)
    null = QUANTILES[1] * statistics['sharpe'] - kappa * rho
    np.testing.assert_allclose(statistics['null_bear_sharpe'], null, atol=1e-12)
    np.testing.assert_allclose(statistics['convexity_premium'],
                               statistics['bear_sharpe'] - null, atol=1e-12)
    # The annual Bear contribution in return units is af times the Bear-month sum over all
    # months, volatility times the Bear-Sharpe contribution.
    np.testing.assert_allclose(statistics['bear_return_pa'],
                               AF * panel[masks['Bear']].sum() / MONTHS, atol=1e-15)
    np.testing.assert_allclose(statistics['bear_return_pa'],
                               statistics['ann_vol'] * statistics['bear_sharpe'], atol=1e-15)
    # Expected returns are the sample means, and the mixture covariance is equation (10)
    # rebuilt from per-regime least squares and the core's regime moments.
    np.testing.assert_allclose(means, AF * panel.mean().reindex(names), atol=1e-15)
    assert list(names) == [CORE, *FUNDS]
    reference_covar, reference_betas, reference_idio = mixture_covariance_reference(panel)
    np.testing.assert_allclose(covar, reference_covar, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(betas[reference_betas.columns], reference_betas, atol=1e-12)
    np.testing.assert_allclose(betas['idio_vol'], reference_idio, atol=1e-12)
    assert np.linalg.eigvalsh(covar).min() > 0.0
    # The page's input table.
    table = statistics.loc[[CORE, *FUNDS], ['sharpe', 'ann_vol', 'rho', 'bear_sharpe',
                                            'null_bear_sharpe', 'convexity_premium',
                                            'bear_return_pa']]
    np.testing.assert_allclose(table.round(3), [
        [0.342, 0.103, 1.000, -0.772, -0.788, 0.017, -0.080],
        [0.450, 0.100, 0.024, 0.452, 0.052, 0.400, 0.045],
        [0.458, 0.120, 0.717, -0.496, -0.531, 0.035, -0.060],
        [0.500, 0.070, 0.107, -0.005, -0.010, 0.005, -0.000],
        [-0.333, 0.120, -0.954, 0.817, 0.751, 0.066, 0.098]], rtol=0, atol=5.1e-4)
    np.testing.assert_allclose((100 * table['bear_return_pa']).round(2),
                               [-7.96, 4.52, -5.95, -0.03, 9.80], rtol=0, atol=5.1e-3)
    # The mechanism: the convex Trend has a large premium at almost zero correlation, the linear
    # Equity L/S and the independent Market neutral have premia near zero, and the Tail hedge's
    # Bear contribution mostly comes from its negative correlation.
    premium = statistics['convexity_premium']
    assert premium['Trend'] > 0.3 and abs(rho['Trend']) < 0.05
    assert abs(premium['Equity L/S']) < 0.05 and abs(premium['Market neutral']) < 0.05
    assert 0.0 < premium['Tail hedge'] < statistics.at['Tail hedge', 'null_bear_sharpe']

    min_weights = pd.Series(0.0, index=names)
    max_weights = pd.Series(OVERLAY_BUDGET, index=names)
    min_weights[CORE] = max_weights[CORE] = 1.0
    base = op.Constraints(
        is_long_only=True, min_weights=min_weights, max_weights=max_weights,
        min_exposure=1.0 + OVERLAY_BUDGET, max_exposure=1.0 + OVERLAY_BUDGET,
    )
    allocations = {}
    for label, coverage in COVERAGES.items():
        floor = None if coverage is None else op.LinearConstraints(
            loadings=bear_contributions.to_frame("bear_coverage"),
            lower=pd.Series({"bear_coverage": (1.0 - coverage) * bear_contributions[CORE]}),
        )
        outcome = op.cvx_maximize_portfolio_sharpe(
            covar=covar.to_numpy(), means=means.to_numpy(),
            constraints=replace(base, linear_constraints=floor),
        )
        if not (outcome.accepted and outcome.compliant):
            raise RuntimeError(f"{label}: {outcome.status}; {outcome.reason}")
        allocations[label] = pd.Series(outcome.weights, index=names)
    weights = pd.DataFrame(allocations)

    # The block is the figure's helper.
    for label, coverage in COVERAGES.items():
        np.testing.assert_allclose(allocate(covar, means, bear_contributions, coverage).weights,
                                   weights[label], rtol=0, atol=1e-12)
    # The capital mandate: the core at 1.0, long-only overlays summing to the budget.
    np.testing.assert_allclose(weights.loc[CORE], 1.0, atol=1e-6)
    np.testing.assert_allclose(weights.loc[FUNDS].sum(), OVERLAY_BUDGET, atol=1e-6)
    assert (weights >= -1e-6).all().all()
    # The no-floor allocation covers 22.1% of the core's Bear-regime loss, so the 0% and 20%
    # floors are slack and leave it unchanged; from 40% on every floor binds at equality.
    a = bear_contributions
    covered = 1.0 - (a @ weights) / a[CORE]
    assert round(covered['No floor'], 3) == 0.221
    for label in ('0%', '20%'):
        np.testing.assert_allclose(weights[label], weights['No floor'], atol=1e-6)
    for label in ('40%', '60%', '80%', '100%'):
        assert abs(covered[label] - COVERAGES[label]) < 1e-6
    # Each allocation is certified by its first-order conditions, and each binding floor also
    # by the homogeneous encoding of the overlay page, a different formulation.
    for label, coverage in COVERAGES.items():
        binding = coverage is not None and coverage > covered['No floor']
        assert kkt_certificate(weights[label], covar, means, a, binding)
        if binding:
            shifted = a - (1.0 - coverage) * a[CORE] / (1.0 + OVERLAY_BUDGET)
            homogeneous = op.cvx_maximize_portfolio_sharpe(
                covar=covar.to_numpy(), means=means.to_numpy(),
                constraints=replace(base, asset_returns=shifted, target_return=0.0))
            np.testing.assert_allclose(homogeneous.weights, weights[label], atol=1e-6)
    # The page's allocation table, in percent of the overlay budget; Equity L/S is never held.
    shown = ['No floor', '20%', '40%', '60%', '80%', '100%']
    np.testing.assert_allclose(100 * weights.loc[FUNDS, shown].round(3), [
        [39.4, 39.4, 35.2, 29.9, 29.8, 34.9],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [60.6, 60.6, 48.4, 35.1, 18.9, 0.0],
        [0.0, 0.0, 16.4, 35.1, 51.3, 65.1]], rtol=0, atol=0.051)
    assert (weights.loc['Equity L/S'] < 1e-6).all()
    # The model Sharpe ratio of the mixture covariance never rises as the floor tightens.
    model = pd.Series({label: float(means @ weights[label]
                                    / np.sqrt(weights[label] @ covar @ weights[label]))
                       for label in COVERAGES})
    assert (np.diff(model.to_numpy()) < 1e-9).all()
    assert round(model['No floor'], 2) == 0.62 and round(model['100%'], 2) == 0.42
    # Theta_max = -W max_i a_i / a_c = 123.2%, the whole budget in the Tail hedge; a 130%
    # floor is infeasible, and the solver rejects it.
    theta_max = -OVERLAY_BUDGET * a.drop(CORE).max() / a[CORE]
    assert a.drop(CORE).idxmax() == 'Tail hedge' and round(theta_max, 3) == 1.232
    infeasible = allocate(covar, means, a, 1.30)
    assert not infeasible.accepted and infeasible.status == 'infeasible'

    stacked = panel @ weights
    single = panel[FUNDS].mul(OVERLAY_BUDGET).add(panel[CORE], axis=0)
    portfolios = pd.concat([panel[CORE], single, stacked], axis=1)
    realised = rg.compute_regime_premium_table(
        rg.create_sampled_returns_with_regime_id(portfolios, benchmark=CORE, q=QUANTILES),
        benchmark=CORE, af=AF, q=QUANTILES,
    )
    realised["coverage"] = 1.0 - realised["bear_return_pa"] / bear_contributions[CORE]

    # The block is the figure's helper, and its statistics are the simulated returns'.
    pd.testing.assert_frame_equal(realised, realised_statistics(panel, weights, a))
    bear = masks['Bear']
    for name in portfolios:
        returns = portfolios[name]
        np.testing.assert_allclose(realised.at[name, 'sharpe'],
                                   np.sqrt(AF) * returns.mean() / returns.std(ddof=1), atol=1e-12)
        np.testing.assert_allclose(realised.at[name, 'coverage'],
                                   1.0 - returns[bear].sum() / panel.loc[bear, CORE].sum(),
                                   atol=1e-12)
    # Proposition 4 in return units: the Bear contribution of every stack is the weighted sum of
    # the inputs' Bear contributions, which is what keeps the floor linear.
    np.testing.assert_allclose(realised.loc[weights.columns, 'bear_return_pa'], a @ weights,
                               atol=1e-15)
    np.testing.assert_allclose(realised.loc[FUNDS, 'bear_return_pa'],
                               a[CORE] + OVERLAY_BUDGET * a[FUNDS], atol=1e-15)
    # Definition 4: Trend, Equity L/S and Market neutral raise both the Sharpe ratio and the
    # Bear-Sharpe ratio of the core; the Tail hedge raises only the second.
    smart = [fund for fund in FUNDS
             if realised.at[fund, 'sharpe'] > realised.at[CORE, 'sharpe']
             and realised.at[fund, 'bear_sharpe'] > realised.at[CORE, 'bear_sharpe']]
    assert smart == ['Trend', 'Equity L/S', 'Market neutral']
    assert realised.at['Tail hedge', 'bear_sharpe'] > 0.0 > realised.at['Tail hedge', 'sharpe']
    # The pitfall: Equity L/S raises the Bear-Sharpe ratio only by doubling volatility, from
    # 10.3% to 20.7%; it adds 6.0% a year to the Bear-regime loss, which grows from 8.0% to
    # 13.9%, and the coverage floor counts that loss.
    assert round(realised.at['Equity L/S', 'ann_vol'], 3) == 0.207
    assert round(a['Equity L/S'], 3) == -0.060
    assert (round(-a[CORE], 3), round(-realised.at['Equity L/S', 'bear_return_pa'], 3)) \
        == (0.080, 0.139)
    # Each overlay alone at the full budget covers 56.7% (Trend), -74.8% (Equity L/S), -0.4%
    # (Market neutral) and 123.2% (Tail hedge) of the Bear-regime loss.
    np.testing.assert_allclose(realised.loc[FUNDS, 'coverage'].round(3),
                               [0.567, -0.748, -0.004, 1.232], rtol=0, atol=5.1e-4)
    # The page's results table.
    rows = ['sharpe', 'bear_sharpe', 'ann_vol', 'bear_return_pa', 'coverage']
    np.testing.assert_allclose(realised.loc[[CORE, *shown], rows].round(3), [
        [0.342, -0.772, 0.103, -0.080, 0.000],
        [0.602, -0.503, 0.123, -0.062, 0.221],
        [0.602, -0.503, 0.123, -0.062, 0.221],
        [0.602, -0.467, 0.102, -0.048, 0.400],
        [0.594, -0.403, 0.079, -0.032, 0.600],
        [0.561, -0.257, 0.062, -0.016, 0.800],
        [0.452, 0.000, 0.055, 0.000, 1.000]], rtol=0, atol=5.1e-4)
    np.testing.assert_allclose(realised.loc[FUNDS, ['sharpe', 'bear_sharpe']].round(3), [
        [0.552, -0.237], [0.436, -0.672], [0.538, -0.611], [-0.126, 0.490]],
        rtol=0, atol=5.1e-4)
    # Floors up to 40% keep the realised Sharpe ratio within 0.001 of the no-floor 0.602 while
    # the realised volatility falls from 12.3% to 10.2%; beyond 40% each 20-point step costs more
    # Sharpe ratio than the one before.
    assert abs(realised.at['40%', 'sharpe'] - realised.at['No floor', 'sharpe']) < 1e-3
    costs = -np.diff(realised.loc[['40%', '60%', '80%', '100%'], 'sharpe'].to_numpy())
    assert (costs > 0.0).all() and (np.diff(costs) > 0.0).all()
    print("app_smart_diversification_overlays: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: the stacked portfolios in the paper's coordinates, and weights.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import qis

    panel = simulate_panel()
    statistics, betas, covar, means, a = estimate(panel)
    weights = pd.DataFrame({label: allocate(covar, means, a, coverage).weights
                            for label, coverage in COVERAGES.items()}, index=covar.index)
    realised = realised_statistics(panel, weights, a)
    floored = [label for label, coverage in COVERAGES.items() if coverage is not None]
    points = realised.loc[[CORE, *FUNDS, 'No floor']]
    groups = pd.Series({CORE: 'Core', **{fund: 'Core + one overlay' for fund in FUNDS},
                        'No floor': 'Allocations'})

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 5.2), layout='constrained')
    qis.plot_overlay_allocation_frontier(
        portfolio_stats=points, frontier_stats=realised.loc[floored], groups=groups,
        benchmark=CORE, group_styles={'Core': dict(label=CORE, color='#52514e')},
        highlights={'No floor': dict(marker='*', color='#009E73', s=180)},
        label_offsets={'Equity L/S': (6, -12)},
        frontier_label='Coverage floors 0% to 100%',
        xlabel='Bear-Sharpe contribution of the stacked portfolio',
        ylabel='Arithmetic excess Sharpe ratio', title='Stacked portfolios (in sample)',
        ax=left)
    qis.plot_bars(weights.loc[FUNDS, floored].T, stacked=True, yvar_format='{:.0%}',
                  x_rotation=0, legend_loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ylabel='Overlay exposure / capital',
                  title='Overlay weights by coverage floor', ax=right)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    covered = 1.0 - (a @ weights) / a[CORE]
    binding = [label for label in floored if COVERAGES[label] > covered['No floor']]
    table = pd.concat([weights.loc[FUNDS].T,
                       realised.loc[weights.columns, ['sharpe', 'bear_sharpe', 'coverage']]],
                      axis=1).rename_axis('allocation')
    checks = {
        'capital_mandate_held': bool(np.allclose(weights.loc[CORE], 1.0, atol=1e-6)
                                     and np.allclose(weights.loc[FUNDS].sum(), OVERLAY_BUDGET,
                                                     atol=1e-6)),
        'slack_floors_keep_the_no_floor_allocation': bool(all(
            np.allclose(weights[label], weights['No floor'], atol=1e-6)
            for label in floored if label not in binding)),
        'binding_floors_cover_exactly': bool(all(
            abs(realised.at[label, 'coverage'] - COVERAGES[label]) < 1e-6 for label in binding)),
        'bear_contributions_add_up': bool(np.allclose(
            realised.loc[weights.columns, 'bear_return_pa'], a @ weights, atol=1e-15)),
        'model_sharpe_never_rises': bool((np.diff([
            float(means @ weights[label] / np.sqrt(weights[label] @ covar @ weights[label]))
            for label in COVERAGES]) < 1e-9).all()),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
