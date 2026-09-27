"""Canonical script of docs/risk_budgeting.md.

The page's three Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: an independent conic solve of the same objective with CVXPY, the closed-form
diagonal solution, the optimality conditions of a binding cap solved by root finding and a
two-asset budget equation. The group-budget example moved with its checks to
``examples/docs/hierarchical_risk_parity_and_cluster_budgets.py``. Risk shares are recomputed as
``w * (Sigma w) / (w' Sigma w)`` rather than read from qis. The script runs offline after
``pip install optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.risk_budgeting

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import logging
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve

ASSETS = ['Equity', 'Bonds', 'Diversifier']
# Annual covariance of the page's three-asset example, in fractional return squared.
COVARIANCE = [[0.040, 0.004, 0.002],
              [0.004, 0.010, 0.001],
              [0.002, 0.001, 0.022]]
RISK_BUDGETS = [0.50, 0.30, 0.20]
SLACK_CAP = 0.80  # the per-asset cap of the first example, slack at its solution
EQUITY_CAP = 0.25  # the cap on Equity alone in the second example, which binds


def risk_shares(weights, covar) -> np.ndarray:
    """Euler risk shares w_i (Sigma w)_i / (w' Sigma w), computed without qis."""
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(covar, dtype=float)
    return w * (sigma @ w) / (w @ sigma @ w)


def conic_reference(covar, budgets, upper) -> np.ndarray:
    """Solve the homogeneous volatility/log objective with CVXPY and CLARABEL."""
    sigma = np.asarray(covar, dtype=float)
    b = np.asarray(budgets, dtype=float)
    y = cp.Variable(len(b))
    total = cp.sum(y)
    root = np.linalg.cholesky(sigma / np.max(np.diag(sigma))).T
    objective = cp.norm(root @ y) - cp.sum(cp.multiply(b / b.sum(), cp.log(y)))
    problem = cp.Problem(cp.Minimize(objective), [y >= 0, y <= np.asarray(upper) * total])
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11,
                  max_iter=500)
    assert problem.status == 'optimal', problem.status
    return y.value / y.value.sum()


def diagonal_reference(variances, budgets) -> np.ndarray:
    """Closed-form weights sqrt(b_i) / sigma_i, normalised, for uncorrelated assets."""
    raw = np.sqrt(np.asarray(budgets, dtype=float)) / np.sqrt(np.asarray(variances, dtype=float))
    return raw / raw.sum()


def capped_kkt_reference(covar, budgets, capped: int, cap: float) -> tuple:
    """Weights and multiplier c from r_i = b_i + c w_i for the free assets, w_capped = cap."""
    sigma = np.asarray(covar, dtype=float)
    b = np.asarray(budgets, dtype=float) / np.sum(budgets)
    free = [i for i in range(len(b)) if i != capped]

    def weights_of(free_weights):
        """Full weight vector with the capped asset at its cap."""
        w = np.empty(len(b))
        w[capped] = cap
        w[free] = free_weights
        return w

    def conditions(x):
        """Stationarity of each free asset and full investment."""
        w = weights_of(x[:-1])
        r = risk_shares(w, sigma)
        return [*(r[free] - b[free] - x[-1] * w[free]), np.sum(w) - 1.0]

    start = diagonal_reference(np.diag(sigma), b)[free]
    start = start / start.sum() * (1.0 - cap)
    solution = fsolve(conditions, [*start, 0.0], xtol=1e-12)
    assert np.abs(conditions(solution)).max() < 1e-13
    return weights_of(solution[:-1]), solution[-1]


def two_asset_reference(covar, budgets) -> np.ndarray:
    """Two-asset weights whose first risk share equals the first normalised budget."""
    target = budgets[0] / np.sum(budgets)
    share = brentq(lambda x: risk_shares([x, 1.0 - x], covar)[0] - target, 1e-9, 1 - 1e-9,
                   xtol=1e-15)
    return np.array([share, 1.0 - share])


def solves(slack_cap: float, equity_cap: float) -> dict:
    """Weights of the page's example with a slack cap, and with the Equity cap binding."""
    import optimalportfolios as opt

    covar = pd.DataFrame(COVARIANCE, index=ASSETS, columns=ASSETS)
    budgets = pd.Series(RISK_BUDGETS, index=ASSETS)
    caps = {'no binding bound': pd.Series(slack_cap, index=ASSETS),
            'Equity capped': pd.Series([equity_cap, slack_cap, slack_cap], index=ASSETS)}
    return {name: opt.wrapper_risk_budgeting(
        pd_covar=covar, constraints=opt.Constraints(is_long_only=True, max_weights=cap),
        risk_budget=budgets) for name, cap in caps.items()}


def assert_raises(error: type, function, **arguments) -> None:
    """Fail unless ``function(**arguments)`` raises ``error``."""
    try:
        function(**arguments)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


class Captured(logging.Handler):
    """Collect the messages of warning records emitted while attached."""

    def __init__(self) -> None:
        """Start with no records."""
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the formatted message."""
        self.messages.append(record.getMessage())


def logged_warnings(function, **arguments) -> tuple:
    """Call ``function`` with package log records captured; return its result and them."""
    package = logging.getLogger('optimalportfolios')
    handler, propagate = Captured(), package.propagate
    package.addHandler(handler)
    package.propagate = False
    try:
        return function(**arguments), handler.messages
    finally:
        package.removeHandler(handler)
        package.propagate = propagate


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import pandas as pd
    import qis
    import optimalportfolios as opt

    assets = ["Equity", "Bonds", "Diversifier"]
    covar = pd.DataFrame(
        [[0.040, 0.004, 0.002],
         [0.004, 0.010, 0.001],
         [0.002, 0.001, 0.022]],
        index=assets,
        columns=assets,
    )
    budgets = pd.Series([0.50, 0.30, 0.20], index=assets)
    constraints = opt.Constraints(
        is_long_only=True,
        max_weights=pd.Series(0.80, index=assets),
    )
    weights = opt.wrapper_risk_budgeting(
        pd_covar=covar,
        constraints=constraints,
        risk_budget=budgets,
    )
    realised_budgets = qis.compute_portfolio_risk_contribution_ratios(
        weights=weights, covar=covar,
    )
    print(pd.concat([weights.rename("weight"),
                     realised_budgets.rename("risk share")], axis=1))

    # The block's inputs are the module constants; 0.040 is an annual volatility of 20%.
    assert assets == ASSETS and covar.to_numpy().tolist() == COVARIANCE
    assert budgets.tolist() == RISK_BUDGETS and constraints.max_weights.eq(SLACK_CAP).all()
    assert np.isclose(np.sqrt(covar.loc["Equity", "Equity"]), 0.20)
    # The weights match an independent conic solve of the same objective, sum to one and meet
    # the budgets exactly; the qis shares equal the Euler shares computed here.
    sigma, target = covar.to_numpy(), budgets.to_numpy()
    np.testing.assert_allclose(weights, conic_reference(sigma, target, [SLACK_CAP] * 3),
                               atol=2e-6, rtol=0.0)
    assert abs(weights.sum() - 1.0) <= 1e-8
    shares = risk_shares(weights, sigma)
    np.testing.assert_allclose(realised_budgets, shares, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(shares, target, atol=1e-6, rtol=0.0)
    # Euler: contributions w_i (Sigma w)_i / sigma_p add up to sigma_p, and a contribution is
    # not the weight times the asset's own volatility.
    w = weights.to_numpy()
    volatility = np.sqrt(w @ sigma @ w)
    contributions = w * (sigma @ w) / volatility
    assert np.isclose(contributions.sum(), volatility, rtol=1e-14)
    assert not np.allclose(contributions, w * np.sqrt(np.diag(sigma)), atol=1e-3)
    # The page's table and prose: about 30.1% of capital carries 50% of the risk.
    np.testing.assert_allclose(weights, [0.301339, 0.441151, 0.257511], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(realised_budgets, [0.50, 0.30, 0.20], atol=1e-6, rtol=0.0)
    assert round(100 * weights["Equity"], 1) == 30.1
    # The 80% cap is slack: without it the solve takes the pure CCD path (default box, no
    # group rows) instead of ADMM and lands on the same weights.
    assert weights.max() < 0.5 < SLACK_CAP
    uncapped = opt.wrapper_risk_budgeting(pd_covar=covar,
                                          constraints=opt.Constraints(is_long_only=True),
                                          risk_budget=budgets)
    np.testing.assert_allclose(uncapped, weights, atol=1e-7, rtol=0.0)
    # Proportional scores define the same target; opt_risk_budgeting is the array entry point
    # below the wrapper; is_long_only=False does not open a short-selling solve.
    scores = opt.wrapper_risk_budgeting(pd_covar=covar,
                                        constraints=opt.Constraints(is_long_only=True),
                                        risk_budget=pd.Series([50, 30, 20], index=assets))
    np.testing.assert_allclose(scores, uncapped, atol=1e-12, rtol=0.0)
    array_weights = opt.opt_risk_budgeting(covar=sigma, constraints=constraints,
                                           risk_budget=target)
    np.testing.assert_allclose(array_weights, weights, atol=1e-12, rtol=0.0)
    not_long_only = opt.wrapper_risk_budgeting(pd_covar=covar,
                                               constraints=opt.Constraints(is_long_only=False),
                                               risk_budget=budgets)
    np.testing.assert_allclose(not_long_only, uncapped, atol=1e-12, rtol=0.0)

    capped_constraints = opt.Constraints(
        is_long_only=True,
        max_weights=pd.Series([0.25, 0.80, 0.80], index=assets),
    )
    capped_weights = opt.wrapper_risk_budgeting(
        pd_covar=covar,
        constraints=capped_constraints,
        risk_budget=budgets,
    )
    capped_shares = qis.compute_portfolio_risk_contribution_ratios(
        weights=capped_weights, covar=covar,
    )
    print(pd.concat([capped_weights.rename("weight"),
                     capped_shares.rename("risk share")], axis=1))

    # The cap binds; the weights match the conic solve and the root of the optimality
    # conditions r_i = b_i + c w_i of the free assets, with a positive multiplier c.
    assert capped_constraints.max_weights.tolist() == [EQUITY_CAP, SLACK_CAP, SLACK_CAP]
    capped = capped_weights.to_numpy()
    upper = capped_constraints.max_weights.to_numpy()
    np.testing.assert_allclose(capped, conic_reference(sigma, target, upper), atol=2e-6,
                               rtol=0.0)
    kkt_weights, multiplier = capped_kkt_reference(sigma, target, capped=0, cap=EQUITY_CAP)
    np.testing.assert_allclose(capped, kkt_weights, atol=1e-8, rtol=0.0)
    assert abs(capped[0] - EQUITY_CAP) <= 1e-9 and abs(capped.sum() - 1.0) <= 1e-9
    capped_share = risk_shares(capped, sigma)
    np.testing.assert_allclose(capped_shares, capped_share, atol=1e-14, rtol=0.0)
    # The page's table: weights 0.250, 0.479, 0.271 and shares 0.394, 0.368, 0.238.
    np.testing.assert_allclose(capped, [0.250, 0.479, 0.271], atol=0.5e-3, rtol=0.0)
    np.testing.assert_allclose(capped_share, [0.394, 0.368, 0.238], atol=0.5e-3, rtol=0.0)
    # Insight: no asset meets its budget. Each free asset overshoots by c = 0.141 times its
    # weight, so Equity falls short by c * (1 - cap) and Bonds takes 1.77 times Diversifier's
    # excess, the ratio of their weights, not the 1.5 of their budgets.
    excess = capped_share - target
    assert excess[0] < -0.1 and (excess[1:] > 0.03).all()
    np.testing.assert_allclose(excess[1:] / capped[1:], multiplier, atol=1e-8, rtol=0.0)
    assert np.isclose(excess[0], -multiplier * (1 - EQUITY_CAP), atol=1e-8)
    assert round(multiplier, 3) == 0.141 and multiplier > 0.0
    assert round(excess[1] / excess[2], 2) == 1.77 == round(capped[1] / capped[2], 2)
    assert np.isclose(target[1] / target[2], 1.5, rtol=1e-15)
    # The solve is not the uncapped solution clipped at the cap with the excess spread pro
    # rata over the free assets.
    clipped = np.r_[EQUITY_CAP, w[1:] / w[1:].sum() * (1 - EQUITY_CAP)]
    assert np.abs(capped - clipped).max() > 5e-3

    diagonal_assets = ["Growth", "Defensive"]
    diagonal_covar = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=diagonal_assets,
        columns=diagonal_assets,
    )
    diagonal_budgets = pd.Series([0.80, 0.20], index=diagonal_assets)
    diagonal_weights = opt.wrapper_risk_budgeting(
        pd_covar=diagonal_covar,
        constraints=opt.Constraints(is_long_only=True),
        risk_budget=diagonal_budgets,
    )
    diagonal_shares = qis.compute_portfolio_risk_contribution_ratios(
        weights=diagonal_weights, covar=diagonal_covar,
    )
    print(diagonal_weights.round(6).tolist())  # [0.5, 0.5]
    print(diagonal_shares.round(6).tolist())   # [0.8, 0.2]

    # The closed form sqrt(b_i) / sigma_i: volatilities 0.20 and 0.10 give equal numerators.
    variances = np.diag(diagonal_covar.to_numpy())
    np.testing.assert_allclose(np.sqrt(variances), [0.20, 0.10], atol=1e-15)
    reference = diagonal_reference(variances, diagonal_budgets)
    np.testing.assert_allclose(reference, [0.5, 0.5], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(diagonal_weights, reference, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(risk_shares(diagonal_weights, diagonal_covar), [0.8, 0.2],
                               atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(diagonal_shares, [0.8, 0.2], atol=1e-6, rtol=0.0)
    assert diagonal_weights.round(6).tolist() == [0.5, 0.5]
    assert diagonal_shares.round(6).tolist() == [0.8, 0.2]
    # Scaling the covariance and the budgets leaves the allocation unchanged while the
    # variances stay above the floor; equal budgets give inverse-volatility, not
    # inverse-variance, weights.
    scaled = opt.wrapper_risk_budgeting(pd_covar=diagonal_covar * 12.0,
                                        constraints=opt.Constraints(is_long_only=True),
                                        risk_budget=diagonal_budgets * 100.0)
    np.testing.assert_allclose(scaled, reference, atol=1e-6, rtol=0.0)
    equal = opt.wrapper_risk_budgeting(pd_covar=diagonal_covar,
                                       constraints=opt.Constraints(is_long_only=True),
                                       risk_budget=pd.Series(0.5, index=diagonal_assets))
    np.testing.assert_allclose(equal, [1 / 3, 2 / 3], atol=1e-6, rtol=0.0)
    assert not np.allclose(equal, [0.2, 0.8], atol=1e-2)

    # Limitations. Zero, negative and missing budgets are excluded from the solve.
    for excluded in ([0.5, 0.0, 0.5], [0.5, -0.2, 0.5], [0.5, np.nan, 0.5]):
        kept = opt.wrapper_risk_budgeting(pd_covar=covar,
                                          constraints=opt.Constraints(is_long_only=True),
                                          risk_budget=pd.Series(excluded, index=assets))
        assert kept["Bonds"] == 0.0
        np.testing.assert_allclose(kept[["Equity", "Diversifier"]],
                                   two_asset_reference(sigma[np.ix_([0, 2], [0, 2])], [1, 1]),
                                   atol=1e-7)
    # A positive variance below 0.001**2 is raised to it; the examples stay above the floor,
    # and expressing the diagonal example in much smaller units activates the floor.
    assert min(np.diag(sigma).min(), variances.min()) > 0.001 ** 2
    pair = ["A", "B"]
    tiny = opt.wrapper_risk_budgeting(
        pd_covar=pd.DataFrame(np.diag([0.04, 1e-8]), index=pair, columns=pair),
        constraints=opt.Constraints(is_long_only=True), risk_budget=pd.Series(0.5, index=pair))
    np.testing.assert_allclose(tiny, diagonal_reference([0.04, 0.001 ** 2], [0.5, 0.5]),
                               atol=1e-9)
    small_units = opt.wrapper_risk_budgeting(
        pd_covar=diagonal_covar * 1e-6, constraints=opt.Constraints(is_long_only=True),
        risk_budget=diagonal_budgets)
    np.testing.assert_allclose(small_units, diagonal_reference([1.0, 1.0], [0.8, 0.2]),
                               atol=1e-7)
    assert np.abs(small_units - reference).max() > 0.1
    # A frozen asset keeps its weight; the others are solved on the reduced covariance and
    # scaled to the remaining capital, so the final shares miss the budgets.
    held = pd.Series([0.40, 0.35, 0.25], index=assets)
    frozen = opt.wrapper_risk_budgeting(pd_covar=covar,
                                        constraints=opt.Constraints(is_long_only=True),
                                        risk_budget=budgets, weights_0=held,
                                        rebalancing_indicators=pd.Series([0, 1, 1], index=assets))
    assert frozen["Equity"] == 0.40
    reduced = two_asset_reference(sigma[1:, 1:], target[1:])
    np.testing.assert_allclose(frozen[["Bonds", "Diversifier"]], 0.60 * reduced, atol=1e-7)
    assert np.abs(risk_shares(frozen, sigma) - target).max() > 0.1
    # With every asset frozen, the wrapper warns and returns zeros, not the frozen book.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        all_frozen = opt.wrapper_risk_budgeting(
            pd_covar=covar, constraints=opt.Constraints(is_long_only=True),
            risk_budget=budgets, weights_0=held,
            rebalancing_indicators=pd.Series(0.0, index=assets))
    assert all_frozen.eq(0.0).all()
    assert any("no valid assets" in str(item.message) for item in caught)
    # The single-asset rolling shortcut returns 100% even for an unusable covariance, and a
    # budget panel must hold a row for every covariance date.
    dates = pd.to_datetime(["2024-03-29", "2024-06-28"])
    prices = pd.DataFrame(100.0, index=pd.bdate_range(dates[0], dates[-1]), columns=assets)
    unusable = covar * np.nan
    single = opt.rolling_risk_budgeting(prices=prices,
                                        constraints=opt.Constraints(is_long_only=True),
                                        risk_budget=pd.Series({"Bonds": 1.0}),
                                        covar_dict={date: unusable for date in dates})
    assert single.index.tolist() == dates.tolist()
    assert single["Bonds"].eq(1.0).all() and single.drop(columns="Bonds").eq(0.0).all().all()
    assert_raises(ValueError, opt.rolling_risk_budgeting, prices=prices,
                  constraints=opt.Constraints(is_long_only=True),
                  risk_budget=pd.DataFrame([budgets], index=dates[:1]),
                  covar_dict={date: covar for date in dates})
    rolling = opt.rolling_risk_budgeting(prices=prices,
                                         constraints=opt.Constraints(is_long_only=True),
                                         risk_budget=budgets,
                                         covar_dict={date: covar for date in dates})
    np.testing.assert_allclose(rolling, [uncapped.to_numpy()] * 2, atol=1e-12)
    # Pitfall: 20% caps on three assets cannot hold a fully invested book. The wrapper logs the
    # solver failure and returns zeros, or the supplied weights_0; it does not raise.
    infeasible = opt.Constraints(is_long_only=True, max_weights=pd.Series(0.20, index=assets))
    assert np.isclose(infeasible.max_weights.sum(), 0.60)
    fallback, messages = logged_warnings(opt.wrapper_risk_budgeting, pd_covar=covar,
                                         constraints=infeasible, risk_budget=budgets)
    assert fallback.eq(0.0).all()
    assert any("sum of upper bounds is below 1" in message for message in messages)
    assert any("falling back to zeros" in message for message in messages)
    previous, _ = logged_warnings(opt.wrapper_risk_budgeting, pd_covar=covar,
                                  constraints=infeasible, risk_budget=budgets, weights_0=held)
    np.testing.assert_allclose(previous, held, atol=0.0)
    # Nonnegative weights can still carry a negative risk contribution.
    hedged = risk_shares([0.2, 0.8], [[0.04, -0.015], [-0.015, 0.01]])
    assert hedged[0] < 0.0 and np.isclose(hedged.sum(), 1.0)
    print("risk_budgeting: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: target and achieved risk shares with and without a binding cap.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import qis

    covar = np.array(COVARIANCE)
    target = np.array(RISK_BUDGETS)
    weights = solves(SLACK_CAP, EQUITY_CAP)
    shares = {name: risk_shares(w, covar) for name, w in weights.items()}
    table = pd.DataFrame({'target_budget': target}, index=ASSETS)
    for name, w in weights.items():
        table[f'{name}: weight'] = w.to_numpy()
        table[f'{name}: risk share'] = shares[name]

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = {'no binding bound': '#2a78d6', 'Equity capped': '#eb6834'}
    labels = {'no binding bound': f'No binding bound ({SLACK_CAP:.0%} caps)',
              'Equity capped': f'Equity capped at {EQUITY_CAP:.0%}'}
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    x = np.arange(len(ASSETS))
    width = 0.36
    for offset, name in zip((-width / 2, width / 2), weights):
        bars = left.bar(x + offset, shares[name], width, color=colours[name], label=labels[name])
        left.bar_label(bars, labels=[f'{value:.0%}' for value in shares[name]], padding=2,
                       fontsize=9, color=ink)
        bars = right.bar(x + offset, weights[name], width, color=colours[name],
                         label=labels[name])
        right.bar_label(bars, labels=[f'{value:.0%}' for value in weights[name]], padding=2,
                        fontsize=9, color=ink)
    for position, value in zip(x, target):
        left.plot([position - 0.46, position + 0.46], [value, value], color=ink, linewidth=1.6,
                  linestyle='--', label='Target budget' if position == 0 else None)
    right.plot([-0.46, 0.46], [EQUITY_CAP, EQUITY_CAP], color=ink, linewidth=1.6,
               linestyle=':', label=f'Equity cap ({EQUITY_CAP:.0%})')
    left.set_title('Share of portfolio risk', loc='left', color=ink)
    right.set_title('Capital weight', loc='left', color=ink)
    left.legend(frameon=False, loc='upper right', fontsize=9, labelcolor=ink)
    # The bar colours are explained on the left; the right legend adds only the cap line.
    cap_line = [line for line in right.get_lines() if line.get_linestyle() == ':']
    right.legend(handles=cap_line, frameon=False, loc='upper right', fontsize=9,
                 labelcolor=ink)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.set_xticks(x, ASSETS)
        axis.set_ylim(0, 0.62)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    free = shares['Equity capped'][1:] - target[1:]
    capped = weights['Equity capped'].to_numpy()
    checks = {
        'slack_solve_meets_budgets': bool(np.allclose(shares['no binding bound'], target,
                                                      atol=1e-6)),
        'shares_sum_to_one': bool(all(np.isclose(s.sum(), 1.0) for s in shares.values())),
        'equity_cap_binds': bool(np.isclose(capped[0], EQUITY_CAP, atol=1e-9)),
        'capped_asset_below_budget': bool(shares['Equity capped'][0] < target[0] - 0.1),
        'free_assets_above_budget': bool((free > 0.0).all()),
        'excess_proportional_to_weight': bool(np.isclose(free[0] / capped[1],
                                                         free[1] / capped[2], atol=1e-8)),
        'qis_shares_match_euler_formula': bool(all(
            np.allclose(qis.compute_portfolio_risk_contribution_ratios(
                weights=w.to_numpy(), covar=covar), shares[name], atol=1e-14)
            for name, w in weights.items())),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
