"""Canonical script of docs/strategic_allocation_targets.md.

The page shows excerpts of this file; every number it quotes is asserted here against a
reference computed a different way. The script runs offline after ``pip install
optimalportfolios``: its data is the monthly multi-asset fixture packaged with the package's
tests, and its estimation settings and constraints are those of
``examples/backtests/multiasset_saa.py``:

    python -m examples.docs.strategic_allocation_targets

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import qis

import optimalportfolios as op
from optimalportfolios.tests.data.multiasset import load_multiasset_data

# Settings of examples/backtests/multiasset_saa.py: a 36-month EWMA covariance of monthly
# returns at year ends from 2005, long-only weights capped at 25% and asset-class bands.
RETURNS_FREQ = 'ME'
EWMA_SPAN = 36
REBALANCING_FREQ = 'YE'
BACKTEST_START = '31Dec2005'
MAX_WEIGHT = 0.25
GROUP_MIN = {'Fixed Income': 0.20}
GROUP_MAX = {'Equity': 0.60, 'Alternatives': 0.40}
DATE = '2025-12-31'
# Stylised capital market assumptions: the cash rate plus one Sharpe ratio times volatility.
CASH_RATE = 0.02
SHARPE_RATIO = 0.30
TARGET_RETURN = 0.045
TARGET_VOL = 0.05
# Targets and penalty weights of the figure.
FRONTIER_RETURNS = [0.03, 0.035, 0.04, 0.045, 0.05, 0.055]
FRONTIER_VOLS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
PENALTY_WEIGHTS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
WEIGHT_TOL = 1e-3  # solver weights agree to this; returns and volatilities to MOMENT_TOL
MOMENT_TOL = 1e-5


def saa_constraints(group_data: pd.Series) -> op.Constraints:
    """Return the long-only constraints with a 25% cap and the asset-class bands."""
    loadings = qis.set_group_loadings(group_data=group_data)
    group_min = pd.Series(GROUP_MIN).reindex(loadings.columns).fillna(0.0)
    group_max = pd.Series(GROUP_MAX).reindex(loadings.columns).fillna(1.0)
    return op.Constraints(
        is_long_only=True, max_weights=pd.Series(MAX_WEIGHT, index=group_data.index),
        group_lower_upper_constraints=op.GroupLowerUpperConstraints(
            group_loadings=loadings, group_min_allocation=group_min,
            group_max_allocation=group_max))


def equal_sharpe_cmas(covar: pd.DataFrame) -> pd.Series:
    """Return the stylised expected returns: cash rate plus Sharpe ratio times volatility."""
    return CASH_RATE + SHARPE_RATIO * pd.Series(np.sqrt(np.diag(covar)), index=covar.index)


def saa_inputs() -> tuple:
    """Return the fixture, the year-end covariances, and the covariance, CMAs and constraints."""
    data = load_multiasset_data()
    estimator = op.EwmaCovarEstimator(returns_freq=RETURNS_FREQ, span=EWMA_SPAN,
                                      rebalancing_freq=REBALANCING_FREQ)
    covar_dict = estimator.fit_rolling_covars(
        prices=data.prices,
        time_period=qis.TimePeriod(start=BACKTEST_START, end=data.prices.index[-1]))
    covar = covar_dict[pd.Timestamp(DATE)]
    cmas = equal_sharpe_cmas(covar)
    constraints = saa_constraints(data.group_data)
    return data, covar_dict, covar, cmas, constraints


def volatility(weights: pd.Series, covar: pd.DataFrame) -> float:
    """Return the portfolio volatility from the explicit quadratic form."""
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ covar.to_numpy() @ w))


def raw_rows(w: cp.Variable, group_data: pd.Series) -> list:
    """Return the budget, long-only, cap and band rows from raw arrays, as a reference."""
    dummies = pd.get_dummies(group_data).astype(float)
    lower = np.array([GROUP_MIN.get(group, 0.0) for group in dummies.columns])
    upper = np.array([GROUP_MAX.get(group, 1.0) for group in dummies.columns])
    loadings = dummies.to_numpy()
    return [cp.sum(w) == 1.0, w >= 0.0, w <= MAX_WEIGHT,
            loadings.T @ w >= lower, loadings.T @ w <= upper]


def frontier_ends(covar: pd.DataFrame, cmas: pd.Series, group_data: pd.Series) -> dict:
    """Solve the frontier's ends with CVXPY on raw arrays, independently of the package."""
    sigma, mu = covar.to_numpy(), cmas.to_numpy()
    w = cp.Variable(len(mu))
    cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), raw_rows(w, group_data)).solve(
        solver='CLARABEL')
    w_mv = w.value.copy()
    cp.Problem(cp.Maximize(mu @ w), raw_rows(w, group_data)).solve(solver='CLARABEL')
    r_max = float(mu @ w.value)
    cp.Problem(cp.Minimize(cp.quad_form(w, sigma)),
               raw_rows(w, group_data) + [mu @ w >= r_max - 1e-9]).solve(solver='CLARABEL')
    return {'w_mv': w_mv, 'r_max': r_max, 'w_top': w.value.copy()}


def shadow_price(covar: pd.DataFrame, cmas: pd.Series, group_data: pd.Series,
                 target_vol: float) -> float:
    """Return the dual value of the variance row of the hard problem, solved on raw arrays."""
    w = cp.Variable(len(cmas))
    variance_row = cp.quad_form(w, covar.to_numpy()) <= target_vol ** 2
    cp.Problem(cp.Maximize(cmas.to_numpy() @ w),
               raw_rows(w, group_data) + [variance_row]).solve(solver='CLARABEL')
    return float(np.squeeze(variance_row.dual_value))


def utility_reference(covar: pd.DataFrame, cmas: pd.Series, group_data: pd.Series,
                      phi: float) -> np.ndarray:
    """Solve max mu'w - phi w'Sigma w on raw arrays, independently of the package."""
    w = cp.Variable(len(cmas))
    cp.Problem(cp.Maximize(cmas.to_numpy() @ w - phi * cp.quad_form(w, covar.to_numpy())),
               raw_rows(w, group_data)).solve(solver='CLARABEL')
    return w.value


def main() -> None:
    """Run the worked example of the page and assert every quoted number."""
    data = load_multiasset_data()
    estimator = op.EwmaCovarEstimator(returns_freq=RETURNS_FREQ, span=EWMA_SPAN,
                                      rebalancing_freq=REBALANCING_FREQ)
    covar_dict = estimator.fit_rolling_covars(
        prices=data.prices,
        time_period=qis.TimePeriod(start=BACKTEST_START, end=data.prices.index[-1]))
    covar = covar_dict[pd.Timestamp(DATE)]
    cmas = equal_sharpe_cmas(covar)
    constraints = saa_constraints(data.group_data)
    assert covar.shape == (19, 19) and len(covar_dict) == 21
    assert np.linalg.eigvalsh(covar.to_numpy()).min() > 1e-5
    assert round(cmas['Cash'], 4) == 0.0215 and round(cmas.max(), 4) == 0.0627
    assert cmas.idxmax() == 'Asia Ex-Japan'
    floored = op.factorize_covariance(covar.to_numpy()).covar  # the matrix the solvers use
    assert np.abs(floored - covar.to_numpy()).max() < 1e-12

    # The ends of the frontier: the minimum-variance portfolio, and the highest return.
    w_mv, _ = op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=cmas, target_return=0.0, constraints=constraints)
    r_mv, vol_mv = cmas @ w_mv, volatility(w_mv, covar)
    ends = frontier_ends(covar, cmas, data.group_data)
    assert np.abs(w_mv.to_numpy() - ends['w_mv']).max() < WEIGHT_TOL
    assert round(r_mv, 4) == 0.0290 and round(vol_mv, 4) == 0.0186
    assert abs(w_mv['Cash'] - MAX_WEIGHT) < 1e-6 and abs(w_mv['Global Bonds'] - MAX_WEIGHT) < 1e-6
    r_max, vol_top = ends['r_max'], volatility(ends['w_top'], covar)
    assert round(r_max, 3) == 0.055 and round(vol_top, 4) == 0.0965

    # Round trip from a target return: its volatility, fed back, returns the same portfolio.
    w_return, outcome = op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=cmas, target_return=TARGET_RETURN,
        constraints=constraints)
    vol_return = volatility(w_return, covar)
    w_back, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=vol_return, constraints=constraints)
    assert outcome.accepted and abs(cmas @ w_return - TARGET_RETURN) < MOMENT_TOL
    assert np.abs(w_back - w_return).max() < WEIGHT_TOL
    assert abs(cmas @ w_back - TARGET_RETURN) < MOMENT_TOL and round(vol_return, 4) == 0.0432

    # Round trip from a target volatility: its return, fed back, returns the same portfolio.
    w_vol, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=TARGET_VOL, constraints=constraints)
    return_vol = cmas @ w_vol
    w_forth, _ = op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=cmas, target_return=return_vol,
        constraints=constraints)
    assert abs(volatility(w_vol, covar) - TARGET_VOL) < MOMENT_TOL
    assert np.abs(w_forth - w_vol).max() < WEIGHT_TOL
    assert abs(volatility(w_forth, covar) - TARGET_VOL) < MOMENT_TOL
    assert round(return_vol, 4) == 0.0481
    assert abs(w_vol.groupby(data.group_data).sum()['Alternatives'] - 0.40) < 1e-6

    # Proposition 1 on every target of the figure, both ways.
    for target in FRONTIER_RETURNS:
        w_1, _ = op.wrapper_min_variance_target_return(
            pd_covar=covar, expected_returns=cmas, target_return=target, constraints=constraints)
        w_2, _ = op.wrapper_max_return_target_vol(
            pd_covar=covar, expected_returns=cmas, target_vol=volatility(w_1, covar),
            constraints=constraints)
        assert np.abs(w_1 - w_2).max() < WEIGHT_TOL and abs(cmas @ w_2 - target) < MOMENT_TOL
    for target in FRONTIER_VOLS:
        w_2, _ = op.wrapper_max_return_target_vol(
            pd_covar=covar, expected_returns=cmas, target_vol=target, constraints=constraints)
        w_1, _ = op.wrapper_min_variance_target_return(
            pd_covar=covar, expected_returns=cmas, target_return=cmas @ w_2,
            constraints=constraints)
        assert np.abs(w_1 - w_2).max() < WEIGHT_TOL
        assert abs(volatility(w_1, covar) - target) < MOMENT_TOL

    # The utility form at the shadow price of the 5% variance row is the hard solution.
    phi_star = shadow_price(covar, cmas, data.group_data, TARGET_VOL)
    soft = constraints.copy(
        constraint_enforcement_type=op.ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        tre_utility_weight=phi_star)
    w_soft, soft_outcome = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=TARGET_VOL, constraints=soft)
    assert soft_outcome.accepted and np.abs(w_soft - w_vol).max() < WEIGHT_TOL
    assert round(phi_star, 2) == 4.46
    w_quadratic, _ = op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=constraints, means=cmas, carra=2.0 * phi_star,
        portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY)
    assert np.abs(w_quadratic - w_vol).max() < WEIGHT_TOL  # risk aversion 2 phi, same portfolio

    # Proposition 2: each soft solution is on the frontier, and its risk falls as phi grows.
    soft_vols, soft_returns = [], []
    for phi in PENALTY_WEIGHTS:
        w_phi, _ = op.wrapper_max_return_target_vol(
            pd_covar=covar, expected_returns=cmas, target_vol=TARGET_VOL,
            constraints=soft.copy(tre_utility_weight=phi))
        w_hard, _ = op.wrapper_max_return_target_vol(
            pd_covar=covar, expected_returns=cmas, target_vol=volatility(w_phi, covar),
            constraints=constraints)
        assert np.abs(w_hard - w_phi).max() < WEIGHT_TOL
        tracking_error = volatility(w_phi - w_mv, covar)
        assert tracking_error <= np.sqrt((r_max - r_mv) / phi)
        soft_vols.append(volatility(w_phi, covar))
        soft_returns.append(cmas @ w_phi)
    assert np.all(np.diff(soft_vols) < 0.0) and np.all(np.diff(soft_returns) < 0.0)
    assert all((vol > TARGET_VOL) == (phi < phi_star)
               for phi, vol in zip(PENALTY_WEIGHTS, soft_vols))
    assert [round(vol, 4) for vol in soft_vols[1:4]] == [0.0707, 0.0599, 0.0460]

    # Pitfall: the default utility form ignores target_vol; it solves with phi = 1.
    utility = constraints.copy(
        constraint_enforcement_type=op.ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    w_3, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=0.03, constraints=utility)
    w_7, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=0.07, constraints=utility)
    assert np.abs(w_3 - w_7).max() < 1e-9 and utility.tre_utility_weight == 1.0
    reference = utility_reference(covar, cmas, data.group_data, phi=1.0)
    assert np.abs(w_3.to_numpy() - reference).max() < WEIGHT_TOL
    assert round(volatility(w_3, covar), 4) == 0.0707

    # Remark: with weights_0 meeting the hard rows, the default turnover penalty keeps it.
    w_held, _ = op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=cmas, target_return=TARGET_RETURN,
        constraints=utility, weights_0=w_vol)
    threshold = np.abs(2.0 * covar.to_numpy() @ w_vol.to_numpy()).max()
    assert np.abs(w_held - w_vol).max() < 1e-6 and utility.turnover_utility_weight == 0.40
    assert round(threshold, 4) == 0.0089 and utility.turnover_utility_weight > threshold
    w_free, _ = op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=cmas, target_return=TARGET_RETURN,
        constraints=utility)
    assert np.abs(w_free - w_return).max() < 1e-6  # no weights_0: no penalty, the hard solve

    # Targets outside the frontier: rejected solves fall back to zeros on a single date.
    w_high, high = op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=cmas, target_return=0.056, constraints=constraints)
    w_low, low = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=0.015, constraints=constraints)
    for weights, result in ((w_high, high), (w_low, low)):
        assert not result.accepted and 'infeasible' in result.status
        assert result.fallback_source == 'zeros' and (weights == 0.0).all()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        w_clamped, clamped = op.wrapper_min_variance_target_return(
            pd_covar=covar, expected_returns=cmas, target_return=0.07,
            constraints=constraints)
    assert any('max asset return (6.2700%). Clamping' in str(item.message) for item in caught)
    assert not clamped.accepted and 'infeasible' in clamped.status and (w_clamped == 0.0).all()
    w_slack, _ = op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=cmas, target_return=0.02, constraints=constraints)
    assert np.abs(w_slack - w_mv).max() < WEIGHT_TOL and round(cmas @ w_slack, 4) == 0.0290
    w_wide, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=0.12, constraints=constraints)
    assert np.abs(w_wide.to_numpy() - ends['w_top']).max() < WEIGHT_TOL
    assert abs(cmas @ w_wide - r_max) < MOMENT_TOL
    assert round(volatility(w_wide, covar), 4) == 0.0965

    # Benchmark-relative forms: tracking error replaces volatility and the round trip holds.
    benchmark = pd.Series(1.0 / len(cmas), index=cmas.index)
    w_te, _ = op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=cmas, target_return=TARGET_RETURN,
        constraints=constraints, benchmark_weights=benchmark)
    te = volatility(w_te - benchmark, covar)
    w_te_back, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=te, constraints=constraints,
        benchmark_weights=benchmark)
    assert np.abs(w_te_back - w_te).max() < WEIGHT_TOL
    assert abs(volatility(w_te_back - benchmark, covar) - te) < MOMENT_TOL

    # Rolling: CMAs per year end and one target, forward-filled to the covariance dates.
    cma_table = pd.DataFrame(
        {date: equal_sharpe_cmas(estimate) for date, estimate in covar_dict.items()}).T
    dates = list(covar_dict)
    saa_return = op.rolling_min_variance_target_return(
        prices=data.prices, expected_returns=cma_table,
        target_returns=pd.Series(TARGET_RETURN, index=dates[:1]),
        constraints=constraints, benchmark_weights=None, covar_dict=covar_dict)
    path_vols = pd.Series(
        {date: volatility(saa_return.loc[date], covar_dict[date]) for date in dates})
    saa_vol = op.rolling_max_return_target_vol(
        prices=data.prices, expected_returns=cma_table, target_vols=path_vols,
        constraints=constraints, benchmark_weights=None, covar_dict=covar_dict)
    assert np.allclose((saa_return * cma_table).sum(axis=1), TARGET_RETURN, atol=MOMENT_TOL)
    assert (saa_vol - saa_return).abs().max().max() < WEIGHT_TOL
    assert dates[0] == pd.Timestamp('2005-12-31') and dates[-1] == pd.Timestamp(DATE)

    # Limitations: dates before the first CMA row get zero expected returns, and dates
    # before the first target get a NaN target, which CVXPY refuses.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        late_cmas = op.rolling_min_variance_target_return(
            prices=data.prices, expected_returns=cma_table.iloc[1:],
            target_returns=pd.Series(TARGET_RETURN, index=dates[:1]),
            constraints=constraints, benchmark_weights=None, covar_dict=covar_dict)
    first_mv, _ = op.wrapper_min_variance_target_return(
        pd_covar=covar_dict[dates[0]], expected_returns=cma_table.loc[dates[0]],
        target_return=0.0, constraints=constraints)
    assert np.abs(late_cmas.loc[dates[0]] - first_mv).max() < WEIGHT_TOL
    assert any('max asset return (0.0000%)' in str(item.message) for item in caught)
    late_vol = op.rolling_max_return_target_vol(
        prices=data.prices, expected_returns=cma_table.iloc[1:],
        target_vols=pd.Series(TARGET_VOL, index=dates[:1]),
        constraints=constraints, benchmark_weights=None, covar_dict=covar_dict)
    assert abs(late_vol.loc[dates[0]].sum() - 1.0) < 1e-6  # feasible, with a zero objective

    # Limitations: a fixed phi does not hold a volatility target across dates.
    fixed_phi = op.rolling_max_return_target_vol(
        prices=data.prices, expected_returns=cma_table,
        target_vols=pd.Series(TARGET_VOL, index=dates[:1]),
        constraints=soft.copy(turnover_utility_weight=None), benchmark_weights=None,
        covar_dict=covar_dict)
    fixed_vols = [volatility(fixed_phi.loc[date], covar_dict[date]) for date in dates]
    assert abs(fixed_vols[-1] - TARGET_VOL) < MOMENT_TOL
    assert round(min(fixed_vols), 3) == 0.041 and round(max(fixed_vols), 3) == 0.053

    # Limitations: with default utility settings, the rolling return floor keeps the drifted
    # portfolio whenever it still meets the floor, the cap and the bands (the Remark).
    held = op.rolling_min_variance_target_return(
        prices=data.prices, expected_returns=cma_table,
        target_returns=pd.Series(TARGET_RETURN, index=dates[:1]),
        constraints=utility, benchmark_weights=None, covar_dict=covar_dict)
    held_years = 0
    for previous, date in zip(dates[:-1], dates[1:]):
        growth = data.prices.loc[:date].iloc[-1] / data.prices.loc[:previous].iloc[-1]
        drifted = held.loc[previous] * growth / (held.loc[previous] * growth).sum()
        bands = drifted.groupby(data.group_data).sum()
        feasible = (cma_table.loc[date] @ drifted >= TARGET_RETURN - 1e-9
                    and drifted.max() <= MAX_WEIGHT + 1e-9
                    and all(bands[group] >= floor - 1e-9 for group, floor in GROUP_MIN.items())
                    and all(bands[group] <= cap + 1e-9 for group, cap in GROUP_MAX.items()))
        assert (np.abs(held.loc[date] - drifted).max() < 1e-6) == feasible
        held_years += feasible
    assert held_years == 8 and len(dates) - 1 == 20
    try:
        op.rolling_min_variance_target_return(
            prices=data.prices, expected_returns=cma_table,
            target_returns=pd.Series(TARGET_RETURN, index=dates[1:2]),
            constraints=constraints, benchmark_weights=None, covar_dict=covar_dict)
    except ValueError as error:
        assert 'NaN' in str(error)
    else:
        raise AssertionError('a target series that starts after the first date should raise')
    print('strategic_allocation_targets: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: three routes to one frontier, and the soft path against phi.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data, _, covar, cmas, constraints = saa_inputs()
    soft = constraints.copy(
        constraint_enforcement_type=op.ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    ends = frontier_ends(covar, cmas, data.group_data)
    vol_mv, r_mv = volatility(ends['w_mv'], covar), float(cmas @ ends['w_mv'])
    vol_top = volatility(ends['w_top'], covar)
    rows, round_trip = [], True
    for target in FRONTIER_RETURNS:
        w_1, _ = op.wrapper_min_variance_target_return(
            pd_covar=covar, expected_returns=cmas, target_return=target, constraints=constraints)
        w_2, _ = op.wrapper_max_return_target_vol(
            pd_covar=covar, expected_returns=cmas, target_vol=volatility(w_1, covar),
            constraints=constraints)
        round_trip &= bool(np.abs(w_1 - w_2).max() < WEIGHT_TOL)
        rows.append(('target return', target, volatility(w_1, covar), cmas @ w_1))
    for target in FRONTIER_VOLS:
        w_2, _ = op.wrapper_max_return_target_vol(
            pd_covar=covar, expected_returns=cmas, target_vol=target, constraints=constraints)
        w_1, _ = op.wrapper_min_variance_target_return(
            pd_covar=covar, expected_returns=cmas, target_return=cmas @ w_2,
            constraints=constraints)
        round_trip &= bool(np.abs(w_1 - w_2).max() < WEIGHT_TOL)
        rows.append(('target volatility', target, volatility(w_2, covar), cmas @ w_2))
    on_frontier = True
    for phi in np.geomspace(0.3, 100.0, 31).tolist() + PENALTY_WEIGHTS:
        w_phi, _ = op.wrapper_max_return_target_vol(
            pd_covar=covar, expected_returns=cmas, target_vol=TARGET_VOL,
            constraints=soft.copy(tre_utility_weight=phi))
        w_hard, _ = op.wrapper_max_return_target_vol(
            pd_covar=covar, expected_returns=cmas, target_vol=volatility(w_phi, covar),
            constraints=constraints)
        on_frontier &= bool(np.abs(w_hard - w_phi).max() < WEIGHT_TOL)
        route = 'utility' if phi in PENALTY_WEIGHTS else 'utility path'
        rows.append((route, phi, volatility(w_phi, covar), cmas @ w_phi))
    phi_star = shadow_price(covar, cmas, data.group_data, TARGET_VOL)
    w_star, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=cmas, target_vol=TARGET_VOL,
        constraints=soft.copy(tre_utility_weight=phi_star))
    rows.append(('utility at shadow price', phi_star, volatility(w_star, covar), cmas @ w_star))
    frontier = []
    for target in np.linspace(r_mv, ends['r_max'], 40):
        w_f, _ = op.wrapper_min_variance_target_return(
            pd_covar=covar, expected_returns=cmas, target_return=float(target),
            constraints=constraints)
        frontier.append((volatility(w_f, covar), cmas @ w_f))
        rows.append(('frontier', float(target), *frontier[-1]))
    table = pd.DataFrame(rows, columns=['route', 'parameter', 'volatility', 'expected_return'])
    frontier = np.array(frontier)

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange, aqua, grey = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b3'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    left.plot(100 * frontier[:, 0], 100 * frontier[:, 1], color=grey, linewidth=2.0, zorder=1,
              label='frontier')
    styles = (('target return', 'minimum variance at a target return',
               {'s': 90, 'marker': 'o', 'facecolors': 'none', 'edgecolors': blue}),
              ('target volatility', 'maximum return at a target volatility',
               {'s': 60, 'marker': 'x', 'color': orange}),
              ('utility', 'utility form at a penalty weight φ',
               {'s': 26, 'marker': 'D', 'color': aqua}))
    for route, label, style in styles:
        points = table[table['route'] == route]
        left.scatter(100 * points['volatility'], 100 * points['expected_return'],
                     linewidths=1.5, zorder=3, label=label, **style)
    left.annotate('minimum variance', (100 * vol_mv, 100 * r_mv), xytext=(3.2, 2.75),
                  color=ink, fontsize=10, arrowprops={'arrowstyle': '-', 'color': muted})
    left.annotate('highest return', (100 * vol_top, 100 * ends['r_max']), xytext=(7.3, 4.75),
                  color=ink, fontsize=10, arrowprops={'arrowstyle': '-', 'color': muted})
    left.set_title('Three routes to one frontier', loc='left', color=ink)
    left.set_xlabel('Volatility, % a year')
    left.set_ylabel('Expected return, % a year')
    left.set_xlim(0.0, 10.5)
    left.set_ylim(2.5, 7.0)
    left.legend(frameon=False, loc='upper left', fontsize=10, labelcolor=ink)

    path_points = table[table['route'] == 'utility path']
    marked = table[table['route'] == 'utility']
    right.plot(path_points['parameter'], 100 * path_points['volatility'], color=aqua,
               linewidth=2.0, zorder=2)
    right.scatter(marked['parameter'], 100 * marked['volatility'], s=26, marker='D', color=aqua,
                  zorder=3, label='utility form at a penalty weight φ')
    right.axhline(100 * TARGET_VOL, color=muted, linestyle='--', linewidth=1.2)
    right.axhline(100 * vol_mv, color=grey, linestyle=':', linewidth=1.2)
    right.scatter([phi_star], [100 * TARGET_VOL], s=110, marker='x', color=orange,
                  linewidths=2.0, zorder=4, label='maximum return at 5% volatility')
    right.text(0.32, 100 * TARGET_VOL + 0.15, 'hard target 5%', color=ink, fontsize=10,
               va='bottom')
    right.text(phi_star * 1.15, 100 * TARGET_VOL + 0.15, f'φ* = {phi_star:.2f}', color=ink,
               fontsize=10, va='bottom')
    right.text(0.32, 100 * vol_mv + 0.15, f'minimum variance {100 * vol_mv:.2f}%', color=ink,
               fontsize=10, va='bottom')
    right.set_xscale('log')
    right.set_xticks([0.3, 1.0, 3.0, 10.0, 30.0, 100.0], ['0.3', '1', '3', '10', '30', '100'])
    right.minorticks_off()
    right.set_title('The utility form crosses the target at φ*', loc='left', color=ink)
    right.set_xlabel('Penalty weight φ, tre_utility_weight (log scale)')
    right.set_ylabel('Volatility, % a year')
    right.set_ylim(0.0, 10.5)
    right.legend(frameon=False, loc='upper right', fontsize=10, labelcolor=ink)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.grid(color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    utility_vols = marked['volatility'].to_numpy()
    checks = {
        'round_trips_return_the_same_weights': round_trip,
        'utility_solutions_are_on_the_frontier': on_frontier,
        'utility_at_shadow_price_is_the_hard_solution': bool(
            abs(volatility(w_star, covar) - TARGET_VOL) < MOMENT_TOL),
        'utility_volatility_falls_as_phi_grows': bool(np.all(np.diff(utility_vols) < 0.0)),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
