"""Canonical script of docs/app_cma_strategic_allocation.md.

The case study follows the workflow that Sepp, Hansen and Kastenholz (2026) describe, from
capital market assumptions (CMAs) built on multi-asset tradable factors to a strategic
allocation. It reproduces none of the paper's data, numbers or results. It builds the workflow
from the package's documented pieces on fixed synthetic inputs: one loading matrix gives both the
CMAs and the covariance, and the strategic allocation maximises the CMA-weighted active return
against each mandate benchmark within a tracking-error budget. Every number and property the page
states is asserted against a reference computed a different way: element-wise sums, a rank and
projection check, an active-set closed form with its optimality conditions, and tracking error as
an explicit quadratic form. The script runs offline after ``pip install optimalportfolios`` and
needs no data file or random seed:

    python -m examples.docs.app_cma_strategic_allocation

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import numpy as np
import pandas as pd

FACTORS = ['Equity', 'Rates', 'Credit', 'Commodities']
ASSETS = ['Govt bonds', 'IG credit', 'HY credit', 'DM equity', 'EM equity', 'Real estate',
          'Hedge funds', 'Commodities']
# Synthetic loadings of each asset (rows) on each factor (columns).
LOADINGS = [
    [0.0, 1.0, 0.0, 0.0],
    [0.1, 0.7, 0.6, 0.0],
    [0.4, 0.2, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [1.2, 0.0, 0.2, 0.2],
    [0.6, 0.4, 0.2, 0.0],
    [0.3, 0.0, 0.2, 0.1],
    [0.2, -0.1, 0.0, 1.0],
]
# Annual factor volatilities and correlations, and the annual premium per unit of loading.
FACTOR_VOLS = [0.15, 0.06, 0.05, 0.18]
FACTOR_CORR = [
    [1.0, -0.2, 0.5, 0.3],
    [-0.2, 1.0, 0.1, -0.1],
    [0.5, 0.1, 1.0, 0.2],
    [0.3, -0.1, 0.2, 1.0],
]
PREMIA = [0.045, 0.010, 0.015, 0.020]
# Annual residual volatilities and the residual adjustments of the CMAs.
RESIDUAL_VOLS = [0.02, 0.02, 0.03, 0.03, 0.05, 0.08, 0.035, 0.06]
ADJUSTMENTS = [0.0, 0.0, 0.0, 0.0, 0.0, -0.01, 0.01, 0.0]
RISK_FREE_RATE = 0.03
TE_BUDGET = 0.01
# Three mandate benchmarks that share the alternatives and high yield and differ in the rest.
BENCHMARKS = {
    'Conservative': [0.35, 0.20, 0.05, 0.15, 0.05, 0.05, 0.10, 0.05],
    'Balanced': [0.20, 0.15, 0.05, 0.30, 0.10, 0.05, 0.10, 0.05],
    'Growth': [0.10, 0.05, 0.05, 0.45, 0.15, 0.05, 0.10, 0.05],
}
SHOWN_MANDATE = 'Balanced'  # the mandate whose active weights the page tabulates and draws
MISSING_ASSET = 'EM equity'  # the asset whose CMA the pitfall leaves out of a vintage
HELD = 1e-6  # a weight above this is a held asset


def model_inputs() -> dict:
    """Return the loadings, premia, factor covariance, residual variances and adjustments."""
    vols = np.array(FACTOR_VOLS)
    return {
        'beta': pd.DataFrame(LOADINGS, index=ASSETS, columns=FACTORS),
        'premia': pd.Series(PREMIA, index=FACTORS),
        'factor_covar': pd.DataFrame(np.outer(vols, vols) * np.array(FACTOR_CORR),
                                     index=FACTORS, columns=FACTORS),
        'residual_var': pd.Series(np.square(RESIDUAL_VOLS), index=ASSETS),
        'adjustments': pd.Series(ADJUSTMENTS, index=ASSETS),
    }


def cma_and_covariance(inputs: dict) -> tuple:
    """Return the CMAs and the covariance that the model inputs imply."""
    beta = inputs['beta']
    cma = RISK_FREE_RATE + beta @ inputs['premia'] + inputs['adjustments']
    covar = beta @ inputs['factor_covar'] @ beta.T + np.diag(inputs['residual_var'])
    return cma, covar


def solve_saa(covar: pd.DataFrame, alphas: pd.Series, benchmark: pd.Series) -> tuple:
    """Solve alpha over tracking error with the page's long-only constraints."""
    import optimalportfolios as op

    return op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
        constraints=op.Constraints(is_long_only=True, tracking_err_vol_constraint=TE_BUDGET))


def explicit_tracking_error(weights, benchmark, covar) -> float:
    """Return sqrt((w - b)' Sigma (w - b)) with NumPy, independently of the package and qis."""
    active = np.asarray(weights, dtype=float) - np.asarray(benchmark, dtype=float)
    return float(np.sqrt(active @ np.asarray(covar, dtype=float) @ active))


def active_set_solution(alphas, covar, benchmark, budget: float, zero: list) -> tuple:
    """Solve max alphas'd subject to d'Sigma d = budget^2, 1'w = 1 and w = 0 on ``zero``.

    With the zero set fixed, stationarity gives the free active weights as s p + q for a scalar
    s = 1 / (2 nu) that the tracking-error row fixes, where nu is its multiplier. The returned
    multipliers of the zero bounds, together with positive free weights and s > 0, certify the
    solution of the long-only problem.

    Args:
        alphas: Expected returns, one per asset.
        covar: Covariance matrix.
        benchmark: Benchmark weights summing to one.
        budget: Tracking-error budget, in the square-root units of the covariance.
        zero: Positions of the assets held at zero weight.

    Returns:
        The weights, the multipliers of the zero bounds and s.
    """
    alphas = np.asarray(alphas, dtype=float)
    covar = np.asarray(covar, dtype=float)
    benchmark = np.asarray(benchmark, dtype=float)
    free = np.ones(len(alphas), dtype=bool)
    free[zero] = False
    d_zero = -benchmark[~free]
    s_ff, s_fz = covar[np.ix_(free, free)], covar[np.ix_(free, ~free)]
    a = np.linalg.solve(s_ff, alphas[free])
    u = np.linalg.solve(s_ff, np.ones(free.sum()))
    g = np.linalg.solve(s_ff, s_fz @ d_zero)
    c = -d_zero.sum()
    direction, offset = np.zeros(len(alphas)), np.zeros(len(alphas))
    direction[free] = a - a.sum() / u.sum() * u
    offset[free] = (c + g.sum()) / u.sum() * u - g
    offset[~free] = d_zero
    qa = direction @ covar @ direction
    qb = 2.0 * direction @ covar @ offset
    qc = offset @ covar @ offset - budget ** 2
    assert qa > 0.0 and qc < 0.0  # one positive root: the budget binds along the direction
    s = (-qb + np.sqrt(qb ** 2 - 4.0 * qa * qc)) / (2.0 * qa)
    active = s * direction + offset
    eta = (a.sum() - (c + g.sum()) / s) / u.sum()
    multipliers = (covar @ active)[~free] / s - alphas[~free] + eta
    return benchmark + active, multipliers, s


def certified(weights: pd.Series, alphas: pd.Series, covar, benchmark: pd.Series) -> np.ndarray:
    """Return the closed-form weights on the solver's zero set after checking optimality."""
    zero = list(np.flatnonzero(weights.to_numpy() <= HELD))
    reference, multipliers, s = active_set_solution(alphas, covar, benchmark, TE_BUDGET, zero)
    free = np.setdiff1d(np.arange(len(reference)), zero)
    assert s > 0.0 and (reference[free] > 10 * HELD).all() and (multipliers > 0.0).all()
    return reference


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import numpy as np
    import pandas as pd
    import optimalportfolios as op

    beta = pd.DataFrame(LOADINGS, index=ASSETS, columns=FACTORS)
    premia = pd.Series(PREMIA, index=FACTORS)
    factor_covar = pd.DataFrame(np.outer(FACTOR_VOLS, FACTOR_VOLS) * np.array(FACTOR_CORR),
                                index=FACTORS, columns=FACTORS)
    residual_var = pd.Series(np.square(RESIDUAL_VOLS), index=ASSETS)
    adjustments = pd.Series(ADJUSTMENTS, index=ASSETS)

    cma = RISK_FREE_RATE + beta @ premia + adjustments
    covar = beta @ factor_covar @ beta.T + np.diag(residual_var)

    # The inputs the page tabulates.
    assert np.round(100 * np.array(FACTOR_VOLS)).tolist() == [15, 6, 5, 18]
    assert np.round(1000 * np.array(PREMIA)).tolist() == [45, 10, 15, 20]
    assert np.round(1000 * np.array(RESIDUAL_VOLS)).tolist() == [20, 20, 30, 30, 50, 80, 35, 60]
    assert np.round(1000 * np.array(ADJUSTMENTS)).tolist() == [0, 0, 0, 0, 0, -10, 10, 0]
    assert RISK_FREE_RATE == 0.03 and TE_BUDGET == 0.01
    for mandate, weights in BENCHMARKS.items():
        assert abs(sum(weights) - 1.0) < 1e-12 and weights[2] == 0.05
        assert weights[5:] == [0.05, 0.10, 0.05]  # the same alternatives in every mandate
    assert [weights[:5] for weights in BENCHMARKS.values()] == [
        [0.35, 0.20, 0.05, 0.15, 0.05], [0.20, 0.15, 0.05, 0.30, 0.10],
        [0.10, 0.05, 0.05, 0.45, 0.15]]

    # One loading matrix: rebuild both from the constants with explicit sums over the factors.
    loadings, n, m = np.array(LOADINGS), len(ASSETS), len(FACTORS)
    sigma_f = np.array([[FACTOR_VOLS[k] * FACTOR_VOLS[j] * FACTOR_CORR[k][j] for j in range(m)]
                        for k in range(m)])
    cma_ref = [RISK_FREE_RATE + sum(LOADINGS[i][k] * PREMIA[k] for k in range(m))
               + ADJUSTMENTS[i] for i in range(n)]
    covar_ref = np.einsum('ik,kl,jl->ij', loadings, sigma_f, loadings) + np.diag(
        np.square(RESIDUAL_VOLS))
    np.testing.assert_allclose(cma, cma_ref, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(covar, covar_ref, rtol=0.0, atol=1e-15)
    assert cma.index.tolist() == ASSETS and covar.index.tolist() == ASSETS
    assert covar.columns.tolist() == ASSETS
    # The factor block of the covariance has rank M = 4, and the factor-implied CMAs lie in its
    # column space: each is the asset's covariance with the factors times Sigma_F^-1 lambda.
    factor_block = (covar - np.diag(residual_var)).to_numpy()
    assert np.linalg.matrix_rank(factor_block, tol=1e-12) == m
    factor_part = (cma - RISK_FREE_RATE - adjustments).to_numpy()
    projection = np.linalg.lstsq(factor_block, factor_part, rcond=None)[0]
    assert np.abs(factor_block @ projection - factor_part).max() < 1e-12
    cross = loadings @ sigma_f  # covariance of each asset with each factor
    np.testing.assert_allclose(cross @ np.linalg.solve(sigma_f, PREMIA), factor_part,
                               rtol=0.0, atol=1e-15)
    # CMAs from another loading matrix fail the test: here government bonds gain credit risk.
    other = loadings.copy()
    other[0, 2] = 0.3
    other_part = other @ np.array(PREMIA)
    other_fit = np.linalg.lstsq(factor_block, other_part, rcond=None)[0]
    assert np.abs(factor_block @ other_fit - other_part).max() > 1e-4
    helper_cma, helper_covar = cma_and_covariance(model_inputs())
    pd.testing.assert_series_equal(helper_cma, cma)
    pd.testing.assert_frame_equal(helper_covar, covar)
    # The CMA table: factor-implied excess returns, CMAs from 4.0% to 9.1%, and volatilities.
    assert np.linalg.eigvalsh(covar_ref).min() > 0.0
    assert np.round(100.0 * factor_part, 2).tolist() == [1.00, 2.05, 3.50, 4.50, 6.10, 3.40,
                                                         1.85, 2.80]
    assert np.round(100.0 * np.array(cma_ref), 2).tolist() == [4.00, 5.05, 6.50, 7.50, 9.10,
                                                               5.40, 5.85, 5.80]
    vols = np.sqrt(np.diag(covar_ref))
    assert np.round(100.0 * vols, 1).tolist() == [6.3, 6.1, 10.0, 15.3, 20.5, 12.4, 6.8, 20.1]

    constraints = op.Constraints(is_long_only=True, tracking_err_vol_constraint=TE_BUDGET)
    saa = {}
    for mandate, weights in BENCHMARKS.items():
        benchmark = pd.Series(weights, index=ASSETS)
        saa[mandate], outcome = op.wrapper_maximise_alpha_over_tre(
            pd_covar=covar, alphas=cma, benchmark_weights=benchmark, constraints=constraints)
        if not (outcome.accepted and outcome.compliant):
            raise RuntimeError(f'{mandate}: {outcome.status}; {outcome.reason}')

    date = pd.Timestamp('2025-12-31')  # Synthetic decision date, not a data cutoff.
    risk_model = op.build_risk_model({date: covar})
    tracking_errors = {
        mandate: risk_model.compute_tre_at_date(
            benchmark_weights=pd.Series(BENCHMARKS[mandate], index=ASSETS),
            portfolio_weights=weights, date=date)
        for mandate, weights in saa.items()}

    # Every mandate: qis agrees with the explicit quadratic form and the 1% budget binds; the
    # weights match the active-set closed form, whose optimality conditions hold, and real
    # estate is the only asset at zero. Excess CMAs give the same weights as total CMAs.
    assert constraints.min_exposure == constraints.max_exposure == 1.0  # fully invested default
    closed = {}
    for mandate, weights in saa.items():
        assert abs(weights.sum() - 1.0) < 1e-8 and (weights >= -1e-8).all()
        benchmark = pd.Series(BENCHMARKS[mandate], index=ASSETS)
        explicit = explicit_tracking_error(weights, benchmark, covar_ref)
        assert abs(tracking_errors[mandate] - explicit) < 1e-12
        assert explicit <= TE_BUDGET + 1e-6 and abs(explicit - TE_BUDGET) < 1e-6
        assert round(100.0 * tracking_errors[mandate], 2) == 1.00
        closed[mandate] = certified(weights, cma, covar_ref, benchmark) - benchmark.to_numpy()
        np.testing.assert_allclose(weights - benchmark, closed[mandate], rtol=0.0, atol=5e-5)
        assert (weights <= HELD).tolist() == [asset == 'Real estate' for asset in ASSETS]
        excess, excess_outcome = solve_saa(covar, cma - RISK_FREE_RATE, benchmark)
        assert excess_outcome.accepted and excess_outcome.compliant
        np.testing.assert_allclose(excess, weights, rtol=0.0, atol=5e-5)
    # The same asset binds at zero with the same benchmark weight in every mandate, so the
    # closed-form active weights are identical, and the solver's agree with them.
    shown = closed[SHOWN_MANDATE]
    for active in closed.values():
        np.testing.assert_allclose(active, shown, rtol=0.0, atol=1e-12)
    # With no asset held at zero, the closed form ignores the benchmark altogether.
    unbounded = [active_set_solution(cma_ref, covar_ref, weights, TE_BUDGET, [])[0]
                 - np.array(weights) for weights in BENCHMARKS.values()]
    for active in unbounded:
        np.testing.assert_allclose(active, unbounded[0], rtol=0.0, atol=1e-12)
    assert np.round(100.0 * shown, 1).tolist() == [-6.3, 5.5, 1.6, -4.6, 7.9, -5.0, 3.4, -2.4]

    # The split of the expected active return: r_f 1'd + lambda'(beta'd) + adj'd, with 1'd = 0.
    assert abs(shown.sum()) < 1e-12
    active_return = float(np.array(cma_ref) @ shown)
    factor_return = float(np.array(PREMIA) @ (loadings.T @ shown))
    adjustment_return = float(np.array(ADJUSTMENTS) @ shown)
    assert abs(active_return - factor_return - adjustment_return) < 1e-15
    assert round(100.0 * active_return, 2) == 0.29 and round(active_return / TE_BUDGET, 2) == 0.29
    assert round(100.0 * adjustment_return, 2) == 0.08
    assert round(100.0 * adjustment_return / active_return) == 29
    for mandate, weights in saa.items():
        solver_active = weights - pd.Series(BENCHMARKS[mandate], index=ASSETS)
        assert abs(float(cma @ solver_active) - active_return) < 1e-6
    # Adjustments average 0.25% in absolute value, against 3.15% for the factor-implied parts.
    assert round(100.0 * np.mean(np.abs(ADJUSTMENTS)), 6) == 0.25
    assert round(100.0 * np.mean(factor_part), 6) == 3.15

    # Factor-implied CMAs alone: residual risk earns nothing, and hedge funds are sold out.
    benchmark = pd.Series(BENCHMARKS[SHOWN_MANDATE], index=ASSETS)
    factor_implied = RISK_FREE_RATE + beta @ premia
    factor_only, factor_outcome = solve_saa(covar, factor_implied, benchmark)
    assert factor_outcome.accepted and factor_outcome.compliant
    factor_active = certified(factor_only, factor_implied, covar_ref, benchmark) - benchmark
    np.testing.assert_allclose(factor_only - benchmark, factor_active, rtol=0.0, atol=5e-5)
    assert (factor_only <= HELD).tolist() == [asset == 'Hedge funds' for asset in ASSETS]
    assert abs(explicit_tracking_error(factor_only, benchmark, covar_ref) - TE_BUDGET) < 1e-6
    assert np.round(100.0 * factor_active, 1).tolist() == [-3.9, 6.5, 3.4, -1.9, 6.1, 0.3,
                                                           -10.0, -0.5]
    print(pd.DataFrame({'benchmark': benchmark, 'active_factor_implied_only': factor_active,
                        'active_with_adjustments': shown}).mul(100.0).round(2))

    # The pitfall: the rolling solver forward-fills CMA vintages onto the covariance dates and
    # sets a missing CMA to zero; the earlier vintage's value is not carried forward.
    quarter_ends = pd.date_range('2025-03-31', '2025-12-31', freq='QE')
    vintages = pd.DataFrame([cma, cma], index=pd.to_datetime(['2024-12-31', '2025-06-30']))
    vintages.loc['2025-06-30', MISSING_ASSET] = np.nan
    prices = pd.DataFrame(100.0, index=quarter_ends, columns=ASSETS)
    path = op.rolling_maximise_alpha_over_tre(
        prices=prices, alphas=vintages, constraints=constraints, benchmark_weights=benchmark,
        covar_dict={quarter_end: covar for quarter_end in quarter_ends})
    zero_filled = cma.copy()
    zero_filled[MISSING_ASSET] = 0.0
    assert cma.min() > 0.0  # zero is below every CMA
    zero_weights, _ = solve_saa(covar, zero_filled, benchmark)
    zero_active = certified(zero_weights, zero_filled, covar_ref, benchmark) - benchmark
    np.testing.assert_allclose(path.iloc[0], saa[SHOWN_MANDATE], rtol=0.0, atol=1e-6)
    for row in range(1, len(quarter_ends)):
        np.testing.assert_allclose(path.iloc[row], zero_weights, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(zero_weights - benchmark, zero_active, rtol=0.0, atol=5e-5)
    assert (zero_weights <= HELD).tolist() == [asset == MISSING_ASSET for asset in ASSETS]
    assert round(100.0 * zero_active[MISSING_ASSET], 1) == -10.0
    assert round(100.0 * zero_active['DM equity'], 1) == 11.9
    assert round(100.0 * shown[ASSETS.index(MISSING_ASSET)], 1) == 7.9
    # The single-date wrapper removes an asset with a NaN CMA, which also leaves it at zero.
    dropped, dropped_outcome = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=vintages.iloc[1], benchmark_weights=benchmark,
        constraints=constraints)
    assert dropped[MISSING_ASSET] == 0.0 and len(dropped_outcome.weights) == len(ASSETS) - 1
    # Filling the gap with the factor-implied CMA restores the allocation: EM equity has no
    # adjustment, so its factor-implied CMA is its CMA.
    filled = vintages.fillna(factor_implied)  # each column's gap takes its factor-implied CMA
    assert filled.loc['2025-06-30', MISSING_ASSET] == cma[MISSING_ASSET]
    repaired = op.rolling_maximise_alpha_over_tre(
        prices=prices, alphas=filled, constraints=constraints, benchmark_weights=benchmark,
        covar_dict={quarter_end: covar for quarter_end in quarter_ends})
    for row in range(len(quarter_ends)):
        np.testing.assert_allclose(repaired.iloc[row], saa[SHOWN_MANDATE], rtol=0.0, atol=1e-6)
    print('app_cma_strategic_allocation: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: the two parts of each CMA and the active weights they imply.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inputs = model_inputs()
    cma, covar = cma_and_covariance(inputs)
    factor_part = inputs['beta'] @ inputs['premia']
    benchmark = pd.Series(BENCHMARKS[SHOWN_MANDATE], index=ASSETS)
    full, full_outcome = solve_saa(covar, cma, benchmark)
    factor_only, factor_outcome = solve_saa(covar, RISK_FREE_RATE + factor_part, benchmark)
    table = pd.DataFrame({'factor_implied_excess': factor_part,
                          'adjustment': inputs['adjustments'], 'cma': cma,
                          'active_factor_implied_only': factor_only - benchmark,
                          'active_with_adjustments': full - benchmark})

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange, aqua = '#2a78d6', '#eb6834', '#1baf7a'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface,
                                      sharey=True)
    rows = np.arange(len(ASSETS))[::-1]
    height = 0.38
    left.barh(rows + height / 2, 100.0 * table['factor_implied_excess'], height, color=blue,
              label='Factor-implied part')
    left.barh(rows - height / 2, 100.0 * table['adjustment'], height, color=orange,
              label='Residual adjustment')
    left.set_yticks(rows, ASSETS)
    left.set_xlim(-1.5, 7.0)
    left.set_xlabel('expected excess return, % a year')
    left.set_title('Synthetic CMAs: factor part and adjustment', loc='left', color=ink)
    right.barh(rows + height / 2, 100.0 * table['active_factor_implied_only'], height,
               color=blue, label='Factor-implied CMAs only')
    right.barh(rows - height / 2, 100.0 * table['active_with_adjustments'], height, color=aqua,
               label='With adjustments')
    right.set_xticks([-10, -5, 0, 5])
    right.set_xlim(-11.5, 9.5)
    right.set_xlabel(f'active weight against the {SHOWN_MANDATE} benchmark, points')
    right.set_title(f'SAA within a {TE_BUDGET:.0%} tracking-error budget', loc='left', color=ink)
    # Room below the last asset keeps the one-row legends clear of the bars.
    for axis in (left, right):
        axis.set_ylim(-1.4, len(ASSETS) - 0.45)
        axis.vlines(0.0, -0.5, len(ASSETS) - 0.45, color=muted, linewidth=0.8)
        axis.legend(loc='lower left', ncol=2, fontsize=9, labelcolor=ink, facecolor=surface,
                    edgecolor='none', framealpha=1.0, handlelength=1.2, columnspacing=1.0)
        axis.set_facecolor(surface)
        axis.grid(axis='x', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis='y', length=0)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    rebuilt = RISK_FREE_RATE + np.array(LOADINGS) @ np.array(PREMIA) + np.array(ADJUSTMENTS)
    checks = {
        'cma_is_factor_part_plus_adjustment': bool(np.allclose(cma, rebuilt, rtol=0.0,
                                                               atol=1e-15)),
        'both_solves_accepted': bool(full_outcome.accepted and factor_outcome.accepted),
        'tracking_error_at_budget': bool(all(
            abs(explicit_tracking_error(w, benchmark, covar) - TE_BUDGET) < 1e-6
            for w in (full, factor_only))),
        'hedge_funds_sold_without_adjustments': bool(factor_only['Hedge funds'] <= HELD),
        'hedge_funds_overweight_with_adjustments': bool(full['Hedge funds']
                                                        > benchmark['Hedge funds']),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
