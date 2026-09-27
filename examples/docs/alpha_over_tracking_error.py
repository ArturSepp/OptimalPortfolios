"""Canonical script of docs/alpha_over_tracking_error.md.

The page shows excerpts of ``main``; every number and property it states is asserted here against
a reference computed a different way: the closed form of Proposition 1 and the yield-floor
solution of Proposition 2 by NumPy linear solves, tracking error as an explicit quadratic form,
and the soft yield-target solves as independent CVXPY problems written from raw arrays. The
script runs offline after ``pip install optimalportfolios`` and needs no data file or random
seed:

    python -m examples.docs.alpha_over_tracking_error

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
from contextlib import contextmanager
import inspect
import logging
import warnings

import cvxpy as cvx
import numpy as np
import pandas as pd
import qis

import optimalportfolios as op
from optimalportfolios.optimization.constraints import BenchmarkBetaConstraint

TICKERS = ['Govt bonds', 'Credit', 'US equity', 'Europe equity', 'EM equity', 'Gold']
# Annual volatilities and correlations of a stylised six-asset universe.
VOLS = [0.05, 0.07, 0.16, 0.17, 0.21, 0.15]
CORR = [
    [1.00, 0.50, -0.20, -0.20, -0.15, 0.10],
    [0.50, 1.00, 0.40, 0.40, 0.45, 0.15],
    [-0.20, 0.40, 1.00, 0.85, 0.75, 0.10],
    [-0.20, 0.40, 0.85, 1.00, 0.80, 0.15],
    [-0.15, 0.45, 0.75, 0.80, 1.00, 0.25],
    [0.10, 0.15, 0.10, 0.15, 0.25, 1.00],
]
BENCHMARK = [0.30, 0.15, 0.25, 0.10, 0.10, 0.10]  # the strategic allocation
# Annual expected active returns: 0.1 x volatility x a score in {-1, -0.5, 0, 0.5, 1}.
ALPHAS = [-0.0025, 0.0070, -0.0160, -0.0085, 0.0210, 0.0]
YIELDS = [0.035, 0.050, 0.015, 0.030, 0.025, 0.0]  # annual yields
TE_BUDGET = 0.02  # annual ex-ante tracking error
YIELD_TARGET = 0.035  # annual portfolio yield
EXHIBIT_BUDGETS = [round(0.0025 * k, 4) for k in range(1, 33)]  # 0.25% to 8%
WEIGHT_TOL = 1e-4  # solver weights against a closed form, in fractions of NAV
TE_TOL = 1e-6  # realised ex-ante tracking error against its budget


def covariance() -> pd.DataFrame:
    """Return the annual covariance from the volatilities and correlations."""
    vols = np.array(VOLS)
    return pd.DataFrame(np.outer(vols, vols) * np.array(CORR), index=TICKERS, columns=TICKERS)


def closed_form(covar: np.ndarray, alphas: np.ndarray, budget: float) -> tuple:
    """Return the active weights of Proposition 1 and the information ratio, by linear solves.

    Args:
        covar: Covariance matrix.
        alphas: Alpha vector.
        budget: Tracking-error budget in the square-root units of ``covar``.

    Returns:
        The active weights ``budget * inv(covar) @ alpha_tilde / IR`` and ``IR``.
    """
    ones = np.ones(len(alphas))
    inv_alpha = np.linalg.solve(covar, alphas)
    inv_ones = np.linalg.solve(covar, ones)
    eta = ones @ inv_alpha / (ones @ inv_ones)  # the alpha of the minimum-variance portfolio
    direction = inv_alpha - eta * inv_ones
    ratio = float(np.sqrt((alphas - eta) @ direction))
    return budget * direction / ratio, ratio


def yield_floor_closed_form(covar: np.ndarray, alphas: np.ndarray, yields: np.ndarray,
                            benchmark: np.ndarray, target: float, budget: float) -> tuple:
    """Return the active weights of Proposition 2 and the smallest budget that meets the floor.

    Args:
        covar: Covariance matrix.
        alphas: Alpha vector.
        yields: Asset yields in the units of ``target``.
        benchmark: Benchmark weights.
        target: Portfolio yield floor, assumed binding.
        budget: Tracking-error budget.

    Returns:
        The active weights ``d_y + s * inv(covar) @ alpha_bar / sqrt(alpha_bar' inv(covar)
        alpha_bar)`` and the tracking error of ``d_y``.
    """
    rows = np.column_stack([np.ones(len(alphas)), yields])
    gap = np.array([0.0, target - yields @ benchmark])
    inv_rows = np.linalg.solve(covar, rows)
    gram = rows.T @ inv_rows
    floor_active = inv_rows @ np.linalg.solve(gram, gap)
    projected = alphas - rows @ np.linalg.solve(gram, inv_rows.T @ alphas)
    inv_projected = np.linalg.solve(covar, projected)
    floor_te = float(np.sqrt(floor_active @ covar @ floor_active))
    spare = np.sqrt(budget ** 2 - floor_te ** 2)
    return floor_active + spare * inv_projected / np.sqrt(projected @ inv_projected), floor_te


def explicit_tracking_error(weights, benchmark, covar) -> float:
    """Return sqrt((w - b)' Sigma (w - b)) with NumPy, independently of the package and qis."""
    active = np.asarray(weights, dtype=float) - np.asarray(benchmark, dtype=float)
    return float(np.sqrt(active @ np.asarray(covar, dtype=float) @ active))


def information_ratios(covar: pd.DataFrame, benchmark: pd.Series, alphas: pd.Series,
                       budgets: list, long_only: bool) -> list:
    """Solve the forced problem at each budget; return alpha over realised tracking error.

    Args:
        covar: Covariance matrix.
        benchmark: Benchmark weights.
        alphas: Alpha vector.
        budgets: Tracking-error budgets.
        long_only: Whether the weights are bounded below by zero.

    Returns:
        One information ratio per budget. Each solve must be accepted and use its budget.
    """
    ratios = []
    for budget in budgets:
        weights, outcome = op.wrapper_maximise_alpha_over_tre(
            pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
            constraints=op.Constraints(is_long_only=long_only,
                                       tracking_err_vol_constraint=budget))
        assert outcome.accepted and outcome.compliant, outcome.reason
        tracking_error = explicit_tracking_error(weights, benchmark, covar)
        assert abs(tracking_error - budget) < TE_TOL
        ratios.append(float(alphas @ (weights - benchmark)) / tracking_error)
    return ratios


def soft_reference(covar: np.ndarray, alphas: np.ndarray, yields: np.ndarray,
                   benchmark: np.ndarray, holdings: np.ndarray, tre_weight: float,
                   turnover_weight) -> np.ndarray:
    """Solve the soft yield-target problem with CVXPY from raw arrays, without the package.

    Args:
        covar: Covariance matrix.
        alphas: Alpha vector.
        yields: Asset yields.
        benchmark: Benchmark weights.
        holdings: Current weights, the turnover baseline.
        tre_weight: Weight of the active variance.
        turnover_weight: Weight of the L1 turnover, or None for no turnover term.

    Returns:
        The long-only, fully invested weights that meet the yield floor.
    """
    w = cvx.Variable(len(alphas))
    active = w - benchmark
    objective = alphas @ active - tre_weight * cvx.quad_form(active, covar)
    if turnover_weight is not None:
        objective = objective - turnover_weight * cvx.norm1(w - holdings)
    problem = cvx.Problem(cvx.Maximize(objective),
                          [cvx.sum(w) == 1.0, w >= 0.0, yields @ w >= YIELD_TARGET])
    problem.solve(solver='CLARABEL')
    assert problem.status == 'optimal', problem.status
    return w.value


def least_turnover_to_floor(yields: np.ndarray, holdings: np.ndarray) -> np.ndarray:
    """Return the long-only, fully invested weights of least L1 turnover that meet the floor."""
    w = cvx.Variable(len(yields))
    problem = cvx.Problem(cvx.Minimize(cvx.norm1(w - holdings)),
                          [cvx.sum(w) == 1.0, w >= 0.0, yields @ w >= YIELD_TARGET])
    problem.solve(solver='CLARABEL')
    assert problem.status == 'optimal', problem.status
    return w.value


@contextmanager
def solver_warnings_silenced():
    """Silence the package's fallback warnings while a deliberately failing solve runs."""
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def assert_raises(error: type, function, match: str = '', **arguments) -> None:
    """Fail unless ``function(**arguments)`` raises ``error`` with ``match`` in its message."""
    try:
        function(**arguments)
    except error as raised:
        assert match in str(raised), str(raised)
        return
    raise AssertionError(f'expected {error.__name__}')


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    covar = covariance()
    benchmark = pd.Series(BENCHMARK, index=TICKERS)
    alphas = pd.Series(ALPHAS, index=TICKERS)
    constraints = op.Constraints(is_long_only=True, tracking_err_vol_constraint=TE_BUDGET)
    weights, outcome = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark, constraints=constraints)
    active = weights - benchmark
    reference, ir = closed_form(covar.to_numpy(), alphas.to_numpy(), TE_BUDGET)
    assert outcome.accepted and outcome.compliant
    assert np.abs(active.to_numpy() - reference).max() < WEIGHT_TOL
    print((100 * pd.DataFrame({'solver': active, 'closed form': reference})).round(1))

    # The inputs the page quotes, and the closed form's table in percentage points.
    sigma = covar.to_numpy()
    assert np.linalg.eigvalsh(sigma).min() > 0.0
    assert abs(benchmark.sum() - 1.0) < 1e-15
    assert np.allclose(ALPHAS, 0.1 * np.array(VOLS) * np.array([-0.5, 1, -1, -0.5, 1, 0]),
                       rtol=0.0, atol=1e-15)
    assert np.round(100 * reference, 1).tolist() == [-14.3, 23.1, -13.1, -5.1, 10.6, -1.2]
    assert abs(reference.sum()) < 1e-14
    assert abs(np.sqrt(reference @ sigma @ reference) - TE_BUDGET) < 1e-15
    # No long-only bound binds: the largest underweight, government bonds, is below its weight.
    assert reference.argmin() == TICKERS.index('Govt bonds')
    assert (benchmark.to_numpy() + reference > 0.0).all()
    # The budget multiplier eta is the alpha of the fully invested minimum-variance portfolio;
    # gold, with zero alpha, sits above it and is still underweighted through its correlations.
    minimum_variance = np.linalg.solve(sigma, np.ones(len(TICKERS)))
    minimum_variance /= minimum_variance.sum()
    eta = float(alphas @ minimum_variance)
    assert round(100 * eta, 2) == -0.53
    assert alphas['Gold'] - eta > 0.0 and reference[TICKERS.index('Gold')] < 0.0
    # The first-order conditions of the Lagrangian: alpha - eta 1 = 2 nu Sigma d with
    # nu = IR / (2 TE), the utility weight of the corollary.
    nu = ir / (2.0 * TE_BUDGET)
    assert np.allclose(alphas.to_numpy() - eta, 2.0 * nu * sigma @ reference, rtol=0.0,
                       atol=1e-15)
    assert round(ir, 3) == 0.336 and round(nu, 2) == 8.41
    # Only the direction of the alphas matters in the forced problem: a positive scale and a
    # common shift leave the weights unchanged.
    for transformed in (100.0 * alphas, alphas + 0.01):
        same, same_outcome = op.wrapper_maximise_alpha_over_tre(
            pd_covar=covar, alphas=transformed, benchmark_weights=benchmark,
            constraints=constraints)
        assert same_outcome.accepted and np.abs(same - weights).max() < WEIGHT_TOL
    # Proposition 1 does not involve the benchmark: another benchmark gives the same active
    # weights while no bound binds.
    other = pd.Series([0.25, 0.20, 0.25, 0.10, 0.10, 0.10], index=TICKERS)
    moved, moved_outcome = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=other, constraints=constraints)
    assert moved_outcome.accepted and np.abs(moved - other - reference).max() < WEIGHT_TOL
    # The benchmark argument replaces a benchmark stored on the constraints.
    stored = constraints.copy(benchmark_weights=other)
    replaced, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark, constraints=stored)
    assert np.abs(replaced - weights).max() < WEIGHT_TOL
    # Covariance units: a monthly covariance with the budget over sqrt(12) gives the same solve.
    monthly, monthly_outcome = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar / 12.0, alphas=alphas, benchmark_weights=benchmark,
        constraints=op.Constraints(tracking_err_vol_constraint=TE_BUDGET / np.sqrt(12.0)))
    assert monthly_outcome.accepted and np.abs(monthly - weights).max() < WEIGHT_TOL
    # The tracking-error row is the second-order cone of the eigen-factorised covariance.
    assert outcome.covar_factorization.n_eigenvalues_floored == 0
    factor = outcome.covar_factorization.factor
    assert abs(np.linalg.norm(factor.T @ active.to_numpy()) - TE_BUDGET) < TE_TOL
    # Without a tracking-error limit the forced problem is a linear program: all in EM equity.
    vertex, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
        constraints=op.Constraints(is_long_only=True))
    assert abs(vertex['EM equity'] - 1.0) < WEIGHT_TOL
    # The closed form trades a spread of 37 points between credit and government bonds.
    assert round(100 * (reference[1] - reference[0])) == 37

    date = pd.Timestamp('2024-12-31')  # a snapshot label for the risk model
    risk_model = op.build_risk_model({date: covar})
    tracking_error = risk_model.compute_tre_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date)
    active_return = float(alphas @ active)
    print(f'{tracking_error:.3%} {active_return:.3%} {active_return / tracking_error:.3f}')
    assert abs(tracking_error - TE_BUDGET) < TE_TOL

    # qis agrees with the explicit quadratic form; the expected active return is IR x budget.
    assert isinstance(risk_model, qis.RiskModel)
    assert abs(tracking_error - explicit_tracking_error(weights, benchmark, covar)) < 1e-12
    assert (f'{tracking_error:.3%} {active_return:.3%} {active_return / tracking_error:.3f}'
            == '2.000% 0.673% 0.336')
    assert abs(active_return - ir * TE_BUDGET) < 1e-6 and round(100 * ir * TE_BUDGET, 2) == 0.67
    # The tracking-error row does not control total risk: the tactical allocation is more
    # volatile than its benchmark.
    total_vols = [explicit_tracking_error(p, np.zeros(len(TICKERS)), covar)
                  for p in (benchmark, benchmark + reference)]
    assert np.round(100 * np.array(total_vols), 2).tolist() == [8.20, 8.64]
    # A hard volatility cap binds in the forced problem; the utility form has no such row.
    vol_capped, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
        constraints=constraints.copy(max_target_portfolio_vol_an=0.083))
    assert abs(explicit_tracking_error(vol_capped, np.zeros(len(TICKERS)), covar) - 0.083) < TE_TOL

    budgets = [0.01, 0.02, 0.03, 0.04, 0.06, 0.07]
    ratios = pd.DataFrame({
        'long-only': information_ratios(covar, benchmark, alphas, budgets, long_only=True),
        'no bounds': information_ratios(covar, benchmark, alphas, budgets, long_only=False)},
        index=budgets)
    print(ratios.round(3))

    # Without bounds the ratio is IR at every budget; long-only bounds start to bind at the
    # budget where the first negative active weight reaches its benchmark weight: US equity.
    assert np.allclose(ratios['no bounds'], ir, rtol=0.0, atol=1e-6)
    unit = reference / TE_BUDGET
    shorts = unit < 0.0
    binding = benchmark.to_numpy()[shorts] / -unit[shorts]
    first_bound = float(binding.min())
    assert TICKERS[int(np.flatnonzero(shorts)[binding.argmin()])] == 'US equity'
    assert round(100 * first_bound, 2) == 3.82
    assert np.allclose(ratios['long-only'][[0.01, 0.02, 0.03]], ir, rtol=0.0, atol=1e-6)
    assert (np.diff(ratios['long-only'][[0.04, 0.06, 0.07]]) < -0.01).all()
    assert (np.round(ratios['long-only'], 3).tolist()
            == [0.336, 0.336, 0.336, 0.335, 0.272, 0.247])
    # At 6%, the long-only solve holds no US equity.
    wide, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
        constraints=op.Constraints(is_long_only=True, tracking_err_vol_constraint=0.06))
    assert wide['US equity'] < 1e-6

    utility = op.Constraints(
        is_long_only=True, tre_utility_weight=ir / (2.0 * TE_BUDGET),
        constraint_enforcement_type=op.ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    utility_weights, utility_outcome = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark, constraints=utility)
    assert np.abs(utility_weights - weights).max() < WEIGHT_TOL

    assert utility_outcome.accepted and utility_outcome.compliant
    # Twice the weight halves the tracking error: the utility form scales as 1 / weight.
    halved, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
        constraints=utility.copy(tre_utility_weight=ir / TE_BUDGET))
    assert abs(explicit_tracking_error(halved, benchmark, covar) - TE_BUDGET / 2.0) < TE_TOL
    # Alphas in percent need a utility weight 100 times larger for the same solve.
    in_percent, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=100.0 * alphas, benchmark_weights=benchmark,
        constraints=utility.copy(tre_utility_weight=100.0 * ir / (2.0 * TE_BUDGET)))
    assert np.abs(in_percent - weights).max() < WEIGHT_TOL
    # The utility form keeps no tracking-error limit and no volatility cap as rows.
    for unenforced in (utility.copy(tracking_err_vol_constraint=0.001),
                       utility.copy(max_target_portfolio_vol_an=0.083)):
        unchanged, _ = op.wrapper_maximise_alpha_over_tre(
            pd_covar=covar, alphas=alphas, benchmark_weights=benchmark, constraints=unenforced)
        assert np.abs(unchanged - weights).max() < WEIGHT_TOL
    # Without the tracking-error term the utility solve is the linear program's vertex.
    no_penalty, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
        constraints=utility.copy(tre_utility_weight=None))
    assert abs(no_penalty['EM equity'] - 1.0) < WEIGHT_TOL
    # A group object replaces the total penalty: one group of all assets with the same weight
    # gives the same solve, whatever tre_utility_weight says.
    everything = op.GroupTrackingErrorConstraint(
        group_loadings=pd.DataFrame({'All': 1.0}, index=TICKERS),
        group_tre_utility_weights=pd.Series({'All': ir / (2.0 * TE_BUDGET)}))
    by_group, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
        constraints=utility.copy(tre_utility_weight=1000.0,
                                 group_tracking_error_constraint=everything))
    assert np.abs(by_group - weights).max() < WEIGHT_TOL
    # alphas=None is pure tracking in the utility form: the solve holds the benchmark.
    tracking, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=None, benchmark_weights=benchmark, constraints=utility)
    assert np.abs(tracking - benchmark).max() < WEIGHT_TOL
    # The forced form needs alphas: None fails when the objective is built.
    assert_raises(AttributeError, op.wrapper_maximise_alpha_over_tre, pd_covar=covar,
                  alphas=None, benchmark_weights=benchmark, constraints=constraints)
    # With current holdings at the benchmark, the default turnover penalty of 0.40 stops the
    # alpha-over-tracking-error utility solve from trading at all.
    assert utility.turnover_utility_weight == 0.40 and utility.tre_utility_weight != 1.0
    assert op.Constraints().tre_utility_weight == 1.0
    frozen, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark, constraints=utility,
        weights_0=benchmark)
    assert np.abs(frozen - benchmark).max() < WEIGHT_TOL
    # Group tracking error: a hard 1% cap on the equity block binds next to the 2% total cap,
    # and costs active return.
    equity = pd.DataFrame({'Equity': [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]}, index=TICKERS)
    grouped = constraints.copy(group_tracking_error_constraint=op.GroupTrackingErrorConstraint(
        group_loadings=equity, group_tre_vols=pd.Series({'Equity': 0.01})))
    group_weights, group_outcome = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark, constraints=grouped)
    group_active = (group_weights - benchmark).to_numpy()
    masked = group_active * equity['Equity'].to_numpy()
    assert group_outcome.accepted and group_outcome.compliant
    assert abs(np.sqrt(masked @ sigma @ masked) - 0.01) < TE_TOL
    assert abs(np.sqrt(group_active @ sigma @ group_active) - TE_BUDGET) < TE_TOL
    assert float(alphas @ group_active) < active_return - 1e-4

    yields = pd.Series(YIELDS, index=TICKERS)
    floored, floored_outcome = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
        constraints=constraints, benchmark_weights=benchmark)
    floored_active = floored - benchmark
    print(f'{yields @ benchmark:.3%} {yields @ weights:.2%} {yields @ floored:.2%}',
          f'{alphas @ floored_active:.2%}')

    # The benchmark and the unfloored solve yield less than the 3.5% floor, so it binds, and
    # Proposition 2 gives the solve, which spends the whole 2% budget.
    assert (f'{yields @ benchmark:.3%} {yields @ weights:.2%} {yields @ floored:.2%} '
            f'{alphas @ floored_active:.2%}') == '2.725% 3.29% 3.50% 0.64%'
    assert float(yields @ (benchmark + reference)) < YIELD_TARGET
    assert floored_outcome.accepted and floored_outcome.compliant
    floor_reference, floor_te = yield_floor_closed_form(
        sigma, alphas.to_numpy(), yields.to_numpy(), benchmark.to_numpy(), YIELD_TARGET,
        TE_BUDGET)
    assert np.abs(floored_active.to_numpy() - floor_reference).max() < WEIGHT_TOL
    assert (benchmark.to_numpy() + floor_reference > 0.0).all()
    assert abs(float(yields @ floored) - YIELD_TARGET) < 1e-6
    assert abs(explicit_tracking_error(floored, benchmark, covar) - TE_BUDGET) < TE_TOL
    # The floor costs 3 basis points of expected active return; reaching it takes 1.61%.
    assert round(1e4 * float(alphas.to_numpy() @ (reference - floor_reference))) == 3
    assert round(100 * floor_te, 2) == 1.61
    # Below that budget the hard path is infeasible and falls back to the current weights.
    with solver_warnings_silenced():
        _, short_outcome = op.wrapper_maximise_alpha_with_target_return(
            pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
            constraints=op.Constraints(tracking_err_vol_constraint=0.015),
            benchmark_weights=benchmark, weights_0=benchmark)
    assert not short_outcome.accepted and short_outcome.fallback_source == 'weights_0'
    # A floor that does not bind leaves the unfloored solve unchanged.
    loose, _ = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=0.03,
        constraints=constraints, benchmark_weights=benchmark)
    assert np.abs(loose - weights).max() < WEIGHT_TOL
    # Without a benchmark the hard path maximises alpha'w, a linear program whose vertex mixes
    # EM equity and credit so that 2.5% a + 5% (1 - a) = 3.5%; a tracking-error limit then
    # cannot be compiled.
    absolute, absolute_outcome = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
        constraints=op.Constraints())
    assert absolute_outcome.accepted
    np.testing.assert_allclose(absolute[['EM equity', 'Credit']], [0.6, 0.4], rtol=0.0,
                               atol=WEIGHT_TOL)
    assert_raises(ValueError, op.wrapper_maximise_alpha_with_target_return,
                  match='benchmark_weights must be given', pd_covar=covar, alphas=alphas,
                  yields=yields, target_return=YIELD_TARGET, constraints=constraints)
    # A missing yield counts as zero, with a warning.
    gapped = yields.where(yields.index != 'Credit')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        zero_yield, _ = op.wrapper_maximise_alpha_with_target_return(
            pd_covar=covar, alphas=alphas, yields=gapped, target_return=0.03,
            constraints=constraints, benchmark_weights=benchmark)
    assert any('NaN yields' in str(item.message) for item in caught)
    filled, _ = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=gapped.fillna(0.0), target_return=0.03,
        constraints=constraints, benchmark_weights=benchmark)
    assert np.abs(zero_yield - filled).max() < WEIGHT_TOL

    soft = op.Constraints(is_long_only=True, tre_utility_weight=ir / (2.0 * TE_BUDGET))
    penalised, penalised_outcome = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
        constraints=soft, benchmark_weights=benchmark, soft_tracking_error=True,
        weights_0=benchmark)
    unpenalised, _ = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
        constraints=soft.copy(turnover_utility_weight=None), benchmark_weights=benchmark,
        soft_tracking_error=True, weights_0=benchmark)
    for portfolio in (penalised, unpenalised):
        print(f'{(portfolio - benchmark).abs().sum():.0%} {alphas @ (portfolio - benchmark):.2%}')

    # Both solves agree with independent CVXPY problems written from the raw arrays; the
    # default penalty is 0.40 per unit of turnover.
    raw = dict(covar=sigma, alphas=alphas.to_numpy(), yields=yields.to_numpy(),
               benchmark=benchmark.to_numpy(), holdings=benchmark.to_numpy(),
               tre_weight=ir / (2.0 * TE_BUDGET))
    assert soft.turnover_utility_weight == 0.40 and penalised_outcome.accepted
    assert np.abs(penalised.to_numpy() - soft_reference(**raw, turnover_weight=0.40)).max() \
        < WEIGHT_TOL
    assert np.abs(unpenalised.to_numpy() - soft_reference(**raw, turnover_weight=None)).max() \
        < WEIGHT_TOL
    trades = [float((p - benchmark).abs().sum()) for p in (penalised, unpenalised)]
    earned = [float(alphas @ (p - benchmark)) for p in (penalised, unpenalised)]
    assert np.round(100 * np.array(trades)).tolist() == [36.0, 81.0]
    assert np.round(100 * np.array(earned), 2).tolist() == [0.25, 0.76]
    # The floor holds in both. The penalised solve is the trade of least turnover that adds the
    # 0.775% of yield the floor needs: into credit, out of US equity and gold.
    for portfolio in (penalised, unpenalised):
        assert abs(float(yields @ portfolio) - YIELD_TARGET) < 1e-6
    assert round(1e5 * (YIELD_TARGET - float(yields @ benchmark))) == 775
    least = least_turnover_to_floor(yields.to_numpy(), benchmark.to_numpy())
    assert np.abs(penalised.to_numpy() - least).max() < WEIGHT_TOL
    moves = (penalised - benchmark).round(4)
    assert moves[moves != 0.0].index.tolist() == ['Credit', 'US equity', 'Gold']
    assert moves['Credit'] > 0.0 and (moves[['US equity', 'Gold']] < 0.0).all()
    assert round(100 * explicit_tracking_error(unpenalised, benchmark, covar), 2) == 2.28
    # Without weights_0 the penalty has no baseline and is skipped.
    no_baseline, _ = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
        constraints=soft, benchmark_weights=benchmark, soft_tracking_error=True)
    assert np.abs(no_baseline - unpenalised).max() < WEIGHT_TOL
    # A hard turnover cap replaces the penalty, and the solve uses the whole cap.
    capped, capped_outcome = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
        constraints=soft.copy(turnover_constraint=0.50), benchmark_weights=benchmark,
        soft_tracking_error=True, weights_0=benchmark)
    assert capped_outcome.accepted and capped_outcome.compliant
    assert abs(float((capped - benchmark).abs().sum()) - 0.50) < 1e-5
    cap_reference = cvx.Variable(len(TICKERS))
    cap_active = cap_reference - benchmark.to_numpy()
    cvx.Problem(cvx.Maximize(alphas.to_numpy() @ cap_active - raw['tre_weight']
                             * cvx.quad_form(cap_active, sigma)),
                [cvx.sum(cap_reference) == 1.0, cap_reference >= 0.0,
                 yields.to_numpy() @ cap_reference >= YIELD_TARGET,
                 cvx.norm1(cap_active) <= 0.50]).solve(solver='CLARABEL')
    assert np.abs(capped.to_numpy() - cap_reference.value).max() < WEIGHT_TOL
    # The soft path ignores a populated tracking-error limit in the solve and the validation.
    ignored, ignored_outcome = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
        constraints=soft.copy(tracking_err_vol_constraint=0.001, turnover_utility_weight=None),
        benchmark_weights=benchmark, soft_tracking_error=True)
    assert ignored_outcome.accepted and np.abs(ignored - unpenalised).max() < WEIGHT_TOL
    # Without a benchmark the flag has no effect: the hard path runs.
    flagged, _ = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=YIELD_TARGET,
        constraints=op.Constraints(), soft_tracking_error=True)
    assert np.abs(flagged - absolute).max() < WEIGHT_TOL

    # In a rolling run the penalty applies from the second date on, when the previous target
    # becomes weights_0 (constant prices, so no drift).
    dates = pd.to_datetime(['2024-03-31', '2024-06-30', '2024-09-30'])
    prices = pd.DataFrame(100.0, index=dates, columns=TICKERS)
    covar_dict = {snapshot: covar for snapshot in dates}
    turned = [0.0070, -0.0025, 0.0210, -0.0085, -0.0160, 0.0]
    soft_rolling = op.rolling_maximise_alpha_with_target_return(
        prices=prices, alphas=pd.DataFrame([ALPHAS, turned], index=dates[:2], columns=TICKERS),
        yields=pd.DataFrame([YIELDS], index=dates[[0]], columns=TICKERS),
        target_returns=pd.Series(YIELD_TARGET, index=dates[[0]]), constraints=soft,
        covar_dict=covar_dict, benchmark_weights=benchmark, soft_tracking_error=True)
    np.testing.assert_allclose(soft_rolling.iloc[0], unpenalised, rtol=0.0, atol=WEIGHT_TOL)
    second, _ = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=pd.Series(turned, index=TICKERS), yields=yields,
        target_return=YIELD_TARGET, constraints=soft, benchmark_weights=benchmark,
        soft_tracking_error=True, weights_0=soft_rolling.iloc[0])
    np.testing.assert_allclose(soft_rolling.iloc[1], second, rtol=0.0, atol=WEIGHT_TOL)
    unanchored, _ = op.wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=pd.Series(turned, index=TICKERS), yields=yields,
        target_return=YIELD_TARGET, constraints=soft, benchmark_weights=benchmark,
        soft_tracking_error=True)
    assert np.abs(second - unanchored).max() > 0.05
    # Rolling yield targets forward-fill alphas, yields and targets and keep one outcome record
    # per date.
    rolling_yield = op.rolling_maximise_alpha_with_target_return(
        prices=prices, alphas=pd.DataFrame([ALPHAS], index=dates[[0]], columns=TICKERS),
        yields=pd.DataFrame([YIELDS], index=dates[[0]], columns=TICKERS),
        target_returns=pd.Series(YIELD_TARGET, index=dates[[0]]), constraints=constraints,
        covar_dict=covar_dict, benchmark_weights=benchmark)
    records = rolling_yield.attrs['optimization_outcomes']
    assert [record['accepted'] for record in records] == [True, True, True]
    np.testing.assert_allclose(rolling_yield, [floored] * 3, rtol=0.0, atol=WEIGHT_TOL)

    # Rolling alpha over tracking error: alphas forward-filled, then a missing alpha set to
    # zero; a Series benchmark used at every date; weights on the price columns without outcome
    # records; the ex-ante tracking error from qis at every date.
    signals = pd.DataFrame([ALPHAS, ALPHAS], index=dates[[0, 2]], columns=TICKERS)
    signals.loc[dates[2], 'Gold'] = np.nan
    rolling = op.rolling_maximise_alpha_over_tre(
        prices=prices, alphas=signals, constraints=constraints, benchmark_weights=benchmark,
        covar_dict=covar_dict)
    assert rolling.index.equals(dates) and rolling.columns.tolist() == TICKERS
    assert 'optimization_outcomes' not in rolling.attrs
    np.testing.assert_allclose(rolling.iloc[:2], [weights, weights], rtol=0.0, atol=WEIGHT_TOL)
    no_gold_view, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas.where(alphas.index != 'Gold', 0.0),
        benchmark_weights=benchmark, constraints=constraints)
    np.testing.assert_allclose(rolling.iloc[2], no_gold_view, rtol=0.0, atol=WEIGHT_TOL)
    history = op.build_risk_model(covar_dict).compute_tre_history(
        benchmark_weights=benchmark, portfolio_weights=rolling)
    assert np.allclose(history, TE_BUDGET, rtol=0.0, atol=TE_TOL)
    # The previous target becomes weights_0: a zero turnover cap holds it at the next date.
    pinned = op.rolling_maximise_alpha_over_tre(
        prices=prices, alphas=pd.DataFrame([ALPHAS, turned], index=dates[:2], columns=TICKERS),
        constraints=constraints.copy(turnover_constraint=0.0), benchmark_weights=benchmark,
        covar_dict=covar_dict)
    np.testing.assert_allclose(pinned, [weights] * 3, rtol=0.0, atol=WEIGHT_TOL)
    # Per-date beta loadings are required with a benchmark-beta constraint.
    assert_raises(ValueError, op.rolling_maximise_alpha_over_tre,
                  match='benchmark_beta_loadings must be given', prices=prices, alphas=signals,
                  constraints=constraints.copy(
                      benchmark_beta_constraint=BenchmarkBetaConstraint(beta_min=0.9)),
                  benchmark_weights=benchmark, covar_dict=covar_dict)
    # The single-date wrappers instead drop an asset whose alpha is missing.
    dropped, dropped_outcome = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas.where(alphas.index != 'Gold'),
        benchmark_weights=benchmark, constraints=constraints)
    assert dropped['Gold'] == 0.0 and len(dropped_outcome.weights) == len(TICKERS) - 1
    # A dated benchmark that starts after the first decision is filled with zeros there: the
    # tracking-error row then caps total volatility, which no fully invested portfolio meets,
    # and the fallback returns that zero benchmark. Later dates forward-fill the benchmark.
    late = pd.DataFrame([BENCHMARK], index=dates[[1]], columns=TICKERS)
    with solver_warnings_silenced():
        late_rolling = op.rolling_maximise_alpha_over_tre(
            prices=prices, alphas=signals, constraints=constraints, benchmark_weights=late,
            covar_dict=covar_dict)
    assert (late_rolling.iloc[0] == 0.0).all()
    np.testing.assert_allclose(late_rolling.iloc[1:], [weights, no_gold_view], rtol=0.0,
                               atol=WEIGHT_TOL)
    minimum_vol = np.sqrt(1.0 / (np.ones(len(TICKERS)) @ np.linalg.solve(sigma,
                                                                         np.ones(len(TICKERS)))))
    assert minimum_vol > TE_BUDGET
    # A benchmark column that is missing is a zero benchmark weight.
    no_gold = pd.DataFrame([BENCHMARK[:-1]], index=dates[[0]], columns=TICKERS[:-1])
    missing_column = op.rolling_maximise_alpha_over_tre(
        prices=prices, alphas=signals.iloc[:1], constraints=op.Constraints(
            is_long_only=True, tracking_err_vol_constraint=0.10),
        benchmark_weights=no_gold, covar_dict=covar_dict)
    zero_gold, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark.where(
            benchmark.index != 'Gold', 0.0),
        constraints=op.Constraints(is_long_only=True, tracking_err_vol_constraint=0.10))
    np.testing.assert_allclose(missing_column.iloc[0], zero_gold, rtol=0.0, atol=WEIGHT_TOL)

    # A rejected solve without current weights returns the benchmark; only the alpha-over-TE
    # wrapper reads the three diagnostic fields and accepts rebalancing indicators.
    with solver_warnings_silenced():
        fallback, rejected = op.wrapper_maximise_alpha_over_tre(
            pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
            constraints=constraints.copy(max_weights=pd.Series(0.1, index=TICKERS)))
    assert not rejected.accepted and rejected.fallback_source == 'benchmark_weights'
    assert np.abs(fallback - benchmark).max() == 0.0
    # A turnover cap from holdings far from the benchmark excludes every portfolio within the
    # budget: the solve is infeasible and returns the holdings.
    far = pd.Series([0.10, 0.10, 0.10, 0.10, 0.50, 0.10], index=TICKERS)
    assert explicit_tracking_error(far, benchmark, covar) > 2.0 * TE_BUDGET
    with solver_warnings_silenced():
        held, held_outcome = op.wrapper_maximise_alpha_over_tre(
            pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
            constraints=constraints.copy(turnover_constraint=0.05), weights_0=far)
    assert held_outcome.status == 'infeasible' and held_outcome.fallback_source == 'weights_0'
    assert np.abs(held - far).max() == 0.0
    over_tre = inspect.getsource(op.wrapper_maximise_alpha_over_tre)
    target_yield = inspect.getsource(op.wrapper_maximise_alpha_with_target_return)
    for field in ('validate_inputs', 'diagnose_infeasibility', 'max_constraint_relaxation',
                  'rebalancing_indicators'):
        assert field in over_tre and field not in target_yield
    assert 'rebalancing_indicators' in inspect.signature(
        op.rolling_maximise_alpha_over_tre).parameters
    assert inspect.signature(op.wrapper_maximise_alpha_over_tre).parameters[
        'optimiser_config'].default.apply_total_to_good_ratio is False
    assert inspect.signature(op.wrapper_maximise_alpha_with_target_return).parameters[
        'optimiser_config'].default.apply_total_to_good_ratio is True
    print('alpha_over_tracking_error: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: active weights against the closed form, and IR against budget.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    covar = covariance()
    benchmark = pd.Series(BENCHMARK, index=TICKERS)
    alphas = pd.Series(ALPHAS, index=TICKERS)
    rows = {}
    for budget in EXHIBIT_BUDGETS:
        row = {}
        for label, long_only in (('long_only', True), ('no_bounds', False)):
            weights, outcome = op.wrapper_maximise_alpha_over_tre(
                pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
                constraints=op.Constraints(is_long_only=long_only,
                                           tracking_err_vol_constraint=budget))
            active = weights - benchmark
            tracking_error = explicit_tracking_error(weights, benchmark, covar)
            row[f'accepted_{label}'] = bool(outcome.accepted and outcome.compliant)
            row[f'te_{label}'] = tracking_error
            row[f'ir_{label}'] = float(alphas @ active) / tracking_error
            if long_only:
                row.update({f'active_long_only: {t}': a for t, a in active.items()})
        reference, ir = closed_form(covar.to_numpy(), alphas.to_numpy(), budget)
        row.update({f'closed_form: {t}': a for t, a in zip(TICKERS, reference)})
        rows[budget] = row
    table = pd.DataFrame.from_dict(rows, orient='index')
    table.index.name = 'te_budget'
    shown = table.loc[TE_BUDGET]
    solver = np.array([shown[f'active_long_only: {t}'] for t in TICKERS])
    reference = np.array([shown[f'closed_form: {t}'] for t in TICKERS])
    unit = reference / TE_BUDGET
    shorts = unit < 0.0
    first_bound = float((benchmark.to_numpy()[shorts] / -unit[shorts]).min())

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange = '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface,
                                      gridspec_kw={'width_ratios': [1.0, 1.1]})
    positions = np.arange(len(TICKERS))[::-1]
    left.barh(positions, 100.0 * solver, height=0.6, color=blue,
              label='package solve, long-only')
    left.scatter(100.0 * reference, positions, marker='D', s=30, color=ink, zorder=3,
                 label='closed form')
    # The zero line stops above the legend, which sits below the last asset.
    left.vlines(0.0, -0.5, len(TICKERS) - 0.4, color=muted, linewidth=0.8)
    left.set_yticks(positions, TICKERS)
    left.set_xlim(-20.0, 30.0)
    left.set_ylim(-1.6, len(TICKERS) - 0.4)
    left.set_xlabel('active weight at a 2% budget, percentage points')
    left.set_title('Active weights follow the closed form', loc='left', color=ink)
    left.legend(loc='lower right', fontsize=9, labelcolor=ink, facecolor=surface,
                edgecolor='none', framealpha=1.0)
    left.grid(axis='x', color=grid, linewidth=0.8)
    left.tick_params(axis='y', length=0)

    budgets = 100.0 * table.index.to_numpy()
    # The two lines coincide until the first bound binds; the dashes keep both visible.
    right.plot(budgets, table['ir_long_only'], color=blue, linewidth=2.4)
    right.plot(budgets, table['ir_no_bounds'], color=orange, linewidth=2.0, linestyle='--')
    right.axvline(100.0 * first_bound, color=muted, linestyle=':', linewidth=1.2)
    right.text(budgets[-1], table['ir_no_bounds'].iloc[-1] + 0.006, 'no weight bounds',
               ha='right', va='bottom', color=ink, fontsize=10)
    right.text(budgets[-1], table['ir_long_only'].iloc[-1] - 0.007, 'long-only',
               ha='right', va='top', color=ink, fontsize=10)
    right.text(100.0 * first_bound - 0.15, 0.27,
               f'US equity reaches\nzero at a {100.0 * first_bound:.2f}%\nbudget', ha='right',
               va='center', color=ink, fontsize=9)
    right.set_xlim(0.0, 8.2)
    right.set_ylim(0.20, 0.36)
    right.set_xlabel('tracking-error budget, % a year')
    right.set_ylabel('information ratio')
    right.set_title('The ratio is flat until a bound binds', loc='left', color=ink)
    right.grid(axis='y', color=grid, linewidth=0.8)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    ir = closed_form(covar.to_numpy(), alphas.to_numpy(), TE_BUDGET)[1]
    below = table.index.to_numpy() < first_bound
    checks = {
        'every_solve_accepted': bool(table['accepted_long_only'].all()
                                     and table['accepted_no_bounds'].all()),
        'te_equals_budget': bool(np.allclose(table['te_long_only'], table.index, atol=TE_TOL)
                                 and np.allclose(table['te_no_bounds'], table.index,
                                                 atol=TE_TOL)),
        'closed_form_at_shown_budget': bool(np.abs(solver - reference).max() < WEIGHT_TOL),
        'ir_flat_without_bounds': bool(np.allclose(table['ir_no_bounds'], ir, atol=1e-6)),
        'ir_flat_until_first_bound': bool(np.allclose(table['ir_long_only'][below], ir,
                                                      atol=1e-6)),
        'ir_falls_after_first_bound': bool(
            (np.diff(table['ir_long_only'][~below].to_numpy()) < 0.0).all()),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
