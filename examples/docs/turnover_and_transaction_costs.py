"""Canonical script of docs/turnover_and_transaction_costs.md.

The page's four Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: a one-dimensional optimality argument for the turnover budget, an exact
enumeration of the first-order conditions of the turnover-penalty problem with explicit NumPy
turnover and tracking error, and an exact rational currency ledger of the backtest. The script
runs offline after ``pip install optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.turnover_and_transaction_costs

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
from contextlib import contextmanager
from fractions import Fraction
import inspect
import itertools
import logging
import warnings

import numpy as np
import pandas as pd

PENALTY_ASSETS = ['Equity', 'Credit', 'Govt bonds', 'Gold']
# Annual volatilities and correlations of a stylised four-asset universe.
PENALTY_VOLS = [0.16, 0.08, 0.05, 0.15]
PENALTY_CORR = [
    [1.0, 0.6, -0.2, 0.1],
    [0.6, 1.0, 0.3, 0.1],
    [-0.2, 0.3, 1.0, 0.2],
    [0.1, 0.1, 0.2, 1.0],
]
PENALTY_BENCHMARK = [0.50, 0.20, 0.25, 0.05]
PENALTY_HOLDINGS = [0.40, 0.25, 0.20, 0.15]
TRACKING_ERROR_WEIGHT = 100.0  # tre_utility_weight, fixed along the path
PENALTY_WEIGHT_MAX = 0.50      # turnover_utility_weight runs from zero to this value
PENALTY_WEIGHT_STEPS = 50      # in equal steps of 0.01
LABELLED_PENALTY_WEIGHTS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]


def penalty_covariance() -> pd.DataFrame:
    """Return the penalty example's annual covariance from its volatilities and correlations."""
    vols = np.array(PENALTY_VOLS)
    return pd.DataFrame(np.outer(vols, vols) * np.array(PENALTY_CORR), index=PENALTY_ASSETS,
                        columns=PENALTY_ASSETS)


def penalty_weights() -> np.ndarray:
    """Return the grid of turnover penalty weights, from zero in equal steps."""
    return np.round(np.linspace(0.0, PENALTY_WEIGHT_MAX, PENALTY_WEIGHT_STEPS + 1), 10)


def penalty_constraints(**weights):
    """Return long-only utility-mode constraints, passing any penalty weights through."""
    import optimalportfolios as opt

    return opt.Constraints(
        is_long_only=True,
        constraint_enforcement_type=opt.ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        **weights)


def solve_penalty(constraints, weights_0=None) -> tuple:
    """Solve the no-alpha utility problem of the four-asset example for one constraint set."""
    import optimalportfolios as opt

    return opt.wrapper_maximise_alpha_over_tre(
        pd_covar=penalty_covariance(), alphas=None,
        benchmark_weights=pd.Series(PENALTY_BENCHMARK, index=PENALTY_ASSETS),
        constraints=constraints, weights_0=weights_0)


def solve_penalty_path() -> tuple:
    """Solve the example at every penalty weight of the grid, from the current holdings.

    Returns:
        Weights with one row per penalty weight, and the solver outcome of each solve.
    """
    holdings = pd.Series(PENALTY_HOLDINGS, index=PENALTY_ASSETS)
    rows, outcomes = {}, []
    for weight in penalty_weights():
        constraints = penalty_constraints(tre_utility_weight=TRACKING_ERROR_WEIGHT,
                                          turnover_utility_weight=float(weight))
        weights, outcome = solve_penalty(constraints, weights_0=holdings)
        rows[float(weight)] = weights
        outcomes.append(outcome)
    return pd.DataFrame(rows).T, outcomes


def explicit_tracking_error(weights, benchmark, covar) -> float:
    """Return sqrt((w - b)' Sigma (w - b)) with NumPy, independently of the package."""
    active = np.asarray(weights, dtype=float) - np.asarray(benchmark, dtype=float)
    return float(np.sqrt(active @ np.asarray(covar, dtype=float) @ active))


def explicit_turnover(weights, baseline) -> float:
    """Return the full L1 change sum(|w - w0|) with NumPy, with no half factor."""
    change = np.asarray(weights, dtype=float) - np.asarray(baseline, dtype=float)
    return float(np.abs(change).sum())


def exact_l1(target, baseline, multipliers=None) -> Fraction:
    """Return sum(|a_i (w_i - w0_i)|) in exact rational arithmetic from percentage points."""
    multipliers = multipliers or [1] * len(target)
    return sum(abs(Fraction(a) * Fraction(w - w0, 100))
               for a, w, w0 in zip(multipliers, target, baseline))


def no_trade_threshold(tracking_error_weight: float = TRACKING_ERROR_WEIGHT) -> float:
    """Return the smallest penalty weight at which the current holdings are optimal.

    With no alpha, unit multipliers and interior holdings, w0 is optimal when some budget
    multiplier mu gives |2 lambda_TE (Sigma d0)_i + mu| <= lambda_TO for every asset, where
    d0 = w0 - b; the best mu centres the vector, so the bound is lambda_TE times its spread.
    """
    marginal = penalty_covariance().to_numpy() @ (np.array(PENALTY_HOLDINGS)
                                                   - np.array(PENALTY_BENCHMARK))
    return float(tracking_error_weight * (marginal.max() - marginal.min()))


def kkt_reference(turnover_weight: float) -> np.ndarray:
    """Solve the no-alpha penalty problem exactly by enumerating its first-order conditions.

    The problem is min lambda_TE (w - b)' Sigma (w - b) + lambda_TO sum|w - w0| subject to
    sum(w) = 1. Each asset buys, sells or holds; for one such pattern the stationarity rows of
    the trading assets, the holds and the budget are linear in (w, mu). The unique optimum is
    the pattern whose solution has the assumed signs and whose held assets satisfy the
    subgradient bound. No solver is used, and the result must be strictly positive, so the
    long-only bound is slack.

    Args:
        turnover_weight: The penalty weight lambda_TO.

    Returns:
        The optimal weights.
    """
    covar = penalty_covariance().to_numpy()
    benchmark = np.array(PENALTY_BENCHMARK)
    holdings = np.array(PENALTY_HOLDINGS)
    n, scale, tol = len(benchmark), 2.0 * TRACKING_ERROR_WEIGHT, 1e-12
    for pattern in itertools.product((-1.0, 0.0, 1.0), repeat=n):
        signs = np.array(pattern)
        if not signs.any():
            # Nothing trades: w = w0, and the best multiplier centres the marginal vector.
            marginal = scale * covar @ (holdings - benchmark)
            if turnover_weight >= (marginal.max() - marginal.min()) / 2.0 - tol:
                return holdings.copy()
            continue
        system, rhs = np.zeros((n + 1, n + 1)), np.zeros(n + 1)
        for i in range(n):
            if signs[i]:
                system[i, :n], system[i, n] = scale * covar[i], 1.0
                rhs[i] = scale * covar[i] @ benchmark - turnover_weight * signs[i]
            else:
                system[i, i], rhs[i] = 1.0, holdings[i]
        system[n, :n], rhs[n] = 1.0, 1.0
        solution = np.linalg.solve(system, rhs)
        weights, mu = solution[:n], solution[n]
        gradient = scale * covar @ (weights - benchmark) + mu
        trading = signs != 0.0
        if ((signs * (weights - holdings))[trading] >= -tol).all() and (
                np.abs(gradient[~trading]) <= turnover_weight + tol).all():
            assert (weights > 0.0).all()
            return weights
    raise AssertionError('no sign pattern satisfies the first-order conditions')


def currency_ledger() -> dict:
    """Build an exact two-trade currency ledger without calling a backtest or cost helper.

    Returns:
        Units after each trade, the first trade's per-asset costs, the pre-cost NAV before the
        second trade and its notional, and the cash costs and post-cost NAV on each price date.
    """
    first_units = [Fraction(60, 102), Fraction(40, 99)]
    first_costs = [units * price / 1000 for units, price in zip(first_units, (102, 99))]
    pre_nav = sum(units * 101 for units in first_units) - sum(first_costs)
    new_units = pre_nav / (2 * 101)
    second_trade = sum(abs(new_units - units) * 101 for units in first_units)
    second_cost = second_trade / 1000
    return {
        'first_units': np.array(first_units, dtype=float),
        'first_costs': np.array(first_costs, dtype=float),
        'new_units': float(new_units), 'pre_nav': float(pre_nav),
        'second_trade': float(second_trade),
        'costs': np.array([0, sum(first_costs), second_cost], dtype=float),
        'nav': np.array([100, 100 - sum(first_costs), pre_nav - second_cost], dtype=float),
    }


@contextmanager
def solver_warnings_silenced():
    """Silence the package's fallback warnings while a deliberately failing solve runs."""
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def assert_raises(error: type, match: str, function, *args, **kwargs) -> None:
    """Fail unless ``function(*args, **kwargs)`` raises ``error`` with ``match`` in its message."""
    try:
        function(*args, **kwargs)
    except error as exc:
        assert match in str(exc), str(exc)
        return
    raise AssertionError(f'expected {error.__name__}')


def recorded_warnings(function, *args, **kwargs) -> tuple:
    """Call ``function`` and return its result with the warnings it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = function(*args, **kwargs)
    return result, list(caught)


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    # Full L1 arithmetic of the methodology, exactly: a 5-point sale and a 5-point purchase use
    # 0.10; 60/40 to 50/50 uses 0.20 although the net change is zero; 60/40 to 55/45 uses 0.10
    # with unit multipliers and 0.15 with multipliers [2, 1].
    assert exact_l1([55, 45], [60, 40]) == Fraction(1, 10)
    assert exact_l1([50, 50], [60, 40]) == Fraction(2, 10)
    assert Fraction(50 - 60, 100) + Fraction(50 - 40, 100) == 0
    assert exact_l1([55, 45], [60, 40], multipliers=[2, 1]) == Fraction(15, 100)

    import pandas as pd
    import optimalportfolios as opt

    current = pd.Series({"A": 0.60, "B": 0.40})
    constraints = opt.Constraints(
        is_long_only=True,
        weights_0=current,
        turnover_constraint=0.10,
        turnover_costs=pd.Series({"A": 1.0, "B": 1.0}),
    )

    # Every other field keeps its default: forced constraints, and utility weights of 1.0 and
    # 0.40 that a forced solve does not read.
    forced = opt.ConstraintEnforcementType.FORCED_CONSTRAINTS
    assert constraints.constraint_enforcement_type == forced
    assert constraints.tre_utility_weight == 1.0 and constraints.turnover_utility_weight == 0.40

    covar = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]], index=current.index, columns=current.index
    )
    optimal_weights, outcome = opt.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=constraints,
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
    )
    if not (outcome.accepted and outcome.compliant and outcome.fallback_source is None):
        raise RuntimeError(f"Unusable solve: {outcome.status}")

    # Independent certificate: with w_B = 1 - w_A the budget 2|w_A - 0.60| <= 0.10 means
    # w_A >= 0.55, and the derivative 0.08 w_A - 0.02 (1 - w_A) of 0.04 w_A^2 + 0.01 w_B^2 is
    # positive on [0.55, 0.65], so the lower end is optimal; unconstrained, it is zero at 0.20.
    assert 0.10 * 0.55 - 0.02 > 0.0 and abs(0.10 * 0.20 - 0.02) < 1e-15
    assert outcome.covar_factorization.n_eigenvalues_floored == 0
    np.testing.assert_allclose(optimal_weights, [0.55, 0.45], atol=2e-7, rtol=0.0)
    assert abs(explicit_turnover(optimal_weights, current) - 0.10) < 4e-7
    # The page's table: baseline, constrained target and absolute change at two decimals.
    assert optimal_weights.index.tolist() == ["A", "B"]
    displayed = np.array([[0.60, 0.55, 0.05], [0.40, 0.45, 0.05]])
    np.testing.assert_allclose(
        displayed, np.c_[current, optimal_weights, (optimal_weights - current).abs()],
        atol=2e-7, rtol=0.0)
    # The forced solve ignores the default penalty weights: None gives the same target.
    unpenalised, _ = opt.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=constraints.copy(turnover_utility_weight=None,
                                                     tre_utility_weight=None))
    np.testing.assert_allclose(unpenalised, optimal_weights, atol=1e-9, rtol=0.0)
    # Without the budget the minimum-variance allocation is 20/80.
    free, free_outcome = opt.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=opt.Constraints(is_long_only=True))
    assert free_outcome.accepted and free_outcome.compliant
    np.testing.assert_allclose(free, [0.20, 0.80], atol=2e-6, rtol=0.0)
    # Multipliers [2, 1] turn the same 0.10 budget into 3|x| <= 0.10: a transfer of 0.10/3.
    weighted = constraints.copy(turnover_costs=pd.Series({"A": 2.0, "B": 1.0}))
    weighted_weights, weighted_outcome = opt.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=weighted)
    assert weighted_outcome.accepted and weighted_outcome.compliant
    np.testing.assert_allclose(weighted_weights, [0.60 - 0.10 / 3, 0.40 + 0.10 / 3], atol=3e-7)
    assert (weighted.turnover_costs * (weighted_weights - current).abs()).sum() <= 0.100001
    # No baseline: the budget row is skipped (20/80), while explicit zero holdings make a 0.10
    # budget infeasible for a fully invested book; zeros are not inferred.
    absent = constraints.copy(weights_0=None)
    skipped, skipped_outcome = opt.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=absent)
    assert skipped_outcome.accepted and skipped_outcome.compliant
    np.testing.assert_allclose(skipped, [0.20, 0.80], atol=2e-6)
    with solver_warnings_silenced():
        _, rejected = opt.wrapper_quadratic_optimisation(
            pd_covar=covar, constraints=absent, weights_0=current * 0.0)
    assert not rejected.accepted
    # A wrapper argument overrides the stored baseline: from 50/50 the optimum is 45/55.
    overridden, overridden_outcome = opt.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=constraints, weights_0=pd.Series({"A": 0.50, "B": 0.50}))
    assert overridden_outcome.accepted and overridden_outcome.compliant
    np.testing.assert_allclose(overridden, [0.45, 0.55], atol=3e-7)

    # The turnover penalty on the four-asset example: its quoted inputs first.
    penalty_covar = penalty_covariance()
    benchmark = np.array(PENALTY_BENCHMARK)
    holdings = np.array(PENALTY_HOLDINGS)
    assert np.round(100 * np.sqrt(np.diag(penalty_covar))).tolist() == [16, 8, 5, 15]
    assert np.round(100 * benchmark).tolist() == [50, 20, 25, 5]
    assert np.round(100 * holdings).tolist() == [40, 25, 20, 15]
    assert abs(benchmark.sum() - 1.0) < 1e-15 and abs(holdings.sum() - 1.0) < 1e-15
    assert np.linalg.eigvalsh(penalty_covar).min() > 0.0
    # One no-alpha utility solve per weight, from 0 to 0.5 in steps of 0.01; turnover and
    # tracking error computed explicitly.
    path, outcomes = solve_penalty_path()
    assert path.index[0] == 0.0 and path.index[-1] == 0.5
    np.testing.assert_allclose(np.diff(path.index), 0.01, rtol=0.0, atol=1e-12)
    assert all(o.accepted and o.compliant and o.fallback_source is None for o in outcomes)
    turnover = np.array([explicit_turnover(row, holdings) for row in path.to_numpy()])
    tracking = np.array([explicit_tracking_error(row, benchmark, penalty_covar)
                         for row in path.to_numpy()])
    # compute_tre_turnover_stats reports the same full L1 turnover and tracking error.
    for row, expected_turnover, expected_tracking in zip(path.to_numpy(), turnover, tracking):
        te_vol, stats_turnover, alpha, _, _ = opt.compute_tre_turnover_stats(
            covar=penalty_covar.to_numpy(),
            benchmark_weights=pd.Series(benchmark, index=PENALTY_ASSETS),
            weights=pd.Series(row, index=PENALTY_ASSETS),
            weights_0=pd.Series(holdings, index=PENALTY_ASSETS))
        assert abs(te_vol - expected_tracking) < 1e-15 and alpha == 0.0
        assert abs(stats_turnover - expected_turnover) < 1e-15
    # A label missing from weights_0 gives a NaN change, which the turnover sum drops.
    _, partial, _, _, _ = opt.compute_tre_turnover_stats(
        covar=penalty_covar.to_numpy(),
        benchmark_weights=pd.Series(benchmark, index=PENALTY_ASSETS),
        weights=pd.Series(benchmark, index=PENALTY_ASSETS),
        weights_0=pd.Series(holdings, index=PENALTY_ASSETS).drop("Gold"))
    assert abs(partial - explicit_turnover(benchmark[:3], holdings[:3])) < 1e-15
    assert abs(partial - 0.20) < 1e-15
    # Every solve matches the exact first-order solution to within the solver's tolerance.
    np.testing.assert_allclose(path, [kkt_reference(weight) for weight in path.index],
                               rtol=0.0, atol=5e-6)
    # As the weight grows, turnover never rises and tracking error never falls.
    assert (np.diff(turnover) <= 1e-6).all() and (np.diff(tracking) >= -1e-7).all()
    # At zero the solve holds the benchmark and trades the whole 0.30 gap.
    assert tracking[0] < 1e-6 and abs(turnover[0] - 0.30) < 1e-6
    assert abs(explicit_turnover(benchmark, holdings) - 0.30) < 1e-15
    # Credit stops trading at about 0.014, government bonds at about 0.039 (exact solutions).
    for asset, before, after in ((1, 0.0140, 0.0141), (2, 0.0391, 0.0392)):
        assert abs(kkt_reference(before)[asset] - holdings[asset]) > 1e-6
        assert abs(kkt_reference(after)[asset] - holdings[asset]) < 1e-12
    assert np.abs(path.loc[0.05].to_numpy()[1:3] - holdings[1:3]).max() < 1e-6
    # At 0.20: turnover 0.0855 and tracking error 1.02%. The figure's description quotes
    # turnover of 17.8%, 13.2% and 8.5% at 0.02, 0.1 and 0.2.
    assert round(turnover[path.index.get_loc(0.20)], 4) == 0.0855
    assert round(100 * tracking[path.index.get_loc(0.20)], 2) == 1.02
    assert [round(100 * turnover[path.index.get_loc(weight)], 1)
            for weight in (0.02, 0.1, 0.2)] == [17.8, 13.2, 8.5]
    # The no-trade threshold: 100 x 0.003851 = 0.3851, with 1.88% tracking error beyond it.
    marginal = penalty_covar.to_numpy() @ (holdings - benchmark)
    assert round(marginal.max() - marginal.min(), 12) == 0.003851
    # The spread is set by gold and equity; credit and government bonds sit inside it, with the
    # two smallest marginal active risks.
    assert PENALTY_ASSETS[marginal.argmax()] == "Gold"
    assert PENALTY_ASSETS[marginal.argmin()] == "Equity"
    assert {PENALTY_ASSETS[i] for i in np.argsort(np.abs(marginal))[:2]} == {"Credit",
                                                                            "Govt bonds"}
    threshold = no_trade_threshold()
    assert round(threshold, 10) == 0.3851
    above = path.index >= threshold
    assert above.any() and (turnover[above] < 1e-6).all() and (turnover[~above] > 1e-3).all()
    assert round(100 * explicit_tracking_error(holdings, benchmark, penalty_covar), 2) == 1.88
    np.testing.assert_allclose(tracking[above], explicit_tracking_error(
        holdings, benchmark, penalty_covar), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(kkt_reference(threshold), holdings, rtol=0.0, atol=1e-12)
    assert abs(kkt_reference(threshold - 1e-4)[0] - holdings[0]) > 1e-6
    # Without weights_0 the penalty is skipped and the solve returns the benchmark.
    unanchored, unanchored_outcome = solve_penalty(penalty_constraints(
        tre_utility_weight=TRACKING_ERROR_WEIGHT, turnover_utility_weight=PENALTY_WEIGHT_MAX))
    assert unanchored_outcome.accepted
    np.testing.assert_allclose(unanchored, benchmark, rtol=0.0, atol=1e-6)
    # Pitfall: with the default weights 1.0 and 0.40 the threshold is 0.003851, which 0.40
    # exceeds more than 100 times, so the utility solve keeps the holdings unchanged.
    defaults = penalty_constraints()
    assert defaults.tre_utility_weight == 1.0 and defaults.turnover_utility_weight == 0.40
    assert round(no_trade_threshold(1.0), 12) == 0.003851
    assert 0.40 / no_trade_threshold(1.0) > 100.0
    frozen, frozen_outcome = solve_penalty(defaults, weights_0=pd.Series(holdings,
                                                                         index=PENALTY_ASSETS))
    assert frozen_outcome.accepted and frozen_outcome.compliant
    np.testing.assert_allclose(frozen, holdings, rtol=0.0, atol=1e-7)

    import pandas as pd
    import qis

    prices = pd.DataFrame(
        {"A": [100.0, 102.0, 101.0], "B": [100.0, 99.0, 101.0]},
        index=pd.date_range("2024-01-02", periods=3, freq="B"),
    )
    targets = pd.DataFrame(
        {"A": [0.60, 0.50], "B": [0.40, 0.50]},
        index=prices.index[:2],
    )
    portfolio = qis.backtest_model_portfolio(
        prices=prices,
        weights=targets,
        rebalancing_costs=0.0010,
        weight_implementation_lag=1,
        ticker="Cost-aware backtest",
    )

    # The targets change by 0.20 in full L1 terms, twice the 0.10 budget above.
    assert exact_l1([50, 50], [60, 40]) == 2 * exact_l1([55, 45], [60, 40])
    assert prices.index.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03",
                                                          "2024-01-04"]
    # The exact rational ledger: the 2 January target enters on 3 January at 102 and 99,
    # buying 60/102 and 40/99 units that cost 0.06 and 0.04; the 3 January target trades on
    # 4 January, sized from the pre-cost NAV of about 100.119846.
    ledger = currency_ledger()
    np.testing.assert_allclose(ledger['first_costs'], [0.06, 0.04], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(portfolio.realized_costs.iloc[1], ledger['first_costs'],
                               rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(portfolio.units.iloc[0], 0.0, atol=0.0)
    np.testing.assert_allclose(portfolio.units.iloc[1], ledger['first_units'], atol=1e-12)
    np.testing.assert_allclose(portfolio.units.iloc[2], ledger['new_units'], atol=1e-12)
    np.testing.assert_allclose(portfolio.nav, ledger['nav'], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(portfolio.realized_costs.sum(axis=1), ledger['costs'],
                               rtol=0.0, atol=1e-12)
    assert portfolio.is_rebalancing.tolist() == [False, True, True]
    assert round(ledger['pre_nav'], 6) == 100.119846
    # The page's trade table at six decimals.
    displayed = np.array([[100.000000, 0.100000, 99.900000],
                          [18.603684, 0.018604, 100.101242]])
    np.testing.assert_allclose(
        displayed, np.c_[[100.0, ledger['second_trade']], ledger['costs'][1:],
                         ledger['nav'][1:]], rtol=0.0, atol=0.5e-6)
    # The second trade is not 0.20 times NAV.
    assert abs(ledger['second_trade'] - 0.20 * ledger['pre_nav']) > 1.0
    # The first cost is charged on the first price observation, before any turnover row exists.
    same_day = qis.backtest_model_portfolio(
        prices, targets.iloc[[0]], rebalancing_costs=0.0010, weight_implementation_lag=0)
    assert abs(same_day.realized_costs.iloc[0].sum() - 0.10) < 1e-12
    assert abs(same_day.nav.iloc[0] - 99.90) < 1e-12
    assert np.isnan(same_day.get_turnover(is_agg=True, roll_period=None).iloc[0])
    # Rates are read on the execution date, and a later schedule row cannot alter past costs.
    schedule = pd.DataFrame({"A": [0.0, 0.001, 0.003], "B": [0.0, 0.002, 0.001]},
                            index=prices.index)
    scheduled = qis.backtest_model_portfolio(
        prices, targets, rebalancing_costs=schedule, weight_implementation_lag=1)
    assert abs(scheduled.realized_costs.iloc[1].sum() - (0.06 * 1 + 0.04 * 2)) < 1e-12
    expected = scheduled.units.diff().abs().mul(prices).mul(schedule)
    np.testing.assert_allclose(scheduled.realized_costs.iloc[1:], expected.iloc[1:], atol=1e-12)
    future = schedule.copy()
    future.loc[pd.Timestamp("2024-01-05")] = [1.0, 1.0]
    later = qis.backtest_model_portfolio(
        prices, targets, rebalancing_costs=future, weight_implementation_lag=1)
    pd.testing.assert_frame_equal(scheduled.realized_costs, later.realized_costs)
    pd.testing.assert_series_equal(scheduled.nav, later.nav)
    # A ticker-indexed Series is a constant rate per column; a one-row DataFrame is
    # forward-filled onto the price grid, so both reproduce the matching constant rates.
    by_ticker = qis.backtest_model_portfolio(
        prices, targets, rebalancing_costs=pd.Series({"A": 0.001, "B": 0.002}),
        weight_implementation_lag=1)
    assert abs(by_ticker.realized_costs.iloc[1].sum() - 0.14) < 1e-12
    first_row = qis.backtest_model_portfolio(
        prices, targets, rebalancing_costs=pd.DataFrame({"A": [0.0010], "B": [0.0010]},
                                                        index=prices.index[:1]),
        weight_implementation_lag=1)
    pd.testing.assert_frame_equal(first_row.realized_costs, portfolio.realized_costs)
    # Dates before the first schedule row are costless, and a missing cell becomes zero.
    late = pd.DataFrame({"A": [0.001], "B": [np.nan]}, index=prices.index[[-1]])
    late_costs = qis.backtest_model_portfolio(
        prices, targets, rebalancing_costs=late, weight_implementation_lag=1).realized_costs
    np.testing.assert_allclose(late_costs.iloc[1], 0.0, atol=0.0)
    assert late_costs.iloc[2, 0] > 0.0 and late_costs.iloc[2, 1] == 0.0
    # A date-indexed Series and a DataFrame without every price column are rejected.
    assert_raises(ValueError, "date-indexed", qis.backtest_model_portfolio, prices, targets,
                  rebalancing_costs=pd.Series(0.001, index=prices.index),
                  weight_implementation_lag=1)
    assert_raises(ValueError, "missing price columns", qis.backtest_model_portfolio, prices,
                  targets, rebalancing_costs=pd.DataFrame({"A": 0.001}, index=prices.index),
                  weight_implementation_lag=1)

    executed_turnover = portfolio.get_turnover(
        is_agg=True, roll_period=None,
        turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_NAV,
    )
    target_turnover = portfolio.get_turnover(
        is_agg=True, roll_period=None,
        turnover_computation_type=qis.TurnoverComputationType.TARGET_WEIGHTS,
    )
    cash_costs = portfolio.realized_costs.sum(axis=1)
    cost_fractions = portfolio.get_costs(is_agg=True, roll_period=None)

    # Executed notional over same-date post-cost NAV, from the ledger: 100 / 99.90 = 1.001001
    # on 3 January and 0.185849 on 4 January; the first row is missing, not zero.
    np.testing.assert_allclose(cash_costs, ledger['costs'], rtol=0.0, atol=1e-12)
    assert np.isnan(executed_turnover.iloc[0])
    np.testing.assert_allclose(executed_turnover.iloc[1:],
                               np.array([100.0, ledger['second_trade']]) / ledger['nav'][1:],
                               rtol=0.0, atol=1e-12)
    assert [round(value, 6) for value in executed_turnover.iloc[1:]] == [1.001001, 0.185849]
    assert round(100 / 99.90, 6) == 1.001001
    # NAV turnover is the default convention of a new PortfolioData.
    assert portfolio.turnover_computation_type == qis.TurnoverComputationType.EXECUTED_NOTIONAL_NAV
    pd.testing.assert_series_equal(portfolio.get_turnover(is_agg=True, roll_period=None),
                                   executed_turnover)
    # The target proxy is dated on the second decision date, 3 January, not on execution.
    assert np.isnan(target_turnover.iloc[0]) and np.isnan(target_turnover.iloc[-1])
    assert abs(target_turnover.loc["2024-01-03"] - 0.20) < 1e-12
    # Cost fractions divide charges by same-date NAV; the scalar rate times NAV turnover gives
    # them wherever turnover is defined, and False returns the currency charges.
    np.testing.assert_allclose(cost_fractions, ledger['costs'] / ledger['nav'], atol=1e-12)
    np.testing.assert_allclose(cost_fractions.iloc[1:], 0.0010 * executed_turnover.iloc[1:],
                               rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(portfolio.get_costs(is_agg=True, roll_period=None,
                                                   is_unit_based_traded_volume=False),
                               ledger['costs'], rtol=0.0, atol=1e-12)
    # Gross-exposure turnover is exactly 1 at entry, below NAV turnover after the cost debit;
    # the deprecated boolean selects it and warns.
    gross, caught = recorded_warnings(
        portfolio.get_turnover, is_agg=True, roll_period=None,
        turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_GROSS)
    assert any(issubclass(item.category, RuntimeWarning)
               and "gross exposure is zero" in str(item.message) for item in caught)
    assert abs(gross.iloc[1] - 1.0) < 1e-12 and executed_turnover.iloc[1] > gross.iloc[1]
    legacy, caught = recorded_warnings(portfolio.get_turnover, is_agg=True, roll_period=None,
                                       is_unit_based_traded_volume=True)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    pd.testing.assert_series_equal(gross, legacy)
    # The default window is 260 observations, so the three-row example is all missing; rolling
    # and resampling sum observations and do not annualise.
    for method in (portfolio.get_turnover, portfolio.get_costs):
        assert inspect.signature(method).parameters["roll_period"].default == 260
    assert portfolio.get_turnover(is_agg=True).isna().all()
    rolled = portfolio.get_turnover(is_agg=True, roll_period=2)
    np.testing.assert_allclose(rolled, executed_turnover.rolling(2).sum(), equal_nan=True)
    monthly = portfolio.get_turnover(is_agg=True, roll_period=None, freq="ME")
    assert len(monthly) == 1 and abs(monthly.iloc[0] - executed_turnover.sum()) < 1e-12
    # With freq, the rolling count applies to the resampled rows: two months are needed.
    assert portfolio.get_turnover(is_agg=True, roll_period=2, freq="ME").isna().all()
    monthly_costs = portfolio.get_costs(is_agg=True, roll_period=None, freq="ME")
    assert abs(monthly_costs.iloc[0] - cost_fractions.sum()) < 1e-12
    # Summed cost fractions differ from the NAV gap to the same backtest without costs.
    costless = qis.backtest_model_portfolio(prices=prices, weights=targets,
                                            weight_implementation_lag=1)
    nav_gap = 1.0 - portfolio.nav.iloc[-1] / costless.nav.iloc[-1]
    assert abs(nav_gap - cost_fractions.sum()) > 1e-7
    print("turnover_and_transaction_costs: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: the turnover and tracking error of each penalty weight.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import optimalportfolios as opt

    weights, outcomes = solve_penalty_path()
    covar = penalty_covariance()
    benchmark = np.array(PENALTY_BENCHMARK)
    holdings = np.array(PENALTY_HOLDINGS)
    table = pd.DataFrame({
        'turnover': [explicit_turnover(row, holdings) for row in weights.to_numpy()],
        'tracking_error': [explicit_tracking_error(row, benchmark, covar)
                           for row in weights.to_numpy()],
    }, index=pd.Index(weights.index, name='turnover_utility_weight'))
    threshold = no_trade_threshold()
    turnover_max = table['turnover'].max()

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue = '#2a78d6'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    labelled = table.loc[LABELLED_PENALTY_WEIGHTS]
    names = {LABELLED_PENALTY_WEIGHTS[0]: 'benchmark', LABELLED_PENALTY_WEIGHTS[-1]: 'holdings'}
    left.plot(100.0 * table['tracking_error'], 100.0 * table['turnover'], color=blue,
              linewidth=1.8)
    left.scatter(100.0 * labelled['tracking_error'], 100.0 * labelled['turnover'], color=blue,
                 s=22, zorder=3)
    for weight, row in labelled.iterrows():
        text = f'{weight:g}: {names[weight]}' if weight in names else f'{weight:g}'
        left.annotate(text, (100.0 * row['tracking_error'], 100.0 * row['turnover']),
                      xytext=(7, 3), textcoords='offset points', fontsize=9, color=ink)
    # Room on the right keeps the last label clear of the curve.
    left.set_xlim(-0.08, 2.6)
    left.set_xlabel('ex-ante tracking error, % a year')
    left.set_ylabel('full L1 turnover, % of NAV')
    left.set_title('Each penalty weight on the frontier', loc='left', color=ink)

    right.plot(table.index, 100.0 * table['turnover'], color=blue, linewidth=1.8)
    right.scatter(labelled.index, 100.0 * labelled['turnover'], color=blue, s=22, zorder=3)
    right.axvline(threshold, color='#b9b8b3', linestyle='--', linewidth=1.2)
    right.annotate(f'no-trade threshold\n{threshold:.4f}', (threshold, 100.0 * turnover_max),
                   xytext=(-6, 0), textcoords='offset points', ha='right', va='top', fontsize=9,
                   color=muted)
    right.set_xlabel(f'turnover_utility_weight (tre_utility_weight {TRACKING_ERROR_WEIGHT:g})')
    right.set_ylabel('full L1 turnover, % of NAV')
    right.set_title('Turnover falls as the penalty grows', loc='left', color=ink)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.grid(color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    turnover, tracking = table['turnover'].to_numpy(), table['tracking_error'].to_numpy()
    stats = [opt.compute_tre_turnover_stats(
        covar=covar.to_numpy(), benchmark_weights=pd.Series(benchmark, index=PENALTY_ASSETS),
        weights=row, weights_0=pd.Series(holdings, index=PENALTY_ASSETS))
        for _, row in weights.iterrows()]
    above = table.index.to_numpy() >= threshold
    checks = {
        'turnover_never_rises': bool((np.diff(turnover) <= 1e-6).all()),
        'tracking_error_never_falls': bool((np.diff(tracking) >= -1e-7).all()),
        'zero_weight_holds_benchmark': bool(tracking[0] < 1e-6 and abs(
            turnover[0] - explicit_turnover(benchmark, holdings)) < 1e-6),
        'no_trade_at_and_above_threshold': bool((turnover[above] < 1e-6).all()),
        'trades_below_threshold': bool((turnover[~above] > 1e-3).all()),
        'matches_exact_first_order_solution': bool(np.allclose(
            weights, [kkt_reference(weight) for weight in weights.index], rtol=0.0, atol=5e-6)),
        'package_stats_match_explicit': bool(np.allclose(
            [[s[0], s[1]] for s in stats], np.c_[tracking, turnover], rtol=0.0, atol=1e-15)),
        'every_solve_accepted_and_compliant': bool(all(
            o.accepted and o.compliant and o.fallback_source is None for o in outcomes)),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
