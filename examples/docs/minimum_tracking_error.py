"""Canonical script of docs/minimum_tracking_error.md.

The page's two Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: a closed-form active-set solution and its first-order conditions, tracking error
as an explicit quadratic form, closed-form budget-constrained solves, hand-computed drift, and an
independent SciPy SLSQP solve of every step of the constraint-cost ladder. The script runs
offline after ``pip install optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.minimum_tracking_error

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
from contextlib import contextmanager
import logging

import numpy as np
import pandas as pd

LADDER_ASSETS = ['US equity', 'Europe equity', 'EM equity', 'Govt bonds', 'Credit']
# Annual volatilities and correlations of a stylised five-asset universe.
LADDER_VOLS = [0.16, 0.18, 0.22, 0.06, 0.08]
LADDER_CORR = [
    [1.00, 0.80, 0.70, -0.10, 0.40],
    [0.80, 1.00, 0.75, -0.05, 0.40],
    [0.70, 0.75, 1.00, 0.00, 0.45],
    [-0.10, -0.05, 0.00, 1.00, 0.50],
    [0.40, 0.40, 0.45, 0.50, 1.00],
]
LADDER_BENCHMARK = [0.35, 0.15, 0.10, 0.25, 0.15]
LADDER_HOLDINGS = [0.30, 0.10, 0.10, 0.35, 0.15]
EXCLUDED_ASSET = 'EM equity'
CAPPED_ASSET = 'US equity'
ASSET_CAP = 0.30
EQUITY_ASSETS = ['US equity', 'Europe equity', 'EM equity']
EQUITY_MAX = 0.50
TURNOVER_MAX = 0.30
HIGHLIGHTED_STEPS = [1, 4]  # the two steps whose active weights the figure shows


def ladder_covariance() -> pd.DataFrame:
    """Return the ladder's annual covariance from its volatilities and correlations."""
    vols = np.array(LADDER_VOLS)
    return pd.DataFrame(np.outer(vols, vols) * np.array(LADDER_CORR), index=LADDER_ASSETS,
                        columns=LADDER_ASSETS)


def ladder_steps() -> dict:
    """Return the cumulative constraints of the ladder, keyed by the step's label."""
    import optimalportfolios as opt

    zeros = pd.Series(0.0, index=LADDER_ASSETS)
    excluded = pd.Series(1.0, index=LADDER_ASSETS)
    excluded[EXCLUDED_ASSET] = 0.0
    capped = excluded.copy()
    capped[CAPPED_ASSET] = ASSET_CAP
    equity = opt.GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame(
            {'Equity': [float(asset in EQUITY_ASSETS) for asset in LADDER_ASSETS]},
            index=LADDER_ASSETS),
        group_min_allocation=None,
        group_max_allocation=pd.Series({'Equity': EQUITY_MAX}))
    return {
        'Fully invested, long-only': opt.Constraints(is_long_only=True),
        f'+ exclude {EXCLUDED_ASSET}': opt.Constraints(
            is_long_only=True, min_weights=zeros, max_weights=excluded),
        f'+ cap {CAPPED_ASSET} at {ASSET_CAP:.0%}': opt.Constraints(
            is_long_only=True, min_weights=zeros, max_weights=capped),
        f'+ equity at most {EQUITY_MAX:.0%}': opt.Constraints(
            is_long_only=True, min_weights=zeros, max_weights=capped,
            group_lower_upper_constraints=equity),
        f'+ turnover at most {TURNOVER_MAX:.0%}': opt.Constraints(
            is_long_only=True, min_weights=zeros, max_weights=capped,
            group_lower_upper_constraints=equity, turnover_constraint=TURNOVER_MAX),
    }


def solve_ladder() -> tuple:
    """Solve every ladder step with the wrapper, starting from the current holdings.

    Returns:
        Weights with one row per step, and the solver outcome of each step.
    """
    import optimalportfolios as opt

    covar = ladder_covariance()
    benchmark = pd.Series(LADDER_BENCHMARK, index=LADDER_ASSETS)
    holdings = pd.Series(LADDER_HOLDINGS, index=LADDER_ASSETS)
    rows, outcomes = {}, []
    for label, constraints in ladder_steps().items():
        weights, outcome = opt.wrapper_minimise_tracking_error(
            pd_covar=covar, benchmark_weights=benchmark, constraints=constraints,
            weights_0=holdings)
        rows[label] = weights
        outcomes.append(outcome)
    return pd.DataFrame(rows).T, outcomes


def explicit_tracking_error(weights, benchmark, covar) -> float:
    """Return sqrt((w - b)' Sigma (w - b)) with NumPy, independently of the package and qis."""
    active = np.asarray(weights, dtype=float) - np.asarray(benchmark, dtype=float)
    return float(np.sqrt(active @ np.asarray(covar, dtype=float) @ active))


def slsqp_reference(step: int) -> np.ndarray:
    """Solve ladder step ``step`` with SciPy SLSQP on explicit linear rows, without CVXPY.

    The variables are the weights and one trade size per asset, which only the turnover limit
    uses.

    Args:
        step: Zero-based position of the step in ``ladder_steps``.

    Returns:
        The reference minimum-tracking-error weights.
    """
    from scipy.optimize import minimize

    covar = ladder_covariance().to_numpy()
    benchmark = np.array(LADDER_BENCHMARK)
    holdings = np.array(LADDER_HOLDINGS)
    n = len(benchmark)
    upper = np.ones(n)
    if step >= 1:
        upper[LADDER_ASSETS.index(EXCLUDED_ASSET)] = 0.0
    if step >= 2:
        upper[LADDER_ASSETS.index(CAPPED_ASSET)] = ASSET_CAP
    zeros, ones, eye = np.zeros(n), np.ones(n), np.eye(n)
    rows, limits = [], []
    if step >= 3:
        rows.append(np.r_[[float(asset in EQUITY_ASSETS) for asset in LADDER_ASSETS], zeros])
        limits.append(EQUITY_MAX)
    if step >= 4:
        rows.extend(np.hstack([eye, -eye]))   # w - t <= holdings
        limits.extend(holdings)
        rows.extend(np.hstack([-eye, -eye]))  # -w - t <= -holdings
        limits.extend(-holdings)
        rows.append(np.r_[zeros, ones])       # total trade size <= turnover limit
        limits.append(TURNOVER_MAX)
    budget = np.r_[ones, zeros]
    constraints = [{'type': 'eq', 'fun': lambda x: budget @ x - 1.0, 'jac': lambda x: budget}]
    if rows:
        matrix, bound = np.array(rows), np.array(limits)
        constraints.append({'type': 'ineq', 'fun': lambda x: bound - matrix @ x,
                            'jac': lambda x: -matrix})

    def variance(x: np.ndarray) -> float:
        """Active variance of the weight block."""
        active = x[:n] - benchmark
        return float(active @ covar @ active)

    def gradient(x: np.ndarray) -> np.ndarray:
        """Gradient of the active variance; the trade sizes do not enter it."""
        return np.r_[2.0 * covar @ (x[:n] - benchmark), zeros]

    result = minimize(variance, np.r_[np.full(n, 1.0 / n), zeros], jac=gradient,
                      bounds=[(0.0, u) for u in upper] + [(0.0, None)] * n,
                      constraints=constraints, method='SLSQP',
                      options={'ftol': 1e-15, 'maxiter': 1000})
    # Status 8 is a line-search stop next to the optimum; feasibility is checked either way.
    assert result.status in (0, 8), result.message
    x = result.x
    assert abs(budget @ x - 1.0) < 1e-9 and (x[:n] >= -1e-9).all() and (x[:n] <= upper + 1e-9).all()
    assert not rows or (matrix @ x <= bound + 1e-9).all()
    return x[:n]


def budget_constrained_solve(covar: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimise (w - target)' Sigma (w - target) subject to sum(w) = 1 in closed form.

    With the budget row alone, the first-order condition gives w = target + lambda Sigma^-1 1,
    with lambda chosen so that the weights sum to one; the caller checks they are non-negative.
    """
    direction = np.linalg.solve(covar, np.ones(len(target)))
    return target + (1.0 - target.sum()) / direction.sum() * direction


@contextmanager
def solver_warnings_silenced():
    """Silence the package's fallback warnings while a deliberately failing solve runs."""
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def assert_raises(error: type, function, **arguments) -> None:
    """Fail unless ``function(**arguments)`` raises ``error``."""
    try:
        function(**arguments)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import numpy as np
    import pandas as pd
    import optimalportfolios as opt

    assets = ["A", "B", "C"]
    covar = pd.DataFrame(
        [[0.040, 0.006, 0.002],
         [0.006, 0.022, 0.003],
         [0.002, 0.003, 0.012]],
        index=assets,
        columns=assets,
    )
    benchmark = pd.Series([0.50, 0.30, 0.20], index=assets)
    benchmark = benchmark.reindex(covar.columns)
    if not np.isfinite(benchmark.to_numpy()).all():
        raise ValueError("The benchmark must cover every covariance asset.")

    constraints = opt.Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=assets),
        max_weights=pd.Series([0.35, 0.80, 0.80], index=assets),
    )
    weights, outcome = opt.wrapper_minimise_tracking_error(
        pd_covar=covar,
        benchmark_weights=benchmark,
        constraints=constraints,
        weights_0=benchmark,
    )
    if not (outcome.accepted and outcome.compliant):
        raise RuntimeError(
            f"Unusable solve: {outcome.status}; {outcome.reason}; "
            f"fallback={outcome.fallback_source}"
        )

    date = pd.Timestamp("2024-01-31")  # Synthetic snapshot label, not a data cutoff.
    risk_model = opt.build_risk_model({date: covar})
    tracking_error = risk_model.compute_tre_at_date(
        benchmark_weights=benchmark,
        portfolio_weights=weights,
        date=date,
    )
    result = pd.DataFrame({
        "Benchmark": benchmark,
        "Portfolio": weights,
        "Active": weights - benchmark,
    })
    print(result.round(6))
    print(f"Annualized tracking error: {tracking_error:.6%}")

    # The coverage guard rejects both an absent label and an explicit NaN. Without it the wrapper
    # fills the absent label with zero and solves against a benchmark summing to 0.8, while an
    # explicit NaN raises.
    for defective in (benchmark.drop("C"), benchmark.mask(benchmark.index == "C")):
        assert not np.isfinite(defective.reindex(covar.columns).to_numpy()).all()
    _, zero_filled = opt.wrapper_minimise_tracking_error(
        pd_covar=covar, benchmark_weights=benchmark.drop("C"), constraints=constraints)
    assert zero_filled.constraints.benchmark_weights.tolist() == [0.5, 0.3, 0.0]
    assert_raises(ValueError, opt.wrapper_minimise_tracking_error, pd_covar=covar,
                  benchmark_weights=benchmark.mask(benchmark.index == "C"),
                  constraints=constraints)

    # Independent certificate: fix A at its cap and solve the B-to-C transfer in closed form.
    sigma = covar.to_numpy()
    assert np.linalg.eigvalsh(sigma).min() > 0.0

    def transfer_derivative(x: float) -> float:
        """(Sigma d)_B - (Sigma d)_C with A at its cap, B = 0.30 + x and C = 0.35 - x."""
        gradient = sigma @ np.array([-0.15, x, 0.15 - x])
        return gradient[1] - gradient[2]

    assert abs(transfer_derivative(0.0) + 0.00195) < 1e-15
    assert abs(transfer_derivative(1.0) - transfer_derivative(0.0) - 0.028) < 1e-15
    transfer = 0.00195 / 0.028
    assert round(transfer, 9) == 0.069642857
    reference = np.array([0.35, 0.30 + transfer, 0.35 - transfer])
    assert abs(reference.sum() - 1.0) < 1e-15 and (reference >= 0.0).all()
    assert (reference <= constraints.max_weights.to_numpy()).all()
    gradient = sigma @ (reference - benchmark.to_numpy())
    assert abs(gradient[1] - gradient[2]) < 1e-14
    assert gradient[0] < gradient[2]  # moving weight back into A raises the objective
    np.testing.assert_allclose(weights, reference, rtol=0.0, atol=2e-6)
    assert outcome.accepted and outcome.compliant and outcome.fallback_source is None
    # A's cap binds; B and C absorb its 15-point reduction without reaching their caps.
    assert abs(weights["A"] - 0.35) < 2e-6 and (weights[["B", "C"]] < 0.80).all()
    # The page's table and tracking error, from the solve and from the closed-form weights.
    displayed = np.array([[0.500000, 0.350000, -0.150000],
                          [0.300000, 0.369643, 0.069643],
                          [0.200000, 0.280357, 0.080357]])
    assert result.index.tolist() == assets
    np.testing.assert_allclose(displayed, result, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        displayed[:, 1:], np.round(np.c_[reference, reference - benchmark.to_numpy()], 6),
        rtol=0.0, atol=1e-12)
    # The solver's stopping point moves the last digits; the exact optimum has 3.072778%.
    reference_te = explicit_tracking_error(reference, benchmark, covar)
    assert abs(100.0 * tracking_error - 3.072781) < 1e-6 and round(tracking_error, 4) == 0.0307
    assert round(100.0 * reference_te, 6) == 3.072778
    assert abs(tracking_error - reference_te) < 1e-7
    # No eigenvalue is floored, so the solver covariance is the supplied one. RiskModel applies
    # the supplied units on the exact grid date and rejects any other date.
    assert outcome.covar_factorization.n_eigenvalues_floored == 0
    np.testing.assert_allclose(outcome.covar_factorization.covar, sigma, rtol=0.0, atol=1e-15)
    assert abs(tracking_error - explicit_tracking_error(weights, benchmark, covar)) < 1e-15
    assert_raises(KeyError, risk_model.compute_tre_at_date, benchmark_weights=benchmark,
                  portfolio_weights=weights, date=pd.Timestamp("2024-01-30"))
    # Scaling the covariance scales tracking error by the square root and keeps the allocation.
    scaled_covar = covar / 12.0
    scaled_te = opt.build_risk_model({date: scaled_covar}).compute_tre_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date)
    assert abs(scaled_te * np.sqrt(12.0) / tracking_error - 1.0) < 1e-12
    scaled_weights, scaled_outcome = opt.wrapper_minimise_tracking_error(
        pd_covar=scaled_covar, benchmark_weights=benchmark, constraints=constraints,
        weights_0=benchmark)
    assert scaled_outcome.accepted and scaled_outcome.compliant
    assert scaled_outcome.covar_factorization.n_eigenvalues_floored == 0
    np.testing.assert_allclose(scaled_weights, weights, rtol=0.0, atol=2e-6)

    # The constraint-cost ladder. Each step keeps the previous constraints, so the minimum
    # tracking error cannot fall; the wrapper agrees with an SLSQP solve of explicit rows.
    ladder, outcomes = solve_ladder()
    ladder_covar = ladder_covariance()
    ladder_benchmark = np.array(LADDER_BENCHMARK)
    holdings = np.array(LADDER_HOLDINGS)
    # The inputs the page quotes: volatilities from 6% to 22%, the lowest in government bonds,
    # a 60/40 benchmark of 35/15/10/25/15 and holdings of 30/10/10/35/15.
    volatilities = np.sqrt(np.diag(ladder_covar))
    assert volatilities.argmin() == LADDER_ASSETS.index("Govt bonds")
    assert round(volatilities.min(), 12) == 0.06 and round(volatilities.max(), 12) == 0.22
    assert round(ladder_benchmark[:3].sum(), 12) == 0.60 and abs(ladder_benchmark.sum() - 1) < 1e-15
    assert np.round(100 * ladder_benchmark).tolist() == [35, 15, 10, 25, 15]
    assert np.round(100 * holdings).tolist() == [30, 10, 10, 35, 15]
    assert all(o.accepted and o.compliant and o.fallback_source is None for o in outcomes)
    costs = np.array([explicit_tracking_error(row, ladder_benchmark, ladder_covar)
                      for row in ladder.to_numpy()])
    references = np.array([slsqp_reference(step) for step in range(len(ladder))])
    np.testing.assert_allclose(ladder, references, rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(
        costs, [explicit_tracking_error(row, ladder_benchmark, ladder_covar)
                for row in references], rtol=0.0, atol=5e-7)
    assert costs[0] < 1e-6 and (np.diff(costs) > 1e-3).all()
    assert (np.round(100.0 * costs, 2) == [0.00, 1.38, 1.57, 1.86, 2.06]).all()
    # The zero cap in closed form: with EM held at zero, the other weights minimise a quadratic
    # centred on the benchmark plus EM's regression on them, subject to the budget alone.
    kept = [asset for asset in LADDER_ASSETS if asset != EXCLUDED_ASSET]
    sigma_kept = ladder_covar.loc[kept, kept].to_numpy()
    excluded_weight = LADDER_BENCHMARK[LADDER_ASSETS.index(EXCLUDED_ASSET)]
    replication = excluded_weight * np.linalg.solve(
        sigma_kept, ladder_covar.loc[kept, EXCLUDED_ASSET].to_numpy())
    zero_cap = budget_constrained_solve(
        sigma_kept, pd.Series(LADDER_BENCHMARK, index=LADDER_ASSETS)[kept].to_numpy()
        + replication)
    assert (zero_cap >= 0.0).all()
    exclusion = ladder.iloc[1] - ladder_benchmark
    np.testing.assert_allclose(ladder.iloc[1][kept], zero_cap, rtol=0.0, atol=2e-6)
    assert np.round(100.0 * exclusion, 1).tolist() == [2.8, 6.0, -10.0, -3.8, 5.0]
    # With every constraint the solve sits at US equity's cap and the equity limit; selling EM
    # into European equity uses 0.20 of the 0.30 turnover, leaving one 5-point switch from
    # government bonds into credit instead of 17.3 points without the limit.
    final = ladder.iloc[4]
    np.testing.assert_allclose(final - ladder_benchmark, [-0.05, 0.05, -0.10, 0.05, 0.05],
                               rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(final - holdings, [0.0, 0.10, -0.10, -0.05, 0.05], atol=1e-5)
    assert abs(np.abs(final - holdings).sum() - TURNOVER_MAX) < 1e-6
    assert round(holdings[3] - ladder.iloc[3]["Govt bonds"], 3) == 0.173

    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    prices = pd.DataFrame(100.0, index=dates, columns=assets)
    benchmarks = pd.DataFrame(
        [[0.50, 0.30, 0.20], [0.20, 0.50, 0.30]],
        index=dates[[0, 2]],
        columns=assets,
    )
    rolling_weights = opt.rolling_minimise_tracking_error(
        prices=prices,
        constraints=opt.Constraints(is_long_only=True),
        benchmark_weights=benchmarks,
        covar_dict={snapshot: covar for snapshot in dates},
    )
    print(rolling_weights.round(6))

    # Each date holds the latest benchmark observation at or before it; the page's table.
    expected = benchmarks.reindex(dates, method="ffill")
    np.testing.assert_allclose(rolling_weights, expected, rtol=0.0, atol=2e-6)
    pd.testing.assert_index_equal(rolling_weights.index, dates)
    pd.testing.assert_index_equal(rolling_weights.columns, prices.columns)
    assert rolling_weights.index.strftime("%Y-%m-%d").tolist() == [
        "2024-01-31", "2024-02-29", "2024-03-31"]
    np.testing.assert_allclose(rolling_weights, [[0.50, 0.30, 0.20], [0.50, 0.30, 0.20],
                                                 [0.20, 0.50, 0.30]], rtol=0.0, atol=1e-6)
    # A different March observation changes March alone: no later observation is borrowed.
    perturbed = benchmarks.copy()
    perturbed.loc[dates[-1]] = [0.10, 0.20, 0.70]
    changed = opt.rolling_minimise_tracking_error(
        prices=prices, constraints=opt.Constraints(is_long_only=True),
        benchmark_weights=perturbed, covar_dict={snapshot: covar for snapshot in dates})
    np.testing.assert_allclose(changed.iloc[:2], rolling_weights.iloc[:2], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(changed.iloc[-1], perturbed.iloc[-1], rtol=0.0, atol=2e-6)
    # The rolling interface rejects a missing column, a NaN at a selected observation and a
    # benchmark that starts after the first decision.
    for defective in (benchmarks.drop(columns="C"), benchmarks.mask(benchmarks == 0.30),
                      benchmarks.iloc[1:]):
        assert_raises(ValueError, opt.rolling_minimise_tracking_error, prices=prices,
                      constraints=opt.Constraints(is_long_only=True),
                      benchmark_weights=defective,
                      covar_dict={snapshot: covar for snapshot in dates})
    # With a zero turnover limit, the first date holds the stored Constraints.weights_0
    # unchanged, or the benchmark when none is stored; the next date holds that target drifted
    # by a 10% rise in A, computed here by hand.
    moving = pd.DataFrame([[100.0] * 3, [110.0, 100.0, 100.0]], index=dates[:2], columns=assets)
    stored = pd.Series([0.20, 0.20, 0.60], index=assets)
    for baseline in (stored, None):
        pinned = opt.rolling_minimise_tracking_error(
            prices=moving,
            constraints=opt.Constraints(is_long_only=True, weights_0=baseline,
                                        turnover_constraint=0.0),
            benchmark_weights=benchmark, covar_dict={snapshot: covar for snapshot in dates[:2]})
        first = benchmark if baseline is None else baseline
        grown = first * np.array([1.10, 1.0, 1.0])
        np.testing.assert_allclose(pinned, [first, grown / grown.sum()], rtol=0.0, atol=1e-6)
    _, pinned_outcome = opt.wrapper_minimise_tracking_error(
        pd_covar=covar, benchmark_weights=benchmark,
        constraints=opt.Constraints(is_long_only=True, turnover_constraint=0.0),
        weights_0=stored)
    assert pinned_outcome.accepted and pinned_outcome.fallback_source is None

    # Input filtering. Excluding EM with an inclusion indicator drops its covariances and its
    # benchmark weight without renormalising; the retained solve, in closed form, puts EM's 10%
    # mostly into government bonds and raises full-benchmark tracking error from 1.38% to 2.10%.
    indicators = pd.Series([float(asset != EXCLUDED_ASSET) for asset in LADDER_ASSETS],
                           index=LADDER_ASSETS)
    filtered, filtered_outcome = opt.wrapper_minimise_tracking_error(
        pd_covar=ladder_covar,
        benchmark_weights=pd.Series(LADDER_BENCHMARK, index=LADDER_ASSETS),
        constraints=opt.Constraints(is_long_only=True), inclusion_indicators=indicators)
    retained_benchmark = filtered_outcome.constraints.benchmark_weights
    assert retained_benchmark.index.tolist() == kept
    assert abs(retained_benchmark.sum() - 0.90) < 1e-15
    retained = budget_constrained_solve(sigma_kept, retained_benchmark.to_numpy())
    assert (retained >= 0.0).all()
    np.testing.assert_allclose(filtered[kept], retained, rtol=0.0, atol=2e-6)
    filtered_active = filtered - ladder_benchmark
    assert round(filtered_active["Govt bonds"], 3) == 0.082
    assert filtered_active["Govt bonds"] > 0.8 * filtered_active[kept].clip(lower=0.0).sum()
    filtered_cost = explicit_tracking_error(filtered, ladder_benchmark, ladder_covar)
    assert round(100.0 * filtered_cost, 2) == 2.10 and filtered_cost > costs[1]
    # The wrapper keeps the covariance order with zero at the excluded asset; the outcome
    # describes the retained universe only. A zero-variance asset is also removed.
    assert filtered.index.tolist() == LADDER_ASSETS and filtered[EXCLUDED_ASSET] == 0.0
    assert len(filtered_outcome.weights) == len(kept)
    with_cash = covar.reindex(index=assets + ["Cash"], columns=assets + ["Cash"]).fillna(0.0)
    cash_weights, cash_outcome = opt.wrapper_minimise_tracking_error(
        pd_covar=with_cash, constraints=opt.Constraints(is_long_only=True),
        benchmark_weights=pd.Series([0.4, 0.3, 0.2, 0.1], index=with_cash.index))
    assert cash_weights["Cash"] == 0.0 and cash_outcome.constraints.benchmark_weights.size == 3
    # An asymmetric covariance and an empty retained universe are rejected.
    asymmetric = covar.copy()
    asymmetric.iloc[0, 1] = 0.007
    assert_raises(ValueError, opt.wrapper_minimise_tracking_error, pd_covar=asymmetric,
                  benchmark_weights=benchmark, constraints=constraints)
    assert_raises(ValueError, opt.wrapper_minimise_tracking_error, pd_covar=covar,
                  benchmark_weights=benchmark, constraints=constraints,
                  inclusion_indicators=pd.Series(0.0, index=assets))

    # Stabilisation: a zero or tiny negative eigenvalue is raised to the 1e-10 floor, and a
    # singular covariance lets another allocation match the benchmark's zero tracking error.
    pair = ["X", "Y"]
    even = pd.Series([0.5, 0.5], index=pair)
    for offset, raw_sign in ((0.0, 0.0), (1e-12, -1.0)):
        singular = pd.DataFrame([[0.04, 0.04 + offset], [0.04 + offset, 0.04]], index=pair,
                                columns=pair)
        _, singular_outcome = opt.wrapper_minimise_tracking_error(
            pd_covar=singular, benchmark_weights=even,
            constraints=opt.Constraints(is_long_only=True))
        factorization = singular_outcome.covar_factorization
        assert np.sign(factorization.raw_min_eigenvalue) == raw_sign
        assert factorization.n_eigenvalues_floored == 1
        assert factorization.stabilized_min_eigenvalue == 1e-10
    assert explicit_tracking_error([1.0, 0.0], even, np.full((2, 2), 0.04)) == 0.0
    # The floor is absolute: a 1e-9 eigenvalue survives in annual units, not in monthly ones.
    tiny = pd.DataFrame(np.diag([0.04, 1e-9]), index=pair, columns=pair)
    for scale, floored in ((1.0, 0), (1.0 / 12.0, 1)):
        _, tiny_outcome = opt.wrapper_minimise_tracking_error(
            pd_covar=tiny * scale, benchmark_weights=even,
            constraints=opt.Constraints(is_long_only=True))
        assert tiny_outcome.covar_factorization.n_eigenvalues_floored == floored
    # A materially negative eigenvalue raises instead.
    indefinite = pd.DataFrame([[0.04, 0.05], [0.05, 0.04]], index=pair, columns=pair)
    assert_raises(ValueError, opt.wrapper_minimise_tracking_error, pd_covar=indefinite,
                  benchmark_weights=even, constraints=opt.Constraints(is_long_only=True))
    # Fallback order: current weights, then the benchmark. Caps summing to 0.6 are infeasible,
    # and neither fallback satisfies them.
    infeasible = opt.Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=assets),
                                 max_weights=pd.Series(0.2, index=assets))
    with solver_warnings_silenced():
        for current, source, returned in ((stored, "weights_0", stored),
                                          (None, "benchmark_weights", benchmark)):
            fallback, failed = opt.wrapper_minimise_tracking_error(
                pd_covar=covar, benchmark_weights=benchmark, constraints=infeasible,
                weights_0=current)
            assert failed.fallback_source == source and not failed.accepted
            assert not failed.compliant
            pd.testing.assert_series_equal(fallback, returned, check_names=False)
    print("minimum_tracking_error: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: tracking error as constraints are added, and two active sets.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ladder, outcomes = solve_ladder()
    covar = ladder_covariance()
    benchmark = np.array(LADDER_BENCHMARK)
    holdings = np.array(LADDER_HOLDINGS)
    costs = pd.Series([explicit_tracking_error(row, benchmark, covar) for row in ladder.to_numpy()],
                      index=ladder.index, name='tracking_error')
    active = ladder - benchmark
    table = pd.concat([costs, active.add_prefix('active: ')], axis=1)

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    highlight = dict(zip(HIGHLIGHTED_STEPS, ['#2a78d6', '#eb6834']))
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    rows = np.arange(len(ladder))[::-1]
    left.barh(rows, 100.0 * costs, height=0.6,
              color=[highlight.get(step, '#b9b8b3') for step in range(len(ladder))])
    for row, value in zip(rows, costs):
        left.text(100.0 * value + 0.04, row, f'{100.0 * value:.2f}%', color=ink, fontsize=9,
                  va='center')
    left.set_yticks(rows, ladder.index)
    left.set_xlim(0.0, 2.6)
    left.set_xlabel('ex-ante tracking error, % a year')
    left.set_title('Each step keeps the constraints above it', loc='left', color=ink)

    positions = np.arange(len(LADDER_ASSETS))[::-1]
    for offset, (step, colour) in zip((0.19, -0.19), highlight.items()):
        right.barh(positions + offset, 100.0 * active.iloc[step], height=0.36, color=colour,
                   label=ladder.index[step])
    # Room below the last asset keeps the legend clear of the bars and of the zero line.
    right.vlines(0.0, -0.5, len(LADDER_ASSETS) - 0.45, color=muted, linewidth=0.8)
    right.set_yticks(positions, LADDER_ASSETS)
    right.set_xlim(-12.0, 12.0)
    right.set_ylim(-1.55, len(LADDER_ASSETS) - 0.45)
    right.set_xlabel('active weight, percentage points')
    right.set_title('Active weights at two steps', loc='left', color=ink)
    right.legend(loc='lower left', fontsize=9, labelcolor=ink, facecolor=surface,
                 edgecolor='none', framealpha=1.0)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.grid(axis='x', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis='y', length=0)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    final = ladder.iloc[-1]
    checks = {
        'tracking_error_never_falls': bool((np.diff(costs) >= -1e-9).all()),
        'benchmark_feasible_at_first_step': bool(costs.iloc[0] < 1e-6),
        'every_step_accepted_and_compliant': bool(all(o.accepted and o.compliant
                                                      for o in outcomes)),
        'matches_slsqp_reference': bool(np.allclose(
            costs, [explicit_tracking_error(slsqp_reference(step), benchmark, covar)
                    for step in range(len(ladder))], rtol=0.0, atol=5e-7)),
        'final_step_meets_every_limit': bool(
            final[EXCLUDED_ASSET] <= 1e-6 and final[CAPPED_ASSET] <= ASSET_CAP + 1e-6
            and final[EQUITY_ASSETS].sum() <= EQUITY_MAX + 1e-6
            and np.abs(final - holdings).sum() <= TURNOVER_MAX + 1e-6),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
