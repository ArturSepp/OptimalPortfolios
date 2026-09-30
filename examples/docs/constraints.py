"""Canonical script of docs/constraints.md.

The page's seventeen Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: hand arithmetic of each small example, explicit NumPy quadratic forms and L1 sums
in place of the residual evaluator, the three binding rows and the dual certificate of the forced
optimum, an independent CVXPY solve of the utility example written from raw arrays, the
closed-form solution of a tracking-error penalty and the rows each compiler emits. The script
runs offline after ``pip install optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.constraints

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
from contextlib import contextmanager
import dataclasses
import logging
import warnings

import cvxpy as cvx
import numpy as np
import pandas as pd

ASSETS = ['Equity', 'Bond', 'Gold']
# Annual covariance of the page's three-asset example, in fractional return squared.
COVARIANCE = [[0.0324, 0.0018, 0.0024],
              [0.0018, 0.0064, 0.0006],
              [0.0024, 0.0006, 0.0144]]
BENCHMARK = [0.45, 0.40, 0.15]
ALPHAS = [0.030, 0.005, 0.010]  # annual alphas of the utility example and of the figure
TRACKING_ERROR_LIMIT = 0.02  # the figure's hard limit on annual tracking error
PENALTY_WEIGHTS = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]  # tre_utility_weight of the penalty solves
# The forced example's remaining inputs.
CURRENT = [0.40, 0.45, 0.15]
EXPECTED_RETURNS = [0.070, 0.035, 0.040]
MIN_WEIGHTS = [0.20, 0.20, 0.05]
MAX_WEIGHTS = [0.60, 0.65, 0.25]
TURNOVER_COSTS = [1.0, 0.5, 2.0]
GROWTH = [1.0, 0.0, 0.0]
DEFENSIVE = [0.0, 1.0, 1.0]
RISK_ASSETS = [1.0, 0.0, 1.0]
INFLATION = [0.5, -0.5, 1.0]
CONSTRAINTS_FIELDS = [
    'is_long_only', 'min_weights', 'max_weights', 'max_exposure', 'min_exposure',
    'benchmark_weights', 'tracking_err_vol_constraint', 'weights_0', 'turnover_constraint',
    'turnover_costs', 'target_return', 'asset_returns', 'max_target_portfolio_vol_an',
    'constraint_enforcement_type', 'tre_utility_weight', 'turnover_utility_weight',
    'group_lower_upper_constraints', 'group_tracking_error_constraint',
    'group_turnover_constraint', 'sector_deviation_constraints', 'style_deviation_constraints',
    'benchmark_beta_constraint',
    'linear_constraints',
]
CONSTRAINTS_LOGGER = 'optimalportfolios.optimization.constraints'
# The reference solves run CLARABEL to tighter tolerances than the page's blocks, so that their
# dual values and weights serve as references.
SOLVER_OPTIONS = {'solver': 'CLARABEL', 'tol_gap_abs': 1e-10, 'tol_gap_rel': 1e-10,
                  'tol_feas': 1e-10, 'max_iter': 500}


def compiled_rows(constraints, n: int, covar=None, utility: bool = False) -> list:
    """Compile ``constraints`` for ``n`` assets with the forced or the generic utility builder."""
    w = cvx.Variable(n)
    wrapped = None if covar is None else cvx.psd_wrap(np.asarray(covar, dtype=float))
    if utility:
        return constraints.set_cvx_utility_objective_constraints(w=w, covar=wrapped)[1]
    return constraints.set_cvx_all_constraints(w=w, covar=wrapped)


def exposure_rows(constraints) -> list:
    """Class names of the long-only, exposure and box rows compiled for two assets."""
    return [type(row).__name__
            for row in constraints.set_cvx_exposure_constraints(w=cvx.Variable(2))]


def record_of(weights, constraints, kind: str, name=None, covar=None):
    """Return the one residual of ``kind`` (and ``name``) that the evaluator reports."""
    from optimalportfolios.optimization.constraints import evaluate_constraint_residuals

    records = [record for record in evaluate_constraint_residuals(
        np.asarray(weights, dtype=float), constraints,
        covar=None if covar is None else np.asarray(covar, dtype=float))
        if record.constraint_type == kind and (name is None or record.name == name)]
    assert len(records) == 1, records
    return records[0]


def scipy_callback_count(constraints, n: int) -> int:
    """Number of SciPy callbacks compiled for ``n`` assets."""
    return len(constraints.set_scipy_constraints(np.eye(n))[0])


def pyrb_rows(constraints, n: int) -> tuple:
    """Compile the risk-budgeting matrix rows for ``n`` assets."""
    return constraints.set_pyrb_constraints(np.eye(n))


def charnes_cooper_rows_hold(constraints, feasible, infeasible) -> bool:
    """Whether rows compiled on y = k w keep a feasible point and reject an infeasible one."""
    y, k = cvx.Variable(len(feasible)), cvx.Variable(nonneg=True)
    rows = constraints.set_cvx_all_constraints(w=y, exposure_scaler=k)
    for scale in (0.5, 4.0):
        y.value, k.value = scale * np.asarray(feasible, dtype=float), scale
        if not all(np.max(row.violation()) < 1e-12 for row in rows):
            return False
        y.value = scale * np.asarray(infeasible, dtype=float)
        if not any(np.max(row.violation()) > 0.1 * scale for row in rows):
            return False
    return True


def explicit_tracking_error(weights, benchmark, covar, loadings=None) -> float:
    """Return sqrt(d' Sigma d) for d = loadings * (w - benchmark), with NumPy alone."""
    active = np.asarray(weights, dtype=float) - np.asarray(benchmark, dtype=float)
    if loadings is not None:
        active = np.asarray(loadings, dtype=float) * active
    return float(np.sqrt(active @ np.asarray(covar, dtype=float) @ active))


def factorization_takes_precedence(constraints, covar, cap: float) -> bool:
    """Whether the volatility row and its residual follow a factorization over a wrong covariance.

    The covariance is diagonal, so the largest first weight under the cap solves the quadratic
    a w^2 + c (1 - w)^2 = cap^2; half the covariance, passed as ``covar``, would allow more.
    """
    from optimalportfolios import factorize_covariance
    from optimalportfolios.optimization.constraints import evaluate_constraint_residuals

    right = np.asarray(covar, dtype=float)
    wrong = 0.5 * right
    factorization = factorize_covariance(right)
    w = cvx.Variable(len(right))
    rows = constraints.set_cvx_all_constraints(w=w, covar=cvx.psd_wrap(wrong),
                                               covar_factorization=factorization)
    problem = cvx.Problem(cvx.Maximize(w[0]), rows)
    problem.solve(**SOLVER_OPTIONS)
    a, c = right[0, 0], right[1, 1]
    largest = (c + np.sqrt(c * c - (a + c) * (c - cap ** 2))) / (a + c)
    record = [record for record in evaluate_constraint_residuals(
        np.array([0.5, 0.5]), constraints, covar=wrong, covar_factorization=factorization)
        if record.constraint_type == 'portfolio_volatility'][0]
    return bool(problem.status == 'optimal' and abs(w.value[0] - largest) < 1e-6
                and abs(record.actual - np.sqrt(0.25 * (a + c))) < 1e-12)


def group_tre_penalty(constraint, benchmark):
    """The group tracking-error utility term that ``constraint`` builds for three assets."""
    return constraint.set_cvx_group_tre_utility(
        w=cvx.Variable(len(benchmark)), benchmark_weights=benchmark,
        covar=cvx.psd_wrap(np.array(COVARIANCE)))


def solve_forced(constraints, objective, covar) -> np.ndarray:
    """Maximise ``objective @ w`` over the hard rows that set_cvx_all_constraints compiles."""
    w = cvx.Variable(len(objective))
    rows = constraints.set_cvx_all_constraints(w=w, covar=cvx.psd_wrap(np.asarray(covar)))
    problem = cvx.Problem(cvx.Maximize(np.asarray(objective) @ w), rows)
    problem.solve(**SOLVER_OPTIONS)
    assert problem.status == 'optimal', problem.status
    return w.value


def solve_utility(constraints, alphas, covar) -> np.ndarray:
    """Maximise the generic utility of ``constraints`` over the rows it keeps hard."""
    w = cvx.Variable(len(alphas))
    utility, rows = constraints.set_cvx_utility_objective_constraints(
        w=w, alphas=np.asarray(alphas), covar=cvx.psd_wrap(np.asarray(covar)))
    problem = cvx.Problem(cvx.Maximize(utility), rows)
    problem.solve(**SOLVER_OPTIONS)
    assert problem.status == 'optimal', problem.status
    return w.value


def utility_reference(beta_loadings) -> np.ndarray:
    """Solve the page's utility example from raw arrays, without the package's compilers.

    The group penalties are 5 on each group's masked active variance, written as a sum of
    squares of the Cholesky factor, and 0.02 on each group's L1 trade.
    """
    sigma = np.array(COVARIANCE)
    root = np.linalg.cholesky(sigma).T
    benchmark = np.array(BENCHMARK)
    w = cvx.Variable(len(ASSETS))
    active = w - benchmark
    objective = np.array(ALPHAS) @ active
    for loading in (np.array(GROWTH), np.array(DEFENSIVE)):
        objective = objective - 5.0 * cvx.sum_squares(root @ cvx.multiply(loading, active))
        objective = objective - 0.02 * cvx.norm1(cvx.multiply(loading, w - np.array(CURRENT)))
    growth, defensive = np.array(GROWTH), np.array(DEFENSIVE)
    rows = [w >= 0.0, cvx.sum(w) == 1.0, w >= np.array(MIN_WEIGHTS), w <= np.array(MAX_WEIGHTS),
            np.array(EXPECTED_RETURNS) @ w >= 0.045,
            growth @ w >= 0.30, growth @ w <= 0.55, defensive @ w >= 0.45, defensive @ w <= 0.70,
            cvx.abs(np.array(RISK_ASSETS) @ active) <= 0.08,
            cvx.abs(np.array(INFLATION) @ active) <= 0.12,
            np.asarray(beta_loadings) @ w >= 0.85, np.asarray(beta_loadings) @ w <= 1.15]
    problem = cvx.Problem(cvx.Maximize(objective), rows)
    problem.solve(**SOLVER_OPTIONS)
    assert problem.status == 'optimal', problem.status
    return w.value


def soft_tracking_error_turnover(turnover_utility_weight, turnover_constraint=None) -> float:
    """L1 turnover of the soft-tracking-error alpha solve on the forced example's inputs."""
    from optimalportfolios import cvx_maximise_alpha_with_target_return
    from optimalportfolios.optimization.constraints import Constraints

    spec = Constraints(benchmark_weights=pd.Series(BENCHMARK, index=ASSETS),
                       weights_0=pd.Series(CURRENT, index=ASSETS),
                       asset_returns=pd.Series(EXPECTED_RETURNS, index=ASSETS),
                       target_return=0.045, turnover_constraint=turnover_constraint,
                       turnover_utility_weight=turnover_utility_weight)
    outcome = cvx_maximise_alpha_with_target_return(
        covar=np.array(COVARIANCE), alphas=np.array(ALPHAS), constraints=spec,
        soft_tracking_error=True)
    assert outcome.accepted, outcome.reason
    return float(np.abs(outcome.weights - np.array(CURRENT)).sum())


def closed_form_active_weights(alphas, covar, penalty_weight: float) -> np.ndarray:
    """Maximise a'd - penalty d' Sigma d subject to sum(d) = 0 alone, in closed form."""
    direction = np.linalg.solve(np.asarray(covar), np.asarray(alphas))
    ones = np.linalg.solve(np.asarray(covar), np.ones(len(alphas)))
    return (direction - direction.sum() / ones.sum() * ones) / (2.0 * penalty_weight)


def tracking_error_path(limit: float, penalty_weights) -> tuple:
    """Solve the figure's problem with a hard tracking-error limit and at each penalty weight.

    The problem is the three-asset example, long-only and fully invested, maximising the active
    expected return ALPHAS @ (w - BENCHMARK) with no other row.

    Args:
        limit: Hard limit on annual tracking error.
        penalty_weights: ``tre_utility_weight`` values of the utility solves.

    Returns:
        A table with one row per solve, and the dual value of the hard limit's row.
    """
    from optimalportfolios.optimization.constraints import (
        ConstraintEnforcementType,
        Constraints,
    )

    sigma = np.array(COVARIANCE)
    benchmark = np.array(BENCHMARK)
    alphas = np.array(ALPHAS)
    forced = Constraints(benchmark_weights=pd.Series(BENCHMARK, index=ASSETS),
                         tracking_err_vol_constraint=limit)
    w = cvx.Variable(len(ASSETS))
    rows = forced.set_cvx_all_constraints(w=w, covar=cvx.psd_wrap(sigma))
    assert len(rows) == 3  # w >= 0, sum(w) == 1 and the tracking-error row, in that order
    problem = cvx.Problem(cvx.Maximize(alphas @ (w - benchmark)), rows)
    problem.solve(**SOLVER_OPTIONS)
    assert problem.status == 'optimal', problem.status
    solves = {'hard limit': (np.nan, w.value, problem.value)}
    for weight in penalty_weights:
        soft = forced.copy(
            constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
            tre_utility_weight=weight)
        weights = solve_utility(soft, alphas, sigma)
        active = weights - benchmark
        solves[f'penalty {weight:g}'] = (weight, weights,
                                         alphas @ active - weight * active @ sigma @ active)
    table = pd.DataFrame(
        [[weight, *weights, explicit_tracking_error(weights, benchmark, sigma),
          alphas @ (weights - benchmark), objective]
         for weight, weights, objective in solves.values()],
        index=list(solves), columns=['penalty_weight', *ASSETS, 'tracking_error',
                                     'active_return', 'objective'])
    table.insert(5, 'excess_over_limit', (table['tracking_error'] - limit).clip(lower=0.0))
    return table, float(np.squeeze(rows[2].dual_value))


def assert_raises(error: type, function, **arguments) -> None:
    """Fail unless ``function(**arguments)`` raises ``error``."""
    try:
        function(**arguments)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def warned(function, **arguments) -> tuple:
    """Call ``function`` and return its result and the messages of the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = function(**arguments)
    return result, [str(item.message) for item in caught]


class Captured(logging.Handler):
    """Collect the log records emitted while attached."""

    def __init__(self) -> None:
        """Start with no records."""
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record."""
        self.records.append(record)


def logged(function, **arguments) -> tuple:
    """Call ``function`` with the constraint logger captured at DEBUG; return result, records."""
    logger = logging.getLogger(CONSTRAINTS_LOGGER)
    handler, level, propagate = Captured(), logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        return function(**arguments), handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate


@contextmanager
def solver_warnings_silenced():
    """Silence the package's fallback warnings while a deliberately failing solve runs."""
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    from optimalportfolios.optimization.constraints import (
        BenchmarkBetaConstraint,
        BenchmarkDeviationConstraints,
        ConstraintEnforcementType,
        Constraints,
        GroupLowerUpperConstraints,
        GroupTrackingErrorConstraint,
        GroupTurnoverConstraint,
        evaluate_constraint_residuals,
    )

    # Two enforcement policies. Constraints() is long-only and fully invested with no boxes, its
    # 23 fields are those of the constraint map, and the utility penalty weights default to 1.0
    # for tracking error and 0.40 for turnover.
    utility_type = ConstraintEnforcementType.UTILITY_CONSTRAINTS
    assert [member.name for member in ConstraintEnforcementType] == [
        'FORCED_CONSTRAINTS', 'UTILITY_CONSTRAINTS']
    default = Constraints()
    assert default.is_long_only and default.min_exposure == default.max_exposure == 1.0
    assert default.min_weights is None and default.max_weights is None
    assert default.constraint_enforcement_type is ConstraintEnforcementType.FORCED_CONSTRAINTS
    assert default.tre_utility_weight == 1.0 and default.turnover_utility_weight == 0.40
    assert [field.name for field in dataclasses.fields(Constraints)] == CONSTRAINTS_FIELDS
    supporting = {'benchmark_weights', 'weights_0', 'asset_returns', 'turnover_costs'}
    assert len(CONSTRAINTS_FIELDS) == 23 and len(set(CONSTRAINTS_FIELDS) - supporting) == 19
    assert exposure_rows(default) == ['Inequality', 'Equality']  # w >= 0 and sum(w) == 1
    # Exposure is the signed net sum: [1.20, -0.20] has net exposure 1.00 and gross 1.40.
    net = record_of([1.20, -0.20], Constraints(is_long_only=False), 'exposure')
    assert net.passed and round(net.actual, 12) == 1.00
    assert round(abs(1.20) + abs(-0.20), 12) == 1.40
    # Units are the caller's: 0.0324 is an 18% volatility in the annual examples.
    assert round(COVARIANCE[0][0] ** 0.5, 12) == 0.18

    import pandas as pd

    assets = pd.Index(["A", "B"])
    box = Constraints(
        is_long_only=True,
        min_exposure=0.80,
        max_exposure=1.00,
        min_weights=pd.Series([0.10, 0.00], index=assets),
        max_weights=pd.Series([0.70, 0.80], index=assets),
    )

    # The candidate [0.60, 0.30] passes: exposure 0.90 lies in [0.80, 1.00], A in [0.10, 0.70]
    # and B in [0.00, 0.80]. The evaluator agrees on all six rows.
    candidate = [0.60, 0.30]
    assert round(sum(candidate), 12) == 0.90 and 0.80 <= 0.90 <= 1.00
    assert 0.10 <= candidate[0] <= 0.70 and 0.00 <= candidate[1] <= 0.80
    box_records = evaluate_constraint_residuals(candidate, box)
    assert len(box_records) == 6 and all(record.passed for record in box_records)
    assert round(box_records[0].actual, 12) == 0.90
    # A band compiles two exposure rows; equal limits one equality; 1.000005 stays a band.
    assert exposure_rows(box) == ['Inequality'] * 5
    assert exposure_rows(Constraints(min_exposure=1.0, max_exposure=1.000005)) == [
        'Inequality'] * 3
    # Constructor validation: min > max + 1e-10 on equal indexes, long-only minima below
    # -1e-10, and three group/box contradictions at a 1e-4 tolerance on positive loadings.
    pair = pd.Index(["A", "B"])
    assert_raises(ValueError, Constraints, min_weights=pd.Series([0.5, 0.0], index=pair),
                  max_weights=pd.Series([0.5 - 2e-10, 1.0], index=pair))
    Constraints(min_weights=pd.Series([0.5, 0.0], index=pair),
                max_weights=pd.Series([0.5 - 5e-11, 1.0], index=pair))
    Constraints(min_weights=pd.Series([0.5, 0.0], index=pair),  # unequal indexes: not checked
                max_weights=pd.Series([1.0, 0.4], index=["B", "A"]))
    assert_raises(ValueError, Constraints, min_weights=pd.Series([-2e-10, 0.0], index=pair))
    Constraints(min_weights=pd.Series([-5e-11, 0.0], index=pair))
    Constraints(is_long_only=False, min_weights=pd.Series([-0.5, 0.0], index=pair))
    only_a = pd.DataFrame({"G": [1.0, 0.0]}, index=pair)
    both = pd.DataFrame({"G": [1.0, 1.0]}, index=pair)

    def one_group(loadings, lower=None, upper=None):
        """A one-group allocation block with optional bounds."""
        return GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=None if lower is None else pd.Series({"G": lower}),
            group_max_allocation=None if upper is None else pd.Series({"G": upper}))

    # 1. Caps cannot reach a floor: 0.4998 < 0.5 is rejected, 0.49995 is inside the tolerance.
    assert_raises(ValueError, Constraints, max_weights=pd.Series([0.4998, 1.0], index=pair),
                  group_lower_upper_constraints=one_group(only_a, lower=0.5))
    Constraints(max_weights=pd.Series([0.49995, 1.0], index=pair),
                group_lower_upper_constraints=one_group(only_a, lower=0.5))
    # 2. Floors already exceed a ceiling; 3. one floor alone does, although the sum does not.
    assert_raises(ValueError, Constraints, min_weights=pd.Series([0.3, 0.3], index=pair),
                  group_lower_upper_constraints=one_group(both, upper=0.5))
    assert_raises(ValueError, Constraints, is_long_only=False,
                  min_weights=pd.Series([0.6, -0.3], index=pair),
                  group_lower_upper_constraints=one_group(both, upper=0.5))

    returns = pd.Series({"A": 0.08, "B": 0.02})
    return_floor = Constraints(asset_returns=returns, target_return=0.05)

    # [0.50, 0.50] earns 0.05 and passes; [0.40, 0.60] earns 0.044 and misses by 0.006.
    assert round(0.08 * 0.50 + 0.02 * 0.50, 12) == 0.05
    assert round(0.08 * 0.40 + 0.02 * 0.60, 12) == 0.044
    passing = record_of([0.50, 0.50], return_floor, 'target_return')
    failing = record_of([0.40, 0.60], return_floor, 'target_return')
    assert passing.passed and abs(passing.actual - 0.05) < 1e-15
    assert not failing.passed and abs(failing.violation - 0.006) < 1e-12
    # A target without asset_returns raises at compilation; the floor stays hard in utility
    # mode, in the compiled rows and in the residuals.
    assert_raises(ValueError, compiled_rows, constraints=Constraints(target_return=0.05), n=2)
    utility_floor = return_floor.copy(constraint_enforcement_type=utility_type,
                                      tre_utility_weight=None)
    assert len(compiled_rows(utility_floor, 2, utility=True)) == len(
        compiled_rows(return_floor, 2)) == 3
    assert record_of([0.40, 0.60], utility_floor, 'target_return').hard

    # Portfolio volatility: diag(0.04, 0.01) at [0.5, 0.5] is sqrt(0.0125) = 0.1118, under 0.12.
    diagonal = [[0.04, 0.0], [0.0, 0.01]]
    variance = 0.5 ** 2 * 0.04 + 0.5 ** 2 * 0.01
    assert abs(variance - 0.0125) < 1e-15 and round(variance ** 0.5, 4) == 0.1118
    vol_cap = Constraints(max_target_portfolio_vol_an=0.12, tre_utility_weight=None)
    volatility = record_of([0.5, 0.5], vol_cap, 'portfolio_volatility', covar=diagonal)
    assert volatility.passed and abs(volatility.actual - variance ** 0.5) < 1e-15
    # The _an suffix annualises nothing: a monthly covariance is held to the same 0.12 limit.
    monthly = record_of([0.5, 0.5], vol_cap, 'portfolio_volatility',
                        covar=[[0.04 / 12, 0.0], [0.0, 0.01 / 12]])
    assert monthly.upper == 0.12 and abs(monthly.actual - (variance / 12) ** 0.5) < 1e-15
    # A factorization takes precedence over a conflicting covariance, in the compiled row and
    # in the residual analytics.
    assert factorization_takes_precedence(vol_cap, diagonal, cap=0.12)
    # The cap is compiled whatever the enum says; the generic utility builder omits it.
    utility_cap = vol_cap.copy(constraint_enforcement_type=utility_type)
    assert len(compiled_rows(utility_cap, 2, covar=diagonal)) == 3
    assert len(compiled_rows(vol_cap, 2, covar=diagonal)) == 3
    assert len(compiled_rows(utility_cap, 2, covar=diagonal, utility=True)) == 2

    from optimalportfolios.optimization.constraints import LinearConstraints

    characteristics = LinearConstraints(
        loadings=pd.DataFrame(
            {"carry": [0.04, 0.01, -0.02], "duration": [0.0, 6.0, 0.0]},
            index=["Equity", "Bond", "Gold"],
        ),
        lower=pd.Series({"carry": 0.015}),
        upper=pd.Series({"duration": 3.0}),
    )
    signed = Constraints(linear_constraints=characteristics)

    # Carry 0.04 * 0.45 + 0.01 * 0.40 - 0.02 * 0.15 = 0.019 and duration 6 * 0.40 = 2.4 pass;
    # [0.20, 0.45, 0.35] earns carry 0.0055, 0.0095 short; [0.30, 0.60, 0.10] has duration 3.6.
    for weights, carry, duration in (([0.45, 0.40, 0.15], 0.019, 2.4),
                                     ([0.20, 0.45, 0.35], 0.0055, 2.7),
                                     ([0.30, 0.60, 0.10], 0.016, 3.6)):
        assert abs(0.04 * weights[0] + 0.01 * weights[1] - 0.02 * weights[2] - carry) < 1e-15
        assert abs(6.0 * weights[1] - duration) < 1e-14
        assert abs(record_of(weights, signed, 'linear', 'carry').actual - carry) < 1e-15
        assert abs(record_of(weights, signed, 'linear', 'duration').actual - duration) < 1e-14
    assert record_of([0.45, 0.40, 0.15], signed, 'linear', 'carry').passed
    assert record_of([0.45, 0.40, 0.15], signed, 'linear', 'duration').passed
    short = record_of([0.20, 0.45, 0.35], signed, 'linear', 'carry')
    assert not short.passed and abs(short.violation - 0.0095) < 1e-15
    capped = record_of([0.30, 0.60, 0.10], signed, 'linear', 'duration')
    assert not capped.passed and abs(capped.violation - 0.6) < 1e-14
    assert (short.lower, short.upper, capped.lower, capped.upper) == (0.015, None, None, 3.0)
    # One row per bounded side: long-only and full investment plus two, hard in utility mode as
    # well; SciPy adds two callbacks; the risk-budgeting matrix helper refuses the rows.
    assert len(compiled_rows(Constraints(), 3)) == 2 and len(compiled_rows(signed, 3)) == 4
    utility_signed = signed.copy(constraint_enforcement_type=utility_type,
                                 tre_utility_weight=None)
    assert len(compiled_rows(utility_signed, 3, utility=True)) == 4
    assert record_of([0.20, 0.45, 0.35], utility_signed, 'linear', 'carry').hard
    assert scipy_callback_count(signed, 3) == scipy_callback_count(Constraints(), 3) + 2
    assert_raises(ValueError, pyrb_rows, constraints=signed, n=3)
    # Both bounds are scaled by the Charnes-Cooper scale k: y = k w meets them at every k > 0.
    assert charnes_cooper_rows_hold(signed, feasible=[0.45, 0.40, 0.15],
                                    infeasible=[0.30, 0.60, 0.10])
    # A policy with neither side emits nothing, and NaN is unbounded.
    unbounded = LinearConstraints(loadings=characteristics.loadings,
                                  lower=pd.Series({"carry": float("nan")}))
    assert list(unbounded.iter_bounds()) == []
    # Alignment reorders, rejects a universe asset without loadings and refuses to drop Gold's
    # carry coefficient; a duration-only policy may drop Equity, whose coefficient is zero.
    reordered = characteristics.update(["Bond", "Equity", "Gold"])
    assert reordered.loadings.index.tolist() == ["Bond", "Equity", "Gold"]
    assert_raises(ValueError, characteristics.update,
                  valid_tickers=["Equity", "Bond", "Gold", "Cash"])
    assert_raises(ValueError, characteristics.update, valid_tickers=["Equity", "Bond"])
    duration_only = LinearConstraints(loadings=characteristics.loadings[["duration"]],
                                      upper=pd.Series({"duration": 3.0}))
    assert duration_only.update(["Bond", "Gold"]).loadings.index.tolist() == ["Bond", "Gold"]
    # Caps [0.30, 0.30, 0.40] allow at most 0.04 * 0.30 + 0.01 * 0.30 = 0.015 carry, since Gold's
    # coefficient is negative and long-only floors it at zero; a 0.02 floor is rejected.
    assert abs(0.04 * 0.30 + 0.01 * 0.30 - 0.015) < 1e-15
    caps = pd.Series([0.30, 0.30, 0.40], index=ASSETS)
    assert_raises(ValueError, Constraints, max_weights=caps,
                  linear_constraints=characteristics.copy(lower=pd.Series({"carry": 0.02})))
    Constraints(max_weights=caps, linear_constraints=characteristics)

    group_tre = GroupTrackingErrorConstraint(
        group_loadings=pd.DataFrame(
            {
                "Growth": [1.0, 0.0, 0.0],
                "Defensive": [0.0, 1.0, 1.0],
            },
            index=["Equity", "Bond", "Gold"],
        ),
        group_tre_vols=pd.Series({"Growth": 0.05, "Defensive": 0.05}),
    )

    # Group TE masks the active vector and keeps the cross-covariance inside the mask; it is
    # not a group's weight times total TE.
    reference_benchmark = pd.Series(BENCHMARK, index=ASSETS)
    masked = Constraints(benchmark_weights=reference_benchmark,
                         group_tracking_error_constraint=group_tre)
    tilted = [0.45, 0.35, 0.20]
    defensive_te = record_of(tilted, masked, 'group_tracking_error', 'Defensive',
                             covar=COVARIANCE)
    assert abs(defensive_te.actual - explicit_tracking_error(
        tilted, BENCHMARK, COVARIANCE, DEFENSIVE)) < 1e-15
    assert round(defensive_te.actual, 12) == 0.007  # sqrt(0.000049), with the -0.000003 term
    assert abs(defensive_te.actual - (0.05 ** 2 * (0.0064 + 0.0144)) ** 0.5) > 2e-4
    assert abs(defensive_te.actual - 0.55 * explicit_tracking_error(
        tilted, BENCHMARK, COVARIANCE)) > 1e-3
    # One of the two series is required; missing coverage warns, then raises at compilation.
    assert_raises(ValueError, GroupTrackingErrorConstraint,
                  group_loadings=group_tre.group_loadings)
    partial, messages = warned(GroupTrackingErrorConstraint,
                               group_loadings=group_tre.group_loadings,
                               group_tre_vols=pd.Series({"Growth": 0.05}))
    assert len(messages) == 1 and 'Defensive' in messages[0]
    assert_raises(KeyError, compiled_rows, n=3, covar=COVARIANCE, constraints=Constraints(
        benchmark_weights=reference_benchmark, group_tracking_error_constraint=partial))
    # A NaN utility coefficient skips that group's penalty: with both NaN there is none.
    skipped = GroupTrackingErrorConstraint(
        group_loadings=group_tre.group_loadings,
        group_tre_utility_weights=pd.Series({"Growth": float("nan"),
                                             "Defensive": float("nan")}))
    penalty, _ = warned(group_tre_penalty, constraint=skipped, benchmark=reference_benchmark)
    assert penalty is None
    # Forced mode emits total and group caps together: one row plus one per group.
    total_te = Constraints(benchmark_weights=reference_benchmark,
                           tracking_err_vol_constraint=0.06)
    assert len(compiled_rows(total_te.copy(group_tracking_error_constraint=group_tre), 3,
                             covar=COVARIANCE)) == len(compiled_rows(total_te, 3,
                                                                     covar=COVARIANCE)) + 2

    group_allocation = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame(
            {
                "Growth": [1.0, 0.0, 0.0],
                "Defensive": [0.0, 1.0, 1.0],
            },
            index=["Equity", "Bond", "Gold"],
        ),
        group_min_allocation=pd.Series({"Growth": 0.30, "Defensive": 0.45}),
        group_max_allocation=pd.Series({"Growth": 0.55, "Defensive": 0.70}),
    )

    # At [0.45, 0.40, 0.15] Growth holds 0.45 and Defensive 0.55, inside both bands.
    point = pd.Series([0.45, 0.40, 0.15], index=["Equity", "Bond", "Gold"])
    assert (group_allocation.group_loadings.T @ point).round(12).tolist() == [0.45, 0.55]
    allocation = Constraints(group_lower_upper_constraints=group_allocation)
    held = [record for record in evaluate_constraint_residuals(point.to_numpy(), allocation)
            if record.constraint_type == 'group_weight']
    assert [round(record.actual, 12) for record in held] == [0.45, 0.55]
    assert all(record.passed for record in held)
    assert len(compiled_rows(allocation, 3)) == 2 + 4  # exposure rows and two sides per group
    # All-zero and all-missing columns are dropped; a column close to zero is kept but emits
    # no row; a missing bound is warned about, stored as NaN and skipped.
    loadings = group_allocation.group_loadings.assign(
        Empty=0.0, Missing=float("nan"), Tiny=[1e-12, 0.0, 0.0])
    widened, messages = warned(
        GroupLowerUpperConstraints, group_loadings=loadings,
        group_min_allocation=pd.Series({"Growth": 0.30, "Tiny": 0.0}),
        group_max_allocation=pd.Series({"Growth": 0.55, "Defensive": 0.70, "Tiny": 0.1}))
    assert widened.group_loadings.columns.tolist() == ["Growth", "Defensive", "Tiny"]
    assert len(messages) == 1 and 'Defensive' in messages[0]
    assert pd.isna(widened.group_min_allocation["Defensive"])
    assert len(compiled_rows(Constraints(group_lower_upper_constraints=widened), 3)) == 2 + 3
    # A reversed band is not rejected when no box check exposes it.
    Constraints(group_lower_upper_constraints=GroupLowerUpperConstraints(
        group_loadings=group_allocation.group_loadings,
        group_min_allocation=pd.Series({"Growth": 0.60, "Defensive": 0.0}),
        group_max_allocation=pd.Series({"Growth": 0.40, "Defensive": 1.0})))
    # Merging renames overlapping groups _1 and _2, fills missing loadings with zero and leaves
    # a missing side as NaN.
    from optimalportfolios.optimization.constraints import merge_group_lower_upper_constraints
    second = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Growth": [1.0, 1.0], "Cash": [0.0, 1.0]},
                                    index=["Equity", "Cash"]),
        group_min_allocation=None,
        group_max_allocation=pd.Series({"Growth": 0.60, "Cash": 0.10}))
    merged = merge_group_lower_upper_constraints(group_allocation, second)
    assert sorted(merged.group_loadings.columns) == ["Cash", "Defensive", "Growth_1", "Growth_2"]
    assert merged.group_loadings.loc["Cash", "Growth_1"] == 0.0
    assert merged.group_loadings.loc["Bond", "Growth_2"] == 0.0
    assert merged.group_min_allocation[["Growth_2", "Cash"]].isna().all()
    assert merged.group_max_allocation.notna().all()

    sector_deviation = BenchmarkDeviationConstraints(
        factor_loading_mat=pd.DataFrame(
            {"Risk assets": [1.0, 0.0, 1.0]},
            index=["Equity", "Bond", "Gold"],
        ),
        factor_max_deviation=pd.Series({"Risk assets": 0.08}),
    )

    style_deviation = BenchmarkDeviationConstraints(
        factor_loading_mat=pd.DataFrame(
            {"Inflation": [0.5, -0.5, 1.0]},
            index=["Equity", "Bond", "Gold"],
        ),
        factor_max_deviation=pd.Series({"Inflation": 0.12}),
    )

    # Benchmark [0.45, 0.40, 0.15], portfolio [0.50, 0.35, 0.15]: active Risk-assets exposure
    # (0.50 + 0.15) - (0.45 + 0.15) = 0.05 passes 0.08. Both families stay hard in utility mode.
    assert round((0.50 + 0.15) - (0.45 + 0.15), 12) == 0.05
    deviations = Constraints(
        benchmark_weights=reference_benchmark, sector_deviation_constraints=sector_deviation,
        style_deviation_constraints=style_deviation, constraint_enforcement_type=utility_type)
    sector = record_of([0.50, 0.35, 0.15], deviations, 'sector_deviation')
    assert sector.hard and sector.passed and sector.upper == 0.08
    assert abs(sector.actual - 0.05) < 1e-15
    assert record_of([0.50, 0.35, 0.15], deviations, 'style_deviation').hard
    assert len(compiled_rows(deviations, 3, covar=COVARIANCE, utility=True)) == 2 + 2
    # A bound label without a loading column warns at construction and fails at compilation.
    unmatched, messages = warned(BenchmarkDeviationConstraints,
                                 factor_loading_mat=sector_deviation.factor_loading_mat,
                                 factor_max_deviation=pd.Series({"Missing": 0.1}))
    assert len(messages) == 1 and 'Missing' in messages[0]
    assert_raises(KeyError, compiled_rows, n=3, constraints=Constraints(
        benchmark_weights=reference_benchmark, sector_deviation_constraints=unmatched))

    group_turnover = GroupTurnoverConstraint(
        group_loadings=pd.DataFrame(
            {
                "Growth": [1.0, 0.0, 0.0],
                "Defensive": [0.0, 1.0, 1.0],
            },
            index=["Equity", "Bond", "Gold"],
        ),
        group_max_turnover=pd.Series({"Growth": 0.15, "Defensive": 0.15}),
    )

    # Moving 10% from A to B is full L1 turnover 0.20; costs [2, 1] make it 0.30.
    start = pd.Series([0.60, 0.40], index=pair)
    assert round(abs(0.50 - 0.60) + abs(0.50 - 0.40), 12) == 0.20
    assert round(2 * abs(0.50 - 0.60) + 1 * abs(0.50 - 0.40), 12) == 0.30
    plain = record_of([0.50, 0.50], Constraints(weights_0=start, turnover_constraint=1.0),
                      'turnover')
    costed = record_of([0.50, 0.50], Constraints(
        weights_0=start, turnover_constraint=1.0,
        turnover_costs=pd.Series([2.0, 1.0], index=pair)), 'turnover')
    assert abs(plain.actual - 0.20) < 1e-15 and abs(costed.actual - 0.30) < 1e-15
    # Without weights_0 both turnover rows are skipped with a debug log.
    rows, records = logged(compiled_rows, n=3, constraints=Constraints(
        turnover_constraint=0.1, group_turnover_constraint=group_turnover))
    assert len(rows) == 2 and [record.levelno for record in records] == [logging.DEBUG] * 2
    assert all(record.getMessage() == 'turnover constraint skipped because weights_0 is absent'
               for record in records)
    # Group turnover uses its loadings, not turnover_costs: Bond's 0.05 sale counts 0.05, not
    # 0.025. Buys and sells inside a group do not net, and overlapping groups count a trade
    # twice.
    trading = Constraints(weights_0=pd.Series(CURRENT, index=ASSETS),
                          turnover_costs=pd.Series(TURNOVER_COSTS, index=ASSETS),
                          turnover_constraint=1.0, group_turnover_constraint=group_turnover)
    assert abs(record_of([0.45, 0.40, 0.15], trading, 'group_turnover', 'Defensive').actual
               - 0.05) < 1e-12
    assert abs(record_of([0.40, 0.40, 0.20], trading, 'group_turnover', 'Defensive').actual
               - 0.10) < 1e-12
    overlapping = GroupTurnoverConstraint(
        group_loadings=group_turnover.group_loadings.assign(All=1.0),
        group_max_turnover=pd.Series({"Growth": 1.0, "Defensive": 1.0, "All": 1.0}))
    counted = [record.actual for record in evaluate_constraint_residuals(
        [0.45, 0.40, 0.15], trading.copy(group_turnover_constraint=overlapping))
        if record.constraint_type == 'group_turnover']
    assert abs(sum(counted) - 0.20) < 1e-12  # Equity's 0.05 buy counts in Growth and in All
    # One of the two series is required; forced mode emits the group rows and the total row.
    assert_raises(ValueError, GroupTurnoverConstraint,
                  group_loadings=group_turnover.group_loadings)
    assert len(compiled_rows(trading, 3)) == 2 + 2 + 1

    from optimalportfolios.optimization.constraints import (
        compute_benchmark_beta_loadings_from_covar,
    )

    assets = pd.Index(["Equity", "Bond", "Gold"])
    covar = pd.DataFrame(
        [
            [0.0324, 0.0018, 0.0024],
            [0.0018, 0.0064, 0.0006],
            [0.0024, 0.0006, 0.0144],
        ],
        index=assets,
        columns=assets,
    )
    benchmark = pd.Series([0.45, 0.40, 0.15], index=assets)

    beta_loadings = compute_benchmark_beta_loadings_from_covar(
        covar=covar,
        benchmark_weights=benchmark,
        asset_tickers=assets.tolist(),
    )
    beta_constraint = BenchmarkBetaConstraint(
        beta_min=0.85,
        beta_max=1.15,
    ).with_loadings(beta_loadings)

    # The inputs are the module constants; h = Sigma b / (b' Sigma b), so h @ b is one.
    assert assets.tolist() == ASSETS and covar.to_numpy().tolist() == COVARIANCE
    assert benchmark.tolist() == BENCHMARK
    explicit_beta = covar @ benchmark / (benchmark @ covar @ benchmark)
    assert (beta_loadings - explicit_beta).abs().max() < 1e-15
    assert abs(beta_loadings @ benchmark - 1.0) < 1e-12
    # Benchmark variance must be positive; a range needs one side; compiling needs loadings.
    assert_raises(ValueError, compute_benchmark_beta_loadings_from_covar, covar=covar,
                  benchmark_weights=0.0 * benchmark, asset_tickers=ASSETS)
    assert_raises(ValueError, BenchmarkBetaConstraint)
    unloaded = Constraints(benchmark_beta_constraint=BenchmarkBetaConstraint(
        beta_min=0.85, beta_max=1.15))
    assert_raises(ValueError, compiled_rows, constraints=unloaded, n=3)
    assert len(compiled_rows(Constraints(benchmark_beta_constraint=beta_constraint), 3)) == 4
    # Beta is absolute portfolio beta h @ w and stays hard in utility mode.
    beta_record = record_of([0.50, 0.35, 0.15], Constraints(
        benchmark_beta_constraint=beta_constraint, constraint_enforcement_type=utility_type),
        'benchmark_beta')
    assert beta_record.hard and abs(beta_record.actual - beta_loadings @ pd.Series(
        [0.50, 0.35, 0.15], index=assets)) < 1e-15
    # The factor-model variant: loadings B F b / (b' F b + benchmark idiosyncratic variance).
    from optimalportfolios.optimization.constraints import compute_benchmark_beta_loadings
    market = pd.DataFrame({"Market": [1.2, 0.2, 0.5]}, index=assets)
    factor_loadings = compute_benchmark_beta_loadings(
        asset_betas=market, benchmark_betas=pd.Series({"Market": 1.0}),
        factor_covar=pd.DataFrame([[0.03]], index=["Market"], columns=["Market"]),
        benchmark_idio_var=0.001)
    assert (factor_loadings - market["Market"] * 0.03 / 0.031).abs().max() < 1e-15

    # Utility mode. The total tracking-error penalty needs benchmark_weights even without
    # alphas; tre_utility_weight=None compiles without a benchmark.
    assert_raises(ValueError, compiled_rows, n=3, covar=COVARIANCE, utility=True,
                  constraints=Constraints(constraint_enforcement_type=utility_type))
    assert len(compiled_rows(Constraints(constraint_enforcement_type=utility_type,
                                         tre_utility_weight=None), 3, utility=True)) == 2
    # A group object keeps one meaning: hard caps alone are not penalties, for tracking error
    # and for turnover, and utility weights alone are not hard rows.
    assert_raises(ValueError, group_tre_penalty, constraint=group_tre,
                  benchmark=reference_benchmark)
    tre_weights = pd.Series({"Growth": 5.0, "Defensive": 5.0})
    assert_raises(AttributeError, compiled_rows, n=3, covar=COVARIANCE, constraints=Constraints(
        benchmark_weights=reference_benchmark,
        group_tracking_error_constraint=GroupTrackingErrorConstraint(
            group_loadings=group_tre.group_loadings, group_tre_utility_weights=tre_weights)))
    assert_raises(ValueError, compiled_rows, n=3, covar=COVARIANCE, utility=True,
                  constraints=trading.copy(benchmark_weights=reference_benchmark,
                                           constraint_enforcement_type=utility_type))
    # The soft-tracking-error alpha solve re-adds a configured turnover cap as a hard row and,
    # without a cap, keeps the total turnover penalty: a weight of 1.0 stops all trading.
    assert soft_tracking_error_turnover(1.0) < 1e-6
    assert soft_tracking_error_turnover(0.0) > 0.5
    assert abs(soft_tracking_error_turnover(1.0, turnover_constraint=0.05) - 0.05) < 1e-6

    import cvxpy as cvx
    import numpy as np
    import pandas as pd

    from optimalportfolios.optimization.constraints import (
        BenchmarkBetaConstraint,
        BenchmarkDeviationConstraints,
        ConstraintEnforcementType,
        Constraints,
        GroupLowerUpperConstraints,
        GroupTrackingErrorConstraint,
        GroupTurnoverConstraint,
        compute_benchmark_beta_loadings_from_covar,
        evaluate_constraint_residuals,
    )

    assets = pd.Index(["Equity", "Bond", "Gold"], name="asset")
    covar = pd.DataFrame(
        [
            [0.0324, 0.0018, 0.0024],
            [0.0018, 0.0064, 0.0006],
            [0.0024, 0.0006, 0.0144],
        ],
        index=assets,
        columns=assets,
    )
    benchmark = pd.Series([0.45, 0.40, 0.15], index=assets)
    current = pd.Series([0.40, 0.45, 0.15], index=assets)
    expected_returns = pd.Series([0.070, 0.035, 0.040], index=assets)

    groups = pd.DataFrame(
        {
            "Growth": [1.0, 0.0, 0.0],
            "Defensive": [0.0, 1.0, 1.0],
        },
        index=assets,
    )
    beta_loadings = compute_benchmark_beta_loadings_from_covar(
        covar=covar,
        benchmark_weights=benchmark,
        asset_tickers=assets.tolist(),
    )

    constraints = Constraints(
        min_weights=pd.Series([0.20, 0.20, 0.05], index=assets),
        max_weights=pd.Series([0.60, 0.65, 0.25], index=assets),
        min_exposure=1.0,
        max_exposure=1.0,
        benchmark_weights=benchmark,
        tracking_err_vol_constraint=0.06,
        weights_0=current,
        turnover_constraint=0.25,
        turnover_costs=pd.Series([1.0, 0.5, 2.0], index=assets),
        target_return=0.045,
        asset_returns=expected_returns,
        max_target_portfolio_vol_an=0.13,
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=groups,
            group_min_allocation=pd.Series({"Growth": 0.30, "Defensive": 0.45}),
            group_max_allocation=pd.Series({"Growth": 0.55, "Defensive": 0.70}),
        ),
        group_tracking_error_constraint=GroupTrackingErrorConstraint(
            group_loadings=groups,
            group_tre_vols=pd.Series({"Growth": 0.05, "Defensive": 0.05}),
            group_tre_utility_weights=pd.Series({"Growth": 5.0, "Defensive": 5.0}),
        ),
        group_turnover_constraint=GroupTurnoverConstraint(
            group_loadings=groups,
            group_max_turnover=pd.Series({"Growth": 0.15, "Defensive": 0.15}),
            group_turnover_utility_weights=pd.Series({"Growth": 0.02, "Defensive": 0.02}),
        ),
        sector_deviation_constraints=BenchmarkDeviationConstraints(
            factor_loading_mat=pd.DataFrame(
                {"Risk assets": [1.0, 0.0, 1.0]}, index=assets
            ),
            factor_max_deviation=pd.Series({"Risk assets": 0.08}),
        ),
        style_deviation_constraints=BenchmarkDeviationConstraints(
            factor_loading_mat=pd.DataFrame(
                {"Inflation": [0.5, -0.5, 1.0]}, index=assets
            ),
            factor_max_deviation=pd.Series({"Inflation": 0.12}),
        ),
        benchmark_beta_constraint=BenchmarkBetaConstraint(
            beta_min=0.85,
            beta_max=1.15,
            beta_loadings=beta_loadings,
        ),
    )

    w = cvx.Variable(len(assets))
    rows = constraints.set_cvx_all_constraints(
        w=w,
        covar=cvx.psd_wrap(covar.to_numpy()),
    )
    problem = cvx.Problem(cvx.Maximize(expected_returns.to_numpy() @ w), rows)
    problem.solve(solver="CLARABEL")

    solution = pd.Series(w.value, index=assets)
    print(solution.round(6))

    # The block's inputs are the module constants.
    assert covar.to_numpy().tolist() == COVARIANCE and benchmark.tolist() == BENCHMARK
    assert current.tolist() == CURRENT and expected_returns.tolist() == EXPECTED_RETURNS
    assert constraints.min_weights.tolist() == MIN_WEIGHTS
    assert constraints.max_weights.tolist() == MAX_WEIGHTS
    assert constraints.turnover_costs.tolist() == TURNOVER_COSTS
    assert groups["Growth"].tolist() == GROWTH and groups["Defensive"].tolist() == DEFENSIVE
    assert problem.status == "optimal"
    # Independent certificate: full investment, the upper Risk-assets deviation
    # w_E + w_G <= 0.60 + 0.08 = 0.68, and the affine minorant of weighted L1 turnover,
    # w_E - w_B / 2 - 2 w_G <= 0.25 + (0.40 - 0.45 / 2 - 2 * 0.15) = 0.125, valid for every
    # allocation. Solving the three rows gives the allocation; their multipliers 0.04, 0.02
    # and 0.01 reproduce the expected returns and bound them by 0.05485.
    binding = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, -0.5, -2.0]])
    rhs = np.array([1.0, 0.45 + 0.15 + 0.08,
                    0.25 + np.array([1.0, -0.5, -2.0]) @ np.array(CURRENT)])
    np.testing.assert_allclose(rhs, [1.0, 0.68, 0.125], rtol=0.0, atol=1e-15)
    reference = np.linalg.solve(binding, rhs)
    np.testing.assert_allclose(solution, reference, rtol=0.0, atol=1e-6)
    multipliers = np.linalg.solve(binding.T, expected_returns.to_numpy())
    np.testing.assert_allclose(multipliers, [0.04, 0.02, 0.01], rtol=0.0, atol=1e-12)
    assert (multipliers[1:] >= 0.0).all()
    upper_bound = float(multipliers @ rhs)
    assert abs(upper_bound - 0.05485) < 1e-12
    assert abs(expected_returns.to_numpy() @ solution.to_numpy() - upper_bound) < 1e-7
    # The page's displayed allocation; Equity is bought, Bond and Gold are sold.
    np.testing.assert_allclose(solution, [0.548333, 0.320000, 0.131667], rtol=0.0, atol=1e-6)
    assert np.sign(solution - current).tolist() == [1.0, -1.0, -1.0]

    residuals = evaluate_constraint_residuals(
        solution.to_numpy(),
        constraints,
        covar=covar.to_numpy(),
    )
    hard_breaches = [r for r in residuals if r.hard and not r.passed]
    assert hard_breaches == []

    # Every applicable row is evaluated: 21 records, all hard in forced mode. Exposure, the
    # Risk-assets deviation and weighted turnover bind, as the certificate says.
    kinds = pd.Series([record.constraint_type for record in residuals]).value_counts()
    assert kinds.sum() == 21 and kinds['instrument_weight'] == 6
    assert all(record.hard for record in residuals)
    by_kind = {(record.constraint_type, record.name): record for record in residuals}
    assert abs(by_kind['exposure', 'total'].actual - 1.0) < 1e-8
    assert abs(by_kind['sector_deviation', 'Risk assets'].actual - 0.08) < 1e-7
    assert abs(by_kind['turnover', 'total_l1'].actual - 0.25) < 1e-7
    assert abs(by_kind['tracking_error', 'total'].actual - explicit_tracking_error(
        solution, benchmark, covar)) < 1e-15

    utility_constraints = constraints.copy(
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
    )

    w_utility = cvx.Variable(len(assets))
    utility, hard_rows = utility_constraints.set_cvx_utility_objective_constraints(
        w=w_utility,
        alphas=np.array([0.030, 0.005, 0.010]),
        covar=cvx.psd_wrap(covar.to_numpy()),
    )
    problem = cvx.Problem(cvx.Maximize(utility), hard_rows)
    problem.solve(solver="CLARABEL")

    utility_solution = pd.Series(w_utility.value, index=assets)
    print(utility_solution.round(6))

    # copy() deep-copies the pandas state, and the dataclass is frozen.
    assert utility_constraints.benchmark_weights is not constraints.benchmark_weights
    assert utility_constraints.benchmark_weights.equals(constraints.benchmark_weights)
    assert_raises(dataclasses.FrozenInstanceError,
                  lambda: setattr(constraints, 'turnover_constraint', 0.30))
    # The page's utility allocation, and an independent solve of the same problem written
    # from raw arrays with the two group penalties.
    assert problem.status == "optimal"
    np.testing.assert_allclose(utility_solution, [0.411341, 0.438660, 0.150000],
                               rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(utility_solution, utility_reference(explicit_beta.to_numpy()),
                               rtol=0.0, atol=1e-6)
    # The group penalties replace the total ones: the total weights do not move the solve,
    # and neither do the group objects' hard caps.
    alphas = np.array(ALPHAS)
    np.testing.assert_allclose(solve_utility(utility_constraints.copy(
        tre_utility_weight=1000.0, turnover_utility_weight=1000.0), alphas, covar),
        utility_solution, rtol=0.0, atol=1e-6)
    tiny = pd.Series({"Growth": 1e-6, "Defensive": 1e-6})
    np.testing.assert_allclose(solve_utility(utility_constraints.copy(
        group_tracking_error_constraint=dataclasses.replace(
            constraints.group_tracking_error_constraint, group_tre_vols=tiny),
        group_turnover_constraint=dataclasses.replace(
            constraints.group_turnover_constraint, group_max_turnover=tiny)), alphas, covar),
        utility_solution, rtol=0.0, atol=1e-6)
    # Pitfall: the enum does not dispatch. set_cvx_all_constraints() compiles the same hard
    # rows under UTILITY_CONSTRAINTS, and the forced example solves to the same allocation.
    assert len(compiled_rows(utility_constraints, 3, covar=covar)) == len(rows) == 20
    np.testing.assert_allclose(solve_forced(utility_constraints, expected_returns, covar),
                               solution, rtol=0.0, atol=1e-7)

    soft_spec = Constraints(
        min_weights=pd.Series([0.20, 0.20, 0.05], index=assets),
        max_weights=pd.Series([0.60, 0.65, 0.25], index=assets),
        benchmark_weights=benchmark,
        tracking_err_vol_constraint=0.01,
        weights_0=current,
        turnover_constraint=0.05,
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
    )
    candidate = np.array([0.35, 0.40, 0.25])
    records = evaluate_constraint_residuals(
        candidate, soft_spec, covar=covar.to_numpy()
    )
    soft_violations = [r for r in records if not r.hard and r.violation > 0]
    [(r.constraint_type, round(r.violation, 6), r.passed) for r in soft_violations]

    # The page's result. By hand: turnover 0.05 + 0.05 + 0.10 = 0.20 exceeds 0.05 by 0.15, and
    # active weights [-0.10, 0, 0.10] give sqrt(0.00042) = 0.020494, 0.010494 over 0.01.
    assert [(r.constraint_type, round(r.violation, 6), r.passed) for r in soft_violations] == [
        ("turnover", 0.15, True), ("tracking_error", 0.010494, True)]
    assert abs(np.abs(candidate - current).sum() - 0.05 - 0.15) < 1e-15
    soft_te = explicit_tracking_error(candidate, benchmark, covar)
    assert abs(soft_te ** 2 - 0.00042) < 1e-15 and round(soft_te - 0.01, 6) == 0.010494
    assert all(not r.hard and r.passed for r in soft_violations)
    # Under forced enforcement the same two records are hard breaches.
    forced_spec = soft_spec.copy(
        constraint_enforcement_type=ConstraintEnforcementType.FORCED_CONSTRAINTS)
    assert [r.constraint_type for r in evaluate_constraint_residuals(
        candidate, forced_spec, covar=covar.to_numpy()) if r.hard and not r.passed] == [
        'turnover', 'tracking_error']

    # What changes when a limit becomes a penalty (the figure). Maximise ALPHAS @ (w - b),
    # long-only and fully invested, with tracking error at most 2% as a hard row, then with
    # tre_utility_weight penalties instead.
    path, shadow_price = tracking_error_path(TRACKING_ERROR_LIMIT, PENALTY_WEIGHTS)
    hard, penalties = path.loc['hard limit'], path.iloc[1:]
    # Closed form with the budget row alone, since no weight reaches zero:
    # d(l) = Sigma^-1 (a - k 1) / (2 l), so tracking error is TE(1) / l.
    unit = closed_form_active_weights(alphas, covar, 1.0)
    unit_te = explicit_tracking_error(unit, 0.0 * unit, covar)
    for weight in PENALTY_WEIGHTS:
        expected = benchmark.to_numpy() + unit / weight
        assert (expected > 0.0).all()
        np.testing.assert_allclose(penalties.loc[f'penalty {weight:g}', ASSETS], expected,
                                   rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(penalties['tracking_error'] * penalties['penalty_weight'],
                               unit_te, rtol=1e-5)
    # The hard solve sits on the limit along the same direction, d = 0.02 d(1) / TE(1).
    assert abs(hard['tracking_error'] - TRACKING_ERROR_LIMIT) < 1e-8
    np.testing.assert_allclose(hard[ASSETS], benchmark.to_numpy()
                               + TRACKING_ERROR_LIMIT / unit_te * unit, rtol=0.0, atol=1e-5)
    # The excess over the limit falls monotonically with the weight and vanishes from 5 on,
    # where the limit is slack and active return is given up.
    excess = penalties['excess_over_limit'].to_numpy()
    assert (np.diff(excess) <= 0.0).all() and (excess[:3] > 0.0).all()
    assert (excess[3:] == 0.0).all() and (penalties['tracking_error'].iloc[3:] < 0.02).all()
    assert (penalties['active_return'].iloc[3:] < hard['active_return']).all()
    assert (penalties['active_return'].iloc[:3] > hard['active_return']).all()
    # Insight: every solve earns 0.133 of active return per unit of tracking error, and the
    # penalty reproduces the hard solve at one weight only, the dual value of the limit's row:
    # TE(1) / 0.02 = 3.33 in closed form.
    assert np.allclose(path['active_return'] / path['tracking_error'], unit @ alphas / unit_te,
                       rtol=1e-5, atol=0.0)
    assert round(unit @ alphas / unit_te, 3) == 0.133
    assert abs(shadow_price / (unit_te / TRACKING_ERROR_LIMIT) - 1.0) < 1e-4
    assert round(shadow_price, 2) == round(unit_te / TRACKING_ERROR_LIMIT, 2) == 3.33
    at_shadow_price = solve_utility(Constraints(
        benchmark_weights=benchmark, tracking_err_vol_constraint=TRACKING_ERROR_LIMIT,
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        tre_utility_weight=shadow_price), alphas, covar)
    np.testing.assert_allclose(at_shadow_price, hard[ASSETS], rtol=0.0, atol=1e-5)
    # The shadow price moves with the inputs: doubling the alphas doubles it, and scaling the
    # covariance by four halves it.
    for scaled_alphas, scaled_covar, factor in ((2 * alphas, covar, 2.0),
                                                (alphas, 4 * covar, 0.5)):
        moved = closed_form_active_weights(scaled_alphas, scaled_covar, 1.0)
        moved_te = explicit_tracking_error(moved, 0.0 * moved, scaled_covar)
        assert abs(moved_te - factor * unit_te) < 1e-15
    # The residual evaluator reports each excess as a soft violation that still passes.
    soft_limit = Constraints(
        benchmark_weights=benchmark, tracking_err_vol_constraint=TRACKING_ERROR_LIMIT,
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    for _, row in penalties.iterrows():
        soft = record_of(row[ASSETS], soft_limit, 'tracking_error', covar=covar)
        assert not soft.hard and soft.passed
        assert abs(soft.violation - row['excess_over_limit']) < 1e-12
    # The page's table: Equity weight, then tracking error, excess and active return in %.
    displayed = [[0.556, 2.00, 0.00, 0.27], [0.803, 6.66, 4.66, 0.89],
                 [0.626, 3.33, 1.33, 0.44], [0.568, 2.22, 0.22, 0.30],
                 [0.521, 1.33, 0.00, 0.18], [0.485, 0.67, 0.00, 0.09],
                 [0.468, 0.33, 0.00, 0.04]]
    assert np.round(path['Equity'], 3).tolist() == [row[0] for row in displayed]
    assert np.round(100 * path[['tracking_error', 'excess_over_limit', 'active_return']],
                    2).to_numpy().tolist() == [row[1:] for row in displayed]

    # Backend coverage. SciPy boxes: [0, 1] per asset for long-only without sides, none for
    # long/short; a supplied side fills the missing lower side with 0 (long-only) or -inf
    # (long/short) and the missing upper side with 1.
    two = np.zeros((2, 2))
    assert Constraints().set_scipy_bounds(two).tolist() == [[0.0, 1.0], [0.0, 1.0]]
    assert Constraints(is_long_only=False).set_scipy_bounds(two) is None
    caps = pd.Series([0.6, 0.7], index=pair)
    assert Constraints(max_weights=caps).set_scipy_bounds(two).tolist() == [[0.0, 0.6],
                                                                            [0.0, 0.7]]
    assert Constraints(is_long_only=False, max_weights=caps).set_scipy_bounds(two).tolist() == [
        [-np.inf, 0.6], [-np.inf, 0.7]]
    assert Constraints(min_weights=0.5 * caps).set_scipy_bounds(two).tolist() == [[0.3, 1.0],
                                                                                  [0.35, 1.0]]
    # SciPy compiles long-only, the exact exposure target as two opposite inequalities and the
    # four group rows. Target returns have their own row; PyRB rows are -L'w <= -l and L'w <= u.
    callbacks, _ = allocation.set_scipy_constraints(np.zeros((3, 3)))
    assert [callback["type"] for callback in callbacks] == ["ineq"] * 7
    returns_only, _ = return_floor.set_scipy_constraints(two)
    assert len(returns_only) == 4
    _, rows_c, rows_d = allocation.set_pyrb_constraints(np.zeros((3, 3)))
    assert rows_c.tolist() == [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, -1.0],
                               [0.0, 1.0, 1.0]]
    assert rows_d.tolist() == [-0.30, 0.55, -0.45, 0.70]

    # Universe alignment. update_with_valid_tickers fills an inserted label with 0.0 for the
    # boxes, weights_0, asset_returns and benchmark_weights and with 1.0 for turnover_costs;
    # an explicit NaN survives.
    flat = Constraints(
        min_weights=pd.Series([0.1, float("nan")], index=pair),
        max_weights=pd.Series([0.9, 0.8], index=pair),
        weights_0=pd.Series([0.5, 0.5], index=pair),
        asset_returns=pd.Series([0.05, 0.03], index=pair),
        benchmark_weights=pd.Series([0.6, 0.4], index=pair),
        turnover_costs=pd.Series([2.0, 2.0], index=pair))
    grown = flat.update_with_valid_tickers(valid_tickers=["A", "B", "C"])
    for field, fill in (("min_weights", 0.0), ("max_weights", 0.0), ("weights_0", 0.0),
                        ("asset_returns", 0.0), ("benchmark_weights", 0.0),
                        ("turnover_costs", 1.0)):
        assert getattr(grown, field).index.tolist() == ["A", "B", "C"]
        assert getattr(grown, field)["C"] == fill
    assert np.isnan(grown.min_weights["B"])
    # A label missing from the rebalancing indicators trades: C keeps its 0.0 cap instead of
    # being pinned at its 0.2 current weight.
    tradable = flat.update_with_valid_tickers(
        valid_tickers=["A", "B", "C"], weights_0=pd.Series([0.4, 0.4, 0.2], index=["A", "B", "C"]),
        rebalancing_indicators=pd.Series([1.0, 1.0], index=pair))
    assert tradable.max_weights["C"] == 0.0
    # total_to_good_ratio scales the turnover limit and the per-name maxima, except a maximum
    # close to 1.0; minima, exposure, group bounds, target return and risk limits stay.
    scaled = Constraints(
        min_weights=pd.Series([0.1, 0.0], index=pair),
        max_weights=pd.Series([0.30, 1.0 - 1e-9], index=pair),
        min_exposure=0.9, max_exposure=1.0, turnover_constraint=0.20,
        asset_returns=pd.Series([0.05, 0.02], index=pair), target_return=0.03,
        benchmark_weights=pd.Series([0.5, 0.5], index=pair), tracking_err_vol_constraint=0.05,
        max_target_portfolio_vol_an=0.10,
        group_lower_upper_constraints=one_group(both, lower=0.5, upper=1.0),
    ).update_with_valid_tickers(valid_tickers=list(pair), total_to_good_ratio=1.5)
    assert np.round(scaled.max_weights, 12).tolist() == [0.45, 1.0 - 1e-9]
    assert round(scaled.turnover_constraint, 12) == 0.30
    assert scaled.min_weights.tolist() == [0.1, 0.0]
    assert (scaled.min_exposure, scaled.max_exposure, scaled.target_return) == (0.9, 1.0, 0.03)
    assert scaled.tracking_err_vol_constraint == 0.05 and scaled.max_target_portfolio_vol_an == 0.10
    assert scaled.group_lower_upper_constraints.group_min_allocation["G"] == 0.5
    assert scaled.group_lower_upper_constraints.group_max_allocation["G"] == 1.0
    # update() aligns only the nested blocks; the flat Series keep their order.
    reordered = Constraints(min_weights=pd.Series([0.1, 0.0], index=pair),
                            group_lower_upper_constraints=one_group(both, upper=1.0),
                            ).update(valid_tickers=["B", "A"])
    assert reordered.min_weights.index.tolist() == ["A", "B"]
    assert reordered.group_lower_upper_constraints.group_loadings.index.tolist() == ["B", "A"]

    from optimalportfolios.optimization.constraints import (
        compute_eligible_rebalancing_bounds,
    )

    assets = ["a", "b", "c", "d"]
    current = pd.Series([0.5, 0.3, 0.0, 0.0], index=assets)
    model = pd.Series([0.2, 0.3, 0.5, 0.0], index=assets)
    lower, upper, indicators = compute_eligible_rebalancing_bounds(
        current_weights=current,
        model_weights=model,
        current_min_weights=pd.Series(0.0, index=assets),
        current_max_weights=pd.Series(1.0, index=assets),
    )

    print(lower.tolist())      # [0.2, 0.3, 0.0, 0.0]
    print(upper.tolist())      # [0.5, 0.3, 0.5, 0.0]
    print(indicators.tolist()) # [1, 1, 1, 0]

    # The printed values. The corridor lies between current and model weights, so holding is
    # possible and the model is never overshot; b, already at its model weight, is pinned.
    assert lower.tolist() == [0.2, 0.3, 0.0, 0.0] and upper.tolist() == [0.5, 0.3, 0.5, 0.0]
    assert indicators.tolist() == [1, 1, 1, 0]
    assert (lower >= np.minimum(current, model)).all()
    assert (upper <= np.maximum(current, model)).all()
    assert ((lower <= current) & (current <= upper)).all() and lower["b"] == upper["b"]
    # The indicator needs an absolute weight strictly above 1e-8 in either portfolio.
    edge = pd.Index(["x", "y", "z"])
    _, _, flags = compute_eligible_rebalancing_bounds(
        current_weights=pd.Series([1e-8, 2e-8, 0.0], index=edge),
        model_weights=pd.Series([0.0, 0.0, -2e-8], index=edge),
        current_min_weights=pd.Series(-1.0, index=edge),
        current_max_weights=pd.Series(1.0, index=edge))
    assert flags.tolist() == [0, 1, 1]

    # Frozen positions: an indicator not close to one pins each configured box side at the
    # current weight; 1 - 1e-9 counts as one and trades, 0.999 and 0 freeze.
    names = ["W", "X", "Y", "Z"]
    freeze = dict(valid_tickers=names, weights_0=pd.Series([0.1, 0.2, 0.3, 0.4], index=names),
                  rebalancing_indicators=pd.Series([1.0, 1.0 - 1e-9, 0.999, 0.0], index=names))
    pinned = Constraints(min_weights=pd.Series(0.0, index=names),
                         max_weights=pd.Series(1.0, index=names)).update_with_valid_tickers(
        **freeze)
    assert pinned.min_weights.tolist() == [0.0, 0.0, 0.3, 0.4]
    assert pinned.max_weights.tolist() == [1.0, 1.0, 0.3, 0.4]
    capped_only = Constraints(max_weights=pd.Series(1.0, index=names)).update_with_valid_tickers(
        **freeze)
    assert capped_only.min_weights is None  # no exact pin without both sides
    assert capped_only.max_weights.tolist() == [1.0, 1.0, 0.3, 0.4]
    # A long-only book clips a tiny negative frozen weight to zero; a long/short book keeps it.
    drifted = dict(valid_tickers=names, weights_0=pd.Series([-1e-9, 0.5, 0.2, 0.3], index=names),
                   rebalancing_indicators=pd.Series([0.0, 1.0, 1.0, 1.0], index=names))
    for long_only, floor, kept in ((True, 0.0, 0.0), (False, -1.0, -1e-9)):
        book = Constraints(is_long_only=long_only, min_weights=pd.Series(floor, index=names),
                           max_weights=pd.Series(1.0, index=names)).update_with_valid_tickers(
            **drifted)
        assert book.min_weights["W"] == book.max_weights["W"] == kept

    assets = pd.Index(["Alternatives", "Liquid"])
    spec = Constraints(
        min_weights=pd.Series(0.0, index=assets),
        max_weights=pd.Series(1.0, index=assets),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=pd.DataFrame({"Illiquid": [1.0, 0.0]}, index=assets),
            group_min_allocation=None,
            group_max_allocation=pd.Series({"Illiquid": 0.20}),
        ),
    )
    aligned = spec.update_with_valid_tickers(
        valid_tickers=assets.tolist(),
        weights_0=pd.Series([0.25, 0.75], index=assets),
        rebalancing_indicators=pd.Series([0, 1], index=assets),
    )
    aligned.group_lower_upper_constraints.group_max_allocation["Illiquid"]
    # 0.25000001

    # The waived ceiling is the frozen 0.25 plus the 1e-8 feasibility cushion.
    waived = aligned.group_lower_upper_constraints.group_max_allocation["Illiquid"]
    assert waived == 0.25 + 1e-8 and round(waived, 8) == 0.25000001
    frozen_args = dict(valid_tickers=assets.tolist(),
                       weights_0=pd.Series([0.25, 0.75], index=assets),
                       rebalancing_indicators=pd.Series([0, 1], index=assets))
    # The 0.05 mismatch is material (at least 1e-4): one INFO record carries the structured
    # relaxation. A 5e-5 mismatch is reconciled all the same, at DEBUG and without a record.
    _, records = logged(spec.update_with_valid_tickers, **frozen_args)
    assert [record.levelno for record in records] == [logging.INFO]
    assert records[0].relaxation.items == (("Illiquid", "group_max", 0.20, 0.25 + 1e-8),)
    small, records = logged(spec.update_with_valid_tickers, **{
        **frozen_args, "weights_0": pd.Series([0.20005, 0.79995], index=assets)})
    assert [record.levelno for record in records] == [logging.DEBUG]
    assert not hasattr(records[0], "relaxation")
    assert small.group_lower_upper_constraints.group_max_allocation["Illiquid"] == 0.20005 + 1e-8
    # max_relaxation_tol escalates the log to ERROR and leaves the waiver in place.
    tolerated, records = logged(spec.update_with_valid_tickers, max_relaxation_tol=0.01,
                                **frozen_args)
    assert [record.levelno for record in records] == [logging.ERROR]
    assert records[0].relaxation.breached_tol
    assert tolerated.group_lower_upper_constraints.group_max_allocation["Illiquid"] == waived
    # Without the waiver, or when the bound was already infeasible before the freeze, the
    # aligned constructor rejects the frozen overhang.
    assert_raises(ValueError, spec.update_with_valid_tickers, relax_frozen_group_bounds=False,
                  **frozen_args)
    assert_raises(ValueError, spec.copy(min_weights=pd.Series([0.20005, 0.0], index=assets))
                  .update_with_valid_tickers, **frozen_args)
    # The symmetric rule lowers a floor to the frozen maxima minus 1e-8.
    floor_spec = spec.copy(group_lower_upper_constraints=GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Core": [0.0, 1.0]}, index=assets),
        group_min_allocation=pd.Series({"Core": 0.80}), group_max_allocation=None))
    lowered = floor_spec.update_with_valid_tickers(
        **{**frozen_args, "rebalancing_indicators": pd.Series([1, 0], index=assets)})
    assert lowered.group_lower_upper_constraints.group_min_allocation["Core"] == 0.75 - 1e-8
    # A negative-only loading column is a valid row but grants no waiver.
    signed = spec.copy(group_lower_upper_constraints=GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Short": [-1.0, 0.0]}, index=assets),
        group_min_allocation=None, group_max_allocation=pd.Series({"Short": -0.30})))
    unwaived = signed.update_with_valid_tickers(**frozen_args)
    assert unwaived.group_lower_upper_constraints.group_max_allocation["Short"] == -0.30
    assert len(compiled_rows(unwaived, 2)) == 4 + 1

    candidate = np.array([0.62, 0.28, 0.10])
    records = evaluate_constraint_residuals(
        candidate,
        constraints,
        covar=covar.to_numpy(),
    )
    frame = pd.DataFrame([vars(record) for record in records])
    breaches = frame.loc[
        frame["hard"] & ~frame["passed"],
        ["constraint_type", "name", "actual", "lower", "upper", "violation"],
    ]

    # The breaches the page lists, each recomputed by hand: the Equity cap, weighted turnover
    # 0.405, group turnover 0.22 and 0.22, Growth 0.62 and Defensive 0.38, active Risk assets
    # 0.12 and beta h @ w above 1.15. Tracking error, return and style pass.
    assert list(zip(breaches["constraint_type"], breaches["name"])) == [
        ("instrument_weight", "Equity"), ("turnover", "total_l1"), ("group_turnover", "Growth"),
        ("group_turnover", "Defensive"), ("group_weight", "Growth"),
        ("group_weight", "Defensive"), ("sector_deviation", "Risk assets"),
        ("benchmark_beta", "portfolio")]
    trade = candidate - np.array(CURRENT)
    hand = [0.62, np.abs(np.array(TURNOVER_COSTS) * trade).sum(), abs(trade[0]),
            np.abs(trade[1:]).sum(), 0.62, 0.38, abs(0.62 + 0.10 - 0.60),
            explicit_beta.to_numpy() @ candidate]
    np.testing.assert_allclose(breaches["actual"], hand, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(hand[1:4], [0.405, 0.22, 0.22], rtol=0.0, atol=1e-12)
    assert hand[-1] > 1.15
    # Tolerances: 1e-6 for long-only and instrument boxes, 1e-4 for every other row.
    tolerances = frame.groupby("constraint_type")["tolerance"].unique()
    assert all(list(values) == ([1e-6] if kind in ("long_only", "instrument_weight") else [1e-4])
               for kind, values in tolerances.items())
    # Risk rows need a covariance; benchmark-relative risk also needs benchmark weights.
    risk_kinds = {"portfolio_volatility", "tracking_error", "group_tracking_error"}
    assert not risk_kinds & {r.constraint_type for r in evaluate_constraint_residuals(
        candidate, constraints)}
    assert {r.constraint_type for r in evaluate_constraint_residuals(
        candidate, constraints.copy(benchmark_weights=None), covar=covar.to_numpy())
        } & risk_kinds == {"portfolio_volatility"}

    # The next block reads a wrapper outcome. The minimum-tracking-error wrapper, run on the
    # forced example's constraints, holds the benchmark, which meets every hard row.
    import optimalportfolios as opt
    _, outcome = opt.wrapper_minimise_tracking_error(
        pd_covar=covar, benchmark_weights=benchmark, constraints=constraints)

    outcome.compliant
    outcome.residuals_frame()
    hard_breaches = [
        residual
        for residual in outcome.constraint_residuals
        if residual.hard and not residual.passed
    ]

    assert outcome.accepted and outcome.compliant and hard_breaches == []
    assert len(outcome.residuals_frame()) == len(outcome.constraint_residuals) == 21
    np.testing.assert_allclose(outcome.weights, benchmark, rtol=0.0, atol=1e-6)
    # accepted and compliant differ: an unreachable return floor makes the solve fail, the
    # wrapper falls back to the current weights, and they do not satisfy the mandate.
    with solver_warnings_silenced():
        fallback, failed = opt.wrapper_minimise_tracking_error(
            pd_covar=covar, benchmark_weights=benchmark,
            constraints=constraints.copy(target_return=0.10))
    assert not failed.accepted and not failed.compliant
    assert failed.fallback_source == "weights_0"
    np.testing.assert_allclose(fallback, CURRENT, rtol=0.0, atol=0.0)

    print("constraints: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: tracking error and active return, hard limit against penalties.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from optimalportfolios.optimization.constraints import (
        ConstraintEnforcementType,
        Constraints,
    )

    table, shadow_price = tracking_error_path(TRACKING_ERROR_LIMIT, PENALTY_WEIGHTS)
    hard, penalties = table.loc['hard limit'], table.iloc[1:]
    weights = penalties['penalty_weight'].to_numpy()

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange = '#2a78d6', '#eb6834'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    panels = ((left, 'tracking_error', 'Tracking error against the 2% limit', 0.075, 0),
              (right, 'active_return', 'Active expected return', 0.010, 1))
    for axis, column, title, top, decimals in panels:
        axis.plot(weights, penalties[column], color=blue, marker='o', linewidth=1.8,
                  label='Penalty on tracking error')
        axis.axhline(hard[column], color=orange, linestyle='--', linewidth=1.8,
                     label='Hard limit (forced solve)')
        # The shadow price marks where the penalty solve reaches the hard solve.
        axis.plot([shadow_price, shadow_price], [0.0, hard[column]], color=muted,
                  linestyle=':', linewidth=1.4)
        axis.text(shadow_price * 1.05, 0.012 * top, f'shadow price {shadow_price:.2f}',
                  color=muted, fontsize=9, va='bottom')
        for weight, value in zip(weights, penalties[column]):
            axis.annotate(f'{100 * value:.2f}%', (weight, value), textcoords='offset points',
                          xytext=(6, 5), fontsize=9, color=ink)
        axis.set_xscale('log')
        axis.set_xticks(weights, [f'{weight:g}' for weight in weights])
        axis.minorticks_off()
        axis.set_xlabel('tre_utility_weight (log scale)')
        axis.set_ylim(0.0, top)
        axis.set_title(title, loc='left', color=ink)
        axis.yaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(1.0, decimals=decimals))
        axis.set_facecolor(surface)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    left.fill_between(weights, TRACKING_ERROR_LIMIT, penalties['tracking_error'],
                      where=penalties['tracking_error'] > TRACKING_ERROR_LIMIT,
                      interpolate=True, color=blue, alpha=0.12, linewidth=0)
    left.text(1.07, TRACKING_ERROR_LIMIT + 0.003, 'excess over\nthe limit', color=ink,
              fontsize=9, va='bottom')
    right.text(weights[-1], hard['active_return'] + 0.0002,
               f'forced solve: {100 * hard["active_return"]:.2f}%', color=ink, fontsize=9,
               ha='right', va='bottom')
    left.legend(frameon=False, loc='upper right', fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    excess = penalties['excess_over_limit'].to_numpy()
    sigma, benchmark = np.array(COVARIANCE), np.array(BENCHMARK)
    unit = closed_form_active_weights(ALPHAS, sigma, 1.0)
    soft = Constraints(benchmark_weights=pd.Series(BENCHMARK, index=ASSETS),
                       tracking_err_vol_constraint=TRACKING_ERROR_LIMIT,
                       constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    records = [record_of(row[ASSETS], soft, 'tracking_error', covar=sigma)
               for _, row in penalties.iterrows()]
    checks = {
        'hard_solve_meets_limit': bool(hard['tracking_error'] <= TRACKING_ERROR_LIMIT + 1e-8),
        'hard_limit_binds': bool(abs(hard['tracking_error'] - TRACKING_ERROR_LIMIT) < 1e-8),
        'excess_falls_with_penalty_weight': bool(
            (np.diff(excess) <= 0.0).all()
            and all(later < earlier for earlier, later in zip(excess, excess[1:]) if earlier > 0)),
        'smallest_weight_exceeds_limit': bool(excess[0] > 0.0),
        'largest_weight_leaves_limit_slack': bool(
            penalties['tracking_error'].iloc[-1] < TRACKING_ERROR_LIMIT),
        'penalty_solves_match_closed_form': bool(all(
            np.allclose(row[ASSETS].to_numpy(), benchmark + unit / row['penalty_weight'],
                        atol=1e-6, rtol=0.0) for _, row in penalties.iterrows())),
        'shadow_price_matches_closed_form': bool(abs(
            shadow_price * TRACKING_ERROR_LIMIT / explicit_tracking_error(unit, 0.0 * unit, sigma)
            - 1.0) < 1e-4),
        'penalty_at_shadow_price_reproduces_hard_solve': bool(np.allclose(
            solve_utility(soft.copy(tre_utility_weight=shadow_price), ALPHAS, sigma),
            hard[ASSETS].to_numpy(), atol=1e-5, rtol=0.0)),
        'excess_reported_as_soft_violation': bool(all(
            not record.hard and record.passed and abs(record.violation - value) < 1e-12
            for record, value in zip(records, excess))),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
