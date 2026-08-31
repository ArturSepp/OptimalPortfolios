"""
the utility (penalty) formulation of the two SAA solvers.

``Constraints`` offers two enforcement types. ``FORCED_CONSTRAINTS`` hands the limits to the
solver as hard constraints; ``UTILITY_CONSTRAINTS`` subtracts them from the objective as
penalties instead. Only the forced path had tests, so the whole
``cvx_*_utility`` branch of ``max_return_target_vol`` and ``min_variance_target_return`` —
about 130 statements — had never run.

The distinction matters in production rather than in theory. A penalised solve *always*
returns something: exceed the vol budget and you pay for it in the objective instead of being
told the problem is infeasible. That is the point of the formulation, and it is also why a
mistake in it is invisible — a wrong penalty weight yields a plausible portfolio, not an
error. So the cases here assert the *direction* the penalty must push: raising the vol penalty
must reduce realised vol, raising the turnover penalty must reduce trading against
``weights_0``, and a zero penalty must reproduce the unpenalised solve.

The universe is the same three-asset panel the other rolling-solver tests use, so a result can
be compared against the forced-constraint answer for the same inputs.
"""
# packages
from typing import Dict
import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
import optimalportfolios.optimization.saa.max_return_target_vol as max_return_solver
import optimalportfolios.optimization.saa.min_variance_target_return as min_variance_solver
from optimalportfolios import Constraints
from optimalportfolios.optimization.constraints import (
    ConstraintEnforcementType,
    GroupLowerUpperConstraints,
)
from optimalportfolios.optimization.saa.max_return_target_vol import (
    cvx_max_return_target_vol_utility, rolling_max_return_target_vol,
    wrapper_max_return_target_vol)
from optimalportfolios.optimization.saa.min_variance_target_return import (
    cvx_min_variance_target_return_utility, rolling_min_variance_target_return)

TICKERS = ['growth', 'balanced', 'defensive']
VOLS = np.array([0.22, 0.14, 0.06])
CORR = np.array([[1.00, 0.45, 0.15],
                 [0.45, 1.00, 0.25],
                 [0.15, 0.25, 1.00]])
COVAR = np.outer(VOLS, VOLS) * CORR
ALPHAS = np.array([0.09, 0.06, 0.02])
REBALANCING_DATES = pd.DatetimeIndex(['2024-03-31', '2024-06-30', '2024-09-30'])
SEED = 20260810


def make_prices(n_days: int = 400) -> pd.DataFrame:
    """A seeded daily price panel, used for column alignment and weight drift."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range('2023-06-01', periods=n_days, freq='B')
    returns = rng.multivariate_normal(np.full(3, 0.0003), COVAR / 260.0, size=n_days)
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=TICKERS)


def make_covar_dict() -> Dict[pd.Timestamp, pd.DataFrame]:
    """The same covariance at every rebalancing date."""
    covar = pd.DataFrame(COVAR, index=TICKERS, columns=TICKERS)
    return {date: covar for date in REBALANCING_DATES}


def make_expected_returns() -> pd.DataFrame:
    """Expected returns per rebalancing date, ordered growth > balanced > defensive."""
    return pd.DataFrame([list(ALPHAS)] * len(REBALANCING_DATES),
                        index=REBALANCING_DATES, columns=TICKERS)


def utility_constraints(**overrides) -> Constraints:
    """A long-only, fully invested constraint set in the penalty formulation.

    Note the two penalty weights are *not* None by default — ``Constraints`` ships
    ``tre_utility_weight=1.0`` and ``turnover_utility_weight=0.4``, so the penalised solve is
    already curved before a caller asks for anything. A case that wants the unpenalised
    problem has to say ``tre_utility_weight=0.0`` explicitly.
    """
    kwargs = dict(is_long_only=True,
                  min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS),
                  constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    kwargs.update(overrides)
    return Constraints(**kwargs)


def realised_vol(weights: np.ndarray) -> float:
    """Portfolio volatility of a weight vector under COVAR."""
    return float(np.sqrt(weights @ COVAR @ weights))


def assert_investable(weights: np.ndarray) -> None:
    """A utility solve still respects the hard budget and box constraints."""
    assert weights is not None
    assert np.all(weights >= -1e-6), "long-only penalised solve produced a short"
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)


def _composition_constraints(benchmark_weights: pd.Series = None) -> Constraints:
    """Return constraints that expose SAA utility objective and hard-row composition."""
    group_loadings = pd.DataFrame(
        {"risky": [1.0, 1.0, 0.0]}, index=TICKERS)
    return utility_constraints(
        max_exposure=1.10,
        min_exposure=0.80,
        benchmark_weights=benchmark_weights,
        weights_0=pd.Series([0.30, 0.30, 0.40], index=TICKERS),
        turnover_costs=pd.Series([2.0, 0.5, 1.5], index=TICKERS),
        turnover_utility_weight=1.75,
        tre_utility_weight=2.50,
        target_return=0.055,
        asset_returns=pd.Series(ALPHAS, index=TICKERS),
        max_target_portfolio_vol_an=0.01,
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=group_loadings,
            group_min_allocation=pd.Series({"risky": 0.30}),
            group_max_allocation=pd.Series({"risky": 0.80}),
        ),
    )


def _capture_problem(monkeypatch, solver_module, solve_call, probe: np.ndarray):
    """Run a CVXPY entry point without a backend solve and return its compiled problem."""
    captured = {}
    sentinel = object()

    def capture_solve(problem, **_kwargs):
        """Record the problem and supply the requested probe as its variable value."""
        captured["problem"] = problem
        variables = problem.variables()
        assert len(variables) == 1
        variables[0].value = probe

    monkeypatch.setattr(cvx.Problem, "solve", capture_solve)
    monkeypatch.setattr(
        solver_module,
        "validate_solution",
        lambda *_args, **_kwargs: sentinel,
    )

    assert solve_call() is sentinel
    return captured["problem"]


def _expected_composition_rows(
        probe: np.ndarray,
        include_target: bool,
) -> list[tuple[object, object]]:
    """Compute the SAA utility mandate-row sequence directly."""
    weight_sum = float(np.sum(probe))
    rows = [
        (np.zeros(len(TICKERS)), probe),
        (weight_sum, 1.10),
        (0.80, weight_sum),
        (np.zeros(len(TICKERS)), probe),
        (probe, np.ones(len(TICKERS))),
    ]
    if include_target:
        rows.append((0.055, float(ALPHAS @ probe)))
    risky_weight = float(probe[0] + probe[1])
    rows.extend([(0.30, risky_weight), (risky_weight, 0.80)])
    return rows


def _assert_compiled_rows(
        rows: list,
        expected: list[tuple[object, object]],
) -> None:
    """Assert compiled CVXPY row order and evaluated argument values."""
    assert len(rows) == len(expected)
    for row, (expected_left, expected_right) in zip(rows, expected):
        np.testing.assert_allclose(
            np.asarray(row.args[0].value, dtype=float),
            np.asarray(expected_left, dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(row.args[1].value, dtype=float),
            np.asarray(expected_right, dtype=float),
            rtol=0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize("has_benchmark", [False, True], ids=["absolute", "benchmark"])
def test_max_return_utility_composition_matches_direct_formula(
        monkeypatch,
        has_benchmark: bool,
) -> None:
    """Max-return utility keeps its formula and every configured mandate row hard."""
    probe = np.array([0.45, 0.25, 0.30])
    benchmark = (
        pd.Series([0.35, 0.35, 0.30], index=TICKERS)
        if has_benchmark else None
    )
    constraints = _composition_constraints(benchmark)
    problem = _capture_problem(
        monkeypatch,
        max_return_solver,
        lambda: max_return_solver.cvx_max_return_target_vol_utility(
            covar=COVAR,
            alphas=ALPHAS,
            constraints=constraints,
            has_benchmark=has_benchmark,
            factorize_covar=False,
        ),
        probe,
    )

    risk_weights = probe - benchmark.to_numpy() if has_benchmark else probe
    alpha_weights = risk_weights if has_benchmark else probe
    weight_change = probe - constraints.weights_0.to_numpy()
    expected_objective = float(ALPHAS @ alpha_weights)
    expected_objective -= 2.50 * float(risk_weights @ COVAR @ risk_weights)
    expected_objective -= 1.75 * float(np.sum(np.abs(
        constraints.turnover_costs.to_numpy() * weight_change)))

    assert float(problem.objective.args[0].value) == pytest.approx(
        expected_objective, abs=1e-12)
    _assert_compiled_rows(
        problem.constraints,
        _expected_composition_rows(probe, include_target=True),
    )


@pytest.mark.parametrize("has_benchmark", [False, True], ids=["absolute", "benchmark"])
def test_min_variance_utility_composition_matches_direct_formula(
        monkeypatch,
        has_benchmark: bool,
) -> None:
    """Min-variance utility keeps its absolute/active risk and positive turnover cost."""
    probe = np.array([0.45, 0.25, 0.30])
    benchmark = (
        pd.Series([0.35, 0.35, 0.30], index=TICKERS)
        if has_benchmark else None
    )
    constraints = _composition_constraints(benchmark)
    problem = _capture_problem(
        monkeypatch,
        min_variance_solver,
        lambda: min_variance_solver.cvx_min_variance_target_return_utility(
            covar=COVAR,
            constraints=constraints,
            has_benchmark=has_benchmark,
            factorize_covar=False,
        ),
        probe,
    )

    risk_weights = probe - benchmark.to_numpy() if has_benchmark else probe
    weight_change = probe - constraints.weights_0.to_numpy()
    expected_objective = float(risk_weights @ COVAR @ risk_weights)
    expected_objective += 1.75 * float(np.sum(np.abs(
        constraints.turnover_costs.to_numpy() * weight_change)))

    assert float(problem.objective.args[0].value) == pytest.approx(
        expected_objective, abs=1e-12)
    _assert_compiled_rows(
        problem.constraints,
        _expected_composition_rows(probe, include_target=True),
    )


# --------------------------------------------------------------------------- #
# max return, vol as a penalty
# --------------------------------------------------------------------------- #
def test_max_return_utility_without_a_penalty_chases_the_highest_alpha() -> None:
    """with the vol penalty switched off the objective is linear and takes a vertex"""
    outcome = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS, constraints=utility_constraints(tre_utility_weight=0.0))
    assert_investable(outcome.weights)
    assert outcome.weights[0] == pytest.approx(1.0, abs=1e-4)   # all in 'growth'


def test_max_return_utility_penalises_volatility_by_default() -> None:
    """the shipped default is a curved objective, not a linear one

    ``Constraints`` defaults ``tre_utility_weight`` to 1.0, so a caller who selects the
    utility formulation and sets nothing else already gets a risk-penalised solve. That is
    easy to miss and changes the answer, so it is pinned here.
    """
    default = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS, constraints=utility_constraints())
    unpenalised = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS, constraints=utility_constraints(tre_utility_weight=0.0))
    assert_investable(default.weights)
    assert realised_vol(default.weights) < realised_vol(unpenalised.weights)


def test_max_return_utility_vol_penalty_reduces_realised_vol() -> None:
    """the penalty is the whole mechanism: more weight on it must buy less volatility"""
    light = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS, constraints=utility_constraints(tre_utility_weight=1.0))
    heavy = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS, constraints=utility_constraints(tre_utility_weight=50.0))
    assert_investable(light.weights)
    assert_investable(heavy.weights)
    assert realised_vol(heavy.weights) < realised_vol(light.weights)
    # and the heavier penalty holds less of the most volatile asset
    assert heavy.weights[0] < light.weights[0]


def test_max_return_utility_turnover_penalty_holds_the_prior_portfolio() -> None:
    """penalising turnover against weights_0 must reduce the distance travelled"""
    weights_0 = pd.Series([0.0, 0.0, 1.0], index=TICKERS)      # start fully defensive
    free = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS,
        constraints=utility_constraints(tre_utility_weight=5.0, weights_0=weights_0))
    sticky = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS,
        constraints=utility_constraints(tre_utility_weight=5.0, weights_0=weights_0,
                                        turnover_utility_weight=10.0))
    assert_investable(sticky.weights)
    free_turnover = np.abs(free.weights - weights_0.to_numpy()).sum()
    sticky_turnover = np.abs(sticky.weights - weights_0.to_numpy()).sum()
    assert sticky_turnover < free_turnover


def test_max_return_utility_applies_per_asset_turnover_costs() -> None:
    """with costs supplied the penalty is cost-weighted, so the dear asset moves least"""
    weights_0 = pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS)
    costs = pd.Series([10.0, 0.001, 0.001], index=TICKERS)     # 'growth' is expensive to trade
    outcome = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS,
        constraints=utility_constraints(tre_utility_weight=2.0, weights_0=weights_0,
                                        turnover_utility_weight=5.0, turnover_costs=costs))
    assert_investable(outcome.weights)
    moves = np.abs(outcome.weights - weights_0.to_numpy())
    assert moves[0] < moves[1] + 1e-9, "the costly asset should have moved least"


def test_max_return_utility_runs_the_benchmark_relative_formulation() -> None:
    """with a benchmark the objective switches to the active-risk utility"""
    benchmark = pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS)
    outcome = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS, has_benchmark=True,
        constraints=utility_constraints(benchmark_weights=benchmark,
                                        tre_utility_weight=5.0))
    assert_investable(outcome.weights)
    assert outcome.weights[0] > benchmark['growth'], "the alpha tilt did not appear"


def test_max_return_utility_without_factorisation_matches_the_factorised_solve() -> None:
    """the covariance square root is an implementation detail, not a different problem"""
    constraints = utility_constraints(tre_utility_weight=10.0)
    factorised = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS, constraints=constraints, factorize_covar=True)
    direct = cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS, constraints=constraints, factorize_covar=False)
    np.testing.assert_allclose(factorised.weights, direct.weights, atol=1e-4)


# --------------------------------------------------------------------------- #
# min variance, return as a penalty
# --------------------------------------------------------------------------- #
def test_min_variance_utility_without_a_return_floor_minimises_variance() -> None:
    """with no floor the objective is variance alone, so the solve is minimum variance"""
    outcome = cvx_min_variance_target_return_utility(
        covar=COVAR, constraints=utility_constraints())
    assert_investable(outcome.weights)
    assert int(np.argmax(outcome.weights)) == 2                 # 'defensive'
    nudged = outcome.weights + np.array([0.05, 0.0, -0.05])
    assert realised_vol(outcome.weights) < realised_vol(nudged)


def test_min_variance_utility_return_floor_is_hard_not_penalised() -> None:
    """unlike turnover, the return target is a constraint — it is met exactly, not traded off

    This is the asymmetry worth pinning: the solver name says 'utility', but only the
    turnover term is a penalty here. The return floor stays a hard constraint, so a portfolio
    that misses it is infeasible rather than merely expensive.
    """
    outcome = cvx_min_variance_target_return_utility(
        covar=COVAR, constraints=utility_constraints(
            target_return=0.05, asset_returns=pd.Series(ALPHAS, index=TICKERS)))
    assert_investable(outcome.weights)
    assert float(outcome.weights @ ALPHAS) >= 0.05 - 1e-6


def test_min_variance_utility_a_higher_floor_costs_variance() -> None:
    """demanding more return forces the solve up the frontier"""
    asset_returns = pd.Series(ALPHAS, index=TICKERS)
    modest = cvx_min_variance_target_return_utility(
        covar=COVAR, constraints=utility_constraints(target_return=0.04,
                                                     asset_returns=asset_returns))
    demanding = cvx_min_variance_target_return_utility(
        covar=COVAR, constraints=utility_constraints(target_return=0.075,
                                                     asset_returns=asset_returns))
    assert_investable(demanding.weights)
    assert realised_vol(modest.weights) <= realised_vol(demanding.weights) + 1e-9
    assert float(demanding.weights @ ALPHAS) >= 0.075 - 1e-6


def test_min_variance_utility_turnover_penalty_holds_the_prior_portfolio() -> None:
    """the turnover term is the penalty on this solver, and it must reduce trading"""
    weights_0 = pd.Series([1.0, 0.0, 0.0], index=TICKERS)
    free = cvx_min_variance_target_return_utility(
        covar=COVAR, constraints=utility_constraints(weights_0=weights_0,
                                                     turnover_utility_weight=0.0))
    sticky = cvx_min_variance_target_return_utility(
        covar=COVAR, constraints=utility_constraints(weights_0=weights_0,
                                                     turnover_utility_weight=25.0))
    assert_investable(sticky.weights)
    assert (np.abs(sticky.weights - weights_0.to_numpy()).sum()
            < np.abs(free.weights - weights_0.to_numpy()).sum())


def test_min_variance_utility_applies_per_asset_turnover_costs() -> None:
    """cost-weighted turnover keeps the expensive asset closer to where it started"""
    weights_0 = pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS)
    costs = pd.Series([0.001, 0.001, 10.0], index=TICKERS)     # 'defensive' is dear to trade
    outcome = cvx_min_variance_target_return_utility(
        covar=COVAR,
        constraints=utility_constraints(weights_0=weights_0, turnover_utility_weight=5.0,
                                        turnover_costs=costs))
    assert_investable(outcome.weights)
    moves = np.abs(outcome.weights - weights_0.to_numpy())
    assert moves[2] < moves[0] + 1e-9


def test_min_variance_utility_runs_the_benchmark_relative_formulation() -> None:
    """the benchmark branch minimises active variance rather than absolute variance"""
    benchmark = pd.Series([0.5, 0.3, 0.2], index=TICKERS)
    outcome = cvx_min_variance_target_return_utility(
        covar=COVAR, has_benchmark=True,
        constraints=utility_constraints(benchmark_weights=benchmark))
    assert_investable(outcome.weights)
    # minimising tracking variance with nothing pulling away returns the benchmark
    np.testing.assert_allclose(outcome.weights, benchmark.to_numpy(), atol=1e-4)


def test_min_variance_utility_without_factorisation_matches_the_factorised_solve() -> None:
    """the covariance square root is an implementation detail, not a different problem"""
    constraints = utility_constraints()
    factorised = cvx_min_variance_target_return_utility(
        covar=COVAR, constraints=constraints, factorize_covar=True)
    direct = cvx_min_variance_target_return_utility(
        covar=COVAR, constraints=constraints, factorize_covar=False)
    np.testing.assert_allclose(factorised.weights, direct.weights, atol=1e-4)


# --------------------------------------------------------------------------- #
# reached through the rolling wrappers
# --------------------------------------------------------------------------- #
def test_rolling_max_return_target_vol_routes_to_the_utility_solver() -> None:
    """the enforcement type selects the penalised branch end to end"""
    weights = rolling_max_return_target_vol(
        prices=make_prices(), expected_returns=make_expected_returns(),
        target_vols=pd.Series(0.10, index=REBALANCING_DATES),
        constraints=utility_constraints(tre_utility_weight=10.0),
        benchmark_weights=None, covar_dict=make_covar_dict())
    assert list(weights.index) == list(REBALANCING_DATES)
    assert list(weights.columns) == TICKERS
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-5)


def test_rolling_min_variance_target_return_routes_to_the_utility_solver() -> None:
    """same for the minimum-variance side"""
    weights = rolling_min_variance_target_return(
        prices=make_prices(), expected_returns=make_expected_returns(),
        target_returns=pd.Series(0.05, index=REBALANCING_DATES),
        constraints=utility_constraints(tre_utility_weight=10.0),
        benchmark_weights=None, covar_dict=make_covar_dict())
    assert list(weights.index) == list(REBALANCING_DATES)
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-5)


def test_wrapper_zero_fills_an_asset_with_no_expected_return() -> None:
    """an asset the return frame never mentions warns and is treated as zero alpha

    Note it must be *absent*, not NaN: ``filter_covar_and_vectors_for_nans`` drops NaN
    entries before this branch is reached, so a NaN in the frame is handled earlier and
    silently. The reachable case is a covariance that covers an asset the expected-return
    frame does not.
    """
    covar = pd.DataFrame(COVAR, index=TICKERS, columns=TICKERS)
    expected_returns = pd.Series(ALPHAS[:2], index=TICKERS[:2])   # 'defensive' absent
    with pytest.warns(UserWarning, match=r"NaN expected returns for \['defensive'\]"):
        weights, _ = wrapper_max_return_target_vol(
            pd_covar=covar, expected_returns=expected_returns, target_vol=0.10,
            benchmark_weights=None, constraints=utility_constraints())
    assert weights.notna().all()
    assert weights['defensive'] == pytest.approx(0.0, abs=1e-6)


def test_rolling_wrappers_accept_a_time_varying_benchmark() -> None:
    """a benchmark DataFrame is forward-filled onto the rebalancing schedule"""
    benchmark = pd.DataFrame([[0.4, 0.3, 0.3]], index=REBALANCING_DATES[:1],
                             columns=TICKERS)
    weights = rolling_max_return_target_vol(
        prices=make_prices(), expected_returns=make_expected_returns(),
        target_vols=pd.Series(0.06, index=REBALANCING_DATES),
        constraints=utility_constraints(tre_utility_weight=10.0),
        benchmark_weights=benchmark, covar_dict=make_covar_dict())
    assert list(weights.index) == list(REBALANCING_DATES)
