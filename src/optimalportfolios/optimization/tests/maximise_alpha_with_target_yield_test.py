"""
alpha maximisation under a hard yield floor, and the soft-TE branch that keeps it feasible.

    max_w  alpha'(w - w_b)   s.t.  y'w >= r,  TE(w, w_b) <= budget,  box/turnover/group

The interesting structure is the interaction between the two risk-side terms. A tight tracking
-error budget and an aggressive yield floor can be jointly infeasible -- a low-yield-tilting
alpha colliding with a high floor -- and the module offers two ways through that, selected by
``soft_tracking_error``. With it off, TE is a hard constraint and the solve can legitimately
come back infeasible. With it on, TE moves into the objective as a utility penalty while the
yield floor stays hard, so the target wins.

A populated ``tracking_err_vol_constraint`` is ignored on that second path -- in the solve and
in ``validate_solution`` alike -- which is what
``test_a_populated_hard_te_budget_is_ignored_by_the_soft_branch`` pins. The two have to agree:
when the solve dropped the budget but validation still re-applied it, an ``optimal`` solve was
rejected for violating a constraint it had been deliberately freed from, and every rebalance
fell back to benchmark weights (issue #49). The test therefore asserts not just acceptance but
that the weights equal the utility-only solve, so a budget silently left in force would show
up as a different optimum rather than only as a different reported reason.

That branch does something subtle enough to be worth a test of its own: the utility builder
would penalise turnover as well, so the soft path nulls ``turnover_utility_weight`` and then
re-adds turnover as a *hard* constraint. Get that wrong and turnover is either double-counted
(penalised and constrained) or silently unconstrained. Both produce a solved portfolio.

The objective also switches form on whether a benchmark is present -- active ``alpha'(w - w_b)``
versus absolute ``alpha'w``. Since ``w_b`` is a constant, the two have the same argmax under
identical constraints, so the tests assert the weights agree rather than pretending the switch
changes the optimum; what it changes is the reported objective value.

The yield floor itself is asserted as a floor: at the optimum the portfolio yield must reach
the target, and raising the target must move the weights toward higher-yielding assets.
"""
# packages
import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
import optimalportfolios.optimization.taa.maximise_alpha_with_target_yield as target_yield_solver
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import Constraints, GroupTurnoverConstraint
from optimalportfolios.optimization.taa.maximise_alpha_with_target_yield import (
    cvx_maximise_alpha_with_target_return,
    rolling_maximise_alpha_with_target_return,
    wrapper_maximise_alpha_with_target_return,
)

TICKERS = pd.Index(['A', 'B', 'C', 'D'])
DATES = pd.DatetimeIndex(['2024-03-31', '2024-06-30'])


def covar_frame() -> pd.DataFrame:
    """A well-conditioned 4-asset covariance in the canonical ticker order."""
    vols = np.array([0.20, 0.12, 0.08, 0.05])
    corr = np.array([
        [1.00, 0.30, 0.10, 0.00],
        [0.30, 1.00, 0.20, 0.10],
        [0.10, 0.20, 1.00, 0.25],
        [0.00, 0.10, 0.25, 1.00],
    ])
    return pd.DataFrame(np.outer(vols, vols) * corr, index=TICKERS, columns=TICKERS)


def alphas_series() -> pd.Series:
    """Alphas that prefer the low-yield end of the universe, so the floor genuinely binds."""
    return pd.Series([0.04, 0.02, -0.01, -0.02], index=TICKERS)


def yields_series() -> pd.Series:
    """Yields ordered opposite to the alphas: A is the alpha pick and the worst carry."""
    return pd.Series([0.01, 0.02, 0.04, 0.05], index=TICKERS)


def make_constraints(**overrides) -> Constraints:
    """Long-only fully-invested bounds, with optional field overrides."""
    kwargs = dict(min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS),
                  min_exposure=1.0,
                  max_exposure=1.0)
    kwargs.update(overrides)
    return Constraints(**kwargs)


def prices_frame() -> pd.DataFrame:
    """A price panel spanning the rebalancing dates, used for column alignment and drift."""
    return pd.DataFrame(100.0, index=pd.date_range('2024-01-31', periods=8, freq='ME'),
                        columns=TICKERS)


# --------------------------------------------------------------------------- #
# the yield floor
# --------------------------------------------------------------------------- #
def test_the_yield_target_is_reached_at_the_optimum() -> None:
    """The floor binds: alphas prefer low-yield A, so the optimum sits on the constraint."""
    target = 0.030
    weights, outcome = wrapper_maximise_alpha_with_target_return(
        pd_covar=covar_frame(), alphas=alphas_series(), yields=yields_series(),
        target_return=target, constraints=make_constraints())
    assert outcome.accepted
    assert float(weights @ yields_series()) >= target - 1e-6


def test_a_higher_target_shifts_weight_toward_the_higher_yielding_assets() -> None:
    """Raising the floor must buy carry, at the cost of alpha."""
    common = dict(pd_covar=covar_frame(), alphas=alphas_series(), yields=yields_series(),
                  constraints=make_constraints())
    low, _ = wrapper_maximise_alpha_with_target_return(target_return=0.020, **common)
    high, _ = wrapper_maximise_alpha_with_target_return(target_return=0.045, **common)
    assert high[['C', 'D']].sum() > low[['C', 'D']].sum()
    assert float(high @ alphas_series()) <= float(low @ alphas_series()) + 1e-9


def test_a_nan_yield_is_treated_as_zero_with_a_warning() -> None:
    """An asset with no yield estimate must not be credited toward the floor.

    Filling with zero is the conservative choice and is warned about; filling with the mean
    would let a missing estimate satisfy the target it was supposed to constrain.
    """
    yields = yields_series().copy()
    yields['D'] = np.nan
    with pytest.warns(UserWarning, match='NaN yields'):
        weights, outcome = wrapper_maximise_alpha_with_target_return(
            pd_covar=covar_frame(), alphas=alphas_series(), yields=yields,
            target_return=0.020, constraints=make_constraints())
    contributed = float(weights @ yields.fillna(0.0))
    assert contributed >= 0.020 - 1e-6


# --------------------------------------------------------------------------- #
# absolute vs active objective
# --------------------------------------------------------------------------- #
def test_without_a_benchmark_the_objective_is_absolute() -> None:
    """No benchmark means the plain alpha'w problem -- the original behaviour."""
    outcome = cvx_maximise_alpha_with_target_return(
        covar=covar_frame().to_numpy(), alphas=alphas_series().to_numpy(),
        constraints=make_constraints(asset_returns=yields_series(), target_return=0.020))
    assert outcome.accepted


def test_a_benchmark_switches_to_the_active_objective_without_moving_the_optimum() -> None:
    """alpha'(w - w_b) and alpha'w differ by a constant, so they share an argmax.

    The switch changes the reported objective value, not the portfolio; asserting the weights
    match is what shows the benchmark is subtracted rather than, say, added.
    """
    benchmark = pd.Series(0.25, index=TICKERS)
    absolute = cvx_maximise_alpha_with_target_return(
        covar=covar_frame().to_numpy(), alphas=alphas_series().to_numpy(),
        constraints=make_constraints(asset_returns=yields_series(), target_return=0.020))
    active = cvx_maximise_alpha_with_target_return(
        covar=covar_frame().to_numpy(), alphas=alphas_series().to_numpy(),
        constraints=make_constraints(asset_returns=yields_series(), target_return=0.020,
                                     benchmark_weights=benchmark))
    np.testing.assert_allclose(absolute.weights, active.weights, atol=1e-5)


# --------------------------------------------------------------------------- #
# the soft tracking-error branch
# --------------------------------------------------------------------------- #
def test_soft_tracking_error_reaches_a_target_the_hard_path_cannot() -> None:
    """The soft branch meets an aggressive yield floor where the hard TE budget goes infeasible.

    This is the whole point of the branch: the floor stays hard and TE becomes a penalty, so
    the target takes priority. Whether the caller also leaves a hard budget set is immaterial
    to that -- see the tests below, which pin it either way.
    """
    benchmark = pd.Series(0.25, index=TICKERS)
    covar, alphas = covar_frame().to_numpy(), alphas_series().to_numpy()

    hard = cvx_maximise_alpha_with_target_return(
        covar=covar, alphas=alphas, soft_tracking_error=False,
        constraints=make_constraints(asset_returns=yields_series(), target_return=0.045,
                                     benchmark_weights=benchmark,
                                     tracking_err_vol_constraint=0.0001))
    assert not hard.accepted and hard.status == 'infeasible'

    soft = cvx_maximise_alpha_with_target_return(
        covar=covar, alphas=alphas, soft_tracking_error=True,
        constraints=make_constraints(asset_returns=yields_series(), target_return=0.045,
                                     benchmark_weights=benchmark,
                                     tre_utility_weight=10.0))
    assert soft.accepted
    assert float(soft.weights @ yields_series().to_numpy()) >= 0.045 - 1e-6


def test_a_populated_hard_te_budget_is_ignored_by_the_soft_branch() -> None:
    """Setting both ``tre_utility_weight`` and ``tracking_err_vol_constraint`` is accepted.

    The budget is dropped from the CVXPY problem *and* from the constraints handed to
    ``validate_solution``, so the two agree on which problem was solved. Comparing against the
    utility-only geometry is the assertion that carries the weight: acceptance alone would
    still pass if the budget were merely relaxed rather than removed, whereas identical weights
    can only come from the same problem. The tracking error of that shared optimum is checked
    to exceed the budget, so the case is one the budget would genuinely have rejected -- which
    is what made issue #49 a silent fallback to benchmark weights on every rebalance.
    """
    benchmark = pd.Series(0.25, index=TICKERS)
    covar, alphas = covar_frame().to_numpy(), alphas_series().to_numpy()
    soft = dict(asset_returns=yields_series(), target_return=0.045,
                benchmark_weights=benchmark, tre_utility_weight=10.0)

    with_budget = cvx_maximise_alpha_with_target_return(
        covar=covar, alphas=alphas, soft_tracking_error=True,
        constraints=make_constraints(tracking_err_vol_constraint=0.0001, **soft))
    utility_only = cvx_maximise_alpha_with_target_return(
        covar=covar, alphas=alphas, soft_tracking_error=True,
        constraints=make_constraints(**soft))

    assert with_budget.status == 'optimal'
    assert with_budget.accepted
    assert with_budget.fallback_source is None
    np.testing.assert_allclose(with_budget.weights, utility_only.weights, atol=1e-7)
    assert with_budget.constraints is not None
    assert with_budget.constraints.tracking_err_vol_constraint is None
    active = with_budget.weights - benchmark.to_numpy()
    assert float(np.sqrt(active @ covar @ active)) > 0.0001
    assert float(with_budget.weights @ yields_series().to_numpy()) >= 0.045 - 1e-8


def test_the_hard_path_still_rejects_that_same_geometry() -> None:
    """Softening TE is opt-in: with the flag off the budget binds and the fallback stands.

    The pair with the test above is the point -- the same constraints must reach opposite
    outcomes depending only on ``soft_tracking_error``, which is what shows the fix loosened
    the soft branch rather than the budget everywhere.
    """
    benchmark = pd.Series(0.25, index=TICKERS)
    outcome = cvx_maximise_alpha_with_target_return(
        covar=covar_frame().to_numpy(), alphas=alphas_series().to_numpy(),
        soft_tracking_error=False,
        constraints=make_constraints(asset_returns=yields_series(), target_return=0.045,
                                     benchmark_weights=benchmark,
                                     tracking_err_vol_constraint=0.0001,
                                     tre_utility_weight=10.0))
    assert not outcome.accepted
    assert outcome.fallback_source == 'benchmark_weights'
    np.testing.assert_allclose(outcome.weights, benchmark.to_numpy())


def test_the_soft_branch_requires_a_benchmark_to_engage() -> None:
    """``soft_tracking_error`` without a benchmark falls through to the hard path.

    There is no TE to soften without ``w_b``, so the flag is inert rather than an error.
    """
    outcome = cvx_maximise_alpha_with_target_return(
        covar=covar_frame().to_numpy(), alphas=alphas_series().to_numpy(),
        constraints=make_constraints(asset_returns=yields_series(), target_return=0.020),
        soft_tracking_error=True)
    assert outcome.accepted


def test_the_soft_branch_keeps_turnover_hard_rather_than_double_counting_it() -> None:
    """Turnover is nulled in the utility objective and re-added as a hard bound.

    If the nulling were dropped, turnover would be both penalised and constrained; if the
    re-add were dropped, it would be unconstrained. Either way the solve still succeeds, so
    the binding budget is what has to be checked -- and checked twice: the realised L1 move
    against the budget, and the recorded residual, which is where a turnover term demoted to a
    penalty (or dropped) would show up as a missing or non-hard entry.
    """
    benchmark = pd.Series(0.25, index=TICKERS)
    weights_0 = pd.Series([1.0, 0.0, 0.0, 0.0], index=TICKERS)
    # starting wholly in the lowest-yielding asset, reaching a 2% floor needs to move >= 0.25
    # out of A, i.e. an L1 budget above 0.50; a tighter budget is infeasible for the floor
    # rather than binding on it, which would test nothing.
    budget = 0.60
    constraints = make_constraints(asset_returns=yields_series(), target_return=0.020,
                                   benchmark_weights=benchmark,
                                   tre_utility_weight=10.0,
                                   turnover_utility_weight=5.0,
                                   weights_0=weights_0,
                                   turnover_constraint=budget)
    outcome = cvx_maximise_alpha_with_target_return(
        covar=covar_frame().to_numpy(), alphas=alphas_series().to_numpy(),
        constraints=constraints, soft_tracking_error=True)
    assert outcome.accepted
    turnover = float(np.sum(np.abs(outcome.weights - weights_0.to_numpy())))
    assert turnover <= budget + 1e-5
    residuals = [residual for residual in outcome.constraint_residuals
                 if residual.constraint_type == 'turnover']
    assert len(residuals) == 1
    assert residuals[0].hard
    assert residuals[0].passed


def test_the_soft_branch_keeps_group_and_global_turnover_hard_together(monkeypatch) -> None:
    """A workbook group limit must not suppress the independent portfolio L1 limit."""
    compilation_order = []
    compile_group_turnover = target_yield_solver._set_cvx_group_turnover_constraints
    compile_total_turnover = target_yield_solver._set_cvx_total_turnover_constraints

    def record_group_turnover(*args, **kwargs):
        """Record and delegate group-turnover compilation."""
        compilation_order.append("group")
        return compile_group_turnover(*args, **kwargs)

    def record_total_turnover(*args, **kwargs):
        """Record and delegate total-turnover compilation."""
        compilation_order.append("total")
        return compile_total_turnover(*args, **kwargs)

    monkeypatch.setattr(
        target_yield_solver,
        "_set_cvx_group_turnover_constraints",
        record_group_turnover,
    )
    monkeypatch.setattr(
        target_yield_solver,
        "_set_cvx_total_turnover_constraints",
        record_total_turnover,
    )
    benchmark = pd.Series(0.25, index=TICKERS)
    weights_0 = benchmark.copy()
    group_turnover = GroupTurnoverConstraint(
        group_loadings=pd.DataFrame(
            {'A only': [1.0, 0.0, 0.0, 0.0]}, index=TICKERS),
        group_max_turnover=pd.Series({'A only': 0.02}),
    )
    constraints = make_constraints(
        asset_returns=yields_series(),
        target_return=0.020,
        benchmark_weights=benchmark,
        tre_utility_weight=10.0,
        turnover_utility_weight=5.0,
        weights_0=weights_0,
        turnover_constraint=0.04,
        group_turnover_constraint=group_turnover,
    )

    outcome = cvx_maximise_alpha_with_target_return(
        covar=covar_frame().to_numpy(),
        alphas=alphas_series().to_numpy(),
        constraints=constraints,
        soft_tracking_error=True,
    )

    assert outcome.accepted
    assert compilation_order == ["group", "total"]
    trade = outcome.weights - weights_0.to_numpy()
    assert abs(trade[0]) <= 0.02 + 1e-6
    assert np.abs(trade).sum() <= 0.04 + 1e-6
    hard_turnover = [
        residual for residual in outcome.constraint_residuals
        if residual.constraint_type in ('turnover', 'group_turnover')
    ]
    assert hard_turnover
    assert all(residual.hard and residual.passed for residual in hard_turnover)


def test_the_soft_branch_warns_when_turnover_has_no_starting_weights() -> None:
    """A turnover budget without ``weights_0`` cannot be enforced, and says so."""
    benchmark = pd.Series(0.25, index=TICKERS)
    constraints = make_constraints(asset_returns=yields_series(), target_return=0.020,
                                   benchmark_weights=benchmark,
                                   tre_utility_weight=10.0,
                                   turnover_constraint=0.30)
    with pytest.warns(UserWarning, match='weights_0 must be given'):
        cvx_maximise_alpha_with_target_return(
            covar=covar_frame().to_numpy(), alphas=alphas_series().to_numpy(),
            constraints=constraints, soft_tracking_error=True)


def test_the_soft_branch_honours_per_asset_turnover_costs() -> None:
    """With costs set, the hard turnover bound is cost-weighted rather than plain L1."""
    benchmark = pd.Series(0.25, index=TICKERS)
    weights_0 = pd.Series([1.0, 0.0, 0.0, 0.0], index=TICKERS)
    costs = pd.Series([2.0, 1.0, 1.0, 1.0], index=TICKERS)
    constraints = make_constraints(asset_returns=yields_series(), target_return=0.020,
                                   benchmark_weights=benchmark,
                                   tre_utility_weight=10.0,
                                   weights_0=weights_0,
                                   turnover_costs=costs,
                                   turnover_constraint=1.00)
    outcome = cvx_maximise_alpha_with_target_return(
        covar=covar_frame().to_numpy(), alphas=alphas_series().to_numpy(),
        constraints=constraints, soft_tracking_error=True)
    assert outcome.accepted
    weighted = float(np.sum(np.abs(costs.to_numpy() * (outcome.weights - weights_0.to_numpy()))))
    assert weighted <= 1.00 + 1e-5


# --------------------------------------------------------------------------- #
# solver failure
# --------------------------------------------------------------------------- #
def test_a_solver_failure_becomes_a_rejected_outcome(monkeypatch) -> None:
    """A degenerate yield/alpha geometry can make CLARABEL raise instead of report.

    The rolling layer calls this once per rebalancing date, so it must degrade to a recorded
    non-accepted outcome rather than propagate and kill the run.
    """
    def fail(self, **kwargs):
        """Stand in for a solver backend that raises instead of returning a status."""
        raise cvx.error.SolverError('degenerate geometry')

    monkeypatch.setattr(cvx.Problem, 'solve', fail)
    outcome = cvx_maximise_alpha_with_target_return(
        covar=covar_frame().to_numpy(), alphas=alphas_series().to_numpy(),
        constraints=make_constraints(asset_returns=yields_series(), target_return=0.020))
    assert not outcome.accepted
    assert outcome.status == 'solver_error'


# --------------------------------------------------------------------------- #
# the rolling layer
# --------------------------------------------------------------------------- #
def rolling_inputs(**overrides) -> dict:
    """Panel inputs for the rolling solver over the two rebalancing dates."""
    kwargs = dict(prices=prices_frame(),
                  alphas=pd.DataFrame([alphas_series()] * len(DATES), index=DATES),
                  yields=pd.DataFrame([yields_series()] * len(DATES), index=DATES),
                  target_returns=pd.Series(0.020, index=DATES),
                  constraints=make_constraints(),
                  covar_dict={date: covar_frame() for date in DATES})
    kwargs.update(overrides)
    return kwargs


def test_the_rolling_solver_returns_one_row_per_rebalancing_date() -> None:
    """The covariance dict's keys define the schedule, and the output follows it."""
    weights = rolling_maximise_alpha_with_target_return(**rolling_inputs())
    assert list(weights.index) == list(DATES)
    assert list(weights.columns) == list(TICKERS)


def test_a_static_benchmark_series_is_broadcast_over_the_schedule() -> None:
    """A Series benchmark is transposed and forward-filled onto every rebalancing date."""
    weights = rolling_maximise_alpha_with_target_return(
        **rolling_inputs(benchmark_weights=pd.Series(0.25, index=TICKERS)))
    assert list(weights.index) == list(DATES)
    assert not weights.isna().any().any()


def test_a_time_varying_benchmark_frame_is_forward_filled() -> None:
    """A DataFrame benchmark starting before the schedule is carried forward, not dropped."""
    frame = pd.DataFrame([[0.25, 0.25, 0.25, 0.25]],
                         index=pd.DatetimeIndex(['2024-01-31']), columns=TICKERS)
    weights = rolling_maximise_alpha_with_target_return(
        **rolling_inputs(benchmark_weights=frame))
    assert list(weights.index) == list(DATES)


def test_alphas_and_yields_are_forward_filled_onto_the_schedule() -> None:
    """Inputs on a slower cadence than the covariances are ffilled, not reindexed to NaN."""
    sparse_dates = pd.DatetimeIndex(['2024-01-31'])
    weights = rolling_maximise_alpha_with_target_return(
        **rolling_inputs(
            alphas=pd.DataFrame([alphas_series()], index=sparse_dates),
            yields=pd.DataFrame([yields_series()], index=sparse_dates),
            target_returns=pd.Series(0.020, index=sparse_dates)))
    assert list(weights.index) == list(DATES)
    assert not weights.isna().any().any()


def test_the_verbose_config_prints_per_date_diagnostics(capsys) -> None:
    """``verbose`` is threaded into the rolling loop, not only into the solver."""
    rolling_maximise_alpha_with_target_return(
        **rolling_inputs(optimiser_config=OptimiserConfig(apply_total_to_good_ratio=True,
                                                          verbose=True)))
    printed = capsys.readouterr().out
    assert 'date=' in printed
    assert 'pd_covar=' in printed


def test_an_infeasible_first_rebalance_resets_the_drift_anchor() -> None:
    """A zero solve clears ``weights_0`` so the next date starts cold rather than drifting zeros.

    An infeasible solve at the first date has no prior weights to fall back on and returns zeros.
    Carrying those zeros forward would make them the anchor the next rebalance drifts from, so the
    subsequent solve would be warm-started and turnover-constrained against a portfolio that was
    never held. The reset is what keeps the following dates equal to a clean solve.
    """
    tickers = pd.Index(['A', 'B', 'C'])
    correlation = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    vols = np.array([0.10, 0.15, 0.22])
    covar = pd.DataFrame(correlation * np.outer(vols, vols), index=tickers, columns=tickers)
    dates = pd.date_range('2024-01-31', periods=3, freq='ME')
    prices = pd.DataFrame(
        100.0, index=pd.date_range('2023-01-31', periods=40, freq='ME'), columns=tickers)
    alphas = pd.DataFrame(0.02, index=dates, columns=tickers)
    yields = pd.DataFrame([[0.03, 0.02, 0.04]] * len(dates), index=dates, columns=tickers)
    config = OptimiserConfig(verbose=False, diagnose_infeasibility=False)
    common = dict(prices=prices, alphas=alphas, yields=yields,
                  constraints=Constraints(is_long_only=True),
                  covar_dict={date: covar for date in dates}, optimiser_config=config)

    # first date is impossible: no yield combination reaches a 500% return
    with_failure = rolling_maximise_alpha_with_target_return(
        target_returns=pd.Series([5.0, 0.035, 0.035], index=dates), **common)
    all_feasible = rolling_maximise_alpha_with_target_return(
        target_returns=pd.Series([0.035, 0.035, 0.035], index=dates), **common)

    assert np.all(with_failure.iloc[0].to_numpy() == 0.0), 'the first solve was expected to fail'
    pd.testing.assert_frame_equal(with_failure.iloc[1:], all_feasible.iloc[1:], atol=1e-8)
