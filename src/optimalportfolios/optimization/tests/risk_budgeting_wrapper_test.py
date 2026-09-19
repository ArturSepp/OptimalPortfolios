"""
the pandas layer around the risk-budgeting solver, and its degenerate inputs.

``risk_budgeting_solver_test.py`` covers the ADMM solver itself against the Richard & Roncalli
tables. What sits above it — ``wrapper_risk_budgeting`` — is the layer that turns a labelled
universe into the arrays that solver takes: it derives eligibility from the budgets, drops
assets the covariance cannot describe, rescales the budgets over whatever is left, and puts
the frozen positions back afterwards. None of that is arithmetic the solver checks.

The cases here are the inputs that layer is there to survive: a budget given as a dict rather
than a Series, a covariance with nothing usable in it, a solve the ADMM refuses, and the
inverse problem (``solve_for_risk_budgets_from_given_weights``) on degenerate universes, a
fixed-point calibration, and failed refinement. A rolling allocation survives one bad covariance
date, while inverse calibration fails explicitly rather than turning a mandate into zero budgets.
"""
# packages
import logging
from typing import Dict
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
from optimalportfolios import Constraints
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import GroupLowerUpperConstraints
from optimalportfolios.optimization.risk_allocation import (
    risk_budgeting as risk_budgeting_module,
)
from optimalportfolios.optimization.risk_allocation.risk_budgeting import (
    average_rolling_weights,
    risk_budget_objective,
    solve_for_risk_budgets_from_given_weights,
    wrapper_risk_budgeting,
)

SEED = 20260810
TICKERS = ['growth', 'balanced', 'defensive']
VOLS = np.array([0.22, 0.14, 0.06])
CORR = np.array([[1.00, 0.45, 0.15],
                 [0.45, 1.00, 0.25],
                 [0.15, 0.25, 1.00]])
COVAR = np.outer(VOLS, VOLS) * CORR
COVAR_DF = pd.DataFrame(COVAR, index=TICKERS, columns=TICKERS)
EQUAL_BUDGET = {ticker: 1.0 / len(TICKERS) for ticker in TICKERS}
REBALANCING_DATES = pd.DatetimeIndex(['2024-03-31', '2024-06-30', '2024-09-30'])


def make_prices(n_days: int = 300) -> pd.DataFrame:
    """A seeded daily price panel over TICKERS."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range('2023-06-01', periods=n_days, freq='B')
    returns = rng.multivariate_normal(np.full(3, 0.0003), COVAR / 260.0, size=n_days)
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=TICKERS)


def make_covar_dict() -> Dict[pd.Timestamp, pd.DataFrame]:
    """The same covariance at every rebalancing date."""
    return {date: COVAR_DF for date in REBALANCING_DATES}


def long_only(**overrides) -> Constraints:
    """Fully invested, long only."""
    kwargs = dict(is_long_only=True, min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS))
    kwargs.update(overrides)
    return Constraints(**kwargs)


# --------------------------------------------------------------------------- #
# how the budgets are stated
# --------------------------------------------------------------------------- #
def test_a_budget_given_as_a_dict_is_the_same_as_one_given_as_a_series() -> None:
    """a plain dict is accepted, because that is how a config file states a budget"""
    from_dict = wrapper_risk_budgeting(pd_covar=COVAR_DF, constraints=long_only(),
                                       risk_budget=EQUAL_BUDGET)
    from_series = wrapper_risk_budgeting(pd_covar=COVAR_DF, constraints=long_only(),
                                         risk_budget=pd.Series(EQUAL_BUDGET))
    pd.testing.assert_series_equal(from_dict, from_series)


def test_a_budget_that_is_neither_a_dict_nor_a_series_is_rejected() -> None:
    """the type is named in the error, because a bare list would otherwise index by position

    A list has no asset labels, so a silent ``pd.Series(list)`` would attach the budgets to
    0, 1, 2 and every asset would end up with a NaN budget — which reads downstream as
    "excluded", not as "misstated".
    """
    with pytest.raises(NotImplementedError, match='list'):
        wrapper_risk_budgeting(pd_covar=COVAR_DF, constraints=long_only(),
                               risk_budget=[0.3, 0.3, 0.4])


def test_a_zero_budget_asset_is_excluded_rather_than_solved_for() -> None:
    """eligibility is derived from the budgets: a zero budget means "do not hold" """
    budget = pd.Series([0.5, 0.5, 0.0], index=TICKERS)
    weights = wrapper_risk_budgeting(pd_covar=COVAR_DF, constraints=long_only(),
                                     risk_budget=budget)
    assert weights[TICKERS[2]] == pytest.approx(0.0, abs=1e-8)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# a universe the covariance cannot describe
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('cash_variance', [1e-10, 0.001**2, 4e-6])
def test_cash_variance_floor_matches_diagonal_risk_budget_solution(cash_variance) -> None:
    """Cash-like risk budgeting matches the independent diagonal closed-form solution."""
    variances = np.array([0.04, 0.01, cash_variance])
    covar = pd.DataFrame(np.diag(variances), index=TICKERS, columns=TICKERS)
    original = covar.copy(deep=True)
    budgets = pd.Series([0.5, 0.49, 0.01], index=TICKERS)

    weights = wrapper_risk_budgeting(
        pd_covar=covar, constraints=long_only(), risk_budget=budgets)

    expected = np.sqrt(budgets.to_numpy() / np.maximum(variances, 0.001**2))
    expected /= expected.sum()
    np.testing.assert_allclose(weights.to_numpy(), expected, atol=1e-6, rtol=0.0)
    pd.testing.assert_frame_equal(covar, original)


@pytest.mark.parametrize('invalid_variance', [0.0, -1e-8, np.nan])
def test_cash_floor_preserves_covariances_and_excludes_invalid_assets(
        monkeypatch, invalid_variance) -> None:
    """Only eligible positive diagonal entries are floored before the solver sees them."""
    covar = pd.DataFrame(
        [[0.04, -1e-5, 0.0], [-1e-5, 1e-8, 0.0], [0.0, 0.0, invalid_variance]],
        index=TICKERS, columns=TICKERS)
    original = covar.copy(deep=True)

    def check_solver_covariance(covar, **kwargs):
        """Inspect the actual solver input after covariance-universe filtering."""
        np.testing.assert_array_equal(covar, [[0.04, -1e-5], [-1e-5, 0.001**2]])
        return np.array([0.4, 0.6])

    monkeypatch.setattr(risk_budgeting_module, 'opt_risk_budgeting', check_solver_covariance)
    weights = wrapper_risk_budgeting(
        pd_covar=covar, constraints=long_only(), risk_budget=pd.Series(EQUAL_BUDGET))
    np.testing.assert_array_equal(weights.to_numpy(), [0.4, 0.6, 0.0])
    pd.testing.assert_frame_equal(covar, original)


def test_a_covariance_with_no_usable_asset_returns_a_flat_zero_portfolio() -> None:
    """with nothing left to allocate to, the date produces no position and says so

    An all-NaN covariance is what a rebalancing date before any asset has history looks like.
    Returning zeros lets the rolling wrapper record "no portfolio here" and carry on; raising
    would end the backtest at its first date.
    """
    dead = COVAR_DF.copy()
    dead.loc[:, :] = np.nan
    with pytest.warns(UserWarning, match='no valid assets in covariance matrix'):
        weights = wrapper_risk_budgeting(pd_covar=dead, constraints=long_only(),
                                         risk_budget=pd.Series(EQUAL_BUDGET))
    assert list(weights.index) == TICKERS
    np.testing.assert_allclose(weights.to_numpy(), 0.0, atol=1e-12)


def test_rescaling_over_the_surviving_assets_can_be_switched_off() -> None:
    """without the rescale the budgets are left as stated over the reduced universe

    ``apply_total_to_good_ratio`` scales the budgets of the survivors by N_total / N_valid so
    a dropped asset's share is redistributed. Off, the raw budgets go through — they no longer
    sum to one over the survivors, and the solver normalises. Either is defensible; what
    matters is that the flag actually reaches the wrapper.
    """
    dead = COVAR_DF.copy()
    dead.loc[TICKERS[2], :] = np.nan
    dead.loc[:, TICKERS[2]] = np.nan
    budget = pd.Series([0.5, 0.5, 0.0], index=TICKERS)
    kwargs = dict(pd_covar=dead, constraints=long_only(), risk_budget=budget)
    rescaled = wrapper_risk_budgeting(
        optimiser_config=OptimiserConfig(apply_total_to_good_ratio=True), **kwargs)
    raw = wrapper_risk_budgeting(
        optimiser_config=OptimiserConfig(apply_total_to_good_ratio=False), **kwargs)
    for weights in (rescaled, raw):
        assert weights[TICKERS[2]] == pytest.approx(0.0, abs=1e-8)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_detailed_output_reports_the_realised_risk_contributions() -> None:
    """the diagnostic form returns the contributions next to the weights, not just weights"""
    detailed = wrapper_risk_budgeting(pd_covar=COVAR_DF, constraints=long_only(),
                                      risk_budget=pd.Series(EQUAL_BUDGET),
                                      detailed_output=True)
    assert isinstance(detailed, pd.DataFrame)
    assert list(detailed.index) == TICKERS
    assert len(detailed.columns) > 1


# --------------------------------------------------------------------------- #
# a solve the ADMM refuses
# --------------------------------------------------------------------------- #
def test_a_solver_rejection_is_logged_and_falls_back(caplog) -> None:
    """an infeasible box makes the ADMM raise, and the wrapper must not propagate it

    ``solve_constrained_risk_budgeting`` raises ``ValueError`` on a box that cannot reach full
    investment. Inside a rolling backtest that is one bad date, not a reason to stop, so it is
    logged with its context and routed into the same validation fallback as any other failure.
    """
    unreachable = long_only(max_weights=pd.Series(0.2, index=TICKERS))
    with caplog.at_level(logging.WARNING,
                         logger='optimalportfolios.optimization.risk_allocation.risk_budgeting'):
        weights = wrapper_risk_budgeting(pd_covar=COVAR_DF, constraints=unreachable,
                                         risk_budget=pd.Series(EQUAL_BUDGET),
                                         context='2024-03-31')
    assert any('opt_risk_budgeting: solver failed' in record.getMessage()
               for record in caplog.records)
    assert any('2024-03-31' in record.getMessage() for record in caplog.records)
    assert np.isfinite(weights.to_numpy()).all()


def test_verbose_reports_the_group_constraint_slack(capsys) -> None:
    """the verbose path prints how much room is left on each group row

    Only reachable with group constraints present — the slack is of ``C w - d``, and without
    a group block there are no rows to report.
    """
    loadings = pd.DataFrame({'risky': [1.0, 1.0, 0.0], 'safe': [0.0, 0.0, 1.0]},
                            index=TICKERS)
    groups = GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series([0.0, 0.0], index=['risky', 'safe']),
        group_max_allocation=pd.Series([0.7, 1.0], index=['risky', 'safe']))
    wrapper_risk_budgeting(
        pd_covar=COVAR_DF, constraints=long_only(group_lower_upper_constraints=groups),
        risk_budget=pd.Series(EQUAL_BUDGET),
        optimiser_config=OptimiserConfig(verbose=True))
    assert 'slack=' in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the scipy objective
# --------------------------------------------------------------------------- #
def test_the_scipy_objective_defaults_to_an_equal_budget() -> None:
    """with no budget stated the objective is equal risk contribution

    ``budget=None`` is not "no target" — it is the ERC target, an equal share for every asset.
    The two are asserted to agree, so the default cannot drift away from the explicit form.
    """
    weights = np.array([0.2, 0.3, 0.5])
    implicit = risk_budget_objective(weights, [COVAR, None])
    explicit = risk_budget_objective(weights, [COVAR, np.full(len(TICKERS),
                                                              1.0 / len(TICKERS))])
    assert implicit == pytest.approx(explicit, abs=1e-15)
    assert implicit > 0.0, 'these weights are not the ERC portfolio, so the gap is not zero'


# --------------------------------------------------------------------------- #
# the inverse problem
# --------------------------------------------------------------------------- #
def test_a_one_asset_universe_skips_the_search_entirely() -> None:
    """the only budget summing to one is 1.0, and the cap of 0.99 would forbid it

    Short-circuiting is not an optimisation here: the search is bounded above by
    ``max_risk_budget=0.99``, so a single asset makes the constrained problem infeasible and
    the solver would return the zero budgets its non-convergence branch falls back to.
    """
    prices = make_prices()[[TICKERS[0]]]
    covar_dict = {date: COVAR_DF.loc[[TICKERS[0]], [TICKERS[0]]]
                  for date in REBALANCING_DATES}
    budgets = solve_for_risk_budgets_from_given_weights(
        prices=prices, given_weights=pd.Series([1.0], index=[TICKERS[0]]),
        covar_dict=covar_dict)
    assert budgets.to_dict() == {TICKERS[0]: 1.0}


def test_a_good_fixed_point_skips_the_slow_slsqp_search(monkeypatch) -> None:
    """a valid inverse budget is returned before the fragile SLSQP refinement

    The inverse objective runs a complete rolling risk-budget solve on every evaluation.
    A bounded fixed-point iteration can recover this well-conditioned case directly, so the
    generic search must not be called and must never replace a useful answer with zeros.
    """
    def fail_if_called(*args, **kwargs):
        """Make an unnecessary SLSQP refinement visible to the test."""
        raise AssertionError('SLSQP should not run after fixed-point convergence')

    monkeypatch.setattr(risk_budgeting_module, 'minimize', fail_if_called)
    prices = make_prices()
    covar_dict = make_covar_dict()
    given = pd.Series([0.5, 0.3, 0.2], index=TICKERS)
    budgets = solve_for_risk_budgets_from_given_weights(
        prices=prices, given_weights=given, covar_dict=covar_dict)
    realised = risk_budgeting_module.rolling_risk_budgeting(
        prices=prices, constraints=Constraints(is_long_only=True),
        risk_budget=budgets, covar_dict=covar_dict).mean(axis=0)
    assert budgets.sum() == pytest.approx(1.0, abs=1e-10)
    assert float(np.mean(np.abs(realised - given))) <= 1e-4
    assert list(budgets.index) == TICKERS


def test_inferred_budgets_reproduce_weights_inside_asset_and_group_bands(
        monkeypatch) -> None:
    """The CCD inverse solution survives a banded ADMM forward solve."""
    def fail_if_called(*args, **kwargs):
        """Make an unnecessary SLSQP refinement visible to the test."""
        raise AssertionError('SLSQP should not run after fixed-point convergence')

    monkeypatch.setattr(risk_budgeting_module, 'minimize', fail_if_called)
    prices = make_prices()
    covar_dict = make_covar_dict()
    target = pd.Series([0.5, 0.3, 0.2], index=TICKERS)
    budgets = solve_for_risk_budgets_from_given_weights(
        prices=prices, given_weights=target, covar_dict=covar_dict)

    min_weights = pd.Series([0.45, 0.25, 0.20], index=TICKERS)
    max_weights = pd.Series([0.50, 0.35, 0.25], index=TICKERS)
    group_loadings = pd.DataFrame(
        {'risky': [1.0, 1.0, 0.0], 'safe': [0.0, 0.0, 1.0]}, index=TICKERS)
    group_mins = pd.Series({'risky': 0.76, 'safe': 0.20})
    group_maxs = pd.Series({'risky': 0.80, 'safe': 0.24})
    groups = GroupLowerUpperConstraints(
        group_loadings=group_loadings,
        group_min_allocation=group_mins,
        group_max_allocation=group_maxs)
    constraints = long_only(
        min_weights=min_weights,
        max_weights=max_weights,
        group_lower_upper_constraints=groups)

    realised = risk_budgeting_module.rolling_risk_budgeting(
        prices=prices,
        constraints=constraints,
        risk_budget=budgets,
        covar_dict=covar_dict)
    average = realised.mean(axis=0)
    errors = (average - target).abs()
    group_weights = realised @ group_loadings

    assert budgets.sum() == pytest.approx(1.0, abs=1e-10)
    assert float(errors.mean()) <= risk_budgeting_module._INVERSE_MEAN_WEIGHT_TOL
    assert float(errors.max()) <= risk_budgeting_module._INVERSE_MAX_WEIGHT_TOL
    assert realised.ge(min_weights - 1e-8).all().all()
    assert realised.le(max_weights + 1e-8).all().all()
    assert group_weights.ge(group_mins - 1e-8).all().all()
    assert group_weights.le(group_maxs + 1e-8).all().all()
    np.testing.assert_allclose(realised['growth'], max_weights['growth'], atol=1e-5)
    np.testing.assert_allclose(realised['defensive'], min_weights['defensive'], atol=1e-5)
    np.testing.assert_allclose(group_weights['risky'], group_maxs['risky'], atol=1e-5)
    np.testing.assert_allclose(group_weights['safe'], group_mins['safe'], atol=1e-5)


def test_fixed_point_applies_multiplicative_updates(monkeypatch) -> None:
    """time-varying average weights are corrected before the second evaluation"""
    given = np.array([0.5, 0.3, 0.2])
    evaluations = iter([
        (0.05, 0.1, np.array([0.4, 0.4, 0.2])),
        (0.0, 0.0, given),
    ])
    monkeypatch.setattr(
        risk_budgeting_module, '_evaluate_inverse_risk_budget',
        lambda **_kwargs: next(evaluations))
    budgets, mean_error, max_error, iteration = (
        risk_budgeting_module._solve_inverse_risk_budget_fixed_point(
            prices=make_prices(),
            given_weights=given,
            covar_dict=make_covar_dict(),
            initial_risk_budgets=np.full(3, 1.0 / 3.0),
            lower_bounds=np.full(3, 1e-4),
            upper_bounds=np.full(3, 0.99)))
    assert iteration == 2
    assert mean_error == 0.0
    assert max_error == 0.0
    assert budgets.sum() == pytest.approx(1.0, abs=1e-12)
    assert budgets[0] > budgets[1]


def test_non_finite_forward_weights_stop_fixed_point_calibration(monkeypatch) -> None:
    """a broken forward solve is retained as an explicit infinite calibration error"""
    monkeypatch.setattr(
        risk_budgeting_module, 'rolling_risk_budgeting',
        lambda **_kwargs: pd.DataFrame(np.nan, index=REBALANCING_DATES, columns=TICKERS))
    initial = np.full(3, 1.0 / 3.0)
    budgets, mean_error, max_error, iteration = (
        risk_budgeting_module._solve_inverse_risk_budget_fixed_point(
            prices=make_prices(),
            given_weights=np.array([0.5, 0.3, 0.2]),
            covar_dict=make_covar_dict(),
            initial_risk_budgets=initial,
            lower_bounds=np.full(3, 1e-4),
            upper_bounds=np.full(3, 0.99)))
    np.testing.assert_allclose(budgets, initial, atol=1e-12)
    assert np.isinf(mean_error)
    assert np.isinf(max_error)
    assert iteration == 0


def test_inverse_budget_bounds_must_reach_full_investment() -> None:
    """an infeasible upper-bound sum raises a direct configuration error"""
    with pytest.raises(ValueError, match='bounds are infeasible'):
        risk_budgeting_module._scale_to_box_simplex(
            values=np.ones(3),
            lower_bounds=np.zeros(3),
            upper_bounds=np.full(3, 0.3))


def test_inverse_target_weights_are_validated_before_calibration() -> None:
    """negative target weights cannot enter the long-only inverse solver"""
    with pytest.raises(ValueError, match='finite, non-negative'):
        solve_for_risk_budgets_from_given_weights(
            prices=make_prices(),
            given_weights=pd.Series([1.1, -0.1, 0.0], index=TICKERS),
            covar_dict=make_covar_dict())


def test_average_rolling_weights_matches_the_explicit_recursion() -> None:
    """the exponential average is the first-row-seeded EWMA recursion at the last rebalance

    The reference is computed a different way from ``qis.compute_ewm``: an explicit Python
    loop m_0 = w_0, m_t = lambda m_{t-1} + (1 - lambda) w_t with lambda = 1 - 2 / (span + 1),
    and, equivalently, the closed-form weights (1 - lambda) lambda^(T-1-t) with the residual
    lambda^(T-1) on the first row.
    """
    dates = pd.date_range('2024-03-31', periods=5, freq='QE')
    path = pd.DataFrame({'growth': [0.6, 0.5, 0.4, 0.3, 0.2],
                         'balanced': [0.2, 0.3, 0.0, 0.3, 0.3],
                         'defensive': [0.2, 0.2, 0.6, 0.4, 0.5]}, index=dates)
    span = 3.0
    decay = 1.0 - 2.0 / (span + 1.0)
    expected = {}
    for column in path.columns:
        values = path[column].to_numpy()
        state = float(values[0])
        for value in values[1:]:
            state = decay * state + (1.0 - decay) * value
        expected[column] = state
        closed_form = (1.0 - decay) * decay ** np.arange(len(values))[::-1]
        closed_form[0] = decay ** (len(values) - 1)
        assert closed_form.sum() == pytest.approx(1.0, abs=1e-12)
        assert float(closed_form @ values) == pytest.approx(state, abs=1e-12)
    averaged = average_rolling_weights(weights=path, ewma_span=span)
    assert list(averaged.index) == list(path.columns)
    np.testing.assert_allclose(averaged.to_numpy(), pd.Series(expected).to_numpy(), atol=1e-12)
    # the most recent rebalance carries the largest weight
    assert averaged['growth'] < path['growth'].mean()
    # None restores the simple mean
    pd.testing.assert_series_equal(average_rolling_weights(weights=path, ewma_span=None),
                                   path.mean(axis=0))


@pytest.mark.parametrize('span', [0.0, -1.0, np.inf, np.nan])
def test_average_rolling_weights_rejects_a_non_positive_span(span) -> None:
    """a span that is not a finite positive number is a configuration error"""
    path = pd.DataFrame(np.full((3, 3), 1.0 / 3.0), index=REBALANCING_DATES, columns=TICKERS)
    with pytest.raises(ValueError, match='ewma_span must be None or a finite positive number'):
        average_rolling_weights(weights=path, ewma_span=span)


def test_inverse_evaluation_fits_the_recent_rebalances(monkeypatch) -> None:
    """the inverse error is measured on the exponentially averaged path, not the flat mean

    A path that reaches the target only in its recent rebalances is a near-perfect fit under a
    short span and a poor fit under the simple mean, which is what lets a regime change in the
    covariance be fitted at all.
    """
    target = np.array([0.5, 0.3, 0.2])
    dates = pd.date_range('2020-03-31', periods=12, freq='QE')
    early = np.tile([0.2, 0.2, 0.6], (8, 1))
    recent = np.tile(target, (4, 1))
    path = pd.DataFrame(np.vstack([early, recent]), index=dates, columns=TICKERS)
    monkeypatch.setattr(risk_budgeting_module, 'rolling_risk_budgeting', lambda **_kwargs: path)
    mean_error_flat, max_error_flat, _ = risk_budgeting_module._evaluate_inverse_risk_budget(
        prices=make_prices(), given_weights=target, covar_dict=make_covar_dict(),
        risk_budgets=np.full(3, 1.0 / 3.0), ewma_span=None)
    mean_error_ewma, max_error_ewma, _ = risk_budgeting_module._evaluate_inverse_risk_budget(
        prices=make_prices(), given_weights=target, covar_dict=make_covar_dict(),
        risk_budgets=np.full(3, 1.0 / 3.0), ewma_span=1.0)
    assert max_error_flat > risk_budgeting_module._INVERSE_MAX_WEIGHT_TOL
    assert mean_error_ewma == pytest.approx(0.0, abs=1e-12)
    assert max_error_ewma == pytest.approx(0.0, abs=1e-12)


def test_inverse_fit_threads_the_span_into_every_evaluation(monkeypatch) -> None:
    """the seed, the fixed point and the SLSQP check all average with the caller's span"""
    seen = []

    def record_evaluation(**kwargs):
        """Capture the span each evaluation was asked to average with."""
        seen.append(kwargs['ewma_span'])
        return 0.0, 0.0, kwargs['given_weights']

    monkeypatch.setattr(risk_budgeting_module, '_evaluate_inverse_risk_budget', record_evaluation)
    budgets = solve_for_risk_budgets_from_given_weights(
        prices=make_prices(), given_weights=pd.Series([0.5, 0.3, 0.2], index=TICKERS),
        covar_dict=make_covar_dict(), ewma_span=7.0)
    assert seen == [7.0]
    assert budgets.sum() == pytest.approx(1.0, abs=1e-10)
    with pytest.raises(ValueError, match='ewma_span must be None or a finite positive number'):
        solve_for_risk_budgets_from_given_weights(
            prices=make_prices(), given_weights=pd.Series([0.5, 0.3, 0.2], index=TICKERS),
            covar_dict=make_covar_dict(), ewma_span=0.0)


# A hedging asset: 'defensive' moves against both risky assets strongly enough that its
# marginal risk contribution at 50/30/20 is negative. The matrix is positive definite.
HEDGE_CORR = np.array([[1.00, 0.50, -0.80],
                       [0.50, 1.00, -0.80],
                       [-0.80, -0.80, 1.00]])
HEDGE_COVAR_DF = pd.DataFrame(np.outer(VOLS, VOLS) * HEDGE_CORR, index=TICKERS, columns=TICKERS)


def test_a_target_held_by_a_hedging_asset_is_refused_before_the_search(monkeypatch) -> None:
    """no non-negative budget holds an asset whose marginal risk contribution is negative

    The reference is the identity w_i (Σw)_i = b_i σ_p²: with (Σw)_i < 0 at the target on
    every date the required budget is negative, so the fit is refused before any solve.
    """
    given = pd.Series([0.5, 0.3, 0.2], index=TICKERS)
    marginal = HEDGE_COVAR_DF.to_numpy() @ given.to_numpy()
    assert np.linalg.eigvalsh(HEDGE_COVAR_DF.to_numpy()).min() > 0.0
    assert marginal[2] < 0.0 and marginal[0] > 0.0 and marginal[1] > 0.0

    def fail_if_called(**kwargs):
        """The search must not start on a target that cannot be reproduced."""
        raise AssertionError('the fixed point must not run on an irreproducible target')

    monkeypatch.setattr(risk_budgeting_module, '_solve_inverse_risk_budget_fixed_point',
                        fail_if_called)
    covar_dict = {date: HEDGE_COVAR_DF for date in REBALANCING_DATES}
    with pytest.raises(ValueError, match='marginal risk contribution of defensive') as excinfo:
        solve_for_risk_budgets_from_given_weights(
            prices=make_prices(), given_weights=given, covar_dict=covar_dict)
    message = str(excinfo.value)
    assert 'defensive: target weight 0.2000' in message
    assert 'negative on 100% of dates' in message
    assert 'growth' not in message.split('targeted assets with')[1]


def test_a_hedging_asset_on_most_dates_warns_but_the_fit_proceeds(monkeypatch) -> None:
    """a negative contribution on a majority of dates is a warning, not a refusal

    The averaged contribution is positive because the short span weights the recent
    dates, where the asset carries risk; the early dates, where it hedges, are the
    majority and are reported.
    """
    given = pd.Series([0.5, 0.3, 0.2], index=TICKERS)
    dates = pd.date_range('2020-03-31', periods=8, freq='QE')
    covar_dict = {date: (HEDGE_COVAR_DF if k < 5 else COVAR_DF) for k, date in enumerate(dates)}
    seen = {}

    def converged_fixed_point(**kwargs):
        """Record the seed and report immediate convergence."""
        seen['seed'] = kwargs['initial_risk_budgets']
        return kwargs['initial_risk_budgets'], 0.0, 0.0, 1

    monkeypatch.setattr(risk_budgeting_module, '_solve_inverse_risk_budget_fixed_point',
                        converged_fixed_point)
    with pytest.warns(UserWarning, match='defensive has a negative marginal risk contribution'):
        budgets = solve_for_risk_budgets_from_given_weights(
            prices=make_prices(), given_weights=given, covar_dict=covar_dict, ewma_span=1.0)
    assert budgets.sum() == pytest.approx(1.0, abs=1e-10)
    assert np.all(seen['seed'] > 0.0)
    # the diagnostics table reports the hedging asset and the share of dates
    diagnostics = risk_budgeting_module._target_risk_contributions(
        given_weights=given, covar_dict=covar_dict, ewma_span=1.0)
    assert diagnostics.loc['defensive', 'negative_rc_share'] == pytest.approx(5 / 8)
    assert diagnostics.loc['defensive', 'average_rc'] > 0.0
    assert diagnostics.loc['growth', 'negative_rc_share'] == 0.0
    assert '62% of dates' in risk_budgeting_module._describe_target_risk_contributions(diagnostics)


def test_a_clean_target_reports_no_hedging_asset() -> None:
    """the description says so when every targeted asset carries risk on every date"""
    diagnostics = risk_budgeting_module._target_risk_contributions(
        given_weights=pd.Series([0.5, 0.3, 0.2], index=TICKERS),
        covar_dict=make_covar_dict(), ewma_span=None)
    assert (diagnostics['negative_rc_share'] == 0.0).all()
    assert diagnostics['average_rc'].sum() == pytest.approx(1.0, abs=1e-12)
    description = risk_budgeting_module._describe_target_risk_contributions(diagnostics)
    assert description == 'no targeted asset has a negative marginal risk contribution on any date'


def test_one_weighted_asset_in_a_larger_panel_has_the_only_budget() -> None:
    """the 0.99 generic cap does not make a one-leg allocation infeasible"""
    budgets = solve_for_risk_budgets_from_given_weights(
        prices=make_prices(),
        given_weights=pd.Series([0.0, 1.0, 0.0], index=TICKERS),
        covar_dict=make_covar_dict())
    assert budgets.to_dict() == {'growth': 0.0, 'balanced': 1.0, 'defensive': 0.0}


def test_slsqp_can_refine_a_fixed_point_outside_tolerance(monkeypatch) -> None:
    """a valid SLSQP refinement is checked with the forward solver before return"""
    candidate = np.array([0.4, 0.35, 0.25])
    monkeypatch.setattr(
        risk_budgeting_module, '_solve_inverse_risk_budget_fixed_point',
        lambda **_kwargs: (candidate, 0.02, 0.04, 50))
    monkeypatch.setattr(
        risk_budgeting_module, '_evaluate_inverse_risk_budget',
        lambda **_kwargs: (0.0, 0.0, np.array([0.5, 0.3, 0.2])))

    class _Success:
        """The shape of a successful ``scipy.optimize`` result."""
        success = True
        status = 0
        message = 'Optimization terminated successfully'
        x = candidate

    def successful_minimize(objective, x0, **_kwargs):
        """Exercise the nested objective before returning a valid result."""
        assert objective(x0) == 0.0
        return _Success()

    monkeypatch.setattr(risk_budgeting_module, 'minimize', successful_minimize)
    budgets = solve_for_risk_budgets_from_given_weights(
        prices=make_prices(),
        given_weights=pd.Series([0.5, 0.3, 0.2], index=TICKERS),
        covar_dict=make_covar_dict())
    np.testing.assert_allclose(budgets.to_numpy(), candidate, atol=1e-12)


def test_failed_inverse_calibration_raises_instead_of_returning_zeros(monkeypatch) -> None:
    """both solver diagnostics are reported when no candidate reproduces the target"""
    candidate = np.array([0.4, 0.35, 0.25])
    monkeypatch.setattr(
        risk_budgeting_module, '_solve_inverse_risk_budget_fixed_point',
        lambda **_kwargs: (candidate, 0.02, 0.04, 50))

    class _Failed:
        """The shape of a failed ``scipy.optimize`` result."""
        success = False
        status = 9
        message = 'Iteration limit reached'
        x = candidate

    monkeypatch.setattr(risk_budgeting_module, 'minimize',
                        lambda *args, **kwargs: _Failed())
    with pytest.raises(RuntimeError, match='No zero risk-budget fallback was returned') as excinfo:
        solve_for_risk_budgets_from_given_weights(
            prices=make_prices(),
            given_weights=pd.Series([0.5, 0.3, 0.2], index=TICKERS),
            covar_dict=make_covar_dict())
    assert str(excinfo.value).endswith(
        'no targeted asset has a negative marginal risk contribution on any date')
