"""
the pandas layer around the risk-budgeting solver, and its degenerate inputs.

``risk_budgeting_solver_test.py`` covers the ADMM solver itself against the Richard & Roncalli
tables. What sits above it — ``wrapper_risk_budgeting`` — is the layer that turns a labelled
universe into the arrays that solver takes: it derives eligibility from the budgets, drops
assets the covariance cannot describe, rescales the budgets over whatever is left, and puts
the frozen positions back afterwards. None of that is arithmetic the solver checks.

The cases here are the inputs that layer is there to survive: a budget given as a dict rather
than a Series, a covariance with nothing usable in it, a solve the ADMM refuses, and the
inverse problem (``solve_for_risk_budgets_from_given_weights``) on a one-asset universe and on
a search that does not converge. Each of them returns a portfolio rather than raising, which
is the point — a backtest must survive one bad rebalancing date — and each of them therefore
fails silently if the fallback is wrong.
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
from optimalportfolios.optimization.general import risk_budgeting as risk_budgeting_module
from optimalportfolios.optimization.general.risk_budgeting import (
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
                         logger='optimalportfolios.optimization.general.risk_budgeting'):
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


def test_a_search_that_does_not_converge_reports_zero_budgets(monkeypatch, caplog) -> None:
    """SLSQP failure is logged with its status and yields zeros rather than ``res.x``

    On failure SLSQP still returns an iterate, and it is one no constraint was enforced on —
    it need not sum to one or stay inside the bounds. Passing it on would give a plausible-
    looking budget vector that reproduces nothing. Non-convergence cannot be provoked from an
    input (the objective is a full rolling backtest per evaluation), so it is injected.
    """
    class _Failed:
        """The shape of a failed ``scipy.optimize`` result."""
        success = False
        status = 4
        message = 'Inequality constraints incompatible'
        x = np.array([5.0, -2.0, 0.5])

    monkeypatch.setattr(risk_budgeting_module, 'minimize',
                        lambda *args, **kwargs: _Failed())
    with caplog.at_level(logging.WARNING,
                         logger='optimalportfolios.optimization.general.risk_budgeting'):
        budgets = solve_for_risk_budgets_from_given_weights(
            prices=make_prices(), given_weights=pd.Series([0.5, 0.3, 0.2], index=TICKERS),
            covar_dict=make_covar_dict())
    assert any('SLSQP did not converge' in record.getMessage()
               for record in caplog.records)
    np.testing.assert_allclose(budgets.to_numpy(), 0.0, atol=1e-12)
    assert list(budgets.index) == TICKERS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
