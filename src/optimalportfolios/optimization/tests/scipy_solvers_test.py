"""
the scipy (SLSQP) solver paths and the CARA objective functions.

The package solves with cvxpy by default; these are the SLSQP alternatives —
``opt_risk_budgeting_scipy`` (documented as "fallback, not recommended") and
``opt_maximize_cara`` — plus the three raw objective functions the latter minimises.

They matter despite being secondary. A fallback that quietly returns its initial guess looks
exactly like a solve: `opt_maximize_cara` does precisely that when SLSQP fails to converge,
and `opt_risk_budgeting_scipy` falls back to ``weights_0`` or zeros. Untested, the difference
between "solved" and "gave up" is invisible in a backtest.

The objectives are pure functions of a weight vector, so they are checked against the formula
in their own docstrings rather than against recorded numbers: the quadratic CARA objective is
``-(μ'w - γ/2 w'Σw)``, the exponential is ``exp(-γμ'w + γ²/2 w'Σw)``, and both must move the
right way when risk aversion rises.
"""
# packages
import logging
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
from optimalportfolios import Constraints
from optimalportfolios.optimization.general.carra_mixture import (
    carra_objective, carra_objective_exp, carra_objective_mixture, opt_maximize_cara)
from optimalportfolios.optimization.risk_allocation.risk_budgeting import (
    opt_risk_budgeting_scipy, risk_budget_objective)

TICKERS = ['growth', 'balanced', 'defensive']
VOLS = np.array([0.22, 0.14, 0.06])
CORR = np.array([[1.00, 0.45, 0.15],
                 [0.45, 1.00, 0.25],
                 [0.15, 0.25, 1.00]])
COVAR = np.outer(VOLS, VOLS) * CORR
MEANS = np.array([0.09, 0.06, 0.02])


def make_constraints(**overrides) -> Constraints:
    """A long-only, fully invested constraint set over TICKERS."""
    kwargs = dict(is_long_only=True,
                  min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS))
    kwargs.update(overrides)
    return Constraints(**kwargs)


def risk_shares(weights: np.ndarray) -> np.ndarray:
    """Normalised risk contributions of a weight vector under COVAR."""
    contributions = weights * (COVAR @ weights)
    return contributions / contributions.sum()


# --------------------------------------------------------------------------- #
# risk budgeting via SLSQP
# --------------------------------------------------------------------------- #
def test_scipy_risk_budgeting_matches_the_requested_budget() -> None:
    """the SLSQP path solves the same problem as the cvxpy one, to looser tolerance"""
    budget = np.array([0.2, 0.3, 0.5])
    weights = opt_risk_budgeting_scipy(covar=COVAR, constraints=make_constraints(),
                                       risk_budget=budget, context='slsqp')
    assert weights is not None
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)
    assert (weights >= -1e-6).all()
    np.testing.assert_allclose(risk_shares(weights), budget, atol=0.03)


def test_scipy_risk_budgeting_defaults_to_equal_budgets() -> None:
    """with no budget every asset contributes the same share of risk"""
    weights = opt_risk_budgeting_scipy(covar=COVAR, constraints=make_constraints(),
                                       risk_budget=None)
    np.testing.assert_allclose(risk_shares(weights), np.full(3, 1 / 3), atol=0.03)


def test_scipy_risk_budgeting_warm_starts_from_prior_weights() -> None:
    """weights_0 seeds the search, so supplying it must not change the answer materially"""
    budget = np.array([0.2, 0.3, 0.5])
    cold = opt_risk_budgeting_scipy(covar=COVAR, constraints=make_constraints(),
                                    risk_budget=budget)
    warm = opt_risk_budgeting_scipy(
        covar=COVAR, risk_budget=budget,
        constraints=make_constraints(weights_0=pd.Series([0.2, 0.3, 0.5], index=TICKERS)))
    np.testing.assert_allclose(cold, warm, atol=0.02)


def test_scipy_risk_budgeting_excludes_a_zero_budget_asset() -> None:
    """a zero budget is turned into NaN so the asset is dropped from the objective"""
    weights = opt_risk_budgeting_scipy(covar=COVAR, constraints=make_constraints(),
                                       risk_budget=np.array([0.5, 0.5, 0.0]))
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)


def test_risk_budget_objective_is_zero_at_the_matching_portfolio() -> None:
    """the objective is a sum of squared deviations, so a perfect match scores zero"""
    budget = np.array([0.2, 0.3, 0.5])
    solved = opt_risk_budgeting_scipy(covar=COVAR, constraints=make_constraints(),
                                      risk_budget=budget)
    at_solution = risk_budget_objective(solved, [COVAR, budget])
    elsewhere = risk_budget_objective(np.array([0.8, 0.1, 0.1]), [COVAR, budget])
    assert at_solution >= 0.0
    assert at_solution < elsewhere


def test_risk_budget_objective_handles_a_nan_budget_entry() -> None:
    """a NaN budget means 'no target', and must not poison the whole objective"""
    value = risk_budget_objective(np.array([0.4, 0.35, 0.25]),
                                  [COVAR, np.array([0.5, np.nan, 0.5])])
    assert np.isfinite(value)


# --------------------------------------------------------------------------- #
# CARA objectives
# --------------------------------------------------------------------------- #
def test_quadratic_cara_objective_matches_its_formula() -> None:
    """the objective is the negated mean-variance utility, checked by hand"""
    w = np.array([0.4, 0.35, 0.25])
    carra = 2.0
    expected = -(MEANS @ w - 0.5 * carra * w @ COVAR @ w)
    assert carra_objective(w, [MEANS, COVAR, carra]) == pytest.approx(expected)


def test_exponential_cara_objective_matches_its_formula() -> None:
    """the exponential form is the certainty-equivalent transform of the same utility"""
    w = np.array([0.4, 0.35, 0.25])
    carra = 2.0
    expected = np.exp(-carra * MEANS @ w + 0.5 * carra ** 2 * w @ COVAR @ w)
    assert carra_objective_exp(w, [MEANS, COVAR, carra]) == pytest.approx(expected)


def test_both_cara_objectives_agree_on_which_portfolio_is_better() -> None:
    """the two forms are monotone transforms, so they must rank portfolios identically"""
    carra = 2.0
    a, b = np.array([0.6, 0.3, 0.1]), np.array([0.1, 0.3, 0.6])
    quadratic_prefers_a = carra_objective(a, [MEANS, COVAR, carra]) < carra_objective(
        b, [MEANS, COVAR, carra])
    exponential_prefers_a = carra_objective_exp(a, [MEANS, COVAR, carra]) < (
        carra_objective_exp(b, [MEANS, COVAR, carra]))
    assert quadratic_prefers_a == exponential_prefers_a


def test_mixture_objective_reduces_to_the_single_component_case() -> None:
    """a one-component mixture is just that component, which pins the weighting"""
    w = np.array([0.4, 0.35, 0.25])
    carra = 2.0
    mixture = carra_objective_mixture(
        w, [[MEANS], [COVAR], np.array([1.0]), carra])
    single = carra_objective_exp(w, [MEANS, COVAR, carra])
    assert mixture == pytest.approx(single, rel=1e-9)


def test_mixture_objective_averages_its_components_by_probability() -> None:
    """two components weighted 50/50 give the mean of the two single-component values"""
    w = np.array([0.4, 0.35, 0.25])
    carra = 2.0
    stressed_means = MEANS - 0.10
    mixture = carra_objective_mixture(
        w, [[MEANS, stressed_means], [COVAR, COVAR], np.array([0.5, 0.5]), carra])
    expected = 0.5 * (carra_objective_exp(w, [MEANS, COVAR, carra])
                      + carra_objective_exp(w, [stressed_means, COVAR, carra]))
    assert mixture == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# CARA maximisation via SLSQP
# --------------------------------------------------------------------------- #
def test_cara_maximisation_returns_an_investable_portfolio() -> None:
    """the quadratic branch solves to a fully invested long-only portfolio"""
    weights = opt_maximize_cara(means=MEANS, covar=COVAR, carra=2.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)
    assert (weights >= -1e-6).all()


def test_higher_risk_aversion_buys_less_volatility() -> None:
    """gamma is the whole knob: raising it must move weight to the low-vol asset"""
    bold = opt_maximize_cara(means=MEANS, covar=COVAR, carra=0.5)
    cautious = opt_maximize_cara(means=MEANS, covar=COVAR, carra=20.0)
    assert cautious[2] > bold[2]                       # more 'defensive'
    assert float(np.sqrt(cautious @ COVAR @ cautious)) < float(
        np.sqrt(bold @ COVAR @ bold))


def test_the_exponential_branch_agrees_with_the_quadratic_one() -> None:
    """the two objectives are monotone transforms, so they find the same optimum"""
    quadratic = opt_maximize_cara(means=MEANS, covar=COVAR, carra=2.0, is_exp=False)
    exponential = opt_maximize_cara(means=MEANS, covar=COVAR, carra=2.0, is_exp=True)
    np.testing.assert_allclose(quadratic, exponential, atol=0.05)


def test_cara_maximisation_respects_box_bounds() -> None:
    """the per-asset bounds are passed to SLSQP as inequality constraints"""
    weights = opt_maximize_cara(means=MEANS, covar=COVAR, carra=0.5,
                                min_weights=np.array([0.0, 0.0, 0.30]),
                                max_weights=np.array([0.40, 1.0, 1.0]))
    assert weights[2] >= 0.30 - 1e-4
    assert weights[0] <= 0.40 + 1e-4


def test_cara_maximisation_falls_back_to_the_initial_guess_when_it_cannot_converge(
        caplog) -> None:
    """a failed SLSQP run returns equal weights and says so, rather than returning junk

    This is the branch that makes the fallback visible. Contradictory bounds — a floor above
    the cap on the same asset — leave SLSQP no feasible point.
    """
    with caplog.at_level(logging.WARNING):
        weights = opt_maximize_cara(means=MEANS, covar=COVAR, carra=0.5,
                                    min_weights=np.array([0.9, 0.9, 0.9]),
                                    max_weights=np.array([0.1, 0.1, 0.1]))
    assert weights is not None
    assert np.all(np.isfinite(weights))
    if 'did not converge' in caplog.text:
        np.testing.assert_allclose(weights, np.full(3, 1 / 3))


def test_cara_maximisation_can_print_its_diagnostics(capsys) -> None:
    """the print branch is a notebook aid and must not raise"""
    opt_maximize_cara(means=MEANS, covar=COVAR, carra=2.0, is_print_log=True)
    printed = capsys.readouterr().out
    assert 'return_p' in printed and 'sigma_p' in printed
