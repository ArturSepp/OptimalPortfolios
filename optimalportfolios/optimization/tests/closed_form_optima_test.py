"""
solvers checked against optima that can be written down.

Every other test in this package checks that a solver runs, returns the right shape, and respects
the constraints it was given. None of them checks that the number it returns is the right number.
A solver that quietly converges to the wrong point satisfies all of those and is the failure mode
`AGENTS.md` warns about: published results depend on these defaults.

Where a problem has an analytic solution, this module solves it twice — once through the package
and once from the formula — and compares. With `Σ` the covariance, `μ` expected returns, `1` a
vector of ones, and `A = 1'Σ⁻¹1`, `B = 1'Σ⁻¹μ`, `C = μ'Σ⁻¹μ`, `D = AC - B²`:

    minimum variance, fully invested          w = Σ⁻¹1 / A
    maximum Sharpe, fully invested            w = Σ⁻¹μ / B
    minimum variance at target return r       w = [(C - rB)Σ⁻¹1 + (rA - B)Σ⁻¹μ] / D
    maximum return at target vol σ            w = Σ⁻¹1/A + k Σ⁻¹(μ - (B/A)1),  k set so vol = σ
    maximum alpha over tracking error τ       w = w_b + τ Σ⁻¹α̃ / √(α̃'Σ⁻¹Σ Σ⁻¹α̃),
                                              α̃ = α - (1'Σ⁻¹α / A)1, the budget-preserving tilt

Two more have a closed form only in a special case, which is enough to pin them: under equal
pairwise correlation both equal risk contribution and maximum diversification reduce to weights
proportional to `1/σ_i`. Away from that case they are checked by the property that defines them.

`MAX_CARA_MIXTURE` has no closed form and is checked by properties only.

The covariance matrices here are built inline rather than taken from the committed fixture. The
references need exact, well-conditioned inputs; a real panel gives neither and would turn an exact
comparison into a tolerance argument. Tolerances below are set from the measured agreement, an
order of magnitude looser than observed, so a genuine regression fails and solver-version jitter
does not.
"""
# packages
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# optimalportfolios
import optimalportfolios as op
from optimalportfolios import Constraints, PortfolioObjective

TICKERS = ['A', 'B', 'C', 'D']
VOLS = np.array([0.10, 0.15, 0.20, 0.25])
EQUAL_RHO = 0.30

# measured agreement at the time of writing, on qis 5.0.5 and 5.3.0:
#   min variance 6.9e-14, max Sharpe 5.7e-11, target return 1.7e-07, target vol 6.9e-10,
#   alpha/TRE 8.1e-10, equal risk contribution 2.2e-07, max diversification 5.2e-06
ANALYTIC_TOL = 1e-8      # solvers that reach the formula to machine-ish precision
ITERATIVE_TOL = 1e-4     # solvers that iterate to a tolerance of their own


def _equal_correlation_universe() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    covariance with equal pairwise correlation and distinct volatilities.

    Returns:
        the covariance as a labelled frame, and as a plain array
    """
    corr = np.full((len(TICKERS), len(TICKERS)), EQUAL_RHO)
    np.fill_diagonal(corr, 1.0)
    sigma = np.outer(VOLS, VOLS) * corr
    return pd.DataFrame(sigma, index=TICKERS, columns=TICKERS), sigma


def _general_universe() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    covariance with unequal correlations, still well conditioned.

    Returns:
        the covariance as a labelled frame, and as a plain array
    """
    corr = np.array([[1.00, 0.20, 0.50, -0.10],
                     [0.20, 1.00, 0.30, 0.15],
                     [0.50, 0.30, 1.00, 0.25],
                     [-0.10, 0.15, 0.25, 1.00]])
    sigma = np.outer(VOLS, VOLS) * corr
    assert np.linalg.eigvalsh(sigma).min() > 0, 'test universe must be positive definite'
    return pd.DataFrame(sigma, index=TICKERS, columns=TICKERS), sigma


def _means() -> pd.Series:
    """expected returns, distinct and ordered so a tilt is visible."""
    return pd.Series([0.03, 0.05, 0.07, 0.09], index=TICKERS)


def _abc(sigma: np.ndarray, mu: np.ndarray) -> Tuple[float, float, float, float]:
    """the four efficient-frontier scalars A, B, C and D = AC - B^2."""
    inv = np.linalg.inv(sigma)
    one = np.ones(len(mu))
    a = float(one @ inv @ one)
    b = float(one @ inv @ mu)
    c = float(mu @ inv @ mu)
    return a, b, c, a * c - b * b


def _long_short_fully_invested() -> Constraints:
    """the constraint set under which the closed forms hold: sum to one, sign unrestricted."""
    return Constraints(is_long_only=False, min_exposure=1.0, max_exposure=1.0)


# ───────────────────────────────────────────────────────────────────────────────
# General solvers
# ───────────────────────────────────────────────────────────────────────────────


def test_min_variance_matches_the_closed_form() -> None:
    """w = Σ⁻¹1 / (1'Σ⁻¹1)."""
    covar, sigma = _general_universe()
    weights, _ = op.wrapper_quadratic_optimisation(
        pd_covar=covar,
        constraints=_long_short_fully_invested(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)
    inv, one = np.linalg.inv(sigma), np.ones(len(TICKERS))
    reference = inv @ one / (one @ inv @ one)
    np.testing.assert_allclose(np.asarray(weights, dtype=float), reference, atol=ANALYTIC_TOL)


def test_min_variance_is_a_minimum_not_merely_a_solution() -> None:
    """
    every budget-preserving perturbation increases the variance.

    A closed-form comparison catches a wrong answer. This catches the case where the formula and
    the solver are wrong in the same way, which a shared derivation makes possible.
    """
    covar, sigma = _general_universe()
    weights = np.asarray(op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=_long_short_fully_invested(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)[0], dtype=float)
    variance = weights @ sigma @ weights
    rng = np.random.default_rng(11)
    for _ in range(25):
        step = rng.normal(size=len(TICKERS))
        step -= step.mean()                      # keep the budget at one
        perturbed = weights + 1e-3 * step
        assert perturbed @ sigma @ perturbed >= variance - 1e-14, (
            'a budget-preserving perturbation reduced the variance, so the reported point is not '
            'the minimum')


def test_max_sharpe_matches_the_tangency_closed_form() -> None:
    """w = Σ⁻¹μ / (1'Σ⁻¹μ)."""
    covar, sigma = _general_universe()
    mu = _means()
    weights, _ = op.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=mu, constraints=_long_short_fully_invested())
    inv, one = np.linalg.inv(sigma), np.ones(len(TICKERS))
    reference = inv @ mu.to_numpy() / (one @ inv @ mu.to_numpy())
    np.testing.assert_allclose(np.asarray(weights, dtype=float), reference, atol=ANALYTIC_TOL)


def test_equal_risk_contribution_is_inverse_vol_under_equal_correlation() -> None:
    """with equal pairwise correlation the ERC portfolio is w_i ∝ 1/σ_i."""
    covar, _ = _equal_correlation_universe()
    weights = op.wrapper_risk_budgeting(pd_covar=covar, constraints=Constraints(is_long_only=True))
    reference = (1.0 / VOLS) / np.sum(1.0 / VOLS)
    np.testing.assert_allclose(np.asarray(weights, dtype=float), reference, atol=ITERATIVE_TOL)


def test_equal_risk_contribution_equalises_risk_contributions() -> None:
    """
    the defining property, away from the special case.

    Risk contribution of asset i is w_i (Σw)_i / (w'Σw); the ERC portfolio makes all four equal.
    """
    covar, sigma = _general_universe()
    weights = np.asarray(op.wrapper_risk_budgeting(
        pd_covar=covar, constraints=Constraints(is_long_only=True)), dtype=float)
    contributions = weights * (sigma @ weights) / (weights @ sigma @ weights)
    np.testing.assert_allclose(contributions, np.full(len(TICKERS), 1.0 / len(TICKERS)),
                               atol=1e-3)


def test_max_diversification_is_inverse_vol_under_equal_correlation() -> None:
    """with equal pairwise correlation the most-diversified portfolio is also w_i ∝ 1/σ_i."""
    covar, _ = _equal_correlation_universe()
    weights = op.wrapper_maximise_diversification(pd_covar=covar,
                                                  constraints=Constraints(is_long_only=True))
    reference = (1.0 / VOLS) / np.sum(1.0 / VOLS)
    np.testing.assert_allclose(np.asarray(weights, dtype=float), reference, atol=ITERATIVE_TOL)


def test_max_diversification_maximises_the_diversification_ratio() -> None:
    """the defining property: w'σ / √(w'Σw) is not improved by a budget-preserving perturbation."""
    covar, sigma = _general_universe()
    weights = np.asarray(op.wrapper_maximise_diversification(
        pd_covar=covar, constraints=Constraints(is_long_only=True)), dtype=float)
    vols = np.sqrt(np.diag(sigma))

    def ratio(w: np.ndarray) -> float:
        return float(w @ vols / np.sqrt(w @ sigma @ w))

    best = ratio(weights)
    rng = np.random.default_rng(23)
    for _ in range(25):
        step = rng.normal(size=len(TICKERS))
        step -= step.mean()
        perturbed = weights + 1e-4 * step
        if np.all(perturbed >= 0.0):             # stay inside the long-only feasible set
            assert ratio(perturbed) <= best + 1e-6


# ───────────────────────────────────────────────────────────────────────────────
# SAA solvers
# ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('target_return', [0.04, 0.06, 0.08])
def test_min_variance_target_return_matches_the_closed_form(target_return: float) -> None:
    """w = [(C - rB)Σ⁻¹1 + (rA - B)Σ⁻¹μ] / D."""
    covar, sigma = _general_universe()
    mu = _means()
    m = mu.to_numpy()
    inv, one = np.linalg.inv(sigma), np.ones(len(TICKERS))
    a, b, c, d = _abc(sigma, m)
    reference = ((c - target_return * b) * (inv @ one)
                 + (target_return * a - b) * (inv @ m)) / d
    weights, _ = op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=mu, target_return=target_return,
        constraints=_long_short_fully_invested())
    np.testing.assert_allclose(np.asarray(weights, dtype=float), reference, atol=1e-6)


@pytest.mark.parametrize('target_return', [0.04, 0.06, 0.08])
def test_min_variance_target_return_hits_its_target(target_return: float) -> None:
    """the constraint the solver was given is the constraint it satisfies."""
    covar, _ = _general_universe()
    mu = _means()
    weights = np.asarray(op.wrapper_min_variance_target_return(
        pd_covar=covar, expected_returns=mu, target_return=target_return,
        constraints=_long_short_fully_invested())[0], dtype=float)
    assert weights @ mu.to_numpy() == pytest.approx(target_return, abs=1e-6)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize('target_vol', [0.10, 0.12, 0.15])
def test_max_return_target_vol_matches_the_closed_form(target_vol: float) -> None:
    """w = Σ⁻¹1/A + k Σ⁻¹(μ - (B/A)1), with k set so the portfolio volatility is the target."""
    covar, sigma = _general_universe()
    mu = _means()
    m = mu.to_numpy()
    inv, one = np.linalg.inv(sigma), np.ones(len(TICKERS))
    a, b, _, _ = _abc(sigma, m)
    w_min_var = inv @ one / a
    direction = inv @ (m - (b / a) * one)        # zero-sum, increases expected return
    k = np.sqrt((target_vol ** 2 - w_min_var @ sigma @ w_min_var) / (direction @ sigma @ direction))
    reference = w_min_var + k * direction
    weights, _ = op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=mu, target_vol=target_vol,
        constraints=_long_short_fully_invested())
    np.testing.assert_allclose(np.asarray(weights, dtype=float), reference, atol=1e-6)


@pytest.mark.parametrize('target_vol', [0.10, 0.12, 0.15])
def test_max_return_target_vol_hits_its_target(target_vol: float) -> None:
    """the realised portfolio volatility equals the budget it was given."""
    covar, sigma = _general_universe()
    weights = np.asarray(op.wrapper_max_return_target_vol(
        pd_covar=covar, expected_returns=_means(), target_vol=target_vol,
        constraints=_long_short_fully_invested())[0], dtype=float)
    assert np.sqrt(weights @ sigma @ weights) == pytest.approx(target_vol, abs=1e-6)


# ───────────────────────────────────────────────────────────────────────────────
# TAA solver
# ───────────────────────────────────────────────────────────────────────────────


def _benchmark_and_alphas() -> Tuple[pd.Series, pd.Series]:
    """an equal-weight benchmark and a signed alpha vector that nets to a non-trivial tilt."""
    benchmark = pd.Series(np.full(len(TICKERS), 1.0 / len(TICKERS)), index=TICKERS)
    alphas = pd.Series([0.01, -0.005, 0.02, -0.01], index=TICKERS)
    return benchmark, alphas


@pytest.mark.parametrize('tracking_error', [0.01, 0.02, 0.03])
def test_maximise_alpha_over_tre_matches_the_closed_form(tracking_error: float) -> None:
    """
    w = w_b + τ Σ⁻¹α̃ / √(α̃'Σ⁻¹ Σ Σ⁻¹α̃).

    α̃ = α - (1'Σ⁻¹α / A)1 is alpha projected so the tilt preserves the budget; without it the
    solution drifts off full investment and the comparison fails for the wrong reason.
    """
    covar, sigma = _general_universe()
    benchmark, alphas = _benchmark_and_alphas()
    inv, one = np.linalg.inv(sigma), np.ones(len(TICKERS))
    a = float(one @ inv @ one)
    alpha_tilde = alphas.to_numpy() - (one @ inv @ alphas.to_numpy()) / a * one
    direction = inv @ alpha_tilde
    scale = tracking_error / np.sqrt(direction @ sigma @ direction)
    reference = benchmark.to_numpy() + scale * direction

    constraints = Constraints(is_long_only=False, min_exposure=1.0, max_exposure=1.0,
                              benchmark_weights=benchmark,
                              tracking_err_vol_constraint=tracking_error)
    weights, _ = op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
        constraints=constraints)
    np.testing.assert_allclose(np.asarray(weights, dtype=float), reference, atol=1e-6)


@pytest.mark.parametrize('tracking_error', [0.01, 0.02, 0.03])
def test_maximise_alpha_over_tre_spends_its_tracking_error_budget(tracking_error: float) -> None:
    """
    the active risk equals the budget.

    The objective is linear in the active weights, so the constraint binds: a solution spending
    less than its budget is leaving alpha on the table and indicates the constraint is not
    reaching the solver.
    """
    covar, sigma = _general_universe()
    benchmark, alphas = _benchmark_and_alphas()
    constraints = Constraints(is_long_only=False, min_exposure=1.0, max_exposure=1.0,
                              benchmark_weights=benchmark,
                              tracking_err_vol_constraint=tracking_error)
    weights = np.asarray(op.wrapper_maximise_alpha_over_tre(
        pd_covar=covar, alphas=alphas, benchmark_weights=benchmark,
        constraints=constraints)[0], dtype=float)
    active = weights - benchmark.to_numpy()
    assert np.sqrt(active @ sigma @ active) == pytest.approx(tracking_error, abs=1e-6)


# ───────────────────────────────────────────────────────────────────────────────
# Constrained problems, where the reference is optimality rather than a formula
# ───────────────────────────────────────────────────────────────────────────────


def test_long_only_min_variance_satisfies_the_kkt_conditions() -> None:
    """
    with a sign constraint there is no formula, but the optimum still has a signature.

    At the optimum of min w'Σw subject to w'1 = 1 and w ≥ 0, the marginal variance (Σw)_i is equal
    across every asset that is held, and no smaller for any asset that is not. An interior asset
    with a lower marginal than a held one would be worth buying, so its presence means the solver
    stopped early.
    """
    covar, sigma = _general_universe()
    weights = np.asarray(op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=Constraints(is_long_only=True),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)[0], dtype=float)
    assert np.all(weights >= -1e-8), f'long-only solution has a negative weight: {weights}'

    marginal = sigma @ weights
    held = weights > 1e-6
    assert held.any(), 'no asset is held'
    np.testing.assert_allclose(marginal[held], marginal[held].mean(), rtol=1e-3)
    if (~held).any():
        assert marginal[~held].min() >= marginal[held].mean() - 1e-6, (
            'an excluded asset has a lower marginal variance than the held ones, so buying it '
            'would have reduced portfolio variance')


def test_weight_bounds_are_respected_and_binding() -> None:
    """a max-weight cap changes the answer and is not exceeded."""
    covar, _ = _general_universe()
    cap = 0.30
    constraints = Constraints(is_long_only=True,
                              max_weights=pd.Series(np.full(len(TICKERS), cap), index=TICKERS))
    weights = np.asarray(op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=constraints,
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)[0], dtype=float)
    assert weights.max() <= cap + 1e-6, f'cap {cap} exceeded: {weights}'
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    unconstrained = np.asarray(op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=Constraints(is_long_only=True),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)[0], dtype=float)
    assert unconstrained.max() > cap, (
        'the unconstrained optimum already respects the cap, so this test proves nothing; '
        'lower the cap')


def test_constrained_optimum_is_worse_than_the_unconstrained_one() -> None:
    """
    adding a constraint cannot improve the objective.

    Trivially true in theory and a sharp check in practice: it fails when a constraint is dropped
    on the way to the solver, which is silent otherwise because the answer still looks reasonable.
    """
    covar, sigma = _general_universe()
    free = np.asarray(op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=_long_short_fully_invested(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)[0], dtype=float)
    capped = np.asarray(op.wrapper_quadratic_optimisation(
        pd_covar=covar,
        constraints=Constraints(is_long_only=True,
                                max_weights=pd.Series(np.full(len(TICKERS), 0.30), index=TICKERS)),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)[0], dtype=float)
    assert capped @ sigma @ capped >= free @ sigma @ free - 1e-12


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
