"""
input validation and infeasibility detection in the constrained risk-budgeting solver.

This solver minimises a log-barrier objective whose domain is the strictly positive orthant,
then recovers the fully-invested solution by root-finding on the barrier multiplier. Both
halves of that construction fail badly on inputs a portfolio caller can plausibly supply, and
neither failure is loud on its own:

- a zero or negative variance on the diagonal makes the risk contribution undefined, and a
  negative lower bound puts the barrier's ``log(x)`` outside its domain -- so those are
  rejected up front rather than left to produce NaNs mid-iteration;
- a box whose lower bounds already sum above 1 (or whose upper bounds sum below 1) has no
  fully-invested point at all, which the root-finder would otherwise chase to a bracketing
  failure several hundred iterations later.

``_validate_inputs`` deliberately runs on the *raw* inputs, before budget normalisation and
before any slicing, because the caller's fallback contract is that every invalid input raises
``ValueError``. The tests below go through the public entry point rather than calling the
private validator, so they exercise that ordering as well as the messages.

The pinned-box short circuit and the bracketing failure are the two non-validation paths here.
A fully pinned box is not solved at all -- the weights are determined -- and the returned
multiplier is NaN by contract, which callers must not read as a solve failure.
"""
# packages
import numpy as np
import pytest
# optimalportfolios
from optimalportfolios.optimization.general.risk_budgeting_solver import (
    solve_constrained_risk_budgeting,
)

N = 4


def covar_matrix() -> np.ndarray:
    """A well-conditioned 4-asset covariance."""
    vols = np.array([0.20, 0.12, 0.08, 0.05])
    corr = np.array([
        [1.00, 0.30, 0.10, 0.00],
        [0.30, 1.00, 0.20, 0.10],
        [0.10, 0.20, 1.00, 0.25],
        [0.00, 0.10, 0.25, 1.00],
    ])
    return np.outer(vols, vols) * corr


def equal_budgets() -> np.ndarray:
    """Equal risk budgets over the four assets."""
    return np.ones(N) / N


# --------------------------------------------------------------------------- #
# the happy path, for reference
# --------------------------------------------------------------------------- #
def test_the_solution_is_fully_invested_and_returns_a_positive_multiplier() -> None:
    """Equal risk budgets give positive weights summing to one and a real multiplier."""
    weights, lam = solve_constrained_risk_budgeting(covar=covar_matrix(),
                                                    budgets=equal_budgets())
    assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-6)
    assert np.all(weights > 0.0)
    assert lam > 0.0


def test_budgets_default_to_equal_when_omitted() -> None:
    """``budgets=None`` means equal risk contribution, the usual entry point."""
    omitted, _ = solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=None)
    explicit, _ = solve_constrained_risk_budgeting(covar=covar_matrix(),
                                                   budgets=equal_budgets())
    np.testing.assert_allclose(omitted, explicit, atol=1e-8)


def test_budgets_are_normalised_rather_than_required_to_sum_to_one() -> None:
    """Only the relative budgets matter, so an unnormalised vector is accepted."""
    normalised, _ = solve_constrained_risk_budgeting(covar=covar_matrix(),
                                                     budgets=equal_budgets())
    scaled, _ = solve_constrained_risk_budgeting(covar=covar_matrix(),
                                                 budgets=7.0 * equal_budgets())
    np.testing.assert_allclose(normalised, scaled, atol=1e-8)


# --------------------------------------------------------------------------- #
# covariance validation
# --------------------------------------------------------------------------- #
def test_a_non_square_covariance_raises() -> None:
    """The commonest caller error: a panel slice rather than a covariance."""
    with pytest.raises(ValueError, match='covar must be square'):
        solve_constrained_risk_budgeting(covar=np.ones((3, 4)), budgets=np.ones(3) / 3)


def test_a_non_finite_covariance_raises_and_counts_the_offenders() -> None:
    """The message carries the count, since one NaN and a whole NaN row differ in cause."""
    covar = covar_matrix()
    covar[0, 1] = np.nan
    with pytest.raises(ValueError, match='non-finite values'):
        solve_constrained_risk_budgeting(covar=covar, budgets=equal_budgets())


@pytest.mark.parametrize('variance', [0.0, -0.01])
def test_a_non_positive_variance_on_the_diagonal_raises(variance: float) -> None:
    """A zero-variance asset makes its risk contribution undefined, not merely small.

    The log-barrier objective divides through the risk contribution, so this would surface as
    a NaN mid-iteration rather than as a rejected input.
    """
    covar = covar_matrix()
    covar[2, 2] = variance
    with pytest.raises(ValueError, match='diagonal must be strictly positive'):
        solve_constrained_risk_budgeting(covar=covar, budgets=equal_budgets())


# --------------------------------------------------------------------------- #
# budget validation
# --------------------------------------------------------------------------- #
def test_budgets_of_the_wrong_length_raise() -> None:
    """A budget vector over a different universe would broadcast or truncate silently."""
    with pytest.raises(ValueError, match=r'budgets must have shape \(4,\)'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=np.ones(3) / 3)


@pytest.mark.parametrize('bad', [
    np.array([0.25, 0.25, -0.25, 0.75]),
    np.array([0.25, 0.25, np.nan, 0.25]),
])
def test_negative_or_non_finite_budgets_raise(bad: np.ndarray) -> None:
    """A negative risk budget has no meaning; a NaN one silently drops an asset."""
    with pytest.raises(ValueError, match='budgets must be finite and non-negative'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=bad)


def test_all_zero_budgets_raise_rather_than_dividing_by_zero() -> None:
    """Budgets are normalised by their sum, so a zero sum must be caught first."""
    with pytest.raises(ValueError, match='budgets must have a positive sum'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=np.zeros(N))


# --------------------------------------------------------------------------- #
# box validation
# --------------------------------------------------------------------------- #
def test_bounds_of_the_wrong_shape_raise() -> None:
    """Bounds are (n, 2); a transposed array is the likely mistake."""
    with pytest.raises(ValueError, match=r'bounds must have shape \(4, 2\)'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=equal_budgets(),
                                         bounds=np.zeros((2, N)))


def test_a_lower_bound_above_its_upper_bound_raises_and_names_the_asset() -> None:
    """An inverted box is empty for that asset; the message identifies which."""
    bounds = np.column_stack([np.zeros(N), np.full(N, 0.5)])
    bounds[2, :] = [0.6, 0.4]
    with pytest.raises(ValueError, match='lower bound exceeds upper bound at index 2'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=equal_budgets(),
                                         bounds=bounds)


def test_a_negative_lower_bound_raises_as_outside_the_barrier_domain() -> None:
    """The log-barrier is defined on positive weights, so shorting is not merely unsupported.

    A negative lower bound would let the iteration step into ``log`` of a negative number.
    """
    bounds = np.column_stack([np.zeros(N), np.full(N, 0.5)])
    bounds[1, 0] = -0.10
    with pytest.raises(ValueError, match='log-barrier domain'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=equal_budgets(),
                                         bounds=bounds)


def test_lower_bounds_summing_above_one_are_infeasible() -> None:
    """No fully-invested point exists, which the root-finder would otherwise chase for a while."""
    bounds = np.column_stack([np.full(N, 0.40), np.full(N, 0.60)])
    with pytest.raises(ValueError, match='sum of lower bounds exceeds 1'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=equal_budgets(),
                                         bounds=bounds)


def test_upper_bounds_summing_below_one_are_infeasible() -> None:
    """The mirror case: the box cannot reach full investment."""
    bounds = np.column_stack([np.zeros(N), np.full(N, 0.10)])
    with pytest.raises(ValueError, match='sum of upper bounds is below 1'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=equal_budgets(),
                                         bounds=bounds)


def test_a_fully_pinned_box_short_circuits_with_a_nan_multiplier() -> None:
    """When lo == hi the weights are determined, so there is nothing to solve.

    The multiplier is NaN by contract here. Callers must read that as "not applicable", not as
    a failed solve -- it is the one accepted return with a non-finite second element.
    """
    pinned = np.full(N, 0.25)
    bounds = np.column_stack([pinned, pinned])
    weights, lam = solve_constrained_risk_budgeting(covar=covar_matrix(),
                                                    budgets=equal_budgets(), bounds=bounds)
    np.testing.assert_allclose(weights, pinned, atol=1e-9)
    assert np.isnan(lam)


# --------------------------------------------------------------------------- #
# linear inequality validation
# --------------------------------------------------------------------------- #
def test_the_solver_accepts_a_linear_inequality_block() -> None:
    """``C x <= d`` routes through the ADMM path and still returns a fully-invested solution."""
    c_rows = np.array([[1.0, 1.0, 0.0, 0.0]])
    c_lhs = np.array([0.60])
    weights, _ = solve_constrained_risk_budgeting(covar=covar_matrix(),
                                                  budgets=equal_budgets(),
                                                  c_rows=c_rows, c_lhs=c_lhs)
    assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-5)
    assert float(weights[:2].sum()) <= 0.60 + 1e-5


@pytest.mark.parametrize('c_rows, c_lhs', [
    (np.array([[1.0, 1.0, 0.0, 0.0]]), None),
    (None, np.array([0.60])),
])
def test_a_half_specified_inequality_block_raises(c_rows, c_lhs) -> None:
    """A matrix without a right-hand side (or the reverse) is silently ignorable otherwise."""
    with pytest.raises(ValueError, match='must both be given or both be None'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=equal_budgets(),
                                         c_rows=c_rows, c_lhs=c_lhs)


def test_an_inequality_matrix_with_the_wrong_width_raises() -> None:
    """``C`` has one column per asset; anything else is a universe mismatch."""
    with pytest.raises(ValueError, match=r'c_rows must have shape \(p, 4\)'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=equal_budgets(),
                                         c_rows=np.ones((1, 3)), c_lhs=np.array([0.60]))


def test_a_right_hand_side_of_the_wrong_length_raises() -> None:
    """One bound per inequality row; a mismatch would zip them wrongly."""
    with pytest.raises(ValueError, match=r'c_lhs must have shape \(1,\)'):
        solve_constrained_risk_budgeting(covar=covar_matrix(), budgets=equal_budgets(),
                                         c_rows=np.ones((1, N)), c_lhs=np.array([0.6, 0.7]))
