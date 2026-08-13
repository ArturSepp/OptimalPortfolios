"""
the two solver paths behind maximum Sharpe, and how the dispatcher chooses between them.

Maximising Sharpe is not a convex problem as written. ``cvx_maximize_portfolio_sharpe``
handles that by dispatching on the exposure constraints: when ``min_exposure ==
max_exposure`` the problem admits the Charnes-Cooper transform, which turns the ratio into a
genuine convex program in a scaled variable; when they differ there is no equality sum
constraint to normalise against, so it falls back to SciPy SLSQP on the raw ratio.

That dispatch is the thing worth pinning. Both branches return the same outcome type and both
produce plausible weights, so a mis-routed problem looks exactly like a solved one -- it is
simply solved by a local method with no convexity guarantee when a global one was available,
or handed to a transform whose derivation assumed a constraint that is not there. The tests
below assert which solver actually ran, via the solver recorded on the outcome, rather than
inferring it from the weights.

The Charnes-Cooper path also unscales its solution by dividing through the auxiliary variable
``k``; the recovered weights are checked against the exposure they are supposed to hit, since
a scaling error there produces weights that are wrong by a constant factor and otherwise
entirely reasonable-looking. The ``SolverError`` fallback is covered for the same reason it is
in the TRE solver: it must degrade to a rejected outcome, not kill a rolling backtest.
"""
# packages
import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.general.max_sharpe import (
    cvx_maximize_portfolio_sharpe,
    wrapper_maximize_portfolio_sharpe,
)

TICKERS = pd.Index(['A', 'B', 'C', 'D'])


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


def means_vector() -> np.ndarray:
    """Expected returns with a clear cross-sectional spread."""
    return np.array([0.08, 0.05, 0.03, 0.02])


def make_constraints(min_exposure: float = 1.0, max_exposure: float = 1.0,
                     **overrides) -> Constraints:
    """Long-only bounds with the given exposure band."""
    kwargs = dict(min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS),
                  min_exposure=min_exposure,
                  max_exposure=max_exposure)
    kwargs.update(overrides)
    return Constraints(**kwargs)


# --------------------------------------------------------------------------- #
# the dispatch
# --------------------------------------------------------------------------- #
def test_a_fixed_exposure_routes_to_the_charnes_cooper_transform() -> None:
    """Equal min and max exposure gives the equality constraint the transform needs."""
    outcome = cvx_maximize_portfolio_sharpe(covar=covar_matrix(), means=means_vector(),
                                            constraints=make_constraints(1.0, 1.0))
    assert outcome.solver != 'SLSQP'
    assert outcome.accepted


def test_an_exposure_band_routes_to_the_scipy_fallback() -> None:
    """Without an equality sum constraint the ratio has nothing to normalise against."""
    outcome = cvx_maximize_portfolio_sharpe(covar=covar_matrix(), means=means_vector(),
                                            constraints=make_constraints(0.5, 1.0))
    assert outcome.solver == 'SLSQP'


def test_both_paths_return_the_same_outcome_type() -> None:
    """The caller cannot tell the paths apart from the return type, only from the solver."""
    fixed = cvx_maximize_portfolio_sharpe(covar=covar_matrix(), means=means_vector(),
                                          constraints=make_constraints(1.0, 1.0))
    banded = cvx_maximize_portfolio_sharpe(covar=covar_matrix(), means=means_vector(),
                                           constraints=make_constraints(0.5, 1.0))
    assert type(fixed) is type(banded)
    assert fixed.weights is not None and banded.weights is not None


# --------------------------------------------------------------------------- #
# the Charnes-Cooper path
# --------------------------------------------------------------------------- #
def test_the_transformed_solution_is_unscaled_back_onto_the_exposure() -> None:
    """The auxiliary variable k must be divided out, or the weights are off by a constant.

    Charnes-Cooper solves in ``z = k*w``; recovering ``w = z[:n] / z[n]`` is what puts the
    answer back on the caller's exposure. A missed division still yields positive, ordered,
    entirely plausible weights.
    """
    outcome = cvx_maximize_portfolio_sharpe(covar=covar_matrix(), means=means_vector(),
                                            constraints=make_constraints(1.0, 1.0))
    assert outcome.accepted
    assert float(np.sum(outcome.weights)) == pytest.approx(1.0, abs=1e-6)


def test_the_transformed_solution_beats_every_naive_alternative_on_sharpe() -> None:
    """The recovered weights must actually maximise the ratio, not merely satisfy the bounds.

    Optimality is the claim the transform makes, and it is the one thing box-and-exposure
    assertions cannot see: a wrong ``k``, a dropped covariance term or a sign slip all leave
    weights that are long-only, sum to one and look sensible. Scoring the result against the
    equal-weight portfolio and against each single-asset portfolio is a cheap lower bound that
    those failures do not clear. Deliberately not asserted: any ordering of the weights against
    the means. The Sharpe optimum here is ordered D > C > B > A, the exact inverse of the means,
    because the low-mean assets are the low-vol ones -- an intuition-shaped assertion would
    pin the wrong portfolio.
    """
    outcome = cvx_maximize_portfolio_sharpe(covar=covar_matrix(), means=means_vector(),
                                            constraints=make_constraints(1.0, 1.0))
    weights = pd.Series(outcome.weights, index=TICKERS)
    assert weights.min() >= -1e-9                       # long-only bounds respected

    covar, means = covar_matrix(), means_vector()

    def sharpe(w: np.ndarray) -> float:
        """Reward-to-risk of a weight vector under the test covariance."""
        return float(w @ means / np.sqrt(w @ covar @ w))

    best = sharpe(weights.to_numpy())
    assert best > sharpe(np.full(len(TICKERS), 1.0 / len(TICKERS)))
    assert best > max(sharpe(row) for row in np.eye(len(TICKERS)))


def test_a_solver_failure_on_the_transform_becomes_a_rejected_outcome(monkeypatch) -> None:
    """CLARABEL can raise rather than report a status when the geometry is degenerate.

    That must route into the same fallback as an honest infeasibility; propagating it would
    kill a rolling backtest at one bad rebalancing date.
    """
    def fail(self, **kwargs):
        """Stand in for a solver backend that raises instead of returning a status."""
        raise cvx.error.SolverError('degenerate geometry')

    monkeypatch.setattr(cvx.Problem, 'solve', fail)
    outcome = cvx_maximize_portfolio_sharpe(covar=covar_matrix(), means=means_vector(),
                                            constraints=make_constraints(1.0, 1.0))
    assert not outcome.accepted
    assert outcome.status == 'solver_error'


# --------------------------------------------------------------------------- #
# the SciPy fallback
# --------------------------------------------------------------------------- #
def test_the_scipy_path_respects_the_weight_bounds() -> None:
    """SLSQP is given the same bounds and must return weights inside them."""
    outcome = cvx_maximize_portfolio_sharpe(covar=covar_matrix(), means=means_vector(),
                                            constraints=make_constraints(0.5, 1.0))
    weights = np.asarray(outcome.weights, dtype=float)
    assert weights.min() >= -1e-6
    assert weights.max() <= 1.0 + 1e-6


def test_the_scipy_objective_is_flat_at_zero_volatility() -> None:
    """A zero-variance covariance makes the ratio undefined; the objective returns 0 instead.

    Without that guard the first SLSQP evaluation divides by zero, and the optimiser walks off
    into NaN rather than reporting a failure.
    """
    outcome = cvx_maximize_portfolio_sharpe(covar=np.zeros((4, 4)), means=means_vector(),
                                            constraints=make_constraints(0.5, 1.0))
    assert outcome.solver == 'SLSQP'
    assert np.all(np.isfinite(np.asarray(outcome.weights, dtype=float)))


# --------------------------------------------------------------------------- #
# the wrapper
# --------------------------------------------------------------------------- #
def test_the_wrapper_returns_weights_indexed_by_ticker() -> None:
    """The wrapper is the layer that puts the ticker index back on the solver's array."""
    weights, outcome = wrapper_maximize_portfolio_sharpe(
        pd_covar=pd.DataFrame(covar_matrix(), index=TICKERS, columns=TICKERS),
        means=pd.Series(means_vector(), index=TICKERS),
        constraints=make_constraints(1.0, 1.0))
    assert list(weights.index) == list(TICKERS)
    assert outcome.accepted
