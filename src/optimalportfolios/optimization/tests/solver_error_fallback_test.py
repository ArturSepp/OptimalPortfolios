"""
what every cvxpy solve does when the backend raises instead of returning a status.

``problem.solve`` normally reports failure as a *status* — 'infeasible', 'unbounded' — which
the validator downstream knows how to route. CLARABEL and the other backends can also raise
``cvx.error.SolverError`` outright when the constraint geometry is numerically degenerate, and
an unhandled raise inside a rolling backtest kills the whole run at one bad rebalancing date
rather than falling back for that date and carrying on.

Every cvxpy solver in this package therefore wraps its solve in the same three lines: catch
``SolverError``, blank the iterate, and hand the validator a ``'solver_error'`` status so the
raise routes into exactly the same fallback path as an honestly-reported infeasibility. There
are seven such call sites across five modules, and none of them is reachable from a solvable
problem — a backend that raises on demand is not something a covariance can be built to
produce. So the raise is injected, once per call site, and what is asserted is the contract
they share: no exception escapes, the outcome is *not* accepted, and the weights that come
back are finite and safe to trade rather than the None the solver left behind.

The fallback itself (weights_0 when there is one, zeros otherwise) is ``validate_solution``'s
and is covered in ``solver_diagnostics_test.py``; what is new here is that these seven sites
reach it at all.
"""
# packages
from typing import Callable
import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
from optimalportfolios import Constraints, PortfolioObjective
from optimalportfolios.optimization.constraints import ConstraintEnforcementType
from optimalportfolios.optimization.general.quadratic import cvx_quadratic_optimisation
from optimalportfolios.optimization.saa.max_return_target_vol import (
    cvx_max_return_target_vol, cvx_max_return_target_vol_utility)
from optimalportfolios.optimization.saa.min_variance_target_return import (
    cvx_min_variance_target_return, cvx_min_variance_target_return_utility)
from optimalportfolios.optimization.taa.maximise_alpha_over_tre import (
    cvx_maximise_alpha_over_tre, cvx_maximise_tre_utility)

TICKERS = ['growth', 'balanced', 'defensive']
VOLS = np.array([0.22, 0.14, 0.06])
CORR = np.array([[1.00, 0.45, 0.15],
                 [0.45, 1.00, 0.25],
                 [0.15, 0.25, 1.00]])
COVAR = np.outer(VOLS, VOLS) * CORR
ALPHAS = np.array([0.09, 0.06, 0.02])
WEIGHTS_0 = pd.Series([0.30, 0.30, 0.40], index=TICKERS)
BENCHMARK = pd.Series([0.40, 0.35, 0.25], index=TICKERS)


def make_constraints(**overrides) -> Constraints:
    """A long-only, fully invested set carrying a benchmark and a prior portfolio."""
    kwargs = dict(is_long_only=True,
                  min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS),
                  benchmark_weights=BENCHMARK,
                  weights_0=WEIGHTS_0,
                  asset_returns=pd.Series(ALPHAS, index=TICKERS),
                  target_return=0.05,
                  max_target_portfolio_vol_an=0.12,
                  tracking_err_vol_constraint=0.03)
    kwargs.update(overrides)
    return Constraints(**kwargs)


def utility_constraints(**overrides) -> Constraints:
    """The same set in the penalty formulation, which takes the ``*_utility`` code path."""
    return make_constraints(
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        **overrides)


@pytest.fixture
def raising_solver(monkeypatch):
    """Make every ``cvx.Problem.solve`` raise the error a degenerate backend raises."""
    def _raise(self, *args, **kwargs):
        """Stand in for a backend that raises rather than reporting a status."""
        raise cvx.error.SolverError('numerically degenerate constraint geometry')

    monkeypatch.setattr(cvx.Problem, 'solve', _raise)


# Each entry solves a different problem, but all seven share the same guard. The callable
# takes no arguments so the assertion below can be written once.
SOLVERS = {
    'quadratic': lambda: cvx_quadratic_optimisation(
        portfolio_objective=PortfolioObjective.MIN_VARIANCE, covar=COVAR,
        constraints=make_constraints()),
    'max_return_target_vol': lambda: cvx_max_return_target_vol(
        covar=COVAR, alphas=ALPHAS, constraints=make_constraints()),
    'max_return_target_vol_utility': lambda: cvx_max_return_target_vol_utility(
        covar=COVAR, alphas=ALPHAS, constraints=utility_constraints()),
    'min_variance_target_return': lambda: cvx_min_variance_target_return(
        covar=COVAR, constraints=make_constraints()),
    'min_variance_target_return_utility': lambda: cvx_min_variance_target_return_utility(
        covar=COVAR, constraints=utility_constraints()),
    'maximise_alpha_over_tre': lambda: cvx_maximise_alpha_over_tre(
        covar=COVAR, alphas=ALPHAS, constraints=make_constraints()),
    'maximise_tre_utility': lambda: cvx_maximise_tre_utility(
        covar=COVAR, alphas=ALPHAS, constraints=utility_constraints()),
}


@pytest.mark.parametrize('name', sorted(SOLVERS))
def test_a_raising_backend_is_routed_into_the_fallback(name: str, raising_solver) -> None:
    """the raise is caught at every call site and reported as a rejected solve"""
    outcome = SOLVERS[name]()
    assert not outcome.accepted, 'a solve that never ran must not be reported as accepted'
    weights = np.asarray(outcome.weights, dtype=float)
    assert weights.shape == (len(TICKERS),)
    assert np.all(np.isfinite(weights)), 'the None iterate reached the caller'
    # the prior portfolio is the safe answer, and it is what the validator falls back to
    np.testing.assert_allclose(weights, WEIGHTS_0.to_numpy(), atol=1e-8)


@pytest.mark.parametrize('name', sorted(SOLVERS))
def test_a_raising_backend_without_a_prior_portfolio_falls_back_to_the_benchmark(
        name: str, raising_solver) -> None:
    """with no prior portfolio the benchmark is the next safe answer down the ladder

    The point is that the ladder is reached at all: whatever the caller supplied, a raise
    produces a tradeable vector rather than the None the solver left in ``w.value``.
    """
    solver: Callable = {
        'quadratic': lambda: cvx_quadratic_optimisation(
            portfolio_objective=PortfolioObjective.MIN_VARIANCE, covar=COVAR,
            constraints=make_constraints(weights_0=None)),
        'max_return_target_vol': lambda: cvx_max_return_target_vol(
            covar=COVAR, alphas=ALPHAS, constraints=make_constraints(weights_0=None)),
        'max_return_target_vol_utility': lambda: cvx_max_return_target_vol_utility(
            covar=COVAR, alphas=ALPHAS, constraints=utility_constraints(weights_0=None)),
        'min_variance_target_return': lambda: cvx_min_variance_target_return(
            covar=COVAR, constraints=make_constraints(weights_0=None)),
        'min_variance_target_return_utility': lambda: cvx_min_variance_target_return_utility(
            covar=COVAR, constraints=utility_constraints(weights_0=None)),
        'maximise_alpha_over_tre': lambda: cvx_maximise_alpha_over_tre(
            covar=COVAR, alphas=ALPHAS, constraints=make_constraints(weights_0=None)),
        'maximise_tre_utility': lambda: cvx_maximise_tre_utility(
            covar=COVAR, alphas=ALPHAS, constraints=utility_constraints(weights_0=None)),
    }[name]
    outcome = solver()
    assert not outcome.accepted
    np.testing.assert_allclose(np.asarray(outcome.weights, dtype=float),
                               BENCHMARK.to_numpy(), atol=1e-8)
