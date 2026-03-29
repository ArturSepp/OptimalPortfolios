"""
Maximum Sharpe ratio portfolio optimisation.

Maximises the Sharpe ratio:

    SR(w) = μ'w / sqrt(w'Σw)

where μ is the vector of expected returns and Σ is the covariance matrix.
The maximum Sharpe portfolio is the tangency portfolio on the mean-variance
efficient frontier, and is invariant to the risk-free rate when μ is
expressed as excess returns.

The fractional (ratio) objective is non-convex, but admits an exact convex
reformulation via the Charnes-Cooper transformation: introduce auxiliary
variables (y, k) with y = k*w, k > 0, and solve the equivalent SOCP:

    min_y  y'Σy   s.t.  μ'y = c,  constraints(y, k)

then recover w* = y / k. This yields the global optimum without the
initialisation sensitivity of direct ratio optimisation.

Uses CVXPY with pre-computed covariance matrices from any CovarEstimator.
Expected returns are provided externally (e.g., from a CMA model) and
forward-filled to the rebalancing schedule.

Reference:
    Cornuejols G. and Tütüncü R. (2007),
    "Optimization Methods in Finance",
    Cambridge University Press, Section 8.3.

    Charnes A. and Cooper W.W. (1962),
    "Programming with Linear Fractional Functionals",
    Naval Research Logistics Quarterly, 9(3-4), 181-186.
"""
import warnings
import numpy as np
import pandas as pd
import cvxpy as cvx
from typing import Dict

from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.config import OptimiserConfig


def rolling_maximize_portfolio_sharpe(prices: pd.DataFrame,
                                      expected_returns: pd.DataFrame,
                                      constraints: Constraints,
                                      covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                      optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                      ) -> pd.DataFrame:
    """
    Maximise portfolio Sharpe ratio at each rebalancing date.

    Args:
        prices: Asset price panel. Used for column alignment.
        expected_returns: Expected returns per asset. Forward-filled to
            rebalancing dates.
        constraints: Portfolio constraints.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    rebalancing_dates = list(covar_dict.keys())
    expected_returns = expected_returns.reindex(index=rebalancing_dates, method='ffill')

    tickers = prices.columns.to_list()
    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():
        weights_ = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                     means=expected_returns.loc[date, :],
                                                     constraints=constraints,
                                                     weights_0=weights_0,
                                                     optimiser_config=optimiser_config)
        weights_0 = weights_
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers).fillna(0.0)
    return weights


def wrapper_maximize_portfolio_sharpe(pd_covar: pd.DataFrame,
                                      means: pd.Series,
                                      constraints: Constraints,
                                      weights_0: pd.Series = None,
                                      optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                      ) -> pd.Series:
    """
    Single-date maximum Sharpe with NaN/zero-variance filtering.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        means: Expected returns per asset.
        constraints: Portfolio constraints.
        weights_0: Previous-period weights for warm-start / fallback.
        optimiser_config: Solver configuration.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    vectors = dict(means=means)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    if optimiser_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    constraints1 = constraints.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=total_to_good_ratio,
                                                         weights_0=weights_0)

    weights = cvx_maximize_portfolio_sharpe(covar=clean_covar.to_numpy(),
                                            means=good_vectors['means'].to_numpy(),
                                            constraints=constraints1,
                                            solver=optimiser_config.solver,
                                            verbose=optimiser_config.verbose)
    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)
    return weights


def cvx_maximize_portfolio_sharpe(covar: np.ndarray,
                                  means: np.ndarray,
                                  constraints: Constraints,
                                  verbose: bool = False,
                                  solver: str = 'CLARABEL'
                                  ) -> np.ndarray:
    """
    Maximise the Sharpe ratio via the Charnes-Cooper transformation.

    The Charnes-Cooper transformation introduces z = [y; k] where y = k*w
    and k > 0. Setting μ'y = c pins the scale and converts the problem to:

        min_z  y'Σy   s.t.  μ'y = c,  constraints(y, k)

    The optimal weights are recovered as w* = y / k.

    The transformation requires a fixed-sum equality constraint on portfolio
    exposure (max_exposure == min_exposure). When the portfolio allows
    variable net exposure (long-short with max_exposure != min_exposure),
    the function falls back to direct ratio optimisation via scipy SLSQP,
    which handles arbitrary exposure bounds at the cost of non-convexity.

    Args:
        covar: Covariance matrix (N x N).
        means: Expected returns vector (N,).
        constraints: Portfolio constraints.
        verbose: If True, print CVXPY solver diagnostics.
        solver: CVXPY solver name (used only for Charnes-Cooper path).

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
    """
    if constraints.max_exposure != constraints.min_exposure:
        # long-short: Charnes-Cooper requires equality sum constraint
        return _scipy_maximize_sharpe(covar=covar, means=means,
                                      constraints=constraints, verbose=verbose)
    else:
        return _cvx_maximize_sharpe_charnes_cooper(covar=covar, means=means,
                                                    constraints=constraints,
                                                    verbose=verbose, solver=solver)


def _cvx_maximize_sharpe_charnes_cooper(covar: np.ndarray,
                                         means: np.ndarray,
                                         constraints: Constraints,
                                         verbose: bool = False,
                                         solver: str = 'CLARABEL'
                                         ) -> np.ndarray:
    """Charnes-Cooper SOCP for fixed-sum (long-only) case."""
    n = covar.shape[0]
    z = cvx.Variable(n+1)
    w = z[:n]
    k = z[n]

    objective = cvx.Minimize(cvx.quad_form(w, cvx.psd_wrap(covar)))
    constraints_ = constraints.set_cvx_all_constraints(
        w=w, covar=cvx.psd_wrap(covar), exposure_scaler=k)
    constraints_ += [means.T @ w == constraints.max_exposure]
    constraints_ += [k >= 0]

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = z.value
    if optimal_weights is not None:
        optimal_weights = optimal_weights[:n] / optimal_weights[n]
    else:
        warnings.warn(f"cvx_maximize_portfolio_sharpe: solver did not converge")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
        else:
            optimal_weights = np.zeros(n)

    return optimal_weights


def _scipy_maximize_sharpe(covar: np.ndarray,
                            means: np.ndarray,
                            constraints: Constraints,
                            verbose: bool = False
                            ) -> np.ndarray:
    """Direct Sharpe ratio maximisation via scipy SLSQP for long-short."""
    from scipy.optimize import minimize

    n = covar.shape[0]
    x0 = np.ones(n) / n

    def neg_sharpe(w):
        port_ret = means @ w
        port_vol = np.sqrt(w @ covar @ w)
        if port_vol < 1e-12:
            return 0.0
        return -port_ret / port_vol

    constraints_, bounds = constraints.set_scipy_constraints(covar=covar)
    res = minimize(neg_sharpe, x0, method='SLSQP',
                   constraints=constraints_,
                   bounds=bounds,
                   options={'disp': verbose, 'ftol': 1e-10, 'maxiter': 500})

    optimal_weights = res.x

    if not res.success or optimal_weights is None:
        warnings.warn(f"_scipy_maximize_sharpe: solver did not converge: {res.message}")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
        else:
            optimal_weights = np.zeros(n)

    return optimal_weights