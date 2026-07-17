"""
Portfolio optimisation using quadratic objective functions.

Implements minimum variance and quadratic utility (mean-variance) portfolio
optimisation via CVXPY, with support for rolling rebalancing, NaN-aware
covariance filtering, and vol-targeting via bisection.

Supported objectives:
    - MIN_VARIANCE: min w' Σ w  s.t. constraints
    - QUADRATIC_UTILITY: max μ'w - (γ/2) w' Σ w  s.t. constraints

The rolling wrapper accepts pre-computed covariance matrices (from any
CovarEstimator) and rebalances at each date in the covar dict.
"""
# packages
import warnings
import numpy as np
import pandas as pd
import cvxpy as cvx
from typing import Tuple, Optional, Dict

# optimalportfolios
from optimalportfolios.config import PortfolioObjective
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.solver_diagnostics import validate_solution
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.utils.weights_drift import apply_drift_to_weights_0
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans


def rolling_quadratic_optimisation(prices: pd.DataFrame,
                                   constraints: Constraints,
                                   covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                   inclusion_indicators: Optional[pd.DataFrame] = None,
                                   portfolio_objective: PortfolioObjective = PortfolioObjective.MIN_VARIANCE,
                                   expected_returns: pd.DataFrame = None,  # QUADRATIC_UTILITY only
                                   carra: float = 1.0,
                                   optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                   ) -> pd.DataFrame:
    """
    Compute rolling quadratic portfolio optimisation at each rebalancing date.

    Args:
        prices: Asset price panel. Used for column alignment.
        constraints: Portfolio constraints.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        inclusion_indicators: Optional binary DataFrame for asset eligibility.
        portfolio_objective: MIN_VARIANCE or QUADRATIC_UTILITY.
        expected_returns: Expected returns per asset. Required for
            QUADRATIC_UTILITY; forward-filled to rebalancing dates.
        carra: Risk aversion coefficient γ for QUADRATIC_UTILITY.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    if portfolio_objective == PortfolioObjective.QUADRATIC_UTILITY and expected_returns is None:
        raise ValueError("expected_returns must be given for QUADRATIC_UTILITY objective")

    rebalancing_schedule = list(covar_dict.keys())
    tickers = prices.columns.to_list()

    if expected_returns is not None:
        expected_returns = expected_returns.reindex(index=rebalancing_schedule, method='ffill')

    if inclusion_indicators is not None:
        inclusion_indicators1 = inclusion_indicators.reindex(columns=tickers)
        inclusion_indicators1 = inclusion_indicators1.reindex(index=rebalancing_schedule, method='ffill')
    else:
        inclusion_indicators1 = pd.DataFrame(1.0, index=rebalancing_schedule, columns=tickers)

    weights = {}
    weights_0 = None
    prev_date = None
    for date, pd_covar in covar_dict.items():
        weights_0 = apply_drift_to_weights_0(
            weights_0=weights_0, prices=prices,
            prev_date=prev_date, date=date,
            use_drifted_weights_0=optimiser_config.use_drifted_weights_0,
        )
        if expected_returns is not None:
            means = expected_returns.loc[date, :]
        else:
            means = None
        weights_ = wrapper_quadratic_optimisation(
            pd_covar=pd_covar,
            constraints=constraints,
            weights_0=weights_0,
            portfolio_objective=portfolio_objective,
            means=means,
            carra=carra,
            inclusion_indicators=inclusion_indicators1.loc[date, :],
            optimiser_config=optimiser_config,
            context=str(pd.Timestamp(date).date())
        )
        weights_0 = weights_
        prev_date = date
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers).fillna(0.0)
    return weights


def wrapper_quadratic_optimisation(pd_covar: pd.DataFrame,
                                   constraints: Constraints,
                                   inclusion_indicators: pd.Series = None,
                                   portfolio_objective: PortfolioObjective = PortfolioObjective.MIN_VARIANCE,
                                   means: pd.Series = None,  # required for QUADRATIC_UTILITY
                                   weights_0: pd.Series = None,
                                   carra: float = 1.0,
                                   optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True),
                                   context: str = ''
                                   ) -> pd.Series:
    """
    Single-date quadratic optimisation with NaN/zero-variance filtering.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        constraints: Portfolio constraints.
        inclusion_indicators: Binary series for asset eligibility.
        portfolio_objective: MIN_VARIANCE or QUADRATIC_UTILITY.
        means: Expected returns per asset. Required for QUADRATIC_UTILITY;
            filtered alongside the covariance for NaN/excluded assets.
        weights_0: Previous-period weights for warm-start / fallback.
        carra: Risk aversion coefficient for QUADRATIC_UTILITY.
        optimiser_config: Solver configuration.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    vectors = dict(means=means) if means is not None else None
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(
        pd_covar=pd_covar,
        vectors=vectors,
        inclusion_indicators=inclusion_indicators
    )
    if means is not None:
        means_np = good_vectors['means'].to_numpy()
    else:
        means_np = None

    if optimiser_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    constraints1 = constraints.update_with_valid_tickers(context=context,
        valid_tickers=clean_covar.columns.to_list(),
        total_to_good_ratio=total_to_good_ratio,
        weights_0=weights_0
    )

    weights = cvx_quadratic_optimisation(
        portfolio_objective=portfolio_objective,
        covar=clean_covar.to_numpy(),
        constraints=constraints1,
        means=means_np,
        carra=carra,
        solver=optimiser_config.solver,
        verbose=optimiser_config.verbose,
        context=context
    )
    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)
    return weights


def cvx_quadratic_optimisation(portfolio_objective: PortfolioObjective,
                               covar: np.ndarray,
                               constraints: Constraints,
                               means: np.ndarray = None,
                               verbose: bool = False,
                               solver: str = 'CLARABEL',
                               carra: float = 1.0,
                               context: str = ''
                               ) -> np.ndarray:
    """
    Solve quadratic portfolio optimisation via CVXPY.

    For MIN_VARIANCE:
        max  -w' Σ w   s.t. constraints

    For QUADRATIC_UTILITY:
        max  μ'w - (γ/2) w' Σ w   s.t. constraints

    Args:
        portfolio_objective: MIN_VARIANCE or QUADRATIC_UTILITY.
        covar: Covariance matrix (N x N).
        constraints: Portfolio constraints.
        means: Expected returns (N,). Required for QUADRATIC_UTILITY.
        verbose: If True, print CVXPY solver diagnostics.
        solver: CVXPY solver name.
        carra: Risk aversion coefficient γ.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
    """
    covar = cvx.psd_wrap(covar)
    n = covar.shape[0]
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    w = cvx.Variable(n, nonneg=nonneg)

    portfolio_var = cvx.quad_form(w, covar)

    if portfolio_objective == PortfolioObjective.MIN_VARIANCE:
        objective_fun = -portfolio_var

    elif portfolio_objective == PortfolioObjective.QUADRATIC_UTILITY:
        if means is None:
            raise ValueError(f"means must be given for QUADRATIC_UTILITY objective")
        objective_fun = means.T @ w - 0.5 * carra * portfolio_var

    else:
        raise ValueError(f"unsupported portfolio_objective: {portfolio_objective}")

    objective = cvx.Maximize(objective_fun)
    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar)
    problem = cvx.Problem(objective, constraints_)
    try:
        problem.solve(verbose=verbose, solver=solver)
        solved_status = problem.status
    except cvx.error.SolverError:
        # CLARABEL (and other backends) can raise rather than return a status when the
        # constraint geometry is numerically degenerate. Route this into the same fallback path
        # as an honestly-reported infeasibility instead of propagating and killing the run.
        w.value = None
        solved_status = 'solver_error'

    optimal_weights, _is_valid = validate_solution(
        w.value, solved_status, constraints, n, solver=solver, context=context)

    return optimal_weights


def solve_analytic_log_opt(covar: np.ndarray,
                           means: np.ndarray,
                           exposure_budget_eq: Tuple[np.ndarray, float] = None,
                           gamma: float = 1.0
                           ) -> np.ndarray:
    """
    Analytic solution for the unconstrained quadratic utility problem.

    Solves: max μ'w - (γ/2) w'Σw  subject to a'w = a₀

    Args:
        covar: Covariance matrix (N x N).
        means: Expected returns (N,).
        exposure_budget_eq: Tuple (a, a₀) defining equality constraint a'w = a₀.
        gamma: Risk aversion coefficient.

    Returns:
        Optimal weights (N,).
    """
    sigma_i = np.linalg.inv(covar)

    if exposure_budget_eq is not None:
        a = exposure_budget_eq[0]
        a0 = exposure_budget_eq[1]
        a_sigma_a = a.T @ sigma_i @ a
        a_sigma_mu = a.T @ sigma_i @ means
        l_lambda = (-gamma * a0 + a_sigma_mu) / a_sigma_a
        optimal_weights = (1.0 / gamma) * sigma_i @ (means - l_lambda * a)
    else:
        optimal_weights = (1.0 / gamma) * sigma_i @ means

    return optimal_weights