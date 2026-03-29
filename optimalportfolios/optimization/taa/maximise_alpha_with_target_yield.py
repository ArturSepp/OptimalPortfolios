"""
Alpha-maximising portfolio optimisation with a target return constraint.

Solves the tactical asset allocation (TAA) problem:

    max_w  α'w

    s.t.   y'w >= r_target        (return constraint, optional)
           w'Σw <= σ²_max         (risk constraint, optional)
           1'w = 1                (full investment)
           w >= 0                 (long-only, optional)
           w_min <= w <= w_max    (weight bounds)

where α is the vector of expected alphas (excess returns from active views),
y is the vector of asset yields or expected returns, r_target is the minimum
portfolio return, and Σ is the covariance matrix.

This formulation separates the alpha signal (what we want to maximise) from
the return constraint (what we need to deliver).

Reference:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation
    for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
    Available at https://www.pm-research.com/content/iijpormgmt/52/4/86
"""
import warnings
import numpy as np
import pandas as pd
import cvxpy as cvx
from typing import Dict

from optimalportfolios import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.config import OptimiserConfig


def rolling_maximise_alpha_with_target_return(prices: pd.DataFrame,
                                              alphas: pd.DataFrame,
                                              yields: pd.DataFrame,
                                              target_returns: pd.Series,
                                              constraints: Constraints,
                                              covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                              optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                              ) -> pd.DataFrame:
    """
    Compute rolling alpha-maximising portfolios with a target return constraint.

    At each rebalancing date (defined by the keys of ``covar_dict``), solves:

        max_w  α_t'w   s.t.  y_t'w >= r_t,  constraints

    Alphas, yields, and target returns are forward-filled to the rebalancing
    schedule.

    Args:
        prices: Asset price panel. Used for column alignment.
        alphas: Alpha signals per asset. Forward-filled to rebalancing dates.
        yields: Expected asset returns or yields. Forward-filled.
        target_returns: Minimum portfolio return at each date. Forward-filled.
        constraints: Portfolio constraints.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    rebalancing_schedule = list(covar_dict.keys())
    alphas = alphas.reindex(index=rebalancing_schedule, method='ffill')
    yields = yields.reindex(index=rebalancing_schedule, method='ffill')
    target_returns = target_returns.reindex(index=rebalancing_schedule, method='ffill')

    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():

        if optimiser_config.verbose:
            print(f"date={date}")
            print(f"pd_covar=\n{pd_covar}")
            print(f"alphas=\n{alphas.loc[date, :]}")
            print(f"yields=\n{yields.loc[date, :]}")
            print(f"target_return=\n{target_returns[date]}")

        weights_ = wrapper_maximise_alpha_with_target_return(
            pd_covar=pd_covar,
            alphas=alphas.loc[date, :],
            yields=yields.loc[date, :],
            target_return=target_returns[date],
            constraints=constraints,
            weights_0=weights_0,
            optimiser_config=optimiser_config
        )

        weights_0 = weights_  # warm-start next period
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns).fillna(0.0)
    return weights


def wrapper_maximise_alpha_with_target_return(pd_covar: pd.DataFrame,
                                              alphas: pd.Series,
                                              yields: pd.Series,
                                              target_return: float,
                                              constraints: Constraints,
                                              weights_0: pd.Series = None,
                                              optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                              ) -> pd.Series:
    """
    Single-date alpha maximisation with NaN/zero-variance filtering.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        alphas: Alpha signal per asset.
        yields: Expected returns per asset for the return constraint.
        target_return: Minimum portfolio return level (y'w >= target_return).
        constraints: Portfolio constraints.
        weights_0: Previous-period weights for warm-start / fallback.
        optimiser_config: Solver configuration.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    vectors = dict(alphas=alphas)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    if optimiser_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    # yields: fill NaN with 0 so they don't break the return constraint
    # assets with NaN yield contribute zero to portfolio yield — conservative
    valid_tickers = clean_covar.columns.to_list()
    yields_clean = yields.reindex(index=valid_tickers)
    nan_yields = yields_clean.index[yields_clean.isna()].tolist()
    if nan_yields:
        warnings.warn(f"NaN yields for {nan_yields}, setting to 0 in return constraint")
        yields_clean = yields_clean.fillna(0.0)

    constraints1 = constraints.update_with_valid_tickers(
        valid_tickers=valid_tickers,
        total_to_good_ratio=total_to_good_ratio,
        weights_0=weights_0,
        asset_returns=yields_clean,
        target_return=target_return
    )

    weights = cvx_maximise_alpha_with_target_return(
        covar=clean_covar.to_numpy(),
        alphas=good_vectors['alphas'].to_numpy(),
        constraints=constraints1,
        solver=optimiser_config.solver,
        verbose=optimiser_config.verbose
    )
    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)

    return weights


def cvx_maximise_alpha_with_target_return(covar: np.ndarray,
                                          alphas: np.ndarray,
                                          constraints: Constraints,
                                          verbose: bool = False,
                                          solver: str = 'CLARABEL'
                                          ) -> np.ndarray:
    """
    Solve alpha-maximising portfolio allocation via CVXPY.

    Solves:

        max_w  α'w

        s.t.   y'w >= r_target    (encoded in constraints)
               w'Σw <= σ²_max     (encoded in constraints, optional)
               1'w = 1,  w >= 0,  bounds

    Args:
        covar: Covariance matrix (N x N).
        alphas: Alpha signal vector (N,).
        constraints: Portfolio constraints including return target.
        verbose: If True, print CVXPY solver diagnostics.
        solver: CVXPY solver name.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
    """
    n = covar.shape[0]
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    w = cvx.Variable(n, nonneg=nonneg)

    objective_fun = alphas.T @ w
    objective = cvx.Maximize(objective_fun)

    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=cvx.psd_wrap(covar))

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        warnings.warn(f"cvx_maximise_alpha_with_target_return: solver did not converge")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
        else:
            optimal_weights = np.zeros(n)

    return optimal_weights
