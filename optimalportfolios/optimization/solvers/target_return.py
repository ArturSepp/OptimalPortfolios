"""
Alpha-maximising portfolio optimisation with a target return constraint.

Solves the tactical asset allocation (TAA) problem:

    max_w  α'w

    s.t.   y'w >= r_target        (return constraint)
           w'Σw <= σ²_max         (risk constraint, optional)
           1'w = 1                (full investment)
           w >= 0                 (long-only, optional)
           w_min <= w <= w_max    (weight bounds)

where α is the vector of expected alphas (excess returns from active views),
y is the vector of asset yields or expected returns, r_target is the minimum
portfolio return, and Σ is the covariance matrix.

This formulation separates the alpha signal (what we want to maximise) from
the return constraint (what we need to deliver). This is typical in
multi-asset TAA where the portfolio must meet a minimum yield or income
target while the alpha overlay tilts toward the best risk-adjusted
opportunities within the feasible set.

The covariance matrices are produced externally by any CovarEstimator
(EwmaCovarEstimator, FactorCovarEstimator, etc.), and the alphas and yields
are provided as time-indexed DataFrames aligned to the rebalancing schedule.

Reference:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation
    for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
    Available at https://www.pm-research.com/content/iijpormgmt/52/4/86
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
from typing import Dict

from optimalportfolios import filter_covar_and_vectors_for_nans, estimate_rolling_ewma_covar
from optimalportfolios.optimization.constraints import Constraints


def rolling_maximise_alpha_with_target_return(prices: pd.DataFrame,
                                              alphas: pd.DataFrame,
                                              yields: pd.DataFrame,
                                              target_returns: pd.Series,
                                              constraints: Constraints,
                                              covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                              solver: str = 'ECOS_BB',
                                              verbose: bool = False
                                              ) -> pd.DataFrame:
    """
    Compute rolling alpha-maximising portfolios with a target return constraint.

    At each rebalancing date (defined by the keys of ``covar_dict``), solves:

        max_w  α_t'w   s.t.  y_t'w >= r_t,  constraints

    where α_t, y_t, and r_t are the alpha signal, asset yields, and target
    return at date t. Previous-period weights are passed as warm-start to
    stabilise turnover.

    Alphas, yields, and target returns are forward-filled to the rebalancing
    schedule, so they can be provided at any frequency (e.g., monthly signals
    with quarterly rebalancing).

    Args:
        prices: Asset price panel. Index=dates, columns=tickers. Used only
            for column alignment of the output weights DataFrame.
        alphas: Alpha signals per asset. Index=dates, columns=tickers.
            Forward-filled to rebalancing dates. Represents the expected
            excess return from active views (e.g., from a CMA model).
        yields: Expected asset returns or yields. Index=dates, columns=tickers.
            Forward-filled to rebalancing dates. Used in the return constraint
            y'w >= r_target (e.g., carry, yield-to-worst, expected income).
        target_returns: Minimum portfolio return at each date. Index=dates.
            Forward-filled to rebalancing dates. The constraint y'w >= r_t
            ensures the portfolio delivers at least this return level.
        constraints: Portfolio constraints (long-only, weight bounds, group
            exposures, vol budget, etc.).
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
            Typically produced by ``estimator.fit_rolling_covars()``.
            The dict keys define the rebalancing schedule.
        solver: CVXPY solver name.
        verbose: If True, print inputs at each rebalancing date for debugging.

    Returns:
        DataFrame of portfolio weights. Index=rebalancing dates from covar_dict,
        columns=tickers aligned to ``prices.columns``.
    """
    # align signals to the rebalancing schedule via forward-fill
    rebalancing_schedule = list(covar_dict.keys())
    alphas = alphas.reindex(index=rebalancing_schedule, method='ffill')
    yields = yields.reindex(index=rebalancing_schedule, method='ffill')
    target_returns = target_returns.reindex(index=rebalancing_schedule, method='ffill')

    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():

        if verbose:
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
            solver=solver
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
                                              solver: str = 'ECOS_BB'
                                              ) -> pd.Series:
    """
    Single-date alpha maximisation with NaN/zero-variance filtering.

    Removes assets with NaN or zero diagonal entries in the covariance
    matrix, updates the constraint set for the reduced universe (including
    the return constraint y'w >= r_target), solves, and maps weights back
    to the full asset universe.

    The return constraint is passed to the constraint object via
    ``update_with_valid_tickers(asset_returns=yields, target_return=...)``,
    which adds it as a linear inequality in the CVXPY problem.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        alphas: Alpha signal per asset. Index=tickers.
        yields: Expected returns per asset for the return constraint.
        target_return: Minimum portfolio return level (y'w >= target_return).
        constraints: Portfolio constraints.
        weights_0: Previous-period weights for warm-start / fallback.
        solver: CVXPY solver name.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    # filter out assets with zero variance or NaN alphas
    vectors = dict(alphas=alphas)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    # update constraints for the valid subset: rescale weight bounds and
    # inject the return constraint y'w >= r_target
    constraints1 = constraints.update_with_valid_tickers(
        valid_tickers=clean_covar.columns.to_list(),
        total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
        weights_0=weights_0,
        asset_returns=yields,
        target_return=target_return
    )

    weights = cvx_maximise_alpha_with_target_return(
        covar=clean_covar.to_numpy(),
        alphas=good_vectors['alphas'].to_numpy(),
        constraints=constraints1,
        solver=solver
    )
    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)

    return weights


def cvx_maximise_alpha_with_target_return(covar: np.ndarray,
                                          alphas: np.ndarray,
                                          constraints: Constraints,
                                          verbose: bool = False,
                                          solver: str = 'ECOS_BB'
                                          ) -> np.ndarray:
    """
    Solve alpha-maximising portfolio allocation via CVXPY.

    Solves the linear-objective convex programme:

        max_w  α'w

        s.t.   y'w >= r_target    (encoded in constraints)
               w'Σw <= σ²_max     (encoded in constraints, optional)
               1'w = 1            (full investment)
               w >= 0             (long-only, if enabled)
               w_min <= w <= w_max

    The objective is linear in w (maximise alpha exposure), while risk
    and return constraints define the feasible set. This is a second-order
    cone programme (SOCP) when the vol constraint is active, and a linear
    programme (LP) when only the return and weight constraints bind.

    Note: the covariance matrix enters only through the constraints
    (vol budget), not through the objective. This separates the alpha
    signal from the risk model — the optimizer tilts toward high-alpha
    assets subject to risk and return feasibility.

    Args:
        covar: Covariance matrix (N x N) as numpy array. Used by constraints
            for vol budget enforcement.
        alphas: Alpha signal vector (N,). The objective maximises α'w.
        constraints: Portfolio constraints including return target and
            optional vol budget. Generated by ``update_with_valid_tickers``.
        verbose: If True, print CVXPY solver diagnostics.
        solver: CVXPY solver name.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros if the
        solver fails (e.g., infeasible return target given risk constraints).
    """
    n = covar.shape[0]
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    w = cvx.Variable(n, nonneg=nonneg)

    # linear objective: maximise alpha exposure
    objective_fun = alphas.T @ w
    objective = cvx.Maximize(objective_fun)

    # all constraints (weight bounds, full investment, return target, vol budget)
    # are assembled by the Constraints object
    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar)
    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        print(f"not solved")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
            print(f"using weights_0")
        else:
            optimal_weights = np.zeros(n)
            print(f"using zero weights")

    return optimal_weights
