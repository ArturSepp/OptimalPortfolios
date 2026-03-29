"""
Alpha-maximising portfolio optimisation with tracking error constraints.

Solves the tactical asset allocation (TAA) problem relative to a benchmark:

    max_w  α'(w - w_b)

    s.t.   (w - w_b)' Σ (w - w_b) <= TE²_max   (tracking error budget)
           ||w - w_0||_1 <= TO_max               (turnover limit, optional)
           1'w = 1                                (full investment)
           w >= 0                                 (long-only, optional)
           w_min <= w <= w_max                    (weight bounds)

Two formulations are supported:

1. **Hard constraints** (``ConstraintEnforcementType.HARD_CONSTRAINTS``):
   The tracking error and turnover limits are enforced as explicit CVXPY
   constraints.

2. **Utility penalties** (``ConstraintEnforcementType.UTILITY_CONSTRAINTS``):
   The tracking error and turnover are penalised in the objective:

       max_w  α'(w - w_b) - λ_TE (w - w_b)'Σ(w - w_b) - λ_TO ||w - w_0||_1

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
from typing import Optional, Union, Dict

# optimalportfolios
from optimalportfolios.optimization.constraints import Constraints, ConstraintEnforcementType
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.utils.portfolio_funcs import compute_portfolio_risk_contribution_outputs
from optimalportfolios.optimization.config import OptimiserConfig


def rolling_maximise_alpha_over_tre(prices: pd.DataFrame,
                                    alphas: Optional[pd.DataFrame],
                                    constraints: Constraints,
                                    benchmark_weights: Union[pd.Series, pd.DataFrame],
                                    covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                    rebalancing_indicators: pd.DataFrame = None,
                                    optimiser_config: OptimiserConfig = OptimiserConfig()
                                    ) -> pd.DataFrame:
    """
    Compute rolling alpha-maximising portfolios with tracking error control.

    Args:
        prices: Asset price panel. Used for column alignment.
        alphas: Alpha signals per asset. Forward-filled. None for pure tracking.
        constraints: Portfolio constraints including TE budget.
        benchmark_weights: SAA benchmark. Series (static) or DataFrame (time-varying).
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        rebalancing_indicators: Optional binary DataFrame for position freezing.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    rebalancing_dates = list(covar_dict.keys())

    if alphas is not None:
        alphas = alphas.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    if isinstance(benchmark_weights, pd.DataFrame):
        benchmark_weights = benchmark_weights.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)
    else:
        benchmark_weights = benchmark_weights.to_frame(
            name=rebalancing_dates[0]).T.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    if rebalancing_indicators is not None:
        rebalancing_indicators = rebalancing_indicators.reindex(index=rebalancing_dates).fillna(0.0)

    weights = {}
    weights_0 = None

    for date, pd_covar in covar_dict.items():
        rebalancing_indicators_t = (
            rebalancing_indicators.loc[date, :] if rebalancing_indicators is not None else None
        )
        alphas_t = alphas.loc[date, :] if alphas is not None else None

        weights_ = wrapper_maximise_alpha_over_tre(
            pd_covar=pd_covar,
            alphas=alphas_t,
            benchmark_weights=benchmark_weights.loc[date, :],
            constraints=constraints,
            rebalancing_indicators=rebalancing_indicators_t,
            weights_0=weights_0,
            optimiser_config=optimiser_config
        )

        if np.all(np.equal(weights_, 0.0)):
            weights_0 = None
        else:
            weights_0 = weights_
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list()).fillna(0.0)
    return weights


def wrapper_maximise_alpha_over_tre(pd_covar: pd.DataFrame,
                                    alphas: Optional[pd.Series],
                                    benchmark_weights: pd.Series,
                                    constraints: Constraints,
                                    weights_0: pd.Series = None,
                                    rebalancing_indicators: pd.Series = None,
                                    detailed_output: bool = False,
                                    optimiser_config: OptimiserConfig = OptimiserConfig()
                                    ) -> Union[pd.Series, pd.DataFrame]:
    """
    Single-date alpha-over-TE optimisation with NaN filtering and routing.

    Routes to ``cvx_maximise_alpha_over_tre`` (hard constraints) or
    ``cvx_maximise_tre_utility`` (utility penalties) based on
    ``constraint_enforcement_type``.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        alphas: Alpha signal per asset. None for pure tracking.
        benchmark_weights: SAA benchmark weights.
        constraints: Portfolio constraints including TE budget.
        weights_0: Previous-period weights for warm-start / turnover.
        rebalancing_indicators: Binary series for position freezing.
        detailed_output: If True, return DataFrame with risk contribution diagnostics.
        optimiser_config: Solver configuration.

    Returns:
        Portfolio weights as pd.Series (or DataFrame if detailed_output=True).
    """
    if alphas is None:
        vectors = None
    else:
        vectors = dict(alphas=alphas)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    if optimiser_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    valid_tickers = clean_covar.columns.to_list()
    constraints1 = constraints.update_with_valid_tickers(
        valid_tickers=valid_tickers,
        total_to_good_ratio=total_to_good_ratio,
        weights_0=weights_0,
        benchmark_weights=benchmark_weights,
        rebalancing_indicators=rebalancing_indicators
    )

    alphas_np = good_vectors['alphas'].to_numpy() if alphas is not None else None

    if constraints.constraint_enforcement_type == ConstraintEnforcementType.UTILITY_CONSTRAINTS:
        weights = cvx_maximise_tre_utility(
            covar=clean_covar.to_numpy(),
            alphas=alphas_np,
            constraints=constraints1,
            solver=optimiser_config.solver,
            verbose=optimiser_config.verbose
        )
    else:
        weights = cvx_maximise_alpha_over_tre(
            covar=clean_covar.to_numpy(),
            alphas=alphas_np,
            constraints=constraints1,
            solver=optimiser_config.solver,
            verbose=optimiser_config.verbose
        )

    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=valid_tickers)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)

    if detailed_output:
        out = compute_portfolio_risk_contribution_outputs(weights=weights, clean_covar=clean_covar)
    else:
        out = weights
    return out


def cvx_maximise_alpha_over_tre(covar: np.ndarray,
                                alphas: np.ndarray,
                                constraints: Constraints,
                                solver: str = 'CLARABEL',
                                verbose: bool = False
                                ) -> np.ndarray:
    """
    Maximise active alpha subject to a hard tracking error constraint.

    Solves:

        max_w  α'(w - w_b)

        s.t.   (w - w_b)' Σ (w - w_b) <= TE²_max
               ||w - w_0||_1 <= TO_max   (if active)
               1'w = 1,  w >= 0,  bounds

    Args:
        covar: Covariance matrix (N x N).
        alphas: Alpha signal vector (N,).
        constraints: Constraints with benchmark_weights and TE budget injected.
        solver: CVXPY solver name.
        verbose: If True, print CVXPY solver diagnostics.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
    """
    n = covar.shape[0]
    nonneg = constraints.is_long_only
    w = cvx.Variable(n, nonneg=nonneg)
    covar = cvx.psd_wrap(covar)

    benchmark_weights = constraints.benchmark_weights.to_numpy()
    objective_fun = alphas.T @ (w - benchmark_weights)
    objective = cvx.Maximize(objective_fun)

    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar)

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        warnings.warn(f"cvx_maximise_alpha_over_tre: solver did not converge")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
        else:
            optimal_weights = np.zeros(n)

    return optimal_weights


def cvx_maximise_tre_utility(covar: np.ndarray,
                             constraints: Constraints,
                             alphas: Optional[np.ndarray] = None,
                             solver: str = 'CLARABEL',
                             verbose: bool = False
                             ) -> np.ndarray:
    """
    Maximise utility with tracking error and turnover penalties.

    Solves:

        max_w  α'(w - w_b) - λ_TE (w - w_b)'Σ(w - w_b) - λ_TO ||w - w_0||_1

        s.t.   1'w = 1,  w >= 0,  bounds

    Args:
        covar: Covariance matrix (N x N).
        constraints: Constraints with benchmark_weights and penalty weights.
        alphas: Alpha signal vector (N,). None for pure benchmark tracking.
        solver: CVXPY solver name.
        verbose: If True, print CVXPY solver diagnostics.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
    """
    n = covar.shape[0]
    nonneg = constraints.is_long_only
    w = cvx.Variable(n, nonneg=nonneg)
    covar = cvx.psd_wrap(covar)

    constraints1 = constraints.copy()
    objective_fun, constraints_ = constraints1.set_cvx_utility_objective_constraints(
        w=w,
        alphas=alphas,
        covar=covar
    )

    problem = cvx.Problem(cvx.Maximize(objective_fun), constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        warnings.warn(f"cvx_maximise_tre_utility: solver did not converge")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
        else:
            optimal_weights = np.zeros(n)

    return optimal_weights
