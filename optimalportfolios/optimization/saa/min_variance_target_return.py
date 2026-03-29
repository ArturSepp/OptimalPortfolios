"""
Minimum-variance portfolio optimisation with a target return constraint.

Solves the strategic asset allocation (SAA) problem:

    min_w  (w - w_b)' Σ (w - w_b)     [if benchmark provided]
    min_w  w' Σ w                      [if no benchmark]

    s.t.   α'w >= r_target             (return floor, always hard)
           ||w - w_0||_1 <= TO_max     (turnover limit, optional)
           1'w = 1                     (full investment)
           w >= 0                      (long-only, optional)
           w_min <= w <= w_max         (weight bounds)
           group constraints           (AC L1/L2 bands)

Two formulations are supported:

1. **Hard constraints** (``ConstraintEnforcementType.FORCED_CONSTRAINTS``):
   Minimise portfolio variance (or tracking error variance) subject to
   a hard return floor.

2. **Utility penalties** (``ConstraintEnforcementType.UTILITY_CONSTRAINTS``):
   The risk minimisation is combined with turnover penalties:

       min_w  (w - w_b)'Σ(w - w_b) + λ_TO ||w - w_0||_1

       s.t.   α'w >= r_target   (still hard)

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
from typing import Union, Dict

from optimalportfolios.optimization.constraints import Constraints, ConstraintEnforcementType
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.config import OptimiserConfig


def rolling_min_variance_target_return(prices: pd.DataFrame,
                                       expected_returns: pd.DataFrame,
                                       target_returns: pd.Series,
                                       constraints: Constraints,
                                       benchmark_weights: Union[pd.Series, pd.DataFrame, None],
                                       covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                       rebalancing_indicators: pd.DataFrame = None,
                                       optimiser_config: OptimiserConfig = OptimiserConfig()
                                       ) -> pd.DataFrame:
    """
    Compute rolling minimum-variance portfolios with a target return floor.

    Args:
        prices: Asset price panel for column alignment.
        expected_returns: Expected returns per asset. Forward-filled to
            rebalancing dates.
        target_returns: Minimum portfolio return at each date.
        constraints: Portfolio constraints.
        benchmark_weights: Optional benchmark. Series (static) or DataFrame
            (time-varying). None for absolute variance minimisation.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        rebalancing_indicators: Optional binary DataFrame for position freezing.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    rebalancing_dates = list(covar_dict.keys())

    expected_returns = expected_returns.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)
    target_returns = target_returns.reindex(index=rebalancing_dates, method='ffill')

    if benchmark_weights is not None:
        if isinstance(benchmark_weights, pd.DataFrame):
            benchmark_weights = benchmark_weights.reindex(
                index=rebalancing_dates, method='ffill').fillna(0.0)
        else:
            benchmark_weights = benchmark_weights.to_frame(
                name=rebalancing_dates[0]).T.reindex(
                index=rebalancing_dates, method='ffill').fillna(0.0)

    if rebalancing_indicators is not None:
        rebalancing_indicators = rebalancing_indicators.reindex(
            index=rebalancing_dates).fillna(0.0)

    weights = {}
    weights_0 = None

    for date, pd_covar in covar_dict.items():
        bw_t = benchmark_weights.loc[date, :] if benchmark_weights is not None else None
        ri_t = rebalancing_indicators.loc[date, :] if rebalancing_indicators is not None else None

        weights_ = wrapper_min_variance_target_return(
            pd_covar=pd_covar,
            expected_returns=expected_returns.loc[date, :],
            target_return=target_returns[date],
            benchmark_weights=bw_t,
            constraints=constraints,
            weights_0=weights_0,
            rebalancing_indicators=ri_t,
            optimiser_config=optimiser_config)

        if np.all(np.equal(weights_, 0.0)):
            weights_0 = None
        else:
            weights_0 = weights_
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns).fillna(0.0)
    return weights


def wrapper_min_variance_target_return(pd_covar: pd.DataFrame,
                                       expected_returns: pd.Series,
                                       target_return: float,
                                       constraints: Constraints,
                                       benchmark_weights: pd.Series = None,
                                       weights_0: pd.Series = None,
                                       rebalancing_indicators: pd.Series = None,
                                       optimiser_config: OptimiserConfig = OptimiserConfig()
                                       ) -> pd.Series:
    """
    Single-date minimum-variance optimisation with return floor.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        expected_returns: Expected returns per asset for the return constraint.
        target_return: Minimum portfolio return (α'w >= target_return).
        constraints: Portfolio constraints.
        benchmark_weights: Optional benchmark for TE-based risk.
        weights_0: Previous-period weights for warm-start / turnover.
        rebalancing_indicators: Binary series for position freezing.
        optimiser_config: Solver configuration.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    # sanity check: target return must be achievable
    max_return = expected_returns.reindex(pd_covar.index).max()
    if target_return > max_return + 1e-8:
        warnings.warn(
            f"target_return={target_return:.4%} exceeds max asset return "
            f"({max_return:.4%}). Clamping to max asset return.")
        target_return = max_return

    vectors = dict(expected_returns=expected_returns)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(
        pd_covar=pd_covar, vectors=vectors)

    if optimiser_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    # expected returns: fill NaN with 0 so they don't break the return constraint
    valid_tickers = clean_covar.columns.to_list()
    er_clean = expected_returns.reindex(index=valid_tickers)
    nan_er = er_clean.index[er_clean.isna()].tolist()
    if nan_er:
        warnings.warn(f"NaN expected returns for {nan_er}, setting to 0 in return constraint")
        er_clean = er_clean.fillna(0.0)

    constraints1 = constraints.update_with_valid_tickers(
        valid_tickers=valid_tickers,
        total_to_good_ratio=total_to_good_ratio,
        weights_0=weights_0,
        benchmark_weights=benchmark_weights,
        asset_returns=er_clean,
        target_return=target_return,
        rebalancing_indicators=rebalancing_indicators)

    if constraints.constraint_enforcement_type == ConstraintEnforcementType.UTILITY_CONSTRAINTS:
        weights = cvx_min_variance_target_return_utility(
            covar=clean_covar.to_numpy(),
            constraints=constraints1,
            has_benchmark=benchmark_weights is not None,
            solver=optimiser_config.solver,
            verbose=optimiser_config.verbose)
    else:
        weights = cvx_min_variance_target_return(
            covar=clean_covar.to_numpy(),
            constraints=constraints1,
            has_benchmark=benchmark_weights is not None,
            solver=optimiser_config.solver,
            verbose=optimiser_config.verbose)

    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=valid_tickers)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)
    return weights


def cvx_min_variance_target_return(covar: np.ndarray,
                                    constraints: Constraints,
                                    has_benchmark: bool = False,
                                    solver: str = 'CLARABEL',
                                    verbose: bool = False
                                    ) -> np.ndarray:
    """
    Minimise portfolio variance subject to a hard return floor.

    Solves:

        min_w  (w - w_b)' Σ (w - w_b)    [if has_benchmark]
        min_w  w' Σ w                     [otherwise]

        s.t.   α'w >= r_target
               1'w = 1,  w >= 0,  bounds,  groups

    Args:
        covar: Covariance matrix (N x N).
        constraints: Constraints with target_return and asset_returns injected.
        has_benchmark: If True, minimise TE variance; else absolute variance.
        solver: CVXPY solver name.
        verbose: If True, print CVXPY solver diagnostics.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
    """
    n = covar.shape[0]
    nonneg = constraints.is_long_only
    w = cvx.Variable(n, nonneg=nonneg)
    covar_psd = cvx.psd_wrap(covar)

    if has_benchmark and constraints.benchmark_weights is not None:
        w_b = constraints.benchmark_weights.to_numpy()
        risk_expr = cvx.quad_form(w - w_b, covar_psd)
    else:
        risk_expr = cvx.quad_form(w, covar_psd)

    objective = cvx.Minimize(risk_expr)
    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar_psd)

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        warnings.warn(f"cvx_min_variance_target_return: solver did not converge")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
        else:
            optimal_weights = np.zeros(n)

    return optimal_weights


def cvx_min_variance_target_return_utility(covar: np.ndarray,
                                            constraints: Constraints,
                                            has_benchmark: bool = False,
                                            solver: str = 'CLARABEL',
                                            verbose: bool = False
                                            ) -> np.ndarray:
    """
    Minimise portfolio variance with turnover penalty and hard return floor.

    Solves:

        min_w  (w - w_b)'Σ(w - w_b) + λ_TO ||w - w_0||_1   [if has_benchmark]
        min_w  w'Σw + λ_TO ||w - w_0||_1                    [otherwise]

        s.t.   α'w >= r_target   (always hard)
               1'w = 1,  w >= 0,  bounds,  groups

    Args:
        covar: Covariance matrix (N x N).
        constraints: Constraints with target_return, asset_returns, and
            optionally benchmark_weights.
        has_benchmark: If True, minimise TE variance; else absolute variance.
        solver: CVXPY solver name.
        verbose: If True, print CVXPY solver diagnostics.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
    """
    n = covar.shape[0]
    nonneg = constraints.is_long_only
    w = cvx.Variable(n, nonneg=nonneg)
    covar_psd = cvx.psd_wrap(covar)

    if has_benchmark and constraints.benchmark_weights is not None:
        w_b = constraints.benchmark_weights.to_numpy()
        risk_expr = cvx.quad_form(w - w_b, covar_psd)
    else:
        risk_expr = cvx.quad_form(w, covar_psd)

    objective_fun = risk_expr

    if constraints.weights_0 is not None and constraints.turnover_utility_weight is not None:
        if constraints.turnover_costs is not None:
            turnover_term = constraints.turnover_utility_weight * cvx.norm(
                cvx.multiply(constraints.turnover_costs.to_numpy(),
                             w - constraints.weights_0.to_numpy()), 1)
        else:
            turnover_term = constraints.turnover_utility_weight * cvx.norm(
                w - constraints.weights_0.to_numpy(), 1)
        objective_fun = objective_fun + turnover_term

    objective = cvx.Minimize(objective_fun)

    constraints_ = constraints.set_cvx_exposure_constraints(w=w)

    if constraints.target_return is not None and constraints.asset_returns is not None:
        constraints_ += [constraints.asset_returns.to_numpy() @ w >= constraints.target_return]

    if constraints.group_lower_upper_constraints is not None:
        constraints_ += constraints.group_lower_upper_constraints.set_cvx_group_lower_upper_constraints(w=w)

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        warnings.warn(f"cvx_min_variance_target_return_utility: solver did not converge")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
        else:
            optimal_weights = np.zeros(n)

    return optimal_weights
