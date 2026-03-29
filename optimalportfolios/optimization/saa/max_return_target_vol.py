"""
Return-maximising portfolio optimisation with a volatility constraint.

Solves the strategic asset allocation (SAA) problem:

    max_w  α'(w - w_b)                [if benchmark provided]
    max_w  α'w                        [if no benchmark]

    s.t.   (w - w_b)' Σ (w - w_b) <= σ²_max   [if benchmark: TE budget]
           w' Σ w <= σ²_max                     [if no benchmark: abs vol]
           ||w - w_0||_1 <= TO_max              (turnover limit, optional)
           1'w = 1                              (full investment)
           w >= 0                               (long-only, optional)
           w_min <= w <= w_max                  (weight bounds)
           group constraints                    (AC L1/L2 bands)

Two formulations are supported:

1. **Hard constraints** (``ConstraintEnforcementType.FORCED_CONSTRAINTS``):
   The volatility budget is enforced as a hard SOCP constraint.

2. **Utility penalties** (``ConstraintEnforcementType.UTILITY_CONSTRAINTS``):
   The volatility and turnover are penalised in the objective.

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

from optimalportfolios.optimization.constraints import Constraints, ConstraintEnforcementType
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.config import OptimiserConfig


def rolling_max_return_target_vol(prices: pd.DataFrame,
                                  expected_returns: pd.DataFrame,
                                  target_vols: pd.Series,
                                  constraints: Constraints,
                                  benchmark_weights: Union[pd.Series, pd.DataFrame, None],
                                  covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                  rebalancing_indicators: pd.DataFrame = None,
                                  optimiser_config: OptimiserConfig = OptimiserConfig()
                                  ) -> pd.DataFrame:
    """
    Compute rolling return-maximising portfolios with a volatility budget.

    Args:
        prices: Asset price panel for column alignment.
        expected_returns: Expected returns per asset. Forward-filled.
        target_vols: Maximum portfolio volatility at each date.
        constraints: Portfolio constraints.
        benchmark_weights: Optional benchmark. None for absolute vol constraint.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        rebalancing_indicators: Optional binary DataFrame for position freezing.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    rebalancing_dates = list(covar_dict.keys())

    expected_returns = expected_returns.reindex(
        index=rebalancing_dates, method='ffill').fillna(0.0)
    target_vols = target_vols.reindex(index=rebalancing_dates, method='ffill')

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

        weights_ = wrapper_max_return_target_vol(
            pd_covar=pd_covar,
            expected_returns=expected_returns.loc[date, :],
            target_vol=target_vols[date],
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


def wrapper_max_return_target_vol(pd_covar: pd.DataFrame,
                                  expected_returns: pd.Series,
                                  target_vol: float,
                                  constraints: Constraints,
                                  benchmark_weights: pd.Series = None,
                                  weights_0: pd.Series = None,
                                  rebalancing_indicators: pd.Series = None,
                                  optimiser_config: OptimiserConfig = OptimiserConfig()
                                  ) -> pd.Series:
    """
    Single-date return maximisation with volatility budget.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        expected_returns: Expected returns per asset (α = CMAs).
        target_vol: Maximum portfolio volatility or tracking error.
        constraints: Portfolio constraints.
        benchmark_weights: Optional benchmark for TE-based risk.
        weights_0: Previous-period weights for warm-start / turnover.
        rebalancing_indicators: Binary series for position freezing.
        optimiser_config: Solver configuration.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    vectors = dict(expected_returns=expected_returns)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(
        pd_covar=pd_covar, vectors=vectors)

    if optimiser_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    valid_tickers = clean_covar.columns.to_list()

    # expected returns: fill NaN with 0 so they don't break the objective
    er_clean = expected_returns.reindex(index=valid_tickers)
    nan_er = er_clean.index[er_clean.isna()].tolist()
    if nan_er:
        warnings.warn(f"NaN expected returns for {nan_er}, setting to 0")
        er_clean = er_clean.fillna(0.0)

    # wire vol budget into constraints based on benchmark presence
    if benchmark_weights is not None:
        constraints1 = constraints.update_with_valid_tickers(
            valid_tickers=valid_tickers,
            total_to_good_ratio=total_to_good_ratio,
            weights_0=weights_0,
            benchmark_weights=benchmark_weights,
            rebalancing_indicators=rebalancing_indicators)
        # override TE constraint with target_vol
        constraints1 = Constraints(**{
            **constraints1._to_dict(),
            'tracking_err_vol_constraint': target_vol})
    else:
        constraints1 = constraints.update_with_valid_tickers(
            valid_tickers=valid_tickers,
            total_to_good_ratio=total_to_good_ratio,
            weights_0=weights_0,
            rebalancing_indicators=rebalancing_indicators)
        # absolute vol constraint
        constraints1 = Constraints(**{
            **constraints1._to_dict(),
            'max_target_portfolio_vol_an': target_vol})

    alphas_np = er_clean.to_numpy()

    if constraints.constraint_enforcement_type == ConstraintEnforcementType.UTILITY_CONSTRAINTS:
        weights = cvx_max_return_target_vol_utility(
            covar=clean_covar.to_numpy(),
            alphas=alphas_np,
            constraints=constraints1,
            has_benchmark=benchmark_weights is not None,
            solver=optimiser_config.solver,
            verbose=optimiser_config.verbose)
    else:
        weights = cvx_max_return_target_vol(
            covar=clean_covar.to_numpy(),
            alphas=alphas_np,
            constraints=constraints1,
            has_benchmark=benchmark_weights is not None,
            solver=optimiser_config.solver,
            verbose=optimiser_config.verbose)

    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=valid_tickers)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)
    return weights


def cvx_max_return_target_vol(covar: np.ndarray,
                               alphas: np.ndarray,
                               constraints: Constraints,
                               has_benchmark: bool = False,
                               solver: str = 'CLARABEL',
                               verbose: bool = False
                               ) -> np.ndarray:
    """
    Maximise expected return subject to a hard volatility constraint.

    Args:
        covar: Covariance matrix (N x N).
        alphas: Expected return vector (N,).
        constraints: Constraints with vol budget already injected.
        has_benchmark: If True, objective is active return α'(w - w_b).
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
        objective_fun = alphas.T @ (w - w_b)
    else:
        objective_fun = alphas.T @ w

    objective = cvx.Maximize(objective_fun)
    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar_psd)

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        warnings.warn(f"cvx_max_return_target_vol: solver did not converge")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
        else:
            optimal_weights = np.zeros(n)

    return optimal_weights


def cvx_max_return_target_vol_utility(covar: np.ndarray,
                                       alphas: np.ndarray,
                                       constraints: Constraints,
                                       has_benchmark: bool = False,
                                       solver: str = 'CLARABEL',
                                       verbose: bool = False
                                       ) -> np.ndarray:
    """
    Maximise return with volatility and turnover penalties.

    Args:
        covar: Covariance matrix (N x N).
        alphas: Expected return vector (N,).
        constraints: Constraints with penalty weights.
        has_benchmark: If True, use benchmark-relative formulation.
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
        constraints1 = constraints.copy()
        objective_fun, constraints_ = constraints1.set_cvx_utility_objective_constraints(
            w=w, alphas=alphas, covar=covar_psd)
    else:
        objective_fun = alphas.T @ w

        if constraints.tre_utility_weight is not None:
            objective_fun = objective_fun - constraints.tre_utility_weight * cvx.quad_form(w, covar_psd)

        if constraints.weights_0 is not None and constraints.turnover_utility_weight is not None:
            if constraints.turnover_costs is not None:
                objective_fun = objective_fun - constraints.turnover_utility_weight * cvx.norm(
                    cvx.multiply(constraints.turnover_costs.to_numpy(),
                                 w - constraints.weights_0.to_numpy()), 1)
            else:
                objective_fun = objective_fun - constraints.turnover_utility_weight * cvx.norm(
                    w - constraints.weights_0.to_numpy(), 1)

        constraints_ = constraints.set_cvx_exposure_constraints(w=w)
        if constraints.group_lower_upper_constraints is not None:
            constraints_ += constraints.group_lower_upper_constraints.set_cvx_group_lower_upper_constraints(w=w)

    objective = cvx.Maximize(objective_fun)

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        warnings.warn(f"cvx_max_return_target_vol_utility: solver did not converge")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
        else:
            optimal_weights = np.zeros(n)

    return optimal_weights
