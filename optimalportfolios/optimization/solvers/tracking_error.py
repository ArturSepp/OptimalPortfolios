"""
optimise alpha over tracking error
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
import qis as qis
from typing import Optional, List, Tuple, Union, Dict

from optimalportfolios import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.utils.covar_matrix import squeeze_covariance_matrix, estimate_rolling_ewma_covar, estimate_rolling_lasso_covar


def rolling_maximise_alpha_over_tre(prices: pd.DataFrame,
                                    alphas: pd.DataFrame,
                                    constraints0: Constraints,
                                    benchmark_weights: Union[pd.Series, pd.DataFrame],
                                    time_period: qis.TimePeriod,  # when we start building portfolios
                                    returns_freq: str = 'W-WED',
                                    rebalancing_freq: str = 'QE',
                                    span: int = 52,  # 1y
                                    squeeze_factor: Optional[float] = None,
                                    solver: str = 'ECOS_BB'
                                    ) -> pd.DataFrame:
    """
    maximise portfolio alpha subject to constraint on tracking tracking error
    """
    # estimate covar at rebalancing schedule
    pd_covars = estimate_rolling_ewma_covar(prices=prices,
                                            time_period=time_period,
                                            returns_freq=returns_freq,
                                            rebalancing_freq=rebalancing_freq,
                                            span=span)
    rebalancing_schedule = list(pd_covars.keys())
    alphas = alphas.reindex(index=rebalancing_schedule, method='ffill')

    tickers = prices.columns.to_list()
    weights = {}
    # extend benchmark weights
    if isinstance(benchmark_weights, pd.DataFrame):
        weights_0 = benchmark_weights.iloc[:, 0]
        benchmark_weights = benchmark_weights.reindex(index=rebalancing_schedule, method='ffill').fillna(0.0)
    else:
        weights_0 = benchmark_weights
        # crate df with weights
        benchmark_weights = benchmark_weights.to_frame(name=rebalancing_schedule[0]).T.reindex(index=rebalancing_schedule, method='ffill').fillna(0.0)

    for date, pd_covar in pd_covars.items():
        weights_ = wrapper_maximise_alpha_over_tre(pd_covar=pd_covar,
                                                   alphas=alphas.loc[date, :],
                                                   benchmark_weights=benchmark_weights.loc[date, :],
                                                   constraints0=constraints0,
                                                   weights_0=weights_0,
                                                   squeeze_factor=squeeze_factor,
                                                   solver=solver)
        weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers)
    return weights


def rolling_maximise_alpha_over_tre_lasso_covar(benchmark_prices: pd.DataFrame,
                                                prices: pd.DataFrame,
                                                alphas: pd.DataFrame,
                                                constraints0: Constraints,
                                                benchmark_weights: Union[pd.Series, pd.DataFrame],
                                                time_period: qis.TimePeriod,  # when we start building portfolios
                                                pd_covars: Dict[pd.Timestamp, pd.DataFrame] = None,
                                                returns_freq: str = 'W-WED',
                                                rebalancing_freq: str = 'QE',
                                                span: int = 52,  # 1y
                                                reg_lambda: float = 1e-8,
                                                squeeze_factor: Optional[float] = None,
                                                solver: str = 'ECOS_BB'
                                                ) -> pd.DataFrame:
    """
    maximise portfolio alpha subject to constraint on tracking tracking error
    """
    # estimate covar at rebalancing schedule
    if pd_covars is None:
        pd_covars = estimate_rolling_lasso_covar(benchmark_prices=benchmark_prices,
                                                 prices=prices,
                                                 time_period=time_period,
                                                 returns_freq=returns_freq,
                                                 rebalancing_freq=rebalancing_freq,
                                                 span=span,
                                                 reg_lambda=reg_lambda,
                                                 squeeze_factor=squeeze_factor)

    rebalancing_schedule = list(pd_covars.keys())
    alphas = alphas.reindex(index=rebalancing_schedule, method='ffill').fillna(0.0)

    tickers = prices.columns.to_list()
    weights = {}
    # extend benchmark weights
    if isinstance(benchmark_weights, pd.DataFrame):
        weights_0 = benchmark_weights.iloc[:, 0]
        benchmark_weights = benchmark_weights.reindex(index=rebalancing_schedule, method='ffill').fillna(0.0)
    else:
        weights_0 = benchmark_weights
        # crate df with weights
        benchmark_weights = benchmark_weights.to_frame(name=rebalancing_schedule[0]).T.reindex(index=rebalancing_schedule, method='ffill').fillna(0.0)

    for date, pd_covar in pd_covars.items():
        weights_ = wrapper_maximise_alpha_over_tre(pd_covar=pd_covar,
                                                   alphas=alphas.loc[date, :],
                                                   benchmark_weights=benchmark_weights.loc[date, :],
                                                   constraints0=constraints0,
                                                   weights_0=weights_0,
                                                   squeeze_factor=None,  # taling care of
                                                   solver=solver)
        if np.all(np.isclose(weights_, 0.0)):
            weights_ = benchmark_weights.loc[date, :]

        weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers)
    return weights


def wrapper_maximise_alpha_over_tre(pd_covar: pd.DataFrame,
                                    alphas: pd.Series,
                                    benchmark_weights: pd.Series,
                                    constraints0: Constraints,
                                    weights_0: pd.Series = None,
                                    squeeze_factor: Optional[float] = None,
                                    solver: str = 'ECOS_BB'
                                    ) -> pd.Series:
    """
    create wrapper accounting for nans or zeros in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    """
    # filter out assets with zero variance or nans
    vectors = dict(alphas=alphas)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    if squeeze_factor is not None and squeeze_factor > 0.0:
        clean_covar = squeeze_covariance_matrix(clean_covar, squeeze_factor=squeeze_factor)

    constraints = constraints0.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
                                                         weights_0=weights_0,
                                                         benchmark_weights=benchmark_weights)

    weights = cvx_maximise_alpha_over_tre(covar=clean_covar.to_numpy(),
                                          alphas=good_vectors['alphas'].to_numpy(),
                                          constraints=constraints,
                                          solver=solver)
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)  # align with tickers
    return weights


def cvx_maximise_alpha_over_tre(covar: np.ndarray,
                                alphas: np.ndarray,
                                constraints: Constraints,
                                solver: str = 'ECOS_BB'
                                ) -> np.ndarray:
    """
    numpy level one step solution of problem
    max alpha@w
    such that
    w @ Sigma @ w.t <= tracking_err_vol_constraint
    sum(abs(w-w_0)) <= turnover_constraint
    sum(w) = 1 # exposure constraint
    w >= 0  # long only constraint

    subject to linear constraints
         1. weight_min <= w <= weight_max
         2. sum(w) = 1
         3. exposure_budget_eq[0]^t*w = exposure_budget_eq[1]
    here we assume that all assets are valid: Sigma is invertable
    """
    # covar1 = cvx.psd_wrap(covar)
    n = covar.shape[0]
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    w = cvx.Variable(n, nonneg=nonneg)

    # set solver
    benchmark_weights = constraints.benchmark_weights.to_numpy()
    objective_fun = alphas.T @ (w - benchmark_weights)
    objective = cvx.Maximize(objective_fun)
    constraints_ = constraints.set_cvx_constraints(w=w, covar=covar)

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=False, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        # raise ValueError(f"not solved")
        print(f"not solved")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
            print(f"using weights_0")
        else:
            optimal_weights = np.zeros(n)
            print(f"using zeroweights")

    return optimal_weights


def minimize_tracking_error(covar: np.ndarray,
                            benchmark_weights: np.ndarray = None,
                            min_weights: np.ndarray = None,
                            max_weights: np.ndarray = None,
                            is_long_only: bool = True,
                            max_leverage: float = None,  # for long short portfolios
                            turnover_constraint: Optional[float] = 0.5,
                            weights_0: np.ndarray = None,
                            group_exposures_min_max: Optional[List[Tuple[np.ndarray, float, float]]] = None,
                            solver: str = 'ECOS_BB',
                            verbose: bool = False
                            ) -> np.ndarray:
    """
    TODO: revise
    max alpha@w
    such that
    w @ Sigma @ w.t <= tracking_err_vol_constraint
    sum(abs(w-w_0)) <= turnover_constraint
    sum(w) = 1 # exposure constraint
    w >= 0  # long only constraint

    subject to linear constraints
         1. weight_min <= w <= weight_max
         2. sum(w) = 1
         3. exposure_budget_eq[0]^t*w = exposure_budget_eq[1]
    """
    n = covar.shape[0]
    w = cvx.Variable(n)
    covar = cvx.psd_wrap(covar)
    tracking_error_var = cvx.quad_form(w-benchmark_weights, covar)

    objective_fun = tracking_error_var

    objective = cvx.Minimize(objective_fun)

    # add constraints
    constraints = []
    # gross_notional = 1:
    constraints = constraints + [cvx.sum(w) == 1]

    # tracking error constraint
    # constraints += [tracking_error_var <= tracking_err_vol_constraint ** 2]  # variance constraint

    # turnover_constraint:
    if turnover_constraint is not None:
        if weights_0 is None:
            raise ValueError(f"weights_0 must be given")
        constraints += [cvx.norm(w-weights_0, 1) <= turnover_constraint]

    if is_long_only:
        constraints = constraints + [w >= 0.0]
    if min_weights is not None:
        constraints = constraints + [w >= min_weights]
    if max_weights is not None:
        constraints = constraints + [w <= max_weights]
    if group_exposures_min_max is not None:
        for group_exposures_min_max_ in group_exposures_min_max:
            constraints = constraints + [group_exposures_min_max_[0] @ w >= group_exposures_min_max_[1]]
            constraints = constraints + [group_exposures_min_max_[0] @ w <= group_exposures_min_max_[2]]

    if max_leverage is not None:
        constraints = constraints + [cvx.norm(w, 1) <= max_leverage]

    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        raise ValueError(f"not solved")

    #if group_exposures_min_max is not None:
    #    for group_exposures_min_max_ in group_exposures_min_max:
    #        print(f"exposure: {group_exposures_min_max_[1]} <= {group_exposures_min_max_[0] @ optimal_weights} <= {group_exposures_min_max_[2]} ")

    return optimal_weights
