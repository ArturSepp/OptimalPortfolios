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
from optimalportfolios.utils.covar_matrix import CovarEstimator


def rolling_maximise_alpha_over_tre(prices: pd.DataFrame,
                                    alphas: pd.DataFrame,
                                    constraints0: Constraints,
                                    benchmark_weights: Union[pd.Series, pd.DataFrame],
                                    time_period: qis.TimePeriod,  # when we start building portfolios
                                    covar_estimator: CovarEstimator = CovarEstimator(),  # default covar estimator
                                    pd_covars: Dict[pd.Timestamp, pd.DataFrame] = None,
                                    rebalancing_indicators: pd.DataFrame = None,
                                    apply_total_to_good_ratio: bool = True,
                                    solver: str = 'ECOS_BB'
                                    ) -> pd.DataFrame:
    """
    maximise portfolio alpha subject to constraint on tracking tracking error
    """
    # estimate covar at rebalancing schedule
    if pd_covars is None:  # use default ewm covar with covar_estimator
        pd_covars = covar_estimator.fit_rolling_covars(prices=prices, time_period=time_period)

    rebalancing_dates = list(pd_covars.keys())
    alphas = alphas.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    weights = {}
    # extend benchmark weights
    if isinstance(benchmark_weights, pd.DataFrame):
        benchmark_weights = benchmark_weights.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)
    else:
        benchmark_weights = benchmark_weights.to_frame(name=rebalancing_dates[0]).T.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    if rebalancing_indicators is not None:  #  need to reindex at pd_covars index
        rebalancing_indicators = rebalancing_indicators.reindex(index=rebalancing_dates).fillna(0.0)  # by default no rebalancing

    weights_0 = None  # it will relax turnover constraint for the first rebalancing
    for date, pd_covar in pd_covars.items():
        if rebalancing_indicators is not None:
            rebalancing_indicators_t = rebalancing_indicators.loc[date, :]
        else:
            rebalancing_indicators_t = None
        weights_ = wrapper_maximise_alpha_over_tre(pd_covar=pd_covar,
                                                   alphas=alphas.loc[date, :],
                                                   benchmark_weights=benchmark_weights.loc[date, :],
                                                   constraints0=constraints0,
                                                   rebalancing_indicators=rebalancing_indicators_t,
                                                   weights_0=weights_0,
                                                   apply_total_to_good_ratio=apply_total_to_good_ratio,
                                                   solver=solver)

        weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list())
    return weights


def wrapper_maximise_alpha_over_tre(pd_covar: pd.DataFrame,
                                    alphas: pd.Series,
                                    benchmark_weights: pd.Series,
                                    constraints0: Constraints,
                                    weights_0: pd.Series = None,
                                    rebalancing_indicators: pd.Series = None,
                                    apply_total_to_good_ratio: bool = True,
                                    solver: str = 'ECOS_BB'
                                    ) -> pd.Series:
    """
    create wrapper accounting for nans or zeros in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    """
    # filter out assets with zero variance or nans
    vectors = dict(alphas=alphas)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)
    if apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = 1.0

    constraints = constraints0.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=total_to_good_ratio,
                                                         weights_0=weights_0,
                                                         benchmark_weights=benchmark_weights,
                                                         rebalancing_indicators=rebalancing_indicators)

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
                                solver: str = 'ECOS_BB',
                                verbose: bool = False
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
    covar = cvx.psd_wrap(covar)

    # set solver
    benchmark_weights = constraints.benchmark_weights.to_numpy()
    objective_fun = alphas.T @ (w - benchmark_weights)
    objective = cvx.Maximize(objective_fun)
    constraints_ = constraints.set_cvx_constraints(w=w, covar=covar)

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

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
