"""
optimise alpha over tracking error
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
import qis as qis
from typing import Optional, Union, Dict

from optimalportfolios import filter_covar_and_vectors_for_nans, compute_portfolio_risk_contribution_outputs
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator


def rolling_maximise_alpha_over_tre(prices: pd.DataFrame,
                                    alphas: Optional[pd.DataFrame],
                                    constraints0: Constraints,
                                    benchmark_weights: Union[pd.Series, pd.DataFrame],
                                    time_period: qis.TimePeriod,  # when we start building portfolios
                                    covar_estimator: CovarEstimator = CovarEstimator(),  # default covar estimator
                                    covar_dict: Dict[pd.Timestamp, pd.DataFrame] = None,
                                    rebalancing_indicators: pd.DataFrame = None,
                                    apply_total_to_good_ratio: bool = True,
                                    is_apply_tre_utility_objective: bool = False,
                                    solver: str = 'ECOS_BB'
                                    ) -> pd.DataFrame:
    """
    maximise portfolio alpha subject to constraint on tracking error
    """
    # estimate covar at rebalancing schedule
    if covar_dict is None:  # use default ewm covar with covar_estimator
        covar_dict = covar_estimator.fit_rolling_covars(prices=prices, time_period=time_period).y_covars
    rebalancing_dates = list(covar_dict.keys())

    if alphas is None:
        is_apply_tre_utility_objective = True
    else:
        alphas = alphas.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    weights = {}
    # extend benchmark weights
    if isinstance(benchmark_weights, pd.DataFrame):
        benchmark_weights = benchmark_weights.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)
    else:  # for series do transformation
        benchmark_weights = benchmark_weights.to_frame(
            name=rebalancing_dates[0]).T.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    if rebalancing_indicators is not None:  # need to reindex at covar_dict index: by default no rebalancing
        rebalancing_indicators = rebalancing_indicators.reindex(index=rebalancing_dates).fillna(0.0)

    weights_0 = None  # it will relax turnover constraint for the first rebalancing
    for date, pd_covar in covar_dict.items():
        if rebalancing_indicators is not None:
            rebalancing_indicators_t = rebalancing_indicators.loc[date, :]
        else:
            rebalancing_indicators_t = None
        alphas_t = alphas.loc[date, :] if alphas is not None else None
        weights_ = wrapper_maximise_alpha_over_tre(pd_covar=pd_covar,
                                                   alphas=alphas_t,
                                                   benchmark_weights=benchmark_weights.loc[date, :],
                                                   constraints0=constraints0,
                                                   rebalancing_indicators=rebalancing_indicators_t,
                                                   weights_0=weights_0,
                                                   apply_total_to_good_ratio=apply_total_to_good_ratio,
                                                   is_apply_tre_utility_objective=is_apply_tre_utility_objective,
                                                   solver=solver)
        if np.all(np.equal(weights_, 0.0)):
            weights_0 = None
        else:
            weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list())
    return weights


def wrapper_maximise_alpha_over_tre(pd_covar: pd.DataFrame,
                                    alphas: Optional[pd.Series],
                                    benchmark_weights: pd.Series,
                                    constraints0: Constraints,
                                    weights_0: pd.Series = None,
                                    rebalancing_indicators: pd.Series = None,
                                    apply_total_to_good_ratio: bool = True,
                                    solver: str = 'ECOS_BB',
                                    detailed_output: bool = False,
                                    is_apply_tre_utility_objective: bool = False,
                                    verbose: bool = False
                                    ) -> Union[pd.Series, pd.DataFrame]:
    """
    create wrapper accounting for nans or zeros in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    """
    # filter out assets with zero variance or nans
    if alphas is None:
        is_apply_tre_utility_objective = True
        vectors = None
    else:
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

    if alphas is not None:
        alphas_np = good_vectors['alphas'].to_numpy()
    else:
        alphas_np = None

    if is_apply_tre_utility_objective:
        weights = cvx_maximise_tre_utility(covar=clean_covar.to_numpy(),
                                           alphas=alphas_np,
                                           constraints=constraints,
                                           solver=solver,
                                           verbose=verbose)
    else:
        weights = cvx_maximise_alpha_over_tre(covar=clean_covar.to_numpy(),
                                              alphas=alphas_np,
                                              constraints=constraints,
                                              solver=solver,
                                              verbose=verbose)

    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)  # align with tickers

    if detailed_output:
        out = compute_portfolio_risk_contribution_outputs(weights=weights, clean_covar=clean_covar)
    else:
        out = weights
    return out


def cvx_maximise_alpha_over_tre(covar: np.ndarray,
                                alphas: np.ndarray,
                                constraints: Constraints,
                                solver: str = 'ECOS_BB',
                                verbose: bool = False
                                ) -> np.ndarray:
    """
    numpy level solution of quadratic problem:
    max alpha@w
    such that
    (w-benchmark_weights) @ Sigma @ (w-benchmark_weights).t <= tracking_err_vol_constraint
    sum(abs(w-w_0)) <= turnover_constraint
    subject to linear constraints
         1. weight_min <= w <= weight_max
         2. sum(w) = 1
         3. exposure_budget_eq[0]^t*w = exposure_budget_eq[1]
    here we assume that all assets are valid: Sigma is invertible
    """
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


def cvx_maximise_tre_utility(covar: np.ndarray,
                             constraints: Constraints,
                             alphas: Optional[np.ndarray] = None,
                             tre_weight: Optional[float] = 1.0,
                             turnover_weight: Optional[float] = 0.1,
                             solver: str = 'ECOS_BB',
                             verbose: bool = False
                             ) -> np.ndarray:
    """
    numpy level solution of quadratic problem with utility weights:
    max { alpha@w - tre_weight * (w-benchmark_weights)@Sigma@(w-benchmark_weights).t - turnover_weight*sum(abs(w-w_0))}
    subject to linear constraints
         1. weight_min <= w <= weight_max
         2. sum(w) = 1
         3. exposure_budget_eq[0]^t*w = exposure_budget_eq[1]
    here we assume that all assets are valid: Sigma is invertible
    """
    n = covar.shape[0]
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    w = cvx.Variable(n, nonneg=nonneg)
    covar = cvx.psd_wrap(covar)

    constraints1 = constraints.copy()
    # set solver
    benchmark_weights = constraints.benchmark_weights.to_numpy()

    # compute tracking error var
    constraints1.tracking_err_vol_constraint = None  # disable from constraints
    tracking_error_var = cvx.quad_form(w - benchmark_weights, covar)

    if alphas is not None:
        objective_fun = alphas.T @ (w - benchmark_weights)
        if tre_weight is not None:
            objective_fun += -1.0*tre_weight*tracking_error_var
    else:
        if tre_weight is None:
            raise ValueError(f"tre_weight must be given for tre without alphas")
        objective_fun = -1.0*tre_weight*tracking_error_var

    # add turover
    if turnover_weight is not None:
        constraints1.turnover_constraint = None  # disable from constraints
        if constraints1.weights_0 is None:
            print(f"weights_0 must be given for turnover_constraint")
        else:
            objective_fun += -1.0*turnover_weight*cvx.norm(w - constraints1.weights_0, 1)

    objective = cvx.Maximize(objective_fun)
    constraints_ = constraints1.set_cvx_constraints(w=w, covar=covar)

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
