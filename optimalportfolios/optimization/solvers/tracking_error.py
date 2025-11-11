"""Optimize alpha over tracking error."""
import numpy as np
import pandas as pd
import cvxpy as cvx
import qis as qis
from typing import Optional, Union, Dict, NamedTuple

from optimalportfolios import filter_covar_and_vectors_for_nans, compute_portfolio_risk_contribution_outputs
from optimalportfolios.optimization.constraints import Constraints, ConstraintEnforcementType
from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator


class OptimiserAlphaOverTreConfig(NamedTuple):
    """Configuration for alpha over tracking error optimization."""
    constraint_enforcement_type: ConstraintEnforcementType = ConstraintEnforcementType.FORCED_CONSTRAINTS
    solver: str = 'ECOS_BB'  # CVXPY solver choice
    # this specify weights of utility function for ConstraintEnforcementType.UTILITY_CONSTRAINTS
    tre_utility_weight: Optional[float] = 1.0  # penalty weight for tracking error in utility
    turnover_utility_weight: Optional[float] = 0.40  # penalty weight for turnover in utility
    apply_total_to_good_ratio: bool = True  # adjust constraints for non-investable assets



def rolling_maximise_alpha_over_tre(prices: pd.DataFrame,
                                    alphas: Optional[pd.DataFrame],
                                    constraints: Constraints,
                                    benchmark_weights: Union[pd.Series, pd.DataFrame],
                                    time_period: qis.TimePeriod,  # when we start building portfolios
                                    covar_estimator: CovarEstimator = CovarEstimator(),  # default covar estimator
                                    covar_dict: Dict[pd.Timestamp, pd.DataFrame] = None,
                                    rebalancing_indicators: pd.DataFrame = None,
                                    optimiser_alpha_over_tre_config: OptimiserAlphaOverTreConfig = OptimiserAlphaOverTreConfig(),
                                    ) -> pd.DataFrame:
    """Maximize portfolio alpha subject to tracking error constraint over rolling periods."""
    # estimate covar at rebalancing schedule
    if covar_dict is None:  # use default ewm covar with covar_estimator
        covar_dict = covar_estimator.fit_rolling_covars(prices=prices, time_period=time_period).y_covars
    rebalancing_dates = list(covar_dict.keys())

    # align alphas with rebalancing dates
    if alphas is not None:
        alphas = alphas.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    weights = {}
    # extend benchmark weights to all rebalancing dates
    if isinstance(benchmark_weights, pd.DataFrame):
        benchmark_weights = benchmark_weights.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)
    else:  # for series do transformation
        benchmark_weights = benchmark_weights.to_frame(
            name=rebalancing_dates[0]).T.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    # align rebalancing indicators with dates
    if rebalancing_indicators is not None:  # need to reindex at covar_dict index: by default no rebalancing
        rebalancing_indicators = rebalancing_indicators.reindex(index=rebalancing_dates).fillna(0.0)

    weights_0 = None  # it will relax turnover constraint for the first rebalancing
    # loop through rebalancing dates and optimize
    for date, pd_covar in covar_dict.items():
        # get rebalancing indicator for this date
        if rebalancing_indicators is not None:
            rebalancing_indicators_t = rebalancing_indicators.loc[date, :]
        else:
            rebalancing_indicators_t = None
        # get alphas for this date
        alphas_t = alphas.loc[date, :] if alphas is not None else None
        # optimize portfolio weights
        weights_ = wrapper_maximise_alpha_over_tre(pd_covar=pd_covar,
                                                   alphas=alphas_t,
                                                   benchmark_weights=benchmark_weights.loc[date, :],
                                                   constraints=constraints,
                                                   rebalancing_indicators=rebalancing_indicators_t,
                                                   weights_0=weights_0,
                                                   optimiser_alpha_over_tre_config=optimiser_alpha_over_tre_config)
        # reset weights_0 if all zeros, otherwise update
        if np.all(np.equal(weights_, 0.0)):
            weights_0 = None
        else:
            weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_

    # convert to dataframe and align with price columns
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list())
    return weights


def wrapper_maximise_alpha_over_tre(pd_covar: pd.DataFrame,
                                    alphas: Optional[pd.Series],
                                    benchmark_weights: pd.Series,
                                    constraints: Constraints,
                                    weights_0: pd.Series = None,
                                    rebalancing_indicators: pd.Series = None,
                                    detailed_output: bool = False,
                                    optimiser_alpha_over_tre_config: OptimiserAlphaOverTreConfig = OptimiserAlphaOverTreConfig(),
                                    verbose: bool = False
                                    ) -> Union[pd.Series, pd.DataFrame]:
    """Wrapper for alpha optimization handling NaNs and zero-variance assets."""
    # filter out assets with zero variance or nans
    if alphas is None:
        vectors = None
    else:
        vectors = dict(alphas=alphas)
    # clean covariance matrix and filter vectors
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    # compute scaling ratio for constraints adjustment
    if optimiser_alpha_over_tre_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    # update constraints with valid tickers only
    constraints1 = constraints.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=total_to_good_ratio,
                                                         weights_0=weights_0,
                                                         benchmark_weights=benchmark_weights,
                                                         rebalancing_indicators=rebalancing_indicators)

    # convert alphas to numpy array
    if alphas is not None:
        alphas_np = good_vectors['alphas'].to_numpy()
    else:
        alphas_np = None

    # choose optimization method based on constraint enforcement type
    if optimiser_alpha_over_tre_config.constraint_enforcement_type == ConstraintEnforcementType.UTILITY_CONSTRAINTS:
        # use utility-based optimization with penalty weights
        weights = cvx_maximise_tre_utility(covar=clean_covar.to_numpy(),
                                           alphas=alphas_np,
                                           constraints=constraints1,
                                           solver=optimiser_alpha_over_tre_config.solver,
                                           tre_utility_weight=optimiser_alpha_over_tre_config.tre_utility_weight,
                                           turnover_utility_weight=optimiser_alpha_over_tre_config.turnover_utility_weight,
                                           verbose=verbose)
    else:
        # use hard constraints optimization
        weights = cvx_maximise_alpha_over_tre(covar=clean_covar.to_numpy(),
                                              alphas=alphas_np,
                                              constraints=constraints1,
                                              solver=optimiser_alpha_over_tre_config.solver,
                                              verbose=verbose)
    # clean up infinite values
    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)  # align with tickers

    # return detailed output or just weights
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
    """Maximize alpha subject to tracking error and linear constraints using CVXPY."""
    n = covar.shape[0]
    # set non-negativity based on long-only constraint
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    # create optimization variable
    w = cvx.Variable(n, nonneg=nonneg)
    covar = cvx.psd_wrap(covar)  # wrap covariance for PSD constraint

    # set objective: maximize active alpha
    benchmark_weights = constraints.benchmark_weights.to_numpy()
    objective_fun = alphas.T @ (w - benchmark_weights)
    objective = cvx.Maximize(objective_fun)

    # build all constraints
    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar)

    # solve optimization problem
    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    # extract solution with fallback to previous weights or zeros
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
                             tre_utility_weight: Optional[float] = 1.0,
                             turnover_utility_weight: Optional[float] = 0.40,
                             solver: str = 'ECOS_BB',
                             verbose: bool = False
                             ) -> np.ndarray:
    """Maximize utility with tracking error and turnover penalties using CVXPY."""
    n = covar.shape[0]
    # set non-negativity based on long-only constraint
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    # create optimization variable
    w = cvx.Variable(n, nonneg=nonneg)
    covar = cvx.psd_wrap(covar)  # wrap covariance for PSD constraint

    constraints1 = constraints.copy()

    # set objective with utility penalties for TRE and turnover
    objective_fun, constraints_ = constraints1.set_cvx_utility_objective_constraints(
        w=w,
        alphas=alphas,
        covar=covar,
        tre_utility_weight=tre_utility_weight,
        turnover_utility_weight=turnover_utility_weight)

    # solve optimization problem
    problem = cvx.Problem(cvx.Maximize(objective_fun), constraints_)
    problem.solve(verbose=verbose, solver=solver)

    # extract solution with fallback to previous weights or zeros
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
