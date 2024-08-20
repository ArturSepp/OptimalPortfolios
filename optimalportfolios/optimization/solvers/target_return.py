"""
optimise alpha with targeting return
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
import qis as qis

from optimalportfolios import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints


def rolling_maximise_alpha_with_target_return(prices: pd.DataFrame,
                                              alphas: pd.DataFrame,
                                              yields: pd.DataFrame,
                                              target_returns: pd.Series,
                                              constraints0: Constraints,
                                              time_period: qis.TimePeriod,  # when we start building portfolios
                                              returns_freq: str = 'W-WED',
                                              rebalancing_freq: str = 'QE',
                                              span: int = 52,  # 1y
                                              solver: str = 'ECOS_BB',
                                              print_inputs: bool = False
                                              ) -> pd.DataFrame:
    """
    maximise portfolio alpha subject to constraint on tracking tracking error
    """
    # compute ewma covar with fill nans in covar using zeros
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    returns_np = returns.to_numpy()
    x = returns_np - qis.compute_ewm(returns_np, span=span)
    covar_tensor_txy = qis.compute_ewm_covar_tensor(a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)
    an_factor = qis.infer_an_from_data(data=returns)

    # create rebalancing schedule: it must much idx in covar_tensor_txy using returns.index
    rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)
    alphas = alphas.reindex(index=rebalancing_schedule.index, method='ffill')
    yields = yields.reindex(index=rebalancing_schedule.index, method='ffill')
    target_returns = target_returns.reindex(index=rebalancing_schedule.index, method='ffill')
    tickers = prices.columns.to_list()
    weights = {}
    weights_0 = None
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if value and date >= time_period.start:
            pd_covar = pd.DataFrame(an_factor*covar_tensor_txy[idx], index=tickers, columns=tickers)
            # call optimiser
            if print_inputs:
                print(f"date={date}")
                print(f"pd_covar=\n{pd_covar}")
                print(f"alphas=\n{alphas.loc[date, :]}")
                print(f"yields=\n{yields.loc[date, :]}")
                print(f"target_return=\n{target_returns[date]}")

            weights_ = wrapper_maximise_alpha_with_target_return(pd_covar=pd_covar,
                                                                 alphas=alphas.loc[date, :],
                                                                 yields=yields.loc[date, :],
                                                                 target_return=target_returns[date],
                                                                 constraints0=constraints0,
                                                                 weights_0=weights_0,
                                                                 solver=solver)

            weights_0 = weights_  # update for next rebalancing
            weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers).fillna(0.0)  # align with tickers
    return weights


def wrapper_maximise_alpha_with_target_return(pd_covar: pd.DataFrame,
                                              alphas: pd.Series,
                                              yields: pd.Series,
                                              target_return: float,
                                              constraints0: Constraints,
                                              weights_0: pd.Series = None,
                                              solver: str = 'ECOS_BB'
                                              ) -> pd.Series:
    """
    create wrapper accounting for nans or zeros in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    """
    # filter out assets with zero variance or nans
    vectors = dict(alphas=alphas)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    constraints = constraints0.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
                                                         weights_0=weights_0,
                                                         asset_returns=yields,
                                                         target_return=target_return)

    weights = cvx_maximise_alpha_with_target_return(covar=clean_covar.to_numpy(),
                                                    alphas=good_vectors['alphas'].to_numpy(),
                                                    constraints=constraints,
                                                    solver=solver)

    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)  # align with tickers

    return weights


def cvx_maximise_alpha_with_target_return(covar: np.ndarray,
                                          alphas: np.ndarray,
                                          constraints: Constraints,
                                          verbose: bool = False,
                                          solver: str = 'ECOS_BB'
                                          ) -> np.ndarray:
    """
    numpy level one step solution of problem
    max alpha @ w
    such that
    yields @ w = target return
    sum(w) = 1 # exposure constraint
    w >= 0  # long only constraint
    w.T @ Sigma @ w <= vol_constraint
    """
    # set up problem
    n = covar.shape[0]
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    w = cvx.Variable(n, nonneg=nonneg)
    # covar = cvx.psd_wrap(covar)

    # set solver
    objective_fun = alphas.T @ w
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
