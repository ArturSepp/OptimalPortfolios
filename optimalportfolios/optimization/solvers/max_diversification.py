"""
implementation of maximum diversification objective
"""
# packages
import warnings
import numpy as np
import pandas as pd
import qis as qis
from scipy.optimize import minimize
from typing import List, Dict

# optimalportfolios
from optimalportfolios.utils.portfolio_funcs import calculate_diversification_ratio
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator


def rolling_maximise_diversification(prices: pd.DataFrame,
                                     constraints: Constraints,
                                     time_period: qis.TimePeriod,  # when we start building portfolios
                                     covar_dict: Dict[pd.Timestamp, pd.DataFrame] = None,  # can be precomputed
                                     covar_estimator: CovarEstimator = CovarEstimator()  # default EWMA estimator
                                     ) -> pd.DataFrame:
    """
    compute rolling maximum diversification portfolios
    covar_dict: Dict[timestamp, covar matrix] can be precomputed
    portolio is rebalances at covar_dict.keys()
    """

    if covar_dict is None:  # use default ewm covar with covar_estimator
        covars = covar_estimator.fit_rolling_covars(prices=prices, time_period=time_period)
        covar_dict = covars.y_covars

    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():
        weights_ = wrapper_maximise_diversification(pd_covar=pd_covar,
                                                    constraints=constraints,
                                                    weights_0=weights_0)
        weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list())
    return weights


def wrapper_maximise_diversification(pd_covar: pd.DataFrame,
                                     constraints: Constraints,
                                     weights_0: pd.Series = None
                                     ) -> pd.Series:
    """
    create wrapper accounting for nans or zeros in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    """
    # filter out assets with zero variance or nans
    vectors = None
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    constraints1 = constraints.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
                                                         weights_0=weights_0)

    weights = opt_maximise_diversification(covar=clean_covar.to_numpy(),
                                           constraints=constraints1)
    weights = pd.Series(weights, index=clean_covar.columns)
    weights = weights.reindex(index=pd_covar.columns).fillna(0.0)  # align with tickers
    return weights


def opt_maximise_diversification(covar: np.ndarray,
                                 constraints: Constraints,
                                 verbose: bool = False,
                                 ftol: float = 1e-8,
                                 maxiter: int = 500
                                 ) -> np.ndarray:
    n = covar.shape[0]
    x0 = np.ones(n) / n

    constraints_, bounds = constraints.set_scipy_constraints(covar=covar)
    res = minimize(max_diversification_objective, x0, args=[covar], method='SLSQP',
                   constraints=constraints_,
                   bounds=bounds,
                   options={'disp': verbose, 'ftol': ftol, 'maxiter': maxiter})
    # print(res)
    optimal_weights = res.x

    if res.success == False or optimal_weights is None:
        # raise ValueError(f"not solved")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0
            mes = f"using weights_0"
        else:
            optimal_weights = np.zeros(n)
            mes = f"using zeroweights"
        warnings.warn(f"opt_maximise_diversification(): problem is not solved, {mes}")

    else:
        if constraints.is_long_only:
            optimal_weights = np.where(optimal_weights > 0.0, optimal_weights, 0.0)

    return optimal_weights


def max_diversification_objective(w: np.ndarray, pars: List[np.ndarray]) -> float:
    covar = pars[0]
    return -calculate_diversification_ratio(w=w, covar=covar)
