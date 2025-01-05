"""
implementation of maximum diversification objective
"""
# packages
import numpy as np
import pandas as pd
import qis as qis
from scipy.optimize import minimize
from typing import List, Dict

# optimalportfolios
from optimalportfolios.utils.portfolio_funcs import calculate_diversification_ratio
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.utils.covar_matrix import CovarEstimator


def rolling_maximise_diversification(prices: pd.DataFrame,
                                     constraints0: Constraints,
                                     time_period: qis.TimePeriod,  # when we start building portfolios
                                     pd_covars: Dict[pd.Timestamp, pd.DataFrame] = None,  # can be precomputed
                                     covar_estimator: CovarEstimator = CovarEstimator()  # default EWMA estimator
                                     ) -> pd.DataFrame:
    """
    compute rolling maximum diversification portfolios
    pd_covars: Dict[timestamp, covar matrix] can be precomputed
    portolio is rebalances at pd_covars.keys()
    """
    if pd_covars is None:  # use default ewm covar with covar_estimator
        pd_covars = covar_estimator.fit_rolling_covars(prices=prices, time_period=time_period)

    weights = {}
    weights_0 = None
    for date, pd_covar in pd_covars.items():
        weights_ = wrapper_maximise_diversification(pd_covar=pd_covar,
                                                    constraints0=constraints0,
                                                    weights_0=weights_0)
        weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list())
    return weights


def wrapper_maximise_diversification(pd_covar: pd.DataFrame,
                                     constraints0: Constraints,
                                     weights_0: pd.Series = None
                                     ) -> pd.Series:
    """
    create wrapper accounting for nans or zeros in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    """
    # filter out assets with zero variance or nans
    vectors = None
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    constraints = constraints0.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
                                                         weights_0=weights_0)

    weights = opt_maximise_diversification(covar=clean_covar.to_numpy(),
                                           constraints=constraints)
    weights = pd.Series(weights, index=clean_covar.columns)
    weights = weights.reindex(index=pd_covar.columns).fillna(0.0)  # align with tickers
    return weights


def opt_maximise_diversification(covar: np.ndarray,
                                 constraints: Constraints,
                                 verbose: bool = False
                                 ) -> np.ndarray:
    n = covar.shape[0]
    x0 = np.ones(n) / n

    constraints_ = constraints.set_scipy_constraints(covar=covar)
    res = minimize(max_diversification_objective, x0, args=[covar], method='SLSQP',
                   constraints=constraints_,
                   options={'disp': verbose, 'ftol': 1e-18, 'maxiter': 200})
    optimal_weights = res.x
    if optimal_weights is None:
        # raise ValueError(f"not solved")
        print(f"not solved")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0
            print(f"using weights_0")
        else:
            optimal_weights = np.zeros(n)
            print(f"using zeroweights")

    else:
        if constraints.is_long_only:
            optimal_weights = np.where(optimal_weights > 0.0, optimal_weights, 0.0)

    return optimal_weights


def max_diversification_objective(w: np.ndarray, pars: List[np.ndarray]) -> float:
    covar = pars[0]
    return -calculate_diversification_ratio(w=w, covar=covar)
