"""
Implementation of carra utility
"""

import numpy as np
import pandas as pd
import qis as qis
from scipy.optimize import minimize
from typing import List
from enum import Enum

from optimalportfolios.utils.gaussian_mixture import fit_gaussian_mixture
from optimalportfolios.utils.portfolio_funcs import (calculate_portfolio_var, calculate_risk_contribution)
from optimalportfolios.optimization.constraints import (Constraints, total_weight_constraint, long_only_constraint)


def rolling_maximize_cara_mixture(prices: pd.DataFrame,
                                  constraints0: Constraints,
                                  time_period: qis.TimePeriod,  # when we start building portfolios
                                  rebalancing_freq: str = 'QE',
                                  roll_window: int = 52*6,  # number of returns in mixture estimation, default is 6y of weekly returns
                                  returns_freq: str = 'W-WED',  # frequency for returns computing mixure distr
                                  carra: float = 0.5,  # carra parameters
                                  n_components: int = 3
                                  ) -> pd.DataFrame:
    """
    solve solvers mixture Carra portfolios
    estimation is applied for the whole period of prices
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    # generate rebalancing dates on the returns index
    rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)

    _, scaler = qis.get_period_days(freq=returns_freq)

    tickers = prices.columns.to_list()
    weights = {}
    weights_0 = None
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if idx >= roll_window-1 and value:
            period = qis.TimePeriod(rebalancing_schedule.index[idx - roll_window+1], date)
            # drop assets with
            rets_ = period.locate(returns).dropna(axis=1, how='any')
            params = fit_gaussian_mixture(x=rets_.to_numpy(), n_components=n_components, scaler=scaler)
            constraints = constraints0.update_with_valid_tickers(valid_tickers=rets_.columns.to_list(),
                                                                 total_to_good_ratio=len(tickers)/len(rets_.columns),
                                                                 weights_0=weights_0)
            weights_ = wrapper_maximize_cara_mixture(means=params.means,
                                                     covars=params.covars,
                                                     probs=params.probs,
                                                     constraints0=constraints,
                                                     tickers=rets_.columns.to_list(),
                                                     carra=carra)
            weights_0 = weights_  # update for next rebalancing
            weights[date] = weights_.reindex(index=tickers).fillna(0.0)
    weights = pd.DataFrame.from_dict(weights, orient='index', columns=prices.columns)
    if time_period is not None:
        weights = time_period.locate(weights)

    return weights


def wrapper_maximize_cara_mixture(means: List[np.ndarray],
                                  covars: List[np.ndarray],
                                  probs: np.ndarray,
                                  constraints0: Constraints,
                                  tickers: List[str],
                                  carra: float = 0.5
                                  ) -> pd.Series:
    """
    wrapper assumes means and covars are valid
    """
    weights = opt_maximize_cara_mixture(means=means,
                                        covars=covars,
                                        probs=probs,
                                        constraints=constraints0,
                                        carra=carra)
    weights = pd.Series(weights, index=tickers)
    return weights


def opt_maximize_cara_mixture(means: List[np.ndarray],
                              covars: List[np.ndarray],
                              probs: np.ndarray,
                              constraints: Constraints,
                              carra: float = 0.5,
                              verbose: bool = False
                              ) -> np.ndarray:

    # set up problem
    n = covars[0].shape[0]
    if Constraints.weights_0 is not None:
        x0 = Constraints.weights_0.to_numpy()
    else:
        x0 = np.ones(n) / n

    constraints_ = constraints.set_scipy_constraints()  # covar is not used for this method
    res = minimize(carra_objective_mixture, x0, args=[means, covars, probs, carra], method='SLSQP',
                   constraints=constraints_,
                   options={'disp': verbose, 'ftol': 1e-12})
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

    return optimal_weights


def opt_maximize_cara(means: np.ndarray,
                      covar: np.ndarray,
                      carra: float = 0.5,
                      min_weights: np.ndarray = None,
                      max_weights: np.ndarray = None,
                      disp: bool = False,
                      is_exp: bool = False,
                      is_print_log: bool = False
                      ) -> np.ndarray:
    n = covar.shape[0]
    x0 = np.ones(n) / n
    cons = [{'type': 'ineq', 'fun': long_only_constraint},
            {'type': 'eq', 'fun': total_weight_constraint}]
    if min_weights is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: x - min_weights})
    if max_weights is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: max_weights - x})

    if is_exp:
        func = carra_objective_exp
    else:
        func = carra_objective
    res = minimize(func, x0, args=[means, covar, carra], method='SLSQP', constraints=cons,
                   options={'disp': disp, 'ftol': 1e-16})
    w_rb = res.x

    if is_print_log:
        print(f'return_p = {w_rb@means}, '
              f'sigma_p = {np.sqrt(calculate_portfolio_var(w_rb, covar))}, weights: {w_rb}, '
              f'risk contrib.s: {calculate_risk_contribution(w_rb, covar).T} '
              f'sum of weights: {sum(w_rb)}')
    return w_rb


def carra_objective(w: np.ndarray, pars: List[np.ndarray]) -> float:
    means, covar, carra = pars[0], pars[1], pars[2]
    v = means.T @ w - 0.5*carra*w.T @ covar @ w
    return -v


def carra_objective_exp(w: np.ndarray, pars: List[np.ndarray]) -> float:
    means, covar, carra = pars[0], pars[1], pars[2]
    v = np.exp(-carra*means.T @ w + 0.5*carra*carra*w.T @ covar @ w)
    return v


def carra_objective_mixture(w: np.ndarray, pars: List[np.ndarray]) -> float:
    means, covars, probs, carra = pars[0], pars[1], pars[2], pars[3]
    v = 0.0
    for idx, prob in enumerate(probs):
        v = v + prob*np.exp(-carra*means[idx].T @ w + 0.5*carra*carra*w.T @ covars[idx] @ w)
    return v


class UnitTests(Enum):
    CARA = 1
    CARA_MIX = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.CARA:
        means = np.array([0.3, 0.1])
        covar = np.array([[0.2 ** 2, 0.01],
                          [0.01, 0.1 ** 2]])
        w_rb = opt_maximize_cara(means=means, covar=covar, carra=10, is_exp=False, disp=True)
        w_rb = opt_maximize_cara(means=means, covar=covar, carra=10, is_exp=True, disp=True)

    elif unit_test == UnitTests.CARA_MIX:
        means = [np.array([0.05, -0.1]), np.array([0.05, 2.0])]
        covars = [np.array([[0.2 ** 2, 0.01],
                          [0.01, 0.2 ** 2]]),
                 np.array([[0.2 ** 2, 0.01],
                           [0.01, 0.2 ** 2]])
                 ]
        probs = np.array([0.95, 0.05])
        optimal_weights = opt_maximize_cara_mixture(means=means, covars=covars, probs=probs,
                                                    constraints=Constraints(),
                                                    carra=20.0, verbose=True)
        print(optimal_weights)


if __name__ == '__main__':

    unit_test = UnitTests.CARA_MIX

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

