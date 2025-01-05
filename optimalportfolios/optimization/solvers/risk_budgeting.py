"""
implementation of risk budgeting optimisation
using ConstrainedRiskBudgeting from pyrb package
"""
from __future__ import division

import numpy as np
import pandas as pd
import qis as qis
from scipy.optimize import minimize
from typing import Dict
from enum import Enum

from optimalportfolios.utils.portfolio_funcs import (calculate_portfolio_var, calculate_risk_contribution)
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.utils.covar_matrix import CovarEstimator

from pyrb import ConstrainedRiskBudgeting


def rolling_risk_budgeting(prices: pd.DataFrame,
                           constraints0: Constraints,
                           time_period: qis.TimePeriod,  # when we start building portfolios
                           pd_covars: Dict[pd.Timestamp, pd.DataFrame] = None,  # can be precomputed
                           covar_estimator: CovarEstimator = CovarEstimator(),  # default covar estimator is ewma
                           risk_budget: pd.Series = None,
                           rebalancing_indicators: pd.DataFrame = None,  # whe assets can be rebalanced
                           apply_total_to_good_ratio: bool = True
                           ) -> pd.DataFrame:
    """
    compute equal risk contribution
    risk_budget sets the risk budgets
    pd_covars: Dict[timestamp, covar matrix] can be precomputed
    portolio is rebalances at pd_covars.keys()
    """
    if pd_covars is None:  # use default ewm covar with covar_estimator
        pd_covars = covar_estimator.fit_rolling_covars(prices=prices, time_period=time_period)

    if rebalancing_indicators is not None:  # need to reindex at pd_covars index
        rebalancing_dates = list(pd_covars.keys())
        rebalancing_indicators = rebalancing_indicators.reindex(index=rebalancing_dates).fillna(0.0)  # by default no rebalancing

    weights = {}
    weights_0 = None
    for date, pd_covar in pd_covars.items():
        if rebalancing_indicators is not None and weights_0 is not None:
            rebalancing_indicators_t = rebalancing_indicators.loc[date, :]
        else:
            rebalancing_indicators_t = None
        weights_ = wrapper_risk_budgeting(pd_covar=pd_covar,
                                          constraints0=constraints0,
                                          weights_0=weights_0,
                                          risk_budget=risk_budget,
                                          rebalancing_indicators=rebalancing_indicators_t,
                                          apply_total_to_good_ratio=apply_total_to_good_ratio)
        weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list())
    return weights


def wrapper_risk_budgeting(pd_covar: pd.DataFrame,
                           constraints0: Constraints,
                           weights_0: pd.Series = None,
                           risk_budget: pd.Series = None,
                           rebalancing_indicators: pd.Series = None,
                           apply_total_to_good_ratio: bool = True,
                           verbouse: bool = False
                           ) -> pd.Series:
    """
    create wrapper accounting for nans or zeros in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    for assets with rebalancing_indicators == False, the min_weight and max_weight is fixed at weights_0
    note that:
     i) assets with rebalancing_indicators == False will impact the portfolio volatility (their weights are set to weights_0)
    ii) assets with inclusion_indicators == False will not impact the portfolio volatility (their weights will be set to zero)
    """
    # assets with zero risk budgets are excluded from optimisation
    if risk_budget is not None:  # exclude assets with risk_badget = 0
        inclusion_indicators = pd.Series(np.where(risk_budget.fillna(0.0) > 0.0, 1.0, 0.0), index=risk_budget.index)
    else:
        inclusion_indicators = pd.Series(1.0, index=pd_covar.columns)

    if rebalancing_indicators is not None and weights_0 is not None:
        # make sure indices are alined, default is rebalanced
        rebalancing_indicators = rebalancing_indicators.reindex(index=inclusion_indicators.index).fillna(1.0)
        weights_0 = weights_0.reindex(index=inclusion_indicators.index).fillna(0.0)
        fixed_weights = weights_0.where(np.isclose(rebalancing_indicators, 0.0), other=0.0)
        inclusion_indicators = inclusion_indicators.where(np.isclose(rebalancing_indicators, 1.0), other=0.0)
    else:
        fixed_weights = None

    vectors = dict(min_weights=constraints0.min_weights, max_weights=constraints0.max_weights, risk_budget=risk_budget)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors,
                                                                  inclusion_indicators=inclusion_indicators)

    if len(clean_covar.columns) == 0:
        print(f"in wrapper_equal_risk_contribution no valid covar")
        return pd.Series(0.0, index=pd_covar.index)

    if apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = 1.0

    if risk_budget is not None:
        risk_budget = risk_budget.loc[clean_covar.columns].fillna(0.0)
        risk_budget *= total_to_good_ratio
        risk_budget_np = risk_budget.to_numpy()
    else:
        risk_budget_np = None

    constraints = constraints0.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=total_to_good_ratio,
                                                         weights_0=weights_0,
                                                         rebalancing_indicators=None,  # don't need to account here
                                                         apply_total_to_good_ratio=apply_total_to_good_ratio)

    weights0 = opt_risk_budgeting(covar=clean_covar.to_numpy(),
                                  constraints=constraints,
                                  risk_budget=risk_budget_np)
    weights = pd.Series(weights0, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)  # align with tickers

    if fixed_weights is not None:
        left_allocation = 1.0 - np.nansum(fixed_weights)
        sum_solved = np.nansum(weights)
        if sum_solved > 0.0:
            weights = weights * left_allocation / np.nansum(weights)
        weights = weights.where(np.isclose(inclusion_indicators, 1.0), other=fixed_weights)

    if verbouse:
        asset_rc = calculate_risk_contribution(weights0, clean_covar.to_numpy())
        asset_rc_ratio = asset_rc / np.nansum(asset_rc)
        df = pd.concat([pd.Series(weights0, index=clean_covar.columns, name='weights'),
                        pd.Series(asset_rc, index=clean_covar.columns, name='asset_rc'),
                        risk_budget.rename('risk_budget'),
                        pd.Series(asset_rc_ratio, index=clean_covar.columns, name='asset_rc_ratio')
                        ], axis=1)
        print(df)

    return weights


def opt_risk_budgeting(covar: np.ndarray,
                       constraints: Constraints,
                       risk_budget: np.ndarray = None,
                       verbose: bool = False
                       ) -> np.ndarray:
    """
    optimiser of constrained risk-budgeting using ConstrainedRiskBudgeting from pyrb package
    """
    n = covar.shape[0]
    # default is equal budget
    if risk_budget is None:
        risk_budget = np.ones(n) / n

    bounds, c_rows, c_lhs = constraints.set_pyrb_constraints(covar=covar)

    this = ConstrainedRiskBudgeting(covar, budgets=risk_budget, bounds=bounds, C=c_rows, d=c_lhs) # , solver='admm_qp'
    this.solve()
    optimal_weights = this.x

    if verbose:
        if c_rows is not None:
            qqq = c_rows @ optimal_weights
            slack = qqq - c_lhs
            print(f"slack={slack}")

    if optimal_weights is None or np.any(np.isnan(optimal_weights)):
        # raise ValueError(f"not solved")
        print(f"equal risk contribution not solved")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0
            print(f"using weights_0")
        else:
            optimal_weights = np.zeros(n)
            print(f"using zeroweights")

    return optimal_weights


def opt_risk_budgeting_scipy(covar: np.ndarray,
                             constraints: Constraints,
                             risk_budget: np.ndarray = None
                             ) -> np.ndarray:
    """
    risk budgeting using scipy solver
    not recommended
    """
    # set x0
    n = covar.shape[0]
    if constraints.weights_0 is not None:
        x0 = constraints.weights_0.to_numpy()
    elif risk_budget is not None:
        x0 = risk_budget
    else:
        x0 = np.ones(n) / n

    # default is equal budget
    if risk_budget is None:
        risk_budget = np.ones(n) / n

    if constraints.min_weights is not None and constraints.max_weights is not None:
        bounds = [(x, y) for x, y in zip(constraints.min_weights.to_numpy(), constraints.max_weights.to_numpy())]
    else:
        bounds = None

    constraints_ = constraints.set_scipy_constraints(covar=covar)

    # set zero risk budget to nan to exlude from computations
    risk_budget = np.where(np.isclose(risk_budget, 0.0), np.nan, risk_budget)
    options = {'ftol': 1e-8, 'maxiter': 200}

    res = minimize(risk_budget_objective, x0, args=[covar, risk_budget], method='SLSQP',
                  constraints=constraints_, bounds=bounds, options=options)

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


def risk_budget_objective(x, pars):
    covar, budget = pars[0], pars[1]
    asset_rc = calculate_risk_contribution(x, covar)
    sig_p = np.sqrt(calculate_portfolio_var(x, covar))
    if budget is not None:
        risk_target = np.where(np.isnan(budget), asset_rc, np.multiply(sig_p, budget))  # budget can be nan f
    else:
        risk_target = np.multiply(sig_p, np.ones_like(asset_rc) / asset_rc.shape[0])
    # sse = np.nansum(np.square(asset_rc - risk_target))
    sse = np.nanmean(np.square(asset_rc - risk_target))
    return sse


class UnitTests(Enum):
    RISK_PARITY = 1
    PYRB = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.RISK_PARITY:
        # risk_budget = np.array([0.50, 0.0, 0.5])
        # risk_budget = np.array([0.333, 0.333, 0.334])
        risk_budget = np.array([0.45, 0.45, 0.1])
        covar = np.array([[0.2 ** 2, 0.5*0.15*0.2, 0.0],
                          [0.5*0.15*0.2, 0.15 ** 2, 0.0],
                          [0.0, 0.0, 0.1**2]])

        covar = np.array([[0.2 ** 2, 0.5*0.15*0.2, -0.01],
                          [0.5*0.15*0.2, 0.15 ** 2, -0.005],
                          [-0.01, -0.005, 0.1**2]])

        print('covar')
        print(covar)
        vol = np.sqrt(np.diag(covar))
        norm = np.outer(1.0 / vol, 1.0 / vol)
        print('corr')
        print(covar*norm)

        w_rb = opt_risk_budgeting_scipy(covar=covar,
                                        constraints=Constraints(is_long_only=True),
                                        risk_budget=risk_budget)

        print(f"risk_budget={risk_budget}")
        print(f"weights={w_rb}")
        asset_rc = calculate_risk_contribution(w_rb, covar)
        print(f"asset_rc={asset_rc/np.nansum(asset_rc)}")

        bounds = np.array([(0.0, 0.0, 0.4), (0.4, 0.4, 0.4)]).T
        print(f"bounds={bounds}")
        from pyrb import ConstrainedRiskBudgeting
        this = ConstrainedRiskBudgeting(covar, budgets=risk_budget, bounds=bounds)
        this.solve()
        # print(this)
        weights = this.x
        print(f"weights={weights}, sum={np.sum(weights)}")
        asset_rc = calculate_risk_contribution(weights, covar)
        print(f"asset_rc_new={asset_rc/np.nansum(asset_rc)}")

    elif unit_test == UnitTests.PYRB:
        pass


if __name__ == '__main__':

    unit_test = UnitTests.RISK_PARITY

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)