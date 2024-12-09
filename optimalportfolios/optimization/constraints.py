"""
implement constraints as dataclass object
to support setting various constrains
"""
from __future__ import annotations, division

import pandas as pd
import numpy as np
import cvxpy as cvx
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class GroupLowerUpperConstraints:
    """
    add constraints that each asset group is group_min_allocation <= sum group weights <= group_max_allocation
    """
    group_loadings: pd.DataFrame  # columns=instruments, index=groups, data=1 if instrument in indexed group else 0
    group_min_allocation: pd.Series  # index=groups, data=group min allocation 
    group_max_allocation: pd.Series  # index=groups, data=group max allocation 

    def update(self, valid_tickers: List[str]) -> GroupLowerUpperConstraints:
        new_self = GroupLowerUpperConstraints(group_loadings=self.group_loadings.loc[valid_tickers, :],
                                              group_min_allocation=self.group_min_allocation,
                                              group_max_allocation=self.group_max_allocation)
        return new_self

    def print(self):
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"min_max:\n{self.group_loadings}")


@dataclass
class Constraints:
    is_long_only: bool = True  # for positive allocation weights
    min_weights: pd.Series = None  # instrument min weights  
    max_weights: pd.Series = None  # instrument max weights
    max_exposure: float = 1.0  # for long short portfolios: for long_portfolios = 1
    min_exposure: float = 1.0  # for long short portfolios: for long_portfolios = 1
    benchmark_weights: pd.Series = None  # for minimisation of tracking error 
    tracking_err_vol_constraint: float = None  # annualised sqrt tracking error
    weights_0: pd.Series = None  # for turnover constraints
    turnover_constraint: float = None  # for turnover constraints
    target_return: float = None  # for optimisation with target return
    asset_returns: pd.Series = None  # for optimisation with target return
    max_target_portfolio_vol_an: float = None  # for optimisation with maximum portfolio volatility target
    min_target_portfolio_vol_an: float = None  # for optimisation with maximum portfolio volatility target
    group_lower_upper_constraints: GroupLowerUpperConstraints = None  # for group allocations constraints
    apply_total_to_good_ratio_for_constraints: bool = True  # for constraint rescale

    def copy(self) -> Constraints:
        this = asdict(self).copy()
        if self.group_lower_upper_constraints is not None:
            gluc = self.group_lower_upper_constraints
            this['group_lower_upper_constraints'] = GroupLowerUpperConstraints(group_loadings=gluc.group_loadings,
                                                                               group_min_allocation=gluc.group_min_allocation,
                                                                               group_max_allocation=gluc.group_max_allocation)
        return Constraints(**this)

    def update(self, valid_tickers: List[str], **kwargs) -> Constraints:
        self_dict = asdict(self)
        self_dict.update(kwargs)
        if self.group_lower_upper_constraints is not None:  # asdict will make is dictionary, need to create object
            group_lower_upper_constraints = self.group_lower_upper_constraints.update(valid_tickers=valid_tickers)
            self_dict['group_lower_upper_constraints'] = group_lower_upper_constraints
        return Constraints(**self_dict)

    def update_with_valid_tickers(self,
                                  valid_tickers: List[str],
                                  total_to_good_ratio: float = 1.0,
                                  weights_0: pd.Series = None,
                                  asset_returns: pd.Series = None,
                                  benchmark_weights: pd.Series = None,
                                  target_return: float = None
                                  ) -> Constraints:
        this = self.copy()
        if this.min_weights is not None:
            this.min_weights = this.min_weights[valid_tickers].fillna(0.0)
        if this.max_weights is not None:
            if self.apply_total_to_good_ratio_for_constraints:
                this.max_weights = total_to_good_ratio*this.max_weights[valid_tickers].fillna(0.0)
            else:
                this.max_weights = this.max_weights[valid_tickers].fillna(0.0)
        if this.group_lower_upper_constraints is not None:
            this.group_lower_upper_constraints = this.group_lower_upper_constraints.update(valid_tickers=valid_tickers)
        if this.turnover_constraint is not None:
            if self.apply_total_to_good_ratio_for_constraints:
                this.turnover_constraint *= total_to_good_ratio
        if weights_0 is not None:
            this.weights_0 = weights_0.reindex(index=valid_tickers).fillna(0.0)
        if asset_returns is not None:
            this.asset_returns = asset_returns.reindex(index=valid_tickers).fillna(0.0)
        if benchmark_weights is not None:
            benchmark_weights_ = benchmark_weights.reindex(index=valid_tickers).fillna(0.0)
            this.benchmark_weights = benchmark_weights_ / np.nansum(benchmark_weights_)
        if target_return is not None:
            this.target_return = target_return
        return this

    def set_cvx_constraints(self,
                            w: cvx.Variable,  # problem variable
                            covar: np.ndarray = None,
                            exposure_scaler: cvx.Variable = None  # can be used for max sharpe contraints
                            ) -> List:
        """
        constraints for cvx solver
        """
        constraints = []

        if self.is_long_only:
            constraints += [w >= 0]

        if exposure_scaler is None:
            if self.max_exposure == self.min_exposure:
                constraints += [cvx.sum(w) == self.max_exposure]
            else:
                constraints += [cvx.sum(w) <= self.max_exposure]
                constraints += [cvx.sum(w) >= self.min_exposure]
        else:  # for max sharpe
            if self.max_exposure == self.min_exposure:
                constraints += [cvx.sum(w) == exposure_scaler]
            else:
                constraints += [cvx.sum(w) >= exposure_scaler]  # scaling

        # min weights
        if self.min_weights is not None:
            if isinstance(self.min_weights, pd.Series):
                min_weights = self.min_weights.to_numpy()
            else:
                min_weights = self.min_weights
            if exposure_scaler is None:
                constraints += [w >= min_weights]
            else:
                constraints += [w >= exposure_scaler * min_weights]

        # max weight
        if self.max_weights is not None:
            if isinstance(self.max_weights, pd.Series):
                max_weights = self.max_weights.to_numpy()
            else:
                max_weights = self.max_weights
            if exposure_scaler is None:
                constraints += [w <= max_weights]
            else:
                constraints += [w <= exposure_scaler*max_weights]

        # target_return
        if self.target_return is not None:
            if self.asset_returns is None:
                raise ValueError(f"asset_returns must be given")
            constraints += [self.asset_returns.to_numpy() @ w >= self.target_return]

        # target vol
        if self.max_target_portfolio_vol_an is not None:
            if covar is None:
                raise ValueError(f"asset_returns must be given")
            constraints += [cvx.quad_form(w, covar) <= self.max_target_portfolio_vol_an**2]
        if self.min_target_portfolio_vol_an is not None:
            if covar is None:
                raise ValueError(f"asset_returns must be given")
            constraints += [cvx.quad_form(w, covar) >= self.min_target_portfolio_vol_an**2]

        # turnover_constraint:
        if self.turnover_constraint is not None:
            if self.weights_0 is None:
                print(f"weights_0 must be given for turnover_constraint")
            else:
                constraints += [cvx.norm(w - self.weights_0, 1) <= self.turnover_constraint]

        # tracking error constraint
        if self.tracking_err_vol_constraint is not None:
            if self.benchmark_weights is None:
                raise ValueError(f"benchmark_weights must be given")
            tracking_error_var = cvx.quad_form(w - self.benchmark_weights.to_numpy(), covar)
            constraints += [tracking_error_var <= self.tracking_err_vol_constraint ** 2]  # variance constraint

        # group constraints
        if self.group_lower_upper_constraints is not None:
            group_lower_upper_constraints = self.group_lower_upper_constraints
            for group in group_lower_upper_constraints.group_loadings.columns:
                group_loading = group_lower_upper_constraints.group_loadings[group].to_numpy()
                if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                    if exposure_scaler is None:
                        constraints += [group_loading @ w >= group_lower_upper_constraints.group_min_allocation[group]]
                        constraints += [group_loading @ w <= group_lower_upper_constraints.group_max_allocation[group]]
                    else:
                        constraints += [group_loading @ w >= exposure_scaler * group_lower_upper_constraints.group_min_allocation[group]]
                        constraints += [group_loading @ w <= exposure_scaler * group_lower_upper_constraints.group_max_allocation[group]]

        return constraints

    def set_scipy_constraints(self, covar: np.ndarray = None) -> List:
        """
        constraints for cvx solver
        """
        constraints = []
        if self.is_long_only:
            constraints += [{'type': 'ineq', 'fun': long_only_constraint}]

        # exposure
        if self.max_exposure == self.min_exposure:
            constraints += [{'type': 'eq', 'fun': lambda x: self.max_exposure - np.sum(x)}]
        else:
            constraints += [{'type': 'ineq', 'fun': lambda x: self.max_exposure - np.sum(x)}]  # >=0
            constraints += [{'type': 'ineq', 'fun': lambda x: np.sum(x) - self.min_exposure}]  # <=0

        # min weights
        if self.min_weights is not None:
            if isinstance(self.min_weights, pd.Series):
                min_weights = self.min_weights.to_numpy()
            else:
                min_weights = self.min_weights
            constraints += [{'type': 'ineq', 'fun': lambda x: x - min_weights}]

        # max weights
        if self.max_weights is not None:
            if isinstance(self.max_weights, pd.Series):
                max_weights = self.max_weights.to_numpy()
            else:
                max_weights = self.max_weights
            constraints += [{'type': 'ineq', 'fun': lambda x: max_weights - x}]

        # group constraints
        if self.group_lower_upper_constraints is not None:
            group_lower_upper_constraints = self.group_lower_upper_constraints
            for group in group_lower_upper_constraints.group_loadings.columns:
                group_loading = group_lower_upper_constraints.group_loadings[group].to_numpy()
                min_weight = group_lower_upper_constraints.group_min_allocation[group]
                max_weight = group_lower_upper_constraints.group_max_allocation[group]
                if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                    constraints += [{'type': 'ineq', 'fun': lambda x: group_loading * x - min_weight}]
                    constraints += [{'type': 'ineq', 'fun': lambda x: max_weight - group_loading * x}]

        return constraints


def total_weight_constraint(x, total: float = 1.0):
    return total - np.sum(x)


def long_only_constraint(x):
    return x
