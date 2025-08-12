"""
implement constraints as dataclass object
to support setting various constrains
"""
from __future__ import annotations, division

import pandas as pd
import numpy as np
import cvxpy as cvx
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Union, Literal
from cvxpy.atoms.affine.wraps import psd_wrap


@dataclass
class GroupLowerUpperConstraints:
    """
    add constraints that each asset group is group_min_allocation <= sum group weights <= group_max_allocation
    """
    group_loadings: pd.DataFrame  # columns=instruments, index=groups, data=1 if instrument in indexed group else 0
    group_min_allocation: Optional[pd.Series]  # index=groups, data=group min allocation
    group_max_allocation: Optional[pd.Series]  # index=groups, data=group max allocation

    def __post_init__(self):
        if self.group_min_allocation is not None:
            assert self.group_min_allocation.index.equals(self.group_loadings.columns)
        if self.group_max_allocation is not None:
            assert self.group_max_allocation.index.equals(self.group_loadings.columns)

    def update(self, valid_tickers: List[str]) -> GroupLowerUpperConstraints:
        new_self = GroupLowerUpperConstraints(group_loadings=self.group_loadings.loc[valid_tickers, :],
                                              group_min_allocation=self.group_min_allocation,
                                              group_max_allocation=self.group_max_allocation)
        return new_self

    def print(self):
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_min_allocation:\n{self.group_min_allocation}")
        print(f"group_max_allocation:\n{self.group_max_allocation}")


def merge_group_lower_upper_constraints(group_lower_upper_constraints1: GroupLowerUpperConstraints,
                                        group_lower_upper_constraints2: GroupLowerUpperConstraints,
                                        duplicated_keep: Literal['last', 'first'] = 'last'
                                        ) -> GroupLowerUpperConstraints:

    # check if columns do overlap
    overlaps = list(set(group_lower_upper_constraints1.group_loadings.columns) & set(group_lower_upper_constraints2.group_loadings.columns))
    if len(overlaps) > 0 :
        overlaps1 = {x: f"{x}_1" for x in overlaps}
        overlaps2 = {x: f"{x}_2" for x in overlaps}
    else:
        overlaps1 = {}
        overlaps2 = {}

    group_loadings = pd.concat([group_lower_upper_constraints1.group_loadings.rename(overlaps1, axis=1),
                                group_lower_upper_constraints2.group_loadings.rename(overlaps2, axis=1)
                                ], axis=1)

    if (group_lower_upper_constraints1.group_min_allocation is not None
            and group_lower_upper_constraints2.group_min_allocation is not None):
        group_min_allocation = pd.concat([group_lower_upper_constraints1.group_min_allocation.rename(overlaps1),
                                          group_lower_upper_constraints2.group_min_allocation.rename(overlaps2)])
    elif (group_lower_upper_constraints1.group_min_allocation is not None
          and group_lower_upper_constraints2.group_min_allocation is None):
        group_min_allocation = group_lower_upper_constraints1.group_min_allocation.rename(overlaps1)
    elif (group_lower_upper_constraints1.group_min_allocation is None
          and group_lower_upper_constraints2.group_min_allocation is not None):
        group_min_allocation = group_lower_upper_constraints2.group_min_allocation.rename(overlaps2)
    else:
        group_min_allocation = None
    if group_min_allocation is not None:  # fill missing will large negative number
        group_min_allocation = group_min_allocation.reindex(index=group_loadings.columns).fillna(-1e16)

    if (group_lower_upper_constraints1.group_max_allocation is not None
            and group_lower_upper_constraints2.group_max_allocation is not None):
        group_max_allocation = pd.concat([group_lower_upper_constraints1.group_max_allocation.rename(overlaps1),
                                          group_lower_upper_constraints2.group_max_allocation.rename(overlaps2)])
    elif (group_lower_upper_constraints1.group_max_allocation is not None
            and group_lower_upper_constraints2.group_max_allocation is None):
        group_max_allocation = group_lower_upper_constraints1.group_max_allocation.rename(overlaps1)
    elif (group_lower_upper_constraints1.group_max_allocation is None
            and group_lower_upper_constraints2.group_max_allocation is not None):
        group_max_allocation = group_lower_upper_constraints2.group_max_allocation.rename(overlaps2)
    else:
        group_max_allocation = None
    if group_max_allocation is not None:  # fill missing will large positive number
        group_max_allocation = group_max_allocation.reindex(index=group_loadings.columns).fillna(1e16)

    group_lower_upper_constraints = GroupLowerUpperConstraints(group_loadings=group_loadings,
                                                               group_min_allocation=group_min_allocation,
                                                               group_max_allocation=group_max_allocation)
    return group_lower_upper_constraints


@dataclass
class GroupTrackingErrorConstraint:
    """
    add constraints that the tracking error for each asset group is
    tracking_error_var = cvx.quad_form(group_loading*(w - self.benchmark_weights.to_numpy()), covar)
    constraints += [tracking_error_var <= self.tracking_err_vol_constraint ** 2]  # variance constraint

    """
    group_loadings: pd.DataFrame  # columns=instruments, index=groups, data=1 if instrument in indexed group else 0
    group_tre_vols: pd.Series  # index=groups, data=group min allocation

    def update(self, valid_tickers: List[str]) -> GroupTrackingErrorConstraint:
        new_self = GroupTrackingErrorConstraint(group_loadings=self.group_loadings.loc[valid_tickers, :],
                                                group_tre_vols=self.group_tre_vols)
        return new_self

    def set_group_tre_constraints(self, w: cvx.Variable, benchmark_weights: pd.Series, covar: np.ndarray) -> List:
        constraints = []
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].copy()
            if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                # aling just in case
                group_loading = group_loading.loc[benchmark_weights.index]
                tracking_error_var = cvx.quad_form(cvx.multiply(group_loading.to_numpy(), w - benchmark_weights.to_numpy()), covar)
                constraints += [tracking_error_var <= self.group_tre_vols.loc[group] ** 2]  # variance constraint
        return constraints

    def print(self):
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_tre_vols:\n{self.group_tre_vols}")


@dataclass
class GroupTurnoverConstraint:
    """
    add constraints that turnover for each asset group is
    [cvx.norm(w - self.weights_0, 1) <= self.turnover_constraint]
    constraints += [cvx.norm(w - self.weights_0, 1) <= self.turnover_constraint]
    """
    group_loadings: pd.DataFrame  # columns=instruments, index=groups, data=1 if instrument in indexed group else 0
    group_max_turnover: pd.Series  # index=groups, data=group max turnover

    def update(self, valid_tickers: List[str]) -> GroupTurnoverConstraint:
        new_self = GroupTurnoverConstraint(group_loadings=self.group_loadings.loc[valid_tickers, :],
                                           group_max_turnover=self.group_max_turnover)
        return new_self

    def set_group_turnover_constraints(self, w: cvx.Variable, weights_0: pd.Series = None) -> List:
        constraints = []
        if weights_0 is None:
            print(f"weights_0 must be given for turnover_constraint")
        else:
            for group in self.group_loadings.columns:
                group_loading = self.group_loadings[group].copy()
                if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                    group_loading = group_loading.loc[weights_0.index]  # aling just in case
                    constraints += [cvx.norm(cvx.multiply(group_loading.to_numpy(), w - weights_0), 1) <= self.group_max_turnover.loc[group]]
        return constraints

    def print(self):
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_tre_vols:\n{self.group_max_turnover}")


@dataclass
class Constraints:
    is_long_only: bool = True  # for positive allocation weights
    min_weights: pd.Series = None  # instrument min weights  
    max_weights: pd.Series = None  # instrument max weights
    max_exposure: float = 1.0  # for long short portfolios: for long_portfolios = 1
    min_exposure: float = 1.0  # for long short portfolios: for long_portfolios = 1
    benchmark_weights: pd.Series = None  # for minimisation of tracking error 
    tracking_err_vol_constraint: float = None  # annualised sqrt tracking error
    weights_0: Optional[pd.Series] = None  # for turnover constraints
    turnover_constraint: float = None  # for turnover constraints
    turnover_costs: pd.Series = None  # for weights of turnover constraints
    target_return: float = None  # for optimisation with target return
    asset_returns: pd.Series = None  # for optimisation with target return
    max_target_portfolio_vol_an: float = None  # for optimisation with maximum portfolio volatility target
    min_target_portfolio_vol_an: float = None  # for optimisation with maximum portfolio volatility target
    group_lower_upper_constraints: GroupLowerUpperConstraints = None  # for group allocations constraints
    group_tracking_error_constraint: GroupTrackingErrorConstraint = None
    group_turnover_constraint: Optional[GroupTurnoverConstraint] = None
    apply_total_to_good_ratio_for_constraints: bool = True  # for constraint rescale

    def copy(self) -> Constraints:
        this = asdict(self).copy()
        if self.group_lower_upper_constraints is not None:
            gluc = self.group_lower_upper_constraints
            this['group_lower_upper_constraints'] = GroupLowerUpperConstraints(group_loadings=gluc.group_loadings,
                                                                               group_min_allocation=gluc.group_min_allocation,
                                                                               group_max_allocation=gluc.group_max_allocation)
        if self.group_tracking_error_constraint is not None:
            gluc = self.group_tracking_error_constraint
            this['group_tracking_error_constraint'] = GroupTrackingErrorConstraint(group_loadings=gluc.group_loadings,
                                                                                    group_tre_vols=gluc.group_tre_vols)

        if self.group_turnover_constraint is not None:
            gluc = self.group_turnover_constraint
            this['group_turnover_constraint'] = GroupTurnoverConstraint(group_loadings=gluc.group_loadings,
                                                                        group_max_turnover=gluc.group_max_turnover)

        return Constraints(**this)

    def update(self, valid_tickers: List[str], **kwargs) -> Constraints:
        self_dict = asdict(self)
        self_dict.update(kwargs)
        if self.group_lower_upper_constraints is not None:  # asdict will make is dictionary, need to create object
            group_lower_upper_constraints = self.group_lower_upper_constraints.update(valid_tickers=valid_tickers)
            self_dict['group_lower_upper_constraints'] = group_lower_upper_constraints
        if self.group_tracking_error_constraint is not None:
            self_dict['group_tracking_error_constraint'] = self.group_tracking_error_constraint.update(valid_tickers=valid_tickers)
        if self.group_turnover_constraint is not None:
            self_dict['group_turnover_constraint'] = self.group_turnover_constraint.update(valid_tickers=valid_tickers)
        return Constraints(**self_dict)

    def update_group_lower_upper_constraints(self,
                                             group_lower_upper_constraints: GroupLowerUpperConstraints
                                             ) -> Constraints:
        this = self.copy()
        if this.group_lower_upper_constraints is not None:
            this.group_lower_upper_constraints = merge_group_lower_upper_constraints(
                group_lower_upper_constraints1=this.group_lower_upper_constraints,
                group_lower_upper_constraints2=group_lower_upper_constraints)
        else:
            this.group_lower_upper_constraints = group_lower_upper_constraints
        return this

    def update_with_valid_tickers(self,
                                  valid_tickers: List[str],
                                  total_to_good_ratio: float = 1.0,
                                  weights_0: pd.Series = None,
                                  asset_returns: pd.Series = None,
                                  benchmark_weights: pd.Series = None,
                                  target_return: float = None,
                                  rebalancing_indicators: pd.Series = None,
                                  apply_total_to_good_ratio: bool = True
                                  ) -> Constraints:
        """
        if rebalancing_indicators == 0, then min and max weights are set to weight0
        """
        this = self.copy()
        with pd.option_context('future.no_silent_downcasting', True):
            if this.min_weights is not None:
                this.min_weights = this.min_weights[valid_tickers].fillna(0.0)
            if this.max_weights is not None:
                if apply_total_to_good_ratio and self.apply_total_to_good_ratio_for_constraints:
                    # do not change max_weight == 1
                    max_weight = this.max_weights[valid_tickers]
                    this.max_weights = max_weight.where(np.isclose(max_weight, 1.0), other=total_to_good_ratio*max_weight).fillna(0.0)
                else:
                    this.max_weights = this.max_weights[valid_tickers].fillna(0.0)
            if this.group_lower_upper_constraints is not None:
                this.group_lower_upper_constraints = this.group_lower_upper_constraints.update(valid_tickers=valid_tickers)
            if this.group_tracking_error_constraint is not None:
                this.group_tracking_error_constraint = this.group_tracking_error_constraint.update(valid_tickers=valid_tickers)
            if this.group_turnover_constraint is not None:
                this.group_turnover_constraint = this.group_turnover_constraint.update(valid_tickers=valid_tickers)
            if this.turnover_constraint is not None:
                if apply_total_to_good_ratio and self.apply_total_to_good_ratio_for_constraints:
                    this.turnover_constraint *= total_to_good_ratio
            if this.turnover_costs is not None:
                this.turnover_costs = this.turnover_costs.reindex(index=valid_tickers).fillna(1.0)
            if weights_0 is not None:
                this.weights_0 = weights_0.reindex(index=valid_tickers).fillna(0.0)
            if asset_returns is not None:
                this.asset_returns = asset_returns.reindex(index=valid_tickers).fillna(0.0)
            if benchmark_weights is not None:
                benchmark_weights_ = benchmark_weights.reindex(index=valid_tickers).fillna(0.0)
                this.benchmark_weights = benchmark_weights_  # / np.nansum(benchmark_weights_)
            if target_return is not None:
                this.target_return = target_return

            # check rebalancing indicators
            if rebalancing_indicators is not None and weights_0 is not None:
                rebalancing_indicators = rebalancing_indicators[this.weights_0.index].fillna(1.0)  # by default rebalance
                is_rebalanced = np.isclose(rebalancing_indicators, 1.0)
                if this.min_weights is not None:
                    this.min_weights = this.min_weights.where(is_rebalanced, other=weights_0)
                if this.max_weights is not None:
                    this.max_weights = this.max_weights.where(is_rebalanced, other=weights_0)

        return this

    def set_cvx_constraints(self,
                            w: cvx.Variable,  # problem variable
                            covar: Union[np.ndarray, psd_wrap] = None,
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

        else:  # for max sharpe only max exposure works
            if self.max_exposure == self.min_exposure:
                constraints += [cvx.sum(w) == exposure_scaler]
            else:
                constraints += [cvx.sum(w) == exposure_scaler*self.max_exposure]

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

        # group turnover_constraint
        if self.group_turnover_constraint is not None:
            if self.weights_0 is None:
                print(f"weights_0 must be given for turnover_constraint")
            else:
                for group in self.group_turnover_constraint.group_loadings.columns:
                    group_loading = self.group_turnover_constraint.group_loadings[group].copy()
                    if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                        group_loading = group_loading.loc[self.weights_0.index]  # aling just in case
                        constraints += [cvx.norm(cvx.multiply(group_loading.to_numpy(), w - self.weights_0), 1) <= self.group_turnover_constraint.group_max_turnover.loc[group]]
        # otherwise implement single constraints
        elif self.turnover_constraint is not None:
            if self.weights_0 is None:
                print(f"weights_0 must be given for turnover_constraint")
            else:
                if self.turnover_costs is not None:
                    constraints += [cvx.norm(cvx.multiply(self.turnover_costs.to_numpy(), w - self.weights_0), 1) <= self.turnover_constraint]
                else:
                    assert w.size == len(self.weights_0.index)
                    constraints += [cvx.norm(w - self.weights_0, 1) <= self.turnover_constraint]

        # tracking error constraint
        if self.group_tracking_error_constraint is not None:
            if self.benchmark_weights is None:
                raise ValueError(f"benchmark_weights must be given")
            for group in self.group_tracking_error_constraint.group_loadings.columns:
                group_loading = self.group_tracking_error_constraint.group_loadings[group].copy()
                if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                    # aling just in case
                    group_loading = group_loading.loc[self.benchmark_weights.index]
                    tracking_error_var = cvx.quad_form(cvx.multiply(group_loading.to_numpy(), w - self.benchmark_weights.to_numpy()), covar)
                    constraints += [tracking_error_var <= self.group_tracking_error_constraint.group_tre_vols[group] ** 2]  # variance constraint
        # otherwise implement self.tracking_err_vol_constraint
        elif self.tracking_err_vol_constraint is not None:
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
                        if group_lower_upper_constraints.group_min_allocation is not None:
                            constraints += [group_loading @ w >= group_lower_upper_constraints.group_min_allocation[group]]
                        if group_lower_upper_constraints.group_max_allocation is not None:
                            constraints += [group_loading @ w <= group_lower_upper_constraints.group_max_allocation[group]]
                    else:
                        if group_lower_upper_constraints.group_min_allocation is not None:
                            constraints += [group_loading @ w >= exposure_scaler * group_lower_upper_constraints.group_min_allocation[group]]
                        if group_lower_upper_constraints.group_max_allocation is not None:
                            constraints += [group_loading @ w <= exposure_scaler * group_lower_upper_constraints.group_max_allocation[group]]

        return constraints

    def set_scipy_constraints(self, covar: np.ndarray = None) -> List:
        """
        constraints for cvx solver
        """
        constraints = []
        if self.is_long_only and self.min_weights is None:
            constraints += [{'type': 'ineq', 'fun': long_only_constraint}]

        # exposure
        #if self.max_exposure == self.min_exposure:
        #    constraints += [{'type': 'eq', 'fun': lambda x: self.max_exposure - np.sum(x)}]
        #else:
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
                if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                    if group_lower_upper_constraints.group_min_allocation is not None:
                        min_weight = group_lower_upper_constraints.group_min_allocation[group]
                        constraints += [{'type': 'ineq', 'fun': lambda x: group_loading * x - min_weight}]
                    if group_lower_upper_constraints.group_max_allocation is not None:
                        max_weight = group_lower_upper_constraints.group_max_allocation[group]
                        constraints += [{'type': 'ineq', 'fun': lambda x: max_weight - group_loading * x}]
        return constraints

    def set_pyrb_constraints(self, covar: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        constraints for pyrb solver
        """
        constraints = []
        if self.is_long_only and self.min_weights is None:
            n = covar.shape[0]
            bounds = np.array([(0.0, 1.0) for _ in np.arange((n))])

        elif self.min_weights is not None and self.max_weights is not None:
            bounds = np.array([(x, y) for x, y in zip(self.min_weights.to_numpy(), self.max_weights.to_numpy())])
        else:
            bounds = None

        # group constraints
        # C*x <= d
        if self.group_lower_upper_constraints is not None:
            group_lower_upper_constraints = self.group_lower_upper_constraints
            c_rows = []
            c_lhs = []
            for group in group_lower_upper_constraints.group_loadings.columns:
                group_loading = group_lower_upper_constraints.group_loadings[group].to_numpy()
                if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                    if group_lower_upper_constraints.group_min_allocation is not None:
                        min_weight = group_lower_upper_constraints.group_min_allocation[group]
                        c_rows.append(-1.0 * group_loading)
                        c_lhs.append(-1.0 * min_weight)
                    if group_lower_upper_constraints.group_max_allocation is not None:
                        max_weight = group_lower_upper_constraints.group_max_allocation[group]
                        c_rows.append(group_loading)
                        c_lhs.append(max_weight)
            c_rows = np.vstack(c_rows)
            c_lhs = np.array(c_lhs)
        else:
            c_rows = None
            c_lhs = None

        return bounds, c_rows, c_lhs


def total_weight_constraint(x, total: float = 1.0):
    return total - np.sum(x)


def long_only_constraint(x):
    return x
