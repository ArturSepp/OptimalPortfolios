"""
implementation of Lasso model estimation using quadratic solver from cvx
"""

from __future__ import annotations

import cvxpy as cvx
import numpy as np
import pandas as pd
import qis as qis
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict
from enum import Enum


class LassoModelType(Enum):
    LASSO = 1
    GROUP_LASSO = 2


@dataclass
class LassoModel:
    """
    wrapper for lasso model
    estimated betas are pd.Dataframe with pd.DataFrame(betas, index=factor_names, columns=asset_names)

    """
    model_type: LassoModelType = LassoModelType.LASSO
    group_data: pd.Series = None
    reg_lambda: float = 1e-8
    span: Optional[int] = None  # for weight
    fill_nans_to_zero: bool = True
    demean: bool = True
    x: pd.DataFrame = None
    y: pd.DataFrame = None
    estimated_betas: pd.DataFrame = None
    solver: str = 'ECOS_BB'
    warm_up_periods: int = 12  # period to start rolling estimation

    def __post_init__(self):
        if self.model_type == LassoModelType.GROUP_LASSO and self.group_data is None:
            raise ValueError(f"group_data must be provided for model_type = ModelType.GROUP_LASSO")

    def copy(self, kwargs: Dict = None) -> LassoModel:
        this = asdict(self).copy()
        if kwargs is not None:
            this.update(kwargs)
        return LassoModel(**this)

    def get_x_y_np(self, x: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.fill_nans_to_zero:
            y = y.fillna(0.0)
            x = x.fillna(0.0)
        x_np = x.to_numpy()
        y_np = y.to_numpy()
        if self.demean:
            if self.span is None:
                x_np = x_np - np.nanmean(x_np, axis=0)
                y_np = y_np - np.nanmean(y_np, axis=0)
            else:
                x_np = x_np - qis.compute_ewm(x_np, span=self.span)
                y_np = y_np - qis.compute_ewm(y_np, span=self.span)
        return x_np, y_np

    def fit(self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            verbose: bool = False,
            apply_independent_nan_filter: bool = True
            ) -> LassoModel:
        """
        estimate model lasso coefficients for full sample
        """
        x_np, y_np = self.get_x_y_np(x=x, y=y)

        if self.model_type == LassoModelType.GROUP_LASSO:
            # create group loadings
            group_loadings = qis.set_group_loadings(group_data=self.group_data[y.columns])
            estimated_beta = solve_group_lasso_cvx_problem(x=x_np,
                                                           y=y_np,
                                                           group_loadings=group_loadings.to_numpy(),
                                                           reg_lambda=self.reg_lambda,
                                                           span=self.span,
                                                           verbose=verbose,
                                                           solver=self.solver)
        else:
            estimated_beta = solve_lasso_cvx_problem(x=x_np,
                                                     y=y_np,
                                                     reg_lambda=self.reg_lambda,
                                                     span=self.span,
                                                     verbose=verbose,
                                                     solver=self.solver,
                                                     apply_independent_nan_filter=apply_independent_nan_filter)
        self.x = x
        self.y = y
        self.estimated_betas = pd.DataFrame(estimated_beta, index=x.columns, columns=y.columns)
        return self

    def estimate_rolling_betas(self,
                               x: pd.DataFrame,
                               y: pd.DataFrame,
                               verbose: bool = False
                               ) -> Tuple[Dict[pd.Timestamp, pd.DataFrame], Dict[pd.Timestamp, pd.Series], Dict[pd.Timestamp, pd.Series]]:
        """
        fit rolling time series of betas
        """
        betas_t = {}
        residual_vars_t = {}
        r2_t = {}
        for idx, date in enumerate(y.index):
            if idx > self.warm_up_periods:
                self.fit(x=x.iloc[:idx, :], y=y.iloc[:idx, :], verbose=verbose)
                betas, residual_vars, r2 = self.get_betas_residual_var_r2()
                betas_t[date] = betas
                residual_vars_t[date] = residual_vars
                r2_t[date] = r2
        return betas_t, residual_vars_t, r2_t

    def get_betas_residual_var_r2(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        if self.estimated_betas is None:
            raise ValueError(f"calibrate model")
        x_np, y_np = self.get_x_y_np(x=self.x, y=self.y)
        residual_vars, r2 = compute_residual_variance_r2(x=x_np,
                                                         y=y_np,
                                                         beta=self.estimated_betas.to_numpy(),
                                                         span=self.span)
        residual_vars = pd.Series(residual_vars, index=self.estimated_betas.columns)
        r2 = pd.Series(r2, index=self.estimated_betas.columns)
        return self.estimated_betas, residual_vars, r2


def solve_lasso_cvx_problem(x: np.ndarray,
                            y: np.ndarray,
                            reg_lambda: float = 1e-8,
                            span: Optional[int] = None,  # for weight
                            verbose: bool = False,
                            solver: str = 'ECOS_BB',
                            nonneg: bool = False,
                            apply_independent_nan_filter: bool = True
                            ) -> np.ndarray:
    """
    solve lasso for n dimensional matrix of dependent variables y
    each column in y is estimated using independent lasso model
    if data is expected to contain nans, apply_independent_nan_filter = True will do recursive estimation with
    size of nonnan x and dependent on column in y
    out[ut array is (n_x, n_y)
    """
    assert y.ndim in [1, 2]
    assert x.ndim in [1, 2]
    assert x.shape[0] == y.shape[0]

    if x.ndim == 1:
        n_x = 1
    else:
        n_x = x.shape[1]
    if y.ndim == 1:
        n_y = 1
    else:
        n_y = y.shape[1]

    if apply_independent_nan_filter and y.ndim == 2:
        # apply recursive 1-d estimation with independent set of non-nan basis
        betas = np.zeros((n_x, n_y))
        for idx in np.arange(n_y):
            betas[:, idx] = solve_lasso_cvx_problem(x=x, y=y[:, idx],
                                                    reg_lambda=reg_lambda,
                                                    span=span,
                                                    verbose=verbose,
                                                    solver=solver,
                                                    nonneg=nonneg)
        return betas

    # select non nan basis
    x, y = qis.select_non_nan_x_y(x=x, y=y)

    # set variables
    if n_y == 1:
        beta = cvx.Variable(n_x, nonneg=nonneg)
        nan_vector = np.full(n_x, np.nan)
    else:
        beta = cvx.Variable((n_x, n_y))
        nan_vector = np.full((n_x, n_y), np.nan)

    # check suffient obs
    t = x.shape[0]
    if t < 5:  # too little observations
        print(f"small number of non nans in lasso t={t}")
        return nan_vector

    # compute weights
    if span is not None:
        weights = qis.compute_expanding_power(n=t, power_lambda=np.sqrt(1.0 - 2.0 / (span+1.0)), reverse_columns=True)
    else:
        weights = np.ones(t)

    if n_y > 1:
        weights = np.tile(weights, (n_y, 1)).T  # map to columns

    objective_fun = (1.0 / t) * cvx.sum_squares(cvx.multiply(weights, x @ beta - y)) + reg_lambda * cvx.norm1(beta)
    objective = cvx.Minimize(objective_fun)
    problem = cvx.Problem(objective)
    problem.solve(verbose=verbose, solver=solver)
    # problem.solve()

    estimated_beta = beta.value
    if estimated_beta is None:
        # raise ValueError(f"not solved")
        print(f"not solved")
        return nan_vector
    else:
        return estimated_beta


def solve_group_lasso_cvx_problem(x: np.ndarray,
                                  y: np.ndarray,
                                  group_loadings: np.ndarray,
                                  reg_lambda: float = 1e-8,
                                  span: Optional[int] = None,  # for weight
                                  verbose: bool = False,
                                  solver: str = 'ECOS_BB'
                                  ) -> np.ndarray:
    """
    solve lasso for n dimensional matrix of dependent variables y
    each column in y is estimated using independent lasso model
    if data is expected to contain nans, apply_independent_nan_filter = True will do recursive estimation with
    size of nonnan x and dependent on column in y
    out[ut array is (n_x, n_y)
    group_loadings is array (n_y, number groups): with l_ij = 1 if instrument i is in group j and zero otherwise
    """
    # assume multifactor model
    assert y.ndim in [2]
    assert x.ndim in [2]
    assert group_loadings.ndim in [2]
    assert x.shape[0] == y.shape[0]
    assert y.shape[1] == group_loadings.shape[0]

    n_x = x.shape[1]
    n_y = y.shape[1]
    n_groups = group_loadings.shape[1]

    # select non nan basis
    x, y = qis.select_non_nan_x_y(x=x, y=y)

    # set variables
    beta = cvx.Variable((n_x, n_y))
    nan_vector = np.full((n_x, n_y), np.nan)

    # check suffient obs
    t = x.shape[0]
    if t < 5:  # too little observations
        print(f"small number of non nans in lasso t={t}")
        return nan_vector

    # compute weights
    if span is not None:
        weights = qis.compute_expanding_power(n=t, power_lambda=np.sqrt(1.0 - 2.0 / (span+1.0)), reverse_columns=True)
    else:
        weights = np.ones(t)

    if n_y > 1:
        weights = np.tile(weights, (n_y, 1)).T  # map to columns

    objective_fun = (1.0 / t) * cvx.sum_squares(cvx.multiply(weights, x @ beta - y))

    # produce group loadings
    group_masks = [np.isclose(group_loadings[:, group_idx], 1.0) for group_idx in np.arange(n_groups)]
    # need to compute the sum of squares per each instrument in y -> sum by axis=1: beta[:, mask] is matrix
    group_norms = cvx.sum([reg_lambda*np.sqrt(np.sum(group_masks))*cvx.sum(cvx.norm2(beta[:, mask], axis=1)) for mask in group_masks])

    objective_fun = objective_fun + group_norms

    objective = cvx.Minimize(objective_fun)
    problem = cvx.Problem(objective)
    problem.solve(verbose=verbose, solver=solver)
    # problem.solve()

    estimated_beta = beta.value
    if estimated_beta is None:
        # raise ValueError(f"not solved")
        print(f"not solved")
        return nan_vector
    else:
        return estimated_beta


def compute_residual_variance_r2(x: np.ndarray,
                                 y: np.ndarray,
                                 beta: np.ndarray,
                                 span: Optional[int] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute residual var for y = x*beta
    """
    assert x.shape[0] == y.shape[0]
    assert beta.shape[0] == x.shape[1]
    assert beta.shape[1] == y.shape[1]

    t = x.shape[0]
    if span is not None:
        weights = qis.compute_expanding_power(n=t, power_lambda=np.sqrt(1.0 - 2.0 / (span+1.0)), reverse_columns=True)
    else:
        weights = np.ones(t)
    if y.ndim == 2:
        weights = np.tile(weights, (y.shape[1], 1)).T  # map to columns
    num_nonnans = np.count_nonzero(~np.isnan(y), axis=0)
    # weighted residuals
    ss_res = np.nansum(np.square(weights * (x @ beta - y)), axis=0) / num_nonnans
    ss_total = np.nansum(np.square(weights * (y - np.nanmean(y, axis=0))), axis=0) / num_nonnans
    r2 = np.divide(ss_res, ss_total, where=ss_total > 0.0)
    return ss_res, r2
