"""
implementation of Lasso model estimation using quadratic solver from cvx
"""

from __future__ import annotations

import cvxpy as cvx
import numpy as np
import pandas as pd
import qis as qis
import scipy.cluster.hierarchy as spc
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict
from enum import Enum


class LassoModelType(Enum):
    LASSO = 1
    GROUP_LASSO = 2  # use defined groups for lass
    GROUP_LASSO_CLUSTERS = 3  # use statistical clusters for lasso


@dataclass
class ClusterDataByDates:
    clusters: Dict[pd.Timestamp, pd.Series]
    linkages: Dict[pd.Timestamp, np.ndarray]
    cutoffs: Dict[pd.Timestamp, float]


@dataclass
class LassoModel:
    """
    wrapper for lasso model
    estimated betas are pd.Dataframe with pd.DataFrame(betas, index=factor_names, columns=asset_names)

    """
    model_type: LassoModelType = LassoModelType.LASSO
    group_data: Optional[pd.Series] = None
    reg_lambda: float = 1e-5
    span: Optional[int] = None  # for weight
    fill_nans_to_zero: bool = True
    demean: bool = True
    x: pd.DataFrame = None
    y: pd.DataFrame = None
    estimated_betas: pd.DataFrame = None
    solver: str = 'ECOS_BB'
    warmup_period: Optional[int] = 12  # period to start rolling estimation
    exclude_zero_betas: bool = True  # eliminate residual for zero betas
    nonneg: bool = False  # restriction for estimated betas
    # computed internally
    clusters: Optional[pd.Series] = None
    linkage: Optional[np.ndarray] = None
    cutoff: float = None

    def __post_init__(self):
        if self.model_type == LassoModelType.GROUP_LASSO and self.group_data is None:
            raise ValueError(f"group_data must be provided for model_type = ModelType.GROUP_LASSO")

    def copy(self, kwargs: Dict = None) -> LassoModel:
        this = asdict(self).copy()
        if kwargs is not None:
            this.update(kwargs)
        return LassoModel(**this)

    def get_x_y_np(self, x: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        if y data is missing it is replaced by zero
        final check if the number of data points is sufficient to estimate beta is given by warmup_period in fit()
        """
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
                # after demean  x_np[0, :] and y_np[0, :] will be zeros
                x_np = x_np[1:, :]
                y_np = y_np[1:, :]
        return x_np, y_np

    def fit(self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            verbose: bool = False,
            apply_independent_nan_filter: bool = True,
            span: Optional[float] = None,
            num_lags_newey_west: Optional[int] = None
            ) -> LassoModel:
        """
        estimate model lasso coefficients for full sample
        """
        x_np, y_np = self.get_x_y_np(x=x, y=y)
        span = span or self.span
        clusters = None
        linkage = None
        cutoff = None
        # also use lasso for 1-d y
        if self.model_type == LassoModelType.LASSO or y_np.shape[1] == 1:
            estimated_beta, _, _ = solve_lasso_cvx_problem(x=x_np,
                                                           y=y_np,
                                                           reg_lambda=self.reg_lambda,
                                                           span=span,
                                                           verbose=verbose,
                                                           solver=self.solver,
                                                           apply_independent_nan_filter=apply_independent_nan_filter,
                                                           nonneg=self.nonneg)

        elif self.model_type == LassoModelType.GROUP_LASSO:
            # create group loadings
            group_loadings = qis.set_group_loadings(group_data=self.group_data[y.columns])
            estimated_beta = solve_group_lasso_cvx_problem(x=x_np,
                                                           y=y_np,
                                                           group_loadings=group_loadings.to_numpy(),
                                                           reg_lambda=self.reg_lambda,
                                                           span=span,
                                                           verbose=verbose,
                                                           solver=self.solver,
                                                           nonneg=self.nonneg)

        elif self.model_type == LassoModelType.GROUP_LASSO_CLUSTERS:
            # create group loadings using ewma corr matrix
            if num_lags_newey_west is not None:
                corr_matrix = qis.compute_ewm_covar_newey_west(a=y_np, span=span, num_lags=num_lags_newey_west, is_corr=True)
            else:
                corr_matrix = qis.compute_ewm_covar(a=y_np, span=span, is_corr=True)

            corr_matrix = pd.DataFrame(corr_matrix, columns=y.columns, index=y.columns)
            clusters, linkage, cutoff = compute_clusters_from_corr_matrix(corr_matrix=corr_matrix)
            # print(f"clusters=\n{clusters}")
            group_loadings = qis.set_group_loadings(group_data=clusters)
            estimated_beta = solve_group_lasso_cvx_problem(x=x_np,
                                                           y=y_np,
                                                           group_loadings=group_loadings.to_numpy(),
                                                           reg_lambda=self.reg_lambda,
                                                           span=span,
                                                           verbose=verbose,
                                                           solver=self.solver,
                                                           nonneg=self.nonneg)

        else:
            raise NotImplementedError(f"{self.model_type}")

        # enforce local  warmup_period for
        if self.warmup_period is not None:
            # set columns_idx_to_exclude only for y with insufficient history
            num_non_nans = np.count_nonzero(~np.isnan(y), axis=0)
            columns_idx_to_exclude = np.where(num_non_nans > self.warmup_period, False, True)
            if np.any(columns_idx_to_exclude):
                estimated_beta[:, columns_idx_to_exclude] = 0.0

        self.x = x
        self.y = y
        self.estimated_betas = pd.DataFrame(estimated_beta, index=x.columns, columns=y.columns)
        self.clusters = clusters
        self.linkage = linkage
        self.cutoff = cutoff
        return self

    def estimate_rolling_betas(self,
                               x: pd.DataFrame,
                               y: pd.DataFrame,
                               verbose: bool = False,
                               span: Optional[float] = None,
                               num_lags_newey_west: Optional[int] = None
                               ) -> Tuple[Dict[pd.Timestamp, pd.DataFrame],
                                          Dict[pd.Timestamp, pd.Series],
                                          Dict[pd.Timestamp, pd.Series],
                                          Dict[pd.Timestamp, pd.Series],
                                          ClusterDataByDates]:
        """
        fit rolling time series of betas
        """
        betas_t = {}
        total_vars_t = {}
        residual_vars_t = {}
        r2_t = {}
        clusters = {}
        linkages = {}
        cutoffs = {}
        for idx, date in enumerate(y.index):
            if idx > self.warmup_period:  # global warm-up period
                self.fit(x=x.iloc[:idx, :], y=y.iloc[:idx, :], verbose=verbose, span=span, num_lags_newey_west=num_lags_newey_west)
                betas, total_vars, residual_vars, r2 = self.compute_residual_alpha_r2(span=span)
                betas_t[date] = betas
                total_vars_t[date] = total_vars
                residual_vars_t[date] = residual_vars
                r2_t[date] = r2
                clusters[date] = self.clusters
                linkages[date] = self.linkage
                cutoffs[date] = self.cutoff
        cluster_data = ClusterDataByDates(clusters=clusters, linkages=linkages, cutoffs=cutoffs)

        return betas_t, total_vars_t, residual_vars_t, r2_t, cluster_data

    def compute_residual_alpha_r2(self,
                                  span: Optional[float] = None
                                  ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        if self.estimated_betas is None:
            raise ValueError(f"calibrate model")
        x_np, y_np = self.get_x_y_np(x=self.x, y=self.y)
        total_vars, residual_vars, r2 = compute_residual_variance_r2(x=x_np,
                                                                     y=y_np,
                                                                     beta=self.estimated_betas.to_numpy(),
                                                                     span=span or self.span,
                                                                     exclude_zero_betas=self.exclude_zero_betas)
        residual_vars = pd.Series(residual_vars, index=self.estimated_betas.columns)
        total_vars = pd.Series(total_vars, index=self.estimated_betas.columns)
        r2 = pd.Series(r2, index=self.estimated_betas.columns)
        return self.estimated_betas, total_vars, residual_vars, r2


def solve_lasso_cvx_problem(x: np.ndarray,
                            y: np.ndarray,
                            reg_lambda: float = 1e-8,
                            span: Optional[int] = None,  # for weight
                            verbose: bool = False,
                            solver: str = 'ECOS_BB',
                            nonneg: bool = False,
                            apply_independent_nan_filter: bool = True
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        alphas = np.zeros(n_y)
        r2s = np.zeros(n_y)
        for idx in np.arange(n_y):
            betas[:, idx], alphas[idx], r2s[idx] = solve_lasso_cvx_problem(x=x, y=y[:, idx],
                                                                           reg_lambda=reg_lambda,
                                                                           span=span,
                                                                           verbose=verbose,
                                                                           solver=solver,
                                                                           nonneg=nonneg)
        return betas, alphas, r2s

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
        return nan_vector, np.nan, np.nan

    # compute weights
    if span is not None:
        weights = qis.compute_expanding_power(n=t, power_lambda=np.sqrt(1.0 - 2.0 / (span+1.0)), reverse_columns=True)
    else:
        weights = np.ones(t)

    constraints = None
    if n_y > 1:
        weights = np.tile(weights, (n_y, 1)).T  # map to columns

    objective_fun = (1.0 / t) * cvx.sum_squares(cvx.multiply(weights, x @ beta - y)) + reg_lambda * cvx.norm1(beta)
    objective = cvx.Minimize(objective_fun)
    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose=verbose, solver=solver)

    estimated_beta = beta.value
    if estimated_beta is None:
        # raise ValueError(f"not solved")
        print(f"not solved")
        return nan_vector, np.nan, np.nan
    else:
        weights2 = np.square(weights)
        norm_weights = weights2 / np.nansum(weights2, axis=0)
        residuals = (x @ estimated_beta - y)
        alpha = np.nansum(norm_weights * residuals, axis=0)
        ss_res = np.nansum(norm_weights * np.square(residuals), axis=0)
        ss_total = np.nansum(norm_weights * np.square((y - np.nanmean(y, axis=0))), axis=0)
        r2 = 1.0 - np.divide(ss_res, ss_total, where=ss_total > 0.0)
        return estimated_beta, alpha, r2


def solve_group_lasso_cvx_problem(x: np.ndarray,
                                  y: np.ndarray,
                                  group_loadings: np.ndarray,
                                  reg_lambda: float = 1e-8,
                                  span: Optional[int] = None,
                                  nonneg: bool = False,
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
    span: defines exponential weight
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
    beta = cvx.Variable((n_x, n_y), nonneg=nonneg)
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
    # need to compute the sum of squares per each column in y -> sum by axis=1: beta[:, mask] is matrix
    # groups are weighted by sqrt(number of members / n_groups)
    group_norms = cvx.sum([reg_lambda*np.sqrt(np.sum(mask)/n_groups)*cvx.sum(cvx.norm2(beta[:, mask], axis=1)) for mask in group_masks])

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
                                 span: Optional[int] = None,
                                 exclude_zero_betas: bool = True
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    compute residual var for y = x*beta
    exclude_zero_betas: when some columns in y have nan values they are replaced by zeros
    if some columns in y are all zeros, estimated betas are zeros, in this case we ignore the residuals as well
    """
    assert x.shape[0] == y.shape[0]
    assert beta.shape[0] == x.shape[1]
    assert beta.shape[1] == y.shape[1]

    t = x.shape[0]
    if span is not None:
        # weights = qis.compute_expanding_power(n=t, power_lambda=np.sqrt(1.0 - 2.0 / (span+1.0)), reverse_columns=True)
        weights = qis.compute_expanding_power(n=t, power_lambda=1.0 - 2.0 / (span + 1.0), reverse_columns=True)
    else:
        weights = np.ones(t)
    if y.ndim == 2:
        weights = np.tile(weights, (y.shape[1], 1)).T  # map to columns
    num_nonnans = np.count_nonzero(~np.isnan(y), axis=0)
    # weighted residuals
    norm_weights = weights / num_nonnans
    norm_weights = norm_weights / np.nansum(norm_weights, axis=0)
    ss_res = np.nansum(norm_weights * np.square((x @ beta - y)), axis=0)
    ss_total = np.nansum(norm_weights * np.square((y - np.nanmean(y, axis=0))), axis=0)
    r2 = 1.0 - np.divide(ss_res, ss_total, where=ss_total > 0.0)
    # filter out staticstic for zero betas which are produces when the data length for column in Y is not sufficient
    if exclude_zero_betas:
        is_betas_zero = np.where(np.count_nonzero(beta, axis=0) == 0, True, False)
        if np.any(is_betas_zero):
            ss_total = np.where(is_betas_zero, np.nan, ss_total)
            ss_res = np.where(is_betas_zero, np.nan, ss_total)
            r2 = np.where(is_betas_zero, np.nan, ss_total)
    return ss_total, ss_res, r2


def compute_clusters_from_corr_matrix(corr_matrix: pd.DataFrame) -> Tuple[pd.Series, np.ndarray, float]:
    corr_matrix = corr_matrix.fillna(0.0)
    pdist = spc.distance.pdist(1.0 - corr_matrix.to_numpy())
    linkage = spc.linkage(pdist, method='ward')
    cutoff = 0.5 * np.max(pdist)
    idx = spc.fcluster(linkage, cutoff, 'distance')
    clusters = pd.Series(idx, index=corr_matrix.columns)
    return clusters, linkage, cutoff
