"""
LASSO and Group LASSO factor model estimation using CVXPY.

Implements sparse factor model estimation with support for:
    - Standard L1 LASSO regression
    - Group LASSO with predefined or cluster-based groups (HCGL method)
    - Sign constraints on factor loadings (non-negative, non-positive, zero, free)
    - Prior-centered regularisation (penalise toward beta0 instead of zero)
    - Exponentially weighted observations
    - NaN-aware estimation via validity masking (assets with different history lengths)

Convention:
    The beta matrix follows the paper convention: beta is (N x M), where N is the
    number of assets and M is the number of factors. The factor model is:
        Y_t = beta X_t + eps_t
    where Y_t is (N x 1) and X_t is (M x 1). In matrix form for T observations:
        Y = X beta' + E
    where Y is (T x N), X is (T x M), beta is (N x M), and E is (T x N).

NaN handling strategy:
    Instead of removing rows with NaN (which discards valid observations for other assets),
    a binary valid_mask (1.0=valid, 0.0=was NaN) is applied element-wise to the observation
    weights. This zeros out the contribution of missing observations in the objective function
    while preserving all valid data for each asset independently.

EWMA weighting in the solver vs diagnostics:
    The solver objective uses ``cvx.sum_squares(cvx.multiply(weights, residuals))``,
    which squares the weighted residuals. To achieve EWMA decay lam^(T-t) on the squared
    residuals, the weights must be ``sqrt(lam)^(T-t)``, hence ``power_lambda=sqrt(lam)``.
    In-sample diagnostics in ``_compute_solver_diagnostics`` use the same solver weights
    normalised per-column (no additional squaring, since manual np.square provides it).

Prior-centered regularisation:
    When ``factors_beta_prior`` (beta0) is provided, the penalty term becomes
    ||beta - beta0|| instead of ||beta||. For LASSO this is lam*||beta - beta0||_1;
    for Group LASSO it is sum_g lam * sqrt(|g|/G) * ||beta[g,:] - beta0[g,:]||_2.
    This shrinks estimates toward the prior rather than toward zero. NaN entries in
    beta0 are treated as zero (standard shrinkage for those elements). The prior is
    subtracted directly inside the penalty norm -- no variable substitution is needed.
"""

from __future__ import annotations

import warnings
import cvxpy as cvx
import numpy as np
import pandas as pd
import qis as qis
import scipy.cluster.hierarchy as spc
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict
from enum import Enum


class LassoModelType(Enum):
    """Enumeration of supported LASSO estimation methods."""
    LASSO = 1                   # Standard L1 LASSO
    GROUP_LASSO = 2             # Group LASSO with user-defined groups
    GROUP_LASSO_CLUSTERS = 3    # Group LASSO with hierarchical clustering (HCGL)


@dataclass
class LassoEstimationResult:
    """
    Output of LASSO / Group LASSO solver functions.

    Attributes:
        estimated_beta: Factor loadings (N x M), following paper convention.
            N = number of assets, M = number of factors. NaN if solver failed.
        alpha: EWMA-weighted mean residual per asset (N,).
        ss_total: EWMA-weighted total variance per asset (N,).
        ss_res: EWMA-weighted residual variance per asset (N,).
        r2: R-squared per asset (N,).
    """
    estimated_beta: np.ndarray
    alpha: np.ndarray
    ss_total: np.ndarray
    ss_res: np.ndarray
    r2: np.ndarray


def _compute_solver_diagnostics(x: np.ndarray,
                                y: np.ndarray,
                                estimated_beta: np.ndarray,
                                weights: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute in-sample diagnostics from the solver's masked weights.

    Args:
        x: Factor matrix (T x M).
        y: Response matrix (T x N).
        estimated_beta: Estimated loadings (N x M), paper convention.
        weights: Masked observation weights (T x N).

    Returns:
        Tuple of (alpha, ss_total, ss_res, r2), each of shape (N,).
    """
    col_sums = np.sum(weights, axis=0)
    norm_weights = np.divide(weights, col_sums, out=np.zeros_like(weights), where=col_sums != 0)

    # Y = X @ beta' => residuals = y - x @ beta.T
    residuals = y - x @ estimated_beta.T
    alpha = np.sum(norm_weights * residuals, axis=0)
    ss_res = np.sum(norm_weights * np.square(residuals), axis=0)
    y_wmean = np.sum(norm_weights * y, axis=0)
    ss_total = np.sum(norm_weights * np.square(y - y_wmean), axis=0)
    r2 = 1.0 - np.divide(ss_res, ss_total, where=ss_total > 0.0)
    return alpha, ss_total, ss_res, r2


def _compute_solver_weights(t: int,
                            n_y: int,
                            span: Optional[int],
                            valid_mask: np.ndarray
                            ) -> np.ndarray:
    """
    Compute observation weights for the solver objective function.

    Args:
        t: Number of time periods.
        n_y: Number of assets (N).
        span: EWMA span. None uses equal weights.
        valid_mask: Binary validity mask (T x N).

    Returns:
        Masked weights array (T x N).
    """
    if span is not None:
        weights = qis.compute_expanding_power(
            n=t, power_lambda=np.sqrt(1.0 - 2.0 / (span + 1.0)), reverse_columns=True
        )
    else:
        weights = np.ones(t)

    if n_y > 1 and valid_mask.ndim == 2:
        weights = np.tile(weights, (n_y, 1)).T

    weights = weights * valid_mask
    return weights


def _clean_beta_prior(factors_beta_prior: Optional[np.ndarray],
                      n_y: int,
                      n_x: int
                      ) -> np.ndarray:
    """
    Return a clean prior matrix in paper convention (N x M): NaN -> 0, None -> zeros.

    Args:
        factors_beta_prior: Prior beta0 matrix (N x M) or None.
        n_y: Number of assets (N).
        n_x: Number of factors (M).

    Returns:
        Prior matrix (N x M) with NaN replaced by 0.0.
    """
    if factors_beta_prior is not None:
        return np.where(np.isnan(factors_beta_prior), 0.0, factors_beta_prior)
    else:
        return np.zeros((n_y, n_x))


@dataclass
class LassoModel:
    """
    Configurable LASSO/Group LASSO factor model estimator.

    Estimates sparse factor loadings beta in the model Y = X beta' + eps using
    L1 (LASSO) or Group L2/L1 (Group LASSO) regularisation via CVXPY.

    Convention:
        beta is (N x M) following the paper: Y_t = beta X_t where Y_t is (N x 1)
        and X_t is (M x 1). The ``estimated_betas`` DataFrame has
        index=asset_names (N rows) and columns=factor_names (M columns).

    Supports sign constraints via ``factors_beta_loading_signs`` (N x M):
        -  0: Beta constrained to zero
        -  1: Beta constrained non-negative
        - -1: Beta constrained non-positive
        - NaN: Beta unconstrained (free)

    Supports prior-centered regularisation via ``factors_beta_prior`` (N x M):
        The penalty becomes ||beta - beta0|| instead of ||beta||.

    After ``fit()``, results are accessible via:
        - ``estimated_betas``: pd.DataFrame(index=asset_names, columns=factor_names)
        - ``estimation_result_``: LassoEstimationResult with alpha, ss_total, ss_res, r2
    """
    model_type: LassoModelType = LassoModelType.LASSO
    group_data: Optional[pd.Series] = None
    reg_lambda: float = 1e-5
    span: Optional[int] = None
    span_freq_dict: Optional[Dict[str, int]] = None
    demean: bool = True
    x: Optional[pd.DataFrame] = None
    y: Optional[pd.DataFrame] = None
    estimated_betas: Optional[pd.DataFrame] = None
    solver: str = 'ECOS_BB'
    warmup_period: Optional[int] = 12
    exclude_zero_betas: bool = True
    nonneg: bool = False
    factors_beta_loading_signs: Optional[pd.DataFrame] = None
    factors_beta_prior: Optional[pd.DataFrame] = None
    # Computed internally by fit()
    estimation_result_: Optional[LassoEstimationResult] = None
    clusters: Optional[pd.Series] = None
    linkage: Optional[np.ndarray] = None
    cutoff: Optional[float] = None
    valid_mask_: Optional[np.ndarray] = None
    effective_span_: Optional[int] = None

    def __post_init__(self):
        if self.model_type == LassoModelType.GROUP_LASSO and self.group_data is None:
            raise ValueError("group_data must be provided for model_type=GROUP_LASSO")

    def copy(self, kwargs: Optional[Dict] = None) -> LassoModel:
        """Create a copy of this model, optionally overriding parameters."""
        this = asdict(self).copy()
        if kwargs is not None:
            this.update(kwargs)
        return LassoModel(**this)

    def fit(self,
            x: pd.DataFrame,
            y: pd.DataFrame,
            verbose: bool = False,
            span: Optional[float] = None,
            num_lags_newey_west: Optional[int] = None
            ) -> LassoModel:
        """
        Estimate LASSO coefficients for the full sample.

        The sign constraints and prior DataFrames are indexed as (assets x factors),
        i.e., ``factors_beta_loading_signs.loc[y.columns, x.columns]`` gives (N x M).

        Args:
            x: Factor returns (T x M). Index=dates, columns=factors.
            y: Asset returns (T x N). Index=dates, columns=assets. May contain NaNs.
            verbose: If True, print solver diagnostics.
            span: Override EWMA span for this fit call.
            num_lags_newey_west: Newey-West lags (GROUP_LASSO_CLUSTERS only).

        Returns:
            Self with updated ``estimated_betas`` (N x M DataFrame).
        """
        effective_span = span or self.span

        x_np, y_np, valid_mask = get_x_y_np(x=x, y=y, span=effective_span, demean=self.demean)

        # Extract sign constraints as (N x M) numpy array
        factors_beta_loading_signs_np = None
        if self.factors_beta_loading_signs is not None:
            factors_beta_loading_signs_np = (
                self.factors_beta_loading_signs.loc[y.columns, x.columns].to_numpy()
            )

        # Extract prior as (N x M) numpy array
        factors_beta_prior_np = None
        if self.factors_beta_prior is not None:
            factors_beta_prior_np = (
                self.factors_beta_prior.loc[y.columns, x.columns].to_numpy()
            )

        clusters = None
        linkage = None
        cutoff = None

        if self.model_type == LassoModelType.LASSO or y_np.shape[1] == 1:
            result = solve_lasso_cvx_problem(
                x=x_np, y=y_np,
                valid_mask=valid_mask,
                reg_lambda=self.reg_lambda,
                span=effective_span,
                verbose=verbose,
                solver=self.solver,
                nonneg=self.nonneg,
                factors_beta_loading_signs=factors_beta_loading_signs_np,
                factors_beta_prior=factors_beta_prior_np
            )

        elif self.model_type == LassoModelType.GROUP_LASSO:
            group_loadings = qis.set_group_loadings(group_data=self.group_data[y.columns])
            result = solve_group_lasso_cvx_problem(
                x=x_np, y=y_np,
                valid_mask=valid_mask,
                group_loadings=group_loadings.to_numpy(),
                reg_lambda=self.reg_lambda,
                span=effective_span,
                verbose=verbose,
                solver=self.solver,
                nonneg=self.nonneg,
                factors_beta_loading_signs=factors_beta_loading_signs_np,
                factors_beta_prior=factors_beta_prior_np
            )

        elif self.model_type == LassoModelType.GROUP_LASSO_CLUSTERS:
            if num_lags_newey_west is not None:
                corr_matrix = qis.compute_ewm_covar_newey_west(
                    a=y_np, span=effective_span, num_lags=num_lags_newey_west, is_corr=True
                )
            else:
                corr_matrix = qis.compute_ewm_covar(a=y_np, span=effective_span, is_corr=True)

            corr_matrix = pd.DataFrame(corr_matrix, columns=y.columns, index=y.columns)
            clusters, linkage, cutoff = compute_clusters_from_corr_matrix(corr_matrix=corr_matrix)
            group_loadings = qis.set_group_loadings(group_data=clusters)
            result = solve_group_lasso_cvx_problem(
                x=x_np, y=y_np,
                valid_mask=valid_mask,
                group_loadings=group_loadings.to_numpy(),
                reg_lambda=self.reg_lambda,
                span=effective_span,
                verbose=verbose,
                solver=self.solver,
                nonneg=self.nonneg,
                factors_beta_loading_signs=factors_beta_loading_signs_np,
                factors_beta_prior=factors_beta_prior_np
            )

        else:
            raise NotImplementedError(f"Unsupported model_type: {self.model_type}")

        # estimated_beta is (N x M)
        estimated_beta = result.estimated_beta

        # Zero out betas for assets with insufficient non-NaN history
        if self.warmup_period is not None:
            num_non_nans = np.count_nonzero(~np.isnan(y.to_numpy()), axis=0)
            insufficient_history = num_non_nans < self.warmup_period
            if np.any(insufficient_history):
                estimated_beta[insufficient_history, :] = 0.0
                if hasattr(result.alpha, '__len__'):
                    result.alpha[insufficient_history] = np.nan
                    result.ss_total[insufficient_history] = np.nan
                    result.ss_res[insufficient_history] = np.nan
                    result.r2[insufficient_history] = np.nan

        # Store results: estimated_betas is (N x M) DataFrame
        self.x = x
        self.y = y
        self.valid_mask_ = valid_mask
        self.effective_span_ = effective_span
        self.estimated_betas = pd.DataFrame(estimated_beta, index=y.columns, columns=x.columns)
        self.estimation_result_ = result
        self.clusters = clusters
        self.linkage = linkage
        self.cutoff = cutoff
        return self


def get_x_y_np(x: pd.DataFrame,
               y: pd.DataFrame,
               span: Optional[int] = None,
               demean: bool = True
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare numpy arrays from factor and asset return DataFrames with NaN masking.

    Args:
        x: Factor returns (T x M). May have all-NaN rows.
        y: Asset returns (T x N). May contain NaNs.
        span: EWMA span for demeaning. None uses simple demeaning.
        demean: If True, subtract (rolling) mean before estimation.

    Returns:
        Tuple of (x_np, y_np, valid_mask). T' = T-1 if EWMA demeaning.
    """
    assert x.index.equals(y.index), (
        f"x and y must have equal indices: "
        f"x.index has {len(x.index)} entries, y.index has {len(y.index)} entries"
    )

    nan_mask_y = y.isna().to_numpy()

    x_all_nan_rows = x.isna().all(axis=1).to_numpy()
    if np.any(x_all_nan_rows):
        nan_mask_y[x_all_nan_rows, :] = True

    y = y.fillna(0.0)
    x = x.fillna(0.0)
    x_np = x.to_numpy()
    y_np = y.to_numpy()

    if demean:
        if span is None:
            x_np = x_np - np.nanmean(x_np, axis=0)
            y_np = y_np - np.nanmean(y_np, axis=0)
        else:
            x_np = x_np - qis.compute_ewm(x_np, span=span)
            y_np = y_np - qis.compute_ewm(y_np, span=span)
            x_np = x_np[1:, :]
            y_np = y_np[1:, :]
            nan_mask_y = nan_mask_y[1:, :]

    valid_mask = (~nan_mask_y).astype(float)
    return x_np, y_np, valid_mask


def _derive_valid_mask_from_y(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Derive validity mask from NaN positions in y and fill NaNs to zero."""
    nan_mask = np.isnan(y)
    valid_mask = (~nan_mask).astype(float)
    y_filled = np.where(nan_mask, 0.0, y)
    return y_filled, valid_mask


def solve_lasso_cvx_problem(x: np.ndarray,
                            y: np.ndarray,
                            valid_mask: np.ndarray = None,
                            reg_lambda: float = 1e-8,
                            span: Optional[int] = None,
                            verbose: bool = False,
                            solver: str = 'ECOS_BB',
                            nonneg: bool = False,
                            factors_beta_loading_signs: np.ndarray = None,
                            factors_beta_prior: np.ndarray = None
                            ) -> LassoEstimationResult:
    """
    Solve L1-regularised (LASSO) regression via CVXPY.

    Minimises: (1/T) || W * (X beta' - Y) ||^2 + lam * ||beta - beta0||_1

    where beta is (N x M), X is (T x M), Y is (T x N).

    Args:
        x: Factor matrix (T x M).
        y: Response matrix (T x N).
        valid_mask: Binary mask (T x N). If None, derived from np.isnan(y).
        reg_lambda: L1 regularisation strength.
        span: EWMA span for observation weighting.
        verbose: If True, print CVXPY solver output.
        solver: CVXPY solver name.
        nonneg: If True, constrain all betas >= 0.
        factors_beta_loading_signs: Sign constraint matrix (N x M).
        factors_beta_prior: Prior beta0 matrix (N x M). NaN entries -> zero prior.

    Returns:
        LassoEstimationResult with estimated_beta (N x M).
    """
    assert y.ndim == 2, f"y must be 2D, got {y.ndim}D"
    assert x.ndim == 2, f"x must be 2D, got {x.ndim}D"
    assert x.shape[0] == y.shape[0]

    t = x.shape[0]
    n_x = x.shape[1]   # M factors
    n_y = y.shape[1]    # N assets

    if factors_beta_loading_signs is not None:
        assert factors_beta_loading_signs.shape == (n_y, n_x)
    if factors_beta_prior is not None:
        assert factors_beta_prior.shape == (n_y, n_x)

    if valid_mask is None:
        y, valid_mask = _derive_valid_mask_from_y(y)

    # beta is (N x M) following paper convention
    constraints = []
    if factors_beta_loading_signs is not None:
        beta = cvx.Variable((n_y, n_x))

        zero_mask = np.isclose(factors_beta_loading_signs, 0.0).astype(float)
        nonneg_mask = np.greater(factors_beta_loading_signs, 0.0).astype(float)
        nonpos_mask = np.less(factors_beta_loading_signs, 0.0).astype(float)

        if np.any(zero_mask > 0):
            constraints.append(cvx.multiply(zero_mask, beta) == 0)
        if np.any(nonneg_mask > 0):
            constraints.append(cvx.multiply(nonneg_mask, beta) >= 0)
        if np.any(nonpos_mask > 0):
            constraints.append(cvx.multiply(nonpos_mask, beta) <= 0)
    else:
        beta = cvx.Variable((n_y, n_x), nonneg=nonneg)

    nan_vector = np.full((n_y, n_x), np.nan)
    nan_result = LassoEstimationResult(
        estimated_beta=nan_vector,
        alpha=np.full(n_y, np.nan), ss_total=np.full(n_y, np.nan),
        ss_res=np.full(n_y, np.nan), r2=np.full(n_y, np.nan)
    )

    if t < 5:
        warnings.warn(f"insufficient total observations in lasso: t={t}")
        return nan_result

    weights = _compute_solver_weights(t=t, n_y=n_y, span=span, valid_mask=valid_mask)
    beta_prior_clean = _clean_beta_prior(factors_beta_prior, n_y, n_x)

    # residuals = y - x @ beta.T, shape (T x N)
    objective_fun = (
        (1.0 / t) * cvx.sum_squares(cvx.multiply(weights, x @ beta.T - y))
        + reg_lambda * cvx.norm1(beta - beta_prior_clean)
    )
    objective = cvx.Minimize(objective_fun)

    if len(constraints) > 0:
        problem = cvx.Problem(objective, constraints)
    else:
        problem = cvx.Problem(objective)
    problem.solve(verbose=verbose, solver=solver)

    estimated_beta = beta.value  # (N x M)
    if estimated_beta is None:
        warnings.warn(f"lasso problem not solved")
        return nan_result

    alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
        x=x, y=y, estimated_beta=estimated_beta, weights=weights
    )
    return LassoEstimationResult(
        estimated_beta=estimated_beta, alpha=alpha,
        ss_total=ss_total, ss_res=ss_res, r2=r2
    )


def solve_group_lasso_cvx_problem(x: np.ndarray,
                                  y: np.ndarray,
                                  group_loadings: np.ndarray,
                                  valid_mask: np.ndarray = None,
                                  reg_lambda: float = 1e-8,
                                  span: Optional[int] = None,
                                  nonneg: bool = False,
                                  verbose: bool = False,
                                  solver: str = 'ECOS_BB',
                                  factors_beta_loading_signs: np.ndarray = None,
                                  factors_beta_prior: np.ndarray = None
                                  ) -> LassoEstimationResult:
    """
    Solve Group LASSO regression via CVXPY.

    Minimises: (1/T) || W * (X beta' - Y) ||^2 + sum_g lam * sqrt(|g|/G) * ||beta[g,:] - beta0[g,:]||_2

    where beta is (N x M), X is (T x M), Y is (T x N), and g indexes groups
    of assets (rows of beta).

    Args:
        x: Factor matrix (T x M).
        y: Asset return matrix (T x N).
        group_loadings: Binary group membership matrix (N x G).
        valid_mask: Binary mask (T x N). If None, derived from np.isnan(y).
        reg_lambda: Group LASSO regularisation strength.
        span: EWMA span for observation weighting.
        nonneg: If True, constrain all betas >= 0.
        verbose: If True, print CVXPY solver output.
        solver: CVXPY solver name.
        factors_beta_loading_signs: Sign constraint matrix (N x M).
        factors_beta_prior: Prior beta0 matrix (N x M). NaN entries -> zero prior.

    Returns:
        LassoEstimationResult with estimated_beta (N x M).
    """
    assert y.ndim == 2
    assert x.ndim == 2
    assert group_loadings.ndim == 2
    assert x.shape[0] == y.shape[0]
    assert y.shape[1] == group_loadings.shape[0]

    t = x.shape[0]
    n_x = x.shape[1]   # M factors
    n_y = y.shape[1]    # N assets
    n_groups = group_loadings.shape[1]

    if factors_beta_loading_signs is not None:
        assert factors_beta_loading_signs.shape == (n_y, n_x)
    if factors_beta_prior is not None:
        assert factors_beta_prior.shape == (n_y, n_x)

    if valid_mask is None:
        y, valid_mask = _derive_valid_mask_from_y(y)

    # beta is (N x M) following paper convention
    constraints = None
    if factors_beta_loading_signs is None:
        beta = cvx.Variable((n_y, n_x), nonneg=nonneg)
    else:
        beta = cvx.Variable((n_y, n_x))
        zero_mask = np.isclose(factors_beta_loading_signs, 0.0).astype(float)
        nonneg_mask = np.greater(factors_beta_loading_signs, 0.0).astype(float)
        nonpos_mask = np.less(factors_beta_loading_signs, 0.0).astype(float)

        constraints = []
        if np.any(zero_mask > 0):
            constraints.append(cvx.multiply(zero_mask, beta) == 0)
        if np.any(nonneg_mask > 0):
            constraints.append(cvx.multiply(nonneg_mask, beta) >= 0)
        if np.any(nonpos_mask > 0):
            constraints.append(cvx.multiply(nonpos_mask, beta) <= 0)

    nan_vector = np.full((n_y, n_x), np.nan)
    nan_result = LassoEstimationResult(
        estimated_beta=nan_vector,
        alpha=np.full(n_y, np.nan), ss_total=np.full(n_y, np.nan),
        ss_res=np.full(n_y, np.nan), r2=np.full(n_y, np.nan)
    )

    if t < 5:
        warnings.warn(f"insufficient total observations in group lasso: t={t}")
        return nan_result

    weights = _compute_solver_weights(t=t, n_y=n_y, span=span, valid_mask=valid_mask)
    beta_prior_clean = _clean_beta_prior(factors_beta_prior, n_y, n_x)

    # residuals = y - x @ beta.T, shape (T x N)
    objective_fun = (1.0 / t) * cvx.sum_squares(cvx.multiply(weights, x @ beta.T - y))

    # Group LASSO penalty: groups are over assets (rows of beta)
    # beta[mask, :] selects rows (assets) in group g, all columns (factors)
    group_masks = [
        np.isclose(group_loadings[:, group_idx], 1.0)
        for group_idx in np.arange(n_groups)
    ]
    group_norms = cvx.sum([
        reg_lambda * np.sqrt(np.sum(mask) / n_groups) * cvx.sum(cvx.norm2(beta[mask, :] - beta_prior_clean[mask, :], axis=1))
        for mask in group_masks
    ])

    objective_fun = objective_fun + group_norms
    objective = cvx.Minimize(objective_fun)

    if constraints is None:
        problem = cvx.Problem(objective)
    else:
        problem = cvx.Problem(objective, constraints)

    problem.solve(verbose=verbose, solver=solver)

    estimated_beta = beta.value  # (N x M)
    if estimated_beta is None:
        warnings.warn(f"group lasso problem not solved")
        return nan_result

    alpha, ss_total, ss_res, r2 = _compute_solver_diagnostics(
        x=x, y=y, estimated_beta=estimated_beta, weights=weights
    )
    return LassoEstimationResult(
        estimated_beta=estimated_beta, alpha=alpha,
        ss_total=ss_total, ss_res=ss_res, r2=r2
    )


def compute_clusters_from_corr_matrix(corr_matrix: pd.DataFrame) -> Tuple[pd.Series, np.ndarray, float]:
    """
    Compute hierarchical clusters from a correlation matrix using Ward's method.

    Converts correlation to distance (1 - corr), applies Ward's agglomerative
    clustering, and cuts the dendrogram at 50% of the maximum pairwise distance.
    """
    corr_matrix = corr_matrix.fillna(0.0)
    pdist = spc.distance.pdist(1.0 - corr_matrix.to_numpy())
    linkage = spc.linkage(pdist, method='ward')
    cutoff = 0.5 * np.max(pdist)
    idx = spc.fcluster(linkage, cutoff, 'distance')
    clusters = pd.Series(idx, index=corr_matrix.columns)
    return clusters, linkage, cutoff
