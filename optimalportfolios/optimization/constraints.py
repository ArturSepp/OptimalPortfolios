"""Portfolio optimization constraints for CVXPY, SciPy, and PyRB taa.

This module provides a comprehensive framework for defining and enforcing portfolio
constraints across multiple optimization backends. It supports individual asset
constraints, group-based constraints, tracking error limits, and turnover controls.

All dataclass containers are immutable (frozen=True). Mutation methods return new instances.
"""
from __future__ import annotations, division
import warnings
import logging
import copy as _copy
import pandas as pd
import numpy as np
import cvxpy as cvx
from dataclasses import dataclass, fields
from typing import List, Tuple, Optional, Union, Callable
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.constraints.nonpos import Inequality
from enum import Enum


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelaxationRecord:
    """Structured record of a frozen-overhang group-bound relaxation.

    Attached to the log record under ``extra={"relaxation": ...}`` so a handler
    can aggregate the per-rebalance relaxations into one run-level tally instead
    of flooding the console. ``items`` is a tuple of (group, kind, old, new)
    where ``kind`` is ``"group_max"`` or ``"group_min"``.
    """
    context: str
    items: Tuple[Tuple[str, str, float, float], ...]
    total_relaxation: float
    max_relaxation: float
    breached_budget: bool
    breached_tol: bool


class ConstraintEnforcementType(Enum):
    """Specification of tracking error and turnover constraint enforcement.

    Attributes:
        FORCED_CONSTRAINTS: Constraints are hard limits enforced by solver.
        UTILITY_CONSTRAINTS: Constraints are added as penalties to objective function.
    """
    FORCED_CONSTRAINTS = 1  # constraints are enforced for qp solver
    UTILITY_CONSTRAINTS = 2  # constraints are added as utility to the objective


def _reindex_optional_series(s: Optional[pd.Series], index: pd.Index, fill_value: float = 0.0) -> Optional[pd.Series]:
    """Reindex a Series to align with given index, filling missing values.

    Args:
        s: Series to reindex (may be None).
        index: Target index to align to.
        fill_value: Value for missing entries.

    Returns:
        Reindexed Series or None if input is None.
    """
    if s is None:
        return None
    return s.reindex(index=index, fill_value=fill_value)


def _copy_optional_series(s: Optional[pd.Series]) -> Optional[pd.Series]:
    """Copy a Series if not None."""
    return s.copy() if s is not None else None


@dataclass(frozen=True)
class GroupLowerUpperConstraints:
    """Group-based allocation constraints with min/max limits.

    Enables portfolio constraints at the group level (e.g., sector, region, asset class)
    rather than individual asset level. Groups are defined via loading matrices (binary or fractional).

    Attributes:
        group_loadings: Matrix (assets x groups) where positive values indicate membership/exposure.
        group_min_allocation: Minimum allocation per group (optional).
        group_max_allocation: Maximum allocation per group (optional).
    """
    group_loadings: pd.DataFrame
    group_min_allocation: Optional[pd.Series]
    group_max_allocation: Optional[pd.Series]

    def __post_init__(self):
        """Validate allocation series indices match group loadings.

        Ensures consistency between group definitions and allocation constraints.
        Uses object.__setattr__ for frozen dataclass initialization.
        """
        group_loadings = self.group_loadings
        group_min_allocation = self.group_min_allocation
        group_max_allocation = self.group_max_allocation

        # drop group_loadings columns where all assets have zero loading
        zero_cols = group_loadings.columns[
            (group_loadings == 0).all(axis=0) | group_loadings.isna().all(axis=0)]
        if len(zero_cols) > 0:
            warnings.warn(
                f"GroupLowerUpperConstraints: group_loadings columns {zero_cols.tolist()} "
                f"have all zero loadings and dropped"
            )
            group_loadings = group_loadings.drop(columns=zero_cols)
            if group_min_allocation is not None:
                group_min_allocation = group_min_allocation.drop(index=zero_cols, errors='ignore')
            if group_max_allocation is not None:
                group_max_allocation = group_max_allocation.drop(index=zero_cols, errors='ignore')

        # if no groups remain, nullify all constraints
        if group_loadings.empty or len(group_loadings.columns) == 0:
            warnings.warn(
                "GroupLowerUpperConstraints: no valid group_loadings remain after dropping "
                "zero-loading columns; setting all group constraints to None"
            )
            group_min_allocation = None
            group_max_allocation = None

        if group_min_allocation is not None:
            this = group_loadings.columns.isin(group_min_allocation.index)
            if not this.all():
                missing = group_loadings.columns[~this]
                warnings.warn(f"in group_min_allocation: loadings in\n{group_loadings.columns} "
                              f"are missing for {missing}")
            group_min_allocation = group_min_allocation.reindex(index=group_loadings.columns)

        if group_max_allocation is not None:
            this = group_loadings.columns.isin(group_max_allocation.index)
            if not this.all():
                missing = group_loadings.columns[~this]
                warnings.warn(f"in group_max_allocation: loadings in\n{group_loadings.columns} "
                              f"are missing for {missing}")
            group_max_allocation = group_max_allocation.reindex(index=group_loadings.columns)

        # assign validated fields via object.__setattr__ (frozen dataclass)
        object.__setattr__(self, 'group_loadings', group_loadings)
        object.__setattr__(self, 'group_min_allocation', group_min_allocation)
        object.__setattr__(self, 'group_max_allocation', group_max_allocation)

    def copy(self) -> GroupLowerUpperConstraints:
        """Create a deep copy of the constraint object.

        Returns:
            GroupLowerUpperConstraints: Independent copy of the constraints.
        """
        return GroupLowerUpperConstraints(
            group_loadings=self.group_loadings.copy(),
            group_min_allocation=_copy_optional_series(self.group_min_allocation),
            group_max_allocation=_copy_optional_series(self.group_max_allocation),
        )

    def update(self, valid_tickers: List[str]) -> GroupLowerUpperConstraints:
        """Filter constraints to valid tickers only.

        Args:
            valid_tickers: List of tickers to retain in constraints.

        Returns:
            GroupLowerUpperConstraints: Filtered constraint object.
        """
        return GroupLowerUpperConstraints(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_min_allocation=_copy_optional_series(self.group_min_allocation),
            group_max_allocation=_copy_optional_series(self.group_max_allocation),
        )

    def drop_constraint(self, name: str) -> GroupLowerUpperConstraints:
        """Remove constraint by group name.

        Args:
            name: Name of group constraint to remove.

        Returns:
            GroupLowerUpperConstraints: Updated constraint object without specified group.
        """
        return GroupLowerUpperConstraints(
            group_loadings=self.group_loadings.drop([name], axis=1),
            group_min_allocation=self.group_min_allocation.drop([name], axis=0) if self.group_min_allocation is not None else None,
            group_max_allocation=self.group_max_allocation.drop([name], axis=0) if self.group_max_allocation is not None else None,
        )

    def set_cvx_group_lower_upper_constraints(self,
                                              w: cvx.Variable,
                                              exposure_scaler: cvx.Variable = None
                                              ) -> List[Inequality]:
        """Generate CVXPY constraints for group allocations.

        Creates linear inequality constraints of the form:
            group_loading @ w >= min_allocation
            group_loading @ w <= max_allocation

        Args:
            w: Portfolio weight variable.
            exposure_scaler: Optional exposure scaling variable for levered portfolios.

        Returns:
            List of CVXPY inequality constraints.
        """
        constraints = []
        multiplier = 1.0 if exposure_scaler is None else exposure_scaler
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].to_numpy()
            if np.any(np.isclose(group_loading, 0.0) == False):
                if self.group_min_allocation is not None:
                    if group in self.group_min_allocation.index:
                        this = self.group_min_allocation.loc[group]
                        if not np.isnan(this):
                            constraints += [group_loading @ w >= multiplier * this]
                    else:
                        warnings.warn(f"no group={group} in group_min_allocation, constraint skipped")
                if self.group_max_allocation is not None:
                    if group in self.group_max_allocation.index:
                        this = self.group_max_allocation.loc[group]
                        if not np.isnan(this):
                            constraints += [group_loading @ w <= multiplier * this]
                    else:
                        warnings.warn(f"no group={group} in group_max_allocation, constraint skipped")
        return constraints

    def print(self):
        """Print constraint details for debugging."""
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_min_allocation:\n{self.group_min_allocation}")
        print(f"group_max_allocation:\n{self.group_max_allocation}")


def merge_group_lower_upper_constraints(
        group_lower_upper_constraints1: GroupLowerUpperConstraints,
        group_lower_upper_constraints2: GroupLowerUpperConstraints
) -> GroupLowerUpperConstraints:
    """Merge two GroupLowerUpperConstraints objects, handling overlaps with suffixes.

    When group names overlap, appends '_1' and '_2' suffixes to distinguish them.
    Missing bounds are filled with specified default values.

    Args:
        group_lower_upper_constraints1: First constraint object.
        group_lower_upper_constraints2: Second constraint object.
        Default for missing min allocations is np.nan so it is ignored by setting constraints
        Default for missing max allocations is np.nan so it is ignored by setting constraints

    Returns:
        Merged GroupLowerUpperConstraints object.
    """
    overlaps = list(set(group_lower_upper_constraints1.group_loadings.columns) &
                    set(group_lower_upper_constraints2.group_loadings.columns))

    if len(overlaps) > 0:
        overlaps1 = {x: f"{x}_1" for x in overlaps}
        overlaps2 = {x: f"{x}_2" for x in overlaps}
    else:
        overlaps1 = {}
        overlaps2 = {}

    duplicates = group_lower_upper_constraints1.group_loadings.index.duplicated()
    if duplicates.any():
        warnings.warn(f"Duplicate values in group_lower_upper_constraints1.group_loadings.index:"
                      f" {group_lower_upper_constraints1.group_loadings.index[duplicates].unique()}")

    duplicates = group_lower_upper_constraints2.group_loadings.index.duplicated()
    if duplicates.any():
        warnings.warn(f"Duplicate values in group_lower_upper_constraints2.group_loadings.index"
                      f" {group_lower_upper_constraints2.group_loadings.index[duplicates].unique()}")

    group_loadings = pd.concat([
        group_lower_upper_constraints1.group_loadings.rename(overlaps1, axis=1),
        group_lower_upper_constraints2.group_loadings.rename(overlaps2, axis=1)
    ], axis=1).fillna(0.0)

    # Merge minimum allocations
    if (group_lower_upper_constraints1.group_min_allocation is not None and
            group_lower_upper_constraints2.group_min_allocation is not None):
        group_min_allocation = pd.concat([
            group_lower_upper_constraints1.group_min_allocation.rename(overlaps1),
            group_lower_upper_constraints2.group_min_allocation.rename(overlaps2)
        ])
    elif group_lower_upper_constraints1.group_min_allocation is not None:
        group_min_allocation = group_lower_upper_constraints1.group_min_allocation.rename(overlaps1)
    elif group_lower_upper_constraints2.group_min_allocation is not None:
        group_min_allocation = group_lower_upper_constraints2.group_min_allocation.rename(overlaps2)
    else:
        group_min_allocation = None

    if group_min_allocation is not None:
        group_min_allocation = group_min_allocation.reindex(index=group_loadings.columns)

    # Merge maximum allocations
    if (group_lower_upper_constraints1.group_max_allocation is not None and
            group_lower_upper_constraints2.group_max_allocation is not None):
        group_max_allocation = pd.concat([
            group_lower_upper_constraints1.group_max_allocation.rename(overlaps1),
            group_lower_upper_constraints2.group_max_allocation.rename(overlaps2)
        ])
    elif group_lower_upper_constraints1.group_max_allocation is not None:
        group_max_allocation = group_lower_upper_constraints1.group_max_allocation.rename(overlaps1)
    elif group_lower_upper_constraints2.group_max_allocation is not None:
        group_max_allocation = group_lower_upper_constraints2.group_max_allocation.rename(overlaps2)
    else:
        group_max_allocation = None

    if group_max_allocation is not None:
        group_max_allocation = group_max_allocation.reindex(index=group_loadings.columns)

    return GroupLowerUpperConstraints(
        group_loadings=group_loadings,
        group_min_allocation=group_min_allocation,
        group_max_allocation=group_max_allocation
    )


@dataclass(frozen=True)
class GroupTrackingErrorConstraint:
    """Group-based tracking error constraints.

    Limits tracking error at the group level relative to a benchmark. Can be enforced
    as hard constraints or as utility penalties.

    Attributes:
        group_loadings: Matrix (assets x groups) where positive values indicate membership.
        group_tre_vols: Maximum tracking error volatility per group.
        group_tre_utility_weights: Utility penalty weights for soft constraints.
    """
    group_loadings: pd.DataFrame
    group_tre_vols: pd.Series = None
    group_tre_utility_weights: pd.Series = None

    def __post_init__(self):
        """Validate that at least one constraint type is specified and aligned."""
        if self.group_tre_vols is not None:
            this = self.group_loadings.columns.isin(self.group_tre_vols.index)
            if not this.all():
                missing = self.group_loadings.columns[~this]
                warnings.warn(f"Missing in group_loadings.columns: {missing}")
        elif self.group_tre_utility_weights is not None:
            this = self.group_loadings.columns.isin(self.group_tre_utility_weights.index)
            if not this.all():
                missing = self.group_loadings.columns[~this]
                warnings.warn(f"Missing in group_loadings.columns: {missing}")
        else:
            raise ValueError(f"group_tre_vols or group_tre_utility_weights must be given")

    def update(self, valid_tickers: List[str]) -> GroupTrackingErrorConstraint:
        """Filter constraints to valid tickers only."""
        return GroupTrackingErrorConstraint(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_tre_vols=_copy_optional_series(self.group_tre_vols),
            group_tre_utility_weights=_copy_optional_series(self.group_tre_utility_weights),
        )

    def set_cvx_group_tre_constraints(
            self,
            w: cvx.Variable,
            benchmark_weights: pd.Series,
            covar: np.ndarray
    ) -> List[Inequality]:
        """Generate CVXPY constraints for group tracking errors.

        Creates quadratic constraints: (group_loading ⊙ (w - bm))' Σ (group_loading ⊙ (w - bm)) ≤ σ²
        where ⊙ denotes element-wise multiplication.
        """
        constraints = []
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].reindex(benchmark_weights.index, fill_value=0.0)
            if np.any(np.isclose(group_loading, 0.0) == False):
                tracking_error_var = cvx.quad_form(
                    cvx.multiply(group_loading.to_numpy(), w - benchmark_weights.to_numpy()),
                    covar
                )
                group_tre_vol = self.group_tre_vols.loc[group]
                constraints += [tracking_error_var <= group_tre_vol ** 2]
        return constraints

    def set_cvx_group_tre_utility(self, w: cvx.Variable,
                                  benchmark_weights: pd.Series,
                                  covar: np.ndarray) -> AddExpression:
        """Add group tracking error as utility penalty to objective function."""
        if self.group_tre_utility_weights is None:
            raise ValueError(f"supply group_tre_utility_weights for GroupTrackingErrorConstraint")
        objective_fun = None
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].reindex(benchmark_weights.index, fill_value=0.0)
            if np.any(np.isclose(group_loading, 0.0) == False):
                group_tre_utility_weight = self.group_tre_utility_weights.loc[group]
                if not np.isnan(group_tre_utility_weight):
                    term = -1.0 * group_tre_utility_weight * cvx.quad_form(
                        cvx.multiply(group_loading.to_numpy(), w - benchmark_weights.to_numpy()), covar)
                    objective_fun = add_term_to_objective_function(objective_fun, term)
        if objective_fun is None:
            warnings.warn(f"objective_fun is None in set_cvx_group_tre_utility()")
        return objective_fun

    def print(self):
        """Print constraint details for debugging."""
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_tre_vols:\n{self.group_tre_vols}")
        print(f"group_tre_utility_weights:\n{self.group_tre_utility_weights}")


@dataclass(frozen=True)
class GroupTurnoverConstraint:
    """Group-based turnover constraints.

    Limits portfolio turnover at the group level to control transaction costs
    and maintain stable exposures.

    Attributes:
        group_loadings: Matrix (assets x groups) where positive values indicate membership.
        group_max_turnover: Maximum L1 turnover per group.
        group_turnover_utility_weights: Utility penalty weights for soft constraints.
    """
    group_loadings: pd.DataFrame
    group_max_turnover: pd.Series = None
    group_turnover_utility_weights: pd.Series = None

    def __post_init__(self):
        """Validate that at least one constraint type is specified and aligned."""
        if self.group_max_turnover is not None:
            this = self.group_loadings.columns.isin(self.group_max_turnover.index)
            if not this.all():
                missing = self.group_loadings.columns[~this]
                warnings.warn(f"Missing in self.group_loadings.columns: {missing}")
        elif self.group_turnover_utility_weights is not None:
            this = self.group_loadings.columns.isin(self.group_turnover_utility_weights.index)
            if not this.all():
                missing = self.group_loadings.columns[~this]
                warnings.warn(f"Missing in self.group_loadings.columns: {missing}")
        else:
            raise ValueError(f"group_max_turnover or group_turnover_utility_weights must be given")

    def update(self, valid_tickers: List[str]) -> GroupTurnoverConstraint:
        """Filter constraints to valid tickers only."""
        return GroupTurnoverConstraint(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_max_turnover=_copy_optional_series(self.group_max_turnover),
            group_turnover_utility_weights=_copy_optional_series(self.group_turnover_utility_weights),
        )

    def set_group_turnover_constraints(
            self,
            w: cvx.Variable,
            weights_0: pd.Series = None
    ) -> List[Inequality]:
        """Generate CVXPY constraints for group turnovers.

        Creates L1 norm constraints: ||group_loading ⊙ (w - w₀)||₁ ≤ max_turnover
        """
        constraints = []
        if weights_0 is None:
            warnings.warn(f"weights_0 must be given for turnover_constraint")
        else:
            for group in self.group_loadings.columns:
                group_loading = self.group_loadings[group].reindex(weights_0.index, fill_value=0.0)
                if np.any(np.isclose(group_loading, 0.0) == False):
                    constraints += [cvx.norm(cvx.multiply(group_loading.to_numpy(), w - weights_0), 1)
                                    <= self.group_max_turnover.loc[group]]
        return constraints

    def set_cvx_group_turnover_utility(self,
                                       w: cvx.Variable,
                                       weights_0: pd.Series
                                       ) -> AddExpression:
        """Add group turnover as utility penalty to objective function."""
        if self.group_turnover_utility_weights is None:
            raise ValueError(f"group_turnover_utility_weights must be supplied")
        objective_fun = None
        if weights_0 is None:
            warnings.warn("weights_0 must be given for group turnover constraint")
        else:
            for group in self.group_loadings.columns:
                group_loading = self.group_loadings[group].reindex(weights_0.index, fill_value=0.0)
                if np.any(np.isclose(group_loading, 0.0) == False):
                    turnover_utility_weight = self.group_turnover_utility_weights.loc[group]
                    if not np.isnan(turnover_utility_weight):
                        term = -1.0 * turnover_utility_weight * cvx.norm(
                            cvx.multiply(group_loading.to_numpy(), w - weights_0), 1)
                        objective_fun = add_term_to_objective_function(objective_fun, term)
        return objective_fun

    def print(self):
        """Print constraint details for debugging."""
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_max_turnover:\n{self.group_max_turnover}")


@dataclass(frozen=True)
class BenchmarkDeviationConstraints:
    """Benchmark Deviation Constraints: can be used for factor-style deviation constraints, industry deviation constraints

    Creates constraints (elementwise): factor_loading_mat ⊙ (w - w₀) ≤ factor_max_deviation    

    Attributes:
        factor_loading_mat: matrix of factor loadings (instruments x groups). E.g. in terms of Sector constraints, it would be matrix of binary values
        factor_max_deviation: Maximum deviation of the factor aggregated by all names
    """
    factor_loading_mat: pd.DataFrame
    factor_max_deviation: pd.Series

    def __post_init__(self):
        """Validate that at least one constraint type is specified and aligned."""
        if self.factor_max_deviation is not None:
            this = self.factor_max_deviation.index.isin(self.factor_loading_mat.columns)
            if not this.all():
                missing = self.factor_max_deviation.index[~this]
                warnings.warn(f"factor_max_deviation entries not in factor_loading_mat.columns: {missing.tolist()}")
        else:
            raise ValueError(f"factor_max_deviation must be given")

    def copy(self) -> BenchmarkDeviationConstraints:
        return BenchmarkDeviationConstraints(
            factor_loading_mat=self.factor_loading_mat.copy(),
            factor_max_deviation=self.factor_max_deviation.copy(),
        )

    def update(self, valid_tickers: List[str]) -> BenchmarkDeviationConstraints:
        """Filter to valid tickers only."""
        new_self = BenchmarkDeviationConstraints(
            factor_loading_mat=self.factor_loading_mat.loc[valid_tickers, :],
            factor_max_deviation=self.factor_max_deviation
        )
        return new_self

    def set_cvx_constraints(
            self,
            w: cvx.Variable,
            benchmark_weights: pd.Series,
    ) -> List[Inequality]:
        constraints = []
        for group in self.factor_max_deviation.index:
            group_loading = self.factor_loading_mat[group]
            if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                # Align indices
                group_loading = group_loading.loc[benchmark_weights.index]
                active_deviation = cvx.sum(cvx.multiply(group_loading.to_numpy(), w - benchmark_weights.to_numpy()))
                constraints += [cvx.abs(active_deviation) <= self.factor_max_deviation.loc[group]]
        return constraints

    def print(self):
        """Print constraint details."""
        print(f"factor_loading_mat:\n{self.factor_loading_mat}")
        print(f"factor_max_deviation:\n{self.factor_max_deviation}")


def compute_benchmark_beta_loadings(asset_betas: pd.DataFrame,
                                    benchmark_betas: pd.Series,
                                    factor_covar: pd.DataFrame,
                                    benchmark_idio_var: float = 0.0,
                                    ) -> pd.Series:
    """Per-asset loadings of portfolio beta to a benchmark under a factor model.

    With joint factor covariance F, asset loadings B_a (assets x factors),
    benchmark loadings b (factors,) and benchmark idiosyncratic variance
    d_idio, the ex-ante beta of portfolio w to the benchmark is linear:

        beta(w) = w' @ beta_loadings,
        beta_loadings = (B_a @ F @ b) / (b' @ F @ b + d_idio)

    The cross-covariance carries no idiosyncratic term (factor-model
    residuals are independent across instruments), so only the benchmark
    variance in the denominator picks up its idio component.

    Args:
        asset_betas: Factor loadings of assets (assets x factors).
        benchmark_betas: Factor loadings of the benchmark (indexed by factor).
        factor_covar: Factor covariance F (factors x factors).
        benchmark_idio_var: Benchmark idiosyncratic variance (same
            periodicity as factor_covar).

    Returns:
        pd.Series of loadings indexed by asset; beta(w) = loadings @ w.
    """
    factors = factor_covar.index
    b = benchmark_betas.reindex(factors).fillna(0.0).to_numpy()
    ba = asset_betas.reindex(columns=factors).fillna(0.0).to_numpy()
    f = factor_covar.to_numpy()
    fb = f @ b
    denom = float(b @ fb) + float(benchmark_idio_var)
    if denom <= 0.0:
        raise ValueError(f"benchmark variance must be positive, got {denom}")
    return pd.Series(ba @ fb / denom, index=asset_betas.index)


def compute_benchmark_beta_loadings_from_covar(covar: pd.DataFrame,
                                               benchmark_weights: pd.Series,
                                               asset_tickers: List[str],
                                               ) -> pd.Series:
    """Per-asset beta loadings sliced from ONE joint covariance matrix.

    The fully consistent variant of ``compute_benchmark_beta_loadings``:
    when the benchmark constituents are members of the same estimated
    covariance as the assets (one joint fit), the loadings are a pure
    slice — the beta the optimiser enforces then derives from the exact
    matrix its TRE terms use:

        c = Sigma[assets, cons] @ b / (b' Sigma[cons, cons] b),
        beta(w) = c' w

    Args:
        covar: Joint covariance (labelled DataFrame) covering assets AND
            benchmark constituents — e.g. one date of the extended-universe
            ``get_y_covars`` dict.
        benchmark_weights: Static benchmark composition indexed by
            constituent ticker (need not sum to 1; the ratio normalises).
        asset_tickers: Portfolio asset order of w.

    Returns:
        pd.Series of loadings indexed by ``asset_tickers``.
    """
    cons = list(benchmark_weights.index)
    missing = [t for t in cons if t not in covar.index]
    if missing:
        raise KeyError(
            f"benchmark constituents missing from joint covariance: {missing} — "
            f"estimate the covariance with include_static_benchmark_assets=True")
    b = benchmark_weights.to_numpy()
    sig_ab = covar.loc[asset_tickers, cons].to_numpy()
    sig_bb = covar.loc[cons, cons].to_numpy()
    denom = float(b @ sig_bb @ b)
    if denom <= 0.0:
        raise ValueError(f"benchmark variance must be positive, got {denom}")
    return pd.Series(sig_ab @ b / denom, index=asset_tickers)


@dataclass(frozen=True)
class BenchmarkBetaConstraint:
    """Range constraint on ex-ante portfolio beta to a (static) benchmark.

    Given per-asset ``beta_loadings`` c with beta(w) = c'w (see
    ``compute_benchmark_beta_loadings``), creates linear constraints:

        beta_min <= c' @ w <= beta_max

    Follows the ``weights_0`` convention for per-rebalance state: the
    (beta_min, beta_max) spec is static, while ``beta_loadings`` depend on
    the rolling covariance and are injected per rebalancing date via
    ``with_loadings`` before ``set_cvx_all_constraints`` is called.

    Attributes:
        beta_min: Lower bound on ex-ante beta (None = unbounded below).
        beta_max: Upper bound on ex-ante beta (None = unbounded above).
        beta_loadings: Per-asset loadings c (indexed by asset). None until
            injected for the current rebalancing date.
    """
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    beta_loadings: Optional[pd.Series] = None

    def __post_init__(self):
        if self.beta_min is None and self.beta_max is None:
            raise ValueError("at least one of beta_min / beta_max must be given")
        if (self.beta_min is not None and self.beta_max is not None
                and self.beta_min > self.beta_max):
            raise ValueError(f"beta_min={self.beta_min} > beta_max={self.beta_max}")

    def copy(self) -> BenchmarkBetaConstraint:
        return BenchmarkBetaConstraint(
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            beta_loadings=self.beta_loadings.copy() if self.beta_loadings is not None else None,
        )

    def with_loadings(self, beta_loadings: pd.Series) -> BenchmarkBetaConstraint:
        """New instance carrying this rebalancing date's beta loadings."""
        return BenchmarkBetaConstraint(
            beta_min=self.beta_min, beta_max=self.beta_max,
            beta_loadings=beta_loadings)

    def update(self, valid_tickers: List[str]) -> BenchmarkBetaConstraint:
        """Filter loadings to valid tickers (dropped names carry zero weight)."""
        if self.beta_loadings is None:
            return self
        return BenchmarkBetaConstraint(
            beta_min=self.beta_min, beta_max=self.beta_max,
            beta_loadings=self.beta_loadings.reindex(valid_tickers).fillna(0.0))

    def set_cvx_beta_constraints(self, w: cvx.Variable) -> List[Inequality]:
        """Two linear inequalities beta_min <= c'w <= beta_max."""
        if self.beta_loadings is None:
            raise ValueError(
                "beta_loadings not set — inject per-rebalance loadings via "
                "with_loadings() before building cvx constraints")
        c = self.beta_loadings.to_numpy()
        constraints = []
        if self.beta_min is not None:
            constraints += [c @ w >= self.beta_min]
        if self.beta_max is not None:
            constraints += [c @ w <= self.beta_max]
        return constraints

    def print(self):
        print(f"beta range: [{self.beta_min}, {self.beta_max}]")
        print(f"beta_loadings:\n{self.beta_loadings}")


@dataclass(frozen=True)
class Constraints:
    """Comprehensive portfolio optimization constraints.

    Unified container for all portfolio constraints including exposure limits,
    tracking error, turnover, group constraints, and target return/volatility.
    Supports multiple optimization backends (CVXPY, SciPy, PyRB).

    Immutable: all mutation methods return new Constraints instances.

    Attributes:
        is_long_only: Enforce non-negative weights (no short positions).
        min_weights: Minimum weight per asset.
        max_weights: Maximum weight per asset.
        max_exposure: Maximum total portfolio exposure.
        min_exposure: Minimum total portfolio exposure.
        benchmark_weights: Benchmark portfolio weights for tracking error.
        tracking_err_vol_constraint: Maximum tracking error volatility.
        weights_0: Current portfolio weights for turnover calculations.
        turnover_constraint: Maximum portfolio-level L1 turnover.
        turnover_costs: Transaction costs per asset (scales turnover).
        target_return: Minimum target portfolio return.
        asset_returns: Expected returns for each asset.
        max_target_portfolio_vol_an: Maximum annualized portfolio volatility.
        min_target_portfolio_vol_an: Minimum annualized portfolio volatility.
        constraint_enforcement_type: How tracking error/turnover constraints are enforced.
        tre_utility_weight: Penalty weight for tracking error in utility optimization.
        turnover_utility_weight: Penalty weight for turnover in utility optimization.
        group_lower_upper_constraints: Group-level allocation constraints.
        group_tracking_error_constraint: Group-level tracking error constraints.
        group_turnover_constraint: Group-level turnover constraints.
        sector_deviation_constraints: Sector deviation constraints relative to benchmark.
        style_deviation_constraints: Style deviation constraints relative to benchmark.
    """
    is_long_only: bool = True
    min_weights: pd.Series = None
    max_weights: pd.Series = None
    max_exposure: float = 1.0
    min_exposure: float = 1.0
    benchmark_weights: pd.Series = None
    tracking_err_vol_constraint: float = None
    weights_0: Optional[pd.Series] = None
    turnover_constraint: Optional[float] = None
    turnover_costs: pd.Series = None
    target_return: float = None
    asset_returns: pd.Series = None
    max_target_portfolio_vol_an: float = None
    min_target_portfolio_vol_an: float = None
    constraint_enforcement_type: ConstraintEnforcementType = ConstraintEnforcementType.FORCED_CONSTRAINTS
    tre_utility_weight: Optional[float] = 1.0
    turnover_utility_weight: Optional[float] = 0.40
    group_lower_upper_constraints: Optional[GroupLowerUpperConstraints] = None
    group_tracking_error_constraint: Optional[GroupTrackingErrorConstraint] = None
    group_turnover_constraint: Optional[GroupTurnoverConstraint] = None
    sector_deviation_constraints: Optional[BenchmarkDeviationConstraints] = None
    style_deviation_constraints: Optional[BenchmarkDeviationConstraints] = None
    benchmark_beta_constraint: Optional[BenchmarkBetaConstraint] = None

    def __post_init__(self):
        """Validate that individual min/max weights are consistent with group constraints.

        The group constraint is: group_loading @ w >= group_min (and <= group_max),
        where group_loading can be fractional (not necessarily binary).

        Checks for three infeasibility conditions:
            * Sum of loading-weighted asset upper bounds < group minimum → can't reach group floor
            * Sum of loading-weighted asset lower bounds > group maximum → can't stay under group ceiling
            * Single asset loading-weighted floor > group ceiling → immediate infeasibility

        Raises:
            ValueError: If any combination of individual and group constraints is infeasible.
        """

        # validate min/max weight consistency
        if self.min_weights is not None and self.max_weights is not None:
            if self.min_weights.index.equals(self.max_weights.index):
                violations = self.min_weights > self.max_weights + 1e-10
                if violations.any():
                    bad = self.min_weights.index[violations].tolist()
                    raise ValueError(
                        f"min_weights > max_weights for assets: {bad}"
                    )

        if self.is_long_only:
            if self.min_weights is not None:
                negative = self.min_weights < -1e-10
                if negative.any():
                    bad = self.min_weights.index[negative].tolist()
                    raise ValueError(
                        f"is_long_only=True but min_weights < 0 for assets: {bad}"
                    )

        # validate group constraint
        if self.group_lower_upper_constraints is None:
            return

        gluc = self.group_lower_upper_constraints
        loadings = gluc.group_loadings
        errors = []

        for group in loadings.columns:
            group_loading = loadings[group]
            members = group_loading.index[group_loading > 0]
            if len(members) == 0:
                continue

            member_loadings = group_loading.loc[members]

            # loading-weighted cumulative max of individual weights in group
            if self.max_weights is not None:
                group_max_sum = (self.max_weights.reindex(members, fill_value=1.0) * member_loadings).sum()
            else:
                upper = 1.0 if self.is_long_only else self.max_exposure
                group_max_sum = (member_loadings * upper).sum()

            # loading-weighted cumulative min of individual weights in group
            if self.min_weights is not None:
                group_min_sum = (self.min_weights.reindex(members, fill_value=0.0) * member_loadings).sum()
            else:
                lower = 0.0 if self.is_long_only else -self.max_exposure
                group_min_sum = (member_loadings * lower).sum()

            # Check 1: loading-weighted sum of asset max_weights must reach group min_allocation
            if gluc.group_min_allocation is not None:
                gmin = gluc.group_min_allocation.get(group, np.nan)
                if not np.isnan(gmin) and group_max_sum < gmin - 1e-4:
                    errors.append(
                        f"Group '{group}': loading-weighted sum of asset max_weights ({group_max_sum:.4f}) "
                        f"< group_min_allocation ({gmin:.4f}). "
                        f"Increase max_weights for assets {members.tolist()} or lower group_min_allocation."
                    )

            # Check 2: loading-weighted sum of asset min_weights must stay under group max_allocation
            if gluc.group_max_allocation is not None:
                gmax = gluc.group_max_allocation.get(group, np.nan)
                if not np.isnan(gmax) and group_min_sum > gmax + 1e-4:
                    errors.append(
                        f"Group '{group}': loading-weighted sum of asset min_weights ({group_min_sum:.4f}) "
                        f"> group_max_allocation ({gmax:.4f}). "
                        f"Lower min_weights for assets {members.tolist()} or increase group_max_allocation."
                    )

            # Check 3: single asset loading-weighted floor must not exceed group max
            if gluc.group_max_allocation is not None and self.min_weights is not None:
                gmax = gluc.group_max_allocation.get(group, np.nan)
                if not np.isnan(gmax):
                    for ticker in members:
                        wmin = self.min_weights.get(ticker, 0.0)
                        if not np.isnan(wmin):
                            weighted_min = wmin * member_loadings.loc[ticker]
                            if weighted_min > gmax + 1e-4:
                                errors.append(
                                    f"Asset '{ticker}': min_weight ({wmin:.4f}) x loading "
                                    f"({member_loadings.loc[ticker]:.4f}) = {weighted_min:.4f} "
                                    f"> group '{group}' max_allocation ({gmax:.4f}). "
                                    f"Lower min_weight for '{ticker}' or increase group_max_allocation "
                                    f"for '{group}'."
                                )

        if errors:
            raise ValueError(
                f"Infeasible constraints detected ({len(errors)} violation(s)):\n"
                + "\n".join(f"  [{i + 1}] {e}" for i, e in enumerate(errors))
            )

    def _to_dict(self) -> dict:
        """Convert to dict preserving original types (no recursive conversion).

        Returns:
            Dictionary of field name to field value.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def copy(self, **overrides) -> Constraints:
        """Create a deep copy of all constraints, optionally overriding specific fields.

        Args:
            **overrides: Field names and new values to replace.

        Returns:
            New Constraints instance (deep-copied, then overridden).
        """
        self_dict = _copy.deepcopy(self)._to_dict()
        self_dict.update(overrides)
        return Constraints(**self_dict)

    def update_min_max_weights(
            self,
            min_weights: Optional[pd.Series] = None,
            max_weights: Optional[pd.Series] = None,
    ) -> Constraints:
        """Return a new Constraints with updated min/max weights, all other fields intact.

        Args:
            min_weights: New minimum weights (None keeps existing). Reindexed to existing index.
            max_weights: New maximum weights (None keeps existing). Reindexed to existing index.

        Returns:
            New Constraints instance with updated bounds.
        """
        self_dict = self._to_dict()
        if min_weights is not None:
            if self.min_weights is not None:
                min_weights = min_weights.reindex(index=self.min_weights.index).fillna(0.0)
            self_dict['min_weights'] = min_weights
        if max_weights is not None:
            if self.max_weights is not None:
                max_weights = max_weights.reindex(index=self.max_weights.index).fillna(0.0)
            self_dict['max_weights'] = max_weights
        return Constraints(**self_dict)

    def update(self, valid_tickers: List[str], **kwargs) -> Constraints:
        """Update constraints with valid tickers and additional parameters.

        Args:
            valid_tickers: List of tickers to retain in constraints.
            **kwargs: Additional constraint parameters to update.

        Returns:
            New Constraints object with updated fields.
        """
        self_dict = self._to_dict()
        self_dict.update(kwargs)

        if self.group_lower_upper_constraints is not None:
            self_dict['group_lower_upper_constraints'] = \
                self.group_lower_upper_constraints.update(valid_tickers=valid_tickers)
        if self.group_tracking_error_constraint is not None:
            self_dict['group_tracking_error_constraint'] = \
                self.group_tracking_error_constraint.update(valid_tickers=valid_tickers)
        if self.group_turnover_constraint is not None:
            self_dict['group_turnover_constraint'] = \
                self.group_turnover_constraint.update(valid_tickers=valid_tickers)
        if self.sector_deviation_constraints is not None:
            self_dict['sector_deviation_constraints'] = \
                self.sector_deviation_constraints.update(valid_tickers=valid_tickers)
        if self.style_deviation_constraints is not None:
            self_dict['style_deviation_constraints'] = \
                self.style_deviation_constraints.update(valid_tickers=valid_tickers)
        if self.benchmark_beta_constraint is not None:
            self_dict['benchmark_beta_constraint'] = \
                self.benchmark_beta_constraint.update(valid_tickers=valid_tickers)
        return Constraints(**self_dict)

    def update_group_lower_upper_constraints(
            self,
            group_lower_upper_constraints: GroupLowerUpperConstraints
    ) -> Constraints:
        """Add or merge group lower/upper constraints.

        Args:
            group_lower_upper_constraints: New group constraints to add/merge.

        Returns:
            New Constraints object with updated group constraints.
        """
        self_dict = self._to_dict()
        if self.group_lower_upper_constraints is not None:
            self_dict['group_lower_upper_constraints'] = merge_group_lower_upper_constraints(
                group_lower_upper_constraints1=self.group_lower_upper_constraints,
                group_lower_upper_constraints2=group_lower_upper_constraints)
        else:
            self_dict['group_lower_upper_constraints'] = group_lower_upper_constraints
        return Constraints(**self_dict)

    def update_with_valid_tickers(
            self,
            valid_tickers: List[str],
            total_to_good_ratio: Optional[float] = None,
            weights_0: pd.Series = None,
            asset_returns: pd.Series = None,
            benchmark_weights: pd.Series = None,
            target_return: float = None,
            rebalancing_indicators: pd.Series = None,
            context: str = '',
            max_relaxation_tol: Optional[float] = None
    ) -> Constraints:
        """Update constraints with valid tickers and rebalancing logic.

        All pd.Series fields are reindexed to valid_tickers to ensure aligned indices.

        Assets with rebalancing_indicators == 0 have fixed min/max weights at current weights,
        effectively preventing trading in those positions.

        Args:
            valid_tickers: List of tickers to retain.
            total_to_good_ratio: Scaling factor for constrained exposure.
            weights_0: Current portfolio weights.
            asset_returns: Expected asset returns.
            benchmark_weights: Benchmark portfolio weights.
            target_return: Target portfolio return.
            rebalancing_indicators: Binary indicators (1=rebalance, 0=hold fixed).

        Returns:
            New Constraints object with all Series aligned to valid_tickers.
        """
        valid_index = pd.Index(valid_tickers)
        self_dict = self._to_dict()

        # Update individual weight constraints — aligned to valid_tickers
        if self.min_weights is not None:
            self_dict['min_weights'] = self.min_weights.reindex(index=valid_index, fill_value=0.0)
        if self.max_weights is not None:
            max_w = self.max_weights.reindex(index=valid_index, fill_value=0.0)
            if total_to_good_ratio is not None:
                max_w = max_w.where(np.isclose(max_w, 1.0), other=total_to_good_ratio * max_w)
            self_dict['max_weights'] = max_w

        # Update group constraints
        if self.group_lower_upper_constraints is not None:
            self_dict['group_lower_upper_constraints'] = \
                self.group_lower_upper_constraints.update(valid_tickers=valid_tickers)
        if self.group_tracking_error_constraint is not None:
            self_dict['group_tracking_error_constraint'] = \
                self.group_tracking_error_constraint.update(valid_tickers=valid_tickers)
        if self.group_turnover_constraint is not None:
            self_dict['group_turnover_constraint'] = \
                self.group_turnover_constraint.update(valid_tickers=valid_tickers)

        # Update turnover constraints with exposure scaling
        if self.turnover_constraint is not None and total_to_good_ratio is not None:
            self_dict['turnover_constraint'] = self.turnover_constraint * total_to_good_ratio
        if self.turnover_costs is not None:
            self_dict['turnover_costs'] = self.turnover_costs.reindex(index=valid_index, fill_value=1.0)

        # Update portfolio universe — all aligned to valid_tickers
        if weights_0 is not None:
            self_dict['weights_0'] = weights_0.reindex(index=valid_index, fill_value=0.0)
        elif self.weights_0 is not None:
            self_dict['weights_0'] = self.weights_0.reindex(index=valid_index, fill_value=0.0)

        if asset_returns is not None:
            self_dict['asset_returns'] = asset_returns.reindex(index=valid_index, fill_value=0.0)
        elif self.asset_returns is not None:
            self_dict['asset_returns'] = self.asset_returns.reindex(index=valid_index, fill_value=0.0)

        if benchmark_weights is not None:
            self_dict['benchmark_weights'] = benchmark_weights.reindex(index=valid_index, fill_value=0.0)
        elif self.benchmark_weights is not None:
            self_dict['benchmark_weights'] = self.benchmark_weights.reindex(index=valid_index, fill_value=0.0)

        if target_return is not None:
            self_dict['target_return'] = target_return

        # Apply rebalancing indicators to freeze certain positions
        resolved_weights_0 = self_dict.get('weights_0')
        if rebalancing_indicators is not None and resolved_weights_0 is not None:
            rebal = rebalancing_indicators.reindex(index=valid_index, fill_value=1.0)
            is_rebalanced = np.isclose(rebal, 1.0)
            # Frozen (non-rebalanced) assets inherit weights_0 as both their
            # lower and upper bound. For a long-only book that bound cannot be
            # negative, but a drifted weights_0 can carry a tiny negative from a
            # prior solve (cvx honours the >= 0 constraint only to ~1e-8), which
            # would set min_weights < 0 here and trip the long-only validation in
            # __post_init__. Floor the frozen bound at 0 for long-only so a lone
            # asset frozen at a numerically-negative weight is pinned to 0.
            frozen_weights_0 = (resolved_weights_0.clip(lower=0.0)
                                if self.is_long_only else resolved_weights_0)
            if self_dict['min_weights'] is not None:
                self_dict['min_weights'] = self_dict['min_weights'].where(is_rebalanced, other=frozen_weights_0)
            if self_dict['max_weights'] is not None:
                self_dict['max_weights'] = self_dict['max_weights'].where(is_rebalanced, other=frozen_weights_0)

        # Relax group bounds to accommodate the frozen-position overhang.
        #
        # The freeze step above pins min/max for non-rebalanced assets to
        # ``weights_0``. When ``weights_0`` comes from a drift step (as in
        # the rolling backtest after the use_drifted_weights_0 patch) or
        # from a live portfolio-management system that is slightly out of
        # compliance, the frozen positions can push a group's loading-
        # weighted min above its group_max_allocation (or, symmetrically,
        # push frozen max below group_min_allocation). The optimiser
        # cannot trade frozen assets, so the only feasible resolution is
        # to relax the group bound for this rebalance.
        #
        # We grant a one-period waiver: raise group_max_allocation to the
        # frozen-min sum (or lower group_min_allocation to the frozen-max
        # sum), with a small tolerance. A warning is emitted so the
        # relaxation is visible in logs. The drift-induced overshoot is
        # typically a few tens of basis points; for live-PMS-induced
        # overshoots this is the equivalent of a compliance waiver.
        gluc = self_dict.get('group_lower_upper_constraints')
        min_w = self_dict.get('min_weights')
        max_w = self_dict.get('max_weights')
        if gluc is not None and (min_w is not None or max_w is not None):
            loadings = gluc.group_loadings
            gmin = gluc.group_min_allocation
            gmax = gluc.group_max_allocation
            new_gmin = _copy_optional_series(gmin)
            new_gmax = _copy_optional_series(gmax)
            tol = 1e-4
            relax_msgs = []
            relax_items = []
            for group in loadings.columns:
                group_loading = loadings[group]
                members = group_loading.index[group_loading > 0]
                if len(members) == 0:
                    continue
                member_loadings = group_loading.loc[members]
                # cap overshoot from frozen min
                if gmax is not None and min_w is not None:
                    gmax_val = gmax.get(group, np.nan)
                    if not np.isnan(gmax_val):
                        group_min_sum = float(
                            (min_w.reindex(members, fill_value=0.0)
                             * member_loadings).sum())
                        if group_min_sum > gmax_val + tol:
                            new_gmax.loc[group] = group_min_sum + tol
                            relax_msgs.append(
                                f"  group '{group}': group_max_allocation "
                                f"{gmax_val:.4f} → {group_min_sum + tol:.4f} "
                                f"(frozen-min overshoot)")
                            relax_items.append(
                                (str(group), "group_max", float(gmax_val),
                                 float(group_min_sum + tol)))
                # floor undershoot from frozen max
                if gmin is not None and max_w is not None:
                    gmin_val = gmin.get(group, np.nan)
                    if not np.isnan(gmin_val):
                        group_max_sum = float(
                            (max_w.reindex(members, fill_value=1.0)
                             * member_loadings).sum())
                        if group_max_sum < gmin_val - tol:
                            new_gmin.loc[group] = group_max_sum - tol
                            relax_msgs.append(
                                f"  group '{group}': group_min_allocation "
                                f"{gmin_val:.4f} → {group_max_sum - tol:.4f} "
                                f"(frozen-max undershoot)")
                            relax_items.append(
                                (str(group), "group_min", float(gmin_val),
                                 float(group_max_sum - tol)))
            if relax_msgs:
                _tag = f"[{context}] " if context else ""
                _msg = (
                    _tag + "Constraints.update_with_valid_tickers: relaxing group "
                    "bounds for frozen-position overhang (drift or live "
                    "PMS state):\n" + "\n".join(relax_msgs))
                max_exposure = self_dict.get('max_exposure', 1.0)
                deltas = [abs(new - old) for _, _, old, new in relax_items]
                total_relaxation = float(sum(deltas))
                max_relax = float(max(deltas)) if deltas else 0.0
                breached_budget = bool(
                    new_gmax is not None
                    and len(new_gmax[new_gmax > max_exposure + tol]) > 0)
                breached_tol = bool(
                    max_relaxation_tol is not None and max_relax > max_relaxation_tol)
                record = RelaxationRecord(
                    context=context, items=tuple(relax_items),
                    total_relaxation=total_relaxation, max_relaxation=max_relax,
                    breached_budget=breached_budget, breached_tol=breached_tol)
                if breached_tol:
                    _msg += (f"\n  max single relaxation {max_relax:.4f} exceeds "
                             f"tolerance {max_relaxation_tol:.4f}")
                # Per-rebalance detail at INFO (file); escalate to ERROR when the
                # relaxation magnitude breaches the tolerance or the budget. A
                # run-level RelaxationSummary aggregates these into one line.
                _level = logging.ERROR if (breached_tol or breached_budget) else logging.INFO
                logger.log(_level, _msg, extra={"relaxation": record})
                if breached_budget:
                    breached = new_gmax[new_gmax > max_exposure + tol]
                    logger.error(
                        _tag + "Constraints.update_with_valid_tickers: frozen "
                        "overhang relaxed group_max above max_exposure "
                        "(%s) for %s; constraints are effectively "
                        "infeasible \u2014 the solve output must be "
                        "validated.", max_exposure, breached.to_dict())
                self_dict['group_lower_upper_constraints'] = \
                    GroupLowerUpperConstraints(
                        group_loadings=loadings,
                        group_min_allocation=new_gmin,
                        group_max_allocation=new_gmax,
                    )

        # Update sector and style deviation constraints
        if self.sector_deviation_constraints is not None:
            self_dict["sector_deviation_constraints"] = \
                self.sector_deviation_constraints.update(valid_tickers=valid_tickers)
        if self.style_deviation_constraints is not None:
            self_dict["style_deviation_constraints"] = \
                self.style_deviation_constraints.update(valid_tickers=valid_tickers)

        return Constraints(**self_dict)

    def set_cvx_exposure_constraints(self,
                                     w: cvx.Variable,
                                     exposure_scaler: cvx.Variable = None
                                     ) -> List[Inequality]:
        """Generate CVXPY exposure constraints.

        Creates constraints for long-only, total exposure, and individual weight bounds.

        Args:
            w: Portfolio weight variable.
            exposure_scaler: Optional exposure scaling for levered portfolios.

        Returns:
            List of CVXPY inequality constraints.
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
        else:
            if self.max_exposure == self.min_exposure:
                constraints += [cvx.sum(w) == exposure_scaler * self.max_exposure]
            else:
                # preserve both bounds in Charnes-Cooper space: k*min ≤ sum(y) ≤ k*max
                constraints += [cvx.sum(w) <= exposure_scaler * self.max_exposure]
                constraints += [cvx.sum(w) >= exposure_scaler * self.min_exposure]

        if self.min_weights is not None:
            min_weights = (self.min_weights.to_numpy()
                           if isinstance(self.min_weights, pd.Series)
                           else self.min_weights)
            if exposure_scaler is None:
                constraints += [w >= min_weights]
            else:
                constraints += [w >= exposure_scaler * min_weights]

        if self.max_weights is not None:
            max_weights = (self.max_weights.to_numpy()
                           if isinstance(self.max_weights, pd.Series)
                           else self.max_weights)
            if exposure_scaler is None:
                constraints += [w <= max_weights]
            else:
                constraints += [w <= exposure_scaler * max_weights]
        return constraints

    def set_cvx_all_constraints(
            self,
            w: cvx.Variable,
            covar: Union[np.ndarray, psd_wrap] = None,
            exposure_scaler: cvx.Variable = None
    ) -> List:
        """Generate all CVXPY constraints for portfolio optimization.

        Comprehensive constraint generation for mean-variance and related optimization problems.

        Args:
            w: Portfolio weight variable.
            covar: Covariance matrix (required for volatility/tracking error constraints).
            exposure_scaler: Optional exposure scaling for levered portfolios.

        Returns:
            List of all CVXPY constraints.

        Raises:
            ValueError: If required universe is missing for specified constraints.
        """
        constraints = self.set_cvx_exposure_constraints(w=w, exposure_scaler=exposure_scaler)

        if self.target_return is not None:
            if self.asset_returns is None:
                raise ValueError("asset_returns must be given for target_return constraint")
            constraints += [self.asset_returns.to_numpy() @ w >= self.target_return]

        if self.max_target_portfolio_vol_an is not None:
            if covar is None:
                raise ValueError("covar must be given for portfolio volatility constraint")
            constraints += [cvx.quad_form(w, covar) <= self.max_target_portfolio_vol_an ** 2]
        if self.min_target_portfolio_vol_an is not None:
            if covar is None:
                raise ValueError("covar must be given for portfolio volatility constraint")
            constraints += [cvx.quad_form(w, covar) >= self.min_target_portfolio_vol_an ** 2]

        if self.group_turnover_constraint is not None:
            constraints += self.group_turnover_constraint.set_group_turnover_constraints(
                w=w, weights_0=self.weights_0)
        elif self.turnover_constraint is not None:
            if self.weights_0 is None:
                warnings.warn("weights_0 must be given for turnover constraint")
            else:
                if self.turnover_costs is not None:
                    constraints += [cvx.norm(cvx.multiply(self.turnover_costs.to_numpy(),
                                                          w - self.weights_0), 1)
                                    <= self.turnover_constraint]
                else:
                    assert w.size == len(self.weights_0.index)
                    constraints += [cvx.norm(w - self.weights_0, 1) <= self.turnover_constraint]

        if self.group_tracking_error_constraint is not None:
            constraints += self.group_tracking_error_constraint.set_cvx_group_tre_constraints(
                w=w, benchmark_weights=self.benchmark_weights, covar=covar)
        elif self.tracking_err_vol_constraint is not None:
            if self.benchmark_weights is None:
                raise ValueError("benchmark_weights must be given for tracking error constraint")
            tracking_error_var = cvx.quad_form(w - self.benchmark_weights.to_numpy(), covar)
            constraints += [tracking_error_var <= self.tracking_err_vol_constraint ** 2]

        if self.group_lower_upper_constraints is not None:
            constraints += self.group_lower_upper_constraints.set_cvx_group_lower_upper_constraints(
                w=w, exposure_scaler=exposure_scaler)

        # add sector and style deviation constraints
        if self.sector_deviation_constraints is not None:
            constraints += self.sector_deviation_constraints.set_cvx_constraints(
                w=w, benchmark_weights=self.benchmark_weights)
        if self.style_deviation_constraints is not None:
            constraints += self.style_deviation_constraints.set_cvx_constraints(
                w=w, benchmark_weights=self.benchmark_weights)

        # benchmark beta range: linear in w given per-date beta_loadings
        if self.benchmark_beta_constraint is not None:
            constraints += self.benchmark_beta_constraint.set_cvx_beta_constraints(w=w)

        return constraints

    def set_cvx_utility_objective_constraints(
            self,
            w: cvx.Variable,
            alphas: Optional[np.ndarray] = None,
            covar: Union[np.ndarray, psd_wrap] = None,
            exposure_scaler: cvx.Variable = None
    ) -> Tuple[AddExpression, List[Inequality]]:
        """Generate CVXPY utility objective with constraints added as utility penalties.

        Constructs objective function that combines alpha signals with soft penalties for
        tracking error and turnover, rather than enforcing them as hard constraints.

        Args:
            w: Portfolio weight variable.
            alphas: Expected excess returns (alpha signals).
            covar: Covariance matrix (required for tracking error penalties).
            exposure_scaler: Optional exposure scaling for levered portfolios.

        Returns:
            Tuple of (objective function expression, list of hard constraints).

        Raises:
            ValueError: If required universe is missing for specified penalties.
        """
        benchmark_weights: pd.Series = self.benchmark_weights

        if alphas is not None:
            objective_fun = alphas.T @ (w - benchmark_weights.to_numpy())
        else:
            objective_fun = None

        # Add group turnover penalty (takes precedence over portfolio-level)
        if self.group_turnover_constraint is not None:
            if self.weights_0 is None:
                warnings.warn("weights_0 must be given for group turnover constraint")
            else:
                term = self.group_turnover_constraint.set_cvx_group_turnover_utility(
                    w=w, weights_0=self.weights_0)
                objective_fun = add_term_to_objective_function(objective_fun, term)

        elif self.turnover_utility_weight is not None:
            if self.weights_0 is None:
                warnings.warn("weights_0 must be given for turnover constraint")
            else:
                if self.turnover_costs is not None:
                    term = -1.0 * self.turnover_utility_weight * cvx.norm(
                        cvx.multiply(self.turnover_costs.to_numpy(), w - self.weights_0), 1)
                else:
                    assert w.size == len(self.weights_0.index)
                    term = -1.0 * self.turnover_utility_weight * cvx.norm(w - self.weights_0, 1)
                objective_fun = add_term_to_objective_function(objective_fun, term)

        # Add group tracking error penalty (takes precedence over portfolio-level)
        if self.group_tracking_error_constraint is not None:
            if self.benchmark_weights is None:
                raise ValueError("benchmark_weights must be given for group tracking error constraint")
            else:
                term = self.group_tracking_error_constraint.set_cvx_group_tre_utility(
                    w=w, benchmark_weights=benchmark_weights, covar=covar)
                objective_fun = add_term_to_objective_function(objective_fun, term)

        elif self.tre_utility_weight is not None:
            if benchmark_weights is None:
                raise ValueError("benchmark_weights must be given for tracking error constraint")
            term = -1.0 * self.tre_utility_weight * cvx.quad_form(
                w - self.benchmark_weights.to_numpy(), covar)
            objective_fun = add_term_to_objective_function(objective_fun, term)

        # Generate hard constraints (exposure, bounds, groups)
        constraints = self.set_cvx_exposure_constraints(w=w, exposure_scaler=exposure_scaler)

        if self.target_return is not None:
            if self.asset_returns is None:
                raise ValueError("asset_returns must be given for target_return constraint")
            constraints += [self.asset_returns.to_numpy() @ w >= self.target_return]

        if self.group_lower_upper_constraints is not None:
            constraints += self.group_lower_upper_constraints.set_cvx_group_lower_upper_constraints(
                w=w, exposure_scaler=exposure_scaler)

        # benchmark beta range stays a HARD bound under utility enforcement:
        # it is a policy limit, linear in w given the per-date beta_loadings
        # (the TRE/turnover terms above remain soft penalties)
        if self.benchmark_beta_constraint is not None:
            constraints += self.benchmark_beta_constraint.set_cvx_beta_constraints(w=w)
        return objective_fun, constraints

    def set_scipy_bounds(self, covar: np.ndarray):
        """Convert weight constraints into (min, max) bounds for scipy solvers.

        Handles all combinations of min_weights, max_weights, and is_long_only.
        When neither bound is provided, returns (0, 1) for long-only or None
        for unconstrained. When either bound is provided, the missing side
        defaults to 0 (long-only) or -inf (unconstrained) for lows, and 1 for highs.

        Args:
            covar: Covariance matrix (N x N), used to infer number of assets.

        Returns:
            Array of (min, max) tuples per asset, or None if unconstrained.
        """
        min_w = self.min_weights
        max_w = self.max_weights

        # no explicit bounds: use long-only defaults or fully unconstrained
        if min_w is None and max_w is None:
            if self.is_long_only:
                n = covar.shape[0]
                bounds = np.array([(0.0, 1.0) for _ in range(n)])
            else:
                bounds = None
        else:
            # at least one bound is provided: fill the missing side with defaults
            n = covar.shape[0]
            lows = min_w.to_numpy() if min_w is not None else np.full(n, 0.0 if self.is_long_only else -np.inf)
            highs = max_w.to_numpy() if max_w is not None else np.ones(n)
            bounds = np.array(list(zip(lows, highs)))

        return bounds

    def set_scipy_constraints(self, covar: np.ndarray) -> Tuple[List, np.ndarray]:
        """Generate SciPy-compatible constraints (inequality form: constraint >= 0).

        Converts constraints to format expected by scipy.optimize.minimize.

        Args:
            covar: Covariance matrix (used for bounds inference if needed).

        Returns:
            Tuple of (constraint dictionaries, bounds array).
        """
        constraints = []

        if self.is_long_only and self.min_weights is None:
            constraints += [{'type': 'ineq', 'fun': long_only_constraint}]

        constraints += [{'type': 'ineq', 'fun': lambda x: self.max_exposure - np.sum(x)}]
        constraints += [{'type': 'ineq', 'fun': lambda x: np.sum(x) - self.min_exposure}]

        if self.group_lower_upper_constraints is not None:
            gluc = self.group_lower_upper_constraints
            for group in gluc.group_loadings.columns:
                group_loading = gluc.group_loadings[group].to_numpy()
                if np.any(np.isclose(group_loading, 0.0) == False):
                    if gluc.group_min_allocation is not None:
                        min_weight = gluc.group_min_allocation.loc[group]
                        if not np.isnan(min_weight):
                            constraints += [{'type': 'ineq',
                                             'fun': make_min_constraint(group_loading, min_weight)}]
                    if gluc.group_max_allocation is not None:
                        max_weight = gluc.group_max_allocation.loc[group]
                        if not np.isnan(max_weight):
                            constraints += [{'type': 'ineq',
                                             'fun': make_max_constraint(group_loading, max_weight)}]

        bounds = self.set_scipy_bounds(covar=covar)
        return constraints, bounds

    def set_pyrb_constraints(
            self,
            covar: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate PyRB-compatible constraints in matrix form (C*x <= d).

        Converts group constraints to matrix inequality form for risk budgeting taa.

        Args:
            covar: Covariance matrix (used for bounds inference if needed).

        Returns:
            Tuple of (bounds array, constraint matrix C, constraint vector d).
        """
        bounds = self.set_scipy_bounds(covar=covar)

        if self.group_lower_upper_constraints is not None:
            gluc = self.group_lower_upper_constraints
            c_rows = []
            c_lhs = []

            for group in gluc.group_loadings.columns:
                group_loading = gluc.group_loadings[group].to_numpy()
                if np.any(np.isclose(group_loading, 0.0) == False):
                    if gluc.group_min_allocation is not None:
                        min_weight = gluc.group_min_allocation.loc[group]
                        if not np.isnan(min_weight):
                            c_rows.append(-1.0 * group_loading)
                            c_lhs.append(-1.0 * min_weight)
                    if gluc.group_max_allocation is not None:
                        max_weight = gluc.group_max_allocation.loc[group]
                        if not np.isnan(max_weight):
                            c_rows.append(group_loading)
                            c_lhs.append(max_weight)

            if c_rows:
                c_rows = np.vstack(c_rows)
                c_lhs = np.array(c_lhs)
            else:
                c_rows = None
                c_lhs = None
        else:
            c_rows = None
            c_lhs = None

        return bounds, c_rows, c_lhs

    def print_constraints(
            self,
            constraints_list:  List[Inequality],
    ) -> None:
        """
            Print CVXPY constraints in a readable format for debugging and verification.

            constraints_list: List of CVXPY constraints to print e.g. outputs of set_cvx_exposure_constraints
        """
        print("=== CVXPY constraints ===")
        for i, c in enumerate(constraints_list):
            print(f"\nConstraint {i}")
            print(f"  as str:    {c}")             # most readable
            print(f"  type:      {type(c)}")
            print(f"  shape:     {c.shape}")
            print("---------------------------")

    def check_constraints_violation(
            self,
            constraints_list: List[Inequality],
    ) -> None:
        """
            Check the violations of CVXPY constraints after optimization
            after getting the optimal weights. This can help identify which constraints are binding and if there are any numerical issues.

            constraints_list: List of CVXPY constraints to print e.g. outputs of set_cvx_exposure_constraints
        """
        print("=== Check the Violations of CVXPY constraints ===")
        for i, c in enumerate(constraints_list):
            v = c.violation()   # numpy array of nonnegative violations
            max_v = v.max() if v.size > 0 else 0.0
            print(f"Constraint {i}: max violation = {max_v}")


def add_term_to_objective_function(objective_fun: AddExpression, term: AddExpression) -> AddExpression:
    """Safely add a term to CVXPY objective function, handling None cases.

    Args:
        objective_fun: Existing objective function (may be None).
        term: New term to add (may be None).

    Returns:
        Updated objective function.
    """
    if objective_fun is None:
        if term is not None:
            objective_fun = term
    else:
        if term is not None:
            objective_fun += term
    return objective_fun


def total_weight_constraint(x: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Total portfolio weight constraint: total - sum(x) = 0."""
    return total - np.sum(x)


def long_only_constraint(x: np.ndarray) -> np.ndarray:
    """Long-only constraint: x >= 0."""
    return x


def make_min_constraint(group_loading: np.ndarray, min_weight: float) -> Callable[[np.ndarray], float]:
    """Create minimum group allocation constraint: group_loading @ x >= min_weight."""
    return lambda x: group_loading @ x - min_weight


def make_max_constraint(group_loading: np.ndarray, max_weight: float) -> Callable[[np.ndarray], float]:
    """Create maximum group allocation constraint: group_loading @ x <= max_weight."""
    return lambda x: max_weight - group_loading @ x