"""Group allocation, tracking-error, and turnover constraint models."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cvxpy as cvx
import numpy as np
import pandas as pd
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.constraints.nonpos import Inequality

from optimalportfolios.optimization._constraint_expressions import (
    _cvx_factor_risk,
    add_term_to_objective_function,
    cvx_covar_variance,
)
from optimalportfolios.optimization.covar_factorization import CovarianceFactorization


logger = logging.getLogger('optimalportfolios.optimization.constraints')


@dataclass(frozen=True)
class DroppedGroupRecord:
    """Zero-loading group columns removed from one aligned constraint set."""

    groups: Tuple[str, ...]
    no_groups_remain: bool = False


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
            dropped = DroppedGroupRecord(groups=tuple(map(str, zero_cols)))
            logger.debug(
                "GroupLowerUpperConstraints dropped all-zero loading columns: %s",
                list(dropped.groups), extra={'dropped_groups': dropped})
            group_loadings = group_loadings.drop(columns=zero_cols)
            if group_min_allocation is not None:
                group_min_allocation = group_min_allocation.drop(index=zero_cols, errors='ignore')
            if group_max_allocation is not None:
                group_max_allocation = group_max_allocation.drop(index=zero_cols, errors='ignore')

        # if no groups remain, nullify all constraints
        if group_loadings.empty or len(group_loadings.columns) == 0:
            dropped = DroppedGroupRecord(
                groups=tuple(map(str, zero_cols)), no_groups_remain=True)
            logger.debug(
                "GroupLowerUpperConstraints has no non-zero group loadings; "
                "group constraints disabled", extra={'dropped_groups': dropped})
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
    ], axis=1, sort=False).fillna(0.0)

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
            raise ValueError("group_tre_vols or group_tre_utility_weights must be given")

    def update(self, valid_tickers: List[str]) -> GroupTrackingErrorConstraint:
        """Filter group TRE loadings to ``valid_tickers``.

        Args:
            valid_tickers: Asset labels retained by the solver wrapper.

        Returns:
            A new aligned group tracking-error constraint.
        """
        return GroupTrackingErrorConstraint(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_tre_vols=_copy_optional_series(self.group_tre_vols),
            group_tre_utility_weights=_copy_optional_series(self.group_tre_utility_weights),
        )

    def set_cvx_group_tre_constraints(
            self,
            w: cvx.Variable,
            benchmark_weights: pd.Series,
            covar: Union[np.ndarray, psd_wrap],
            covar_factorization: Optional[CovarianceFactorization] = None,
    ) -> List[Inequality]:
        """Generate CVXPY constraints for group tracking errors.

        With a covariance factor ``B``, creates the SOC constraint
        ``||B.T @ (group_loading ⊙ (w - bm))||₂ <= σ``. Without one,
        preserves the legacy quadratic-form constraint.

        Args:
            w: CVXPY portfolio-weight variable.
            benchmark_weights: Ticker-aligned benchmark weights.
            covar: Covariance representation used by the legacy path.
            covar_factorization: Optional precomputed covariance square root.

        Returns:
            One hard upper-TRE constraint per non-empty group.
        """
        constraints = []
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].reindex(
                benchmark_weights.index, fill_value=0.0)
            if np.any(~np.isclose(group_loading, 0.0)):
                group_active_weights = cvx.multiply(
                    group_loading.to_numpy(), w - benchmark_weights.to_numpy()
                )
                group_tre_vol = self.group_tre_vols.loc[group]
                if covar_factorization is not None:
                    group_risk = _cvx_factor_risk(
                        group_active_weights, covar_factorization)
                    constraints += [cvx.norm(group_risk, 2) <= group_tre_vol]
                else:
                    tracking_error_var = cvx_covar_variance(
                        active_weights=group_active_weights,
                        covar=covar,
                    )
                    constraints += [tracking_error_var <= group_tre_vol ** 2]
        return constraints

    def set_cvx_group_tre_utility(
            self,
            w: cvx.Variable,
            benchmark_weights: pd.Series,
            covar: Union[np.ndarray, psd_wrap],
            covar_factorization: Optional[CovarianceFactorization] = None,
    ) -> AddExpression:
        """Build the sum of group tracking-error utility penalties.

        Args:
            w: CVXPY portfolio-weight variable.
            benchmark_weights: Ticker-aligned benchmark weights.
            covar: Covariance representation used by the legacy path.
            covar_factorization: Optional precomputed covariance square root.

        Returns:
            A negative scalar utility expression, or None when no group has a
            usable loading/penalty.

        Raises:
            ValueError: If group utility weights were not configured.
        """
        if self.group_tre_utility_weights is None:
            raise ValueError(
                "supply group_tre_utility_weights for GroupTrackingErrorConstraint")
        objective_fun = None
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].reindex(
                benchmark_weights.index, fill_value=0.0)
            if np.any(~np.isclose(group_loading, 0.0)):
                group_tre_utility_weight = self.group_tre_utility_weights.loc[group]
                if not np.isnan(group_tre_utility_weight):
                    group_active_weights = cvx.multiply(
                        group_loading.to_numpy(), w - benchmark_weights.to_numpy())
                    group_tre_variance = cvx_covar_variance(
                        active_weights=group_active_weights,
                        covar=covar,
                        covar_factorization=covar_factorization,
                    )
                    term = -1.0 * group_tre_utility_weight * group_tre_variance
                    objective_fun = add_term_to_objective_function(objective_fun, term)
        if objective_fun is None:
            warnings.warn("objective_fun is None in set_cvx_group_tre_utility()")
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
            raise ValueError("group_max_turnover or group_turnover_utility_weights must be given")

    def update(self, valid_tickers: List[str]) -> GroupTurnoverConstraint:
        """Filter group turnover loadings to ``valid_tickers``.

        Args:
            valid_tickers: Asset labels retained by the solver wrapper.

        Returns:
            A new aligned group turnover constraint.
        """
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
        """Generate hard CVXPY constraints for group turnovers.

        Args:
            w: CVXPY portfolio-weight variable.
            weights_0: Ticker-aligned starting weights. When None, no group
                turnover constraints are emitted.

        Returns:
            One L1 turnover constraint per configured group.
        """
        constraints = []
        if weights_0 is None:
            logger.debug("turnover constraint skipped because weights_0 is absent")
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
        """Build the sum of group-turnover utility penalties.

        Args:
            w: CVXPY portfolio-weight variable.
            weights_0: Ticker-aligned starting weights.

        Returns:
            Negative scalar utility expression, or None when no usable group
            penalty is available.

        Raises:
            ValueError: If group turnover utility weights were not configured.
        """
        if self.group_turnover_utility_weights is None:
            raise ValueError("group_turnover_utility_weights must be supplied")
        objective_fun = None
        if weights_0 is None:
            logger.debug("group turnover constraint skipped because weights_0 is absent")
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
