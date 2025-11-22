"""Portfolio optimization constraints for CVXPY, SciPy, and PyRB solvers.

This module provides a comprehensive framework for defining and enforcing portfolio
constraints across multiple optimization backends. It supports individual asset
constraints, group-based constraints, tracking error limits, and turnover controls.
"""
from __future__ import annotations, division
import warnings
import pandas as pd
import numpy as np
import cvxpy as cvx
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Union, Callable
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.constraints.nonpos import Inequality
from enum import Enum


class ConstraintEnforcementType(Enum):
    """Specification of tracking error and turnover constraint enforcement.

    Attributes:
        FORCED_CONSTRAINTS: Constraints are hard limits enforced by solver.
        UTILITY_CONSTRAINTS: Constraints are added as penalties to objective function.
    """
    FORCED_CONSTRAINTS = 1  # constraints are enforced for qp solver
    UTILITY_CONSTRAINTS = 2  # constraints are added as utility to the objective


@dataclass
class GroupLowerUpperConstraints:
    """Group-based allocation constraints with min/max limits.

    Enables portfolio constraints at the group level (e.g., sector, region, asset class)
    rather than individual asset level. Groups are defined via binary loading matrices.

    Attributes:
        group_loadings: Binary matrix (assets x groups) where 1 indicates membership.
        group_min_allocation: Minimum allocation per group (optional).
        group_max_allocation: Maximum allocation per group (optional).
    """
    group_loadings: pd.DataFrame
    group_min_allocation: Optional[pd.Series]
    group_max_allocation: Optional[pd.Series]

    def __post_init__(self):
        """Validate allocation series indices match group loadings.

        Ensures consistency between group definitions and allocation constraints.
        Fills missing values with zeros to prevent constraint violations.
        """
        if self.group_min_allocation is not None:
            this = self.group_min_allocation.index.isin(self.group_loadings.columns)
            if not this.all():
                warnings.warn(f"{self.group_min_allocation.index} missing in\n{self.group_loadings.columns}")
            # temp fix to ensure that min allocation is zero for given group loadings
            self.group_min_allocation.reindex(index=self.group_loadings.columns).fillna(0)
        if self.group_max_allocation is not None:
            this = self.group_max_allocation.index.isin(self.group_loadings.columns)
            if not this.all():
                warnings.warn(f"{self.group_max_allocation.index} missing in\n{self.group_loadings.columns}")
            # temp fix to ensure that max allocation is zero for given group loadings
            self.group_max_allocation.reindex(index=self.group_loadings.columns).fillna(0)

    def copy(self) -> GroupLowerUpperConstraints:
        """Create a deep copy of the constraint object.

        Returns:
            GroupLowerUpperConstraints: Independent copy of the constraints.
        """
        new_self = GroupLowerUpperConstraints(
            group_loadings=self.group_loadings.copy(),
            group_min_allocation=self.group_min_allocation.copy() if self.group_min_allocation is not None else None,
            group_max_allocation=self.group_max_allocation.copy() if self.group_max_allocation is not None else None
        )
        return new_self

    def update(self, valid_tickers: List[str]) -> GroupLowerUpperConstraints:
        """Filter constraints to valid tickers only.

        Args:
            valid_tickers: List of tickers to retain in constraints.

        Returns:
            GroupLowerUpperConstraints: Filtered constraint object.
        """
        new_self = GroupLowerUpperConstraints(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_min_allocation=self.group_min_allocation,
            group_max_allocation=self.group_max_allocation
        )
        return new_self

    def drop_constraint(self, name: str) -> GroupLowerUpperConstraints:
        """Remove constraint by group name.

        Args:
            name: Name of group constraint to remove.

        Returns:
            GroupLowerUpperConstraints: Updated constraint object without specified group.
        """
        group_loadings = self.group_loadings.drop([name], axis=1)
        if self.group_min_allocation is not None:
            group_min_allocation = self.group_min_allocation.drop([name], axis=0)
        else:
            group_min_allocation = None
        if self.group_max_allocation is not None:
            group_max_allocation = self.group_max_allocation.drop([name], axis=0)
        else:
            group_max_allocation = None
        new_self = GroupLowerUpperConstraints(group_loadings=group_loadings,
                                              group_min_allocation=group_min_allocation,
                                              group_max_allocation=group_max_allocation)
        return new_self

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
        # Determine constraint multiplier based on exposure scaling
        if exposure_scaler is None:
            multiplier = 1.0
        else:
            multiplier = exposure_scaler
        # Generate constraints for each group with non-zero loadings
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].to_numpy()
            if np.any(np.isclose(group_loading, 0.0) == False):
                if self.group_min_allocation is not None:
                    if group in self.group_min_allocation:
                        this = self.group_min_allocation[group]
                        if this is not None:
                            constraints += [group_loading @ w >= multiplier * this]
                    else:
                        warnings.warn(f"no group={group} in group_min_allocation, constraint skipped")
                if self.group_max_allocation is not None:
                    if group in self.group_max_allocation:
                        this = self.group_max_allocation[group]
                        if this is not None:
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
        group_lower_upper_constraints2: GroupLowerUpperConstraints,
        filling_value_for_missing_lower_bound: float = -10.0,
        filling_value_for_missing_upper_bound: float = 10.0
) -> GroupLowerUpperConstraints:
    """Merge two GroupLowerUpperConstraints objects, handling overlaps with suffixes.

    When group names overlap, appends '_1' and '_2' suffixes to distinguish them.
    Missing bounds are filled with specified default values.

    Args:
        group_lower_upper_constraints1: First constraint object.
        group_lower_upper_constraints2: Second constraint object.
        filling_value_for_missing_lower_bound: Default for missing min allocations.
        filling_value_for_missing_upper_bound: Default for missing max allocations.

    Returns:
        Merged GroupLowerUpperConstraints object.
    """
    # Check for overlapping column names and create rename mappings
    overlaps = list(set(group_lower_upper_constraints1.group_loadings.columns) & set(group_lower_upper_constraints2.group_loadings.columns))

    if len(overlaps) > 0:
        overlaps1 = {x: f"{x}_1" for x in overlaps}
        overlaps2 = {x: f"{x}_2" for x in overlaps}
    else:
        overlaps1 = {}
        overlaps2 = {}

    # Merge group loadings with duplicate checking
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

    # Merge minimum allocations with proper handling of None cases
    if (group_lower_upper_constraints1.group_min_allocation is not None and
            group_lower_upper_constraints2.group_min_allocation is not None):
        group_min_allocation = pd.concat([
            group_lower_upper_constraints1.group_min_allocation.rename(overlaps1),
            group_lower_upper_constraints2.group_min_allocation.rename(overlaps2)
        ])
    elif (group_lower_upper_constraints1.group_min_allocation is not None and
          group_lower_upper_constraints2.group_min_allocation is None):
        group_min_allocation = group_lower_upper_constraints1.group_min_allocation.rename(overlaps1)
    elif (group_lower_upper_constraints1.group_min_allocation is None and
          group_lower_upper_constraints2.group_min_allocation is not None):
        group_min_allocation = group_lower_upper_constraints2.group_min_allocation.rename(overlaps2)
    else:
        group_min_allocation = None

    if group_min_allocation is not None:
        group_min_allocation = group_min_allocation.reindex(index=group_loadings.columns
                                                            ).fillna(filling_value_for_missing_lower_bound)

    # Merge maximum allocations with proper handling of None cases
    if (group_lower_upper_constraints1.group_max_allocation is not None and
            group_lower_upper_constraints2.group_max_allocation is not None):
        group_max_allocation = pd.concat([
            group_lower_upper_constraints1.group_max_allocation.rename(overlaps1),
            group_lower_upper_constraints2.group_max_allocation.rename(overlaps2)
        ])
    elif (group_lower_upper_constraints1.group_max_allocation is not None and
          group_lower_upper_constraints2.group_max_allocation is None):
        group_max_allocation = group_lower_upper_constraints1.group_max_allocation.rename(overlaps1)
    elif (group_lower_upper_constraints1.group_max_allocation is None and
          group_lower_upper_constraints2.group_max_allocation is not None):
        group_max_allocation = group_lower_upper_constraints2.group_max_allocation.rename(overlaps2)
    else:
        group_max_allocation = None

    if group_max_allocation is not None:
        group_max_allocation = group_max_allocation.reindex(
            index=group_loadings.columns
        ).fillna(filling_value_for_missing_upper_bound)

    return GroupLowerUpperConstraints(
        group_loadings=group_loadings,
        group_min_allocation=group_min_allocation,
        group_max_allocation=group_max_allocation
    )


@dataclass
class GroupTrackingErrorConstraint:
    """Group-based tracking error constraints.

    Limits tracking error at the group level relative to a benchmark. Can be enforced
    as hard constraints or as utility penalties.

    Attributes:
        group_loadings: Binary matrix (assets x groups) where 1 indicates membership.
        group_tre_vols: Maximum tracking error volatility per group.
        group_tre_utility_weights: Utility penalty weights for soft constraints.
    """
    group_loadings: pd.DataFrame
    group_tre_vols: pd.Series = None
    group_tre_utility_weights: pd.Series = None

    def __post_init__(self):
        """Validate that at least one constraint type is specified and aligned."""
        if self.group_tre_vols is not None:
            this = self.group_tre_vols.index.isin(self.group_loadings.columns)
            if not this.all():
                missing = self.group_tre_vols.index[~this]
                warnings.warn(f"Missing in group_tre_vols.index: {missing}")
        elif self.group_tre_utility_weights is not None:
            this = self.group_tre_utility_weights.index.isin(self.group_loadings.columns)
            if not this.all():
                missing = self.group_tre_utility_weights.index[~this]
                warnings.warn(f"Missing in group_tre_utility_weights.index: {missing}")
        else:
            raise ValueError(f"group_tre_vols or group_tre_utility_weights must be given")

    def update(self, valid_tickers: List[str]) -> GroupTrackingErrorConstraint:
        """Filter constraints to valid tickers only.

        Args:
            valid_tickers: List of tickers to retain.

        Returns:
            Filtered GroupTrackingErrorConstraint object.
        """
        new_self = GroupTrackingErrorConstraint(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_tre_vols=self.group_tre_vols,
            group_tre_utility_weights=self.group_tre_utility_weights
        )
        return new_self

    def set_cvx_group_tre_constraints(
            self,
            w: cvx.Variable,
            benchmark_weights: pd.Series,
            covar: np.ndarray
    ) -> List[Inequality]:
        """Generate CVXPY constraints for group tracking errors.

        Creates quadratic constraints: (group_loading ⊙ (w - bm))' Σ (group_loading ⊙ (w - bm)) ≤ σ²
        where ⊙ denotes element-wise multiplication.

        Args:
            w: Portfolio weight variable.
            benchmark_weights: Benchmark portfolio weights.
            covar: Covariance matrix of asset returns.

        Returns:
            List of CVXPY inequality constraints.
        """
        constraints = []
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].copy()
            group_loading = group_loading.loc[benchmark_weights.index]  # align
            if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
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
        """Add group tracking error as utility penalty to objective function.

        Penalizes tracking error with group-specific weights rather than hard constraints.

        Args:
            w: Portfolio weight variable.
            benchmark_weights: Benchmark portfolio weights.
            covar: Covariance matrix of asset returns.

        Returns:
            CVXPY expression for tracking error penalties.

        Raises:
            ValueError: If group_tre_utility_weights is not specified.
        """
        if self.group_tre_utility_weights is None:
            raise ValueError(f"supply group_tre_utility_weights for GroupTrackingErrorConstraint")
        objective_fun = None  # Initialize as None
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].copy().loc[benchmark_weights.index]
            if np.any(np.isclose(group_loading, 0.0) == False):
                # First align with benchmark_weights index
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


@dataclass
class GroupTurnoverConstraint:
    """Group-based turnover constraints.

    Limits portfolio turnover at the group level to control transaction costs
    and maintain stable exposures.

    Attributes:
        group_loadings: Binary matrix (assets x groups) where 1 indicates membership.
        group_max_turnover: Maximum L1 turnover per group.
        group_turnover_utility_weights: Utility penalty weights for soft constraints.
    """
    group_loadings: pd.DataFrame
    group_max_turnover: pd.Series = None
    group_turnover_utility_weights: pd.Series = None

    def __post_init__(self):
        """Validate that at least one constraint type is specified and aligned."""
        if self.group_max_turnover is not None:
            this = self.group_max_turnover.index.isin(self.group_loadings.columns)
            if not this.all():
                missing = self.group_max_turnover.index[~this]
                warnings.warn(f"Missing in group_max_turnover.index: {missing}")
        elif self.group_turnover_utility_weights is not None:
            this = self.group_turnover_utility_weights.index.isin(self.group_loadings.columns)
            if not this.all():
                missing = self.group_turnover_utility_weights.index[~this]
                warnings.warn(f"Missing in group_turnover_utility_weights.index: {missing}")
        else:
            raise ValueError(f"group_max_turnover or group_turnover_utility_weights must be given")

    def update(self, valid_tickers: List[str]) -> GroupTurnoverConstraint:
        """Filter constraints to valid tickers only.

        Args:
            valid_tickers: List of tickers to retain.

        Returns:
            Filtered GroupTurnoverConstraint object.
        """
        new_self = GroupTurnoverConstraint(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_max_turnover=self.group_max_turnover,
            group_turnover_utility_weights=self.group_turnover_utility_weights
        )
        return new_self

    def set_group_turnover_constraints(
            self,
            w: cvx.Variable,
            weights_0: pd.Series = None
    ) -> List[Inequality]:
        """Generate CVXPY constraints for group turnovers.

        Creates L1 norm constraints: ||group_loading ⊙ (w - w₀)||₁ ≤ max_turnover

        Args:
            w: Target portfolio weight variable.
            weights_0: Current portfolio weights.

        Returns:
            List of CVXPY inequality constraints.
        """
        constraints = []
        if weights_0 is None:
            warnings.warn(f"weights_0 must be given for turnover_constraint")
        else:
            for group in self.group_loadings.columns:
                group_loading = self.group_loadings[group].copy()
                if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                    group_loading = group_loading.loc[weights_0.index]  # align indices
                    constraints += [ cvx.norm( cvx.multiply(group_loading.to_numpy(), w - weights_0), 1)
                                     <= self.group_max_turnover.loc[group]]
        return constraints

    def set_cvx_group_turnover_utility(self,
                                       w: cvx.Variable,
                                       weights_0: pd.Series
                                       ) -> AddExpression:
        """Add group turnover as utility penalty to objective function.

        Penalizes turnover with group-specific weights rather than hard constraints.

        Args:
            w: Target portfolio weight variable.
            weights_0: Current portfolio weights.

        Returns:
            CVXPY expression for turnover penalties.

        Raises:
            ValueError: If group_turnover_utility_weights is not specified.
        """
        if self.group_turnover_utility_weights is None:
            raise ValueError(f"group_turnover_utility_weights must be supplied")
        objective_fun = None  # Initialize as None
        if weights_0 is None:
            warnings.warn("weights_0 must be given for group turnover constraint")
        else:
            for group in self.group_loadings.columns:
                group_loading = self.group_loadings[group].loc[weights_0.index]
                if np.any(np.isclose(group_loading, 0.0) == False):
                    turnover_utility_weight = self.group_turnover_utility_weights[group]
                    if not np.isnan(turnover_utility_weight):
                        term = -1.0 * turnover_utility_weight * cvx.norm(cvx.multiply(group_loading.to_numpy(), w - weights_0), 1)
                        objective_fun = add_term_to_objective_function(objective_fun, term)
        return objective_fun

    def print(self):
        """Print constraint details for debugging."""
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_max_turnover:\n{self.group_max_turnover}")


@dataclass
class Constraints:
    """Comprehensive portfolio optimization constraints.

    Unified container for all portfolio constraints including exposure limits,
    tracking error, turnover, group constraints, and target return/volatility.
    Supports multiple optimization backends (CVXPY, SciPy, PyRB).

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
        tre_utility_weight: Penalty weight for tracking error in utility optimization.
        turnover_utility_weight: Penalty weight for turnover in utility optimization.
        group_lower_upper_constraints: Group-level allocation constraints.
        group_tracking_error_constraint: Group-level tracking error constraints.
        group_turnover_constraint: Group-level turnover constraints.
    """
    is_long_only: bool = True
    min_weights: pd.Series = None
    max_weights: pd.Series = None
    max_exposure: float = 1.0
    min_exposure: float = 1.0
    benchmark_weights: pd.Series = None
    tracking_err_vol_constraint: float = None
    weights_0: Optional[pd.Series] = None
    turnover_constraint: float = None
    turnover_costs: pd.Series = None
    target_return: float = None
    asset_returns: pd.Series = None
    max_target_portfolio_vol_an: float = None
    min_target_portfolio_vol_an: float = None
    tre_utility_weight: Optional[float] = 1.0  # penalty weight for tracking error in utility
    turnover_utility_weight: Optional[float] = 0.40  # penalty weight for turnover in utility
    group_lower_upper_constraints: GroupLowerUpperConstraints = None
    group_tracking_error_constraint: GroupTrackingErrorConstraint = None
    group_turnover_constraint: Optional[GroupTurnoverConstraint] = None

    def copy(self) -> Constraints:
        """Create a deep copy of all constraints.

        Returns:
            Independent copy of the Constraints object.
        """
        this = asdict(self).copy()
        if self.group_lower_upper_constraints is not None:
            gluc = self.group_lower_upper_constraints
            this['group_lower_upper_constraints'] = GroupLowerUpperConstraints(
                group_loadings=gluc.group_loadings,
                group_min_allocation=gluc.group_min_allocation,
                group_max_allocation=gluc.group_max_allocation
            )
        if self.group_tracking_error_constraint is not None:
            gluc = self.group_tracking_error_constraint
            this['group_tracking_error_constraint'] = GroupTrackingErrorConstraint(
                group_loadings=gluc.group_loadings,
                group_tre_vols=gluc.group_tre_vols,
                group_tre_utility_weights=gluc.group_tre_utility_weights
            )
        if self.group_turnover_constraint is not None:
            gluc = self.group_turnover_constraint
            this['group_turnover_constraint'] = GroupTurnoverConstraint(
                group_loadings=gluc.group_loadings,
                group_turnover_utility_weights=gluc.group_turnover_utility_weights,
                group_max_turnover=gluc.group_max_turnover
            )
        return Constraints(**this)

    def update(self, valid_tickers: List[str], **kwargs) -> Constraints:
        """Update constraints with valid tickers and additional parameters.

        Args:
            valid_tickers: List of tickers to retain in constraints.
            **kwargs: Additional constraint parameters to update.

        Returns:
            Updated Constraints object.
        """
        self_dict = asdict(self)
        self_dict.update(kwargs)
        if self.group_lower_upper_constraints is not None:
            group_lower_upper_constraints = self.group_lower_upper_constraints.update(
                valid_tickers=valid_tickers
            )
            self_dict['group_lower_upper_constraints'] = group_lower_upper_constraints
        if self.group_tracking_error_constraint is not None:
            self_dict['group_tracking_error_constraint'] = \
                self.group_tracking_error_constraint.update(valid_tickers=valid_tickers)
        if self.group_turnover_constraint is not None:
            self_dict['group_turnover_constraint'] = \
                self.group_turnover_constraint.update(valid_tickers=valid_tickers)
        return Constraints(**self_dict)

    def update_group_lower_upper_constraints(
            self,
            group_lower_upper_constraints: GroupLowerUpperConstraints,
            filling_value_for_missing_lower_bound: float = -10.0,
            filling_value_for_missing_upper_bound: float = 10.0
    ) -> Constraints:
        """Add or merge group lower/upper constraints.

        Args:
            group_lower_upper_constraints: New group constraints to add/merge.
            filling_value_for_missing_lower_bound: Default for missing min allocations.
            filling_value_for_missing_upper_bound: Default for missing max allocations.

        Returns:
            Constraints object with updated group constraints.
        """
        this = self.copy()
        if this.group_lower_upper_constraints is not None:
            this.group_lower_upper_constraints = merge_group_lower_upper_constraints(
                group_lower_upper_constraints1=this.group_lower_upper_constraints,
                group_lower_upper_constraints2=group_lower_upper_constraints,
                filling_value_for_missing_lower_bound=filling_value_for_missing_lower_bound,
                filling_value_for_missing_upper_bound=filling_value_for_missing_upper_bound)
        else:
            this.group_lower_upper_constraints = group_lower_upper_constraints
        return this

    def update_with_valid_tickers(
            self,
            valid_tickers: List[str],
            total_to_good_ratio: Optional[float] = None,
            weights_0: pd.Series = None,
            asset_returns: pd.Series = None,
            benchmark_weights: pd.Series = None,
            target_return: float = None,
            rebalancing_indicators: pd.Series = None
    ) -> Constraints:
        """Update constraints with valid tickers and rebalancing logic.

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
            Updated Constraints object.
        """
        this = self.copy()

        with pd.option_context('future.no_silent_downcasting', True):
            # Update individual weight constraints
            if this.min_weights is not None:
                this.min_weights = this.min_weights[valid_tickers].fillna(0.0)
            if this.max_weights is not None:
                if total_to_good_ratio is not None:
                    max_weight = this.max_weights[valid_tickers]
                    this.max_weights = max_weight.where(
                        np.isclose(max_weight, 1.0),
                        other=total_to_good_ratio * max_weight
                    ).fillna(0.0)
                else:
                    this.max_weights = this.max_weights[valid_tickers].fillna(0.0)

            # Update group constraints
            if this.group_lower_upper_constraints is not None:
                this.group_lower_upper_constraints = \
                    this.group_lower_upper_constraints.update(valid_tickers=valid_tickers)
            if this.group_tracking_error_constraint is not None:
                this.group_tracking_error_constraint = \
                    this.group_tracking_error_constraint.update(valid_tickers=valid_tickers)
            if this.group_turnover_constraint is not None:
                this.group_turnover_constraint = \
                    this.group_turnover_constraint.update(valid_tickers=valid_tickers)

            # Update turnover constraints with exposure scaling
            if this.turnover_constraint is not None:
                if total_to_good_ratio is not None:
                    this.turnover_constraint *= total_to_good_ratio
            if this.turnover_costs is not None:
                this.turnover_costs = this.turnover_costs.reindex(index=valid_tickers).fillna(1.0)

            # Update portfolio data
            if weights_0 is not None:
                this.weights_0 = weights_0.reindex(index=valid_tickers).fillna(0.0)
            if asset_returns is not None:
                this.asset_returns = asset_returns.reindex(index=valid_tickers).fillna(0.0)
            if benchmark_weights is not None:
                benchmark_weights_ = benchmark_weights.reindex(index=valid_tickers).fillna(0.0)
                this.benchmark_weights = benchmark_weights_
            if target_return is not None:
                this.target_return = target_return

            # Apply rebalancing indicators to freeze certain positions
            if rebalancing_indicators is not None and weights_0 is not None:
                rebalancing_indicators = rebalancing_indicators[this.weights_0.index].fillna(1.0)
                is_rebalanced = np.isclose(rebalancing_indicators, 1.0)
                if this.min_weights is not None:
                    this.min_weights = this.min_weights.where(is_rebalanced, other=weights_0)
                if this.max_weights is not None:
                    this.max_weights = this.max_weights.where(is_rebalanced, other=weights_0)
        return this


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
        # Long-only constraint
        if self.is_long_only:
            constraints += [w >= 0]

        # Exposure constraints (sum of weights)
        if exposure_scaler is None:
            if self.max_exposure == self.min_exposure:
                constraints += [cvx.sum(w) == self.max_exposure]
            else:
                constraints += [cvx.sum(w) <= self.max_exposure]
                constraints += [cvx.sum(w) >= self.min_exposure]
        else:  # for max sharpe optimization
            if self.max_exposure == self.min_exposure:
                constraints += [cvx.sum(w) == exposure_scaler]
            else:
                constraints += [cvx.sum(w) == exposure_scaler * self.max_exposure]

        # Individual weight constraints
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
            ValueError: If required data is missing for specified constraints.
        """
        # Start with exposure constraints
        constraints = self.set_cvx_exposure_constraints(w=w, exposure_scaler=exposure_scaler)

        # Target return constraint
        if self.target_return is not None:
            if self.asset_returns is None:
                raise ValueError("asset_returns must be given for target_return constraint")
            constraints += [self.asset_returns.to_numpy() @ w >= self.target_return]

        # Portfolio volatility constraints
        if self.max_target_portfolio_vol_an is not None:
            if covar is None:
                raise ValueError("covar must be given for portfolio volatility constraint")
            constraints += [cvx.quad_form(w, covar) <= self.max_target_portfolio_vol_an ** 2]
        if self.min_target_portfolio_vol_an is not None:
            if covar is None:
                raise ValueError("covar must be given for portfolio volatility constraint")
            constraints += [cvx.quad_form(w, covar) >= self.min_target_portfolio_vol_an ** 2]

        # Group turnover constraints (takes precedence over portfolio-level)
        if self.group_turnover_constraint is not None:
            constraints += self.group_turnover_constraint.set_group_turnover_constraints(w=w, weights_0=self.weights_0)

        # Portfolio-level turnover constraint
        elif self.turnover_constraint is not None:
            if self.weights_0 is None:
                warnings.warn("weights_0 must be given for turnover constraint")
            else:
                if self.turnover_costs is not None:
                    constraints += [cvx.norm(cvx.multiply(self.turnover_costs.to_numpy(), w - self.weights_0), 1)
                                    <= self.turnover_constraint]
                else:
                    assert w.size == len(self.weights_0.index)
                    constraints += [cvx.norm(w - self.weights_0, 1) <= self.turnover_constraint]

        # Group tracking error constraints (takes precedence over portfolio-level)
        if self.group_tracking_error_constraint is not None:
            constraints += self.group_tracking_error_constraint.set_cvx_group_tre_constraints(w=w,
                                                                                              benchmark_weights=self.benchmark_weights,
                                                                                              covar=covar)
        # Portfolio-level tracking error constraint
        elif self.tracking_err_vol_constraint is not None:
            if self.benchmark_weights is None:
                raise ValueError("benchmark_weights must be given for tracking error constraint")
            tracking_error_var = cvx.quad_form(w - self.benchmark_weights.to_numpy(), covar)
            constraints += [tracking_error_var <= self.tracking_err_vol_constraint ** 2]

        # Group allocation constraints
        if self.group_lower_upper_constraints is not None:
            constraints += self.group_lower_upper_constraints .set_cvx_group_lower_upper_constraints(w=w,
                                                                                                     exposure_scaler=exposure_scaler)

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
            ValueError: If required data is missing for specified penalties.
        """
        benchmark_weights: pd.Series = self.benchmark_weights

        # Initialize objective with alpha component
        if alphas is not None: # alphas are added
            objective_fun = alphas.T @ (w - benchmark_weights.to_numpy())
        else:
            objective_fun = None

        # Add group turnover penalty (takes precedence over portfolio-level)
        if self.group_turnover_constraint is not None:
            if self.weights_0 is None:
                warnings.warn("weights_0 must be given for group turnover constraint")
            else:
                term = self.group_turnover_constraint.set_cvx_group_turnover_utility(w=w, weights_0=self.weights_0)
                objective_fun = add_term_to_objective_function(objective_fun, term)

        # Add portfolio-level turnover penalty
        elif self.turnover_utility_weight is not None:
            if self.weights_0 is None:
                warnings.warn("weights_0 must be given for turnover constraint")
            else:
                if self.turnover_costs is not None:
                    term = -1.0*self.turnover_utility_weight * cvx.norm(cvx.multiply(self.turnover_costs.to_numpy(), w - self.weights_0), 1)
                else:
                    assert w.size == len(self.weights_0.index)
                    term = -1.0*self.turnover_utility_weight * cvx.norm(w - self.weights_0, 1)
                objective_fun = add_term_to_objective_function(objective_fun, term)

        # Add group tracking error penalty (takes precedence over portfolio-level)
        if self.group_tracking_error_constraint is not None:
            if self.benchmark_weights is None:
                raise ValueError("benchmark_weights must be given for group tracking error constraint")
            else:
                term = self.group_tracking_error_constraint.set_cvx_group_tre_utility(
                    w=w, benchmark_weights=benchmark_weights, covar=covar)
                objective_fun = add_term_to_objective_function(objective_fun, term)

        # Add portfolio-level tracking error penalty
        elif self.tre_utility_weight is not None:
            if benchmark_weights is None:
                raise ValueError("benchmark_weights must be given for tracking error constraint")
            term = -1.0*self.tre_utility_weight * cvx.quad_form(w - self.benchmark_weights.to_numpy(), covar)
            objective_fun = add_term_to_objective_function(objective_fun, term)

        # Generate hard constraints (exposure, bounds, groups)
        constraints = self.set_cvx_exposure_constraints(w=w, exposure_scaler=exposure_scaler)

        # Target return constraint
        if self.target_return is not None:
            if self.asset_returns is None:
                raise ValueError("asset_returns must be given for target_return constraint")
            constraints += [self.asset_returns.to_numpy() @ w >= self.target_return]

        # Group allocation constraints
        if self.group_lower_upper_constraints is not None:
            constraints += self.group_lower_upper_constraints.set_cvx_group_lower_upper_constraints(w=w,
                                                                                                     exposure_scaler=exposure_scaler)
        return objective_fun, constraints

    def set_scipy_bounds(self, covar: np.ndarray = None):
        """Set up bounds for scipy optimization.

        Converts weight constraints into (min, max) bounds for each asset.

        Args:
            covar: Covariance matrix (used to infer number of assets if needed).

        Returns:
            Array of (min, max) tuples for each asset.
        """
        if self.is_long_only and self.min_weights is None:
            n = covar.shape[0]
            bounds = np.array([(0.0, 1.0) for _ in range(n)])
        elif self.min_weights is not None and self.max_weights is not None:
            bounds = np.array([
                (x, y) for x, y in zip(
                    self.min_weights.to_numpy(),
                    self.max_weights.to_numpy()
                )
            ])
        elif self.min_weights is not None and self.max_weights is None:
            bounds = np.array([
                (x, y) for x, y in zip(
                    self.min_weights.to_numpy(),
                    [1.0] * len(self.min_weights.to_numpy())
                )
            ])
        elif self.min_weights is None and self.max_weights is not None:
            bounds = np.array([
                (x, y) for x, y in zip(
                    [0.0] * len(self.max_weights.to_numpy()),
                    self.max_weights.to_numpy())
            ])
        else:
            bounds = None
        return bounds

    def set_scipy_constraints(self, covar: np.ndarray = None) -> Tuple[List, np.ndarray]:
        """Generate SciPy-compatible constraints (inequality form: constraint >= 0).

        Converts constraints to format expected by scipy.optimize.minimize.

        Args:
            covar: Covariance matrix (used for bounds inference if needed).

        Returns:
            Tuple of (constraint dictionaries, bounds array).
        """
        constraints = []

        # Long-only constraint
        if self.is_long_only and self.min_weights is None:
            constraints += [{'type': 'ineq', 'fun': long_only_constraint}]

        # Exposure constraints (reformulated as inequality: max - sum(x) >= 0)
        constraints += [{'type': 'ineq', 'fun': lambda x: self.max_exposure - np.sum(x)}]
        constraints += [{'type': 'ineq', 'fun': lambda x: np.sum(x) - self.min_exposure}]

        # Group allocation constraints
        if self.group_lower_upper_constraints is not None:
            group_lower_upper_constraints = self.group_lower_upper_constraints
            for group in group_lower_upper_constraints.group_loadings.columns:
                group_loading = group_lower_upper_constraints.group_loadings[group].to_numpy()
                if np.any(np.isclose(group_loading, 0.0) == False):
                    if group_lower_upper_constraints.group_min_allocation is not None:
                        min_weight: float = group_lower_upper_constraints.group_min_allocation[group]
                        constraints += [{ 'type': 'ineq', 'fun': make_min_constraint(group_loading, min_weight)}]

                    if group_lower_upper_constraints.group_max_allocation is not None:
                        max_weight: float = group_lower_upper_constraints.group_max_allocation[group]
                        constraints += [{ 'type': 'ineq', 'fun': make_max_constraint(group_loading, max_weight)}]

        bounds = self.set_scipy_bounds(covar=covar)
        return constraints, bounds

    def set_pyrb_constraints(
            self,
            covar: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate PyRB-compatible constraints in matrix form (C*x <= d).

        Converts group constraints to matrix inequality form for risk budgeting solvers.

        Args:
            covar: Covariance matrix (used for bounds inference if needed).

        Returns:
            Tuple of (bounds array, constraint matrix C, constraint vector d).
        """
        # Set up bounds
        bounds = self.set_scipy_bounds(covar=covar)

        # Set up group constraints in matrix form (C*x <= d)
        if self.group_lower_upper_constraints is not None:
            group_lower_upper_constraints = self.group_lower_upper_constraints
            c_rows = []
            c_lhs = []

            for group in group_lower_upper_constraints.group_loadings.columns:
                group_loading = group_lower_upper_constraints.group_loadings[group].to_numpy()
                if np.any(np.isclose(group_loading, 0.0) == False):
                    # Minimum allocation: -group_loading * x <= -min_weight
                    if group_lower_upper_constraints.group_min_allocation is not None:
                        min_weight = group_lower_upper_constraints.group_min_allocation[group]
                        c_rows.append(-1.0 * group_loading)
                        c_lhs.append(-1.0 * min_weight)
                    # Maximum allocation: group_loading * x <= max_weight
                    if group_lower_upper_constraints.group_max_allocation is not None:
                        max_weight = group_lower_upper_constraints.group_max_allocation[group]
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
    """Total portfolio weight constraint: total - sum(x) = 0.

    Args:
        x: Portfolio weights.
        total: Target total exposure (default 1.0 for fully invested).

    Returns:
        Constraint value (should equal 0).
    """
    return total - np.sum(x)


def long_only_constraint(x: np.ndarray) -> np.ndarray:
    """Long-only constraint: x >= 0.

    Args:
        x: Portfolio weights.

    Returns:
        Constraint value (should be non-negative).
    """
    return x


def make_min_constraint( group_loading: np.ndarray, min_weight: float) -> Callable[[np.ndarray], float]:
    """Create minimum group allocation constraint: group_loading @ x >= min_weight.

    Args:
        group_loading: Binary group membership vector.
        min_weight: Minimum allocation to group.

    Returns:
        Constraint function for scipy optimizer.
    """
    return lambda x: group_loading @ x - min_weight


def make_max_constraint(group_loading: np.ndarray, max_weight: float) -> Callable[[np.ndarray], float]:
    """Create maximum group allocation constraint: group_loading @ x <= max_weight.

    Args:
        group_loading: Binary group membership vector.
        max_weight: Maximum allocation to group.

    Returns:
        Constraint function for scipy optimizer.
    """
    return lambda x: max_weight - group_loading @ x
