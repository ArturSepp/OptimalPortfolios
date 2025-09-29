"""Portfolio optimization constraints implementation.

This module implements constraints as dataclass objects to support setting
various constraints for portfolio optimization using CVXPY, SciPy, and PyRB solvers.
"""
from __future__ import annotations, division
import warnings
import pandas as pd
import numpy as np
import cvxpy as cvx
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Union, Literal
from cvxpy.atoms.affine.wraps import psd_wrap


@dataclass
class GroupLowerUpperConstraints:
    """Group-based allocation constraints for portfolio optimization.
    This class implements constraints that enforce minimum and maximum allocation
    limits for predefined asset groups (e.g., sectors, regions, asset classes).

    Attributes:
        group_loadings (pd.DataFrame): Binary matrix where columns are instruments
            and index are groups. Data is 1 if instrument belongs to the indexed
            group, 0 otherwise.
        group_min_allocation (Optional[pd.Series]): Minimum allocation for each
            group. Index should match group_loadings columns.
        group_max_allocation (Optional[pd.Series]): Maximum allocation for each
            group. Index should match group_loadings columns.

    Raises:
        AssertionError: If group_min_allocation or group_max_allocation indices
            don't match group_loadings columns.
    """
    group_loadings: pd.DataFrame
    group_min_allocation: Optional[pd.Series]
    group_max_allocation: Optional[pd.Series]

    def __post_init__(self):
        """Validate that allocation series indices match group loadings columns."""
        if self.group_min_allocation is not None:
            this = self.group_min_allocation.index.isin(self.group_loadings.columns)
            if not this.all():
                print(f"{self.group_min_allocation.index} missing in\n{self.group_loadings.columns}")
            # temp fix to ensure that min allocation is zero for given group loadings
            self.group_min_allocation.reindex(index=self.group_loadings.columns).fillna(0)
        if self.group_max_allocation is not None:
            this = self.group_max_allocation.index.isin(self.group_loadings.columns)
            if not this.all():
                print(f"{self.group_max_allocation.index} missing in\n{self.group_loadings.columns}")
            # temp fix to ensure that max allocation is zero for given group loadings
            self.group_max_allocation.reindex(index=self.group_loadings.columns).fillna(0)

    def copy(self) -> GroupLowerUpperConstraints:
        """Create a copy.

        Returns:
            GroupLowerUpperConstraints: New instance with filtered group loadings.
        """
        new_self = GroupLowerUpperConstraints(
            group_loadings=self.group_loadings.copy(),
            group_min_allocation=self.group_min_allocation.copy() if self.group_min_allocation is not None else None,
            group_max_allocation=self.group_max_allocation.copy() if self.group_max_allocation is not None else None
        )
        return new_self

    def update(self, valid_tickers: List[str]) -> GroupLowerUpperConstraints:
        """Filter constraints to only include valid tickers.

        Args:
            valid_tickers (List[str]): List of valid instrument tickers to retain.

        Returns:
            GroupLowerUpperConstraints: New instance with filtered group loadings.
        """
        new_self = GroupLowerUpperConstraints(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_min_allocation=self.group_min_allocation,
            group_max_allocation=self.group_max_allocation
        )
        return new_self

    def drop_constraint(self, name: str) -> GroupLowerUpperConstraints:
        """Filter constraints to only include valid tickers.

        Args:
            valid_tickers (List[str]): List of valid instrument tickers to retain.

        Returns:
            GroupLowerUpperConstraints: New instance with filtered group loadings.
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

    def print(self):
        """Print constraint details for debugging and inspection."""
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_min_allocation:\n{self.group_min_allocation}")
        print(f"group_max_allocation:\n{self.group_max_allocation}")


def merge_group_lower_upper_constraints(
        group_lower_upper_constraints1: GroupLowerUpperConstraints,
        group_lower_upper_constraints2: GroupLowerUpperConstraints,
        filling_value_for_missing_lower_bound: float = -10.0,
        filling_value_for_missing_upper_bound: float = 10.0
) -> GroupLowerUpperConstraints:
    """Merge two GroupLowerUpperConstraints objects.
    Combines group loadings and allocation bounds from two constraint objects.
    Handles overlapping group names by renaming them with suffixes (_1, _2).

    Args:
        group_lower_upper_constraints1 (GroupLowerUpperConstraints): First constraint set.
        group_lower_upper_constraints2 (GroupLowerUpperConstraints): Second constraint set.
        filling_value_for_missing_lower_bound (float, optional): value used to
            fill missing bounds. Defaults to -10.
        filling_value_for_missing_upper_bound (float, optional): value used to
            fill missing bounds. Defaults to -10.
    Returns:
        GroupLowerUpperConstraints: Merged constraint object with combined groups
            and appropriately filled missing bounds.
    """
    # Check for overlapping column names and create rename mappings
    overlaps = list(set(group_lower_upper_constraints1.group_loadings.columns) &
                    set(group_lower_upper_constraints2.group_loadings.columns))

    if len(overlaps) > 0:
        overlaps1 = {x: f"{x}_1" for x in overlaps}
        overlaps2 = {x: f"{x}_2" for x in overlaps}
    else:
        overlaps1 = {}
        overlaps2 = {}

    # Merge group loadings
    group_loadings = pd.concat([
        group_lower_upper_constraints1.group_loadings.rename(overlaps1, axis=1),
        group_lower_upper_constraints2.group_loadings.rename(overlaps2, axis=1)
    ], axis=1)

    # Merge minimum allocations
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
        group_min_allocation = group_min_allocation.reindex(
            index=group_loadings.columns
        ).fillna(filling_value_for_missing_lower_bound)

    # Merge maximum allocations
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
    """Group-based tracking error constraints for portfolio optimization.
    Constrains the tracking error (deviation from benchmark) for specific asset
    groups to be below specified volatility thresholds.

    Attributes:
        group_loadings (pd.DataFrame): Binary matrix where columns are instruments
            and index are groups. Data is 1 if instrument belongs to group, 0 otherwise.
        group_tre_vols (pd.Series): Maximum tracking error volatility for each group.
            Index should match group_loadings columns.
    """
    group_loadings: pd.DataFrame
    group_tre_vols: pd.Series

    def update(self, valid_tickers: List[str]) -> GroupTrackingErrorConstraint:
        """Filter constraints to only include valid tickers.

        Args:
            valid_tickers (List[str]): List of valid instrument tickers to retain.

        Returns:
            GroupTrackingErrorConstraint: New instance with filtered group loadings.
        """
        new_self = GroupTrackingErrorConstraint(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_tre_vols=self.group_tre_vols
        )
        return new_self

    def set_group_tre_constraints(
            self,
            w: cvx.Variable,
            benchmark_weights: pd.Series,
            covar: np.ndarray
    ) -> List:
        """Generate CVXPY constraints for group tracking errors.
        Creates quadratic constraints that limit the variance of group-level
        deviations from benchmark weights.

        Args:
            w (cvx.Variable): Portfolio weight variable from CVXPY.
            benchmark_weights (pd.Series): Benchmark portfolio weights.
            covar (np.ndarray): Asset covariance matrix.

        Returns:
            List: List of CVXPY constraint objects.
        """
        constraints = []
        for group in self.group_loadings.columns:
            group_loading = self.group_loadings[group].copy()
            if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                # Align indices
                group_loading = group_loading.loc[benchmark_weights.index]
                tracking_error_var = cvx.quad_form(
                    cvx.multiply(group_loading.to_numpy(), w - benchmark_weights.to_numpy()),
                    covar
                )
                constraints += [tracking_error_var <= self.group_tre_vols.loc[group] ** 2]
        return constraints

    def print(self):
        """Print constraint details for debugging and inspection."""
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_tre_vols:\n{self.group_tre_vols}")


@dataclass
class GroupTurnoverConstraint:
    """Group-based turnover constraints for portfolio optimization.
    Constrains the turnover (sum of absolute weight changes) for specific asset
    groups to be below specified thresholds.

    Attributes:
        group_loadings (pd.DataFrame): Binary matrix where columns are instruments
            and index are groups. Data is 1 if instrument belongs to group, 0 otherwise.
        group_max_turnover (pd.Series): Maximum turnover allowed for each group.
            Index should match group_loadings columns.
    """
    group_loadings: pd.DataFrame
    group_max_turnover: pd.Series

    def update(self, valid_tickers: List[str]) -> GroupTurnoverConstraint:
        """Filter constraints to only include valid tickers.

        Args:
            valid_tickers (List[str]): List of valid instrument tickers to retain.

        Returns:
            GroupTurnoverConstraint: New instance with filtered group loadings.
        """
        new_self = GroupTurnoverConstraint(
            group_loadings=self.group_loadings.loc[valid_tickers, :],
            group_max_turnover=self.group_max_turnover
        )
        return new_self

    def set_group_turnover_constraints(
            self,
            w: cvx.Variable,
            weights_0: pd.Series = None
    ) -> List:
        """Generate CVXPY constraints for group turnovers.
        Creates L1-norm constraints that limit the sum of absolute weight changes
        within each group.

        Args:
            w (cvx.Variable): Portfolio weight variable from CVXPY.
            weights_0 (pd.Series, optional): Current portfolio weights for
                turnover calculation.

        Returns:
            List: List of CVXPY constraint objects.

        Note:
            Prints warning if weights_0 is None as it's required for turnover constraints.
        """
        constraints = []
        if weights_0 is None:
            print(f"weights_0 must be given for turnover_constraint")
        else:
            for group in self.group_loadings.columns:
                group_loading = self.group_loadings[group].copy()
                if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                    group_loading = group_loading.loc[weights_0.index]  # align indices
                    constraints += [
                        cvx.norm(
                            cvx.multiply(group_loading.to_numpy(), w - weights_0), 1
                        ) <= self.group_max_turnover.loc[group]
                    ]
        return constraints

    def print(self):
        """Print constraint details for debugging and inspection."""
        print(f"group_loadings:\n{self.group_loadings}")
        print(f"group_max_turnover:\n{self.group_max_turnover}")


@dataclass
class Constraints:
    """Comprehensive portfolio optimization constraints container.
    This class consolidates all types of constraints that can be applied to
    portfolio optimization problems, including individual asset constraints,
    portfolio-level constraints, and group-based constraints.

    Attributes:
        is_long_only (bool): If True, enforces non-negative weights. Defaults to True.
        min_weights (pd.Series, optional): Minimum weight for each asset.
        max_weights (pd.Series, optional): Maximum weight for each asset.
        max_exposure (float): Maximum total portfolio exposure. Defaults to 1.0.
        min_exposure (float): Minimum total portfolio exposure. Defaults to 1.0.
        benchmark_weights (pd.Series, optional): Benchmark portfolio weights for
            tracking error calculations.
        tracking_err_vol_constraint (float, optional): Maximum annualized tracking
            error volatility.
        weights_0 (pd.Series, optional): Current portfolio weights for turnover
            calculations.
        turnover_constraint (float, optional): Maximum portfolio turnover allowed.
        turnover_costs (pd.Series, optional): Transaction costs for turnover
            calculations.
        target_return (float, optional): Target portfolio return for optimization.
        asset_returns (pd.Series, optional): Expected asset returns.
        max_target_portfolio_vol_an (float, optional): Maximum annualized portfolio
            volatility target.
        min_target_portfolio_vol_an (float, optional): Minimum annualized portfolio
            volatility target.
        group_lower_upper_constraints (GroupLowerUpperConstraints, optional):
            Group allocation constraints.
        group_tracking_error_constraint (GroupTrackingErrorConstraint, optional):
            Group tracking error constraints.
        group_turnover_constraint (GroupTurnoverConstraint, optional): Group
            turnover constraints.
        apply_total_to_good_ratio_for_constraints (bool): Whether to apply total-to-good
            ratio scaling to constraints. Defaults to True.
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
    group_lower_upper_constraints: GroupLowerUpperConstraints = None
    group_tracking_error_constraint: GroupTrackingErrorConstraint = None
    group_turnover_constraint: Optional[GroupTurnoverConstraint] = None
    apply_total_to_good_ratio_for_constraints: bool = True

    def copy(self) -> Constraints:
        """Create a deep copy of the constraints object.

        Returns:
            Constraints: New instance with copied attributes and constraint objects.
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
                group_tre_vols=gluc.group_tre_vols
            )
        if self.group_turnover_constraint is not None:
            gluc = self.group_turnover_constraint
            this['group_turnover_constraint'] = GroupTurnoverConstraint(
                group_loadings=gluc.group_loadings,
                group_max_turnover=gluc.group_max_turnover
            )
        return Constraints(**this)

    def update(self, valid_tickers: List[str], **kwargs) -> Constraints:
        """Update constraints with new valid tickers and additional parameters.

        Args:
            valid_tickers (List[str]): List of valid tickers to retain.
            **kwargs: Additional constraint parameters to update.

        Returns:
            Constraints: New constraints instance with updated parameters.
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
            group_lower_upper_constraints (GroupLowerUpperConstraints): New group
                constraints to add or merge.
            filling_value_for_missing_lower_bound
            filling_value_for_missing_upper_bound
        Returns:
            Constraints: New instance with updated group constraints.
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
            total_to_good_ratio: float = 1.0,
            weights_0: pd.Series = None,
            asset_returns: pd.Series = None,
            benchmark_weights: pd.Series = None,
            target_return: float = None,
            rebalancing_indicators: pd.Series = None,
            apply_total_to_good_ratio: bool = True
    ) -> Constraints:
        """Comprehensive update of constraints with valid tickers and rebalancing logic.
        Updates all constraint components to only include valid tickers and applies
        rebalancing logic where assets with rebalancing_indicators == 0 have their
        min/max weights fixed to current weights.

        Args:
            valid_tickers (List[str]): List of valid tickers to retain.
            total_to_good_ratio (float): Scaling factor for constraint adjustment.
                Defaults to 1.0.
            weights_0 (pd.Series, optional): Current portfolio weights.
            asset_returns (pd.Series, optional): Expected asset returns.
            benchmark_weights (pd.Series, optional): Benchmark weights.
            target_return (float, optional): Target portfolio return.
            rebalancing_indicators (pd.Series, optional): Binary indicators where
                0 means don't rebalance (fix weights), 1 means rebalance normally.
            apply_total_to_good_ratio (bool): Whether to apply ratio scaling.
                Defaults to True.

        Returns:
            Constraints: New constraints instance with updated valid tickers.
        """
        this = self.copy()

        with pd.option_context('future.no_silent_downcasting', True):
            # Update individual weight constraints
            if this.min_weights is not None:
                this.min_weights = this.min_weights[valid_tickers].fillna(0.0)
            if this.max_weights is not None:
                if apply_total_to_good_ratio and self.apply_total_to_good_ratio_for_constraints:
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

            # Update turnover constraints
            if this.turnover_constraint is not None:
                if apply_total_to_good_ratio and self.apply_total_to_good_ratio_for_constraints:
                    this.turnover_constraint *= total_to_good_ratio
            if this.turnover_costs is not None:
                this.turnover_costs = this.turnover_costs.reindex(
                    index=valid_tickers
                ).fillna(1.0)

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

            # Apply rebalancing indicators
            if rebalancing_indicators is not None and weights_0 is not None:
                rebalancing_indicators = rebalancing_indicators[this.weights_0.index].fillna(1.0)
                is_rebalanced = np.isclose(rebalancing_indicators, 1.0)
                if this.min_weights is not None:
                    this.min_weights = this.min_weights.where(is_rebalanced, other=weights_0)
                if this.max_weights is not None:
                    this.max_weights = this.max_weights.where(is_rebalanced, other=weights_0)
        return this

    def set_cvx_constraints(
            self,
            w: cvx.Variable,
            covar: Union[np.ndarray, psd_wrap] = None,
            exposure_scaler: cvx.Variable = None
    ) -> List:
        """Generate CVXPY constraints for portfolio optimization.
        Converts all constraint specifications into CVXPY constraint objects
        that can be used in convex optimization problems.

        Args:
            w (cvx.Variable): Portfolio weight variable from CVXPY.
            covar (Union[np.ndarray, psd_wrap], optional): Asset covariance matrix
                for volatility and tracking error constraints.
            exposure_scaler (cvx.Variable, optional): Scaling variable for maximum
                Sharpe ratio optimization.

        Returns:
            List: List of CVXPY constraint objects.

        Raises:
            ValueError: If required data (asset_returns, benchmark_weights, covar)
                is missing for specific constraint types.
        """
        constraints = []

        # Long-only constraint
        if self.is_long_only:
            constraints += [w >= 0]

        # Exposure constraints
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

        # Group turnover constraints
        if self.group_turnover_constraint is not None:
            if self.weights_0 is None:
                warnings.warn("weights_0 must be given for group turnover constraint")
            else:
                for group in self.group_turnover_constraint.group_loadings.columns:
                    group_loading = self.group_turnover_constraint.group_loadings[group].copy()
                    if np.any(np.isclose(group_loading, 0.0) == False):
                        group_loading = group_loading.loc[self.weights_0.index]
                        constraints += [
                            cvx.norm(
                                cvx.multiply(group_loading.to_numpy(), w - self.weights_0), 1
                            ) <= self.group_turnover_constraint.group_max_turnover.loc[group]
                        ]
        # Portfolio-level turnover constraint
        elif self.turnover_constraint is not None:
            if self.weights_0 is None:
                print("weights_0 must be given for turnover constraint")
            else:
                if self.turnover_costs is not None:
                    constraints += [
                        cvx.norm(
                            cvx.multiply(self.turnover_costs.to_numpy(), w - self.weights_0), 1
                        ) <= self.turnover_constraint
                    ]
                else:
                    assert w.size == len(self.weights_0.index)
                    constraints += [cvx.norm(w - self.weights_0, 1) <= self.turnover_constraint]

        # Group tracking error constraints
        if self.group_tracking_error_constraint is not None:
            if self.benchmark_weights is None:
                raise ValueError("benchmark_weights must be given for group tracking error constraint")
            for group in self.group_tracking_error_constraint.group_loadings.columns:
                group_loading = self.group_tracking_error_constraint.group_loadings[group].copy()
                if np.any(np.isclose(group_loading, 0.0) == False):
                    group_loading = group_loading.loc[self.benchmark_weights.index]
                    tracking_error_var = cvx.quad_form(
                        cvx.multiply(group_loading.to_numpy(),
                                     w - self.benchmark_weights.to_numpy()),
                        covar
                    )
                    constraints += [
                        tracking_error_var <=
                        self.group_tracking_error_constraint.group_tre_vols[group] ** 2
                    ]
        # Portfolio-level tracking error constraint
        elif self.tracking_err_vol_constraint is not None:
            if self.benchmark_weights is None:
                raise ValueError("benchmark_weights must be given for tracking error constraint")
            tracking_error_var = cvx.quad_form(w - self.benchmark_weights.to_numpy(), covar)
            constraints += [tracking_error_var <= self.tracking_err_vol_constraint ** 2]

        # Group allocation constraints
        if exposure_scaler is None:
            multiplier = 1.0
        else:
            multiplier = exposure_scaler
        if self.group_lower_upper_constraints is not None:
            group_lower_upper_constraints = self.group_lower_upper_constraints
            for group in group_lower_upper_constraints.group_loadings.columns:
                group_loading = group_lower_upper_constraints.group_loadings[group].to_numpy()
                if np.any(np.isclose(group_loading, 0.0) == False):
                    if group_lower_upper_constraints.group_min_allocation is not None:
                        if group in group_lower_upper_constraints.group_min_allocation:
                            this = group_lower_upper_constraints.group_min_allocation[group]
                            if this is not None:
                                constraints += [ group_loading @ w >= multiplier * this]
                        else:
                            warnings.warn(f"no group={group} in group_min_allocation, constraint skipped")
                    if group_lower_upper_constraints.group_max_allocation is not None:
                        if group in group_lower_upper_constraints.group_max_allocation:
                            this = group_lower_upper_constraints.group_max_allocation[group]
                            if this is not None:
                                constraints += [ group_loading @ w <= multiplier * this ]
                        else:
                            warnings.warn(f"no group={group} in group_max_allocation, constraint skipped")

        return constraints

    def set_scipy_constraints(self, covar: np.ndarray = None) -> List:
        """Generate SciPy-compatible constraints for portfolio optimization.
        Converts constraint specifications into SciPy constraint dictionaries
        that can be used with scipy.optimize.minimize.

        Args:
            covar (np.ndarray, optional): Asset covariance matrix for volatility
                constraints (currently not implemented).

        Returns:
            List: List of constraint dictionaries in SciPy format.

        Note:
            SciPy constraints use inequality form where constraint function >= 0.
        """
        constraints = []

        # Long-only constraint
        if self.is_long_only and self.min_weights is None:
            constraints += [{'type': 'ineq', 'fun': long_only_constraint}]

        # Exposure constraints
        constraints += [{'type': 'ineq', 'fun': lambda x: self.max_exposure - np.sum(x)}]
        constraints += [{'type': 'ineq', 'fun': lambda x: np.sum(x) - self.min_exposure}]

        # Individual weight constraints
        if self.min_weights is not None:
            min_weights = (self.min_weights.to_numpy()
                           if isinstance(self.min_weights, pd.Series)
                           else self.min_weights)
            constraints += [{'type': 'ineq', 'fun': lambda x: x - min_weights}]

        if self.max_weights is not None:
            max_weights = (self.max_weights.to_numpy()
                           if isinstance(self.max_weights, pd.Series)
                           else self.max_weights)
            constraints += [{'type': 'ineq', 'fun': lambda x: max_weights - x}]

        # Group allocation constraints
        if self.group_lower_upper_constraints is not None:
            group_lower_upper_constraints = self.group_lower_upper_constraints
            for group in group_lower_upper_constraints.group_loadings.columns:
                group_loading = group_lower_upper_constraints.group_loadings[group].to_numpy()
                if np.any(np.isclose(group_loading, 0.0) == False):
                    if group_lower_upper_constraints.group_min_allocation is not None:
                        min_weight = group_lower_upper_constraints.group_min_allocation[group]
                        constraints += [{'type': 'ineq', 'fun': lambda x: group_loading @ x - min_weight}]
                    if group_lower_upper_constraints.group_max_allocation is not None:
                        max_weight = group_lower_upper_constraints.group_max_allocation[group]
                        constraints += [{'type': 'ineq', 'fun': lambda x: max_weight - group_loading @ x}]

        return constraints

    def set_pyrb_constraints(
            self,
            covar: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate PyRB-compatible constraints for portfolio optimization.
        Converts constraint specifications into bounds and inequality matrices
        compatible with PyRB (Python Risk Budgeting) optimization.

        Args:
            covar (np.ndarray, optional): Asset covariance matrix for determining
                problem size.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - bounds: Array of (min, max) tuples for each asset
                - c_rows: Constraint matrix (C in C*x <= d)
                - c_lhs: Right-hand side vector (d in C*x <= d)

        Note:
            PyRB uses matrix form constraints where C*x <= d.
        """
        # Set up bounds
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
        else:
            bounds = None

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


def total_weight_constraint(x, total: float = 1.0):
    """Constraint function for total portfolio weight.

    Args:
        x (np.ndarray): Portfolio weights.
        total (float): Target total weight. Defaults to 1.0.

    Returns:
        float: Difference between target and actual total weight.

    Note:
        Returns total - sum(x), which equals 0 when constraint is satisfied.
    """
    return total - np.sum(x)


def long_only_constraint(x):
    """Constraint function for long-only portfolio weights.

    Args:
        x (np.ndarray): Portfolio weights.

    Returns:
        np.ndarray: Portfolio weights (constraint satisfied when all >= 0).

    Note:
        For SciPy optimization where constraint function should be >= 0.
    """
    return x
