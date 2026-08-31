"""Portfolio optimization constraints for CVXPY, SciPy, and PyRB taa.

This module provides a comprehensive framework for defining and enforcing portfolio
constraints across multiple optimization backends. It supports individual asset
constraints, group-based constraints, tracking error limits, and turnover controls.
It also provides the current/model implementation corridor used to derive
eligible instrument bounds and rebalancing indicators for live proposals.

Compatible CVXPY risk expressions accept a precomputed
``CovarianceFactorization``. When supplied, variance is expressed as a squared
factor norm and hard upper-risk limits as second-order-cone constraints, so one
controlled covariance decomposition is reused throughout a solve.

All dataclass containers are immutable (frozen=True). Mutation methods return new instances.

Weights, budgets, and exposure loadings are dimensionless; volatility, tracking-error, and
turnover limits use the caller's units, with no resampling or annualisation in this module.
``Constraints`` and the group-constraint dataclasses are the main entry points. Boundary:
optimiser objectives, covariance estimation, and performance reporting are owned elsewhere.
"""
from __future__ import annotations, division
import copy as _copy
import pandas as pd
import numpy as np
import cvxpy as cvx
from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Union
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.constraints.nonpos import Inequality
from enum import Enum

from optimalportfolios.optimization.constraints.alignment import (
    align_nested_constraint_fields,
    build_valid_ticker_constraint_fields,
)
from optimalportfolios.optimization.constraints.analytics import (
    _construction_group_reachability_errors,
)
from optimalportfolios.optimization.constraints.backends import (
    set_cvx_all_constraints as _set_cvx_all_constraints,
    set_cvx_exposure_constraints as _set_cvx_exposure_constraints,
    set_cvx_utility_objective_constraints as _set_cvx_utility_objective_constraints,
    set_pyrb_constraints as _set_pyrb_constraints,
    set_scipy_bounds as _set_scipy_bounds,
    set_scipy_constraints as _set_scipy_constraints,
)
from optimalportfolios.optimization.constraints.benchmarks import (
    BenchmarkBetaConstraint,
    BenchmarkDeviationConstraints,
)
from optimalportfolios.optimization.constraints.groups import (
    GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
    merge_group_lower_upper_constraints,
)
from optimalportfolios.optimization.covar_factorization import CovarianceFactorization


class ConstraintEnforcementType(Enum):
    """Specification of tracking error and turnover constraint enforcement.

    Attributes:
        FORCED_CONSTRAINTS: Constraints are hard limits enforced by solver.
        UTILITY_CONSTRAINTS: Constraints are added as penalties to objective function.
    """
    FORCED_CONSTRAINTS = 1  # constraints are enforced for qp solver
    UTILITY_CONSTRAINTS = 2  # constraints are added as utility to the objective


@dataclass(frozen=True)
class Constraints:
    """Comprehensive portfolio optimization constraints.

    Unified container for all portfolio constraints including exposure limits,
    tracking error, turnover, group constraints, and target return/volatility.
    Supports multiple optimization backends (CVXPY, SciPy, PyRB).

    Sector and style deviations share ``BenchmarkDeviationConstraints`` and both impose
    ``|L_g.T @ (w - benchmark_weights)| <= d_g``. Sector loadings are normally binary
    membership indicators, while style loadings are normally continuous factor exposures.

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
        constraint_enforcement_type: How tracking error/turnover constraints are enforced.
        tre_utility_weight: Penalty weight for tracking error in utility optimization.
        turnover_utility_weight: Penalty weight for turnover in utility optimization.
        group_lower_upper_constraints: Group-level allocation constraints.
        group_tracking_error_constraint: Group-level tracking error constraints.
        group_turnover_constraint: Group-level turnover constraints.
        sector_deviation_constraints: Benchmark-relative limits using normally binary sector
            membership loadings; deviations are active sector weights.
        style_deviation_constraints: Benchmark-relative limits using normally continuous style
            loadings; deviation units follow the scaling of those loadings.
        benchmark_beta_constraint: Benchmark-relative beta range constraint.
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

        errors = _construction_group_reachability_errors(self, atol=1e-4)
        if errors:
            raise ValueError(
                f"Infeasible constraints detected ({len(errors)} violation(s)):\n"
                + "\n".join(f"  [{i + 1}] {e}" for i, e in enumerate(errors))
            )

    def copy(self, **overrides) -> Constraints:
        """Create a deep copy of all constraints, optionally overriding specific fields.

        Args:
            **overrides: Field names and new values to replace.

        Returns:
            New Constraints instance (deep-copied, then overridden).
        """
        return replace(_copy.deepcopy(self), **overrides)

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
        overrides = {}
        if min_weights is not None:
            if self.min_weights is not None:
                min_weights = min_weights.reindex(index=self.min_weights.index).fillna(0.0)
            overrides['min_weights'] = min_weights
        if max_weights is not None:
            if self.max_weights is not None:
                max_weights = max_weights.reindex(index=self.max_weights.index).fillna(0.0)
            overrides['max_weights'] = max_weights
        return replace(self, **overrides)

    def update(self, valid_tickers: List[str], **kwargs) -> Constraints:
        """Update constraints with valid tickers and additional parameters.

        Args:
            valid_tickers: List of tickers to retain in constraints.
            **kwargs: Additional constraint parameters to update.

        Returns:
            New Constraints object with updated fields.
        """
        overrides = dict(kwargs)
        overrides.update(align_nested_constraint_fields(
            constraint_spec=self,
            valid_tickers=valid_tickers,
        ))
        return replace(self, **overrides)

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
        if self.group_lower_upper_constraints is not None:
            group_constraints = merge_group_lower_upper_constraints(
                group_lower_upper_constraints1=self.group_lower_upper_constraints,
                group_lower_upper_constraints2=group_lower_upper_constraints)
        else:
            group_constraints = group_lower_upper_constraints
        return replace(self, group_lower_upper_constraints=group_constraints)

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
            max_relaxation_tol: Optional[float] = None,
            relax_frozen_group_bounds: bool = True,
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
            context: Rebalance label used in any constraint-relaxation logs.
            max_relaxation_tol: Optional maximum permitted relative relaxation
                when fixed-position constraints must be reconciled.
            relax_frozen_group_bounds: Whether frozen positions may widen group
                allocation bounds. Disable for execution-policy projection,
                where an infeasible selected trade set must remain visible.

        Returns:
            New Constraints object with all Series aligned to valid_tickers.
        """
        aligned_fields = build_valid_ticker_constraint_fields(
            constraint_spec=self,
            valid_tickers=valid_tickers,
            total_to_good_ratio=total_to_good_ratio,
            weights_0=weights_0,
            asset_returns=asset_returns,
            benchmark_weights=benchmark_weights,
            target_return=target_return,
            rebalancing_indicators=rebalancing_indicators,
            context=context,
            max_relaxation_tol=max_relaxation_tol,
            relax_frozen_group_bounds=relax_frozen_group_bounds,
        )
        return replace(self, **aligned_fields)

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
        return _set_cvx_exposure_constraints(
            constraint_spec=self,
            w=w,
            exposure_scaler=exposure_scaler,
        )

    def set_cvx_all_constraints(
            self,
            w: cvx.Variable,
            covar: Union[np.ndarray, psd_wrap] = None,
            exposure_scaler: cvx.Variable = None,
            covar_factorization: Optional[CovarianceFactorization] = None,
    ) -> List:
        """Generate all CVXPY constraints for portfolio optimization.

        Comprehensive constraint generation for mean-variance and related optimization problems.

        Args:
            w: Portfolio weight variable.
            covar: Covariance matrix (required for volatility/tracking error constraints).
            exposure_scaler: Optional exposure scaling for levered portfolios.
            covar_factorization: Optional precomputed covariance square root.
                When supplied, volatility and tracking-error upper bounds use
                norm constraints instead of quadratic forms.

        Returns:
            List of all CVXPY constraints.

        Raises:
            ValueError: If required universe is missing for specified constraints.
        """
        return _set_cvx_all_constraints(
            constraint_spec=self,
            w=w,
            covar=covar,
            exposure_scaler=exposure_scaler,
            covar_factorization=covar_factorization,
        )

    def set_cvx_utility_objective_constraints(
            self,
            w: cvx.Variable,
            alphas: Optional[np.ndarray] = None,
            covar: Union[np.ndarray, psd_wrap] = None,
            exposure_scaler: cvx.Variable = None,
            covar_factorization: Optional[CovarianceFactorization] = None,
    ) -> Tuple[AddExpression, List[Inequality]]:
        """Generate CVXPY utility objective with constraints added as utility penalties.

        Constructs objective function that combines alpha signals with soft penalties for
        tracking error and turnover, rather than enforcing them as hard constraints.

        Args:
            w: Portfolio weight variable.
            alphas: Expected excess returns (alpha signals).
            covar: Covariance matrix (required for tracking error penalties).
            exposure_scaler: Optional exposure scaling for levered portfolios.
            covar_factorization: Optional precomputed covariance square root.
                When supplied, tracking-error variance is expressed as a
                factorized sum of squares.

        Returns:
            Tuple of (objective function expression, list of hard constraints).

        Raises:
            ValueError: If required universe is missing for specified penalties.
        """
        return _set_cvx_utility_objective_constraints(
            constraint_spec=self,
            w=w,
            alphas=alphas,
            covar=covar,
            exposure_scaler=exposure_scaler,
            covar_factorization=covar_factorization,
        )

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
        return _set_scipy_bounds(constraint_spec=self, covar=covar)

    def set_scipy_constraints(self, covar: np.ndarray) -> Tuple[List, np.ndarray]:
        """Generate SciPy-compatible constraints (inequality form: constraint >= 0).

        Converts constraints to format expected by scipy.optimize.minimize.

        Args:
            covar: Covariance matrix (used for bounds inference if needed).

        Returns:
            Tuple of (constraint dictionaries, bounds array).
        """
        return _set_scipy_constraints(constraint_spec=self, covar=covar)

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
        return _set_pyrb_constraints(constraint_spec=self, covar=covar)

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
