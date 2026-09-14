"""Portfolio policy specifications and constraint-compiler entry points.

Constraints stores policy and delegates universe alignment and backend
compilation to their owning modules. CVXPY supports the full constraint
families; SciPy and the PyRB-compatible matrix format support subsets.
The matrix helper supplies inputs to the risk-budgeting solver, not a solve.

Dataclasses are frozen, while their contained pandas objects remain mutable.
Update methods return replacement instances; copy() deep-copies existing
state before applying caller-supplied overrides.

Weights and exposure loadings are dimensionless. Risk limits use the square
root of the supplied covariance units, and turnover uses weight-change units
or their configured cost scaling. No method resamples or annualizes inputs.

The complete mathematical, backend and alignment contract is in
``docs/constraints.md`` in the source checkout. Portfolio objectives,
covariance estimation and generic performance reporting are owned elsewhere.
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
    """Select the constraint policy used by supported solver wrappers and diagnostics.

    The enum describes policy; it does not dispatch a low-level compiler.
    Calling set_cvx_all_constraints() directly always builds hard rows.
    Utility compilation retains hard mandate rows and uses supported risk/trading
    terms in the objective; backend coverage still applies.

    Attributes:
        FORCED_CONSTRAINTS: Enforce configured limits as hard solver rows where supported.
        UTILITY_CONSTRAINTS: Treat turnover, tracking-error and maximum-volatility
            diagnostics as soft. Generic utility compilation adds risk/trading
            penalties but no maximum-volatility cap; solver objectives may differ.
    """
    FORCED_CONSTRAINTS = 1  # constraints are enforced for qp solver
    UTILITY_CONSTRAINTS = 2  # constraints are added as utility to the objective


@dataclass(frozen=True)
class Constraints:
    """Portfolio policy in an ordered asset universe.

    Backend compilers enforce their supported families; a populated field alone
    does not establish backend coverage. SciPy compiles boxes, net exposure and
    group allocation. The PyRB-compatible helper compiles boxes and group rows;
    the risk-budgeting solver owns its full-investment contract.

    Sector and style deviations share BenchmarkDeviationConstraints. Sector
    loadings are normally binary membership indicators; style loadings are
    normally continuous exposures. Their units follow the loading scale.

    The dataclass is frozen, but contained pandas objects are mutable. Treat them
    as policy inputs and use copy() or an update method to create replacements.
    Align all vectors and both covariance axes before compiling or evaluating.

    Attributes:
        is_long_only: Require nonnegative weights where supported.
        min_weights: Per-asset lower bounds in the ordered universe.
        max_weights: Per-asset upper bounds in the ordered universe.
        max_exposure: Upper bound on the sum of weights, not gross absolute exposure.
        min_exposure: Lower bound on the sum of weights; exact equality with
            max_exposure selects a single CVXPY exposure equality.
        benchmark_weights: Benchmark weights for active risk and deviations.
        tracking_err_vol_constraint: Total tracking-error limit in covariance-root units.
        weights_0: Current implemented weights for trading and freezing.
        turnover_constraint: Total L1 weight-change limit; no half-turnover factor.
        turnover_costs: Per-asset scaling inside total L1 turnover; units must match
            turnover_constraint. Group turnover uses its own loadings.
        target_return: Minimum portfolio expected return where supported.
        asset_returns: Per-asset expected returns in the same units as target_return.
        max_target_portfolio_vol_an: Portfolio-volatility limit. It is annual only
            when the supplied covariance is annualized; no conversion is applied.
        constraint_enforcement_type: Wrapper/diagnostic policy; select the matching
            low-level compiler when compiling directly.
        tre_utility_weight: Total tracking-error penalty coefficient; an existing
            group-TE object takes precedence in generic utility compilation.
        turnover_utility_weight: Total turnover penalty coefficient; an existing
            group-turnover object takes precedence in generic utility compilation.
        group_lower_upper_constraints: Absolute group allocation limits.
        group_tracking_error_constraint: Group tracking-error limits.
        group_turnover_constraint: Group L1 turnover limits.
        sector_deviation_constraints: Benchmark-relative limits on sector loadings.
        style_deviation_constraints: Benchmark-relative limits on style loadings.
        benchmark_beta_constraint: Benchmark-beta range with supplied beta loadings.
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
        """Check selected instrument-box and group-reachability conditions.

        Per-name minimum/maximum consistency is checked when both Series have exactly
        matching indexes. Long-only minima below -1e-10 are rejected. Group checks use
        positive loadings and a 1e-4 tolerance; they do not establish feasibility of
        every signed-loading, exposure, risk or trading combination.

        Raises:
            ValueError: If a checked box or group-reachability condition fails.
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
        """Deep-copy existing policy state, then apply overrides.

        Args:
            **overrides: Fields to replace after the deep copy. Supplied replacement
                objects are not themselves deep-copied by this method.

        Returns:
            A new Constraints instance with constructor validation applied. This
            operation does not align vectors or nested blocks to a new universe.
        """
        return replace(_copy.deepcopy(self), **overrides)

    def update_min_max_weights(
            self,
            min_weights: Optional[pd.Series] = None,
            max_weights: Optional[pd.Series] = None,
    ) -> Constraints:
        """Replace supplied box sides, retaining other policy fields.

        Args:
            min_weights: New lower side; None retains the existing side. If a lower
                side already exists, reindex to it and replace missing/NaN values by zero.
            max_weights: New upper side; None retains the existing side. If an upper
                side already exists, reindex to it and replace missing/NaN values by zero.

        Returns:
            A replacement Constraints instance. This is not a full-universe alignment
            or a deep copy of unchanged pandas fields.
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
        """Align registered nested blocks and apply other field overrides.

        Flat Series such as min_weights, weights_0 and benchmark_weights are not
        automatically reindexed here. Use update_with_valid_tickers() for the full
        alignment and rebalancing path.

        Args:
            valid_tickers: Ordered universe passed to each existing nested block.
            **kwargs: Replacement fields. Aligned nested blocks from the original
                specification take precedence over overrides of those same fields.

        Returns:
            A replacement Constraints instance with constructor validation applied.
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
        """Align policy inputs and apply the configured freezing and waiver rules.

        Flat Series and registered nested blocks follow valid_tickers order. Inserted
        labels receive field-specific defaults; existing explicit NaNs generally
        survive reindexing. The caller must supply data available at the decision date
        and align the covariance and candidate separately.

        A rebalancing indicator not numerically close to one freezes each configured
        box side at the resolved current weight. Both sides must exist for an exact
        pin. Long-only frozen bounds clip negative weights to zero. Optional group
        waivers reconcile mismatches introduced by freezing and are logged; they do
        not certify that the resulting mandate is feasible.

        Args:
            valid_tickers: Ordered solver universe.
            total_to_good_ratio: Optional multiplier for total turnover and per-name
                maxima, except maxima close to 1.0. Exposure limits, minima and group
                bounds are not scaled.
            weights_0: Current weights; None retains and aligns the existing field.
            asset_returns: Expected returns; None retains and aligns the existing field.
            benchmark_weights: Benchmark weights; None retains and aligns the existing field.
            target_return: Replacement minimum return; None retains the existing value.
            rebalancing_indicators: Values close to one permit trading; other values
                freeze configured box sides when current weights are available.
                Missing labels are treated as tradable.
            context: Rebalance label attached to any relaxation logs.
            max_relaxation_tol: Optional absolute single-group-bound change threshold
                for ERROR logging. It does not cap, reject or undo a waiver.
            relax_frozen_group_bounds: Whether to reconcile eligible group-bound
                mismatches introduced by freezing. False keeps the original group policy.

        Returns:
            A replacement Constraints instance with aligned fields and constructor
            validation applied. The original specification is not updated in place.
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
        """Compile CVXPY long-only, net-exposure and instrument-box rows.

        Exactly equal stored exposure limits produce an equality; otherwise both
        sides are compiled. Inputs must already share the solver's asset order.

        Args:
            w: Ordered portfolio-weight variable.
            exposure_scaler: Optional multiplier of exposure and box limits.

        Returns:
            A list of CVXPY constraints, including an equality when appropriate.
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
        """Compile configured hard CVXPY rows without solving.

        This method does not switch to utility compilation when the enforcement
        enum changes. Group and total turnover/TE limits are independent hard rows.
        Required analytical inputs must be supplied; total turnover is omitted when
        weights_0 is absent.

        Args:
            w: Portfolio variable in aligned constraint order.
            covar: Ordered covariance for configured volatility and tracking-error
                rows when no factorization is supplied.
            exposure_scaler: Optional scaling for exposure, boxes and group allocation;
                it does not uniformly scale every policy family.
            covar_factorization: Optional existing solver factorization. Its stabilized
                covariance takes precedence and upper-risk rows use factor norms.

        Returns:
            A constraint list to combine with a caller-owned CVXPY objective.

        Raises:
            ValueError: If a delegated compiler rejects missing required inputs.
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
        """Build the generic utility expression and remaining hard CVXPY rows.

        Configured group risk/trading penalties take precedence over their total
        counterparts. Exposure, boxes, target return, group allocation, benchmark
        deviations and beta remain hard where configured. This generic method adds
        no maximum-volatility cap. It neither solves nor dispatches on the enum.

        Args:
            w: Portfolio variable in aligned constraint order.
            alphas: Optional ordered alpha vector. Its active-return term requires
                benchmark_weights.
            covar: Ordered covariance for tracking-error penalties when no
                factorization is supplied.
            exposure_scaler: Optional multiplier for supported exposure, box and
                group-allocation rows, not a uniform policy rescaling.
            covar_factorization: Optional existing factorization whose stabilized
                covariance is used for factorized risk penalties.

        Returns:
            Utility expression and hard constraint list. Maximize the expression or
            combine it with the selected solver's objective. The expression can be
            None when no alpha term or applicable penalty is constructed.

        Raises:
            ValueError: If a delegated compiler rejects missing required inputs.
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
        """Compile supported SciPy callbacks and bounds without solving.

        The callbacks use nonnegative feasibility values and cover net exposure
        and group allocation, plus long-only when no explicit minimum is supplied.
        Boxes use set_scipy_bounds(). Other policy families are not compiled.
        An exact exposure target remains two opposing inequalities.

        Args:
            covar: Ordered covariance used to infer the asset count for bounds.

        Returns:
            Constraint-dictionary list and bounds array, or None for unbounded boxes.
        """
        return _set_scipy_constraints(constraint_spec=self, covar=covar)

    def set_pyrb_constraints(
            self,
            covar: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compile boxes and group rows in the PyRB-compatible matrix format.

        The returned group matrix uses C*x <= d. Net-exposure bands, risk, return,
        trading and benchmark-deviation limits are not compiled here. Full investment
        is owned by the risk-budgeting solver and its validator.

        Args:
            covar: Ordered covariance used to infer the asset count for bounds.

        Returns:
            Bounds, group matrix C and right-hand-side vector d. Bounds can be None;
            C and d are both None when there are no applicable group rows. The method
            does not run a solver.
        """
        return _set_pyrb_constraints(constraint_spec=self, covar=covar)

    def print_constraints(
            self,
            constraints_list:  List[Inequality],
    ) -> None:
        """Print CVXPY row representations, types and shapes for inspection.

        Args:
            constraints_list: Compiled CVXPY rows, including any exposure equality.

        Returns:
            None. Diagnostic text is written to standard output.
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
        """Print maximum CVXPY row violations at the current variable values.

        This diagnostic reports numerical violations, not mandate acceptance or a
        list of binding constraints. Use evaluate_constraint_residuals() and the
        solver outcome for structured policy diagnostics.

        Args:
            constraints_list: Compiled rows whose variables have values, normally
                after solving.

        Returns:
            None. Diagnostic text is written to standard output.
        """
        print("=== Check the Violations of CVXPY constraints ===")
        for i, c in enumerate(constraints_list):
            v = c.violation()   # numpy array of nonnegative violations
            max_v = v.max() if v.size > 0 else 0.0
            print(f"Constraint {i}: max violation = {max_v}")
