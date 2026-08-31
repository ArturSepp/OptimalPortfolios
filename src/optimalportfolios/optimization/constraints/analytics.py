"""Pure evaluation of portfolio constraints against candidate weights.

This module owns solver-independent residual records and their evaluation. It
does not compile constraints, invoke solvers, mutate specifications, or emit
logs. Weights and exposures are dimensionless; covariance inputs retain the
caller's variance units without resampling or annualisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from optimalportfolios.optimization.constraints.core import Constraints
    from optimalportfolios.optimization.covar_factorization import CovarianceFactorization


_INDEXED_CONSTRAINT_FIELDS = (
    (None, "benchmark_weights"),
    (None, "max_weights"),
    (None, "min_weights"),
    (None, "weights_0"),
    (None, "asset_returns"),
    (None, "turnover_costs"),
    ("group_lower_upper_constraints", "group_loadings"),
    ("group_tracking_error_constraint", "group_loadings"),
    ("group_turnover_constraint", "group_loadings"),
    ("sector_deviation_constraints", "factor_loading_mat"),
    ("style_deviation_constraints", "factor_loading_mat"),
    ("benchmark_beta_constraint", "beta_loadings"),
)


@dataclass(frozen=True)
class _ExposureFacts:
    """Literal exposure limits and their equality policy."""

    minimum: float
    maximum: float
    is_exact_equality: bool


@dataclass(frozen=True)
class _BudgetBoxResiduals:
    """Raw budget and box quantities shared by validation and telemetry."""

    exposure: _ExposureFacts
    sum_weights: float
    budget_residual: float
    minimum_weight: float
    long_only_violation: float
    minimum_weights: Optional[np.ndarray]
    maximum_weights: Optional[np.ndarray]
    minimum_difference: Optional[float]
    minimum_position: Optional[int]
    maximum_overage: Optional[float]
    maximum_position: Optional[int]
    max_box_violation: float


@dataclass(frozen=True)
class _FiniteGroupBound:
    """One loadings column and its stated, non-missing allocation bounds."""

    name: Any
    loadings: pd.Series
    lower: Optional[float]
    upper: Optional[float]


@dataclass(frozen=True)
class _GroupReachability:
    """Reachable endpoints and bound failures for one group loading vector."""

    minimum: float
    maximum: float
    lower_unreachable: bool
    upper_unreachable: bool


@dataclass(frozen=True)
class _StaticReachabilityFinding:
    """One pure pre-solve constraint finding and its diagnostic classification."""

    kind: str
    message: str
    code: str
    name: Optional[str] = None
    position: Optional[int] = None


def _resolve_asset_index(
        constraints: Constraints,
        n_assets: Optional[int] = None,
) -> Optional[pd.Index]:
    """Resolve the first indexed constraint field in canonical solver order.

    Args:
        constraints: Constraint specification carrying optional indexed fields.
        n_assets: Optional fallback size when no field carries an index.

    Returns:
        Ordered asset index, a numeric fallback index, or ``None`` when neither
        an indexed field nor a fallback size is available.
    """
    for container_name, indexed_name in _INDEXED_CONSTRAINT_FIELDS:
        container = (
            constraints if container_name is None
            else getattr(constraints, container_name, None)
        )
        indexed = getattr(container, indexed_name, None)
        if indexed is not None and hasattr(indexed, "index"):
            return pd.Index(indexed.index)
    if n_assets is None:
        return None
    return pd.RangeIndex(n_assets)


def _optional_series_to_array(
        values: Optional[pd.Series],
        index: Sequence[Any],
) -> Optional[np.ndarray]:
    """Align optional Series values or preserve a plain array's given order."""
    if values is None:
        return None
    if isinstance(values, pd.Series):
        return values.reindex(index=pd.Index(index)).to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def _exposure_facts(constraints: Constraints) -> _ExposureFacts:
    """Return literal exposure bounds and their exact equality fact.

    Args:
        constraints: Portfolio constraints containing min/max exposure.

    Returns:
        Exposure limits whose equality follows the stored values literally.
    """
    max_exp = float(getattr(constraints, "max_exposure", 1.0))
    min_exp = float(getattr(constraints, "min_exposure", max_exp))
    return _ExposureFacts(
        minimum=min_exp,
        maximum=max_exp,
        is_exact_equality=min_exp == max_exp,
    )


def _budget_box_residuals(
        weights: np.ndarray,
        constraints: Constraints,
        index: Sequence[Any],
) -> _BudgetBoxResiduals:
    """Compute budget, long-only, and instrument-bound residual quantities.

    Args:
        weights: Finite candidate weights in solver order.
        constraints: Budget and box constraint specification.
        index: Asset order used to align labelled bounds.

    Returns:
        Raw residual quantities without applying acceptance tolerances.
    """
    w = np.asarray(weights, dtype=float)
    exposure = _exposure_facts(constraints)
    sum_weights = float(np.sum(w))
    if exposure.is_exact_equality or sum_weights > exposure.maximum:
        budget_residual = sum_weights - exposure.maximum
    elif sum_weights < exposure.minimum:
        budget_residual = sum_weights - exposure.minimum
    else:
        budget_residual = 0.0

    minimum_weight = float(np.min(w))
    is_long_only = bool(getattr(constraints, "is_long_only", False))
    long_only_violation = float(-minimum_weight) if is_long_only else 0.0
    maximum_weights = _optional_series_to_array(
        getattr(constraints, "max_weights", None), index)
    minimum_weights = _optional_series_to_array(
        getattr(constraints, "min_weights", None), index)

    maximum_overage = None
    maximum_position = None
    minimum_difference = None
    minimum_position = None
    max_box_violation = 0.0
    if is_long_only:
        max_box_violation = max(max_box_violation, long_only_violation)
    if maximum_weights is not None:
        overages = w - maximum_weights
        maximum_overage = float(np.max(overages))
        maximum_position = int(np.argmax(overages))
        max_box_violation = max(max_box_violation, maximum_overage)
    if minimum_weights is not None:
        differences = w - minimum_weights
        minimum_difference = float(np.min(differences))
        minimum_position = int(np.argmin(differences))
        max_box_violation = max(max_box_violation, -minimum_difference)

    return _BudgetBoxResiduals(
        exposure=exposure,
        sum_weights=sum_weights,
        budget_residual=float(budget_residual),
        minimum_weight=minimum_weight,
        long_only_violation=long_only_violation,
        minimum_weights=minimum_weights,
        maximum_weights=maximum_weights,
        minimum_difference=minimum_difference,
        minimum_position=minimum_position,
        maximum_overage=maximum_overage,
        maximum_position=maximum_position,
        max_box_violation=max(0.0, max_box_violation),
    )


def _bound_violation(
        actual: float,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
) -> float:
    """Return the non-negative distance by which ``actual`` misses its bounds."""
    violation = 0.0
    if lower is not None:
        violation = max(violation, float(lower) - float(actual))
    if upper is not None:
        violation = max(violation, float(actual) - float(upper))
    return max(0.0, float(violation))


def _iter_finite_group_bounds(
        group_constraints: Any,
        index: Optional[pd.Index] = None,
) -> Iterator[_FiniteGroupBound]:
    """Yield loadings columns with at least one stated, non-missing bound.

    Infinite numeric values remain stated bounds, matching the established
    compiler behavior; only absent and NaN values are omitted.

    Args:
        group_constraints: Group allocation constraint block, or ``None``.
        index: Optional asset index for reindexing and zero-filling loadings.

    Yields:
        Group loading columns and normalized optional lower/upper bounds.
    """
    if group_constraints is None:
        return
    loadings = getattr(group_constraints, "group_loadings", None)
    if loadings is None:
        return
    if index is not None:
        loadings = loadings.reindex(index=index).fillna(0.0)
    lower_bounds = getattr(group_constraints, "group_min_allocation", None)
    upper_bounds = getattr(group_constraints, "group_max_allocation", None)
    for group in loadings.columns:
        lower = lower_bounds.get(group) if lower_bounds is not None else None
        upper = upper_bounds.get(group) if upper_bounds is not None else None
        if lower is not None and pd.isna(lower):
            lower = None
        if upper is not None and pd.isna(upper):
            upper = None
        if lower is not None or upper is not None:
            yield _FiniteGroupBound(
                name=group,
                loadings=loadings[group],
                lower=None if lower is None else float(lower),
                upper=None if upper is None else float(upper),
            )


def _group_reachability(
        loadings: Sequence[float],
        floors: Sequence[float],
        caps: Sequence[float],
        lower: Optional[float],
        upper: Optional[float],
        atol: float,
) -> _GroupReachability:
    """Evaluate current loading-weighted endpoints against one group band."""
    loading_values = np.asarray(loadings, dtype=float)
    minimum = float(np.sum(loading_values * np.asarray(floors, dtype=float)))
    maximum = float(np.sum(loading_values * np.asarray(caps, dtype=float)))
    return _GroupReachability(
        minimum=minimum,
        maximum=maximum,
        lower_unreachable=(
            lower is not None and maximum < float(lower) - atol),
        upper_unreachable=(
            upper is not None and minimum > float(upper) + atol),
    )


def _construction_group_reachability_errors(
        constraints: Constraints,
        atol: float = 1e-4,
) -> Tuple[str, ...]:
    """Return constructor-time group/box consistency errors without side effects."""
    errors = []
    group_constraints = getattr(constraints, "group_lower_upper_constraints", None)
    for bound in _iter_finite_group_bounds(group_constraints):
        group_loading = bound.loadings
        members = group_loading.index[group_loading > 0]
        if len(members) == 0:
            continue
        member_loadings = group_loading.loc[members]

        max_weights = getattr(constraints, "max_weights", None)
        if max_weights is not None:
            caps = max_weights.reindex(members, fill_value=1.0).to_numpy(dtype=float)
            caps = np.where(np.isnan(caps), 0.0, caps)
        else:
            upper = 1.0 if constraints.is_long_only else constraints.max_exposure
            caps = np.full(len(members), upper, dtype=float)

        min_weights = getattr(constraints, "min_weights", None)
        if min_weights is not None:
            floors = min_weights.reindex(members, fill_value=0.0).to_numpy(dtype=float)
            floors = np.where(np.isnan(floors), 0.0, floors)
        else:
            lower = 0.0 if constraints.is_long_only else -constraints.max_exposure
            floors = np.full(len(members), lower, dtype=float)

        reachability = _group_reachability(
            member_loadings.to_numpy(dtype=float),
            floors,
            caps,
            bound.lower,
            bound.upper,
            atol,
        )
        if reachability.lower_unreachable:
            errors.append(
                f"Group '{bound.name}': loading-weighted sum of asset max_weights "
                f"({reachability.maximum:.4f}) < group_min_allocation "
                f"({bound.lower:.4f}). Increase max_weights for assets "
                f"{members.tolist()} or lower group_min_allocation."
            )
        if reachability.upper_unreachable:
            errors.append(
                f"Group '{bound.name}': loading-weighted sum of asset min_weights "
                f"({reachability.minimum:.4f}) > group_max_allocation "
                f"({bound.upper:.4f}). Lower min_weights for assets "
                f"{members.tolist()} or increase group_max_allocation."
            )
        if bound.upper is not None and min_weights is not None:
            for ticker in members:
                wmin = min_weights.get(ticker, 0.0)
                if not np.isnan(wmin):
                    weighted_min = wmin * member_loadings.loc[ticker]
                    if weighted_min > bound.upper + atol:
                        errors.append(
                            f"Asset '{ticker}': min_weight ({wmin:.4f}) x loading "
                            f"({member_loadings.loc[ticker]:.4f}) = {weighted_min:.4f} "
                            f"> group '{bound.name}' max_allocation "
                            f"({bound.upper:.4f}). Lower min_weight for '{ticker}' or "
                            f"increase group_max_allocation for '{bound.name}'."
                        )
    return tuple(errors)


def _static_reachability_findings(
        constraints: Constraints,
        index: pd.Index,
        atol: float,
) -> Tuple[_StaticReachabilityFinding, ...]:
    """Return cheap box, group, and benchmark reachability findings."""
    findings = []
    exposure = _exposure_facts(constraints)
    maximum_weights = _optional_series_to_array(
        getattr(constraints, "max_weights", None), index)
    minimum_weights = _optional_series_to_array(
        getattr(constraints, "min_weights", None), index)

    if maximum_weights is not None:
        cap_sum = float(np.sum(maximum_weights))
        if cap_sum < exposure.minimum - atol:
            message = (
                f"box caps sum to {cap_sum:.4f} < budget "
                f"{exposure.minimum:.4f}: full investment is infeasible")
            findings.append(_StaticReachabilityFinding(
                kind="structural", message=message, code="box_caps_unreachable"))
    if minimum_weights is not None:
        floor_sum = float(np.sum(minimum_weights))
        if floor_sum > exposure.maximum + atol:
            message = (
                f"box floors sum to {floor_sum:.4f} > budget "
                f"{exposure.maximum:.4f}: constraints are infeasible")
            findings.append(_StaticReachabilityFinding(
                kind="structural", message=message, code="box_floors_overshoot"))

    group_constraints = getattr(constraints, "group_lower_upper_constraints", None)
    caps = maximum_weights if maximum_weights is not None else np.ones(len(index))
    floors = minimum_weights if minimum_weights is not None else np.zeros(len(index))
    for bound in _iter_finite_group_bounds(group_constraints, index=index):
        loading_values = bound.loadings.to_numpy(dtype=float)
        if not np.any(loading_values > 0):
            continue
        reachability = _group_reachability(
            loading_values, floors, caps, bound.lower, bound.upper, atol)
        if reachability.lower_unreachable:
            findings.append(_StaticReachabilityFinding(
                kind="group",
                name=str(bound.name),
                code="floor_unreachable",
                message=(
                    f"group '{bound.name}' floor {bound.lower:.4f} > max reachable "
                    f"{reachability.maximum:.4f} given box caps"),
            ))
        if reachability.upper_unreachable:
            findings.append(_StaticReachabilityFinding(
                kind="group",
                name=str(bound.name),
                code="cap_too_low",
                message=(
                    f"group '{bound.name}' cap {bound.upper:.4f} < min forced "
                    f"{reachability.minimum:.4f} given box floors"),
            ))

    benchmark = _optional_series_to_array(
        getattr(constraints, "benchmark_weights", None), index)
    if benchmark is not None:
        if maximum_weights is not None:
            cap_differences = benchmark - maximum_weights
            if float(np.max(cap_differences)) > atol:
                position = int(np.argmax(cap_differences))
                findings.append(_StaticReachabilityFinding(
                    kind="benchmark",
                    position=position,
                    code="cap_exceeded",
                    message=(
                        f"benchmark weight {benchmark[position]:.4f} at index "
                        f"{position} exceeds its cap {maximum_weights[position]:.4f}"),
                ))
        if minimum_weights is not None:
            floor_differences = minimum_weights - benchmark
            if float(np.max(floor_differences)) > atol:
                position = int(np.argmax(floor_differences))
                findings.append(_StaticReachabilityFinding(
                    kind="benchmark",
                    position=position,
                    code="below_floor",
                    message=(
                        f"benchmark weight {benchmark[position]:.4f} at index "
                        f"{position} below its floor {minimum_weights[position]:.4f}"),
                ))
        benchmark_sum = float(np.sum(benchmark))
        if (benchmark_sum < exposure.minimum - atol
                or benchmark_sum > exposure.maximum + atol):
            findings.append(_StaticReachabilityFinding(
                kind="benchmark",
                position=-1,
                code="sum_out_of_band",
                message=(
                    f"benchmark sums to {benchmark_sum:.4f}, outside budget band "
                    f"[{exposure.minimum:.4f}, {exposure.maximum:.4f}]"),
            ))
    return tuple(findings)


@dataclass(frozen=True)
class ConstraintResidual:
    """One evaluated constraint with bounds, tolerance, and pass/fail state."""

    constraint_type: str
    name: str
    actual: float
    lower: Optional[float]
    upper: Optional[float]
    violation: float
    tolerance: float
    hard: bool
    passed: bool


def _group_allocation_residuals(
        weights: np.ndarray,
        constraints: Constraints,
        index: Sequence[Any],
        tolerance: float = 1e-4,
) -> Tuple[ConstraintResidual, ...]:
    """Evaluate every active group-allocation row in compiler order."""
    w = np.asarray(weights, dtype=float).ravel()
    residuals = []
    group_bounds = getattr(constraints, "group_lower_upper_constraints", None)
    for bound in _iter_finite_group_bounds(group_bounds, index=pd.Index(index)):
        loading_values = bound.loadings.to_numpy(dtype=float)
        if not np.any(~np.isclose(loading_values, 0.0)):
            continue
        actual = float(loading_values @ w)
        violation = _bound_violation(
            actual=actual,
            lower=bound.lower,
            upper=bound.upper,
        )
        residuals.append(ConstraintResidual(
            constraint_type="group_weight",
            name=str(bound.name),
            actual=actual,
            lower=bound.lower,
            upper=bound.upper,
            violation=violation,
            tolerance=float(tolerance),
            hard=True,
            passed=violation <= tolerance,
        ))
    return tuple(residuals)


def evaluate_constraint_residuals(
        weights: np.ndarray,
        constraints: Constraints,
        covar: Optional[np.ndarray] = None,
        covar_factorization: Optional[CovarianceFactorization] = None,
        tolerance: float = 1e-4,
) -> Tuple[ConstraintResidual, ...]:
    """Evaluate solver and policy constraints on one weight vector.

    The covariance is the stabilized solver covariance when a factorization is
    supplied, ensuring the audit measures exactly the geometry enforced by the
    optimizer. Soft utility terms are included with ``hard=False`` for
    reporting but do not determine compliance.

    Args:
        weights: Candidate portfolio weights in aligned constraint order.
        constraints: Ticker-aligned constraint set used by the solver.
        covar: Covariance used when no factorization is available.
        covar_factorization: Optional exact factorization used by the solver;
            takes precedence over ``covar``.
        tolerance: Default absolute tolerance for aggregate constraints.

    Returns:
        Immutable residual records for all applicable hard and soft terms.
    """
    w = np.asarray(weights, dtype=float).ravel()
    index = _resolve_asset_index(constraints, n_assets=len(w))
    utility = (
        getattr(constraints.constraint_enforcement_type, "name", None)
        == "UTILITY_CONSTRAINTS"
    )
    risk_covar = (
        covar_factorization.covar
        if covar_factorization is not None else covar
    )
    if risk_covar is not None:
        risk_covar = np.asarray(risk_covar, dtype=float)

    residuals: List[ConstraintResidual] = []

    def add(kind: str, name: str, actual: float,
            lower: Optional[float] = None,
            upper: Optional[float] = None,
            hard: bool = True,
            atol: float = tolerance) -> None:
        """Record one residual, deriving its violation from the supplied bounds."""
        violation = _bound_violation(actual=actual, lower=lower, upper=upper)
        residuals.append(ConstraintResidual(
            constraint_type=kind,
            name=str(name),
            actual=float(actual),
            lower=None if lower is None else float(lower),
            upper=None if upper is None else float(upper),
            violation=violation,
            tolerance=float(atol),
            hard=bool(hard),
            passed=(not hard) or violation <= atol,
        ))

    budget_box = _budget_box_residuals(w, constraints, index)
    add(
        'exposure',
        'total',
        budget_box.sum_weights,
        budget_box.exposure.minimum,
        budget_box.exposure.maximum,
    )
    if constraints.is_long_only:
        add('long_only', 'minimum_weight', budget_box.minimum_weight, lower=0.0,
            atol=1e-6)

    for pos, ticker in enumerate(index):
        if (budget_box.minimum_weights is not None
                and np.isfinite(budget_box.minimum_weights[pos])):
            add(
                'instrument_weight',
                ticker,
                w[pos],
                lower=budget_box.minimum_weights[pos],
                atol=1e-6,
            )
        if (budget_box.maximum_weights is not None
                and np.isfinite(budget_box.maximum_weights[pos])):
            add(
                'instrument_weight',
                ticker,
                w[pos],
                upper=budget_box.maximum_weights[pos],
                atol=1e-6,
            )

    if constraints.target_return is not None and constraints.asset_returns is not None:
        actual_return = float(
            constraints.asset_returns.reindex(index).fillna(0.0).to_numpy() @ w)
        add('target_return', 'portfolio', actual_return,
            lower=constraints.target_return)

    if risk_covar is not None:
        portfolio_var = max(float(w @ risk_covar @ w), 0.0)
        portfolio_vol = float(np.sqrt(portfolio_var))
        if constraints.max_target_portfolio_vol_an is not None:
            add('portfolio_volatility', 'maximum', portfolio_vol,
                upper=constraints.max_target_portfolio_vol_an, hard=not utility)

    if constraints.weights_0 is not None:
        w0 = constraints.weights_0.reindex(index).fillna(0.0).to_numpy(dtype=float)
        trade = w - w0
        if constraints.turnover_constraint is not None:
            if constraints.turnover_costs is not None:
                costs = constraints.turnover_costs.reindex(
                    index).fillna(1.0).to_numpy()
                actual_turnover = float(np.abs(costs * trade).sum())
            else:
                actual_turnover = float(np.abs(trade).sum())
            add('turnover', 'total_l1', actual_turnover,
                upper=constraints.turnover_constraint, hard=not utility)

        group_turnover = constraints.group_turnover_constraint
        if group_turnover is not None and group_turnover.group_max_turnover is not None:
            loadings = group_turnover.group_loadings.reindex(index).fillna(0.0)
            for group, limit in group_turnover.group_max_turnover.items():
                if group not in loadings.columns or pd.isna(limit):
                    continue
                actual = float(np.abs(loadings[group].to_numpy() * trade).sum())
                add('group_turnover', group, actual, upper=float(limit),
                    hard=not utility)

    benchmark = constraints.benchmark_weights
    if benchmark is not None and risk_covar is not None:
        benchmark_np = benchmark.reindex(index).fillna(0.0).to_numpy(dtype=float)
        active = w - benchmark_np
        total_tre = float(np.sqrt(max(float(active @ risk_covar @ active), 0.0)))
        if constraints.tracking_err_vol_constraint is not None:
            add('tracking_error', 'total', total_tre,
                upper=constraints.tracking_err_vol_constraint, hard=not utility)
        group_tre = constraints.group_tracking_error_constraint
        if group_tre is not None and group_tre.group_tre_vols is not None:
            loadings = group_tre.group_loadings.reindex(index).fillna(0.0)
            for group, limit in group_tre.group_tre_vols.items():
                if group not in loadings.columns or pd.isna(limit):
                    continue
                group_active = loadings[group].to_numpy() * active
                actual = float(np.sqrt(max(
                    float(group_active @ risk_covar @ group_active), 0.0)))
                add('group_tracking_error', group, actual, upper=float(limit),
                    hard=not utility)

    residuals.extend(_group_allocation_residuals(
        weights=w,
        constraints=constraints,
        index=index,
        tolerance=tolerance,
    ))

    for kind, deviation in (
            ('sector_deviation', constraints.sector_deviation_constraints),
            ('style_deviation', constraints.style_deviation_constraints)):
        if deviation is None or benchmark is None:
            continue
        active = w - benchmark.reindex(index).fillna(0.0).to_numpy(dtype=float)
        loadings = deviation.factor_loading_mat.reindex(index).fillna(0.0)
        for group, limit in deviation.factor_max_deviation.items():
            if group not in loadings.columns or pd.isna(limit):
                continue
            actual = float(loadings[group].to_numpy() @ active)
            add(kind, group, abs(actual), upper=float(limit))

    beta_constraint = constraints.benchmark_beta_constraint
    if beta_constraint is not None and beta_constraint.beta_loadings is not None:
        actual_beta = float(
            beta_constraint.beta_loadings.reindex(index).fillna(0.0).to_numpy() @ w)
        add('benchmark_beta', 'portfolio', actual_beta,
            lower=beta_constraint.beta_min, upper=beta_constraint.beta_max)

    return tuple(residuals)


__all__ = ["ConstraintResidual", "evaluate_constraint_residuals"]
