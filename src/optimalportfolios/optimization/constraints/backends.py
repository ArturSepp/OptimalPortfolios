"""Compile portfolio constraint specifications for supported solver backends.

The functions in this module translate an aligned ``Constraints`` specification
into CVXPY expressions, SciPy callbacks and bounds, or PyRB matrix inequalities.
They preserve the caller's weight, covariance, volatility, tracking-error, and
turnover units and perform no resampling or annualisation. Constraint policy,
universe alignment, and post-solve diagnostics remain owned by their respective
modules.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import cvxpy as cvx
import numpy as np
import pandas as pd
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.constraints.nonpos import Inequality

from optimalportfolios.optimization.constraints.expressions import (
    _cvx_factor_risk,
    add_term_to_objective_function,
    cvx_covar_variance,
)
from optimalportfolios.optimization.covar_factorization import CovarianceFactorization

if TYPE_CHECKING:
    from optimalportfolios.optimization.constraints.core import Constraints


logger = logging.getLogger("optimalportfolios.optimization.constraints")


def set_cvx_exposure_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
        exposure_scaler: cvx.Variable = None,
) -> List[Inequality]:
    """Generate CVXPY exposure constraints.

    Creates constraints for long-only, total exposure, and individual weight bounds.

    Args:
        constraint_spec: Portfolio constraint specification to compile.
        w: Portfolio weight variable.
        exposure_scaler: Optional exposure scaling for levered portfolios.

    Returns:
        List of CVXPY inequality constraints.
    """
    constraints = []
    if constraint_spec.is_long_only:
        constraints += [w >= 0]

    if exposure_scaler is None:
        if constraint_spec.max_exposure == constraint_spec.min_exposure:
            constraints += [cvx.sum(w) == constraint_spec.max_exposure]
        else:
            constraints += [cvx.sum(w) <= constraint_spec.max_exposure]
            constraints += [cvx.sum(w) >= constraint_spec.min_exposure]
    else:
        if constraint_spec.max_exposure == constraint_spec.min_exposure:
            constraints += [cvx.sum(w) == exposure_scaler * constraint_spec.max_exposure]
        else:
            # preserve both bounds in Charnes-Cooper space: k*min ≤ sum(y) ≤ k*max
            constraints += [cvx.sum(w) <= exposure_scaler * constraint_spec.max_exposure]
            constraints += [cvx.sum(w) >= exposure_scaler * constraint_spec.min_exposure]

    if constraint_spec.min_weights is not None:
        min_weights = (constraint_spec.min_weights.to_numpy()
                       if isinstance(constraint_spec.min_weights, pd.Series)
                       else constraint_spec.min_weights)
        if exposure_scaler is None:
            constraints += [w >= min_weights]
        else:
            constraints += [w >= exposure_scaler * min_weights]

    if constraint_spec.max_weights is not None:
        max_weights = (constraint_spec.max_weights.to_numpy()
                       if isinstance(constraint_spec.max_weights, pd.Series)
                       else constraint_spec.max_weights)
        if exposure_scaler is None:
            constraints += [w <= max_weights]
        else:
            constraints += [w <= exposure_scaler * max_weights]
    return constraints


def _set_cvx_target_return_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
) -> List:
    """Compile the optional minimum-return row."""
    if constraint_spec.target_return is None:
        return []
    if constraint_spec.asset_returns is None:
        raise ValueError("asset_returns must be given for target_return constraint")
    return [constraint_spec.asset_returns.to_numpy() @ w >= constraint_spec.target_return]


def _set_cvx_portfolio_volatility_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
        covar: Union[np.ndarray, psd_wrap] = None,
        covar_factorization: Optional[CovarianceFactorization] = None,
) -> List:
    """Compile the optional portfolio-volatility upper bound."""
    if constraint_spec.max_target_portfolio_vol_an is None:
        return []
    if covar_factorization is not None:
        portfolio_risk = _cvx_factor_risk(w, covar_factorization)
        return [cvx.norm(portfolio_risk, 2)
                <= constraint_spec.max_target_portfolio_vol_an]
    if covar is None:
        raise ValueError("covar must be given for portfolio volatility constraint")
    return [cvx.quad_form(w, covar)
            <= constraint_spec.max_target_portfolio_vol_an ** 2]


def _set_cvx_group_turnover_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
) -> List:
    """Compile hard group-turnover rows when configured."""
    if constraint_spec.group_turnover_constraint is None:
        return []
    return constraint_spec.group_turnover_constraint.set_group_turnover_constraints(
        w=w, weights_0=constraint_spec.weights_0)


def _set_cvx_total_turnover_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
) -> List:
    """Compile the independent whole-portfolio turnover row."""
    if constraint_spec.turnover_constraint is None:
        return []
    if constraint_spec.weights_0 is None:
        logger.debug("turnover constraint skipped because weights_0 is absent")
        return []
    if constraint_spec.turnover_costs is not None:
        return [cvx.norm(
            cvx.multiply(
                constraint_spec.turnover_costs.to_numpy(),
                w - constraint_spec.weights_0,
            ),
            1,
        ) <= constraint_spec.turnover_constraint]
    assert w.size == len(constraint_spec.weights_0.index)
    return [cvx.norm(w - constraint_spec.weights_0, 1)
            <= constraint_spec.turnover_constraint]


def _set_cvx_group_tracking_error_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
        covar: Union[np.ndarray, psd_wrap] = None,
        covar_factorization: Optional[CovarianceFactorization] = None,
) -> List:
    """Compile hard group tracking-error rows when configured."""
    return constraint_spec.group_tracking_error_constraint.set_cvx_group_tre_constraints(
        w=w,
        benchmark_weights=constraint_spec.benchmark_weights,
        covar=covar,
        covar_factorization=covar_factorization,
    )


def _set_cvx_total_tracking_error_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
        covar: Union[np.ndarray, psd_wrap] = None,
        covar_factorization: Optional[CovarianceFactorization] = None,
) -> List:
    """Compile the whole-portfolio tracking-error row when configured."""
    if constraint_spec.benchmark_weights is None:
        raise ValueError("benchmark_weights must be given for tracking error constraint")
    active_weights = w - constraint_spec.benchmark_weights.to_numpy()
    if covar_factorization is not None:
        active_risk = _cvx_factor_risk(active_weights, covar_factorization)
        return [cvx.norm(active_risk, 2)
                <= constraint_spec.tracking_err_vol_constraint]
    tracking_error_var = cvx_covar_variance(active_weights=active_weights, covar=covar)
    return [tracking_error_var <= constraint_spec.tracking_err_vol_constraint ** 2]


def _set_cvx_group_allocation_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
        exposure_scaler: cvx.Variable = None,
) -> List:
    """Compile hard group allocation floors and ceilings."""
    if constraint_spec.group_lower_upper_constraints is None:
        return []
    return (
        constraint_spec.group_lower_upper_constraints.set_cvx_group_lower_upper_constraints(
            w=w, exposure_scaler=exposure_scaler)
    )


def _set_cvx_deviation_constraints(
        deviation_constraint,
        w: cvx.Variable,
        benchmark_weights: pd.Series,
) -> List:
    """Compile one optional family of benchmark-deviation rows."""
    if deviation_constraint is None:
        return []
    return deviation_constraint.set_cvx_constraints(
        w=w, benchmark_weights=benchmark_weights)


def _set_cvx_beta_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
) -> List:
    """Compile the optional benchmark-beta range rows."""
    if constraint_spec.benchmark_beta_constraint is None:
        return []
    return constraint_spec.benchmark_beta_constraint.set_cvx_beta_constraints(w=w)


def _set_cvx_utility_hard_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
        exposure_scaler: cvx.Variable = None,
) -> List:
    """Compile mandate rows that remain hard under utility enforcement."""
    constraints = constraint_spec.set_cvx_exposure_constraints(
        w=w, exposure_scaler=exposure_scaler)
    constraints += _set_cvx_target_return_constraints(constraint_spec, w)
    constraints += _set_cvx_group_allocation_constraints(
        constraint_spec, w, exposure_scaler=exposure_scaler)
    constraints += _set_cvx_deviation_constraints(
        constraint_spec.sector_deviation_constraints,
        w,
        constraint_spec.benchmark_weights,
    )
    constraints += _set_cvx_deviation_constraints(
        constraint_spec.style_deviation_constraints,
        w,
        constraint_spec.benchmark_weights,
    )
    constraints += _set_cvx_beta_constraints(constraint_spec, w)
    return constraints


def _make_cvx_group_turnover_penalty(
        constraint_spec: Constraints,
        w: cvx.Variable,
) -> Optional[cvx.Expression]:
    """Build the optional group-turnover utility penalty."""
    if constraint_spec.weights_0 is None:
        logger.debug("group turnover utility skipped because weights_0 is absent")
        return None
    return constraint_spec.group_turnover_constraint.set_cvx_group_turnover_utility(
        w=w, weights_0=constraint_spec.weights_0)


def _make_cvx_total_turnover_penalty(
        constraint_spec: Constraints,
        w: cvx.Variable,
) -> Optional[cvx.Expression]:
    """Build the optional whole-portfolio turnover utility penalty."""
    if constraint_spec.weights_0 is None:
        logger.debug("turnover utility skipped because weights_0 is absent")
        return None
    if constraint_spec.turnover_costs is not None:
        return -1.0 * constraint_spec.turnover_utility_weight * cvx.norm(
            cvx.multiply(
                constraint_spec.turnover_costs.to_numpy(),
                w - constraint_spec.weights_0,
            ),
            1,
        )
    assert w.size == len(constraint_spec.weights_0.index)
    return -1.0 * constraint_spec.turnover_utility_weight * cvx.norm(
        w - constraint_spec.weights_0, 1)


def _make_cvx_group_tracking_error_penalty(
        constraint_spec: Constraints,
        w: cvx.Variable,
        covar: Union[np.ndarray, psd_wrap] = None,
        covar_factorization: Optional[CovarianceFactorization] = None,
) -> Optional[cvx.Expression]:
    """Build the optional group tracking-error utility penalty."""
    if constraint_spec.benchmark_weights is None:
        raise ValueError(
            "benchmark_weights must be given for group tracking error constraint")
    return constraint_spec.group_tracking_error_constraint.set_cvx_group_tre_utility(
        w=w,
        benchmark_weights=constraint_spec.benchmark_weights,
        covar=covar,
        covar_factorization=covar_factorization,
    )


def _make_cvx_total_tracking_error_penalty(
        constraint_spec: Constraints,
        w: cvx.Variable,
        covar: Union[np.ndarray, psd_wrap] = None,
        covar_factorization: Optional[CovarianceFactorization] = None,
) -> Optional[cvx.Expression]:
    """Build the optional whole-portfolio tracking-error utility penalty."""
    if constraint_spec.benchmark_weights is None:
        raise ValueError("benchmark_weights must be given for tracking error constraint")
    tracking_error_variance = cvx_covar_variance(
        active_weights=w - constraint_spec.benchmark_weights.to_numpy(),
        covar=covar,
        covar_factorization=covar_factorization,
    )
    return -1.0 * constraint_spec.tre_utility_weight * tracking_error_variance


def set_cvx_all_constraints(
        constraint_spec: Constraints,
        w: cvx.Variable,
        covar: Union[np.ndarray, psd_wrap] = None,
        exposure_scaler: cvx.Variable = None,
        covar_factorization: Optional[CovarianceFactorization] = None,
) -> List:
    """Generate all CVXPY constraints for portfolio optimization.

    Comprehensive constraint generation for mean-variance and related optimization problems.

    Args:
        constraint_spec: Portfolio constraint specification to compile.
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
    constraints = constraint_spec.set_cvx_exposure_constraints(
        w=w, exposure_scaler=exposure_scaler)

    constraints += _set_cvx_target_return_constraints(constraint_spec, w)
    constraints += _set_cvx_portfolio_volatility_constraints(
        constraint_spec,
        w,
        covar=covar,
        covar_factorization=covar_factorization,
    )
    constraints += _set_cvx_group_turnover_constraints(constraint_spec, w)
    # Group limits and the whole-portfolio limit are independent controls. Identity loadings
    # can cap every name while ``turnover_constraint`` caps aggregate trading. The former
    # ``elif`` silently disabled the portfolio cap whenever any group cap was present.
    constraints += _set_cvx_total_turnover_constraints(constraint_spec, w)

    # Group and whole-portfolio tracking-error limits are independent policy controls,
    # matching the additive treatment of group and total turnover limits above.
    if constraint_spec.group_tracking_error_constraint is not None:
        constraints += _set_cvx_group_tracking_error_constraints(
            constraint_spec,
            w,
            covar=covar,
            covar_factorization=covar_factorization,
        )
    if constraint_spec.tracking_err_vol_constraint is not None:
        constraints += _set_cvx_total_tracking_error_constraints(
            constraint_spec,
            w,
            covar=covar,
            covar_factorization=covar_factorization,
        )

    constraints += _set_cvx_group_allocation_constraints(
        constraint_spec, w, exposure_scaler=exposure_scaler)
    constraints += _set_cvx_deviation_constraints(
        constraint_spec.sector_deviation_constraints,
        w,
        constraint_spec.benchmark_weights,
    )
    constraints += _set_cvx_deviation_constraints(
        constraint_spec.style_deviation_constraints,
        w,
        constraint_spec.benchmark_weights,
    )
    constraints += _set_cvx_beta_constraints(constraint_spec, w)

    return constraints


def set_cvx_utility_objective_constraints(
        constraint_spec: Constraints,
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
        constraint_spec: Portfolio constraint specification to compile.
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
    benchmark_weights: pd.Series = constraint_spec.benchmark_weights

    if alphas is not None:
        objective_fun = alphas.T @ (w - benchmark_weights.to_numpy())
    else:
        objective_fun = None

    # Utility mode intentionally gives group penalties precedence over aggregate turnover.
    if constraint_spec.group_turnover_constraint is not None:
        term = _make_cvx_group_turnover_penalty(constraint_spec, w)
        objective_fun = add_term_to_objective_function(objective_fun, term)
    elif constraint_spec.turnover_utility_weight is not None:
        term = _make_cvx_total_turnover_penalty(constraint_spec, w)
        objective_fun = add_term_to_objective_function(objective_fun, term)

    # Group tracking error likewise keeps precedence over the portfolio-level penalty.
    if constraint_spec.group_tracking_error_constraint is not None:
        term = _make_cvx_group_tracking_error_penalty(
            constraint_spec,
            w,
            covar=covar,
            covar_factorization=covar_factorization,
        )
        objective_fun = add_term_to_objective_function(objective_fun, term)
    elif constraint_spec.tre_utility_weight is not None:
        term = _make_cvx_total_tracking_error_penalty(
            constraint_spec,
            w,
            covar=covar,
            covar_factorization=covar_factorization,
        )
        objective_fun = add_term_to_objective_function(objective_fun, term)

    constraints = _set_cvx_utility_hard_constraints(
        constraint_spec,
        w,
        exposure_scaler=exposure_scaler,
    )
    return objective_fun, constraints


def set_scipy_bounds(constraint_spec: Constraints, covar: np.ndarray):
    """Convert weight constraints into (min, max) bounds for scipy solvers.

    Handles all combinations of min_weights, max_weights, and is_long_only.
    When neither bound is provided, returns (0, 1) for long-only or None
    for unconstrained. When either bound is provided, the missing side
    defaults to 0 (long-only) or -inf (unconstrained) for lows, and 1 for highs.

    Args:
        constraint_spec: Portfolio constraint specification to compile.
        covar: Covariance matrix (N x N), used to infer number of assets.

    Returns:
        Array of (min, max) tuples per asset, or None if unconstrained.
    """
    min_w = constraint_spec.min_weights
    max_w = constraint_spec.max_weights

    # no explicit bounds: use long-only defaults or fully unconstrained
    if min_w is None and max_w is None:
        if constraint_spec.is_long_only:
            n = covar.shape[0]
            bounds = np.array([(0.0, 1.0) for _ in range(n)])
        else:
            bounds = None
    else:
        # at least one bound is provided: fill the missing side with defaults
        n = covar.shape[0]
        lows = min_w.to_numpy() if min_w is not None else np.full(
            n, 0.0 if constraint_spec.is_long_only else -np.inf)
        highs = max_w.to_numpy() if max_w is not None else np.ones(n)
        bounds = np.array(list(zip(lows, highs)))

    return bounds


def set_scipy_constraints(
        constraint_spec: Constraints,
        covar: np.ndarray,
) -> Tuple[List, np.ndarray]:
    """Generate SciPy-compatible constraints (inequality form: constraint >= 0).

    Converts constraints to format expected by scipy.optimize.minimize.

    Args:
        constraint_spec: Portfolio constraint specification to compile.
        covar: Covariance matrix (used for bounds inference if needed).

    Returns:
        Tuple of (constraint dictionaries, bounds array).
    """
    constraints = []

    if constraint_spec.is_long_only and constraint_spec.min_weights is None:
        constraints += [{'type': 'ineq', 'fun': long_only_constraint}]

    constraints += [{
        'type': 'ineq',
        'fun': lambda x: constraint_spec.max_exposure - np.sum(x),
    }]
    constraints += [{
        'type': 'ineq',
        'fun': lambda x: np.sum(x) - constraint_spec.min_exposure,
    }]

    if constraint_spec.group_lower_upper_constraints is not None:
        gluc = constraint_spec.group_lower_upper_constraints
        for group in gluc.group_loadings.columns:
            group_loading = gluc.group_loadings[group].to_numpy()
            if np.any(np.isclose(group_loading, 0.0) == False):
                if gluc.group_min_allocation is not None:
                    min_weight = gluc.group_min_allocation.loc[group]
                    if not np.isnan(min_weight):
                        constraints += [{
                            'type': 'ineq',
                            'fun': make_min_constraint(group_loading, min_weight),
                        }]
                if gluc.group_max_allocation is not None:
                    max_weight = gluc.group_max_allocation.loc[group]
                    if not np.isnan(max_weight):
                        constraints += [{
                            'type': 'ineq',
                            'fun': make_max_constraint(group_loading, max_weight),
                        }]

    bounds = constraint_spec.set_scipy_bounds(covar=covar)
    return constraints, bounds


def set_pyrb_constraints(
        constraint_spec: Constraints,
        covar: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate PyRB-compatible constraints in matrix form (C*x <= d).

    Converts group constraints to matrix inequality form for risk budgeting taa.

    Args:
        constraint_spec: Portfolio constraint specification to compile.
        covar: Covariance matrix (used for bounds inference if needed).

    Returns:
        Tuple of (bounds array, constraint matrix C, constraint vector d).
    """
    bounds = constraint_spec.set_scipy_bounds(covar=covar)

    if constraint_spec.group_lower_upper_constraints is not None:
        gluc = constraint_spec.group_lower_upper_constraints
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


def total_weight_constraint(x: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Total portfolio weight constraint: total - sum(x) = 0."""
    return total - np.sum(x)


def long_only_constraint(x: np.ndarray) -> np.ndarray:
    """Long-only constraint: x >= 0."""
    return x


def make_min_constraint(
        group_loading: np.ndarray,
        min_weight: float,
) -> Callable[[np.ndarray], float]:
    """Create minimum group allocation constraint: group_loading @ x >= min_weight."""
    return lambda x: group_loading @ x - min_weight


def make_max_constraint(
        group_loading: np.ndarray,
        max_weight: float,
) -> Callable[[np.ndarray], float]:
    """Create maximum group allocation constraint: group_loading @ x <= max_weight."""
    return lambda x: max_weight - group_loading @ x
