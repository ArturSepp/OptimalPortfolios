"""
Risk budgeting portfolio optimisation.

Implements constrained risk budgeting (CRB) where each asset's contribution
to portfolio risk matches a prescribed risk budget:

    RC_i(w) = w_i (Σw)_i / sqrt(w'Σw) = b_i * sqrt(w'Σw)

where RC_i is asset i's risk contribution, b_i is the risk budget, and
Σ is the covariance matrix. The optimisation finds weights w such that
the risk contribution of each asset is proportional to its budget, subject
to portfolio constraints (long-only, weight bounds, group exposures).

The primary solver is the internal CCD / ADMM-CCD implementation of
Richard & Roncalli (2019) in ``risk_budgeting_solver.py``, which supports
box bounds and linear inequality constraints on the weights. A scipy SLSQP
fallback is also provided but not recommended for production use.

Special features:
    - Date-varying budgets: rolling allocation accepts either one static budget
      Series or a date-by-asset budget DataFrame.
    - Rebalancing indicators: assets can be frozen at previous weights while
      remaining assets are re-optimised. Frozen assets still contribute to
      portfolio risk but their weights are not changed.
    - Zero risk budgets: assets with b_i = 0 are excluded from the optimisation
      and receive zero weight.
    - NaN-aware filtering: assets with NaN or zero variance in the covariance
      matrix are automatically excluded and receive zero weight.

Reference:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
    Available at https://www.pm-research.com/content/iijpormgmt/52/4/86

Covariance is consumed in caller-supplied variance units; weights and risk budgets are
dimensionless, budgets are normalised by the solver, and no frequency conversion occurs here.
Main entry points are ``rolling_risk_budgeting``, ``wrapper_risk_budgeting``, and
``opt_risk_budgeting``. Boundary: covariance estimation, risk-budget design, and reporting are
outside this module.
"""
from __future__ import division

import warnings
import logging
import numpy as np
import pandas as pd
import qis as qis
from scipy.optimize import minimize
from typing import Dict, Union

from optimalportfolios.utils.portfolio_funcs import (compute_portfolio_variance,
                                                     compute_portfolio_risk_contributions,
                                                     compute_portfolio_risk_contribution_outputs)
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.utils.weights_drift import apply_drift_to_weights_0
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.risk_allocation.risk_budgeting_solver import (
    solve_constrained_risk_budgeting)
from optimalportfolios.optimization.solver_diagnostics import (
    validate_scipy_solution, validate_rb_solution)

logger = logging.getLogger(__name__)


def rolling_risk_budgeting(prices: pd.DataFrame,
                           constraints: Constraints,
                           risk_budget: Union[pd.Series, pd.DataFrame],
                           covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                           rebalancing_indicators: pd.DataFrame = None,
                           optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                           ) -> pd.DataFrame:
    """
    Compute rolling risk-budgeted portfolios at each rebalancing date.

    At each date in ``covar_dict``, solves the constrained risk budgeting
    problem using the pre-computed covariance matrix. The risk budget
    specifies the target fraction of portfolio risk contributed by each asset.

    Args:
        prices: Asset price panel. Used for column alignment.
        constraints: Portfolio constraints.
        risk_budget: Static target budgets as an asset-indexed Series, or point-in-time
            budgets as a date-by-asset DataFrame. Assets with budget 0 are excluded.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        rebalancing_indicators: Optional binary DataFrame for position freezing.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    # Single-asset explicit budget: trivial 100% allocation at every rebalancing
    # date. With risk_budget=None (equal budgets) the full path handles any
    # universe size, including a single asset.
    if risk_budget is not None and not isinstance(risk_budget, (pd.Series, pd.DataFrame)):
        raise TypeError("risk_budget must be a pandas Series, DataFrame, or None")
    if isinstance(risk_budget, pd.Series) and not risk_budget.index.is_unique:
        raise ValueError("risk_budget asset labels must be unique")
    if isinstance(risk_budget, pd.DataFrame):
        if not risk_budget.index.is_unique:
            raise ValueError("risk_budget observation labels must be unique")
        if not risk_budget.columns.is_unique:
            raise ValueError("risk_budget asset labels must be unique")
        missing_dates = pd.Index(covar_dict).difference(risk_budget.index)
        if not missing_dates.empty:
            raise ValueError(
                "risk_budget is missing covariance dates: "
                f"{missing_dates[:5].tolist()}"
            )

    if isinstance(risk_budget, pd.Series) and len(risk_budget) == 1:
        asset = risk_budget.index[0]
        weights = pd.DataFrame(1.0,
                               index=pd.DatetimeIndex(list(covar_dict.keys())),
                               columns=[asset])
        return weights.reindex(columns=prices.columns.to_list()).fillna(0.0)

    if rebalancing_indicators is not None:
        rebalancing_dates = list(covar_dict.keys())
        rebalancing_indicators = rebalancing_indicators.reindex(index=rebalancing_dates).fillna(0.0)

    weights = {}
    weights_0 = None
    prev_date = None
    for date, pd_covar in covar_dict.items():
        if isinstance(risk_budget, pd.DataFrame):
            risk_budget_t = risk_budget.loc[date]
        else:
            risk_budget_t = risk_budget
        if rebalancing_indicators is not None and weights_0 is not None:
            rebalancing_indicators_t = rebalancing_indicators.loc[date, :]
        else:
            rebalancing_indicators_t = None
        # align covariance to risk budget ordering (no-op with equal budgets)
        if risk_budget_t is not None:
            pd_covar = pd_covar.reindex(index=risk_budget_t.index).reindex(
                columns=risk_budget_t.index)
        # drift weights_0 to current date (no-op when prices/prev_date missing)
        weights_0 = apply_drift_to_weights_0(
            weights_0=weights_0, prices=prices,
            prev_date=prev_date, date=date,
            use_drifted_weights_0=optimiser_config.use_drifted_weights_0,
        )
        weights_ = wrapper_risk_budgeting(pd_covar=pd_covar,
                                          constraints=constraints,
                                          weights_0=weights_0,
                                          risk_budget=risk_budget_t,
                                          rebalancing_indicators=rebalancing_indicators_t,
                                          optimiser_config=optimiser_config,
                                          context=str(pd.Timestamp(date).date()))
        weights_0 = weights_  # warm-start next period
        prev_date = date
        weights[date] = weights_
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list()).fillna(0.0)
    return weights


def wrapper_risk_budgeting(pd_covar: pd.DataFrame,
                           constraints: Constraints,
                           weights_0: pd.Series = None,
                           risk_budget: Union[pd.Series, Dict[str, float]] = None,
                           rebalancing_indicators: pd.Series = None,
                           optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True),
                           detailed_output: bool = False,
                           context: str = ''
                           ) -> Union[pd.Series, pd.DataFrame]:
    """
    Single-date risk budgeting with NaN filtering and rebalancing controls.

    Handles three layers of asset filtering:

    1. **Zero risk budgets** (b_i = 0): asset excluded, receives zero weight.
    2. **Rebalancing indicators** (rebal_i = 0): asset frozen at previous weight.
    3. **NaN/zero variance**: asset excluded via covariance filtering.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        constraints: Portfolio constraints.
        weights_0: Previous-period weights for warm-start / fallback / freezing.
        risk_budget: Target risk budgets. Dict or pd.Series.
        rebalancing_indicators: Binary series for position freezing.
        optimiser_config: Solver configuration.
        detailed_output: If True, return DataFrame with risk contribution diagnostics.

    Returns:
        Portfolio weights as pd.Series (or DataFrame if detailed_output=True).
    """
    # assets with zero risk budgets are excluded from optimisation
    if risk_budget is not None:
        if isinstance(risk_budget, dict):
            risk_budget = pd.Series(risk_budget)
        elif isinstance(risk_budget, pd.Series):
            pass
        else:
            raise NotImplementedError(f"{type(risk_budget)}")
        inclusion_indicators = pd.Series(np.where(risk_budget.fillna(0.0) > 0.0, 1.0, 0.0), index=risk_budget.index)
    else:
        inclusion_indicators = pd.Series(1.0, index=pd_covar.columns)

    # handle frozen assets: fix their weights at weights_0 and exclude from optimisation
    if rebalancing_indicators is not None and weights_0 is not None:
        rebalancing_indicators = rebalancing_indicators.reindex(index=inclusion_indicators.index).fillna(1.0)
        weights_0 = weights_0.reindex(index=inclusion_indicators.index).fillna(0.0)
        fixed_weights = weights_0.where(np.isclose(rebalancing_indicators, 0.0), other=0.0)
        inclusion_indicators = inclusion_indicators.where(np.isclose(rebalancing_indicators, 1.0), other=0.0)
    else:
        fixed_weights = None

    # filter covariance for NaN/zero-variance assets
    vectors = dict(min_weights=constraints.min_weights, max_weights=constraints.max_weights, risk_budget=risk_budget)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors,
                                                                  inclusion_indicators=inclusion_indicators)

    if len(clean_covar.columns) == 0:
        warnings.warn("wrapper_risk_budgeting: no valid assets in covariance matrix, returning zero weights")
        return pd.Series(0.0, index=pd_covar.index)

    # rescale risk budgets for reduced universe
    if optimiser_config.apply_total_to_good_ratio:
        n_eligible = int(inclusion_indicators.sum())
        n_valid = len(clean_covar.columns)
        total_to_good_ratio1 = n_eligible / n_valid if n_valid > 0 else 1.0
        total_to_good_ratio = total_to_good_ratio1
    else:
        total_to_good_ratio1 = 1.0
        total_to_good_ratio = None

    if risk_budget is not None:
        risk_budget = risk_budget.loc[clean_covar.columns].fillna(0.0)
        risk_budget *= total_to_good_ratio1
        risk_budget_np = risk_budget.to_numpy()
    else:
        risk_budget_np = None

    constraints1 = constraints.update_with_valid_tickers(context=context, valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=total_to_good_ratio,
                                                         weights_0=weights_0,
                                                         rebalancing_indicators=None)

    weights0 = opt_risk_budgeting(covar=clean_covar.to_numpy(),
                                  constraints=constraints1,
                                  risk_budget=risk_budget_np,
                                  verbose=optimiser_config.verbose,
                                  context=context)
    weights0[np.isinf(weights0)] = 0.0
    weights = pd.Series(weights0, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)

    # re-integrate frozen assets: rescale solved weights to fill remaining allocation
    if fixed_weights is not None:
        left_allocation = 1.0 - np.nansum(fixed_weights)
        sum_solved = np.nansum(weights)
        if sum_solved > 0.0:
            weights = weights * left_allocation / np.nansum(weights)
        weights = weights.where(np.isclose(inclusion_indicators, 1.0), other=fixed_weights)

    if detailed_output:
        df = compute_portfolio_risk_contribution_outputs(weights=weights, clean_covar=clean_covar, risk_budget=risk_budget)
    else:
        df = weights

    return df


def opt_risk_budgeting(covar: np.ndarray,
                       constraints: Constraints,
                       risk_budget: np.ndarray = None,
                       verbose: bool = False,
                       context: str = ''
                       ) -> np.ndarray:
    """
    Solve constrained risk budgeting using the internal CCD / ADMM-CCD solver.

    Args:
        covar: Covariance matrix (N x N).
        constraints: Portfolio constraints.
        risk_budget: Target risk budgets (N,). If None, equal budgets used.
        verbose: If True, print constraint slack diagnostics after solving.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
    """
    n = covar.shape[0]
    if risk_budget is None:
        risk_budget = np.ones(n) / n

    bounds, c_rows, c_lhs = constraints.set_pyrb_constraints(covar=covar)

    try:
        optimal_weights, _lambda_star = solve_constrained_risk_budgeting(covar=covar,
                                                                         budgets=risk_budget,
                                                                         bounds=bounds,
                                                                         c_rows=c_rows,
                                                                         c_lhs=c_lhs)
    except ValueError as exc:
        tag = f"[{context}] " if context else ""
        logger.warning(f"{tag}opt_risk_budgeting: solver failed ({exc})")
        optimal_weights = None

    if verbose and optimal_weights is not None and c_rows is not None:
        slack = c_rows @ optimal_weights - c_lhs
        print(f"slack={slack}")

    optimal_weights, _is_valid = validate_rb_solution(
        optimal_weights, constraints, n,
        c_rows=c_rows, c_lhs=c_lhs, context=context)

    return optimal_weights


def opt_risk_budgeting_scipy(covar: np.ndarray,
                             constraints: Constraints,
                             risk_budget: np.ndarray = None,
                             context: str = ''
                             ) -> np.ndarray:
    """
    Risk budgeting via scipy SLSQP (fallback solver, not recommended).

    Args:
        covar: Covariance matrix (N x N).
        constraints: Portfolio constraints.
        risk_budget: Target risk budgets (N,). If None, equal budgets used.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros if not solved.
    """
    n = covar.shape[0]
    if constraints.weights_0 is not None:
        x0 = constraints.weights_0.to_numpy()
    elif risk_budget is not None:
        x0 = risk_budget
    else:
        x0 = np.ones(n) / n

    if risk_budget is None:
        risk_budget = np.ones(n) / n

    constraints_, bounds = constraints.set_scipy_constraints(covar=covar)

    risk_budget = np.where(np.isclose(risk_budget, 0.0), np.nan, risk_budget)
    options = {'ftol': 1e-8, 'maxiter': 200}

    res = minimize(risk_budget_objective, x0, args=[covar, risk_budget], method='SLSQP',
                  constraints=constraints_, bounds=bounds, options=options)

    optimal_weights, _is_valid = validate_scipy_solution(
        res.x, res, constraints, n, solver='SLSQP', context=context)

    return optimal_weights


def risk_budget_objective(x, pars) -> float:
    """Risk budget deviation objective for scipy minimisation."""
    covar, budget = pars[0], pars[1]
    asset_rc = compute_portfolio_risk_contributions(x, covar)
    sig_p = np.sqrt(compute_portfolio_variance(x, covar))
    if budget is not None:
        risk_target = np.where(np.isnan(budget), asset_rc, np.multiply(sig_p, budget))
    else:
        risk_target = np.multiply(sig_p, np.ones_like(asset_rc) / asset_rc.shape[0])
    sse = np.nanmean(np.square(asset_rc - risk_target))
    return sse


_INVERSE_MEAN_WEIGHT_TOL = 1e-4
_INVERSE_MAX_WEIGHT_TOL = 1e-3


def _scale_to_box_simplex(values: np.ndarray,
                          lower_bounds: np.ndarray,
                          upper_bounds: np.ndarray
                          ) -> np.ndarray:
    """Scale values proportionally onto a unit simplex with box bounds."""
    if np.sum(lower_bounds) > 1.0 or np.sum(upper_bounds) < 1.0:
        raise ValueError(
            "inverse risk-budget bounds are infeasible: their sums do not contain 1.0")

    active = upper_bounds > lower_bounds
    positive_floor = np.where(lower_bounds > 0.0, lower_bounds, 1e-12)
    values = np.where(active, np.maximum(values, positive_floor), lower_bounds)
    low = 0.0
    high = float(np.max(np.divide(
        upper_bounds, values, out=np.zeros_like(values), where=values > 0.0)))
    for _ in range(100):
        midpoint = 0.5 * (low + high)
        scaled = np.clip(midpoint * values, lower_bounds, upper_bounds)
        if np.sum(scaled) < 1.0:
            low = midpoint
        else:
            high = midpoint

    return np.clip(0.5 * (low + high) * values, lower_bounds, upper_bounds)


def _evaluate_inverse_risk_budget(prices: pd.DataFrame,
                                  given_weights: np.ndarray,
                                  covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                  risk_budgets: np.ndarray
                                  ) -> tuple[float, float, np.ndarray]:
    """Return errors and average weights for candidate risk budgets."""
    risk_budget_weights = rolling_risk_budgeting(
        prices=prices,
        covar_dict=covar_dict,
        risk_budget=pd.Series(risk_budgets, index=prices.columns),
        constraints=Constraints(is_long_only=True))
    average_weights = risk_budget_weights.mean(axis=0).reindex(prices.columns).to_numpy()
    if not np.all(np.isfinite(average_weights)):
        return np.inf, np.inf, average_weights
    errors = np.abs(average_weights - given_weights)
    return float(np.mean(errors)), float(np.max(errors)), average_weights


def _solve_inverse_risk_budget_fixed_point(
        prices: pd.DataFrame,
        given_weights: np.ndarray,
        covar_dict: Dict[pd.Timestamp, pd.DataFrame],
        initial_risk_budgets: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        max_iterations: int = 50
        ) -> tuple[np.ndarray, float, float, int]:
    """Calibrate inverse budgets with bounded multiplicative fixed-point updates."""
    risk_budgets = _scale_to_box_simplex(
        initial_risk_budgets, lower_bounds, upper_bounds)
    best_budgets = risk_budgets.copy()
    best_mean_error = np.inf
    best_max_error = np.inf
    best_iteration = 0

    for iteration in range(1, max_iterations + 1):
        mean_error, max_error, average_weights = _evaluate_inverse_risk_budget(
            prices=prices,
            given_weights=given_weights,
            covar_dict=covar_dict,
            risk_budgets=risk_budgets)
        if mean_error < best_mean_error:
            best_budgets = risk_budgets.copy()
            best_mean_error = mean_error
            best_max_error = max_error
            best_iteration = iteration
        if (mean_error <= _INVERSE_MEAN_WEIGHT_TOL
                and max_error <= _INVERSE_MAX_WEIGHT_TOL):
            break
        if not np.isfinite(mean_error):
            break

        safe_average_weights = np.maximum(average_weights, 1e-12)
        ratios = np.divide(given_weights, safe_average_weights,
                           out=np.ones_like(given_weights), where=given_weights > 0.0)
        ratios = np.clip(ratios, 1e-3, 1e3)
        proposal = _scale_to_box_simplex(
            risk_budgets * np.square(ratios), lower_bounds, upper_bounds)
        risk_budgets = _scale_to_box_simplex(
            0.5 * risk_budgets + 0.5 * proposal, lower_bounds, upper_bounds)

    return best_budgets, best_mean_error, best_max_error, best_iteration


def solve_for_risk_budgets_from_given_weights(prices: pd.DataFrame,
                                              given_weights: pd.Series,
                                              covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                              min_risk_budget: float = 1e-4,
                                              max_risk_budget: float = 0.99
                                              ) -> pd.Series:
    """
    Inverse risk budgeting: find budgets that reproduce given target weights.

    Args:
        prices: Asset price panel.
        given_weights: Target portfolio weights to reproduce.
        covar_dict: Pre-computed covariance matrices.
        min_risk_budget: Lower bound on each non-zero risk budget.
        max_risk_budget: Upper bound on each risk budget.

    Returns:
        Optimal risk budgets as pd.Series. Budgets sum to 1.
    """
    # Single-asset universe: the only budget consistent with sum=1 is 1.0
    # on the lone asset. Skip the solver — it would be infeasible under the
    # max_risk_budget=0.99 cap anyway.
    if prices.shape[1] == 1:
        return pd.Series(1.0, index=prices.columns)

    given_weights = given_weights.reindex(prices.columns)
    given_weights_np = given_weights.to_numpy()
    if (not np.all(np.isfinite(given_weights_np))
            or np.any(given_weights_np < 0.0)
            or not np.isclose(np.sum(given_weights_np), 1.0)):
        raise ValueError(
            "given_weights must be finite, non-negative, aligned to prices, and sum to 1.0")
    if np.count_nonzero(given_weights_np > 0.0) == 1:
        return pd.Series(np.where(given_weights_np > 0.0, 1.0, 0.0), index=prices.columns)

    def objective_function(risk_budgets: np.ndarray) -> float:
        """Mean absolute gap between the backtested average weights and the targets."""
        mean_error, _, _ = _evaluate_inverse_risk_budget(
            prices=prices,
            given_weights=given_weights_np,
            covar_dict=covar_dict,
            risk_budgets=risk_budgets)
        return mean_error

    is_use_avg_rc = True
    if is_use_avg_rc:
        portfolio_rc = {}
        for date, pd_covar in covar_dict.items():
            rc = qis.compute_portfolio_risk_contributions(w=given_weights, covar=pd_covar)
            portfolio_rc[date] = rc / np.nansum(rc)
        avg_portfolio_rc = pd.DataFrame.from_dict(portfolio_rc, orient='index').mean(axis=0)
        x0 = np.nan_to_num(avg_portfolio_rc.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # Not covered, and unreachable as written: `is_use_avg_rc` is assigned the literal True
        # directly above and is never reassigned anywhere in the package, so this branch is dead.
        # Left in place rather than deleted because it records the alternative seeding -- the raw
        # target weights instead of their average risk contributions -- which is a numerical
        # choice, not a refactor to make silently.
        x0 = given_weights.to_numpy()  # pragma: no cover

    enforce_min_max = np.where(np.greater(given_weights_np, 0.0), 1.0, 0.0)
    min_rbs = min_risk_budget * enforce_min_max
    max_rbs = max_risk_budget * enforce_min_max

    fixed_point, fixed_point_mean_error, fixed_point_max_error, iteration = (
        _solve_inverse_risk_budget_fixed_point(
            prices=prices,
            given_weights=given_weights_np,
            covar_dict=covar_dict,
            initial_risk_budgets=x0,
            lower_bounds=min_rbs,
            upper_bounds=max_rbs))
    if (fixed_point_mean_error <= _INVERSE_MEAN_WEIGHT_TOL
            and fixed_point_max_error <= _INVERSE_MAX_WEIGHT_TOL):
        logger.info(
            "solve_for_risk_budgets_from_given_weights: fixed point converged in %s "
            "iterations (mean weight error=%.6g, max weight error=%.6g)",
            iteration, fixed_point_mean_error, fixed_point_max_error)
        return pd.Series(fixed_point, index=prices.columns)

    bounds = [(x, y) for x, y in zip(min_rbs, max_rbs)]
    options = {'ftol': 1e-8, 'maxiter': 100}
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    res = minimize(objective_function, fixed_point, method='SLSQP',
                   constraints=constraints, bounds=bounds, options=options)

    risk_budgets = res.x
    slsqp_error_text = "SLSQP candidate was unavailable"
    if res.success and risk_budgets is not None and np.all(np.isfinite(risk_budgets)):
        mean_error, max_error, _ = _evaluate_inverse_risk_budget(
            prices=prices,
            given_weights=given_weights_np,
            covar_dict=covar_dict,
            risk_budgets=risk_budgets)
        slsqp_error_text = (
            f"SLSQP mean/max weight errors were {mean_error:.6g}/{max_error:.6g}")
        if (mean_error <= _INVERSE_MEAN_WEIGHT_TOL
                and max_error <= _INVERSE_MAX_WEIGHT_TOL):
            return pd.Series(risk_budgets, index=prices.columns)

    raise RuntimeError(
        "inverse risk-budget calibration failed: fixed-point best mean/max weight errors "
        f"were {fixed_point_mean_error:.6g}/{fixed_point_max_error:.6g}; "
        f"SLSQP status={res.status}: {res.message}; {slsqp_error_text}. "
        "No zero risk-budget fallback was returned.")
