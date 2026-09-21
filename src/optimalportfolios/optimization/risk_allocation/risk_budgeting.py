"""
Risk budgeting portfolio optimisation.

Implements risk budgeting (RB), targeting each asset's prescribed risk contribution:

    RC_i(w) = w_i (Σw)_i / sqrt(w'Σw) = b_i * sqrt(w'Σw)

where RC_i is asset i's risk contribution, b_i is the risk budget, and
Σ is the covariance matrix. Without binding constraints the optimisation reproduces
these budgets. With binding bounds it minimizes log(sigma(w)) - sum(b_i log(w_i))
on the fully invested feasible set; risk contributions need not equal the budgets.

The primary solver is the scale-consistent CCD / ADMM-CCD formulation in
``risk_budgeting_solver.py``, which represents instrument bounds, group bounds
and full investment jointly. The separate scipy SLSQP entry point is not an
automatic fallback and is not recommended for production use.

Special features:
    - Date-varying budgets: rolling allocation accepts either one static budget
      Series or a date-by-asset budget DataFrame.
    - Rebalancing indicators: assets can be frozen at previous weights while
      remaining assets are re-optimised. Frozen assets still contribute to
      portfolio risk but their weights are not changed.
    - Zero risk budgets: assets with b_i = 0 are excluded unless an explicit
      positive min=max weight pins them inside the full-covariance solve.
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
from typing import Dict, Optional, Sequence, Union

from optimalportfolios.utils.portfolio_funcs import (compute_portfolio_variance,
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
                           optimiser_config: OptimiserConfig = OptimiserConfig(
                               apply_total_to_good_ratio=True)
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
            budgets as a date-by-asset DataFrame. Zero-budget assets are excluded
            unless a positive instrument bound pins their weight.
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
                           optimiser_config: OptimiserConfig = OptimiserConfig(
                               apply_total_to_good_ratio=True),
                           detailed_output: bool = False,
                           context: str = ''
                           ) -> Union[pd.Series, pd.DataFrame]:
    """
    Single-date risk budgeting with NaN filtering and rebalancing controls.

    Handles three layers of asset filtering:

    1. **Zero risk budgets** (b_i = 0): asset excluded, except when an explicit
       positive min=max instrument bound fixes its weight in the joint solve.
    2. **Rebalancing indicators** (rebal_i = 0): asset frozen at previous weight.
    3. **NaN/non-positive variance**: asset excluded via covariance filtering.

    Remaining positive variances are floored at ``0.001**2`` (0.1% volatility
    for annualised covariance). Off-diagonal covariances are unchanged. This also
    applies to rolling allocations and inverse calibration through this wrapper.

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
    # A positive exact instrument pin overrides zero-budget exclusion. Keep the
    # pinned asset in the same covariance solve so its cross-covariances affect
    # the freely allocated assets; a rebalancing freeze would not do this.
    if risk_budget is not None:
        if isinstance(risk_budget, dict):
            risk_budget = pd.Series(risk_budget)
        elif isinstance(risk_budget, pd.Series):
            pass
        else:
            raise NotImplementedError(f"{type(risk_budget)}")
        inclusion_indicators = pd.Series(
            np.where(risk_budget.fillna(0.0) > 0.0, 1.0, 0.0), index=risk_budget.index
        )
        if constraints.min_weights is not None and constraints.max_weights is not None:
            minimum = constraints.min_weights.reindex(risk_budget.index)
            maximum = constraints.max_weights.reindex(risk_budget.index)
            pinned = (minimum.gt(0.0) & np.isclose(minimum, maximum, atol=1e-12))
            inclusion_indicators.loc[pinned] = 1.0
    else:
        inclusion_indicators = pd.Series(1.0, index=pd_covar.columns)

    # handle frozen assets: fix their weights at weights_0 and exclude from optimisation
    if rebalancing_indicators is not None and weights_0 is not None:
        rebalancing_indicators = rebalancing_indicators.reindex(
            index=inclusion_indicators.index
        ).fillna(1.0)
        weights_0 = weights_0.reindex(index=inclusion_indicators.index).fillna(0.0)
        fixed_weights = weights_0.where(np.isclose(rebalancing_indicators, 0.0), other=0.0)
        inclusion_indicators = inclusion_indicators.where(
            np.isclose(rebalancing_indicators, 1.0), other=0.0
        )
    else:
        fixed_weights = None

    # Filter invalid assets and floor cash-like variances for risk budgeting only.
    vectors = dict(
        min_weights=constraints.min_weights,
        max_weights=constraints.max_weights,
        risk_budget=risk_budget,
    )
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(
        pd_covar=pd_covar, vectors=vectors,
        inclusion_indicators=inclusion_indicators, variance_floor=0.001**2)

    if len(clean_covar.columns) == 0:
        warnings.warn(
            "wrapper_risk_budgeting: no valid assets in covariance matrix, returning zero weights"
        )
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

    constraints1 = constraints.update_with_valid_tickers(
        context=context,
        valid_tickers=clean_covar.columns.to_list(),
        total_to_good_ratio=total_to_good_ratio,
        weights_0=weights_0,
        rebalancing_indicators=None,
    )

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
        df = compute_portfolio_risk_contribution_outputs(
            weights=weights, clean_covar=clean_covar, risk_budget=risk_budget
        )
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
    asset_rc = qis.compute_portfolio_risk_contributions(w=x, covar=covar)
    sig_p = np.sqrt(compute_portfolio_variance(x, covar))
    if budget is not None:
        risk_target = np.where(np.isnan(budget), asset_rc, np.multiply(sig_p, budget))
    else:
        risk_target = np.multiply(sig_p, np.ones_like(asset_rc) / asset_rc.shape[0])
    sse = np.nanmean(np.square(asset_rc - risk_target))
    return sse


_INVERSE_MEAN_WEIGHT_TOL = 1e-4
_INVERSE_MAX_WEIGHT_TOL = 1e-3
# A targeted asset whose marginal risk contribution is negative on at least this share of
# covariance dates is flagged before the fit: it hedges the target portfolio there, and a
# non-negative budget cannot hold it on those dates.
_NEGATIVE_RC_SHARE_WARNING = 0.5
# Retained for callers that explicitly opt into the former 12-rebalance span;
# it is no longer the inverse-fit default.
INVERSE_EWMA_SPAN = 12


def average_rolling_weights(weights: pd.DataFrame,
                            ewma_span: Optional[float] = None
                            ) -> pd.Series:
    """average a rolling weight path over its rebalance dates

    With ``ewma_span=None`` the result is the simple mean over dates. Otherwise it is the
    last value of the ``qis.compute_ewm`` recursion seeded at the first row
    (``InitType.X0``), with lambda = 1 - 2 / (ewma_span + 1):

        m_0 = w_0,   m_t = lambda m_{t-1} + (1 - lambda) w_t,   w̄ = m_{T-1}

    so row t carries weight (1 - lambda) lambda^(T-1-t) and the first row the residual
    lambda^(T-1); the recursion is causal at every step. Rows are taken in index order, so
    the path must be sorted by date; a NaN row is carried forward by the recursion (qis
    ``NanBackfill.FFILL``).

    Args:
        weights: Weight path indexed by rebalance date, one column per asset.
        ewma_span: Exponential span in rebalances, or None (default) for the simple mean.

    Returns:
        Averaged weights indexed by the columns of ``weights``.

    Raises:
        ValueError: If ``ewma_span`` is not None and not a finite positive number.
    """
    if ewma_span is None:
        return weights.mean(axis=0)
    if not np.isfinite(ewma_span) or ewma_span <= 0.0:
        raise ValueError(
            f"ewma_span must be None or a finite positive number, got {ewma_span!r}")
    if weights.isna().all().all():
        return weights.mean(axis=0)  # nothing to seed the recursion with: all-NaN, as the mean
    return qis.compute_ewm(data=weights, span=ewma_span, init_type=qis.InitType.X0).iloc[-1]


def _target_risk_contributions(given_weights: pd.Series,
                               covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                               ewma_span: Optional[float]
                               ) -> pd.DataFrame:
    """risk-contribution shares of the target weights over the covariance dates

    On each date the shares are RC_i / Σ_j RC_j with RC_i = w_i (Σ w)_i at the target
    weights w, so a negative share is a negative marginal risk contribution (Σ w)_i: the
    asset reduces the risk of the target portfolio on that date.

    Args:
        given_weights: Target weights, indexed by asset.
        covar_dict: Covariance matrices keyed by rebalance date in order.
        ewma_span: Span for the averaged share, as in the inverse fit; None for the mean.

    Returns:
        Frame indexed by asset with ``target_weight``, ``average_rc`` (the share averaged
        over dates with ``average_rolling_weights``) and ``negative_rc_share`` (the
        fraction of dates with a negative share).
    """
    shares = {}
    for date, pd_covar in covar_dict.items():
        rc = qis.compute_portfolio_risk_contributions(w=given_weights, covar=pd_covar)
        shares[date] = rc / np.nansum(rc)
    shares_by_date = pd.DataFrame.from_dict(shares, orient='index').sort_index()
    shares_by_date = shares_by_date.reindex(columns=given_weights.index)
    return pd.DataFrame({
        'target_weight': given_weights,
        'average_rc': average_rolling_weights(weights=shares_by_date, ewma_span=ewma_span),
        'negative_rc_share': shares_by_date.lt(0.0).mean(axis=0)})


def _describe_target_risk_contributions(diagnostics: pd.DataFrame) -> str:
    """one line per targeted asset with a negative marginal contribution on any date"""
    flagged = diagnostics[(diagnostics['target_weight'] > 0.0)
                          & (diagnostics['negative_rc_share'] > 0.0)]
    if flagged.empty:
        return 'no targeted asset has a negative marginal risk contribution on any date'
    flagged = flagged.sort_values('negative_rc_share', ascending=False)
    lines = [f"{asset}: target weight {row.target_weight:.4f}, averaged risk contribution "
             f"{row.average_rc:+.4f}, marginal contribution negative on "
             f"{100.0 * row.negative_rc_share:.0f}% of dates"
             for asset, row in flagged.iterrows()]
    return 'targeted assets with a negative marginal risk contribution: ' + '; '.join(lines)


def _identify_inverse_fixed_weights(
        given_weights: pd.Series,
        covar_dict: Dict[pd.Timestamp, pd.DataFrame],
        ewma_span: Optional[float],
) -> tuple[pd.Series, pd.DataFrame]:
    """Identify non-positive average-RC targets to hold at their original weights."""
    diagnostics = _target_risk_contributions(
        given_weights=given_weights, covar_dict=covar_dict, ewma_span=ewma_span)
    targeted = diagnostics[diagnostics['target_weight'] > 0.0]
    if not np.isfinite(targeted['average_rc'].to_numpy()).all():
        raise ValueError(
            'inverse risk-budget calibration: average target risk contributions '
            'must be finite on the covariance dates')
    non_positive = targeted[targeted['average_rc'] <= 0.0]
    if len(non_positive) == len(targeted):
        raise ValueError(
            'inverse risk-budget calibration: no positive-risk target assets remain '
            'after pinning non-positive average marginal contributions')
    if not non_positive.empty:
        details = '; '.join(
            f'{asset}: central weight fixed at {row.target_weight:.4f}, '
            f'risk budget set to 0 (average risk contribution {row.average_rc:+.4f})'
            for asset, row in non_positive.iterrows())
        warnings.warn(
            'inverse risk-budget calibration pinned assets with non-positive '
            'average marginal risk contributions; their weights remain at the '
            f'original central weights: {details}',
            UserWarning,
            stacklevel=2,
        )
    return given_weights.where(given_weights.index.isin(non_positive.index), 0.0), diagnostics


def _check_target_risk_contributions(diagnostics: pd.DataFrame,
                                     fixed_assets: pd.Index = pd.Index([])) -> None:
    """Reject unfixed non-positive contributions; warn on intermittent hedges.

    A risk-budget portfolio holds asset i only where (Σ w)_i > 0, because
    w_i (Σ w)_i = b_i σ_p² with b_i ≥ 0. A targeted asset whose averaged risk
    contribution at the target weights is not positive therefore has no admissible
    budget. Fixed-weight assets instead keep their target weight with zero budget.
    An asset negative on a majority of dates is only warned about: the averaged
    path may still reach the target.

    Raises:
        ValueError: If a targeted asset has a non-positive averaged risk contribution.
    """
    targeted = diagnostics[(diagnostics['target_weight'] > 0.0)
                           & ~diagnostics.index.isin(fixed_assets)]
    not_reproducible = targeted[targeted['average_rc'] <= 0.0]
    if not not_reproducible.empty:
        raise ValueError(
            'inverse risk-budget calibration cannot reproduce the given weights: the '
            'marginal risk contribution of '
            f"{', '.join(map(str, not_reproducible.index))} at the target weights is not "
            'positive on average, so no non-negative risk budget holds the asset there '
            '(the asset hedges the target portfolio). '
            + _describe_target_risk_contributions(diagnostics))
    hedging = targeted[targeted['negative_rc_share'] >= _NEGATIVE_RC_SHARE_WARNING]
    if not hedging.empty:
        warnings.warn(
            'inverse risk-budget calibration: '
            f"{', '.join(map(str, hedging.index))} has a negative marginal risk "
            'contribution at the target weights on at least '
            f'{100.0 * _NEGATIVE_RC_SHARE_WARNING:.0f}% of dates; a non-negative budget '
            'cannot hold it on those dates and the fit may not converge. '
            + _describe_target_risk_contributions(diagnostics))


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
                                  risk_budgets: np.ndarray,
                                  ewma_span: Optional[float] = None,
                                  fixed_weights: Optional[pd.Series] = None
                                  ) -> tuple[float, float, np.ndarray]:
    """Return errors and averaged weights for candidate risk budgets."""
    if fixed_weights is None or fixed_weights.empty:
        constraints = Constraints(is_long_only=True)
    else:
        minimum = pd.Series(0.0, index=prices.columns)
        maximum = pd.Series(1.0, index=prices.columns)
        minimum.loc[fixed_weights.index] = fixed_weights
        maximum.loc[fixed_weights.index] = fixed_weights
        constraints = Constraints(is_long_only=True, min_weights=minimum,
                                  max_weights=maximum)
    risk_budget_weights = rolling_risk_budgeting(
        prices=prices,
        covar_dict=covar_dict,
        risk_budget=pd.Series(risk_budgets, index=prices.columns),
        constraints=constraints)
    average_weights = average_rolling_weights(
        weights=risk_budget_weights, ewma_span=ewma_span).reindex(prices.columns).to_numpy()
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
        max_iterations: int = 50,
        ewma_span: Optional[float] = None,
        fixed_weights: Optional[pd.Series] = None
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
            risk_budgets=risk_budgets,
            ewma_span=ewma_span,
            fixed_weights=fixed_weights)
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


def _inverse_boundary_overweights(budgets: np.ndarray,
                                  average_weights: np.ndarray,
                                  target_weights: pd.Series,
                                  lower_bounds: np.ndarray,
                                  active: pd.Series) -> pd.Series:
    """Rank free assets overweight despite a fitted budget at its lower bound."""
    budget = pd.Series(budgets, index=target_weights.index)
    average = pd.Series(average_weights, index=target_weights.index)
    floor = pd.Series(lower_bounds, index=target_weights.index)
    floor_slack = np.maximum(1e-10, 1e-4 * floor)
    overweight = average - target_weights
    candidates = active & budget.le(floor + floor_slack) & overweight.gt(
        _INVERSE_MAX_WEIGHT_TOL)
    return overweight[candidates].sort_values(ascending=False)


def _probe_inverse_budget_floor(asset: str,
                                budgets: np.ndarray,
                                lower_bounds: np.ndarray,
                                upper_bounds: np.ndarray,
                                prices: pd.DataFrame,
                                given_weights: np.ndarray,
                                covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                ewma_span: Optional[float],
                                fixed_weights: pd.Series) -> Optional[float]:
    """Return an asset's forward average with its budget forced to the positive floor."""
    position = prices.columns.get_loc(asset)
    probe_upper = upper_bounds.copy()
    probe_upper[position] = lower_bounds[position]
    if np.sum(probe_upper) < 1.0:
        return None
    probe_budgets = _scale_to_box_simplex(budgets, lower_bounds, probe_upper)
    _, _, average_weights = _evaluate_inverse_risk_budget(
        prices=prices, given_weights=given_weights, covar_dict=covar_dict,
        risk_budgets=probe_budgets, ewma_span=ewma_span,
        fixed_weights=fixed_weights)
    return float(average_weights[position])


def solve_for_risk_budgets_from_given_weights(prices: pd.DataFrame,
                                              given_weights: pd.Series,
                                              covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                              min_risk_budget: float = 1e-4,
                                              max_risk_budget: float = 0.99,
                                              ewma_span: Optional[float] = None,
                                              fixed_weight_assets: Optional[Sequence[str]] = None
                                              ) -> pd.Series:
    """
    Inverse risk budgeting: find budgets that reproduce given target weights.

    The candidate budgets are run through the long-only rolling risk-budget solve over
    ``covar_dict`` and the resulting weight path is averaged over its rebalance dates with
    ``average_rolling_weights``; the budgets are fitted so that this average matches
    ``given_weights``. The default is the simple mean over the whole path; an
    explicit ``ewma_span`` emphasizes recent rebalances. A positive target asset
    with non-positive average risk contribution, or one explicitly named in
    ``fixed_weight_assets``, is pinned at its original central weight in every
    forward solve using the full covariance and assigned zero reported budget.
    If the remaining fit fails, a free overweight asset is probed at its minimum
    positive budget, even if the inverse search stopped above that boundary.
    It is trial-pinned only when that probe remains overweight, and accepted
    only if the complete refit meets both weight tolerances. Only the budgets
    of the other assets are fitted.

    Args:
        prices: Asset price panel.
        given_weights: Target portfolio weights to reproduce.
        covar_dict: Pre-computed covariance matrices, keyed by rebalance date in order.
        min_risk_budget: Lower bound on each non-zero risk budget.
        max_risk_budget: Upper bound on each risk budget.
        ewma_span: Exponential span, in rebalances, for averaging the weight path;
            None (default) for the simple mean.
        fixed_weight_assets: Additional positive-weight asset labels to pin at
            their target weights, even if their average contribution is positive.

    Returns:
        Optimal risk budgets as pd.Series. Budgets sum to 1.

    Raises:
        ValueError: If ``given_weights`` are invalid, ``ewma_span`` is not positive,
            or no admissible positive-risk target remains after pinning.
        RuntimeError: If no candidate reproduces the averaged target weights. The message
            lists the targeted assets whose marginal risk contribution is negative on some
            dates, with the share of such dates.
    """
    # Single-asset universe: the only budget consistent with sum=1 is 1.0
    # on the lone asset. Skip the solver — it would be infeasible under the
    # max_risk_budget=0.99 cap anyway.
    if prices.shape[1] == 1:
        if fixed_weight_assets:
            raise ValueError('the sole target asset cannot have a zero risk budget')
        return pd.Series(1.0, index=prices.columns)

    given_weights = given_weights.reindex(prices.columns)
    given_weights_np = given_weights.to_numpy()
    if (not np.all(np.isfinite(given_weights_np))
            or np.any(given_weights_np < 0.0)
            or not np.isclose(np.sum(given_weights_np), 1.0)):
        raise ValueError(
            "given_weights must be finite, non-negative, aligned to prices, and sum to 1.0")
    if isinstance(fixed_weight_assets, str):
        raise TypeError('fixed_weight_assets must be a sequence of asset labels, not a string')
    explicit_fixed = pd.Index(() if fixed_weight_assets is None else tuple(fixed_weight_assets))
    if not explicit_fixed.is_unique:
        raise ValueError('fixed_weight_assets must not contain duplicate labels')
    unknown = explicit_fixed.difference(prices.columns)
    if not unknown.empty:
        raise ValueError(f'fixed_weight_assets are not in prices: {unknown.tolist()}')
    zero_target = explicit_fixed[given_weights.reindex(explicit_fixed).le(0.0)]
    if not zero_target.empty:
        raise ValueError(
            f'fixed_weight_assets must have positive target weights: {zero_target.tolist()}')
    if np.count_nonzero(given_weights_np > 0.0) == 1 and explicit_fixed.empty:
        return pd.Series(np.where(given_weights_np > 0.0, 1.0, 0.0), index=prices.columns)

    # Pin hedging assets without changing any target weight. In particular, do
    # not redistribute their central weights onto the active risk-budget sleeve.
    fixed_weight_targets, rc_diagnostics = _identify_inverse_fixed_weights(
        given_weights=given_weights, covar_dict=covar_dict, ewma_span=ewma_span)
    fixed_weight_targets.loc[explicit_fixed] = given_weights.loc[explicit_fixed]
    fixed_weights = fixed_weight_targets[fixed_weight_targets > 0.0]
    active = given_weights.gt(0.0) & ~given_weights.index.isin(fixed_weights.index)
    if not active.any():
        raise ValueError('inverse risk-budget calibration needs at least one unfixed target asset')
    if active.sum() == 1:
        return pd.Series(np.where(active, 1.0, 0.0), index=prices.columns)

    def objective_function(risk_budgets: np.ndarray) -> float:
        """Mean absolute gap between the backtested average weights and the targets."""
        mean_error, _, _ = _evaluate_inverse_risk_budget(
            prices=prices,
            given_weights=given_weights_np,
            covar_dict=covar_dict,
            risk_budgets=risk_budgets,
            ewma_span=ewma_span,
            fixed_weights=fixed_weights)
        return mean_error

    # Risk contributions of the original target seed the search and diagnose
    # intermittent hedges even when their average contribution remains positive.
    _check_target_risk_contributions(rc_diagnostics, fixed_assets=fixed_weights.index)
    is_use_avg_rc = True
    if is_use_avg_rc:
        x0 = np.nan_to_num(rc_diagnostics['average_rc'].to_numpy(),
                           nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # Not covered, and unreachable as written: `is_use_avg_rc` is assigned the literal True
        # directly above and is never reassigned anywhere in the package, so this branch is dead.
        # Left in place rather than deleted because it records the alternative seeding -- the raw
        # target weights instead of their average risk contributions -- which is a numerical
        # choice, not a refactor to make silently.
        x0 = given_weights.to_numpy()  # pragma: no cover

    enforce_min_max = active.to_numpy(dtype=float)
    min_rbs = min_risk_budget * enforce_min_max
    max_rbs = max_risk_budget * enforce_min_max

    fixed_point, fixed_point_mean_error, fixed_point_max_error, iteration = (
        _solve_inverse_risk_budget_fixed_point(
            prices=prices,
            given_weights=given_weights_np,
            covar_dict=covar_dict,
            initial_risk_budgets=x0,
            lower_bounds=min_rbs,
            upper_bounds=max_rbs,
            ewma_span=ewma_span,
            fixed_weights=fixed_weights))
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
    best_average_weights = None
    best_budgets = fixed_point
    if res.success and risk_budgets is not None and np.all(np.isfinite(risk_budgets)):
        mean_error, max_error, slsqp_average_weights = _evaluate_inverse_risk_budget(
            prices=prices,
            given_weights=given_weights_np,
            covar_dict=covar_dict,
            risk_budgets=risk_budgets,
            ewma_span=ewma_span,
            fixed_weights=fixed_weights)
        slsqp_error_text = (
            f"SLSQP mean/max weight errors were {mean_error:.6g}/{max_error:.6g}")
        if mean_error <= fixed_point_mean_error:
            best_average_weights = slsqp_average_weights
            best_budgets = risk_budgets
        if (mean_error <= _INVERSE_MEAN_WEIGHT_TOL
                and max_error <= _INVERSE_MAX_WEIGHT_TOL):
            return pd.Series(risk_budgets, index=prices.columns)

    if best_average_weights is None:
        _, _, best_average_weights = _evaluate_inverse_risk_budget(
            prices=prices, given_weights=given_weights_np, covar_dict=covar_dict,
            risk_budgets=fixed_point, ewma_span=ewma_span, fixed_weights=fixed_weights)
    boundary_candidates = _inverse_boundary_overweights(
        budgets=best_budgets, average_weights=best_average_weights,
        target_weights=given_weights, lower_bounds=min_rbs, active=active)
    gaps = pd.Series(best_average_weights - given_weights_np, index=prices.columns)
    overweight_assets = gaps[active & gaps.gt(_INVERSE_MAX_WEIGHT_TOL)].sort_values(
        ascending=False)
    candidate_assets = list(boundary_candidates.index) + [
        asset for asset in overweight_assets.index if asset not in boundary_candidates.index]
    rejected_trials = []
    for asset in candidate_assets:
        if asset in boundary_candidates.index:
            floor_average_weight = float(best_average_weights[prices.columns.get_loc(asset)])
        else:
            floor_average_weight = _probe_inverse_budget_floor(
                asset=asset, budgets=best_budgets, lower_bounds=min_rbs,
                upper_bounds=max_rbs, prices=prices,
                given_weights=given_weights_np, covar_dict=covar_dict,
                ewma_span=ewma_span, fixed_weights=fixed_weights)
        if (floor_average_weight is None or not np.isfinite(floor_average_weight)
                or floor_average_weight <= given_weights.loc[asset] + _INVERSE_MAX_WEIGHT_TOL):
            rejected_trials.append(f'{asset}: floor probe not overweight or infeasible')
            continue
        try:
            with warnings.catch_warnings():
                # The outer call already announced the same average-RC pins.
                warnings.filterwarnings(
                    'ignore', message='inverse risk-budget calibration pinned assets '
                    'with non-positive average marginal risk contributions',
                    category=UserWarning)
                trial_budgets = solve_for_risk_budgets_from_given_weights(
                    prices=prices, given_weights=given_weights, covar_dict=covar_dict,
                    min_risk_budget=min_risk_budget, max_risk_budget=max_risk_budget,
                    ewma_span=ewma_span,
                    fixed_weight_assets=(*explicit_fixed, asset))
        except (RuntimeError, ValueError) as exc:
            rejected_trials.append(f'{asset}: {type(exc).__name__}')
            continue
        trial_fixed_weights = given_weights[
            given_weights.gt(0.0) & trial_budgets.reindex(given_weights.index).eq(0.0)]
        trial_mean_error, trial_max_error, _ = _evaluate_inverse_risk_budget(
            prices=prices, given_weights=given_weights_np, covar_dict=covar_dict,
            risk_budgets=trial_budgets.to_numpy(), ewma_span=ewma_span,
            fixed_weights=trial_fixed_weights)
        if (trial_mean_error <= _INVERSE_MEAN_WEIGHT_TOL
                and trial_max_error <= _INVERSE_MAX_WEIGHT_TOL):
            warnings.warn(
                'inverse risk-budget calibration pinned a positive-average-risk asset '
                f'at the budget boundary: {asset}, target weight '
                f'{given_weights.loc[asset]:.4f}, unpinned fit '
                f'{given_weights.loc[asset] + gaps.loc[asset]:.4f}, '
                f'positive-floor probe {floor_average_weight:.4f}; '
                'its central weight remains fixed with zero reported budget',
                UserWarning, stacklevel=2)
            return trial_budgets
        rejected_trials.append(f'{asset}: refit max error {trial_max_error:.6g}')
    worst = gaps.abs().nlargest(5).index
    gap_details = '; '.join(
        f'{asset}: target {given_weights.loc[asset]:.4f}, fit '
        f'{best_average_weights[prices.columns.get_loc(asset)]:.4f}' for asset in worst)

    raise RuntimeError(
        "inverse risk-budget calibration failed: fixed-point best mean/max weight errors "
        f"were {fixed_point_mean_error:.6g}/{fixed_point_max_error:.6g}; "
        f"SLSQP status={res.status}: {res.message}; {slsqp_error_text}. "
        f"Largest average-weight gaps: {gap_details}. "
        f"Boundary pin trials rejected: {', '.join(rejected_trials) or 'none'}. "
        "No zero risk-budget fallback was returned. "
        + _describe_target_risk_contributions(rc_diagnostics))
