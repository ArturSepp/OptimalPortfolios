"""
Risk budgeting portfolio optimisation.

Implements constrained risk budgeting (CRB) where each asset's contribution
to portfolio risk matches a prescribed risk budget:

    RC_i(w) = w_i (Σw)_i / sqrt(w'Σw) = b_i * sqrt(w'Σw)

where RC_i is asset i's risk contribution, b_i is the risk budget, and
Σ is the covariance matrix. The optimisation finds weights w such that
the risk contribution of each asset is proportional to its budget, subject
to portfolio constraints (long-only, weight bounds, group exposures).

The primary solver uses ``ConstrainedRiskBudgeting`` from the pyrb package,
which supports linear inequality constraints on the weights. A scipy SLSQP
fallback is also provided but not recommended for production use.

Special features:
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
"""
from __future__ import division

import warnings
import numpy as np
import pandas as pd
import qis as qis
from scipy.optimize import minimize
from typing import Dict, Union

from optimalportfolios.utils.portfolio_funcs import (compute_portfolio_variance,
                                                     compute_portfolio_risk_contributions,
                                                     compute_portfolio_risk_contribution_outputs)
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints
from pyrb import ConstrainedRiskBudgeting


def rolling_risk_budgeting(prices: pd.DataFrame,
                           constraints: Constraints,
                           risk_budget: pd.Series,
                           covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                           rebalancing_indicators: pd.DataFrame = None,
                           apply_total_to_good_ratio: bool = True
                           ) -> pd.DataFrame:
    """
    Compute rolling risk-budgeted portfolios at each rebalancing date.

    At each date in ``covar_dict``, solves the constrained risk budgeting
    problem using the pre-computed covariance matrix. The risk budget
    specifies the target fraction of portfolio risk contributed by each asset.

    The covariance matrices are produced externally by any CovarEstimator
    (EwmaCovarEstimator, FactorCovarEstimator, etc.), decoupling the
    estimation step from the optimisation step.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers. Used only
            for column alignment of the output weights DataFrame.
        constraints: Portfolio constraints (long-only, weight bounds, group
            exposures, etc.).
        risk_budget: Target risk budgets per asset. Index=tickers, values=budgets.
            Budgets are relative (need not sum to 1 — they are rescaled internally).
            Assets with budget 0 are excluded from optimisation.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
            Typically produced by ``estimator.fit_rolling_covars()``.
            The dict keys define the rebalancing schedule.
        rebalancing_indicators: Optional binary DataFrame (index=dates, columns=tickers)
            indicating whether each asset is eligible for rebalancing at each date.
            Assets with value 0 are frozen at their previous weights (from weights_0).
            Frozen assets still contribute to portfolio risk via the covariance matrix.
            If None, all assets are rebalanced at every date.
        apply_total_to_good_ratio: If True, rescale risk budgets and constraints
            proportionally when some assets are excluded due to NaN or zero variance.
            This preserves the intended risk allocation across the valid subset.

    Returns:
        DataFrame of portfolio weights. Index=rebalancing dates from covar_dict,
        columns=tickers aligned to ``prices.columns``.
    """
    if rebalancing_indicators is not None:
        rebalancing_dates = list(covar_dict.keys())
        rebalancing_indicators = rebalancing_indicators.reindex(index=rebalancing_dates).fillna(0.0)

    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():
        if rebalancing_indicators is not None and weights_0 is not None:
            rebalancing_indicators_t = rebalancing_indicators.loc[date, :]
        else:
            rebalancing_indicators_t = None
        # align covariance to risk budget ordering
        pd_covar = pd_covar.reindex(index=risk_budget.index).reindex(columns=risk_budget.index)
        weights_ = wrapper_risk_budgeting(pd_covar=pd_covar,
                                          constraints=constraints,
                                          weights_0=weights_0,
                                          risk_budget=risk_budget,
                                          rebalancing_indicators=rebalancing_indicators_t,
                                          apply_total_to_good_ratio=apply_total_to_good_ratio)
        weights_0 = weights_  # warm-start next period
        weights[date] = weights_
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list())
    return weights


def wrapper_risk_budgeting(pd_covar: pd.DataFrame,
                           constraints: Constraints,
                           weights_0: pd.Series = None,
                           risk_budget: Union[pd.Series, Dict[str, float]] = None,
                           rebalancing_indicators: pd.Series = None,
                           apply_total_to_good_ratio: bool = True,
                           detailed_output: bool = False
                           ) -> Union[pd.Series, pd.DataFrame]:
    """
    Single-date risk budgeting with NaN filtering and rebalancing controls.

    Handles three layers of asset filtering:

    1. **Zero risk budgets** (b_i = 0): asset excluded from optimisation,
       receives zero weight. Does not contribute to portfolio risk.
    2. **Rebalancing indicators** (rebal_i = 0): asset frozen at previous
       weight (weights_0). Still contributes to portfolio risk via Σ, but
       its weight is not optimised. The remaining allocation (1 - Σ frozen)
       is distributed among rebalanced assets.
    3. **NaN/zero variance**: asset excluded from optimisation (via
       ``filter_covar_and_vectors_for_nans``), receives zero weight.

    When ``apply_total_to_good_ratio=True``, risk budgets are rescaled by
    N_eligible / N_valid (where N_eligible is the number of assets with
    positive risk budget and N_valid is the subset of those with valid
    covariance data) so that the valid assets absorb the full risk allocation.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        constraints: Portfolio constraints.
        weights_0: Previous-period weights for warm-start / fallback / freezing.
        risk_budget: Target risk budgets. Dict or pd.Series. Assets with budget 0
            are excluded. If None, equal risk budgets are used.
        rebalancing_indicators: Binary series. Assets with value 0 are frozen
            at weights_0. If None, all assets are rebalanced.
        apply_total_to_good_ratio: If True, rescale budgets for excluded assets.
        detailed_output: If True, return a DataFrame with risk contribution
            diagnostics instead of just weights.

    Returns:
        Portfolio weights as pd.Series (or DataFrame if detailed_output=True),
        aligned to pd_covar.index.
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
        warnings.warn(f"wrapper_risk_budgeting: no valid assets in covariance matrix, returning zero weights")
        return pd.Series(0.0, index=pd_covar.index)

    # rescale risk budgets for reduced universe
    # n_eligible counts assets with positive risk budget (before NaN filtering)
    # n_valid counts eligible assets that also have valid covariance
    # the ratio rescales budgets so valid assets absorb the full allocation
    if apply_total_to_good_ratio:
        n_eligible = int(inclusion_indicators.sum())  # assets with positive risk budget
        n_valid = len(clean_covar.columns)             # eligible minus NaN/zero-variance
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

    constraints1 = constraints.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=total_to_good_ratio,
                                                         weights_0=weights_0,
                                                         rebalancing_indicators=None)

    weights0 = opt_risk_budgeting(covar=clean_covar.to_numpy(),
                                  constraints=constraints1,
                                  risk_budget=risk_budget_np)
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
                       verbose: bool = False
                       ) -> np.ndarray:
    """
    Solve constrained risk budgeting using pyrb's ConstrainedRiskBudgeting.

    Finds weights w such that each asset's marginal risk contribution matches
    the prescribed budget:

        RC_i(w) = w_i (Σw)_i / sqrt(w'Σw) = b_i * sqrt(w'Σw)

    subject to linear inequality constraints C @ w >= d and box bounds on w.

    Uses the ADMM-based solver from the pyrb package, which handles linear
    constraints natively without reformulating as a penalty.

    Args:
        covar: Covariance matrix (N x N) as numpy array.
        constraints: Portfolio constraints providing bounds and linear constraints
            via ``set_pyrb_constraints()``.
        risk_budget: Target risk budgets (N,). Must be positive for included assets.
            If None, equal budgets (1/N) are used.
        verbose: If True, print constraint slack diagnostics after solving.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros if the solver
        fails or returns NaN.
    """
    n = covar.shape[0]
    if risk_budget is None:
        risk_budget = np.ones(n) / n

    bounds, c_rows, c_lhs = constraints.set_pyrb_constraints(covar=covar)

    this = ConstrainedRiskBudgeting(covar, budgets=risk_budget, bounds=bounds, C=c_rows, d=c_lhs)
    this.solve()
    optimal_weights = this.x

    if verbose:
        if c_rows is not None:
            qqq = c_rows @ optimal_weights
            slack = qqq - c_lhs
            print(f"slack={slack}")

    if optimal_weights is None or np.any(np.isnan(optimal_weights)):
        warnings.warn(f"opt_risk_budgeting: pyrb solver failed or returned NaN")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0
            warnings.warn(f"opt_risk_budgeting: falling back to weights_0")
        else:
            optimal_weights = np.zeros(n)
            warnings.warn(f"opt_risk_budgeting: falling back to zero weights")

    return optimal_weights


def opt_risk_budgeting_scipy(covar: np.ndarray,
                             constraints: Constraints,
                             risk_budget: np.ndarray = None
                             ) -> np.ndarray:
    """
    Risk budgeting via scipy SLSQP (fallback solver, not recommended).

    Minimises the sum of squared deviations between actual and target
    risk contributions:

        min_w  Σ_i (RC_i(w) - b_i * σ_p(w))²

    This formulation is non-convex and sensitive to initialisation. The pyrb
    solver (``opt_risk_budgeting``) is preferred for production use.

    Assets with zero risk budget are set to NaN internally to exclude them
    from the objective without affecting the constraint structure.

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

    # set zero risk budget to NaN to exclude from objective computation
    risk_budget = np.where(np.isclose(risk_budget, 0.0), np.nan, risk_budget)
    options = {'ftol': 1e-8, 'maxiter': 200}

    res = minimize(risk_budget_objective, x0, args=[covar, risk_budget], method='SLSQP',
                  constraints=constraints_, bounds=bounds, options=options)

    optimal_weights = res.x

    if optimal_weights is None:
        warnings.warn(f"opt_risk_budgeting_scipy: SLSQP solver failed")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0
            warnings.warn(f"opt_risk_budgeting_scipy: falling back to weights_0")
        else:
            optimal_weights = np.zeros(n)
            warnings.warn(f"opt_risk_budgeting_scipy: falling back to zero weights")

    return optimal_weights


def risk_budget_objective(x, pars) -> float:
    """
    Risk budget deviation objective for scipy minimisation.

    Computes mean squared error between actual risk contributions and
    target budgets:

        f(w) = (1/N) Σ_i (RC_i - b_i σ_p)²

    where RC_i = w_i (Σw)_i / σ_p is the risk contribution and b_i σ_p
    is the target. Assets with NaN budgets are excluded from the sum.

    Args:
        x: Portfolio weights (N,).
        pars: [covar, budget] — covariance matrix and risk budgets.

    Returns:
        Mean squared deviation (scalar).
    """
    covar, budget = pars[0], pars[1]
    asset_rc = compute_portfolio_risk_contributions(x, covar)
    sig_p = np.sqrt(compute_portfolio_variance(x, covar))
    if budget is not None:
        # NaN budgets are preserved: their RC matches itself, contributing 0 to SSE
        risk_target = np.where(np.isnan(budget), asset_rc, np.multiply(sig_p, budget))
    else:
        risk_target = np.multiply(sig_p, np.ones_like(asset_rc) / asset_rc.shape[0])
    sse = np.nanmean(np.square(asset_rc - risk_target))
    return sse


def solve_for_risk_budgets_from_given_weights(prices: pd.DataFrame,
                                              given_weights: pd.Series,
                                              time_period: qis.TimePeriod,
                                              covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                              min_risk_budget: float = 1e-4,
                                              max_risk_budget: float = 0.99
                                              ) -> pd.Series:
    """
    Inverse risk budgeting: find budgets that reproduce given target weights.

    Solves for risk budgets b such that ``rolling_risk_budgeting(b)`` produces
    weights matching ``given_weights`` as closely as possible, measured by
    mean absolute deviation of average rolling weights from the target.

    This is useful for reverse-engineering the implicit risk allocation of
    an existing portfolio (e.g., a strategic benchmark) into risk budget form,
    enabling risk-budgeted rebalancing that maintains the intended allocation.

    Initialisation uses the average realised risk contributions of the target
    weights across all dates in covar_dict, which is typically close to the
    solution for stationary covariance structures.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        given_weights: Target portfolio weights to reproduce. Index=tickers.
        time_period: Period for rolling backtest within the objective function.
        covar_dict: Pre-computed covariance matrices.
        min_risk_budget: Lower bound on each non-zero risk budget.
        max_risk_budget: Upper bound on each risk budget.

    Returns:
        Optimal risk budgets as pd.Series. Index=tickers. Budgets sum to 1.
        Assets with zero target weight receive zero budget.
    """
    given_weights_np = given_weights.to_numpy()

    def objective_function(risk_budgets: np.ndarray) -> float:
        risk_budgets = pd.Series(risk_budgets, index=prices.columns)
        risk_budget_weights = rolling_risk_budgeting(prices=prices,
                                                     covar_dict=covar_dict,
                                                     risk_budget=risk_budgets,
                                                     constraints=Constraints(is_long_only=True))
        sse = np.nanmean(np.abs(np.nanmean(risk_budget_weights, axis=0) - given_weights_np))
        return sse

    # initialise from average realised risk contributions of target weights
    is_use_avg_rc = True
    if is_use_avg_rc:
        portfolio_rc = {}
        for date, pd_covar in covar_dict.items():
            rc = qis.compute_portfolio_risk_contributions(w=given_weights, covar=pd_covar)
            portfolio_rc[date] = rc / np.nansum(rc)
        avg_portfolio_rc = pd.DataFrame.from_dict(portfolio_rc, orient='index').mean(0)
        x0 = avg_portfolio_rc.to_numpy()
    else:
        x0 = given_weights.to_numpy()

    # enforce zero budget for zero-weight assets
    enforce_min_max = np.where(np.greater(given_weights_np, 0.0), 1.0, 0.0)
    min_rbs = min_risk_budget * enforce_min_max
    max_rbs = max_risk_budget * enforce_min_max

    bounds = [(x, y) for x, y in zip(min_rbs, max_rbs)]

    options = {'ftol': 1e-8, 'maxiter': 100}
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    res = minimize(objective_function, x0, method='SLSQP',
                   constraints=constraints, bounds=bounds, options=options)

    risk_budgets = res.x

    if risk_budgets is None:
        warnings.warn(f"solve_for_risk_budgets_from_given_weights: solver failed, using zero budgets")
        risk_budgets = np.zeros_like(x0)
    risk_budgets = pd.Series(risk_budgets, index=prices.columns)
    return risk_budgets
