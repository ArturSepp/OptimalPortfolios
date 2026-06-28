"""
Alpha-maximising portfolio optimisation with a target return constraint.

Solves the tactical asset allocation (TAA) problem:

    max_w  α'w

    s.t.   y'w >= r_target        (return constraint, optional)
           w'Σw <= σ²_max         (risk constraint, optional)
           1'w = 1                (full investment)
           w >= 0                 (long-only, optional)
           w_min <= w <= w_max    (weight bounds)

where α is the vector of expected alphas (excess returns from active views),
y is the vector of asset yields or expected returns, r_target is the minimum
portfolio return, and Σ is the covariance matrix.

This formulation separates the alpha signal (what we want to maximise) from
the return constraint (what we need to deliver).

Reference:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation
    for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
    Available at https://www.pm-research.com/content/iijpormgmt/52/4/86
"""
import warnings
import dataclasses
import numpy as np
import pandas as pd
import cvxpy as cvx
from typing import Dict, Optional, Union

from optimalportfolios import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.solver_diagnostics import validate_solution
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.utils.weights_drift import apply_drift_to_weights_0


def rolling_maximise_alpha_with_target_return(prices: pd.DataFrame,
                                              alphas: pd.DataFrame,
                                              yields: pd.DataFrame,
                                              target_returns: pd.Series,
                                              constraints: Constraints,
                                              covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                              benchmark_weights: Optional[Union[pd.Series, pd.DataFrame]] = None,
                                              soft_tracking_error: bool = False,
                                              optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                              ) -> pd.DataFrame:
    """
    Compute rolling alpha-maximising portfolios with a target return constraint.

    At each rebalancing date (defined by the keys of ``covar_dict``), solves:

        max_w  α_t'(w - w_b)   s.t.  y_t'w >= r_t,  TE(w, w_b) <= budget,  constraints

    Alphas, yields, and target returns are forward-filled to the rebalancing
    schedule.

    Args:
        prices: Asset price panel. Used for column alignment.
        alphas: Alpha signals per asset. Forward-filled to rebalancing dates.
        yields: Expected asset returns or yields. Forward-filled.
        target_returns: Minimum portfolio return at each date. Forward-filled.
        constraints: Portfolio constraints.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        benchmark_weights: Optional TAA benchmark, against which tracking error
            and active alpha are measured. Series (static) or DataFrame
            (time-varying), forward-filled to the rebalancing schedule. When
            None the problem stays absolute (objective α'w, no TE term) — the
            original behaviour. When provided, the objective becomes active
            (α'(w - w_b)) and any TE budget on ``constraints`` is enforced
            relative to these weights — mirroring ``maximise_alpha_over_tre``.
        soft_tracking_error: When True (requires benchmark_weights), the return
            target stays hard but TE is enforced as a utility penalty
            (``tre_utility_weight``) rather than a hard constraint, so the yield
            target always takes priority and the solve never goes infeasible on
            a tight TE budget. When False, TE is a hard constraint if a budget
            is set on ``constraints``.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    rebalancing_schedule = list(covar_dict.keys())
    alphas = alphas.reindex(index=rebalancing_schedule, method='ffill')
    yields = yields.reindex(index=rebalancing_schedule, method='ffill')
    target_returns = target_returns.reindex(index=rebalancing_schedule, method='ffill')

    # align benchmark to the rebalancing schedule, mirroring
    # rolling_maximise_alpha_over_tre: Series -> broadcast, DataFrame -> ffill.
    if benchmark_weights is not None:
        if isinstance(benchmark_weights, pd.DataFrame):
            benchmark_weights = benchmark_weights.reindex(
                index=rebalancing_schedule, method='ffill').fillna(0.0)
        else:
            benchmark_weights = benchmark_weights.to_frame(
                name=rebalancing_schedule[0]).T.reindex(
                index=rebalancing_schedule, method='ffill').fillna(0.0)

    weights = {}
    weights_0 = None
    prev_date = None
    for date, pd_covar in covar_dict.items():

        if optimiser_config.verbose:
            print(f"date={date}")
            print(f"pd_covar=\n{pd_covar}")
            print(f"alphas=\n{alphas.loc[date, :]}")
            print(f"yields=\n{yields.loc[date, :]}")
            print(f"target_return=\n{target_returns[date]}")

        weights_0 = apply_drift_to_weights_0(
            weights_0=weights_0, prices=prices,
            prev_date=prev_date, date=date,
            use_drifted_weights_0=optimiser_config.use_drifted_weights_0,
        )
        benchmark_weights_t = (
            benchmark_weights.loc[date, :] if benchmark_weights is not None else None
        )
        weights_ = wrapper_maximise_alpha_with_target_return(
            pd_covar=pd_covar,
            alphas=alphas.loc[date, :],
            yields=yields.loc[date, :],
            target_return=target_returns[date],
            constraints=constraints,
            benchmark_weights=benchmark_weights_t,
            soft_tracking_error=soft_tracking_error,
            weights_0=weights_0,
            optimiser_config=optimiser_config,
            context=str(pd.Timestamp(date).date())
        )

        if np.all(np.equal(weights_, 0.0)):
            weights_0 = None
            prev_date = None
        else:
            weights_0 = weights_
            prev_date = date
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list()).fillna(0.0)

    return weights


def wrapper_maximise_alpha_with_target_return(pd_covar: pd.DataFrame,
                                              alphas: pd.Series,
                                              yields: pd.Series,
                                              target_return: float,
                                              constraints: Constraints,
                                              benchmark_weights: Optional[pd.Series] = None,
                                              soft_tracking_error: bool = False,
                                              weights_0: pd.Series = None,
                                              optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True),
                                              context: str = ''
                                              ) -> pd.Series:
    """
    Single-date alpha maximisation with NaN/zero-variance filtering.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        alphas: Alpha signal per asset.
        yields: Expected returns per asset for the return constraint.
        target_return: Minimum portfolio return level (y'w >= target_return).
        constraints: Portfolio constraints.
        benchmark_weights: Optional TAA benchmark weights. When provided they
            are injected into the constraints (``update_with_valid_tickers``),
            which makes the active objective α'(w - w_b) and enables the TE
            term relative to these weights. When None the problem is absolute
            (α'w, no TE term) — the original behaviour.
        soft_tracking_error: When True, the return target stays hard but TE is
            enforced as a utility penalty (``tre_utility_weight``) rather than a
            hard constraint — so yield always takes priority and the solve
            never goes infeasible on a tight TE budget. Requires
            benchmark_weights. When False, TE is hard (if a budget is set).
        weights_0: Previous-period weights for warm-start / fallback.
        optimiser_config: Solver configuration.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    vectors = dict(alphas=alphas)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    if optimiser_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    # yields: fill NaN with 0 so they don't break the return constraint
    # assets with NaN yield contribute zero to portfolio yield — conservative
    valid_tickers = clean_covar.columns.to_list()
    yields_clean = yields.reindex(index=valid_tickers)
    nan_yields = yields_clean.index[yields_clean.isna()].tolist()
    if nan_yields:
        warnings.warn(f"NaN yields for {nan_yields}, setting to 0 in return constraint")
        yields_clean = yields_clean.fillna(0.0)

    constraints1 = constraints.update_with_valid_tickers(context=context, 
        valid_tickers=valid_tickers,
        total_to_good_ratio=total_to_good_ratio,
        weights_0=weights_0,
        asset_returns=yields_clean,
        target_return=target_return,
        benchmark_weights=benchmark_weights,
    )

    weights = cvx_maximise_alpha_with_target_return(
        covar=clean_covar.to_numpy(),
        alphas=good_vectors['alphas'].to_numpy(),
        constraints=constraints1,
        soft_tracking_error=soft_tracking_error,
        solver=optimiser_config.solver,
        verbose=optimiser_config.verbose,
        context=context
    )

    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=valid_tickers)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)

    return weights


def cvx_maximise_alpha_with_target_return(covar: np.ndarray,
                                          alphas: np.ndarray,
                                          constraints: Constraints,
                                          soft_tracking_error: bool = False,
                                          verbose: bool = False,
                                          solver: str = 'CLARABEL',
                                          context: str = ''
                                          ) -> np.ndarray:
    """
    Solve alpha-maximising portfolio allocation via CVXPY.

    Two formulations, selected by ``soft_tracking_error``:

    Hard tracking error (default, soft_tracking_error=False):

        max_w  α'(w - w_b)
        s.t.   y'w >= r_target                      (hard return target)
               (w - w_b)'Σ(w - w_b) <= TE²_max      (hard TE, if set on constraints)
               1'w = 1,  w >= 0,  bounds

    Soft tracking error (soft_tracking_error=True) — TE drops from a hard
    constraint to a penalty so the return target always takes priority:

        max_w  α'(w - w_b) - λ_TE (w - w_b)'Σ(w - w_b) - λ_TO ||w - w_0||_1
        s.t.   y'w >= r_target                      (hard return target)
               1'w = 1,  w >= 0,  bounds

    The soft path reuses Constraints.set_cvx_utility_objective_constraints, so
    the penalty weights are the calibrated ``tre_utility_weight`` /
    ``turnover_utility_weight`` already on the constraints object, and the
    return target (asset_returns @ w >= target_return) is still enforced as a
    hard constraint by that builder. The caller should NOT set a hard
    ``tracking_err_vol_constraint`` in the soft case (it would be ignored here).

    Args:
        covar: Covariance matrix (N x N).
        alphas: Alpha signal vector (N,).
        constraints: Portfolio constraints including return target.
        soft_tracking_error: If True, enforce TE as a utility penalty and keep
            only the return target hard. If False, use the hard-constraint form.
        verbose: If True, print CVXPY solver diagnostics.
        solver: CVXPY solver name.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
    """
    n = covar.shape[0]
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    w = cvx.Variable(n, nonneg=nonneg)
    covar_psd = cvx.psd_wrap(covar)

    if soft_tracking_error and constraints.benchmark_weights is not None:
        # Soft TE only: TE becomes a utility penalty (tre_utility_weight) while
        # the return target and turnover stay HARD. The utility builder would
        # otherwise also penalise turnover (turnover_utility_weight) — we null
        # that out so turnover isn't double-counted, then add the hard turnover
        # constraint explicitly below. Yield target is added hard by the builder.
        constraints_soft = dataclasses.replace(
            constraints,
            turnover_utility_weight=None,
            group_turnover_constraint=None,
        )
        objective_fun, constraints_ = constraints_soft.set_cvx_utility_objective_constraints(
            w=w, alphas=alphas, covar=covar_psd,
        )
        objective = cvx.Maximize(objective_fun)
        # keep turnover HARD if a budget is set, mirroring set_cvx_all_constraints
        if constraints.group_turnover_constraint is None and constraints.turnover_constraint is not None:
            if constraints.weights_0 is None:
                warnings.warn("weights_0 must be given for turnover constraint")
            elif constraints.turnover_costs is not None:
                constraints_ += [cvx.norm(cvx.multiply(
                    constraints.turnover_costs.to_numpy(), w - constraints.weights_0.to_numpy()), 1)
                    <= constraints.turnover_constraint]
            else:
                constraints_ += [cvx.norm(w - constraints.weights_0.to_numpy(), 1)
                                 <= constraints.turnover_constraint]
    else:
        # Hard path: active objective α'(w - w_b) when a benchmark is injected,
        # else absolute α'w. set_cvx_all_constraints enforces the hard TE
        # (if set), the return target, and the box/turnover/group constraints.
        if constraints.benchmark_weights is not None:
            benchmark_weights = constraints.benchmark_weights.to_numpy()
            objective_fun = alphas.T @ (w - benchmark_weights)
        else:
            objective_fun = alphas.T @ w
        objective = cvx.Maximize(objective_fun)
        constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar_psd)

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights, _is_valid = validate_solution(
        w.value, problem.status, constraints, n, solver=solver, context=context)

    return optimal_weights
