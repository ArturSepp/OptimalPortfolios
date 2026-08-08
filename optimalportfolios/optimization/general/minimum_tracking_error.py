"""Minimum-tracking-error portfolio optimization.

The solver finds the hard-constraint-compliant portfolio closest to a supplied
benchmark in covariance risk:

    minimize  (w - w_b)' Sigma (w - w_b)

Unlike Euclidean target projection, cash and low-volatility instruments enter
through the full covariance matrix. The caller owns policy-specific bounds;
this module owns covariance factorization, CVXPY construction, validation, and
structured diagnostics.
"""
from typing import Dict, Optional, Tuple, Union

import cvxpy as cvx
import numpy as np
import pandas as pd

from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import Constraints, cvx_covar_variance
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.solver_diagnostics import (
    OptimizationOutcome,
    validate_solution,
)
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.utils.weights_drift import apply_drift_to_weights_0


def rolling_minimise_tracking_error(
        prices: pd.DataFrame,
        constraints: Constraints,
        benchmark_weights: Union[pd.Series, pd.DataFrame],
        covar_dict: Dict[pd.Timestamp, pd.DataFrame],
        inclusion_indicators: Optional[pd.DataFrame] = None,
        optimiser_config: OptimiserConfig = OptimiserConfig(),
) -> pd.DataFrame:
    """Compute rolling minimum-tracking-error portfolio weights.

    Args:
        prices: Asset price panel, used for column alignment and weight drift.
        constraints: Hard portfolio constraints applied at every rebalance.
        benchmark_weights: Static Series or time-varying DataFrame benchmark.
        covar_dict: Covariance matrices keyed by rebalancing date.
        inclusion_indicators: Optional binary asset-eligibility panel.
        optimiser_config: Solver and rolling-weight configuration.

    Returns:
        Portfolio weights indexed by the covariance rebalancing dates.
    """
    rebalancing_dates = list(covar_dict.keys())
    tickers = prices.columns
    if not rebalancing_dates:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers, dtype=float)

    if isinstance(benchmark_weights, pd.DataFrame):
        benchmarks = benchmark_weights.reindex(
            index=rebalancing_dates, method='ffill'
        ).reindex(columns=tickers)
    elif isinstance(benchmark_weights, pd.Series):
        benchmark = benchmark_weights.reindex(tickers)
        benchmarks = pd.DataFrame(
            np.tile(benchmark.to_numpy(), (len(rebalancing_dates), 1)),
            index=rebalancing_dates,
            columns=tickers,
        )
    else:
        raise TypeError('benchmark_weights must be a Series or DataFrame')

    if benchmarks.isna().any().any() or not np.isfinite(benchmarks.to_numpy()).all():
        raise ValueError('benchmark_weights must be finite and complete at every rebalance')

    if inclusion_indicators is not None:
        indicators = inclusion_indicators.reindex(
            index=rebalancing_dates, method='ffill'
        ).reindex(columns=tickers)
    else:
        indicators = None

    weights = {}
    weights_0 = None
    prev_date = None
    for date, pd_covar in covar_dict.items():
        weights_0 = apply_drift_to_weights_0(
            weights_0=weights_0,
            prices=prices,
            prev_date=prev_date,
            date=date,
            use_drifted_weights_0=optimiser_config.use_drifted_weights_0,
        )
        weights_t, _ = wrapper_minimise_tracking_error(
            pd_covar=pd_covar,
            benchmark_weights=benchmarks.loc[date],
            constraints=constraints,
            weights_0=weights_0,
            inclusion_indicators=(indicators.loc[date] if indicators is not None else None),
            optimiser_config=optimiser_config,
            context=str(pd.Timestamp(date).date()),
        )
        weights[date] = weights_t
        weights_0 = weights_t
        prev_date = date

    return pd.DataFrame.from_dict(weights, orient='index').reindex(
        columns=tickers
    ).fillna(0.0)


def wrapper_minimise_tracking_error(
        pd_covar: pd.DataFrame,
        benchmark_weights: pd.Series,
        constraints: Constraints,
        weights_0: pd.Series = None,
        inclusion_indicators: pd.Series = None,
        optimiser_config: OptimiserConfig = OptimiserConfig(),
        context: str = '',
) -> Tuple[pd.Series, OptimizationOutcome]:
    """Minimize covariance tracking error to one benchmark portfolio.

    Args:
        pd_covar: Finite covariance matrix with the canonical asset order.
        benchmark_weights: Raw model/benchmark weights in the same universe.
        constraints: Hard portfolio constraints, including final instrument
            bounds and any group, beta, volatility, or tracking-error limits.
        weights_0: Optional current portfolio used by diagnostics and retained
            turnover constraints.
        inclusion_indicators: Optional binary asset-eligibility vector.
        optimiser_config: Solver and covariance-factorization configuration.
        context: Solve label included in diagnostics.

    Returns:
        Minimum-TRE weights and their structured solver outcome.
    """
    if not isinstance(pd_covar, pd.DataFrame) or pd_covar.empty:
        raise ValueError('pd_covar must be a non-empty DataFrame')
    if not pd_covar.index.equals(pd_covar.columns):
        raise ValueError('pd_covar index and columns must match in the same order')
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(
        pd_covar=pd_covar,
        vectors={'benchmark': benchmark_weights},
        inclusion_indicators=inclusion_indicators,
    )
    assets = clean_covar.index
    covariance = clean_covar.to_numpy(dtype=float)
    if not np.isfinite(covariance).all():
        raise ValueError('pd_covar must be finite')
    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError('pd_covar must be symmetric')

    benchmark = good_vectors['benchmark'].astype(float)
    if benchmark.isna().any() or not np.isfinite(benchmark.to_numpy()).all():
        raise ValueError('benchmark_weights must be finite and complete')
    current = None
    if weights_0 is not None:
        current = weights_0.reindex(assets).astype(float)
        if current.isna().any() or not np.isfinite(current.to_numpy()).all():
            raise ValueError('weights_0 must be finite and complete')

    aligned_constraints = constraints.update_with_valid_tickers(
        valid_tickers=assets.tolist(),
        weights_0=current,
        benchmark_weights=benchmark,
        context=context,
        relax_frozen_group_bounds=False,
    )
    outcome = cvx_minimise_tracking_error(
        covar=covariance,
        constraints=aligned_constraints,
        solver=optimiser_config.solver,
        verbose=optimiser_config.verbose,
        factorize_covar=optimiser_config.factorize_covar,
        context=context,
    )
    weights = pd.Series(outcome.weights, index=assets, name='minimum_tracking_error')
    weights = weights.reindex(pd_covar.index).fillna(0.0)
    return weights, outcome


def cvx_minimise_tracking_error(
        covar: np.ndarray,
        constraints: Constraints,
        solver: str = 'CLARABEL',
        verbose: bool = False,
        factorize_covar: bool = True,
        context: str = '',
) -> OptimizationOutcome:
    """Solve one hard-constrained minimum-tracking-error problem.

    Args:
        covar: Finite symmetric covariance matrix.
        constraints: Ticker-aligned constraints carrying benchmark weights.
        solver: CVXPY solver name.
        verbose: Emit solver output when true.
        factorize_covar: Factorize covariance once before building objective
            and compatible risk constraints.
        context: Solve label included in diagnostics.

    Returns:
        Structured accepted solution or documented fallback outcome.
    """
    raw_covar = np.asarray(covar, dtype=float)
    if raw_covar.ndim != 2 or raw_covar.shape[0] != raw_covar.shape[1]:
        raise ValueError('covar must be a square matrix')
    if not np.isfinite(raw_covar).all():
        raise ValueError('covar must be finite')
    if constraints.benchmark_weights is None:
        raise ValueError('constraints.benchmark_weights is required')
    benchmark = constraints.benchmark_weights.to_numpy(dtype=float)
    n = raw_covar.shape[0]
    if benchmark.shape != (n,):
        raise ValueError('benchmark_weights length does not match covariance')

    covar_factorization = (
        factorize_covariance(raw_covar) if factorize_covar else None
    )
    solver_covar = (
        covar_factorization.covar
        if covar_factorization is not None else raw_covar
    )
    covar_psd = cvx.psd_wrap(solver_covar)
    w = cvx.Variable(n, nonneg=constraints.is_long_only)
    active = w - benchmark
    objective = cvx.Minimize(cvx_covar_variance(
        active_weights=active,
        covar=covar_psd,
        covar_factorization=covar_factorization,
    ))
    constraints_ = constraints.set_cvx_all_constraints(
        w=w,
        covar=covar_psd,
        covar_factorization=covar_factorization,
    )
    problem = cvx.Problem(objective, constraints_)
    solver_options = {'max_iter': 1000} if solver.upper() == 'CLARABEL' else {}
    try:
        problem.solve(solver=solver, verbose=verbose, **solver_options)
        status = problem.status
    except cvx.error.SolverError:
        status = 'solver_error'
    return validate_solution(
        optimal_weights=None if w.value is None else np.asarray(w.value).ravel(),
        problem_status=status,
        constraints=constraints,
        n=n,
        solver=solver,
        context=context,
        covar=solver_covar,
        covar_factorization=covar_factorization,
    )
