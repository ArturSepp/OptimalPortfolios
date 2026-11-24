"""Experimental supplied-CMA allocation layer; existing optimiser defaults are unchanged.

Inputs are annual expected excess returns and annual covariance in matching asset
order. A sparse policy benchmark is a reference portfolio, not an eligible-asset
mask. Statistical estimation belongs upstream; QIS owns the risk diagnostics.
This module is deliberately separate from the public production dispatchers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis

from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import (
    Constraints, ConstraintEnforcementType, GroupLowerUpperConstraints,
)
from optimalportfolios.optimization.taa.maximise_alpha_over_tre import (
    wrapper_maximise_alpha_over_tre,
)


def solve_cma_tre(means, covariance, benchmark, tracking_error, date, solver='CLARABEL',
                  *, max_weights=None, group_loadings=None, group_min=None, group_max=None):
    """Maximise supplied annual CMA subject to full investment, long-only and TRE.

    Zero benchmark weights do not prohibit investment. Reject nonfinite or
    misaligned inputs and any inaccurate, fallback or infeasible solution.
    Return weights and raw/effective risk diagnostics; no resampling or fitting
    is performed here. The ordinary OP covariance factorization is retained.
    Optional name caps and group bands are explicit absolute portfolio limits;
    the policy benchmark need not satisfy the portfolio's individual name caps.
    """
    index = means.index
    if (len(index) < 2 or index.has_duplicates or index.hasnans
            or not covariance.index.equals(index) or not covariance.columns.equals(index)
            or not benchmark.index.equals(index)):
        raise ValueError('CMA, covariance and benchmark require identical unique asset labels')
    for value in (means, covariance, benchmark):
        if not np.isfinite(value.to_numpy()).all():
            raise ValueError('Nonfinite CMA inputs; no filling or asset dropping')
    if ((benchmark < 0).any() or not np.isclose(benchmark.sum(), 1., atol=1e-10, rtol=0)):
        raise ValueError('Benchmark must be nonnegative and sum to one; no normalization')
    if not np.isfinite(tracking_error) or tracking_error <= 0:
        raise ValueError('Annual tracking error must be positive')
    if (not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0)
            or np.any(np.diag(covariance) <= 0)):
        raise ValueError('Covariance must be symmetric with positive variances')
    caps = pd.Series(1., index=index) if max_weights is None else max_weights
    if (not isinstance(caps, pd.Series) or not caps.index.equals(index)
            or not np.isfinite(caps.to_numpy()).all() or (caps < 0).any()
            or (caps > 1).any() or caps.sum() < 1.-1e-10):
        raise ValueError(
            'Name caps require matching labels, finite bounds and full-investment capacity')
    groups = None
    if any(v is not None for v in (group_loadings, group_min, group_max)):
        if not all(v is not None for v in (group_loadings, group_min, group_max)):
            raise ValueError('Group loadings and both allocation bounds must be supplied together')
        if (not isinstance(group_loadings, pd.DataFrame)
                or not group_loadings.index.equals(index) or group_loadings.columns.has_duplicates
                or group_loadings.columns.hasnans or group_loadings.shape[1] == 0
                or not np.isfinite(group_loadings.to_numpy()).all()
                or (group_loadings < 0).any().any() or (group_loadings == 0).all().any()):
            raise ValueError('Require explicit aligned, finite and nonempty group loadings')
        for bound in (group_min, group_max):
            if (not isinstance(bound, pd.Series) or not bound.index.equals(group_loadings.columns)
                    or not np.isfinite(bound.to_numpy()).all()):
                raise ValueError('Group bounds must exactly match all declared group columns')
        if (group_min < 0).any() or (group_max > 1).any() or (group_min > group_max).any():
            raise ValueError('Require ordered group bounds between zero and one')
        groups = GroupLowerUpperConstraints(group_loadings, group_min, group_max)
    constraints = Constraints(
        is_long_only=True, min_exposure=1., max_exposure=1.,
        min_weights=pd.Series(0., index=index), max_weights=caps,
        benchmark_weights=benchmark, tracking_err_vol_constraint=float(tracking_error),
        turnover_constraint=None, group_lower_upper_constraints=groups,
        constraint_enforcement_type=ConstraintEnforcementType.FORCED_CONSTRAINTS,
    )
    weights, outcome = wrapper_maximise_alpha_over_tre(
        pd_covar=covariance, alphas=means, benchmark_weights=benchmark,
        constraints=constraints, weights_0=None,
        optimiser_config=OptimiserConfig(solver=solver, apply_total_to_good_ratio=False),
        context=f'CMA TRE {tracking_error:g} {date}',
    )
    if (not outcome.accepted or outcome.status != 'optimal'
            or outcome.fallback_source is not None or len(outcome.weights) != len(index)):
        raise RuntimeError(f'Uncertified CMA optimum: {outcome.status}, {outcome.reason}')
    effective = covariance if outcome.covar_factorization is None else pd.DataFrame(
        outcome.covar_factorization.covar, index=index, columns=index)
    raw_risk = qis.RiskModel(covar={date: covariance})
    effective_risk = qis.RiskModel(covar={date: effective})
    raw_te = float(raw_risk.compute_tre_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date))
    te = float(effective_risk.compute_tre_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date))
    variance = float(effective_risk.compute_tre_at_date(
        benchmark_weights=benchmark*0, portfolio_weights=weights, date=date))**2
    allocations = {} if groups is None else (group_loadings.T @ weights).to_dict()
    group_error = 0. if groups is None else max(
        float((group_min-pd.Series(allocations)).max()),
        float((pd.Series(allocations)-group_max).max()), 0.)
    cap_error = max(float((weights-caps).max()), 0.)
    error = max(abs(weights.sum()-1.), -float(weights.min()), cap_error, group_error,
                te-tracking_error, raw_te-tracking_error, 0.)
    if not np.isfinite(weights).all() or error > 2e-6:
        raise RuntimeError(f'CMA portfolio violates its constraints: {error}')
    return weights, dict(
        tracking_error=te, raw_tracking_error=raw_te, tracking_error_limit=tracking_error,
        max_constraint_error=error, forecast_annual_variance=variance,
        expected_excess_return_pa=float(means @ weights),
        expected_active_return_pa=float(means @ (weights-benchmark)),
        covariance_max_adjustment=float(np.max(np.abs(effective-covariance))),
        max_asset_cap_violation=cap_error, max_group_violation=group_error,
        group_allocations=allocations,
        solver=outcome.solver, status=outcome.status,
    )
