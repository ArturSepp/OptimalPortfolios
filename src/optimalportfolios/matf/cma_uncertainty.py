"""Opt-in conditional GLS inference, separate from frozen CMA accounting results."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t

from optimalportfolios.matf import sharpe_accounting as sa
from optimalportfolios.matf.cma_diagnostics import (
    CmaDiagnosticInputs, CmaDiagnosticSpec, _labelled,
)


@dataclass(frozen=True)
class CmaConditionalGlsResult:
    """Labelled pointwise intervals, joint covariance and explicit working assumptions."""

    factors: pd.DataFrame
    covariance: pd.DataFrame
    summary: dict


# factorlasso.linear_confidence_intervals supports normal and bootstrap-t calibration,
# not finite-df Student-t GLS. Reuse OP's SVD geometry and scipy's t quantile here.
def compute_conditional_gls_intervals(inputs: CmaDiagnosticInputs, *, confidence_level=.95,
                                      spec: CmaDiagnosticSpec | None = None
                                      ) -> CmaConditionalGlsResult:
    """Infer pointwise premia under y = B lambda + e, e ~ N(0, tau^2 diagonal D).

    B, D, cash and already applied admitted alpha are treated as fixed. Estimate
    tau^2 from weighted residual SSE / (N-K), with no intercept. The confidence
    level is conditional on this working model, not calibrated provider accuracy.
    Exposure/alpha estimation uncertainty and correlated forecast errors are absent.
    Rank deficiency, no residual degrees of freedom and numerically zero residual
    scale produce explicit unavailable intervals, never pseudoinverse certainty.
    Calling this function does not modify inputs or existing diagnostic results.
    """
    if not isinstance(inputs, CmaDiagnosticInputs):
        raise TypeError('inputs must be CmaDiagnosticInputs')
    spec = CmaDiagnosticSpec() if spec is None else spec
    if not isinstance(spec, CmaDiagnosticSpec):
        raise TypeError('spec must be CmaDiagnosticSpec')
    if (not np.isscalar(confidence_level) or isinstance(confidence_level, (str, bool))
            or not np.isfinite(confidence_level) or not 0 < confidence_level < 1):
        raise ValueError('confidence_level must be a finite scalar strictly between zero and one')
    if not isinstance(inputs.betas, pd.DataFrame):
        raise ValueError('betas must be a labelled DataFrame')
    ids, factors = inputs.betas.index, inputs.betas.columns
    _labelled(inputs.betas, pd.DataFrame, ids, 'betas')
    _labelled(inputs.residual_variances, pd.Series, ids, 'residual_variances')
    _labelled(inputs.excess_cma, pd.Series, ids, 'excess_cma')
    geometry = sa._gls_geometry(inputs.betas, inputs.residual_variances, rcond=spec.rcond)
    target = inputs.excess_cma.to_numpy(dtype=float)
    white_target = np.linalg.solve(geometry.chol, target)
    estimate = geometry.white_pinv @ white_target
    residual = geometry.residual_projector @ target
    white_residual = np.linalg.solve(geometry.chol, residual)
    sse = float(white_residual @ white_residual)
    n, k, rank = len(ids), len(factors), geometry.rank
    df = n-rank
    status = ('unavailable_rank_deficient' if rank < k else
              'unavailable_residual_df' if df <= 0 else
              'unavailable_zero_residual_scale'
              if sse <= max(float(white_target @ white_target), 1e-300)*1e-24 else
              'conditional_working_model')
    covariance = np.full((k, k), np.nan)
    se, lower, upper = (np.full(k, np.nan) for _ in range(3))
    scale, critical = None, None
    if status == 'conditional_working_model':
        scale = sse/df
        critical = float(t.isf((1.-confidence_level)/2., df))
        covariance = scale*(geometry.white_pinv @ geometry.white_pinv.T)
        se = np.sqrt(np.diag(covariance))
        lower, upper = estimate-critical*se, estimate+critical*se
        if not all(np.isfinite(x).all() for x in (covariance, se, lower, upper)):
            raise ValueError('Conditional GLS intervals overflow; check units and identification')
    table = pd.DataFrame(dict(estimate=estimate if rank == k else np.full(k, np.nan),
                              standard_error=se, lower=lower, upper=upper, status=status),
                         index=factors)
    summary = dict(method='conditional_gls_student_t', confidence_level=float(confidence_level),
                   scope='pointwise', status=status, n_assets=n, n_factors=k, rank=rank,
                   residual_df=df, weighted_residual_sse=sse, scale_squared=scale,
                   critical_value=critical, relative_cutoff=geometry.relative_cutoff,
                   return_units='annual_arithmetic_excess_decimal',
                   assumptions='y = B lambda + e; e ~ N(0, tau^2 D). B and diagonal D are '
                   'fixed; cross-asset forecast errors are independent with variance '
                   'proportional to D. Cash and already applied admitted alpha are fixed. '
                   'D is residual return risk; its use for forecast-error shape is a working '
                   'assumption. tau^2 = weighted residual SSE / (N-K), with no intercept.',
                   excluded_uncertainty='Estimated exposures, residual-risk estimation, '
                   'cash, admitted alpha, factor definitions and provider error correlation.',
                   interpretation='Conditional pointwise confidence intervals, not simultaneous '
                   'coverage, adviser dispersion, realized-return prediction or '
                   'leave-one-out ranges.')
    return CmaConditionalGlsResult(table, pd.DataFrame(covariance, index=factors,
                                                      columns=factors), summary)
