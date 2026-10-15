"""Opt-in conditional GLS inference, separate from frozen CMA accounting results."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from optimalportfolios.matf.gls_diagnostic import infer_factor_premia

from optimalportfolios.matf.cma_diagnostics import (
    CmaDiagnosticInputs, CmaDiagnosticSpec, _labelled,
)


@dataclass(frozen=True)
class CmaConditionalGlsResult:
    """Labelled pointwise intervals, joint covariance and explicit working assumptions."""

    factors: pd.DataFrame
    covariance: pd.DataFrame
    summary: dict


# gls_diagnostic owns risk-premia estimation and Student-t inference; this is a CMA adapter.
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
    ids = inputs.betas.index
    _labelled(inputs.betas, pd.DataFrame, ids, 'betas')
    _labelled(inputs.residual_variances, pd.Series, ids, 'residual_variances')
    _labelled(inputs.excess_cma, pd.Series, ids, 'excess_cma')
    inferred = infer_factor_premia(
        inputs.betas, inputs.residual_variances, inputs.excess_cma, cash_rate=0.,
        confidence_level=confidence_level, rcond=spec.rcond)
    table = inferred.factors[['estimate', 'standard_error', 'lower', 'upper', 'status']].copy()
    keys = ('method', 'confidence_level', 'scope', 'status', 'n_assets', 'n_factors', 'rank',
            'residual_df', 'weighted_residual_sse', 'scale_squared', 'critical_value',
            'relative_cutoff')
    summary = dict({key: inferred.summary[key] for key in keys},
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
    return CmaConditionalGlsResult(table, inferred.covariance.copy(), summary)
