"""Factor-coordinate Sharpe accounting; MATF is an illustration, not a required factor set."""

from optimalportfolios.matf.cma_diagnostics import (
    AdmissionPolicy,
    CmaDiagnosticInputs,
    CmaDiagnosticResult,
    CmaDiagnosticSpec,
    compute_cma_diagnostics,
    without_admitted_alpha,
)

from optimalportfolios.matf.sharpe_accounting import (
    CompletionResult,
    CompletionStatus,
    achievable_sharpe2,
    ceiling_premium_correlation_split,
    completion_sharpe2,
    factor_ceiling,
    factor_mimicking_portfolios,
    fixed_exposure_fpir,
    fpir,
    gls_projector,
    gls_split,
    holdings_decomposition,
    hurdle_premia,
    ir_bound,
    overlay_value,
    partial_overlay_value,
    tangency_access_split,
)

from optimalportfolios.matf.cma_uncertainty import (
    CmaConditionalGlsResult,
    compute_conditional_gls_intervals,
)

from optimalportfolios.matf.cma_identification import (
    CmaIdentificationResult,
    compute_cma_identification,
)
