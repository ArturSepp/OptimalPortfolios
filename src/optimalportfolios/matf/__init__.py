"""Factor-coordinate Sharpe accounting; MATF is an illustration, not a required factor set."""

from optimalportfolios.matf.gls_diagnostic import (
    GlsGeometry,
    GlsPremiaResult,
    GlsResolutionResult,
    GlsUncertaintyScenario,
    GlsLoadingStabilityResult,
    ResidualOpportunity,
    compute_gls_geometry,
    compute_gls_resolution,
    compute_gls_uncertainty,
    compute_gls_reference_uncertainty,
    compute_gls_adviser_disagreement,
    compute_gls_provider_contrast,
    compute_gls_loading_stability,
    compute_gls_loading_stability_contrast,
    summarize_gls_loading_stability,
    compute_residual_opportunity,
    infer_factor_premia,
)

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
