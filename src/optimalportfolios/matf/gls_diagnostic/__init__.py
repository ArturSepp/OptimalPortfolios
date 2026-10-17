"""GLS risk-premia inference conditional on an estimated factor model.

Provider CMAs, cash, alpha admission and forecast uncertainty belong to this
portfolio-analytics layer. Historical loading estimation remains in FactorLasso.
"""

from optimalportfolios.matf.gls_diagnostic._inference import (
    GlsGeometry,
    GlsPremiaResult,
    compute_gls_geometry,
    factor_mimicking_portfolios,
    gls_projector,
    gls_split,
    infer_factor_premia,
)
from optimalportfolios.matf.gls_diagnostic._opportunity import (
    ResidualOpportunity,
    compute_residual_opportunity,
)
from optimalportfolios.matf.gls_diagnostic._resolution import (
    GlsResolutionResult,
    compute_gls_resolution,
)
from optimalportfolios.matf.gls_diagnostic._scenarios import (
    GlsUncertaintyScenario,
    compute_gls_uncertainty,
    compute_gls_reference_uncertainty,
    compute_gls_adviser_disagreement,
    compute_gls_provider_contrast,
)
from optimalportfolios.matf.gls_diagnostic._stability import (
    GlsLoadingStabilityResult,
    summarize_gls_loading_stability,
    compute_gls_loading_stability,
    compute_gls_loading_stability_contrast,
)
