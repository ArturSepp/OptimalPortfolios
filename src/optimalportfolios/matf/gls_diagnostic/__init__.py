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
