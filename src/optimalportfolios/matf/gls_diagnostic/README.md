---
myst:
  html_meta:
    description: >-
      Contributor guide to OP's risk-premia GLS package: ownership, input and result
      contracts, conditional intervals, forecast sensitivities and report integration.
---

# GLS risk-premia diagnostics

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

This package infers the factor risk premia implicit in supplied capital market
assumptions (CMAs), conditional on an already estimated factor model. It owns the
generalized least-squares (GLS) geometry, factor-mimicking portfolios, conditional
confidence intervals and forecast-box sensitivity bounds. The mathematical
derivation is in the [GLS methodology guide](../../../../docs/gls_factor_premia.md).
This README describes the source and integration contracts.

## Ownership and workflow

| Stage | Owner | Contract |
|---|---|---|
| Historical estimation | [FactorLasso](https://github.com/ArturSepp/FactorLasso), configured by the application | Estimate loadings and residual risk from historical returns; retain historical residual means for alpha admission. Provider forecasts do not enter the fit. |
| Input preparation | Rosaa or another consuming application | Supply tracking-index mappings, factor definitions, prices, cash/currency conventions, cutoff, annualization and the policy-admitted alpha vector. |
| Risk-premia inference | `optimalportfolios.matf.gls_diagnostic` | Project provider forecasts into the supplied factor span and expose rank, uncertainty and influence diagnostics. |
| CMA accounting and identification | OP's adjacent `cma_diagnostics`, `cma_uncertainty` and `cma_identification` modules | Reconcile forecasts and channels, compute contribution/deletion tables and test individual or paired factor identification. |
| Presentation and persistence | OP reports and the application | Render completed views using QIS; save full-precision tables, mappings and provenance beside the PDF. |

The numerical core imports NumPy, pandas and SciPy. It does not call FactorLasso,
Rosaa, a data vendor or paper replication code. OP continues to use FactorLasso
elsewhere for estimation; this ownership does not remove that package dependency.
See the [FactorLasso citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff)
and [QIS citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

## Public entry points

Import from `optimalportfolios.matf.gls_diagnostic`; `_inference.py` is private.

| Entry point | Result |
|---|---|
| `infer_factor_premia(B, Omega, total_returns, *, cash_rate, ...)` | `GlsPremiaResult`: labelled estimates, reconciliation, intervals, mimicking weights and optional sensitivity bounds. |
| `compute_gls_geometry(B, Omega, *, rcond=None)` | `GlsGeometry`: Cholesky whitening, singular values and retained rank, pseudoinverse and asset-space residual operator. |
| `gls_projector(B, Omega)` | The residual operator, not the fitted-return projector; uses the default rank cutoff. |
| `gls_split(m, B, Omega)` | Minimum-norm factor coordinates and asset residuals for an already cash/alpha-adjusted forecast vector. |
| `factor_mimicking_portfolios(B, Omega)` | Asset-by-factor unit-exposure weights; requires full column rank. |
| `compute_residual_opportunity(B, residual_variances, alpha, *, controls=None, alpha_covariance=None, residual_correlation=None, repair_correlation=False, rcond=None)` | `ResidualOpportunity`: squared residual-alpha capacity, conditional noise floor and effective residual breadth. |
| `compute_gls_resolution(B, Omega, *, factor_volatility=None, conditional_result=None, ...)` | `GlsResolutionResult`: residual visibility, normalized geometric resolution, current conditional precision and diagnostic settings. |
| `compute_gls_uncertainty(B, Omega, premia, forecast_covariance, ...)` | `GlsUncertaintyScenario`: explicit covariance propagation; default `distribution='none'` provides standard errors without interval endpoints. |
| `compute_gls_reference_uncertainty(B, Omega, premia, *, scale_squared, reference_provenance, ...)` | Known-scale normal or independently estimated-reference Student-t scenario; never changes the default own-scale CI. |
| `compute_gls_adviser_disagreement(B, Omega, premia, lower_quartile, upper_quartile, *, source_provenance, ...)` | IQR-derived adviser disagreement scenario, with explicit cross-asset correlation and no adviser-count divisor. |
| `compute_gls_provider_contrast(B_left, Omega_left, premia_left, B_right, Omega_right, premia_right, *, covariance_left, covariance_right, cross_covariance, ...)` | Complete paired covariance propagated through separate provider maps. |
| `compute_gls_loading_stability(B, Omega, total_returns, *, B_draws, Omega_draws, cash_rate, baseline_provenance, ...)` | Provider-keyed `GlsLoadingStabilityResult` objects from paired estimator draws; one SVD per draw. |
| `summarize_gls_loading_stability(premia_draws, baseline, *, ranks, alpha_policy, baseline_provenance, ...)` | Stability bands from existing premium draws, preserving rank and solver failures. |
| `compute_gls_loading_stability_contrast(left, right, ...)` | Paired draw differences using shared draw identities and joint availability. |

`GlsGeometry`, `GlsPremiaResult`, `compute_gls_geometry` and `infer_factor_premia`
are also exported by `optimalportfolios.matf` and the package root. Existing
`sharpe_accounting` projection and mimicking-portfolio entry points delegate here,
preserving their import paths.

`ResidualOpportunity` and `compute_residual_opportunity` are also available from
`optimalportfolios.matf`. They reuse the same whitened SVD and numerical rank
policy. With diagonal residual-return variances `D` and whitened residual
projector `M`, capacity is `||M D^-1/2 alpha||^2`; the conditional estimation-noise
floor is `trace(M D^-1/2 V D^-1/2)`. `V` is the supplied covariance of estimated
alpha, usually produced by FactorLasso's uncertainty layer. It is not residual
return covariance. Additional columns such as funding or asset-class indicators
must be supplied explicitly through `controls`.

Effective breadth is the participation ratio of `M R M`, where `R` is residual
return correlation. Identity correlation gives `N - rank`; an exhausted subspace
or numerically zero projected covariance gives zero. An indefinite correlation
requires explicit `repair_correlation=True`, which clips negative eigenvalues
once and rescales once to unit diagonal, reporting both the raw ratio and repair
magnitude. There is no factor fit, shrinkage, or hidden covariance estimator.
Capacity minus its noise floor remains signed. These conditional universe
descriptors are not long-only portfolio Sharpe ratios or confidence intervals;
portfolio constraints belong to `matf.opportunity`.

## Input contract

`B` is an asset-by-factor matrix. `Omega` is either a vector of strictly positive
residual variances or a symmetric positive-definite residual covariance matrix.
Arrays are positional. Pandas axes must be unique, nonmissing and exactly aligned;
the calculation never silently sorts, reindexes, imputes or repairs covariance.

Supply arithmetic total-return forecasts in decimal units and one currency.
`cash_rate` uses the same horizon and annualization convention. Annual inputs give
annual premium outputs; no conversion is performed. `alpha` is the policy-admitted
historical vector, including any asset-specific admission weights. The additional
`alpha_weight` lies in [0, 1] and defaults to zero. A nonzero multiplier requires
an explicit alpha vector. Thus the target is
`total_returns - cash_rate - alpha_weight * alpha`.

For example, `alpha_weight=1` credits all of the supplied admitted vector, not
necessarily all raw historical alpha. The common views 0, 0.5 and 1 hold the fitted
model and forecasts fixed while changing this multiplier.

Optional `lower_returns` and `upper_returns` must be supplied together in the same
asset order and units. `cash_bounds` is one shared cash interval and requires those
forecast endpoints; it is not a separate independent cash shock for each asset.

## Results and interpretation

| `GlsPremiaResult` field | Meaning |
|---|---|
| `premia` | Factor coordinates; at deficient rank these are a minimum-norm convention, not uniquely identified premia. |
| `assets` | Total forecast, cash, admitted and credited alpha, GLS target, fitted factor return and residual. Their components reconcile the supplied forecast. |
| `factors` | Identified estimate, standard error, lower/upper conditional interval, status, variance inflation and gross mimicking weight. |
| `covariance` | Joint conditional premium covariance, for contrasts that must retain off-diagonal terms. |
| `mimicking_weights` | Risky asset weights with unit exposure to one factor and zero exposure to the others. They can be short or leveraged; cash completes financing. |
| `sensitivity_bounds` | Signed marginal forecast-box bounds, separate from statistical intervals. |
| `summary` | Rank/cutoff, residual degrees of freedom and scale, alpha/cash settings, confidence level, assumptions and exclusions. |

Rank is determined on the whitened loadings, without squaring their condition
number. The default relative SVD cutoff is machine epsilon times the larger design
dimension. A custom `rcond` changes the retained model span and is recorded.

Conditional pointwise Student-t intervals assume Gaussian cross-asset forecast
errors with covariance proportional to the supplied residual-return covariance.
Loadings, residual risk, alpha, cash and factor definitions are held fixed. This
working assumption is additional to the historical risk model; these intervals
exclude first-stage estimation uncertainty and do not measure provider accuracy.
Residual degrees of freedom are the asset count minus retained rank, with no
intercept, forecast-horizon divisor or adviser-count divisor.

At deficient factor rank, identified estimates, intervals, mimicking weights and
box endpoints are unavailable. Full-rank fits with no residual degrees of freedom
or numerically zero residual scale also have unavailable CIs. Inspect `summary`
and `factors.status` before using endpoints. Forecast boxes, 25 bp perturbations
and leave-one-out ranges answer sensitivity questions and remain separate from CIs.

## Extension contracts

The original `infer_factor_premia` implementation and default conditional interval
result remain unchanged. New diagnostics are separate results; no renderer or
adapter automatically chooses a narrower uncertainty model.

`GlsResolutionResult` contains `assets`, `factors` and `summary`. Visibility is the
whitened residual-projector diagonal; with diagonal residual covariance its sum
is residual dimension and its entries identify individual assets. Dense covariance
uses ordered Cholesky coordinates, which must not be described as individual
asset visibility. Normalized resolution divides residual mimicking standard
deviation by fixed factor volatility and is independent of alpha treatment.
Normalized conditional half-width additionally depends on the forecast residual
scale. Annual 0.01 and normalized 0.25 are diagnostic targets, not constraints.

`GlsUncertaintyScenario` contains `factors`, `covariance` and `summary`. Input
forecast covariance may be singular but must be positive semidefinite with exact
labels, ordering and annual decimal units. Normal intervals require the declared
Gaussian model; estimated-reference Student-t requires its degrees of freedom
and `independent_reference=True`. Provenance arguments are nonempty strings.
Adviser IQR scenarios describe disagreement, not consensus-mean error or accuracy.
Provider contrasts require explicit cross covariance and a positive semidefinite
joint covariance; zeros assert independence. Unknown dependence belongs in
labelled sensitivity scenarios, not an implicit significance test.

`GlsLoadingStabilityResult` contains `bands`, `draws`, `draw_status` and `summary`.
The high-level replay accepts loading draws shaped `(draw, asset, factor)`,
residual variance draws `(draw, asset)` or covariance draws `(draw, asset, asset)`,
and optional already policy-admitted alpha draws `(draw, asset)`. All axes retain
the matching baseline's order. `fixed_admission` holds baseline alpha fixed;
`paired_admission` uses the alpha draw paired with loadings and residual risk.
The diagnostic multiplier is applied once to either admitted vector.

Supply the baseline fit on the same common history as the draws. Save its sampling
convention and source hashes in provenance. Finite, successful, full-rank draws
determine the percentile bands; every draw remains in quality and availability
counts. Optional seed labels retain seed-specific endpoints. Native production
estimates are not recentered, and percentile bands are not calibrated CIs for
constrained or selected first-stage estimates. No addition to default conditional
variance is performed without a separate joint model.

## CMA and provider report integration

The [CMA diagnostics guide](../../../../docs/cma_diagnostics.md) owns the adapter
and rendering contracts. `infer_factor_premia` subtracts cash and credited alpha
from total forecasts. In contrast, `CmaDiagnosticInputs.excess_cma` is an already
prepared target: supplying its `admitted_alpha` field records a channel and does
not subtract that channel. Apply the chosen adjustment once before constructing
provider inference views. `compute_conditional_gls_intervals` passes this prepared
target to the core with zero cash and no further alpha subtraction.

`CmaInferenceView` and `CmaInferenceReport` are reporting containers, not aliases
of `GlsPremiaResult`. Applications populate them from the completed CMA audit,
conditional intervals and identification tables; `render_cma_inference` does no
fitting or target preparation. Contributions in the CMA tables explain revisions
from calibrated to implied premia, not factor loadings or portfolio risk shares.

To select the extended three-page layout, attach `resolution` and unchanged
`conditional_intervals` to every view. Optional `uncertainty_scenarios`,
`loading_stability`, report-level `provider_contrasts`, `loading_stability_contrasts`
and `extension_notes`
remain explicitly typed and labelled. `validate_cma_inference_report` validates
the same contracts without rendering. The application saves full-precision
sidecars, including the cash/alpha target ledger and draw-failure counts.

Consensus and JPM are consuming examples, not special cases in this package.
Their adapters retain provider/index mappings, horizon and alpha scenarios,
licensed inputs and source hashes outside `src`. Consensus compares 10-year and
20-year forecasts at alpha weights 0, 0.5 and 1; JPM also evaluates universe/proxy
exclusions. Exact reproduction requires the separately maintained local data.
The public example below is synthetic and does not reproduce either provider.

## Development and verification

From an OP checkout, install into an external environment and run the canonical
[offline example](../../../../examples/docs/gls_factor_premia.py):

```console
python -m pip install -e .
python examples/docs/gls_factor_premia.py
python -m pytest src/optimalportfolios/matf/gls_diagnostic/tests/inference_test.py
```

On the maintainer's Windows checkout, first use the repository environment launcher
and prescribed C-local interpreter as described in
[AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md);
do not create an environment inside OneDrive.
The example fits synthetic index/factor histories with FactorLasso and independently
checks OP estimates, covariance and intervals using weighted QR. Source-adjacent
Markdown is a contributor guide; do not assume it ships inside installed wheels.

Keep default inference in `_inference.py`, geometry diagnostics in `_resolution.py`,
explicit covariance scenarios in `_scenarios.py`, and paired replay in `_stability.py`.
Public imports belong in `__init__.py` and automated checks in adjacent `tests/`.
Update the methodology and CMA
contracts together when changing an interface or assumption. Preserve accepted
replication snapshots; an implementation change requires a separate parity check,
not replacement of a baseline.

## Related documentation

- [GLS derivation and canonical example](../../../../docs/gls_factor_premia.md)
- [CMA diagnostics, identification and PDF reports](../../../../docs/cma_diagnostics.md)
- [Software design and package boundaries](../../../../docs/software_design.md)
- [Factor estimation with HCGL](../../../../docs/factor_covariance_hcgl.md)
