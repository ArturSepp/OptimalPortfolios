---
myst:
  html_meta:
    description: >-
      Audit completed capital market assumptions with a rank-aware GLS projection,
      factor and residual Sharpe accounting, explicit admission policies and a
      compact PDF, without changing the forecasts or estimating a new model.
---

# CMA diagnostics and reports

*Author: [Artur Sepp](https://github.com/ArturSepp)*

[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios) audits an already
computed expected-return vector against its supplied factor model. It returns
complete numerical tables separately from a two- or three-page PDF. The audit
does not estimate CMAs, change an admission weight or replace published forecasts.

Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Inputs and computation

Import `CmaDiagnosticInputs`, `CmaDiagnosticSpec`, `AdmissionPolicy` and
`compute_cma_diagnostics` from `optimalportfolios.matf`. The input requires labelled
`betas`, strictly positive `residual_variances` and `excess_cma`, all in exactly the
same asset order. The factor columns define the order of optional
`factor_covariance` and calibrated `factor_premia`. Expected returns are annual
arithmetic excess decimals in one currency; covariances are annualized decimals.
The caller supplies the cash convention and provenance. No annualization or
currency conversion is inferred.

Optional `channels` explain the difference between calibrated factor-implied and
published excess returns. They include admitted historical alpha and any regional,
discretionary, funding, pricing-override or currency adjustments. A table of
`scenarios` contains absolute annual excess returns with the same asset index.
`asset_metadata` supplies display fields such as name, asset class and observation
count. These descriptive fields cannot shadow calculated result columns.

`CmaDiagnosticResult` holds `summary`, `assets`, `factors`, `channels`,
`policy_checks`, `scenarios`, `validation`, `provenance`, `factor_contributions`,
`factor_sensitivity` and `factor_leave_one_out`.
It keeps numerical
precision; rendering handles display rounding. Missing optional inputs produce
explicit unavailable metrics. An inconsistent construction bridge raises an error.

The GLS residual projector uses the inverse-residual-covariance metric. Projection,
forecast splitting and factor-mimicking portfolios use one whitened SVD rank rule:
singular values must exceed machine epsilon times the larger matrix dimension times
the largest singular value. `CmaDiagnosticSpec.rcond` can select another relative
cutoff, recorded in the result. Truncating nonzero directions is a model
approximation; failed covariance accounting remains visible.

The numerical GLS implementation belongs to OP's
[`matf.gls_diagnostic`](gls_factor_premia.md): `infer_factor_premia` accepts
provider total-return forecasts, cash, admitted alpha, estimated loadings and
residual risk. It returns implied premia, conditional intervals, mimicking
portfolios and optional forecast-box sensitivities. OP delegates its existing
projection functions to that implementation and retains CMA accounting, policy
diagnostics, identification contrasts and PDF rendering. FactorLasso supplies the
historical estimation stage; GLS requires no additional FactorLasso API or release.

For provider inference, estimate historical loadings and alpha from tracking-index
prices first, then apply the GLS map to the provider's forecasts. Do not substitute
historical means or model-generated asset forecasts for the provider's target.
Cash is subtracted once, and the alpha multiplier is recorded for every view.

### Choose the input interface

| Entry point | Forecast input | Cash and alpha handling | Residual metric |
|---|---|---|---|
| `infer_factor_premia` | Provider arithmetic **total** forecasts | Subtracts explicit `cash_rate` and `alpha_weight * alpha`; alpha already includes asset admission weights. | Positive variances or full positive-definite covariance. |
| `compute_cma_diagnostics` | Prepared **excess** forecast target | Does not subtract cash or `admitted_alpha`; channels explain the supplied target. | Diagonal residual variances. |
| `compute_conditional_gls_intervals` and `compute_cma_identification` | The same prepared target used for the audit | No further adjustment; the interval adapter calls the GLS core with zero cash. | The audit's diagonal residual variances and rank specification. |

For provider comparisons, hold estimated loadings and residual risk fixed and
prepare one target for each horizon and alpha multiplier. A full-alpha view means
the entire **CMA-spec admitted** vector, which can differ from raw historical alpha.
Keep a production audit of the published forecast distinct from a provider view
whose admitted alpha has already been removed. Reconcile the construction channels
to the target actually supplied.

The [GLS module README](https://github.com/ArturSepp/OptimalPortfolios/blob/main/src/optimalportfolios/matf/gls_diagnostic/README.md)
lists public entry points, result fields and contributor commands. The
[methodology guide](gls_factor_premia.md) gives the equations and conditional model.

## Interpretation

The total unconstrained squared excess Sharpe splits into the systematic score at
the forecast's **implied** GLS premia and its full residual score. The calibrated
systematic score is a different metric, used to measure access to the factor
ceiling and to define the optional admission budget.

Raw admitted-alpha score, projected admission residual and full-forecast residual
are three separate quantities. Regional and other adjustments can offset or
reinforce admitted alpha. Channel residual attributions are signed and additive;
standalone channel squared scores are not generally additive.

The default `AdmissionPolicy` is explicitly named `paper_illustrative`. It limits
positive admitted information ratios to 0.50, admission shares of positive excess
forecasts to 50%, and raw admission score to half of calibrated achievable
systematic squared Sharpe. Passing no policy requests no policy verdict. Nonpositive
forecast denominators are not automatically passes, and negative admissions are
not subjected to an invented absolute-value cap.

An asset set with full row rank is saturated: its residual is mechanically zero
and has no residual diagnostic power. With deficient factor rank, fitted asset
means and residuals remain useful but implied factor coordinates use a minimum-norm
convention. Identification scales are not standard errors or confidence intervals.
Residuals can reflect deliberate views, omitted risks or estimation error.

Every subset must be projected again. Do not slice residuals from a larger
universe. Do not include cash with invented variance, or treat a synthetic
portfolio and its component sleeves as independent opportunities. The CMA audit
uses diagonal residual covariance; it is distinct from the funding-neutral
opportunity metric and from correlated production portfolio risk.

## Rendering and persistence

`optimalportfolios.reports.cma_diagnostics.render_cma_diagnostics` consumes a result
and an explicit PDF output path. It shows forecast consistency, admission and model
quality, and a third scenario page when scenarios exist. Large universes show
selected contributors with the remaining coverage disclosed. Save the complete
result tables and input provenance alongside the PDF in the application's storage
layer. Report rendering performs no fitting or portfolio optimisation.

For provider studies, `optimalportfolios.reports.cma_inference` exposes
`CmaInferenceView`, `CmaInferenceReport` and `render_cma_inference`. The adapter
supplies up to six completed views, a headline view, source vintage, model date,
horizon, cash and alpha multiplier. The shared three-page PDF compares premia
and scenario diagnostics, shows the headline's top two positive and bottom two
negative contributors for every factor, and reports forecast sensitivity and
leave-one-out ranges. It uses the same QIS rendering primitives as the audit.
The renderer does not construct scenarios, estimate exposures or subtract alpha.
Provider adapters must subtract the already policy-admitted alpha exactly once
before computing the supplied results. Forecast and model dates stay distinct.

Views must share their calibrated factor basis. Missing or nonidentified
factor coordinates appear as `N/A`, including Private Equity when a provider's
benchmark-only universe has no exposure. This is not a zero premium. Failed
accounting, duplicate view identities and oversized compact reports are rejected.
Complete scenario tables, mapping qualifications and provider forecast-band
assumptions remain in the caller's data bundle; the headline pages do not replace
those records. PDF publication is atomic and leaves existing output intact if
rendering fails.

For admission sensitivity, `without_admitted_alpha(inputs, spec=spec)` returns an
owned counterfactual input. It subtracts only the supplied admitted alpha from
base and scenario excess CMAs, zeros its construction channel, and keeps all
other channels, risk inputs and scenario shocks fixed. Missing admission raises
an error. Compute this input with the same diagnostic specification and policy,
then pass its result as `zero_alpha_result` to `render_cma_diagnostics` alongside
the original result. The original production forecasts remain unchanged.

With three pages, this view compares forecast accounting and all displayed
factor premia, then gives each view a contributor page. With two pages, it gives
the two contributor pages. Top and bottom mean the five largest positive and
five most negative **signed GLS residuals**, respectively; fewer rows are shown
when a sign has fewer than five assets. Ties preserve input order. Residual
squared-Sharpe contributions and shares are also displayed; these are
nonnegative for both signs. This residual ranking is distinct from the compact
factor-contribution table below it. Each factor row shows its revision and the
two largest positive and two most negative asset contributions, in annual basis
points. Full channels, policy checks and scenario tables belong in the
saved bundle; the paired PDF replaces the separate scenario page.

The numerical `factor_contributions` table has assets in rows and factors in
columns. With GLS coefficient map `A`, each entry is
`A[k, i] * (excess_cma[i] - (betas @ factor_premia)[i])`. These are contributions
to the **revision from calibrated to GLS-implied premia**, not beta exposures,
portfolio risk contributions or full asset returns. All asset entries sum to
each revision at full factor rank; the four displayed tails generally do not.
With deficient rank, the additional factor-column `revision_attribution_offset`
is `(A @ betas - I) @ factor_premia`. This non-asset coordinate offset remains
explicit, and asset sums plus offset reconcile to the revision. Missing
calibrated premia leave the contribution table empty. The numerical validation
includes this reconciliation. Legacy frozen results without the table retain
their existing layout.

## Factor identification and forecast influence

The long-form `factor_sensitivity` table separates each asset contribution into
its forecast gap and GLS multiplier. The gap is `excess_cma - betas @ factor_premia`;
the product with `gls_multiplier` reproduces the asset's premium-revision
contribution. The audit projects the supplied forecast exactly once: callers
comparing alpha-adjusted consensus views subtract their chosen alpha before
constructing the input. Supplying `admitted_alpha` alone does not subtract it.

`premium_change_25bp` is the change in the factor premium when one asset's annual
excess forecast increases by 0.0025, holding cash, alpha and the fitted risk model
fixed. For example, a multiplier of -4 means a 25 bp higher asset forecast lowers
the factor premium by 100 bp. This derivative is distinct from the asset's beta
and from its contribution at the current forecast. Large offsetting multipliers
can produce a plausible premium that remains sensitive to small forecast changes.

`weighted_loading_overlap` in `factors` measures the share of a factor's whitened
loading column explained by the remaining columns, without an intercept. This is
cross-sectional loading overlap, not time-series return correlation.
`conditional_information_share` allocates the remaining loading signal across
assets and sums to one for an identified factor. `gross_gls_multiplier` and
`max_abs_25bp_shift` report the size of the inverse mapping and its most sensitive
single-asset forecast. These are deterministic geometry measures, not standard
errors or statistical confidence statements.

`factor_leave_one_out` removes each asset and recomputes GLS under the same rank
policy, holding the remaining forecasts, betas and residual variances fixed. It
does not reestimate historical loadings. The saved table identifies the excluded
asset, retained rank, premium and change from the full universe. The factor table
summarizes identified deletion estimates in `loo_low`, `loo_high` and
`loo_max_abs_change`, with the worst excluded asset and explicit available and
unavailable counts. A nonunique base fit, rank-deficient deletion or empty retained
universe has unavailable comparative premia; minimum-norm coordinates never enter
these ranges. Deletion ranges are sensitivity checks, not confidence intervals.

The compact PDF displays maximum absolute 25 bp shifts and identified deletion
ranges next to factor premia and contributor rows. Full gap/multiplier and deletion
tables belong in the saved bundle. Legacy frozen results without these tables keep
their previous layout and are never recomputed during rendering.

The renderer uses [QIS](https://github.com/ArturSepp/QuantInvestStrats) reporting
primitives; see its [software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
Rosaa supplies its completed-result and saved-cut adapters, source hashes and
transactional report bundles. The generic OP layer has no dependency on Rosaa or
the local paper replication workspace.

## Conditional GLS confidence intervals

`compute_conditional_gls_intervals(inputs, confidence_level=0.95, spec=spec)` is
an opt-in numerical calculation exported from `optimalportfolios.matf`. Its
`CmaConditionalGlsResult` contains `factors`, joint `covariance` and `summary`.
It neither modifies nor extends the frozen `CmaDiagnosticResult` tables. The
caller applies cash and admitted alpha to the forecast before calling it, using
the same inputs and SVD cutoff as the audit.

The working model is `y = B @ lambda + e`, with zero-mean Gaussian forecast errors
having covariance `tau_squared * D`. Loadings, diagonal residual-return covariance,
cash and admitted alpha are fixed. Cross-asset forecast errors are independent;
using return-risk D to specify their relative variances is an explicit assumption,
not an empirical estimate of provider forecast accuracy.

For full column rank K and N greater than K, the scale estimate is weighted
residual SSE divided by N-K. Premium covariance is that scale times the inverse
of `B.T @ inv(D) @ B`, evaluated through whitened SVD. Each interval uses the
Student-t critical value with N-K residual degrees of freedom. Forecast horizon
and adviser count do not set those degrees of freedom. Confidence is pointwise,
not simultaneous across factors or scenarios. The result includes the confidence
level, critical value, estimated scale and assumptions so a saved table remains
interpretable without its PDF.

Rank-deficient factors, zero residual degrees of freedom and numerically zero
residual scale have unavailable interval endpoints and explicit status codes.
A minimum-norm coefficient must not acquire a finite interval. These intervals
exclude uncertainty in fitted exposures, residual risk, cash, admitted alpha and
factor definitions, and do not represent realized-return prediction bands.

Pass the completed result as `CmaInferenceView.conditional_intervals` for every
view, or omit it for every view to retain the previous layout. The common report
shows estimates and intervals for all scenarios, their residual degrees of
freedom and the working assumptions. Existing 25 bp sensitivities, leave-one-out
ranges and adviser forecast-box envelopes retain their separate interpretations.

## Individual and pair identification

The `compute_cma_identification` entry point is called as
`compute_cma_identification(inputs, factor_pair=('Carry G10', 'Carry EM'),
precision_target=None, equivalence_margin=None, confidence_level=0.95, spec=spec)`
returns a separate `CmaIdentificationResult` containing `factors`, `contrasts`,
`directions`, `pair` and `summary`. Supply the same already alpha-adjusted input
as for the audit and conditional intervals. Existing results are unchanged.

Individual conditional information is the squared norm of a whitened factor
column after removing the other columns, without an intercept. Its reciprocal
is residual mimicking variance. Variance inflation compares raw and distinct
loading information; it does not detect weak absolute exposure by itself.
The optional `precision_target` is a native annual decimal CI half-width:
0.01 means a one-percentage-point precision target. `weak_for_target` means
the interval exceeds this declared requirement, not failure of a rank test.

Pair geometry removes all remaining factors before taking the two-column SVD.
Columns are multiplied by their fitted factor volatilities, so coordinates
are premia divided by volatility. The minimum/maximum information, their ratio,
partial loading cosine, mutual variance inflation and weak/strong directions
separate lack of exposure from pair redundancy. Eigenvector signs are fixed
by a positive largest component; repeated eigenvalues have nonunique directions.

Common is `(first + second)/2`; spread is `first - second`. Native contrasts
are annual excess decimals. Normalized contrasts apply the same operations to
premia divided by their respective factor volatilities. They use the full joint
GLS covariance, including off-diagonal entries. These are unrestricted-model
contrasts, not estimates from a pooled factor model. Risk factors remain separate.

The equality F test imposes zero normalized spread. Non-rejection is not
equivalence. An explicit `equivalence_margin`, in normalized-premium units,
enables two one-sided tests at level `1-confidence_level`. At 95% confidence,
equivalence requires the 90% spread CI to lie inside the declared bounds.
Otherwise status is `not_established`; no threshold means `not_assessed`.
All inference inherits fixed B, D, cash and alpha assumptions and additionally
holds factor volatilities fixed. It does not address estimated-loading uncertainty
or establish that two realized risk factors can be merged.

Pass `CmaInferenceView.identification` for every view alongside its matching
`conditional_intervals`. The three-page report displays individual precision,
native and normalized carry contrasts, pair diagnostics and equivalence status.
Full information, direction intervals, TOST bounds and settings belong in CSV/JSON
exports; existing leave-one-out and forecast sensitivity ranges stay distinct.
Missing pair factors, missing factor covariance, rank/df failures and zero error
scale are explicitly unavailable. Native contrasts do not require factor covariance.

## Extended inference reports

The extended layout keeps `conditional_gls_student_t` as the default interval
method. `CmaInferenceView.resolution` accepts a completed `GlsResolutionResult`
from `compute_gls_resolution`. Supply resolution and conditional intervals for
every view to select this layout. Existing report calls without resolution retain
their earlier layout and numerical contracts.

The three pages cover premia and precision, benchmark revision attribution and
residual visibility, then loading stability and explicit comparisons. The report
retains top and bottom two signed benchmark revision contributors for every
factor, forecast shocks and leave-one-out ranges. These contributions sum to the
revision from calibrated to inferred premia; they do not measure premium bias.
Annual precision 0.01 and normalized precision 0.25 are diagnostic targets. Print
the continuous half-width beside each flag, especially near a threshold.

`CmaInferenceView.uncertainty_scenarios` accepts a tuple of separately computed
`GlsUncertaintyScenario` results. Each carries its method, projected target,
confidence level, covariance, calibration status, reference provenance and
assumptions. Known forecast covariance, independently estimated reference scale
and adviser disagreement remain distinct. A covariance-only result has no
statistical interval endpoints. The renderer neither estimates a reference nor
replaces the default intervals when an alternative is narrower.

`CmaInferenceView.loading_stability` accepts `GlsLoadingStabilityResult`. The
report shows the matching common-history baseline, percentile endpoints,
full-rank availability and failure counts. It keeps the native estimate distinct
from the bootstrap baseline and median. Constrained-estimator percentile bands
are stability diagnostics, not calibrated confidence intervals. Use paired alpha
draws for policy-consistent admission; label a fixed-admission comparison explicitly.

`CmaInferenceReport.provider_contrasts` accepts explicitly paired covariance
scenarios; `loading_stability_contrasts` accepts paired bootstrap differences
with their matching baseline and joint draw availability. `extension_notes` records
missing optional inputs, mapping qualifications and treatment of shared or filled
forecasts. Missing paired draws must remain unavailable; do not reuse a different
universe's bootstrap or invent a consensus reference for another currency.

`validate_cma_inference_report(report)` checks completed results without drawing
figures or writing files. It uses the same validation as `render_cma_inference`:
CI, factor, geometry, alpha-view and explicit-scenario identities must agree.
Applications can therefore validate numerical-only bundles before publication.

### Persistence and replay

Save each target input with total forecast, cash, already policy-weighted alpha,
the diagnostic admission multiplier and the resulting GLS target. Assert that
the target equals total forecast minus cash minus credited alpha. Regenerate
descriptive alpha metadata from that reconciliation after a historical refit;
stale provenance must never change or silently contradict the numerical target.

Keep full-precision asset visibility and factor resolution tables, default
conditional covariance and assumptions, explicit scenario covariances, loading
draw status, seed and baseline provenance, and source hashes beside the PDF.
Adviser forecast boxes retain their signed worst-case interpretation. Supplied
provider lines and cells filled from another source require separate flags.

Report-only replay consumes saved estimates and forecasts. It does not change
historical fitting, factor definitions, inclusion flags or the valuation cutoff.
Archived results remain immutable; new diagnostics are separate outputs rather
than replacement regression baselines. The mathematical definitions and scenario
assumptions are in [GLS inference](gls_factor_premia.md).

## References

- Sepp, A., Hansen, E. and Kastenholz, M. (2026). *Capital Market Assumptions and
  Strategic Asset Allocation Using Multi-Asset Tradable Factors*. Working paper,
  [SSRN 6785958](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958).
- [Conventions](conventions.md) and [portfolio risk analytics](portfolio_risk_analytics.md).
- [Statsmodels linear-restriction F tests](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.f_test.html)
  and [the two-one-sided-test equivalence principle](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttost_ind.html).
