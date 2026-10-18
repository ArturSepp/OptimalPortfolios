---
myst:
  html_meta:
    description: >-
      Infer factor risk premia from asset forecasts using estimated factor loadings,
      explicit cash and alpha, rank-revealing GLS and conditional uncertainty.
---

# GLS inference of factor risk premia

*Author: [Artur Sepp](https://github.com/ArturSepp)*

TradableFactors supplies the inference used by
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios) consumers.
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Generalized least squares (GLS) projects asset expected returns onto a fitted factor
loading matrix. This second stage infers the premia implicit in a provider's capital
market assumptions (CMAs). Historical returns estimate exposures and alpha; the provider's
forecasts supply the expected-return target.

## Overview

Estimate a factor model from index and factor return histories, retain its loading matrix,
historical residual means and residual covariance, then infer premia from the provider's
forecasts. The inference stage does not recalibrate those forecasts or infer cash.
Production price preparation, currency conversion, factor definitions and provider mappings
belong to the consuming application. TradableFactors supplies CMA diagnostics and PDF rendering.
Historical regression estimation
uses [FactorLasso](https://github.com/ArturSepp/FactorLasso), with its
[software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Arithmetic excess returns for estimation; provider total forecasts minus explicit cash and credited alpha for inference |
| Estimation grid | Monthly synthetic histories in the example; a fixed fitted model in GLS |
| Rebalancing grid | None; a cross-sectional inference at one valuation date |
| Covariance units | Annual residual-return covariance; monthly example variances multiplied by 12 |
| Expected returns | Annual arithmetic decimal forecasts in one currency |
| Weight state | Factor-mimicking weights are inference portfolios, with cash completing financing |
| Solver | CLARABEL for the example LASSO; Cholesky and rank-revealing SVD for GLS |

| Input | Meaning | Units and convention |
|---|---|---|
| $B$ | Estimated asset-by-factor loadings | $N$ assets by $M$ factors |
| $D$ | Residual-return covariance or diagonal variances | Positive definite, same asset order |
| $\mu$ and $r_f$ | Provider total forecasts and common cash forecast | Arithmetic decimal returns, same currency and annualisation |
| $a$ | Declared historical alpha, with asset admission weights already applied | Same return units as forecasts |
| $w$ | Diagnostic multiplier on declared alpha | Zero, one half or one are common views |

Historical total-return index prices must first be converted to excess returns using the
declared cash and currency conventions. Funded factor NAVs and excess-return factor NAVs
require different preparation. Missing series, proxy mappings and delayed observations
must be resolved explicitly before inference. Exact pandas axis alignment is required.

## Methodology

The two stages are

$$
Y_t=\alpha+B X_t+\varepsilon_t,\qquad y_w=\mu-r_f\boldsymbol 1-w a.
$$

For full column rank, define the premium map and residual operator:

$$
G=B^\top D^{-1}B,\quad A=G^{-1}B^\top D^{-1},\quad
\widehat\lambda=A y_w,\quad P_\perp=I-BA.
$$

Every forecast reconciles as $\mu=r_f\boldsymbol1+w a+B\widehat\lambda+P_\perp y_w$.
The implementation whitens with the Cholesky factor of $D$ and takes the singular value
decomposition of the whitened design, avoiding a rank decision on the squared Gram matrix.
The default relative cutoff is machine epsilon times the larger design dimension.

The columns of $Q=A^\top$ are minimum-residual-variance factor-mimicking portfolios:
$B^\top Q=I$. Their risky weights need not sum to one; cash completes financing. Large
offsetting weights explain why small forecast changes can cause large premium revisions.
The factor variance-inflation diagnostic is $G_{jj}(G^{-1})_{jj}$.

Conditional intervals require the additional working model
$y_w=B\lambda_*+e$, with $e\sim\mathcal N(0,\tau^2D)$. With $\nu=N-M$ and
$s^2=y_w^\top P_\perp^\top D^{-1}P_\perp y_w/\nu$, a pointwise interval is

$$
\widehat\lambda_j\ \pm\ t_{\nu,1-\delta/2}\sqrt{s^2(G^{-1})_{jj}}.
$$

No forecast-horizon or adviser-count divisor is added. A diagonal $D$ assumes independent
cross-asset forecast errors; a full $D$ imposes the supplied correlation shape.

Optional provider return boxes are propagated with the signs of $Q$. A negative weight
uses the upper asset endpoint for the lower factor bound. One shared cash interval enters
through $-\sum_i Q_{ij}$ once. These are sensitivity bounds, not confidence intervals or
factor quantiles; endpoint combinations need not represent any individual provider.

### Geometric resolution and residual visibility

Write the residual covariance as $D=LL^\top$ and the whitened design as
$X=L^{-1}B$. The orthogonal whitened residual projector is
$M_\perp=I-XX^+$. Its diagonal is residual visibility; one minus that diagonal
is fitted leverage. Visibility lies between zero and one and sums to
$N-\operatorname{rank}(X)$. With diagonal $D$, whitened coordinates correspond
to individual assets. For dense $D$, the implementation uses ordered Cholesky
coordinates, which mix assets; their visibility is not an individual original
asset quantity.

For a full-rank design and fixed annual factor volatility $\sigma_j$, define
normalized resolution per unit error scale as

$$
q_j=\frac{\sqrt{(G^{-1})_{jj}}}{\sigma_j}.
$$

This compares design precision across consistent factor scalings. It excludes
the provider residual scale. The normalized conditional interval half-width is
$t_{\nu,1-\delta/2}s q_j$, which additionally depends on the forecast and alpha
admission. Changes in alpha can therefore change conditional precision without
changing geometric resolution. Annual half-width 0.01 and normalized half-width
0.25 are default diagnostic targets, not estimation constraints or rank tests.

Low residual visibility means that a sleeve's view is largely absorbed by the
factor fit. A small residual on that sleeve provides little evidence of agreement
with the factor model. It does not imply that the view has little influence on
premia. Gross absolute mimicking weights measure that influence separately.

### Explicit covariance scenarios and provider contrasts

For a declared covariance $C_y$ of the alpha-adjusted forecast, propagate

$$
V_\lambda=A C_y A^\top.
$$

A covariance alone supports standard errors. Normal intervals additionally assume
centered Gaussian forecast errors with that known covariance. An independently
estimated proportional reference scale requires its own sampling model and
reference degrees of freedom before Student-t intervals are justified. These
alternatives are separately labelled scenarios; they never replace the default
own-residual-scale intervals automatically.

The estimand is the projected provider mean. If the mean forecast includes a
deliberate shift $u$, the projected mean shifts by $A u$. Coverage of this mean
does not establish coverage of an unshifted structural premium. A shift in the
factor span can change premia without changing the residual score, so residual
disagreement does not identify premium bias.

For providers $p$ and $q$, with possibly different maps $A_p$ and $A_q$, the
covariance of their premium difference is

$$
V_{p-q}=A_p C_{pp}A_p^\top+A_q C_{qq}A_q^\top-A_p C_{pq}A_q^\top-A_q C_{qp}A_p^\top.
$$

Supply the cross covariance explicitly, including zeros when asserting
independence. The complete joint covariance must be positive semidefinite.
When dependence is unmeasured, use labelled feasible correlation scenarios and
avoid a single significance claim. Provider-filled or shared forecast cells
must remain identified in the consuming application's provenance.

Adviser quartiles can supply a disagreement scenario: divide marginal IQRs by
the normal IQR, approximately 1.3489795, then apply an explicit correlation model.
The result describes propagated adviser dispersion, not estimation error of the
consensus average or forecast accuracy. Approximate conversions from geometric
to arithmetic forecasts, cash treatment and cross-asset correlation assumptions
must be stated. Neither adviser count nor forecast horizon is an implicit divisor.

### Paired loading and alpha stability

Loading stability replays saved, paired draws of loadings, residual covariance
and, where specified, admitted alpha. The same draw serves every provider, so
paired provider differences preserve cross-provider dependence. The fixed-alpha
and jointly resampled-alpha policies are separate diagnostics.

Percentile bands use only finite, successful, full-rank draws while retaining
every attempted draw in the availability denominator. Rank losses, solver and
geometry failures, minimum singular values and optional seed-specific endpoints
accompany the bands. The baseline must use the same common history and estimator
conventions as the draws. A native production estimate using different histories
is a separate comparison and is not recentered to the bootstrap median.

These are loading stability bands, not calibrated confidence intervals for a
constrained or selected estimator. They are not added to conditional covariance
without an explicit joint uncertainty model. Compare widths at the same nominal
level and retain deterministic forecast boxes alongside both objects.

## Worked example

The [offline script](../examples/docs/gls_factor_premia.py) constructs synthetic monthly
simple excess-return NAVs, recovers their returns, fits LASSO and estimates historical
residual means. It uses 240 observations, eight assets, three factors, uniform observation
weights, no missing data and CLARABEL. The factor design has rank three and five residual
degrees of freedom. It is a teaching example, not a production MATF calibration.

```python
    model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5,
                       span=None, demean=True, solver="CLARABEL").fit(x=x, y=y)
    betas = model.coef_
    historical_residual = y - x @ betas.T
    # Arithmetic annualisation of monthly simple excess returns; no compounding bridge.
    alpha = 12*historical_residual.mean()
    residual_variances = 12*historical_residual.var(ddof=0)
    provider_cma = pd.Series([.08, .065, .06, .07, .05, .09, .055, .075], index=assets)
    views = {}
    for weight in (0., .5, 1.):
        views[weight] = infer_factor_premia(
            betas, residual_variances, provider_cma, cash_rate=.03,
            alpha=alpha, alpha_weight=weight)
    result = views[1.]
```

The script independently checks estimates and intervals using weighted QR and verifies that
the change from zero to full alpha credit equals the mimicking portfolio's exposure to alpha.
Run `python examples/docs/gls_factor_premia.py` from a checkout.

## Implementation in optimalportfolios

The supported inference import is now `tradablefactors.premia`; the old OP
analytics imports have been removed. Install TradableFactors alongside OP for
this composed example. Its `premia/README.md` describes module responsibilities
and contributor checks; `_inference.py` is private. OP consumes the resulting
expected returns and covariance for constrained optimisation.

- `infer_factor_premia` returns `GlsPremiaResult`: exact asset reconciliation, premium
  estimates, conditional covariance and intervals, mimicking weights, VIFs and optional boxes.
- `compute_gls_geometry` returns `GlsGeometry`, including the singular values and common cutoff.
- `gls_projector` returns the asset-space residual operator. `gls_split` accepts an already
  cash/alpha-adjusted target and returns minimum-norm coordinates and residuals.
- `factor_mimicking_portfolios` returns asset-by-factor weights and rejects deficient rank.
- `compute_gls_resolution` returns `GlsResolutionResult`, with asset visibility,
  factor resolution, current conditional half-widths and diagnostic settings.
- `compute_gls_uncertainty` returns `GlsUncertaintyScenario` for an explicitly
  supplied positive semidefinite forecast covariance. Its default distribution
  is `none`, so covariance alone does not create interval endpoints.
- `compute_gls_reference_uncertainty` supplies a known-scale normal or explicitly
  independent estimated-reference Student-t scenario with recorded provenance.
- `compute_gls_adviser_disagreement` supplies a labelled IQR-based Gaussian
  disagreement scenario; `compute_gls_provider_contrast` propagates complete
  paired forecast covariance through the providers' separate maps.
- `compute_gls_loading_stability` solves all providers together for each paired
  estimator draw. `summarize_gls_loading_stability` accepts existing premium
  draws. Both return `GlsLoadingStabilityResult` objects with bands, draws,
  draw status and assumptions. `compute_gls_loading_stability_contrast` takes
  paired result differences using the intersection of available draws.

These are exported from `tradablefactors.premia` and the TradableFactors package
root. There are no OP forwarding exports. The GLS calculation uses NumPy, pandas
and SciPy and does not call
FactorLasso. All return units are supplied by the caller;
annual inputs produce annual premium outputs. Forecasts and input frames are not modified.

### Total forecasts and prepared audit targets

`infer_factor_premia` prepares the target from total forecasts. Its `alpha` argument
must already contain any per-asset admission weights; `alpha_weight=1` credits that
vector in full, not automatically every historical residual mean. Changing the
multiplier alone leaves the historical fit unchanged.

The [CMA audit](cma_diagnostics.md) instead consumes a prepared `excess_cma` vector.
For a provider view, apply cash and the selected alpha adjustment before building
that input. Its `admitted_alpha` field is descriptive and does not subtract alpha.
`compute_conditional_gls_intervals` delegates to this GLS core with zero cash and
no further alpha adjustment, retaining the CMA result schema and rank cutoff.

`GlsPremiaResult` is not a PDF report container. Provider adapters supply completed
CMA accounting, conditional intervals and identification results to
`CmaInferenceView` and `CmaInferenceReport`; `render_cma_inference` only presents
them. The core accepts full residual covariance or diagonal variances, while the
CMA audit adapter uses diagonal residual variances. Keep that metric choice fixed
when comparing estimates and intervals.

### Provider workflows

For each provider, the application maps line items to tracking indices, estimates
the historical model, prepares the provider's arithmetic forecasts and admitted
alpha, and computes each horizon/alpha view before rendering. Consensus examples
compare 10-year and 20-year forecasts at weights 0, 0.5 and 1. JPM examples add
universe and proxy exclusions. Those mappings, model definitions, source vintages
and licensed data belong to the consuming workflow; neither provider is hard-coded
in the inference package. Save full-precision tables and input/source hashes beside
the report. The public script above uses synthetic data only.

## Interpretation and limitations

Full rank is necessary for unique factor coordinates, not proof of useful precision. At
deficient rank the minimum-norm coordinates remain available as a convention in `premia`,
while identified estimates, intervals and mimicking weights are marked unavailable. No
residual degrees of freedom or numerically zero residual scale also suppresses intervals.

The conditional intervals hold loadings, residual risk, alpha, cash and factor definitions
fixed. They exclude first-stage estimation uncertainty and do not measure provider accuracy.
Changing alpha credit can narrow an interval without validating that admission policy.
Use first-stage resampling and independent sensitivity analyses for different questions.
The optional covariance scenarios and percentile stability bands do not establish
empirical coverage by their presence in a report. A consensus residual scale is
not automatically the error scale of an individual provider, and a residual-score
ratio is descriptive unless its joint sampling model has been established.
This adaptation of GLS adds neither an intercept nor an economically identified premium
where the supplied factor span cannot identify one.

## See also

- [CMA diagnostics and reports](cma_diagnostics.md)
- [Factor covariance estimation](factor_covariance_hcgl.md)
- [Return and risk conventions](conventions.md)

## References

- Aitken, A. C. (1935). On Least Squares and Linear Combination of Observations.
  *Proceedings of the Royal Society of Edinburgh*, 55, 42-48.
  [Publisher record](https://doi.org/10.1017/S0370164600014346).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
