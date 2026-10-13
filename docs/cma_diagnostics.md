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
`policy_checks`, `scenarios`, `validation` and `provenance`. It keeps numerical
precision; rendering handles display rounding. Missing optional inputs produce
explicit unavailable metrics. An inconsistent construction bridge raises an error.

The GLS residual projector uses the inverse-residual-covariance metric. Projection,
forecast splitting and factor-mimicking portfolios use one whitened SVD rank rule:
singular values must exceed machine epsilon times the larger matrix dimension times
the largest singular value. `CmaDiagnosticSpec.rcond` can select another relative
cutoff, recorded in the result. Truncating nonzero directions is a model
approximation; failed covariance accounting remains visible.

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

The renderer uses [QIS](https://github.com/ArturSepp/QuantInvestStrats) reporting
primitives; see its [software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
Rosaa supplies its completed-result and saved-cut adapters, source hashes and
transactional report bundles. The generic OP layer has no dependency on Rosaa or
the local paper replication workspace.

## References

- Sepp, A., Hansen, E. and Kastenholz, M. (2026). *Capital Market Assumptions and
  Strategic Asset Allocation Using Multi-Asset Tradable Factors*. Working paper,
  [SSRN 6785958](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958).
- [Conventions](conventions.md) and [portfolio risk analytics](portfolio_risk_analytics.md).
