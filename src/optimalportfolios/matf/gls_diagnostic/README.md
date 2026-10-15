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

`GlsGeometry`, `GlsPremiaResult`, `compute_gls_geometry` and `infer_factor_premia`
are also exported by `optimalportfolios.matf` and the package root. Existing
`sharpe_accounting` projection and mimicking-portfolio entry points delegate here,
preserving their import paths.

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

Keep numerical changes in `_inference.py`, public imports in `__init__.py`, and
automated checks in `tests/inference_test.py`. Update the methodology and CMA
contracts together when changing an interface or assumption. Preserve accepted
replication snapshots; an implementation change requires a separate parity check,
not replacement of a baseline.

## Related documentation

- [GLS derivation and canonical example](../../../../docs/gls_factor_premia.md)
- [CMA diagnostics, identification and PDF reports](../../../../docs/cma_diagnostics.md)
- [Software design and package boundaries](../../../../docs/software_design.md)
- [Factor estimation with HCGL](../../../../docs/factor_covariance_hcgl.md)
