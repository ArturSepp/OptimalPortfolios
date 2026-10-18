---
myst:
  html_meta:
    description: >-
      An offline FactorLasso, MATF and QIS workflow separating conditional alpha uncertainty,
      ex-ante risk, forward attribution, trading costs and unexplained accounting effects.
---

# Factor diagnostics and attribution uncertainty

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Estimation and inference use [FactorLasso](https://github.com/ArturSepp/FactorLasso)
([citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff)); execution, risk and
reporting use [qis](https://github.com/ArturSepp/QuantInvestStrats)
([citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)).

This source-only guide connects existing factor estimates, MATF alpha geometry and QIS risk
and attribution. It keeps uncertainty about an estimated residual mean separate from future
return risk and from accounting reconciliation. It introduces no new estimator or risk API.
Use the matching source snapshot; this combined example has no released-version guarantee.

## Run the workflow

The [canonical offline script](../examples/docs/factor_diagnostics_workflow.py) generates a
fixed synthetic monthly sample and runs without a vendor, network connection or optional
comparator. From a C-local source export, using the prescribed external interpreter:

```console
python -m examples.docs.factor_diagnostics_workflow
```

`run_workflow()` returns labelled diagnostics, interval/risk tables, QIS reporting objects
and provenance. `assert_references(result)` checks independent calculations;
`coverage_experiment(result)` runs the stated synthetic coverage experiment.
`render_report(result)` returns a figure assembled with QIS table and NAV plotting APIs.
The script writes no files. A caller may save intentional output on C-local storage.

## Timing, data and units

| Item | Contract in this example |
|---|---|
| Data | Six assets and two factors, 240 monthly simple fractional returns, one valuation currency. |
| Beta fit | First 72 months, existing `LassoModel`, fixed regularization and no later refit. |
| Alpha and risk | Next 120 months, conditional on the frozen fitted loadings. |
| Forward period | Last 48 months, untouched by estimation and selection. |
| Statistical alpha | EWMA of original-unit residuals `y - X @ beta.T`, span 36 observations. It is not the demeaned solver diagnostic or the regression intercept. |
| Moment units | Arithmetic annualisation: alpha times 12; return covariance times 12; alpha estimation covariance times 144. This is not compounded annual growth. |
| Inference calendar | Monthly observation ordinals, Bartlett bandwidth six months; estimator weights retain the native recursion and initialization. |
| Portfolio policy | Fixed, externally chosen long-only books; four decisions, actual executed state from L2, implementation lag one observation. |
| Cash and costs | Initial cash 100; zero cash/funding rate; proportional trade fees of 10 bp, applied by QIS. |

The fitted snapshot records its availability date. Every alpha/risk observation ends by the
first decision close; the first execution is at the next monthly close. The forward report
uses actual beginning-period weights after drift and earlier trades. No future observation
enters training. Post-beta training and held-out residual-diagonality diagnostics are reported
separately; a failed diagnostic remains visible and is not silently repaired.

## What each diagnostic measures

| Quantity | Existing owner and meaning | What it does not establish |
|---|---|---|
| Statistical alpha | FactorLasso `estimate_alpha_uncertainty`: initialized EWMA residual mean and joint calendar HAC covariance. | Skill, an economic forecast, or a post-selection significance test. |
| Factor-spanned alpha | MATF `gls_split` and `gls_projector`: projection onto the fitted loading span under diagonal residual-risk weighting. | Realised factor contribution or an independent source of portfolio return. |
| Factor-orthogonal alpha | The complementary residual-mean vector; its weighted inner products with the factor loadings vanish. | Pure idiosyncratic risk or persistent investment skill. |
| Conditional book interval | MATF `fixed_portfolio_alpha_interval` and FactorLasso `linear_confidence_intervals`: uncertainty of a fixed active book's estimated mean. The joint covariance is retained across targets. | Prediction intervals for next month's portfolio return. |
| Ex-ante tracking error | `OpportunityModel` and `portfolio_opportunity_metrics`, delegating to canonical `qis.RiskModel`. Factor and diagonal residual return variances sum to total model variance. | A guarantee that the model's diagonal residual assumption is correct, or that realised TE will equal it. |
| Realised TE | QIS `compute_te_ir_errors` and `compute_ewma_realised_tracking_error` on implemented NAV return differences. | Alpha estimation error or another estimate of the factor-spanned mean. |
| Accounting residual | Active NAV return less realised factor/residual contributions and the canonical QIS trade-cost effect. | Alpha or evidence of skill. A nonzero gap calls for accounting/convention investigation. |

`OpportunityModel`, `portfolio_opportunity_metrics` and `fixed_portfolio_alpha_interval` are
existing documented module imports from `optimalportfolios.matf.opportunity` and
`optimalportfolios.matf.uncertainty`; this guide adds no root re-exports.
The [risk-model adapter](https://github.com/ArturSepp/tradablefactors/blob/main/src/tradablefactors/risk.py),
[MATF opportunity diagnostics](../src/optimalportfolios/matf/opportunity.py) and
[MATF uncertainty propagation](../src/optimalportfolios/matf/uncertainty.py) retain numerical ownership.

## Intervals, dependence and calibration

The workflow's normal/HAC intervals are **approximate and conditional** on the fitted
loadings, residual-risk geometry and factor definitions. They exclude LASSO selection,
beta estimation, risk-model estimation, changing means and forecast uncertainty. A separate
beta sample avoids reusing the same observations to fit loadings and evaluate residual means;
it does not remove uncertainty in the estimated loadings.

The existing FactorLasso dependent-wild bootstrap uses shared Gaussian calendar multipliers
for the whole residual panel. It preserves the relevant joint score dependence and the fixed
estimator weights; each draw re-estimates its HAC standard error. Independently shuffling dates
or resampling each asset separately would break these contracts. The bootstrap-t table retains
`bootstrap_approximation_unvalidated`: a covariance/decomposition check does not confer nominal
coverage, posterior credibility or validity after selecting a portfolio.

The independent coverage experiment generates 3,000 complete Gaussian AR(1) panels per temporal
correlation, with 96 monthly observations, six assets, cross-asset correlation 0.25, monthly
residual volatility 1%, and temporal coefficients -0.3 and 0.5. Loadings, residual-risk geometry,
three linear contrasts and EWMA weights of span 24 are fixed throughout this experiment.
FactorLasso's known-shape Gaussian critical value receives the **true** temporal covariance
shape. Pointwise 95% coverage must lie within five binomial standard errors of 95% for every
contrast and temporal coefficient. The script also measures ordinary normal/HAC coverage,
without requiring it to be 95%. Substituting an estimated shape would lose the known-shape
calibration interpretation.

This experiment validates the stated fixed-model Gaussian setting. It does not establish
unconditional coverage for fitted loadings, a changing endpoint, missing-data designs or
bootstrap-t intervals. Its tolerance is declared before observing the fixed-seed results.

## Realised attribution and verification

The forward calculation applies frozen factor loadings to realised factor returns and actual
lagged active weights. Realised asset residuals are the difference between observed asset
returns and that factor component. QIS `PortfolioData.get_instruments_pnl` supplies gross and
net arithmetic contributions, so their difference identifies the trade-cost effect without
another cash ledger. The remaining NAV reconciliation gap is shown separately.

The displayed totals are sums of monthly arithmetic contributions, not compounded relative
wealth. Zero funding/management fees, complete quotes and a single currency make the synthetic
accounting identity exact to rounding. In production, financing, carry, FX, missing marks or
other fees need explicit reconciliation; they must not be relabelled residual alpha.

The script checks GLS decomposition and covariance propagation against independent whitened
QR, risk against an independent matrix calculation, and realised TE against a sample standard
deviation. Regression checks cover future shocks, held-out timestamps, annualisation, actual
state, bootstrap joint targets and the prespecified coverage gate. The report uses QIS rendering;
no new displayed analytics figure or reviewed exhibit is added to this page.

## See also

- [Factor covariance estimation](factor_covariance_hcgl.md) and [risk analytics](portfolio_risk_analytics.md)
- [Existing MATF GLS diagnostics](gls_factor_premia.md) and [CMA diagnostics](cma_diagnostics.md)
- [Executed-state decisions](rolling_backtests.md#optional-executed-state-decisions)
- [FactorLasso alpha uncertainty implementation](https://github.com/ArturSepp/FactorLasso/blob/main/src/factorlasso/covariance/_alpha_uncertainty.py)
- [FactorLasso dependent-wild bootstrap](https://github.com/ArturSepp/FactorLasso/blob/main/src/factorlasso/inference/_resampling.py)
- [QIS canonical risk model](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/risk_model.py)
- [QIS ex-post tracking error](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/risk/ex_post_tracking_error.py)
