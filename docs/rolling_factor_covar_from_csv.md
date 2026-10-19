---
myst:
  html_meta:
    description: The TradableFactors owner and the supplied-input handoff to OptimalPortfolios for Rolling factor risk model from CSV.
---

# Rolling factor risk model from CSV

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented upstream in [TradableFactors](supplied_risk_inputs.md#tradablefactors-availability),
consumed by [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Upstream citation: `CITATION.cff` in an authorised TradableFactors checkout.

The versioned six-file bundle loader, financial preparation and rolling factor-model fit belong to the installed TradableFactors API. Loading, fitting in memory and explicit persistence are separate operations.

## Canonical method and runnable workflow

The full methodology and implementation guide has moved to TradableFactors with
its original authorship and source history. TF is private and unreleased; see the
[availability prerequisite](supplied_risk_inputs.md#tradablefactors-availability).
In an authorised TF checkout, `docs/rolling_factor_covar_from_csv.md`
contains the estimator, units, date cutoffs, diagnostics and runnable examples.
This local page records the owner and OP handoff, not the full private methodology.
The old OP estimator and inference exports are removed in OP 8; there are no forwarding APIs.

OP is complete once it produces the allocation and its solver outcome. See the
[completed optimisation output](completed_output.md), [supplied risk inputs](supplied_risk_inputs.md)
and [solver outcomes](solver_numerics_and_outcomes.md). The retained
[EWMA estimator](covariance_estimators.md) is an explicit OP preparation utility.

## Earlier links

The former section anchors are retained below so links from existing OP guides
arrive at this ownership handoff. The authorised TF guide contains the corresponding method.

<a id="overview"></a>
<a id="pipeline-at-a-glance"></a>
<a id="inputs-notation-and-assumptions"></a>
<a id="the-six-file-csv-contract"></a>
<a id="yahoo-demonstration-basis"></a>
<a id="methodology"></a>
<a id="currency-conversion-and-observation-timing"></a>
<a id="numerical-reconstruction-check"></a>
<a id="worked-example"></a>
<a id="create-an-offline-six-file-bundle"></a>
<a id="reload-every-input"></a>
<a id="convert-asset-returns"></a>
<a id="fit-the-rolling-decomposition"></a>
<a id="implementation-in-optimalportfolios"></a>
<a id="install-and-run"></a>
<a id="fetch-and-persist"></a>
<a id="persistence-boundary"></a>
<a id="interpretation-and-limitations"></a>
<a id="replace-yahoo-with-delivered-matf-data"></a>
<a id="validation-boundary"></a>
<a id="see-also"></a>
<a id="references"></a>
