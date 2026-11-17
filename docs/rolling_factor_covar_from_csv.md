---
myst:
  html_meta:
    description: The FactorModels owner and the supplied-input handoff to OptimalPortfolios for Rolling factor risk model from CSV.
---

# Rolling factor risk model from CSV

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-06](https://github.com/ArturSepp/OptimalPortfolios/commit/d8628f883d87bcb7e7006319bab8a804d899d949)*

Documented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Implemented upstream in [FactorModels](supplied_risk_inputs.md#factormodels-availability),
consumed by [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Upstream citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

Load factor histories with pandas and pass prepared DataFrames to the installed
FactorModels estimator. The current API has no six-file bundle loader. FM's runnable
CSV example prepares input returns, fits in memory and exports results only when requested.

## Canonical method and runnable workflow

The public [FactorModels method guide](https://factormodels.readthedocs.io/en/latest/rolling_factor_covar_from_csv.html)
explains the inputs, units, date cutoffs, diagnostics and runnable examples.
See the [installation and handoff](supplied_risk_inputs.md#factormodels-availability)
for composing FM with OP. This page retains the OP handoff and earlier section links.

OP is complete once it produces the allocation and its solver outcome. See the
[completed optimisation output](completed_output.md), [supplied risk inputs](supplied_risk_inputs.md)
and [solver outcomes](solver_numerics_and_outcomes.md). The retained
[EWMA estimator](covariance_estimators.md) is an explicit OP preparation utility.

## Earlier links

The former section anchors are retained below so links from existing OP guides
arrive at this ownership handoff. The linked FM guide contains the corresponding method.

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
