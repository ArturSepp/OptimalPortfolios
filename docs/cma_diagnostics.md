---
myst:
  html_meta:
    description: The FactorModels owner and the supplied-input handoff to OptimalPortfolios for CMA diagnostics and reports.
---

# CMA diagnostics and reports

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-13](https://github.com/ArturSepp/OptimalPortfolios/commit/00eadd14a0687486efba43ae9f8f575bf491e6ce)*

Documented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Implemented upstream in [FactorModels](supplied_risk_inputs.md#factormodels-availability),
consumed by [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Upstream citation: [CITATION.cff](https://github.com/ArturSepp/FactorModels/blob/main/CITATION.cff).

CMA accounting, identification, sensitivity and method-specific reports belong to FactorModels. General portfolio risk and stress reporting belongs to QIS; OP solves the supplied portfolio problem.

## Canonical method and runnable workflow

The public [FactorModels method guide](https://factormodels.readthedocs.io/en/latest/cma_diagnostics.html)
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

<a id="inputs-and-computation"></a>
<a id="choose-the-input-interface"></a>
<a id="interpretation"></a>
<a id="rendering-and-persistence"></a>
<a id="factor-identification-and-forecast-influence"></a>
<a id="conditional-gls-confidence-intervals"></a>
<a id="individual-and-pair-identification"></a>
<a id="extended-inference-reports"></a>
<a id="persistence-and-replay"></a>
<a id="references"></a>
