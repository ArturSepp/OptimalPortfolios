---
myst:
  html_meta:
    description: The TradableFactors owner and the supplied-input handoff to OptimalPortfolios for CMA diagnostics and reports.
---

# CMA diagnostics and reports

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented upstream in [TradableFactors](https://github.com/ArturSepp/tradablefactors),
consumed by [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Upstream citation: [TradableFactors CITATION.cff](https://github.com/ArturSepp/tradablefactors/blob/main/CITATION.cff).

CMA accounting, identification, sensitivity and method-specific reports belong to TradableFactors. General portfolio risk and stress reporting belongs to QIS; OP solves the supplied portfolio problem.

## Canonical method and runnable workflow

The full [methodology and implementation guide](https://github.com/ArturSepp/tradablefactors/blob/main/docs/cma_diagnostics.md)
has moved to TradableFactors with its original authorship and source history.
Use that guide for the estimator, units, date cutoffs, diagnostics and runnable examples.
The old OP estimator and inference exports are removed in OP 8; there are no forwarding APIs.

OP is complete once it produces the allocation and its solver outcome. See the
[completed optimisation output](completed_output.md), [supplied risk inputs](supplied_risk_inputs.md)
and [solver outcomes](solver_numerics_and_outcomes.md). The retained
[EWMA estimator](covariance_estimators.md) is an explicit OP preparation utility.

## Earlier links

The former section anchors are retained below so links from existing OP guides
arrive at this ownership handoff. The linked TF page contains the corresponding method.

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
