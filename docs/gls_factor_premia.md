---
myst:
  html_meta:
    description: The FactorModels owner and the supplied-input handoff to OptimalPortfolios for GLS inference of factor risk premia.
---

# GLS inference of factor risk premia

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-16](https://github.com/ArturSepp/OptimalPortfolios/commit/ccb486a6a1ab1d3949d2b6408ee85dcf9dc687a8)*

Documented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Implemented upstream in [FactorModels](supplied_risk_inputs.md#factormodels-availability),
consumed by [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Upstream citation: `CITATION.cff` in an authorised FactorModels checkout.

GLS projection, premium inference, factor-mimicking portfolios and conditional uncertainty belong to FactorModels. OP consumes the prepared expected returns and covariance for constrained optimisation.

## Canonical method and runnable workflow

The full methodology and implementation guide has moved to FactorModels with
its original authorship and source history. FM is private and unreleased; see the
[availability prerequisite](supplied_risk_inputs.md#factormodels-availability).
In an authorised FM checkout, `docs/gls_factor_premia.md`
contains the estimator, units, date cutoffs, diagnostics and runnable examples.
This local page records the owner and OP handoff, not the full private methodology.
The old OP estimator and inference exports are removed in OP 8; there are no forwarding APIs.

OP is complete once it produces the allocation and its solver outcome. See the
[completed optimisation output](completed_output.md), [supplied risk inputs](supplied_risk_inputs.md)
and [solver outcomes](solver_numerics_and_outcomes.md). The retained
[EWMA estimator](covariance_estimators.md) is an explicit OP preparation utility.

## Earlier links

The former section anchors are retained below so links from existing OP guides
arrive at this ownership handoff. The authorised FM guide contains the corresponding method.

<a id="overview"></a>
<a id="inputs-notation-and-assumptions"></a>
<a id="methodology"></a>
<a id="geometric-resolution-and-residual-visibility"></a>
<a id="explicit-covariance-scenarios-and-provider-contrasts"></a>
<a id="paired-loading-and-alpha-stability"></a>
<a id="worked-example"></a>
<a id="implementation-in-optimalportfolios"></a>
<a id="total-forecasts-and-prepared-audit-targets"></a>
<a id="provider-workflows"></a>
<a id="interpretation-and-limitations"></a>
<a id="see-also"></a>
<a id="references"></a>
