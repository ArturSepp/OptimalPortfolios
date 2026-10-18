---
myst:
  html_meta:
    description: The TradableFactors owner and the supplied-input handoff to OptimalPortfolios for GLS inference of factor risk premia.
---

# GLS inference of factor risk premia

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented upstream in [TradableFactors](https://github.com/ArturSepp/tradablefactors),
consumed by [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Upstream citation: [TradableFactors CITATION.cff](https://github.com/ArturSepp/tradablefactors/blob/main/CITATION.cff).

GLS projection, premium inference, factor-mimicking portfolios and conditional uncertainty belong to TradableFactors. OP consumes the prepared expected returns and covariance for constrained optimisation.

## Canonical method and runnable workflow

The full [methodology and implementation guide](https://github.com/ArturSepp/tradablefactors/blob/main/docs/gls_factor_premia.md)
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
