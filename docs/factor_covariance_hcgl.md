---
myst:
  html_meta:
    description: The TradableFactors owner and the supplied-input handoff to OptimalPortfolios for Factor covariance with HCGL.
---

# Factor covariance with HCGL

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented upstream in [TradableFactors](supplied_risk_inputs.md#tradablefactors-availability),
consumed by [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Upstream citation: `CITATION.cff` in an authorised TradableFactors checkout.

Factor covariance construction, loadings, annualisation, residual policy and completed factor models belong to TradableFactors. OP consumes the selected dated covariance and supplied expected returns.

## Canonical method and runnable workflow

The full methodology and implementation guide has moved to TradableFactors with
its original authorship and source history. TF is private and unreleased; see the
[availability prerequisite](supplied_risk_inputs.md#tradablefactors-availability).
In an authorised TF checkout, `docs/factor_covariance_hcgl.md`
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
<a id="inputs-notation-and-assumptions"></a>
<a id="methodology"></a>
<a id="the-factor-model-and-its-covariance"></a>
<a id="sparse-loadings-hcgl-and-cadence-penalties"></a>
<a id="factors-as-clustering-references"></a>
<a id="residual-covariance-orthogonal-and-empirical"></a>
<a id="the-residual-weight"></a>
<a id="the-point-in-time-contract"></a>
<a id="worked-example"></a>
<a id="construct-factor-prices-and-asset-returns"></a>
<a id="fit-hcgl-at-an-explicit-cutoff"></a>
<a id="a-completed-model-at-an-explicit-cutoff"></a>
<a id="residual-policy-belongs-to-the-completed-object"></a>
<a id="supplied-factor-covariance-has-availability-metadata"></a>
<a id="rolling-fits-are-point-in-time"></a>
<a id="rolling-fits-retain-the-same-completed-object-contract"></a>
<a id="implementation-in-optimalportfolios"></a>
<a id="interpretation-and-limitations"></a>
<a id="see-also"></a>
<a id="references"></a>

## Historical exhibit

This synthetic exhibit remains with its existing reviewed publication receipt and unchanged
OP producer. The current method and numerical example are maintained in TradableFactors.
The image compares systematic and residual variance, including the effect of retaining
residual correlation in the synthetic private-asset sleeve.

![The fixed synthetic example's systematic and residual asset variances and the residual variance of an equal-weight private-asset portfolio at different residual-correlation retention weights.](images/factor_covariance_variance_split.png)
