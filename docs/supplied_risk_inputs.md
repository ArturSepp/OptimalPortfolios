---
myst:
  html_meta:
    description: The completed optimisation result and explicit upstream risk inputs in OP 8.
---

# Supplied risk inputs and completed optimisation results

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

OP accepts dated covariance matrices, caller-supplied alphas and constraints. Its
completed outputs are target weights, solver status and feasibility diagnostics.
Holdings simulation and general risk/stress reporting are separate QIS operations.
The retained EWMA estimator and signal formulas remain available as explicit utilities.
Optimisers do not fit loadings or infer alphas.

The [completed-output guide](completed_output.md) provides the canonical base-install
example, the actual result and separate solver-outcome fields, and explicit table export.

## Result context

`PortfolioOptimisationResult(weights=..., benchmark_weights=...)` is usable for
weight inspection without a fitted factor object. Add either `covariance=...` or
`risk_model=...` to enable risk calculations. A RiskModel date is an exact requested
date, or the latest available date at or before `optimisation_date` when omitted.
Asset labels are checked and aligned. Covariance units are supplied unchanged.
Risk methods require risk context; missing data is an error, not a zero-risk book.

`asset_diagnostics` and `asset_volatilities` accept prepared labelled tables.
They are optional and do not turn the result into a statistical estimator.
The result owns copies of its selected numerical context. In-memory pandas objects
remain inspectable; there is no new universal serialisation schema.

## Factor-model composition

Install TradableFactors separately when constructing factor inputs. A completed TF
snapshot supplies `tf.prepare_portfolio_context(snapshot)` as keyword arguments to
`PortfolioOptimisationResult`, alongside independently obtained weights and benchmark.
The adapter does not import OP. Full selected covariance remains authoritative for
portfolio variance and tracking error. For correlated residuals, a requested
`diagonal_reference=True` view labels approximate component rows with `reference`;
those components need not sum to total risk.

`tradablefactors.analytics.FactorOpportunityModel.optimisation_inputs()` returns
labelled arrays for `optimalportfolios.matf.OpportunityModel`. Constrained solving
stays in OP. GLS decomposition, uncertainty estimation and universe diagnostics
stay in TF/FL; any projected alpha covariance is prepared before optimisation.
Cluster mappings and optional pooled signal panels are likewise prepared upstream.

## Breaking migration

OP 8 removes the FactorLasso dependency, root aliases, `risk_labelling` shim,
factor-estimator modules and statistical MATF modules. There is no factor extra or
legacy `covar_data` result constructor. Import statistical objects from their owning
package and supply completed inputs. The [TF migration guide](https://github.com/ArturSepp/tradablefactors/blob/main/docs/migration.md)
lists canonical owners. Existing research snapshots retain their original environment.

## Verification

The base CI lane imports and tests an installed OP wheel with neither TF nor FL
installed. A separate composed lane installs both candidate OP and TF wheels and
runs the shipped `optimalportfolios.tests.factor_integration_test` suite without
skips. Repository examples are classified from their imports into base offline,
composed offline and network lanes. The [EWMA quickstart](quickstart.md) remains a
base-install example.
