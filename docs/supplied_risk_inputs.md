---
myst:
  html_meta:
    description: The completed optimisation result and explicit upstream risk inputs in OP 8.
---

# Supplied risk inputs and completed optimisation results

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-18](https://github.com/ArturSepp/OptimalPortfolios/commit/42c248224d9acc31d03ff45ca1d7e96a3e88e118)*

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

<a id="tradablefactors-availability"></a>

## FactorModels availability

The [FactorModels distribution](https://pypi.org/project/factormodels/) is published
separately as `factormodels`. The current development examples and integration tests
require the **unreleased FM 0.3.0 candidate**, installed from its reviewed wheel;
an older indexed release does not provide their completed-result contract. Its source repository
and method guides retain their existing access requirements. The base OP installation
and EWMA quickstart do not require FM or FactorLasso.

FM consumes price or level histories with explicit return transformations. It owns
factor definitions, NAV construction and completed fitted loadings and
covariance for a specified asset universe. Its finite output is a completed model
snapshot with an explicit cutoff, labelled assets and factors, return transforms,
annualisation, residual policy and provenance. Prepared expected returns and
diagnostic tables are additional upstream inputs where a workflow needs them.
OP consumes the selected dated covariance, supplied alphas, constraints and optional
QIS RiskModel context; it does not acquire factor data, fit loadings or infer premia.
The OP boundary ends at target weights, solver status and feasibility diagnostics.

The local [HCGL](factor_covariance_hcgl.md), [GLS](gls_factor_premia.md),
[CMA](cma_diagnostics.md) and [CSV bundle](rolling_factor_covar_from_csv.md) pages
identify owners and handoff contracts. They are not copies of the private method
guides. Paths labelled as FM sources or documentation below refer to an authorised
FM checkout, not to publicly accessible web pages. That checkout retains the
upstream authorship, source history and software citation in `CITATION.cff`.

## Factor-model composition

Use the separately installed FM package when constructing factor inputs. A completed FM
snapshot supplies `fm.prepare_portfolio_context(snapshot)` as keyword arguments to
`PortfolioOptimisationResult`, alongside independently obtained weights and benchmark.
The adapter does not import OP. Full selected covariance remains authoritative for
portfolio variance and tracking error. For correlated residuals, a requested
`diagonal_reference=True` view labels approximate component rows with `reference`;
those components need not sum to total risk.

`factormodels.analytics.FactorOpportunityModel.optimisation_inputs()` returns
labelled arrays for `optimalportfolios.optimization.opportunity.OpportunityModel`. Constrained solving
stays in OP. GLS decomposition, uncertainty estimation and universe diagnostics
stay in FM/FL; any projected alpha covariance is prepared before optimisation.
Cluster mappings and optional pooled signal panels are likewise prepared upstream.

## Breaking migration

OP 8 removes the FactorLasso dependency, root aliases, `risk_labelling` shim,
factor-estimator modules and statistical MATF modules. There is no factor extra or
legacy `covar_data` result constructor. Import statistical objects from their owning
package and supply completed inputs. The FM migration guide at `docs/migration.md`
in an authorised FM checkout lists canonical owners. Existing research snapshots
retain their original environment.

The development version moves `OpportunityModel`, `OpportunitySolution`,
`solve_opportunity` and `portfolio_opportunity_metrics` from
`optimalportfolios.matf.opportunity` to `optimalportfolios.optimization.opportunity`.
Update imports directly; the `matf` package is removed without forwarding modules.
Inputs, solver defaults, tolerances and numerical results are unchanged. Historical
environments retain the original import path for their recorded runs.

## Verification

The base CI lane imports and tests an installed OP wheel with neither FM nor FL
installed. A separate composed lane installs both candidate OP and FM wheels and
runs the shipped `optimalportfolios.tests.factor_integration_test` suite without
skips. Repository examples are classified from their imports into base offline,
composed offline and network lanes. The [EWMA quickstart](quickstart.md) remains a
base-install example.
