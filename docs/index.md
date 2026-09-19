---
myst:
  html_meta:
    description: >-
      OptimalPortfolios documentation: start with an offline portfolio backtest,
      then explore signals, covariance estimation, constraints, and risk reporting.
---

<a id="optimalportfolios-production-portfolio-construction-and-rolling-backtesting"></a>

# optimalportfolios

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-09](https://github.com/ArturSepp/OptimalPortfolios/commit/254505981ed43e0dbc12a19c98c034d351b8d059)*

Documentation for [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

OptimalPortfolios is a Python library for multi-asset portfolio construction and rolling
backtesting. It turns dated estimates, investment objectives and constraints into target
weights, then delegates holdings simulation, transaction costs and reporting to
[QIS](https://github.com/ArturSepp/QuantInvestStrats).

The package implements the ROSAA framework described by Sepp, Ossa and Kastenholz (2026);
the [papers below](#papers) provide its research context.

## Start here

| Your task | Begin with |
|---|---|
| Install the package and choose optional features | [Installation](installation.md) |
| Estimate covariance, compute constrained weights and backtest them | [Offline quickstart](quickstart.md) |
| Find a complete workflow and its data prerequisites | [Examples guide](examples_readme.md) |
| Inspect reproducible synthetic portfolio and risk exhibits | [Analytics gallery](analytics_gallery.md) |

The quickstart computes a result from a fixed monthly fixture shipped in the wheel. After
installation, the calculation runs offline. The guide explains how to run a saved script
or use the repository example; the `examples/` tree itself is not installed with the wheel.
Its Colab option needs a network connection for setup.

## Overview

A typical workflow connects four steps:

1. **Estimate signals.** The [alpha guide](alphas_module_readme.md) covers momentum, carry,
   low-beta, residual momentum and reversal, including cross-sectional and within-cluster
   scoring.
2. **Estimate risk.** The [covariance guide](covariance_estimators.md) explains EWMA and
   factor covariance. Sparse factor fitting and hierarchical clustering group lasso (HCGL)
   are supplied by [FactorLasso](https://github.com/ArturSepp/FactorLasso).
3. **Construct target weights.** The [optimization guide](optimization_module_readme.md)
   maps objectives to their solvers. The shared [constraints contract](constraints.md)
   describes the supported conditions and their backend-specific limits. Implementations
   use CVXPY, SciPy or dedicated risk-budgeting routines, depending on the objective.
4. **Simulate and report.** [Rolling backtests](rolling_backtests.md) connect dated targets
   to QIS holdings, price drift, implementation lag and transaction costs. Target weights
   and executed holdings are distinct states.

The [software design guide](software_design.md) explains these package boundaries. Cite
the [QIS software record](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
for its analytics and simulation, and the
[FactorLasso software record](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff)
for its factor estimation.

## Signals and risk estimates

- [Alpha signals](alphas_module_readme.md): definitions, scoring and estimation timing.
- [Covariance estimators](covariance_estimators.md): EWMA and factor-model inputs and outputs.
- [Mixed-frequency data](mixed_frequency_data.md): align assets observed at different cadences.
- [Incomplete histories](incomplete_histories.md): handle eligibility, missing prices and warmup.
- [Rolling factor covariance from CSV](rolling_factor_covar_from_csv.md): load dated estimates
  and connect them to the QIS risk model.

## Construction and constraints

- [Optimization guide](optimization_module_readme.md): objectives, dispatch, configuration,
  return types and solver outcomes.
- [Constraints](constraints.md): units, alignment, feasibility and backend support.
- [Minimum tracking error](minimum_tracking_error.md): construct allocations relative to a
  supplied benchmark.
- [Risk budgeting](risk_budgeting.md): allocate risk using specified contribution budgets.
- [Overlay tail floor](overlay_tail_floor.md): combine fixed exposure with an optimized sleeve
  under a downside floor.

## Backtests and applied examples

- [Analytics gallery](analytics_gallery.md): six synthetic exhibits with sample dates, conventions,
  producer source links and reviewed provenance.

- [Rolling backtests](rolling_backtests.md): estimation dates, target weights and execution.
- [Turnover and transaction costs](turnover_and_transaction_costs.md): distinguish construction
  penalties from realized trades and cash costs.
- [Stress testing with options](stress_testing_with_options.md): assess a supplied stock-and-option
  portfolio using factor scenarios and option repricing. This example requires network data and
  the local prerequisites listed in its guide; it does not optimize the supplied positions.

## Reference and project guidance

- [Software design](software_design.md): component responsibilities and integration boundaries.
- [Package comparison](package_comparison.md): a dated comparison of portfolio-library capabilities.
- [Documentation standard](documentation_standard.md): article structure, notation, citations and
  reproducible analytics.
- [API reference](api.rst): the public import surface, signatures and docstrings.

## Papers

- Sepp, A. (2023). *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*.
  [Risk Magazine, 6 October 2023](https://www.risk.net/cutting-edge/7957914/optimal-allocation-to-cryptocurrencies-in-diversified-portfolios).
  [SSRN 4217841](https://ssrn.com/abstract=4217841).
- Sepp, A., Ossa, I. and Kastenholz, M. (2026). *Robust Optimization of Strategic and Tactical
  Asset Allocation for Multi-Asset Portfolios*.
  [The Journal of Portfolio Management, 52(4), 86–120](https://www.pm-research.com/content/iijpormgmt/52/4/86).
  An [author-shared publisher copy](https://eprints.pm-research.com/17511/143431/index.html)
  is also available.
- Sepp, A., Hansen, E. H. and Kastenholz, M. (2026). *Capital Market Assumptions and Strategic
  Asset Allocation Using Multi-Asset Tradable Factors*. Working paper, 17 May 2026.
  [SSRN 6785958](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958).

The [author's research catalogue](https://artursepp.com/research/) provides the related
bibliographic records. Cite a paper for its method and the software records above for the
packages used in an implementation.

## Project links

- [PyPI](https://pypi.org/project/optimalportfolios/) and
  [rendered documentation](https://optimalportfolios.readthedocs.io/en/latest/).
- [Source repository](https://github.com/ArturSepp/OptimalPortfolios) and
  [issue tracker](https://github.com/ArturSepp/OptimalPortfolios/issues).
- [Governance, maintenance and support](https://github.com/ArturSepp/OptimalPortfolios/blob/main/GOVERNANCE.md).
- [Changelog](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CHANGELOG.md).

<!-- The sidebar mirrors the ordinary article links above. Keep each document in one tree. -->

```{toctree}
:hidden:
:maxdepth: 2
:caption: Start here

installation
quickstart
examples_readme
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Signals and risk estimates

alphas_module_readme
covariance_estimators
mixed_frequency_data
incomplete_histories
rolling_factor_covar_from_csv
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Construction and constraints

optimization_module_readme
constraints
minimum_tracking_error
risk_budgeting
overlay_tail_floor
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Backtests and applied examples

analytics_gallery
rolling_backtests
turnover_and_transaction_costs
stress_testing_with_options
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference and project

software_design
package_comparison
documentation_standard
api
```
