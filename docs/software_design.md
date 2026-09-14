---
myst:
  html_meta:
    description: >-
      OptimalPortfolios architecture: dated estimates, rolling construction, solver
      backends, constraint handling, and delegation to FactorLasso and QIS.
---

# Software design and boundaries

*[author / affiliation / date — placeholder]*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

OptimalPortfolios coordinates portfolio construction from dated estimates and constraints.
Its rolling functions produce target weights; [QIS](https://github.com/ArturSepp/QuantInvestStrats)
simulates the resulting holdings and transaction costs. The distinction between a construction
decision and its execution is central to the design.

This guide maps those responsibilities to the current source. The [quickstart](quickstart.md)
provides a runnable offline workflow; the [optimization guide](optimization_module_readme.md)
documents the detailed solver interfaces and limitations.

## Composition across the open-source stack

| Responsibility | Owning layer | Integration in OptimalPortfolios |
|---|---|---|
| Sparse factor fitting, clustering and factor-covariance containers | [FactorLasso](https://github.com/ArturSepp/FactorLasso) | [Factor estimator](../src/optimalportfolios/covar_estimation/factor_covar_estimator.py): prepare financial inputs, estimation dates, return frequencies and annualization. |
| Return transformations and EWMA covariance primitives | QIS | [EWMA estimator](../src/optimalportfolios/covar_estimation/ewma_covar_estimator.py): select the sample and schedule, then expose dated covariance matrices. |
| Objectives, constraints, asset eligibility and construction state | OptimalPortfolios | [Rolling dispatcher](../src/optimalportfolios/optimization/wrapper_rolling_portfolios.py) and [constraints facade](../src/optimalportfolios/optimization/constraints/__init__.py). |
| Unit-based simulation, cash costs, performance analytics and factsheets | QIS | Pass dated targets, prices, costs and implementation lag to `qis.backtest_model_portfolio`. |
| Canonical ex-ante risk and tracking-error analytics | `qis.RiskModel` | [Risk-model adapter](../src/optimalportfolios/covar_estimation/risk_model_adapter.py) and [result container](../src/optimalportfolios/optimization/portfolio_result.py). |

These boundaries keep shared calculations in their owning package. FactorLasso owns generic
factor estimation, QIS owns the reusable analytics and holdings simulator, and OptimalPortfolios
owns portfolio construction and its financial input preparation. Some historical utility
functions remain in this repository; their presence does not establish a separate canonical
analytics layer.

The ordinary workflow is:

```text
prices and dated inputs
          |
          v
estimation: OptimalPortfolios + FactorLasso/QIS
          |
          v
rolling construction <--- previous target + price drift
          |          <--- constraints and dealing inputs
          v
dated target weights
          |
          v
QIS execution: units, implementation lag and cash costs
          |
          v
PortfolioData and reports
```

The rolling construction state uses prior targets and supplied prices. The convenience
backtest function computes the targets first and then invokes QIS; executed QIS holdings are
not automatically fed back into each optimizer decision. Use explicit current holdings when
calling a single-date wrapper for a live decision.

## Architectural boundaries

### Point-in-time estimation

The [`CovarEstimator` interface](../src/optimalportfolios/covar_estimation/covar_estimator.py)
separates estimation from construction. EWMA and factor estimators expose
`fit_rolling_covars(...)` as a dictionary from decision dates to labeled, annualized covariance
matrices. Their inputs differ: EWMA accepts an asset-price panel; the factor estimator also
needs factor prices and asset returns organized by estimation frequency.

That common output permits covariance-based objectives to reuse estimates. It does not make
every dispatcher branch an identical experiment:

| Dispatcher objective | Estimation performed by the dispatch path |
|---|---|
| `MIN_VARIANCE`, `MAX_DIVERSIFICATION`, `EQUAL_RISK_CONTRIBUTION` | Consume the supplied dated covariance matrices. |
| `QUADRATIC_UTILITY`, `MAXIMUM_SHARPE_RATIO` | Consume those matrices and estimate EWMA expected returns from prices, using the selected return frequency and span. |
| `MAX_CARA_MIXTURE` | Fits rolling mixture distributions from prices using its own window and schedule. This branch does not use the supplied covariance dictionary. |

The dispatcher covers the six members of
[`PortfolioObjective`](../src/optimalportfolios/config.py). Minimum tracking error,
alpha-over-tracking-error, target-return allocation and hierarchical risk parity have separate
public entry points. The [optimization guide](optimization_module_readme.md) maps those routes.

A date key identifies the intended information date; it does not prove that the supplied data
were available then. Supply covariance keys in chronological order, with consistent asset labels,
and account for publication delays, revisions and stale observations. Current-estimate methods
operate on the data supplied to them; restrict their input for a historical decision.
See [covariance estimation](covariance_estimators.md), [mixed-frequency data](mixed_frequency_data.md)
and the [alpha guide's timing qualifications](alphas_module_readme.md).

### Holdings as state

With `OptimiserConfig.use_drifted_weights_0=True`, rolling solvers drift the previous target
to the next decision date using price changes before passing it as `weights_0`.
The [drift helper](../src/optimalportfolios/utils/portfolio_funcs.py) retains the previous weights
when its prerequisites are unavailable. Setting the option to `False` reuses the prior target.

This reference state supports turnover constraints, penalties and warm starts where the selected
solver implements them. It is a decision-date approximation to current allocation, not a complete
copy of the simulated execution ledger. An implementation lag can move trades to a later price
observation, and costs can further change the simulated holdings.

[`backtest_rolling_optimal_portfolio`](../src/optimalportfolios/optimization/wrapper_rolling_portfolios.py)
calls the dispatcher, optionally restricts the target rows with `perf_time_period`, truncates
prices to the retained start, and passes targets and execution arguments to QIS. Its return
value is `qis.PortfolioData`. Set costs and lag explicitly for a reproducible experiment;
a lag counts price observations, not an assumed number of business days.

The [rolling-backtest guide](rolling_backtests.md) and
[turnover and transaction-cost guide](turnover_and_transaction_costs.md) distinguish target
changes, drift-adjusted decision turnover, actual traded volume and cash charges.

### Mixed-frequency scheduling

Estimation cadence, portfolio decision dates and asset dealing availability are distinct inputs.
An asset can use monthly return observations while the wider portfolio is evaluated more often.
Eligible-universe and rebalancing indicators help selected wrappers exclude or freeze positions.

Support is specific to the entry point: the convenience dispatcher does not expose every
eligibility or dealing argument accepted by individual rolling functions. Pass the required
panels to the appropriate public solver rather than assuming that a schedule applies uniformly
to all objectives. Forward filling and missing-price behavior also require explicit assumptions.

See [mixed-frequency data](mixed_frequency_data.md), [incomplete histories](incomplete_histories.md)
and the solver-by-solver [constraints contract](constraints.md).

### Shared constraints

`Constraints` carries common specifications for weights, exposure, groups, tracking error,
turnover, return targets and other supported restrictions. The constraint package separates
data models, alignment, backend translation and analytical residual checks; its
[`__init__.py` facade](../src/optimalportfolios/optimization/constraints/__init__.py)
preserves the public import path.

Most solver families separate three responsibilities:

1. A rolling function selects dated inputs and maintains its prior-weight state.
2. A labeled wrapper filters or aligns the active universe and adjusts the constraint problem.
3. A numerical solver constructs the objective, applies the supported restrictions and
   validates the returned candidate.

A shared specification does not imply identical enforcement by every backend. Some fields are
restricted to particular CVXPY paths, and some configuration flags are currently consumed only by
the alpha-over-tracking-error wrapper. Consult the enforcement matrix before switching objectives.

Likewise, returned weights alone do not prove that a solve was accepted. Low-level functions can
return structured outcomes, wrappers can return weights with diagnostics, and rolling functions
commonly expose only the weight panel. Check the documented return contract and fallback behavior
in the [optimization guide](optimization_module_readme.md).

## Optimization backends

The implementation uses several numerical routes:

| Route | Current use | Source |
|---|---|---|
| CVXPY with a selected solver | Quadratic and conic formulations, including minimum variance, minimum tracking error and benchmark-relative allocation. | [Quadratic solver](../src/optimalportfolios/optimization/general/quadratic.py), [minimum tracking error](../src/optimalportfolios/optimization/general/minimum_tracking_error.py) |
| CVXPY or SciPy SLSQP | Maximum Sharpe uses the Charnes–Cooper route when exposure is fixed; unequal minimum and maximum exposure select SLSQP. | [Maximum-Sharpe implementation](../src/optimalportfolios/optimization/general/max_sharpe.py) |
| SciPy SLSQP | Maximum diversification and CARA-mixture utility. | [Diversification](../src/optimalportfolios/optimization/general/max_diversification.py), [mixture utility](../src/optimalportfolios/optimization/general/carra_mixture.py) |
| Internal CCD/ADMM with quadprog projection | Constrained risk budgeting; quadprog solves the projection subproblem. | [Risk-budgeting solver](../src/optimalportfolios/optimization/risk_allocation/risk_budgeting_solver.py) |
| Linkage-based recursive allocation | Hierarchical risk parity accepts a supplied linkage and allocates recursively. | [HRP implementation](../src/optimalportfolios/optimization/risk_allocation/hierarchical_risk_parity.py) |

[`OptimiserConfig`](../src/optimalportfolios/optimization/config.py) selects the CVXPY solver
(default `CLARABEL`) and shared controls. Its `solver` field does not replace the fixed SciPy
or risk-budgeting algorithms. Parameter names and supported constraints must therefore be checked
at the chosen entry point. This architecture description makes no comparative runtime claim.

## Data and result contracts

| Boundary | Representation | Reader check |
|---|---|---|
| Price input | DataFrame: observation dates in rows, assets in columns. | State price/return convention, frequency and information cutoff. |
| Rolling covariance | Dictionary keyed by decision date; each value is an asset-by-asset DataFrame. | Keep label alignment, chronological keys and annual covariance units explicit. |
| Rolling target weights | DataFrame: decision dates in rows, assets in columns. | Weights are dimensionless; read the selected solver's diagnostics and fallback contract. |
| QIS execution result | `qis.PortfolioData` from the backtest wrapper. | Inspect the actual trading calendar, costs and lag. |
| Construction/risk snapshot | `PortfolioOptimisationResult`: assets in rows and portfolios in columns for a multi-portfolio weight input. | This orientation differs from a rolling weight panel; provide its factor-model and benchmark context. |

`PortfolioOptimisationResult` combines construction output and risk context; it does not perform
portfolio optimization. Its risk calculations delegate to `qis.RiskModel`.
The public `build_risk_model(covar_data)` adapter accepts rolling factor data, dated factor
snapshots or a dated dictionary of asset covariance matrices. A covariance-only input creates
a covariance-only risk model; it does not reconstruct missing factor loadings.

For a complete executable example, follow the [canonical quickstart](quickstart.md) or
[example catalogue](examples_readme.md). The [CSV workflow](rolling_factor_covar_from_csv.md)
shows how factor inputs and risk-model context cross the package boundary.

## Core installation and integrations

The numerical pipeline depends on QIS and FactorLasso as core stack packages.
The [packaging metadata](../pyproject.toml) declares the remaining numerical dependencies
and three published extras:

| Extra | Role |
|---|---|
| `data` | Yahoo Finance integration through yfinance. |
| `reports` | Optional pybloqs rendering. QIS/matplotlib analytics and factsheets are already used by the core workflows. |
| `docs` | Sphinx, theme and Markdown tooling for building documentation. |

The `all` contributor dependency group is separate from a published extra.
The [installation guide](installation.md) describes released installs, contributor groups,
the external Windows environment and the pending lockfile reconciliation.

Core imports keep optional integrations out of their import path. A dedicated optional module
or a repository example can still require its declared integration: inspect the relevant
prerequisites before importing it. Repository examples and component `run_local` diagnostics
are excluded from built distributions. Package tests and their frozen fixture are shipped for
the supported installed-package checks.

## Why this is a separate package

The package's organizing abstraction is a dated construction workflow: estimates and input
availability, prior allocation, eligibility/dealing restrictions, target generation, execution
assumptions and QIS evaluation. A standalone objective function addresses one step in that
workflow. Keeping construction here makes its financial conventions and dependency boundaries
visible together.

Other portfolio libraries organize related capabilities differently.
The [package comparison](package_comparison.md) records a **21 August 2026** version snapshot
and describes those differences. Treat it as a dated comparison, not a current release inventory
or a universal ranking. Shared generic capabilities continue to belong in their owning packages.

## Intentional exclusions

Portfolio construction does not include broker connectivity, order management, execution routing,
market-data licensing or proprietary production orchestration. An implementation lag in a research
backtest is an execution assumption, not a broker execution model.

The public dispatcher is also not a catalogue of every risk objective or every model-selection
procedure. Its current objective set is explicit; non-quadratic risk measures and generic
cross-validation/search frameworks are outside that interface. Consult the dated comparison when
evaluating a different workflow.

These are scope boundaries, not guarantees that every existing helper has already been moved to
its canonical owner. Method-specific qualifications and known timing limits remain in the linked
guides.

## See also

- [Quickstart](quickstart.md) and [examples](examples_readme.md)
- [Optimization guide](optimization_module_readme.md) and [constraints](constraints.md)
- [Covariance estimators](covariance_estimators.md) and [backtest timing](rolling_backtests.md)
- [Documentation standard](documentation_standard.md)

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff)
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff)
