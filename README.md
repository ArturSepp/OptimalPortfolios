---
myst:
  html_meta:
    description: >-
      Multi-asset portfolio construction and rolling backtesting in Python, with
      offline examples, constrained optimization, covariance models and analytics.
---

# optimalportfolios

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2023-07-08](https://github.com/ArturSepp/OptimalPortfolios/commit/6950e6d9310f70280891ddc4094be78c6f178e24)*

Source: [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).

**Production multi-asset portfolio construction and rolling backtesting in Python — constrained
optimisation from supplied alphas, covariances and constraints to dated target weights and solver
outcomes.**

The [completed-output guide](docs/completed_output.md) identifies the result, its units,
dates, status and explicit export boundary. Factor-model construction belongs to
[FactorModels](docs/supplied_risk_inputs.md#factormodels-availability); sparse estimation to
[FactorLasso](https://github.com/ArturSepp/FactorLasso); general risk, stress testing and
reporting to QIS. OP retains explicit EWMA and signal utilities and rolling orchestration.

**Install:** `pip install optimalportfolios` · **Import:** `optimalportfolios` · **Status:** Stable

[![PyPI](https://img.shields.io/pypi/v/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/)
[![Python](https://img.shields.io/pypi/pyversions/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/)
[![CI](https://github.com/ArturSepp/OptimalPortfolios/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/OptimalPortfolios/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/optimalportfolios/badge/?version=latest)](https://optimalportfolios.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ArturSepp/OptimalPortfolios.svg?style=flat-square)](https://github.com/ArturSepp/OptimalPortfolios/blob/main/LICENSE.txt)
[![Downloads](https://static.pepy.tech/badge/optimalportfolios)](https://pepy.tech/project/optimalportfolios)
[![Monthly](https://static.pepy.tech/badge/optimalportfolios/month)](https://pepy.tech/project/optimalportfolios)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/OptimalPortfolios/blob/main/examples/getting_started/production_quickstart.ipynb)

**Papers:** Sepp, A. (2023), *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*,
Risk Magazine · Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of Strategic
and Tactical Asset Allocation for Multi-Asset Portfolios*, The Journal of Portfolio Management
52(4) · Sepp, A., Hansen, E. and Kastenholz, M. (2026), *Capital Market Assumptions and Strategic
Asset Allocation Using Multi-Asset Tradable Factors* · Sepp, A. and Kastenholz, M. (2026), *The
Convexity Premium of Portfolio Overlays*, Journal of Investment Management. See
[Statement of need](#statement-of-need) for what each paper contributes and [Citation](#citation)
for the BibTeX entries.

**Documentation:** [optimalportfolios.readthedocs.io](https://optimalportfolios.readthedocs.io/en/latest/) ·
[analytics gallery](https://optimalportfolios.readthedocs.io/en/latest/analytics_gallery.html) ·
[production quickstart](examples/getting_started/production_quickstart.py)

---

## Why optimalportfolios

PyPortfolioOpt, Riskfolio-Lib, and skfolio all provide substantial portfolio
optimisation capabilities. Their documented design centres emphasise, respectively,
compact classical allocation, breadth across risk measures and portfolio families,
and scikit-learn-compatible model selection. `optimalportfolios` is organised around
a different primary abstraction: a dated state transition from estimates and current
holdings to constrained targets and realised backtests.

**The completed OP object is an allocation and its audit:** supplied alphas and
covariance → optimise under constraints → inspect target weights, solver status and
feasibility. A separate QIS operation simulates execution and produces reports.
Rolling orchestration supports incomplete data, mixed-frequency assets and illiquid positions.

### Key differentiators

- **Production multi-asset pipeline.** Factor-model covariance, risk-budgeted strategic
  allocation, alpha signals, tracking-error-constrained tactical allocation and a rolling
  backtest share one dated roll-forward state, with equities rebalancing monthly and
  alternatives quarterly. See the [ROSAA case study](https://optimalportfolios.readthedocs.io/en/latest/app_rosaa_multi_asset_allocation.html).
- **Supplied risk inputs.** A dated covariance can come from any upstream model.
  FactorModels supplies a complete factor model using FactorLasso; OP consumes the selected
  covariance. The [HCGL ownership handoff](docs/factor_covariance_hcgl.md) explains availability.
- **Cluster-aware risk allocation.** Clusters, sectors or asset classes become asset-level risk
  budgets, static or dated, and canonical HRP runs from a supplied linkage. See
  [hierarchical risk parity and cluster risk budgets](https://optimalportfolios.readthedocs.io/en/latest/hierarchical_risk_parity_and_cluster_budgets.html).
- **Prepared expected returns.** Forecasts and alphas enter the selected objective directly.
  FM GLS inference and CMA diagnostics are separate upstream workflows, complete before
  optimisation; see their [GLS](docs/gls_factor_premia.md) and
  [CMA](docs/cma_diagnostics.md) ownership handoffs.
- **Drift-aware rolling backtests.** Turnover limits and cost penalties act on the drifted
  holdings rather than the previous target (`OptimiserConfig.use_drifted_weights_0`, default
  `True`). See [rolling backtests](https://optimalportfolios.readthedocs.io/en/latest/rolling_backtests.html).
- **Incomplete and illiquid histories.** Assets enter when their history suffices, missing
  prices get zero weight, and rebalancing indicators freeze illiquid positions. See
  [incomplete histories and frozen positions](https://optimalportfolios.readthedocs.io/en/latest/incomplete_histories.html).
- **Research-backed.** The reference implementation of the ROSAA framework in *The Journal of
  Portfolio Management*; every number on the methodology pages is asserted by a script that CI
  runs. See [research papers](https://optimalportfolios.readthedocs.io/en/latest/research_papers.html).

### When to use it — and when not

Use `optimalportfolios` when you need a dated roll-forward pipeline from point-in-time covariance
and alpha estimates to constrained targets, scheduled rebalancing, and drift-aware backtests with
transaction costs across incomplete or mixed-frequency multi-asset panels.

The solvers use quadratic and conic objectives (variance, tracking error, Sharpe ratio,
diversification ratio, CARA utility). Choose another package when the core problem is a
non-quadratic risk measure such as CVaR, MAD, or drawdown constraints; the
[package comparison](https://optimalportfolios.readthedocs.io/en/latest/package_comparison.html)
points to Riskfolio-Lib and skfolio for those workflows and records the versioned evidence for the
field comparison. Use `factorlasso` directly when you need its standalone sparse multi-output
factor-model estimator rather than portfolio construction. Each solver lives in its own module and
plugs into the rolling backtester through one dispatch function; the
[software-design guide](https://optimalportfolios.readthedocs.io/en/latest/software_design.html)
explains these boundaries.

## Installation

Install from PyPI:

```bash
pip install optimalportfolios
```

Upgrade with:

```bash
pip install --upgrade optimalportfolios
```

Clone the repository with:

```bash
git clone https://github.com/ArturSepp/OptimalPortfolios.git
```

The core package supports Python >=3.11; `pyproject.toml` is the source of truth for dependency
floors, and the [installation guide](https://optimalportfolios.readthedocs.io/en/latest/installation.html) describes the locked environment.
Optional extras keep network-data and reporting integrations out of the core installation:

| Extra | Adds |
| --- | --- |
| `data` | `yfinance` for free-data example loaders. |
| `reports` | `pybloqs` for HTML/PDF report backends. |
| `riskbudget` | `quadprog` projection backend for risk-budgeting solves (8.0.0 development series). |
| `docs` | Sphinx, Furo and MyST for documentation builds. |

For both runtime integrations, install `optimalportfolios[data,reports]`. In the unreleased
8.0.0 development series, risk-budgeting solves require
`pip install "optimalportfolios[riskbudget]"`. This installs the existing GPL-licensed
`quadprog` projection backend; OP's first-party source remains MIT. Ordinary imports,
supplied-input optimisation with other solvers, and the offline quickstart use the base
installation. See [installation and dependency licences](docs/installation.md). There is no
`jupyter` or `dev` extra; tests and static checks are the PEP 735 `test` and `lint` dependency
groups described under [Testing](#testing).

## Table of contents

1. [Why optimalportfolios](#why-optimalportfolios)
2. [Installation](#installation)
3. [Statement of need](#statement-of-need)
4. [Overview](#overview)
5. [Five-minute quickstart](#five-minute-quickstart)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Changelog](#changelog)
9. [Feedback & contributing](#feedback-contributing)
10. [Acknowledgments](#acknowledgments)
11. [Citation](#citation)
12. [Ecosystem](#ecosystem)
13. [License](#license)
14. [Disclaimer](#disclaimer)

## Statement of need

Multi-asset allocation in production is a dated state transition: point-in-time estimates and
current holdings go in, constrained targets, solver outcomes and a realised backtest come out, and
the next date starts from the drifted result. `optimalportfolios` implements that transition for
panels with incomplete histories, mixed rebalancing frequencies and illiquid positions, with the
covariance and expected returns supplied by upstream models rather than estimated inside the
optimiser. The papers below define the methods; the package is their reference implementation,
and every number on the methodology pages is asserted by a script that CI runs.

| Paper | Implemented analytics | Where |
|---|---|---|
| Sepp, A. (2023), *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*, Risk Magazine, October 2023, 1–6, [SSRN 4217841](https://ssrn.com/abstract=4217841) | The four allocation methods of the paper, run offline on the replication design | [`papers/crypto_allocation_risk_2023`](papers/crypto_allocation_risk_2023/README.md), [cryptocurrency case study](https://optimalportfolios.readthedocs.io/en/latest/app_crypto_allocation.html) |
| Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios*, [The Journal of Portfolio Management, 52(4), 86–120](https://www.pm-research.com/content/iijpormgmt/52/4/86) | The ROSAA framework: factor-model covariance, risk-budgeted strategic allocation, alpha signals, tracking-error-constrained tactical allocation and the rolling backtest on one dated roll-forward state | [`papers/robust_optimisation_jpm_2026`](papers/robust_optimisation_jpm_2026/README.md), [ROSAA case study](https://optimalportfolios.readthedocs.io/en/latest/app_rosaa_multi_asset_allocation.html) |
| Sepp, A., Hansen, E. and Kastenholz, M. (2026), *Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors*, [SSRN 6785958](https://ssrn.com/abstract=6785958) | Strategic allocation from capital market assumptions on a supplied factor-model covariance; the factor model itself is prepared upstream in FactorModels | [CMA strategic allocation](docs/app_cma_strategic_allocation.md), [supplied risk inputs](docs/supplied_risk_inputs.md) |
| Sepp, A. and Kastenholz, M. (2026), *The Convexity Premium of Portfolio Overlays*, Journal of Investment Management, forthcoming | The Section II coverage-floor allocation: fixed-core maximum Sharpe ratio with a named linear row | [`papers/smart_diversification_joim_2026`](papers/smart_diversification_joim_2026/README.md), [overlay page](https://optimalportfolios.readthedocs.io/en/latest/overlay_tail_floor.html), [smart diversification case study](https://optimalportfolios.readthedocs.io/en/latest/app_smart_diversification_overlays.html) |

## Overview

<a id="package-overview"></a>
<a id="group-risk-budgets"></a>
<a id="hierarchical-risk-parity"></a>
<a id="naming-convention"></a>
<a id="available-signals"></a>
<a id="mixed-frequency-support"></a>
<a id="alphasdata-container"></a>
<a id="1-implementation-structure"></a>
<a id="2-example-of-implementation-for-maximum-diversification-solver"></a>
<a id="3-constraints"></a>
<a id="4-wrapper-for-implemented-rolling-portfolios"></a>
<a id="5-adding-an-optimiser"></a>
<a id="6-default-parameters"></a>
<a id="7-price-time-series-data"></a>
<a id="8-drift-aware-rolling-backtests-v531"></a>

| Subpackage | What it holds | Articles |
| --- | --- | --- |
| `covar_estimation` | EWMA covariance estimation; factor models are prepared upstream in FM | [covariance estimators](https://optimalportfolios.readthedocs.io/en/latest/covariance_estimators.html), [supplied risk inputs](docs/supplied_risk_inputs.md) |
| `alphas` | Momentum, low-beta, carry, residual and managers' alpha signals, `AlphasData`, profiling and diagnostics | [alpha signals](https://optimalportfolios.readthedocs.io/en/latest/alphas_module_readme.html), [signal diagnostics](https://optimalportfolios.readthedocs.io/en/latest/signal_diagnostics_and_profiling.html) |
| `optimization` | The rolling dispatcher, the general, risk-allocation, SAA and TAA solvers, `Constraints`, `OptimiserConfig` and solver diagnostics | [choosing an objective](https://optimalportfolios.readthedocs.io/en/latest/optimization_module_readme.html), [constraints](https://optimalportfolios.readthedocs.io/en/latest/constraints.html), [solver outcomes](https://optimalportfolios.readthedocs.io/en/latest/solver_numerics_and_outcomes.html), [risk budgeting](https://optimalportfolios.readthedocs.io/en/latest/risk_budgeting.html), [HRP and cluster budgets](https://optimalportfolios.readthedocs.io/en/latest/hierarchical_risk_parity_and_cluster_budgets.html), [rolling backtests](https://optimalportfolios.readthedocs.io/en/latest/rolling_backtests.html) |
| `universe` | `UniverseData` and its transforms, such as unsmoothing | [universe data](https://optimalportfolios.readthedocs.io/en/latest/universe_data_and_unsmoothing.html) |
| `utils` | Risk contributions, benchmark betas, weight drift, NaN filtering, Gaussian mixtures | [risk contributions and betas](https://optimalportfolios.readthedocs.io/en/latest/portfolio_risk_analytics.html) |
| `reports` | Result plots, marginal backtests and optional PyBloqs reports | [analytics gallery](https://optimalportfolios.readthedocs.io/en/latest/analytics_gallery.html) |

The [software-design guide](https://optimalportfolios.readthedocs.io/en/latest/software_design.html) draws the module imports, and the
[API reference](https://optimalportfolios.readthedocs.io/en/latest/api.html) lists every public object.

OP produces constrained target weights and solver outcomes from supplied alphas and
covariances. `PortfolioOptimisationResult` can inspect those weights on their own,
with a supplied covariance matrix, or with a QIS `RiskModel` and prepared diagnostic
tables. Neither `factormodels` nor `factorlasso` is an OP dependency.
Factor model construction and GLS inference belong to FactorModels; statistical
estimation belongs to FactorLasso. See the [breaking migration contract](docs/supplied_risk_inputs.md).

**Architecture: completed outputs across the stack.**
[`factorlasso`](https://github.com/ArturSepp/factorlasso) is the domain-agnostic sparse
factor-model estimator, with sign constraints, prior-centered regularisation and HCGL clustering;
it produces statistical estimates. `factormodels` constructs definitions and NAVs,
fits the asset universe through FL and completes the dated loading/covariance model.
OP receives supplied alphas and covariances and produces constrained weights. QIS
provides general risk/stress analytics, backtesting and reporting. Each completed
object can be inspected before its separate downstream use.

### Analytics at a glance

| Area | Current user-facing analytics |
| --- | --- |
| Alpha construction | Momentum, low beta, risk-adjusted carry, managers alpha, residual momentum, residual reversal and rolling EWMA means; fixed-group and time-varying cluster scoring are supported. |
| Alpha evaluation | Rank-portfolio profiling, cross-backtests, `AlphasData`, IC/IR panels, component diagnostics and comparison tables. |
| Covariance inputs | Current/rolling EWMA or supplied covariance matrices, including completed FM estimates. |
| Cluster inputs | Supplied static or dated cluster mappings and prepared scores. Lineage and statistical clustering remain in FactorLasso; OP aliases are removed. |
| General optimisation | Minimum variance, quadratic utility, maximum Sharpe, maximum diversification, CARA Gaussian-mixture utility and minimum tracking error. |
| Risk allocation | Constrained risk budgeting, point-in-time group risk budgets, date-varying rolling budgets, group Euler-risk attribution and external-linkage hierarchical risk parity. |
| SAA and TAA optimisation | Minimum variance at target return, maximum return at target volatility, alpha over tracking error and alpha at target portfolio return. |
| Constraints and implementation | Instrument/group bounds, exposure, turnover, tracking error, target return/volatility, benchmark-relative sector/style/beta limits, frozen holdings and current-to-model eligibility corridors. |
| Solver controls and diagnostics | One covariance factorization per compatible CVXPY solve, input-contract validation, structured `OptimizationOutcome`/`ConstraintResidual` output, infeasibility diagnosis and run-level warning summaries. |
| Portfolio and risk results | `PortfolioOptimisationResult` provides weights/trades, volatility, turnover, tracking error, factor/residual risk, group attribution, factor exposures, efficient-frontier data and report tables using `qis.RiskModel`. |
| Universe, backtest and reporting | Validated `UniverseData`, metadata/group-loadings persistence and transforms, drift-aware rolling weights, transaction-cost backtests through `qis`, efficient-frontier plots, marginal portfolio backtests and optional PyBloqs HTML/PDF reports. |

This table groups the analytics by workflow. The exact package-root import inventory and callable
signatures are maintained in the [API reference](https://optimalportfolios.readthedocs.io/en/latest/api.html).

<a id="quick-start-offline-rolling-backtest"></a>

## Five-minute quickstart

The [production quickstart](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/getting_started/production_quickstart.py) is the authoritative
source for the first-use workflow. It runs entirely offline on the multi-asset fixture shipped in
the wheel and writes no files:

```bash
pip install optimalportfolios
python examples/getting_started/production_quickstart.py
```

For a zero-setup trial, [open the mechanically checked mirror in
Colab](https://colab.research.google.com/github/ArturSepp/OptimalPortfolios/blob/main/examples/getting_started/production_quickstart.ipynb).
The notebook installs the latest PyPI release, prints its version, and adds no notebook dependency
to the package.

The script uses a documented six-asset slice, a point-in-time 24-month EWMA covariance estimator,
quarterly constrained minimum-variance weights, a one-month implementation lag, and 10 basis
points of transaction costs. It prints the data range, rolling-weight dimensions, final weights,
final NAV, and measured runtime. The
[rendered quickstart documentation](https://optimalportfolios.readthedocs.io/en/latest/quickstart.html)
includes this same file directly, so the example and documentation cannot drift.

### A minimal executable example

The script above remains the authoritative first-use workflow. The shorter version below exists so
that the README's own code is executed rather than trusted:

```python
import qis
from optimalportfolios import (
    Constraints,
    EwmaCovarEstimator,
    PortfolioObjective,
    compute_rolling_optimal_weights,
)
from optimalportfolios.tests.data.multiasset import load_multiasset_data

prices = load_multiasset_data().prices.iloc[-120:, :4]
time_period = qis.TimePeriod(prices.index[0], prices.index[-1])

# estimate covariance → optimise → get rolling weights
estimator = EwmaCovarEstimator(returns_freq='ME', span=24, rebalancing_freq='QE')
covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)
weights = compute_rolling_optimal_weights(prices=prices,
                                          portfolio_objective=PortfolioObjective.MAX_DIVERSIFICATION,
                                          constraints=Constraints(is_long_only=True),
                                          time_period=time_period,
                                          covar_dict=covar_dict)

# backtest with transaction costs
portfolio = qis.backtest_model_portfolio(prices=prices.loc[weights.index[0]:], weights=weights,
                                         rebalancing_costs=0.001, ticker='MaxDiv')

print(f"assets: {list(weights.columns)}")
print(f"rebalance dates: {len(weights.index)}")
print(f"long only: {bool((weights >= -1e-6).all().all())}")
print(f"fully invested: {bool(weights.sum(axis=1).round(6).eq(1.0).all())}")
print(f"nav name: {portfolio.nav.name}")
```

```result
assets: ['Global Bonds', 'Global IG Bonds', 'US Treasuries', 'US TIPs']
rebalance dates: 39
long only: True
fully invested: True
nav name: MaxDiv
```

`readme_test.py` executes the block above and diffs its output against that
`result` fence, so this example cannot drift from what the package actually
does. Structural facts are asserted rather than weights: a solver-version change
may move an allocation by 1e-9, but it must not change the rebalance schedule,
break long-only, or stop the book being fully invested.

The committed multi-asset fixture keeps this example offline. The same pipeline
supports price panels with NaNs and different start dates, while preserving
roll-forward estimation (no hindsight bias) and drift-aware turnover accounting.

## Examples

The `examples/` folder is organised by purpose. The
[examples and recipes guide](https://optimalportfolios.readthedocs.io/en/latest/examples_readme.html) maps every task to its article, canonical
script and standalone examples, with each example's offline, network or local-data lane:

```text
examples/
├── getting_started/       Canonical offline quickstart and its notebook mirror
├── data/                  Shared Yahoo loaders and local universe builders
├── solvers/               Objective-specific examples
├── backtests/             Complete rolling workflows
├── comparisons/           Comparisons of methods or configurations, including the drift policy
├── covar_estimation/      Covariance and factor-model examples
├── alphas/                Signal profiling
├── reports/               Manually prepared portfolio reports
├── docs/                  Canonical scripts of the documentation pages
└── figures/               Existing documentation previews
```

### Recommended reading order for newcomers

1. [Quickstart](https://optimalportfolios.readthedocs.io/en/latest/quickstart.html) — the smallest offline portfolio workflow.
2. [`examples/backtests/multiasset_saa.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/backtests/multiasset_saa.py) — another offline workflow with group metadata and objective choices.
3. [`examples/data/universe.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/data/universe.py) and [`examples/backtests/minimal_backtest.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/backtests/minimal_backtest.py) — downloaded prices and reporting.
4. [`examples/solvers/min_variance.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/solvers/min_variance.py) and [`examples/solvers/minimum_tracking_error.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/solvers/minimum_tracking_error.py) — covariance-based construction.
5. [`examples/solvers/tracking_error.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/solvers/tracking_error.py) — benchmark-relative allocation with a signal.
6. [`examples/comparisons/optimisers.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/comparisons/optimisers.py) — how objectives differ on the same universe.

### Highlighted examples

The previews in this section are offline teaching exhibits on a fixed synthetic sample ending
31 December 2025, produced by the scripts named with each; the
[shared provenance record](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/figures/analytics_manifest.json) records their
inputs, configuration, software versions and visual review.

#### Optimal portfolio backtest

[`examples/backtests/minimal_backtest.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/backtests/minimal_backtest.py) fetches eight
ETFs, estimates an EWMA covariance, solves maximum diversification each quarter, backtests with
transaction costs and writes a qis factsheet.

[![Synthetic maximum-diversification portfolio growth and drawdowns](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/example_portfolio_factsheet1.PNG)](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/example_portfolio_factsheet1.PNG)
[![Synthetic maximum-diversification target weights and contributions to annualized volatility](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/example_portfolio_factsheet2.PNG)](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/example_portfolio_factsheet2.PNG)

#### Customised reporting

`PortfolioData` from [qis](https://github.com/ArturSepp/QuantInvestStrats) plots NAV, weights and
return scatters. The preview from [`portfolio_reports.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/tools/docs_analytics/portfolio_reports.py)
shows quarterly target weights and realised trading costs.

[![Synthetic portfolio target weights and quarterly trading costs](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/example_customised_report.PNG)](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/example_customised_report.PNG)

#### Parameter sensitivity backtest

[`examples/comparisons/parameter_sensitivity.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/comparisons/parameter_sensitivity.py)
backtests one method across estimation parameters; the preview from
[`span_sensitivity.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/tools/docs_analytics/span_sensitivity.py) compares five EWMA spans.

[![Synthetic maximum-diversification performance and trading costs across EWMA spans](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/max_diversification_span.PNG)](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/max_diversification_span.PNG)

#### Multi-optimiser cross-backtest

[`examples/comparisons/optimisers.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/comparisons/optimisers.py) runs several objectives
through `compute_rolling_optimal_weights()`; the preview from
[`optimiser_comparison.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/tools/docs_analytics/optimiser_comparison.py) compares minimum
variance, maximum diversification and equal risk budgets on shared inputs.

[![Synthetic net performance and trading costs for three covariance-only objectives](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/multi_optimisers_backtest.PNG)](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/multi_optimisers_backtest.PNG)

#### Multi-covariance-estimator backtest

[`examples/comparisons/covar_estimators.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/comparisons/covar_estimators.py) backtests one
objective with several covariance estimators; the preview from
[`covariance_comparison.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/tools/docs_analytics/covariance_comparison.py) compares six estimators
on one known-factor simulation, which does not establish an estimator ranking.

[![Synthetic minimum-variance performance and covariance errors for six estimators](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)](https://raw.githubusercontent.com/ArturSepp/OptimalPortfolios/main/examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)

### From method to implementation

| Method | Documentation page | Canonical script |
|---|---|---|
| ROSAA: risk-budgeted SAA and tracking-error-constrained TAA on one roll-forward state | [ROSAA case study](https://optimalportfolios.readthedocs.io/en/latest/app_rosaa_multi_asset_allocation.html) | [`examples/docs/app_rosaa_multi_asset_allocation.py`](examples/docs/app_rosaa_multi_asset_allocation.py) |
| Optimal allocation to cryptocurrencies: the four methods of the paper | [cryptocurrency case study](https://optimalportfolios.readthedocs.io/en/latest/app_crypto_allocation.html) | [`examples/docs/app_crypto_allocation.py`](examples/docs/app_crypto_allocation.py) |
| CMA-based strategic allocation on a supplied factor-model covariance | [CMA strategic allocation](docs/app_cma_strategic_allocation.md) | [`examples/docs/app_cma_strategic_allocation.py`](examples/docs/app_cma_strategic_allocation.py) |
| Coverage-floor overlay allocation (convexity premium, Section II) | [overlay page](https://optimalportfolios.readthedocs.io/en/latest/overlay_tail_floor.html), [smart diversification case study](https://optimalportfolios.readthedocs.io/en/latest/app_smart_diversification_overlays.html) | [`examples/docs/overlay_tail_floor.py`](examples/docs/overlay_tail_floor.py), [`examples/docs/app_smart_diversification_overlays.py`](examples/docs/app_smart_diversification_overlays.py) |
| Drift-aware turnover under a binding budget (`use_drifted_weights_0` True vs False) | [turnover and transaction costs](https://optimalportfolios.readthedocs.io/en/latest/turnover_and_transaction_costs.html) | [`examples/comparisons/drift_policy.py`](examples/comparisons/drift_policy.py) |

## Testing

```bash
uv sync --locked --group test
uv run --no-sync pytest
uv run --locked --only-group lint ruff check --select TID251,TID253,ICN,F src/optimalportfolios/
```

After installing `pytest`, verify an installed wheel with `python -m pytest --pyargs optimalportfolios`.
`readme_test.py` executes the minimal example above and diffs its output against the `result`
fence. Tests that need an optional extra skip rather than fail, so a core install stays green.
[CONTRIBUTING.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CONTRIBUTING.md)
describes the contributor environment.

## Changelog

Release history is maintained in [CHANGELOG.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CHANGELOG.md).

## Feedback & contributing <a name="feedback-contributing"></a>

- **Bug:** use the [bug-report form](https://github.com/ArturSepp/OptimalPortfolios/issues/new?template=bug_report.yml) with the package version, Python/platform, a minimal public-data reproducer, and expected versus actual output.
- **Feature:** use the [feature-request form](https://github.com/ArturSepp/OptimalPortfolios/issues/new?template=feature_request.yml) and describe the user goal, current workaround, and smallest useful API. In particular: which constraint, report, or portfolio workflow cannot be expressed today?
- **Question or methodology:** search or open an [issue](https://github.com/ArturSepp/OptimalPortfolios/issues) and name the paper, example, or convention involved.
- **Contribution:** follow [CONTRIBUTING.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CONTRIBUTING.md); focused work is listed under [`good first issue`](https://github.com/ArturSepp/OptimalPortfolios/labels/good%20first%20issue) and [`help wanted`](https://github.com/ArturSepp/OptimalPortfolios/labels/help%20wanted).

Project decisions, maintenance expectations, release policy, and best-effort support routes are
documented in [GOVERNANCE.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/GOVERNANCE.md).

## Acknowledgments

- [Thomas Schmelzer](https://github.com/tschm), creator of
  [Jebel-Quant/rhiza](https://github.com/Jebel-Quant/rhiza), for substantial contributions to test
  coverage, cross-platform CI/CD, dependency auditing, example validation, packaging, and
  built-wheel verification.

## Citation

A machine-readable citation is available in [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

If you use optimalportfolios in your research, please cite it as:

```bibtex
@software{sepp2026optimalportfolios,
  author={Sepp, Artur},
  title={optimalportfolios: point-in-time multi-asset portfolio construction and rolling backtesting in Python},
  year={2026},
  version={8.0.0},
  url={https://github.com/ArturSepp/OptimalPortfolios}
}
```

```bibtex
@article{sepp2023,
  title={Optimal allocation to cryptocurrencies in diversified portfolios},
  author={Sepp, Artur},
  journal={Risk Magazine},
  pages={1--6},
  month={October},
  year={2023},
  url={https://ssrn.com/abstract=4217841}
}
```

```bibtex
@article{sepp2026rosaa,
  author={Sepp, Artur and Ossa, Ivan and Kastenholz, Mika},
  title={Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios},
  journal={The Journal of Portfolio Management},
  volume={52},
  number={4},
  pages={86--120},
  year={2026}
}
```

```bibtex
@article{sepphansenkastenholz2026,
  title={Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors},
  author={Sepp, Artur and Hansen, Emilie H. and Kastenholz, Mika},
  journal={The Journal of Portfolio Management},
  note={Forthcoming},
  year={2027}
}
```

```bibtex
@article{seppkastenholz2026convexity,
  title={The Convexity Premium of Portfolio Overlays},
  author={Sepp, Artur and Kastenholz, Mika},
  journal={Journal of Investment Management},
  year={2026}
}
```

## Ecosystem

This package is part of an open-source Python stack for quantitative finance. The
[ArturSepp profile](https://github.com/ArturSepp) is the canonical catalogue:

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, backtesting, and factsheet reporting |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) *(this package)* | Portfolio construction and rolling backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor-model estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data in pandas DataFrames |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain data and queries |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Numba-vectorised BSM and Bachelier pricing |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic-volatility pricing and calibration |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Closed-form trend-following analytics |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Multi-factor PME for private assets |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Goal-based allocation under regime-switching jump-diffusions |

Within the stack, `optimalportfolios` directly depends on `qis` for analytics and reporting.
`factormodels` and `factorlasso` are separately installed upstream tools, not dependencies; see
[supplied risk inputs](docs/supplied_risk_inputs.md#factormodels-availability) for the
`factormodels` availability note.

## License

MIT — see [LICENSE.txt](https://github.com/ArturSepp/OptimalPortfolios/blob/main/LICENSE.txt).

## Disclaimer

`optimalportfolios` is distributed FREE & WITHOUT ANY WARRANTY under the MIT License. See
[LICENSE.txt](https://github.com/ArturSepp/OptimalPortfolios/blob/main/LICENSE.txt) in the
release for details. Use the routes in [Feedback & contributing](#feedback-contributing) for
bugs, feature requests and methodology questions.
