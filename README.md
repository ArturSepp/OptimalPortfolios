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
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Analytics and holdings simulation use [qis](https://github.com/ArturSepp/QuantInvestStrats);
cite its [software record](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Explore the [analytics gallery](https://optimalportfolios.readthedocs.io/en/latest/analytics_gallery.html) for reproducible synthetic examples
with sample dates, conventions, producer links and reviewed provenance.

**Production multi-asset portfolio construction and rolling backtesting in Python — from
point-in-time covariance and alpha estimation through constrained optimisation, rebalancing,
transaction costs, and reporting.**

**Install:** `pip install optimalportfolios` · **Import:** `optimalportfolios` · **Status:** Stable

[![PyPI](https://img.shields.io/pypi/v/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/)
[![Python](https://img.shields.io/pypi/pyversions/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/)
[![CI](https://github.com/ArturSepp/OptimalPortfolios/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/OptimalPortfolios/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/optimalportfolios/badge/?version=latest)](https://optimalportfolios.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ArturSepp/OptimalPortfolios.svg?style=flat-square)](https://github.com/ArturSepp/OptimalPortfolios/blob/main/LICENSE.txt)
[![Downloads](https://static.pepy.tech/badge/optimalportfolios)](https://pepy.tech/project/optimalportfolios)
[![Monthly](https://static.pepy.tech/badge/optimalportfolios/month)](https://pepy.tech/project/optimalportfolios)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/OptimalPortfolios/blob/main/examples/getting_started/production_quickstart.ipynb)

**Papers:** Sepp, A. (2023), *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*, Risk Magazine — [SSRN 4217841](https://ssrn.com/abstract=4217841) · Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios*, [The Journal of Portfolio Management, 52(4), 86–120](https://www.pm-research.com/content/iijpormgmt/52/4/86) · Sepp, A., Hansen, E. and Kastenholz, M. (2026), *Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors* — [SSRN 6785958](https://ssrn.com/abstract=6785958) · Sepp, A. and Kastenholz, M. (2026), *The Convexity Premium of Portfolio Overlays*, Journal of Investment Management, forthcoming — [companion](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/smart_diversification_joim_2026). See [References](#references).

---

## Why optimalportfolios

PyPortfolioOpt, Riskfolio-Lib, and skfolio all provide substantial portfolio
optimisation capabilities. Their documented design centres emphasise, respectively,
compact classical allocation, breadth across risk measures and portfolio families,
and scikit-learn-compatible model selection. `optimalportfolios` is organised around
a different primary abstraction: a dated state transition from estimates and current
holdings to constrained targets and realised backtests.

**optimalportfolios solves the production problem end-to-end:**
estimate covariance → compute alpha signals → optimise with constraints →
rebalance on schedule → backtest with transaction costs — all in a single
roll-forward pipeline that handles incomplete data, mixed-frequency assets,
and illiquid positions.

### Key differentiators

- **Production multi-asset pipeline.** Factor-model covariance, risk-budgeted strategic
  allocation, alpha signals, tracking-error-constrained tactical allocation and a rolling
  backtest share one dated roll-forward state, with equities rebalancing monthly and
  alternatives quarterly. See the [ROSAA case study](https://optimalportfolios.readthedocs.io/en/latest/app_rosaa_multi_asset_allocation.html).
- **HCGL factor covariance.** Sparse, structured covariance for heterogeneous universes, fitted
  by [`factorlasso`](https://github.com/ArturSepp/factorlasso) and assembled point in time by
  `FactorCovarEstimator` across return frequencies. See
  [factor covariance with HCGL](https://optimalportfolios.readthedocs.io/en/latest/factor_covariance_hcgl.html).
- **Cluster-aware risk allocation.** Clusters, sectors or asset classes become asset-level risk
  budgets, static or dated, and canonical HRP runs from a supplied linkage. See
  [hierarchical risk parity and cluster risk budgets](https://optimalportfolios.readthedocs.io/en/latest/hierarchical_risk_parity_and_cluster_budgets.html).
- **Drift-aware rolling backtests.** Turnover limits and cost penalties act on the drifted
  holdings rather than the previous target (`OptimiserConfig.use_drifted_weights_0`, default
  `True`). See [rolling backtests](https://optimalportfolios.readthedocs.io/en/latest/rolling_backtests.html).
- **Incomplete and illiquid histories.** Assets enter when their history suffices, missing
  prices get zero weight, and rebalancing indicators freeze illiquid positions. See
  [incomplete histories and frozen positions](https://optimalportfolios.readthedocs.io/en/latest/incomplete_histories.html).
- **Research-backed.** The reference implementation of the ROSAA framework in *The Journal of
  Portfolio Management*; every number on the methodology pages is asserted by a script that CI
  runs. See [research papers](https://optimalportfolios.readthedocs.io/en/latest/research_papers.html).

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

### Design scope

The solvers use quadratic and conic objectives (variance, tracking error, Sharpe ratio,
diversification ratio, CARA utility); non-quadratic risk measures such as CVaR, MAD or drawdown
constraints are out of scope, and Riskfolio-Lib or skfolio cover them. Each solver lives in its
own module and plugs into the rolling backtester through one dispatch function: the
[software-design guide](https://optimalportfolios.readthedocs.io/en/latest/software_design.html) explains these boundaries, and the
[package comparison](https://optimalportfolios.readthedocs.io/en/latest/package_comparison.html) records the versioned evidence for the field
comparison.

## When to use it — and when not

Use `optimalportfolios` when you need a dated roll-forward pipeline from point-in-time covariance
and alpha estimates to constrained targets, scheduled rebalancing, and drift-aware backtests with
transaction costs across incomplete or mixed-frequency multi-asset panels.

Choose another package when the core problem is a non-quadratic risk measure such as CVaR, MAD,
or drawdown constraints; the package comparison points to Riskfolio-Lib and skfolio for those
workflows. Use `factorlasso` directly when you need its standalone sparse multi-output factor-model
estimator rather than portfolio construction.

## Package overview

| Subpackage | What it holds | Articles |
| --- | --- | --- |
| `covar_estimation` | EWMA and HCGL factor covariance estimators, the `qis.RiskModel` adapter, covariance reports | [covariance estimators](https://optimalportfolios.readthedocs.io/en/latest/covariance_estimators.html), [factor covariance](https://optimalportfolios.readthedocs.io/en/latest/factor_covariance_hcgl.html), [risk contributions and betas](https://optimalportfolios.readthedocs.io/en/latest/portfolio_risk_analytics.html) |
| `alphas` | Momentum, low-beta, carry, residual and managers' alpha signals, `AlphasData`, profiling and diagnostics | [alpha signals](https://optimalportfolios.readthedocs.io/en/latest/alphas_module_readme.html), [signal diagnostics](https://optimalportfolios.readthedocs.io/en/latest/signal_diagnostics_and_profiling.html) |
| `optimization` | The rolling dispatcher, the general, risk-allocation, SAA and TAA solvers, `Constraints`, `OptimiserConfig` and solver diagnostics | [choosing an objective](https://optimalportfolios.readthedocs.io/en/latest/optimization_module_readme.html), [constraints](https://optimalportfolios.readthedocs.io/en/latest/constraints.html), [solver outcomes](https://optimalportfolios.readthedocs.io/en/latest/solver_numerics_and_outcomes.html) |
| `universe` | `UniverseData` and its transforms, such as unsmoothing | [universe data](https://optimalportfolios.readthedocs.io/en/latest/universe_data_and_unsmoothing.html) |
| `utils` | Risk contributions, benchmark betas, weight drift, NaN filtering, Gaussian mixtures | [risk contributions and betas](https://optimalportfolios.readthedocs.io/en/latest/portfolio_risk_analytics.html) |
| `reports` | Result plots, marginal backtests and optional PyBloqs reports | [analytics gallery](https://optimalportfolios.readthedocs.io/en/latest/analytics_gallery.html) |

The [software-design guide](https://optimalportfolios.readthedocs.io/en/latest/software_design.html) draws the module imports, and the
[API reference](https://optimalportfolios.readthedocs.io/en/latest/api.html) lists every public object.

### Analytics at a glance

| Area | Current user-facing analytics |
| --- | --- |
| Alpha construction | Momentum, low beta, risk-adjusted carry, managers alpha, residual momentum, residual reversal and rolling EWMA means; fixed-group and time-varying cluster scoring are supported. |
| Alpha evaluation | Rank-portfolio profiling, cross-backtests, `AlphasData`, IC/IR panels, component diagnostics and comparison tables. |
| Covariance and dependence | Current/rolling EWMA and HCGL sparse factor covariance; Pearson, Spearman and Gerber dependence choices, configurable correlation-distance transforms through `factorlasso`, and current/rolling covariance diagnostic reports. |
| Risk-cluster analytics | Persistent cluster lineage, births/deaths/splits/merges and report tables/figures through `factorlasso.cluster_lineage` (`analyze_cluster_lineage()` and `run_cluster_lineage_report()`); the `analyze_risk_clusters()` and `run_risk_label_report()` aliases in `optimalportfolios.covar_estimation.risk_labelling` are deprecated. |
| General optimisation | Minimum variance, quadratic utility, maximum Sharpe, maximum diversification, CARA Gaussian-mixture utility and minimum tracking error. |
| Risk allocation | Constrained risk budgeting, point-in-time group risk budgets, date-varying rolling budgets, group Euler-risk attribution and external-linkage hierarchical risk parity. |
| SAA and TAA optimisation | Minimum variance at target return, maximum return at target volatility, alpha over tracking error and alpha at target portfolio return. |
| Constraints and implementation | Instrument/group bounds, exposure, turnover, tracking error, target return/volatility, benchmark-relative sector/style/beta limits, frozen holdings and current-to-model eligibility corridors. |
| Solver controls and diagnostics | One covariance factorization per compatible CVXPY solve, input-contract validation, structured `OptimizationOutcome`/`ConstraintResidual` output, infeasibility diagnosis and run-level warning summaries. |
| Portfolio and risk results | `PortfolioOptimisationResult` provides weights/trades, volatility, turnover, tracking error, factor/residual risk, group attribution, factor exposures, efficient-frontier data and report tables using `qis.RiskModel`. |
| Universe, backtest and reporting | Validated `UniverseData`, metadata/group-loadings persistence and transforms, drift-aware rolling weights, transaction-cost backtests through `qis`, efficient-frontier plots, marginal portfolio backtests and optional PyBloqs HTML/PDF reports. |

This table groups the analytics by workflow. The exact package-root import inventory and callable
signatures are maintained in the [API reference](https://optimalportfolios.readthedocs.io/en/latest/api.html).

**Architecture: factorlasso vs optimalportfolios**

[`factorlasso`](https://github.com/ArturSepp/factorlasso) is the domain-agnostic sparse
factor-model estimator, with sign constraints, prior-centred regularisation and HCGL clustering;
it knows nothing about asset returns, frequencies or rebalancing. `optimalportfolios` adds the
finance layer: `estimate_lasso_factor_covar_data()` computes factor returns from prices, fits
the factor model to the asset returns of each frequency and annualises the decomposition, and
`FactorCovarEstimator` runs it on a rolling schedule. See [factor covariance with HCGL](https://optimalportfolios.readthedocs.io/en/latest/factor_covariance_hcgl.html).

## Cluster-aware risk allocation

<a id="group-risk-budgets"></a>
<a id="hierarchical-risk-parity"></a>

See [hierarchical risk parity and cluster risk budgets](https://optimalportfolios.readthedocs.io/en/latest/hierarchical_risk_parity_and_cluster_budgets.html) for group risk budgets and hierarchical risk parity, including date-by-asset cluster labels, and the [risk-budgeting guide](https://optimalportfolios.readthedocs.io/en/latest/risk_budgeting.html) for the solver that group budgets feed.

## Alpha signals module

<a id="naming-convention"></a>
<a id="available-signals"></a>
<a id="mixed-frequency-support"></a>
<a id="alphasdata-container"></a>

See the [alpha signals guide](https://optimalportfolios.readthedocs.io/en/latest/alphas_module_readme.html) for the signal catalogue, mixed-frequency inputs, cluster scoring, and the `AlphasData` container.

## Table of contents

1. [Why optimalportfolios](#why-optimalportfolios)
2. [Package overview](#package-overview)
3. [Cluster-aware risk allocation](#cluster-aware-risk-allocation)
4. [Alpha signals module](#alpha-signals-module)
5. [Installation](#installation)
6. [Portfolio Optimisers](#portfolio-optimisers)
7. [Examples](#examples)
8. [Updates](#updates)
9. [Disclaimer](#disclaimer)

## Installation

Install from PyPI:

```bash
pip install optimalportfolios
```

After installing `pytest`, verify the installed wheel with `python -m pytest --pyargs optimalportfolios`.

Upgrade with:

```bash
pip install --upgrade optimalportfolios
```

Clone the repository with:

```bash
git clone https://github.com/ArturSepp/OptimalPortfolios.git
```

The core package supports Python >=3.10; `pyproject.toml` is the source of truth for dependency
floors, and the [installation guide](https://optimalportfolios.readthedocs.io/en/latest/installation.html) describes the locked environment.
Optional extras keep network-data and reporting integrations out of the core installation:

| Extra | Adds |
| --- | --- |
| `data` | `yfinance` for free-data example loaders. |
| `reports` | `pybloqs` for HTML/PDF report backends. |
| `docs` | Sphinx, Furo and MyST for documentation builds. |

For both runtime integrations, install `optimalportfolios[data,reports]`. There is no `jupyter`
or `dev` extra; tests and static checks are the PEP 735 `test` and `lint` dependency groups, and
[CONTRIBUTING.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CONTRIBUTING.md) describes the contributor environment.

## Portfolio optimisers

<a id="1-implementation-structure"></a>
<a id="2-example-of-implementation-for-maximum-diversification-solver"></a>
<a id="3-constraints"></a>
<a id="4-wrapper-for-implemented-rolling-portfolios"></a>
<a id="5-adding-an-optimiser"></a>
<a id="6-default-parameters"></a>
<a id="7-price-time-series-data"></a>
<a id="8-drift-aware-rolling-backtests-v531"></a>

See the [optimisation module guide](https://optimalportfolios.readthedocs.io/en/latest/optimization_module_readme.html) for solver architecture, constraints, backends, and configuration. The [rolling backtest guide](https://optimalportfolios.readthedocs.io/en/latest/rolling_backtests.html) covers rebalancing and transaction costs; [supported examples](https://optimalportfolios.readthedocs.io/en/latest/examples_readme.html) provide complete runnable workflows.

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
├── comparisons/           Comparisons of methods or configurations
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

### Highlighted demos

#### Optimal portfolio backtest

[`examples/backtests/minimal_backtest.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/backtests/minimal_backtest.py) fetches eight
ETFs, estimates an EWMA covariance, solves maximum diversification each quarter, backtests with
transaction costs and writes a qis factsheet. The previews in this section are offline teaching
exhibits on a fixed synthetic sample ending 31 December 2025, produced by the scripts named with
each; the [shared provenance record](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/figures/analytics_manifest.json) records their
inputs, configuration, software versions and visual review.

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

#### Drift-policy comparison (new in v5.3.1)

[`examples/comparisons/drift_policy.py`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/comparisons/drift_policy.py) compares
`OptimiserConfig.use_drifted_weights_0 = True` (the default) with `False` under a binding
turnover budget; see [turnover and transaction costs](https://optimalportfolios.readthedocs.io/en/latest/turnover_and_transaction_costs.html).

#### Optimal allocation to cryptocurrencies

The paper's replication code is in
[`papers/crypto_allocation_risk_2023`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/papers/crypto_allocation_risk_2023/README.md); the
[cryptocurrency case study](https://optimalportfolios.readthedocs.io/en/latest/app_crypto_allocation.html) reports its design and results and runs
the four methods offline.

#### Robust optimisation of strategic and tactical asset allocation

The paper's example is in
[`papers/robust_optimisation_jpm_2026`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/papers/robust_optimisation_jpm_2026/README.md); the
[ROSAA case study](https://optimalportfolios.readthedocs.io/en/latest/app_rosaa_multi_asset_allocation.html) reports the framework, its study
design and results, and runs the same configuration offline.

#### The convexity premium of portfolio overlays

The paper's synthetic companion is in
[`papers/smart_diversification_joim_2026`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/papers/smart_diversification_joim_2026/README.md);
its Section II coverage-floor allocation is the fixed-core maximum Sharpe ratio with a named
linear row described in the
[overlay page](https://optimalportfolios.readthedocs.io/en/latest/overlay_tail_floor.html), section *The coverage floor*.
The [smart diversification case study](https://optimalportfolios.readthedocs.io/en/latest/app_smart_diversification_overlays.html)
builds the workflow on the companion's synthetic design.

## Updates

See the [changelog](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CHANGELOG.md) for release history and migration notes.

## Acknowledgments

- [Thomas Schmelzer](https://github.com/tschm), creator of
  [Jebel-Quant/rhiza](https://github.com/Jebel-Quant/rhiza), for substantial contributions to test
  coverage, cross-platform CI/CD, dependency auditing, example validation, packaging, and
  built-wheel verification.

## References

Sepp A. (2023),
"Optimal Allocation to Cryptocurrencies in Diversified Portfolios",
*Risk Magazine*, October 2023, 1-6.
Available at <https://ssrn.com/abstract=4217841>

Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
*The Journal of Portfolio Management*, 52(4), 86-120.
[Paper link](https://eprints.pm-research.com/17511/143431/index.html)

Sepp A., Hansen E., and Kastenholz M. (2026),
"Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors",
*Under revision at the Journal of Portfolio Management*.
Available at <https://ssrn.com/abstract=6785958>

Sepp A., and Kastenholz M. (2026),
"The Convexity Premium of Portfolio Overlays",
*Journal of Investment Management*, forthcoming.
Companion code at <https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/smart_diversification_joim_2026>

## Ecosystem

This package is part of an open-source Python stack for quantitative finance. The
[ArturSepp profile](https://github.com/ArturSepp) is the canonical full catalogue:

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, factsheets, and visualisation |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) *(this package)* | Portfolio construction and backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor models and factor covariance estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data fetching |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain normalisation, reconstruction, querying, and visualisation |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Vectorised vanilla option pricers and implied volatility fitters |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) | Stochastic volatility pricing analytics |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Trend-following systems: closed-form theory and replication |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Money-weighted multi-factor alpha from private-asset cash flows |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Dynamic MV allocation under regime-switching jump-diffusions |

Within the stack, `optimalportfolios` directly depends on `qis` for analytics and reporting and on
`factorlasso` for sparse factor covariance estimation. The profile catalogue explains the other
packages and their distinct boundaries.

## Feedback & contributing

- **Bug:** use the [bug-report form](https://github.com/ArturSepp/OptimalPortfolios/issues/new?template=bug_report.yml) with the package version, Python/platform, a minimal public-data reproducer, and expected versus actual output.
- **Feature:** use the [feature-request form](https://github.com/ArturSepp/OptimalPortfolios/issues/new?template=feature_request.yml) and describe the user goal, current workaround, and smallest useful API. In particular: which constraint, report, or portfolio workflow cannot be expressed today?
- **Question or methodology:** search or open an [issue](https://github.com/ArturSepp/OptimalPortfolios/issues) and name the paper, example, or convention involved.
- **Contribution:** follow [CONTRIBUTING.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CONTRIBUTING.md); focused work is listed under [`good first issue`](https://github.com/ArturSepp/OptimalPortfolios/labels/good%20first%20issue) and [`help wanted`](https://github.com/ArturSepp/OptimalPortfolios/labels/help%20wanted).

Project decisions, maintenance expectations, release policy, and best-effort support routes are
documented in [GOVERNANCE.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/GOVERNANCE.md).

## Citation

A machine-readable citation is available in [`CITATION.cff`](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

If you use optimalportfolios in your research, please cite it as:

```bibtex
@software{sepp2026optimalportfolios,
  author={Sepp, Artur},
  title={optimalportfolios: point-in-time multi-asset portfolio construction and rolling backtesting in Python},
  year={2026},
  version={7.10.2},
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
  journal={Working Paper},
  year={2026}
}
```

```bibtex
@article{seppkastenholz2026convexity,
  title={The Convexity Premium of Portfolio Overlays},
  author={Sepp, Artur and Kastenholz, Mika},
  journal={Journal of Investment Management},
  note={Forthcoming},
  year={2026}
}
```

## License

MIT — see [LICENSE.txt](https://github.com/ArturSepp/OptimalPortfolios/blob/main/LICENSE.txt).

## Disclaimer

OptimalPortfolios package is distributed FREE & WITHOUT ANY WARRANTY under the MIT License.

See the [LICENSE.txt](https://github.com/ArturSepp/OptimalPortfolios/blob/main/LICENSE.txt) in the release for details.

Use the dedicated routes in [Feedback & contributing](#feedback-contributing) for bugs, feature
requests, and methodology questions.
