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

Explore the [analytics gallery](docs/analytics_gallery.md) for reproducible synthetic examples
with sample dates, conventions, producer links and reviewed provenance.

**Production multi-asset portfolio construction and rolling backtesting in Python — from
point-in-time covariance and alpha estimation through constrained optimisation, rebalancing,
transaction costs, and reporting.**

**Install:** `pip install optimalportfolios` · **Import:** `optimalportfolios` · **Status:** Stable

[![PyPI](https://img.shields.io/pypi/v/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/)
[![Python](https://img.shields.io/pypi/pyversions/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/)
[![CI](https://github.com/ArturSepp/OptimalPortfolios/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/OptimalPortfolios/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/optimalportfolios/badge/?version=latest)](https://optimalportfolios.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ArturSepp/OptimalPortfolios.svg?style=flat-square)](LICENSE.txt)
[![Downloads](https://static.pepy.tech/badge/optimalportfolios)](https://pepy.tech/project/optimalportfolios)
[![Monthly](https://static.pepy.tech/badge/optimalportfolios/month)](https://pepy.tech/project/optimalportfolios)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/OptimalPortfolios/blob/main/examples/getting_started/production_quickstart.ipynb)

**Papers:** Sepp, A. (2023), *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*, Risk Magazine — [SSRN 4217841](https://ssrn.com/abstract=4217841) · Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios*, [The Journal of Portfolio Management, 52(4), 86–120](https://www.pm-research.com/content/iijpormgmt/52/4/86) · Sepp, A., Hansen, E. and Kastenholz, M. (2026), *Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors* — [SSRN 6785958](https://ssrn.com/abstract=6785958). See [References](#references).

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

**Production multi-asset portfolio construction.**
The package implements the full pipeline from the ROSAA framework: factor model
covariance estimation (via [`factorlasso`](https://github.com/ArturSepp/factorlasso))
→ risk-budgeted SAA → alpha signal computation →
TE-constrained TAA → rolling backtest. In this pipeline, equities can rebalance
monthly while alternatives rebalance quarterly, and an asset can enter the
allocation set only when sufficient return history is available. Weight bounds,
group allocation limits, tracking error budgets, turnover controls, and rebalancing
indicators for frozen positions share the same dated roll-forward state.

**HCGL factor covariance estimation.**
The Hierarchical Clustering Group LASSO factor model (published in JPM, 2026)
produces sparse, structured covariance matrices for heterogeneous multi-asset
universes. The LASSO/Group LASSO/HCGL solver is implemented in the standalone
[`factorlasso`](https://github.com/ArturSepp/factorlasso) package — a
general-purpose sparse factor model estimator with sign constraints,
prior-centred regularisation, and scikit-learn-compatible API.
`optimalportfolios` builds on top of `factorlasso` with finance-specific
functionality: `FactorCovarEstimator` handles multi-frequency asset returns,
rolling estimation schedules, factor covariance assembly
(Σ_y = β Σ_x β' + D), and integration with `qis` for performance attribution.
The separation means the LASSO solver can be used independently for any
multi-output regression problem (genomics, macro-econometrics), while the
portfolio-specific rolling pipeline stays in `optimalportfolios`.

**Cluster-aware risk allocation.**
Statistical clusters can be used after covariance estimation as an allocation
structure rather than only as a modelling diagnostic. `compute_group_risk_budgets()`
maps point-in-time clusters, sectors, or asset classes into asset-level risk
budgets; `rolling_risk_budgeting()` accepts either one static budget Series or a
date-by-asset budget panel; and `compute_hierarchical_risk_parity_weights()`
implements canonical HRP from an externally supplied linkage. Cluster formation,
distance transforms, De-PC1 diagnostics, and linkage estimation remain in
[`factorlasso`](https://github.com/ArturSepp/factorlasso); OptimalPortfolios owns
the conversion from that structure into portfolio weights and risk attribution.

**Drift-aware rolling backtests (new in v5.3.1).**
Turnover constraints and transaction-cost penalties act on the realised
current holdings, not the previous target. This eliminates the "phantom
turnover budget" issue where the optimiser thinks it's trading X but the
NAV simulator actually trades X·(1 + drift fraction). Controlled by
`OptimiserConfig.use_drifted_weights_0` (default `True`); set to `False`
to reproduce pre-v5.3.1 behaviour for legacy comparisons.

**NaN-aware rolling backtesting.**
The three-layer architecture (solver / wrapper / rolling) automatically handles
real-world data: assets with missing prices receive zero weight, assets entering
the universe mid-sample are included when sufficient history is available, and
the rebalancing indicator system freezes illiquid positions at their current
weight while re-optimising the liquid portion. When the freeze produces
group-constraint overshoots due to drift, the constraint is relaxed for that
rebalance with a logged warning rather than aborting. No data cleaning or
pre-filtering required.

**Research-backed methodology.**
The package is the reference implementation for the ROSAA framework published in
*The Journal of Portfolio Management* (Sepp, Ossa, Kastenholz, 2026). Its
optimisation solvers, covariance estimators, and alpha signals are covered by
offline tests and public worked examples.

<a id="quick-start-offline-rolling-backtest"></a>

## Five-minute quickstart

The [production quickstart](examples/getting_started/production_quickstart.py) is the authoritative
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

The optimisation solvers use quadratic and conic objective functions (variance,
tracking error, Sharpe ratio, diversification ratio, CARA utility). The package
does not implement non-quadratic risk measures (CVaR, MAD, drawdown constraints).
For these, use Riskfolio-Lib or skfolio. The solver architecture (three-layer:
mathematical / wrapper / rolling) makes it straightforward to add new solvers —
each solver lives in its own module in `optimization/general`,
`optimization/risk_allocation`, `optimization/saa`, or `optimization/taa` and
plugs into the rolling backtester via a single dispatch function. The
[software-design guide](https://optimalportfolios.readthedocs.io/en/latest/software_design.html)
explains these boundaries and the alternatives considered; the
[package comparison](https://optimalportfolios.readthedocs.io/en/latest/package_comparison.html)
records the versioned evidence for the field comparison.

## When to use it — and when not

Use `optimalportfolios` when you need a dated roll-forward pipeline from point-in-time covariance
and alpha estimates to constrained targets, scheduled rebalancing, and drift-aware backtests with
transaction costs across incomplete or mixed-frequency multi-asset panels.

Choose another package when the core problem is a non-quadratic risk measure such as CVaR, MAD,
or drawdown constraints; the package comparison points to Riskfolio-Lib and skfolio for those
workflows. Use `factorlasso` directly when you need its standalone sparse multi-output factor-model
estimator rather than portfolio construction.

## Package overview

```
src/optimalportfolios/
├── config.py                      # PortfolioObjective enum
├── alphas/                        # Alpha signal computation
│   ├── signals/                   # risk-adjusted/classic momentum, carry, low_beta,
│   │                              #   residual momentum/reversal, managers_alpha, rolling_ewma_mean
│   ├── profile/                   # Signal profiling
│   ├── alpha_data.py              # AlphasData container
│   ├── backtest_alphas.py         # Signal backtesting tool
│   └── signal_diagnostics.py      # Signal IC-IR and risk-contribution diagnostics
├── covar_estimation/              # Covariance matrix estimation
│   ├── covar_estimator.py         # CovarEstimator ABC
│   ├── ewma_covar_estimator.py    # EwmaCovarEstimator
│   ├── factor_covar_estimator.py  # FactorCovarEstimator (uses factorlasso)
│   ├── risk_model_adapter.py      # Canonical qis.RiskModel adapter
│   ├── risk_labelling.py          # Deprecated shim; canonical lineage is in factorlasso
│   └── covar_reporting.py         # Rolling covariance diagnostics
├── optimization/                  # Portfolio optimisation
│   ├── constraints/               # Canonical public facade and constraint owners
│   │   ├── core.py                # Constraints aggregate and enforcement enum
│   │   ├── alignment.py           # Universe alignment and frozen-bound relaxation
│   │   ├── analytics.py           # Pure residual and feasibility analytics
│   │   ├── backends.py            # CVXPY, SciPy and risk-budgeting translations
│   │   ├── benchmarks.py          # Benchmark-deviation and beta constraints
│   │   ├── expressions.py         # Shared CVXPY risk and objective expressions
│   │   └── groups.py              # Group allocation, TRE and turnover constraints
│   ├── config.py                  # OptimiserConfig (incl. use_drifted_weights_0)
│   ├── covar_factorization.py     # Stabilised covariance and square-root factor
│   ├── solver_diagnostics.py      # Input contracts, outcomes, fallback and run summaries
│   ├── portfolio_result.py        # PortfolioOptimisationResult
│   ├── wrapper_rolling_portfolios.py  # compute_rolling_optimal_weights()
│   ├── general/                   # Objective-driven solvers
│   │   ├── quadratic.py           # min variance, max quadratic utility
│   │   ├── minimum_tracking_error.py  # closest feasible portfolio to benchmark
│   │   ├── max_sharpe.py          # maximum Sharpe ratio
│   │   ├── max_diversification.py # maximum diversification ratio
│   │   └── carra_mixture.py       # CARA utility under Gaussian mixture
│   ├── risk_allocation/           # Risk-based portfolio construction
│   │   ├── risk_budgeting.py      # constrained and rolling risk budgeting
│   │   ├── risk_budgeting_solver.py  # internal CCD/ADMM solver
│   │   ├── group_risk_budgeting.py   # group-to-asset risk budgets
│   │   └── hierarchical_risk_parity.py  # external-linkage HRP
│   ├── saa/                       # Strategic solvers with return/vol targets
│   │   ├── min_variance_target_return.py
│   │   └── max_return_target_vol.py
│   └── taa/                       # Tactical solvers with alpha and TE constraints
│       ├── maximise_alpha_over_tre.py
│       └── maximise_alpha_with_target_yield.py
├── universe/                      # Validated universe data containers and transforms
│   ├── universe_data.py           # UniverseData: prices, metadata and group loadings
│   └── universe_transforms.py     # e.g. copy with unsmoothed prices
├── utils/                         # Auxiliary analytics
│   ├── benchmark_beta.py          # Benchmark-beta loadings and dated portfolio beta
│   ├── filter_nans.py             # NaN-aware covariance/vector filtering
│   ├── portfolio_funcs.py         # Risk contributions, diversification ratio
│   ├── weights_drift.py           # apply_drift_to_weights_0
│   └── gaussian_mixture.py        # Gaussian mixture fitting (numpy/scipy EM)
└── reports/                       # Performance reporting
    ├── marginal_backtest.py       # Marginal asset contribution analysis
    ├── portfolio_result_plots.py  # Optimisation result plots
    └── portfolio_result_pybloqs.py  # Optional HTML/PDF result reports
examples/                          # Repository-only worked examples
├── data/                          # Shared universe fixtures
├── solvers/                       # One demo per single-objective solver
├── backtests/                     # End-to-end rolling workflows
├── comparisons/                   # A-vs-B sweeps (incl. drift_policy)
├── covar_estimation/              # Covariance estimator demos
└── alphas/                        # Alpha signal profiling demos
# factorlasso (pip install factorlasso)
#   └── LassoModel, solve_lasso_cvx_problem, solve_group_lasso_cvx_problem
#       Sign-constrained LASSO/Group LASSO/HCGL solver (domain-agnostic)
#       https://github.com/ArturSepp/factorlasso
```

### Analytics at a glance

| Area | Current user-facing analytics |
| --- | --- |
| Alpha construction | Momentum, low beta, risk-adjusted carry, managers alpha, residual momentum, residual reversal and rolling EWMA means; fixed-group and time-varying cluster scoring are supported. |
| Alpha evaluation | Rank-portfolio profiling, cross-backtests, `AlphasData`, IC/IR panels, component diagnostics and comparison tables. |
| Covariance and dependence | Current/rolling EWMA and HCGL sparse factor covariance; Pearson, Spearman and Gerber dependence choices, configurable correlation-distance transforms through `factorlasso`, and current/rolling covariance diagnostic reports. |
| Risk-cluster analytics | Persistent cluster lineage, births/deaths/splits/merges and report tables/figures through `analyze_risk_clusters()` and `run_risk_label_report()`. |
| General optimisation | Minimum variance, quadratic utility, maximum Sharpe, maximum diversification, CARA Gaussian-mixture utility and minimum tracking error. |
| Risk allocation | Constrained risk budgeting, point-in-time group risk budgets, date-varying rolling budgets, group Euler-risk attribution and external-linkage hierarchical risk parity. |
| SAA and TAA optimisation | Minimum variance at target return, maximum return at target volatility, alpha over tracking error and alpha at target portfolio return. |
| Constraints and implementation | Instrument/group bounds, exposure, turnover, tracking error, target return/volatility, benchmark-relative sector/style/beta limits, frozen holdings and current-to-model eligibility corridors. |
| Solver controls and diagnostics | One covariance factorization per compatible CVXPY solve, input-contract validation, structured `OptimizationOutcome`/`ConstraintResidual` output, infeasibility diagnosis and run-level warning summaries. |
| Portfolio and risk results | `PortfolioOptimisationResult` provides weights/trades, volatility, turnover, tracking error, factor/residual risk, group attribution, factor exposures, efficient-frontier data and report tables using `qis.RiskModel`. |
| Universe, backtest and reporting | Validated `UniverseData`, metadata/group-loadings persistence and transforms, drift-aware rolling weights, transaction-cost backtests through `qis`, efficient-frontier plots, marginal portfolio backtests and optional PyBloqs HTML/PDF reports. |

This table groups the analytics by workflow. The exact package-root import inventory and callable
signatures are maintained in the [API reference](docs/api.rst).

**Architecture: factorlasso vs optimalportfolios**

[`factorlasso`](https://github.com/ArturSepp/factorlasso) is the **domain-agnostic
LASSO solver** — it estimates sparse factor loadings β in Y_t = α + β X_t + ε_t with sign
constraints, prior-centered regularisation, and HCGL clustering. It provides
`LassoModel` (scikit-learn compatible estimator), `CurrentFactorCovarData`
(single-date covariance decomposition Σ_y = β Σ_x β' + D), and
`RollingFactorCovarData` (time-indexed collection). It knows nothing about
finance, asset returns, frequencies, or rebalancing schedules.

`optimalportfolios` adds two finance-specific covariance-integration layers on top:

**`estimate_lasso_factor_covar_data()`** — the core estimation function in
`covar_estimation/factor_covar_estimator.py`. It handles everything between
raw market data and the `factorlasso` solver:

* Computes factor returns from prices at the specified frequency
* Estimates annualised factor covariance Σ_x via EWMA
* Calls `factorlasso.LassoModel.fit()` separately per frequency for
  mixed-frequency universes (e.g., monthly equities + quarterly alternatives)
* Annualises residual variances, R², and alphas across frequencies
* Merges multi-frequency betas into a single (N × M) loading matrix
* Returns a `factorlasso.CurrentFactorCovarData` with the full decomposition

**`FactorCovarEstimator`** — a `CovarEstimator` subclass that wraps
`estimate_lasso_factor_covar_data()` in a rolling estimation schedule using
`qis.TimePeriod` and `qis.generate_dates_schedule`. It provides two APIs:

* `fit_rolling_covars()` → `Dict[Timestamp, DataFrame]` (plain covariance
  matrices, plug into any solver)
* `fit_rolling_factor_covars()` → `RollingFactorCovarData` (full
  decomposition with betas, R², clusters, residuals over time)

## Cluster-aware risk allocation

<a id="group-risk-budgets"></a>
<a id="hierarchical-risk-parity"></a>

See the [risk-budgeting guide](https://optimalportfolios.readthedocs.io/en/latest/risk_budgeting.html) for group risk budgets and hierarchical risk parity, including rolling cluster labels.

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

The core package supports Python >=3.10. Its current dependency floors are NumPy >=2.0,
SciPy >=1.12, pandas >=2.2, Matplotlib >=3.8, seaborn >=0.13, openpyxl >=3.1,
PyYAML >=6.0, CVXPY >=1.5.2, SCS >=3.2.4.post3 (excluding 3.3.0),
quadprog >=0.1.11, `qis` >=5.26.0 and
`factorlasso` >=0.17.0. `pyproject.toml` is the source of truth.

Optional extras keep network-data and reporting integrations out of the core
installation. The default risk-lineage matcher is implemented with core NumPy/SciPy code.

| Extra | Adds |
| --- | --- |
| `data` | `yfinance` for free-data example loaders. |
| `reports` | `pybloqs` for HTML/PDF report backends. |
| `docs` | Sphinx, Furo and MyST for documentation builds. |

The runtime integration extras, `data` and `reports`, correspond to features that import their
dependencies. There is no `jupyter` extra: the package imports none of the Jupyter stack, and the
repository-only Colab quickstart uses Google's hosted runtime. Install notebook tooling separately
for local notebooks. The `docs` extra is the documentation toolchain. Tests and static checks are
contributor tooling in the PEP 735 `test` and `lint` dependency groups; there is no `dev` extra.

From a repository checkout, reproduce the locked test and lint environments with:

```bash
uv sync --locked --group test
uv run --no-sync pytest
uv run --locked --only-group lint ruff check src/optimalportfolios/
```

To run the repository-root examples that use free Yahoo data, add `--extra data` to the sync
command. For a user installation with both runtime integrations, install
`optimalportfolios[data,reports]`.

Automated checks are package modules ending in `*_test.py`. Component development runners sit
beside their owning analytics in `src/optimalportfolios/**/run_local/` and end in `_run.py`; invoke
one explicitly, for example with
`python -m optimalportfolios.optimization.general.run_local.quadratic_run`. They are not collected
by pytest or included in built distributions. Repository-root `examples/` are reserved for larger
analytical workflows.

For example:

```bash
pip install optimalportfolios
pip install "optimalportfolios[data,reports]"
```

## Portfolio optimisers

<a id="1-implementation-structure"></a>
<a id="2-example-of-implementation-for-maximum-diversification-solver"></a>
<a id="3-constraints"></a>
<a id="4-wrapper-for-implemented-rolling-portfolios"></a>
<a id="5-adding-an-optimiser"></a>
<a id="6-default-parameters"></a>
<a id="7-price-time-series-data"></a>
<a id="8-drift-aware-rolling-backtests-v531"></a>

See the [optimisation module guide](https://optimalportfolios.readthedocs.io/en/latest/optimization_module_readme.html) for solver architecture, constraints, backends, and configuration. The [rolling backtest guide](https://optimalportfolios.readthedocs.io/en/latest/rolling_backtests.html) covers rebalancing and transaction costs; [supported examples](docs/examples_readme.md) provide complete runnable workflows.

## Examples

The `examples/` folder is organised into six purpose-folders. The
[examples guide](docs/examples_readme.md) maps every demo to its
role; the headlines are:

```
examples/
├── data/                  Universe fixtures (fetch_benchmark_universe_data, fetch_minimal_universe_data)
├── solvers/               One demo per single-objective solver
├── backtests/             End-to-end rolling backtest workflows
├── comparisons/           A-vs-B sweeps (covar / optimiser / parameter / drift policy)
├── covar_estimation/      Covariance estimator demos
└── alphas/                Alpha signal profiling demos (rank-based profiler)
```

### Recommended reading order for newcomers

1. [`examples/data/universe.py`](examples/data/universe.py) — understand the shared fixture.
2. [`examples/backtests/minimal_backtest.py`](examples/backtests/minimal_backtest.py) — see one full workflow end-to-end.
3. [`examples/solvers/min_variance.py`](examples/solvers/min_variance.py) — minimal solver demo with both single-date and rolling forms.
4. [`examples/solvers/minimum_tracking_error.py`](examples/solvers/minimum_tracking_error.py) — covariance-closest feasible portfolio relative to a benchmark.
5. [`examples/solvers/tracking_error.py`](examples/solvers/tracking_error.py) — the production TAA pattern (alpha + benchmark + TE constraint).
6. [`examples/comparisons/optimisers.py`](examples/comparisons/optimisers.py) — see how objectives differ on the same universe.

### Highlighted demos

#### Optimal portfolio backtest

See script [`examples/backtests/minimal_backtest.py`](examples/backtests/minimal_backtest.py).

```python +SKIP
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis

from optimalportfolios import (compute_rolling_optimal_weights, PortfolioObjective,
                               Constraints, EwmaCovarEstimator)
from examples.data.universe import fetch_minimal_universe_data


# 1. fetch universe (8 ETFs across 6 asset-class groups)
prices, benchmark_prices, group_data = fetch_minimal_universe_data()
time_period = qis.TimePeriod('31Dec2004', '15Mar2026')

# 2. define optimisation setup
portfolio_objective = PortfolioObjective.MAX_DIVERSIFICATION
returns_freq = 'W-WED'
rebalancing_freq = 'QE'
span = 52
constraints = Constraints(is_long_only=True,
                          min_weights=pd.Series(0.0, index=prices.columns),
                          max_weights=pd.Series(0.5, index=prices.columns))

# 3. estimate covariance, then optimise
ewma_estimator = EwmaCovarEstimator(returns_freq=returns_freq, span=span,
                                     rebalancing_freq=rebalancing_freq)
covar_dict = ewma_estimator.fit_rolling_covars(prices=prices, time_period=time_period)
weights = compute_rolling_optimal_weights(prices=prices,
                                          portfolio_objective=portfolio_objective,
                                          constraints=constraints,
                                          time_period=time_period,
                                          rebalancing_freq=rebalancing_freq,
                                          covar_dict=covar_dict)

# 4. backtest with transaction costs (drift-aware under v5.3.1 defaults)
portfolio_data = qis.backtest_model_portfolio(prices=prices.loc[weights.index[0]:, :],
                                              weights=weights,
                                              ticker='MaxDiversification',
                                              weight_implementation_lag=1,
                                              rebalancing_costs=0.0010)

# 5. generate factsheet
portfolio_data.set_group_data(group_data=group_data,
                              group_order=list(group_data.unique()))
figs = qis.generate_strategy_factsheet(portfolio_data=portfolio_data,
                                       benchmark_prices=benchmark_prices,
                                       time_period=time_period,
                                       **qis.fetch_default_report_kwargs(time_period=time_period))
qis.save_figs_to_pdf(figs=figs, file_name=f"{portfolio_data.nav.name}_portfolio_factsheet",
                     orientation='landscape', local_path="output/")
```

The two previews below are offline teaching exhibits produced by
[`portfolio_reports.py`](tools/docs_analytics/portfolio_reports.py), with OptimalPortfolios
construction and qis analytics. They use a fixed synthetic sample ending 31 December 2025;
the performance window starts 1 April 2015 and the risk snapshot is dated 1 October 2025.
The [shared provenance record](examples/figures/analytics_manifest.json) records the input,
configuration, software versions and visual review. The full factsheet produced by the code
above has its own layout.

[![Synthetic maximum-diversification portfolio growth and drawdowns](examples/figures/example_portfolio_factsheet1.PNG)](examples/figures/example_portfolio_factsheet1.PNG)
[![Synthetic maximum-diversification target weights and contributions to annualized volatility](examples/figures/example_portfolio_factsheet2.PNG)](examples/figures/example_portfolio_factsheet2.PNG)

#### Customised reporting

Portfolio data class `PortfolioData` is implemented in
[QIS package](https://github.com/ArturSepp/QuantInvestStrats).

```python +SKIP
def run_customised_reporting(portfolio_data) -> plt.Figure:
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), tight_layout=True)
    perf_params = qis.PerfParams(freq='W-WED', freq_reg='ME')
    kwargs = dict(x_date_freq='YE', framealpha=0.8, perf_params=perf_params)
    portfolio_data.plot_nav(ax=axs[0], **kwargs)
    portfolio_data.plot_weights(ncol=len(prices.columns)//3,
                                legend_stats=qis.LegendStats.AVG_LAST,
                                title='Portfolio weights',
                                freq='QE', ax=axs[1], **kwargs)
    portfolio_data.plot_returns_scatter(benchmark_price=benchmark_prices.iloc[:, 0],
                                        ax=axs[2], **kwargs)
    return fig
```

The offline preview from
[`portfolio_reports.py`](tools/docs_analytics/portfolio_reports.py) shows quarterly target
weights and realized trading costs over 1 April 2015 to 31 December 2025 on the same synthetic
sample. Costs include entry; quarterly sums of cost divided by NAV are descriptive.

[![Synthetic portfolio target weights and quarterly trading costs](examples/figures/example_customised_report.PNG)](examples/figures/example_customised_report.PNG)

#### Parameter sensitivity backtest

Cross-sectional backtests test the sensitivity of an optimisation method to
estimation or solver parameters.

See [`examples/comparisons/parameter_sensitivity.py`](examples/comparisons/parameter_sensitivity.py).

The synthetic preview is generated by
[`span_sensitivity.py`](tools/docs_analytics/span_sensitivity.py). It compares net performance
and trading costs for five EWMA spans over 1 April 2015 to 31 December 2025, using common assets,
constraints and implementation dates.

[![Synthetic maximum-diversification performance and trading costs across EWMA spans](examples/figures/max_diversification_span.PNG)](examples/figures/max_diversification_span.PNG)

#### Multi-optimiser cross-backtest

Multiple optimisation methods can be analysed using
`compute_rolling_optimal_weights()`.

See [`examples/comparisons/optimisers.py`](examples/comparisons/optimisers.py).

The synthetic preview from
[`optimiser_comparison.py`](tools/docs_analytics/optimiser_comparison.py) compares minimum
variance, maximum diversification and equal risk budgets over 1 April 2015 to 31 December 2025.
Covariance, assets, constraints and trade dates are shared; the comparison is illustrative.

[![Synthetic net performance and trading costs for three covariance-only objectives](examples/figures/multi_optimisers_backtest.PNG)](examples/figures/multi_optimisers_backtest.PNG)

#### Multi-covariance-estimator backtest

Multiple covariance estimators can be backtested for the same optimisation method.

See [`examples/comparisons/covar_estimators.py`](examples/comparisons/covar_estimators.py).

The preview from
[`covariance_comparison.py`](tools/docs_analytics/covariance_comparison.py) uses a fixed
known-factor Gaussian simulation. It shows net minimum-variance performance over 3 January 2024
to 31 December 2025 and covariance-estimation error at the last decision, 1 October 2025.
OptimalPortfolios constructs portfolios, factorlasso fits sparse factor models, and qis computes
backtests and analytics. One simulated path does not establish an estimator ranking.

[![Synthetic minimum-variance performance and covariance errors for six estimators](examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)](examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)

#### Drift-policy comparison (new in v5.3.1)

Compares `OptimiserConfig.use_drifted_weights_0 = True` (production default)
vs `False` (legacy) using `rolling_quadratic_optimisation` with a binding L1
turnover budget. Shows that under the legacy convention the realised turnover
exceeds the optimiser's apparent turnover by ~23%; under the new default the
two agree.

See [`examples/comparisons/drift_policy.py`](examples/comparisons/drift_policy.py).

#### Optimal allocation to cryptocurrencies

Computations and visualisations for the paper "Optimal Allocation to
Cryptocurrencies in Diversified Portfolios" are maintained as replication code under
[`papers/crypto_allocation_risk_2023`](papers/crypto_allocation_risk_2023/README.md).

Published reference: Sepp A. (2023), "Optimal Allocation to Cryptocurrencies in
Diversified Portfolios", *Risk Magazine*, October 2023, 1-6. Available at
[SSRN](https://ssrn.com/abstract=4217841).

#### Robust optimisation of strategic and tactical asset allocation

Computations and visualisations for the paper "Robust Optimization of Strategic
and Tactical Asset Allocation for Multi-Asset Portfolios" are maintained under
[`papers/robust_optimisation_jpm_2026`](papers/robust_optimisation_jpm_2026/README.md).

The paper presents the ROSAA framework — a unified approach to strategic and
tactical asset allocation for multi-asset portfolios. Key contributions: the
HCGL (Hierarchical Clustering Group LASSO) factor covariance estimator for
heterogeneous multi-asset universes, constrained risk budgeting for SAA with
group allocation limits, and alpha-over-tracking-error optimisation for TAA.
The framework handles real-world challenges including mixed-frequency assets,
incomplete return histories, and illiquid positions requiring rebalancing
indicators. The `optimalportfolios` package is the reference implementation of
the full ROSAA pipeline.

Published reference: Sepp A., Ossa I., and Kastenholz M. (2026), "Robust
Optimization of Strategic and Tactical Asset Allocation for Multi-Asset
Portfolios", *The Journal of Portfolio Management*, 52(4), 86-120.
[Paper link](https://eprints.pm-research.com/17511/143431/index.html).

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
- **Contribution:** follow [CONTRIBUTING.md](CONTRIBUTING.md); focused work is listed under [`good first issue`](https://github.com/ArturSepp/OptimalPortfolios/labels/good%20first%20issue) and [`help wanted`](https://github.com/ArturSepp/OptimalPortfolios/labels/help%20wanted).

Project decisions, maintenance expectations, release policy, and best-effort support routes are
documented in [GOVERNANCE.md](GOVERNANCE.md).

## Citation

A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

If you use optimalportfolios in your research, please cite it as:

```
@software{sepp2026optimalportfolios,
  author={Sepp, Artur},
  title={optimalportfolios: point-in-time multi-asset portfolio construction and rolling backtesting in Python},
  year={2026},
  version={7.7.0},
  url={https://github.com/ArturSepp/OptimalPortfolios}
}
```

```
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

```
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

```
@article{sepphansenkastenholz2026,
  title={Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors},
  author={Sepp, Artur and Hansen, Emilie H. and Kastenholz, Mika},
  journal={Working Paper},
  year={2026}
}
```

## License

MIT — see [LICENSE.txt](LICENSE.txt).

## Disclaimer

OptimalPortfolios package is distributed FREE & WITHOUT ANY WARRANTY under the MIT License.

See the [LICENSE.txt](LICENSE.txt) in the release for details.

Use the dedicated routes in [Feedback & contributing](#feedback-contributing) for bugs, feature
requests, and methodology questions.
