# 🚀 **Optimal Portfolios Construction and Backtesting: optimalportfolios**

> Production-grade multi-asset portfolio construction and backtesting in Python — from covariance estimation to rolling optimisation to factsheet reporting, in a single pipeline that handles real-world data

---

| 📊 Metric | 🔢 Value |
| --- | --- |
| PyPI Version | [![PyPI](https://img.shields.io/pypi/v/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/) |
| Python Versions | [![Python](https://img.shields.io/pypi/pyversions/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/) |
| License | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt) |
| CI Status | [![CI](https://github.com/ArturSepp/OptimalPortfolios/actions/workflows/test.yml/badge.svg)](https://github.com/ArturSepp/OptimalPortfolios/actions) |

### 📈 Package Statistics

| 📊 Metric | 🔢 Value |
| --- | --- |
| Total Downloads | [![Total](https://pepy.tech/badge/optimalportfolios)](https://pepy.tech/project/optimalportfolios) |
| Monthly | [![Monthly](https://pepy.tech/badge/optimalportfolios/month)](https://pepy.tech/project/optimalportfolios) |
| Weekly | [![Weekly](https://pepy.tech/badge/optimalportfolios/week)](https://pepy.tech/project/optimalportfolios) |
| GitHub Stars | [![GitHub stars](https://img.shields.io/github/stars/ArturSepp/OptimalPortfolios?style=flat-square&logo=github)](https://github.com/ArturSepp/OptimalPortfolios/stargazers) |
| GitHub Forks | [![GitHub forks](https://img.shields.io/github/forks/ArturSepp/OptimalPortfolios?style=flat-square&logo=github)](https://github.com/ArturSepp/OptimalPortfolios/network/members) |

---

## **Why optimalportfolios**

Most Python portfolio optimisation packages (PyPortfolioOpt, Riskfolio-Lib, skfolio)
solve single-period allocation problems: given a covariance matrix and expected
returns, find the optimal weights. This is useful for textbook exercises but
insufficient for running a real multi-asset portfolio.

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
TE-constrained TAA → rolling backtest. No other open-source package handles
universes where equities rebalance monthly, alternatives rebalance quarterly,
and private equity enters the allocation set only when sufficient return history
is available. The constraint system (weight bounds, group allocation limits,
tracking error budgets, turnover controls, rebalancing indicators for frozen
positions) matches what real institutional PM teams need.

**HCGL factor covariance estimation.**
The Hierarchical Clustering Group LASSO factor model (published in JPM, 2026)
produces sparse, structured covariance matrices for heterogeneous multi-asset
universes. The LASSO/Group LASSO/HCGL solver is implemented in the standalone
[`factorlasso`](https://github.com/ArturSepp/factorlasso) package — a
general-purpose sparse factor model estimator with sign constraints,
prior-centered regularisation, and scikit-learn compatible API.
`optimalportfolios` builds on top of `factorlasso` with finance-specific
functionality: `FactorCovarEstimator` handles multi-frequency asset returns,
rolling estimation schedules, factor covariance assembly
(Σ_y = β Σ_x β' + D), and integration with `qis` for performance attribution.
The separation means the LASSO solver can be used independently for any
multi-output regression problem (genomics, macro-econometrics), while the
portfolio-specific rolling pipeline stays in `optimalportfolios`.

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
*The Journal of Portfolio Management* (Sepp, Ossa, Kastenholz, 2026). The
optimisation solvers, covariance estimators, and alpha signals are battle-tested
on live multi-asset portfolios.

### Quick-start: rolling backtest in 10 lines

```python
import qis as qis
from optimalportfolios import (EwmaCovarEstimator, Constraints,
                               PortfolioObjective, compute_rolling_optimal_weights)

prices = ...  # pd.DataFrame of asset prices (may have NaNs, different start dates)
time_period = qis.TimePeriod('31Dec2004', '15Mar2026')

# estimate covariance → optimise → get rolling weights
estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)
weights = compute_rolling_optimal_weights(prices=prices,
                                          portfolio_objective=PortfolioObjective.MAX_DIVERSIFICATION,
                                          constraints=Constraints(is_long_only=True),
                                          time_period=time_period,
                                          covar_dict=covar_dict)

# backtest with transaction costs
portfolio = qis.backtest_model_portfolio(prices=prices, weights=weights,
                                         rebalancing_costs=0.001, ticker='MaxDiv')
```

That's it — from prices to backtested portfolio in 10 lines, with automatic NaN
handling, roll-forward estimation (no hindsight bias), drift-aware turnover
accounting, and any optimisation objective.

### Design scope

The optimisation solvers use quadratic and conic objective functions (variance,
tracking error, Sharpe ratio, diversification ratio, CARA utility). The package
does not implement non-quadratic risk measures (CVaR, MAD, drawdown constraints).
For these, use Riskfolio-Lib or skfolio. The solver architecture (three-layer:
mathematical / wrapper / rolling) makes it straightforward to add new solvers —
each solver lives in its own module in `optimization/solvers/` and plugs into the
rolling backtester via a single dispatch function.

## **Package overview**

```
optimalportfolios/
├── alphas/                        # Alpha signal computation
│   ├── signals/
│   │   ├── momentum.py            # compute_momentum_alpha()
│   │   ├── low_beta.py            # compute_low_beta_alpha()
│   │   └── managers_alpha.py      # compute_managers_alpha()
│   ├── alpha_data.py              # AlphasData container
│   ├── backtest_alphas.py         # Signal backtesting tool
│   └── tests/
│       └── signals_test.py
├── covar_estimation/              # Covariance matrix estimation
│   ├── covar_estimator.py         # CovarEstimator ABC
│   ├── ewma_covar_estimator.py    # EwmaCovarEstimator
│   ├── factor_covar_estimator.py  # FactorCovarEstimator (uses factorlasso)
│   ├── factor_covar_data.py       # CurrentFactorCovarData, RollingFactorCovarData
│   └── covar_reporting.py         # Rolling covariance diagnostics
├── optimization/                  # Portfolio optimisation
│   ├── constraints.py             # Constraints, GroupLowerUpperConstraints
│   ├── config.py                  # OptimiserConfig (incl. use_drifted_weights_0)
│   ├── wrapper_rolling_portfolios.py  # compute_rolling_optimal_weights()
│   └── solvers/
│       ├── quadratic.py           # min variance, max quadratic utility
│       ├── risk_budgeting.py      # constrained risk budgeting (pyrb)
│       ├── max_diversification.py # maximum diversification ratio
│       ├── max_sharpe.py          # maximum Sharpe ratio
│       ├── tracking_error.py      # alpha-over-tracking-error
│       ├── target_return.py       # alpha with target return constraint
│       └── carra_mixure.py        # CARA utility under Gaussian mixture
├── utils/                         # Auxiliary analytics
│   ├── filter_nans.py             # NaN-aware covariance/vector filtering
│   ├── portfolio_funcs.py         # Risk contributions, diversification ratio
│   ├── weights_drift.py           # apply_drift_to_weights_0 (new in v5.3.1)
│   ├── gaussian_mixture.py        # Gaussian mixture fitting (pure numpy/scipy EM)
│   └── returns_unsmoother.py      # AR(1) return unsmoothing for PE/PD
├── reports/                       # Performance reporting
│   └── marginal_backtest.py       # Marginal asset contribution analysis
└── examples/                      # Worked examples — see examples/README.md
    ├── data/                      # Shared universe fixtures
    ├── solvers/                   # One demo per single-objective solver
    ├── backtests/                 # End-to-end rolling workflows
    ├── comparisons/               # A-vs-B sweeps (incl. drift_policy)
    └── covar_estimation/          # Covariance estimator demos

# External dependency:
# factorlasso (pip install factorlasso)
#   └── LassoModel, solve_lasso_cvx_problem, solve_group_lasso_cvx_problem
#       Sign-constrained LASSO/Group LASSO/HCGL solver (domain-agnostic)
#       https://github.com/ArturSepp/factorlasso
```

**Architecture: factorlasso vs optimalportfolios**

[`factorlasso`](https://github.com/ArturSepp/factorlasso) is the **domain-agnostic
LASSO solver** — it estimates sparse factor loadings β in Y_t = α + β X_t + ε_t with sign
constraints, prior-centered regularisation, and HCGL clustering. It provides
`LassoModel` (scikit-learn compatible estimator), `CurrentFactorCovarData`
(single-date covariance decomposition Σ_y = β Σ_x β' + D), and
`RollingFactorCovarData` (time-indexed collection). It knows nothing about
finance, asset returns, frequencies, or rebalancing schedules.

`optimalportfolios` adds two finance-specific layers on top:

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

## **Alpha signals module**

**New in v4.1.1.** The `alphas` module provides standalone alpha signal
computation functions with a consistent interface. Each function handles
single-frequency and mixed-frequency universes, supports within-group
cross-sectional scoring, and returns both a dimensionless score and the
raw signal for diagnostics.

### Naming convention

| Stage | What it is | Example |
| --- | --- | --- |
| **Raw signal** | Observable quantity with units | Cumulative return, EWMA beta, regression residual |
| **Score** | Cross-sectional z-score, dimensionless | Momentum rank, negated beta rank |
| **Alpha** | Portfolio-ready signal after CDF mapping | Combined score mapped to [-1, 1] |

Pipeline: **raw signal → score → alpha**.

### Available signals

**Momentum** (`compute_momentum_alpha`) — EWMA-filtered risk-adjusted excess returns relative to a benchmark, converted to cross-sectional scores.

```python
from optimalportfolios.alphas import compute_momentum_alpha

score, raw_momentum = compute_momentum_alpha(
    prices=prices, benchmark_price=benchmark, returns_freq='ME',
    group_data=asset_class_groups, long_span=12)
```

**Low Beta** (`compute_low_beta_alpha`) — EWMA regression beta to benchmark, negated and cross-sectionally scored ("betting against beta").

```python
from optimalportfolios.alphas import compute_low_beta_alpha

score, raw_beta = compute_low_beta_alpha(
    prices=prices, benchmark_price=benchmark, returns_freq='ME',
    group_data=asset_class_groups, beta_span=12)
```

**Managers Alpha** (`compute_managers_alpha`) — factor model regression residuals using pre-estimated betas from `FactorCovarEstimator`, EWMA-smoothed and cross-sectionally scored.

```python
from optimalportfolios.alphas import compute_managers_alpha

score, raw_alpha = compute_managers_alpha(
    prices=asset_prices, risk_factor_prices=factor_prices,
    estimated_betas=rolling_data.get_y_betas(),
    returns_freq='ME', alpha_span=12)
```

### Mixed-frequency support

All signal functions accept `returns_freq` as a string (uniform) or a `pd.Series` (per-asset frequency). When mixed, the function groups by frequency, computes per group, and merges.

```python
# equities monthly, alternatives quarterly
returns_freq = pd.Series({'SPY': 'ME', 'EZU': 'ME', 'HF_Macro': 'QE', 'PE': 'QE'})
score, raw = compute_momentum_alpha(prices, returns_freq=returns_freq, ...)
```

### AlphasData container

`AlphasData` holds the combined alpha scores and all intermediate components:

```python
from optimalportfolios.alphas import AlphasData

data = AlphasData(alpha_scores=combined, momentum_score=mom, beta_score=beta, ...)
snapshot = data.get_alphas_snapshot(date=pd.Timestamp('2024-12-31'))
```

See the [alphas module README](optimalportfolios/docs/alphas_module_readme.md) for full documentation.

# Table of contents

1. [Why optimalportfolios](#why-optimalportfolios)
2. [Package overview](#package-overview)
3. [Alpha signals module](#alpha-signals-module)
4. [Installation](#installation)
5. [Portfolio Optimisers](#portfolio-optimisers)
   1. [Implementation structure](#1-implementation-structure)
   2. [Example of implementation for Maximum Diversification Solver](#2-example-of-implementation-for-maximum-diversification-solver)
   3. [Constraints](#3-constraints)
   4. [Wrapper for implemented rolling portfolios](#4-wrapper-for-implemented-rolling-portfolios)
   5. [Adding an optimiser](#5-adding-an-optimiser)
   6. [Default parameters](#6-default-parameters)
   7. [Price time series data](#7-price-time-series-data)
   8. [Drift-aware rolling backtests (v5.3.1)](#8-drift-aware-rolling-backtests-v531)
6. [Examples](#examples)
7. [Updates](#updates)
8. [Disclaimer](#disclaimer)

## **Installation**

install using

```
pip install optimalportfolios
```

upgrade using

```
pip install --upgrade optimalportfolios
```

clone using

```
git clone https://github.com/ArturSepp/OptimalPortfolios.git
```

Core dependencies:
python = ">=3.9",
numba = ">=0.60.0",
numpy = ">=2.0",
scipy = ">=1.12.0",
pandas = ">=2.2.0",
matplotlib = ">=3.8.0",
seaborn = ">=0.13.0",
cvxpy = ">=1.3.0",
quadprog = ">=0.1.11",
qis = ">=3.5.7",
factorlasso = ">=0.1.0"

Optional dependencies:
yfinance ">=0.2.40" (for getting test price data),
pybloqs ">=1.2.13" (for producing html and pdf factsheets)

## **Portfolio optimisers**

### 1. Implementation structure

The implementation of each solver is split into 3 layers:

1. **Mathematical layer** which takes clean inputs, formulates the optimisation
   problem and solves it using Scipy or CVXPY solvers.
   The logic of this layer is to solve the problem algorithmically by taking clean inputs.
2. **Wrapper layer** which takes inputs potentially containing NaNs,
   filters them out, and calls the solver in layer 1). The output weights of filtered out
   assets are set to zero. Includes rebalancing indicator support for freezing
   specific assets at their previous weights, and (as of v5.3.1) automatic
   relaxation of group bounds when frozen-position drift causes overshoot.
3. **Rolling layer** which takes price time series as inputs and implements
   the estimation of covariance matrix and other inputs on a roll-forward basis.
   For each update date the rolling layer calls the wrapper layer 2) with estimated
   inputs as of the update date. As of v5.3.1, the rolling layer also drifts
   `weights_0` between rebalances using realised price returns, so that turnover
   constraints and transaction-cost penalties measure actual trades rather than
   notional trades against a stale baseline.

For rolling level function, the estimated covariance matrix can be passed as `Dict[pd.Timestamp, pd.DataFrame]`
with DataFrames containing covariance matrices for the universe and with keys being rebalancing times.

Covariance can be estimated using `EwmaCovarEstimator` (simple EWMA) or
`FactorCovarEstimator` (HCGL factor model using
[`factorlasso.LassoModel`](https://github.com/ArturSepp/factorlasso) for sparse
beta estimation, with finance-specific annualisation, multi-frequency returns,
and rolling schedule management).

**Important design principle (v4.1.1):** covariance estimation is separated from
portfolio optimisation. The recommended workflow is to estimate covariance
matrices first, then pass them as `covar_dict` to any solver:

```python
from optimalportfolios import EwmaCovarEstimator, FactorCovarEstimator

# estimate once
estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)

# reuse across multiple solvers
weights_rb = rolling_risk_budgeting(prices=prices, covar_dict=covar_dict, ...)
weights_md = rolling_maximise_diversification(prices=prices, covar_dict=covar_dict, ...)
weights_te = rolling_maximise_alpha_over_tre(prices=prices, covar_dict=covar_dict, ...)
```

This separation provides three benefits: (1) the same covariance matrices can be
reused across multiple solvers without re-estimation, (2) covariance diagnostics
and reporting can be inspected independently of the optimiser, and (3) different
covariance estimators can be swapped in without modifying the solver code.
For the HCGL factor model, use `FactorCovarEstimator` with `asset_returns_dict`
for mixed-frequency universes (e.g., monthly equities + quarterly alternatives).

The recommended usage is as follows.

Layer 2) is used for live portfolios or for backtests which are implemented using
data augmentation.

Layer 3) is applied for roll forward backtests where all available data is processed
using roll forward analysis.

### 2. Example of implementation for Maximum Diversification Solver

Using example of `optimization.solvers.max_diversification.py`

1. Scipy solver `opt_maximise_diversification()` which takes "clean" inputs of the
   covariance matrix of type `np.ndarray` without NaNs and
   `Constraints` dataclass which implements constraints for the solver.

The lowest level of each optimisation method is `opt_...` or `cvx_...` function taking clean inputs and producing the optimal weights.

The logic of this layer is to implement pure quant logic for the optimiser with cvx solver.

2. Wrapper function `wrapper_maximise_diversification()` which takes inputs
   covariance matrix of type `pd.DataFrame`
   potentially containing NaNs or assets with zero variance (when their time series are missing in the
   estimation period) and filters out non-NaN "clean" inputs and
   updates constraints for OPT/CVX solver in layer 1.

The intermediary level of each optimisation method is `wrapper_...` function taking
"dirty" inputs, filtering inputs, and producing the optimal weights. This wrapper can be called either
by rolling backtest simulations or by live portfolios for rebalancing.

The logic of this layer is to filter out data and to be an interface for portfolio implementations.

3. Rolling optimiser function `rolling_maximise_diversification()` takes the time series of data
   and slices these accordingly and at each rebalancing step calls the wrapper in layer 2.
   In the end, the function outputs the time series of optimal weights of assets in the universe.
   Price data of assets may have gaps and NaNs which is taken care of in the wrapper level.

The backtesting of each optimisation method is implemented with `rolling_...` method which produces the time series of
optimal portfolio weights.

The logic of this layer is to facilitate the backtest of portfolio optimisation method and to produce
time series of portfolio weights using a Markovian setup. These weights are applied for the backtest
of the optimal portfolio and the underlying strategy.

Each module in `optimization.solvers` implements specific optimisers and estimators for their inputs.

### 3. Constraints

Dataclass `Constraints` in `optimization.constraints` implements
optimisation constraints in solver-independent way.

The following inputs for various constraints are implemented.

```python
@dataclass
class Constraints:
    is_long_only: bool = True  # for positive allocation weights
    min_weights: pd.Series = None  # instrument min weights
    max_weights: pd.Series = None  # instrument max weights
    max_exposure: float = 1.0  # for long short portfolios: for long_portfolios = 1
    min_exposure: float = 1.0  # for long short portfolios: for long_portfolios = 1
    benchmark_weights: pd.Series = None  # for minimisation of tracking error
    tracking_err_vol_constraint: float = None  # annualised sqrt tracking error
    weights_0: pd.Series = None  # for turnover constraints
    turnover_constraint: float = None  # for turnover constraints
    target_return: float = None  # for optimisation with target return
    asset_returns: pd.Series = None  # for optimisation with target return
    max_target_portfolio_vol_an: float = None  # for optimisation with maximum portfolio volatility target
    min_target_portfolio_vol_an: float = None  # for optimisation with maximum portfolio volatility target
    group_lower_upper_constraints: GroupLowerUpperConstraints = None  # for group allocations constraints
```

Dataclass `GroupLowerUpperConstraints` implements asset class loading and min and max allocations

```python
@dataclass
class GroupLowerUpperConstraints:
    """
    add constraints that each asset group is group_min_allocation <= sum group weights <= group_max_allocation
    """
    group_loadings: pd.DataFrame  # columns=instruments, index=groups, data=1 if instrument in indexed group else 0
    group_min_allocation: pd.Series  # index=groups, data=group min allocation
    group_max_allocation: pd.Series  # index=groups, data=group max allocation
```

Constraints are updated on the wrapper level to include the valid tickers

```python
    def update_with_valid_tickers(self,  valid_tickers: List[str]) -> Constraints:
```

On the solver layer, the constants for the solvers are requested as follows.

For Scipy: `set_scipy_constraints(self, covar: np.ndarray = None) -> List`

For CVXPY: `set_cvx_constraints(self, w: cvx.Variable, covar: np.ndarray = None) -> List`

**Frozen-position relaxation (new in v5.3.1).** When `rebalancing_indicators`
freeze illiquid positions for a given rebalance date, `update_with_valid_tickers`
pins their `min_weights` and `max_weights` to the current (drifted) `weights_0`.
If the resulting group-loading sum exceeds `group_max_allocation` (or falls below
`group_min_allocation`), the group bound is automatically relaxed for that
rebalance and a `UserWarning` is emitted. This prevents `ValueError: Infeasible
constraints detected` errors that would otherwise occur when illiquid sleeves
drift over their group cap between low-frequency rebalances. The relaxation is
audit-trailable: each event surfaces in logs with the group name, the original
bound, and the relaxed bound.

### 4. Wrapper for implemented rolling portfolios

Module `optimisation.wrapper_rolling_portfolios.py` wraps implementation of
of the following solvers enumerated in `config.py`

Using the wrapper function allows for cross-sectional analysis of different
backtest methods and for sensitivity analysis to parameters of
estimation and solver methods.

```python
class PortfolioObjective(Enum):
    """
    implemented portfolios in rolling_engine
    """
    # risk-based:
    MAX_DIVERSIFICATION = 1  # maximum diversification measure
    EQUAL_RISK_CONTRIBUTION = 2  # implementation in risk_parity
    MIN_VARIANCE = 3  # min w^t @ covar @ w
    # return-risk based
    QUADRATIC_UTILITY = 4  # max means^t*w- 0.5*gamma*w^t*covar*w
    MAXIMUM_SHARPE_RATIO = 5  # max means^t*w / sqrt(*w^t*covar*w)
    # return-skeweness based
    MAX_CARA_MIXTURE = 6  # carra for mixture distributions
```

See examples in the [examples folder](#examples) and the
[`examples/README.md`](optimalportfolios/docs/examples_readme.md) for the full
demo index.

### 5. Adding an optimiser

1. Add analytics for computing rolling weights using a new estimator in
   subpackage `optimization.solvers`. Any third-party packages can be used
2. For cross-sectional analysis, add new optimiser type
   to `config.py` and link implemented
   optimiser in wrapper function `compute_rolling_optimal_weights()` in
   `optimisation.wrapper_rolling_portfolios.py`

### 6. Default parameters

Key parameters include the specification of the estimation sample.

1. `returns_freq` defines the frequency of returns for covariance matrix estimation. This parameter affects all methods.

The default (assuming daily price data) is weekly Wednesday returns `returns_freq = 'W-WED'`.

For price data with monthly observations
(such as hedge funds), monthly returns should be used `returns_freq = 'ME'`.

2. `span` defines the estimation span for EWMA covariance matrix. This parameter affects all methods which use
   EWMA covariance matrix:

```
PortfolioObjective in [MAX_DIVERSIFICATION, EQUAL_RISK_CONTRIBUTION, MIN_VARIANCE]
```

and

```
PortfolioObjective in [QUADRATIC_UTILITY, MAXIMUM_SHARPE_RATIO]
```

The span is defined as the number of returns
for the half-life of EWMA filter: `ewma_lambda = 1 - 2 / (span+1)`. `span=52` with weekly returns means that
last 52 weekly returns (one year of data) contribute 50% of weight to estimated covariance matrix

The default (assuming weekly returns) is 52: `span=52`.

For monthly returns, I recommend to use `span=12` or `span=24`.

3. `rebalancing_freq` defines the frequency of weights update. This parameter affects all methods.

The default value is quarterly rebalancing `rebalancing_freq='QE'`.

For the following methods

```
PortfolioObjective in [QUADRATIC_UTILITY, MAXIMUM_SHARPE_RATIO, MAX_CARA_MIXTURE]
```

Rebalancing frequency is also the rolling sample update frequency when mean returns and mixture distributions are estimated.

4. `roll_window` defines the number of past returns applied for estimation of rolling mean returns and mixture distributions.

This parameter affects the following optimisers

```
PortfolioObjective in [QUADRATIC_UTILITY, MAXIMUM_SHARPE_RATIO, MAX_CARA_MIXTURE]
```

and it is linked to `rebalancing_freq`.

Default value is `roll_window=20` which means that data for past 20 (quarters) are used in the sample
with `rebalancing_freq='QE'`

For monthly rebalancing, I recommend to use `roll_window=60` which corresponds to using past 5 years of data

### 7. Price time series data

The input to all optimisers is dataframe prices which contains dividend and split adjusted prices.

The price data can include assets with prices starting and ending at different times.

All optimisers will set maximum weight to zero for assets with missing prices in the estimation sample period.

### 8. Drift-aware rolling backtests (v5.3.1)

Every rolling optimiser carries `weights_0` forward from one rebalance date to
the next so that turnover constraints (`Constraints.turnover_constraint`,
`turnover_costs`, `turnover_utility_weight`, `group_turnover_constraint`) act
on a sensible baseline. The choice of baseline matters.

**Legacy behaviour (pre-v5.3.1, also `use_drifted_weights_0=False`).** `weights_0`
at each rebalance equals the previous-period *target* weights, with no adjustment
for realised drift over the holding period. The optimiser's L1 turnover budget
constrains `||w_new − w_prev_target||_1`, but the simulator actually trades
`||w_new − w_drift||_1`. The two differ by the realised one-period drift, which
is typically 1–3% of NAV for diversified portfolios at quarterly frequency.
Cumulative effect: realised turnover exceeds the optimiser's budget by
roughly the same fraction.

**New default (v5.3.1, `use_drifted_weights_0=True`).** Before each rebalance,
the helper `apply_drift_to_weights_0` (in `utils/weights_drift.py`) drifts
the previous-period target weights to the current date using realised price
returns under the self-financing identity

```
w_drift_i = w_i · (1 + r_i) / (1 + Σ_j w_j · r_j)
```

The denominator is portfolio NAV growth. For long-only fully-invested portfolios
this reduces to the conventional `gross / sum(gross)` form, but the formula
remains correct for long-short and variable-exposure mandates. The helper is
constraint-agnostic and silently falls back to passing `weights_0` unchanged
whenever any input is missing or pathological (NaN prices, NAV collapse, zero
weights_0, first rebalance, etc.).

**Empirical comparison.** On a min-variance rolling backtest of the
15-ETF benchmark universe with a binding L1 turnover budget of 0.08/quarter:

| | Policy A (legacy, drift off) | Policy B (new default, drift on) |
| --- | --- | --- |
| Apparent turnover (ann.) | 0.2767 | 0.2766 |
| **Realised turnover (ann.)** | **0.3403** | **0.2814** |
| Realised / apparent | 1.23 | 1.02 |
| Cumulative TC drag (bps) | 19.4 | 16.0 |

Under (A), the optimiser believes it's hitting the 0.08/quarter cap but is
actually trading 0.085/quarter — the budget is **leaky**. Under (B), realised
trading sits at 0.070/quarter, comfortably under the cap. Cost drag drops by
17.5% relative for the same nominal constraint. See
[`examples/comparisons/drift_policy.py`](optimalportfolios/examples/comparisons/drift_policy.py)
for the reproducible demonstration.

**Toggling for legacy comparisons.** To reproduce pre-v5.3.1 behaviour
exactly (e.g. to validate against published backtest numbers from earlier
papers or reports), set:

```python
from optimalportfolios import OptimiserConfig

cfg = OptimiserConfig(use_drifted_weights_0=False)
weights = rolling_quadratic_optimisation(prices=prices, covar_dict=covar_dict,
                                          constraints=constraints,
                                          optimiser_config=cfg)
```

## **Examples**

The `examples/` folder is organised into five purpose-folders. The
[examples README](optimalportfolios/docs/examples_readme.md) maps every demo to its
role; the headlines are:

```
examples/
├── data/                  Universe fixtures (fetch_benchmark_universe_data, fetch_minimal_universe_data)
├── solvers/               One demo per single-objective solver
├── backtests/             End-to-end rolling backtest workflows
├── comparisons/           A-vs-B sweeps (covar / optimiser / parameter / drift policy)
├── covar_estimation/      Covariance estimator demos
└── sp500_universe.py      S&P 500 universe loader (top level)
```

### Recommended reading order for newcomers

1. [`examples/data/universe.py`](optimalportfolios/examples/data/universe.py) — understand the shared fixture.
2. [`examples/backtests/minimal_backtest.py`](optimalportfolios/examples/backtests/minimal_backtest.py) — see one full workflow end-to-end.
3. [`examples/solvers/min_variance.py`](optimalportfolios/examples/solvers/min_variance.py) — minimal solver demo with both single-date and rolling forms.
4. [`examples/solvers/tracking_error.py`](optimalportfolios/examples/solvers/tracking_error.py) — the production TAA pattern (alpha + benchmark + TE constraint).
5. [`examples/comparisons/optimisers.py`](optimalportfolios/examples/comparisons/optimisers.py) — see how objectives differ on the same universe.

### Highlighted demos

#### Optimal portfolio backtest

See script [`optimalportfolios/examples/backtests/minimal_backtest.py`](optimalportfolios/examples/backtests/minimal_backtest.py).

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis

from optimalportfolios import (compute_rolling_optimal_weights, PortfolioObjective,
                               Constraints, EwmaCovarEstimator)
from optimalportfolios.examples.data.universe import fetch_minimal_universe_data


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

[![image info](optimalportfolios/examples/figures/example_portfolio_factsheet1.PNG)](optimalportfolios/examples/figures/example_portfolio_factsheet1.PNG)
[![image info](optimalportfolios/examples/figures/example_portfolio_factsheet2.PNG)](optimalportfolios/examples/figures/example_portfolio_factsheet2.PNG)

#### Customised reporting

Portfolio data class `PortfolioData` is implemented in
[QIS package](https://github.com/ArturSepp/QuantInvestStrats).

```python
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

[![image info](optimalportfolios/examples/figures/example_customised_report.PNG)](optimalportfolios/examples/figures/example_customised_report.PNG)

#### Parameter sensitivity backtest

Cross-sectional backtests test the sensitivity of an optimisation method to
estimation or solver parameters.

See [`optimalportfolios/examples/comparisons/parameter_sensitivity.py`](optimalportfolios/examples/comparisons/parameter_sensitivity.py).

[![image info](optimalportfolios/examples/figures/max_diversification_span.PNG)](optimalportfolios/examples/figures/max_diversification_span.PNG)

#### Multi-optimiser cross-backtest

Multiple optimisation methods can be analysed using
`compute_rolling_optimal_weights()`.

See [`optimalportfolios/examples/comparisons/optimisers.py`](optimalportfolios/examples/comparisons/optimisers.py).

[![image info](optimalportfolios/examples/figures/multi_optimisers_backtest.PNG)](optimalportfolios/examples/figures/multi_optimisers_backtest.PNG)

#### Multi-covariance-estimator backtest

Multiple covariance estimators can be backtested for the same optimisation method.

See [`optimalportfolios/examples/comparisons/covar_estimators.py`](optimalportfolios/examples/comparisons/covar_estimators.py).

[![image info](optimalportfolios/examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)](optimalportfolios/examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)

#### Drift-policy comparison (new in v5.3.1)

Compares `OptimiserConfig.use_drifted_weights_0 = True` (production default)
vs `False` (legacy) using `rolling_quadratic_optimisation` with a binding L1
turnover budget. Shows that under the legacy convention the realised turnover
exceeds the optimiser's apparent turnover by ~23%; under the new default the
two agree.

See [`optimalportfolios/examples/comparisons/drift_policy.py`](optimalportfolios/examples/comparisons/drift_policy.py).

#### Optimal allocation to cryptocurrencies

Computations and visualisations for the paper "Optimal Allocation to
Cryptocurrencies in Diversified Portfolios" are implemented in
`optimalportfolios.examples.crypto_allocation`. See the
[README in that module](optimalportfolios/examples/crypto_allocation/README.md).

Published reference: Sepp A. (2023), "Optimal Allocation to Cryptocurrencies in
Diversified Portfolios", *Risk Magazine*, October 2023, 1-6. Available at
[SSRN](https://ssrn.com/abstract=4217841).

#### Robust optimisation of strategic and tactical asset allocation

Computations and visualisations for the paper "Robust Optimization of Strategic
and Tactical Asset Allocation for Multi-Asset Portfolios" are implemented in
`optimalportfolios.examples.robust_optimisation_saa_taa`. See the
[README in that module](optimalportfolios/examples/robust_optimisation_saa_taa/README.md).

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

## **Updates**

#### May 2026, Version 5.3.1 released

**Drift-aware `weights_0` in rolling backtests (default behaviour change).**

Every rolling optimiser now drifts the previous-period weights to the current
rebalance date using realised price returns before passing them as `weights_0`
to the next single-date optimisation. The new helper
`apply_drift_to_weights_0` in `utils/weights_drift.py` implements the
self-financing identity `w_drift_i = w_i · (1 + r_i) / (1 + Σ_j w_j · r_j)`,
which is correct for long-only, long-short, and variable-exposure mandates.

Controlled by `OptimiserConfig.use_drifted_weights_0` — default `True`.
Set to `False` to reproduce pre-v5.3.1 behaviour for legacy comparisons.
The change affects all nine rolling optimisers: `rolling_risk_budgeting`,
`rolling_maximize_portfolio_sharpe`, `rolling_maximize_cara_mixture`,
`rolling_maximise_diversification`, `rolling_quadratic_optimisation`,
`rolling_max_return_target_vol`, `rolling_min_variance_target_return`,
`rolling_maximise_alpha_with_target_return`, `rolling_maximise_alpha_over_tre`.

**Impact:** for backtests with a binding turnover constraint or non-zero
transaction-cost penalty, realised turnover and TC drag will differ from
pre-v5.3.1 numbers. The optimiser now constrains `||w_new − w_drift||_1`
rather than `||w_new − w_prev_target||_1`, which matches what the NAV
simulator actually trades. On a min-variance / L1 0.08-per-quarter
backtest: legacy realised/apparent turnover ratio 1.23, new default 1.02;
cumulative TC drag drops from 19.4 bps to 16.0 bps (17.5% relative reduction).

For backtests without a turnover-related constraint or penalty, `weights_0`
only affects the CVXPY warm-start and the SciPy convergence path. Numerical
differences may exist but are typically below 1 bp/year on Sharpe.

**Frozen-position overshoot relaxation in `Constraints.update_with_valid_tickers`.**

When `rebalancing_indicators` freeze illiquid positions (PE, HF, CAT bonds,
private credit) over multiple TAA rebalance dates, the frozen positions can
drift above their group's `group_max_allocation`. Previously
`Constraints.__post_init__` raised `ValueError: Infeasible constraints
detected`. The new behaviour automatically relaxes the offending bound by
the overshoot amount with an audit-trail `UserWarning`, treating the
rebalance as a one-period compliance waiver — the optimiser can no longer
trade frozen assets, and the relaxed cap prevents tradable members from
adding more on top of the inherited overhang.

This change is independent of the drift policy: it also helps when
`weights_0` comes from a live PMS that is slightly out of compliance due to
intra-period flows, corporate actions, or settlement gaps. Each relaxation
event surfaces in logs with the group name, original bound, and relaxed
bound.

**Examples folder reorganised.**

The flat `examples/` layout has been replaced with five purpose-folders:

| Old path | New path |
| --- | --- |
| `examples.universe` | `examples.data.universe` |
| `examples.optimal_portfolio_backtest` | `examples.backtests.minimal_backtest` |
| `examples.solve_risk_budgets_balanced_portfolio` | `examples.backtests.balanced_risk_budgets` |
| `examples.computation_of_tracking_error` | `examples.backtests.tracking_error_decomposition` |
| `examples.multi_optimisers_backtest` | `examples.comparisons.optimisers` |
| `examples.multi_covar_estimation_backtest` | `examples.comparisons.covar_estimators` |
| `examples.parameter_sensitivity_backtest` | `examples.comparisons.parameter_sensitivity` |
| `examples.risk_budgeting_pyrb_vs_scipy` | `examples.comparisons.pyrb_vs_scipy` |
| `examples.sp500_minvar` | `examples.comparisons.sp500_minvar_spans` |
| `examples.long_short_optimisation` | `examples.solvers.long_short` |
| `examples.sp500_universe` | unchanged (kept at top level) |

The new layout adds an [`examples/README.md`](optimalportfolios/docs/examples_readme.md)
indexing every demo. Six wrong docstrings in `solvers/` corrected
(carra_mixture, max_diversification, max_sharpe, min_variance, risk_budgeting,
tracking_error — all were boilerplate copies of "example of minimization of
tracking error" regardless of the file's contents). Two helpers in
`data/universe.py`: `fetch_benchmark_universe_data()` (15-ETF universe,
6-tuple return) and `fetch_minimal_universe_data()` (8-ETF universe, 3-tuple
return) replace the inline loaders previously duplicated across
`minimal_backtest.py` and `long_short.py`.

**Migration from v5.0.x:**

* If you have notebooks or scripts referencing the old `examples.*` paths,
  see the table above. The package public API (everything under
  `from optimalportfolios import ...`) is unchanged.
* If your backtests rely on the legacy weights_0 behaviour for reproducibility
  (e.g. validating against published numbers), pass
  `OptimiserConfig(use_drifted_weights_0=False)`.
* If you previously caught `ValueError: Infeasible constraints detected`
  from a long-running backtest of illiquid universes, those backtests will
  now run to completion with `UserWarning` messages instead. Consider
  capturing the warnings at the runner level and emitting a summary line
  rather than per-event logs.

#### March 2026, Version 5.0.4 released

**Removed `scikit-learn` dependency.**
The Gaussian mixture model in `utils/gaussian_mixture.py` previously used
`sklearn.mixture.GaussianMixture`. This has been replaced with a pure
numpy/scipy EM implementation (`fit_gmm`) using `scipy.stats.multivariate_normal`
for the E-step and `scipy.cluster.vq.kmeans2` for K-means initialisation.
The public API (`fit_gaussian_mixture`, `Params`, `plot_mixure1`, `plot_mixure2`,
`estimate_rolling_mixture`) is unchanged.

This removes the last `scikit-learn` import from `optimalportfolios`, eliminating
the transitive dependency on `joblib`, `threadpoolctl`, and the scikit-learn
binary itself — a meaningful reduction in install footprint.

#### March 2026, Version 5.0.0 released

**LASSO estimator extracted to [`factorlasso`](https://github.com/ArturSepp/factorlasso) package.**
The `lasso/` module has been removed from `optimalportfolios`. The LASSO/Group
LASSO/HCGL solver is now in the standalone `factorlasso` package — a
domain-agnostic sparse factor model estimator with sign constraints,
prior-centered regularisation, NaN-aware estimation, and scikit-learn
compatible API (`fit` / `predict` / `score` / `coef_` / `intercept_`).
`factorlasso` is a required dependency of `optimalportfolios` v5.0.0.
All existing imports (`from optimalportfolios import LassoModel`) continue
to work via re-exports.

**License changed from GPL-3.0 to MIT.**

**Dependencies cleaned:**

* Removed `easydev`, `pyarrow`, `fsspec`, `statsmodels`, `ecos` (unused)
* `yfinance`, `pandas-datareader` moved to `[data]` optional
* `numpy` unpinned from `==2.2.6` to `>=2.0`
* Build system simplified (removed unused `poetry-core`, `hatchling`)
* Dev tooling: `black`/`flake8`/`isort`/`mypy` replaced with `ruff`

**CI added:** GitHub Actions test pipeline across Python 3.10–3.12.

**Migration from v4.x:** No code changes required. All existing imports
(`from optimalportfolios import LassoModel, LassoModelType`) continue to work
via re-exports from `factorlasso`. The only exception: if your code imports
directly from the deleted module path
(`from optimalportfolios.lasso.lasso_estimator import ...`), change to
`from optimalportfolios import ...`.

#### March 2026, Version 4.1.1 released

**Alpha signals module** (`optimalportfolios.alphas`):

* New `alphas/` package with three standalone signal functions: `compute_momentum_alpha`, `compute_low_beta_alpha`, `compute_managers_alpha`
* Each function handles single-frequency and mixed-frequency universes via `returns_freq` (string or per-asset `pd.Series`)
* Within-group cross-sectional scoring via `group_data` parameter
* `AlphasData` container moved from `utils/manager_alphas.py` to `alphas/alpha_data.py`
* `backtest_alphas.py` moved from `reports/` to `alphas/` with fixed function names (typo corrections: `backtest_alpha_signas` → `backtest_alpha_signals`, etc.)
* Comprehensive test suite in `alphas/tests/signals_test.py`

**Deprecated and removed:**

* `utils/factor_alphas.py` — all functions migrated to `alphas/signals/`. The 9-function variant explosion (3 signal types × 3 frequency variants) is replaced by 3 functions, each handling all dispatch modes internally
* `utils/manager_alphas.py` — `AlphasData` moved to `alphas/alpha_data.py`. `compute_joint_alphas()` is replaced by external aggregation (see migration guide below)
* `reports/backtest_alphas.py` — moved to `alphas/backtest_alphas.py`

**Risk budgeting fixes:**

* Fixed `total_to_good_ratio` computation in `wrapper_risk_budgeting`: previously used `len(pd_covar.columns) / len(clean_covar.columns)` which over-inflated budgets when zero-budget and NaN assets coexisted. Now uses `n_eligible / n_valid` where `n_eligible` counts assets with positive risk budget
* Replaced all `print()` fallback messages with `warnings.warn()` for proper logging
* Removed unused `FactorCovarEstimator` import

**Solver docstrings:**

* Full docstrings added to all optimisation solvers (quadratic, risk_budgeting, max_diversification, max_sharpe, tracking_error, target_return, cara_mixture)
* Full docstrings for the rolling portfolio dispatcher

**Covariance estimation separation:**

* Covariance estimation is now clearly separated from portfolio optimisation. The recommended workflow is to estimate covariance matrices upfront using `EwmaCovarEstimator` or `FactorCovarEstimator`, then pass the resulting `covar_dict` to any solver. This enables reusing the same covariance across multiple solvers, inspecting covariance diagnostics independently, and swapping estimators without modifying solver code.

#### 05 January 2025, Version 3.1.1 released

Added Lasso estimator and Group Lasso estimator using cvxpy quadratic problems.

Added covariance estimator using factor model with Lasso betas.

Estimated covariance matrices can be passed to rolling solvers, CovarEstimator type is added for different covariance estimators.

Risk budgeting is implemented using pyrb package with pyrb forked for optimalportfolios package.

#### 18 August 2024, Version 2.1.1 released

Refactor the implementation of solvers with the 3 layers.

Add new solvers for tracking error and target return optimisations.

Add examples of running all solvers.

#### 2 September 2023, Version 1.0.8 released

Added subpackage `optimisation.rolling_engine` with optimisers grouped by the type of inputs and
data they require.

#### 8 July 2023, Version 1.0.1 released

Implementation of optimisation methods and data considered in
"Optimal Allocation to Cryptocurrencies in
Diversified Portfolios" by A. Sepp published in Risk Magazine, October 2023, 1-6. The draft is available at SSRN: <https://ssrn.com/abstract=4217841>

## **Disclaimer**

OptimalPortfolios package is distributed FREE & WITHOUT ANY WARRANTY under the MIT License.

See the [LICENSE.txt](LICENSE.txt) in the release for details.

Please report any bugs or suggestions by opening an [issue](https://github.com/ArturSepp/OptimalPortfolios/issues).

## **References**

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

## BibTeX Citations for optimalportfolios Package

If you use optimalportfolios in your research, please cite it as:

```
@software{sepp2024optimalportfolios,
  author={Sepp, Artur},
  title={OptimalPortfolios: Implementation of optimisation analytics for constructing and backtesting optimal portfolios in Python},
  year={2024},
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
