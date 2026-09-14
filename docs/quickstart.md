---
myst:
  html_meta:
    description: >-
      Run the offline OptimalPortfolios quickstart: monthly EWMA covariance, constrained
      minimum-variance weights and a QIS backtest with explicit trade timing and costs.
---

# Quickstart

*[author / affiliation / date — placeholder]*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

The production quickstart demonstrates a complete portfolio workflow: load the packaged monthly
test fixture, estimate covariance, compute constrained minimum-variance weights, and simulate
holdings and costs through [QIS](https://github.com/ArturSepp/QuantInvestStrats).
It uses the core package and prints a compact result. The fixed sample makes the workflow
repeatable without a data download or credentials.

## Run the example

Use the interpreter selected in the [installation guide](installation.md). For this repository's
Windows/OneDrive checkout, first follow its external-environment and generated-state setup.

```console
python -m pip install optimalportfolios
python examples/getting_started/production_quickstart.py
```

The script path above is relative to a repository checkout. A wheel installation includes the
fixture but excludes the repository's `examples/` directory. To run outside a checkout,
save a copy of the [canonical Python script](../examples/getting_started/production_quickstart.py)
and run that saved file with the same interpreter. No `data` extra is needed for this example.
The script writes no files; it prints input dates, weight dimensions, last weights, final NAV
and measured runtime.

For a hosted trial, [open the same workflow in Colab](https://colab.research.google.com/github/ArturSepp/OptimalPortfolios/blob/main/examples/getting_started/production_quickstart.ipynb).
The [notebook source](../examples/getting_started/production_quickstart.ipynb) installs the latest
released package from PyPI and displays its version. That initial setup needs network access;
the analytical workflow then uses the packaged fixture. The notebook carries no saved outputs.
Its tagged Python cell is mechanically checked against the canonical script.

A release or Colab installation can differ from the working source used to review this page.
The [installation guide](installation.md#current-development-verification-boundary) records the
current lockfile reconciliation limitation.

## Inputs and portfolio decisions

The six assets are Global Bonds, US Treasuries, Global HY Bonds, MSCI World USD, Hedge Funds,
and Commodities EX-Precious. The [fixture loader](../src/optimalportfolios/tests/data/multiasset.py)
builds a total-return price panel from monthly simple returns in decimal units. The covariance
estimator converts those prices to monthly **log returns**.

| Setting | Quickstart value | Meaning |
|---|---|---|
| Price sample | 2010-01-31 to 2022-12-31 | 156 monthly price observations for six assets. |
| Decision period | 2015-03-31 to 2022-09-30 | 31 quarter-end covariance dates and target-weight rows. |
| Return frequency | `ME` | Monthly observations used for covariance estimation. |
| Covariance span | `24` | EWMA decay in monthly observations; earlier history supplies warmup. |
| Decision frequency | `QE` | Quarter-end labels selected from the covariance path. |
| Objective | `PortfolioObjective.MIN_VARIANCE` | Minimize annualized variance for the selected universe. |
| Target exposure | `1.0` | Fully invested target weights, using the default equality budget. |
| Asset bounds | `0.0` to `0.35` | Long-only targets with a 35% cap for each asset. |
| Implementation lag | `1` | Trade one observation of the supplied monthly price index after each decision. |
| Trading cost rate | `0.001` | 10 basis points on each asset's absolute traded notional, including entry. |
| Initial NAV | `100.0` | QIS default portfolio value before the opening trade. |

At decision date $t$, let $w_i$ be asset $i$'s target fraction of portfolio value and
$\Sigma_t$ the estimated annual covariance matrix of decimal log returns. The problem is

$$
\begin{aligned}
\min_w \quad & w^\mathsf{T}\Sigma_t w \\
\text{subject to}\quad & \sum_i w_i = 1, \\
& 0 \leq w_i \leq 0.35.
\end{aligned}
$$

No expected-return forecast is required for this objective. The [constraints contract](constraints.md)
and [solver guide](optimization_module_readme.md) describe the broader construction interfaces.

The estimator's default demeaning subtracts a contemporaneous EWMA mean seeded from the first
return. It then computes the EWMA second moment of those demeaned returns and annualizes the
monthly covariance by 12. With span 24, the decay is $\lambda = 1 - 2/(24+1) = 0.92$.
Span is neither a fixed lookback nor a half-life: earlier observations retain declining weight.
The price history before the first decision provides more than five years of warmup.

For this complete monthly panel and the default unnormalized-return path, estimates through a
decision date agree with estimates recomputed from prices cut off at that date. Changing later
prices also leaves earlier estimates unchanged. This software timing property assumes that each
input observation was available on its label date; a publication delay requires its own data policy.
See [covariance estimation](covariance_estimators.md) for other modes and their limitations.

## Canonical Python implementation

The [Python source](../examples/getting_started/production_quickstart.py) is authoritative.
Sphinx includes that file below; ordinary Markdown readers can open the same source link.
The [notebook parity check](../.github/scripts/check_quickstart_notebook.py) verifies its Colab mirror.

```{literalinclude} ../examples/getting_started/production_quickstart.py
:language: python
:linenos:
```

The estimator supplies a dictionary of labeled covariance matrices. OptimalPortfolios returns
a DataFrame of target weights. QIS owns holdings simulation, transaction costs and NAV calculation;
the example does not implement a second backtester.

## Trade timing and costs

A decision date, execution date and holding return are different timestamps. Lag `1` means one
row of the supplied price index, so it means one month for this fixture. It does not mean one
calendar day and does not move the price observations.

| Event | Observation date | Result |
|---|---|---|
| First decision | `2015-03-31` | Covariance and target weights use information through this date. |
| First execution | `2015-04-30` | Targets buy units at April's price; opening costs are charged. |
| First invested return endpoint | `2015-05-31` | Holdings earn the April-to-May price movement. |
| Last decision | `2022-09-30` | Final target-weight row. |
| Last execution | `2022-10-31` | Final targets enter at October's price. |
| End of sample | `2022-12-31` | The final holdings remain invested through this observation. |

NAV starts at 100 on 31 March 2015 with no units held. With no funding rate supplied, the
initial cash earns no interest. The fully invested opening trade costs 0.1000 NAV units and
leaves NAV at 99.9000 on 30 April; it earns none of the March-to-April asset return.

Between executions, QIS holds units and lets portfolio weights drift with prices. The 35% cap
applies to optimized target weights; drift and the deduction of trading costs can move measured
holdings weights away from those targets. Costs apply to buys and sells at execution prices.
A full switch between disjoint fully invested portfolios would trade approximately twice NAV.

No funding rate, management fee, extra carry, taxes or market-impact model is supplied by this
script. Its final NAV includes the specified proportional trading costs. See
[rolling backtests](rolling_backtests.md) and [turnover and transaction costs](turnover_and_transaction_costs.md)
for the complete conventions.

## Read the result

The 2026-09-14 review used Python 3.12.14, OptimalPortfolios 7.6.0 working source, QIS 5.26.0
and FactorLasso 0.18.0. The unchanged script printed the following, with elapsed runtime omitted:

```text
Price history: 2010-01-31 to 2022-12-31
Rolling weights: 31 dates x 6 assets
Last rebalance: 2022-09-30
Global Bonds               0.30
US Treasuries              0.35
Global HY Bonds            0.00
MSCI World USD             0.00
Hedge Funds                0.35
Commodities EX-Precious    0.00
Final NAV after 10 bp transaction costs: 105.0409
```

Weights are rounded to four decimals before printing, so a displayed zero can represent a small
positive allocation. In this review, all 31 single-date solves were accepted and compliant with
no fallback. The dispatcher itself returns weights without a per-date diagnostic history;
a finite result alone does not certify every future configuration. Use the
[solver outcome guidance](optimization_module_readme.md#three-layer-solver-pattern) when diagnostic
records are needed.

The final NAV summarizes this fixed software example. It is not an annualized return or a live
analytics update. Runtime and last digits can differ across dependency versions and solver
platforms. The fixture loader still marks its source attribution as TODO; resolve that provenance
before using the panel as evidence in an empirical publication.

## What to change first

Choose the change to the question being modeled, then supply the inputs its implementation uses.

| Change | Required follow-through |
|---|---|
| Objective | Verify the dispatcher's route, parameters and the chosen backend's constraint support; see the cases below. |
| Constraints | Keep labels aligned with the asset panel and check feasibility; a common Constraints object does not make every field active in every backend. |
| Covariance estimator | Supply factor prices, asset-return panels and a configured FactorLasso model for the factor-estimator interface. |
| Decision cadence | Regenerate the covariance dictionary with the desired `rebalancing_freq`, such as `YE`, and retain an appropriate execution lag. |
| Price frequency | Reconsider return sampling, span, annualization and the meaning of a one-observation implementation lag together. |

The [rolling dispatcher](../src/optimalportfolios/optimization/wrapper_rolling_portfolios.py)
supports six `PortfolioObjective` members:

- `MIN_VARIANCE` and `MAX_DIVERSIFICATION` consume the covariance dictionary and constraints.
  Diversification uses a different backend, so its supported constraints still need review.
- `EQUAL_RISK_CONTRIBUTION` also accepts `risk_budget`. Omitting it requests equal budgets;
  specify and align it when a different risk allocation is intended.
- `QUADRATIC_UTILITY` and `MAXIMUM_SHARPE_RATIO` estimate expected returns internally.
  Set `returns_freq="ME"` and the intended mean-estimation `span` for this monthly panel,
  instead of silently inheriting the dispatcher's weekly return-sampling default.
  Quadratic utility additionally uses `carra` for risk aversion.
- `MAX_CARA_MIXTURE` fits rolling return mixtures using `time_period`, `returns_freq`,
  `rebalancing_freq`, `roll_window`, `carra` and the public spelling `n_mixures`.
  Its route does not use the supplied covariance dictionary as its risk estimate.
  Check its estimation sample and mixture settings explicitly.

For covariance-driven routes, the dictionary's dates determine when weights are computed.
Changing only the dispatcher's `rebalancing_freq` does not rebuild that dictionary.

`FactorCovarEstimator.fit_rolling_covars` accepts `risk_factor_prices`,
`asset_returns_dict` and `time_period`, rather than the quickstart's `prices` call.
Configure the `lasso_model` and compatible return frequencies for the intended factor method;
HCGL is an explicit model choice. Generic factor estimation belongs to
[FactorLasso](https://github.com/ArturSepp/FactorLasso).
Follow the [factor covariance guide](covariance_estimators.md) for a complete input example.

For tracking-error or alpha-over-tracking-error construction, use the dedicated interfaces
documented in the solver guide; they are not additional members of this six-value enum.

## References and next steps

- [Installation](installation.md), [examples](examples_readme.md), and [documentation standard](documentation_standard.md).
- [Canonical Python script](../examples/getting_started/production_quickstart.py) and [Colab notebook](../examples/getting_started/production_quickstart.ipynb).
- [EWMA estimator source](../src/optimalportfolios/covar_estimation/ewma_covar_estimator.py).
- [Factor estimator source](../src/optimalportfolios/covar_estimation/factor_covar_estimator.py).
- [QIS backtester source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/backtester.py).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
