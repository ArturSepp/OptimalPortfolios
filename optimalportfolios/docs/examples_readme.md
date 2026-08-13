# Examples

Repository-root scripts illustrating every solver, covariance estimator, and end-to-end workflow
in the package. Run them from the repository root as `python -m examples.<path>` so their shared
`examples.data` imports resolve consistently; `*_local.py` diagnostics require local CSV data or
a Bloomberg terminal.

## Layout

```
examples/
├── data/                  fixtures (asset universes loaded once, reused everywhere)
├── solvers/               one demo per single-objective solver
├── backtests/             end-to-end rolling backtest workflows
├── comparisons/           A-vs-B examples (covar / optimiser / parameter / config)
├── covar_estimation/      covariance estimator demos (EWMA, LASSO, GLASSO, factor model)
├── alphas/                alpha signal profiling demos (rank-based profiler)
└── data/sp500_universe_local.py  local S&P 500 universe builder and loader
```

The folder split follows three orthogonal axes:

- **What** the script demonstrates — a *solver*, an *estimator*, or a *full workflow*.
- **How many configurations** it runs — a single one (`solvers/`, `backtests/`) or a sweep over several (`comparisons/`).
- **Where the data comes from** — a shared fixture in `data/`, or a script-local download.

Most files expose a `LocalTests` enum and a `run_local_test()` function so individual
demos can be selected without editing code.

---

## `data/` — fixtures

Shared universe-loading helpers imported by demos elsewhere in the tree. Place new
fixture-style helpers here so that any example referencing them can use one stable
import path.

| File | Purpose |
|---|---|
| `universe.py` | Two helpers for example demos. `fetch_benchmark_universe_data()` returns a 15-ETF universe across 5 asset classes (Equities, Bonds, IG, HY, Commodities) with asset-class loadings and benchmark weights — used by most `solvers/`, `backtests/`, and `comparisons/` files. `fetch_minimal_universe_data()` returns a compact 8-ETF universe with a 3-tuple `(prices, benchmark_prices, group_data)` — used by `backtests/minimal_backtest` and `solvers/long_short`. Both helpers fetch via `yfinance`. |

**Local S&P 500 fixture:**

| File | Purpose |
|---|---|
| `sp500_universe_local.py` | Builds and loads local S&P 500 historical constituents with point-in-time inclusion indicators from [fja05680/sp500](https://github.com/fja05680/sp500). Its Yahoo and Bloomberg variants persist CSVs outside the repository, so the unattended examples workflow skips it. |

---

## `solvers/` — one demo per single-objective solver

Each file shows both a single-date solve via `wrapper_*` and a rolling backtest via
`rolling_*` for the same objective. Useful as a Rosetta stone between objectives.

| File | Solver | Method |
|---|---|---|
| `min_variance.py` | `rolling_quadratic_optimisation` (MIN_VARIANCE) | CVXPY QP. Minimise `w′Σw`. |
| `minimum_tracking_error.py` | `rolling_minimise_tracking_error` | CVXPY QP. Minimise `(w − w_b)′Σ(w − w_b)` relative to a static or rolling benchmark. |
| `max_sharpe.py` | `rolling_maximize_portfolio_sharpe` | CVXPY SOCP via Charnes–Cooper transformation. Rolling EWMA mean + covar. |
| `max_diversification.py` | `rolling_maximise_diversification` | SciPy SLSQP. Maximise `DR(w) = w′σ / sqrt(w′Σw)`. |
| `risk_budgeting.py` | `rolling_risk_budgeting` | Internal CCD/ADMM. Equal or specified risk contributions. |
| `carra_mixture.py` | `rolling_maximize_cara_mixture` | SciPy SLSQP. Expected CARA utility under K-component Gaussian mixture. |
| `tracking_error.py` | `rolling_maximise_alpha_over_tre` | CVXPY QCQP. Maximise α′(w − w_b) subject to TE budget. EWMA momentum signal vs ETF benchmark. |
| `target_return.py` | `rolling_maximise_alpha_with_target_return` | CVXPY. Maximise alpha subject to a target portfolio return (yield + price-return). |
| `long_short.py` | dispatcher `compute_rolling_optimal_weights` | Demonstrates the long-short constraint flow (`is_long_only=False`, `min_exposure`/`max_exposure` set explicitly). |

All these use the `data/universe.py` fixture except `target_return.py`, which
builds its own universe inline (it needs extra columns: yields, dividends, and
target returns not provided by the shared fixture).

---

## `backtests/` — end-to-end rolling workflows

Full SAA-style workflows: load universe → estimate covariance → run rolling solver → generate factsheet PDF. Pick one as a starting template.

| File | What it shows |
|---|---|
| `minimal_backtest.py` | Smallest end-to-end example: defines a universe, runs `compute_rolling_optimal_weights` for one objective, prints/plots NAVs. Best starting point for new users. |
| `balanced_risk_budgets.py` | Illustrates `solve_for_risk_budgets_from_given_weights`: given a static 60/40 weight, back out the equivalent risk-budget portfolio and compare weights vs risk contributions. Useful for translating between mandate languages (weight-based ↔ risk-based). |
| `tracking_error_decomposition_local.py` | Computes per-asset *tracking-error contributions* of a portfolio vs benchmark from a locally persisted Dow 30 panel. Two modes: marginal TE contributions (sum to total TE) and independent (diagonal) TE contributions. |

---

## `comparisons/` — A-vs-B sweeps

Each file runs the same underlying workflow under several configurations and
compares the results in a single factsheet or table. Use these as templates when
calibrating production parameters.

| File | Axis being swept |
|---|---|
| `optimisers.py` | Several `PortfolioObjective` values on the same universe and covariance estimator. Compares NAV, turnover, group exposure across objectives. |
| `covar_estimators.py` | EWMA vs LASSO vs Group LASSO factor covariance, with and without vol-normalisation. Same optimiser throughout. |
| `parameter_sensitivity.py` | One-method, multiple parameter values (e.g. carra grid, span grid). Backtester sensitivity panel. |
| `risk_budgeting_ccd_vs_scipy.py` | Two implementations of constrained risk budgeting: the internal CCD/ADMM and a naive SciPy SLSQP. Demonstrates why the convex reformulation is the production backend. |
| `sp500_minvar_spans_local.py` | Min-variance on a locally persisted S&P 500 panel across EWMA spans of 26 / 52 / 104 / 208 weeks (half-lives 6m / 1y / 2y / 4y). Imports `load_sp500_universe_yahoo` from `data/sp500_universe_local.py`. |
| `drift_policy.py` | Compares `OptimiserConfig.use_drifted_weights_0 = True` (production default, B) vs `False` (legacy, A) using `rolling_quadratic_optimisation` with a binding L1 turnover budget. Shows that under (A) the realised turnover exceeds the optimiser's apparent turnover by ~20%; under (B) the two agree. |

---

## `covar_estimation/` — estimator demos

Focused on the covariance side of the workflow; no portfolio optimisation involved.
Useful as inputs / diagnostics for the backtest examples above.

| File | What it shows |
|---|---|
| `simulate_factor_returns.py` | Simulates a factor model (`Y = X β + ε`) with controllable correlation and noise structure. Used as ground truth for the LASSO estimator below. |
| `lasso_covar_estimation.py` | Fits the LASSO / Group LASSO factor model on simulated and real data; compares to a vanilla EWMA covariance. Demonstrates `FactorCovarEstimator`. |
| `demo_covar_different_estimation_freqs.py` | Same estimator, different return frequencies (D / W-WED / ME). Shows annualisation factor and sample-size trade-off. |

---

## `alphas/` — signal profiling demos

Demos of the rank-based alpha profiler in `optimalportfolios/alphas/profile/`. The profiler holds
the top quantile of assets ranked by a signal, equal-weighted, against an equal-weight-all
benchmark — no optimiser and no covariance — so what it reports is the selection power of the
signal on its own. The [alphas module README](../alphas/README.md) documents the API.

| File | What it shows |
|---|---|
| `profile_alpha_signals.py` | Carry, low-beta and momentum on a 12-ETF bond universe: all three jointly via `profile_alpha_signals`, carry alone via `profile_carry`, and carry swept across top-quantiles 0.25 / 1/3 / 0.5. The carry panel is a real trailing-12m distribution yield built from `yfinance` dividend history rather than a static yield, so it also shows how to assemble a carry input. Writes a multi-strategy factsheet PDF through `generate_alpha_profile_report`. |

---

## Recommended reading order for newcomers

1. `data/universe.py` — understand the test fixture everything builds on.
2. `backtests/minimal_backtest.py` — see one full workflow end-to-end.
3. `solvers/min_variance.py` — minimal solver demo with both single-date and rolling forms.
4. `solvers/minimum_tracking_error.py` — covariance-closest feasible portfolio relative to a benchmark.
5. `solvers/tracking_error.py` — the production TAA pattern (alpha + benchmark + TE constraint).
6. `comparisons/optimisers.py` — see how objectives differ on the same universe.
7. `covar_estimation/lasso_covar_estimation.py` — when EWMA isn't enough, this is the next step.

---

## Conventions used across the demos

- **Universe loading.** Demos share two helpers in `data/universe.py`: `fetch_benchmark_universe_data` (15-ETF universe with asset-class loadings and benchmark weights, used by `solvers/`, `backtests/balanced_risk_budgets`, and most of `comparisons/`) and `fetch_minimal_universe_data` (compact 8-ETF universe with a 3-tuple return, used by `backtests/minimal_backtest` and `solvers/long_short`). `solvers/target_return.py` defines its own loader inline because it needs extra columns (dividends, yields, target returns) the shared fixtures don't provide.
- **Time period.** Most demos use `qis.TimePeriod('31Jan2007', '17Apr2025')`. Adjust as needed; covariance estimators warm up over the early part of this window.
- **Rebalancing.** Default is `'QE'` for SAA-style examples; faster cadences are tuned per file when relevant.
- **Transaction costs.** Where simulated, `rebalancing_costs=0.0003` (3 bp). Adjust to match the realism of your asset class.
- **Output.** Factsheet PDFs go to `optimalportfolios.local_path.get_output_path()`. Console diagnostics print summary statistics directly.
- **Solver config.** Most demos rely on `OptimiserConfig()` defaults. Override `solver`, `verbose`, or `use_drifted_weights_0` (production default `True`) as needed.

---

## Migration note

This layout reorganises the previous flat structure. If you have notebooks or
scripts referencing the old paths, update as follows:

| Old path | New path |
|---|---|
| `examples.universe` | `examples.data.universe` |
| `examples.optimal_portfolio_backtest` | `examples.backtests.minimal_backtest` |
| `examples.solve_risk_budgets_balanced_portfolio` | `examples.backtests.balanced_risk_budgets` |
| `examples.computation_of_tracking_error` | `examples.backtests.tracking_error_decomposition_local` |
| `examples.multi_optimisers_backtest` | `examples.comparisons.optimisers` |
| `examples.multi_covar_estimation_backtest` | `examples.comparisons.covar_estimators` |
| `examples.parameter_sensitivity_backtest` | `examples.comparisons.parameter_sensitivity` |
| `examples.risk_budgeting_pyrb_vs_scipy` | `examples.comparisons.risk_budgeting_ccd_vs_scipy` |
| `examples.sp500_minvar` | `examples.comparisons.sp500_minvar_spans_local` |
| `examples.long_short_optimisation` | `examples.solvers.long_short` |
| `examples.sp500_universe` | `examples.data.sp500_universe_local` |
| `alphas.profile.profile_alpha_signals` | `examples.alphas.profile_alpha_signals` |

The last row is a different kind of move: that file was never library code, only an example that happened to sit inside the `alphas` package. It also shadowed the exported function of the same name.
