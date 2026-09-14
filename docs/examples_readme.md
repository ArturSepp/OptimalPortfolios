---
myst:
  html_meta:
    description: >-
      Find runnable OptimalPortfolios examples by workflow and data requirements:
      offline fixtures, Yahoo downloads, local datasets, outputs and development runners.
---

# Examples

*[author / affiliation / date — placeholder]*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

The repository's examples demonstrate portfolio construction, covariance estimation and
reporting. This guide maps each Python example to its data requirements and purpose.
[QIS](https://github.com/ArturSepp/QuantInvestStrats) supplies backtesting, analytics and plots;
[FactorLasso](https://github.com/ArturSepp/FactorLasso) supplies sparse factor estimation and
clustering used by the relevant examples.

## Run an example

Start with the [offline quickstart](quickstart.md). It uses the packaged monthly multiasset
test fixture and prints results without downloading data or writing files. The fixture's
original-source attribution remains incomplete; it is a fixed test dataset.

Use the interpreter selected in the [installation guide](installation.md), with the package
installed. The commands below assume the repository root is the working directory:
`examples/` is excluded from wheels, and downloading one script may omit its shared imports.
On this Windows/OneDrive checkout, follow the installation guide's external-environment
setup and use a C-local source export for execution.

```console
python -m examples.getting_started.production_quickstart
python -m examples.backtests.multiasset_saa
python -m examples.comparisons.hrp_linkage_semantics
```

The [quickstart notebook](../examples/getting_started/production_quickstart.ipynb) mirrors the
canonical Python script. Its hosted setup installs the released package and needs a network
connection; the analysis then uses the packaged fixture.

For a Yahoo-backed solver example, install the published `data` extra first:

```console
python -m pip install "optimalportfolios[data]"
python -m examples.solvers.min_variance
```

Downloaded prices and generated results can change on later runs. Check the selected script's
main guard, data window and output paths before running it. Module execution selects its default
scenario; it does not execute every branch of a `Locals` enum.

### Unattended execution lanes

The [classifier and runner](../.github/scripts/run_examples.py) derives lanes from Python imports.
Its current inventory is **7 offline, 19 network, 26 unattended examples**. The **4 local-data workflows** ending in
`*_local.py` are excluded. Each catalogue row below states the classification.

```console
python .github/scripts/run_examples.py --list
python .github/scripts/run_examples.py --lane offline --jobs 2
```

`--list` classifies all examples and runs none. A network label means `yfinance` is reachable
directly or through intra-`examples` imports, including imports inside optional branches.
This is a dependency classification, not a network sandbox or proof that every possible mode is
offline. For example, the HRP comparison's optional `--check-pypfopt` mode needs an additional
comparison package; its default run uses core dependencies.

The runner starts each selected module in a separate interpreter, sets `MPLBACKEND=Agg` unless
a backend is already set, and defaults to a 900-second timeout per module. It returns failure
if an example fails, times out, or the selected lane is empty. Headless execution suppresses
interactive plot windows; it does not suppress file writes.

The [examples workflow](../.github/workflows/examples.yml) defines a core-only offline PR gate
on Linux, Windows and macOS with Python 3.12. Its network lane is scheduled daily and advisory
(`continue-on-error`). These are workflow definitions, not a claim that this documentation
review reran the hosted jobs. The [installation guide](installation.md#contributor-groups-and-the-lockfile)
records the pending lockfile reconciliation.

## Layout

```text
examples/
  getting_started/   canonical offline quickstart and its notebook mirror
  data/              shared Yahoo loaders and local universe builders
  solvers/           objective-specific examples
  backtests/         complete rolling workflows
  comparisons/      comparisons of methods or configurations
  covar_estimation/  covariance and factor-model examples
  alphas/            signal profiling
  reports/           manually prepared portfolio reports
  figures/           existing documentation previews
```

The catalogue is organized by analytical purpose. The lane describes execution prerequisites,
independently of the folder: an estimator example can use synthetic inputs, and a backtest can
use either a packaged fixture or live downloads.

## Examples versus development runners

Automated contracts live in `src/optimalportfolios/**/tests/*_test.py`. Component development
runners live beside production modules in `run_local/*_run.py`. For example:

```console
python -m optimalportfolios.optimization.general.run_local.quadratic_run
python -m optimalportfolios.alphas.signals.run_local.signals_run
python -m optimalportfolios.utils.run_local.gaussian_mixture_run
```

These source-checkout tools expose `Locals` and `run_local(local=...)`, share development data
under `optimalportfolios.run_local.data`, and are excluded from built distributions. Inspect
their prerequisites before selecting a scenario. They are separate from the unattended example
lanes. The four root-level `*_local.py` workflows below also require manual preparation.

## `getting_started/` — offline entry point

| Source | Lane | Purpose |
|---|---|---|
| [Production quickstart](../examples/getting_started/production_quickstart.py) | Offline | Six assets from the packaged monthly fixture; EWMA covariance, constrained minimum variance and a QIS backtest. Prints weights, NAV and runtime. See the [calculation and execution conventions](quickstart.md). |

## `data/` — fixtures

These helpers include downloaded market data; they are not all frozen test fixtures.

| Source | Lane | Purpose and prerequisites |
|---|---|---|
| [Shared ETF universes](../examples/data/universe.py) | Network | `fetch_benchmark_universe_data()` downloads 15 ETFs across five groups and returns six objects: prices, benchmark prices, group loadings, benchmark weights, group labels and factor proxies. `fetch_minimal_universe_data()` downloads eight ETFs and returns prices, benchmark prices and group labels. |
| [S&P 500 universe builder](../examples/data/sp500_universe_local.py) | Local | Reads the historical-constituents CSV named by `SP500_FILE` under the resource directory's `sp500/` subdirectory. It creates Yahoo or Bloomberg price/inclusion bundles. The main guard selects `CREATE_UNIVERSE_DATA_WITH_BLOOMBERG`, which requires the Bloomberg integration and access; select the Yahoo scenario explicitly for its free-data path. |

The S&P 500 constituent history is sourced from the
[historical-components repository](https://github.com/fja05680/sp500). Inspect the pinned filename
in the builder and supply that input before execution. Inclusion indicators do not recover
delisted price histories missing from the chosen data provider.

## `solvers/` — one demo per single-objective solver

The files illustrate selected single-date or rolling entry points. Their branches and supported
constraints differ; consult the [optimization guide](optimization_module_readme.md) and
[constraints contract](constraints.md) before changing objectives.

| Source | Lane | Purpose |
|---|---|---|
| [Minimum variance](../examples/solvers/min_variance.py) | Network | `rolling_quadratic_optimisation` with `MIN_VARIANCE`. |
| [Minimum tracking error](../examples/solvers/minimum_tracking_error.py) | Network | `wrapper_minimise_tracking_error` and `rolling_minimise_tracking_error`; find a feasible portfolio close to a benchmark. |
| [Maximum Sharpe](../examples/solvers/max_sharpe.py) | Network | `rolling_maximize_portfolio_sharpe` with estimated means and covariance. |
| [Maximum diversification](../examples/solvers/max_diversification.py) | Network | `rolling_maximise_diversification`; compare portfolio diversification under constraints. |
| [Risk budgeting](../examples/solvers/risk_budgeting.py) | Network | `rolling_risk_budgeting` with equal or specified risk budgets. |
| [CARA mixture](../examples/solvers/carra_mixture.py) | Network | `rolling_maximize_cara_mixture`; expected utility under a fitted Gaussian mixture. The filename retains the historical `carra` spelling. |
| [Alpha with a tracking-error budget](../examples/solvers/tracking_error.py) | Network | `rolling_maximise_alpha_over_tre`; alpha allocation relative to an ETF benchmark. |
| [Alpha with a target return](../examples/solvers/target_return.py) | Network | `rolling_maximise_alpha_with_target_return`; includes yield and price-return inputs from its own loader. |
| [Long-short allocation](../examples/solvers/long_short.py) | Network | `compute_rolling_optimal_weights` with `is_long_only=False` and explicit exposure bounds. |
| [Overlay tail floor](../examples/solvers/overlay_tail_floor.py) | Offline | Synthetic core and overlay inputs with a homogeneous return-floor constraint. See the [overlay methodology](overlay_tail_floor.md). |

## `backtests/` — end-to-end rolling workflows

| Source | Lane | Purpose and outputs |
|---|---|---|
| [Minimal ETF backtest](../examples/backtests/minimal_backtest.py) | Network | Download eight ETFs, estimate covariance, solve and produce QIS factsheets. Writes a PDF and three preview images; see output handling below. |
| [Multiasset strategic allocation](../examples/backtests/multiasset_saa.py) | Offline | Uses the packaged monthly fixture, a 36-observation EWMA span and annual decisions. The default `OBJECTIVE_SWEEP` prints final weights for three covariance-based objectives; other branches compare NAVs or apply group constraints. |
| [Balanced risk budgets](../examples/backtests/balanced_risk_budgets.py) | Network | Infer risk budgets from a 60/40 allocation and compare weights with risk contributions. |
| [Tracking-error decomposition](../examples/backtests/tracking_error_decomposition_local.py) | Local | Loads `dow30_prices.csv` through `get_resource_path()`; configure the resource path or prepare the input there. The copy under `examples/data/` is not selected automatically. Imports `yfinance` even when its download helper is not called. Compares total-risk contributions with standalone diagonal-risk magnitudes. |

For current risk-model analytics, use the QIS implementation described in the
[minimum-tracking-error guide](minimum_tracking_error.md). The local decomposition script is a
historical illustration with its own formulas. Its signed diagonal quantities are not an
additive decomposition of total tracking error.

## `comparisons/` — A-vs-B sweeps

These are implementation and sensitivity examples; a comparison does not establish future
performance or a general ranking of optimizers.

| Source | Lane | Purpose and prerequisites |
|---|---|---|
| [Portfolio objectives](../examples/comparisons/optimisers.py) | Network | Compare objectives on shared prices; writes a multi-portfolio report and preview. |
| [Covariance estimators](../examples/comparisons/covar_estimators.py) | Network | Compare EWMA and factor-model configurations with a common objective; writes a report and preview. |
| [EWMA span sensitivity](../examples/comparisons/parameter_sensitivity.py) | Network | Maximum diversification with weekly EWMA spans 5, 13, 26, 52, 104; writes a report and preview. |
| [Risk-budgeting formulations](../examples/comparisons/risk_budgeting_ccd_vs_scipy.py) | Offline | A fixed three-asset covariance compares the internal risk-budgeting solver with a squared-risk-contribution SLSQP formulation. The result is specific to these inputs and formulations. |
| [HRP linkage semantics](../examples/comparisons/hrp_linkage_semantics.py) | Offline | Fixed four-asset inputs isolate direct correlation distances versus distances between distance profiles, using the same allocation routine. The optional external-package comparison is outside the default run. |
| [Local S&P 500 span comparison](../examples/comparisons/sp500_minvar_spans_local.py) | Local | Weekly EWMA spans 26, 52, 104, 208. Requires the Yahoo CSV bundle from the S&P 500 builder and downloads SPY for benchmarking. Writes a factsheet. |
| [Drift-policy comparison](../examples/comparisons/drift_policy.py) | Network | Compare previous target weights with drift-adjusted decision-date weights under a minimum-variance turnover budget. Its diagnostic is measured at decision dates; consult the [turnover guide](turnover_and_transaction_costs.md) for actual lagged trades. |

The span numbers above are EWMA smoothing parameters measured in return observations,
not half-lives or fixed rolling windows. The [covariance guide](covariance_estimators.md)
defines the decay convention. Legacy source comments and calendar-style labels can be
imprecise; use the configured frequency and numerical span. No fixed percentage turnover
improvement is implied by the drift-policy example.

## `covar_estimation/` — estimator demos

| Source | Lane | Purpose |
|---|---|---|
| [Simulated factor returns](../examples/covar_estimation/simulate_factor_returns.py) | Offline | Generate synthetic factors and assets with a known covariance structure. |
| [LASSO covariance](../examples/covar_estimation/lasso_covar_estimation.py) | Network | Fits LASSO factor models to Yahoo asset and benchmark prices downloaded at module import; compares frequencies and estimated exposures. |
| [Portable CSV factor-risk model](../examples/covar_estimation/rolling_factor_covar_from_csv.py) | Network | The default `all` mode fetches six CSV inputs and then fits a rolling factor-risk model. The `load` mode uses an existing bundle without downloading; it still needs that bundle. See the [CSV workflow](rolling_factor_covar_from_csv.md). |
| [Mixed estimation frequencies](../examples/covar_estimation/demo_covar_different_estimation_freqs.py) | Offline | Synthetic factor data compare business-daily, Wednesday-weekly and month-end asset/factor frequencies, with covariance heatmaps. See [mixed-frequency data](mixed_frequency_data.md). |

A script's network classification remains unchanged when a particular mode can reuse local inputs.
Follow the CSV workflow's separate fetch/load commands and explicit C-local `--data-dir`.

## `alphas/` — signal profiling demos

| Source | Lane | Purpose |
|---|---|---|
| [Alpha-signal profiling](../examples/alphas/profile_alpha_signals.py) | Network | Carry, low-beta and momentum on a bond ETF universe; builds trailing distribution yields from Yahoo dividend history and produces a QIS multi-strategy report. |

The profiler compares equal-weighted portfolios selected by signal rank with an equal-weight
universe benchmark. It isolates a selection rule without solving a portfolio optimization.
Read the [alpha-signal guide](alphas_module_readme.md), including its current beta-initialization
timing limitation, before interpreting the example as a historical trading result.

## `reports/` — portfolio reports

| Source | Lane | Purpose and prerequisites |
|---|---|---|
| [Options stress report](../examples/reports/stress_testing_with_options_local.py) | Local | A stock-and-option stress report using FCGL clusters, QIS stress-report APIs and VOP pricing. Requires `vanilla_option_pricers` and `yfinance`; fetches Yahoo prices unless a verified cache is available. Use explicit C-local cache and fresh output paths. See the [stress-testing guide](stress_testing_with_options.md). |

## Recommended reading order for newcomers

1. [Quickstart](quickstart.md) — the smallest offline portfolio workflow and its calculation contract.
2. [Multiasset allocation](../examples/backtests/multiasset_saa.py) — another offline workflow with group metadata and objective choices.
3. [Shared ETF data](../examples/data/universe.py) and the [minimal backtest](../examples/backtests/minimal_backtest.py) — transition to downloaded prices and reporting.
4. [Minimum variance](../examples/solvers/min_variance.py) and [minimum tracking error](../examples/solvers/minimum_tracking_error.py) — covariance-based construction.
5. [Alpha with a tracking-error budget](../examples/solvers/tracking_error.py) — benchmark-relative allocation with a signal.
6. [Objective comparisons](../examples/comparisons/optimisers.py) — inspect how construction choices change an otherwise shared workflow.
7. [LASSO covariance](../examples/covar_estimation/lasso_covar_estimation.py) and the [CSV risk model](rolling_factor_covar_from_csv.md) — factor-based estimation and portable inputs.

## Conventions used across the demos

Each script specifies its own sample, return frequency, estimation span, decision cadence,
implementation lag and transaction costs. There is no single date window or cost rate shared
by all examples. The [rolling-backtest guide](rolling_backtests.md) explains why a lag counts
price observations and why drift at a decision date can differ from holdings at execution.

### Output paths and preview images

Console-only and plot-only examples need no factsheet file. Scripts that call
`optimalportfolios.local_path.get_output_path()` use the configured output destination;
inspect that resolved path before execution rather than assuming it is outside the checkout.

Four scripts also write directly to `examples/figures/` relative to their own source files:
the [minimal backtest](../examples/backtests/minimal_backtest.py),
[objective comparison](../examples/comparisons/optimisers.py),
[span comparison](../examples/comparisons/parameter_sensitivity.py), and
[covariance comparison](../examples/comparisons/covar_estimators.py).
Running them in the primary checkout can overwrite existing documentation previews.
Run such examples in a disposable C-local source export on this host.

The [documentation analytics pipeline](documentation_standard.md#analytical-conventions-and-figures)
provides the separate, registered route for generating and reviewing all six synthetic
documentation previews together. An example run alone is not a reviewed publication or a
refresh of every displayed image. Keep data as-of dates, fixed sample dates and generation
timestamps distinct.

## Migration note

This layout reorganises the previous flat structure. If you have notebooks or
scripts referencing the old paths, update as follows:

| Old module → Current module |
|---|
| `examples.universe` → `examples.data.universe` |
| `examples.optimal_portfolio_backtest` → `examples.backtests.minimal_backtest` |
| `examples.solve_risk_budgets_balanced_portfolio` → `examples.backtests.balanced_risk_budgets` |
| `examples.computation_of_tracking_error` → `examples.backtests.tracking_error_decomposition_local` |
| `examples.multi_optimisers_backtest` → `examples.comparisons.optimisers` |
| `examples.multi_covar_estimation_backtest` → `examples.comparisons.covar_estimators` |
| `examples.parameter_sensitivity_backtest` → `examples.comparisons.parameter_sensitivity` |
| `examples.risk_budgeting_pyrb_vs_scipy` → `examples.comparisons.risk_budgeting_ccd_vs_scipy` |
| `examples.sp500_minvar` → `examples.comparisons.sp500_minvar_spans_local` |
| `examples.long_short_optimisation` → `examples.solvers.long_short` |
| `examples.sp500_universe` → `examples.data.sp500_universe_local` |
| `alphas.profile.profile_alpha_signals` → `examples.alphas.profile_alpha_signals` |

The last row moves an example out of the library's alpha package; it had also shadowed the
exported profiling function.

## See also

- [Installation and contributor environment](installation.md)
- [Optimization entry points](optimization_module_readme.md)
- [Constraints](constraints.md), [covariance estimation](covariance_estimators.md) and [backtest timing](rolling_backtests.md)
- [Documentation and analytics standard](documentation_standard.md)

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff)
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff)
