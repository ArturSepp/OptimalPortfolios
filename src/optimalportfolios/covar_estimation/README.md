---
myst:
  html_meta:
    description: >-
      Contributor guide to OptimalPortfolios covariance estimation: module ownership,
      estimator interfaces, factor containers, offline examples, and diagnostic workflows.
---

# Covariance Estimation Module

*[author / affiliation / date — placeholder]*

Contributor documentation for [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

The covariance layer prepares return inputs, coordinates direct EWMA or sparse factor fits,
and supplies dated covariance matrices to portfolio construction. It delegates sparse fitting
and decomposition containers to [FactorLasso](https://github.com/ArturSepp/FactorLasso), and
return sampling, EWMA kernels, risk analytics and generic plotting to
[QIS](https://github.com/ArturSepp/QuantInvestStrats).

The [covariance methodology](../../../docs/covariance_estimators.md) is authoritative for
formulas, units and timing limitations. This README maps the contributor interfaces and
workflows; it does not define a second calculation contract.

## Architecture

| Module | Responsibility |
|---|---|
| [Exports](./__init__.py) | Estimator classes, fitting/reporting helpers and the risk-model adapter. |
| [covar_estimator.py](./covar_estimator.py) | Abstract current/rolling interface and estimation schedule configuration. |
| [ewma_covar_estimator.py](./ewma_covar_estimator.py) | Direct EWMA orchestration using QIS kernels; current-fit helper and a QIS rolling-helper re-export. |
| [factor_covar_estimator.py](./factor_covar_estimator.py) | Per-cadence factor fitting, annualization, rolling input cutoffs and cluster-reference integration. |
| [utils.py](./utils.py) | Return preparation through QIS, including the estimator's EWMA demeaning convention. |
| [risk_model_adapter.py](./risk_model_adapter.py) | Covariance dictionaries or factor decompositions to `qis.RiskModel`. |
| [covar_reporting.py](./covar_reporting.py) | Extract fitted diagnostics and assemble figures with QIS. |
| [risk_labelling.py](./risk_labelling.py) | Deprecated compatibility imports; cluster-lineage work belongs in `factorlasso.cluster_lineage`. |

### Estimator hierarchy

`CovarEstimator` is an abstract base in its owning module. It is not re-exported from
`optimalportfolios` or `optimalportfolios.covar_estimation`.

| Concrete class | Inputs | Current result | Rolling result |
|---|---|---|---|
| `EwmaCovarEstimator` | Asset prices | Annual covariance DataFrame | Date-keyed covariance dictionary |
| `FactorCovarEstimator` | Factor prices and asset-return buckets | Annual covariance DataFrame | Date-keyed covariance dictionary |

Both concrete classes are exported at the package root and covariance-subpackage level.
Their input signatures differ. The shared output shape does not imply identical estimation
dates, warmup rules or parameter meanings.

The factor-specific `fit_current_factor_covars` and `fit_rolling_factor_covars` methods retain
the decomposition in `CurrentFactorCovarData` and `RollingFactorCovarData`. These containers,
`VarianceColumns`, `LassoModel` and `LassoModelType` belong to FactorLasso. OptimalPortfolios
also re-exports them at its root, but the covariance subpackage does not. Prefer imports from
the owning package when extending generic factor functionality.

### Why factorlasso is a separate package

FactorLasso owns sparse regression, grouping and factor-decomposition APIs, with its own
release metadata, tests and software citation. OptimalPortfolios owns their integration into
portfolio estimation and construction. Consult each package's metadata for dependencies and
supported interfaces; fixed dependency counts and publication plans are not interface guarantees.

### Separation of concerns: factorlasso vs covar_estimation

`estimate_lasso_factor_covar_data` aligns factor prices to each asset-return bucket, computes
log returns, calls the configured FactorLasso fit, annualizes the relevant variance and alpha
outputs, and merges asset rows. Betas and R² are dimensionless; **R² is not annualized**.

`FactorCovarEstimator` adds estimation schedules, per-date input slicing, optional causal
cluster paths and access to plain matrices or full decompositions. The helper leaves the
supplied `LassoModel` carrying the final bucket's fitted state. Use returned containers for
the combined result, rather than treating that model object as a history of all fits.

## Mathematical Framework

The following sections preserve existing entry links and point to the authoritative
methodology. Keep new derivations there and verify them against the implementation.

### Factor Model Covariance Decomposition

See [factor and HCGL covariance](../../../docs/covariance_estimators.md#factor-and-hcgl-covariance).
Loadings use assets on rows and factors on columns. The residual term is diagonal;
`residual_var_weight` changes that term without refitting the model.

### Variance Decomposition per Asset

The same [factor methodology](../../../docs/covariance_estimators.md#factor-and-hcgl-covariance)
distinguishes systematic and residual variance. The stored R² diagnostic comes from the
regression fit, with the adapter's missing-value filling and lower clipping; it need not
equal a ratio recomputed from the separately estimated annual factor covariance.

### Mixed-Frequency Annualisation

See [mixed-frequency data](../../../docs/mixed_frequency_data.md) and the
[covariance units contract](../../../docs/covariance_estimators.md#factor-and-hcgl-covariance).
Factor covariance and residual variances use their respective cadences. A supplied
`x_covar` must already have compatible annual units and factor labels. Observation frequency,
smoothing span and covariance output schedule are separate settings.

### LASSO Estimation Methods

`LassoModelType` is owned by FactorLasso. Common choices include `LASSO`, `GROUP_LASSO`,
`HIERARCHICAL_CLUSTER_GROUP_LASSO` (HCGL) and `FACTOR_CLUSTER_GROUP_LASSO` (FCGL).
The enum also contains other model families; this is not an exhaustive list.
Use [FactorLasso's documentation](https://github.com/ArturSepp/FactorLasso) for penalties,
sign constraints, priors, clustering and observation weighting.

### EWMA Covariance

See [EWMA covariance](../../../docs/covariance_estimators.md#ewma-covariance) for the recursion,
initialization, demeaning, missing updates and span/half-life distinction.
The class has no shrinkage-to-identity parameter.
`estimate_rolling_ewma_covar` is a QIS re-export whose defaults differ from the class.

The direct rolling normalized-return option has a documented full-array initialization
limitation. For historical work, follow the
[timing qualifications](../../../docs/covariance_estimators.md#current-fits-and-rolling-dates):
an ordinary current factor fit does not truncate inputs merely because `estimation_date`
was supplied. The factor estimator's top-level `demean` field also does not control the
internal factor-covariance demeaning. Keep these numerical corrections separate from documentation.

## CSV-only factor-risk example

The [repository example](../../../examples/covar_estimation/rolling_factor_covar_from_csv.py)
separates `fetch` (Yahoo teaching proxies) from `load` (saved inputs, no download).
The [CSV guide](../../../docs/rolling_factor_covar_from_csv.md) defines all six input files,
reference-currency and FX-hedging conventions, cutoff handling and the MATF handoff.

From a configured source checkout, set `$dataDirectory` to an explicit absolute directory
outside the checkout. On the maintainer's Windows host, use the C-local task area described
in [AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md).
These commands write/read input files; `fetch` also requires the `data` extra and network access.

```powershell
python -m examples.covar_estimation.rolling_factor_covar_from_csv fetch --data-dir $dataDirectory
python -m examples.covar_estimation.rolling_factor_covar_from_csv load --data-dir $dataDirectory
```

The files persist source inputs and settings, not a serialized `RollingFactorCovarData`.
The load stage reconstructs the fitted snapshots and adapts them through `opt.build_risk_model`
to `qis.RiskModel`. Keep the delivered factor panel's labels and currency basis when replacing
teaching proxies. The guide includes an offline round trip and the loader's validation limits.

## Data Containers (from factorlasso)

### `CurrentFactorCovarData`

Let N denote assets, M factors and T the residual time index.

| Field | Shape and meaning |
|---|---|
| `x_covar` | M-by-M annual factor covariance supplied by this adapter. |
| `y_betas` | N-by-M dimensionless loadings; rows are assets, columns are factors. |
| `y_variances` | Asset-indexed diagnostics; four numerical fit columns plus cluster labels when present. |
| `residuals` | T-by-N panel scaled by each bucket's annualization factor, without fitted-intercept subtraction. |
| `estimation_date` | Snapshot metadata; not a substitute for a historical input cutoff. |
| `clusters`, `linkages`, `cutoffs` | Optional frequency-prefixed memberships, stacked linkage rows and per-frequency cut distances. |
| `derived_signs` | Optional asset-by-factor sign requirements; missing requirements are distinct from zero-loading constraints. |

Do not treat the residual panel as an unscaled per-observation series or recover residual
variance by taking its ordinary sample variance. See the
[implementation qualifications](../../../docs/covariance_estimators.md#implementation-in-optimalportfolios).

| Operation on a current snapshot | Result |
|---|---|
| `y_covar` / `get_y_covar(residual_var_weight=...)` | Default/scaled assembled asset covariance. |
| `get_snapshot()` / `get_model_vols()` | Combined diagnostics / total, systematic and residual volatilities. |
| `estimate_alpha(alpha_span=...)` | Smoothed residual alpha; supply cadence information when using cadence-specific spans. |
| `filter_on_tickers(assets)` | A filtered snapshot. |
| `save(path)` / `CurrentFactorCovarData.load(path)` | FactorLasso's Excel component persistence, requiring a pandas-compatible Excel engine. |

Persistence is explicit file I/O. Use an absolute caller-selected output path outside the
checkout and check the owning implementation's saved fields before relying on a round trip.

### `RollingFactorCovarData`

This container holds `data: dict[Timestamp, CurrentFactorCovarData]`.

| Operation | Result |
|---|---|
| `get_y_covars()` | Date-keyed annual covariance matrices. |
| `get_r2()`, `get_systematic_vars()`, `get_residual_vars()` | Date-by-asset diagnostic panels. |
| `get_total_vols()`, `get_residual_vols()` | Date-by-asset volatility panels. |
| `get_alphas(alpha_span=...)` | Residual alpha panel; cadence-specific spans need `asset_frequencies`. |
| `get_beta(factor=...)` | Date-by-asset loadings for a factor label present in the model. |
| `get_snapshot()` | Timestamp-keyed dictionary of diagnostic DataFrames. |
| `get_latest()` | The latest stored snapshot; not a historical as-of selection. |

`opt.build_risk_model(rolling_data)` retains full factor decomposition with residual weight 1.
Passing a plain matrix dictionary creates a covariance-only QIS model. Use the adapter and
QIS risk methods for downstream analysis.

### `VarianceColumns` Enum

Use enum values when selecting stored fields; optional and derived columns are not four
mandatory columns in every snapshot.

| Enum | Column | Role |
|---|---|---|
| `EWMA_VARIANCE` | `ewma_var` | Annualized fitted response variance. |
| `RESIDUAL_VARS` | `residual_var` | Annualized fitted residual variance. |
| `INSAMPLE_ALPHA` | `insample_alpha` | Annualized fitted alpha. |
| `R2` | `r2` | Dimensionless fit diagnostic. |
| `ALPHA` | `stat_alpha` | Derived smoothed residual alpha. |
| `TOTAL_VOL`, `SYST_VOL`, `RESID_VOL` | `total_vol`, `sys_vol`, `resid_vol` | Derived annual volatility columns. |
| `CLUSTER` | `cluster` | Optional membership, also mirrored into `y_variances` by FactorLasso. |

## Usage Examples

Run the four ordinary Python blocks below in order after installing the core package.
They use deterministic teaching prices. ETF-like labels preserve the earlier example's
mapping; **none of these series is observed ETF data**. The examples exercise interfaces,
not investment performance. No frozen fixture or random seed is changed.

### EWMA covariance (simplest case)

```python
import numpy as np
import pandas as pd
import qis
import optimalportfolios as opt

dates = pd.bdate_range("2018-01-01", periods=1260)
t = np.arange(len(dates), dtype=float)
tickers = ["SPY", "EZU", "EEM", "TLT", "HYG", "GLD"]
prices = pd.DataFrame({
    ticker: 100.0 * np.exp(
        0.00015 * (j + 1) * t + 0.03 * np.sin(t / (13 + j))
        + 0.012 * np.cos(t / (23 + j)))
    for j, ticker in enumerate(tickers)
}, index=dates)
time_period = qis.TimePeriod("2021-12-31", "2022-06-30")

estimator = opt.EwmaCovarEstimator(
    returns_freq="W-WED", span=52, rebalancing_freq="QE",
)
current_covar = estimator.fit_current_covar(prices=prices)
covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)
print(current_covar.shape)
print(current_covar.index.equals(current_covar.columns))
```

Expected output is `(6, 6)` followed by `True`. QIS samples log returns on Wednesdays;
the covariance is annualized. The current fit uses the full price sample, while rolling
outputs select dates within `time_period`. Their final matrices need not coincide.

### Factor LASSO covariance

Continue with monthly asset log returns and two synthetic factor-price series:

```python
from factorlasso import LassoModel, LassoModelType

asset_prices = prices
factor_prices = prices[["SPY", "TLT"]].rename(
    columns={"SPY": "Growth", "TLT": "Rates"},
)
lasso_model = LassoModel(
    model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
    reg_lambda=1e-5, span=36, warmup_period=12,
)
factor_estimator = opt.FactorCovarEstimator(
    lasso_model=lasso_model, factor_returns_freq="ME", rebalancing_freq="QE",
)
asset_returns_dict = qis.compute_asset_returns_dict(
    prices=asset_prices, is_log_returns=True, returns_freqs="ME",
)
factor_covar_dict = factor_estimator.fit_rolling_covars(
    risk_factor_prices=factor_prices, asset_returns_dict=asset_returns_dict,
    time_period=time_period,
)
rolling_data = factor_estimator.fit_rolling_factor_covars(
    risk_factor_prices=factor_prices, asset_returns_dict=asset_returns_dict,
    time_period=time_period,
)
data = rolling_data.get_latest()
r2_panel = rolling_data.get_r2()
sys_vars = rolling_data.get_systematic_vars()
total_vols = rolling_data.get_total_vols()
risk_model = opt.build_risk_model(rolling_data)
print(data.y_betas.shape)
print(isinstance(risk_model, qis.RiskModel))
```

Expected output is `(6, 2)` followed by `True`. The two fitting calls demonstrate alternative
outputs and repeat estimation; keep the full container and use `get_y_covars()` when both
views are needed. Each rolling fit slices its inputs at its own estimation date. The two
factors reuse teaching paths already in the asset panel; this is not an independent factor study.

### Mixed-frequency asset returns

Continue with the original daily/weekly/monthly mapping:

```python
returns_freqs = pd.Series({
    "SPY": "B", "EZU": "B", "EEM": "B",
    "TLT": "W-WED", "HYG": "W-WED", "GLD": "ME",
})
mixed_returns = qis.compute_asset_returns_dict(
    prices=asset_prices, is_log_returns=True, returns_freqs=returns_freqs,
)
mixed_estimator = opt.FactorCovarEstimator(
    lasso_model=lasso_model, factor_returns_freq="W-WED", rebalancing_freq="QE",
)
mixed_covars = mixed_estimator.fit_rolling_covars(
    risk_factor_prices=factor_prices, asset_returns_dict=mixed_returns,
    time_period=time_period,
)
print(all(matrix.shape == (6, 6) for matrix in mixed_covars.values()))
```

Expected output is `True`. This retains `span=36` and `warmup_period=12` in each bucket,
so the calendar horizons differ. Betas are not annualized; variance components use their
cadence-specific scales. See the mixed-frequency guide for matched-horizon configurations.
Reusing `lasso_model` also replaces its attached last-fit state.

### Diagnostic reporting

This block computes report tables without generating figures:

```python
from optimalportfolios.covar_estimation.covar_reporting import (
    plot_current_covar_data, run_rolling_covar_report,
)

figs, dfs = run_rolling_covar_report(
    risk_factor_prices=factor_prices, prices=asset_prices,
    covar_estimator=factor_estimator, time_period=time_period,
    asset_returns_dict=asset_returns_dict, is_plot=False,
)
print(len(figs) == 0)
print(all(isinstance(frame, pd.DataFrame) for frame in dfs.values()))
```

Expected output is two `True` lines. The report function refits the rolling model and
keys its diagnostic tables by formatted date strings; the container's `get_snapshot()`
uses timestamps. `prices` supplies the asset universe when `assets` is omitted.

To draw the latest monthly snapshot, this optional fragment continues the example and creates
matplotlib figures. It does not write a file:

```python +SKIP
figures = plot_current_covar_data(covar_data=data)
```

Close figures after use. With `is_plot=True`, the rolling report draws each date and calls
`plt.close("all")`; take account of other open figures in an interactive session.
Any saved documentation preview must follow the
[analytics provenance process](../../../docs/documentation_standard.md).

### Cluster plot ownership

QIS owns `plot_dendrogram` and `plot_clusters`. OptimalPortfolios extracts membership,
linkage and cutoff mappings from FactorLasso snapshots and calls `qis.plot_clusters`.
The former `covar_reporting.plot_clusters` wrapper has been removed; use QIS directly.
Its plotting interface supports caller-owned axes and multiple cadences.

## Factor references for cluster discovery

`FactorCovarEstimator(include_factors_in_clustering=True)` adds factor returns as clustering
references for HCGL/FCGL; the default is `False`. Sampling follows each asset bucket.
Correlation settings, linkage rules and causal smoothing belong to the configured `LassoModel`.

Only original assets enter the response fit, pooled sign derivation, group penalties and
portfolio covariance. The reported asset tree removes reference leaves while retaining
induced merge heights. Explicit precomputed partitions take precedence over discovery;
provide clusters, linkages and cutoffs together.

Use `factor_clustering_freqs=["ME"]` to restrict enabled references to monthly responses;
`None` applies them to every bucket. References can change both merges and a fractional
distance cutoff. Inspect cluster counts, stability and held-out reconstruction errors;
a clearer partition alone does not establish better portfolio performance. Repeated or
closely replicated factors can overrepresent one exposure. The
[clustering tests](./tests/factor_clustering_test.py) cover these contracts and timing.

## Development workflow

| Work area | Existing checks |
|---|---|
| EWMA, annualization and returned matrices | [EWMA tests](./tests/ewma_covar_estimator_test.py) and [property tests](./tests/covar_properties_test.py). |
| Factor fitting, configuration and cutoffs | [API tests](./tests/factor_estimator_api_test.py), [guards](./tests/factor_estimator_guards_test.py) and [smoothing](./tests/cluster_smoothing_test.py). |
| QIS risk integration and reports | [Risk adapter tests](./tests/risk_model_adapter_test.py) and [report tests](./tests/covar_reporting_test.py). |
| Public calculation contracts | [Covariance article tests](../tests/covariance_estimators_documentation_test.py) and [mixed-frequency article tests](../tests/mixed_frequency_data_documentation_test.py). |

The [EWMA runner](./run_local/ewma_covar_estimator_run.py) defaults to
`ROLLING_COVAR_PROPERTIES`. Despite its name,
[factor_covar_estimator_run.py](./run_local/factor_covar_estimator_run.py) currently contains
EWMA diagnostics and defaults to `CURRENT_COVAR`; it is not a factor-fitting runner.
Both use `Locals` / `run_local(local=...)`, read the configured local ETF CSV through the
[development data helper](../run_local/data/etf_prices.py), and display diagnostics.
They are source-checkout tools, excluded from distributions.

Use the configured external interpreter and C-local setup from
[AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md).
After setup, run the selected commands from the C-local source export:

```text
python tools/check_docs.py --files src/optimalportfolios/covar_estimation/README.md
python -m pytest src/optimalportfolios/covar_estimation/tests
python -m pytest src/optimalportfolios/tests/covariance_estimators_documentation_test.py
python -m pytest src/optimalportfolios/tests/mixed_frequency_data_documentation_test.py
```

For the manual runner, only after providing its local-data prerequisite:

```text
python -m optimalportfolios.covar_estimation.run_local.ewma_covar_estimator_run
```

Preserve numerical defaults, historical cutoffs, frozen fixtures and seeds. Verify proposed
numerical corrections separately with independent references.

## References

- [Covariance methodology and implementation references](../../../docs/covariance_estimators.md).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
- Sepp, A., Ossa, I., and Kastenholz, M. (2026). “Robust Optimization of Strategic and Tactical
  Asset Allocation for Multi-Asset Portfolios.” *The Journal of Portfolio Management*,
  52(4), 86–120. [Publisher reprint](https://eprints.pm-research.com/17511/143431/index.html);
  [author's bibliographic record](https://artursepp.com/research/).
- Sepp, A., Hansen, E. H., and Kastenholz, M. (2026). “Capital Market Assumptions and Strategic
  Asset Allocation Using Multi-Asset Tradable Factors.”
  [Working paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958).
