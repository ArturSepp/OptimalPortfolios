---
myst:
  html_meta:
    description: >-
      Alpha signal construction in OptimalPortfolios: momentum, beta, residual signals,
      carry, scoring conventions, time-varying clusters, and reproducible diagnostics.
---

# Alpha signals — `optimalportfolios.alphas`

*[author / affiliation / date — placeholder]*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

An alpha signal is a dated characteristic used to compare assets when forming a portfolio.
The `optimalportfolios.alphas` layer computes raw characteristics, transforms them into
scores, and provides containers and evaluation tools. A score is not automatically an
expected return, a portfolio weight, or evidence of investment skill.

## Overview

The signal constructors separate the calculation of an asset characteristic from the
choice of comparison universe. Most return `(score, raw_signal)`. Portfolio objectives,
risk constraints, signal blends and any mapping into bounded alpha values are decisions
made by the consuming application. `AlphasData` stores those decisions; it does not apply
a CDF, enforce a range, or choose combination weights.

### Signal Comparison

| Family | Raw characteristic and interpretation |
|---|---|
| EWMA momentum | Exponentially filtered returns, optionally relative to a benchmark and volatility-normalised. |
| Classic momentum | A fixed number of log returns, with an explicit hard skip of recent periods. |
| Low beta | Estimated beta to a benchmark; lower beta receives a higher score. |
| Residual momentum | Continuation in returns after subtracting a lagged single-benchmark exposure. |
| Residual reversal | Negative residual momentum with a shorter default horizon; recent residual losses receive higher scores. |
| Risk-adjusted carry | Supplied annual carry divided by annualised estimated volatility. |
| Managers alpha | Smoothed returns after subtracting supplied, dated multi-factor exposures. |

These are related signal definitions, not interchangeable replications of the published
strategies in the references. In particular, the residual signal constructors use one
benchmark; the cited residual-return studies use their own estimation and portfolio rules.

## Inputs, notation, and assumptions

Supply positive finite adjusted price or NAV levels in a consistent currency and economic
return basis. A price panel has dates in its index and asset identifiers in its columns.
Group, cadence and loading labels must match those identifiers. Missing or stale prices
need an explicit observation and eligibility policy before portfolio construction.

### Naming Conventions

| Object | Convention in this package |
|---|---|
| Raw signal $x_{i,t}$ | Method-specific units: log return, beta, filtered risk-adjusted return, or carry per unit volatility. |
| Score $z_{i,t}$ | A cross-sectional transformation; its formula depends on the constructor and scoring mode. |
| Combined alpha | An application-supplied objective characteristic; no automatic range or return-unit calibration. |

For native observation date $t$, the log return is
$r_{i,t}=\log(P_{i,t}/P_{i,t-1})$. A supplied benchmark produces relative log returns
$r_{i,t}-r_{b,t}$. This benchmark subtraction is different from subtracting a cash rate.

Use signals formed with information available at $t$ for subsequent holding periods.
The EWMA signal wrappers use contemporaneous volatility weights (`weight_lag=0`);
they do not lag the finished signal for trading. A lagged regression beta is a separate
timing choice. Price timestamps, publication availability and execution time must all agree.

### Mixed-Frequency Support

The paired signal constructors accept a cadence string or an asset-indexed cadence Series.
For EWMA signals, positive integer spans can also be mappings such as `{"ME": 12, "QE": 4}`.
They specify observation counts, not identical calendar windows. Classic momentum has separate
lookback and skip mappings. A 12-month hard lookback is distinct from an EWMA span of 12.

Fixed-group constructors compute scores within each cadence and, if supplied, each group,
then merge and forward-fill the output. Cluster constructors compute raw signals by cadence,
merge/forward-fill them, and then apply cluster scoring. Their fallback statistics can compare
assets across cadences. Managers alpha has its own residual-merge path and a scalar
`alpha_span`; it does not share all these grouping and span options.

The legacy carry-only score helper and `estimate_rolling_ewma_means` use a single cadence.
See [mixed-frequency data](mixed_frequency_data.md) for the observation, estimation and
rebalance clocks, and for the limits of treating filled prices as fresh information.

### Within-Group Scoring (Fixed Groups)

`group_data` defines a comparison set; it does not impose portfolio allocation constraints.
An explicit benchmark keeps the regression reference common across groups. With no supplied
benchmark, low-beta and residual constructors use the equal-weight mean log return of the
assets passed into each fit. In the mixed-frequency fixed-group path, that fit can be a
cadence-by-group subset, so changing groups can also change the implicit benchmark.

EWMA momentum with `benchmark_price=None` uses unsubtracted asset returns. Despite the
current standard constructor's docstring, it does **not** subtract an equal-weight benchmark.

## Methodology

### Signal Functions

#### Momentum

For a long span $L$, the decay is $\lambda=1-2/(L+1)$. The QIS long/short filter is
normalised to unit variance for unit-variance white-noise input; it is not just an EWMA mean.
With no short leg, zero initial state, and $u_0=0$, its complete-observation form is

$$
m_t=\sqrt{1-\lambda^2}\sum_{j=1}^{t}\lambda^{t-j}u_j.
$$

By default $u_t$ is the benchmark-relative log return divided by its contemporaneous
per-period EWMA volatility. Without a benchmark it is based on the asset's own return.
`vol_span=None` disables this normalisation for momentum and residual signals.
A short leg subtracts another exponentially filtered component with the joint normalisation
defined in [QIS's EWMA implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ewm.py).
It does not discard a fixed number of recent returns.

Momentum defaults are `long_span=12`, `short_span=None`, `vol_span=13` and
`mean_adj_type=qis.MeanAdjType.NONE`. The filter masks an initial warm-up period.
With volatility normalisation the raw signal is measured in units of per-period risk;
it is not an annual expected return.

#### Classic momentum

For a lookback of $K$ observations and a skip of $S$, the complete window is

$$
m^{\mathrm{classic}}_{i,t}
=\sum_{j=S}^{S+K-1}r_{i,t-j}
=\log\left(\frac{P_{i,t-S}}{P_{i,t-S-K}}\right).
$$

The defaults include exactly 12 log returns and exclude the most recent one
(`lookback_periods=12`, `skip_periods=1`). There is no benchmark subtraction or
volatility scaling. An incomplete fixed window remains missing.

#### Low Beta

An EWMA regression estimates each asset's beta to the supplied benchmark, or to the
fit's equal-weight mean log return when the benchmark is absent. `beta_span=12` and
`mean_adj_type=EWMA` are the defaults. The standard score is the **negative** of the
cross-sectional beta score; the second return value remains the unnegated beta.

The raw-beta path replaces exact zero loadings with NaN. A zero or missing score therefore
needs interpretation in its scoring context. A low-beta characteristic does not itself
construct a beta-neutral or leveraged betting-against-beta portfolio.

**Timing limitation in the verified environment:** these beta-based constructors call
`qis.EwmLinearModel.fit` without overriding its `init_type=MEAN` default. With
`mean_adj_type=EWMA`, the running means start from the full-sample mean. Later observations
can therefore alter earlier low-beta, residual-momentum and residual-reversal signals.
A warm-up reduces this dependence but does not remove it. The current default examples are
descriptive demonstrations, not verified point-in-time backtest inputs. Explicit
`mean_adj_type=qis.MeanAdjType.NONE` bypasses this mean-initialisation path and passes the
future-observation checks on this fixture; it also changes the estimator's meaning.
Correcting the default initialisation requires a separate numerical change.

#### Residual Momentum

The single-benchmark residual uses the preceding observation's fitted beta:

$$
e_{i,t}=r_{i,t}-\widehat{\beta}_{i,t-1}r_{b,t}.
$$

The residual then enters the same QIS volatility-normalisation and long/short filter.
Default beta and long spans are 12, with `vol_span=13` and no short leg.
`mean_adj_type` controls the beta regression; the residual-volatility leg uses `NONE`.

Residual reversal negates the filtered residual signal. Its default long span is 1.
When all settings, including the long span, are held equal, reversal is the negative
of residual momentum. Their default outputs are not negatives because their default
long spans differ.

#### Managers Alpha

For factor return vector $f_t$, let $\tau(t)$ be the most recent supplied beta snapshot
dated at or before the preceding asset-return date. The manager residual is

$$
e_{i,t}=r_{i,t}-B_{i,\tau(t)}f_t.
$$

Factor returns are measured over the asset's own return periods. Dates without a prior
loading snapshot or a complete factor vector are skipped. A missing manager return stays
missing for that asset. Supply chronologically ordered snapshot keys and aligned factor labels.

The default multiplies residuals by periods per year, then smooths with `alpha_span=12`.
Its score divides by the cross-sectional population standard deviation **without centering**:

$$
a_{i,t}=\operatorname{EWMA}_{12}(A e_{i,t}), \qquad
z^{\mathrm{manager}}_{i,t}=\frac{a_{i,t}}
{\operatorname{std}_{0}(a_{\cdot,t})}.
$$

Here $A=12$ for monthly data; `annualise=False` omits that multiplication. The numerator
is an annual-scaled log-return residual, not a compounded annual forecast. The function
has no `group_data` argument, no cluster variant and no clipping/CDF mapping. A common
positive residual mean remains in the scores; zero dispersion can produce non-finite results.
A residual can reflect omitted risks or model error as well as skill.

#### Risk-adjusted carry and rolling means

Carry requires a separate, dated annual-decimal yield panel. For example, `0.03` means 3%
per year. The carry pair computes yield divided by annualised EWMA log-return volatility,
then scores that ratio. Its default return cadence is `W-WED` and its volatility span is 13.
Supply carry values known at formation time. This ratio does not remove all duration,
credit or other economic risk.

Unlike the momentum filter, `vol_span=None` in carry still estimates volatility: it passes
through to QIS's default decay of 0.94. Use an explicit positive span for a defined horizon.

`estimate_rolling_ewma_means` returns EWMA **log-return means**, not cross-sectional scores.
Defaults are `returns_freq="W-WED"`, `span=52` and `annualize=True`. Annualisation multiplies
by the inferred periods per year. Requested dates between observations receive the latest
available estimate; dates before the return sample remain missing.

### Cluster-Based Scoring

#### Motivation

Fixed groups express a supplied classification; clusters express a partition inferred from
data or supplied by another model. Each chooses which assets form the comparison set.
A different score does not by itself establish that one partition improves investment results.

#### How Clusters Are Derived

For HCGL, FactorLasso normally estimates an **asset-response dependence matrix**, clusters
assets before solving the group-penalised regression, and supplies those labels with the
fitted model. It does not cluster a factor correlation matrix after estimating betas.
Dependence measure, clustering horizon, linkage, distance, external partitions and rolling
smoothing settings can change the resulting labels.

`extract_rolling_clusters` reads the stored asset labels from each covariance snapshot;
it does not estimate clusters. Labels can include cadence prefixes such as `ME:1`.
`align_rolling_clusters` provides overlap-based label alignment for interpretation through
time. Relabeling a partition does not change its memberships or basic within-cluster scores.

#### Scoring Logic

Standard fixed-group scoring delegates to `qis.df_to_cross_sectional_score`. It clips
the raw input to `[-5, 5]`, then uses the non-missing comparison values and population
standard deviation:

$$
c_{i,t}=\operatorname{clip}(x_{i,t},-5,5), \qquad
z_{i,t}=\frac{c_{i,t}-\overline{c}_{g,t}}{\operatorname{std}_{0}(c_{g,t})}.
$$

Low-beta scores reverse the sign. Raw return values are not overwritten by score clipping.
Clipping is applied **before**, not after, standardisation, so a score is not bounded by
either `[-1, 1]` or the raw-input clipping interval. A singleton or zero-dispersion fixed
group normally produces NaN.

With non-empty dated cluster assignments, `score_within_clusters` instead uses sample
standard deviation (`ddof=1`), without the standard path's clipping:

| Situation | Basic cluster-scoring behavior |
|---|---|
| Before the first assignment | All scores are 0, including unavailable raw signals. |
| Cluster size greater than `min_cluster_size` | Score using that cluster's available raw values. |
| Cluster size at most `min_cluster_size` | Use statistics of all currently assigned assets in the signal panel. |
| Only one cluster or too few assigned assets | Use the assigned-universe fallback; degenerate values become 0. |
| Asset without an assignment | Score remains 0. |
| Empty assignment dictionary | Use the standard QIS score, including clipping and `ddof=0`. |

The default threshold is 3: a three-member cluster still uses fallback statistics.
Small clusters and singletons do **not** generally receive zero. Missing assigned raw values
can remain NaN in nondegenerate calculations; zero in this table is not proof of tradability.
Optional stability pooling is an explicit alternate path delegated to FactorLasso and is
not exercised by the basic examples below.

## Worked example

### Fixed synthetic inputs

Run the blocks below in order with core dependencies. Eight synthetic assets, two factors,
a benchmark and annual carry inputs cover 97 monthly price dates, 2016-12-31 through
2024-12-31. The paths are deterministic and require no data files, network or random seed.
All prices share one illustrative currency and total-return basis.

```python
import numpy as np
import pandas as pd
import qis
import optimalportfolios as opt
import optimalportfolios.alphas as alphas
import optimalportfolios.alphas.signals as signals

dates = pd.date_range("2016-12-31", "2024-12-31", freq="ME")
step = np.arange(len(dates), dtype=float)
factor_changes = np.column_stack((
    0.005 + 0.025 * np.sin(0.7 * step),
    0.002 + 0.012 * np.cos(0.4 * step),
))
factor_prices = pd.DataFrame(
    100 * np.exp(np.cumsum(factor_changes, axis=0)),
    index=dates, columns=["Growth", "Rates"],
)
loadings = np.array([
    [1.2, 0.1], [0.9, 0.3], [0.7, 0.2], [0.5, 0.4],
    [0.2, 1.1], [0.3, 0.8], [0.4, 0.6], [0.6, 0.5],
])
specific_changes = (
    0.003 * np.sin(step[:, None] * np.arange(0.9, 1.7, 0.1)[None, :])
    + np.arange(8)[None, :] * 0.0002
)
prices = pd.DataFrame(
    100 * np.exp(np.cumsum(factor_changes @ loadings.T + specific_changes, axis=0)),
    index=dates, columns=list("ABCDEFGH"),
)
asset_prices = prices
benchmark = factor_prices["Growth"]
carry = pd.DataFrame(
    np.broadcast_to(np.linspace(0.02, 0.05, 8), prices.shape),
    index=dates, columns=prices.columns,
)
```

### Standard signal usage

The saved raw and score names make the components available to the later container example.

```python
from optimalportfolios.alphas import compute_momentum_alpha

score, raw = compute_momentum_alpha(
    prices=prices,
    benchmark_price=benchmark,
    returns_freq='ME',
    long_span=12,
)
mom_score, raw_momentum = score, raw
```

```python
from optimalportfolios.alphas import compute_low_beta_alpha

score, raw_beta = compute_low_beta_alpha(
    prices=prices,
    benchmark_price=benchmark,
    returns_freq='ME',
    beta_span=12,
)
beta_score = score
```

```python
from optimalportfolios.alphas import compute_residual_momentum_alpha

score, raw_residual = compute_residual_momentum_alpha(
    prices=prices,
    benchmark_price=benchmark,
    returns_freq='ME',
    beta_span=12,
    long_span=12,
    vol_span=13,
)
res_score = score
```

The newer families use the `optimalportfolios.alphas.signals` exports. The plural legacy
carry helper returns only a score; the singular helper returns the score/raw pair.

```python
classic_score, raw_classic = signals.compute_classic_momentum_alpha(
    prices, returns_freq="ME", lookback_periods=12, skip_periods=1,
)
reversal_score, raw_reversal = signals.compute_residual_reversal_alpha(
    prices, benchmark_price=benchmark, returns_freq="ME", long_span=1,
)
carry_score, raw_carry = signals.compute_ra_carry_alpha(
    prices, carry=carry, returns_freq="ME", vol_span=13,
)
legacy_carry_score = alphas.compute_ra_carry_alphas(
    prices, carry=carry, returns_freq="ME", vol_span=13,
)
```

### A factor model for managers' residuals

This small HCGL fit supplies dated loadings and clusters. The first manager price endpoint
is retained before the first beta date so manager returns can begin in January 2023.

```python
factor_estimator = opt.FactorCovarEstimator(
    rebalancing_freq="YE", factor_returns_freq="ME", factor_covar_span=36,
    lasso_model=opt.LassoModel(
        model_type=opt.LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=1.0e-5, span=36, warmup_period=36, demean=True, solver="CLARABEL",
    ),
)
rolling_data = factor_estimator.fit_rolling_factor_covars(
    risk_factor_prices=factor_prices,
    asset_returns_dict={
        "ME": qis.to_returns(prices, freq="ME", is_log_returns=True, drop_first=True)
    },
    assets=prices.columns,
    time_period=qis.TimePeriod("2022-12-31", "2024-12-31"),
)
taa_covar_data = rolling_data
asset_prices = prices.loc["2022-11-30":]
```

```python
from optimalportfolios.alphas import compute_managers_alpha

score, raw_alpha = compute_managers_alpha(
    prices=asset_prices,
    risk_factor_prices=factor_prices,
    estimated_betas=rolling_data.get_y_betas(),
    returns_freq='ME',
    alpha_span=12,
)
mgr_score = score
```

### Cluster Signal Usage

The cluster-extraction example uses the same fitted collection and the explicit asset list.

```python
from optimalportfolios.alphas.signals import extract_rolling_clusters

rolling_clusters = extract_rolling_clusters(
    rolling_covar_data=taa_covar_data,
    assets=prices.columns.tolist(),
)
# Dict[pd.Timestamp, pd.Series]  →  {date: pd.Series(ticker → cluster_id)}
```

The direct utility call and signal-specific wrapper use the same raw momentum:

```python
from optimalportfolios.alphas.signals.utils import score_within_clusters

cluster_score = score_within_clusters(
    raw_signal=raw_momentum,        # T × N DataFrame
    rolling_clusters=rolling_clusters,
)
```

```python
from optimalportfolios.alphas import compute_momentum_cluster_alpha

score, raw = compute_momentum_cluster_alpha(
    prices=prices,
    benchmark_price=benchmark,
    rolling_clusters=rolling_clusters,
    returns_freq='ME',
    long_span=12,
)
mom_cluster_score, raw_momentum_cluster = score, raw
beta_cluster_score, raw_beta_cluster = signals.compute_low_beta_cluster_alpha(
    prices, benchmark_price=benchmark, rolling_clusters=rolling_clusters,
)
res_cluster_score, raw_residual_cluster = signals.compute_residual_momentum_cluster_alpha(
    prices, benchmark_price=benchmark, rolling_clusters=rolling_clusters,
)
classic_cluster_score, raw_classic_cluster = signals.compute_classic_momentum_cluster_alpha(
    prices, rolling_clusters=rolling_clusters,
)
reversal_cluster_score, raw_reversal_cluster = signals.compute_residual_reversal_cluster_alpha(
    prices, benchmark_price=benchmark, rolling_clusters=rolling_clusters,
)
carry_cluster_score, raw_carry_cluster = signals.compute_ra_carry_cluster_alpha(
    prices, carry=carry, returns_freq="ME", rolling_clusters=rolling_clusters,
)
```

### A visible scoring comparison

Use six deliberately simple raw values to isolate scoring from price estimation. Assets A–D
form a four-member cluster; E–F form a two-member cluster. Under the default threshold,
A–D use within-cluster sample statistics and E–F use the full assigned-universe sample statistics.

```python
probe_date = pd.Timestamp("2024-12-31")
raw_probe = pd.DataFrame(
    [[-4.0, -2.0, 0.0, 2.0, 4.0, 8.0]], index=[probe_date], columns=list("ABCDEF"),
)
probe_clusters = {
    probe_date: pd.Series(["Large"] * 4 + ["Small"] * 2, index=raw_probe.columns),
}
standard_probe = qis.df_to_cross_sectional_score(raw_probe)
cluster_probe = signals.score_within_clusters(raw_probe, probe_clusters, min_cluster_size=3)
score_comparison = pd.DataFrame({
    "Raw": raw_probe.loc[probe_date],
    "Standard": standard_probe.loc[probe_date],
    "Cluster": cluster_probe.loc[probe_date],
})
print(score_comparison.round(6))
```

| Asset | Raw | Standard score | Cluster score |
|---|---:|---:|---:|
| A | -4 | -1.517929 | -1.161895 |
| B | -2 | -0.889821 | -0.387298 |
| C | 0 | -0.261712 | 0.387298 |
| D | 2 | 0.366397 | 1.161895 |
| E | 4 | 0.994505 | 0.617213 |
| F | 8 | 1.308560 | 1.543033 |

F's input is clipped to 5 only in the standard calculation. Its nonzero cluster score
comes from fallback statistics, despite membership in a small cluster.

### Cadence and fixed-group examples

G and H now report only quarterly. The hard skip is one **native** period for each bucket.

```python
mixed_prices = prices.copy()
mixed_prices.loc[~mixed_prices.index.is_quarter_end, ["G", "H"]] = np.nan
return_frequencies = pd.Series("ME", index=prices.columns)
return_frequencies.loc[["G", "H"]] = "QE"
mixed_score, mixed_raw = signals.compute_classic_momentum_alpha(
    mixed_prices, returns_freq=return_frequencies,
    lookback_periods={"ME": 12, "QE": 4}, skip_periods={"ME": 1, "QE": 1},
)
```

An explicit benchmark leaves the raw single-cadence momentum unchanged when fixed groups
are introduced; only its comparison set changes.

```python
group_data = pd.Series(["Group 1"] * 4 + ["Group 2"] * 4, index=prices.columns)
group_score, group_raw = signals.compute_momentum_alpha(
    prices, benchmark_price=benchmark, returns_freq="ME", group_data=group_data,
)
```

### AlphasData

This illustrative 50/50 blend is supplied by the example. It is not an aggregation default
or a forecast calibration. The container retains all the existing component fields.

```python
# An illustrative application-level blend; the container does not choose this rule.
combined_scores = 0.5 * mom_score + 0.5 * beta_score
cluster_assignments = pd.DataFrame.from_dict(rolling_clusters, orient="index")
cluster_assignments = cluster_assignments.reindex(prices.index, method="ffill")
from optimalportfolios.alphas import AlphasData

data = AlphasData(
    alpha_scores=combined_scores,                           # (T × N) — input to optimiser
    momentum_score=mom_score,                               # fixed-group component scores
    momentum_cluster_score=mom_cluster_score,               # cluster component scores
    beta_score=beta_score,
    beta_cluster_score=beta_cluster_score,
    managers_scores=mgr_score,
    residual_momentum_score=res_score,
    residual_momentum_cluster_score=res_cluster_score,
    momentum=raw_momentum,                                  # raw signals
    momentum_cluster=raw_momentum_cluster,
    beta=raw_beta,
    beta_cluster=raw_beta_cluster,
    managers_alphas=raw_alpha,
    residual_momentum=raw_residual,
    residual_momentum_cluster=raw_residual_cluster,
    clusters=cluster_assignments,                           # T × N cluster IDs
)

# snapshot at a single date (all available components)
snapshot = data.get_alphas_snapshot(date=pd.Timestamp('2024-12-31'))

# export to dict (only non-None fields, safe for Excel)
output = data.to_dict()
```

At 2024-12-31 the snapshot has eight asset rows and 16 populated component columns.
`to_dict()` returns those 16 populated fields without writing a file. The container has
no dedicated classic-momentum, reversal or carry fields; keep additional panels in a named
dictionary when evaluating those signals.

### Profiling and diagnostics

The rank profiler compares top-quarter selections with an equal-weight-all benchmark.
This example uses quarterly rebalancing and the default zero transaction costs. Every leg
runs through QIS's holdings backtester. The separate diagnostic call evaluates signals against
future log-return horizons retrospectively; these estimates are not inputs known at formation time.

```python
profiles = alphas.backtest_alpha_rank_portfolio(
    prices=prices, alpha_scores={"Momentum": mom_score, "Low beta": beta_score},
    quantile=0.25, rebalancing_freq="QE",
    time_period=qis.TimePeriod("2021-12-31", "2024-12-31"),
)
component_panels = alphas.signal_diagnostics_panel(data)
diagnostics = alphas.run_signal_diagnostics(
    asset_returns_dict={
        "ME": qis.to_returns(prices, freq="ME", is_log_returns=True, drop_first=True)
    },
    signal=mom_score, horizons=(1, 3), is_log_returns=True,
)
mean_dates = pd.date_range("2023-03-31", "2024-12-31", freq="QE")
annual_log_means = alphas.estimate_rolling_ewma_means(
    prices, rebalancing_dates=list(mean_dates), returns_freq="ME", span=12, annualize=True,
)
```

The profile contains two signal strategies and one benchmark. The diagnostic panel enumerates
eight populated score components; the rolling-mean result has eight requested dates and eight assets.
These structural results establish that the workflow executes, not that the signals predict returns.

## Implementation in optimalportfolios

### Architecture

The paired standard and cluster constructors now live together in their owning signal module.

```text
src/optimalportfolios/alphas/
  signals/
    momentum.py, classic_momentum.py, low_beta.py
    residual_momentum.py, residual_reversal.py, managers_alpha.py
    carry.py, rolling_ewma_mean.py, utils.py
    tests/                  offline signal contracts
    run_local/signals_run.py source-checkout diagnostics
  alpha_data.py             AlphasData
  profile/                  rank-selection backtests and reports
  signal_diagnostics.py     adapters for QIS signal diagnostics
  backtest_alphas.py         additional alpha-backtest helpers
  tests/                    cross-cutting contracts
```

[Signal exports](../src/optimalportfolios/alphas/signals/__init__.py) and
[alpha-layer exports](../src/optimalportfolios/alphas/__init__.py) are the import contracts.
The development runner is `python -m optimalportfolios.alphas.signals.run_local.signals_run`
in a checkout; its `Locals` enum selects manual scenarios. It is not the offline verification
entry point for this article.

### Signal Matrix

Import the paired constructors from `optimalportfolios.alphas.signals`.
All 13 constructors below return `(score, raw)`; managers alpha has no cluster counterpart.

| Family | Standard and cluster entry points |
|---|---|
| EWMA momentum | `compute_momentum_alpha`; `compute_momentum_cluster_alpha` |
| Classic momentum | `compute_classic_momentum_alpha`; `compute_classic_momentum_cluster_alpha` |
| Low beta | `compute_low_beta_alpha`; `compute_low_beta_cluster_alpha` |
| Residual momentum | `compute_residual_momentum_alpha`; `compute_residual_momentum_cluster_alpha` |
| Residual reversal | `compute_residual_reversal_alpha`; `compute_residual_reversal_cluster_alpha` |
| Carry | `compute_ra_carry_alpha`; `compute_ra_carry_cluster_alpha` |
| Managers alpha | `compute_managers_alpha` |

`compute_classic_momentum_from_returns` returns only a raw fixed-window sum.
`compute_ra_carry_alphas` returns only the legacy global carry score and is also exported
from `optimalportfolios.alphas`. The singular carry pair is exported from the `signals`
subpackage; do not assume every subpackage export exists at its parent.

See the [momentum](../src/optimalportfolios/alphas/signals/momentum.py),
[classic momentum](../src/optimalportfolios/alphas/signals/classic_momentum.py),
[low-beta](../src/optimalportfolios/alphas/signals/low_beta.py),
[residual-momentum](../src/optimalportfolios/alphas/signals/residual_momentum.py),
[reversal](../src/optimalportfolios/alphas/signals/residual_reversal.py),
[carry](../src/optimalportfolios/alphas/signals/carry.py), and
[managers-alpha](../src/optimalportfolios/alphas/signals/managers_alpha.py) sources for
individual signatures. They do not share one universal interface.

### AlphasData Fields

| Stored fields | Contents |
|---|---|
| `alpha_scores` | Required application-supplied combined panel. |
| `momentum` / `momentum_score` | Raw/scored EWMA momentum. |
| `momentum_cluster` / `momentum_cluster_score` | Cluster momentum components. |
| `beta` / `beta_score` | Raw beta and low-beta score. |
| `beta_cluster` / `beta_cluster_score` | Cluster low-beta components. |
| `managers_alphas` / `managers_scores` | Smoothed residual and its uncentered score. |
| `residual_momentum` / `residual_momentum_score` | Residual momentum components. |
| `residual_momentum_cluster` / `residual_momentum_cluster_score` | Cluster residual components. |
| `clusters` | Dated labels; values may be strings such as `ME:1`. |

Only `alpha_scores` is required. The
[container source](../src/optimalportfolios/alphas/alpha_data.py) defines the exact fields.
`get_alphas_snapshot` requires the requested date in `alpha_scores`. If that date is absent
from another populated component, it takes that component's **last row**, even if it is later.
Align components explicitly before historical snapshots; this convenience method is not a causal
as-of join.

### Evaluation entry points and verification context

[Profiling](../src/optimalportfolios/alphas/profile/core.py) provides
`backtest_alpha_rank_portfolio`, `compute_top_quantile_equal_weights`,
`compute_alpha_rank_analysis_table` and `generate_alpha_profile_report`.
Signal-specific `profile_*` functions and `profile_alpha_signals` are exported from
`optimalportfolios.alphas`. Report output belongs in an explicit local output directory.

[Signal diagnostics](../src/optimalportfolios/alphas/signal_diagnostics.py) provides
`signal_diagnostics_panel`, `run_signal_diagnostics`,
`run_signal_diagnostics_per_component` and `compare_signal_diagnostics`.
Statistical calculations and plotting remain in
[QIS](https://github.com/ArturSepp/QuantInvestStrats); factor fitting and cluster discovery
remain in [FactorLasso](https://github.com/ArturSepp/FactorLasso).

On the maintainer's Windows host, first run the repository's C-local setup and use
`C:\Python\OptimalPortfolios312\Scripts\python.exe`. Execute source checks and tests from a
C-local source export, as specified by [AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md):

```text
python tools/check_docs.py --files docs/alphas_module_readme.md
python -m pytest src/optimalportfolios/tests/alpha_signals_documentation_test.py
```

The local 2026-09-14 verification uses OptimalPortfolios 7.6.0 working source,
QIS 5.26.0, FactorLasso 0.18.0, pandas 3.0.5, NumPy 2.5.2, CVXPY 1.9.2 and CLARABEL 0.11.1.
This does not certify the existing lockfile's QIS 5.22.3 environment.

## Interpretation and limitations

Use sufficiently long histories and distinguish missing observations from zero signals.
Warm-up, zero-beta masking, degenerate group dispersion and cluster fallback have different
effects. The ranking helper currently checks non-missing prices and scores, not full positivity
and finiteness; prevalidate inputs rather than treating this mask as a complete eligibility rule.
Ties are resolved by column order. A top-quantile profile ranks and equal-weights assets; it
does not optimise their covariance or control factor exposures.

Avoid `MeanAdjType.INSAMPLE` in historical trading signals: it uses the full-sample mean.
Future-observation checks pass for EWMA momentum, classic momentum and carry, including
their cluster variants, on this fixture. The three beta-based families have the default
initialisation limitation described under Low Beta; their explicit `NONE` variants pass.
These checks cannot establish when a supplied price, carry input or factor loading was
actually available. The pending default-path checks remain strict expected failures so a
future fix triggers a documentation review.
Changing the reporting cadence changes both the observations and the meaning of scalar spans.

### Empirical Findings

A performance claim needs a reproducible universe, sample, data vintage, signal settings,
benchmark, rebalance rule, cost convention and comparison statistic. The worked examples here
verify definitions and execution. They do not establish that cluster scoring outperforms fixed
groups or that residual signals identify manager skill. Use the profiling and diagnostics
interfaces to evaluate a specified dataset and disclose the resulting design choices.

## See also

- [Mixed-frequency data](mixed_frequency_data.md): native cadences, hard lookbacks and timing.
- [Covariance estimators](covariance_estimators.md): factor models and annualisation.
- [CSV factor risk model](rolling_factor_covar_from_csv.md): persisted input and loading contracts.
- [Rolling backtests](rolling_backtests.md): applying formation-date decisions to holdings.
- [Examples](examples_readme.md): profiling workflows and other entry points.
- [API reference](api.rst): generated public signatures.

## References

- Blitz, D., Huij, J. and Martens, M. (2011).
  [Residual Momentum](https://repub.eur.nl/pub/22252/).
  *Journal of Empirical Finance*, 18(3), 506–521.
- Blitz, D., Huij, J., Lansdorp, S. and Verbeek, M. (2013).
  [Short-Term Residual Reversal](https://pure.eur.nl/en/publications/short-term-residual-reversal/).
  *Journal of Financial Markets*, 16(3), 477–504.
- Sepp, A., Ossa, I. and Kastenholz, M. (2026).
  [Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios](https://www.pm-research.com/content/iijpormgmt/52/4/86).
  *The Journal of Portfolio Management*, 52(4), 86–120.
- Sepp, A., Hansen, E. and Kastenholz, M. (2026).
  [Capital Market Assumptions and Strategic Asset Allocation Using Multi-Asset Tradable Factors](https://ssrn.com/abstract=6785958).
  Working paper.
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
