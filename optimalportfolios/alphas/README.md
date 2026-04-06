# Alphas Module — `optimalportfolios.alphas`

Alpha signal building blocks for the `optimalportfolios` package.

This module provides individual alpha signal computation functions and
the `AlphasData` container. Each signal function is a standalone building
block that computes a cross-sectional score from asset prices — ready for
use in any portfolio optimisation workflow.

**This module does not contain aggregation logic.** The routing of assets
to signals, combination rules, and product-specific configurations belong
in the consuming application (e.g., a private `rosaa` package). This
module provides the bricks; the application builds the house.

## Architecture

```
optimalportfolios/alphas/
├── signals/
│   ├── momentum.py                    # compute_momentum_alpha()
│   ├── momentum_cluster.py            # compute_momentum_cluster_alpha()
│   ├── low_beta.py                    # compute_low_beta_alpha()
│   ├── low_beta_cluster.py            # compute_low_beta_cluster_alpha()
│   ├── residual_momentum.py           # compute_residual_momentum_alpha()
│   ├── residual_momentum_cluster.py   # compute_residual_momentum_cluster_alpha()
│   │                                  # + extract_rolling_clusters()
│   ├── managers_alpha.py              # compute_managers_alpha()
│   ├── carry.py                       # compute_ra_carry_alphas()
│   ├── rolling_ewma_mean.py           # estimate_rolling_ewma_means()
│   ├── utils.py                       # score_within_clusters()
│   └── tests/
│       └── signals_test.py
├── alpha_data.py                      # AlphasData container
├── backtest_alphas.py                 # Backtesting harness
└── README.md
```

## Naming Conventions

The module distinguishes three stages in the alpha pipeline:

| Stage | What it is | Example |
|-------|-----------|---------| 
| **Raw signal** | Observable quantity with units | Cumulative return, EWMA beta, regression residual |
| **Score** | Cross-sectional rank/z-score, dimensionless | Momentum z-score, negated beta rank |
| **Alpha** | Portfolio-ready signal after combination and CDF mapping | Combined score mapped to [-1, 1] |

Pipeline: **raw signal → score → alpha**.

Functions are named `compute_*_alpha()` because their primary output is
a score ready for aggregation into the final alpha vector. The raw signal
is returned as the second element for diagnostics.


## Signal Matrix

The module provides 8 signal functions across two scoring modes:

| Signal | Fixed groups | Cluster-scored | Parameters |
|--------|:---:|:---:|---|
| Momentum | `compute_momentum_alpha` | `compute_momentum_cluster_alpha` | `long_span`, `short_span`, `vol_span` |
| Low beta | `compute_low_beta_alpha` | `compute_low_beta_cluster_alpha` | `beta_span` |
| Residual momentum | `compute_residual_momentum_alpha` | `compute_residual_momentum_cluster_alpha` | `beta_span`, `long_span`, `short_span`, `vol_span` |
| Managers alpha | `compute_managers_alpha` | — | `alpha_span` |

**Fixed-group** signals score assets within user-defined groups (e.g.,
"Equity", "Fixed Income") passed via `group_data: pd.Series`.

**Cluster-scored** signals score within time-varying statistical clusters
extracted from the HCGL/LASSO covariance estimator. See the
[Cluster-Based Scoring](#cluster-based-scoring) section below.


## Signal Functions

Each signal function has the same interface:

```python
def compute_*_alpha(
    prices: pd.DataFrame,
    ...,                                    # signal-specific inputs
    returns_freq: Union[str, pd.Series],    # single or mixed frequency
    group_data: Optional[pd.Series],        # for within-group scoring
    **signal_params,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (score, raw_signal)"""
```

Cluster variants replace `group_data` with `rolling_clusters`:

```python
def compute_*_cluster_alpha(
    prices: pd.DataFrame,
    ...,
    rolling_clusters: Dict[pd.Timestamp, pd.Series],
    returns_freq: Union[str, pd.Series],
    **signal_params,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (cluster_score, raw_signal)"""
```


### Momentum

Cross-sectional momentum score from EWMA-filtered risk-adjusted excess returns.

```python
from optimalportfolios.alphas import compute_momentum_alpha

score, raw = compute_momentum_alpha(
    prices=prices,
    benchmark_price=benchmark,
    returns_freq='ME',
    long_span=12,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `long_span` | 12 | EWMA span for the long momentum signal |
| `short_span` | None | Short-term reversal subtraction (None = disabled) |
| `vol_span` | 13 | EWMA span for vol normalisation (None = disabled) |

Pipeline: `returns → excess returns (vs benchmark) → EWMA long/short filtered RA returns → cross-sectional score`


### Low Beta

Cross-sectional low-beta score ("betting against beta").

```python
from optimalportfolios.alphas import compute_low_beta_alpha

score, raw_beta = compute_low_beta_alpha(
    prices=prices,
    benchmark_price=benchmark,
    returns_freq='ME',
    beta_span=12,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta_span` | 12 | EWMA span for beta estimation |

Pipeline: `returns → EWMA regression beta → negate → cross-sectional score`


### Residual Momentum

Cross-sectional residual momentum score. Strips out single-benchmark
beta exposure from asset returns, then applies EWMA long/short filtering
(same as momentum.py). Sits between total-return momentum (no beta
adjustment) and managers alpha (full MATF adjustment).

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
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta_span` | 12 | EWMA span for benchmark beta estimation |
| `long_span` | 12 | EWMA span for the long momentum signal |
| `short_span` | None | Short-term reversal subtraction (None = disabled) |
| `vol_span` | 13 | EWMA span for vol normalisation (None = disabled) |

Pipeline: `returns → EWMA beta (lagged) → residual = r_t - β̂_{t-1}·r_bench_t → EWMA long/short filtered RA returns → cross-sectional score`

References: Blitz, Huij & Martens (2011), "Residual Momentum", *Journal of Empirical Finance*, 18, 506-521.


### Managers Alpha

Cross-sectional score from MATF factor model regression residuals.

```python
from optimalportfolios.alphas import compute_managers_alpha

score, raw_alpha = compute_managers_alpha(
    prices=asset_prices,
    risk_factor_prices=factor_prices,
    estimated_betas=rolling_data.get_y_betas(),
    returns_freq='ME',
    alpha_span=12,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_span` | 12 | EWMA span for smoothing regression residuals |


## Cluster-Based Scoring

The `_cluster` signal variants use time-varying statistical clusters
instead of fixed user-defined groups for cross-sectional scoring.

### Motivation

Fixed asset-class groups (Equity, Fixed Income, Alternatives) impose a
subjective structure on the cross-sectional scoring. Within-group z-scores
assume assets in the same group are comparable — but a global equity ETF
and a regional small-cap fund may behave very differently.

Cluster-based scoring replaces this with data-driven groups that evolve
over time: assets with similar factor exposures are clustered together,
and scoring happens within each cluster. This naturally handles regime
shifts (e.g., a credit fund behaving more like equity during stress) and
avoids the problem of heterogeneous groups inflating scores for outliers.

### How Clusters Are Derived

Clusters come from the HCGL/LASSO covariance estimator
(`FactorCovarEstimator` with `LassoModelType.GROUP_LASSO_CLUSTERS`):

1. At each estimation date, the LASSO solver produces sparse factor
   loadings for each asset.
2. Hierarchical clustering on the factor correlation matrix groups assets
   with similar systematic risk profiles.
3. These clusters change at each rebalancing date as factor exposures
   evolve.

The clusters are stored in `RollingFactorCovarData` and extracted via
`extract_rolling_clusters()`:

```python
from optimalportfolios.alphas.signals.residual_momentum_cluster import extract_rolling_clusters

rolling_clusters = extract_rolling_clusters(
    rolling_covar_data=taa_covar_data,
    assets=universe_data.get_taa_prices().columns.tolist(),
)
# Dict[pd.Timestamp, pd.Series]  →  {date: pd.Series(ticker → cluster_id)}
```

### Scoring Logic

The shared `score_within_clusters()` function in `utils.py` handles the
row-by-row scoring:

```python
from optimalportfolios.alphas.signals.utils import score_within_clusters

cluster_score = score_within_clusters(
    raw_signal=raw_momentum,        # T × N DataFrame
    rolling_clusters=rolling_clusters,
)
```

For each date:
- Look up the cluster assignment at the most recent estimation date.
- Score within each cluster independently (cross-sectional z-score).
- Singleton clusters (1 member) receive score 0.0.
- Dates before the first cluster estimation receive score 0.0.
- If `rolling_clusters` is empty, falls back to global scoring.

### Cluster Signal Usage

```python
from optimalportfolios.alphas import compute_momentum_cluster_alpha

score, raw = compute_momentum_cluster_alpha(
    prices=prices,
    benchmark_price=benchmark,
    rolling_clusters=rolling_clusters,
    returns_freq='ME',
    long_span=12,
)
```

### Empirical Findings

In ROSAA APAC equity backtests (2005-2026), cluster-scored signals showed
marginal improvement over fixed groups with approximately 5pp turnover
increase (55% → 60% p.a.). The cluster variants are most useful for
equity-only universes where regional groupings may not capture the true
return structure; for diversified multi-asset portfolios the improvement
is smaller.


## Mixed-Frequency Support

All signal functions accept `returns_freq` as string or per-asset `pd.Series`:

```python
# mixed: equities monthly, alternatives quarterly
returns_freq = pd.Series({'SPY': 'ME', 'EZU': 'ME', 'HF_Macro': 'QE', 'PE': 'QE'})
score, raw = compute_momentum_alpha(prices, returns_freq=returns_freq, ...)
```

Mixed-frequency dispatch computes returns at each asset's native frequency,
then concatenates and aligns before scoring.


## Within-Group Scoring (Fixed Groups)

When `group_data` is provided to the fixed-group variants, cross-sectional
scoring is computed within each group independently:

```python
group_data = pd.Series({'SPY': 'Equity', 'TLT': 'Bonds', 'GLD': 'Alts'})
score, raw = compute_momentum_alpha(prices, group_data=group_data, ...)
```


## Signal Comparison

| Signal | Beta adjustment | Scoring | Use case |
|--------|----------------|---------|----------|
| **Momentum** | None (benchmark-relative excess) | Fixed groups | General trend following |
| **Momentum Cluster** | None (benchmark-relative excess) | Data-driven clusters | Trend following with adaptive grouping |
| **Low Beta** | Single benchmark EWMA beta | Fixed groups | Defensive tilt |
| **Low Beta Cluster** | Single benchmark EWMA beta | Data-driven clusters | Defensive tilt with adaptive grouping |
| **Residual Momentum** | Single benchmark EWMA beta | Fixed groups | Beta-neutral momentum for multi-asset TAA |
| **Residual Momentum Cluster** | Single benchmark EWMA beta | Data-driven clusters | Beta-neutral momentum with adaptive grouping |
| **Managers Alpha** | Full MATF factor betas | Fixed groups | Fund/manager skill isolation |


## AlphasData

Container for alpha computation results:

```python
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

### AlphasData Fields

| Field | Type | Description |
|-------|------|-------------|
| `alpha_scores` | DataFrame | Final combined scores — input to TAA optimiser |
| `momentum` / `momentum_score` | DataFrame | Raw / scored momentum (fixed groups) |
| `momentum_cluster` / `momentum_cluster_score` | DataFrame | Raw / scored momentum (clusters) |
| `beta` / `beta_score` | DataFrame | Raw / scored low-beta (fixed groups) |
| `beta_cluster` / `beta_cluster_score` | DataFrame | Raw / scored low-beta (clusters) |
| `managers_alphas` / `managers_scores` | DataFrame | Raw / scored managers alpha |
| `residual_momentum` / `residual_momentum_score` | DataFrame | Raw / scored residual momentum (fixed groups) |
| `residual_momentum_cluster` / `residual_momentum_cluster_score` | DataFrame | Raw / scored residual momentum (clusters) |
| `clusters` | DataFrame | Time-varying cluster assignments (T × N) |

All fields except `alpha_scores` are optional (default None).


## References

Blitz D., Huij J., Martens M. (2011),
"Residual Momentum",
*Journal of Empirical Finance*, 18, 506-521.

Sepp A., Ossa I., and Kastenholz M. (2026),
"Robust Optimization of Strategic and Tactical Asset Allocation
for Multi-Asset Portfolios",
*The Journal of Portfolio Management*, 52(4), 86-120.

Sepp A., Hansen E., and Kastenholz M. (2026),
"Capital Market Assumptions and Strategic Asset Allocation
Using Multi-Asset Tradable Factors",
*Working Paper*.