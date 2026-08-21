# Futures ROSAA-production signal specification grid — execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

The requested exact-production momentum grid was run for the 30/30/30/10 futures
long-short strategy at 10 bp one-way costs. The varied dimensions were:

- short/reversal span: `None`, 1, 2, 3;
- volatility span: 13, 26, 52;
- mean adjustment: `MeanAdjType.NONE`, `MeanAdjType.EWMA`;
- cluster fallback: 5, 7, 10;
- q: 20% primary and 25% robustness;
- cluster treatment: baseline and M1-star.

Every cluster row was compared with a global-within-sleeve rank using exactly the same
short span, volatility span, mean adjustment, q, window, and cost. Across the 96 unique
cluster comparisons (24 signal specifications x 2 q values x 2 cluster methods):

| acceptance question | measured |
|---|---:|
| cluster net-return wins versus matched global | **0 / 96** |
| cluster Sharpe wins versus matched global | **0 / 96** |
| cluster mean-variance wins versus matched global | **0 / 96** |

Changing the signal parameters can make the cluster strategy profitable, but none of the
requested cells makes it outperform its fair same-signal global benchmark.

## Best specification for the stated objective

The closest cluster-to-global result is:

| component | value |
|---|---|
| cluster method | M1-star |
| q | 25% |
| short span | None |
| volatility span | 13 |
| mean adjustment | EWMA |
| fallback | 5, 7, or 10 — identical portfolio |

| metric | M1-star cluster | matched global | cluster minus global |
|---|---:|---:|---:|
| net return/year | 0.3670% | 0.5913% | **-22.42 bp** |
| RF=0 Sharpe | 0.0934 | 0.1037 | -0.0103 |
| volatility/year | 4.2353% | 8.1231% | -3.8878 pp |
| one-way turnover/year | 3.0171 | 3.3757 | -0.3586 |

Baseline clusters under the same signal are only slightly farther behind: 0.3569% net/year
and 0.0899 Sharpe, versus global's 0.5913% and 0.1037. The net-return deficit is 23.43 bp.

This no-reversal/vol-13/EWMA cell is the best choice only if the objective is to minimise
the cluster-minus-global gap. It is an exploratory in-sample leader, not a new frozen
production setting.

## Highest standalone cluster return

The cluster portfolio with the highest absolute net return is a different cell:

| component | value |
|---|---|
| cluster method | baseline |
| q | 25% |
| short span | 3 |
| volatility span | 26 |
| mean adjustment | EWMA |

| metric | baseline cluster | matched global | cluster minus global |
|---|---:|---:|---:|
| net return/year | **1.1886%** | 3.2333% | -204.47 bp |
| RF=0 Sharpe | 0.2990 | 0.4252 | -0.1261 |
| volatility/year | 4.3385% | 8.3618% | -4.0233 pp |
| one-way turnover/year | 2.5158 | 2.1381 | +0.3776 |

It is attractive in isolation but is not evidence for cluster outperformance: the same
signal improves global ranks much more strongly.

The highest global return in the grid is q=20%, short span 3, volatility span 13, and
`MeanAdjType.NONE`: 3.9949% net/year with 0.4755 Sharpe. In that cell, baseline clustering
earns 0.7025% with 0.1855 Sharpe and M1-star earns 0.5622% with 0.1549 Sharpe.

## Parameter effects

### Short/reversal span

No reversal is decisively best for staying competitive with global, especially at q=25%.
Average cluster-minus-global net-return gaps across volatility and mean adjustments are:

| q | cluster | no short | short 1 | short 2 | short 3 |
|---:|---|---:|---:|---:|---:|
| 20% | baseline | -106.68 bp | -117.19 bp | -211.87 bp | -211.44 bp |
| 20% | M1-star | -117.48 bp | -125.78 bp | -242.22 bp | -227.12 bp |
| 25% | baseline | **-59.47 bp** | -164.16 bp | -177.19 bp | -197.92 bp |
| 25% | M1-star | **-38.89 bp** | -175.71 bp | -202.30 bp | -216.63 bp |

The reversal component can raise absolute returns in selected cells, but it raises matched
global returns even more. It therefore works against the paper's cluster-outperformance
objective.

### Volatility span

Longer volatility spans improve the average relative gap. For example, at q=25% the
M1-star mean deficit improves from -173.13 bp at vol 13 to -143.92 bp at vol 52; baseline
improves from -161.93 bp to -134.58 bp. However, interactions matter: the single closest
cell uses vol 13 with no reversal and EWMA mean adjustment.

### Mean adjustment

`MeanAdjType.EWMA` is modestly better on average than `NONE`. At q=25%, the average
M1-star deficit is -157.27 bp under EWMA versus -159.50 bp under NONE; baseline is
-145.42 bp versus -153.95 bp. This improvement is real but far too small to produce a win.

### Cluster fallback

Fallback 5, 7, and 10 produce exactly the same portfolio weights in all 192 pairwise
fallback checks:

| diagnostic | measured | tolerance | result |
|---|---:|---:|---|
| maximum score difference versus fallback 5 | 3.2981 | descriptive | scores change |
| maximum portfolio-weight difference | **0.0** | <= 1e-12 | PASS |

The reason is structural. The fallback changes normalization of small-cluster scores, but
the portfolio subsequently applies percentile ranks inside the same clusters. Positive
affine normalization changes score magnitudes without changing within-cluster order. The
fallback dimension therefore creates no new investment portfolio under this construction.

## Grid accounting

| item | measured |
|---|---:|
| raw signal specifications | 24 |
| q values | 2 |
| global unique portfolios | 48 |
| cluster unique portfolios | 96 |
| emitted performance rows including fallback labels | 336 |
| cluster/global comparison rows including fallback labels | 288 |
| fallback invariance rows | 192 |
| worker processes | 4 |
| runtime per complete pass | 726.34 seconds |

Fallback-invariant portfolios were backtested once and then emitted under all three
requested fallback labels only after their weights matched exactly. This avoids pretending
that identical portfolios are independent payoff experiments.

## Acceptance and verification

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| daily source to frozen W-WED reconstruction | 4.649e-16 | <= 1e-15 | PASS |
| monthly signal-return round trip | 1.624e-15 | <= 1e-12 | PASS |
| signal specifications with no look-ahead | 24 / 24 | 100% | PASS |
| performance rows | 336 | exact | PASS |
| construction acceptance rows | 336 / 336 PASS | 100% | PASS |
| maximum within-sleeve group-budget error | 0 | <= 1e-15 | PASS |
| maximum top-level sleeve-budget error | 1.665e-16 | <= 1e-12 | PASS |
| maximum net-exposure error | 2.220e-16 | <= 1e-12 | PASS |
| maximum gross-exposure error | 2.665e-15 | <= 1e-12 | PASS |
| `CUA1 Comdty` maximum absolute weight | 0 | 0 | PASS |
| standalone four-sleeve reconstructions | 336 / 336 PASS | 100% | PASS |
| maximum standalone reconstruction error | 5.551e-17 | <= 1e-12 | PASS |
| fallback weight-invariance checks | 192 / 192 PASS | 100% | PASS |
| base exact-production regression | 6 / 6 PASS | <= 1e-12 | PASS |
| maximum base-regression error | 4.547e-13 | <= 1e-12 | PASS |
| deterministic numerical artifacts | 12 / 12 byte-identical | 100% | PASS |
| focused pytest | 3 / 3 passed | all pass | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |
| EW-all payoff comparison | 0 | 0 | PASS |

The complete four-worker grid was run twice. Numerical artifacts were sorted before
serialization and reproduced byte-for-byte despite differing worker completion order.

The fail-before-pass checkpoint produced the expected missing-runner failure:

```text
ModuleNotFoundError: No module named
'papers.cluster_lineage_2026.replication.run_futures_prod_signal_grid_30303010_10bp'
```

Final verification output:

```text
...                                                                      [100%]
All checks passed!
Futures ROSAA production signal grid: PASS (12/12 deterministic)
```

## Deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_futures_prod_signal_grid_30303010_10bp.py`

Focused regression checks:

- `papers/cluster_lineage_2026/replication/futures_prod_signal_grid_30303010_10bp_test.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\futures_prod_signal_grid_30_30_30_10_10bp_u1_window
```

Primary machine-readable files are `performance.csv`, `comparison_vs_global.csv`,
`comparison_unique_portfolios.csv`, `grid_summary.csv`, `grid_leaders.csv`,
`fallback_invariance.csv`, `signal_diagnostics.csv`, `acceptance.csv`,
`standalone_weight_reconstruction.csv`, `base_spec_regression.csv`,
`source_preflight.csv`, `design.csv`, `runtime.csv`, and `determinism.csv`.

No covariance or cluster model was refit, no cache was altered, no EW-all payoff
comparison was introduced, and no file was staged or pushed.
