# Stability-pooled z-score S4 mechanism report

**Date:** 2026-08-20  
**Executor:** Sol  
**Proposer:** Ben (Monday TAA meeting)  
**Status:** COMPLETE — boundary channel confirmed; turnover channel is mixed

## Outcome

Low stability is concentrated on assets that subsequently change clusters, validating the
boundary interpretation behind Ben's proposal. At the 36-month window, reassigned assets have
mean asset stability 0.506 versus 0.697 for stable assets, and the reassignment rate is 31.74% in
the bottom stability quartile versus 4.91% in the top quartile.

For the selected full-sample V3/36 cell, the direct reassignment component falls by 0.264602 per
year and the direct signal component falls by 0.206420. Reassignment is the larger reduction, but
only 56.2% of the sum of the two absolute direct reductions. The mechanism is therefore
directionally consistent but not isolated: a material 43.8% sits in the signal component. The
signed trade-interaction term offsets 0.400151 of those direct reductions.

## Turnover decomposition

The frozen Metric-11 construction compares actual targets, current scores under the prior
partition, and price-drifted prior targets. Its two direct legs form a triangle bound rather than
an additive identity; the signed residual is reported as the trade-interaction term. The
`total` column below is this target-weight counterfactual quantity, not the native rolling
12-period production turnover reported in S3.

All values are annualized. Deltas are versus V0.

| cell | reassignment | delta | signal | delta | total | delta | trade interaction | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 1.430121 | 0.000000 | 1.645008 | 0.000000 | 0.555076 | 0.000000 | -2.520053 | 0.000000 |
| V1_1x | 1.270849 | -0.159272 | 1.508289 | -0.136718 | 0.513760 | -0.041316 | -2.265379 | +0.254674 |
| V1_2x | 1.325871 | -0.104250 | 1.567097 | -0.077911 | 0.515199 | -0.039877 | -2.377769 | +0.142284 |
| V2_1x | 1.206989 | -0.223132 | 1.451109 | -0.193898 | 0.501825 | -0.053251 | -2.156273 | +0.363780 |
| V2_2x | 1.313234 | -0.116887 | 1.546195 | -0.098813 | 0.508807 | -0.046269 | -2.350622 | +0.169431 |
| V3_1x | 1.165519 | -0.264602 | 1.438587 | -0.206420 | 0.484204 | -0.070871 | -2.119902 | +0.400151 |
| V3_2x | 1.218697 | -0.211424 | 1.496902 | -0.148106 | 0.488320 | -0.066756 | -2.227280 | +0.292774 |

Every pooled cell reduces both direct components. The reassignment reduction is larger than the
signal reduction in all six cells, but none is a pure reassignment result.

## Stability distribution and coverage

| window | granularity | observations | mean w | median w | q05 | q95 | share w<0.50 | share w<0.75 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 36 | asset | 34,041 | 0.690811 | 0.733333 | 0.243056 | 1.000000 | 21.10% | 52.66% |
| 36 | cluster | 3,145 | 0.741706 | 0.776455 | 0.305556 | 1.000000 | 17.49% | 46.77% |
| 72 | asset | 34,041 | 0.652397 | 0.685185 | 0.216342 | 1.000000 | 25.46% | 61.33% |
| 72 | cluster | 3,145 | 0.716546 | 0.720800 | 0.279773 | 1.000000 | 19.43% | 53.35% |

Coverage is 100% of active assets on every date for both windows. The first 11 partition dates
use the mandated `w=1` short-history fallback; estimated pooling begins on date 12. The 22
undefined size correlations are exactly those two sets of 11 constant-`w` fallback dates, which
also explains the two NumPy constant-series warnings in the run log.

## Boundary and granularity diagnostics

| window | mean w, reassigned | mean w, stable | reassignment, bottom-w quartile | reassignment, top-w quartile | mean within-cluster asset-w std |
|---:|---:|---:|---:|---:|---:|
| 36 | 0.505973 | 0.696774 | 31.74% | 4.91% | 0.099912 |
| 72 | 0.488598 | 0.654348 | 29.17% | 6.07% | 0.105558 |

The within-cluster asset-level dispersion near 0.10 is the information V2 retains and V1
deliberately smears across a cluster. The boundary spread is economically large at both windows.

## Cluster-size confound

| window | finite dates | median corr(cluster size, w) | share with abs(corr)>0.5 | persistent flag |
|---:|---:|---:|---:|---|
| 36 | 273 | -0.189313 | 15.38% | false |
| 72 | 273 | -0.214754 | 20.51% | false |

The predeclared persistent-confound threshold is not breached. The negative median association is
reported rather than interpreted away, but its magnitude is well below 0.5.

## Acceptance

| check | measured | tolerance | status |
|---|---:|---:|---|
| stability coverage | 100% on every date | 100% | PASS |
| short-history fallback | first 11 dates per window | fewer than 12 | PASS |
| reassignment reduction, selected cell | -0.264602/year | below 0 | PASS |
| reassignment larger than signal reduction | 56.2% vs 43.8% | reassignment share >50% | PASS WITH MIXED MECHANISM |
| boundary concentration, 36 months | 31.74% vs 4.91% reassignment | bottom > top | PASS |
| boundary concentration, 72 months | 29.17% vs 6.07% reassignment | bottom > top | PASS |
| persistent size-w confound | false for 2/2 windows | false | PASS |

Evidence files are `turnover_decomposition.csv`, `turnover_decomposition_panel.csv`,
`stability_distribution.csv`, `stability_coverage.csv`, `stability_long.csv`,
`boundary_diagnostic.csv`, `within_cluster_dispersion.csv`, `size_w_correlation.csv`, and
`size_w_confound_summary.csv` under the S3 output root. No files were staged, committed, pushed,
tagged, or released.
