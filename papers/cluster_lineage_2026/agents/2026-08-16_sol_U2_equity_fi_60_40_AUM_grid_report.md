# U2 Equity/Fixed-Income 60/40 long-short AUM grid

**Date:** 2026-08-16  
**Status:** complete; 9/9 acceptance checks pass

## Outcome

U2 is restricted to Equity and Fixed Income funds. Commodity, Multi Asset, Digital Assets,
Real Estate, and Cash funds are removed before signal benchmarking, covariance estimation,
cluster discovery, ranking, and backtesting. Both signed sides carry 60% Equity and 40%
Fixed Income gross exposure.

The class restriction does not make the cluster-relative book beat its matched global-sleeve
control. Cluster net return, Sharpe, and turnover are worse at every AUM cutoff, although
cluster volatility is lower. No AUM cutoff is the least-bad cluster-relative result; USD50m
is the worst, and USD100m partially recovers. Performance is not monotone in cutoff size.

## Fixed specification

- Included official classes: Equity and Fixed Income only.
- Gross budget on each side: 60% Equity / 40% Fixed Income.
- Signal: ROSAA production risk-adjusted momentum.
- Clusters: W-THU covariance returns, EWMA span 156.
- Selection: q=25% on long and short sides.
- Cluster construction: equal budget across available clusters within each retained sleeve.
- Global control: rank within each retained sleeve and equal-weight selected funds.
- Rebalancing: every two months, lag 1.
- Costs: 20 bp one way.
- Headline window: 2009-08-31 through 2026-06-30.

The partitions are newly fit on the class-restricted point-in-time universe; removed funds do
not influence the retained funds' correlations or cluster assignments.

## Performance as a function of AUM cutoff

| AUM cutoff | Global net | Cluster net | Cluster − global | Global vol | Cluster vol | Global Sharpe | Cluster Sharpe | Global turnover | Cluster turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| None | -0.528% | -1.323% | **-79.47 bp** | 6.755% | **4.307%** | -0.045 | -0.287 | **2.123x** | 2.610x |
| >USD50m | -0.675% | -2.417% | **-174.13 bp** | 6.709% | **4.527%** | -0.067 | -0.517 | **2.134x** | 2.620x |
| >USD100m | -0.722% | -2.185% | **-146.30 bp** | 6.666% | **4.699%** | -0.075 | -0.446 | **2.154x** | 2.581x |

### Gross return and costs

| AUM cutoff | Global gross | Cluster gross | Gross gap | Additional cluster cost drag | Net gap |
|---|---:|---:|---:|---:|---:|
| None | 1.173% | 0.761% | -41.23 bp | +38.24 bp | -79.47 bp |
| >USD50m | 1.033% | -0.344% | -137.66 bp | +36.47 bp | -174.13 bp |
| >USD100m | 1.002% | -0.141% | -114.25 bp | +32.05 bp | -146.30 bp |

The result is not solely a trading-cost problem. Cluster gross return already trails global
at every cutoff. Higher turnover then widens the gap by approximately 32–38 bp/year.

Cluster volatility rises as the AUM cutoff tightens—4.307%, 4.527%, then 4.699%—while global
volatility declines slightly. The AUM restriction therefore removes some of the cluster
book's diversification benefit without improving its momentum payoff.

## Eligible breadth

Counts are headline start / median headline date / headline end.

| AUM cutoff | Equity | Fixed Income | Total |
|---|---:|---:|---:|
| None | 125 / 205 / 286 | 24 / 62 / 151 | **149 / 267 / 437** |
| >USD50m | 108 / 178 / 250 | 19 / 49 / 123 | **127 / 227 / 373** |
| >USD100m | 93 / 167 / 238 | 19 / 44 / 111 | **112 / 211 / 349** |

The thinner Fixed Income cross-section is especially important at the start of the sample:
only 19 eligible funds remain under either positive cutoff. Nevertheless, both signed sleeves
remain populated and the exact 60/40 budget checks pass on every date.

## Acceptance

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Included official classes | Equity / Fixed Income | exact | PASS |
| Excluded-class eligible observations | 0 | 0 | PASS |
| Maximum excluded-class weight | 0 | 1e-12 | PASS |
| Partition eligible-member count error | 0 | 0 | PASS |
| Eligible memberships missing from partitions | 0 | 0 | PASS |
| Maximum weight, exposure, and sleeve-budget error | 8.660e-15 | 1e-12 | PASS |
| Maximum signal lookahead days | 0 | 0 | PASS |
| Performance rows | 18 | 18 | PASS |
| Headline sensitivity rows | 3 | 3 | PASS |

All 8/8 deterministic CSV artifacts were byte-identical across two complete cache-first
replays. The final replay took 35.42 seconds. Explicit E/F/W lint reports
`All checks passed!`.

## Reproduction

Runner:

- `papers/cluster_lineage_2026/replication/run_u2_equity_fi_long_short_aum_grid.py`

External output directory:

- `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/covariance_frequency_span_grid/blackrock_us_etfs/equity_fi_60_40_long_short_aum_grid_20260816/`

No files were staged or pushed. The complete `papers/cluster_lineage_2026/` tree remains
gitignored as owner instructed.
