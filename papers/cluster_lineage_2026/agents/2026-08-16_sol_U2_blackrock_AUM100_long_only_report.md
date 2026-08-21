# U2 BlackRock AUM100 long-only signal-ranking report

**Date:** 2026-08-16  
**Status:** complete diagnostic; 6/6 acceptance checks pass; superseded as the primary U2
signal experiment by the owner's return to long-short on 2026-08-16

The numerical result remains valid as a labelled long-only diagnostic. It is not the current
U2 primary specification. The replacement grid and attribution are in
`2026-08-16_sol_U2_long_short_AUM_attribution_report.md`.

## Outcome

Under the owner-selected long-only U2 construction, the matched 50/30/20 global-sleeve
rank clearly outperforms the cluster-contained rank. Over 2009-08-31 through 2026-06-30,
global earns 5.636% net annualized versus 3.419% for clusters. Global also has lower
volatility, higher Sharpe, and lower turnover.

This is a clean ranking comparison, not a covariance-based allocation test. Both legs use
the same eligible funds, ROSAA risk-adjusted momentum, q=25%, strategic sleeve budgets,
monthly decisions, one-period implementation lag, and 20 bp one-way costs. Only the peer
set used for ranking differs.

## Frozen specification

- Eligibility: arithmetic average of the latest 12 completed month-end Bloomberg AUM
  observations strictly greater than USD100m; missing or incomplete histories are ineligible.
- Signal: ROSAA production risk-adjusted momentum.
- Strategic budgets: 50% Equity / 30% Fixed Income / 20% Rest.
- Cluster cell: ME covariance returns, EWMA span 12.
- Selection: q=25%, long-only.
- Cluster construction: equal budget across available clusters within each strategic sleeve;
  equal weight among selected funds inside each cluster.
- Global control: rank within each strategic sleeve and equal-weight selected funds.
- Rebalancing and costs: monthly, lag 1, 20 bp one way.
- Headline window: 2009-08-31 through 2026-06-30.

## Results

All returns and volatility are annualized; turnover is annualized one-way turnover.

| Window | Leg | Net return | Gross return | Volatility | RF=0 Sharpe | Turnover | Cost drag |
|---|---|---:|---:|---:|---:|---:|---:|
| 2009-08-31..2017-12-31 | Global sleeve rank | 6.812% | 8.185% | 9.057% | 0.775 | 3.218x | 137.29 bp |
| 2009-08-31..2017-12-31 | Cluster-contained rank | 4.900% | 6.973% | 9.296% | 0.562 | 4.913x | 207.35 bp |
| 2018-01-31..2026-06-30 | Global sleeve rank | 4.643% | 5.932% | 12.084% | 0.436 | 3.080x | 128.97 bp |
| 2018-01-31..2026-06-30 | Cluster-contained rank | 2.133% | 4.162% | 12.600% | 0.231 | 4.924x | 202.85 bp |
| **2009-08-31..2026-06-30** | **Global sleeve rank** | **5.636%** | **6.954%** | **10.703%** | **0.567** | **3.120x** | **131.75 bp** |
| **2009-08-31..2026-06-30** | Cluster-contained rank | 3.419% | 5.465% | 11.095% | 0.359 | 4.913x | 204.68 bp |

### Cluster minus global

| Window | Net-return delta | Gross-return delta | Volatility delta | Sharpe delta | Turnover delta | Cost-drag delta |
|---|---:|---:|---:|---:|---:|---:|
| 2009-08-31..2017-12-31 | -191.21 bp | -121.15 bp | +0.239 pp | -0.2129 | +1.696x | +70.06 bp |
| 2018-01-31..2026-06-30 | -250.95 bp | -177.07 bp | +0.516 pp | -0.2051 | +1.844x | +73.88 bp |
| **Headline** | **-221.75 bp** | **-148.82 bp** | **+0.393 pp** | **-0.2078** | **+1.793x** | **+72.93 bp** |

The underperformance is not caused only by trading costs: the headline gross-return gap is
already -148.82 bp/year. Higher turnover adds another 72.93 bp/year of relative cost drag.
The same direction in both fixed subwindows makes this a persistent negative result for this
cluster-contained long-only specification.

## What “peer-contained ranking” means

It is simply a controlled change in the comparison group:

- global: a fund's score is compared with every eligible fund in the same broad sleeve;
- cluster: a fund's score is compared only with eligible funds in its correlation cluster
  inside that broad sleeve.

There is no risk optimizer in either leg. The test asks whether narrower, correlation-based
peer groups improve selection. For U2 long-only, the measured answer is no.

## Acceptance

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Eligible memberships missing from partition | 0 | 0 | PASS |
| AUM <= USD100m eligible observations | 0 | 0 | PASS |
| Maximum weight and sleeve-budget error | 2.22045e-16 | 1e-12 | PASS |
| Maximum signal lookahead days | 0 | 0 | PASS |
| Performance rows | 6 | 6 | PASS |
| Comparison rows | 3 | 3 | PASS |

The cache-first replay completed in 10.72 seconds. All 6/6 deterministic CSV artifacts were
byte-identical across two replays. Explicit E/F/W lint reported `All checks passed!`.

## Reproduction

Runner:

- `papers/cluster_lineage_2026/replication/run_u2_blackrock_aum100_long_only.py`

External output directory:

- `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/covariance_frequency_span_grid/blackrock_us_etfs/aum100_long_only_20260816/`

No files were staged or pushed. The complete `papers/cluster_lineage_2026/` tree remains
gitignored as owner instructed.
