# U2 BlackRock funds — Bloomberg AUM eligibility and corrected backtest execution

**Date:** 2026-08-16  
**Executor:** sol  
**Status:** COMPLETE  
**Interpretation of owner threshold:** “50 men” = strictly greater than USD 50 million  
**Repository scope:** ignored `papers/cluster_lineage_2026/` tree only; no staging or push

## Outcome

Bloomberg supplies a usable historical fund-size series.  The field is
`FUND_TOTAL_ASSETS`; Bloomberg reports it in millions of the currency identified by
`FUND_TOTAL_ASSETS_CRNCY`.  All 480 funds in the frozen U.S. iShares cohort report USD
and all 480 have historical observations.

The implemented point-in-time rule is:

1. take the last reported `FUND_TOTAL_ASSETS` observation in each completed calendar
   month, without forward-filling a missing month;
2. require 12 non-missing completed monthly observations;
3. compute their arithmetic mean in USD millions;
4. require the mean to be **strictly greater than 50**;
5. intersect that result with the existing 12-W-WED-return history warmup.

The same eligibility set feeds the correlation partition, cluster ranks, and matched
global ranks.  Missing/incomplete AUM is ineligible.  The accepted price-history-only
caches and outputs were not overwritten.

The filter is a useful liquidity/data-quality rule, but it does **not** make a pure
cluster long-short fund portfolio beat global at 20 bp one-way cost.  Across the corrected
113-cell search, there are zero full-window net-return or Sharpe wins.  The best pure
cluster gap is -32.92 bp/year.  For that specification clustering beats global gross by
+19.16 bp/year, but its additional cost drag is 52.08 bp/year.

A targeted cluster-assisted hybrid does beat global over the full window: the owner-base
global-long/cluster-short book rebalanced every two months earns -0.5343% net versus
-0.8728% for global, a **+33.85 bp/year** edge, with Sharpe -0.0496 versus -0.0807.
It loses before 2018 and wins after 2018, so it remains descriptive and regime-dependent,
not evidence of a stable cluster premium.

## Runners and artifacts

### Source and tests

- `papers/cluster_lineage_2026/data/fetch_blackrock_etf_aum.py`
- `papers/cluster_lineage_2026/replication/run_u2_blackrock_aum_filter.py`
- `papers/cluster_lineage_2026/replication/u2_blackrock_aum_filter_test.py`
- small reusable hooks added to
  `papers/cluster_lineage_2026/replication/run_u2_blackrock_long_short_search.py`
  for an injected partition loader and injected monthly eligibility;
- a supersession warning was added to
  `papers/cluster_lineage_2026/agents/2026-08-16_sol_U2_blackrock_long_short_improvement_search_report.md`.

### Bloomberg data

| artifact | shape | bytes | SHA-256 |
|:--|--:|--:|:--|
| `blackrock_etf_aum_usd_millions_daily.csv` | 4,462 × 480 | 11,594,022 | `71413ed403cbfb047707a67740ae906ee0f55d6c380389931d68996069f62a33` |
| `blackrock_etf_aum_usd_millions_monthly.csv` | 252 × 480 | 661,799 | `5de4e3d00c8b7db13d40ea0973ca243a76c8a7bc379596e5dd8b4a168f678621` |
| `blackrock_etf_aum_audit.csv` | 480 × 10 | 65,539 | `13e33cbeda6e987c3fcd25af1e13d642d4d81ae3efdfa0e054de216f755d7b1` |

The monthly completed-history panel runs from 2005-08-31 through 2026-07-31.
The native daily extraction includes data through 2026-08-14; the partial August month is
deliberately excluded from the monthly panel so it cannot be labelled as completed data.

Current-value unit validation used `IVV US Equity`, `AGG US Equity`, and
`IWM US Equity`.  Bloomberg returned 905,559.99, 138,016.09, and 83,427.75 respectively,
all in USD millions.  AGG independently matched the BlackRock screener value divided by
one million.  Across all funds the median Bloomberg-versus-BlackRock current relative
error is 0.00000303%.

### Output and caches

Root:

`$CLUSTER_LINEAGE_OUTPUT_DIR/e5b/covariance_frequency_span_grid/blackrock_us_etfs/aum50_filter_20260816/`

Filtered partition cache:

`.../aum50_filter_20260816/partitions/<frequency>_span_<span>.pkl`

The root contains the rolling-AUM panel, point-in-time AUM panel, eligibility panel,
per-date and per-fund eligibility diagnostics, all 28 filtered partitions, full search
tables, matched unfiltered controls, cost/component diagnostics, holding-period results,
and the corrected hybrid recheck.  The final 113-candidate payoff run took 856.70 seconds
with all 28 partition caches hit.

## Eligibility measurements

| decision date | history eligible | AUM eligible | combined eligible | removed by AUM | retained share |
|:--|--:|--:|--:|--:|--:|
| 2009-08-31 | 162 | 134 | 133 | 29 | 82.10% |
| 2017-12-31 | 281 | 243 | 242 | 39 | 86.12% |
| 2018-01-31 | 283 | 244 | 243 | 40 | 85.87% |
| 2026-06-30 | 474 | 400 | 400 | 74 | 84.39% |

Across all 240 schedule dates the median retained share is 87.40%.  Seventy-five of the
480 current-cohort funds never pass the trailing-average threshold during the study.
The minimum admitted rolling average is 50.005833 million, confirming strict `>` rather
than `>=`.

This filter does not repair survivorship bias: the input list remains the 2026-08-15
current iShares cohort, so products liquidated before that retrieval date are absent.

## Corrected-window defect and resolution

The investigation exposed a separate defect in the earlier exploratory U2 search.
`run_u2_blackrock_etf_grid._window_prices()` retains the initial pre-decision mark but
always extends the price panel through `FULL_END` (2026-07-31).  Therefore:

- the labelled 2009-2017 selection backtests continued through 2026 while holding the
  final training-window portfolio;
- the labelled headline window ending 2026-06-30 included July 2026 returns.

The new AUM runner uses `_closed_window_prices`: one mark on or before the first decision,
and no performance observation after the declared window end.  The maximum end-date
escape is -4 days (the final W-WED mark is four days before the relevant month end).

The earlier report's split-sample selection, stability statements, and headline payoff
statements are superseded.  The legacy output files and accepted code path remain intact
for auditability.  Future publication tables must use the corrected closed-window path.

## Corrected pure-cluster search

The grid retains 65 marginal candidates and 48 train-selected interactions, for 113
unique candidates.  Cluster and global use the same signal, q, sleeve weights, dates,
+1/-1 exposure, implementation lag, AUM-filtered eligible set, and 20 bp one-way costs.

| analysis window | candidates | net wins | Sharpe wins | wins on both | best net delta |
|:--|--:|--:|--:|--:|--:|
| selection 2009-08-31–2017-12-31 | 113 | 0 | 0 | 0 | -17.03 bp/yr |
| evaluation 2018-01-31–2026-06-30 | 113 | 5 | 0 | 0 | +65.00 bp/yr |
| headline 2009-08-31–2026-06-30 | 113 | 0 | 0 | 0 | -32.92 bp/yr |

The best descriptive full-window pure-cluster cell is:

- signal: classic 12m-ex-1m;
- covariance: W-WED, EWMA span 156;
- q: 15%;
- sleeves: Equity 40% / Fixed Income 40% / Rest 20%;
- cluster budget: square-root selected-group-size.

| metric | cluster | matched global | cluster − global |
|:--|--:|--:|--:|
| gross annual return | 0.3497% | 0.1581% | +19.16 bp |
| net annual return, 20 bp | -2.5784% | -2.2492% | -32.92 bp |
| volatility | 4.5092% | 9.1348% | -4.6256 pp |
| Sharpe, rf=0 | -0.5559 | -0.2031 | -0.3529 |
| one-way turnover/year | 3.6968 | 3.0377 | +0.6592 |
| cost drag/year | 292.81 bp | 240.73 bp | +52.08 bp |

Cost sensitivity confirms the mechanism.  At zero cost the cluster edge is +19.16
bp/year and its Sharpe edge is +0.0372.  At 10 bp one-way the net edge is already -7.18
bp/year; the interpolated break-even cost is about 7.3 bp one-way.  The frozen 20 bp fund
cost therefore rejects the pure-cluster payoff claim even though gross selection is
positive.

## What the AUM filter changes

The matched control uses the same corrected windows and the same selected specification;
only AUM eligibility is switched.

| specification | unfiltered net edge | AUM50 net edge | AUM effect on edge |
|:--|--:|--:|--:|
| owner-base, group-equal | -139.90 bp/yr | -169.54 bp/yr | -29.63 bp/yr |
| owner-base, asset-equal | -69.96 bp/yr | -76.34 bp/yr | -6.38 bp/yr |
| best classic full cell | -49.35 bp/yr | -32.92 bp/yr | +16.43 bp/yr |
| corrected train-selected ROSAA ME/52 cell | -128.78 bp/yr | -77.15 bp/yr | +51.63 bp/yr |

For the best classic cell, the relative improvement does not come from raising cluster
returns: AUM50 changes cluster net return by -6.89 bp/year and global by -23.32 bp/year.
Global deteriorates more, improving the relative spread by 16.43 bp/year.  The effect is
specification-dependent rather than a universal reduction of cluster noise.

## Holding-period and hybrid diagnostics

Slower pure-cluster rebalancing reduces turnover but destroys enough momentum freshness
to worsen the full-window comparison.  Monthly remains best at -32.92 bp/year; the best
two-month result is -56.58 bp/year and the best quarterly result is -76.07 bp/year.
The training-selected quarterly group-equal variant wins by +76.92 bp/year before 2018
but loses by -249.07 bp/year after 2018 and -98.12 bp/year overall.

The exact earlier global-long/cluster-short hypothesis was rerun under AUM50 and closed
windows.  The best full-window row among the targeted schedules is the owner-base signal
with every-two-month rebalancing:

| window | hybrid net | global net | net edge | hybrid Sharpe | global Sharpe | Sharpe edge |
|:--|--:|--:|--:|--:|--:|--:|
| 2009-2017 selection | -0.2194% | 0.3176% | -53.69 bp | -0.0049 | 0.0796 | -0.0844 |
| 2018-2026 evaluation | -0.3919% | -0.5855% | +19.36 bp | -0.0243 | -0.0392 | +0.0149 |
| 2009-2026 headline | -0.5343% | -0.8728% | **+33.85 bp** | -0.0496 | -0.0807 | **+0.0312** |

Zero of the 12 targeted candidate/side/schedule combinations wins net return in both
halves.  The hybrid is therefore a full-sample descriptive result driven by the later
regime, not a stable out-of-sample result.

## Acceptance measurements

| check | measured | tolerance | status |
|:--|--:|:--|:--|
| Bloomberg AUM currencies equal USD | 480 | 480 | PASS |
| funds with Bloomberg AUM history | 480 | 480 | PASS |
| completed monthly AUM end date | 2026-07-31 | 2026-07-31 | PASS |
| minimum AUM admitted | 50.005833m | > 50m | PASS |
| filtered partition cells | 28 | 28 | PASS |
| dates per partition cell | 240 | 240 | PASS |
| eligible memberships missing from partitions | 0 | 0 | PASS |
| maximum weight/exposure error | 1.4211e-14 | <= 1e-12 | PASS |
| signal timing/reconstruction rows green | 25 | 25 | PASS |
| maximum performance-window end escape | -4 days | <= 0 | PASS |
| holding monthly regression error | 4.9738e-14 | <= 1e-12 | PASS |
| declared holding comparison rows | 27 | 27 | PASS |
| matched unfiltered rows | 12 | 12 | PASS |
| matched-control maximum weight error | 9.3259e-15 | <= 1e-12 | PASS |
| hybrid maximum exposure error | 7.1054e-15 | <= 1e-12 | PASS |
| hybrid comparison rows | 36 | 36 | PASS |

## Verification

Focused regression command:

```text
python -m pytest \
  papers/cluster_lineage_2026/replication/u2_blackrock_aum_filter_test.py \
  papers/cluster_lineage_2026/replication/u2_blackrock_long_short_search_test.py -q
```

Output:

```text
..........                                                               [100%]
```

Targeted lint command:

```text
ruff check --isolated --select E,F,W --line-length 100 \
  papers/cluster_lineage_2026/data/fetch_blackrock_etf_aum.py \
  papers/cluster_lineage_2026/replication/run_u2_blackrock_aum_filter.py \
  papers/cluster_lineage_2026/replication/run_u2_blackrock_long_short_search.py \
  papers/cluster_lineage_2026/replication/u2_blackrock_aum_filter_test.py
```

Output:

```text
All checks passed!
```

The AUM rolling mean also has an independent regression: the stored AGG value at
2026-06-30 is reproduced by directly averaging its final 12 raw monthly observations.

## Recommendation

Adopt AUM50 as the U2 point-in-time liquidity eligibility rule.  It removes economically
small funds cleanly and reproducibly, and the Bloomberg coverage is complete for the
frozen current cohort.

Do not claim that AUM50 makes pure cluster ranking outperform global: the corrected
evidence rejects that claim at 20 bp costs.  The paper-safe result is that clustering
materially lowers long-short volatility, while its gross edge is consumed by turnover.
The global-long/cluster-short hybrid is a labelled exploratory full-window win, with its
failed split-sample stability shown beside it.

