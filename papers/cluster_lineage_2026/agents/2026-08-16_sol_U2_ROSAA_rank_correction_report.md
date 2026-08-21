# U2 ROSAA score/rank correction report — 2026-08-16

## Outcome

The U2 Equity/Fixed-Income 60/40 long-short AUM grid now uses the production ROSAA
risk-adjusted-momentum score path and the canonical OptimalPortfolios
`compute_top_quantile_equal_weights` selection/weighting function. The cluster treatment differs
from the global control only in score standardisation. Correlation clusters no longer define a
second ranking universe or receive capital budgets.

No covariance or partition was refit. The frozen point-in-time partitions were consumed
cache-first. Costs remain 20 bp one way, `q=0.25`, lag 1, and the every-two-month schedule.

## Runners and outputs

- Main runner:
  `papers/cluster_lineage_2026/replication/run_u2_equity_fi_long_short_aum_grid.py`
- Exact fund attribution:
  `papers/cluster_lineage_2026/replication/run_u2_equity_fi_fund_pnl_attribution.py`
- EPOL trace:
  `papers/cluster_lineage_2026/replication/diagnose_u2_epol_history.py`
- Cache/output root:
  `papers/cluster_lineage_2026/data/local_outputs/e5b/covariance_frequency_span_grid/blackrock_us_etfs/equity_fi_60_40_long_short_aum_grid_20260816/`
- EPOL diagnostic root:
  `papers/cluster_lineage_2026/data/diagnostics/epol_history/`

The completed `partitions.pkl` was copied byte-for-byte from the external frozen cache into the
gitignored local-data output root because this session cannot write to the legacy external output
directory. This was a cache relocation only, not a refit.

## Corrected construction

For each date and broad sleeve (Equity and Fixed Income):

1. The global leg consumes the global score returned by the existing
   `optimalportfolios.alphas.signals.momentum.compute_momentum_alpha` path.
2. The cluster leg consumes the raw production momentum signal standardised by existing
   `optimalportfolios.alphas.signals.utils.score_within_clusters`, with
   `min_cluster_size=5`. Clusters of size `<=5` use the global cross-sectional mean and standard
   deviation.
3. Each long side is produced by
   `optimalportfolios.alphas.compute_top_quantile_equal_weights(scores, prices, q=0.25)`.
4. Each short side is produced by the same function applied to `-scores`.
5. Sleeve weights are 60% Equity and 40% Fixed Income on each side. There is no within-cluster
   rank and no cluster-equal budget.

## Headline results

Annualised net return and Sharpe use the existing frozen paper conventions.

| AUM rule | Global net | Cluster net | Cluster − global | Global Sharpe | Cluster Sharpe | Δ Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| No cutoff | -0.587% | -1.268% | -0.682 pp | -0.053 | -0.246 | -0.193 |
| > USD50m | -0.710% | -1.090% | -0.381 pp | -0.072 | -0.202 | -0.130 |
| > USD100m | -0.760% | -1.137% | -0.378 pp | -0.081 | -0.210 | -0.129 |

The corrected cluster leg still does not beat the matched global rank in U2, but the performance
gap is materially smaller than under the extra cluster-ranking/group-budget layer. For the primary
USD100m rule, the annualised net-return gap narrows from -1.463 pp to -0.378 pp.

## EPOL correction

- Eligible decision dates: 91.
- Cluster-score leg: 16 long, 36 short, 39 flat.
- Maximum absolute EPOL target weight: 1.9355% (the global leg has the same maximum).
- Average conditional cluster-score weight: +1.3324% long and 1.4988% absolute short.
- EPOL cluster-minus-global net P&L contribution: -0.2442% of starting NAV.

The former ±10% positions are eliminated. EPOL's correlation cluster is retained only as a
descriptive diagnostic; it no longer controls EPOL's rank universe or budget.

## Acceptance checks

### Main backtest

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Declared included official classes | Equity / Fixed Income | Equity / Fixed Income | PASS |
| Excluded-class eligible observations | 0 | 0 | PASS |
| Maximum excluded-class weight | 0 | 1e-12 | PASS |
| Partition eligible-member count error | 0 | 0 | PASS |
| Eligible memberships missing from partitions | 0 | 0 | PASS |
| Maximum exposure/sleeve-budget error | 9.77e-15 | 1e-12 | PASS |
| Maximum signal look-ahead days | 0 | 0 | PASS |
| Performance rows | 18 | 18 | PASS |
| Headline filter-sensitivity rows | 3 | 3 | PASS |
| Independent rank-reference error, all 6 method/filter rows | 0 | 1e-12 | PASS |
| Long/short overlap assets | 0 | 0 | PASS |
| Maximum absolute target weight, whole grid | 8.0% | reported | PASS |
| Deterministic CSV replay | 8 / 8 byte-identical | 8 / 8 | PASS |

### Attribution and EPOL trace

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Attribution partition cache status | hit | hit | PASS |
| Instrument P&L accounting error | 1.94e-13 | 1e-10 | PASS |
| Fund component reconciliation error | 1.50e-15 | 1e-10 | PASS |
| Fund total-to-portfolio delta error | 2.81e-13 | 1e-10 | PASS |
| Attribution deterministic replay | 6 / 6 byte-identical | 6 / 6 | PASS |
| EPOL decision rows | 102 | 102 | PASS |
| EPOL weight outside eligibility | 0 | 1e-12 | PASS |
| EPOL attribution rows | 1 | 1 | PASS |

## Verification commands

The main runner and attribution runner were executed twice from the local output root with the
frozen cache. The canonical OptimalPortfolios profiler suite was also run with a workspace-local
pytest temporary directory:

```text
.................                                                        [100%]
17 passed
```

The first profiler-suite attempt reached 16 passing tests and then failed only because pytest's
default user temporary directory was inaccessible to the sandbox. Re-running with an unused
workspace-local `--basetemp` produced the green result above.

## Deviations and open items

- Numerical deviation from the prior exploratory output is intentional: the removed second
  cluster rank and group budget were the defect being corrected.
- The existing group-equal E5b artifacts remain separate historical allocation evidence; they are
  not used by this corrected signal-isolation backtest.
- No files were staged or pushed.
- Open items: none.
