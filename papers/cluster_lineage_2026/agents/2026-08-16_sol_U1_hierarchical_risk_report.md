# U1 hierarchical-risk execution report

**Date:** 2026-08-16  
**Status:** COMPLETE — all runner and independent-validation checks pass  
**Scope:** U1 only; standard well-known risk-allocation methods; nested cluster risk
budgeting/NCO excluded by owner ruling.

## Outcome and paper recommendation

The selected U1 cluster structure produces a strong, easy-to-explain risk result without a
new optimiser:

1. In the accepted long-short momentum construction, cluster-relative ranking reduces mean
   ex-ante volatility from **7.76% to 4.49% (-42.1%)**, increases the effective number of Ward
   risk clusters from **21.55 to 30.20 (+40.1%)**, and reduces the mean largest absolute
   cluster-risk share from **13.52% to 9.74% (-28.0%)** versus global ranking. Its Ward-cluster
   net-exposure L1 is zero to numerical precision, versus 1.377 for global ranking.
2. Among the standard long-only risk methods, **Ward-HRP is the primary result**. It improves
   net return, volatility, and net Sharpe simultaneously relative to flat ERC. It also narrowly
   improves all three against canonical single-link HRP, directly isolating the selected Ward
   structure's incremental value.
3. Cluster risk budgeting is the mechanism exhibit. Moving from the flat asset budget
   (alpha=1) through sqrt-size (alpha=0.5) to equal-cluster (alpha=0) monotonically reduces
   ex-ante volatility and cluster-risk concentration. The more aggressive versions give back
   realised return because they trade more and materially change capital allocation.
4. **Owner ruling:** Ward-HERC is excluded from the paper. Its internal diagnostic remains in
   the execution cache for auditability, but it appears in no paper comparison row or exhibit.

For the paper, use the long-short risk-decomposition exhibit as the main risk illustration and
Ward-HRP versus flat ERC plus canonical single-HRP as the familiar-method confirmation. Keep
cluster risk budgeting as the transparent mechanism table. No nested allocation is needed.

## Paper terminology and consolidated comparison

The non-proprietary snapshot method is **rolling EWMA-Ward correlation clustering**, shortened
to **Rolling-Ward** in exhibits. Its two portfolio applications are **Rolling-Ward HRP** and
**Rolling-Ward cluster-relative momentum**. When the partition-bonus smoother is used, call the
full method **noise-calibrated Rolling-Ward clustering**. MCF is the separate lineage tracker;
it is not part of the clustering name.

| Panel | Paper label | Net return | Volatility | RF=0 Sharpe | One-way turnover |
|---|---|---:|---:|---:|---:|
| Long-only | Flat ERC | 7.505% | 12.247% | 0.656 | 0.548x |
| Long-only | Canonical HRP (single linkage) | 7.578% | 11.969% | 0.674 | 1.146x |
| Long-only | **Rolling-Ward HRP** | **7.675%** | **11.852%** | **0.687** | 1.329x |
| Long-short | Global momentum rank | -4.402% | 12.791% | -0.286 | 3.296x |
| Long-short | **Rolling-Ward cluster-relative momentum** | **-1.615%** | **6.789%** | **-0.206** | 4.846x |

All rows cover the U1 headline window and are net of 10 bp one-way costs. The long-only
comparison changes allocation while holding the HCGL covariance fixed. The long-short
comparison changes the ranking groups and group-equal construction while holding the signal
and investable cross-section fixed. It supports relative return and risk-compression claims;
the long-short standalone return remains negative.

## Frozen design

- Universe/window: U1 MSCI US point-in-time members, 203 monthly estimation dates,
  2009-08-31 through 2026-06-30.
- Hierarchy: the already selected raw U1 covariance-clustering cell, ME returns, EWMA span 36,
  Pearson correlation, `1-rho` distance, Ward linkage, cutoff 0.60.
- Risk matrix: frozen E2 baseline HCGL/FF6 covariance snapshot on the exact same asset set at
  every date. No covariance or factor model was refitted.
- Portfolio mechanics: long-only, fully invested, monthly decisions, implementation lag 1,
  10 bp one-way cost.
- Benchmark: flat ERC for allocation performance. EW-all is used only for the already frozen
  alpha/beta reference columns and is never a performance yardstick.
- Long-short diagnostic: exact accepted U1 ROSAA-production signal and q=0.25 construction;
  selected Ward groups versus one global group.
- Alpha=1 cluster risk budgets are an exact flat-ERC control and therefore are not duplicated
  as a separate performance row.

## Methods

| ID | Definition |
|---|---|
| Flat ERC | Production `optimalportfolios.wrapper_risk_budgeting`, equal asset risk budgets |
| Cluster RB sqrt(n) | Production risk-budgeting solver; cluster budget proportional to sqrt(cluster size), equal within cluster |
| Cluster RB equal | Production risk-budgeting solver; equal risk budget per Ward cluster, equal within cluster |
| Ward-HRP | Canonical inverse-variance recursive bisection using the selected Ward linkage |
| Single-HRP | Canonical HRP using single linkage on the same point-in-time correlation input |

The HERC implementation was definition-audited after its concentration result. It matches the
[published variance-HERC sequence](https://ssrn.com/abstract=3237540): inverse-variance
allocation inside terminal clusters, cluster variance as the terminal risk, sum of terminal
risks for higher nodes, and inverse-risk top-down division. This is also the algorithm in the
current [CRAN `HierPortfolios` reference implementation](https://rdrr.io/cran/HierPortfolios/src/R/HERC_Portfolio.R).
The result is retained rather than silently replaced.

## Long-only performance

All returns and Sharpes are net of 10 bp one-way costs; Sharpe uses rf=0.

| Method | Net return | Realised vol | Net Sharpe | One-way turnover | Cost drag/year |
|---|---:|---:|---:|---:|---:|
| Flat ERC | 7.505% | 12.247% | 0.656 | 0.548x | 11.68 bp |
| **Ward-HRP** | **7.675%** | **11.852%** | **0.687** | 1.329x | 28.41 bp |
| Single-HRP | 7.578% | 11.969% | 0.674 | 1.146x | 24.46 bp |
| Cluster RB sqrt(n) | 6.805% | 11.888% | 0.617 | 1.185x | 25.09 bp |
| Cluster RB equal | 6.033% | 11.605% | 0.566 | 2.018x | 42.44 bp |

Ward-HRP versus flat ERC: +17.0 bp annual net return, -39.6 bp realised volatility, and
+0.031 net Sharpe. Ward-HRP versus canonical single-HRP: +9.8 bp annual net return,
-11.8 bp realised volatility, and +0.013 net Sharpe.

These are descriptive full-window results, not a new statistical-superiority claim.

## Ex-ante risk through the selected Ward structure

| Method | Mean ex-ante vol | Effective risk clusters | Largest absolute cluster-risk share | Diversification ratio |
|---|---:|---:|---:|---:|
| Flat ERC | 11.92% | 39.71 | 6.30% | 2.374 |
| Ward-HRP | 11.30% | 38.32 | 6.89% | 2.248 |
| Single-HRP | 11.56% | 42.14 | 5.86% | 2.220 |
| Cluster RB sqrt(n) | 11.34% | 53.99 | 3.47% | 2.484 |
| **Cluster RB equal** | **10.87%** | **60.40** | **1.71%** | **2.594** |

The cluster-risk-budget sequence behaves exactly as designed. Alpha=0 equalises Ward-cluster
risk budgets, so the mean effective risk-cluster count is approximately the mean number of
available Ward clusters and the largest share is approximately `1/G` at each date. This makes
it the cleanest mechanism check, even though it is not the best realised-performance portfolio.

Ward-HERC's internal diagnostic was definition-audited but is not part of the paper evidence
set following the owner ruling.

## Accepted long-short signal risk

| Diagnostic | Cluster rank | Global rank | Cluster change |
|---|---:|---:|---:|
| Mean ex-ante volatility | 4.49% | 7.76% | -42.1% |
| Effective Ward risk clusters | 30.20 | 21.55 | +40.1% |
| Largest absolute cluster-risk share | 9.74% | 13.52% | -28.0% |
| Ward-cluster net-exposure L1 | 0.000 | 1.377 | -100.0% |
| Effective assets, each side | 139.24 | 151.54 | -8.1% |

The lower risk is not produced by selecting more names: the cluster leg holds fewer effective
names on each side. It comes from neutralising net exposure inside each discovered Ward group,
whereas a global sort can accumulate large long/short tilts to whole correlation blocks.

## Acceptance — primary runner

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Headline allocation dates | 203 | 203 | PASS |
| Covariance/Ward asset-set match share | 1.000 | 1.000 | PASS |
| Ward cache/frozen ME36 partition match share | 1.000 | 1.000 | PASS |
| Alpha=1 versus flat ERC max weight error | 1.67e-16 | <=1e-10 | PASS |
| Maximum allocation weight-sum error | 9.59e-11 | <=1e-10 | PASS |
| Minimum hierarchical allocation weight | 2.36e-10 | >=0 | PASS |
| Weight outside point-in-time eligibility | 0 | <=1e-10 | PASS |
| Maximum Euler risk reconciliation error | 6.66e-16 | <=1e-10 | PASS |
| Maximum risk-budget target error | 1.09e-6 | <=2e-5 | PASS |
| Risk rows | 1,218 | 1,218 | PASS |
| Maximum signal look-ahead days | 0 | <=0 | PASS |
| Paper comparison rows | 5 | 5 | PASS |
| Ward-HERC paper rows | 0 | 0 | PASS |
| Deterministic numerical artifacts | 22/22 | 100% | PASS |

## Independent validation

The separate validator reads the persisted artifacts and caches; it does not call the runner.
It reconstructs flat ERC and equal-cluster risk budgeting directly with the production solver
on the first, middle, and final dates.

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Performance methods | 6 | 6 | PASS |
| Allocation dates | 203 | 203 | PASS |
| Risk rows | 1,218 | 1,218 | PASS |
| Maximum persisted weight-sum error | 9.58e-11 | <=1e-10 | PASS |
| Minimum persisted weight | 0 | >=0 | PASS |
| Maximum weight outside eligibility | 0 | <=1e-10 | PASS |
| Sampled flat ERC independent solver error | 0 | <=2e-8 | PASS |
| Sampled equal-cluster RB independent solver error | 0 | <=2e-8 | PASS |
| Persisted Euler contribution reconciliation | 3.33e-15 | <=1e-10 | PASS |
| Deterministic replay share | 1.000 | 1.000 | PASS |
| EW-all performance comparisons | 0 | 0 | PASS |
| Paper comparison rows | 5 | 5 | PASS |
| Ward-HERC paper rows | 0 | 0 | PASS |
| Paper long-short source-table error | 1.78e-15 | <=1e-10 | PASS |

The focused unit suite also passes 7/7. The defect-first proof was observed before the module
was added: the new test file initially failed collection with `ModuleNotFoundError`, then passed
after implementation.

## Runtime and footprint

- Cold allocation-cache build: 203 pickles in 476.6 seconds (timestamp span), 17.63 MB.
- Complete cache-first run including six backtests, signal-risk decomposition, CSVs, and
  exhibits: 47.9 seconds.
- The observed scale is operationally unproblematic for the approximately 600-asset U1 panel.

## Deliverables

Runner scripts:

- `papers/cluster_lineage_2026/replication/hierarchical_risk_allocations.py`
- `papers/cluster_lineage_2026/replication/run_u1_hierarchical_risk.py`
- `papers/cluster_lineage_2026/replication/validate_u1_hierarchical_risk.py`
- `papers/cluster_lineage_2026/replication/hierarchical_risk_allocations_test.py`

External artifact/cache root:

- `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/risk_allocation/u1_hierarchical_20260816/`

Primary tables:

- `performance.csv`
- `paper_comparison.csv`
- `comparison_vs_flat_erc.csv`
- `risk_summary.csv`
- `signal_risk_summary.csv`
- `acceptance.csv`
- `independent_validation.csv`
- `determinism.csv`

Paper-ready exhibits:

- `u1_allocation_performance.PNG`
- `u1_allocation_risk_structure.PNG`
- `u1_signal_risk_structure.PNG`

## Verification commands

```powershell
python -m pytest papers/cluster_lineage_2026/replication/hierarchical_risk_allocations_test.py -q
# ....... [100%]

ruff check --isolated --select E,F,W --line-length 100 `
  papers/cluster_lineage_2026/replication/hierarchical_risk_allocations.py `
  papers/cluster_lineage_2026/replication/hierarchical_risk_allocations_test.py `
  papers/cluster_lineage_2026/replication/run_u1_hierarchical_risk.py `
  papers/cluster_lineage_2026/replication/validate_u1_hierarchical_risk.py
# All checks passed!

python -m papers.cluster_lineage_2026.replication.run_u1_hierarchical_risk
# U1 hierarchical risk allocation: PASS (22/22 deterministic)

python -m papers.cluster_lineage_2026.replication.validate_u1_hierarchical_risk
# 14/14 PASS
```

## Deviations and open items

- No refits, no package-level numerical changes, no NCO, no EW-all performance comparison.
- The experiment uses the frozen E2 HCGL covariance for portfolio risk and the selected raw
  ME/span36 correlation hierarchy for grouping. This deliberate separation tests the cluster
  structure as an allocation input while holding the production risk model fixed.
- Statistical intervals were not added because the owner requested an illustration using
  familiar methods, not a new inferential claim. Existing E6 inference remains unchanged.
- No git staging or push was performed.
