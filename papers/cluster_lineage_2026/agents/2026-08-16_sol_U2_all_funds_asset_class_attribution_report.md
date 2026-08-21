# U2 all-funds asset-class attribution report — 2026-08-16

## Outcome

The corrected U2 ROSAA score-ranking comparison was rerun on the complete 480-column BlackRock
fund universe. The owner-selected point-in-time USD100m AUM rule remains primary. No official
asset class is excluded: Equity, Fixed Income, Multi Asset, Digital Assets, Commodity, Real
Estate, and Cash all enter eligibility, clustering, score formation, ranking, backtesting, and
attribution.

Strategic gross budgets are 50% Equity, 30% Fixed Income, and 20% Rest on each side. The official
classes inside Rest are not assigned separate budgets; their exposures are outcomes of the
canonical rank selection. The global and cluster-score legs use identical
`optimalportfolios.alphas.compute_top_quantile_equal_weights` construction. Correlation clusters
affect only production score standardisation.

## Runner, cache, and outputs

- Runner:
  `papers/cluster_lineage_2026/replication/run_u2_all_funds_asset_class_attribution.py`
- Output root:
  `papers/cluster_lineage_2026/data/local_outputs/e5b/covariance_frequency_span_grid/blackrock_us_etfs/all_funds_aum100_asset_class_attribution_20260816/`
- Frozen all-fund partition cache:
  `papers/cluster_lineage_2026/data/local_outputs/e5b/covariance_frequency_span_grid/blackrock_us_etfs/aum50_filter_20260816/threshold_sensitivity/partitions.pkl`
- Cache SHA-256:
  `288C05AE8A9B9A51B3A7203C3DFF649DDFB5F9D1B170160CE9E5D9BBFF68F708`

The cache hash matches the previously completed external all-fund cache byte-for-byte. No
partition was refit.

## Specification

- Window: 2009-08-31 through 2026-06-30.
- Signal: ROSAA production risk-adjusted momentum.
- Cluster fallback: global cross-sectional mean and standard deviation for cluster size `<=5`.
- Rank selection: canonical OP top/bottom 25%, independently within Equity, Fixed Income, and
  Rest.
- Portfolio: +100% / -100%, with 50/30/20 gross budgets per side.
- Rebalancing: every two months, implementation lag 1.
- Costs: 20 bp one way.
- Eligibility: return-history warmup and latest 12-completed-month average AUM strictly above
  USD100m.

## Eligible breadth

All 480 columns remain in scope; 381 funds are eligible on at least one headline date.

| Official class | Columns | Ever eligible | Eligible start | Median | Eligible end |
|---|---:|---:|---:|---:|---:|
| Equity | 288 | 246 | 93 | 167 | 238 |
| Fixed Income | 154 | 113 | 19 | 44 | 111 |
| Multi Asset | 17 | 5 | 0 | 5 | 5 |
| Digital Assets | 4 | 2 | 0 | 0 | 2 |
| Commodity | 7 | 6 | 3 | 4 | 6 |
| Real Estate | 8 | 7 | 3 | 6 | 6 |
| Cash | 2 | 2 | 0 | 0 | 2 |

## Portfolio results

| Leg | Annual net return | Volatility | Sharpe | Annual turnover | Cost drag/year |
|---|---:|---:|---:|---:|---:|
| Global score | -0.960% | 7.601% | -0.089 | 2.265 | 181.0 bp |
| Cluster-standardised score | -1.434% | 5.726% | -0.223 | 2.504 | 199.7 bp |
| Cluster − global | -0.474 pp | -1.876 pp | -0.135 | +0.240 | +18.7 bp |

The cluster-score leg again has materially lower volatility, but it does not beat the matched
global rank on return or Sharpe.

## Exact asset-class P&L attribution

Numbers below are cumulative percentage points of beginning portfolio NAV over the headline
window, not annualised returns. The class contributions sum exactly to each portfolio's total P&L.

| Official class | Global net P&L | Cluster net P&L | Cluster − global | Long component | Short component | Cost effect |
|---|---:|---:|---:|---:|---:|---:|
| Equity | -8.912% | -14.625% | -5.713 pp | -10.657 pp | +4.890 pp | +0.054 pp |
| Commodity | +0.664% | -4.871% | -5.535 pp | -4.979 pp | -0.753 pp | +0.197 pp |
| Multi Asset | +5.183% | +3.263% | -1.920 pp | -1.181 pp | -0.582 pp | -0.156 pp |
| Cash | -0.006% | -0.005% | +0.001 pp | 0.000 pp | -0.000 pp | +0.001 pp |
| Fixed Income | -0.801% | -0.534% | +0.267 pp | -1.144 pp | +2.001 pp | -0.590 pp |
| Digital Assets | -1.051% | +0.073% | +1.124 pp | +1.055 pp | +0.041 pp | +0.028 pp |
| Real Estate | -10.061% | -4.874% | +5.187 pp | +1.861 pp | +3.026 pp | +0.300 pp |
| **Total** | **-14.984%** | **-21.573%** | **-6.588 pp** |  |  |  |

The two largest sources of underperformance are Equity and Commodity. Commodity is primarily a
long-selection problem: -4.979 pp of its -5.535 pp gap comes from the long book. Real Estate is
the strongest offset, improving the cluster leg by +5.187 pp relative to global. Multi Asset is a
smaller negative contributor; Digital Assets and Fixed Income are positive offsets; Cash is
immaterial.

## Acceptance checks

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Partition cache status | hit | hit | PASS |
| Eligible memberships missing from partitions | 0 | 0 | PASS |
| Maximum signal look-ahead days | 0 | 0 | PASS |
| Maximum exposure/sleeve-budget error | 8.44e-15 | 1e-12 | PASS |
| Maximum weight outside eligibility | 0 | 1e-12 | PASS |
| Maximum instrument accounting error | 8.53e-14 | 1e-10 | PASS |
| Maximum class component error | 7.11e-15 | 1e-10 | PASS |
| Maximum class-delta component error | 8.88e-15 | 1e-10 | PASS |
| Maximum portfolio P&L attribution error | 8.88e-14 | 1e-10 | PASS |
| Cluster-global delta attribution error | 8.08e-14 | 1e-10 | PASS |
| Official class rows | 14 | 14 | PASS |
| Official classes with an eligible fund | 7 | 7 | PASS |
| Deterministic replay | 10 / 10 byte-identical | 10 / 10 | PASS |

## Deliverables

- `performance.csv`
- `asset_class_pnl.csv`
- `asset_class_delta_vs_global.csv`
- `asset_class_weight_summary.csv`
- `asset_class_eligibility.csv`
- `instrument_pnl.csv`
- `weight_diagnostics.csv`
- `partition_diagnostics.csv`
- `acceptance.csv`
- `determinism.csv`

No files were staged or pushed. Open items: none.
