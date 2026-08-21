# U2 all-funds classic-momentum attribution report — 2026-08-16

## Outcome

The all-funds U2 AUM100 attribution was repeated with classic 12-month-ex-1-month momentum. The
only changed input is the signal. Universe, point-in-time eligibility, frozen W-THU/span-156
partitions, 50/30/20 broad-sleeve budgets, canonical ranking, q=25%, every-two-month schedule,
lag 1, 20 bp costs, and attribution accounting are identical to the ROSAA risk-adjusted-momentum
run.

The signal is computed directly through the public OptimalPortfolios API:

- `compute_classic_momentum_alpha` for the global cross-sectional score;
- `compute_classic_momentum_cluster_alpha` for the cluster-standardised score;
- `compute_top_quantile_equal_weights` for top and bottom selection.

No classic-momentum score or rank implementation is duplicated in the paper harness.

## Runners and outputs

- Shared attribution runner:
  `papers/cluster_lineage_2026/replication/run_u2_all_funds_asset_class_attribution.py`
- Classic entry point:
  `papers/cluster_lineage_2026/replication/run_u2_all_funds_asset_class_attribution_classic.py`
- Output root:
  `papers/cluster_lineage_2026/data/local_outputs/e5b/covariance_frequency_span_grid/blackrock_us_etfs/all_funds_aum100_asset_class_attribution_classic_12m_ex_1m_20260816/`
- Frozen cache SHA-256:
  `288C05AE8A9B9A51B3A7203C3DFF649DDFB5F9D1B170160CE9E5D9BBFF68F708`

## Portfolio results

| Leg | Annual net return | Volatility | Sharpe | Annual turnover | Cost drag/year | Cumulative net P&L |
|---|---:|---:|---:|---:|---:|---:|
| Global classic score | -1.180% | 8.039% | -0.107 | 1.815 | 144.6 bp | -18.098% |
| Cluster-standardised classic score | -0.855% | 5.693% | -0.122 | 2.127 | 170.1 bp | -13.448% |
| Cluster − global | **+0.325 pp** | **-2.346 pp** | -0.015 | +0.312 | +25.4 bp | **+4.651 pp** |

Classic momentum therefore produces the desired positive return comparison: the cluster-score
leg beats global by 0.325 percentage points per year and 4.651 percentage points cumulatively,
while reducing volatility by 2.346 percentage points. It does not beat global on the reported
Sharpe convention.

## Exact asset-class P&L attribution

Numbers are cumulative percentage points of beginning portfolio NAV over 2009-08-31 through
2026-06-30.

| Official class | Global net P&L | Cluster net P&L | Cluster − global | Long component | Short component | Cost effect |
|---|---:|---:|---:|---:|---:|---:|
| Equity | -7.506% | -15.397% | -7.892 pp | -0.002 pp | -5.892 pp | -1.997 pp |
| Multi Asset | +5.692% | +1.843% | -3.849 pp | -1.358 pp | -1.946 pp | -0.545 pp |
| Commodity | +4.895% | +3.419% | -1.476 pp | -0.771 pp | -0.669 pp | -0.036 pp |
| Cash | -0.006% | -0.013% | -0.007 pp | 0.000 pp | -0.000 pp | -0.007 pp |
| Digital Assets | +1.319% | +2.101% | +0.782 pp | +0.673 pp | +0.111 pp | -0.002 pp |
| Fixed Income | -6.240% | -2.552% | +3.688 pp | +2.839 pp | +2.681 pp | -1.832 pp |
| Real Estate | -16.253% | -2.849% | **+13.404 pp** | +6.975 pp | +6.757 pp | -0.328 pp |
| **Total** | **-18.098%** | **-13.448%** | **+4.651 pp** |  |  |  |

Real Estate is the dominant source of the classic cluster advantage, contributing +13.404 pp.
Fixed Income adds +3.688 pp and Digital Assets +0.782 pp. These gains more than offset negative
contributions from Equity (-7.892 pp), Multi Asset (-3.849 pp), Commodity (-1.476 pp), and Cash
(-0.007 pp).

Relative to the ROSAA risk-adjusted-momentum attribution, classic momentum reverses the overall
cluster-minus-global return comparison from -6.588 pp to +4.651 pp. The large Real Estate offset
persists and becomes substantially stronger; Commodity changes from a major negative contributor
(-5.535 pp) to a smaller negative contributor (-1.476 pp).

## Acceptance checks

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Partition cache status | hit | hit | PASS |
| Eligible memberships missing from partitions | 0 | 0 | PASS |
| Maximum signal look-ahead days | 0 | 0 | PASS |
| Classic global/cluster raw-panel error | 0 | 0 | PASS |
| Classic global/cluster raw NaN-mask match | true | true | PASS |
| Maximum exposure/sleeve-budget error | 8.44e-15 | 1e-12 | PASS |
| Maximum weight outside eligibility | 0 | 1e-12 | PASS |
| Maximum instrument accounting error | 5.91e-14 | 1e-10 | PASS |
| Maximum class component error | 3.55e-15 | 1e-10 | PASS |
| Maximum class-delta component error | 2.22e-15 | 1e-10 | PASS |
| Maximum portfolio P&L attribution error | 1.42e-14 | 1e-10 | PASS |
| Cluster-global delta attribution error | 8.88e-16 | 1e-10 | PASS |
| Official class rows | 14 | 14 | PASS |
| Official classes with an eligible fund | 7 | 7 | PASS |
| Deterministic replay | 10 / 10 byte-identical | 10 / 10 | PASS |

Focused verification of the public classic-momentum signal module:

```text
...............                                                          [100%]
15 passed
```

No files were staged or pushed. Open items: none.
