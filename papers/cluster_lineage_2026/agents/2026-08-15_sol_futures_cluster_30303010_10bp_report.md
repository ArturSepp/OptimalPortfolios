# Futures cluster 30/30/30/10 at 10 bp — execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

The baseline and M1-star cluster-relative futures long-short strategies were run under
the same specification as the completed global control: 30% Equity / 30% Fixed Income /
30% Commodities / 10% FX on each signed side, q=20% primary plus q=25% robustness, the
corrected U1 calendar window, and 10 bp per one-way traded notional.

Neither cluster method beats the matched global ranks. The best cluster result is the
q=25% baseline, but it earns only 0.1301% net/year with a 0.0492 RF=0 Sharpe, compared
with 2.2994% and 0.3102 for global ranks.

| q | method | gross return/year | net return/year | volatility/year | RF=0 Sharpe | one-way turnover/year | cost drag/year |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20% | global | 3.0640% | **1.7548%** | 9.4861% | **0.2285** | 3.2029 | 130.92 bp |
| 20% | baseline cluster | 1.2886% | -0.0912% | 4.5548% | 0.0006 | 3.4284 | 137.98 bp |
| 20% | M1-star cluster | 1.0822% | -0.1021% | **4.5464%** | 0.0040 | **2.9461** | **118.43 bp** |
| 25% | global | 3.4944% | **2.2994%** | 8.4368% | **0.3102** | 2.9099 | 119.50 bp |
| 25% | baseline cluster | 1.4079% | 0.1301% | **4.1608%** | 0.0492 | 3.1708 | 127.78 bp |
| 25% | M1-star cluster | 0.9030% | -0.1785% | 4.1745% | -0.0202 | **2.6930** | **108.14 bp** |

The q=20% M1-star geometric return is slightly negative while its RF=0 Sharpe is slightly
positive. This is not a computation inconsistency: annual return is the compounded CAGR,
whereas the frozen Sharpe uses the annualised arithmetic mean of periodic returns.

## Cluster-minus-global differences

| q | cluster | net-return difference | volatility difference | Sharpe difference | turnover difference |
|---:|---|---:|---:|---:|---:|
| 20% | baseline | -184.60 bp/year | -4.9313 pp | -0.2279 | +0.2256 |
| 20% | M1-star | -185.69 bp/year | -4.9397 pp | -0.2244 | -0.2568 |
| 25% | baseline | -216.93 bp/year | -4.2761 pp | -0.2610 | +0.2609 |
| 25% | M1-star | -247.78 bp/year | -4.2624 pp | -0.3304 | -0.2168 |

The underperformance is generated before costs. At q=20%, the baseline and M1-star gross
return deficits are respectively 177.53 and 198.17 bp/year. Their cost-drag differences
versus global are only +7.06 and -12.49 bp/year. At q=25%, the corresponding gross deficits
are 208.65 and 259.14 bp/year, versus cost-drag differences of only +8.28 and -11.36 bp.

Thus the cluster construction approximately halves volatility, but group-equal
cluster-relative ranking dilutes the futures momentum spread too severely. Transaction
costs are not the reason it fails to beat global ranks.

## Matched cost sensitivity

| q | method | net return/year at 10 bp | net return/year at 20 bp | 10 bp improvement |
|---:|---|---:|---:|---:|
| 20% | global | 1.7548% | 0.4591% | +1.2957 pp |
| 20% | baseline cluster | -0.0912% | -1.4558% | +1.3646 pp |
| 20% | M1-star cluster | -0.1021% | -1.2754% | +1.1733 pp |
| 25% | global | 2.2994% | 1.1155% | +1.1839 pp |
| 25% | baseline cluster | 0.1301% | -1.1347% | +1.2648 pp |
| 25% | M1-star cluster | -0.1785% | -1.2507% | +1.0722 pp |

Reducing costs materially helps every method, but it does not change the ranking.

## Construction and available groups

The cluster legs use the accepted `group_equal` construction. Within each strategic
sleeve, every available hierarchical cluster receives an equal share of that sleeve's
budget. Selected top- or bottom-q contracts split their cluster's budget equally. The
hierarchical identifier is broad sleeve plus cached correlation-cluster id, which prevents
a correlation cluster from crossing a strategic sleeve budget.

| method | sleeve | mean available groups | standard deviation | minimum | maximum |
|---|---|---:|---:|---:|---:|
| baseline | Equity | 2.833 | 1.025 | 1 | 5 |
| baseline | Fixed Income | 2.517 | 0.624 | 1 | 4 |
| baseline | Commodities | 10.700 | 2.349 | 7 | 15 |
| baseline | FX | 3.123 | 0.580 | 2 | 5 |
| M1-star | Equity | 2.793 | 1.075 | 2 | 5 |
| M1-star | Fixed Income | 2.522 | 0.600 | 2 | 4 |
| M1-star | Commodities | 10.394 | 2.384 | 7 | 15 |
| M1-star | FX | 3.414 | 0.594 | 2 | 5 |

The global control has exactly one available group inside each sleeve on every date.
The eligible universe is recomputed point in time and ranges from 88 to 94 contracts:
Equity 29, Fixed Income 18--21, Commodities 31--33, and FX 10--11. `CUA1 Comdty` is
excluded. The existing partial-history convention remains unchanged: the production
48-week-minus-4-week signal uses `sum(min_count=1)` and does not require a complete
48-week history.

## Acceptance and verification

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| decision dates | 203 | 203 | PASS |
| portfolio rows | 6 | 3 methods x 2 q values | PASS |
| cluster/global comparison rows | 4 | 2 clusters x 2 q values | PASS |
| pre/post-window measured NAV rows | 0 / 0 | 0 / 0 | PASS |
| primary cost | 10 bp one-way | exact | PASS |
| owner-excluded `CUA1 Comdty` maximum absolute weight | 0 | 0 | PASS |
| maximum within-sleeve group-budget error | 0 | <= 1e-15 | PASS |
| maximum top-level sleeve-budget error | 1.665e-16 | <= 1e-12 | PASS |
| maximum net-exposure error | 1.804e-16 | <= 1e-12 | PASS |
| maximum gross-exposure error | 2.665e-15 | <= 1e-12 | PASS |
| standalone four-sleeve reconstruction error | 5.551e-17 | <= 1e-12 | PASS |
| global control vs dedicated global run | 4.547e-13 | <= 1e-12 | PASS |
| deterministic numerical artifacts | 11/11 byte-identical | 100% | PASS |
| focused pytest | 3/3 passed | all pass | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |
| EW-all payoff comparison | 0 | 0 | PASS |

One complete pass over all three methods, both q values, and both cost rates took
41.96 seconds and was replayed in full. The combined weights were independently rebuilt
as the weighted sum of four standalone asset-class books; all six reconstructions passed.

The fail-before-pass checkpoint produced the expected missing-runner failure:

```text
ModuleNotFoundError: No module named
'papers.cluster_lineage_2026.replication.run_futures_cluster_30303010_10bp'
```

Final verification output:

```text
...                                                                      [100%]
All checks passed!
Futures cluster 30/30/30/10 at 10 bp: PASS (11/11 deterministic)
```

## Deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_futures_cluster_30303010_10bp.py`

Focused regression checks:

- `papers/cluster_lineage_2026/replication/futures_cluster_30303010_10bp_test.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\futures_cluster_30_30_30_10_10bp_u1_window
```

The machine-readable output contains `performance.csv`, `comparison.csv`,
`cost_sensitivity.csv`, `acceptance.csv`, `allocation_diagnostics.csv`,
`horizon_diagnostic.csv`, `standalone_weight_reconstruction.csv`,
`available_group_counts_by_date.csv`, `available_group_count_summary.csv`,
`global_control_regression.csv`, `design.csv`, `runtime.csv`, and `determinism.csv`.

No cluster cache was altered, no EW-all payoff comparison was introduced, and no file was
staged or pushed.
