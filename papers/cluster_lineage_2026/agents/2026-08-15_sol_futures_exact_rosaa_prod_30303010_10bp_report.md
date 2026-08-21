# Futures exact ROSAA-production 30/30/30/10 at 10 bp — execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE; supersedes the 48w/4w futures results for production-signal interpretation  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

The futures global, baseline-cluster, and M1-star-cluster long-short portfolios were
rerun with the exact ROSAA production momentum primitive used by the U1 and BlackRock
U2 production grids:

- monthly return cadence;
- EWMA momentum long span 12;
- EWMA volatility span 13;
- no short-term reversal component;
- `MeanAdjType.NONE`;
- point-in-time eligible equal-weight benchmark for excess returns;
- production five-name fallback for rolling-cluster scores.

The portfolio specification is unchanged: 30% Equity / 30% Fixed Income /
30% Commodities / 10% FX on each signed side, q=20% primary and q=25% robustness,
the corrected U1 window, one W-WED implementation lag, and 10 bp per one-way traded
notional.

Global ranks remain profitable, but neither cluster method beats global. The closest
cluster result is M1-star at q=25%: -0.0206% net/year and 0.0030 RF=0 Sharpe versus
+0.4770% and 0.0874 for global.

| q | method | gross return/year | net return/year | volatility/year | RF=0 Sharpe | one-way turnover/year | cost drag/year |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20% | global | **2.5596%** | **0.9856%** | 9.1328% | **0.1445** | 3.8739 | 157.40 bp |
| 20% | baseline cluster | 1.1758% | -0.4010% | **4.6276%** | -0.0839 | 3.9274 | 157.68 bp |
| 20% | M1-star cluster | 0.8222% | -0.5952% | 4.6963% | -0.1218 | **3.5397** | **141.74 bp** |
| 25% | global | **1.9211%** | **0.4770%** | 8.0164% | **0.0874** | 3.5725 | 144.41 bp |
| 25% | baseline cluster | 1.1972% | -0.2830% | **4.3260%** | -0.0590 | 3.6844 | 148.01 bp |
| 25% | M1-star cluster | 1.2868% | -0.0206% | 4.3697% | 0.0030 | **3.2503** | **130.74 bp** |

The q=25% M1-star geometric return is slightly negative while its RF=0 Sharpe is
slightly positive. Annual return is the compounded CAGR; the frozen Sharpe uses the
annualised arithmetic mean of periodic returns.

## Cluster-minus-global differences

| q | cluster | net-return difference | gross-return difference | volatility difference | Sharpe difference | turnover difference |
|---:|---|---:|---:|---:|---:|---:|
| 20% | baseline | -138.66 bp/year | -138.38 bp/year | -4.5052 pp | -0.2284 | +0.0535 |
| 20% | M1-star | -158.08 bp/year | -173.74 bp/year | -4.4366 pp | -0.2664 | -0.3342 |
| 25% | baseline | -75.99 bp/year | -72.39 bp/year | -3.6904 pp | -0.1464 | +0.1119 |
| 25% | M1-star | -49.76 bp/year | -63.42 bp/year | -3.6468 pp | -0.0844 | -0.3222 |

As in the raw-weekly robustness, clustering roughly halves realised volatility. The
failure to beat global is principally pre-cost signal dilution: every cluster gross
return is below its matched global gross return. M1-star's lower turnover narrows the
q=25% deficit from 63.42 bp gross to 49.76 bp net, but does not reverse it.

## Cost sensitivity

| q | method | net return/year at 10 bp | net return/year at 20 bp | 10 bp improvement |
|---:|---|---:|---:|---:|
| 20% | global | 0.9856% | -0.5687% | +1.5543 pp |
| 20% | baseline cluster | -0.4010% | -1.9578% | +1.5567 pp |
| 20% | M1-star cluster | -0.5952% | -1.9966% | +1.4014 pp |
| 25% | global | 0.4770% | -0.9506% | +1.4275 pp |
| 25% | baseline cluster | -0.2830% | -1.7454% | +1.4624 pp |
| 25% | M1-star cluster | -0.0206% | -1.3144% | +1.2938 pp |

At 20 bp, every exact-production portfolio is negative. The owner's 10 bp convention
is therefore material, but it does not change the cluster-versus-global ordering.

## Signal validation

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| daily source to frozen W-WED return reconstruction | 4.649e-16 | <= 1e-15 | PASS |
| daily/W-WED NaN-pattern match | true | true | PASS |
| monthly signal-return round trip | 1.624e-15 | <= 1e-12 | PASS |
| global score independent reconstruction | 0 | <= 1e-12 | PASS |
| global/raw signal timestamp agreement | true | true | PASS |
| cluster signal timestamp agreement | 2/2 methods | 100% | PASS |
| maximum signal look-ahead | 0 days | <= 0 days | PASS |
| production signal parameters | ME / 12 / 13 / no short / NONE | exact | PASS |
| production cluster fallback | 5 names | exact | PASS |
| valid production scores per date | 88 / 91 / 94 min/median/max | nonempty | PASS |
| `CUA1 Comdty` valid production scores | 0 | 0 | PASS |

The signal uses the daily `futures_log_returns.csv` source and aggregates it directly to
month-end. It is not approximated from weekly observations. Every decision consumes the
latest signal timestamp not after that decision.

## Portfolio acceptance and verification

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
| standalone four-sleeve reconstruction error | 2.776e-17 | <= 1e-12 | PASS |
| deterministic numerical artifacts | 12/12 byte-identical | 100% | PASS |
| focused pytest | 3/3 passed | all pass | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |
| EW-all payoff comparison | 0 | 0 | PASS |

One complete pass over all methods, q values, and cost rates took 42.37 seconds and was
replayed in full. The six measured net returns and Sharpes are frozen in the focused
regression test.

The fail-before-pass checkpoint produced the expected missing-runner failure:

```text
ModuleNotFoundError: No module named
'papers.cluster_lineage_2026.replication.run_futures_prod_cluster_30303010_10bp'
```

Final verification output:

```text
...                                                                      [100%]
All checks passed!
Futures exact ROSAA production 30/30/30/10 at 10 bp: PASS
(12/12 deterministic)
```

## Supersession ruling for executor outputs

The earlier 2026-08-15 futures global and cluster 10 bp reports used the paper's
raw-weekly 48-week signal with a four-week skip. Those artifacts remain valid as a
labelled signal robustness, but they are not ROSAA-production results. This report and
its artifacts supersede them for every production-signal conclusion.

The old runner docstrings have been corrected from “production” to “paper raw-weekly.”
No old numerical artifact or cluster cache was deleted or overwritten.

## Deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_futures_prod_cluster_30303010_10bp.py`

Focused regression checks:

- `papers/cluster_lineage_2026/replication/futures_prod_cluster_30303010_10bp_test.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\futures_prod_cluster_30_30_30_10_10bp_u1_window
```

The machine-readable output contains `performance.csv`, `comparison.csv`,
`cost_sensitivity.csv`, `signal_diagnostics.csv`, `cluster_signal_diagnostics.csv`,
`acceptance.csv`, `allocation_diagnostics.csv`, `horizon_diagnostic.csv`,
`standalone_weight_reconstruction.csv`, `available_group_counts_by_date.csv`,
`available_group_count_summary.csv`, `design.csv`, `runtime.csv`, and
`determinism.csv`.

No cluster cache was altered, no EW-all payoff comparison was introduced, and no file
was staged or pushed.
