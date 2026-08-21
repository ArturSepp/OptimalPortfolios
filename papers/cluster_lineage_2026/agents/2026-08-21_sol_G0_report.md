# G0 report — U1 headline-window cached-series re-score

**Date:** 2026-08-21  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v3  
**Status:** COMPLETE; AWAITING OWNER GATE G0

## Execution and outputs

Runner: `papers/cluster_lineage_2026/replication/run_g0_u1_window_rescore.py`.  
Focused tests: `replication/g0_u1_window_rescore_test.py` and
`replication/f6_bootstrap_test.py`.

Output directory:
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/g0/`.

The runner read the two existing U1 NAV panels whose F0 content-addressed fingerprints are:

| Input | F0 manifest SHA-256 | Fingerprint match |
|---|---|---|
| U1 signal NAVs | `26f9f2325aff7aa01ca198f40588167b7f263acb344868de0315ef3c1dfcf432` | PASS |
| U1 risk NAVs | `af14338cc8ae5be46d2da656003753502fca9c9a0a3b0ec7c1854e134d5e9b01` | PASS |

It emitted `u1_windowed_performance.csv`, `u1_windowed_cis.csv`, and
`u1_reconciliation.csv`, with supporting `source_manifest.csv`, `acceptance.csv`, and
`determinism.csv`. It did not call a backtest, optimizer, covariance estimator, or clustering
estimator, and it generated no NAV, weight, partition, or covariance series.

The labelled window is 2009-08-31 through 2026-06-30. Under the existing U2/U3 monthly
convention, the last NAV observation on or before the start date (2009-08-26) seeds the first
return and the final observation is 2026-06-24. This gives 202 monthly returns in every row.

## Windowed U1 performance

| Leg | Annual net return | Annual volatility | RF=0 Sharpe |
|---|---:|---:|---:|
| Signal: cluster | -3.8703% | 9.1696% | -0.3831 |
| Signal: global | -3.9365% | 13.0992% | -0.2394 |
| Signal: BICS sector | -3.5160% | 10.3615% | -0.2926 |
| Risk: Rolling-Ward HRP | 8.9984% | 12.8295% | 0.7398 |
| Risk: flat ERC | 8.7863% | 13.2687% | 0.7051 |
| Risk: single-link HRP | 8.8561% | 12.9548% | 0.7236 |

## Windowed U1 comparison intervals

The bootstrap is the frozen joint moving-block procedure: block length 6, 2,000 draws, seed
20260813, percentile 95% intervals. The same sampled block indices are applied jointly to
both legs of each comparison.

| Comparison | Metric | Point | 95% CI | Excludes zero |
|---|---|---:|---:|---|
| Cluster - global | annual net return | +0.000661 | [-0.027206, +0.024238] | No |
| | annual volatility | -0.039297 | [-0.049525, -0.028188] | Yes |
| | RF=0 Sharpe | -0.143708 | [-0.335611, +0.066921] | No |
| Cluster - BICS sector | annual net return | -0.003544 | [-0.026535, +0.017893] | No |
| | annual volatility | -0.011919 | [-0.020520, -0.003145] | Yes |
| | RF=0 Sharpe | -0.090499 | [-0.321866, +0.150546] | No |
| Rolling-Ward HRP - flat ERC | annual net return | +0.002120 | [-0.008500, +0.012396] | No |
| | annual volatility | -0.004392 | [-0.008684, -0.000812] | Yes |
| | RF=0 Sharpe | +0.034749 | [-0.045051, +0.131292] | No |
| Rolling-Ward HRP - single-link HRP | annual net return | +0.001423 | [-0.003761, +0.006639] | No |
| | annual volatility | -0.001253 | [-0.003125, +0.000131] | No |
| | RF=0 Sharpe | +0.016233 | [-0.024192, +0.070120] | No |

The gated U1 signal conclusion is lower volatility, not higher return or Sharpe. Clustering
reduces annualised volatility by 3.930 percentage points versus global ranks and 1.192 points
versus BICS-sector ranks; both intervals exclude zero. Both return intervals and both Sharpe
intervals include zero. For risk allocation, Rolling-Ward HRP's 0.439 percentage-point
volatility reduction versus flat ERC excludes zero; the other risk intervals include zero.

## Reconciliation to the historical quotations

| Comparison | Metric | G0 window | F6 full range | 2026-08-17 narrative | Residual G0 - narrative |
|---|---|---:|---:|---:|---:|
| Cluster - global | annual net return | +0.000661 | +0.003227 | +0.006910 | -0.006249 |
| | annual volatility | -0.039297 | -0.040019 | -0.019200 | -0.020097 |
| | RF=0 Sharpe | -0.143708 | -0.126946 | +0.000300 | -0.144008 |
| Cluster - BICS sector | annual net return | -0.003544 | -0.001004 | +0.002680 | -0.006224 |
| | annual volatility | -0.011919 | -0.014115 | +0.006700 | -0.018619 |
| | RF=0 Sharpe | -0.090499 | -0.075833 | +0.051000 | -0.141499 |
| Rolling-Ward HRP - flat ERC | annual net return | +0.002120 | +0.001701 | +0.001700 | +0.000420 |
| | annual volatility | -0.004392 | -0.003958 | -0.003960 | -0.000432 |
| | RF=0 Sharpe | +0.034749 | +0.031123 | +0.031000 | +0.003749 |
| Rolling-Ward HRP - single-link HRP | annual net return | +0.001423 | +0.000977 | +0.000980 | +0.000443 |
| | annual volatility | -0.001253 | -0.001176 | -0.001180 | -0.000073 |
| | RF=0 Sharpe | +0.016233 | +0.013372 | +0.013000 | +0.003233 |

The headline-window correction does **not** reproduce the lost 2026-08-17 U1 signal
quotations. Its residual gaps are material and cannot be attributed to the full-range versus
headline-window mismatch. Under the roadmap's number-traceability rule, those historical
signal quotations must not be used; the manuscript should use the G0 values if the owner
freezes them. The U1 risk values remain directionally and numerically close to the earlier
record, but G0 is the correctly labelled headline-window source for those rows as well.

## Acceptance and verification

| Acceptance check | Measured | Tolerance | Result |
|---|---:|---:|---|
| F0 NAV fingerprints matched | 2/2 | 2/2 | PASS |
| Windowed performance rows | 10 | 10 | PASS |
| Windowed CI rows | 12 | 12 | PASS |
| Reconciliation rows | 12 | 12 | PASS |
| Monthly observations per row | 202 | 202 | PASS |
| Maximum point recomputation error | 0.0 | `<=1e-12` | PASS |
| Bootstrap block length | 6 | 6 | PASS |
| Bootstrap draws | 2,000 | 2,000 | PASS |
| Bootstrap seed | 20260813 | 20260813 | PASS |
| Backtest/optimizer/estimator calls | 0 | 0 | PASS |
| Files written outside `finalisation/g0/` | 0 | 0 | PASS |
| U2/U3 and frozen F6 guard files changed | 0/5 | 0/5 | PASS |
| Deterministic artifacts | 5/5 byte-identical | 5/5 | PASS |
| Focused pytest | 9/9 passed | all pass | PASS |
| Isolated Ruff E/F/W | 0 findings | 0 | PASS |
| Git staging, commit, tag, or push | 0 | 0 | PASS |

The independent numerical check recomputed annualised return, volatility, and RF=0 Sharpe
from a synthetic monthly NAV without calling the F6 statistic implementation. The window
test was also proved fail-before-pass by deliberately selecting the first, rather than the
last, pre-start observation: it failed on 2009-08-19 versus the required 2009-08-26, then
passed after restoration. One verification invocation omitted the required output-root
environment variable and stopped at that explicit precondition; the final environment-bound
run passed all nine tests and did not alter empirical outputs.

## Deviations and open items

There was no numerical-path deviation, refit, specification change, or output-scope breach.
The only open item is the binding owner decision on which traceable U1 rows enter the
manuscript. F8 has not started.

## GATE REQUEST

Please approve G0 and freeze the U1 rows of `tab:signal` and `tab:risk`, their CI companion
columns, and the U1 manuscript narrative to the values in
`finalisation/g0/u1_windowed_performance.csv` and `u1_windowed_cis.csv`. This ruling retires
the unreproduced 2026-08-17 U1 signal quotations. Per the roadmap, F8 remains blocked until
this gate is recorded.
