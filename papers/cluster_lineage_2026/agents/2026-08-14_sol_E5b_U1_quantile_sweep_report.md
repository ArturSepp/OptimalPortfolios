# U1 quantile-threshold backtest report

**Date:** 2026-08-14  
**Status:** COMPLETE  
**Runner:** `papers/cluster_lineage_2026/replication/run_u1_quantile_sweep.py`  
**Validator:** `papers/cluster_lineage_2026/replication/validate_u1_quantile_sweep.py`

## Scope and frozen conventions

The U1 MSCI US backtest was run at selection fractions `q = 0.30, 0.25, 0.20,
0.15, 0.10`. A threshold `q` selects within each group using rank `>= 1-q`.
The current E5b `group_equal` construction is primary for taxonomy and cluster legs;
the global leg retains its asset-equal construction. Both the headline window
(2009-08-31 through 2026-06-30) and the full panel are reported separately.

Only `q` varies. Momentum scores, E2 partitions, point-in-time eligibility, ME schedule,
10 bp costs, one-period implementation lag, and the pair `{baseline,
M1_delta_0.02}` are fixed. Global rank and taxonomy rank are the only payoff
yardsticks. EW-all is used only as the alpha/beta market reference and is not included
in a payoff comparison.

No clustering was re-estimated. The unchanged caches were read from
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/msci_us/<config>/YYYYMMDD.pkl`.
Results are under
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/quantile_sweep/msci_us/`.

## Acceptance and verification

| Check | Measured | Tolerance | Verdict |
|:--|--:|--:|:--|
| requested q grid | 0.30, 0.25, 0.20, 0.15, 0.10 | exact | PASS |
| coverage | 5 q x 2 windows x 4 ranking legs = 40 rows | 40 | PASS |
| portfolio construction rows | 40/40 PASS | 40/40 | PASS |
| maximum weight-sum absolute error | 5.107026e-15 | <= 1e-12 | PASS |
| maximum group-budget absolute error | 1.387779e-17 | <= 1e-15 | PASS |
| selected-asset monotonicity in q | 8/8 window-leg series | 8/8 | PASS |
| q=0.20 regression to accepted E5b | maximum absolute metric error 0.000e+00 | <= 1e-12 | PASS |
| deterministic replay | 5/5 numerical CSV artifacts byte-identical | 5/5 | PASS |
| EW payoff comparisons | 0 | 0 | PASS |

Independent-validator output:

```text
U1 quantile sweep independent validation: PASS
grid: 5 q values x 2 windows x 4 ranking legs = 40 rows
construction: 40/40 rows PASS
selection monotonicity: 8/8 window-leg series PASS
q=0.20 E5b regression max absolute error: 0.000e+00
determinism: 5/5 numerical CSV artifacts byte-identical
```

One complete pass took 345.6 seconds. The full schedule contains one final weight date
that cannot be implemented with lag 1 before the price history ends; it is dropped, and
the last traded weight date is 2026-06-30. As in the accepted U1 backtest, instruments
without a usable price on their selected trade date remain in cash. These are preserved
engine conventions, not sweep-specific deviations.

## Full results

Net return, volatility, alpha, and cost drag are annualised. Sharpe uses the frozen
RF=0 convention. Alpha and beta are versus the EW market reference only.

| window | q | leg | net return | vol | Sharpe | alpha vs EW | beta vs EW | turnover | cost drag bp/yr |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| headline | 0.30 | global | 0.066409 | 0.147009 | 0.512965 | -0.005454 | 0.890232 | 2.321200 | 49.0617 |
| headline | 0.30 | taxonomy | 0.102809 | 0.143517 | 0.757569 | 0.027294 | 0.897964 | 2.296410 | 50.2327 |
| headline | 0.30 | cluster_baseline | 0.055881 | 0.136310 | 0.470017 | -0.014045 | 0.859536 | 3.831930 | 80.3635 |
| headline | 0.30 | cluster_M1_delta_0.02 | 0.054210 | 0.136931 | 0.456800 | -0.015849 | 0.862600 | 3.081640 | 64.4951 |
| headline | 0.25 | global | 0.061740 | 0.150557 | 0.475102 | -0.009910 | 0.896364 | 2.522760 | 53.1266 |
| headline | 0.25 | taxonomy | 0.104168 | 0.145046 | 0.759511 | 0.028580 | 0.899716 | 2.493270 | 54.6104 |
| headline | 0.25 | cluster_baseline | 0.054480 | 0.136247 | 0.460423 | -0.015082 | 0.856180 | 3.886830 | 81.4108 |
| headline | 0.25 | cluster_M1_delta_0.02 | 0.053901 | 0.137628 | 0.453056 | -0.016356 | 0.865993 | 3.143480 | 65.7584 |
| headline | 0.20 | global | 0.058203 | 0.157025 | 0.440462 | -0.013286 | 0.907319 | 2.726820 | 57.2348 |
| headline | 0.20 | taxonomy | 0.106818 | 0.148999 | 0.759496 | 0.030289 | 0.914014 | 2.705630 | 59.4156 |
| headline | 0.20 | cluster_baseline | 0.056295 | 0.137181 | 0.470865 | -0.013238 | 0.856468 | 4.070640 | 85.4129 |
| headline | 0.20 | cluster_M1_delta_0.02 | 0.051731 | 0.137863 | 0.437578 | -0.018120 | 0.863013 | 3.330950 | 69.5698 |
| headline | 0.15 | global | 0.058275 | 0.165756 | 0.425976 | -0.012630 | 0.916040 | 2.964030 | 62.2548 |
| headline | 0.15 | taxonomy | 0.113020 | 0.154697 | 0.773364 | 0.034838 | 0.935404 | 2.919860 | 64.4415 |
| headline | 0.15 | cluster_baseline | 0.051936 | 0.139603 | 0.435392 | -0.017836 | 0.864919 | 4.301690 | 89.9429 |
| headline | 0.15 | cluster_M1_delta_0.02 | 0.045488 | 0.139772 | 0.390968 | -0.024430 | 0.869661 | 3.549110 | 73.7476 |
| headline | 0.10 | global | 0.051694 | 0.183414 | 0.367697 | -0.019030 | 0.951348 | 3.171190 | 66.2218 |
| headline | 0.10 | taxonomy | 0.113650 | 0.160972 | 0.752863 | 0.034989 | 0.950846 | 3.224710 | 71.2180 |
| headline | 0.10 | cluster_baseline | 0.049346 | 0.140399 | 0.416091 | -0.020545 | 0.868603 | 4.421660 | 92.2371 |
| headline | 0.10 | cluster_M1_delta_0.02 | 0.042785 | 0.139997 | 0.372038 | -0.026702 | 0.866369 | 3.680730 | 76.2888 |
| full | 0.30 | global | 0.033933 | 0.175073 | 0.282851 | -0.021273 | 0.891633 | 2.687550 | 55.2705 |
| full | 0.30 | taxonomy | 0.085780 | 0.168971 | 0.576985 | 0.027379 | 0.885646 | 2.661050 | 57.5078 |
| full | 0.30 | cluster_baseline | 0.027090 | 0.161114 | 0.251154 | -0.027132 | 0.851020 | 4.585880 | 93.9125 |
| full | 0.30 | cluster_M1_delta_0.02 | 0.023051 | 0.161082 | 0.226315 | -0.030769 | 0.846130 | 3.766670 | 76.7990 |
| full | 0.25 | global | 0.028499 | 0.179643 | 0.251157 | -0.026438 | 0.900990 | 2.921370 | 59.7933 |
| full | 0.25 | taxonomy | 0.087004 | 0.170416 | 0.580044 | 0.028782 | 0.885106 | 2.891230 | 62.5615 |
| full | 0.25 | cluster_baseline | 0.024864 | 0.161519 | 0.237585 | -0.029120 | 0.849621 | 4.655320 | 95.1382 |
| full | 0.25 | cluster_M1_delta_0.02 | 0.023228 | 0.161956 | 0.227109 | -0.030679 | 0.849124 | 3.835670 | 78.1955 |
| full | 0.20 | global | 0.019464 | 0.187979 | 0.201717 | -0.035009 | 0.917830 | 3.159750 | 64.1239 |
| full | 0.20 | taxonomy | 0.088510 | 0.174134 | 0.579445 | 0.030134 | 0.894011 | 3.145200 | 68.1706 |
| full | 0.20 | cluster_baseline | 0.025122 | 0.163105 | 0.238585 | -0.028755 | 0.851847 | 4.884170 | 99.8614 |
| full | 0.20 | cluster_M1_delta_0.02 | 0.020921 | 0.162739 | 0.213133 | -0.032865 | 0.850132 | 4.074100 | 82.9339 |
| full | 0.15 | global | 0.012659 | 0.197994 | 0.167875 | -0.040525 | 0.928087 | 3.444600 | 69.5013 |
| full | 0.15 | taxonomy | 0.092827 | 0.181490 | 0.585486 | 0.033433 | 0.920158 | 3.405360 | 74.0660 |
| full | 0.15 | cluster_baseline | 0.019836 | 0.166234 | 0.206424 | -0.034149 | 0.861578 | 5.178030 | 105.3928 |
| full | 0.15 | cluster_M1_delta_0.02 | 0.008551 | 0.165670 | 0.138999 | -0.045197 | 0.857989 | 4.356330 | 87.7125 |
| full | 0.10 | global | -0.000640 | 0.218289 | 0.112113 | -0.052491 | 0.967814 | 3.677730 | 73.2939 |
| full | 0.10 | taxonomy | 0.089692 | 0.188406 | 0.555596 | 0.030636 | 0.935332 | 3.747530 | 81.3013 |
| full | 0.10 | cluster_baseline | 0.015056 | 0.166489 | 0.177725 | -0.038881 | 0.861419 | 5.348760 | 108.3795 |
| full | 0.10 | cluster_M1_delta_0.02 | 0.003755 | 0.166538 | 0.110534 | -0.049870 | 0.858420 | 4.534500 | 90.8971 |

## Baseline versus smoothed cluster leg

| window | q | baseline return | M1 return | M1 - base return | baseline Sharpe | M1 Sharpe | M1 - base Sharpe | baseline turnover | M1 turnover | M1 - base turnover |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| headline | 0.30 | 0.055881 | 0.054210 | -0.001671 | 0.470017 | 0.456800 | -0.013216 | 3.831930 | 3.081640 | -0.750283 |
| headline | 0.25 | 0.054480 | 0.053901 | -0.000579 | 0.460423 | 0.453056 | -0.007367 | 3.886830 | 3.143480 | -0.743348 |
| headline | 0.20 | 0.056295 | 0.051731 | -0.004565 | 0.470865 | 0.437578 | -0.033287 | 4.070640 | 3.330950 | -0.739689 |
| headline | 0.15 | 0.051936 | 0.045488 | -0.006448 | 0.435392 | 0.390968 | -0.044424 | 4.301690 | 3.549110 | -0.752575 |
| headline | 0.10 | 0.049346 | 0.042785 | -0.006561 | 0.416091 | 0.372038 | -0.044053 | 4.421660 | 3.680730 | -0.740926 |
| full | 0.30 | 0.027090 | 0.023051 | -0.004039 | 0.251154 | 0.226315 | -0.024838 | 4.585880 | 3.766670 | -0.819211 |
| full | 0.25 | 0.024864 | 0.023228 | -0.001636 | 0.237585 | 0.227109 | -0.010476 | 4.655320 | 3.835670 | -0.819647 |
| full | 0.20 | 0.025122 | 0.020921 | -0.004202 | 0.238585 | 0.213133 | -0.025451 | 4.884170 | 4.074100 | -0.810069 |
| full | 0.15 | 0.019836 | 0.008551 | -0.011285 | 0.206424 | 0.138999 | -0.067425 | 5.178030 | 4.356330 | -0.821698 |
| full | 0.10 | 0.015056 | 0.003755 | -0.011300 | 0.177725 | 0.110534 | -0.067191 | 5.348760 | 4.534500 | -0.814261 |

## Findings

1. The original `q=0.20` remains the headline baseline cluster leg's best point on this
   grid by both net return (5.63%) and Sharpe (0.471). For the smoothed M1 leg, `q=0.30`
   is best on the headline window by net return (5.42%) and Sharpe (0.457).
2. Narrower selections generally increase turnover and cost drag. From `q=0.30` to
   `q=0.10`, headline turnover rises from 3.83 to 4.42 for baseline and from 3.08 to
   3.68 for M1. The same direction holds on the full panel.
3. M1 smoothing reduces annualised turnover at every threshold: 0.740–0.753 in the
   headline window and 0.810–0.822 in the full panel. The turnover benefit is therefore
   robust to the selection threshold.
4. Taxonomy rank has the highest Sharpe among the four ranking legs at every q in both
   windows. Cluster-leg net returns remain below taxonomy at every threshold.
5. Against global rank, headline cluster-baseline Sharpe is higher at q=0.20, 0.15,
   and 0.10, while M1 Sharpe is higher only at q=0.10. No statistical inference was
   requested for this sweep, so these are descriptive differences only.

## Artifacts and repository state

- `performance.csv`: all 40 absolute performance rows.
- `comparison.csv`: 20 cluster rows with explicit deltas versus global and taxonomy.
- `selection_diagnostics.csv`: selected-asset and available-group diagnostics.
- `acceptance.csv`: per-window, per-q, per-leg construction checks.
- `ew_reference.csv`: separate EW market-reference block.
- `runtime.csv`: execution timing.
- `determinism.csv`: replay hashes.

Nothing was staged or pushed. `papers/cluster_lineage_2026/` remains ignored, so this
execution introduced no tracked change. At handoff the shared checkout had been switched
by a separate concurrent task to `codex/readme-truth-pass`, with an unrelated tracked
`README.md` modification; this execution did not touch or reconcile that work.
