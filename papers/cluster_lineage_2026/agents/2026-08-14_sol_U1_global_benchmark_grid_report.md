# U1 original-universe global-benchmark grid report

**Date:** 2026-08-14  
**Status:** COMPLETE  
**Runner:** `papers/cluster_lineage_2026/replication/run_u1_global_grid.py`  
**Validator:** `papers/cluster_lineage_2026/replication/validate_u1_global_grid.py`

## Scope

The grid uses the original point-in-time eligible U1 universe, including securities without
GICS metadata. Global momentum rank is the sole payoff benchmark. No taxonomy leg is
consumed or reported. EW-all is used only as the market reference for alpha/beta columns.

The primary group-equal construction is fixed for cluster legs. The grid crosses:

- q = 0.30, 0.25, 0.20, 0.15, 0.10;
- baseline, M0_quarterly_hold, M1_delta_0.02, M1_delta_0.05,
  M1_delta_0.10, M1_star, M2_lambda_0.5, and M2_lambda_0.7;
- headline 2009-08-31 through 2026-06-30 and the separately labelled full panel.

Scores, point-in-time eligibility, 10 bp costs, lag 1, and ME schedules are frozen. All
cluster partitions are read from the existing 238-date E2 caches under
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/msci_us/<config>/`.
No clustering was re-estimated.

Outputs are under
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/global_benchmark_grid/msci_us/`.

## Acceptance

| Check | Measured | Tolerance | Verdict |
|:--|--:|--:|:--|
| cluster-grid coverage | 8 configs x 5 q x 2 windows = 80 | 80 | PASS |
| global rows | 5 q x 2 windows = 10 | 10 | PASS |
| maximum weight-sum absolute error | 2.220446e-16 | <= 1e-12 | PASS |
| maximum group-budget absolute error | 3.469447e-18 | <= 1e-15 | PASS |
| construction acceptance | 80/80 | 80/80 | PASS |
| overlap regression to completed q sweep | maximum error 0.000e+00 | <= 1e-12 | PASS |
| numerical deterministic replay | 6/6 artifacts byte-identical | 6/6 | PASS |
| taxonomy rows or comparison columns | 0 | 0 | PASS |

One complete pass took 424.0 seconds. The deterministic two-pass execution completed without
an acceptance failure.

```text
U1 global-benchmark grid independent validation: PASS
grid: 8 configs x 5 q values x 2 windows = 80 cluster rows
construction: 80/80 PASS
overlap regression max absolute error: 0.000e+00
headline raw winner: M1_delta_0.10 at q=0.30 (fidelity rejected)
headline admissible winner: M0_quarterly_hold at q=0.30
headline admissible configs beating global on return and Sharpe: 0
determinism: 6/6 numerical artifacts byte-identical
```

## Headline result

There is no fidelity-admissible group-equal configuration that beats its same-q global
benchmark on both annualised net return and Sharpe in the headline window. The global q=0.30
leg also has higher return and Sharpe than every cluster row anywhere in the grid.

| candidate | status | q | net return | Sharpe | turnover | return vs same-q global | Sharpe vs same-q global | turnover vs same-q global |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| global | benchmark | 0.30 | 0.066409 | 0.512965 | 2.321201 | — | — | — |
| M1_delta_0.10 | REJECTED_FIDELITY | 0.30 | 0.060148 | 0.496417 | 2.446908 | -0.006261 | -0.016548 | 0.125707 |
| M0_quarterly_hold | IN_BAND | 0.30 | 0.056430 | 0.471525 | 2.885653 | -0.009979 | -0.041439 | 0.564452 |
| baseline | REFERENCE | 0.20 | 0.056295 | 0.470865 | 4.070638 | -0.001908 | 0.030403 | 1.343819 |
| M1_delta_0.02 | IN_BAND | 0.30 | 0.054210 | 0.456800 | 3.081643 | -0.012199 | -0.056164 | 0.760442 |

The raw highest-Sharpe cluster row is M1_delta_0.10 at q=0.30, but it loses to global at
the same q and was rejected by E3b fidelity. The highest-Sharpe admissible cluster row is
M0_quarterly_hold at q=0.30; it also loses to global materially.

## Same-q rows that beat global on both headline metrics

Only three headline rows beat their same-q global leg on both return and Sharpe, and every
one is fidelity-rejected:

| config | q | return | Sharpe | turnover | return delta | Sharpe delta | turnover delta | fidelity |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| M1_delta_0.05 | 0.20 | 0.058374 | 0.481991 | 2.931361 | 0.000170 | 0.041529 | 0.204542 | REJECTED |
| M1_delta_0.10 | 0.20 | 0.058946 | 0.485095 | 2.723840 | 0.000743 | 0.044633 | -0.002979 | REJECTED |
| M1_delta_0.10 | 0.10 | 0.055386 | 0.453826 | 3.096616 | 0.003693 | 0.086128 | -0.074576 | REJECTED |

M1_delta_0.10 at q=0.20 is the strongest same-threshold raw candidate: it adds 7 bp/year
of return and 0.045 Sharpe while turnover is essentially unchanged. The return edge is very
small and this is an in-sample grid winner; no statistical or multiple-testing-adjusted
claim is made.

## Best q by config

The E3b fidelity status is binding. `IN_BAND_FULL_ONLY` is not admissible in the headline
window.

| window | config | status | best q by Sharpe | return | Sharpe | turnover | return vs global | Sharpe vs global | turnover vs global |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| headline | M1_delta_0.10 | REJECTED | 0.30 | 0.060148 | 0.496417 | 2.446908 | -0.006261 | -0.016548 | 0.125707 |
| headline | M1_delta_0.05 | REJECTED | 0.25 | 0.058612 | 0.483502 | 2.751302 | -0.003127 | 0.008399 | 0.228541 |
| headline | M0_quarterly_hold | IN_BAND | 0.30 | 0.056430 | 0.471525 | 2.885653 | -0.009979 | -0.041439 | 0.564452 |
| headline | baseline | REFERENCE | 0.20 | 0.056295 | 0.470865 | 4.070638 | -0.001908 | 0.030403 | 1.343819 |
| headline | M1_star | REJECTED | 0.30 | 0.056680 | 0.468466 | 2.514564 | -0.009729 | -0.044498 | 0.193363 |
| headline | M2_lambda_0.7 | IN_BAND | 0.30 | 0.054661 | 0.458291 | 3.461493 | -0.011748 | -0.054674 | 1.140292 |
| headline | M1_delta_0.02 | IN_BAND | 0.30 | 0.054210 | 0.456800 | 3.081643 | -0.012199 | -0.056164 | 0.760442 |
| headline | M2_lambda_0.5 | IN_BAND | 0.15 | 0.053089 | 0.444482 | 4.069597 | -0.005186 | 0.018505 | 1.105570 |
| full | M1_delta_0.10 | REJECTED | 0.30 | 0.033235 | 0.288296 | 2.963753 | -0.000698 | 0.005444 | 0.276207 |
| full | M1_delta_0.05 | IN_BAND_FULL_ONLY | 0.30 | 0.032495 | 0.283732 | 3.271975 | -0.001437 | 0.000881 | 0.584429 |
| full | M1_star | REJECTED | 0.30 | 0.031018 | 0.274033 | 3.047546 | -0.002914 | -0.008819 | 0.360000 |
| full | baseline | REFERENCE | 0.30 | 0.027090 | 0.251154 | 4.585877 | -0.006843 | -0.031698 | 1.898331 |
| full | M0_quarterly_hold | IN_BAND | 0.30 | 0.025021 | 0.238310 | 3.408872 | -0.008912 | -0.044542 | 0.721326 |
| full | M2_lambda_0.5 | IN_BAND | 0.30 | 0.024866 | 0.237935 | 4.312910 | -0.009067 | -0.044917 | 1.625364 |
| full | M2_lambda_0.7 | IN_BAND | 0.30 | 0.024714 | 0.236864 | 4.165024 | -0.009219 | -0.045987 | 1.477478 |
| full | M1_delta_0.02 | IN_BAND | 0.25 | 0.023228 | 0.227109 | 3.835671 | -0.005271 | -0.024048 | 0.914296 |

## Interpretation

1. Stronger partition-bonus smoothing improves the raw payoff frontier: M1_delta_0.05 and
   M1_delta_0.10 occupy the best headline rows. However, that is exactly the region rejected
   by the E3b taxonomy-fidelity band. The grid exposes a stability/payoff versus fidelity
   trade-off; it does not identify an admissible global-beating method.
2. Among admissible smoothed methods, M0 quarterly hold at q=0.30 has the highest absolute
   headline Sharpe and much lower turnover than baseline, but global q=0.30 still dominates
   it on return, Sharpe, and turnover.
3. Baseline q=0.20 improves Sharpe versus same-q global by 0.030 but loses 19 bp/year of
   return and adds 1.34 annual turns. It does not dominate global.
4. The full-panel series is less stable as a selection guide. M1_delta_0.05 is the best
   admissible full-panel smoother, while it fails headline fidelity. Full-panel findings
   remain labelled warmup/robustness and do not override the headline result.
5. The retained asset-equal construction previously allowed baseline and M1_delta_0.02 at
   q=0.20 to beat same-q global on return and Sharpe. This grid uses the binding group-equal
   primary construction; weighting construction, not only clustering method, materially
   changes the payoff conclusion.

## Deliverables

- `performance.csv`: 90 global and cluster performance rows.
- `comparison_vs_global.csv`: all 80 cluster-minus-global rows.
- `rankings.csv`: raw and fidelity-admissible rankings.
- `config_summary.csv`: best q by Sharpe for each config/window.
- `construction_diagnostics.csv`: group counts, selected breadth, effective holdings, and
  numerical errors.
- `acceptance.csv`: all 80 construction verdicts.
- `runtime.csv`: execution timing.
- `determinism.csv`: numerical artifact hashes.

No accepted output or clustering cache was changed. Nothing was staged or pushed; the
runner, validator, report, and outputs remain local under the ignored paper tree.
