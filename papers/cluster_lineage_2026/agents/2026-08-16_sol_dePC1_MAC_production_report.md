# De-PC1 MAC constrained production-backtest report

**Date:** 2026-08-16  
**Executor:** sol  
**Status:** COMPLETE  
**Scope:** owner-requested production diagnostic following the de-PC1 FactorLasso implementation  
**Repository actions:** no staging, commit, push, tag, publication, or release

## Outcome

The frozen `MAC_CONSTRAINED_BATCH` production row completed with the local FactorLasso
development source and `ClusterCorrelationTransform.REMOVE_PC1`. The standard 26-page A4
factsheet was produced successfully. Execution Excel and execution-policy outputs were disabled,
and the output directory contains zero `.xlsx` files.

Relative to the latest accepted raw-correlation MAC production factsheet
`mac_constraint_20260812_0602.pdf`, de-PC1 lowered TAA return and risk, but the return reduction
was larger: the QE-frequency factsheet Sharpe moved from 1.15 to 1.09. This is consistent with the
paper experiment's classification of de-PC1 as a robustness specification rather than a proposed
production replacement.

## Frozen run specification

| item | value |
|---|---|
| production batch | `MAC_CONSTRAINED_BATCH` imported from `rosaa.products.funds.run_production` |
| mandate | `MAC` |
| constrained | `True` |
| signal | `PROD_MOM_BETA_CLUSTER` |
| returns input | `20260810_APAC_ROSAA_Fund_and_Index_Data` |
| factor model | `MATF_CUSTOM` |
| only numerical model-field change | `cluster_correlation_transform: none -> remove_pc1` |
| production date specification | `FUND_BACKTEST_DATES` |
| run id | `20260816_depc1` |
| execution Excel | `False` |
| execution-policy outputs | `False` |
| output-only product name | `mac_depc1` |

The runner copied the nested `LassoModel` and asserted that the transform was the only changed
model field before dispatching the ordinary `run_production_backtest` entry point. The baseline
product name was not used, so accepted production artifacts were not overwritten.

## Factsheet comparison

The table uses the QE-frequency convention printed in both factsheets over
31 December 2004 through 31 July 2026. Baseline numbers are read from the latest accepted
production factsheet; deltas are de-PC1 minus baseline. Display rounding is therefore the same as
the factsheets.

| TAA statistic | accepted raw | de-PC1 | delta |
|---|---:|---:|---:|
| cumulative total return | 490% | 419% | -71 pp |
| annual return | 8.6% | 7.9% | -0.7 pp |
| volatility | 7.5% | 7.3% | -0.2 pp |
| Sharpe, rf=0 | 1.15 | 1.09 | -0.06 |
| excess Sharpe | 0.90 | 0.84 | -0.06 |
| maximum drawdown | -23% | -24% | -1 pp |
| annual alpha vs benchmark | 3.5% | 3.0% | -0.5 pp |
| beta vs benchmark | 0.86 | 0.83 | -0.03 |
| R-squared | 88% | 88% | 0 pp |
| average four-quarter turnover | 101% | 108% | +7 pp |
| latest four-quarter turnover | 67% | 64% | -3 pp |

SAA was much less affected at displayed precision: annual return, volatility, and Sharpe remained
5.6%, 9.1%, and 0.62, while cumulative total return moved from 225% to 227%. The benchmark stayed
at 212% cumulative return. The cumulative TAA-minus-SAA active result moved from 57% to 44%.

## Deliverables and provenance

**Output directory:**
`C:/Users/artur/OneDrive/analytics/outputs/depc1_mac_production_20260816/`

| artifact | SHA-256 |
|---|---|
| `mac_depc1_constraint_20260816_depc1.pdf` | `28f6f193bd652f85604ae07342a65fe106a665c282f2043ee5af8424dd47e8b5` |
| `navs.csv` | `c2cb8c81b4add5b91a48fbd9507795afe05d4e43190f6eea4ba9deac1110b430` |
| `performance_table.csv` | `67dc3b98fee6f0feaa09f9faad836fab16ce543add291c6f03bb98be682b7096` |
| `clusters.csv` | `2aa5ce381586eef9f492eb343003a4fc27d68708a146dfc8e734fbeac28c5c9c` |
| FactorLasso `cluster_utils.py` | `86bf04e5965a787ab6d2b5bf6a8f914127c723575686c42cd902ccde24f350b3` |
| QIS `strategy_benchmark_tre_factsheet.py` | `6a2812a1a8493159eb7b26b3d8bbbb7fbecb456e95b713b56a74e9fa8dc246d9` |
| `run_production_depc1.py` | `1fabf9fcc6b6f7c52f766515f0218e0118a861ebfdcdef0df6652fd44fd77cb1` |

Additional artifacts are `latest_clusters.csv`, `run_manifest.csv`, and
`prod_run_20260816_depc1.log`. The manifest records the local FactorLasso import path, runner and
source hashes, transform, snapshot dates, and factsheet hash.

## Acceptance and verification

| check | measured | tolerance | status |
|---|---:|---:|---|
| frozen production batch unit tests | 5 passed | all pass | PASS |
| runner lint | 0 findings | 0 | PASS |
| local FactorLasso import | local development checkout | exact path | PASS |
| only `LassoModel` field changed | 1: `cluster_correlation_transform` | exactly 1 | PASS |
| production pipeline exit code | 0 | 0 | PASS |
| covariance snapshots | 284 | every scheduled snapshot | PASS |
| covariance dates | 2002-12-31 to 2026-07-31 | recorded schedule | PASS |
| snapshots with clusters | 284/284 | 100% | PASS |
| report issues | `{}` | empty | PASS |
| factsheet pages | 26 | complete report | PASS |
| factsheet page size | A4 | A4 | PASS |
| execution Excel files | 0 | 0 | PASS |
| rendered pages inspected | 26/26 | 100% | PASS |

Visual inspection exposed a QIS presentation defect unrelated to de-PC1: the tracking-error
report drew the performance table twice on one axis, overlaying the raw floats on the formatted
table. A focused regression test first failed with two table artists. The report-only path was
then changed to retain the raw frame without rendering it a second time. The focused QIS suite
passed (`5 passed`), the table-artist count became one, and the raw performance-frame SHA-256
remained exactly
`06af6f5152eec02fa7cec6f5420fb65370f1ab3afd44481f0bb04674fec9b97e` before and after the fix.
The pipeline was rerun and all 26 final pages were re-rendered; the corrected table is legible.

Expected production warnings were retained in the log: early-history assets with fewer than the
frozen 12-observation warmup were zeroed, and unpriced legacy benchmark legs remained in cash.
Neither warning stopped the run or introduced a report issue.

### Non-fatal production diagnostics

The solver summary is not silently treated as clean merely because `report_issues` is empty:

- 380 solver calls completed with 0 rejected solves, 0 numerical blow-ups, 0 infeasible
  fallbacks, and a 0.0% fallback rate;
- 10 rebalances relaxed a group bound, with maximum single relaxation 0.0060;
- the raw covariance input was reported ill-conditioned on 285/285 solver checks, with worst
  minimum eigenvalue `-5.63e-15`, at numerical-zero scale;
- the benchmark lay outside the pointwise optimisation box on 285/285 rebalances; the most
  frequent finding was index 152 below its floor on 284 dates;
- 95 aligned constraint sets dropped zero-loading groups, most frequently Liquidity;
- 527 warnings were captured, of which 522 were the frozen FactorLasso warmup-zeroing warning.

These are production input/constraint diagnostics, not report-generation failures. The run
accepted every solve and produced the requested backtest and factsheet. No claim is made here
that the benchmark-box and group-relaxation findings are caused by de-PC1; a matched same-process
raw diagnostic run would be required to attribute them. The accepted raw run's archived log
already shows the same broad input pattern: covariance ill-conditioning on 285/285 checks,
benchmark outside the box on 285/285 rebalances, 95 zero-loading group drops, and 522 warmup
warnings. That raw run recorded six infeasible-fallback solver outcomes versus zero here, but the
logs come from different run dates/code environments, so this difference is reported without a
de-PC1 causal interpretation.

## Runner

`../../../rosaa/products/funds/analysis/run_production_depc1.py`
