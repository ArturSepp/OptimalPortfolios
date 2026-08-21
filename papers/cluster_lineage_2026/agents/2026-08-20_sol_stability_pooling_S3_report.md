# Stability-pooled z-score S3 MAC diagnostic report

**Date:** 2026-08-20  
**Executor:** Sol  
**Proposer:** Ben (Monday TAA meeting)  
**Status:** COMPLETE — frozen seven-cell grid executed; baseline-reference deviation disclosed

## Outcome

All seven frozen cells completed on the constrained MAC production pipeline with zero rejected
solves in the main grid. The strongest full-sample cell is the V3 comparison arm at the 36-month
co-association window: TAA Sharpe rises from 1.068978 to 1.138897 and annual turnover falls from
0.992292 to 0.834140. V3 also increases ex-post tracking error by 0.003102 and ex-ante TRE by
0.002303, and it breaks strict asset-class neutrality by construction. It is not silently treated
as a production-eligible variance-only arm.

V1 and V2 preserve the cluster-mean numerator. Their 36-month cells both improve full-sample
Sharpe and turnover: V1 by +0.029652 Sharpe and -0.088383 annual turnover; V2 by +0.018823 and
-0.107604 respectively. V2/72 is the only cell with a negative full-sample Sharpe delta.

## Frozen run

- Runner: `papers/cluster_lineage_2026/replication/run_stability_pooling_mac.py`.
- Output/cache root:
  `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/stability_pooling/mac/`.
- Shared fitted-input cache: `shared_pipeline_inputs.pkl` (100,901,170 bytes).
- MAC row: `MAC_CONSTRAINED_BATCH`, constrained, `PROD_MOM_BETA_CLUSTER`,
  `MATF_CUSTOM`, returns vintage `20260810_APAC_ROSAA_Fund_and_Index_Data`.
- Risk model: raw correlation, `ClusterCorrelationTransform.NONE`; no de-PC1 interaction.
- TAA window/frequency: 2004-12-31 through 2026-07-31, month end.
- Co-association windows: 36 and 72 observed monthly partitions, derived from production span 36.
- Production rebalancing cost: 0.0, which is the current MAC production setting.
- Performance convention: quarterly RF=0 Sharpe; cached IRX rates are supplied to the production
  report parameters for the rates-aware excess-alpha convention.
- Runtime: 3,906.04 seconds. The covariance/SAA fit was performed once and shared.

## Measured grid

Tracking-risk figures and turnover are annualized. Deltas are cell minus V0.

| cell | pooling | window | Sharpe | ex-post TE | ex-ante TRE | annual turnover | max DD | Sharpe delta | TE delta | TRE delta | turnover delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | none | 36 | 1.068978 | 0.031291 | 0.028320 | 0.992292 | -0.249967 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| V1_1x | cluster variance | 36 | 1.098630 | 0.031705 | 0.028285 | 0.903909 | -0.222094 | +0.029652 | +0.000415 | -0.000036 | -0.088383 |
| V1_2x | cluster variance | 72 | 1.081120 | 0.031685 | 0.028520 | 0.905934 | -0.223713 | +0.012142 | +0.000394 | +0.000200 | -0.086358 |
| V2_1x | asset variance | 36 | 1.087801 | 0.031266 | 0.028189 | 0.884688 | -0.226478 | +0.018823 | -0.000025 | -0.000131 | -0.107604 |
| V2_2x | asset variance | 72 | 1.066526 | 0.031089 | 0.028221 | 0.897479 | -0.229495 | -0.002452 | -0.000201 | -0.000100 | -0.094813 |
| V3_1x | cluster mean + variance | 36 | 1.138897 | 0.034392 | 0.030623 | 0.834140 | -0.211537 | +0.069920 | +0.003102 | +0.002303 | -0.158152 |
| V3_2x | cluster mean + variance | 72 | 1.100027 | 0.034656 | 0.031087 | 0.838762 | -0.223252 | +0.031049 | +0.003365 | +0.002767 | -0.153530 |

Both V3 rows are comparison arms and are flagged in `metrics.csv` as breaking strict asset-class
neutrality.

## Production-reference reconciliation

The roadmap records the 12 August accepted raw-production Sharpe of 1.15. The current V0 is
1.068978, a difference of -0.081022. This is not a hidden de-PC1 run: the runtime model transform
is `NONE`. V0 agrees with the newer raw-production factsheet
`C:/Users/artur/OneDrive/analytics/outputs/mac_constraint_20260819_0908.pdf` at every displayed
headline field: 447% cumulative return, 1.07 Sharpe, 99% four-quarter average turnover, and -25%
maximum drawdown. The roadmap reference is therefore a dated production-vintage reference, not
the current production result. All experimental deltas use the exact same-process V0 above.

## Acceptance

| check | measured | tolerance | status |
|---|---:|---:|---|
| frozen grid | 7 cells | 7 | PASS |
| main-grid rejected/infeasible solver fallbacks | 0 | 0 | PASS |
| V0 exact alpha replay | byte-exact frame | exact | PASS |
| BAB negation | 7/7 cells exact | 7/7 | PASS |
| co-association windows | [36, 72] | runtime span x {1,2} | PASS |
| production transform | `NONE` | no de-PC1 | PASS |
| V3 neutrality disclosure | 2/2 V3 rows flagged | all V3 rows | PASS |
| rates data present | yes | required | PASS |
| current production factsheet match | 1.068978, displays 1.07 | displays 1.07 | PASS |
| roadmap 12-Aug Sharpe reference | 1.068978; difference -0.081022 | 1.15 | DEVIATION — superseded production vintage |

Generated evidence is in `metrics.csv`, `bab_sign.csv`, `run_parameters.csv`, the per-cell
`cells/` directories, the run log, and `source_manifest.csv` under the output root. No files were
staged, committed, pushed, tagged, or released.
