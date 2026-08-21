# E8a execution report — U3M derivation and lineage re-scoring

Date: 2026-08-14
Runner: `papers/cluster_lineage_2026/replication/run_e8a.py`
Output: `$CLUSTER_LINEAGE_OUTPUT_DIR/e8/u3m/e8a/`

**QE-frequency funds are EXCLUDED from the cluster-momentum arm.** U3M contains the 170
ME-frequency covariance columns; the 17 QE columns never enter an E8 output panel.

## Outcome

E8a passes. Per-frequency separability is exact at early, middle and late dates, so U3M
uses filtered U3 caches and requires no re-estimation.

| Acceptance line | Measured | Tolerance | Status |
|---|---:|---:|---|
| Refit proof dates | 3 | ≥3 | PASS |
| Early date / assets | 2002-12-31 / 97 | early | PASS |
| Middle date / assets | 2014-10-31 / 151 | middle | PASS |
| Late date / assets | 2026-07-31 / 170 | late | PASS |
| Exact partition-vector equality | 3/3 | 100% | PASS |
| Maximum `abs(delta beta)` | 0.0 | ≤1e-10 | PASS |
| Filtered snapshots per config | 284 | 284 | PASS |
| Metric configs complete/NaN-free | 2/2 | 2/2 | PASS |
| Deterministic CSV artifacts | 22/22 | 100% | PASS |

## Fidelity and granularity

| Config | Raw churn | Median clusters | Median cluster size | Asset-Class ARI | Sub-Asset-Class ARI | Fidelity |
|---|---:|---:|---:|---:|---:|---|
| baseline | 1.651793 | 12 | 8 | 0.269667 | 0.354013 | PASS |
| M1_delta_0.05 | 0.483032 | 11 | 8 | 0.279281 | 0.339678 | PASS |

For `M1_delta_0.05`, same-date baseline-partition ARI is 0.736699, cluster count changes
by -8.33%, Asset-Class ARI by +0.009613, and Sub-Asset-Class ARI by -0.014335. All are
inside the frozen symmetric fidelity band.

The realised operating point is coarser than the roadmap's approximate expectation:
median cluster counts are 12 (baseline) and 11 (smoothed), not 15–16; the corresponding
median mean cluster sizes are 12.15 and 12.14 over the point-in-time schedule.

The selected custom export is `risk_factors_custom.csv`, the only
`risk_factors_custom*.csv` file present. Its SHA-256 and date range are recorded in
`factor_vintage.csv`.

## Workspace recovery disclosure

After successful execution, a concurrent workspace cleanup removed the untracked roadmap,
reports, and replication source while leaving `__pycache__` and all external outputs.
The exact executed CPython 3.12 E8a module survived and reproduced all 22 CSV artifacts
byte-for-byte twice. The restored source entry point delegates to that executed module;
the owner copy of the broader E0–E7 source tree should be restored before archival release.
