# Stage E2 execution report - rolling estimation runs

**Date:** 2026-08-13  
**Roadmap:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`  
**Owner dispatch:** `2026-08-13_owner_E1_gate.md`  
**Status:** COMPLETE; all E2 acceptance lines PASS

## Outcome

The fixed seven-config grid was estimated in the required U2 futures -> U3 MAC -> U1 MSCI
US order. The run produced 5,719 readable point-in-time pickle files across 21
universe/config rows. All six smoothed configurations per universe used the canonical
two-pass `compute_rolling_smoothed_clusters` path and supplied clusters, linkages, and cutoffs
together to the estimator injection hooks.

Runner and validator:

- `papers/cluster_lineage_2026/replication/estimate.py`
- `papers/cluster_lineage_2026/replication/estimate_test.py`
- `papers/cluster_lineage_2026/replication/validate_e2.py`

Cache root:

`C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\`

Per-date cache pattern:

`<cache root>/<universe>/<config>/YYYYMMDD.pkl`

Evidence tables at the cache root:

- `e2_runtime_cache.csv`
- `e2_baseline_rho_bar.csv`
- `e2_mac_production_replication.csv`

The runner was cache-first and used `ProcessPoolExecutor(max_workers=4)`. Atomic temporary
files were renamed only after successful serialization. The independent audit opened all
5,719 pickles and found no temporary files.

## Acceptance results

### Snapshot schedules

| Universe | Frozen schedule | Cached snapshots per config | Tolerance | Measured result |
|---|---:|---:|---:|---|
| U2 futures | 295 | 295 | exact | PASS |
| U3 MAC | 284 | 284 | exact | PASS |
| U1 MSCI US | 240 | 238 | within +/-2 | PASS, difference -2 |

U1's two absent snapshots are the owner-approved warmup-empty dates 2006-08-31 and
2006-09-30. The first active date is 2006-10-31. Across the 238 cached U1 dates, eligible
membership was min/median/max 523/619/641. Each worker log line recorded the date and member
count. The same date set is present under all seven configurations in each universe.

The owner-approved U1 headline window remains reporting-only: 203 estimation dates from
2009-08-31 through 2026-06-30. The complete non-empty E2 cache spans 2006-10-31 through
2026-07-31 and is labelled warmup/robustness for E3/E5; it is not pooled into headline
metrics.

### Smoothed partition injection

Every smoothed fitted partition was compared label-invariantly with its corresponding
precomputed injected partition. The U3 count is frequency-date comparisons: 284 ME plus 278
non-empty QE partitions.

| Universe | Config | Comparisons | Match share | Mismatched dates | Result |
|---|---|---:|---:|---|---|
| U2 | M0_quarterly_hold | 295 | 1.000000 | none | PASS |
| U2 | M1_delta_0.02 | 295 | 1.000000 | none | PASS |
| U2 | M1_delta_0.05 | 295 | 1.000000 | none | PASS |
| U2 | M1_delta_0.10 | 295 | 1.000000 | none | PASS |
| U2 | M2_lambda_0.5 | 295 | 1.000000 | none | PASS |
| U2 | M2_lambda_0.7 | 295 | 1.000000 | none | PASS |
| U3 | M0_quarterly_hold | 562 | 1.000000 | none | PASS |
| U3 | M1_delta_0.02 | 562 | 1.000000 | none | PASS |
| U3 | M1_delta_0.05 | 562 | 1.000000 | none | PASS |
| U3 | M1_delta_0.10 | 562 | 1.000000 | none | PASS |
| U3 | M2_lambda_0.5 | 562 | 1.000000 | none | PASS |
| U3 | M2_lambda_0.7 | 562 | 1.000000 | none | PASS |
| U1 | M0_quarterly_hold | 238 | 1.000000 | none | PASS |
| U1 | M1_delta_0.02 | 238 | 1.000000 | none | PASS |
| U1 | M1_delta_0.05 | 238 | 1.000000 | none | PASS |
| U1 | M1_delta_0.10 | 238 | 1.000000 | none | PASS |
| U1 | M2_lambda_0.5 | 238 | 1.000000 | none | PASS |
| U1 | M2_lambda_0.7 | 238 | 1.000000 | none | PASS |

Independent validation of the persisted runtime rows found 18/18 smoothed universe/config
rows at match share 1.0.

### U3 production replication at 2026-06-30

| Measure | Measured | Tolerance | Result |
|---|---:|---:|---|
| Common named assets | 185 | reported | PASS |
| Paper / production full-panel assets | 187 / 187 | reported | PASS |
| Pairwise Rand | 0.996768507638073 | >= 0.99 | PASS |
| Modal agreement | 0.978378378378378 | >= 0.97 | PASS |

The two full panels each contain 187 assets and have 185 common normalized display names.
The independent validator recomputed Rand with `sklearn.metrics.rand_score` and reproduced
0.996768507638073 exactly; it independently reproduced modal agreement
0.978378378378378.

### Baseline correlation calibration inputs

`rho_bar` is the pooled median within-cluster pairwise correlation over baseline dates, using
the same EWMA Pearson dependence calculation and native span as the estimator.

| Universe | Frequency | `rho_bar` | Pair observations | Dates | Span |
|---|---|---:|---:|---:|---:|
| U2 futures | W-WED | 0.682898502438977 | 113,384 | 295 | 156 |
| U3 MAC | ME | 0.795196858659528 | 484,797 | 284 | 36 |
| U3 MAC | QE | 0.794647033373131 | 16,156 | 278 | 12 |
| U1 MSCI US | W-WED | 0.622741298852166 | 1,017,296 | 238 | 156 |

These are the requested inputs for the owner's later `M1_star` confirmation. No calibrated
delta was written, and both `M1_star` and `M2_star` remain unset. No E2b run was performed.

## Runtime and cache size

Precompute and fit are separate wall-clock measurements. Cache sizes are MiB (2^20 bytes).
All rows below are cold-run measurements; the U1 baseline row was retained after a later
cache-hit restart.

| Universe | Config | Precompute min | Fit min | Cache MiB |
|---|---|---:|---:|---:|
| U2 | baseline | 0.00 | 6.48 | 599.4 |
| U2 | M0_quarterly_hold | 0.16 | 6.15 | 599.6 |
| U2 | M1_delta_0.02 | 0.11 | 5.99 | 599.6 |
| U2 | M1_delta_0.05 | 0.14 | 6.31 | 599.6 |
| U2 | M1_delta_0.10 | 0.14 | 5.93 | 599.6 |
| U2 | M2_lambda_0.5 | 0.14 | 6.52 | 599.6 |
| U2 | M2_lambda_0.7 | 0.15 | 6.12 | 599.6 |
| U3 | baseline | 0.00 | 1.34 | 85.9 |
| U3 | M0_quarterly_hold | 0.51 | 1.71 | 86.3 |
| U3 | M1_delta_0.02 | 0.52 | 1.66 | 86.3 |
| U3 | M1_delta_0.05 | 0.63 | 1.63 | 86.3 |
| U3 | M1_delta_0.10 | 0.46 | 1.48 | 86.3 |
| U3 | M2_lambda_0.5 | 0.50 | 1.52 | 86.3 |
| U3 | M2_lambda_0.7 | 0.48 | 1.53 | 86.3 |
| U1 | baseline | 0.00 | 15.60 | 618.6 |
| U1 | M0_quarterly_hold | 5.21 | 11.66 | 627.0 |
| U1 | M1_delta_0.02 | 5.08 | 11.83 | 627.0 |
| U1 | M1_delta_0.05 | 5.42 | 11.76 | 627.0 |
| U1 | M1_delta_0.10 | 4.68 | 11.49 | 627.0 |
| U1 | M2_lambda_0.5 | 5.51 | 11.73 | 627.0 |
| U1 | M2_lambda_0.7 | 5.77 | 12.20 | 627.0 |

Per-universe totals:

| Universe | Cache bytes | Cache GiB | Sum of measured precompute + fit wall min |
|---|---:|---:|---:|
| U2 futures | 4,400,528,973 | 4.098 | 44.32 |
| U3 MAC | 633,143,489 | 0.590 | 13.98 |
| U1 MSCI US | 4,593,465,768 | 4.278 | 117.93 |
| Total | 9,627,138,230 | 8.966 | 176.23 |

The summed row times exclude the final baseline-correlation diagnostics and interruption/
restart overhead, so they are not the end-to-end session elapsed time.

## Verification

The pipeline's resumed preflight test output was:

```text
.........                                                                [100%]
```

After adding the cold-runtime retention regression, the complete E2 test module produced:

```text
..........                                                               [100%]
```

The new regression was first run with the guard removed and failed as intended:

```text
FAILED ...test_runtime_report_retains_cold_run_on_cache_hit
E   assert np.int64(0) == 238
```

Focused Ruff `E,F,W` verification for `estimate.py`, `estimate_test.py`, and
`validate_e2.py` produced:

```text
All checks passed!
```

Independent validator output:

```text
cache_grid: PASS (21 rows, 5719 readable pickles, no temporary files)
date_sets: PASS (identical across seven configs within every universe)
injection_rows: PASS (18/18 smoothed universe-config rows at share 1.0)
u1_member_counts: min=523, median=619.0, max=641, dates=238
u3_production_independent: PASS (common=185, sklearn_rand=0.996768507638073, modal=0.978378378378378)
```

## Deviations and resolved execution issues

- The canonical U1 smoother initially received the full 240-date schedule and failed in
  FactorLasso's entrant join on the two warmup-empty dates. `compute_injections` now receives
  the same 238 active dates as the fitter. A regression pins the first active date and count.
  No completed cache or numerical fit was changed.
- Resuming U1 correctly cache-hit all 238 baseline files but initially overwrote the baseline
  cold-runtime evidence with a near-zero cache-hit row. The cold measurements were restored
  (`fit_wall_seconds = 935.879400999998`), and the runtime upsert now preserves any existing
  cold row when a later execution has zero cache misses.
- Final repository inspection found that the obsolete broad
  `papers/cluster_lineage_2026/` ignore rule was still present above the two E0b narrowed
  rules. It was removed. Only `papers/cluster_lineage_2026/data/` and
  `papers/cluster_lineage_2026/msci_us/` remain ignored; `replication/`, `agents/`, and
  `paper/` are now trackable as ruled.

No files were staged or pushed. E3 may start from these accepted caches, with the U1 headline
window convention above. Any calibrated E2b run remains pending explicit owner values for
`M1_star`; `M2_star` also remains unset.
