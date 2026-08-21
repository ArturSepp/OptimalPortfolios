# Stage E2b execution report - calibrated M1-star extension

**Date:** 2026-08-13  
**Roadmap:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`  
**Owner dispatch:** `2026-08-13_owner_E2_gate.md`  
**Status:** COMPLETE; all E2b acceptance lines PASS

## Calibration used

The exact owner-confirmed four-decimal level-form deltas are recorded in
`replication/configs.py`:

| Universe | Frequency | Delta |
|---|---|---:|
| U2 futures | W-WED | 0.0691 |
| U3 MAC | ME | 0.0830 |
| U3 MAC | QE | 0.1609 |
| U1 MSCI US | W-WED | 0.0866 |

U3 ran one canonical smoothing precompute per frequency with its own delta. The harness did
not average the two values. A regression test observes the actual `LassoModel.smoother_delta`
passed to each precompute and asserts `{ME: 0.0830, QE: 0.1609}`. Reintroducing the prohibited
`QE = ME = 0.0830` defect made both the registry and precompute-path tests fail before the
confirmed value was restored. `M2_star` remains unset.

## Acceptance and runtime

Runner: `papers/cluster_lineage_2026/replication/estimate.py`  
Independent validator: `papers/cluster_lineage_2026/replication/validate_e2b.py`  
Cache root: `C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\`

| Universe | Schedule | Snapshots | Injection comparisons | Match share | Precompute s | Fit s | Cache bytes | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| U2 futures | 295 | 295 | 295 | 1.000000 | 7.683137 | 381.419653 | 628,676,902 | PASS |
| U3 MAC | 284 | 284 | 562 | 1.000000 | 32.820190 | 93.830625 | 90,518,232 | PASS |
| U1 MSCI US | 240 | 238 | 238 | 1.000000 | 326.570380 | 777.663857 | 657,475,848 | PASS |

No mismatched dates or temporary files were present. U1's difference of -2 is the approved
warmup-empty schedule convention; its 203-date headline window remains reporting-only.
Runtime rows were appended to `e2_runtime_cache.csv` without replacing the cold E2 rows.

## Verification

Preflight output:

```text
All checks passed!
............                                                             [100%]
```

Independent validation opened all 817 E2b pickles and produced:

```text
m1_star_cache_grid: PASS (3 rows, 817 readable pickles, no temporary files)
m1_star_injections: PASS (295 U2, 562 U3, 238 U1; all share 1.0)
u3_frequency_deltas: PASS (ME=0.0830, QE=0.1609; no average)
```

No files were staged or pushed. E3 may consume the complete eight-config caches. No gate is
requested; the owner already dispatched E3.
