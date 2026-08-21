# Stage E3 execution report - stability and theory validation

**Date:** 2026-08-14  
**Roadmap:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`  
**Owner dispatch:** `2026-08-13_owner_E2_gate.md`  
**Numerical status:** COMPLETE; all numerical acceptance checks PASS  
**Artifact status:** BLOCKED only for the three XLSX wrappers; deterministic CSV evidence is complete

## Execution surface

Runner: `papers/cluster_lineage_2026/replication/run_stability.py`  
Independent validator: `papers/cluster_lineage_2026/replication/validate_e3.py`  
E2 cache root: `C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\<universe>\<config>\`  
E3 evidence root: `C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\stability\<universe>\`

Each universe directory contains `metric_suite.csv`, the combined per-date panel,
asset-date margins, 40-bin histogram data per frequency, predicted/realised summaries and
per-date panels, correlations, frequency scaling, kurtosis checks, the five-point frontier,
and a manifest. U1 also contains the exhaustive membership-flow panel. The U1 manifest has
16 metric rows, 21,104 per-date rows, 145,428 margin rows, 7,024 prediction-date rows, 10
frontier rows, and 3,485 flow rows. U2/U3 manifests contain respectively 8/8 metric rows,
14,144/14,352 per-date rows, and 26,617/42,118 margin rows.

The requested workbooks `futures_stability_20260813.xlsx`,
`mac_stability_20260813.xlsx`, and `msci_us_stability_20260813.xlsx` were not emitted. The
required `@oai/artifact-tool` workbook runtime is absent from both supported execution
contexts, and no approved dependency loader is exposed. No unsupported Excel library was
substituted. Full details and the bounded remaining task are recorded in
`2026-08-14_sol_escalation_E3_workbook_runtime.md`.

## Frozen margin and calibration inputs

| Universe/frequency | span | step k | kappa | level delta | innovation overlay |
|---|---:|---:|---:|---:|---:|
| U2 futures W-WED | 156 | 52/12 | 1.612290 | 0.0691 | 0.0227 |
| U3 MAC ME | 36 | 1 | 0.836854 | 0.0830 | 0.0273 |
| U3 MAC QE | 12 | 1 | 1.287959 | 0.1609 | 0.0893 |
| U1 MSCI US W-WED | 156 | 52/12 | 2.124418 | 0.0866 | 0.0285 |

Margins use baseline partitions and rebuilt baseline Pearson distances. U3 ME and QE are
scored separately. The frontier contains exactly `{0, 0.02, 0.05, M1_star, 0.10}` and marks
both owner-frozen calibrations; innovation values are annotations only and caused no fits.

## Stability and fidelity

### U2 futures - full panel, 295 snapshots

| Config | Raw churn | Median same-date ARI | Fidelity | Residual guard max change |
|---|---:|---:|---|---:|
| baseline | 0.666134 | 1.000000 | PASS | 0.000000 |
| M0 quarterly hold | 0.379678 | 0.980967 | PASS | 0.000648 |
| M1 0.02 | 0.313608 | 0.947981 | PASS | 0.004534 |
| M1 0.05 | 0.179205 | 0.883712 | PASS | 0.002591 |
| M1 0.10 | 0.096390 | 0.783991 | PASS | 0.006951 |
| M2 0.5 | 0.464755 | 0.976081 | PASS | 0.003092 |
| M2 0.7 | 0.356146 | 0.942826 | PASS | 0.005421 |
| M1 star | 0.146169 | 0.866148 | PASS | 0.005829 |

All eight configurations are in-band. M1-star cuts raw churn by 78.1% from baseline while
remaining in-band.

### U3 MAC - full panel, 284 snapshots

| Config | Raw churn | Median same-date ARI | Fidelity | Residual guard max change |
|---|---:|---:|---|---:|
| baseline | 1.525649 | 1.000000 | PASS | 0.000000 |
| M0 quarterly hold | 0.779294 | 0.858976 | PASS | 0.003106 |
| M1 0.02 | 0.760281 | 0.836632 | PASS | 0.007764 |
| M1 0.05 | 0.456865 | 0.750677 | PASS | 0.005737 |
| M1 0.10 | 0.292704 | 0.656960 | PASS | 0.003106 |
| M2 0.5 | 1.259993 | 0.850393 | PASS | 0.001553 |
| M2 0.7 | 1.078693 | 0.819236 | PASS | 0.003106 |
| M1 star | 0.322162 | 0.726470 | REJECTED | 0.013975 |

M1-star cuts raw churn by 78.9%, but is correctly REJECTED: the asset-class taxonomy ARI
change is +0.039024, outside the absolute 0.03 tolerance. Its cluster-count change is -6.67%,
inside the 15% tolerance.

### U1 MSCI US - headline and full panel kept separate

| Window | Config | Raw churn | Median same-date ARI | Fidelity |
|---|---|---:|---:|---|
| full, 238 | baseline | 3.545476 | 1.000000 | PASS |
| full, 238 | M0 quarterly hold | 1.777280 | 0.398534 | REJECTED |
| full, 238 | M1 0.02 | 2.545068 | 0.428511 | REJECTED |
| full, 238 | M1 0.05 | 1.216692 | 0.405638 | REJECTED |
| full, 238 | M1 0.10 | 0.618264 | 0.344788 | REJECTED |
| full, 238 | M2 0.5 | 4.403778 | 0.411021 | REJECTED |
| full, 238 | M2 0.7 | 4.204580 | 0.397695 | REJECTED |
| full, 238 | M1 star | 0.696193 | 0.363017 | REJECTED |
| headline, 203 | baseline | 3.212423 | 1.000000 | PASS |
| headline, 203 | M0 quarterly hold | 1.726611 | 0.399345 | REJECTED |
| headline, 203 | M1 0.02 | 2.239534 | 0.428727 | REJECTED |
| headline, 203 | M1 0.05 | 0.980818 | 0.407812 | REJECTED |
| headline, 203 | M1 0.10 | 0.478271 | 0.348714 | REJECTED |
| headline, 203 | M2 0.5 | 4.280261 | 0.412865 | REJECTED |
| headline, 203 | M2 0.7 | 4.085077 | 0.402866 | REJECTED |
| headline, 203 | M1 star | 0.546106 | 0.366789 | REJECTED |

No U1 smoothed configuration is in-band. Headline M1-star cuts raw churn by 83.0%, but its
sector / industry-group / industry ARI changes are -0.027409 / -0.042296 / -0.047013; the
latter two exceed 0.03. Full-panel values are reported separately and were never pooled.

## Predicted versus realised churn

Correlation across all eight configurations:

| Universe/window | Including singletons | Excluding singletons |
|---|---:|---:|
| U2 full | 0.864768 | 0.867473 |
| U3 full | 0.911033 | 0.909712 |
| U1 full | 0.857861 | 0.858027 |
| U1 headline | 0.862727 | 0.862925 |

Both singleton conventions are now independently scored. A pre-report audit caught that
pandas reindexing had made the singleton mask object-typed, causing the first exclusion output
to equal inclusion. Explicit boolean conversion fixed the defect; all affected tables were
regenerated. For illustration, U2 M1-star predicted / realised churn is 0.309914 / 0.146169
including and 0.315166 / 0.141744 excluding singleton asset-dates.

Kurtosis/noise-floor check:

| Universe/frequency/window | Pooled excess kurtosis | Median asset excess kurtosis | Gaussian predicted | Realised | Multiplier |
|---|---:|---:|---:|---:|---:|
| U2 W-WED full | 14.817696 | 4.848083 | 0.824198 | 0.666134 | 0.808221 |
| U3 ME full | 6.250994 | 2.341433 | 1.250197 | 1.651793 | 1.321226 |
| U3 QE full | 5.544236 | 3.891030 | 0.159423 | 0.343325 | 2.153557 |
| U1 W-WED full | 1471.263340 | 6.335902 | 3.169685 | 3.545476 | 1.118558 |
| U1 W-WED headline | 1471.263340 | 6.335902 | 3.109880 | 3.212423 | 1.032973 |

The pooled U1 kurtosis is dominated by extreme cross-sectional observations; the approved
calibration input remains the median asset statistic/kappa from E1.

## Frequency scaling

The requested comparison is recorded for every config. Baseline and M1-star endpoints are:

| Universe/window | Config | Realised QE/ME churn ratio | Predicted noise ratio |
|---|---|---:|---:|
| U2 full, cached ME vs subsampled QE | baseline | 0.574356 | 1.685466 |
| U2 full | M1 star | 0.972572 | 1.685466 |
| U3 full, native ME vs QE | baseline | 0.207850 | 1.687055 |
| U3 full | M1 star | 0.323961 | 1.687055 |
| U1 full, cached ME vs subsampled QE | baseline | 0.405165 | 1.685466 |
| U1 full | M1 star | 0.784948 | 1.685466 |
| U1 headline | baseline | 0.411850 | 1.685466 |
| U1 headline | M1 star | 0.820114 | 1.685466 |

Observed annualised churn ratios do not support equality with the simple noise-step scaling;
all ratios lie below the predicted value. U1/U2 used cache subsampling only; U3 used its native
ME and QE sleeves; no refits were made.

## Risk-model invariance and flow identity

Every smoothed row passes the frozen residual-diagonality guard of 0.05. Maxima across
smoothed configs are:

| Universe/window | Rel. Frobenius | Max rel. entry | Abs EW vol change | Residual change | Tolerance/result |
|---|---:|---:|---:|---:|---|
| U2 full | 0.057222 | 0.027889 | 0.000466 | 0.006951 | <=0.05 PASS |
| U3 full | 0.048398 | 0.064093 | 0.000268 | 0.013975 | <=0.05 PASS |
| U1 full | 0.047978 | 0.043548 | 0.000252 | 0.005917 | <=0.05 PASS |
| U1 headline | 0.033037 | 0.036820 | 0.000197 | 0.003509 | <=0.05 PASS |

The U1 membership-flow assertion passes on every emitted row:
`total_transitions = index_entry + index_exit + warmup_entry + clusterer_reassignment +
unclassified`. Baseline full-panel totals are 803 / 914 / 82 / 49,022 / 262, summing to
51,083. Nothing is silently dropped.

## Acceptance and verification

Independent validation output:

```text
futures_metric_grid: PASS (8 rows; windows={'full_panel': 295})
futures_fidelity: PASS (8 PASS, 0 REJECTED; all marked)
futures_residual_guard: PASS (max smoothed relative change=0.006950759 <= 0.05)
futures_theory_panels: PASS (16 predictions, 5 frontier rows, 8 scaling rows)
mac_metric_grid: PASS (8 rows; windows={'full_panel': 284})
mac_fidelity: PASS (7 PASS, 1 REJECTED; all marked)
mac_residual_guard: PASS (max smoothed relative change=0.013975155 <= 0.05)
mac_theory_panels: PASS (32 predictions, 5 frontier rows, 8 scaling rows)
msci_us_metric_grid: PASS (16 rows; windows={'full_panel': 238,
  'headline_20090831_20260630': 203})
msci_us_fidelity: PASS (2 PASS, 14 REJECTED; all marked)
msci_us_residual_guard: PASS (max smoothed relative change=0.005917160 <= 0.05)
msci_us_theory_panels: PASS (32 predictions, 10 frontier rows, 16 scaling rows)
```

The validator independently asserts finite/no-NaN numeric metric tables, complete config and
window grids, exact level/innovation markers, both singleton conventions, complete scaling and
kurtosis grids, and the U1 flow identity.

Focused regression and lint output:

```text
....................                                                     [100%]
All checks passed!
```

The numerical second pass also proved threaded residual diagnostics exactly equal the
sequential calculation. Current-code full CSV hash replays passed for U2 and U3; the corrected
singleton-sensitive replay passed for all 9 affected files across U1/U2/U3. The detailed U1
tables were independently revalidated after the correction. No files were staged or pushed.

## Deviations and open item

1. The theory outcomes contain genuine rejections: U3 M1-star and every U1 smoothed row are
   outside the frozen fidelity band. They remain in all tables and are not reclassified.
2. Frequency scaling is empirically rejected as an equality at the reported endpoints.
3. The three workbook wrappers remain blocked solely by the missing required artifact-tool
   runtime. Their complete deterministic source tables are available in the evidence root.

## GATE REQUEST

The owner must rule on:

1. Whether the observed theory verdicts are accepted: strong churn reduction with in-band
   fidelity for U2; U3 M1-star rejected; no in-band U1 smoother; and frequency-scaling equality
   not supported by observed annualised churn ratios.
2. Whether the numerical E3 evidence may proceed to interpretation despite the separately
   escalated workbook-runtime blocker, or whether GATE E3 must remain open until the three XLSX
   wrappers can be assembled and visually verified.
