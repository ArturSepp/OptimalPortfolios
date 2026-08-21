# F3 report — membership, lineage, and interpretability consolidation

**Date:** 2026-08-20  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v2  
**Status:** COMPLETE

## Execution and outputs

Runner: `papers/cluster_lineage_2026/replication/run_f3_membership.py`.
Focused test: `replication/f3_membership_test.py`.

Output directory:
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/f3/`.

The principal artifacts are `churn_fidelity.csv`, `adopted_cell_verdicts.csv`,
`interpretability.csv`, and `case_study_tracks.csv`. `source_manifest.csv`,
`acceptance.csv`, and `determinism.csv` record provenance and execution checks. The
consolidation reads the corrected E3b evidence and accepted E4 lineage reports. It does not
read the superseded pre-E3b equity smoother outputs.

## Churn and fidelity consolidation

`churn_fidelity.csv` contains all 32 corrected panel/window/config rows with raw and
lineage churn, median same-date ARI, taxonomy dARI at every applicable level, median
cluster-count change, and the frozen symmetric-band verdict. Verdict counts are:

| Panel/window | PASS | REJECTED |
|---|---:|---:|
| Equity, full | 6 | 2 |
| Equity, headline | 5 | 3 |
| Futures, full | 8 | 0 |
| Fund, full | 7 | 1 |

The rejected rows reproduce the gated E3b findings. Equity `M1_delta_0.10` and `M1_star`
degrade all three taxonomy levels on both relevant windows; `M1_delta_0.05` breaches the
headline industry-group band by -0.03129 while remaining in-band on the full panel. Fund
`M1_star` breaches in the positive direction: asset-class ARI rises by +0.03902, the
previously frozen over-consolidation toward the taxonomy. The band remains symmetric.

## Adopted application cells

Both adopted cells were re-scored directly from cached partitions against their own
unsmoothed application-cell baseline. The U1 ME/36 cache was regenerated under the owner's
missing-data rerun instruction before F0 closed; this stage reads it without refitting.

| Cell | Dates | Median same-date ARI | Maximum absolute taxonomy dARI | Median cluster-count change | Verdict |
|---|---:|---:|---:|---:|---|
| U1 ME/36, delta 0.0866 | 203 | 0.370724 | 0.014694 | +0.116667 | PASS |
| U3 W-WED/156, delta 0.0691 | 295 | 0.866148 | 0.009307 | 0.000000 | PASS |

Each row carries the baseline and candidate paths plus explicit F0 manifest fingerprints.
The U1 candidate also carries its internal data-and-specification fingerprint
`5c3d8ddc552a17dbc8b056f494d888fd5b89ac8210fe205080d1839b4f9ee848`.

## Interpretability and case tracks

The accepted E4 metrics show that smoothing lowers annualised label-string churn from
1.158 to 0.635 for equities, 0.537 to 0.103 for futures, and 1.344 to 0.373 for funds.
Track modal-taxonomy purity rises from 0.532 to 0.555, 0.857 to 0.868, and 0.689 to 0.733,
respectively. The share of track life under its modal assigned label is 1.0 in all six
reported rows. `interpretability.csv` also preserves every taxonomy-level ARI and names the
peak level rather than pooling non-comparable hierarchies.

`case_study_tracks.csv` contains exactly three cached lineage tracks per panel, with date
range, membership-path summary, modal label, and source paths. Minimum coverage is 0.7042;
the other eight tracks range from 0.8908 to 1.0.

## Min-cost-flow matcher as implemented

The canonical implementation is
`FactorLasso/src/factorlasso/cluster_lineage.py`. `analyze_cluster_lineage` fingerprints
each raw cluster and delegates the default `method='mcf'` path to `_match_panel_mcf`.
That function constructs all qualifying consecutive and bridge edges, weights bridge gaps
by `bridge_decay ** (gap - 1)`, and calls `solve_max_weight_matching`, the deterministic
sparse bipartite solver, to obtain a global maximum-weight vertex-disjoint path cover.
Walking each predecessor-free chain assigns one persistent derived id. A selected edge is
tagged `continue` at a one-date gap and `bridge` at a longer gap; unmatched qualifying
alternatives determine split and merge annotations, with the remaining endpoints marked as
births and deaths. The bridge-edge property is therefore structural: the joint panel solve
can route identity around a temporary split/merge over the configured six-date window,
rather than irrevocably handing identity to the locally best consecutive match.

This lineage analysis is offline full-panel reporting only. It supplies no portfolio score
or weight.

## Acceptance checks

| Check | Measured | Tolerance | Result |
|---|---:|---:|---|
| F0 sources resolved once | 8/8 | 8/8 | PASS |
| Churn-fidelity rows | 32 | 32 | PASS |
| Adopted-cell rows | 2 | 2 | PASS |
| Published-overlap regression error | `1.110223e-16` | `<=1e-12` | PASS |
| Interpretability rows | 6 | 6 | PASS |
| Case-study tracks per panel | 3 | 3 | PASS |
| Minimum case-study coverage | 0.704225 | `>=0.70` | PASS |
| NaNs across deliverables | 0 | 0 | PASS |
| Deterministic artifacts | 6/6 byte-identical | 6/6 | PASS |
| Focused pytest | 3 passed | all pass | PASS |
| Isolated Ruff E/F/W | 0 findings | 0 | PASS |

No estimator was fit and no source cache was modified. No git staging or push occurred.
