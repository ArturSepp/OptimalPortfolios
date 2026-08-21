# Stage E4 execution report - U2/U3 lineage and interpretability

**Date:** 2026-08-14  
**Roadmap:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`  
**Owner dispatch:** `2026-08-14_owner_E3_gate.md`  
**Status:** COMPLETE for the dispatched U2/U3 scope

## Execution surface

Runner: `papers/cluster_lineage_2026/replication/run_interpretability.py`  
Independent validator: `papers/cluster_lineage_2026/replication/validate_e4_e5.py`  
Cache root: `C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\<universe>\<config>\`  
Evidence root: `C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\interpretability\`

The owner-frozen pairs were U2 `{baseline, M1_star}` and U3
`{baseline, M1_delta_0.05}`. U1 was explicitly outside this E4 dispatch pending E3b. Each of
the four runs emitted the canonical FactorLasso lineage tables and two canonical figures,
metric set 12, track classifications, label panels, taxonomy-ARI-by-date panels,
vocabularies, and case-study paths. The aggregate evidence root contains 66 files totalling
10,474,809 bytes.

## Metric set 12

| Universe | Config | coarse ARI | fine ARI | modal taxonomy purity | core tracks | label churn | modal-label life share | labels | clusters | primary-factor variance share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U2 | baseline | 0.494241 | 0.072169 | 0.857275 | 8 | 0.536709 | 1.000000 | 15 | 16 | 0.858102 |
| U2 | M1_star | 0.503548 | 0.069224 | 0.867505 | 8 | 0.103178 | 1.000000 | 14 | 16 | 0.880035 |
| U3 | baseline | 0.291965 | 0.362630 | 0.688540 | 1 | 1.344082 | 1.000000 | 37 | 15 | 0.537375 |
| U3 | M1_delta_0.05 | 0.304323 | 0.348962 | 0.732843 | 2 | 0.373044 | 1.000000 | 36 | 14 | 0.616703 |

For U2, “coarse/fine” denotes `asset_class/ac_geography`; for U3 it denotes
`Asset Class/Sub Asset Class`. Thus the U2 peak taxonomy alignment is at asset class. U3's
peak remains at sub-asset class, although smoothing shifts alignment toward the coarser asset
class. Both best configurations materially reduce label-string churn and raise modal taxonomy
purity. No run produced an `Idio` modal label share above zero.

## Vocabulary and case studies

The cross-universe vocabulary table contains 15/14 U2 labels for baseline/M1-star and 37/36
U3 labels for baseline/M1-0.05, including mean MATF exposure profiles, systematic variance,
and the count required to cover 90% of non-Idio systematic variance. Five labels reach that
90% threshold in each U2 run; U3 requires 16 labels at baseline and 15 under M1-0.05.

Exactly three case-study tracks per universe were retained, all with coverage at least 0.70.
U2's three selected commodity tracks each have coverage 1.00 and span all 295 snapshots. U3
contains one baseline rates track at coverage 1.00 plus two M1-0.05 tracks at coverage
0.883803 and 0.704225. Membership and loading paths are present for every selected track.

## Futures equity-beta bucket proposal

The asset-level time-median Equity beta quantiles from the U2 baseline are:

| Quantile | beta |
|---:|---:|
| 5% | -0.000000094 |
| 10% | -0.000000083 |
| 25% | 0.000000019 |
| 50% | 0.000000493 |
| 75% | 0.668656 |
| 90% | 0.841910 |
| 95% | 0.933007 |

The proposal is a leverage-free low/mid/high bucketing at the 25th and 75th percentiles:
`0.000000019` and `0.668656`. It is recorded as
`PROPOSAL_ONLY_OWNER_RULING_REQUIRED`; it was not adopted anywhere in the analysis.

## Acceptance and verification

- Purity/persistence tables: PASS, all four owner-dispatched runs complete.
- Vocabulary exhibit: PASS, all four universe/config pairs present.
- Case studies: PASS, 3 per universe and minimum coverage 0.704225 versus tolerance 0.70.
- U2 threshold proposal: PASS, seven quantiles reported and status remains proposal-only.
- Canonical FactorLasso output: PASS, tables and figures present for all four runs.

Independent validation output:

```text
E4: PASS (4 runs; vocabulary present; 3 coverage>=0.70 cases per universe)
E4 deterministic CSV rerun: PASS (58 files)
```

No files were staged or pushed.
