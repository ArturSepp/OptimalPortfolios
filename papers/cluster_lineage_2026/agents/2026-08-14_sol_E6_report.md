# E6 execution report — inference layer

Date: 2026-08-14
Universes: U1 MSCI US, U2 futures, U3 MAC
Output: `$CLUSTER_LINEAGE_OUTPUT_DIR/inference/`

## Frozen inference specification

Moving-block bootstrap uses block length 6, 2,000 draws and seed 20260813. Taxonomy nulls
use 500 size-matched label permutations per date with the same seed. The Greene comparator
uses Jaccard threshold 0.3. No confidence interval or interpretation compares a ranking leg
against EW-all.

## Acceptance lines

| Acceptance line | Measured | Tolerance | Status |
|---|---:|---:|---|
| Bootstrap seed | 20260813 | exactly 20260813 | PASS |
| Bootstrap draws | 2,000 | exactly 2,000 | PASS |
| Block length | 6 | exactly 6 | PASS |
| Permutation draws | 500 | exactly 500 | PASS |
| Greene Jaccard threshold | 0.3 | exactly 0.3 | PASS |
| E3 stability intervals | 75/75 required in-band churn/taxonomy rows | 75 | PASS |
| E5 payoff intervals | 36/36 rows | 36; exactly 3 contrasts; no EW contrast | PASS |
| Theory correlations | 8 | 8 | PASS |
| P4 gap intervals | 32 | 32 | PASS |
| U3 taxonomy-drift breach | +0.039024 | reproduces accepted +0.039 direction/magnitude | PASS |
| Taxonomy permutation nulls | 10/10 observed medians exceed null 95th percentile | 10/10 | PASS |
| Lineage comparator rows | 9/9 | 3 methods × 3 universes | PASS |
| Deterministic replay | 7/7 CSV tables byte-identical | 7/7 | PASS |
| Workbook verification | 24,443 bytes; `run_parameters` re-read | workbook opens and frozen parameters match | PASS |

The exact recovered validator reports:

```
E6 parameters: PASS (seed=20260813, draws=2000, block=6, permutations=500)
E6 E3 intervals: PASS (75/75 required in-band churn/taxonomy rows)
E6 E5 intervals: PASS (36/36 rows; 3 contrasts; no EW-all contrast)
E6 theory: PASS (8 correlations, 32 P4 gaps, U3 breach=0.039024)
E6 nulls/lineage: PASS (10/10 null rejections; 9/9 lineage rows)
E6 workbook: PASS (run_parameters re-read; 24443 bytes)
```

## Theory evidence and research-log remark

The realised-minus-predicted P4 gap increases with delta in the aggregate research-log
diagnostic. Estimated gap-on-delta slopes are 2.3109 for futures, 2.2867 for MAC, 0.1604 for
the U1 full panel and 0.5624 for the U1 headline window; corresponding Spearman associations
are 0.8, 1.0, 0.4 and 0.4. This is recorded only as being consistent with a
noise-plus-structural-breaks churn decomposition. It is a research-log item and candidate
paper remark, not a new claim.

U1 predicted-versus-realised churn correlation is 0.8631 on the full panel including
singletons (95% CI [0.8502, 0.8716]) and 0.8717 in the headline window (95% CI
[0.8650, 0.8784]). The full set of eight correlation intervals and 32 P4-gap intervals is
in `theory_bootstrap.csv`.

## Permutation nulls and lineage comparators

All ten taxonomy tests reject the permutation null at the prescribed comparison. For
example, U1 headline GICS-sector median ARI is 0.18879 versus a null 95th percentile of
0.00022; U2 asset-class ARI is 0.49424 versus 0.00022; and U3 Asset Class ARI is 0.29196
versus -0.00002.

| Universe | Method | Tracks | Annual churn | Fragmentation tracks/asset |
|---|---|---:|---:|---:|
| U1 | MCF default | 1,696 | 4.0858 | 23.7481 |
| U1 | Greene Jaccard 0.3 | 3,578 | 3.9897 | 29.7696 |
| U1 | MONIC overlap | 10,436 | 8.2395 | 74.2919 |
| U2 | MCF default | 80 | 0.7440 | 7.4526 |
| U2 | Greene Jaccard 0.3 | 164 | 0.6765 | 10.0000 |
| U2 | MONIC overlap | 659 | 2.0301 | 48.2211 |
| U3 | MCF default | 122 | 1.6984 | 15.5829 |
| U3 | Greene Jaccard 0.3 | 302 | 1.5672 | 19.4545 |
| U3 | MONIC overlap | 1,391 | 5.7087 | 114.9947 |

## Deliverables

- `cluster_lineage_inference_20260814.xlsx`
- `stability_bootstrap.csv`
- `payoff_bootstrap.csv`
- `theory_bootstrap.csv`
- `taxonomy_permutation_nulls.csv`
- `lineage_baselines.csv`
- `p4_structural_break_research_log.csv`
- `run_parameters.csv`

No Git staging or push was performed.
