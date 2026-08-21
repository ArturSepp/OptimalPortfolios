# Sol to-do: stability pooling S5b — EWMA weights, cluster statistics utility, V1/V2 robustness

**Date:** 2026-08-20
**Author:** Claude, from owner rulings in today's chat
**Executor:** Sol
**Proposer of the method:** Ben (Monday TAA meeting)
**Amends:** `ROADMAP_stability_pooled_zscore.md` (2026-08-20). Read the S0-S6 reports and the
2026-08-20 Claude review before starting. Routing unchanged: FactorLasso changes under OSS
instructions (additive only, no release), harness and backtests under research instructions v2.6.

## Owner rulings (2026-08-20)

1. **V3 is dropped.** The S6 REJECT stands for the mean-plus-variance comparison arm and that
   question is closed. Remove `CLUSTER_MEAN_VARIANCE` from the public enum — nothing was
   released, so removal is clean, not a deprecation. The open verdict is V1/V2 only.
2. **w becomes EWMA-weighted co-association.** The flat trailing-window mean is replaced by an
   exponentially weighted co-association frequency. The span is keyed by estimation frequency
   from a pinned map: `{ME: 36, QE: 18}`. This map is an explicit configuration constant, not a
   derivation from the lasso model. Disclosure: S0 measured the MAC lasso per-frequency spans as
   `ME: 36, QE: 12`, so the QE entry deliberately differs — record it as an owner ruling. For
   MAC the co-association panel is monthly-indexed, so only span 36 binds in S5b. The roadmap's
   {1x, 2x} window sweep axis is retired (36 beat 72 in every S3 pair).
3. **Stability analytics move to a FactorLasso utility, computed once per run.** Momentum-cluster
   and low-beta-cluster both consume the same precomputed object. Signals never construct it.

Symbol discipline unchanged: `w` in code and reports. Lambda stays reserved for the EWMA decay in
the theory draft. In code, parameterise by `span` only.

## D1. FactorLasso: `ClusterStabilityStatistics` (additive, OSS routing)

New module `src/factorlasso/cluster_statistics.py`:

- Constructor `compute_cluster_stability_statistics(partitions, span_by_freq, min_history=12)`
  from the rolling operating partitions (raw partitions when a smoother is active, per the S0
  pinned definition).
- EWMA definition: pandas convention `alpha = 2 / (span + 1)` over observed partition dates
  through `t`, `adjust=True` so weights renormalise over available history. Point in time is
  unchanged: partitions through `t` only. Short history unchanged: `w = 1` until 12 observed
  partition dates.
- Exposes: per-asset stability panel `w_i` (dates by assets), per-cluster panel `w_g`, the
  public co-association accessor (reuse `compute_co_association_panel`, do not duplicate), and a
  per-date coverage frame.
- Diagnostics as methods, migrated from the S4 harness code: boundary statistics
  (reassignment rate by stability quartile), size-versus-w correlation, within-cluster
  asset-w dispersion.
- `score_with_stability_pooled_clusters` keeps its contract and consumes precomputed weights.
  Default remains an exact bypass. The `min_cluster_size` global fallback keeps precedence.

Tests, fail-before-pass as in S2: hand-computed EWMA reference on a small panel, span-map
resolution by frequency, `w = 1` and `w = 0` endpoint identities, causality under the EWMA
weighting (future partition perturbations leave the panel through `t` byte-identical),
short-history fallback, V0 byte-identity retained, and removal of V3 verified across the
public API surface. Full FactorLasso and OptimalPortfolios suites green before S5b runs.

## D2. Compute-once wiring

- The statistics object is constructed once per run at the estimation layer, next to the
  covariance fit that produces `RollingClusterData`, and passed into both signal legs through
  the existing optional arguments of `score_within_clusters`.
- The harness asserts a construction count of exactly one per cell in the run log.
- Production wiring on adoption (NOT implemented now): the pipeline input carries the object and
  `AlphaAggregator` passes it to both legs. No ROSAA source edit in this workstream. If one
  appears unavoidable, stop and report.

## S5b runs

Harness, baseline, costs, constraints, BAB negation, and output root unchanged from S3. Reuse
`shared_pipeline_inputs.pkl` and recompute alphas only.

1. **Cells:** V1/EWMA-36 and V2/EWMA-36.
2. **Transition check:** report each cell against its flat-36 S3 counterpart (Sharpe, turnover,
   TE, TRE deltas of EWMA minus flat). The expectation is small differences. A Sharpe move above
   0.02 in absolute value between weighting schemes is reported prominently, not smoothed over.
3. **Full-sample metrics** as in S3: TAA Sharpe, ex-post TE, ex-ante TRE at the production grid
   point, annual turnover, max drawdown, deltas to V0.
4. **Robustness, frozen S5 protocol on both cells:** the same two half-windows
   (2004-12-31 to 2015-09-30, 2015-10-31 to 2026-07-31) and the paired circular moving-block
   bootstrap (block 6, 2,000 draws, seed 20260813) on the Sharpe and turnover deltas.
5. **Mechanism:** rerun the S4 reassignment-versus-signal decomposition on the better cell under
   EWMA weights.

## Decision rule (predeclared)

Matched risk is pinned as `abs(TE delta) <= 0.001` and `abs(TRE delta) <= 0.001` against V0.

- **ADOPT-CANDIDATE** for a cell requires all of: Sharpe delta at or above zero in both
  half-windows, turnover 95% CI entirely below zero, matched risk as pinned, and the S4
  reassignment share of direct reductions above 50%.
- A full-sample Sharpe CI covering zero does NOT by itself reject when both half-window Sharpe
  deltas are nonnegative and turnover holds. The adoption claim is a turnover reduction at held
  Sharpe and matched risk, not an alpha claim.
- Any negative half-window Sharpe delta is a **REJECT** for that cell. If both cells reject, the
  workstream closes with the family REJECT confirmed.
- If both cells pass, report both and stop for owner selection. Do not pick.

## Standing constraints (unchanged)

No FactorLasso release, tag, or version bump. No production configuration change. No ROSAA
source edits. Caches under `$CLUSTER_LINEAGE_OUTPUT_DIR/stability_pooling/`. Nothing written
into the frozen U1/U2/U3 paper trees. Ben recorded as proposer in every report and eventual
exhibit. Report named `YYYY-MM-DD_sol_stability_pooling_S5b_report.md` in `agents/`, conclusions
first, three decisive numbers in the first paragraph.

## Acceptance checklist

| check | tolerance |
|---|---|
| V3 removed from public API | 0 remaining references |
| EWMA hand-computed reference | exact to explicit arithmetic |
| span map resolution | `{ME: 36, QE: 18}`, pinned, not derived |
| V0 byte-identity retained | exact |
| causality under EWMA | byte-identical panel through `t` |
| statistics constructions per cell | exactly 1, asserted in log |
| ROSAA source edits | 0 |
| full suites | 0 failures/errors |
| split windows per cell | 2 |
| bootstrap parameters | block 6, 2,000 draws, seed 20260813 |
| release/tag/version actions | 0 |
