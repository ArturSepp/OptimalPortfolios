# Stability-pooled z-score S1 implementation report

**Date:** 2026-08-20  
**Executor:** Sol  
**Proposer:** Ben (Monday TAA meeting)  
**Status:** COMPLETE — additive implementation and isolated MAC wiring landed locally

## Outcome

The stability-pooled standardisation is implemented as an opt-in FactorLasso API. Existing
production defaults and ROSAA source are unchanged. The MAC diagnostic is isolated under the
cluster-lineage replication tree and replaces only the two scorer references used by the
production momentum-cluster and low-beta-cluster legs for the duration of each experimental run.

The implementation uses `w` consistently. V3 is explicitly labelled as breaking strict
asset-class neutrality wherever the grid is emitted.

## Implementation

### FactorLasso

- `src/factorlasso/cluster_standardization.py` adds:
  - `StabilityPoolingType.NONE` (V0 exact bypass);
  - `CLUSTER_VARIANCE` (V1);
  - `ASSET_VARIANCE` (V2);
  - `CLUSTER_MEAN_VARIANCE` (V3 comparison arm);
  - `score_with_stability_pooled_clusters`.
- `src/factorlasso/cluster_smoothing.py` promotes the existing private co-association calculation
  through `compute_co_association_panel`. The existing private caller is untouched.
- `src/factorlasso/__init__.py` exports the new public enum and functions.

The small-cluster global fallback takes precedence over pooling. Missing entrant weights are one,
and the first 11 partition dates are forced to one by `min_history=12`. When all applicable
weights equal one, the scorer enters the existing arithmetic branch exactly instead of relying on
floating-point equivalence after a pooled calculation.

### OptimalPortfolios

`src/optimalportfolios/alphas/signals/utils.py::score_within_clusters` has two additive optional
arguments: `stability_pooling_type=NONE` and `stability_weights=None`. Its existing default body is
unchanged. Only a non-NONE mode delegates to FactorLasso.

### Isolated MAC harness

`papers/cluster_lineage_2026/replication/run_stability_pooling_mac.py`:

- asserts the OneDrive FactorLasso checkout at runtime;
- derives the 36/72 partition windows from the runtime Lasso model;
- fits production covariance and SAA once, then recomputes only alphas per cell;
- restores patched scorer references after each use;
- sends actual and prior-partition counterfactual scores through the existing ROSAA optimizer and
  qis backtester;
- keeps production costs, signal spans, implementation mechanics, constraints, and BAB negation;
- writes all generated outputs below
  `$CLUSTER_LINEAGE_OUTPUT_DIR/stability_pooling/mac/`.

No ROSAA source file or production configuration was edited.

## Acceptance

| check | measured | tolerance | status |
|---|---:|---:|---|
| FactorLasso public accessor | 1 documented additive accessor | 1 | PASS |
| pooling variants | V0/V1/V2/V3 | exactly four frozen modes | PASS |
| default production path change | 0 | 0 | PASS |
| ROSAA source edits | 0 | 0 | PASS |
| MAC frozen cells declared | 7 | 7 | PASS |
| runtime FactorLasso source | local `FactorLasso/src/factorlasso/__init__.py` | local checkout | PASS |
| dedicated runner lint | 0 findings | 0 | PASS |
| FactorLasso changed-file lint | 0 findings | 0 | PASS |
| release/tag/version actions | 0 | 0 | PASS |

No files were staged, committed, pushed, tagged, or released.
