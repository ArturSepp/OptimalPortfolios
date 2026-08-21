# F4 report — synthetic flip approximation and Ward verification

**Date:** 2026-08-21  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v2  
**Status:** COMPLETE

## Execution and outputs

Runner: `papers/cluster_lineage_2026/replication/run_f4_simulation.py`.  
Focused test: `replication/f4_simulation_test.py`.

Output directory:
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/f4/`.

The principal artifacts are `simulation_results.csv`, `flip_approximation.pdf`, and
`ward_verification.csv`. Supporting artifacts are `acceptance.csv`,
`monotonicity_violations.csv`, `run_parameters.csv`, `runtime.csv`,
`runtime_first_compute.csv`, `determinism.csv`, and 144 resumable cell checkpoints below
`cells/`.

The fixed grid contains 144 DGP/estimator cells and 1,152 method/delta rows: dimensions
50/100, groups 5/10, separations 0.10/0.20/0.30, three frozen Gaussian-GARCH cells, spans
36/156, estimation steps 13/3 and 13, flat and Ward methods, and four shared-path delta
arms. Each cell has 500 paths, burn-in 5N, 24 estimation dates, and seed
`20260817 + cell_index`. The 13/3 schedule is the exact repeating 4/4/5 pattern.

Equal population blocks tie every asset's population margin within a cell. The output
therefore records one honest within-cell margin bucket and the three pooled margin levels,
rather than fabricating ten deciles.

## Proportionality-constant convention

The boundary proposition states that the transition-noise expression is proportional to
the margin-statistic noise. In the simulation this is implemented as one Gaussian churn
multiplier fitted on the zero-delta arm of each DGP/estimator/method cell and then held
fixed across the innovation, level, and double-level arms. Both the unscaled probability
and the held cell constant are retained in `simulation_results.csv`.

Two pre-acceptance harness defects were found and corrected without changing any random
path, parameter, or checkpoint:

1. Monotonicity had been evaluated in semantic label order even though innovation and
   level calibrations can reverse numerically. It now sorts by the actual delta value.
2. The code had calculated a different realised/predicted ratio on every row and had not
   applied any constant. It now fits once on the zero arm and holds that constant across
   the other arms.

Both fixes have fail-before-pass regression tests. The first-compute run used the prior
in-memory checks and therefore rejected after all checkpoints had safely landed. The
corrected acceptance pass was cache-only (144/144 hits); no synthetic path was recomputed.

## Results

For the proved Gaussian i.i.d. flat-cut cells with separation at least 0.20, the complete-
grid predicted-versus-realised correlation is **0.979820**, above the required 0.90, and
realised churn is non-increasing in numerical delta in every one of the 144 flat cells.

Ward remains descriptive as specified. Across its 576 rows, the measured correlation is
**0.974863** and mean absolute prediction error is **0.032827** after the same method-cell
zero-arm calibration. The corresponding flat-method mean absolute error over all 576 rows
is **0.008980**. No Ward acceptance threshold was imposed.

The first computation used 67,610.8 aggregate worker-seconds across the 144 checkpoints;
the slowest cell took 2,127.9 seconds. `runtime_first_compute.csv` preserves those measured
times, while `runtime.csv` records the final 144/144 cache-hit acceptance replay.

## Acceptance checks

| Check | Measured | Tolerance | Result |
|---|---:|---:|---|
| DGP/estimator cells | 144 | 144 | PASS |
| Simulation rows | 1,152 | 1,152 | PASS |
| Replications per cell | 500 | 500 | PASS |
| Estimation dates per path | 24 | 24 | PASS |
| Gaussian flat predicted-realised correlation, separation >= 0.20 | 0.979820 | >= 0.90 | PASS |
| Flat numerical-delta monotonicity violations | 0 | 0 | PASS |
| NaNs in simulation results | 0 | 0 | PASS |
| Deterministic CSV/PDF artifacts | 6/6 byte-identical | 6/6 | PASS |
| Focused pytest | 7 passed | all pass | PASS |
| Isolated Ruff E/F/W | 0 findings | 0 | PASS |

The simulation used synthetic data only. No market-data estimator was fit, no empirical
cache was modified, and no git staging or push occurred.
