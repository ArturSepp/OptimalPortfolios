# Recovered source copies — STALE VINTAGES, do not run as current code

These files are byte-exact copies recovered from the owner-side session archive after the
2026-08-14 workspace deletion of the untracked `replication/` source tree. They are NOT the
executed state. The canonical executed state survives as CPython 3.12 bytecode in
`replication/recovery_bytecode/`. Use these copies as diffable starting points for source
reconciliation only.

| File | Vintage | Known gaps vs executed state |
|---|---|---|
| `configs.py` | post-E0, pre-E0b | lacks `M1_delta_0.02`; `M1_star`/`M2_star` values unset |
| `local_path.py` | post-E0, pre-E0b | likely current or near-current |
| `metrics.py` | post-E0, pre-E0b | lacks `kappa` in `assignment_margins`, the `tracks_per_asset` redefinition, the `unclassified` flow column, parameterised `signal_rank_metrics` |
| `metrics_test.py` | post-E0, pre-E0b | lacks the E0b regression targets (8.906 etc.) |
| `validate_e0.py` | post-E0, pre-E0b | pre-E0b serialisation (884-byte, not 923-byte) |
| `validate_e0_independent.py` | post-E0, pre-E0b | likely current or near-current |
| `methods.py` | pre-E0 | imports `rosaa.local_path` — violates the frozen no-rosaa constraint; do not run |
| `run_sweep.py` | pre-E0 | imports rosaa; do not run |
| `sp500_baseline.py` | pre-E0 (19,908 bytes) | imports rosaa and optimalportfolios shim; superseded by the 10,264-byte E0 refactor |
| `lineage_matching_validation.py` | pre-E0 | imports rosaa and the deprecation shim; do not run |
| `reproduce_sp500.py` | pre-E0 | probably unchanged through E0; verify against bytecode |

Reconciliation rule: the recovery bytecode wins on any disagreement. Delete this folder once
the live source tree is rebuilt and committed.
