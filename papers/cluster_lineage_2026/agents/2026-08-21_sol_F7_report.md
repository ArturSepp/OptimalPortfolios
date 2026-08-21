# F7 report — readable-source reconstruction

**Date:** 2026-08-21  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v2  
**Status:** COMPLETE

## What changed

The exact executed logic in the surviving CPython 3.12 modules was reconstructed as
human-readable source in:

- `replication/configs.py`;
- `replication/run_backtests.py`.

The owner-frozen futures liquidity ruling is now part of
`run_backtests._investable_eligibility`, rather than a wrapper over executed bytecode. Its
one canonical set contains eleven Bloomberg tickers. The historical alias is preserved as
`MMR1 Curncy -> BMR1 Curncy`, and each canonical exclusion retains the
`low_liquidity_owner_ruling` provenance.

`replication/pyc_compat.py` is now a comment-marked archival file and has zero live imports.
The narrowly scoped loader for other historical modules whose source remains lost moved to
`replication/recovery_loader.py`; ordinary readable source always takes precedence. This
preserves unrelated historical E-stage entry points without putting either reconstructed F7
module on a bytecode execution path.

## Differential verification

Runner: `replication/f7_reconstruction_test.py`.

Frozen inputs were read from:

- `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/msci_us/`;
- `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/futures/`;
- `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/mac/`;
- the approved point-in-time universe panels under `papers/cluster_lineage_2026/data/` and
  `papers/cluster_lineage_2026/msci_us/` through `load_universe`.

The differential reference was the surviving executed `configs.cpython-312.pyc` and
`run_backtests.cpython-312.pyc`. For futures, the owner-frozen exclusion wrapper was applied
to that bytecode reference because it was the effective behavior of the former source shim.
No estimator was fitted.

| Acceptance check | Measured | Tolerance | Result |
|---|---:|---:|---|
| Reconstructed requested modules | 2/2 | 2/2 | PASS |
| Universe enum entries recovered | 3/3 | 3/3 | PASS |
| Smoother enum entries recovered | 9/9 | 9/9 | PASS |
| Universe specifications recovered | 3/3 exact | 3/3 | PASS |
| Smoother specifications recovered | 9/9 exact | 9/9 | PASS |
| M1-star frequency calibrations | 4/4 exact | 4/4 | PASS |
| Production momentum registry fields | 8/8 exact | 8/8 | PASS |
| Canonical futures exclusions | 11/11 exact | 11/11 | PASS |
| Alias mappings | 1/1 exact | 1/1 | PASS |
| Frozen universes in differential test | 3/3 | 3/3 | PASS |
| Primary target-weight panels | 26/26 exact | 26/26 | PASS |
| Prior-partition counterfactual panels | 17/17 exact | 17/17 | PASS |
| E5 artifact-frame types | 11/11 per universe | 11/11 | PASS |
| Artifact frames across universes | 33/33 exact | 33/33 | PASS |
| Deterministic CSV serializations | 33/33 byte-identical | 33/33 | PASS |
| Non-byte-identical differential artifacts | 0 | 0 | PASS |
| Maximum numerical discrepancy | 0.0 | `<= 1e-12` | PASS |
| Live imports of `pyc_compat` | 0 | 0 | PASS |

The eleven differentially regenerated artifact-frame types were `performance`,
`alpha_rank_analysis`, `turnover_decomposition`, `turnover_decomposition_per_date`,
`crisis_windows`, `robustness`, `score_identity`, `weights`, `navs`, `monthly_returns`, and
`target_turnover_per_date`. Because both the frames and their deterministic `%.15g`, LF CSV
serializations were identical, there are no timestamp-container exceptions or other
non-byte-identical files to list.

## Regression and lint evidence

The new futures-constant regression was first run with the canonical `BMR1 Curncy` entry
deliberately misspelled. It failed with one extra and one missing set member. The defect was
restored and the same assertion passed, satisfying the fail-before-pass verification rule.

| Check | Measured | Required | Result |
|---|---:|---:|---|
| F7 differential tests | 8/8 passed | 8/8 | PASS |
| Existing focused tests | 14/14 passed | 14/14 | PASS |
| Isolated Ruff E/F/W on touched F7/import-path files | 0 findings | 0 | PASS |
| Market refits | 0 | 0 | PASS |
| Missing frozen inputs encountered | 0 | 0 | PASS |
| Git staging, commit, tag, or push | 0 | 0 | PASS |

Focused tests were `e5b_test.py`, `futures_best_relative_pnl_scatter_test.py`,
`futures_commodity_pnl_attribution_test.py`, and `futures_sleeve_grid_test.py`. F7 therefore
closes the reconstruction defect with no behavioral change and no gate.
