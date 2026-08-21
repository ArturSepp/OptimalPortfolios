# F0 report — provenance freeze and cache inventory

**Date:** 2026-08-20  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v2  
**Status:** COMPLETE after the owner's 2026-08-20 instruction to rerun missing artifacts.

## What ran

- Inventory: `replication/run_f0_inventory.py`.
- Missing signal-series recovery: `run_u1_bics_sector_comparison_classic.py`,
  `run_u2_55_35_10_signal_grid.py`, and `run_u3_rosaa_min10_short_span_sweep.py`.
- Missing risk-NAV recovery: `run_u1_hierarchical_risk.py` and
  `run_u3_hierarchical_risk.rebuild_navs_from_frozen_weights`.
- Missing adopted-cell cache: `run_u1_me36_adopted_cache.py`.

The owner instruction, "if data is missing you need to do re-run", superseded the F0
no-rerun stop only for these nine inventoried artifacts. No parameter search, universe
change, or specification change was made. U3 risk NAVs preserve the accepted seven-exclusion
risk vintage by consuming its frozen decision weights; the later eleven-exclusion signal
vintage was not substituted.

Inventory of record:

`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/f0/cache_inventory.csv`

SHA-256: `3B2E76DD51998E6690C04A426987CDA583EAE512FCB01224FB12256C02D4589B`.

## Current-state verification

| Check | Measured | Required | Result |
|---|---:|---:|---|
| F-stage/QFEX reports present before F0 started | 0 | 0 | PASS |
| Live manuscript `paper/paper.tex` present | 1 | 1 | PASS |
| Frozen archive consumed | 0 files | 0 | PASS |
| Stability-pooling inputs consumed | 0 | 0 | PASS |
| Local FactorLasso import | `FactorLasso/src/factorlasso/__init__.py` | local checkout | PASS |
| Local OptimalPortfolios import | `OptimalPortfolios/src/optimalportfolios/__init__.py` | local checkout | PASS |
| Matplotlib backend | `Agg` | `Agg` | PASS |
| Output location | `finalisation/f0/` | `finalisation/f0/` | PASS |

## Recovery evidence

| Artifact family | Measured | Tolerance | Result |
|---|---:|---:|---|
| U1 signal deterministic artifacts | 13/13 byte-identical | 13/13 | PASS |
| U2 signal deterministic artifacts | 8/8 byte-identical | 8/8 | PASS |
| U3 signal deterministic artifacts | 8/8 byte-identical | 8/8 | PASS |
| U1 hierarchical-risk deterministic artifacts | 23/23 byte-identical | 23/23 | PASS |
| U3 recovered NAV performance error vs frozen table | `6.279699e-15` | `<=1e-12` | PASS |
| U1 ME/36 adopted cache snapshot count | 203 | 203 | PASS |
| U1 baseline asset-set match share | 1.000 | 1.000 | PASS |
| U1 smoothed asset-set match share | 1.000 | 1.000 | PASS |
| U1 injected/fitted partition match share | 1.000 | 1.000 | PASS |
| U1 adopted smoother delta | 0.0866 | 0.0866 | PASS |
| U1 adopted-cache deterministic artifacts | 205/205 byte-identical | 205/205 | PASS |

The U1 adopted cache fingerprint is
`5c3d8ddc552a17dbc8b056f494d888fd5b89ac8210fe205080d1839b4f9ee848`.
The cache is at
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/msci_us/ME_span_036_M1_star_delta_0.0866/`.

## Final inventory acceptance

The 65 logical inputs reference 8,157 files and 12,482,001,948 bytes. Totals include
intentional overlap where a separately required file also sits below an inventoried
directory.

| Acceptance check | Measured | Tolerance | Result |
|---|---:|---:|---|
| Required logical inputs resolving exactly once | 65/65 | 65/65 | PASS |
| Missing inputs | 0 | 0 | PASS |
| Ambiguous inputs | 0 | 0 | PASS |
| E2/E3b panel/config cache directories | 24/24 | 24/24 | PASS |
| Equity snapshots per E2/E3b config | 238 | 238 | PASS |
| Futures snapshots per E2/E3b config | 295 | 295 | PASS |
| Fund snapshots per E2/E3b config | 284 | 284 | PASS |
| Signal NAV/weight files | 6/6 | 6/6 | PASS |
| Required risk NAV files | 2/2 | 2/2 | PASS |
| Adopted-cell cache directories | 4/4 | 4/4 | PASS |
| Stability-pooling inputs consumed | 0 | 0 | PASS |
| Deterministic inventory replay | identical SHA-256 | byte-identical | PASS |

## Constraint audit

- Narrow owner-authorized market refits: one adopted U1 ME/36 smoothing cell only.
- Backtest reruns: three frozen signal grids and U1 risk; U3 risk NAVs reconstructed from
  the frozen accepted weights and unchanged price panel.
- New empirical searches, spec changes, and substitutions: 0.
- dePC1 or stability-pooling inputs consumed: 0.
- Changes to FactorLasso, OptimalPortfolios, qis, or rosaa source: 0.
- Git staging, commit, tag, push, or release actions: 0.

F0 therefore passes and releases F1 under the dispatch sequence.
