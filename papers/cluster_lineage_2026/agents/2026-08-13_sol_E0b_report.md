# Stage E0b execution report — owner amendments

**Date:** 2026-08-13  
**Roadmap:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`  
**Owner dispatch:** `2026-08-13_claude_E0_review_and_E1_instructions.md`  
**Status:** COMPLETE; E0 metric definitions and registry are now frozen

## Outcome

All owner-approved E0b amendments landed with the frozen E0 numerical targets unchanged.

- `assignment_margins` accepts `kappa=0.0` and multiplies its noise scale by
  `sqrt(1+kappa)`. An explicit zero is frame-identical to the omitted default.
- `tracks_per_asset` is now the cross-asset mean of distinct non-null derived ids. The old
  quotient remains available only as `track_to_asset_ratio`.
- `signal_rank_metrics` accepts `lookback_periods`, `skip_periods`, and `freq`. Its defaults
  retain the exact pre-E0b inclusive weekly timestamp window.
- Membership flows now include `unclassified`, enabling exhaustive transition accounting.
- `M1_delta_0.02` is registered; `M1_star` and `M2_star` remain unset and raising.
- `.gitignore` now excludes only `cluster_lineage_2026/data/` and `msci_us/`; `replication/`,
  `agents/`, and a future `paper/` are trackable. Nothing was staged or pushed.

## New frozen target

On the 60-snapshot cached S&P 500 baseline panel:

| Metric | Measured frozen value | Tolerance | Result |
|---|---:|---:|---|
| Mean distinct tracks per asset | 8.9061876247505 | 1e-12 | PASS |
| Track-to-asset ratio | 0.4311377245508982 | 1e-12 | PASS |

The first value is computed as `membership_panel.nunique(axis=0, dropna=True)` averaged
over the 501 asset columns with at least one membership. Its distribution has median 8,
interquartile range 4–13, and maximum 28.

## Verification

Runner scripts:

- `papers/cluster_lineage_2026/replication/metrics_test.py`
- `papers/cluster_lineage_2026/replication/validate_e0.py`

Cache directory:

`C:\Users\artur\OneDrive\analytics\outputs\cluster_smoothing\sp500_baseline\baseline\`

Command:

```powershell
$env:CLUSTER_LINEAGE_OUTPUT_DIR='C:\Users\artur\OneDrive\analytics\outputs'
python -m pytest papers\cluster_lineage_2026\replication\metrics_test.py -q
```

Verbatim pytest output:

```text
....                                                                     [100%]
```

Command:

```powershell
python -m papers.cluster_lineage_2026.replication.validate_e0
```

Result: all six original frozen checks PASS; deterministic rerun PASS with two identical
923-byte serialisations. The original headline values remain lineage churn 3.2114693774,
216 tracks, matcher churn 0.4857547101, and ARIs 0.2029670733 / 0.2970123965 /
0.3319349338.

Focused Ruff `E,F,W` audit after the final line-wrap cleanup: `All checks passed!`

## Trackability after ignore narrowing

The following categories now become visible to Git: the existing paper replication Python
files, agent roadmap/review/report markdown files, and any future `paper/` manuscript files.
The raw `data/` and `msci_us/` payload trees remain ignored. No Git staging or push occurred.

## Deviations and open items

- None from the owner dispatch.
- The singleton convention is frozen for E3 reporting: predicted-versus-realised churn will
  be reported both including and excluding singleton assets.
- `M1_star` awaits E1 `kappa_hat` plus E2 baseline `rho_bar`; `M2_star` remains unset.
