# F8 report — final exhibits and traceability index

**Date:** 2026-08-21  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v3  
**Status:** COMPLETE

## Gate record and execution

The owner's `continue` instruction immediately after the mandatory G0 stop is recorded as
OWNER GATE G0 approval. The U1 rows and CI companion fields therefore use the gated G0
headline-window values, and the unreproduced 2026-08-17 U1 signal quotations are not used in
the F8 artifacts.

Runner: `papers/cluster_lineage_2026/replication/build_final_exhibits.py`.  
Focused test: `replication/final_exhibits_test.py`.

Output directory:
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/exhibits/`.

The runner reads only frozen F0-F6/G0 artifacts, recorded stability diagnostics, recorded
risk-allocation outputs, and four existing robustness-grid summaries. It runs no backtest,
optimizer, covariance estimator, clustering estimator, or bootstrap. The F4 simulation is
read from its frozen output rather than rerun.

## Exhibit package

`exhibit_index.csv` contains exactly 16 manuscript exhibits:

| Category | Exhibits | Measured count |
|---|---|---:|
| Figures | F1-F6, each EPS plus PDF | 6 |
| Existing body tables | `tab:universes`, `tab:signal`, `tab:risk`, `tab:concentration` | 4 |
| New body tables | TA panel summary, TB calibration bridge, TC churn/fidelity, TD scorecard | 4 |
| Appendix tables | TE and TF selection grids | 2 |

Every index row carries a takeaway title, manuscript section and label, claim family/panel,
the F8 builder, exact source path or paths, output artifact path, caption notes, agent report
of record, and pre-F9 commit provenance. All source and artifact path tokens resolve.

The figure package is:

1. F1 — baseline versus calibrated membership churn through time;
2. F2 — churn-fidelity frontier with level and innovation calibrations and the curvature knee;
3. F3 — cached flat-cut and production-Ward flip-probability verification;
4. F4 — margin distribution and realised flip rate by margin decile;
5. F5 — cumulative net signal NAVs, with U1 sourced from G0;
6. F6 — effective risk clusters through time, flat ERC versus equal-cluster budgets.

The six plots use one 38pc style, common colors and line conventions, and ASCII-compatible
labels. Each index row contains the binding takeaway-title caption and notes covering source,
sample, units, costs where applicable, and provenance.

## Companion CIs and reconciliations

`table_existing_signal.csv` embeds the four predeclared signal comparison intervals in the
candidate leg's `ci_companion` field. Three cluster rows carry those four comparisons because
U1 carries both global and BICS-sector controls. `table_existing_risk.csv` embeds the three
predeclared risk comparison intervals in two candidate rows because U1 Rolling-Ward HRP has
two controls. No standalone manuscript CI table and no CI against EW-all was created.

The G0 values flow into F5, `tab:signal`, and the U1 CI-bearing rows of `tab:risk`. U2/U3
remain on their frozen F6 values. The exact endpoint and concentration checks are in
`precision_reconciliation.csv`:

| Reconciliation | Rows | Maximum absolute error | Tolerance | Result |
|---|---:|---:|---:|---|
| F5 plotted NAV metrics vs `tab:signal` | 21 | `1.387779e-16` | `<=1e-12` | PASS |
| F6 time-series averages vs `tab:concentration` | 6 | `4.796163e-14` | `<=1e-12` | PASS |
| Combined visible-number maximum | 27 | `4.796163e-14` | `<=1e-12` | PASS |

## Robustness-grid consolidation

Only the four summary CSVs used by appendix tables TE/TF were copied. Original files were
left in place and no partition, NAV, ranking, or weight cache was duplicated:

| Consolidated summary | Selected operating point | Source/destination hash |
|---|---|---|
| U2 eligibility | USD 50m cutoff | byte-identical |
| U1 minimum cluster size | 10 | byte-identical |
| U3 short span | 3 | byte-identical |
| U1 covariance frequency/span | ME/span 36 | byte-identical |

Every grid has exactly one `selected_operating_point` row. Every other row is labelled
`selection_record_not_independent_confirmation`.

## Acceptance checks

| Acceptance check | Measured | Tolerance | Result |
|---|---:|---:|---|
| Indexed manuscript exhibits | 16 | 16 | PASS |
| Figure exhibits | 6 | 6 | PASS |
| EPS figure files | 6 | 6 | PASS |
| PDF figure files | 6 | 6 | PASS |
| Existing body tables | 4 | 4 | PASS |
| New body tables | 4 | 4 | PASS |
| Appendix tables | 2 | 2 | PASS |
| Index rows with source provenance | 16 | 16 | PASS |
| Missing indexed source paths | 0 | 0 | PASS |
| Missing indexed exhibit artifacts | 0 | 0 | PASS |
| U1 performance/CI exhibits sourcing G0 | 3/3 | 3/3 | PASS |
| Copied robustness summaries byte-identical | 4/4 | 4/4 | PASS |
| Maximum visible-number reconciliation error | `4.796163e-14` | `<=1e-12` | PASS |
| Signal rows carrying CI companion fields | 3 | 3 | PASS |
| Risk rows carrying CI companion fields | 2 | 2 | PASS |
| Selection grids with exactly one selected row | 4/4 | 4/4 | PASS |
| Deterministic payload replay | 37/37 byte-identical | 37/37 | PASS |
| Backtest/optimizer/estimator runs | 0 | 0 | PASS |
| Files written outside `finalisation/exhibits/` | 0 | 0 | PASS |

## Verification and visual QA

The exhibit-budget regression was proved fail-before-pass by changing its expected index
count from 16 to 15. It failed on the actual 16-row index; restoring the correct assertion
passed. The complete focused suite (`final_exhibits_test.py`, `g0_u1_window_rescore_test.py`,
and `f6_bootstrap_test.py`) passed 15/15 tests. Isolated Ruff E/F/W reported zero findings.

All six one-page PDFs were checked with Poppler (`pdfinfo` and 180-dpi `pdftoppm` renders).
The review found and corrected three presentation defects before the final replay: the Funds
F1 panel had joined ME and QE observations into one path; F6's edge year collided with its
lowest U3 y-axis label; and transparent artists would not render identically in EPS. The
final renders have no clipped text, overlapping labels, broken glyphs, or illegible panels.

One initial F8 execution also stopped at the U3 `short_span` label because the CSV serialized
the selected value as numeric `3.0`. The selector and appendix selection tag now normalize
that label numerically. This changed no return, weight, NAV, or plotted empirical value.

## Open integration item

F8 provides the complete exhibit tree and traceability inputs. The current `paper/paper.tex`
still contains the W3 prose/TODO integration work described by the roadmap, including stale
pre-G0 U1 narrative text. F8 deliberately did not rewrite owner/Claude prose. Those old
quotations must be replaced with the G0-backed `table_existing_signal.csv` values when W3 is
integrated; they are not part of the accepted F8 exhibit package.

No git staging, commit, tag, or push occurred in F8. Per the roadmap, F8 itself has no gate.
