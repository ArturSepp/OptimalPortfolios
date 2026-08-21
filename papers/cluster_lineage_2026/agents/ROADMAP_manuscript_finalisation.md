# ROADMAP_manuscript_finalisation — closing stages to submission for the cluster-lineage manuscript

**Date:** 2026-08-21 (v3; replaces the 2026-08-20 v2 by the one-active-roadmap rule)
**Repository:** OptimalPortfolios, `papers/cluster_lineage_2026/` only
**Canonical location:** `papers/cluster_lineage_2026/agents/ROADMAP_manuscript_finalisation.md`
**Executor input:** this file + the repository. Read `AGENTS.md` first (shared agent core:
verification loop, escalation, conventions).
**Predecessor roadmaps:** `paper/ROADMAP_cluster_lineage_empirics.md` (E0–E8, complete) and
`agents/ROADMAP_factorlasso_depc1_clustering_and_backtest.md` (D0–D6, complete; dePC1 is OUT
of the manuscript by owner ruling).
**Owner context:** `agents/2026-08-17_claude_paper_status_and_next_steps.md` (rev 2, rulings),
`agents/2026-08-17_claude_empirical_data_plan.md`, and
`agents/2026-08-17_claude_sol_todo_qf_baseline_exhibits.md` (the binding exhibit list and
budget for F8). The consuming document is `paper/paper.tex` (the QF rQUF2e build;
`cluster_lineage_manuscript.tex` is the frozen revision-2 archive).

## Changes in this revision (v3, 2026-08-21)

1. **Execution status: F0–F7 and F10 are COMPLETE** (reports of record dated
   2026-08-20/21 in `agents/`; one escalation, resolved by owner instruction, retained as
   the audit trail for the F0 missing-data reruns). Completed stage definitions are
   compressed below to their record; their full specifications live in the v2 text and in
   each stage report.
2. **Owner ruling (2026-08-21, recorded from chat):** the 2026-08-17 elected U1/U2/U3
   specifications remain frozen and are reconfirmed — they were not changed by any F-stage
   and are not to be re-run. No further backtest, optimizer, or estimator run of any kind
   takes place in this workstream. Finalisation proceeds on the achieved pipeline outputs.
   The F0 missing-data rerun exception is CLOSED and has no successor.
3. **Joint F5+F6 gate — disposition under that ruling.** (a) The nine scorecard verdicts
   are frozen as measured. (b) The U2/U3 signal rows and the U3 risk rows of the CI tables
   are frozen as measured — their point deltas reproduce the 2026-08-17 recorded values
   exactly. (c) The U1 rows are frozen only after stage G0 below re-scores the existing,
   already-frozen U1 series over their labelled headline window. G0 is a cached-series
   statistic recomputation permitted by global constraint 1; it is not a backtest and runs
   no estimator. The 2026-08-17 U1 narrative values, whose source NAV file was lost in the
   2026-08-14 workspace loss, are quotable only if G0 reproduces them; otherwise the
   manuscript quotes the G0 values under the number-traceability gate.
4. **New stage G0** inserted before F8. F8 and F9 are otherwise unchanged, except that F8's
   U1 rows and CI companion columns consume G0's output and the F9 tag carries the actual
   date.

## Reporting protocol (binding, unchanged)

ALL executor communications live in `papers/cluster_lineage_2026/agents/`, one dated markdown
file each, never appended to this roadmap:

- Stage reports: `YYYY-MM-DD_sol_<stage>_report.md` — what ran, deliverable paths, every
  acceptance check with its measured value against its tolerance, deviations, open items.
- Gate requests: end the stage report with a `GATE REQUEST` section. Do not start a gated
  stage before the ruling is recorded.
- Escalations: `YYYY-MM-DD_sol_escalation_<topic>.md`.
- Every report names the runner scripts and cache directories used.

## Global constraints (binding, all remaining stages)

1. **No estimator refits on market data and, per the 2026-08-21 ruling, no reruns of any
   elected backtest.** Every partition, covariance, weight, and return series is consumed
   from the existing caches and recorded outputs. "Re-scoring" (computing new statistics
   FROM cached partitions, margins, scores, weights, and NAV series) is allowed and is all
   that stage G0 does.
2. **No new empirical search.** No signal, quantile, span, cutoff, delta, AUM, sleeve,
   universe, or cost is selected again. The elected operating points are final. A worse or
   better number found along the way is reported, never acted on.
3. **No dePC1 work.** The transform stays in the FactorLasso package; it enters no exhibit.
4. **No split-window analysis** of the paper's applications.
5. All new code lives in `papers/cluster_lineage_2026/replication/`. Do not modify
   `optimalportfolios/`, `factorlasso`, `qis`, or anything under `rosaa/`.
6. Cache-first, deterministic: identical reruns produce byte-identical non-timing artifacts;
   every stochastic step records its seed in the output.
7. Bootstrap convention: moving-block bootstrap, block length 6, 2,000 draws, seed 20260813,
   percentile 95% intervals, resampled block indices drawn ONCE per draw and applied jointly
   to every series in the comparison.
8. **No git push, no branch deletion, no release actions.** Stage F9 commits and tags
   locally; it does not push.
9. Outputs under `$CLUSTER_LINEAGE_OUTPUT_DIR/finalisation/<stage>/`. Nothing is written
   into existing cache directories.
10. Expected non-errors: the FactorLasso warmup-zeroing warning remains normal.

## Naming conventions (unchanged)

The manuscript reserves U1/U2/U3 for the investment universes and names the theory panels in
words. Output files use `equity_panel`, `futures_panel`, `fund_panel` (theory track) and
`u1`, `u2`, `u3` (applications track). Config ids keep their cache names; the manuscript
renders `M1_star` as the calibrated bonus delta-star.

---

## Completed stages — record of execution (2026-08-20/21)

| stage | status | report of record | frozen headline values |
|---|---|---|---|
| F0 inventory | COMPLETE | `2026-08-20_sol_F0_report.md` + resolved escalation | 65/65 inputs resolve once; nine lost artifacts regenerated at frozen specs under the closed owner exception; inventory SHA-256 `3B2E76DD…4589B` |
| F1 stability/bridge | COMPLETE | `2026-08-20_sol_F1_report.md` | P1 correlations 0.863 / 0.872 / 0.865 / 0.911 with block CIs; knee = calibrated delta in 3/4 frontiers (equity headline knee = fixed 0.05, reported); bridge reproduces frozen calibrations; U1 own-cell implied level 0.1753 vs adopted transferred 0.0866 |
| F2 P4 revised | COMPLETE | `2026-08-20_sol_F2_report.md` | c calibrated on baseline only; cross-config correlations 0.992 / 0.993 / 0.939; supported as ordering, not levels; funds descriptive |
| F3 membership | COMPLETE | `2026-08-20_sol_F3_report.md` | 32-row churn-fidelity table reproduces E3b verdicts; both adopted cells PASS own-baseline bands (max taxonomy dARI 0.0147 / 0.0093); MCF paragraph input delivered |
| F4 simulation | COMPLETE | `2026-08-21_sol_F4_report.md` | flat-cut correlation 0.980 (target 0.90), 0 monotonicity violations; Ward descriptive 0.975, MAE 0.033; seeds recorded, cache-replay deterministic |
| F5 scorecard | COMPLETE, gated per v3 change 3 | `2026-08-21_sol_F5_report.md` | P1, P2, P5, P6 SUPPORTED; P3, P4-revised SUPPORTED-REVISED; P4-original REJECTED; P7 REJECTED as conjunction with reassignment mechanism supported (all three reassignment CIs exclude zero, negative) |
| F6 CIs | COMPLETE, gated per v3 change 3 | `2026-08-20_sol_F6_report.md` | 21 rows; all four signal volatility deltas exclude zero (negative); all signal return and Sharpe CIs include zero; U1 RW-HRP vol reduction vs flat ERC excludes zero; U1 rows subject to G0 |
| F7 source reconstruction | COMPLETE | `2026-08-21_sol_F7_report.md` | readable `configs.py` / `run_backtests.py`; `pyc_compat` retired; 33/33 differential artifacts byte-identical; max discrepancy 0.0 |
| F10 replication statement | COMPLETE | `2026-08-21_sol_F10_replication_statement.md` | request-based paragraph drafted; W2 sign-off pending |

The 2026-08-17 elected specifications were not altered by any stage above. The regenerated
U2/U3 signal series and the U1 risk series reproduce the 2026-08-17 recorded deltas
exactly. The single open numerical item is the U1 signal quotation, resolved by G0.

## Stage G0 — U1 headline-window re-scoring (effort: low; re-scoring only)

**Why.** The regenerated U1 performance helpers label their rows `headline_20090831_20260630`
but compute over the full 2006-08-02..2026-08-05 NAV range (240 monthly observations),
including warmup, while U2/U3 use the intended headline window. The manuscript cannot quote
headline-labelled numbers computed on a different sample. This stage recomputes statistics
from the EXISTING frozen series. It runs no backtest, fits no estimator, generates no weight
or NAV, and touches no elected specification.

**Deliverable.** `replication/run_g0_u1_window_rescore.py` producing, under
`finalisation/g0/`:

1. `u1_windowed_performance.csv`: the U1 signal legs (cluster, global, BICS sector) and U1
   risk legs (Rolling-Ward HRP, flat ERC, single-linkage HRP) sliced to the labelled
   headline window 2009-08-31..2026-06-30 on the same monthly convention as the U2/U3 rows,
   with annualised net return, volatility, and RF=0 Sharpe per leg and per comparison delta.
2. `u1_windowed_cis.csv`: the frozen bootstrap (convention above, same seed) re-run on the
   four U1 comparisons only, same columns as F6.
3. `u1_reconciliation.csv`: for each U1 comparison and metric — the G0 windowed value, the
   F6 full-range value, and the 2026-08-17 narrative value, with the gap the window explains
   and the residual gap attributable to the lost 2026-08-14 artifact. The outcome is
   reported, never acted on.

**Acceptance.** Input NAV files match the F0 inventory fingerprints (nothing regenerated);
zero files written outside `finalisation/g0/`; U2/U3 artifacts untouched; windowed U1
observation counts match the U2/U3 windowed convention; no call into any backtest execution
entry point (loader functions only); deterministic replay byte-identical; focused pytest and
isolated Ruff E/F/W green.

**Report, then OWNER GATE G0** — the U1 rows of `tab:signal`/`tab:risk`, their CI companion
columns, and the manuscript's U1 narrative freeze on the gated G0 values.

## Stage F8 — exhibit rendering, consolidation, and the traceability index (effort: high)

Unchanged from v2 except as noted.
`agents/2026-08-17_claude_sol_todo_qf_baseline_exhibits.md` is the BINDING exhibit list and
budget: figures F1–F6 (vector EPS plus PDF copy, rQUF 38pc measure, one consistent style),
body tables TA–TD, appendix tables TE–TF, and the bootstrap CIs as companion columns inside
`tab:signal` and `tab:risk` (never separate CI tables). Any exhibit not on that list
requires an owner ruling before it is built.

**Deliverable.**

1. `replication/build_final_exhibits.py`: regenerates every manuscript exhibit artifact into
   one tree `finalisation/exhibits/`, consuming the F1–F6 outputs, **the G0 U1 rows**, and
   the four robustness exhibits consolidated from EXISTING grid outputs with no new runs
   (U2 eligibility grid, U1 minimum-cluster-size grid, U3 short-span sweep, covariance
   frequency/span grids), each carrying its `selection_role` column.
2. `exhibit_index.csv` (replaces the E7-era file): one row per manuscript exhibit —
   takeaway title, manuscript section and label, claim family, panel or universe, source
   script, source artifact path, and the agents-report of record.
3. Consolidation of the scattered late-August output locations (`local_outputs/`,
   `data/local_outputs/`) into the external root, copy-only with hashes recorded; originals
   left in place.

**Acceptance.** Every number quoted in `paper/paper.tex` resolves through
`exhibit_index.csv` to a script and artifact, with every U1 number resolving to a G0
artifact; F5-figure NAV endpoints reconcile with `tab:signal`; F6-figure averages reconcile
with `tab:concentration`; exhibit count equals the budget; no orphan exhibits;
deterministic replay.

**Report, no gate.**

## Stage F9 — first commit and tag (effort: low)

Unchanged from v2. Adjust `.gitignore` so that `replication/`, `agents/`, and `paper/` are
tracked while data, caches, outputs, and regenerable build products remain ignored. One
commit containing the three tracked folders, message referencing this roadmap. One annotated
tag `cluster-lineage-exhibits-<actual date>` marking the exhibit vintage. **No push.**

**Acceptance.** `git status` clean for the tracked set; no data file, cache, output, or
editorial correspondence staged (staged file count and total bytes in the report; above
25 MB staged is an automatic escalation); the tag resolves to the commit.

**Report, then OWNER GATE F9.**

---

## Execution order

```text
G0 → OWNER GATE G0 → F8 → F9 → OWNER GATE F9
```

Nothing else remains on the executor track. F7 and F10 are complete; no other computation of
any kind is authorized.

## Manuscript TODO map (updated)

| Manuscript TODO (`paper/paper.tex`) | Stage | Artifact |
|---|---|---|
| abstract numbers | F5 + F6 + G0 | scorecard + CI tables (U1 rows from G0) |
| MCF description paragraph inputs | F3 | stage-report subsection |
| calibration bridge table | F1 | `calibration_bridge.csv` |
| membership/interpretability exhibit + adopted-cell band verdicts | F3 | `churn_fidelity.csv`, `adopted_cell_verdicts.csv`, `interpretability.csv` |
| theory scorecard table | F5 | `theory_scorecard.csv` |
| P4 revised re-evaluation | F2 | `p4_revised.csv` |
| frontier-knee, ergodicity, turnover rows | F1 + F5 | `frontier.csv`, `ergodicity.csv`, scorecard |
| simulation study | F4 | `flip_approximation.pdf`, `ward_verification.csv` |
| signal CI companion table + interval quotes | F6 + G0 | `signal_cis.csv`, `u1_windowed_cis.csv` |
| risk CI intervals | F6 + G0 | `risk_cis.csv`, `u1_windowed_cis.csv` |
| U1 headline performance quotes | G0 | `u1_windowed_performance.csv` |
| robustness exhibits | F8 | exhibit tree + `exhibit_index.csv` |
| sample-vintage captions | F0 + F8 | inventory + index (vintage columns) |
| replication statement | F10 | statement draft (W2 sign-off) |

## Stability-pooling interlude (2026-08-20, CLOSED — record)

The stability-pooled z-score workstream (S0–S6 + S5b; proposer Ben, Monday TAA meeting)
closed with a family REJECT: V3 at S6; V1/V2 at S5b on negative evaluation-half Sharpe
deltas (−0.052, −0.086) under the predeclared rule, with the turnover reduction real but
short of the held-Sharpe bar. No frozen specification, production configuration, or paper
tree was touched; nothing in this roadmap consumes its caches. FactorLasso residue
(unreleased `cluster_statistics` module, public co-association accessor, V3 enum removal)
is OSS-project scope, pending sign-off alongside 0.15.0. One OPTIONAL owner ruling stays
open: the boundary diagnostic (reassigned assets carry mean stability 0.526 vs 0.733;
bottom-quartile assets reassign 33.5% vs 4.2% top-quartile) as one descriptive paragraph in
Section 4.1, credited to Ben. Default is OUT; it enters only by a dated owner note.

## Owner and Claude track to submission

Division of labour unchanged: owner writes topic sentences, crux sentences, and mechanism
passages; Claude extends, integrates numbers, and runs the sweeps.

**W1 — owner items independent of the executor stages (can close now).**
Gate 0 title confirmation (the named object leads); author list, affiliations, email;
keywords and JEL confirmation; acknowledgements, disclosure statement, funding text; review
of the two %% TOPIC-CANDIDATE introduction paragraphs; the mechanism (%% MECHANISM)
passages seeded by the owner central claim; the MCF one-paragraph description (input in the
F3 report); the [TODO] citation check (multivariate ARCH filtering, J. Econometrics 71,
1996: verify or drop); the factorlasso release-version pin for the implementation footnote.

**W2 — owner items gated on stages.**
P4 revised proposition text from the F2 verdict (supported as a cross-configuration
ordering, not levels); data availability statement sign-off on the F10 draft;
sample-vintage sentence sign-off (`tab:risk` note) against the F0 inventory; notation table
decision once the model sections stabilise; G0 gate ruling on the U1 quotation.

**W3 — Claude integration (after the G0 gate, before the W4 sweeps).**
Insert the frozen Part A / Part B numbers and exhibits into `paper.tex` with the seam
protocol. Three content updates are mandatory, all consequences of the frozen record:
(1) the signal-evidence prose quotes the gated traceable values, and the cross-universe
robust claim is stated as the volatility reduction — all four signal volatility CIs exclude
zero while every signal return and Sharpe CI includes zero — with no unconditional Sharpe
claim; (2) one disclosure sentence on the calibration transfer — the adopted U1 delta
0.0866 is the weekly-theory-cell calibration transferred to the ME/36 application, whose
own-cell implied level threshold is 0.1753 (F1 bridge); (3) the P7 presentation separates
the supported reassignment mechanism (all three reassignment CIs negative) from the
rejected full conjunction. Finalise the abstract numbers after the G0 gate; expand
references toward ~40 through the prior-art sweep; delete each `[TODO: ...]` only when its
number traces through `exhibit_index.csv`.

**W4 — closure sweeps on the assembled manuscript (order fixed).**
(1) number-traceability sweep against `exhibit_index.csv` (every quoted number, all
differently worded repetitions); (2) voice fingerprint report + AI-tell sweep + full
copyedit pass; (3) talk test: ten-slide skeleton and five hostile expert questions;
(4) one hostile-referee review-mode pass. Findings from (3)–(4) feed edits, then (1)–(2)
re-run on the touched sections.

**W5 — submission package (after OWNER GATE F9).**
Anonymous PDF per the rQUF guide (\maketitle moved), stated word count, zipped LaTeX
sources with .eps figures, the generative-AI declaration (required by the 2026 QF
instructions; the header provenance block is the input), submission fee (USD 162), cover
letter. Push and submission are owner actions outside every roadmap.

**Gate order to submission:**
G0 → OWNER GATE G0 (U1 quotation frozen) → F8 exhibit tree + index complete → W3
integration closes → OWNER GATE F9 (commit + tag reviewed) → W1/W2 owner text closed →
W4 sweeps → W5 package.

## Out of scope

Any rerun of an elected backtest (2026-08-21 ruling); dePC1 in any form; split-window
analysis of the paper's applications; any refit of estimators on market data; any new
signal, universe, or parameter search; edits to `factorlasso`, `optimalportfolios`, `qis`,
`rosaa`; stability-pooling adoption work (closed); git push, publication, or release
actions.
