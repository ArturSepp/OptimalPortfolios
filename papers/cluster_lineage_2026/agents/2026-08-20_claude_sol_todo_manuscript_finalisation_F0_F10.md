# Sol execution order — manuscript finalisation, stages F0–F10

**Date:** 2026-08-20
**From:** Claude (owner-side), at owner instruction. **To:** Sol.
**Executes:** `agents/ROADMAP_manuscript_finalisation.md` (v2, 2026-08-20). That file is the
binding stage specification. This file is the dispatch order only: it fixes the execution
sequence, the stop points, and the current-state facts. It defines no new stage, no new
tolerance, and no new exhibit. If this file and the roadmap conflict, the roadmap wins and
the conflict is escalated before proceeding.

## Read first, in this order

1. `AGENTS.md` (shared agent core: verification loop, escalation, conventions).
2. `agents/ROADMAP_manuscript_finalisation.md` (v2, 2026-08-20) — all of it, including the
   "Changes in this revision" block and the "Stability-pooling interlude" section.
3. `agents/2026-08-17_claude_sol_todo_qf_baseline_exhibits.md` — the binding exhibit list
   and budget consumed by stage F8.
4. `agents/2026-08-17_claude_paper_status_and_next_steps.md` (rev 2, owner rulings) and
   `agents/2026-08-17_claude_empirical_data_plan.md` (Part A / Part B split).

## Current-state facts (verify before F0, report in the F0 report)

1. **Fresh start.** No `*_sol_F*_report.md` and no `*_sol_QFEX_report.md` exists in
   `agents/` as of this dispatch. If you find one, stop and escalate — the dispatch is then
   stale.
2. **The live manuscript is `paper/paper.tex`** (QF rQUF2e build). Assert its presence.
   `cluster_lineage_manuscript.tex` is the frozen revision-2 archive; consume it for
   nothing.
3. **Stability pooling is CLOSED (family REJECT, S5b 2026-08-20).** Do not resume any
   S-stage, do not consume `stability_pooling/` caches in any F-stage, and do not build the
   boundary-diagnostic exhibit — that owner ruling is open with default OUT.
4. **Environment as in the S5b run:** OneDrive FactorLasso and OptimalPortfolios checkouts
   first on `PYTHONPATH` with the runtime assertion on `factorlasso.__file__`,
   `CLUSTER_LINEAGE_OUTPUT_DIR` set, `MPLBACKEND=Agg`. Outputs under
   `$CLUSTER_LINEAGE_OUTPUT_DIR/finalisation/<stage>/` per the roadmap.

## Execution sequence (this dispatch)

**Tranche 1 — theory, inference, and scorecard:**

```text
F0 → F1 → F2 → F3 → F6 → F4 → F5 → GATE REQUEST (F5+F6 jointly) → STOP tranche 1
```

End the F5 stage report with one `GATE REQUEST` section covering F5 and F6 together
(the roadmap permits gating them jointly). Do not treat the gate as ruled until the ruling
is recorded in a dated owner note in `agents/`.

**Interleaved exhibit rendering (draft status until the gate).** Per roadmap v2, cache-only
exhibits (figures F1/F5/F6, tables TA/TE/TF) may be rendered any time after F0, and
stage-gated exhibits as their inputs land (figures F2/F4 and table TB after F1; figure F3
after F4; table TD after F5; CI companion columns after F6). Every render follows the QFEX
budget, format (vector EPS plus PDF copy, rQUF 38pc measure, consistent styles), and
caption standard. All renders are DRAFT until the F5+F6 ruling; the final regeneration and
the acceptance object is stage F8. Building any exhibit not on the QFEX list requires an
owner ruling first.

**Tranche 2 — while awaiting the F5+F6 ruling:**

```text
F7 (source reconstruction) → F10 (replication statement)
```

Both are independent of the gate and of each other's outputs.

**Held until rulings — do not start:**

- **F8** starts only after the F5+F6 gate is ruled (the exhibit tree freezes numbers the
  gate must first confirm).
- **F9** starts only after F8 and ends with its own `OWNER GATE F9` request. No push.

## Reporting protocol (binding, from the roadmap)

- One dated report per stage: `YYYY-MM-DD_sol_F<stage>_report.md` in `agents/`, never
  batched — one file per stage even if several stages complete in one session.
- Every acceptance check reported with its measured value against its tolerance.
- Every report names the runner scripts and cache directories used.
- Escalations: `YYYY-MM-DD_sol_escalation_<topic>.md`. Escalate rather than improvise.

## Hard escalation triggers (stop the stage, file the escalation)

1. F0: any input required by F1–F6 that is missing or resolves to more than one path.
2. F3: a required adopted-cell baseline cache does not exist — never fit one.
3. F4: a flat-cut verification target fails (correlation ≥ 0.9, churn monotone in delta) —
   never tune the simulation toward the target.
4. Any number that cannot be traced to a named cache or report.
5. Any change that would touch `factorlasso`, `optimalportfolios`, `qis`, or `rosaa`
   source, or any step that would refit an estimator on market data.
6. Anything the roadmap's "Global constraints" forbid: new empirical search, dePC1,
   split-window analysis, git push, release actions.

## Open owner rulings (context only — none blocks tranche 1; act on none)

Boundary-diagnostic inclusion (default OUT); Gate 0 title; the W1/W2 owner text items in
the roadmap. These proceed on the owner track in parallel.

## Definition of done for this dispatch

F0–F5 stage reports filed with the joint F5+F6 gate request pending; F7 and F10 reports
filed; draft exhibit set rendered within the QFEX budget; zero F8/F9 work; zero
escalation-worthy events unescalated. Nothing further until the owner rules the gate.
