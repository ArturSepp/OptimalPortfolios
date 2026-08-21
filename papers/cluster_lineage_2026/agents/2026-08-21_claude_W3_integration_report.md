# W3 integration report — gated numbers into paper.tex (revision 3)

**Date:** 2026-08-21
**Author:** Claude (owner-side), per `ROADMAP_manuscript_finalisation.md` v3, track W3.
**Consumes:** the gated G0/F1–F8/F10 stage reports. **Produces:** `paper/paper.tex`
revision 3. No computation ran; every inserted number is quoted from a gated stage report.

## Gate rulings recorded

1. **OWNER GATE G0 — approved.** The owner's `continue` instruction immediately after the
   mandatory G0 stop (recorded by Sol in the F8 report). The U1 rows of
   `tab:signal`/`tab:risk`, their CI companions, and the U1 narrative are frozen on the G0
   headline-window values. The 2026-08-17 U1 signal quotations are retired.
2. **OWNER GATE F9 — approved.** Owner chat, 2026-08-21 ("go ahead", ruling the gate
   presented in the session summary). The single local commit and the annotated tag
   `cluster-lineage-exhibits-2026-08-21` stand as reviewed. No push (outside every roadmap).

## Edits made (all delimited with W3 BEGIN/END markers in the file)

1. **Abstract**: final numbers (churn 89%; scorecard 4 supported / 2 revised / 1 rejected;
   volatility reduction 0.7–3.9 pp with all intervals excluding zero and no return or
   Sharpe interval doing so). TODO removed.
2. **MCF paragraph** (Section 2): drafted from the F3 report subsection; owner review W1.
3. **Calibration bridge**: table TB (`tab:bridge`) inserted from F1 with the six
   panel/application rows, plus the calibration-transfer disclosure (adopted 0.0866
   transferred from the weekly theory cell; own-cell implied level 0.175).
4. **Membership section**: adopted-cell band verdicts (max dARI 0.015 / 0.009, both PASS)
   and interpretability numbers (label churn 1.16→0.64, 0.54→0.10, 1.34→0.37; purity up on
   every panel) from F3.
5. **Scorecard section**: table TD (`tab:scorecard`, nine rows) inserted from F5; full
   verdict prose for P1–P7 including P2 (knee on 3/4, equity-headline exception), P4
   revised (0.99/0.99/0.94, ordering not levels, from F2), P6 (ARI ranges), and P7
   (conjunction rejected, reassignment mechanism supported with intervals excluding zero).
6. **Simulation section**: F4 results (flat-cut correlation 0.98, zero monotonicity
   violations; Ward 0.97, MAE 0.033 against 0.009 flat).
7. **Signal section**: topic sentence and mechanism paragraph re-framed to the
   volatility-reduction claim; U1 table rows re-based to G0 (−3.87/9.17/−0.38,
   −3.52/10.36/−0.29, −3.94/13.10/−0.24); comparison paragraph rewritten with the G0/F6
   deltas and 95% intervals, marking which exclude zero.
8. **Risk section**: three G0-covered U1 rows re-based (8.79/13.27/0.71, 9.00/12.83/0.74,
   8.86/12.95/0.72); findings paragraph rewritten with G0/F6 intervals (U1 HRP vol −0.44 pp
   excludes zero; U3 RB +39 bp and +0.15 pp vol both exclude zero, Sharpe interval does
   not).
9. **Robustness section**: grid inventory paragraph (four grids, selection-role framing).
10. **Conclusion**: quantified contribution list; TODO removed.
11. **Data availability**: F10 draft statement inserted (its two semicolon sentences split
    per the house rule, content unchanged); owner sign-off W2.
12. **P4 comment blocks** updated: F2 complete, only the owner proposition text pending.

Compile check: pdflatex + bibtex clean, 16 pages, no errors, no undefined citations.

## Voice fingerprint of the inserted prose (drift-control rule)

64 sentences across the ten W3 blocks: median length 21 (band 19–25), share over 25 words
41% (band ≤40% — at the band edge; the long sentences are interval-quoting data sentences),
We + verb openings 6% (band ≥6%), passive share ~16% (band ≤25%), prose semicolons 0,
em-dash parentheticals 0. No L2 table: no owner-written text was edited.

## Remaining number gaps (narrow TODOs in the file, all F8-artifact traces)

- U1 turnover cells in `tab:signal` (3) and `tab:risk` (5), and the two U1 budget rows
  (cluster budgets alpha=0.5, equal-cluster) — quote from `table_existing_risk.csv` /
  `table_existing_signal.csv` in `finalisation/exhibits/` if they carry the G0-convention
  windowed values, else one G0-style re-score of the frozen weight/NAV series (re-scoring
  only, no rerun).
- Exhibit inserts from `finalisation/exhibits/` (F8): tables TA, TC, TE, TF and figures
  F1–F6 (`\includegraphics` blocks with the F8 EPS files).
- These are the only quantitative items between revision 3 and the W4 sweeps.

## Owner items open (unchanged from the roadmap)

W1: title (Gate 0), author block, keywords/JEL confirmation, back-matter texts,
TOPIC-CANDIDATE and MECHANISM passages, MCF paragraph review, [TODO] citation check,
factorlasso version pin. W2: P4 revised proposition text (F2 verdict is in), data
availability sign-off, sample-vintage sentence sign-off, notation table decision. Then W4
sweeps (traceability, fingerprint + AI-tell + copyedit, talk test, hostile referee) and the
W5 submission package.
