# TODO — exhibit production for the QF manuscript, capped at the QF baseline

**Date:** 2026-08-17. **From:** Claude (owner-side). **To:** Sol.
**Scope:** produce the exhibits for `paper/cluster_lineage_manuscript.tex` revision 2 (2026-08-17). This file is the binding exhibit list. Read `AGENTS.md` first; the reporting protocol of `agents/ROADMAP_cluster_lineage_empirics.md` applies (stage report in `agents/`, acceptance checks with measured values, gate requests at the end).

## Why the caps exist

The owner benchmarked seven recent QF papers (`agents/2026-08-17_claude_qf_structure_review.md`): median 7 figures, 6 body tables, 25 typeset pages. The manuscript follows that baseline. The failure mode to avoid is exhibit proliferation — one figure per concept, one table per claim family, nothing exploratory in the body. **The lists below are exhaustive. Any exhibit not listed here requires an owner ruling before you build it. When in doubt, consolidate rather than add.**

## Hard budget

- Figures: **exactly 6** (F1–F6). No sub-figure sprawl: a figure may carry up to three panels (one per panel/universe), never a grid of variants.
- Body tables: **8** = the 4 already in the manuscript (`tab:universes`, `tab:signal`, `tab:risk`, `tab:concentration`) + TA, TB, TC, TD below. The 4 existing tables keep their numbers unchanged; your only additions to them are the Part B bootstrap confidence intervals as **companion columns in the same tables**, never separate CI tables.
- Appendix tables: **2** (TE, TF), each consolidating two selection grids.
- No new runs anywhere except the F3 simulation study (Part A, already specified in the roadmap, seeded and recorded). Everything else comes from existing caches.

## Figures

Common requirements: vector **EPS** as the deliverable format (the rQUF class incorporates .eps via epstopdf, and T&F production asks for the original .eps sources rather than generated PDFs — see `.private/qf_author_instructions.txt` and rQUFguide sec. 4.2), plus a PDF copy of each for convenience; one consistent font and size across all six, sized for the rQUF single-column measure (38pc text width), matplotlib through the existing paper harness (no seaborn, no new plotting stack, per the code instructions). Colors and line styles consistent across figures (baseline = one style everywhere, calibrated/treatment = one style everywhere).

- **F1 — churn through time.** Membership churn per estimation date, unsmoothed baseline vs calibrated bonus, one panel per estimation panel (equities / futures / funds). Source: cached E3b stability workbooks. This is the visual carrier of the 89% headline.
- **F2 — churn–fidelity frontier (the paper's signature figure).** Churn against taxonomy-ARI (or the fidelity functional as reported in E3b) traced by the bonus grid, with vertical overlays at delta*_lvl and delta*_inn computed from each panel's span, frequency, and measured kurtosis, and the knee marked. One panel per estimation panel. Source: E3b sweep caches + the Corollary 1 formulas evaluated at measured panel values (same numbers as table TB).
- **F3 — flip-probability verification (simulation).** Predicted flip probability from equation (12) against realized flip frequency, flat cut and Ward, from the synthetic G-block + GARCH(1,1) study. The one new computation; fixed seeds recorded in the output. Design grid per the roadmap.
- **F4 — margins and flips.** Margin histogram with flip rate by margin decile overlaid, per panel (P1). Source: cached E3/E3b margin diagnostics.
- **F5 — cumulative signal performance.** Cumulative NAV of cluster-score vs global-score legs (U1 adds the sector leg), one panel per universe, headline window 2009-08-31..2026-06-30, net of stated costs. Source: the cached backtest NAVs behind `tab:signal`. Numbers must reconcile with the table's net returns.
- **F6 — risk-concentration mechanism.** Effective risk clusters (inverse Herfindahl of cluster-risk shares) through time, flat ERC vs equal-cluster budgets, one panel per universe. Source: the cached risk-allocation runs behind `tab:concentration`. Time-series form preferred; if the caches only carry period averages, report that in the stage report and deliver the bar-chart form instead — do not re-run.

## Tables

- **TA — panel summary statistics** (Section 5.1). ONE table, one row per estimation panel (equities W / futures W / funds ME / funds QE sleeve): asset count, frequency, span N, sample range, number of estimation dates, measured kappa_hat. The kappa_hat values quoted in the prose (2.12 / 1.61 / 0.84 / 1.29) move into this table; they must match the E1 record exactly.
- **TB — calibration bridge** (Section 4). ONE table across panels, NOT one table per panel: rows = panels, columns = N, k, measured kappa_hat, measured rho_bar, implied delta*_lvl, implied delta*_inn, adopted delta (equity cell 0.0866; futures cell 0.0691), sweep knee location. Replaces the stale 2026-08-13 worked example (0.060 / 0.020 / 0.05) — do not carry those numbers forward.
- **TC — churn and fidelity across configurations** (Section 4.1). ONE consolidated table: rows = panel x smoothing configuration (baseline, quarterly-hold, fixed 0.02, calibrated), columns = churn per asset-year, taxonomy-ARI change vs baseline, band verdict, cluster-count change. Include the band verdict of every adopted operating point. Track purity / persistence: fold the headline lineage numbers into this table or the caption notes — no separate lineage table. One case-study track per panel goes in the caption notes or the F1 caption, not as an exhibit.
- **TD — theory scorecard** (Section 5.2, the central theory exhibit; QF precedent: Ratliff-Crain et al. Table 1). One row per prediction P1–P7: test statistic, measured value, bootstrap CI or permutation null where applicable, verdict (supported / rejected / revised-supported). The already-gated values quoted in the manuscript (P1: 0.86 / 0.87; P3 constant c in [0.81, 2.15]; P4 equality rejected, gaps 0.20–0.42; P5: 0.014) must re-trace through this table from the workbooks. The P4 row carries the revised absorbed-constant re-evaluation (owner ruling 10). Ergodicity subsample means (P6) fold into this table's P6 row, not a separate table.
- **TE / TF — selection grids** (Appendix B). Four grids consolidated two per table: (TE) U2 eligibility grid (no cutoff / USD 50m / USD 100m) + U1 minimum cluster-size grid; (TF) U3 short-span sweep + covariance frequency and span grids. Each labelled with its selection role ("selection record, not independent confirmation"). Appendix placement is deliberate — do not promote to the body.

## Caption standard (all exhibits)

Two parts, both mandatory: (1) a takeaway title sentence stating the finding, not the topic; (2) Notes: data source, sample period, units, cost and return conventions where relevant, and the provenance reference (runner script + `exhibit_index.csv` row). Example shape: "Calibrated smoothing cuts membership churn by 89% at a bounded fidelity cost. Notes: U1 = point-in-time MSCI US constituents, weekly, span 156, 2009-08-31 to 2026-06-30; churn in changes per asset-year; source `run_...py`, exhibit index row F1-U1."

## Acceptance checks (report each with its measured value)

1. Every exhibit appears in the rebuilt `exhibit_index.csv` with script, cache path, and commit provenance.
2. Every number visible in an exhibit reconciles with its source workbook/cache to the printed precision; list any mismatch rather than adjusting either side.
3. F5 NAV endpoints reconcile with `tab:signal` net returns; F6 averages reconcile with `tab:concentration`.
4. Simulation (F3): fixed seeds recorded; identical rerun reproduces the figure byte-identically at the data level.
5. Exhibit count equals the budget: 6 figures, 4 new body tables, 2 appendix tables, 0 others.
6. Bootstrap CIs (Part B, E6 protocol) delivered as companion columns for `tab:signal` and `tab:risk` comparisons named in the manuscript TODOs, plus the TD rows — no standalone CI tables.

## Out of scope for this TODO

Manuscript prose (owner + Claude only), the P4 proposition restatement (owner ruling), the mechanism paragraphs, the prior-art sweep, dePC1 exhibits (separate gate, ruling 9), and any exhibit not listed above. If a cache is missing or a number cannot be traced, escalate per protocol (`YYYY-MM-DD_sol_escalation_<topic>.md`) instead of re-running.

## Deliverables

One stage report `YYYY-MM-DD_sol_QFEX_report.md` in `agents/` with the acceptance-check values, the exhibit files under the output root (`CLUSTER_LINEAGE_OUTPUT_DIR/paper_exhibits/`), and the rebuilt `exhibit_index.csv`. End with a GATE REQUEST listing anything that needs an owner ruling.
