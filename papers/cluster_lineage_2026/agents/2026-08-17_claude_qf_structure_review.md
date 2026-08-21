# QF structural benchmark and review of cluster_lineage_manuscript.tex

**Date:** 2026-08-17. **Reviewer:** Claude (owner-side), review mode per project instructions v2.6.
**Sources:** the seven published QF papers in `Research Papers/20_Journal_Collections/Quantitative Finance/`, measured programmatically from the PDFs; `paper/cluster_lineage_manuscript.tex` (2026-08-17 revision 1, 14 pp. compiled). Every number below traces to those files. Word counts from `pdftotext` extraction are approximate to a few percent; reference counts are pattern-based estimates.

---

## 1. Structural anatomy of the seven QF papers

| paper | pages | words | abstract | intro words | top-level sections | figures | tables | numbered eqs | appendices | refs (approx.) |
|---|---|---|---|---|---|---|---|---|---|---|
| De Blasis et al., network MTD portfolio optimization | 20 | 11.5k | 182 w / 9 sent | 1,911 | 4 | 3 | 6 | 14 | 2 | 26 |
| Herculano, Betting Against (Bad) Beta | 11 | 7.5k | 109 w / 7 sent | 1,065 | 5 | 6 | 6 | 13 | 0 | 25 |
| Wilinski et al., classifying/clustering trading agents | 24 | 15.4k | 203 w / 9 sent | 1,149 | 4 | 16 | 9 | 2 | 0 | 45 |
| Zhang et al., ClusterLOB | 26 | 12.5k | 269 w / 10 sent | 920 | 6 | 7 | 6 | 25 | 1 | 26 |
| Fathi & Grobys, GK power-laws | 27 | 18.7k | 146 w / 5 sent | 1,990 | 8 | 1 | 17 | 27 | ≥6 (to F) | 59 |
| Ratliff-Crain et al., revisiting Cont | 32 | 22.8k | 227 w / 8 sent | 4,484 | 4 | 11 | 3 | 0 | 8 | 73 |
| Yang et al., non-linear ESG premium | 25 | 17.2k | 141 w / 8 sent | 1,850 | 7 | 9 | 9 | 18 | 2 | 54 |
| **median** | **25** | **15.4k** | **182 w** | **1,850** | **5** | **7** | **6** | **14** | — | **45** |

### Norms distilled from the sample

1. **Canonical skeleton.** Introduction (contributions + roadmap) → Background or Literature → Data → Methodology → Results → Robustness → Discussion/Conclusion. Every one of the seven gives data a dedicated, subsectioned home with a summary-statistics table.
2. **Figure-rich.** Median 7 figures; the two clustering papers carry 7 and 16. The only near-figureless paper (GK, 1 figure) compensates with 17 tables. No paper has zero figures.
3. **Subsection discipline.** Results sections are always subsectioned, frequently to three levels (3.1.2, 4.2.1). Per-result subsections are the QF idiom: Ratliff-Crain devote one subsection to each of Cont's facts (3.1–3.8); Yang et al. run 4.2.1 Static test / 4.2.2 Dynamic test.
4. **No theorem environments.** None of the seven uses lemma/proposition/proof environments in the body. Formal and secondary material goes to appendices (5 of 7 have appendices; Ratliff-Crain have eight).
5. **Front matter.** Keywords 7/7 (5–7 terms); JEL codes 5/7. Full affiliations with addresses, corresponding author marked, received/accepted dates.
6. **Back matter.** Disclosure statement 7/7; Funding 4/7; Acknowledg(e)ments 4/7; ORCID 3/7; Data availability statement or Taylor & Francis Open Scholarship badge section 4/7 (ClusterLOB, GK, and ESG carry the badge section for shared code/materials; the network-MTD paper carries a data availability statement).
7. **Titles.** 4–17 words, median 8. Colon constructions are common ("ClusterLOB: enhancing...", "Modeling variance risk...: new evidence from...").
8. **Abstracts.** 109–269 words, single paragraph, no citations (one exception), concrete findings stated with numbers or counts ("conclusive evidence for eight of Cont's original facts").
9. **Roadmap paragraph.** Present in 4/7, always the last introduction paragraph.
10. **Verdict-table precedent.** Ratliff-Crain's Table 1 is exactly a prediction scorecard (fact-by-fact evidence verdicts in clock-time and event-time). Your planned theory scorecard has a direct QF precedent as a headline exhibit.
11. **References.** 25–73, median 45. The methodology-plus-application papers cluster at 45–59.

---

## 2. Where the draft sits against these norms

| metric | draft | QF sample | verdict |
|---|---|---|---|
| pages (compiled) | 14 (article class) | 11–32 typeset | in range; BABB (11 pp., 7.5k words) proves compact is publishable — do not pad |
| prose words | ~4.3k (+ 18 TODO blocks) | 7.5k–22.8k | will land ~8–10k after Parts A/B; fine |
| abstract | ~200 w | 109–269, median 182 | in range |
| introduction | ~450 w | 920–4,484, median 1,850 | **short by QF norms** |
| top-level sections | **11** | 4–8 | **over; consolidate** |
| figures | **0** | 1–16, median 7 | **the largest gap** |
| tables | 4 | 3–17, median 6 | in range after Part A/B exhibits |
| numbered equations | 15 | 0–27; 13–27 for methodology papers | in range |
| theorem environments | 8 (+3 proofs in body) | 0 in all seven | **move proofs to appendix** |
| appendices | 0 | 5 of 7 have them | add (proofs + robustness detail) |
| keywords / JEL | none | 7/7 and 5/7 | **add** |
| disclosure/data/ORCID block | none | 7/7 disclosure; 5/7 data statement | **add skeleton** |
| title | 16 words | 4–17, median 8 | at the long extreme; shorten |
| references | 23 | 25–73, median 45 | expand via the prior-art sweep |
| roadmap paragraph | present | 4/7 | keep |

---

## 3. Actionable edits, in priority order

**E1. Build the figure set (largest structural gap).** Zero figures against a QF median of seven. Six figures, all already latent in your TODOs and predictions — no new analysis beyond Part A/B is required:

- F1 (Section 5): churn through time, baseline vs calibrated smoothing, one panel per estimation panel. Carries the 89% claim visually.
- F2 (Section 5, the signature exhibit): churn–fidelity frontier traced by δ, with δ*_lvl and δ*_inn overlaid and the knee marked (Prediction 2). This is the figure a reader lifts into their own talk — the canonicalization exhibit for the noise-floor calibration.
- F3 (Section 5): simulation study — predicted flip probability Φ(−(m+δ)/√2σ) against realized, flat cut and Ward (the planned P1b figure).
- F4 (Section 5): margin histograms and flip rate by margin decile (Prediction 1).
- F5 (Section 7): cumulative NAV of cluster-score vs global-score legs per universe. Direct precedents: ESG Figure 2 ("Cumulative value of ESG factors"), BABB Figure 3 (risk-return profile). Table 2 alone is not how QF papers present backtests.
- F6 (Section 8): risk-concentration mechanism — effective risk clusters or largest cluster-risk share through time, flat ERC vs equal-cluster. Turns Table 4's universal finding into the visual takeaway. Optionally one lineage case-study panel for interpretability (precedent: Wilinski et al.'s cluster visualizations).

**E2. Consolidate 11 sections to 8.** No QF sample paper exceeds 8. Suggested mapping, everything else unchanged:

- §6 Design + §7 Signal + §8 Risk → one section "The investment applications" with subsections 6.1 Design and universes, 6.2 Signal evidence, 6.3 Risk evidence. QF precedent: ESG §5–6 hold multi-experiment material as subsections.
- §9 Robustness + §10 Limitations → "Robustness and limitations" (precedent: Wilinski §4 Discussion contains 4.1 Conclusions, 4.2 Limitations, 4.3 Future directions; ESG §6 "Robustness and further results").
- Result: Intro / Rolling-Ward / Theory / Smoothing / Confirmation / Applications / Robustness and limitations / Conclusion.

**E3. Add a Data subsection with a summary-statistics table (7/7 QF papers have one).** Natural home: Section 5.1 "Estimation panels and data". One table: per panel — asset count, frequency, span, sample range, estimation dates, and the measured κ̂ values that currently sit in prose (2.12 / 1.61 / 0.84 / 1.29). This also serves the reviewer-affordances gate (data vintages stated). The investment-universe table (current Table 1) stays in the applications section.

**E4. Move proofs to an appendix; keep statements in the body.** None of the seven QF papers carries in-text proof environments; five carry appendices. Keep Lemma 1, Propositions 1–2, Corollary 1 (and the surrounding Hamilton-test prose) in the body; move the three proofs to Appendix A. The difficulty-signposting sentence about Ward vs flat cut (the honest-division paragraph) stays in the body — it is the crux, not proof detail. Appendix B can absorb the robustness grids of §9.

**E5. Shorten the title.** 16 words against a QF sample of 4–17 (median 8) — inside the observed range but at its extreme. The named object should lead. Suggested: *"Rolling-Ward clustering: noise-calibrated stability for rolling correlation-based clusters"* (10 words). The applications belong in the abstract, not the title. (Gate 0 naming ruling still applies first.)

**E6. Add keywords and JEL codes (7/7 and 5/7).** Suggested, for your selection: Keywords — Correlation clustering; Hierarchical clustering; Cluster stability; EWMA; Hierarchical risk parity; Cross-sectional momentum. JEL — C38, C58, G11.

**E7. Add the Taylor & Francis back-matter skeleton now** (with TODOs), so submission doesn't discover it: Acknowledgements, Disclosure statement (7/7), Funding, ORCID, Data availability statement (4/7 carry a data statement or badge section). The data statement is load-bearing for this paper: MSCI/Bloomberg inputs cannot ship, `factorlasso` and the replication harness can. ClusterLOB, GK, and ESG carry Open Scholarship badge sections for exactly this split (code/materials badge without raw data) — worth pursuing, and it aligns with the Phase C public-reproduction ruling. Also verify the current T&F/QF policy on AI-assistance disclosure at submission; the manuscript header records the drafting provenance, and the journal policy determines what must be stated.

**E8. Deepen the introduction from ~450 to ~1,200–1,500 words.** Median QF intro is 1,850; the current intro is below every paper in the sample. Two paragraphs are missing, both endable with the house-rule contribution sentence:

- A QF-native lineage paragraph: Mantegna's correlation trees, the dynamics-of-correlation-networks line, HRP in portfolio construction, and the recent QF clustering papers — ending with: that literature studies the partition at a point in time; we treat the sequence. This is also where the "successor to the QF-native tradition" positioning (knowledge file, Gate 0) becomes visible to the editor and referees.
- A stability-literature paragraph: clustering-stability results (ben-David/von Luxburg), evolutionary clustering (Chakrabarti, Xu) — ending with: those objectives are tuned; we derive the temporal weight from the estimator's noise floor. The citations already exist in the bib; this is repositioning from Remark 3 and §4 into the introduction, where QF referees expect to find it.

The contributions paragraph ("We make four contributions...") is already the QF idiom (ClusterLOB does exactly this) — keep.

**E9. Expand references from 23 toward ~40.** Sample range 25–73, median 45; the methodology-plus-application papers sit at 45–59. The prior-art sweep gate will drive most of this. The visible holes given the positioning: the correlation-network dynamics literature between Mantegna (1999) and Marti et al. (2016) — the QF-native line the paper claims succession to — plus the momentum and risk-budgeting literatures beyond the single current citation each (Jegadeesh-Titman; Maillard et al.).

**E10. Subsection the confirmation section.** QF results sections are always subsectioned. Suggested: 5.1 Estimation panels and data (E3's table), 5.2 The theory scorecard (verdict table — Ratliff-Crain Table 1 is the precedent), 5.3 Simulation study. The per-prediction verdict prose then anchors to scorecard rows rather than running as one block.

**E11. Caption standard, confirmed by precedent.** BABB's captions carry a "Notes:" block naming the construction, equations, and sample ("News shocks are calculated as outlined in equations (8)-(9) ... until 2023:07"). Template for your exhibit TODOs: takeaway title sentence (house rule) + Notes: data source, sample period, units, cost convention, provenance script. The house takeaway-title rule goes beyond QF practice (their captions are descriptive); QF does not mandate caption style, so the house rule stands — flagged per precedence rule 4.

**E12. Acronym audit, measured (notation-budget rule).** Body counts excluding comments: HRP 12, ERC 12, EWMA 9, GARCH 8, ARI 4, VI 3, AUM 3, MSCI 1. Under the fewer-than-five rule: ARI and VI survive only as math operators with the notation table (planned) carrying them; spell out AUM at its three prose uses (table cells exempt); MSCI is a proper name, not an acronym to expand. MCF no longer appears in the body — the TODO list's "MCF" entry can be struck. Newly coined acronym count: zero (Rolling-Ward is a name, not an acronym) — compliant.

---

## 4. Voice fingerprint of the draft (reported per the drift-control rule)

Measured on the 198 extracted prose sentences of the manuscript body (comments stripped):

| metric | draft | band | verdict |
|---|---|---|---|
| median sentence length | 20 | 19–25 | in band |
| sentences over 25 words | 31.8% | ≤ 40% | in band |
| passive share (approx.) | ~9% | ≤ 25% | in band |
| We + verb openings | 6.1% | ≥ 6% | in band |
| prose semicolons | 0 in body prose | 0 | in band (all matches are in LaTeX comments and TODO placeholder text — keep them out of finalized captions) |
| em-dash parentheticals | 0 | 0 | in band |
| imported vocabulary, classes 1–7 | 0 detected | ≤ 0.05/1k, class 3 = 0 | in band |

Copyedit-pass scan: no complementation errors (allows to / enable to / suggest to), no "the both", no doubled words (the r_t r_t' matches are outer products). No L2-error table required — the body is a generated block; the seam-protocol joins around the %% OWNER passage are where the final copyedit pass should read either side.

---

## 5. Already in line — do not touch

The abstract length (in the QF 109–269 band), the roadmap paragraph (4/7 precedent), the numbered-equation count (15 vs 13–27 for methodology papers), equation punctuation as sentences, author-date natbib, the contributions paragraph form, the comparative-claim honesty in §7 (the ESG paper's hedged comparative register is the same), and the Limitations content (the five items match the honesty ledger). The compact overall length is defensible: BABB published at 11 pages — resolve the TODOs, add the six figures, and resist padding beyond ~10k words.

## 6. Suggested execution order

E5/E6/E7 are mechanical and can land today. E2/E3/E4/E10 are LaTeX restructuring with no numerical content — safe before Part A/B numbers arrive. E1 figures and the scorecard consume Part A/B outputs (Sol's Phase R/T deliverables). E8/E9 are owner-plus-Claude writing work gated on Gate 0 (naming, central claim) and the prior-art sweep.
