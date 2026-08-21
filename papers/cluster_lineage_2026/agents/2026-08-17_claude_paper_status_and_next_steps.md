# Cluster-lineage paper: consolidated status and finalisation plan

**Date:** 2026-08-17 (revision 2 — owner Gate 0 and phase rulings incorporated)
**Author:** Claude (owner-side review agent); rulings by owner, recorded verbatim below
**Inputs:** paper/cluster_distribution_theory.tex + refs (2026-08-13 vintage); paper/ROADMAP_cluster_lineage_empirics.md (incl. E8 extension); agents/ROADMAP_factorlasso_depc1_clustering_and_backtest.md; Sol reports 2026-08-14 through 2026-08-17 (E3b, three-universe signal comparison, dePC1 D0–D6, MAC production + TRE sweeps, U2 rank correction, hierarchical-risk summaries, 08-16 evidence summary, 08-17 pipeline summary); replication/ code review.
**Status:** Gate 0 closed. Phases R/T/C scoped per the rulings. Phase M (manuscript assembly) is ACTIVE. Execution instructions for Sol are the next deliverable and are NOT yet issued.

---

## 1 Where the theory stands

The draft `cluster_distribution_theory.tex` (8 pp., compiles clean, frozen 2026-08-13) carries the complete formal layer:

- **Lemma 1** — stationarity/ergodicity of the partition process (EWMA as contracting iterated random function; pushforward; IGARCH survives via the log-moment condition).
- **Proposition 1** — increment law: effective sample size exactly N; k-step variance factor 2(1−λ^k); elliptical correlation noise floor se(ρ) = √(1+κ)(1−ρ²)/√N; closed-form GARCH(1,1) kurtosis multiplier.
- **Proposition 2** — boundary-flip approximation (proved for the flat cut under a G-block model; Ward deferred to simulation, per the Carlsson–Mémoli honesty argument), with expected churn = Σ Φ(−(m_i+δ)/(√2 σ_i)).
- **Corollary** — two δ* calibrations (level, k-independent; innovation, k-dependent), the designed-hysteresis reading of the smoothing parameter.
- Constrained design objective (churn min s.t. fidelity band) with the evolutionary-clustering positioning: the contribution is the *calibration*, not the objective.
- Predictions P1–P7 as numbered environments, each with its named test.

**Validation verdicts already measured (E3/E3b, owner-gated 2026-08-14):**

| Prediction | Verdict | Measured basis |
|---|---|---|
| P1 margins/predicted churn | **Supported** | cross-config predicted-vs-realised correlation 0.863 (U1 full) / 0.872 (U1 headline); per-universe values in the E3 workbooks |
| P3 kurtosis ordering | **Restated** | baseline constant c = realised/Gaussian-predicted absorbed by owner ruling: 0.81 (U2 W-WED), 1.03–1.12 (U1), 1.32 (U3 ME), 2.15 (U3 QE) |
| P4 frequency scaling | **Rejected as an equality; hysteresis component to be REVISED (owner ruling 10)** | realised annualised ratio exceeds predicted on every U1/U2 config row; mean absolute gaps 0.20–0.42; cross-config correlations 0.54–0.62 |
| P5 risk-model invariance | **Supported** | residual-diagonality guard ≤ 0.014 across smoothed configs on all universes |
| P2 knee, P6 ergodicity, P7 turnover | measured in E3/E5/E6 outputs | verdicts consolidate into the theory-scorecard exhibit (Phase R) |

**Open theory items.** (1) %% MECHANISM passages are owner-only; the owner has now supplied the central-claim text (section 4, ruling 3), which seeds them. (2) P1b simulation study (synthetic G-block + GARCH; Ward verification) — approved, to run. (3) One [TODO] citation (multivariate ARCH filtering, J. Econometrics 71, 1996) to verify or drop. (4) The worked example (δ*_lvl ≈ 0.060, δ*_inn ≈ 0.020, swept 0.05) must be re-derived per universe with measured κ̂ and ρ̄ so text and empirics quote one set (adopted deltas: U1 0.0866, U3 0.0691). (5) The P4 revision itself: restate the hysteresis component with the per-universe proportionality constant absorbed, and re-express P4 in its revised form.

---

## 2 Where the empirics stand

### 2.1 Final architecture

**U1 = MSCI US equities, U2 = BlackRock funds, U3 = futures.** The methodology name is **Rolling-Ward** (owner ruling 2). MCF remains the separate lineage-tracker name. Two experiments share one clustering layer:

1. **Signal experiment** — clustering changes cross-sectional score standardisation only (no cluster capital budgets, after the 2026-08-16 U2 rank correction);
2. **Risk experiment** — signal-free, long-only: HRP on the Ward hierarchy and cluster risk budgets B_g ∝ n_g^α against flat ERC and canonical single-link HRP.

Interpretability (former C3) is IN the paper, presented as part of the cluster-smoothing and membership analysis, not as a standalone claim family (owner ruling 4).

### 2.2 Selected signal results (common window 2009-08-31..2026-06-30, RF=0, lag 1)

| | U1 equities | U2 funds | U3 futures |
|---|---|---|---|
| spec | classic 12m−1m, M1-star δ 0.0866, L/S q=25%, min-size-10 fallback, 10 bp | classic 12m−1m, AUM>50m, 55/35/10 long-only, 2M rebalance, 20 bp | ROSAA short-span-3, vol-normalised sizing (15%/σ cap 5), L/S q=25%, 10 bp |
| cluster vs global | +69 bp net, −1.9 pp vol, Sharpe ≈ equal | +2 bp net, −0.68 pp vol, +0.029 Sharpe | −60 bp net, −2.4 pp vol, +0.091 Sharpe |
| cluster vs BICS (U1 only) | +27 bp net, +0.051 Sharpe | — | — |

Supported claim: cluster-standardised scoring improves Sharpe in all three universes and reduces volatility everywhere; no unconditional return claim (U1 standalone long-short return is negative; U3 cluster return trails global).

### 2.3 Risk-allocation results

Positives: U1 Rolling-Ward HRP beats flat ERC (+17 bp, −40 bp vol, +0.031 Sharpe) and canonical single-HRP; U3 equal-cluster RB beats flat ERC (+39 bp, +0.175 Sharpe). U2 is an **informative failure**: unconstrained HRP concentrates 98.9% in low-vol fixed income. Universal mechanism, all three universes: equal-cluster RB raises effective risk clusters (39.7→60.4, 5.9→14.0, 9.4→16.2) and cuts the largest cluster-risk share (6.3→1.7%, 32.5→8.2%, 21.7→6.3%). Report U1/U3 as positive applications and U2 as a limitation.

### 2.4 dePC1 — OUT of the paper (owner ruling 1)

The de-PC1 experiment (D0–D6, robustness-grade result) does not enter the manuscript — it consumes space without carrying the central claim. The `ClusterCorrelationTransform.REMOVE_PC1` feature stays in the FactorLasso package (0.15.0 in tree). The D-series reports and caches remain the archived record.

### 2.5 The honesty ledger (disclosure items for the manuscript)

1. **Sample vintages are fixed, not reconciled (owner ruling 5).** The accepted risk runs stand on their recorded samples (U2 risk on AUM100, U3 risk on the seven-exclusion universe) while the final signal universes are AUM50 and eleven exclusions. No reruns; the manuscript states each table's universe vintage explicitly.
2. **Selection provenance.** The final U2/U3 specs were selected across grids; the grids are disclosed as selection material. No split-window exercise (owner ruling 6): the rolling-forward backtest is the robustness argument.
3. **U2 survivorship** — current-vintage BlackRock catalogue.
4. **U3 frozen-partition caveat** — accepted M1-star partitions predate the liquidity exclusions (exact-universe refit matches 19.3% of dates); the frozen-strategy convention is stated.
5. **Fidelity-band status of adopted cells** — E3b rejected U1 M1-star at the W-WED/156 cell on the taxonomy band; the adopted ME/36 cell quotes its own band verdict.
6. **Inference** — bootstrap CIs on the final headline deltas are the one permitted addition (rulings 5, 7); kept short.

### 2.6 Replication code

Strengths: cache-first determinism with byte-identical replay proofs, acceptance tables, SHA-256 manifests, no-lookahead asserts, focused pytest + Ruff throughout, qis/OptimalPortfolios consumed for backtests and stats. To fix (delegated to Sol, rulings 12–14):

1. `run_backtests.py` and `configs.py` are pyc-recovered shims (`pyc_compat.load_executed`) from the 08-14 workspace loss — reconstruct human-readable source, diff-proven byte-identical.
2. Runner sediment (private cross-imports across ~15 modules) — consolidate the final-spec pipeline into named exhibit scripts.
3. Duplicate futures-exclusion constants (7-ticker vintage vs 11-ticker `OWNER_FROZEN_2026-08-15` layer) — one canonical constant.
4. Rebuild `exhibit_index.csv` for the final claims.
5. First commit + tag of the paper tree (next step, ruling 14).
6. Public reproduction is **request-based** (owner ruling 15) — no public data bundle; a replication statement in the paper.

---

## 3 Owner rulings record (2026-08-17)

| # | Item | Ruling |
|---|---|---|
| 1 | dePC1 classification | Not in the paper; feature retained in FactorLasso package |
| 2 | Named object | **"Rolling-Ward"** |
| 3 | Central claim | Owner-drafted, recorded in section 4 |
| 4 | Interpretability (C3) | In the paper, as part of cluster smoothing and membership analysis |
| 5 | Reconciliation reruns | Rejected. Analysis stays fixed on the elected eligible universes and signals; only bootstrapping is allowed |
| 6 | Split-window discipline | Rejected. The rolling-forward backtest is robust enough |
| 7 | Bootstrap inference | Approved; keep it short |
| 8 | Theory-scorecard exhibit | Approved; "really necessary" |
| 9 | P1b simulation | Approved |
| 10 | P4 treatment | **Revise** the hysteresis component (not a bare rejection) |
| 11 | Mechanism passages, [TODO] citation, prior-art sweep | Approved |
| 12 | Source reconstruction (pyc shims) | Approved; delegated to Sol |
| 13 | Exhibit consolidation + index | Approved; delegated to Sol |
| 14 | First commit + tag | Approved; next step |
| 15 | Public-reproduction split | Request-based |
| 16 | Manuscript skeleton | Agreed |
| 17 | Submission gates | Agreed |

## 4 Central claim (owner-drafted, 2026-08-17)

> Estimator noise may elevate the membership churn in clusters produced from rolling correlation. The noise is amplified at assignment boundaries. Deriving the smoothing threshold from the estimator's noise floor stabilises clusters at bounded fidelity cost. The stabilised clusters improve the risk properties of cluster-scored signals and cluster-budgeted allocations.

(Correction applied to the source text: "if" → "is" in the second sentence. No other edits.)

Subsumption statement that follows: the noise-floor calibration replaces the tuned temporal weight of evolutionary clustering and ad hoc recluster-frequency conventions.

---

## 5 Revised finalisation plan

### Phase R — inference and scorecard (Sol; bootstrap only, no reruns)

- R1. Compact E6-protocol bootstrap CIs (moving-block, block 6, 2,000 draws, seed 20260813) on the headline deltas of sections 2.2–2.3, on the recorded samples. Keep the output short: one CI table per experiment.
- R2. Theory-scorecard exhibit: one table of P1–P7 verdicts with measured statistics, sourced from the E3/E3b/E5/E6 workbooks, with P4 shown in its revised form once Phase T lands.

### Phase T — theory closure

- T1. P1b simulation study (synthetic G-block + GARCH; flat cut proved / Ward verified) — the flip-approximation figure.
- T2. P4 revision: restate the hysteresis component with the per-universe proportionality constant absorbed (consistent with the E3b constant-c ruling); update the Corollary worked example to the adopted deltas (U1 0.0866, U3 0.0691) with measured κ̂ and ρ̄.
- T3. Owner mechanism passages (seeded by section 4); resolve the [TODO] citation; prior-art sweep for the five nearest results with the stated-differences table.

### Phase C — code and traceability (Sol)

- C1. Reconstruct source for `run_backtests.py`/`configs.py`; retire `pyc_compat`; byte-identical proof.
- C2. Consolidate the final-spec pipeline into named exhibit scripts + rebuilt `exhibit_index.csv` (one row per manuscript exhibit: takeaway title, claim, universe, script, workbook/sheet); one canonical futures-exclusion constant.
- C3. First commit of the paper tree and a tag for the exhibit vintage.
- C4. Replication statement (request-based) drafted for the manuscript.

### Phase M — manuscript assembly (ACTIVE)

Skeleton (updated for the rulings): 1 Introduction (central claim, subsumption, roadmap) · 2 Rolling-Ward clustering and MCF lineage · 3 Distribution theory (extended draft + scorecard) · 4 Noise-calibrated smoothing, membership and interpretability analysis (C3 lives here) · 5 Empirical design (three universes, two experiments, point-in-time controls) · 6 Signal evidence · 7 Risk-allocation evidence · 8 Robustness from existing grids (AUM cutoffs, min-cluster-size, short-span sweep, covariance frequency/span; no dePC1, no split-window) · 9 Limitations (honesty ledger) · 10 Conclusion with quantified contributions.

Division of labour: owner writes topic sentences, crux sentences, and mechanism passages; Claude extends the theoretical note into the manuscript, integrates Sol's empirical results, and applies the house-style, fingerprint, and copyedit passes. Submission gates unchanged: number-traceability sweep against the exhibit index, notation table, AI-tell sweep (the theory draft is one marked AI block), talk test + five hostile questions, one hostile-referee pass.

---

## 6 Immediate next actions (confirmed 2026-08-17; NOT yet executed)

**I. Sol execution instructions.** Claude drafts one detailed research-execution roadmap for Sol covering exactly the approved items: R1–R2 (bootstrap CIs + theory scorecard), T1 (P1b simulation), C1–C4 (source reconstruction, exhibit consolidation + index, commit + tag, replication statement). No reruns, no new spec search, no dePC1 work. The roadmap follows the binding reporting protocol (dated stage reports in agents/, acceptance rows, determinism proofs).

**II. Manuscript assembly.** Begins now, in parallel: owner skeleton first, then Claude extends the theoretical note, adds Sol's empirical results as they land, and feeds any gap discovered while drafting back into (I) as an amendment rather than ad hoc requests.
