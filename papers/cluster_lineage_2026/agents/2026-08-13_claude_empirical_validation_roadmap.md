# Empirical validation roadmap — cluster-lineage paper (draft for owner review)

**Date:** 2026-08-13
**Author:** Claude (research assistant), for Artur Sepp
**Status:** draft. On sign-off this becomes `roadmap/ROADMAP_cluster_lineage_empirics.md` in the OptimalPortfolios repository for staged dispatch to Sol, following the `ROADMAP_cluster_smoothing.md` pattern (stage instructions, acceptance checks, stage reports, owner gates).
**Incorporates the 2026-08-13 rulings:** constrained objective adopted (minimise churn subject to a fidelity band), scope fixed (propositions + simulation in the QF paper), theory track P1a–P1c running in parallel and feeding calibrated parameters into this programme.

---

## 1. Purpose and claims under test

The empirical programme serves the paper's central claim: stable, economically labelled risk clusters are obtainable inside a rolling factor-model estimation at zero cost to the risk model. Three claim families, each with its universe of evidence:

- **C1 (stability):** temporal smoothing reduces membership and lineage churn by a large factor at bounded fidelity cost, across three universes, and the reduction is predicted by the boundary-flip theory (margins vs the estimator noise floor).
- **C2 (significance):** stable clusters reduce reassignment-driven turnover in a cluster-relative momentum overlay, improving net-of-cost performance at the stated cost levels, against both a no-cluster control and a static-taxonomy control.
- **C3 (interpretability):** the labelling grammar produces taxonomy-aligned, persistent, parsimonious labels, with a shared vocabulary across the two MATF universes.

Theory-validation checks are embedded (stage E3) so the propositions from the parallel theory track are confirmed or falsified on the same runs that produce the paper exhibits.

## 2. The three universes (fixed design)

| # | universe | factor model | frequency | est. dates | costs | taxonomy for (iii) |
|---|---|---|---|---|---|---|
| U1 | MSCI US (S&P 500 proxy), point-in-time membership | FF6 (Mkt, SMB, HML, RMW, CMA, MOM) | W-WED, span 156 | ME | 10 bp | GICS sector / industry group / industry |
| U2 | Global futures, 95 contracts | MATF custom | W-WED, span 156 | ME | 20 bp | asset_class × geography |
| U3 | MAC funds, 170 ME + 17 QE | MATF custom | ME span 36 + QE span 12 | ME | 50 bp | Asset Class / Sub Asset Class |

Fixed across universes: FCGL at production settings (reg_lambda 1e-5, ward, one_minus_rho, Pearson, cutoff 0.6, normalized group penalty, auto-sign t=1.0, CLARABEL), expanding-window point-in-time estimation, factor covar span 52 at W-WED (U1, U2) and the production MAC configuration for U3. U3 replicates the mac_apac production setup exactly (Rand 0.997 replication already on record), so the paper's practice case is the production estimator, not a variant.

**Config grid per universe** (the treatment axis, identical everywhere):
`baseline` (NONE), `M0` HOLD at QE, `M1` δ ∈ {0.05, 0.10}, `M2` λ ∈ {0.5, 0.7}, plus two calibrated entries once the theory track delivers them: `M1*` at δ\*(span, freq, kurtosis) and `M2*` at λ\*(span, rebalancing step). Backtest-only controls: `no-cluster` (whole-universe ranks) and `static-taxonomy` (ranks within GICS industry / asset class / Sub Asset Class).

**Sample decisions needed (D1–D3, section 8):** U1 estimation start (recommend 2012-01, ≈ 175 ME snapshots, universe ≈ 500–630 names per date); U2 start (recommend 2002-01, ≈ 295 snapshots, factors begin 1998-12); U3 fixed at the production 284 snapshots 2002-12..2026-07.

## 3. Test (i): the stability metric suite — definitions

One metrics module, written once at E0, reused by every stage, frozen before any sweep runs. Definitions below are the paper's definitions; the module docstring carries them verbatim.

**A. Partition stability.**

1. **Raw membership churn** — per-asset cluster-membership changes per asset-year, counted over consecutive dates where the asset is clustered at both, with cluster ids matched greedily by maximum member overlap. (Existing `annualized_churn` ∘ `greedy_membership_panel`; U1 additionally conditions on index membership at both dates, so entry/exit turnover is excluded — see metric 7.)
2. **Lineage churn and lineage stats** — derived-id changes per asset-year through `factorlasso.analyze_cluster_lineage` at module defaults, plus: number of derived tracks, tracks per asset, matcher-attributable churn (cohort retention ≥ 0.60), mean continuation-link overlap, median track life (dates).
3. **Consecutive-partition distances** — per-transition ARI(Q_t, Q_{t−1}) and variation of information VI(Q_t, Q_{t−1}); report medians and IQRs. VI is the theory-side metric (a true metric on partitions); churn is the practice-side metric; both appear so the theory and the exhibits speak the same language.
4. **Resolution and shape** — per-date cluster count (median/min/max), median cluster size, singleton share, largest-cluster share (mega-cluster guard, flag > 0.25 of universe), entropy of the size distribution. Guards against stability bought by degeneracy — the known failure mode of unconstrained stability optimisation.
5. **Co-association confidence** — trailing 6-date peer co-clustering frequency per assignment (existing factorlasso panel); report the cross-sectional median and the share below 0.5 (weakly attached assets).
6. **Margin distribution (theory link)** — per asset and date, the assignment margin m_i: gap between distance-to-own and distance-to-nearest-other cluster at the cut, expressed in units of the per-transition distance noise σ_d = √(2(1−λ^k))·(1−ρ̂²)/√N_eff. Deliverables: the margin histogram, and predicted churn Σ_i Φ(−(m_i+δ)/(√2σ_d)) vs realised churn per config. This is the direct empirical test of the boundary-flip proposition.
7. **Membership-flow decomposition (U1)** — total asset-date reassignments split into {index entry/exit, warmup entry, clusterer reassignment}; only the last is churn. Prevents index turnover from contaminating the headline metric.

**B. Fidelity and cost guards (the adopted constraint, made operational).**

8. **Fidelity band** — median per-date ARI of each smoothed partition against the same-date baseline partition; the constraint is |ΔARI_taxonomy| ≤ 0.03 against baseline on every taxonomy level (the S1 guard, retained) plus cluster-count median within ±15% of baseline.
9. **Risk-model invariance** — per-date relative Frobenius distance of the fitted covariance vs baseline, max relative entry change, and the absolute difference in ex-ante EW-portfolio vol; plus the residual-diagonality guard (mean diagnostics within 5% of baseline). This generalises the covariance-invariance KEY RESULT from S&P 500 to all three universes and is the "zero cost to the risk model" evidence.
10. **Signal-rank stability** — month-over-month Spearman of within-cluster percentile momentum ranks, mean absolute rank change, and the reassignment-attributable rank-MAD gap (recompute ranks under the prior partition; the gap isolates partition-driven rank movement). Existing definitions, unchanged.

## 4. Test (ii): the momentum backtest arm

**Mechanics (mirrors `run_cross_mandate_analysis` / `profile_alpha_signals`).** For each universe and each config: build the momentum score panel, rank within the peer-group structure, hold the top quantile equal-weighted, compare all legs against the equal-weight-all benchmark in one profile. The public engine is `optimalportfolios.alphas.profile_alpha_signals(prices, alpha_scores, quantile, rebalancing_freq)` + `compute_alpha_rank_analysis_table` + `generate_alpha_profile_report` — no rosaa import needed in the paper harness. Costs enter at the stated per-universe levels (10/20/50 bp); if the profiler does not expose `rebalancing_costs`, the top-quantile weights are passed to `qis.backtest_model_portfolio(weights, rebalancing_costs=..., weight_implementation_lag=1)` so cost handling is uniform. Low-beta is excluded by ruling (benchmark ambiguity); momentum only.

**Signal definition (D4).** Total log return over the trailing window skipping the recent period: 48w skip 4w (U1, U2); 12m skip 1m (U3 ME sleeve). Recommend the plain unscaled score for the paper (transparent, matches the S12 spec) with the production vol-normalised variant (long_span 12, vol_span 13) as a robustness row. Quantile 1/3 per the cross-mandate default (D5).

**Legs per universe profile:** EW-all benchmark; `momentum` (no-cluster whole-universe rank — the control that justifies clustering at all); `momentum_taxonomy` (rank within static GICS industry / asset class / Sub Asset Class — the free-alternative control); `momentum_cluster` under each config {baseline, M0, M1 grid, M1\*, M2 grid, M2\*}. Rebalancing ME everywhere; U3 QE-sleeve assets rebalance at QE within the ME schedule (D6).

**Control points, as specified:**

- **Turnover** — annualised one-way turnover per leg; cost drag in bp/yr; and the decomposition into signal-driven vs reassignment-driven trades (trade the same signal under the prior-date partition; the difference in traded notional is the reassignment component). The prediction is monotone: reassignment turnover falls with smoothing, signal turnover is invariant.
- **Performance** — net-of-cost total return, vol, Sharpe, and IR vs the EW-all benchmark via `qis.PerfParams` / the sanctioned qis estimators; crisis-window breakdown (GFC 2008 where the sample allows, COVID 2020, rate shock 2022) per the cross-mandate pattern; net Sharpe/IR deltas vs baseline clusters with block-bootstrap CIs from E6.

**Claim discipline.** The claim is implementability: smoothing cuts reassignment turnover and cost drag at unchanged or better net performance and unchanged fidelity. Gross alpha differences between configs are expected to be statistically indistinguishable, and the paper says so; significance lives in the turnover/cost channel and its confidence intervals. At 50 bp (U3) the cost channel is largest — the practice case is where the framework matters most, which is the right story for QF.

## 5. Test (iii): interpretability metrics

Interpretability is measured, not asserted:

11. **Taxonomy alignment** — median per-date ARI (and AMI, which is better behaved for many small clusters) against each taxonomy level; the *location of the peak* (industry vs sector on U1) is itself an exhibit: clusters recover fine structure, not just sectors.
12. **Track purity** — member-weighted modal-taxonomy share per track; coverage-weighted average across tracks; the count of "core" tracks (coverage ≥ 0.7) that read as nameable economic groups.
13. **Label persistence** — label-string churn per asset-year (the reporting view), share of track life spent under the modal label, and the label-vintage flip rate across successive runs.
14. **Label parsimony and coverage** — distinct labels vs cluster count; primary-factor variance share per track; 'Idio' share (grammar coverage: how much of the universe the vocabulary describes).
15. **Cross-universe vocabulary (headline exhibit)** — U2 and U3 share the MATF panel, so the grammar emits the same label space: one table of label sets side by side with exposure profiles, demonstrating the coordinate-system property. U1 (FF6) demonstrates the grammar ports to a different factor panel. Futures equity-beta bucket thresholds get their per-universe review here (leverage-free betas).
16. **Case studies** — three named tracks per universe (birth/merge/split narrative, loadings path, membership), the qualitative exhibit a PM reads first.

## 6. Stages

Each stage: deliverable, acceptance checks, then an owner gate before the next. All code in `papers/cluster_lineage_2026/` (canonical environment per the pending P0 ruling), consuming factorlasso/optimalportfolios/qis only — no rosaa imports, no package changes, deterministic, causal.

**E0 — Metric library and config registry.** `replication/metrics.py` implementing section 3 and 5 definitions with module-docstring formulas; a config registry enum for the grid; regression tests reproducing the frozen S&P 500 baseline numbers (lineage churn 3.2115, 216 tracks, ARI medians) from the new module. *Acceptance:* regression pass; definitions doc extracted for the paper's methods section.

**E1 — Data layer.** Universe builders from `data/`: U1 point-in-time membership masking + FF6 collection (Ken French daily → W-WED compounding → factor NAVs; excess-return convention per D7) + MSCI US index series as market proxy; U2 futures panel + MATF factor NAVs from `risk_factors_custom.csv` (confirm 11-vs-12 factor set, D8) + roll-methodology provenance note; U3 production-replica ME/QE panels + classification of the 19 metadata-uncovered columns (D9). *Acceptance:* per-universe data-quality report (coverage, gaps, outlier scan, convention statement), config snapshot committed.

**E2 — Rolling estimation runs.** Cached FCGL fits per universe × config. Runtime budget: U1 dominates (~175 dates × ~30 s × grid ≈ hours per config; parallelise 2–4 workers, cache-first); U2/U3 light. Calibrated M1\*/M2\* slots filled when P1a delivers formulas (E2b, no re-architecture). *Acceptance:* U3 baseline replicates production clusters at 2026-06-30 (pairwise Rand ≥ 0.99); injected == fitted partitions on every date for every smoothed config; runtime/cache report.

**E3 — Stability results + theory validation.** Full metric suite per universe × config; the margin exhibit and predicted-vs-realised churn (proposition test); the frequency-scaling test (U3 ME vs QE natively; U1/U2 re-run at QE estimation dates — churn ∝ √(1−λ^k) prediction); the kurtosis check (U2 fat tails ⇒ higher noise floor ⇒ larger δ\*, cross-universe comparison); risk-model invariance on all three universes. *Acceptance:* every config satisfies the fidelity band or is flagged rejected; theory verdicts stated per prediction. **Owner gate: theory confirmed/amended before the backtest arm is interpreted.**

**E4 — Lineage, labelling, interpretability.** Lineage reports per universe × {baseline, best smoothed}; section 5 metrics; futures beta-bucket threshold review; cross-universe vocabulary exhibit; case studies. *Acceptance:* purity/persistence tables complete; vocabulary table covers ≥ 90% of non-Idio variance in both MATF universes.

**E5 — Momentum backtest arm.** Section 4 in full: score panels, profiler legs, per-universe costs, turnover decomposition, crisis windows, summary tables (one per universe + one cross-universe). *Acceptance:* turnover decomposition sums to total; control ordering sanity (no-cluster and static-taxonomy legs present in every profile); net-of-cost tables with CIs deferred to E6 marked as such.

**E6 — Inference layer (S11 executed).** Stationary block bootstrap over dates (block length ~6 months) for CIs on churn deltas, ARI deltas, Sharpe/IR deltas, turnover deltas; permutation nulls for taxonomy ARI against size-matched random partitions; Greene-style Jaccard-threshold and MONIC-style transition-tracker baselines run on the same panels as lineage comparators. *Acceptance:* every headline delta in E3–E5 carries a CI or a null; baseline-comparator table complete.

**E7 — Exhibit assembly and traceability.** The paper's exhibit list, one row per exhibit: takeaway title, script path, workbook, universe(s). Canonical output workbooks per universe. *Acceptance:* every number destined for the manuscript traces to a named script in this folder (number-traceability gate); a one-page empirical summary per claim family C1–C3.

**E8 — Adversarial pass (owner + Claude).** Hostile-referee review of the empirical set only: weakest exhibit, missing control, alternative explanation per claim family. Output feeds P5 (manuscript) with any patch list for E2–E6.

Dependencies: E0 → E1 → E2 → {E3, E4, E5} (parallelisable once caches exist) → E6 → E7 → E8. The theory track (P1a–P1c) feeds E2b and E3 but blocks nothing before E3's gate.

## 7. What each claim family consumes

- **C1** ← E3 (churn/ARI/VI tables with CIs, margin exhibit, frequency and kurtosis scaling, risk-model invariance) — confirms the constrained-optimisation theory and the calibrated δ\*.
- **C2** ← E5 + E6 (turnover decomposition, net-of-cost tables with CIs, cost-level contrast 10/20/50 bp across universes).
- **C3** ← E4 (+E6 permutation nulls for taxonomy alignment).

## 8. Decisions needed before dispatch (D1–D9)

1. **D1** U1 estimation start: 2012-01 (~175 snapshots) or 2010-01 (~198, heavier).
2. **D2** U2 estimation start: 2002-01 recommended.
3. **D3** confirm U3 stays the exact production configuration (spans {ME:36, QE:12}, 284 snapshots).
4. **D4** momentum score: plain 48w-skip-4w / 12m-skip-1m for the paper, vol-normalised as robustness — or production vol-normalised as primary.
5. **D5** quantile 1/3 (cross-mandate default) or quintile.
6. **D6** U3 QE-sleeve handling in the backtest: QE-rebalanced sleeve within the ME schedule, or ME-only universe for the backtest arm (QE assets excluded from the overlay, clustered only).
7. **D7** U1 return convention: excess returns vs RF (matching FF6) or total returns with Mkt-RF+RF reconstruction — recommend excess, stated once.
8. **D8** confirm the 11-column `risk_factors_custom.csv` is the intended MATF set for the paper (vs the 12-factor `MATF_CUSTOM`), and its citable name.
9. **D9** classify the 19 MAC columns without metadata (benchmarks vs universe members) before E1 freezes the U3 universe.

On sign-off of D1–D9 I convert this draft into the repository roadmap with per-stage executor instructions in the `ROADMAP_cluster_smoothing.md` format (binding constraints, exact estimator configs, acceptance assertions with tolerances) for dispatch to Sol, stage by stage with owner gates.
