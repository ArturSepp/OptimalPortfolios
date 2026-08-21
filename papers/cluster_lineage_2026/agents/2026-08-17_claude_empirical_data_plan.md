# Empirical data plan: theory confirmation vs investment applications

**Date:** 2026-08-17
**Author:** Claude (owner-side)
**Status:** for owner sign-off. This plan refines Phase R and the empirical half of Phase M in `2026-08-17_claude_paper_status_and_next_steps.md` (rev 2). After sign-off, Part A becomes the Sol execution roadmap; Part B needs only the bootstrap layer. Nothing here is dispatched yet.

## 0 Design principle (owner ruling, 2026-08-17)

The empirical evidence splits into two parts with different epistemic roles and different freeze rules:

- **Part A — theory confirmation.** Purpose: confirm or falsify the distribution theory (P1–P7 and the flip approximation). Runs on the cached estimation panels of the E-programme. Permitted computation: consolidation and re-scoring of cached snapshots, the P1b simulation (the one genuinely new compute), and bootstrap. No refits of estimators on data.
- **Part B — application to investing strategies.** Purpose: show that stabilised clusters improve the risk properties of cluster-scored signals and cluster-budgeted allocations. **FIXED**: elected universes, signals, and constructions are final; every number stands as recorded in the 2026-08-16/17 summaries. The only addition is CI/bootstrap analysis.

### Panel-vs-universe naming (decision point 1)

The theory panels and the application universes overlap but are not identical, and the manuscript must not blur them:

| Part A estimation panels (cached, E-series) | Part B investment universes (final) |
|---|---|
| MSCI US equities, FF6, W-WED/156 (= U1 asset panel) | U1 equities (ME/36 clustering cell, classic momentum) |
| Global futures, MATF, W-WED/156 (= U3 asset panel) | U2 BlackRock funds (W-THU/156, AUM50, 55/35/10) |
| MAC multi-asset funds, MATF, ME/36 + QE/12 (production replica) | U3 futures (M1-star δ 0.0691, ROSAA short-3, vol-normalised) |

The MAC panel appears only in Part A (it carries the native ME/QE frequency-scaling test); the BlackRock panel appears only in Part B. Proposal to protect the notation budget: reserve U1/U2/U3 for the Part B investment universes and name the Part A panels in words ("the equity panel", "the futures panel", "the multi-asset fund panel"), stating the asset-level overlap once. Owner to confirm.

## Part A — theory confirmation (Sol roadmap to be drafted from this table)

| Prediction | Test statistic | Existing source (cached) | Gap → Sol action | New compute |
|---|---|---|---|---|
| P1 churn at small margins | margin histograms; flip rate by margin decile; predicted-vs-realised churn correlation across configs (0.863/0.872 on equities) | E3/E3b workbooks; metric 6 panels | consolidate into scorecard; flip-rate-by-decile exhibit; block-bootstrap CI on the correlation | re-scoring only |
| P2 frontier knee | churn-fidelity frontier over δ grid {0, 0.02, 0.05, 0.10, δ*}; knee vs overlaid δ*_lvl, δ*_inn | E3 config grid + fidelity verdicts | frontier exhibit per panel with calibration overlays recomputed from measured κ̂, ρ̄ | none |
| P3 kurtosis ordering | cross-panel churn vs √(1+κ̂) with per-panel constant c absorbed (c ∈ [0.81, 2.15]) | E3b constant table; E1 κ̂ | scorecard row restated in the absorbed-constant form | none |
| P4 frequency scaling (**REVISED**, ruling 10) | restate the hysteresis component with the per-panel proportionality constant absorbed; re-evaluate ME→QE scaling on the cached re-scored panels against the revised prediction | E3b scaling tables (ME/QE re-scoring already cached) | recompute the revised prediction analysis-side; record the revised verdict; the manuscript reports the original equality rejection AND the revised form | analysis only |
| P5 risk-model invariance | relative Frobenius, max entry change, residual diagonality (≤ 0.014 measured) | E3 invariance tables | scorecard row | none |
| P6 ergodicity | subsample means of cluster count, size entropy, consecutive ARI/VI vs bootstrap bands; crisis windows separate | E3 per-date panels | scorecard row + bands | bootstrap only |
| P7 turnover attribution | reassignment turnover monotone in smoothing strength; signal turnover invariant; net performance non-decreasing in band | E5/E5b metric-11 decompositions; E6 CIs for E-series configs | scorecard row; reuse E6 CIs where the config matches, else one short bootstrap | bootstrap only |
| Flip approximation (P1b) | simulated flip probabilities vs Φ(−(m+δ)/(√2σ)) — flat cut confirmed, Ward verified | none | **the one new compute**: synthetic G-block + GARCH(1,1) panels; grid over separation, span N, step k, δ; seeds recorded; produces the flip-approximation figure and the Ward-verification table | simulation |
| Calibration bridge | per-panel table: N, k, κ̂, ρ̄ → δ*_lvl, δ*_inn; adopted δ (0.0866 equity cell, 0.0691 futures cell); sweep knee location | E1/E2 caches; sweep outputs | the bridge exhibit between theory and applications; replaces the draft's stale worked example (0.060/0.020/0.05) | none |
| Membership and interpretability analysis (ruling 4) | taxonomy-ARI by level (peak location), track purity/persistence, label churn, case-study tracks | E4 outputs | fold into Part A as the descriptive membership analysis of smoothing; no separate claim family | none |

Notes. (i) Equity-panel smoothed results use the E3b-corrected caches; the original E3 U1 smoothed rows are superseded and must not be cited. (ii) The consolidated deliverable is ONE theory-scorecard table (P1–P7, statistic, verdict, CI where applicable) plus four exhibits: margin/flip-rate figure, frontier-knee figure with calibration overlays, revised frequency-scaling table, simulation figure. (iii) All Part A work respects the standing protocol: cache-first, deterministic replay, dated stage reports in `agents/`.

## Part B — applications (FIXED; CI/bootstrap only)

**Freeze statement.** The elected specifications, eligibility rules, and all recorded performance numbers are final: signal tables per the 2026-08-17 pipeline summary, risk tables per the 2026-08-16/17 summaries on their recorded samples (risk: AUM100 funds universe, seven-exclusion futures universe — vintages disclosed in captions, not reconciled, per ruling 5). No refits, no reruns, no spec changes, no dePC1, no split-window.

**Bootstrap layer (keep short, ruling 7).** Joint moving-block bootstrap on the aligned per-period net-return series of each leg pair (block length 6, 2,000 draws, seed 20260813), preserving cross-leg correlation. Report 95% CIs on three deltas per comparison: annualised net return, volatility, RF=0 Sharpe. Two output tables only:

| Table | Comparisons |
|---|---|
| Signal CIs | U1 cluster − global; U1 cluster − BICS; U2 cluster − global; U3 cluster − global |
| Risk CIs | U1 Rolling-Ward HRP − flat ERC; U1 Rolling-Ward HRP − canonical single-HRP; U3 equal-cluster RB − flat ERC |

The U2 risk limitation stays narrative (concentration statistics are structural, not sampling questions); no CI row. The risk-concentration mechanism table (effective clusters, largest share) also stays descriptive.

**Robustness exhibits** come from existing grids only, labelled as sensitivity material with their selection role disclosed: U2 AUM cutoffs (none/50/100), U1 min-cluster-size grid, U3 short-span sweep, covariance frequency/span grids. No CIs on grids.

## Mapping to the manuscript skeleton

| Manuscript section | Fed by |
|---|---|
| 3 Distribution theory | Part A scorecard + P4 revision + simulation figure |
| 4 Noise-calibrated smoothing, membership analysis | Part A frontier/calibration bridge + membership/interpretability analysis |
| 6 Signal evidence | Part B signal tables + signal CI table |
| 7 Risk-allocation evidence | Part B risk tables + risk CI table + mechanism table |
| 8 Robustness | Part B grids (sensitivity-labelled) |
| 9 Limitations | honesty ledger (sample vintages, survivorship, frozen-partition caveat, band status) |

## Decision points before the Sol roadmap is drafted

1. **Panel naming**: reserve U1/U2/U3 for Part B and name Part A panels in words (recommended), or extend U-labels to the theory panels.
2. **BlackRock panel in Part A**: default is NO (theory panels stay the three E-series panels; the W-THU fund panel never went through the stability metric suite). A descriptive churn/margin row from its existing caches would be cheap but adds a fourth panel — recommend against.
3. **P7 CI source**: reuse E6 outputs where the config matches, one short bootstrap otherwise (recommended default).
4. **Simulation design**: grid and seeds will be specified in the Sol roadmap for approval within the dispatch, not separately.

On sign-off, the Sol roadmap covers: Part A rows (consolidation, P4 revision, P1b simulation), the Part B bootstrap layer, and the Phase C items already ruled (source reconstruction, exhibit index, first commit + tag, replication statement).
