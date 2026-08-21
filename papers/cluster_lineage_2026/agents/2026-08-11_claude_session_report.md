# Cluster lineage programme — comprehensive session report

**Date:** 2026-08-11 (single working day)
**Author:** Claude (owner-side review, design, and verification agent), for Artur Sepp
**Executor:** Sol (ChatGPT/Codex), via `roadmap/ROADMAP_cluster_smoothing.md`, stages S1–S10
**Purpose:** the complete record of the discussion and work that produced the cluster-lineage
method, its implementation across factorlasso/optimalportfolios, the empirical evidence set,
and the publication strategy for the Quantitative Finance paper. This file is the starting
context for the `papers/cluster_lineage_2026` manuscript work.

---

## 1. Starting point and code review

The session began as a review of `optimalportfolios/covar_estimation/risk_labelling.py` — the
module that tracks raw risk clusters from the rolling factor covariance estimation
(FACTOR_CLUSTER_GROUP_LASSO in factorlasso) into persistent, named lineages — using the MAC
fund optimisation workbook (`mac_unconstraint_20260811_0620.xlsx`, sheet `cluster_labels`) as
the worked example.

Review verdict: the min-cost-flow matcher construction was verified correct (equal to
brute-force max-weight matching on 30 random instances; preserves 2/2 identities through a
transient merge/re-split where the per-transition Hungarian keeps 1/2). Five defects found and
verified on constructed panels:

1. The gated link test ignored `overlap_band[0]` — membership-disjoint clusters could inherit
   an identity on beta proximity alone.
2. Hungarian bridge revival took the first-inserted dormant track, not the best-weight match.
3. `label_tracks`: `mean() or 0.0` NaN guard was inert (NaN is truthy).
4. `factor_labels`: `order[1]` IndexError for single-factor models.
5. Zero-beta clusters were labelled by `factors[0]` (spurious `Equity-Defensive`); fixed to an
   `'Idio'` sentinel.

Empirical observation that shaped everything after: in the MAC workbook, per-asset label churn
was ~30 changes per asset over 284 months, and every change is a raw-cluster membership
reassignment passing through the labeller — the tracker stabilises cluster identity, not asset
membership.

## 2. Matcher configuration sweep — mac_apac

Production estimation replicated from `inputs_store/mac_20260630/config_snapshot.csv`
(FCGL, spans {ME: 36, QE: 12}, cutoff 0.6, ward/one_minus_rho/pearson, CLARABEL, auto-sign
t=1.0, W-WED factor covar span 52, ME rebalancing) on the exported mac_apac panels (170 ME +
17 QE assets), 284 monthly snapshots 2002-12-31..2026-07-31. Replication check at 2026-06-30
vs the production clusters: pairwise Rand 0.997, modal agreement 97.8%.

41 configurations swept (objective: track-identity churn with coherence guards). Adopted as
module defaults: `overlap_band=(0.15, 0.60)`, `spread_vol_cut=0.015`, `bridge_window=6`,
`bridge_decay=0.5`, `method='mcf'`. Versus the previous effective defaults: total churn flat
(1.709 → 1.713 changes/asset/yr), matcher-attributable churn −6% (0.508 → 0.476), derived
tracks −34% (197 → 131), tracks per asset −22% (18.5 → 14.5), coherence guards intact.
Ablations: Jaccard much worse than the overlap coefficient (churn 2.36, 366 tracks);
inv-vol ≈ equal weighting; Hungarian slightly lower total churn (1.69) but far more
fragmented (266 tracks). Reporting-view (label-string) churn moves slightly AGAINST
consolidation (1.28 → 1.34) because fragmented adjacent tracks share label strings —
track-identity churn is the honest metric.

Structural finding: total churn is set upstream by the clusterer and is flat in every matcher
parameter; the matcher's lever is lineage consolidation, with `bridge_window` the knob that
matters and `bridge_decay ≥ 0.7` pathological (gap edges outcompete consecutive continuations
in the global matching: churn 1.80 at 0.7, 2.17 at 0.9).

## 3. Second universe — S&P 500 constituents

User-provided data: 503 current constituents (survivorship accepted by design), daily
adjusted close 2005–2026, GICS at four levels. Setup: W-WED log returns (2017+ history),
market factor = EW average of constituent weekly returns (no index series in the drop —
swap when SPX provided), FCGL span {W-WED: 156}, cutoff 0.6, auto-sign, ME rebalancing
2021-08-31..2026-07-31 (60 snapshots; median 72 raw clusters/date, range 36–90, median size ~5).

All mac_apac findings replicate: churn flat across configs (3.13–3.21/asset/yr — ~2× mac,
driven by cluster granularity), bridge consolidates (302 → 216 tracks bw1→bw6), decay 0.7
harmful (3.38). The lower-band gate fix matters MOST in the single-factor setting: pre-fix,
beta-only links between disjoint clusters faked consolidation (145 tracks, link overlap 0.906
vs 0.937, ~3.4× candidate edges). External validation: per-date ARI vs GICS peaks at industry
level (median 0.332 > industry group 0.297 > sector 0.203); track modal-industry purity 0.51
weighted; the 36 core tracks (coverage ≥ 0.7) read as clean industries (Utilities,
Diversified Financials, Oil & Gas, Residential vs Specialized REITs, A&D, Insurance).
Churn spikes align with cluster-count jumps (corr 0.46; worst month 39% reassigned,
2022-02-28) — the fraction-of-max tree cut is scale-sensitive.

## 4. Literature positioning (established by search, cited in the module header)

The machinery is standard data association: snapshot-partition event tracking (Greene et al.
2010; MONIC, Spiliopoulou et al. 2006; Asur et al. 2007), global min-cost-flow matching from
multi-object tracking (Zhang, Li & Nevatia, CVPR 2008; muSSP), gap closing from the LAP
tracking framework (Jaqaman et al., Nature Methods 2008). Upstream smoothing descends from
evolutionary clustering (Chakrabarti–Kumar–Tomkins 2006; Chi et al. spectral variants; AFFECT,
Xu–Kliger–Hero) and jump-model persistence penalties (Nystrup et al.). Finance context:
the dynamic-asset-trees / market-network tradition (Onnela et al.; Marti et al. survey).
No published treatment was found of persistent identity + economic labelling for rolling
factor-model risk clusters.

## 5. Cluster smoothing programme (roadmap stages S1–S8)

Motivated by the churn-floor finding and by the signal use-case (cluster-relative
cross-sectional scores churn with peer groups). Four methods specified and executed by Sol:

- Tier-1 (partition-level, sp500): baseline raw churn 2.8695; M0 quarterly-hold 1.189
  (−59%); M1 partition-bonus δ=0.05 → 0.5255 (−82%), δ=0.10 → 0.258 (ARI guard breached),
  δ=0.20 → 0.107 (over-frozen, cluster count collapses); M2 similarity-EWMA λ=0.7 → 2.076.
  Within-cluster momentum rank stability: month-over-month Spearman 0.748 → 0.819 (M1), and
  the reassignment-attributable mean-absolute rank change 0.0427 → 0.0106 (−75%).
- Tier-2 was correctly BLOCKED by Sol: the `precomputed_clusters` hook switched the model to
  GROUP_LASSO, degrading FCGL's cluster-factor penalty — injection would have confounded
  smoothing with an objective change. Verified in source. Two roadmap defects owned by
  Claude: the hook assumption, and an invalid "raw ≥ lineage churn" assertion (S1 data:
  greedy 2.87 < lineage 3.21 — the global matcher optimises weight, not per-asset
  continuity), withdrawn and corrected.
- OWNER DECISION: hook semantics changed to PRESERVE the model type (degradation removed, not
  flag-gated). Consequence, led in the CHANGELOG: non-USD CMA runs with precomputed clusters
  change numerics and become consistent with USD runs.
- S6–S8 delivered: factorlasso `ClusterSmootherType` (NONE/HOLD/PARTITION_BONUS/
  SIMILARITY_EWMA) + four declarative `LassoModel` fields (spec-carried, hence backtestable
  and snapshot-recorded), `external_clusters` on the FCGL/HCGL fit paths preserving penalty
  geometry, stateless `compute_rolling_smoothed_clusters` (state recomputed from history —
  live fits reproduce backtest partitions exactly), two-pass rolling in the estimator.
  Tier-2 through the spec path: M1 δ=0.05 → lineage churn 3.211 → 0.557 (−82.6%), tracks
  216 → 155, matcher churn 0.486 → 0.129, link overlap 0.994, residual-diagonality change
  ≤ 0.01% (guard 5%).

KEY RESULT for the paper: at production regularisation (reg_lambda=1e-5) the fitted
covariance is nearly insensitive to the partition, so cluster stabilisation is effectively
free at the risk-model level — the benefit lands entirely in labels and cluster-relative
signal peer groups. Owner-side verification reproduced the decision numbers exactly
cross-platform (churn 0.5574, 155 tracks; injected == fitted partitions 60/60), with one low
finding: the smoothing pass labels ~6 warmup-marginal assets/date that the fit zeroes
(harmless on injection; fix or document before release).

## 6. Architecture decisions and migration (stages S9–S10)

- OWNER DECISION: replace networkx. The matching problem is unit-capacity max-weight
  bipartite matching; Sol implemented it on SciPy's sparse LAPJVsp with per-left dummy
  columns and a deterministic ordered tie perturbation. Verified: 120-panel brute-force
  oracle; sp500 relabel byte-identical to the networkx result; one exact equal-weight tie on
  mac_apac (2014-06-30) — disclosed, deterministic, objective unchanged. Runtime −26%
  (sp500) / −52% (mac_apac). The `[clustering]` extra is gone; networkx is dev-only.
- OWNER DECISION: move the module into factorlasso as `factorlasso/cluster_lineage.py`
  (canonical `analyze_cluster_lineage` / `run_cluster_lineage_report`; numpydoc; MIT → GPL-3
  relicensing of the single-author module recorded per stack policy A5). Rationale: the
  module consumes only factorlasso types and, post-networkx, only factorlasso's dependency
  surface; smoothing already lives there; the paper reproduces from factorlasso alone.
  optimalportfolios keeps an identity-preserving deprecation shim at the old path (rosaa
  unchanged). `FactorLasso/papers/cluster_lineage/reproduce_sp500.py` assembles the pipeline
  from factorlasso primitives only, asserts optimalportfolios is never imported, and
  reproduces the OP-pipeline baseline to 0.0000% (M1 assembly delta 1.5e-07); excluded from
  sdist/wheel. Owner-side verification: `analyze_cluster_lineage` on independently estimated
  snapshots reproduces the networkx-era relabel byte-identically (4,153 rows, 216 tracks).

Code state at session end: factorlasso 0.14.0 and optimalportfolios 6.16.0 in-tree,
UNPUBLISHED and UNCOMMITTED; all production smoothers remain NONE; adoption of
PARTITION_BONUS δ=0.05 pending owner sign-off (mac_apac transfer recommended first for MAC).

## 7. Labelling mechanics (for the methods section)

Labels are inferred from the estimated model, memberships are not: clusters are cut on the
RESIDUAL correlation (what the factor model fails to explain) and named from the loadings
(what it explains) — "partitioned on the unexplained, labelled by the explained". The
matcher's affinity bridges the two: membership overlap gated by the beta-spread volatility
sqrt((Δβ)' Σ_F (Δβ)).

The label grammar, per track: life-mean beta vector b̄; factor contributions
contrib_j = max(b̄_j (Σ_F b̄)_j, 0) (Euler decomposition of systematic variance); primary =
largest contributor, secondary appended at share ≥ 0.35; qualifier from the beta itself
(equity buckets 0.70/0.30; duration sign for rates; 'short' for negative loadings); vol
regime from sqrt(b̄' Σ_F b̄) at 5%/12% cutoffs. Factor NAMES are inherited from the factor
return panel — the method selects among given semantics.

Three label clocks: within a run a track's label is constant by construction (life
averages); per asset, labels change only through membership reassignment (the churn that
smoothing addresses); across successive runs, near-threshold tracks can flip buckets
(label-vintage property of an offline diagnostic; hysteresis is the remedy if needed).

Multi-universe property: under a shared factor model the label VOCABULARY is common across
universes (same names, units, thresholds) while each universe realises its own label set —
labels form a cross-universe coordinate system (fund book, equity book, futures book
comparable by exposure profile). Caveat: equity-beta buckets were calibrated on a long-only
multi-asset universe; leverage-free futures betas need a per-universe threshold review.

## 8. Publication strategy

OWNER DECISION: one paper, target QUANTITATIVE FINANCE, JFDS as fallback. Positioning:
successor to the QF-native correlation-clustering/market-network tradition — identify,
track, name, and stabilise latent risk clusters inside an estimated factor model.

Agreed contribution tier (stated honestly; no new-core-algorithm claims):
1. Labelling grammar — deterministic naming from the Euler variance-contribution
   decomposition; risk-native automatic cluster labelling with the cross-universe
   coordinate-system property (cleanest quasi-algorithmic contribution).
2. The two-evidence affinity — overlap band arbitrated by the Σ_F-metric beta distance, with
   ablations (Jaccard failure, missing-band failure, cut calibration).
3. The formulation duality — partition on residuals, identity/labels from exposures.
4. Evaluation methodology — matcher-vs-clusterer churn attribution (cohort retention) and
   the reassignment-attribution gap for cross-sectional signal ranks.
5. Penalty-preserving smoothing placement + the covariance-invariance finding (what cluster
   smoothing is FOR in risk pipelines).
Plus algorithm-design cautions: bridge-decay pathology in global matching on slow panels;
overlap coefficient over Jaccard under asymmetric splits.

QF gap-closing programme (next roadmap stages, pending dispatch):
- S11 — inference layer (block-bootstrap CIs over dates; permutation nulls for ARI vs
  size-matched random partitions; CIs on smoothing deltas) + named literature baselines
  (Greene-style Jaccard threshold; MONIC-style transition tracker). Ready to write.
- S12 — economic-payoff arm; proposed design AWAITING OWNER CONFIRMATION: sp500, 48-week
  momentum skipping 4, within-cluster percentile scores baseline vs δ=0.05, rank-weighted
  long-short quintiles, monthly, qis backtest at 10 bp costs; the claim is implementability
  (turnover, net-of-cost IR), not alpha.
- S13 — global futures third universe, AWAITING OWNER DATA (prices, sector metadata, factor
  mapping); membership-stable, hence survivorship-free; headline exhibit for the shared
  label vocabulary.
- Manuscript in the Publications project; reframed as identification/labelling of latent
  structure; look-ahead stated as an estimation choice with the live variant described.
- Submission numbers must come from ONE canonical environment:
  `FactorLasso/papers/cluster_lineage/` is that place (the mac_apac third-decimal drift
  between environments — 1.7131 vs 1.7091 via a disclosed tie and cache regeneration — is
  the cautionary example).

## 9. Decision log (owner decisions, all 2026-08-11)

1. Sweep objective: track-identity churn with resolution/coherence guards; labelling layer
   only (estimation pinned at production config).
2. Adopt swept matcher defaults into `analyze_risk_clusters`.
3. Land fixes + defaults + docs + runner in the repos (stage instructions 1–3).
4. Hook semantics: preserve model type; remove GROUP_LASSO degradation outright.
5. Replace networkx with a scipy-only matcher; drop the `[clustering]` extra.
6. Move the lineage module into factorlasso (`cluster_lineage`); accept MIT → GPL-3 for the
   moved single-author module; keep an OP deprecation shim.
7. Publication: single paper targeted at Quantitative Finance, JFDS fallback.
8. Pending: δ=0.05 production adoption; S12 design confirmation; futures data for S13;
   releases and commits.

## 10. Artifact index

Workbooks (in `analytics/outputs/`): `mac_apac_risk_label_sweep_20260811.xlsx`,
`sp500_risk_label_stability_20260811.xlsx`, `sp500_cluster_smoothing_sweep_20260811.xlsx`
(tier1_grid, tier2_grid, m3_confidence), `mac_apac_risk_label_covar_data.pkl`,
`risk_labelling_fixes.patch`. Repo documentation: implementation notes
`2026-08-11_mac_apac_risk_label_config_sweep.md` and `2026-08-11_cluster_smoothing_sweep.md`
(+ INDEX rows); runners `rosaa/research/analysis/risk_label_config_sweep.py` and
`rosaa/research/cluster_smoothing/`; roadmap `roadmap/ROADMAP_cluster_smoothing.md` with
stage reports S1–S10 and `CLUSTER_SMOOTHING_OWNER_REVIEW.md` (both owner reviews);
reproduction `FactorLasso/papers/cluster_lineage/`. Cross-session context:
project doc `claude/RISK_LABELLING_CONFIG_SWEEP.md` in the OSS Quant Stack project.
