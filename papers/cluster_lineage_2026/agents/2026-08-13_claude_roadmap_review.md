# Roadmap review: cluster-lineage paper for Quantitative Finance

**Date:** 2026-08-13
**Author:** Claude (research assistant), for Artur Sepp
**Reviewed:** `2026-08-11_paper_and_research_roadmap.docx` against the 2026-08-11 session report, the project knowledge file (v1.0), the `papers/cluster_lineage_2026` replication code and data, `factorlasso/cluster_smoothing.py`, `factorlasso/cluster_lineage.py`, and `rosaa/data/factors/matf`.

---

## 1. State verified before this review

What I checked, so the comments below rest on the actual trees rather than recalled state.

**Code.** The replication folder `OptimalPortfolios/papers/cluster_lineage_2026/replication/` carries the S&P 500 baseline harness (60 ME snapshots 2021-08..2026-07, FCGL span-156 W-WED, ward/one_minus_rho/0.6, baseline lineage churn asserted at 3.2115), the four-method smoothing grid (baseline, M0 hold, M1 partition bonus, M2 similarity EWMA), and the networkx-vs-LAPJVsp matcher validation. `factorlasso 0.14.0` carries the canonical `cluster_smoothing` (four `ClusterSmootherType` modes, causal, with an exact O(TN²) Pearson EWMA recursion) and `cluster_lineage` (fingerprint, gated two-evidence affinity, MCF path cover, labelling grammar, offline-diagnostic boundary stated in the header).

**Releases (resolves knowledge-file ruling 3).** PyPI now carries factorlasso 0.14.0 and optimalportfolios 6.17.0, one past the 6.16.0 recorded as pending. Both trees shipped. The knowledge file v1.0 should be updated on this point.

**Data in `papers/cluster_lineage_2026/data/`.** All three universes are present and aligned by column:

| universe | file(s) | shape | span | metadata |
|---|---|---|---|---|
| MAC funds | `mac_log_returns_ME/QE.csv` | 332×170 ME + 107×17 QE | 1998-12..2026-07 | 168 rows; **15 ME + 4 QE columns have no metadata row** (benchmark-type indices: LUATTRUU, NDDUUS, HFRXGL, ...) |
| MSCI US | `msci_us_log_returns.csv` + inclusion indicators | 1,358 securities, daily | 2006-08..2026-07 constituent span | full GICS 4-level; point-in-time membership matrix present |
| Futures | `futures_log_returns.csv` | 17,508×95 daily | 1959-07..2026-08 | 95 rows, fully matched; 7 asset classes (29 equities, 20 ags, 16 bonds, 11 FX, 8 energy, 6 metals, 5 STIR) |
| Factors | `risk_factors_custom.csv` | 7,204×11 daily | 1998-12..2026-08 | Equity, Rates, Credit, Credit EM, Carry G10, Carry EM, Inflation, Commodities, Private Equity, Rates Vol, Fx |

Two data checks to close: (a) whether the 19 uncovered MAC columns are intentional (benchmark series, not universe members); (b) `risk_factors_custom.csv` carries 11 factors while `rosaa/data/factors/matf` documents `MATF_CUSTOM` as "the production twelve" — confirm which model the CSV is, so the paper names its factor model correctly. Also note the docx pointer `rosaa/data/matf` resolves to `rosaa/data/factors/matf`.

---

## 2. Comments on the outline, section by section

### 2.1 Motivation (outline section 1)

The four motivating points are right, and the third one (churn increases turnover) is where the paper's arithmetic crux lives. My suggestion for the opening quantitative hook, computable from the existing baseline: between consecutive monthly estimation dates, a span-156 weekly EWMA replaces only about 5% of its weight with new data (1 − λ^4.33 with λ = 1 − 2/157), yet the baseline reassigns roughly 24% of assets per month (raw churn 2.87/asset/yr). A 5% information innovation producing a 24% membership reassignment is the cleanest possible statement that the dendrogram cut amplifies estimation noise, and it motivates both smoothing and lineage in one sentence. I recommend Artur draft this mechanism paragraph himself per the drafting-order rule, but the numbers are ready.

Two wording items: "HCRP" should be HRP (López de Prado 2016) and HERC (Raffinot 2017), and both "check it" claims need narrowing (section 4 below gives the prior-art status). "Never been studied" will not survive a QF referee. "Has not been studied as temporal identity inside a rolling factor-model estimation, with the turnover consequence quantified" will.

### 2.2 Contribution (outline section 2) and the boldness question

The docx asks for a bold claim, "ideally an original smoothing/labelling algo that bears my name specifically". Three constraints push back, and I think they push toward a better version of the same ambition:

1. **Eponymy is conferred, not claimed.** No field names a method after its author at submission. What you control is a memorable, consistently reused object name plus a reference implementation carrying that name. The project instructions say exactly this (named-objects section), and the vehicle already exists: `factorlasso.cluster_lineage`. My recommendation for the named object is **cluster lineage** itself (or "risk-cluster lineage"): it is the smallest reusable unit a stranger can lift, the module already carries the name, and it can be reused across your papers for a decade. One name, one paper, per the notation budget.
2. **The 2026-08-11 owner decision already fixed the honest contribution tier** ("no new-core-algorithm claims"): the machinery is standard data association, and the novelty is the fingerprint, the Σ_F-metric affinity, the labelling grammar, the penalty-preserving smoothing placement, and the covariance-invariance finding. The docx's bolder framing is in tension with that decision. I do not think you need to reopen it, because:
3. **There is a genuinely original piece available that the docx gestures at but does not name** — the span-linked calibration of the smoother (section 3.1 below). If it works out, it converts M1's δ from a swept constant into a derived quantity, which is a claim no adjacent literature makes and which carries your intellectual signature (it is the same noise-floor logic as the Paper A gate: act only on changes that clear the estimator's sampling noise). That is the boldness the paper can afford.

Also: the contribution items are currently phrased as questions ("could we...?"). For the manuscript they must become declarative contributions with numbers, per house style. Fine for a working note, but the conversion should happen at outline-freeze.

### 2.3 Method section (outline section 3)

Ten pages is right if the section is organised as one architecture rather than a list of techniques. Suggested skeleton:

1. **Setup**: rolling FCGL estimation, what a partition is, the three-layer separation (estimation → causal smoothing → offline lineage/labelling). The layer boundary is a real design principle: smoothing is causal and backtestable, lineage is a full-panel offline diagnostic. State this once, early, and the look-ahead question referees always raise is preempted.
2. **Temporal smoothing as input regularisation.** Present M0/M1/M2 as one family: each modifies the *input* to an unchanged Ward cut (schedule, distances, similarity), never the clustering algorithm or the penalty geometry. M1 is a soft must-link prior in the constrained-clustering sense; M2 is AFFECT-style similarity smoothing. Say so, and the honesty costs nothing because the calibration theory (3.1) is yours.
3. **Calibration theory** (the new material, section 3.1 below).
4. **Lineage**: fingerprints, two-evidence affinity, MCF path cover, with the ablation evidence (Jaccard failure, band failure, bridge-decay pathology) as design cautions.
5. **Labelling grammar as Algorithm 2** (section 3.3 below).

### 2.4 Empirical section (outline section 4) — see section 5 below for per-universe detail.

### 2.5 What the outline is missing

Five absences, all cheap to fix at outline time and expensive later:

- **The central claim is not stated.** One paper, one claim. My candidate: *stable, economically labelled risk clusters are obtainable inside a rolling factor-model estimation at zero cost to the risk model* — the covariance-invariance finding is the surprising result that carries it, and every section (smoothing, lineage, labelling, payoff arm) supports it.
- **The inference layer (S11)** — block-bootstrap CIs, permutation nulls for ARI, Greene/MONIC baselines — does not appear in the outline. QF will require it; it belongs in the empirical section, not an appendix.
- **The look-ahead statement** for lineage labels (offline diagnostic, live variant described) — agreed on 2026-08-11, absent from the outline.
- **Canonicalization vehicle declared at outline time** (project instructions): name `factorlasso.cluster_lineage` + `cluster_smoothing` as the reference implementation in the outline, not after acceptance.
- **A no-cluster control** in the payoff arm (section 5.4): without it, a referee asks "why cluster at all?" before asking "why smooth?".

---

## 3. The three methodology ideas, scrutinised

### 3.1 "Cluster smoothing must be linked to the span of the covariance estimator" — yes, and here is the concrete route

This is the best idea in the docx. Two derivable links, one per smoother:

**M1 (partition bonus): δ as the noise floor of the distance estimator.** For an EWMA with span s, the effective sample size is n_eff = (1+λ)/(1−λ) ≈ s. The standard error of an estimated correlation is approximately (1−ρ²)/√n_eff, and d = 1−ρ inherits it. At s = 156 and within-cluster ρ ≈ 0.5–0.7, SE(d̂) ≈ 0.04–0.06. **The swept optimum δ = 0.05 is one standard error of the estimated distance.** That is either a pleasant coincidence or the paper's central design principle: *retain a co-membership unless the data moves the pair's distance by more than its own sampling noise*, i.e. δ(s, ρ) = z·(1−ρ²)/√s with z ≈ 1. This (i) explains why δ = 0.10 breached the ARI guard (two standard errors over-freezes), (ii) predicts how δ must scale when the span or frequency changes (testable across the three universes at their different spans — a strong out-of-sample check of the theory, not just of the number), and (iii) is the same statistical logic as the Paper A noise-floor gate, which gives your two papers one recognisable signature. I recommend a short proposition plus a simulation panel (known partition, EWMA sampling noise, show δ = 1·SE maximises retention without absorbing true breaks).

**M2 (similarity EWMA): λ_s as span extension.** Two nested EWMAs compose: monthly smoothing with weight λ_s on top of a weekly span-s estimator yields a kernel whose effective memory is the estimator span plus roughly Δ·λ_s/(1−λ_s) where Δ is the rebalancing step. So M2 is equivalent to a longer-span estimator *for the clustering input only*, while the fitted covariance keeps its span. That framing answers the natural referee question "why not just lengthen the span?" — because lengthening the span degrades covariance responsiveness, whereas smoothing only the clustering input decouples the two time scales. Derivable in half a page.

One caution: these derivations make δ and λ_s functions of (span, frequency). The empirical section should then report the *calibrated* smoothers alongside the swept grid, and the cross-universe consistency of the calibration is itself an exhibit.

### 3.2 "Adjust by the first PC and cluster the covariance of residuals" — this exists, and the honest version is stronger

Removing the market mode before clustering is **detoning** (López de Prado 2020, Machine Learning for Asset Managers) with roots in the random-matrix cleaning tradition (Laloux et al. 1999; Plerou et al. 2002): the market eigenvector compresses all pairwise correlations toward one level and masks group structure. So the idea cannot be claimed as new. But your pipeline supports a generalisation that is worth studying and fits the paper's existing narrative: **cluster the factor-model residual correlation** (remove the estimated MATF/FF exposures rather than PC1). PC1-detoning is the one-statistical-factor special case; the FCGL setting makes the removed component economic and multi-factor. This also aligns the code with the `cluster_lineage` header's own description ("partitioned on the unexplained, labelled by the explained") — note that the current replication clusters the *raw* demeaned return correlation, so the header's claim and the code presently disagree, and this arm would resolve the disagreement in the header's favour.

The empirical question is exactly the right one: is the residual partition more stable? Plausibly yes cross-sectionally (no market-level compression of the distance distribution, so the cut is better separated) but not automatically over time (residual correlations are smaller, so relative sampling noise is larger; the noise-floor calculus of 3.1 applies with lower ρ and predicts *more* need for smoothing, not less — a testable prediction). Design it as a 2×2: {raw, residual} × {unsmoothed, calibrated M1}, on churn, ARI vs GICS/asset-class, and cut separation.

Two cautions. First, changing the clustered object changes FCGL numerics, so this is a research arm on the paper harness only, never a silent production change (checkpoint rule). Second, this arm is the one place the outline risks scope creep: if it grows, it is a second paper. I would timebox it and let the results decide whether it enters as a subsection or a follow-up.

### 3.3 "Can labelling be presented as an actual algo?" — yes, it already is one

The grammar is fully deterministic: Euler decomposition contrib_j = max(b̄_j(Σ_F b̄)_j, 0), primary/secondary at the 0.35 share, beta-bucket and vol-regime qualifiers, 'Idio' sentinel. Write it as **Algorithm 2** (pseudocode, ~15 lines) with its three stated properties: determinism given the factor panel, the three label clocks, and the cross-universe vocabulary property under a shared factor model. That last property is the exhibit-worthy one, and it is only demonstrable if at least two universes share a factor model — which argues for MATF on both MAC and futures (section 5.3). One honest limitation to carry from the session report: equity-beta bucket thresholds were calibrated long-only multi-asset; futures need a per-universe threshold review, which the paper can present as the vocabulary's one universe-specific dial.

A further method item worth one paragraph of research: the churn floor is set upstream by the clusterer, and the session report already located a mechanism (the fraction-of-max cut is scale-sensitive; churn spikes co-move with cluster-count jumps at corr 0.46). A cut normalisation that does not depend on the single deepest merge (fixed-quantile of merge heights, or median-height normalisation) attacks the floor that no matcher parameter can touch. Cheap to test on the existing 60-snapshot panel, and it would make the method section's smoothing story two-sided: robustify the cut, then smooth what remains.

---

## 4. Prior-art status of the two "check it" claims

Per the prior-art gate, the five nearest results for each claim, from the session's own literature search plus my additions.

**"Stability of clusters has never been studied for portfolio optimisation."** Nearest: (1) evolutionary clustering, Chakrabarti–Kumar–Tomkins 2006 — temporal smoothness as an explicit objective, no finance; (2) AFFECT, Xu–Kliger–Hero 2014 — adaptive similarity smoothing, the direct ancestor of M2; (3) dynamic asset trees, Onnela et al. 2003–04 — temporal survival of correlation-cluster structure in equities, descriptive, no portfolio consequence; (4) Marti et al. — bootstrap stability of financial time-series clustering ("how long is enough?", 2015; survey 2017/2021) — cross-sectional stability under resampling, not temporal identity; (5) Nystrup et al. jump models — persistence penalties for temporal *regime* clustering used in allocation, the closest in spirit on the smoothing side. Add HRP/HERC robustness discussions as the motivation anchor. Honest gap statement: no published treatment tracks *asset-cluster identity through a rolling estimation and prices its instability in turnover*. That claim survived the 2026-08-11 search and I found nothing against it, subject to a fresh sweep at submission.

**"Cluster interpretability — nothing available for finance."** Nearest: sector/industry taxonomies as exogenous labels (GICS itself), the market-network literature's post-hoc naming of communities by inspection, and general interpretable-ML clustering. Automatic, risk-native labelling from estimated factor exposures with a stable vocabulary: nothing found. This one is safe if phrased as "automatic economic labelling of statistical risk clusters", not "interpretability" broadly.

---

## 5. Empirical design comments, per universe

### 5.1 MAC funds (ME + QE)

The production-anchored universe and the practice case. Pin the frequency at ME (production replication exists: Rand 0.997 vs production clusters). Two flags. First, **licensing**: the data folder ships Bloomberg index returns and LGT fund identifiers; the public replication bundle almost certainly cannot carry them. The paper should state data availability honestly (code public, MAC inputs proprietary) and lean on MSCI US + futures for reproduction. This needs an explicit decision before the repository structure hardens. Second, the 19 metadata-uncovered columns (section 1) need classifying as benchmarks or omissions.

### 5.2 MSCI US (the equity universe)

Switching the equity arm from current-constituent S&P 500 to MSCI US with inclusion indicators is a major upgrade: it converts survivorship from an accepted defect into a solved one, and referees will notice. Three design notes:

- **Point-in-time membership interacts with churn metrics.** Entering/exiting assets create mechanical reassignment that is not the clusterer's fault. Metrics must condition on continuing members (the pairwise churn already does), and the universe-size drift reactivates the scale-sensitive-cut issue — another reason to include the cut-robustification arm.
- **FF factors at W-WED**: Ken French publishes daily factors; compound daily to W-WED and build factor NAVs via `qis.returns_to_nav` (the harness consumes factor prices). Decide FF5+MOM vs FF3 once; W-WED is the natural frequency, ME only if you want to mirror the MAC pipeline.
- Swap the equal-weight market proxy for the actual MSCI US index series (`msci_us_index_total_return_timeseries.csv` is already in the tree).

### 5.3 Futures (the global universe)

Recommend **MATF as the factor model**, not equity-regional: MAC and futures then share the factor panel, which is the only way to show the cross-universe label-vocabulary property (the headline exhibit of the labelling contribution). The 95-contract, 7-asset-class universe maps naturally onto Equity/Rates/Commodities/FX/Carry.

On "frequency can be 'B'": for a global futures universe, daily returns carry an asynchronous-close bias (US, Europe, Asia close hours apart), which mechanically shrinks cross-region correlations and will distort clusters toward regional blocks for the wrong reason. W-WED substantially removes it. I would fix W-WED as the primary frequency for futures and MSCI US, treat 'B' as a robustness note at most (with the asynchronicity caveat stated), and note that the noise-floor calibration of 3.1 predicts how δ must move with frequency — which turns the frequency comparison from a chore into a test of the theory.

Data provenance to state for the paper: roll methodology behind `futures_log_returns.csv` (back-adjusted front contracts?), and the same Bloomberg-licensing question as MAC.

### 5.4 The payoff arm (cross-mandate-style portfolios)

The docx design (per-universe clustering → within-cluster momentum and low-beta ranks → top-quantile long-only portfolio → compare clustering methods on performance and turnover) is close to the S12 proposal but long-only rather than long-short. Comments:

- **Keep the claim implementability, not alpha** (as agreed for S12). A long-only top-quantile portfolio is dominated by market beta, so headline Sharpe differences across clustering configs will be noise. The honest exhibits are turnover, cost drag at 10 bp, net-of-cost IR vs the universe benchmark, and the decomposition of turnover into signal-driven vs reassignment-driven trades (the reassignment-attribution machinery already exists).
- **Fix the signal, vary only the partition.** Same momentum window (48-4 as specified for S12), same quantile, same rebalancing; then every performance delta is attributable to clustering.
- **Include two controls**: no-cluster plain cross-sectional ranks (justifies clustering at all), and a static-taxonomy partition (GICS for equities, asset class for futures — justifies *estimated* clusters vs the free alternative a practitioner already has). These two controls are what make the exhibit persuasive to both referees and PMs.
- **Momentum primary, low-beta secondary.** The ROSAA experience (BAB's intentional negative return-IC) says low-beta ranks muddy return attribution in a long-only setting; run it, but as a robustness table.
- Backtests via `qis.backtest_model_portfolio` at 10 bp with `weight_implementation_lag=1`, TE/IR via the sanctioned qis estimators, per project conventions.

For MAC specifically, this arm doubles as the MATF-CMA application shown in the docx (custom risk model), which keeps the production face of the method visible without depending on unpublished companion work.

---

## 6. Consistency flags (process, not science)

1. **Canonical environment conflict.** Decision 2026-08-11: submission numbers come from ONE canonical place, named as `FactorLasso/papers/cluster_lineage/`. The docx and the new tree put replication in `OptimalPortfolios/papers/cluster_lineage_2026/`, and its code imports optimalportfolios, qis, and rosaa (`local_path`, cache dirs). Both cannot be canonical. My read: the OP folder is becoming the *research* home (it needs the full stack and private data) and the FactorLasso folder remains the *public reproduction* (factorlasso-only, asserts OP is never imported). That split is fine, but it should be declared, and the paper's numbers must then come from the OP folder with the FactorLasso script reproducing the public subset. Needs a ruling.
2. **Parallel-copy drift.** `replication/methods.py` re-implements the M0/M1/M2 logic that now lives canonically in `factorlasso.cluster_smoothing` (it predates S6–S8 landing). The Tier-2 runner already goes through the package. The Tier-1 generators should either call the package or be explicitly labelled as the independent cross-check implementation (which has value — but then say so in the module header). Unlabelled duplication is the exact drift pattern the project instructions warn about.
3. **rosaa imports in replication code** (`rosaa.local_path` in `methods.py`, `run_sweep.py`, `sp500_baseline.py`) tie the paper harness to the private repo. Fine for the research home, must be stripped from anything public.
4. **Knowledge file update due**: ruling 3 resolved (0.14.0 and 6.17.0 on PyPI); the S12 design is now amended by this docx (long-only cross-mandate variant); the third-universe data (S13) has arrived. I can draft v1.1 on request.
5. `run_cross_mandate_analysis.py` (the template named in the docx) sits in rosaa; the paper version needs a stack-only reimplementation for the two public universes.

---

## 7. Proposed roadmap

Stages ordered so that owner decisions unblock agent work. Executor suggestions in brackets; every stage ends at an owner gate per the operating regime.

**P0 — Rulings (owner, this week).** (1) Canonical-environment split per flag 6.1. (2) Data licensing for the public bundle (MAC, futures). (3) Factor model per universe: MATF for MAC+futures, FF for MSCI US — confirm. (4) Frequencies: ME (MAC), W-WED (MSCI US, futures) — confirm. (5) Named object: adopt "cluster lineage" as the paper's name. (6) Confirm the payoff-arm design of 5.4 (supersedes S12's long-short quintiles). (7) Scope ruling on the detoning/residual arm: in-paper subsection with a timebox, or follow-up paper.

**P1 — Method formalisation (Claude, ~days).** Draft the two calibration propositions (δ = z·SE(d̂); λ_s span composition) with simulation validation on synthetic partitions. Draft Algorithm 1 (lineage) and Algorithm 2 (labelling) in paper notation. Specify the cut-robustification experiment. Output: a methods note the manuscript can absorb, plus runnable experiment specs.

**P2 — Data infrastructure (Sol, parallelisable with P1).** FF daily → W-WED factor NAVs; MSCI US point-in-time harness (inclusion-indicator masking, index-series market factor); futures universe prep (roll provenance documented, MATF mapping, per-universe beta-bucket threshold review); close the two data checks from section 1.

**P3 — Stability core (Sol executes, Claude verifies).** Baseline + M0/M1/M2 at swept *and calibrated* parameters, on all three universes at their pinned frequencies. Metrics: raw/lineage/matcher churn, ARI vs GICS and asset class, rank-stability, residual diagonality guard. Plus S11 inference: block-bootstrap CIs over dates, permutation nulls for ARI against size-matched random partitions, Greene-style and MONIC-style baselines. Plus the cut-robustification arm and (if ruled in) the 2×2 raw-vs-residual arm.

**P4 — Payoff arm (Sol executes, Claude verifies).** The 5.4 design: three universes, fixed signal, {no-cluster, static-taxonomy, baseline, calibrated-M1} partitions, qis backtests at 10 bp, turnover attribution. Claim: implementability.

**P5 — Manuscript (Artur leads per drafting order).** Outline freeze with central claim, contribution tier, canonicalization vehicle, and exhibit list declared. Artur writes topic sentences, crux sentences, and mechanism paragraphs; Claude expands, then fingerprint + AI-tell + copyedit passes per project instructions. Number traceability: every exhibit names its script in the canonical environment.

**P6 — Canonicalization (after P3 locks numbers).** Public reproduction script(s) in the FactorLasso paper folder for MSCI US + futures; replication README with data availability statement; knowledge file v1.1; consider whether the calibrated-δ default enters `factorlasso` as a documented option (additive, checkpoint rule applies).

The critical path is P0 → P2 → P3; P1 is parallel and feeds both P3 (calibrated parameters) and P5. Nothing in P3/P4 should start before the P0 rulings on environment, licensing, and factor models, because those choices shape every cached artifact.

---

## 8. Open questions put to the owner (compact)

1. Canonical environment: OP folder for research numbers + FactorLasso folder for public reproduction — approve this split?
2. May the public bundle carry MAC and futures returns, or code-plus-availability-statement only?
3. MATF for futures (shared-vocabulary exhibit) — yes?
4. Frequencies pinned at ME / W-WED / W-WED — yes, with 'B' as robustness-only?
5. "Cluster lineage" as the single named object — yes?
6. Payoff arm: long-only top-quantile with the two controls (no-cluster, static taxonomy), momentum primary — approve as the S12 replacement?
7. Residual/detoned clustering: in-paper timeboxed subsection or follow-up paper?
8. Update the project knowledge file to v1.1 with the resolved ruling 3 and the amended S12 — shall I draft it?
