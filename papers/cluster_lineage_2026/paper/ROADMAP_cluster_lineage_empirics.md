# ROADMAP_cluster_lineage_empirics — three-universe empirical validation for the cluster-lineage paper

**Date:** 2026-08-13
**Repository:** OptimalPortfolios, `papers/cluster_lineage_2026/` only
**Canonical location:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`
(a pointer stub lives at `roadmap/ROADMAP_cluster_lineage_empirics.md`).
**Executor input:** this file + the repository. Read `AGENTS.md` first; it carries the shared
agent core (verification loop, escalation, conventions).

## Reporting protocol (binding)

ALL executor communications live in `papers/cluster_lineage_2026/agents/`, one dated markdown
file each, never appended to this roadmap and never placed anywhere else:

- Stage reports: `YYYY-MM-DD_sol_E<stage>_report.md` — what ran, deliverable paths, every
  acceptance check with its measured value against its tolerance, deviations, open items.
- Gate requests: end the stage report with a `GATE REQUEST` section listing exactly what the
  owner must rule on. Do not start a gated stage before the ruling is recorded in the agents
  folder (`YYYY-MM-DD_owner_E<stage>_gate.md` or an owner note in the stage report).
- Escalations and data requests (e.g. the FF6 fallback): `YYYY-MM-DD_sol_escalation_<topic>.md`.
- Every report names the runner scripts and cache directories it used, so the owner-side
  verification agent can reproduce any number without reading code first.

## Purpose

Produce the empirical evidence set for the Quantitative Finance cluster-lineage paper on three
universes, testing three claim families: **C1 stability** (temporal smoothing cuts membership
and lineage churn at bounded fidelity cost, and the reduction is predicted by the noise-floor
theory), **C2 significance** (stable clusters cut reassignment-driven turnover in a
cluster-relative momentum overlay, improving net-of-cost performance against two yardsticks),
**C3 interpretability** (the labelling grammar produces taxonomy-aligned, persistent labels
with a shared vocabulary across the two MATF universes). The theory-validation checks are
embedded so the paper's propositions are confirmed or falsified on the same runs that produce
its exhibits.

## Global constraints (binding, all stages)

- **All new code lives in `papers/cluster_lineage_2026/replication/`** (plus data-preparation
  scripts under `papers/cluster_lineage_2026/`). Do not modify `optimalportfolios/` (package),
  `factorlasso`, `qis`, or anything under `rosaa/`.
- **No rosaa imports anywhere in the paper harness.** The existing `rosaa.local_path` imports
  in `replication/methods.py`, `run_sweep.py`, `sp500_baseline.py` are refactored out in E0.
  Output/cache location comes from the environment variable `CLUSTER_LINEAGE_OUTPUT_DIR`
  (default `~/OneDrive/analytics/outputs/cluster_lineage_2026/`), via one paper-local
  `replication/local_path.py` helper. Caches stay OUTSIDE the repository.
- **Where an analytic exists in factorlasso or optimalportfolios, consume it.** The inventory
  table below states, per analytic, whether it exists or is NEW; NEW analytics are implemented
  in `replication/metrics.py`, never in the packages.
- **Causality.** Every partition, score, and weight at date t uses data up to and including t
  only. Lineage labelling (`analyze_cluster_lineage`) is the one full-panel offline diagnostic;
  it never feeds weights or scores.
- **Determinism.** No randomness except the seeded resampling in E6 (fixed seeds recorded in
  the output). Identical reruns produce identical tables.
- **Do not change numerical results of existing code paths.** Baselines come from the
  unmodified estimator; smoothing enters only through the declarative `LassoModel` smoother
  fields / `compute_rolling_smoothed_clusters` + the `precomputed_*` injection hooks.
- **No git push, no branch deletion, no release actions.**
- Expected non-errors: factorlasso warns and zeroes assets below `warmup_period` valid
  observations; this is normal.

## Data and reference environment

All inputs are in `papers/cluster_lineage_2026/data/` (and `msci_us/`):

- `msci_us_log_returns.csv` — daily log EXCESS returns (owner-confirmed convention; do not
  subtract a risk-free rate), 1,358 securities keyed by `index_symbol`.
- `msci_us_inclusion_indicators.csv` — daily 0/1 point-in-time index membership, same columns.
- `msci_us_metadata.csv` — GICS at four levels + identifiers.
- `msci_us/msci_us_index_total_return_timeseries.csv` — the index series (reference only; the
  backtest benchmark is EW, see below).
- `futures_log_returns.csv` — daily log returns, 95 contracts; `futures_metadata.csv` —
  `asset_class` (7 groups), `geography`, `ac_geography`, carry fields.
- `mac_log_returns_ME.csv` (332×170), `mac_log_returns_QE.csv` (107×17), `mac_metadata.csv`
  (168 rows; 15 ME + 4 QE return columns have no metadata row — E1 classifies them from the
  metadata sources and states the inclusion rule in the data report).
- `risk_factors_custom.csv` — daily log returns of the 11 MATF factors (owner-confirmed as
  the paper's MATF set): Equity, Rates, Credit, Credit EM, Carry G10, Carry EM, Inflation,
  Commodities, Private Equity, Rates Vol, Fx.
- FF6 factors — NOT yet in the tree; E1 collects them (daily FF5 + daily Momentum from the
  Ken French Data Library, combined to Mkt-RF, SMB, HML, RMW, CMA, MOM). If network access is
  unavailable, stop and request the two CSVs from the owner; do not substitute another source.

### Universe specifications (fixed by owner ruling, 2026-08-13)

| | U1 MSCI US | U2 global futures | U3 MAC funds |
|---|---|---|---|
| factor model | FF6 | MATF 11-factor | MATF 11-factor |
| asset frequency | W-WED | W-WED | ME + QE |
| lasso spans | {W-WED: 156} | {W-WED: 156} | {ME: 36, QE: 12} |
| estimation dates (ME) | 2006-08-31 .. 2026-07-31 (~240) | 2002-01-31 .. 2026-07-31 (~294) | 2002-12-31 .. 2026-07-31 (284) |
| universe at date t | index members at t with warmup history | all listed contracts with warmup | production universe |
| costs (backtest) | 10 bp | 20 bp | 50 bp |
| taxonomy (ranking yardstick) | `gics_sector` | `asset_class` | Asset Class (level 0) |
| taxonomy (interpretability) | sector / industry group / industry | asset_class, ac_geography | Asset Class / Sub Asset Class |
| momentum | 48w skip 4w | 48w skip 4w | ME: 12m skip 1m; QE: 4Q skip 1Q |

Weekly returns are the SUM of daily log returns within the W-WED week (`resample('W-WED').sum(min_count=1)`).
Factor NAVs for the estimator are built from factor log returns via `qis.returns_to_nav` on
the W-WED grid. The coarse taxonomy is the ranking yardstick because within-group quintile
selection needs group sizes well above five; the finer levels are used in E4 interpretability
metrics only.

### Estimator (U1/U2; U3 differs only in the marked fields)

```python
lasso_model = LassoModel(model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                         reg_lambda=1e-05,
                         span=156, span_freq_dict={'W-WED': 156},   # U3: {'ME': 36, 'QE': 12}
                         cutoff_fraction=0.6, linkage_method='ward',
                         distance_transform=DistanceTransform.ONE_MINUS_RHO,
                         dependence_measure=DependenceMeasure.PEARSON,
                         group_penalty='normalized', l1_weight=0.0, demean=True,
                         solver='CLARABEL', warmup_period=12, nonneg=False,
                         auto_sign_constraints=True, auto_sign_threshold_t=1.0,
                         auto_sign_adaptive_weights=True, auto_sign_adaptive_gamma=1.0,
                         auto_sign_adaptive_floor=0.5,
                         unilasso_loo=True, unilasso_non_negative=True,
                         **smoother_fields)
estimator = FactorCovarEstimator(rebalancing_freq='ME', lasso_model=lasso_model,
                                 factor_returns_freq='W-WED', factor_covar_span=52,
                                 is_apply_vol_normalised_returns=False, demean=True)
```

U3 must replicate the production mac configuration from `inputs_store/mac_20260630/config_snapshot.csv`
exactly (the acceptance check below pins it numerically).

### Configuration grid (the treatment axis, identical on all universes)

| config | smoother fields |
|---|---|
| `baseline` | `cluster_smoother_type=NONE` |
| `M0_quarterly_hold` | `HOLD, recluster_freq='QE'` |
| `M1_delta_0.05` / `M1_delta_0.10` | `PARTITION_BONUS, smoother_delta=0.05 / 0.10` |
| `M2_lambda_0.5` / `M2_lambda_0.7` | `SIMILARITY_EWMA, smoother_lambda=0.5 / 0.7` |
| `M1_star` / `M2_star` | calibrated values, supplied by the owner from the theory track; build the slots now, fill on instruction |

Backtest-only yardstick legs (no estimation run needed): `global` (whole-universe rank) and
`taxonomy` (rank within the taxonomy column above). Every profile also carries the EW-all
benchmark, which is BOTH the performance benchmark and the market benchmark for the beta and
alpha computations of every leg.

## Analytics inventory (what exists vs what Sol implements)

| analytic | source | status |
|---|---|---|
| rolling FCGL fits, smoothing, injection hooks | `factorlasso` (`LassoModel` smoother fields, `compute_rolling_smoothed_clusters`), `optimalportfolios.FactorCovarEstimator` | exists |
| lineage: tracks, churn, matcher stats, labels | `factorlasso.analyze_cluster_lineage` / `run_cluster_lineage_report` | exists |
| co-association confidence | `factorlasso` (`RollingClusterData.co_association`) | exists |
| residual diagonality | `factorlasso.diagnose_residuals` | exists |
| raw churn, greedy panel, ARI, rank-stability, partition equality | `replication/sp500_baseline.py` | exists — move into `metrics.py` unchanged |
| backtest, perf stats, alphas/betas vs benchmark | `qis.backtest_model_portfolio`, `qis.PerfParams`, `optimalportfolios.alphas.profile_alpha_signals` + `compute_alpha_rank_analysis_table` + `generate_alpha_profile_report` | exists |
| variation of information VI(Q_a, Q_b) | — | NEW (`metrics.py`) |
| size/shape metrics (count stats, singleton share, largest-cluster share, size entropy) | — | NEW |
| assignment margins + predicted churn (theory check) | — | NEW |
| membership-flow decomposition (U1 entry/exit/warmup vs clusterer) | — | NEW |
| covariance-invariance metrics (Frobenius, max entry, EW ex-ante vol) | — | NEW |
| turnover decomposition (signal vs reassignment) | — | NEW |
| block bootstrap + permutation nulls | — | NEW |
| Greene-style and MONIC-style lineage baselines | — | NEW |

## Stage E0 — paths, metric library, config registry (recommended reasoning effort: high)

**Deliverable.** `replication/local_path.py` (env-var output root, no rosaa);
`replication/metrics.py` implementing every metric below with the formula in the function
docstring; `replication/configs.py` (universe specs + config grid as enums/dataclasses, one
place); refactor of the three existing rosaa-importing modules onto the new path helper.

**Metric definitions (frozen here; the module docstring is the paper's methods text).**

1. *Raw membership churn*: per-asset membership changes per asset-year over consecutive dates
   where the asset is clustered at both, cluster ids matched greedily by maximum member
   overlap (existing `annualized_churn` ∘ `greedy_membership_panel`).
2. *Lineage churn + lineage stats*: derived-id changes per asset-year from
   `analyze_cluster_lineage` at module defaults; n tracks, tracks/asset, matcher-attributable
   churn (cohort retention ≥ 0.60), mean link overlap, median track life.
3. *Consecutive-partition distances*: per-transition ARI and VI; medians and IQRs.
   VI(Q_a,Q_b) = H(Q_a|Q_b) + H(Q_b|Q_a) on the common clustered assets.
4. *Size/shape*: per-date cluster count (median/min/max), median cluster size, singleton
   share, largest-cluster share (flag > 0.25), entropy of the size distribution.
5. *Co-association*: cross-sectional median and share < 0.5 of the trailing-6 confidence panel.
6. *Assignment margins*: per asset-date, m_i = mean distance to nearest other cluster minus
   mean distance to own cluster (average-distance proxy for the cut margin; state the proxy in
   the docstring), in units of σ_d = sqrt(2·(1−λ^k)) · (1−ρ̂_i²)/sqrt(span), λ the weekly
   (resp. ME) EWMA decay, k the estimation step in span units, ρ̂_i the asset's mean
   within-cluster correlation. Outputs: margin histogram per universe; predicted churn
   Σ_i Φ(−(m_i+δ)/(√2·σ_d)) vs realised churn per config.
7. *Membership-flow decomposition (U1)*: each asset-date reassignment classified
   {index entry/exit, warmup entry, clusterer reassignment}; churn counts only the last.
8. *Fidelity band*: per-date ARI vs same-date baseline partition (median); taxonomy-ARI
   deltas vs baseline ≤ 0.03 in absolute value at every level; median cluster count within
   ±15% of baseline. A config outside the band is REJECTED and reported as such.
9. *Risk-model invariance*: per-date relative Frobenius distance of fitted covariance vs
   baseline, max relative entry change, |Δ ex-ante EW vol|; residual-diagonality means within
   5% of baseline (existing guard).
10. *Signal-rank stability*: existing three metrics (MoM Spearman, rank MAD,
    reassignment-attributable rank-MAD gap), quintile-consistent.
11. *Turnover decomposition*: at each rebalance, w̃_t = weights from the same scores under the
    prior-date partition; reassignment turnover = 0.5·Σ|w_t − w̃_t|, signal turnover =
    0.5·Σ|w̃_t − w_drift,t−1|; report both plus the residual to total (the split is a bound,
    not an identity — say so in the docstring).
12. *Interpretability set*: taxonomy ARI per level; track modal-taxonomy purity
    (member-weighted); core-track count (coverage ≥ 0.7); label-string churn per asset-year;
    share of track life under modal label; distinct-label count vs cluster count; primary
    factor variance share; 'Idio' share.

**Acceptance.** Regression test: the metrics module reproduces the frozen S&P 500 baseline
numbers from the cached `sp500_baseline` panel (lineage churn 3.2115 ± 0.0001, 216 tracks,
matcher churn 0.486 ± 0.005, ARI medians within 0.005) — regression only, not paper numbers.
All three refactored modules import without rosaa. Deterministic rerun of the metric suite is
byte-identical.

**Report, then OWNER GATE E0** (metric definitions frozen).

## Stage E1 — data layer (recommended reasoning effort: high)

**Deliverable.** `replication/universes.py`: one loader per universe returning a common
container (asset returns dict by frequency, factor NAVs, point-in-time eligibility masks,
taxonomy series, EW benchmark constructor). Plus `replication/fetch_ff6.py` (Ken French
daily FF5 + Momentum → W-WED log factor returns → NAVs; writes
`data/ff6_factors_wwed.csv` with a provenance header: URL, vintage date, transformation).

Specifics: U1 eligibility at date t = inclusion indicator 1 at t AND ≥ `warmup_period`
weekly observations; membership changes between MEs take effect at the next estimation date.
U2/U3 factor NAVs from `risk_factors_custom.csv` resampled W-WED. U3: classify the 19
metadata-uncovered return columns from the metadata files; state the rule (universe member vs
benchmark series vs excluded) per column in the data report; do not silently drop.

**Acceptance.** Per-universe data-quality report (CSV + printed table): date span, column
count, missing-data share, eligibility counts per estimation date (U1 min/median/max members),
weekly-return outlier scan (|r| > 50% listed), factor panel span check, conventions stated
(log, excess for U1, W-WED sum). FF6 file has ≥ 6 columns, daily source dates ≥ 2003-01-01,
and its W-WED sums reconcile against daily compounding on 10 sampled weeks to 1e-12.

**Report, then OWNER GATE E1** (data conventions sign-off).

## Stage E2 — rolling estimation runs (recommended reasoning effort: medium)

**Deliverable.** `replication/estimate.py`: per universe × config, expanding-window fits at
the estimation dates, one pickle per date under
`$CLUSTER_LINEAGE_OUTPUT_DIR/<universe>/<config>/YYYYMMDD.pkl`, cache-first, parallel over
dates (`ProcessPoolExecutor`, default 4 workers; U1 is the heavy leg — budget hours, reuse
caches aggressively). Smoothed configs run the two-pass pattern: partitions from
`compute_rolling_smoothed_clusters`, injected through `precomputed_clusters` /
`precomputed_linkages` / `precomputed_cutoffs` (supplied together, keyed by frequency).

**Acceptance.**
- U3 baseline replication: at 2026-06-30 the clusters match the production run from
  `inputs_store/mac_20260630` at pairwise Rand ≥ 0.99 and modal agreement ≥ 0.97.
- Injected == fitted partitions on 100% of dates for every smoothed config (assert, listing
  mismatched dates on failure).
- U1 snapshot count within ±2 of the schedule length; per-date member counts logged.
- Runtime and cache report per universe × config.

**Report** (no gate; E3–E5 may start as caches complete per universe).

## Stage E3 — stability results and theory validation (recommended reasoning effort: high)

**Deliverable.** `replication/run_stability.py` producing, per universe, one workbook
`<universe>_stability_20260813.xlsx`: metric-suite table (rows = configs, columns = metrics
1–10), per-date panels, the margin histogram data, predicted-vs-realised churn per config,
and the two scaling tests:

- *Frequency scaling*: U3 ME vs QE natively; U1 and U2 re-scored at QE estimation dates
  (subsample the cached ME snapshots — no refit). Compare churn ratios against the
  sqrt(2(1−λ^k)) prediction.
- *Kurtosis check*: per universe, excess kurtosis of pooled weekly (resp. ME) returns, and
  realised churn vs Gaussian-predicted churn; report the multiplier.
- *Risk-model invariance* (metric 9) for every smoothed config on all three universes.

**Acceptance.** No NaNs in metric tables; every config marked PASS/REJECTED against the
fidelity band (metric 8); predicted-vs-realised churn correlation across configs reported per
universe; deterministic rerun byte-identical.

**Report, then OWNER GATE E3** (theory verdicts recorded before E5 results are interpreted).

## Stage E4 — lineage, labelling, interpretability (recommended reasoning effort: medium)

**Deliverable.** `replication/run_interpretability.py`: per universe × {baseline, best
in-band smoothed config}, the lineage report (`run_cluster_lineage_report`), metric set 12,
and three exhibits: (a) taxonomy-ARI-by-level table (the peak-level location is the finding),
(b) the cross-universe label-vocabulary table for U2 and U3 (same MATF panel: label sets side
by side with exposure profiles; count labels covering ≥ 90% of non-Idio systematic variance),
(c) three case-study tracks per universe (id, life span, membership path, loadings path,
label). For U2, propose leverage-free equity-beta bucket thresholds from the cross-sectional
beta distribution (report quantiles; the owner rules on the thresholds — do not adopt
silently).

**Acceptance.** Purity/persistence tables complete for all six runs; vocabulary table
present; case-study tracks have coverage ≥ 0.7; threshold proposal table for U2 delivered.

**Report** (no gate; feeds the manuscript directly).

## Stage E5 — momentum backtest arm (recommended reasoning effort: high)

**Deliverable.** `replication/run_backtests.py`. Per universe, ONE profile containing:
EW-all benchmark; `global` rank leg; `taxonomy` rank leg; one `cluster_<config>` leg per
in-band config. Mechanics:

- Momentum score per the universe table (log-return sums over the stated windows). A second
  score variant `momentum_vol_adj` (same window return divided by annualised realised vol,
  span 13 at the asset frequency) runs as a robustness pass for the three headline legs
  (global, taxonomy, cluster_baseline, cluster_best).
- Ranking: percentile rank within the leg's group structure; selection = rank ≥ 1 − q with
  q = 0.20 (quintile) primary; q parameterised and q = 1/3 run as robustness. Singleton and
  tiny groups: the rule rank ≥ 1 − q applies as-is (a singleton always selects); state this
  in the docstring.
- Weights: equal weight across all selected assets. U3: ME sleeve reselects monthly, QE
  sleeve reselects only at QE dates (selection frozen in between); the EW combination spans
  both sleeves.
- Backtest: `qis.backtest_model_portfolio(prices, weights, rebalancing_costs=<10/20/50 bp>,
  weight_implementation_lag=1, ticker=<leg>)`; if `profile_alpha_signals` supports the cost
  and lag parameters directly, it may be used instead — verify, and use ONE path for all legs.
- Outputs per universe: `compute_alpha_rank_analysis_table`-style summary + a table with
  net-of-cost total return, vol, Sharpe, alpha and beta vs the EW benchmark, annualised
  one-way turnover, cost drag (bp/yr), reassignment vs signal turnover split (metric 11,
  cluster legs only), and the crisis-window breakdown (GFC 2008 where the sample covers it,
  COVID 2020, rate shock 2022). `generate_alpha_profile_report` PDF per universe.

**Acceptance.** Every profile carries both yardstick legs and the benchmark; cluster-leg
turnover split reported with residual < 10% of total; identical scores across legs at each
date (only the grouping differs) — assert on a sampled date; deterministic rerun.

**Report, then OWNER GATE E5** (payoff verdicts).

## Stage E6 — inference layer (recommended reasoning effort: high)

**Deliverable.** `replication/run_inference.py`:

- Moving-block bootstrap over estimation dates (block length 6, 2,000 draws, seed 20260813)
  for CIs on: churn deltas vs baseline (per config), taxonomy-ARI deltas, net Sharpe deltas
  and turnover deltas vs baseline cluster leg (bootstrap the monthly return/turnover series
  jointly per leg pair).
- Permutation nulls for taxonomy ARI: per date, 500 size-matched random partitions
  (permute labels holding cluster sizes), null distribution of ARI; report the median
  observed ARI against the null 95th percentile.
- Lineage baselines: Greene-style Jaccard-threshold tracker (threshold 0.3, consecutive
  dates) and MONIC-style transition tracker (match/split/merge events on member overlap),
  both run on the baseline panels of all three universes; compare track counts, churn, and
  fragmentation against `analyze_cluster_lineage` MCF defaults in one table.

**Acceptance.** Every headline delta quoted in E3/E5 carries a CI or a null in the output
tables; seeds and draw counts recorded in the workbook; rerun with the same seed
byte-identical.

**Report** (no gate).

## Stage E7 — exhibit assembly and traceability (recommended reasoning effort: medium)

**Deliverable.** `replication/build_exhibits.py` writing one canonical workbook per universe
plus `exhibit_index.csv`: one row per prospective paper exhibit — takeaway title, claim
family (C1/C2/C3/theory), universe(s), source script, source workbook/sheet. A one-page
summary table per claim family (the numbers the manuscript will quote, nothing else).

**Acceptance.** Every number in the claim-family summaries traces to a script + workbook in
this folder (the manuscript's number-traceability gate); no orphan exhibits.

**Report, then OWNER GATE E7** — hand-off to the manuscript stage and the adversarial pass.

All stage reports and gate rulings above follow the Reporting protocol: dated files in
`papers/cluster_lineage_2026/agents/`, nothing appended to this roadmap.

## Execution order and parallelism

E0 → E1 → E2 → {E3, E4, E5 in any order per universe as caches land} → E6 → E7.
The calibrated `M1_star`/`M2_star` values arrive from the owner mid-programme: build the grid
slots in E0, run them as an E2 increment plus E3/E5 deltas when the values are supplied. U2
and U3 are cheap (95 and 187 assets) — run them first end-to-end to shake out the pipeline,
then the heavy U1.

---

# Stage extension (2026-08-14): E8 — U3M funds production-fidelity momentum arm

Owner rulings recorded (2026-08-14), following the E4/E5-U1 read-out:

1. The E5-U1 cluster-vs-taxonomy gap is hypothesised to be a peer-group **granularity**
   artifact (median 83–86 clusters of single-digit size vs 11 sectors), not a signal defect.
   The production operating point for funds — coarse clusters (~15–16 of ~11 names) — is
   where cross-sectional cluster momentum works in production. E8 tests the production
   operating point end to end.
2. **QE-frequency funds are EXCLUDED from the cluster-momentum arm.** In production these
   funds are scored by manager alpha, not cluster momentum, and their small sample (17 assets
   in 2–7-name clusters) distorts within-cluster cross-sections. The exclusion rule is
   stated verbatim in every E8 output.
3. The production factor set is the CUSTOM set — for U3 this is already
   `data/risk_factors_custom.csv` (the 11-factor MATF panel); no factor change is needed. If
   the owner drops a newer custom export into `data/`, E8 picks it up and reports the vintage.
4. The production momentum signal (rosaa `MOMENTUM_CLUSTER`) is adopted as a signal leg,
   REPLICATED in the paper harness from pinned parameters — the no-rosaa-import constraint
   stands: read the rosaa source and `inputs_store/mac_20260630/config_snapshot.csv` to pin
   the parameters, record them in `replication/configs.py` with file-level provenance
   comments, implement the score in the replication package.

**U3M definition** (new universe variant; U1/U2/U3 tables above are unchanged): U3 restricted
to the 170 ME-frequency funds. Estimator, factors, costs (50 bp), estimation dates
(2002-12-31..2026-07-31, 284 ME), taxonomy columns and momentum windows are U3's. The QE
sleeve of the E5-U3 backtest does not exist in U3M.

## Stage E8a — U3M derivation, separability proof, lineage re-scoring (effort: high)

**Key efficiency claim, to be PROVEN not assumed:** per-frequency separability. The estimator
fits each frequency's panel independently (per-freq LASSO and per-freq clustering share only
the factor panel), so the ME-sleeve clusters and betas of the cached U3 snapshots should be
IDENTICAL to a ME-only re-estimation, and U3M requires NO new estimation runs.

**Deliverable.** `replication/run_e8a.py`:

1. Separability proof: refit at least 3 estimation dates (early / middle / late) with
   ME-only inputs; assert exact ME partition equality and max |Δbeta| ≤ 1e-10 against the
   cached U3 snapshots. If the proof fails, STOP and escalate — do not silently re-estimate.
2. U3M panels: filter the cached U3 snapshots (all 284 dates, both configs of the accepted
   U3 pair) to ME assets; drop QE clusters. The accepted U3 in-band smoothed config carries
   over unchanged (its ME partitions are identical by separability).
3. Re-score on the ME-only panel: metric suite 1–10 and 12, lineage report per config,
   fidelity band re-evaluated, granularity table (expect median ~15–16 clusters, median size
   ~11 — report actuals).

**Acceptance.** Separability assertions pass (listing the refit dates and max deltas);
metric tables complete and NaN-free for both configs; fidelity-band verdicts stated;
deterministic rerun byte-identical.

**Report** (`YYYY-MM-DD_sol_E8a_report.md`), no gate — E8b may start on acceptance.

## Stage E8b — production-signal backtests and inference deltas (effort: high)

**Deliverable.** Extend `replication/run_backtests.py` (or add `run_e8b.py` reusing its
mechanics) and `replication/validate_e8.py`. One U3M profile per analysis window (the same
two-window convention as the accepted U3 E5 output, separately labelled, never pooled):

- **Signal legs**: `S_raw` (U3's 12m-skip-1m log-return momentum, unchanged); `S_voladj`
  (the established vol-adjusted robustness spec); `S_prod` (the replicated rosaa
  `MOMENTUM_CLUSTER` signal — risk adjustment, spans, winsorisation/z-scoring, and the
  `min_cluster_size` Global-bucket fallback exactly as pinned; every parameter cited to its
  rosaa source location in `configs.py`).
- **Ranking legs per signal**: `global`, `taxonomy` (Asset Class, per the U3 yardstick
  ruling), `cluster_baseline`, `cluster_<accepted in-band config>`. Yardstick discipline
  verbatim from the E3b/E4/E5 gate: global and taxonomy are the only performance yardsticks;
  the EW block is reference-only for NAV statistics and per-leg alpha/beta.
- **Robustness**: q = 1/3 for the headline legs; one Sub-Asset-Class rank leg under `S_prod`
  as the granularity diagnostic (analogue of U1's finer-taxonomy check).
- Mechanics per the E5 protocol: 50 bp costs, `weight_implementation_lag=1`, quintile
  primary, crisis windows, Metric 11 per cluster leg (trade-interaction term labelled, no
  residual guard), point-in-time price-gap disclosure, PDFs, deterministic reruns.
- **Inference deltas** (E6 protocol: moving-block bootstrap, block length 6, 2,000 draws,
  seed 20260813): CIs for (a) `S_prod` vs `S_raw` within the in-band cluster leg (net Sharpe
  and turnover deltas), (b) the in-band cluster leg vs taxonomy under each signal.

**Acceptance.** Raw scores per signal identical across ranking legs (sampled-date assert);
every profile carries both yardsticks and the EW reference block; Metric-11 identity within
1e-14; all CIs present with seeds recorded; validator PASS lines for completeness,
determinism, and the QE-exclusion statement in every output; no rosaa import anywhere
(AST-grep the replication package).

**Report** (`YYYY-MM-DD_sol_E8b_report.md`), then **OWNER GATE E8**: rulings on (i) whether
the production operating point closes the taxonomy gap on U3M, (ii) the `S_prod` vs `S_raw`
verdict, (iii) whether to dispatch the U1 granularity-confirmation stage (prod-scale
clustering on equities) as E9.

Execution note: E8 runs off existing U3 caches (no estimation budget); it may run in
parallel with E7 exhibit assembly, but E7's claim-family summaries must not cite E8 numbers
until the E8 gate is ruled.
