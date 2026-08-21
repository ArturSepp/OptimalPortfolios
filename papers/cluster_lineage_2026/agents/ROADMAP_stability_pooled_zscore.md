# ROADMAP: stability-pooled z-score standardisation (S0-S6)

Location on adoption: `OptimalPortfolios/papers/cluster_lineage_2026/agents/ROADMAP_stability_pooled_zscore.md`
Date: 2026-08-20. Owner: Artur. Executor: Sol. Proposer of the method: Ben (Monday TAA meeting).
Routing: factorlasso changes under OSS instructions v2.4 (hard invariants, release discipline). Harness and backtests under research instructions v2.6, code regime.

Read before starting: `agents/2026-08-17_claude_paper_status_and_next_steps.md`, both Sol summaries of 2026-08-16/17, and the latest dePC1 stage report. This workstream is separate from the QF paper. The frozen U1/U2/U3 paper specs are not to be touched or extended by these results.

## 0. The method under test

Cluster-stability-weighted variance pooling inside cross-sectional z-scoring. For asset i in cluster g at date t:

z_i = (x_i - mu_g) / sqrt(w * var_g + (1 - w) * var_global)

with w in [0, 1] a trailing co-cluster stability weight derived from the co-association panel. w = 1 must reproduce `score_within_clusters` exactly. w = 0 gives the cluster-demeaned, globally-scaled score. The numerator stays the cluster mean in the primary variants, so asset-class neutrality is preserved.

Symbol discipline: use `w` (stability weight) in code and reports. Do not use lambda. Lambda is the EWMA decay in the theory draft and the collision will corrupt later consolidation.

## 1. Variants

| id | w granularity | pooling target | notes |
|-|-|-|-|
| V0 | n/a | none | exact bypass, byte-identical to current `score_within_clusters` output |
| V1 | per cluster | variance only | Ben's proposal. w_g = mean trailing co-association within cluster g |
| V2 | per asset | variance only | w_i = asset i's trailing co-cluster frequency with its current peers. Targets boundary assets directly |
| V3 | per cluster | variance and mean | mu_i = w_g * mu_g + (1 - w_g) * mu_global. Comparison arm only. Breaks strict asset-class neutrality. Flag this in every exhibit it appears in |

The `min_cluster_size` global fallback is unchanged in all variants and takes precedence over pooling. Production defaults are untouched. Everything is opt-in.

## 2. Pinned definitions (confirm or amend in the S0 report before coding)

1. Trailing co-association window: aligned to the production clustering span, read from `ProductConfig` at runtime, never hardcoded. Sweep axis in S3: {1x span, 2x span}.
2. Partitions feeding w: the operating partitions of the run under test. In MAC production the smoother is NONE, so operating and raw coincide. If any run uses a smoothed partition, compute w on the raw partitions and state so. Stability pooling on top of smoothed partitions double-counts stability.
3. Point in time: w at date t uses partitions up to and including t, then the standard signal implementation lag applies. No lookahead. Add a causality test.
4. Short history: fewer than 12 trailing partition dates gives w = 1, which is the current behaviour. Report coverage of the w panel per date.
5. Small-cluster bias: report the cross-sectional correlation of cluster size and w_g at each date. If |corr| > 0.5 persistently, flag it in the S4 report as a confound.

## 3. Stages

### S0. Environment and read-in
OneDrive FactorLasso checkout first on PYTHONPATH with the runtime assertion on `factorlasso.__file__`. Locate every call site of `score_within_clusters` and of `_co_association_panel` across factorlasso, optimalportfolios and rosaa (three-way usage diff). Report the call-site map and the proposed wiring point for the MAC diagnostic before writing any code. Confirm or amend the pinned definitions above. Stop for owner ruling only if a definition cannot be met as written.

### S1. Implementation (factorlasso, additive only)
1. Promote the co-association panel to a public accessor. Additive, documented, no change to the private path.
2. Add the pooling as an opt-in mode on the standardisation layer. Default off is an exact no-op, same contract as `ClusterCorrelationTransform.NONE`. No signature breaks, no default changes, no new hard dependencies.
3. Wire the MAC diagnostic harness at the scoring layer only. No production code path changes in rosaa. If a rosaa edit appears unavoidable, stop and report.

### S2. Tests (fail-before-pass checkpoint, per the dePC1 test pattern)
1. Hand-computed reference on a small panel: V1, V2, V3 against explicit arithmetic.
2. V0 byte-identity: pooling off reproduces current `score_within_clusters` bit for bit on a production-shaped panel.
3. w = 1 reproduces V0 exactly. w = 0 reproduces cluster-demeaned global-variance scoring exactly.
4. `min_cluster_size` fallback precedence under all variants.
5. Causality: perturbing partitions after t leaves w at t unchanged.
6. Short-history fallback to w = 1.

### S3. MAC diagnostic backtest (primary arm)
Harness: the `MAC_CONSTRAINED_BATCH` diagnostic used for dePC1 D6, rates-aware excess-alpha convention (`add_rates_data=True`), production costs. Baseline: accepted production configuration, reference TAA Sharpe 1.15. Signals: TAA momentum and low-beta. BAB sign stays pinned under every variant.
Grid: {V1, V2, V3} x {1x span, 2x span}. Six cells plus baseline. No wider grid without a further ruling.
Report per cell: TAA Sharpe, ex-post TE, ex-ante TRE at the production grid point, annual turnover, max drawdown, and the deltas to baseline. Numbers, not adjectives.

### S4. Mechanism diagnostics
1. Reassignment-vs-signal turnover decomposition per cell. The adoption story requires the turnover gain to sit in the reassignment component. If it sits in the signal component, say so plainly.
2. Distribution of w over time and cross-section, per-cluster and per-asset. Identify whether low-w mass sits on boundary assets (V2 motivation) or is smeared cluster-wide (V1 limitation).
3. The size-vs-w confound check from pinned definition 5.

### S5. Robustness
1. Split-window on the best cell and V0, same protocol as the U2 split-window discipline.
2. E6-protocol bootstrap CIs on the headline Sharpe and turnover deltas of the best cell.
3. Optional generalisation arm, only if MAC results are positive: rerun the best variant on U1 and U3 at their frozen operating points and selected signal specs. Read-only with respect to the paper trees. Results go in this workstream's reports, not in the paper exhibits.

### S6. Consolidated report and recommendation
One report in `agents/`, conclusions first. Recommend exactly one of ADOPT-CANDIDATE / ROBUSTNESS / REJECT with the three decisive numbers stated in the first paragraph. Adoption bar: net-of-cost turnover reduction with Sharpe held at or above baseline at matched risk, mechanism confirmed by the S4 decomposition, surviving split-window and CIs. A turnover gain that costs Sharpe at matched risk is a REJECT.

## 4. Standing constraints

1. No factorlasso release, tag or version bump. Implementation stays in tree pending owner sign-off, as with 0.15.0.
2. Breaking numerical changes require an explicit checkpoint conversation. None are expected, everything here is opt-in.
3. Backward compatibility with optimalportfolios and rosaa on every public change. The S0 three-way diff is the gate.
4. Caches under `$CLUSTER_LINEAGE_OUTPUT_DIR/stability_pooling/`. Nothing written into the paper trees.
5. Stage reports named `YYYY-MM-DD_sol_stability_pooling_S<k>_report.md` in `agents/`. Read the latest report before continuing any stage.
6. Attribution: record Ben as the proposer in every report and any eventual exhibit.

## 5. Out of scope

The QF paper's frozen specs and exhibits. Any production configuration change. Mean-pooling as a default (V3 is a comparison arm only). Sweeping the co-association decay beyond the two pinned windows. dePC1 interaction runs.
