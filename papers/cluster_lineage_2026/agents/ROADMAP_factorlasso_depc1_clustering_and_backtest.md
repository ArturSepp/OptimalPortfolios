# Roadmap: FactorLasso de-PC1 clustering analytics and strategy backtest

**Date:** 2026-08-16  
**Status:** proposed; roadmap only, no implementation or backtest mutation yet  
**Repositories:** local `FactorLasso` development checkout plus the ignored
`OptimalPortfolios/papers/cluster_lineage_2026/` empirical harness

## Objective

Add an opt-in FactorLasso clustering mode that removes the dominant principal component from
the point-in-time clustering correlation matrix, then measure—without retuning any signal or
portfolio parameter—how that changes:

1. cluster topology, taxonomy alignment, stability, and lineage;
2. the frozen long-short strategies for U1 stocks, U2 BlackRock funds, and U3 futures;
3. performance relative to each strategy's already frozen global-rank comparator.

The first principal component is commonly interpreted as the market-wide or common mode in
equity correlation matrices. Plerou et al. identify the largest correlation eigenvalue as an
influence common to stocks, while remaining deviating eigenvectors reveal sector structure.
MacMahon and Garlaschelli explain why removing system-wide dependence can reveal mesoscopic
communities. The implementation is narrower than their full RMT filtering: it removes PC1 only;
it does not remove the random eigenvalue bulk.

Primary references:

- Plerou et al. (2002), [Random matrix approach to cross correlations in financial
  data](https://doi.org/10.1103/PhysRevE.65.066126).
- MacMahon and Garlaschelli (2015), [Community Detection for Correlation
  Matrices](https://doi.org/10.1103/PhysRevX.5.021006).

## Frozen interpretation of “de-PC1”

FactorLasso currently clusters a signed dependence/correlation matrix, not a covariance matrix.
The first implementation therefore removes PC1 from that correlation matrix. A covariance-basis
PC removal is a separate experiment and is out of scope for this roadmap.

For a current eligible-asset correlation matrix `R` with dominant eigenpair
`(lambda_1, v_1)`:

```text
Q       = R - lambda_1 v_1 v_1'
d_i     = Q_ii
R_dePC1 = diag(d)^(-1/2) Q diag(d)^(-1/2)
```

`R_dePC1` is the residual correlation matrix after rank-one deflation and restandardization.
Restandardization is binding: clustering must not treat residual covariances with heterogeneous
diagonals as correlations. The output is symmetric, has unit diagonal, and is bounded in
`[-1, 1]` up to numerical tolerance.

The transform is applied in this order at every estimation date:

```text
history through t
  -> raw signed dependence matrix
  -> exact point-in-time eligible asset subset
  -> remove PC1 and restandardize
  -> optional temporal cluster smoother
  -> distance transform, linkage, and dendrogram cut
```

This ordering is essential. PC1 removal does not commute with asset restriction. A future-listed,
pre-inclusion, below-AUM, or owner-excluded instrument must never influence the PC1 estimated for
the investable set at date `t`.

The transform affects cluster discovery only. It does **not** remove PC1 from:

- the return panel supplied to momentum;
- the global-rank comparator;
- realised portfolio P&L;
- the final factor covariance or optimizer input directly;
- the FF6 or MATF factor panels.

## Frozen API design

Add to FactorLasso:

```python
class ClusterCorrelationTransform(str, Enum):
    NONE = "none"
    REMOVE_PC1 = "remove_pc1"


def remove_first_principal_component(
    corr_matrix: pd.DataFrame,
) -> ClusterCorrelationTransformResult:
    ...


def apply_cluster_correlation_transform(
    corr_matrix: pd.DataFrame,
    transform: ClusterCorrelationTransform | str = ClusterCorrelationTransform.NONE,
) -> pd.DataFrame:
    ...
```

`ClusterCorrelationTransformResult` records the transformed correlation plus the removed
eigenvalue, `lambda_1 / trace(R)`, minimum residual variance, and the number of missing
off-diagonal pairs set to the package's existing neutral-correlation value of zero.

Add the declarative estimator field:

```python
LassoModel.cluster_correlation_transform = ClusterCorrelationTransform.NONE
```

Default `NONE` must bypass all new numerical operations so every existing regression remains
bit-for-bit unchanged. Strings are accepted through enum validation, matching the existing
`dependence_measure` and `distance_transform` conventions.

Extend the rolling interface with an optional exact eligibility panel:

```python
compute_rolling_smoothed_clusters(
    y,
    estimation_dates,
    lasso_model,
    eligibility: pd.DataFrame | None = None,
)
```

When supplied, `eligibility` must be Boolean, cover every estimation date, and use the same
columns as `y`. It is combined with FactorLasso's data warmup mask. There is no forward-fill or
implicit membership inference. With `eligibility=None` and transform `NONE`, current behaviour
is unchanged.

## Numerical edge-case contract

- Symmetrize the matrix before eigendecomposition; preserve labels and ordering.
- Use `numpy.linalg.eigh` and the largest algebraic eigenpair. Eigenvector sign is irrelevant
  because only `v_1 v_1'` enters.
- Preserve the current clustering convention for unavailable pair correlations: missing
  off-diagonal values become zero and the diagonal is one before decomposition.
- Clip only floating-point excursions within a fixed tolerance. Materially negative residual
  variances or transformed correlations materially outside `[-1, 1]` raise with diagnostics;
  do not silently project to an unrelated nearest-correlation matrix.
- A one-asset matrix returns the one-asset identity and zero removed share.
- Assets with residual variance at the numerical floor are retained as isolated residual
  series: off-diagonal correlations zero, diagonal one. Nothing is silently dropped.
- Record the top eigengap. A tied dominant eigenspace is allowed but flagged as non-unique in
  diagnostics; no economic interpretation of that individual PC is made.
- The same transform is valid for any signed dependence matrix accepted by FactorLasso, but the
  empirical run is Pearson only. No Spearman/Gerber grid is authorized here.

## Frozen empirical operating points

No signal, quantile, cost, universe, smoother, or classification is selected again.

### U1 stocks

- Universe and point-in-time membership: current U1 MSCI US panel.
- Clustering cell: `ME`, EWMA span 36, Ward, `1-rho`, Pearson, existing cutoff.
- Signal and construction: frozen `U1_OPTIMAL_SPEC` ROSAA production configuration, q=25%,
  group-equal cluster leg.
- Comparators: global rank and Bloomberg BICS equal-sector rank; sector leg is unchanged by
  de-PC1.
- Costs: 10 bp one way.
- Primary window: 2009-08-31 through 2026-06-30; full schedule remains labelled separately if
  emitted.

### U2 BlackRock funds

- Universe: BlackRock fund panel with the newly frozen primary eligibility rule
  `12 completed month-end average AUM > USD 100m`.
- Clustering cell: `W-THU`, EWMA span 156, Ward, `1-rho`, Pearson.
- Signal: ROSAA production risk-adjusted momentum, q=25%.
- Sleeve budgets: Equity 50%, Fixed Income 30%, Rest 20% on each side.
- Primary portfolio: global-rank long / group-equal cluster-rank short, every-two-month
  rebalance.
- Comparator: matched global rank only; pure cluster is retained as a diagnostic.
- Costs: 20 bp one way.
- Selection provenance remains post-sensitivity; de-PC1 does not trigger a new AUM search.

### U3 futures

- Universe: the owner-frozen eligible futures panel after the seven low-liquidity exclusions.
- Clustering cell: `W-WED`, EWMA span 156, selected M1-star partition-bonus configuration.
- Hold the existing M1-star delta fixed; do not recalibrate it on de-PC1 distances in the
  primary experiment.
- Signal: frozen ROSAA monthly 12-month configuration, no short span, volatility span 13,
  EWMA mean adjustment, fallback 5, q=25%.
- Portfolio: +1/-1 long-short with 30/30/30/10 Equity / Fixed Income / Commodities / FX gross
  budgets per side.
- Comparator: matched within-sleeve global rank.
- Costs: 10 bp one way.
- Window: 2009-08-31 through 2026-06-30.

## Stage D0 — freeze baseline and local-development provenance

1. Confirm the FactorLasso checkout is on `main` and record its source commit, dirty status,
   package version, and hashes of the clustering modules.
2. Preserve the existing untracked `.claude/settings.local.json`; do not stage or modify it.
3. Run the FactorLasso clustering-focused tests and full suite before editing.
4. Freeze hashes and measured rows for the three currently selected raw-correlation strategies.
5. Activate the local development source for all later commands by putting
   `C:\Users\artur\OneDrive\analytics\my_github\FactorLasso` first on `PYTHONPATH`.
6. Assert at runtime that `Path(factorlasso.__file__).resolve()` is inside that checkout, not the
   installed `site-packages` copy.

Acceptance:

- pre-change FactorLasso tests and Ruff pass;
- local source path assertion passes;
- current raw-cluster selected strategy results reproduce within `1e-12`;
- no baseline cache is written or changed.

Deliverable: `2026-08-XX_sol_dePC1_D0_baseline_report.md` in the paper agents directory.

## Stage D1 — implement the pure FactorLasso matrix transform

Files:

- `FactorLasso/factorlasso/cluster_utils.py`;
- `FactorLasso/factorlasso/__init__.py`;
- new `FactorLasso/tests/test_cluster_pc1_removal.py`.

Work:

1. Add the enum, diagnostic result dataclass, pure PC1-removal function, and dispatch wrapper.
2. Include the Plerou and MacMahon/Garlaschelli sources and state that FactorLasso removes only
   the dominant common mode, not the RMT noise bulk.
3. Validate labels, symmetry, finite diagonal, residual variances, and output bounds.
4. Keep `NONE` an exact no-op.

Tests:

- hand-computed rank-one-plus-block matrix;
- sign invariance of the eigenvector outer product;
- symmetry, unit diagonal, bounds, label/order preservation;
- one asset, identity, perfectly common assets, missing pairs, and near-singular residuals;
- invalid transform and malformed matrices raise clear errors;
- synthetic strong-common-factor panel: de-PC1 improves recovery of known blocks at a matched
  cluster count;
- fail-before-pass checkpoint by running the new tests before implementation.

Independent numerical reference:

For complete uniformly weighted and EWMA-weighted panels, standardize returns, remove the PC1
score directly from the weighted observation matrix, recompute residual correlations, and
compare with matrix deflation. Maximum absolute error must be `<= 1e-12`.

## Stage D2 — thread the transform through estimator and rolling analytics

Files:

- `FactorLasso/factorlasso/lasso_estimator.py`;
- `FactorLasso/factorlasso/cluster_smoothing.py`;
- `FactorLasso/tests/test_cluster_pc1_removal.py`;
- existing focused tests only where a regression belongs beside an existing contract.

Work:

1. Add `cluster_correlation_transform` to `LassoModel`, its docstring, validation, cloning,
   `get_params`, and `set_params` coverage.
2. In direct HCGL/FCGL fits, transform the dependence matrix after the fit universe is known and
   before distance/linkage.
3. In rolling clustering, calculate raw dependence causally, restrict to the exact eligible
   assets at `t`, transform that submatrix, then apply smoothing.
4. Apply `SIMILARITY_EWMA` to the sequence of residual correlations. Do not smooth raw
   correlations and remove a single PC afterwards.
5. Apply `PARTITION_BONUS` after residual-correlation distance construction.
6. Use the residual correlation for held-partition entrant assignment.
7. External injected partitions bypass discovery exactly as before.

Acceptance:

- `NONE` direct fits, linkages, cutoffs, clusters, coefficients, and rolling outputs are
  byte-identical to the D0 baseline;
- direct de-PC1 model partitions equal an independent call to the pure transform followed by
  `compute_clusters_from_corr_matrix` on 100% of test panels;
- rolling `NONE` equals fitted partitions as before;
- rolling de-PC1 equals date-by-date fitted de-PC1 partitions on 100% of dates when smoothing is
  `NONE` and the same eligibility is supplied;
- appending future observations leaves every earlier residual correlation and partition
  unchanged;
- future/ineligible columns cannot change the current de-PC1 result;
- all smoother zero-strength and scheduled-anchor invariants still pass.

## Stage D3 — public API, versioning, and full FactorLasso verification

Because the estimator and rolling signatures are public, update in the same change:

- `FactorLasso/CHANGELOG.md`;
- version `0.14.0 -> 0.15.0` in `pyproject.toml`, `CITATION.cff`, and the README BibTeX entry;
- README clustering example and API description.

Do not publish, tag, push, or create a release.

Verification commands from the FactorLasso root:

```powershell
pytest tests/test_cluster_pc1_removal.py tests/test_cluster_smoothing.py `
  tests/test_dependence_measures.py -v
pytest --cov=factorlasso --cov-report=term-missing -q
ruff check factorlasso/ tests/
python -c "import factorlasso, sys; print(factorlasso.__file__); `
print([m for m in ('qis','optimalportfolios','sklearn') if m in sys.modules])"
```

Acceptance:

- focused and full suites pass;
- coverage remains at least 90%;
- Ruff is green;
- no banned stack or module-level scikit-learn import appears;
- three version locations equal `0.15.0`;
- build/install test is deferred unless the owner asks for a FactorLasso release.

Deliverable: `2026-08-XX_sol_dePC1_factorlasso_implementation_report.md` in the paper agents
directory, with exact commands and outputs.

## Stage D4 — isolated three-universe partition experiment

Add paper-local code only under:

- `papers/cluster_lineage_2026/replication/run_depc1_cluster_comparison.py`;
- `papers/cluster_lineage_2026/replication/depc1_cluster_comparison_test.py`.

Cache under a new root only:

```text
$CLUSTER_LINEAGE_OUTPUT_DIR/depc1/<universe>/<raw_or_depc1>/<config>/YYYYMMDD.pkl
```

Every cache fingerprint includes:

- input data hashes and date schedule;
- point-in-time eligibility hash;
- local FactorLasso source hash/version;
- raw versus `remove_pc1` transform;
- frequency, span, dependence measure, distance transform, linkage, cutoff;
- smoother type and unchanged smoother parameter.

Per-date diagnostics:

- eligible asset count and identity;
- raw PC1 eigenvalue, explained share, eigengap, and loading concentration;
- raw versus residual mean/median off-diagonal correlation;
- minimum residual variance and any isolated zero-residual assets;
- raw/de-PC1 cluster counts, singleton share, median cluster size;
- pairwise Rand and ARI between raw and de-PC1 partitions;
- matched-granularity ARI using the raw date-specific cluster count as a diagnostic only;
- taxonomy ARI for U1 BICS sector, U2 BlackRock asset class/sub-asset class, and U3 futures
  asset class;
- lineage churn, tracks per asset, fragmentation, births, deaths, splits, and merges.

Primary comparison is the fixed operating-point cut. Matched-granularity partitions are
diagnostic only and never feed the strategy backtest.

Acceptance:

- raw and de-PC1 use identical dates and asset sets on 100% of snapshots;
- every excluded or ineligible asset has zero partition membership and strategy weight;
- no PC1 calculation includes an asset outside the current eligible set;
- raw partitions reproduce frozen cache partitions exactly;
- de-PC1 injected partitions equal de-PC1 fitted partitions on 100% of dates;
- cache-first replay is byte-identical;
- no existing E2/E3/E5 cache is touched.

## Stage D5 — fixed-strategy backtests

Add:

- `papers/cluster_lineage_2026/replication/run_depc1_strategy_backtests.py`;
- `papers/cluster_lineage_2026/replication/depc1_strategy_backtests_test.py`.

For each universe, run exactly:

1. frozen raw-correlation cluster strategy;
2. same strategy with de-PC1 clusters and every other parameter unchanged;
3. frozen matched global comparator;
4. U1 sector comparator and U2 pure-cluster diagnostic where already declared.

Report gross and net annual return, RF=0 Sharpe, volatility, beta/alpha where already defined,
one-way turnover, cost drag, total P&L, and per-side/per-asset contribution. Explicit deltas:

- de-PC1 cluster minus raw cluster;
- de-PC1 cluster minus global;
- raw cluster minus global;
- for U1 only, each cluster leg minus sector.

The global weights, turnover, costs, and NAV must be byte-identical between the raw and de-PC1
runs. EW-all remains a market/alpha reference only and is never a ranking-performance yardstick.

Mechanics acceptance:

- maximum signal lookahead days `<= 0`;
- weights/exposures sum to their frozen targets within `1e-12`;
- costs equal 10 bp for U1/U3 and 20 bp for U2;
- U2 uses AUM strictly above USD 100m on every decision date;
- U3 applies all seven frozen exclusions on every decision date;
- global comparator numerical difference across arms `<= 1e-12`;
- two complete cache-first runs produce byte-identical numerical artifacts;
- focused tests and isolated Ruff pass.

There is deliberately no performance acceptance requirement. “de-PC1 must beat global” is the
hypothesis under test, not a software gate. A loss or deterioration is a valid result and must
not trigger parameter search.

## Stage D6 — exhibits and report

Machine-readable deliverables per universe:

- `pc1_diagnostics.csv`;
- `partition_comparison.csv`;
- `cluster_metric_summary.csv`;
- `lineage_comparison.csv`;
- `performance.csv`;
- `performance_comparison.csv`;
- `turnover_and_costs.csv`;
- `instrument_pnl.csv`;
- `acceptance.csv`, `runtime.csv`, `determinism.csv`, and source manifest.

Exhibits:

1. PC1 explained share through time;
2. raw versus de-PC1 cluster count and taxonomy ARI;
3. raw/de-PC1 partition heat map at the final date;
4. raw cluster, de-PC1 cluster, and global cumulative net NAV;
5. de-PC1-minus-raw instrument P&L contribution scatter/table.

Final report:

`papers/cluster_lineage_2026/agents/2026-08-XX_sol_dePC1_cluster_and_strategy_report.md`

It must distinguish:

- common-mode removal changing correlation geometry;
- changes caused only by cluster-count/granularity;
- turnover/cost effects versus gross-payoff effects;
- full-window evidence versus independent split-window robustness;
- equity “market mode” interpretation versus the more neutral “dominant common mode” for funds
  and futures.

End with an owner gate asking only whether de-PC1 becomes a primary, robustness, or rejected
clustering specification. No automatic adoption follows from this roadmap.

## Execution order

```text
D0 baseline/provenance
  -> D1 pure transform
  -> D2 estimator + rolling integration
  -> D3 FactorLasso verification/version record
  -> D4 partitions and cluster analytics (U2 -> U3 -> U1)
  -> D5 fixed strategy backtests (U2 -> U3 -> U1)
  -> D6 exhibits/report/owner gate
```

U2 runs first because the AUM100 primary rule is now frozen and its universe is smaller than U1.
U3 follows to exercise M1-star smoothing and frozen exclusions. U1 runs last because it is the
largest panel and provides the strongest taxonomy test.

## Out of scope

- covariance-basis PC removal;
- removing more than one component;
- RMT noise-bulk filtering or nearest-correlation projection;
- signal, q, AUM, sleeve-weight, smoother-delta, cutoff, or cost optimization;
- refitting the global comparator;
- modifying frozen baseline caches;
- editing FactorLasso estimator defaults other than adding the default-off field;
- staging, committing, tagging, pushing, publishing, or releasing either repository.
