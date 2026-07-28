# Changelog

All notable changes to optimalportfolios are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [6.5.0] - 2026-07-28

**6.3.0 and 6.4.0 were written up below but never published.** No tag and no
PyPI release was cut for either, so the last version installable from PyPI is
6.2.0 (2026-07-17) and everything documented under 6.3.0 and 6.4.0 — the
`DistanceTransform`, `DependenceMeasure`, `compute_dependence_matrix` and
`compute_gerber_matrix` re-exports from factorlasso — first reaches users here.
**No library behaviour changes in 6.5.0 itself**: the solver, covariance and
backtest paths are unchanged from the 6.4.0 tree.

### Fixed
- `alphas/tests/test_signal_diagnostics.py` imported `plot_signal_diagnostics`
  and `plot_signal_diagnostics_per_component`, which 6.3.0 removed from
  `alphas/signal_diagnostics.py` when the compute-and-plot wrappers moved to
  qis. The module has raised `ImportError` on collection since then; nothing
  reported it because `testpaths` pointed at a directory that is not in the
  repository and CI ran six files by hand. The nine tests covering the moved
  wrappers are dropped — they belong to qis now — and the seventeen covering the
  `AlphasData` adapter, the per-component sweep and the comparison aggregation
  are collected again.
- `yfinance` is no longer imported at module scope in
  `examples/data/test_data.py` and `universe/tests/universe_data_test.py`. Both
  imports are function-local and raise `ImportError` naming the `[data]` extra,
  so a core install collects the suite instead of erroring. `yfinance` remains a
  test-and-example dependency; no library module imports it.
- `[tool.ruff] exclude`, `[tool.coverage.run] omit` and
  `[tool.setuptools.packages.find] exclude` still named `paper_code`, which
  6.2.0 renamed to `papers`. ruff was linting the replication code the
  configuration intends to skip.
- The 6.4.0 entry below recorded the factorlasso floor as `>=0.10.0,<0.11` and
  the 6.3.0 entry as `>=0.9.0,<0.10`. Neither bound was ever in
  `pyproject.toml`, which went from `>=0.8.0,<0.9` to `>=0.10.1` in one step.
  Both lines are corrected in place. The floor is and remains `>=0.10.1`, which
  is where `compute_clusters_from_corr_matrix` stopped raising when
  `n_clusters` exceeds the universe size — the failure mode of a rolling
  factor-covariance fit over a growing universe. No upper bound is declared,
  matching the `qis` floor, whose `<6` cap was dropped in 6.2.0.

### Added
- `optimalportfolios/tests/version_metadata_test.py`: `pyproject.toml`,
  `CITATION.cff` and the `@software` BibTeX entry in `README.md` must carry the
  same version, and `date-released` must be an ISO date. At 6.4.0 the three read
  6.3.0, 6.2.0 and versionless.
- `CITATION.cff` carries the author ORCID iD 0000-0002-7038-1748; the
  `@software` entry in `README.md` carries `version` and a current year.

### Changed
- `[tool.pytest.ini_options] testpaths` is `["optimalportfolios"]`, replacing a
  pointer at a top-level `tests/` directory that is not in the repository. A
  bare `pytest` at the repository root collects 180 tests across seven modules
  and passes on a core install with no data, network or terminal access. The
  other sixteen `*_test.py` files are `run_local_test` diagnostic scripts and
  contribute no collected tests; they are imported during collection and must
  stay importable without the optional extras.
- CI runs `pytest` instead of six hand-picked `python <file>.py` invocations.
  The import-verification and factorlasso re-export checks stay as separate
  steps, and the suite runs on a core install as well as on `[dev]`.
- ruff no longer selects `I`. Import order in this stack groups the scientific
  stack before project packages, which isort's ordering contradicts; the rule
  was selected here and in no other repository of the stack.

## [6.4.0] - 2026-07-24

### Added
- `DependenceMeasure`, `compute_dependence_matrix` and
  `compute_gerber_matrix` re-exported from factorlasso (>= 0.10.0):
  the dependence measure that builds the clustering correlation matrix
  (`pearson` default, `spearman`, `gerber`). No behavioural change —
  the `LassoModel` fields `dependence_measure`, `gerber_threshold` and
  `n_clusters` flow through `FactorCovarEstimator` and `LassoModel.copy`
  generically, and the defaults reproduce the pre-0.10.0 clustering
  exactly.

### Changed
- factorlasso dependency floor raised to `>=0.10.1` (was `>=0.8.0,<0.9`) to
  admit the dependence-measure parameters. Corrected in 6.5.0: this line read
  `>=0.10.0,<0.11` (was `>=0.9.0,<0.10`), a bound `pyproject.toml` never
  carried.

## [6.3.0] - 2026-07-22

### Added
- `DistanceTransform` re-exported from factorlasso (>= 0.9.0) alongside
  `LassoModel` / `LassoModelType`: the correlation-to-distance transform
  for the clustering step of the factor covariance estimation
  (`one_minus_rho` default, `chord`, `arccos`). No behavioural change:
  the `LassoModel.distance_transform` field flows through
  `FactorCovarEstimator` and `LassoModel.copy` generically, and the
  default reproduces the pre-0.9.0 clustering exactly.

### Changed
- `distance_transform` requires factorlasso >= 0.9.0. Corrected in 6.5.0: this
  line read "floor raised to `>=0.9.0,<0.10` (was `>=0.8.0,<0.9`)", but
  `pyproject.toml` still declared `>=0.8.0,<0.9` at this point and moved
  straight to `>=0.10.1` in 6.4.0.

## [6.2.0] - 2026-07-17

### Added
- `rolling_quadratic_optimisation` gains `expected_returns` and
  `wrapper_quadratic_optimisation` gains `means` (both keyword, default `None`):
  the expected-return panel / vector for `PortfolioObjective.QUADRATIC_UTILITY`,
  filtered alongside the covariance for NaN / excluded assets. Both are additive
  keyword arguments; existing callers are unaffected.
- Repo-root `tests/` headless suite (69 tests) collected by bare `pytest`,
  covering the public API surface, general / SAA / TAA solvers against
  closed-form optima, the rolling dispatcher contract, EWMA and factor
  covariance estimation, and the utility layer.
  **Correction, 2026-07-28: this suite was never committed.**
  `git log --all --diff-filter=A -- "tests/*"` returns nothing — no `tests/`
  directory has existed at any point in this repository's history, and the
  `[tool.pytest.ini_options] testpaths = ["tests"]` that 6.5.0 replaced was
  pointing at it. The fixture and loader described in the bullet below did land
  and are in the wheel. The five areas named here are genuinely uncovered: at
  6.5.0 they hold 1,443 statements at 31% line coverage. They are being rebuilt
  in the package's own `<subpackage>/tests/` directories rather than at the
  repository root; this entry is left in place rather than rewritten, because a
  published release note that quietly changes is worth less than one that
  carries its own correction.
- Offline multi-asset universe fixture `examples/data/multiasset_returns.csv`
  (19 instruments across Fixed Income / Equity / Alternatives / Liquidity,
  monthly, with Asset Class and Sub Asset Class metadata) and loader
  `examples.data.multiasset.load_multiasset_data` returning a frozen
  `MultiAssetData(returns, prices, group_data, sub_group_data)`. No network
  access, unlike the yfinance-based loaders. The `examples.data` package data
  now ships in the wheel (`*.csv`), which also fixes the previously unshipped
  `dow30_prices.csv`.

### Changed
- `opt_risk_budgeting` backend switched from the vendored `pyrb` fork to an
  internal solver, `optimization.general.risk_budgeting_solver`
  (`solve_constrained_risk_budgeting`): a pure-NumPy CCD / ADMM-CCD
  implementation of the log-barrier formulation of Richard & Roncalli (2019).
  Weights match `pyrb` to within ADMM tolerance (~5e-5 in weight space); parity
  against frozen `pyrb` baselines and the paper's published tables is pinned in
  `optimization/tests/risk_budgeting_solver_test.py`. `set_pyrb_constraints`
  keeps its name and `(bounds, C, d)` contract; `validate_pyrb_solution` is
  renamed `validate_rb_solution` (internal) and adds a group-inequality-row
  check at the ADMM primal-residual scale.
- `compute_rolling_optimal_weights` estimates rolling EWMA means for
  `QUADRATIC_UTILITY` as it already does for `MAXIMUM_SHARPE_RATIO`.

### Fixed
- `QUADRATIC_UTILITY` through the rolling dispatcher raised `ValueError` at the
  first rebalance because `means` were never forwarded to the solver.
- `rolling_risk_budgeting(risk_budget=None)` crashed on the single-asset guard
  (`len(None)`) and on the covariance-to-budget reindex; the equal-budget
  default now works for any universe size.
- `EwmaCovarEstimator.fit_rolling_covars` and `estimate_rolling_ewma_covar`
  ignored `time_period.end`, running the schedule to the end of the price panel;
  both bounds now apply (`end=None` is unbounded). No change when `end` equals
  the last price date.
- `solve_constrained_risk_budgeting` validated inputs after normalising budgets
  and slicing bounds, so a zero-sum budget emitted a NumPy `RuntimeWarning` and
  malformed bounds raised `IndexError` instead of `ValueError`; validation now
  runs on raw inputs, so every invalid input raises `ValueError` (the caller's
  fallback contract) cleanly.

### Removed
- The vendored `pyrb` package (`optimalportfolios/pyrb/`) and its `numba`
  dependency. `quadprog` is retained for the ADMM Euclidean-projection QP. The
  `examples/comparisons/pyrb_vs_scipy.py` demo is renamed
  `risk_budgeting_ccd_vs_scipy.py`.

## [5.4.3] - 2026-06-28

### Added
- `optimalportfolios.optimization.solver_diagnostics` — new module providing
  post-solve feasibility validation and pre-solve conditioning / input checks
  for the CVXPY, scipy and pyrb optimisers. `validate_solution` inspects
  `problem.status` and tests the returned `w.value` against the constraint set,
  rejecting a populated-but-grossly-infeasible iterate (e.g. weights summing to
  ~1.5e6) that the previous `if optimal_weights is None` check accepted;
  `validate_scipy_solution` (SLSQP) and `validate_pyrb_solution` cover the
  non-CVXPY solvers. `check_covar_conditioning` adds a pre-solve
  covariance-conditioning warning, `diagnose_infeasibility` runs an elastic
  minimum-violation LP reporting which box / group bounds must relax (and by how
  much) to make a rejected rebalance solvable while holding full investment and
  long-only fixed, and `validate_solver_inputs` is a pre-solve input contract.
  Run-level logging aggregators (`SolverRejectionSummary`, `RelaxationSummary`,
  `InputContractSummary`) plus `configure_run_logging` / `log_environment` keep
  per-rebalance noise out of the console.
- `OptimiserConfig` fields `diagnose_infeasibility` (default True — on a rejected
  solve run the cheap second analysis; one extra LP per *rejected* rebalance
  only), `validate_inputs` (default True — pre-solve input contract), and
  `max_constraint_relaxation` (escalate a frozen-overhang group-bound relaxation
  to an ERROR log when a single relaxation exceeds the given magnitude; None
  applies no bound).
- `constraints.RelaxationRecord` — frozen dataclass capturing a frozen-overhang
  group-bound relaxation (total / max relaxation and per-group
  `(group, kind, old, new)` items, `kind` in {`group_max`, `group_min`}),
  attached to log records under `extra={"relaxation": ...}` so a handler can roll
  per-rebalance relaxations into one run-level tally instead of flooding the
  console.

### Changed
- Every solver wrapper (`max_sharpe`, `quadratic`, `max_diversification`,
  `carra_mixture`, `risk_budgeting`, both SAA solvers, both TAA tracking-error
  solvers, `maximise_alpha_with_target_yield`) now routes its output through the
  relevant `validate_*_solution` and, when enabled, runs `validate_solver_inputs`
  pre-solve and `diagnose_solver_failure` on rejection. The scipy path derives
  `status` from `res.success` instead of only null-checking the result.
- Raised dependency floors: `qis>=4.3.2` (was 4.2.7) and `factorlasso>=0.7.2`
  (was 0.5.4).
- Top-level `README.md` slimmed (~1,240 to ~120 lines), with the detailed
  optimiser documentation consolidated out of the package tree.

### Fixed
- GROWM (tre=100, turnover=0.2) blow-up. A near-collinear private-asset block
  (two proxies at corr 1.00) made the covariance rank-deficient (cond ~5e12);
  CLARABEL returned a non-optimal iterate summing to ~1.5e6; the `is None` check
  accepted it and one 2021 quarter poisoned every second-moment backtest
  statistic downstream. Because `cvx.psd_wrap` asserts PSD to CVXPY (suppressing
  the DCP convexity check) and modern CLARABEL stays feasible even at cond ~6e14,
  neither a convexity check nor provoking the solver catches it — the fix
  validates the output unconditionally on every solve.

### Removed
- `optimalportfolios/optimization/README.md` (folded into the top-level README).

## [5.4.2] - 2026-06-14

Tags the same commit as 5.4.0 (`41c1fd2`); no source changes.

## [5.4.0] - 2026-06-14

Refactor of the alpha / signal modules to consolidate cluster-aware logic, with
improved API consistency and backward compatibility.

### Added
- New `residual_reversal` signal and a `covar_estimation/risk_labelling` module.
- Cluster scoring via `score_within_clusters`; `extract_rolling_clusters`
  re-exported from `utils` for backward compatibility.
- `optimalportfolios.alphas` and `optimalportfolios.alphas.signals` now export
  the new cluster-aware symbols and helpers.

### Changed
- Folded the cluster-specific modules into their parents: `momentum`,
  `low_beta` and `residual_momentum` now expose both standard and cluster-aware
  entry points (`compute_*_alpha` and `compute_*_cluster_alpha`). Package-level
  imports are preserved.
- Added shared raw-signal helpers and mixed-frequency support across momentum,
  low-beta and residual-momentum to remove duplication and handle per-frequency
  processing.
- `managers_alpha`: rolling-regression excess-return computation now uses an
  as-of (lagged) beta lookup, matches factor returns to asset return periods,
  handles mixed-frequency groups, and avoids KeyErrors when a block produces no
  data. Annualisation is applied conditionally.

### Removed
- The separate `*_cluster.py` modules. Code importing them by file path must
  update; package-level imports are unaffected.

---

Versions prior to 5.4.0 have not been backfilled. Run `git log --tags --oneline`
for release-by-release commit history.
