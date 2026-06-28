# Changelog

All notable changes to optimalportfolios are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
