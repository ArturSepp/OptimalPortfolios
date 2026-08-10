# Changelog

All notable changes to optimalportfolios are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `examples/solvers/risk_budgeting.py` built a `GroupLowerUpperConstraints` and never passed it
  to `Constraints`, so the example's stated 10%–30% asset-class bounds were not applied to the
  solve it demonstrates. Found as an unused-variable finding, not by reading the output.
### Changed
- Modernized packaging licence metadata to the PEP 639 SPDX form (`license = "MIT"` and
  `license-files = ["LICENSE.txt"]`), with no change to the legal licence or package behavior;
  originally proposed in PR #9 by @tschm.
- `examples/data/test_data.py` is renamed to `examples/data/etf_prices_local.py`. It matched
  pytest's `test_*.py` pattern, so it was imported at collection while contributing no tests —
  the mechanism behind two past CI failures from module-level optional-extra imports. The
  `load_test_data` and `update_test_prices` functions are unchanged; only the module path moves,
  and the six `*_local.py` diagnostics that import it are updated. Every file matching either of
  pytest's default patterns now collects at least one test.
- Ported three content improvements from PR #38 by @tschm into the retained Sphinx pages: the
  6.12.0 optional-extras truth, the quickstart's "What to change first" guidance, and the
  landing-page overview and publication links.

## [6.12.0] - 2026-08-10

**Behaviour change:** Dispatching `AlphaSignal.RESIDUAL_MOMENTUM` and
`RESIDUAL_MOM_AND_BETA` through `compute_signal_scores` previously raised
`TypeError: compute_residual_momentum_alpha() got an unexpected keyword argument
'momentum_span'`. PR #23 by @tschm makes both routes run and honour the requested
`momentum_span`. At span 4 versus the receiver default 12, every overlapping score changed;
the measured mean absolute differences were 0.381535895168 score units for residual momentum
and 0.269786618740 for the composite signal.

### Added
- Added a module-scope root-package import-cycle regression guard, adapted from PR #22 by
  @tschm, while retaining function- and method-local import support.
- Restored 37 lineage-event tests under a non-colliding name and expanded the dev suite to
  1,111 tests in PR #23 by @tschm. Fresh line coverage is 89.696766627576%, and the ratchet
  floor is 88 (`floor(measured) - 1`).
- Added a pinned `pip-audit` CI gate over the dependency tree resolved from `pyproject.toml`
  in PR #31 by @tschm.
- ROSAA model execution supports an inert-by-default desk-instruction correction dead-band.
  Current Min/Max and explicit minimum, maximum or fixed-target breaches inside the configured
  tolerance are retained and audited without weakening product or lifecycle constraints.

### Fixed
- Three-or-more-dimensional covariance arrays are labelled as DataFrames before
  `qis.covar_to_corr`, preserving asset labels, in PR #23 by @tschm.
- `plot_mixure2` now unpacks the Matplotlib figure and axes correctly in PR #23 by @tschm.

### Changed
- Every subpackage `__init__.py` now declares an `__all__`, stating its export surface
  explicitly for the first time. The public API is unchanged — all 125 names `optimalportfolios`
  exposed before are exposed after — but `from optimalportfolios.<sub> import *` now re-exports
  a written-down list rather than whatever was left in the namespace.
- Ruff configuration moved from `[tool.ruff]` in `pyproject.toml` to a top-level `ruff.toml`.
  Rule selection and behaviour are unchanged.
- CI now gates the `F` (pyflakes) family alongside the three stack invariants. The ~380 `E501`
  line-length findings remain ungated and untouched.
- The S&P 500 span comparison example now uses the project-local output-path helper instead of
  the undeclared private `quant_strats` package, resolving issue #27.
- The Bloomberg S&P 500 universe example now raises an actionable, exception-chained installation
  message when `bbg-fetch` is unavailable, resolving issue #27.
- Renamed the signal-diagnostics test to the repository's `*_test.py` convention in PR #26 by
  @tschm.
- Extended the CI test matrix through Python 3.13 in PR #30 by @tschm; Python 3.12 remains the
  single coverage-gated matrix entry.
- Covariance factorization is now owned exclusively by each low-level CVXPY solver. Wrappers pass
  only `factorize_covar`; solver APIs no longer accept a precomputed factorization, and
  `resolve_covariance_factorization` has been removed. Each enabled solve calls
  `factorize_covariance` directly once and reuses the result internally.
- Factorization-capable single-date optimiser wrappers now have one fixed return contract:
  `(weights, outcome)`. The optional outcome and detailed-output branches were removed; callers
  compute presentation-specific risk-contribution tables separately.
- `validate_solution` now always returns `OptimizationOutcome`. Its legacy tuple return and
  `return_outcome` switch were removed; validated weights and acceptance state are available as
  `outcome.weights` and `outcome.accepted`.

### Removed
- Removed the unused `plotly` dependency and `visualization` extra because no shipped package code
  imports Plotly, resolving issue #27.
- Removed `pandas-datareader` from the `data` extra because no shipped package code imports it,
  resolving issue #27.
- Removed the direct `jinja2` declaration because `pybloqs` already installs it transitively,
  resolving issue #27.
- Removed the orphaned `uv.lock` in PR #32 by @tschm; project dependencies remain declared in
  `pyproject.toml`.

## [6.11.0] - 2026-08-09

**Behaviour change:** risk-cluster matching now has a dedicated `clustering` extra. The default
`mcf` matcher raises a guarded `ImportError` that names `optimalportfolios[clustering]` and the
dependency-free `hungarian` alternative when NetworkX is absent, instead of leaking a raw
`ModuleNotFoundError`.

`estimate_rolling_mixture` now defaults to `n_components=2` (the previous default of three
raised `IndexError` in its two-regime extraction), raises an explicit `ValueError` for other
component counts, and documents and tests its ascending-by-mean output order.

### Added
- The `clustering` extra installs NetworkX for minimum-cost-flow risk-cluster matching.
- Risk-labelling tests compare both matchers with SciPy and brute-force independent references;
  package coverage is now ratcheted at 61% after measuring 62.97%.
- A Sphinx documentation build provides installation guidance, an offline quickstart, and an
  autosummary inventory of every package-root public name; Read the Docs builds it on Python 3.11.

### Changed
- Numerical module headers now state units, conventions, entry points, and package boundaries.
  Docstring coverage remains 100%, with an AST gate enforcing the package-wide Google style.
- CI action revisions and the Ruff minor series are pinned. CI gates stack import invariants,
  docstring coverage, the coverage floor, the full Python 3.10–3.12 suite, and a core install.
- Package-internal root imports were replaced by direct module imports, breaking avoidable import
  cycles without changing the public API.
- Fifteen local plotting and diagnostic scripts no longer use pytest's `*_test.py` naming.
- MIT licensing and package metadata were audited for JOSS and PEP 639 alignment. The existing
  OSI-approved MIT file and consistent `license = {text = "MIT"}` metadata are retained by explicit
  maintainer decision; this release does not change the licence.

## [6.10.0] - 2026-08-09

**No numerical behaviour changes to existing entry points.** The delegated results are pinned
against pre-change characterisation goldens and independent covariance identities at
`rtol=1e-12`.

### Added
- `build_risk_model` adapts rolling factor covariance data, dated current factor snapshots, and
  covariance-only dictionaries to the canonical `qis.RiskModel` computation layer.
- Minimum-tracking-error optimisation now provides paired single-date and rolling entry points;
  the rolling pipeline supports static or time-varying benchmarks, prior-weight drift, and
  point-in-time asset inclusion indicators.
- `compute_eligible_rebalancing_bounds` derives auditable current-to-model implementation
  corridors and rebalancing indicators from instrument-level limits.

### Changed
- `PortfolioOptimisationResult` delegates tracking error, factor/residual tracking-error
  decomposition, and factor exposures to `qis.RiskModel`; output values, labels, and shapes are
  unchanged and characterisation-tested.
- The minimum `qis` version is now 5.7.0, which provides `qis.RiskModel` and `qis.WEIGHT_TOL`.

## [6.9.0] - 2026-08-03

### Added
- `CovarianceFactorization`, `factorize_covariance` and
  `resolve_covariance_factorization` provide one reusable covariance eigendecomposition per
  optimisation solve. The result contains the symmetric positive-semidefinite covariance and its
  square-root factor, together with the raw and stabilised minimum eigenvalues, condition numbers,
  eigenvalue floor and maximum adjustment for audit reporting.
- Structured solver diagnostics through `ConstraintResidual`, `OptimizationOutcome` and
  `evaluate_constraint_residuals`. Callers can request the complete optimisation outcome while
  existing callers continue to receive the original weight output by default.
- `RunDiagnostics` aggregates rejected solver attempts, relaxed group constraints, covariance and
  input-contract diagnostics, zero-loading groups and deduplicated Python warnings for production
  reporting.
- Focused covariance-factorisation tests cover positive-definite, singular and slightly indefinite
  covariance inputs, factor reuse, and equivalence with the legacy quadratic-form geometry.

### Changed
- `OptimiserConfig.factorize_covar` defaults to `True`. Supported CVXPY optimisers now factorise
  the covariance once and reuse that exact factor in their objective, risk constraints and
  post-solve validation: quadratic utility, maximum Sharpe, maximum return at target volatility,
  minimum variance at target return, maximum alpha at target yield, and maximum alpha over
  tracking error. Set the flag to `False` to retain legacy `quad_form` construction.
- Covariance-based constraints accept an optional precomputed factorisation. Volatility, tracking
  error and group tracking-error limits use second-order-cone norms when it is supplied; utility
  tracking-error penalties use the same factorised sum-of-squares representation.
- Input validation reports both the raw covariance condition and the matrix actually supplied to
  the optimiser. Run logging is idempotent, writes UTF-8 output and summarises repeated warnings.

### Fixed
- Singular and numerically indefinite covariance matrices are stabilised with a controlled
  eigenvalue floor before CVXPY canonicalisation. This keeps optimisation and compliance checks on
  identical risk geometry and avoids discrepancies caused by independently refactorising the same
  covariance matrix.
- Solver failure messages now identify the rejected solver attempts and exact residuals instead of
  losing the useful diagnostics behind a generic validation error.

## [6.8.0] - 2026-08-02

**No number moves for any existing caller.** Every span parameter still accepts the scalar it
took before and applies it unchanged at every reporting cadence; the per-cadence mapping is
opt-in and additive.

### Added
- Signal spans accept a per-cadence mapping. `long_span`, `short_span`, `vol_span` and
  `beta_span` on `compute_momentum_alpha`, `compute_low_beta_alpha`,
  `compute_residual_momentum_alpha`, `compute_residual_reversal_alpha`,
  `compute_ra_carry_alpha` and their `*_cluster_alpha` siblings now take either an `int` or a
  `Mapping[str, int]` keyed by reporting cadence. A span is a number of PERIODS and every one
  of these signals estimates one frequency bucket at a time
  (`_compute_raw_*_mixed_freq` loops `qis.get_group_dict`), so a single scalar meant different
  calendar time in each bucket: `long_span=12` is one year of monthly returns and three years
  of quarterly ones, with nothing in the signature saying the unit changed between calls.
  A mapping such as `{'ME': 12, 'QE': 4}` gives every cadence the same calendar horizon.
  Resolution happens at each site that has a scalar cadence in scope and is about to call a
  `_compute_raw_*_single_freq`: inside the mixed-frequency bucket loop, and on the
  single-frequency branch of each entry point. Those raw functions keep an `int` signature and
  never see a mapping. The cross-sectional score is unaffected - it is still computed across the
  whole universe after the buckets are merged.
- `resolve_span(span, freq, name)` in `optimalportfolios.alphas.signals.utils`, the one place
  that decides. `None` passes through, so an optional span stays disableable; a cadence absent
  from the mapping raises rather than inheriting another cadence's entry, which is the scalar
  behaviour being removed.
- `optimalportfolios/alphas/signals/tests/per_cadence_spans_test.py`: 50 checks. Two of them
  are the contract - a scalar is bit-identical to a flat mapping, and a mapping whose `'ME'`
  entry equals the old scalar leaves every monthly column bit-identical while moving the
  quarterly ones. Both are asserted across all five signals and across BOTH dispatch branches:
  `compute_*_alpha` reaches `_compute_*_alpha_mixed_freq`, while `compute_*_cluster_alpha` is
  the only caller of `_compute_raw_*_mixed_freq`, so the cluster entry points are the only
  coverage its bucket loop has. `warmup_period` is derived from the span inside each signal, so
  a per-cadence span carries a per-cadence warmup with it and no second table is needed.
- `optimalportfolios/tests/concat_sort_convention_test.py`: an `axis=1` `pd.concat` in library
  code without an explicit `sort=` fails the suite. What the union of two DatetimeIndexes does
  when the argument is absent has changed twice in two major pandas versions, and the difference
  is a scrambled time axis rather than an error.

### Fixed
- Every `axis=1` `pd.concat` in library code states `sort=` explicitly - 40 call sites in 14
  modules. `sort=True` where the joined index is dates - navs, signal scores, residuals -
  and `sort=False` where it is assets, groups or mixture clusters. pandas 2.2 sorted the union
  of two DatetimeIndexes whatever the argument said; pandas 3.0 honours an explicit
  `sort=False` and leaves the union in appearance order, and pandas 4 drops the implicit sort
  too. Two sites join date indexes that differ by construction - the per-frequency residual
  blocks in `estimate_lasso_factor_covar_data` and the per-frequency excess returns in
  `managers_alpha` - and would have handed an unsorted panel downstream with nothing raising.
  No number moves: the suite is unchanged.

## [6.7.0] - 2026-07-31

### Added
- `align_rolling_clusters(rolling_clusters)` in `optimalportfolios.alphas.signals.utils`, exported
  from the package. `compute_clusters_from_corr_matrix` returns `scipy.cluster.hierarchy.fcluster`
  labels, whose numbering follows the dendrogram traversal and is re-derived independently at
  every estimation date — so `'QE:4'` at one date and `'QE:4'` at the next are unrelated, and a
  time series of the raw labels shows migrations that never happened. This walks the dates forward
  and relabels each partition to the previous one by maximum overlap
  (`scipy.optimize.linear_sum_assignment` on the contingency matrix), returning the aligned
  assignments plus a per-date count of instruments whose cluster genuinely changed. Alignment runs
  within each frequency prefix, because the estimator partitions each frequency bucket separately.
  Measured on the MAC universe (185 responses, 283 monthly estimation dates): the raw labels imply
  140.5 moves per date, the aligned ones 20.5 — 85% of the apparent churn is renumbering.

### Changed

- `optimalportfolios/alphas/profile/profile_alpha_signals.py` moved to
  `optimalportfolios/examples/alphas/profile_alpha_signals.py`. It was an example — its own
  docstring said so — living inside a library package directory, with a module-level `yfinance`
  import in a tree that must import on a core install. Nothing imported it, so no public name
  changes; `optimalportfolios.alphas.profile_alpha_signals` now unambiguously resolves to the
  exported function rather than being shadowed by a same-named module.
- Ruff now enforces three stack invariants that were previously prose: `TID251` bans an import of
  a subject package, `TID253` bans a module-level import of an optional extra, and `ICN` pins the
  `numpy`/`pandas` aliases. Green on the repository as it stands; see `AGENTS.md`.

## [6.6.0] - 2026-07-28

**`estimate_rolling_ewma_covar` is now `qis.estimate_rolling_ewma_covar`.** This package carried
its own implementation of a function `qis` already exports and documents in its core API, with a
near-identical signature — two same-named estimators, in two packages, one depending on the other,
free to drift apart with nothing failing. The local copy is deleted and the name is re-exported
from `qis`, so `from optimalportfolios import estimate_rolling_ewma_covar` keeps working and now
resolves to one implementation instead of two.

Measured before the swap, on the committed `multiasset` fixture over 2007-02 to 2026-05, 77
rebalancing dates:

| configuration | max abs difference |
|---|---|
| `returns_freq='ME'`, `rebalancing_freq='QE'`, `span=24` | **0.0** — bit-identical |
| `returns_freq='ME'`, `rebalancing_freq='YE'`, `span=36` | **0.0** — bit-identical |
| as above with `demean=False` | **0.0** — bit-identical |
| as above with `is_apply_vol_normalised_returns=True` | 5.1e-05 (3.1e-04 relative) |

The one difference is an EWM warm-up artefact and its cause is known: this package's
`compute_returns_from_prices` drops one extra leading row, because after demeaning the first row
is structurally zero, while `qis` keeps it. The demeaned series are bit-identical over their
common tail. The plain covariance path has forgotten the extra observation long before the first
rebalancing date, which is why three of the four configurations agree exactly; the vol-normalised
path divides by a rolling volatility that carries the warm-up in a ratio, so a residual survives.

**A result computed with `is_apply_vol_normalised_returns=True` will move in the fourth decimal.**
Nothing in this package took that path: every internal call site passes the default `False`, and
`EwmaCovarEstimator` and `FactorCovarEstimator` have their own vol-normalisation code that this
change does not touch. Two examples pass `True` and are diagnostic scripts.

Two smaller behaviour changes come with the swap. The local version accepted `**kwargs` and
silently ignored unknown keywords; the `qis` function does not, so a misspelled argument now
raises instead of being dropped. The local version raised `ValueError` when the rebalancing
schedule came out empty; `qis` returns an empty dict instead, so a caller relying on that message
gets a quieter failure.

### Removed
- `optimalportfolios.covar_estimation.ewma_covar_estimator.estimate_rolling_ewma_covar`, the
  local implementation. The name still resolves — it is `qis`'s function now.

### Changed
- `optimalportfolios/tests/public_api_test.py` drops its `estimate_rolling_ewma_covar` entry from
  `QIS_COLLISION_ALLOWLIST`. That test is what found the duplicate, and its staleness check is
  what forced the entry out once the two names became one object.

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
  `examples/data/test_data.py` and `universe/tests/universe_data_local.py`. Both
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
