# Changelog

All notable changes to optimalportfolios are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

**Coverage floor raised (2026-08-13):** `fail_under` rises from `95` to `99`; measured coverage
over the scope narrowed in 6.17.0 is 99.09%, on a suite grown from 1145 to 1277 tests. The
floor rises whenever measured coverage rises, and lowering it requires a dated note here.

### Added

- A `wheel` job in `ci.yml`, the only check that runs against the built artifact rather than the
  editable installation. A subpackage or data file missing from the packaging declarations can
  still escape tests that execute against the `src/` tree. The job builds the wheel, asserts it carries
  its test modules and the offline fixture, installs it with no extras plus pytest, and runs
  `pytest --pyargs optimalportfolios` from outside the checkout — the post-install check AGENTS.md
  documents for users, which nothing had been running. It is what makes shipping the tests inside
  the wheel load-bearing.
- Tests for the three signal-backtest factsheet builders in `alphas/backtest_alphas.py`,
  including a sweep assertion on the recorded target weights rather than on the leg labels,
  which are derived from the sweep values and would look correct even if the span never
  reached `compute_signal_scores`.
- Validation coverage across all three `minimum_tracking_error` entry points, and both
  `max_sharpe` solver paths with an assertion on which solver actually ran — the
  Charnes-Cooper transform and the SciPy SLSQP fallback return the same outcome type, so a
  mis-routed problem is otherwise indistinguishable from a solved one.
- Tests for `residual_reversal`'s four dispatch branches, pinning the raw signal as the exact
  negation of `compute_residual_momentum_alpha`. A dropped negation leaves a signal that still
  scores and still backtests, while buying momentum winners under a module named reversal.
- Tests for the target-yield solver's soft-tracking-error branch, the risk-budgeting solver's
  input validation and box-infeasibility guards, and the mixed-cadence-with-groups path shared
  by `carry`, `low_beta`, `momentum` and `residual_momentum` — the one path that partitions the
  universe twice and merges the cells back, where a dropped or duplicated ticker still yields a
  full-looking score panel.

### Changed

- Adopted the standard `src/` layout proposed in issue #60, moving the import package to
  `src/optimalportfolios/` while preserving the `optimalportfolios` import name and public API.
  Packaging, tests, linting, coverage, documentation and CI now resolve the new source root; the
  clean-wheel job continues to verify the actual distribution independently of the editable install.
- Declared `permissions: {contents: read}` in all four workflows. They previously carried no
  `permissions:` block at all, so every job inherited the repository's default `GITHUB_TOKEN`
  scope — up to write access on contents, issues and packages — across sixteen jobs that only check
  out a tree and run a tool. No step in any workflow uses the token: there is no `gh` call, no
  artifact upload, no PR comment and no push, and the network lane's report is a `$GITHUB_STEP_SUMMARY`
  file write, which needs no permission. Declared at workflow level so a new job inherits the
  restriction rather than the default.
- Moved the repository-only worked examples from `optimalportfolios/examples/` to root-level
  `examples/`, making them visible beside the package while keeping them out of wheels and sdists.
  Example and local-diagnostic imports, documentation links, the workflow runner, and packaging,
  lint, and coverage exclusions now follow the new location.
- Replaced `constraints.txt` with a committed `uv.lock`. The release step that regenerated the
  constraints file is now `uv lock`. The ubuntu/3.12 coverage cell syncs `--locked`, which also
  fails when the lock has drifted from `pyproject.toml`; the remaining matrix cells sync
  `--upgrade` and keep floating.
- Moved `ruff` and `interrogate` out of the `dev` extra into a `lint` dependency-group. They were
  previously declared twice, as open floors in `pyproject.toml` (`ruff>=0.4`) and as the real pins
  in the workflow (`ruff~=0.16.0`), so a contributor's ruff and CI's ruff could disagree about the
  same file. CI now takes its versions from the group. The group is not an extra and is not synced
  by default, so it reaches neither a user install nor the test matrix.
- Split CI into four workflows by what each check depends on: `static.yml` (ruff, interrogate) on
  every pull request and repository branch push, `audit.yml` on a daily schedule and dependency
  changes, `examples.yml` for unattended examples, and `ci.yml` for the test matrix. Moved the
  installer onto uv, dropping `actions/setup-python`.
- Removed the inline public-import-surface step from `ci.yml`. It imported five public names and
  asserted that one of the nine factorlasso re-exports was its source object;
  `src/optimalportfolios/tests/public_api_test.py` checks every public name and all nine re-exports on
  every matrix cell, so the inline check was strictly weaker in a job the suite already covers.

- Added `examples.yml`, which executes the example scripts — the one part of the tree nothing else
  runs, since `examples/` is excluded from wheels, dropped by `[tool.coverage.run] omit` and never
  collected by pytest. Two lanes, classified by `.github/scripts/run_examples.py` walking each
  example's import closure rather than from a hand-kept list: 5 offline examples gate pull
  requests on all three runners against a *core* install, and 18 that download from Yahoo Finance
  run daily, advisory-only. Three local-data or Bloomberg diagnostics were renamed `*_local.py`
  and are excluded from both lanes, leaving 23 unattended examples.
- Removed the `core-install` job and the `multiasset_saa.py` smoke test from `ci.yml`.
  `core-install` existed to show the suite passes with no optional extras, which mattered while
  `[dev]` pulled in `optimalportfolios[data]`; now that `[dev]` is pytest and pytest-cov over the
  core tree, all twelve cells of `test` establish that directly. The Python 3.12 cell on each
  operating system also asserts that all seven banned optional modules are absent, retaining the
  platform-specific dependency guarantee without running the complete suite twice. `audit.yml`
  complements that installed-environment check with scheduled fresh core and `[dev]` resolutions,
  which can detect dependency drift without a repository event. The example smoke test moved to
  `examples.yml`, which runs it and four others on three runners rather than on one cell.

### Removed

- Removed the `jupyter` extra (`jupyter`, `notebook`, `jupyterlab`) and dropped it from `[all]`,
  which is now `[data,reports]`. Nothing in this package imports any of the three and the
  repository contains no notebook, so the extra declared notebook tooling rather than a runtime
  integration. It also pulled the unused notebook stack into the daily `--all-extras` audit.
  Users who want a notebook should install one alongside the package. The remaining runtime
  integration extras, `data` and `reports`, correspond to features that import their dependencies;
  `dev` and `docs` remain contributor toolchains.

### Fixed

- Kept a missing manager return local to that manager in the rolling regression-alpha signal,
  instead of deleting the same date from every manager in its reporting-cadence bucket. The old
  row-wide guard shortened all histories to the latest common starting date and could silently
  suppress otherwise valid optimiser alpha observations.
- Rewrote the extras section of `docs/installation.rst`, which documented two extras that do not
- Rewrote the extras section of `docs/installation.rst`, which documented two extras that do not
  exist. It described a `clustering` extra and told users to install it "when using the `mcf`
  cluster matcher" — but that matcher's NetworkX backend was replaced by a SciPy bipartite
  assignment in 6.13.0 and the analytics moved to `factorlasso`, so `mcf` has run on a core
  install ever since; the advice sent users after a package that no longer exists to enable
  something already working. It also claimed `[dev]` bundles `data` and `clustering` plus the lint
  and docstring tools, none of which has been true since `[dev]` was narrowed to pytest and
  pytest-cov and the lint tools moved to the `lint` dependency-group. The page now lists the five
  real extras and states why `jupyter` and `clustering` are absent.

- Added an opt-in strict mode to the shared rolling-solver NaN filter, and enabled it for objective
  vectors in the quadratic, maximum-Sharpe and alpha optimisers. Non-finite expected returns or
  alphas now exclude their asset before reaching CVXPY, while constraint validators and callers
  with an explicit zero-fill contract retain their prior behaviour. This repairs the live
  optimiser and target-return examples while preserving their full histories.
- Fixed `LassoModelType.GROUP_LASSO_CLUSTERS`, which no longer exists — it is
  `HIERARCHICAL_CLUSTER_GROUP_LASSO`. The stale name appeared in two examples, a docstring in
  shipped code (`covar_estimation/factor_covar_estimator.py`), `covar_estimation/README.md`,
  `docs/alphas_module_readme.md` and a `_local.py` dispatcher. Nothing caught it: no test executes
  `examples/`, and to a linter an enum member that was renamed upstream is a valid attribute
  access. `papers/` is left as-is per AGENTS.md.
- Rewrote `examples/alphas/profile_alpha_signals.py` for the current `alpha_scores` mapping API;
  it now computes carry, low-beta and momentum through their canonical signal functions before
  passing the three named panels to the joint profiler.
- Renamed the tracking-error decomposition, S&P 500 span sweep and S&P 500 universe builder to
  `*_local.py`, because each requires persisted local CSV data or a Bloomberg terminal and cannot
  run unattended. The S&P 500 resource path now uses `pathlib` instead of doubled separators.
- Pinned the three gating tools exactly instead of to a series: `ruff==0.16.2`,
  `interrogate==1.7.0` and `pip-audit==2.10.1`, the last moved out of a `uvx --from` range in
  `audit.yml` into a new `audit` dependency-group. `~=0.16.0` admits any patch release, and a ruff
  patch may add or fix a rule, so unchanged source could change verdict with nothing in this
  repository having moved — the reproducibility the comments claimed was not what the specifiers
  delivered. `pip-audit` additionally ran through `uvx`, which resolves fresh and bypasses
  `uv.lock` entirely; it now runs `uv run --locked --only-group audit`.
- Removed the job-level fork guard from `static.yml` and made `pull_request` unconditional. The
  guard skipped the job for same-repository PRs, and GitHub reports a conditionally skipped job as
  **successful** — so the required PR check could go green without ruff or interrogate having run.
  A same-repo PR branch is now checked on both push and pull_request; the duplicate is a
  seconds-long job with no install, which is the correct thing to pay for a sound gate.
- Retried `uv sync` in the CI matrix, which has failed a cell outright on transient DNS resolution
  errors against `files.pythonhosted.org`. uv's own per-request retries do not cover a resolver
  that gives up mid-flight, so the retry wraps the command: three attempts with 15s and 45s
  backoff.
- Preserved the three-OS optional-dependency guarantee after removing the duplicate core suite:
  the Python 3.12 test cells now check all seven `banned-module-level-imports` names in the actual
  Linux, Windows and macOS environments, while the daily audit independently resolves fresh trees.
- Reduced the `dev` extra to `pytest` and `pytest-cov`. It also carried `networkx` and
  `optimalportfolios[data]`; neither enabled a single test, with collection at 1277 either way.
  The `data` extra was there for the examples rather than the suite — no test imports yfinance,
  and the eleven files that do live under root-level `examples/`, which is excluded from
  wheels and never collected — so every cell of the test matrix was installing yfinance for
  nothing, and could not have established that an optional package was absent. `networkx` was
  orphaned in 6.16.0 when the risk-lineage analytics moved to FactorLasso and has no reference
  left in the repository. Run the examples with `[dev,data]` or `[all]`.
- Added a second `pip-audit` pass over `uv export --locked --all-extras`, and narrowed the stated
  contract of the first. `uv pip compile --all-extras --python-version 3.12` audits one
  resolution — the newest tree resolvable on Linux/CPython 3.12 — not every version the
  open-ended floors permit. The lock was a path trigger whose content no step read; it is now
  audited directly, which is the tree a pin can strand on a vulnerable version.
- Widened `pip-audit` from the core tree to every extra. A bare `pip-audit .` audits only the core
  dependencies and silently omits the optional ones, leaving user-facing integrations and
  contributor toolchains ungated. The audit continues to cover every remaining extra after the
  unused `jupyter` convenience extra was removed.
- Widened the optional-module absence check from three of the seven names ruff bans at module
  level to all of them, and derived the list from `banned-module-level-imports` rather than
  restating it. It runs against the three installed 3.12 environments in `ci.yml` and fresh daily
  core and `[dev]` resolutions in `audit.yml`.

- Made soft tracking error ignore a populated hard tracking-error budget during both the solve
  and post-solve validation, instead of rejecting an optimal soft solution and silently falling
  back to benchmark weights. Reported by @tschm in issue #49.

## [6.17.0] - 2026-08-12

**Coverage scope change (2026-08-12):** `[tool.coverage.run] omit` now drops `reports/`
alongside `tests/`, `examples/` and `papers/`. The reporting layer renders factsheets through
`qis` and `pybloqs` and is reviewed by eye rather than by assertion; measured at 3.9% it
contributed 223 of the 597 missed lines and did nothing but dilute the ratchet. Anything with
a numerical contract belongs outside `reports/`, where it is still measured. The floor rises
from `fail_under = 88` to `95`: measured coverage over the narrowed scope is 96.18%, up from
92.99% at the moment of the scope change, on a suite grown from 1077 to 1145 tests.

### Added

- Added the governed 2026-Q2 custom eleven-factor MATF-CMA snapshot and its
  release-pinned replication inputs.
- Exposed `optimalportfolios.__version__` from installed distribution metadata and added
  per-object Sphinx API pages with full signatures and rendered argument documentation.
- Tests for five previously unexercised modules: the rank-based alpha profiler
  (`alphas/profile/core.py`, `alphas/profile/signal_profilers.py`), the alpha container
  (`alphas/alpha_data.py`), the HCGL covariance report (`covar_estimation/covar_reporting.py`),
  the CVXPY covariance stabiliser (`optimization/covar_factorization.py`), and the
  settings-path accessors (`local_path.py`).

### Fixed

- Made the shipped `pytest --pyargs optimalportfolios` harness select `Agg` even when package
  discovery imports Matplotlib before pytest loads `conftest.py`, and made its path assertions
  honour the documented installed-package fallback.
- Made resource and output paths platform-neutral with `pathlib`; a missing or placeholder
  `settings.yaml` now selects a writable checkout-aware default instead of a literal Windows
  separator, and the offline multi-asset example now imports its shipped fixture. The defect
  was reported by @tschm in issue #43.
- Reconstructed factorlasso's flat, persisted cluster/linkage/cutoff fields by cadence before
  plotting, so `plot_current_covar_data` and `run_rolling_covar_report(is_plot=True)` render
  multi-asset universes again. The defect was discovered by @tschm in PR #44.

### Changed

- Repinned the MATF-CMA replication harness and rebuilt its committed exhibits and manuscript
  numbers; package optimisation and backtest behaviour are unchanged.
- Clarified the public reproduction boundary and recorded known environment versions for every
  paper folder in `papers/README.md`; the package classifier now matches the documented
  Production/Stable maturity.

## [6.16.0] - 2026-08-12

**Risk-label tie disclosure:** the new deterministic matcher preserves the maximum matched
weight and aggregate lineage churn on both roadmap panels. On mac_apac, one exact legacy tie
changes identity assignment: 2014-06-30 `ME:12` links to 2014-07-31 `ME:3` instead of `ME:4`;
both candidate edges have integer-scaled weight 20,000. S&P 500 relabel and lineage frames are
byte-identical to the former backend.

**Packaging behaviour change (issue #39, reported by @tschm):** wheels no longer ship the
examples tree; the complete test suite now ships with its offline fixture and supports
`pytest --pyargs optimalportfolios` as a post-install check. The ubuntu/Python 3.12 coverage
gate installs from the release-refreshed `constraints.txt`; the remaining matrix and audit
resolutions continue to float. Removing the examples package marker also removes the accidental
`optimalportfolios.examples` root-module binding that full-suite collection formerly created;
the clean-import public API is unchanged. The shipped pytest configuration defaults Matplotlib
to the non-interactive `Agg` backend while respecting an explicitly selected backend.

### Changed

- Replaced the default `mcf` risk-lineage matcher's NetworkX min-cost-flow backend with a
  deterministic sparse SciPy maximum-weight bipartite assignment using ordered tie
  perturbations and free unmatched vertices. Plain `optimalportfolios` installs now run the
  default matcher.
- Moved canonical cluster-lineage analytics to `factorlasso.cluster_lineage` and raised the
  factorlasso floor to 0.14.0. The former
  `optimalportfolios.covar_estimation.risk_labelling` path remains as a thin compatibility shim,
  re-exports identical objects, and emits one `DeprecationWarning` naming the new package path.
- Added an independent brute-force oracle over 120 seeded sparse panels, a NetworkX development
  cross-check, and cached 60-date S&P 500 / 284-date mac_apac golden and runtime validation.

### Removed

- Removed the `clustering` optional extra. NetworkX is retained only by the `dev` extra as an
  independent test oracle and is no longer imported by production code.

## [6.15.0] - 2026-08-11

non-USD CMA runs using USD-anchored precomputed clusters change numerical results — they previously ran a row-grouped GROUP_LASSO objective and now run the spec's FCGL cluster-factor objective, making USD and non-USD CMA runs consistent for the first time.

### Added

- Added declarative causal cluster smoothing configuration to the rosaa covariance spec and
  two-pass rolling execution for HOLD, partition-bonus, and similarity-EWMA variants.

### Changed

- Precomputed cluster partitions now retain the configured FCGL or HCGL penalty semantics through
  factorlasso's external-cluster fit path instead of changing the estimator to GROUP_LASSO.
- Raised the factorlasso dependency floor to 0.13.0 for external cluster partitions and causal
  rolling smoothing.
- CI test jobs now run on ubuntu-latest, windows-latest and macos-latest across Python
  3.10–3.13, adopted from PR #40 by @tschm; the coverage gate remains ubuntu + Python 3.12.

## [6.14.0] - 2026-08-11

**Risk-label behaviour change:** the default full-panel lineage matcher now enforces the lower
overlap gate and uses the calibrated `(0.15, 0.60)` overlap band, `0.015` factor-spread-vol cut,
six-date bridge window and `0.5` bridge decay. On the 284-month mac_apac sweep, derived lineages
fell from 184 to 131 and distinct tracks per asset from 18.5 to 14.5; total track-id churn was
effectively flat at 1.709 versus 1.713 moves per asset-year, while matcher-attributable churn
fell from 0.508 to 0.476. These labels remain offline, full-panel reporting diagnostics and are
not point-in-time backtest signals.

### Changed

- Corrected five risk-lineage edge cases: zero-overlap clusters no longer link on beta proximity
  alone; Hungarian bridging selects the highest-affinity dormant track; zero-beta clusters use
  the `Idio` sentinel; NaN equity beta labels fall back safely; and single-factor models no
  longer assume a secondary factor. `bridge_decay` is now retained in report provenance.
- Refreshed the README architecture, analytics inventory, installation extras, constraint
  semantics, solver workflow and release-history descriptions to match the current package.

### Removed

- Removed `Constraints.min_target_portfolio_vol_an`. Its quadratic volatility lower bound was
  non-convex and every supported CVXPY solver rejected it with `DCPError`; no successful solve
  could use the field. The supported `max_target_portfolio_vol_an` constraint is unchanged.

## [6.13.0] - 2026-08-11

**Example behaviour fix:** `examples/solvers/risk_budgeting.py` now passes its constructed
10%–30% asset-class constraints into the demonstrated solve, in PR #36 by @tschm. On the
committed 19-asset fixture, Cash falls from 44.7552286661% to the 30% cap; all 19 weights
change, with 0.295104561020 L1 weight difference (14.7552280510% one-way turnover).

### Changed
- Added the Read the Docs site and excluded local-only trees from root docstring scans.
- CI now gates the `F` (pyflakes) family alongside the three stack invariants, all green on the
  package. The ~380 `E501` line-length findings remain ungated and untouched. Ruff configuration
  stays in `[tool.ruff]` in `pyproject.toml`, which remains the stack's single configuration home.
- `F401` and `F403` are ignored for `"__init__.py"` in `[tool.ruff.lint.per-file-ignores]`, since
  an import there is a re-export. No `__all__` is introduced: adding a name to a subpackage's
  public API stays a single edit, with no second list to maintain beside the import. The public
  surface is unchanged — the 125 names `optimalportfolios` exposes, and every name reachable
  through `from optimalportfolios.<sub> import *`.
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
