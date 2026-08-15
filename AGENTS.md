# AGENTS.md

Guidance for AI coding agents working in the **OptimalPortfolios** repository.

## Project overview

`optimalportfolios` implements the production pipeline for multi-asset portfolio construction and backtesting: alpha signals -> covariance estimation (EWMA or the HCGL factor model from `factorlasso`) -> constrained optimisation (risk budgeting, maximum diversification, maximum Sharpe, alpha over tracking error, and others) -> rolling backtest and reporting through `qis`.

It is the reference implementation of the ROSAA framework published in *The Journal of Portfolio Management* (Sepp, Ossa and Kastenholz, 2026). Distribution and import name `optimalportfolios`. Licensed MIT (`LICENSE.txt`).

## Ecosystem position

This package is one of eight open-source Python libraries maintained at [github.com/ArturSepp](https://github.com/ArturSepp). Before implementing anything non-trivial, check whether it already exists in one of these:

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | Performance analytics, factsheets, visualisation |
| `optimalportfolios` | OptimalPortfolios | Portfolio construction and backtesting |
| `factorlasso` | factorlasso | Sparse factor models and factor covariance estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data fetching |
| `trendfollowing` | TrendFollowingSystems | Trend-following systems: closed-form theory and replication |
| `goal-based-allocation` | GoalBasedAllocation | Dynamic MV allocation under regime-switching jump-diffusions |
| `stochvolmodels` | StochVolModels | Stochastic volatility pricing analytics |
| `vanilla-option-pricers` | VanillaOptionPricers | Vanilla option pricers and implied volatility fitters |

Actual package dependencies within the stack: `optimalportfolios` depends on `qis` and `factorlasso`; `trendfollowing` depends on `qis`; `stochvolmodels` has an optional `research` extra that pulls in `qis`. The others are independent.

Do not vendor or copy code between these packages. If functionality belongs in a sibling package, say so rather than reimplementing it here.

### `rosaa` dependency floors

`rosaa/` is gitignored and carries no `pyproject.toml`, so its floors have nowhere else to live and are recorded here. They are not advisory: each names a symbol or keyword `rosaa` calls that does not exist below the floor.

| Package | Floor | What `rosaa` needs at it |
|---|---|---|
| `qis` | **>= 5.5.0** | `load_df_from_csv` / `load_df_dict_from_csv` take `float_precision`; the inputs store cannot round-trip a float exactly without it |
| `factorlasso` | **>= 0.14.0** | Canonical cluster-lineage analytics live in `factorlasso.cluster_lineage`; it also includes the earlier per-frequency alpha forwarding required by rosaa |
| `optimalportfolios` | **>= 6.8.0** | signal spans accept a per-cadence `Mapping[str, int]`; below it `product_config.SIGNALS` raises, since it passes dicts |

`optimalportfolios 6.7.0` was tagged in `CITATION.cff` but never published — its `pyproject.toml` stayed at 6.6.0 — so a fresh `pip install optimalportfolios` before 6.8.0 gives a package `rosaa` cannot run on. Verified with `pip index versions optimalportfolios`, not from the changelog.

## Repository layout

```
src/optimalportfolios/
  alphas/            alpha signal construction
  covar_estimation/  covariance estimators (EWMA, factor/HCGL via factorlasso)
  optimization/      optimisers, constraints, solvers
  universe/          instrument universes
  reports/           reporting built on qis
  tests/             cross-cutting tests and their offline data fixtures
  utils/, docs/, config.py, local_path.py, settings.yaml
examples/            repository-only examples (excluded from distributions)
papers/              code accompanying the published papers (excluded from ruff)
```

`src/optimalportfolios/covar_estimation/risk_labelling.py` is a deprecated compatibility shim for
the canonical `factorlasso.cluster_lineage` module; keep rosaa imports working, but add lineage
features and tests in FactorLasso.

Tests live inside the source package as `src/optimalportfolios/<subpackage>/tests/*_test.py`; there is no top-level `tests/` directory. The wheel includes these test packages and their fixture under `optimalportfolios/tests/data/`, so `pytest --pyargs optimalportfolios` is the supported post-install check — and is enforced by the `wheel` job in `ci.yml` on every pull request, so it cannot rot into a claim the artifact no longer supports. That job is also the justification for the nested layout: it is what turns shipped tests into a check that the built wheel is complete. The shipped `conftest.py` defaults `MPLBACKEND` to the non-interactive `Agg` backend while preserving an explicitly selected backend. The `examples/` tree remains in the repository but is excluded from wheels. Nineteen `*_local.py` files are local-data diagnostics: run them manually when their price data or terminal is available; pytest and the unattended examples workflow skip them. Eighteen are `run_local_test` dispatchers. `examples/backtests/tracking_error_decomposition_local.py` is the one direct diagnostic script, and `examples/data/etf_prices_local.py` is also the shared loader those dispatchers import their price panel from.

Every file matching pytest's default patterns — `*_test.py` **and** `test_*.py` — collects at least one test, so the file count is a usable proxy for the suite. Keep it that way: a new diagnostic script belongs in `*_local.py`, not under a test-shaped name. A name matching either pattern is *imported* at collection even when it contributes no tests, which is how a module-level import of an optional extra has twice broken CI.

## Commands

```bash
uv sync --extra dev                                      # editable install, versions from uv.lock
uv run pytest                                            # run the test suite (1336 tests, ~3 min)
uv run pytest src/optimalportfolios/optimization/tests/constraints_test.py -v
uv run --only-group lint ruff check src/optimalportfolios/  # lint (papers/ is excluded)
uv run --only-group lint interrogate                     # docstring coverage, must stay at 100%
```

`pip install -e ".[dev]"` still works, but it resolves fresh rather than from `uv.lock`, so it is
not what CI gates the pinned cell against; the floating matrix cells reach the same effect with
`uv sync --extra dev --upgrade`. The lint
tools are deliberately **not** in the `dev` extra: they are declared once in the `lint`
dependency-group, which is where the workflow takes its versions from too, so a local `ruff` and
CI's `ruff` cannot disagree about the same file. `--only-group` installs that group alone, without
the project or the compiled scientific stack.

The three tools that decide whether a change may land are pinned **exactly**, not to a series:
`ruff==0.16.2` and `interrogate==1.7.0` in the `lint` group, `pip-audit==2.10.1` in the `audit`
group. A range does not make a verdict reproducible — `ruff~=0.16.0` admits any 0.16.x, and a ruff
patch release may add or fix a rule, so unchanged source could pass today and fail tomorrow with
nothing here having moved. Exact pins make a verdict change a reviewed commit rather than an
upstream event; bumping one is a deliberate PR (raise the version, `uv lock`, fix what the new
release reports). Note what this does *not* fix: `pip-audit`'s advisory database is remote and
updates continuously, so the audit's answer is expected to change on its own — that is the point of
running it daily. The pin fixes the scanner, not the verdict.

*Note: Terminal execution should be compatible with Windows PowerShell within PyCharm.*

Optional extras: `data`, `reports`, `docs`, `dev`, `all`. The runtime integration extras are `data` and `reports`; `docs` and `dev` are contributor toolchains. A convenience `jupyter` extra was removed because the package imports none of its dependencies; the repository-only Colab quickstart uses the hosted runtime and is mechanically checked against the Python example. Supported Python is >= 3.10; `ci.yml` runs 3.10 – 3.13 on a `[dev]` install, which must be green: no test may need data, network or a Bloomberg terminal. It runs on `ubuntu-latest`, `windows-latest` and `macos-latest`, so a fix that only holds on POSIX paths or POSIX line endings fails the matrix. CI installs with `uv`, which supplies the interpreter as well, so there is no `setup-python` step; `.python-version` is gitignored, so every job states its Python version explicitly rather than inheriting one from the tree. Pinning lives in `uv.lock` and nowhere else — `constraints.txt` is gone, and the release step that regenerated it is now `uv lock`. The ubuntu/Python 3.12 coverage cell syncs `--locked`, which also fails if `uv.lock` has drifted from `pyproject.toml`, so a dependency edit cannot land without a re-lock; the remaining matrix cells sync `--upgrade` and so deliberately float, keeping the matrix honest about what a user installing today actually gets. The two legs differ by that one flag rather than by a whole step. There is no separate `core-install` job: `[dev]` is now only pytest and pytest-cov over the core tree, so the twelve test cells already run that suite without optional extras. On Python 3.12, all three operating systems additionally ask the import system to prove that every `banned-module-level-imports` name is absent; this preserves the platform-specific dependency guarantee without running the complete suite twice.

A thirteenth job, `wheel`, is the only thing in the repository that tests the **built artifact** rather than the editable installation from `src/`. The src layout prevents the repository root from directly shadowing an installed package, but an editable install still does not prove that package discovery or package-data declarations produce a complete wheel. The job builds the wheel, asserts it still carries its test modules and the offline fixture, installs it into a clean environment with **no extras** plus pytest, and runs `pytest --pyargs optimalportfolios` from `$RUNNER_TEMP` outside the checkout. This is what makes shipping the tests inside the wheel load-bearing rather than incidental, and it is the same command the post-install note below documents for users. It runs on one runner and one interpreter: a `py3-none-any` wheel's contents do not vary by platform, and platform behaviour is already covered twelve ways. Like `static.yml` it carries no job-level `if`, for the same reason — a skipped job reports success. It also catches the module-level-optional-import trap from the consumer's side: such an import breaks collection, which ruff's TID253 gate and the absence assertion both check in the source but neither checks in an installed environment.

Three further workflows gate the repository without installing it as a dependency of the test matrix. `static.yml` holds the source-only gates — the three ruff stack invariants and `interrogate` docstring coverage at 100% — and runs unconditionally on every pull request, plus on pushes to branches in this repository. The fork guard sits on the push trigger, not on the job: GitHub reports a conditionally skipped job as *successful*, so a job-level `if` would let the PR check go green without ruff or interrogate ever running. `audit.yml` runs daily, plus on pushes and PRs touching `pyproject.toml` or `uv.lock`, and holds the checks whose answers depend on the outside world rather than on the source. It runs `pip-audit` over two different trees. First `uv pip compile --all-extras`, rather than a bare `pip-audit .`, because that form covers only the core dependencies and silently omits every extra — including the user-facing `data` and `reports` ones, and the `docs` toolchain. State that contract narrowly: it audits **one** resolution — the newest tree resolvable today, on Linux, for CPython 3.12 — not every version the open-ended floors permit, and not what Windows, macOS or another interpreter would resolve to. Second, `uv export --locked --all-extras`, which audits the exact pinned set `uv sync --locked` installs; a pin can sit on a vulnerable version long after the floor would resolve past it, and that is invisible to the first. This second tree is why `uv.lock` is a path trigger — without it the trigger would name an input no step consumed. The workflow also resolves fresh core and `[dev]` trees and fails if either contains one of the banned optional modules. This complements, rather than replaces, the three-platform installed-environment assertion in `ci.yml`: the matrix catches platform-specific arrivals, while the scheduled fresh resolution catches dependency drift when no repository event occurs. Run `interrogate` from the repository root — the `papers/` exclusion in `[tool.interrogate]` is resolved against the working directory.

`examples.yml` executes the example scripts, which nothing else does: `examples/` is excluded from wheels, dropped by `[tool.coverage.run] omit`, and never collected by pytest, so an example can call an API that no longer exists and stay broken indefinitely. Ruff does not close the gap either — the rot these attract is attribute-level, not name-level. The workflow was added after `LassoModelType.GROUP_LASSO_CLUSTERS` was found broken in two examples, a shipped docstring and three documents; an enum member that was renamed upstream is still a valid attribute access to a linter.

It has two lanes, and the split is **derived rather than listed**: `.github/scripts/run_examples.py` walks each unattended example's intra-`examples` import closure and calls it network-bound if `yfinance` is reachable at all. Of 23 unattended examples, 18 are network-bound — 7 import it directly and the rest reach it through `examples/data/universe.py`, whose `fetch_benchmark_universe_data()` downloads 15 tickers back to 2003. The **offline** lane runs the other 5 on all three runners and gates pull requests; it syncs the *core* environment, so it shows the examples work for someone who ran a plain `pip install optimalportfolios`. The **network** lane runs the 18 on a daily schedule only, `continue-on-error`, with a step-summary report — gating a PR on dozens of live Yahoo downloads would fail on Yahoo's availability far more often than on the diff. Files named `*_local.py` are excluded from both lanes because their required local CSV or Bloomberg preconditions cannot be met on a runner. The script fails on an empty lane, so a classification bug cannot report success by running nothing.

That `network` job carries a job-level `if`, which `static.yml` deliberately does not. The distinction matters: a skipped job reports *success*, so a guard is only ever safe on a job that must never gate. That one qualifies twice — excluded from the PR path by the condition, and `continue-on-error` on top. Do not add it to branch protection.

Line coverage is **100.00%** on the 1336-test dev suite, and the ubuntu/3.12 matrix entry gates
`pytest --cov=optimalportfolios` at `fail_under = 100`. This is no longer a ratchet: at 100% an
uncovered line is always something the change under review introduced, which is the same argument
the 100% `interrogate` bar rests on. Lowering the floor requires a dated `CHANGELOG.md` note.
Eight lines carry `# pragma: no cover`, each with a comment at the site naming why — two defensive
raises in `risk_budgeting_solver.py` that only a pinned solver pathology would reach; three
branches that are dead as written (`risk_budgeting.py` seeds from a literal `True`,
`factor_covar_estimator.py` guards a schedule that `include_end_date=True` already guarantees, and
`alphas/signals/utils.py` filters a group that was already filtered); and three that are reachable
only from an environment this cell is not — `__init__.py`'s `PackageNotFoundError` fallback wants
an uninstalled source tree, and `conftest.py`'s `_find_root` miss and the `root` fixture's skip
want an installed wheel with no checkout around it, which is the `wheel` job's path and measures
no coverage. Adding a ninth is a reviewable decision; it is not the way to make a red run green.
Reconciling that count against the tool needs two facts: the eight pragmas suppress ten *statement*
lines, two of them sitting on multi-line statements, and `coverage json` reports fourteen excluded
lines because it also excludes, with no pragma present, the `...` bodies of the two abstract
methods in `covar_estimation/covar_estimator.py`.
The measured scope is not the whole package: `[tool.coverage.run] omit` drops `reports/` alongside
`tests/`, `examples/` and `papers/`, because the reporting layer renders through `qis` and `pybloqs`
and is reviewed by eye rather than by assertion. Put anything with a numerical contract outside
`reports/`, where it is measured. Measure on a `[dev]` install.

`[dev]` is pytest and pytest-cov, and nothing else. It previously also carried `networkx` and
`optimalportfolios[data]`; neither enabled a single test — collection is 1336 either way. The
`data` extra was there for the **examples**, not the suite: no test imports yfinance, and the
eleven files that do all live under root-level `examples/`, which is excluded from distributions
and never collected. `networkx` was orphaned when the risk-lineage analytics moved to FactorLasso
and has no reference left in this repository. To run the examples, install what they need:
`[dev,data]`, or `[all]`.

## Conventions

- Test files are named `*_test.py` and live in a `tests/` directory inside the subpackage under test.
- Line length 100 (`ruff`, rules `E`, `F`, `W`); `papers/` is excluded from linting on purpose. `I` is deliberately not selected anywhere in the stack: imports group the scientific stack before project packages, which isort's ordering contradicts.
- **Ruff is configured in `[tool.ruff]` in `pyproject.toml`**, alongside pytest, coverage and interrogate. `pyproject.toml` is the stack's single configuration home; do not add a `ruff.toml`, which Ruff would read in preference and silently shadow this config.
- **Four rule sets are enforced by ruff rather than written down**: the three stack invariants below and the whole `F` family. All are green on the package, so a finding is always something you just introduced. `E`/`W` stay ungated because of the 216 `E501` line-length findings in the older modules:
  - `TID251` fails an import of `trendfollowing`, `privateassets`, `stochvolmodels`, `goal_based_allocation` or `vanilla_option_pricers`. This package depends on `qis` and `factorlasso` and on nothing else in the stack; subject packages never import each other. `qis` and `factorlasso` are of course not banned — they are declared dependencies, and importing them is the point.
  - `TID253` fails a **module-level** import of an optional extra (`yfinance`, `pandas_datareader`, `pybloqs`, `plotly`, `pyarrow`, `psycopg2`, `sqlalchemy`); the same import inside a function passes, which is the pattern the collection note above requires. `examples/**` and `src/optimalportfolios/reports/portfolio_result_pybloqs.py` are named in `per-file-ignores` — add to that list only for a module `src/optimalportfolios/__init__.py` cannot reach.
  - `ICN` pins `import numpy as np` and `import pandas as pd`. Ruff's default alias map is replaced rather than extended, so `matplotlib` stays free to be both `mpl` and `plt`.
- **Every module, class, method and function carries a docstring.** `interrogate` is configured in `pyproject.toml` with `fail-under = 100` and, like ruff, excludes `papers/`. The bar is 100% rather than a partial target for the same reason the invariants above are lint: at 100% a miss is always something you just introduced. Nested closures and one-line properties count too — a short single line stating what the thing returns is enough; reserve the `Args:`/`Returns:` block for public entry points.
- **README code blocks are executed, so the fences are an interface.** `src/optimalportfolios/tests/readme_test.py` joins every non-skipped ` ```python ` block into one script with an explicit newline between blocks, runs it in a subprocess under a 300s timeout, and diffs stdout against every ` ```result ` block joined the same way. It fails closed on an empty parse: no executable block, or no result block, is an error rather than a vacuous `"" == ""` pass, so renaming or `+SKIP`-ing the fences cannot silently retire the gate. The newline join (rather than `"".join`) keeps a fence body that lacks a trailing newline from splicing onto the next block, and the timeout means a quick-start that hangs fails this test instead of holding the job open to its outer limit. Two consequences: a new ` ```result ` fence anywhere in `README.md` joins that single expected output and will fail the test unless the executed script also prints it, and a new ` ```python ` fence is executed by default — mark an illustrative one `` ```python +SKIP ``, which is what the other twelve blocks carry because they reference names defined nowhere, restate a dataclass, or fetch from the network. Print structural facts (schedule length, long-only, fully-invested), never weights: a solver-backend update moves an allocation by 1e-9 and would make the block flaky across the twelve matrix cells. The skip that keeps `ci.yml`'s `wheel` job green is at the *checkout* level rather than the file level: `conftest.py`'s `root` fixture finds no `pyproject.toml` outside a checkout — neither it nor the root `README.md` is wheel content — and skips before a README is ever requested. Once a checkout has been found the README must be there, and its absence asserts rather than skips: a checkout carrying a `pyproject.toml` but no `README.md` is a broken documentation contract, not a case this harness has nothing to say about.
- **Docstring `>>>` examples are executed too.** `src/optimalportfolios/tests/doctest_test.py` imports every module under `src/` and runs `doctest.testmod` with `ELLIPSIS | NORMALIZE_WHITESPACE`, so a `>>>` line is a test, not decoration. Before this existed, 10 of the package's 13 examples raised `NameError` on a `prices` panel or `time_period` the prose described but the docstring never built; there are now 70 passing doctests. This gate also fails closed in two ways worth knowing before you touch it: an `ImportError` from any module fails the test unless that module is named in `OPTIONAL_EXTRA_MODULES` with its reason (only `reports/portfolio_result_pybloqs.py` is, for the module-level `pybloqs` import a `[dev]` install cannot satisfy), so a newly broken first-party module cannot drop out of the run behind a warning; and zero discovered doctests is a failure, not a skip. Adding an allowlist entry means accepting that the module's `>>>` examples never run on a core install — prefer a function-level import. Three rules keep the examples themselves honest:
  - **Build what the example uses.** A `>>>` block gets its own synthetic inputs — a closed-form price path (`np.exp(0.0004 * np.arange(260))`), not the offline fixture and never an undefined name. Anything genuinely unrunnable — file I/O, or the ~4s factorlasso HCGL fit in `factor_covar_estimator.py` — is marked `# doctest: +SKIP` **with the reason stated in the docstring**, not left to fail.
  - **Assert structure, not floats.** `.tolist()` and `.columns.tolist()` rather than a bare Series, because a pandas repr change would break the expected block; `float(...)` around any scalar, because NumPy 2 reprs it as `np.float64(1.4)` and NumPy 1 as `1.4`. Prefer shapes, column lists, symmetry and exact decimals over mantissa-length numbers.
  - **Document the contract, not the happy path.** The examples worth writing are the ones a reader would otherwise take on faith: that `round_weights_to_pct` sums to exactly 100, that `apply_drift_to_weights_0` is a true passthrough on every failure gate, that `filter_covar_and_vectors_for_nans` *clamps* a cash-like variance but *drops* a NaN one.
- Optimisation problems are expressed with `cvxpy`; `quadprog` is used where a dedicated QP solver is faster. Do not introduce a third optimisation backend.
- Enums and dataclasses carry configuration (optimiser type, constraint sets, estimation settings) — extend the existing enum rather than passing raw strings.
- Time series are pandas objects with a `DatetimeIndex`; the backtest layer is NaN-aware by design, so preserve NaN handling when refactoring.
- Reporting and plotting go through `qis`; do not add a parallel plotting layer here.

## Implementation Directives

- **Preserve Core Logic:** Maintain the existing optimiser defaults, constraint semantics, and rebalancing conventions, as published results heavily depend on them.
- **Respect Linting Exclusions:** Leave `papers/` exactly as-is; it is deliberately excluded from linting to preserve published code.
- **Ensure Offline Execution:** Ensure all examples run on free data. Never add a hard dependency on Bloomberg data.
- **Maintain Clean Commits:** Prevent backtest outputs, factsheets, or generated figures from being committed to version control.

<!-- ===== SHARED AGENT CORE (consumer variant) — begin =====
     Generated from SHARED_AGENT_CORE.md in the maintainer's project knowledge. Do not hand-edit
     between these markers — propose the change to the maintainer instead. Variants: builder
     (qis) / consumer / standalone. Last synced 2026-08-08, agent core v1.1. -->

## Domain invariants

Not inferable from any single file, and the source of numerically wrong code that runs clean:

- **No look-ahead, anywhere in a backtest path.** A weight decided at *t* is applied over
  *[t, t+1]*. Estimation is point-in-time: `MeanAdjType.INSAMPLE` subtracts a full-sample mean
  and is therefore forward-looking — correct for a descriptive exhibit, wrong inside a backtest.
- **Return convention is stated, never implied** — `qis.to_returns(..., is_log_returns=...)`.
  Annualisation follows from the frequency; never silently switch convention, frequency, or
  annualisation factor.
- **Sharpe has three explicitly labelled conventions** in `qis`; excess variants need
  `PerfParams.rates_data`. State which one a number uses.
- **`qis.BootstrapType.STATIONARY` wraps circularly from qis 5.1.0.** Any result resampled under
  an earlier version does not reproduce.
- One convention per concept across the stack. If two packages disagree, that is a bug to
  report, not a difference to accommodate.

## Use the stack before you write it

This package consumes `qis` (analytics, backtesting, reporting) and `factorlasso` (factor
covariance). Reimplementing a capability they export is a defect, not a convenience.
Triggers — stop and check the export list before writing: backtest, rebalance, turnover,
drawdown, Sharpe, volatility target, bootstrap, resample, unsmooth, covariance, correlation,
regime, hedge ratio, factsheet, tracking error, risk contribution.

- **The hard stop:** a `for` loop over dates accumulating a position, a weight or a P&L is
  `qis.backtest_model_portfolio`. The hand-rolled version gets drift adjustment wrong — `qis`
  holds *units* between rebalancings, not weights.
- **Never invent a symbol.** If a function, class, or keyword argument is not in the export
  list, it does not exist. Check in one line —
  `python -c "import qis; print([n for n in dir(qis) if 'unsmooth' in n.lower()])"`;
  `qis.api.CORE_API` is the documented core and `help(qis.<symbol>)` gives the arguments. Say a
  symbol is missing rather than producing code that calls it.
- **If you genuinely must reimplement**, name the rejected stack symbol and why, in a comment on
  the line above the definition — that turns a silent divergence into a reviewable decision.
- Never introduce `quantstats`, `pyfolio`, `empyrical`, `ffn`, `bt`, or an ad-hoc statistics
  layer.

## Verification loop

- Plan → patch → verify. Name the verification command and its result when proposing a patch.
- A second pass is mandatory where a plausible patch can be numerically wrong and still run
  clean: estimation windows, weight normalisation, annualisation, constraint construction,
  anything resampled. Verify against a reference computed a different way, and say which.
- Prove a new test fails before trusting that it passes: reintroduce the defect, watch it fail,
  restore.

## Escalation and scope

- Stop and propose before proceeding when a change would exceed roughly five files, alter a
  public signature, or touch a numerical path.
- Never change numerical results, random seeds, or computed values unless the change is the
  request.
- A public-signature change carries a `CHANGELOG.md` entry and a version bump in the same
  change. Removing a keyword argument from a function taking `**kwargs` is a silent break — the
  caller's keyword is swallowed and nothing raises. Treat it as breaking.
- Do not refactor beyond the requested scope. Propose the wider change; do not perform it.

## Concurrent sessions

More than one agent or session may work on this checkout at the same time, so a file can change
between your read of it and your write.

- Re-read a file from disk immediately before editing it. Never write a file from an earlier
  read: a whole-file write from a stale copy silently reverts another session's work.
- Prefer minimal anchored edits over whole-file replacement. If the on-disk content is not what
  you expected, stop and reconcile your change onto the current content rather than overwrite.

## Roadmap execution

Feature roadmaps live at the repository root as `ROADMAP_<feature>.md`. An execution request
names the file and the stage. A stage is complete when its stated verification command passes;
its out-of-scope list is binding.

<!-- ===== SHARED AGENT CORE — end ===== -->

The JOSS alignment execution contract is `roadmap/ROADMAP_JOSS_ALIGNMENT.md`.

## Replication contract

`papers/` reproduces results from the published papers. If a change alters optimiser behaviour, covariance estimation, or backtest mechanics, re-run the relevant scripts in `papers/` and confirm the outputs still match the published tables before proposing the change.

## Release checklist

A release touches three version locations. All three must agree, and `src/optimalportfolios/tests/version_metadata_test.py` fails when they do not:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the `@software` BibTeX entry in `README.md`

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release with the same tag. Do not bump versions as part of an unrelated change, and do not publish without the maintainer explicitly asking for a release.

## Known issues

- The previous `CLAUDE.md` described version 4.1.1 and a black/isort/flake8/mypy toolchain; the project has since moved to `ruff` and this file supersedes it.
- `ruff check src/optimalportfolios/` reports 225 baseline findings: 216 `E501` line-length, 8 `E712` true-false-comparison and 1 `E402` module-import-not-at-top-of-file. CI gates TID251/TID253/ICN **and `F`**, all green; `E`/`W` remain ungated by policy. Fix only the lines your specific change touches; a repository-wide reflow is not wanted. The `W` family is now clean — the 14 `W291`/`W292` findings were auto-fixed in #72 — so `W` findings a run reports are yours.
- **`F401` in an `__init__.py` is a re-export, not an unused import.** `F401` and `F403` are therefore off for `"__init__.py"` in `[tool.ruff.lint.per-file-ignores]`, rather than answered file by file with `# noqa`. That keeps the rule this package has always followed: a subpackage's public surface is the imports in its own `__init__.py`, and adding a name to it is one edit — no `__all__` or other second list to maintain beside the import. Never `ruff --fix` F401 across `__init__.py` with that ignore removed: it would delete the re-exports and break `from optimalportfolios import Constraints` for every consumer.
- **The offline multiasset fixture is live test infrastructure, not an unused artifact.** `src/optimalportfolios/tests/data/multiasset_returns.csv`, loaded by `optimalportfolios.tests.data.multiasset.load_multiasset_data`, feeds three collected suites: `src/optimalportfolios/optimization/tests/rolling_dispatcher_test.py`, `src/optimalportfolios/utils/tests/portfolio_funcs_properties_test.py` and `src/optimalportfolios/covar_estimation/tests/covar_properties_test.py`. Treat the CSV and loader as frozen test data: do not modify, move or delete them without updating those suites, and expect numerical assertions to change if the data changes. (An earlier version of this file wrongly described the fixture as unused.)
