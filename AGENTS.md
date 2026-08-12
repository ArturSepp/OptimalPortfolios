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
optimalportfolios/
  alphas/            alpha signal construction
  covar_estimation/  covariance estimators (EWMA, factor/HCGL via factorlasso)
  optimization/      optimisers, constraints, solvers
  universe/          instrument universes
  reports/           reporting built on qis
  tests/             cross-cutting tests and their offline data fixtures
  utils/, docs/, config.py, local_path.py, settings.yaml
  examples/          repository-only examples (excluded from wheels)
papers/              code accompanying the published papers (excluded from ruff)
```

`optimalportfolios/covar_estimation/risk_labelling.py` is a deprecated compatibility shim for
the canonical `factorlasso.cluster_lineage` module; keep rosaa imports working, but add lineage
features and tests in FactorLasso.

Tests live inside the package as `optimalportfolios/<subpackage>/tests/*_test.py`; there is no top-level `tests/` directory. The wheel includes these test packages and their fixture under `optimalportfolios/tests/data/`, so `pytest --pyargs optimalportfolios` is the supported post-install check. The shipped `conftest.py` defaults `MPLBACKEND` to the non-interactive `Agg` backend while preserving an explicitly selected backend. The `examples/` tree remains in the repository but is excluded from wheels. Sixteen `*_local.py` files are `run_local_test` diagnostic dispatchers: run them manually when the required local price data is available; pytest never collects them, and the test suite is exactly what bare `pytest` collects. The sixteenth is `examples/data/etf_prices_local.py`, which is also the shared loader those dispatchers import their price panel from.

Every file matching pytest's default patterns — `*_test.py` **and** `test_*.py` — collects at least one test, so the file count is a usable proxy for the suite. Keep it that way: a new diagnostic script belongs in `*_local.py`, not under a test-shaped name. A name matching either pattern is *imported* at collection even when it contributes no tests, which is how a module-level import of an optional extra has twice broken CI.

## Commands

```bash
pip install -e ".[dev]"                                  # editable install with dev tools
pytest                                                   # run the test suite (1142 tests, ~60 s)
pytest optimalportfolios/optimization/tests/constraints_test.py -v
ruff check optimalportfolios/                            # lint (papers/ is excluded)
interrogate                                              # docstring coverage, must stay at 100%
```

*Note: Terminal execution should be compatible with Windows PowerShell within PyCharm.*

Optional extras: `data`, `reports`, `jupyter`, `dev`, `all`. Supported Python is >= 3.10; CI runs 3.10 – 3.13 on a `[dev]` install and 3.12 again on a core install, which must be green: no test may need data, network or a Bloomberg terminal. Both of those jobs run on `ubuntu-latest`, `windows-latest` and `macos-latest`, so a fix that only holds on POSIX paths or POSIX line endings fails the matrix. The ubuntu/Python 3.12 coverage cell alone installs against `constraints.txt`, regenerated at each release; the remaining matrix cells, core installs and audit resolution deliberately float. Separate jobs gate the three ruff stack invariants, `interrogate` docstring coverage at 100%, and `pip-audit` over the dependency tree resolved from `pyproject.toml`. Run `interrogate` from the repository root — the `papers/` exclusion in `[tool.interrogate]` is resolved against the working directory.

Line coverage measured **96.12%** on the 1142-test dev suite. The
ubuntu/3.12 matrix entry gates `pytest --cov=optimalportfolios` at `fail_under = 95`; this floor rises
whenever measured coverage rises, and lowering it requires a dated `CHANGELOG.md` note.
The measured scope is not the whole package: `[tool.coverage.run] omit` drops `reports/` alongside
`tests/`, `examples/` and `papers/`, because the reporting layer renders through `qis` and `pybloqs`
and is reviewed by eye rather than by assertion. Put anything with a numerical contract outside
`reports/`, where it is measured. Measure on a `[dev]` install. NetworkX remains a development dependency solely for independent
matcher cross-checks; the production `mcf` matcher and its regression tests run in a core install.

## Conventions

- Test files are named `*_test.py` and live in a `tests/` directory inside the subpackage under test.
- Line length 100 (`ruff`, rules `E`, `F`, `W`); `papers/` is excluded from linting on purpose. `I` is deliberately not selected anywhere in the stack: imports group the scientific stack before project packages, which isort's ordering contradicts.
- **Ruff is configured in `[tool.ruff]` in `pyproject.toml`**, alongside pytest, coverage and interrogate. `pyproject.toml` is the stack's single configuration home; do not add a `ruff.toml`, which Ruff would read in preference and silently shadow this config.
- **Four rule sets are enforced by ruff rather than written down**: the three stack invariants below and the whole `F` family. All are green on the package, so a finding is always something you just introduced. `E`/`W` stay ungated because of the ~380 `E501` line-length findings in the older modules:
  - `TID251` fails an import of `trendfollowing`, `privateassets`, `stochvolmodels`, `goal_based_allocation` or `vanilla_option_pricers`. This package depends on `qis` and `factorlasso` and on nothing else in the stack; subject packages never import each other. `qis` and `factorlasso` are of course not banned — they are declared dependencies, and importing them is the point.
  - `TID253` fails a **module-level** import of an optional extra (`yfinance`, `pandas_datareader`, `pybloqs`, `plotly`, `pyarrow`, `psycopg2`, `sqlalchemy`); the same import inside a function passes, which is the pattern the collection note above requires. `optimalportfolios/examples/**` and `reports/portfolio_result_pybloqs.py` are named in `per-file-ignores` — add to that list only for a module `optimalportfolios/__init__.py` cannot reach.
  - `ICN` pins `import numpy as np` and `import pandas as pd`. Ruff's default alias map is replaced rather than extended, so `matplotlib` stays free to be both `mpl` and `plt`.
- **Every module, class, method and function carries a docstring.** `interrogate` is configured in `pyproject.toml` with `fail-under = 100` and, like ruff, excludes `papers/`. The bar is 100% rather than a partial target for the same reason the invariants above are lint: at 100% a miss is always something you just introduced. Nested closures and one-line properties count too — a short single line stating what the thing returns is enough; reserve the `Args:`/`Returns:` block for public entry points.
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

A release touches three version locations. All three must agree, and `optimalportfolios/tests/version_metadata_test.py` fails when they do not:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the `@software` BibTeX entry in `README.md`

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release with the same tag. Do not bump versions as part of an unrelated change, and do not publish without the maintainer explicitly asking for a release.

## Known issues

- The previous `CLAUDE.md` described version 4.1.1 and a black/isort/flake8/mypy toolchain; the project has since moved to `ruff` and this file supersedes it.
- `ruff check optimalportfolios/` reports 429 baseline findings: 378 `E501` line-length, and the small `W292`/`E702`/`E712`/`W291`/`E402` remainder. CI gates TID251/TID253/ICN **and `F`**, all green; `E`/`W` remain ungated by policy. Fix only the lines your specific change touches; a repository-wide reflow is not wanted.
- **`F401` in an `__init__.py` is a re-export, not an unused import.** `F401` and `F403` are therefore off for `"__init__.py"` in `[tool.ruff.lint.per-file-ignores]`, rather than answered file by file with `# noqa`. That keeps the rule this package has always followed: a subpackage's public surface is the imports in its own `__init__.py`, and adding a name to it is one edit — no `__all__` or other second list to maintain beside the import. Never `ruff --fix` F401 across `__init__.py` with that ignore removed: it would delete the re-exports and break `from optimalportfolios import Constraints` for every consumer.
- **The offline multiasset fixture is live test infrastructure, not an unused artifact.** `tests/data/multiasset_returns.csv`, loaded by `tests.data.multiasset.load_multiasset_data`, feeds three collected suites: `optimization/tests/rolling_dispatcher_test.py`, `utils/tests/portfolio_funcs_properties_test.py` and `covar_estimation/tests/covar_properties_test.py`. Treat the CSV and loader as frozen test data: do not modify, move or delete them without updating those suites, and expect numerical assertions to change if the data changes. (An earlier version of this file wrongly described the fixture as unused.)
