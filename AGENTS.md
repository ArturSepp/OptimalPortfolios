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
| `factorlasso` | **>= 0.11.0** | `RollingFactorCovarData.get_alphas` forwards `asset_frequencies` / `default_freq`; below it a per-frequency `alpha_span` silently applies the `'ME'` entry to every quarterly asset |
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
  tests/             cross-cutting tests (release metadata agreement)
  utils/, examples/, docs/, config.py, local_path.py, settings.yaml
papers/              code accompanying the published papers (excluded from ruff)
```

Tests live inside the package as `optimalportfolios/<subpackage>/tests/*_test.py`; there is no top-level `tests/` directory. Not every `*_test.py` is a pytest module: sixteen of them are `run_local_test` diagnostic scripts that print and plot, contribute no collected tests, and need the author's local price data. They are still imported during collection, so they must stay importable on a core install — put an optional import inside the function that needs it and raise `ImportError` naming the extra.

## Commands

```bash
pip install -e ".[dev]"                                  # editable install with dev tools
pytest                                                   # run the test suite (180 tests, ~9 s)
pytest optimalportfolios/optimization/tests/constraints_test.py -v
ruff check optimalportfolios/                            # lint (papers/ is excluded)
```

*Note: Terminal execution should be compatible with Windows PowerShell within PyCharm.*

Optional extras: `data`, `reports`, `visualization`, `jupyter`, `dev`, `all`. Supported Python is >= 3.10; CI runs 3.10 – 3.12 on a `[dev]` install and 3.12 again on a core install, which must be green: no test may need data, network or a Bloomberg terminal.

## Conventions

- Test files are named `*_test.py` and live in a `tests/` directory inside the subpackage under test.
- Line length 100 (`ruff`, rules `E`, `F`, `W`); `papers/` is excluded from linting on purpose. `I` is deliberately not selected anywhere in the stack: imports group the scientific stack before project packages, which isort's ordering contradicts.
- **Three stack invariants are enforced by ruff rather than written down.** Unlike `E`/`F`/`W`, which report ~780 legacy findings, these are green on the whole package, so a violation is always something you just introduced:
  - `TID251` fails an import of `trendfollowing`, `privateassets`, `stochvolmodels`, `goal_based_allocation` or `vanilla_option_pricers`. This package depends on `qis` and `factorlasso` and on nothing else in the stack; subject packages never import each other. `qis` and `factorlasso` are of course not banned — they are declared dependencies, and importing them is the point.
  - `TID253` fails a **module-level** import of an optional extra (`yfinance`, `pandas_datareader`, `pybloqs`, `plotly`, `pyarrow`, `psycopg2`, `sqlalchemy`); the same import inside a function passes, which is the pattern the collection note above requires. `optimalportfolios/examples/**` and `reports/portfolio_result_pybloqs.py` are named in `per-file-ignores` — add to that list only for a module `optimalportfolios/__init__.py` cannot reach.
  - `ICN` pins `import numpy as np` and `import pandas as pd`. Ruff's default alias map is replaced rather than extended, so `matplotlib` stays free to be both `mpl` and `plt`.
- Optimisation problems are expressed with `cvxpy`; `quadprog` is used where a dedicated QP solver is faster. Do not introduce a third optimisation backend.
- Enums and dataclasses carry configuration (optimiser type, constraint sets, estimation settings) — extend the existing enum rather than passing raw strings.
- Time series are pandas objects with a `DatetimeIndex`; the backtest layer is NaN-aware by design, so preserve NaN handling when refactoring.
- Reporting and plotting go through `qis`; do not add a parallel plotting layer here.

## Implementation Directives

- **Leverage the Stack:** Always import `factorlasso` for covariance estimation and `qis` for analytics and factsheets. Do not reimplement these functionalities.
- **Preserve Core Logic:** Maintain the existing optimiser defaults, constraint semantics, and rebalancing conventions, as published results heavily depend on them.
- **Respect Linting Exclusions:** Leave `papers/` exactly as-is; it is deliberately excluded from linting to preserve published code.
- **Ensure Offline Execution:** Ensure all examples run on free data. Never add a hard dependency on Bloomberg data.
- **Maintain Clean Commits:** Prevent backtest outputs, factsheets, or generated figures from being committed to version control.

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
- `ruff check optimalportfolios/` reports around 780 findings, almost all `E501` line-length in the older modules. CI does not gate on lint. Fix only the lines your specific change touches; a repository-wide reflow is not wanted.
- **Offline Fixture Anomaly:** The 6.2.0 changelog mentions a 69-test suite and an offline fixture (`examples/data/multiasset_returns.csv` with `examples.data.multiasset.load_multiasset_data`). This fixture is committed but unused. Ignore it completely and do not attempt to integrate it into the current 180-test suite.
