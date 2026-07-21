# AGENTS.md

Guidance for AI coding agents working in the **OptimalPortfolios** repository.

## Project overview

`optimalportfolios` implements the production pipeline for multi-asset portfolio
construction and backtesting: alpha signals -> covariance estimation (EWMA or the HCGL
factor model from `factorlasso`) -> constrained optimisation (risk budgeting, maximum
diversification, maximum Sharpe, alpha over tracking error, and others) -> rolling
backtest and reporting through `qis`.

It is the reference implementation of the ROSAA framework published in *The Journal of
Portfolio Management* (Sepp, Ossa and Kastenholz, 2026). Distribution and import name
`optimalportfolios`. Licensed MIT (`LICENSE.txt`).

## Ecosystem position

This package is one of eight open-source Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Before implementing anything
non-trivial, check whether it already exists in one of these:

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

Actual package dependencies within the stack: `optimalportfolios` depends on `qis`
and `factorlasso`; `trendfollowing` depends on `qis`; `stochvolmodels` has an
optional `research` extra that pulls in `qis`. The others are independent.

Do not vendor or copy code between these packages. If functionality belongs in a
sibling package, say so rather than reimplementing it here.

## Repository layout

```
optimalportfolios/
  alphas/            alpha signal construction
  covar_estimation/  covariance estimators (EWMA, factor/HCGL via factorlasso)
  optimization/      optimisers, constraints, solvers
  universe/          instrument universes
  reports/           reporting built on qis
  utils/, examples/, docs/, config.py, local_path.py, settings.yaml
paper_code/          code accompanying the published papers (excluded from ruff)
```

Tests live inside the package as `optimalportfolios/<subpackage>/tests/*_test.py`;
there is no top-level `tests/` directory.

## Commands

```bash
pip install -e ".[dev]"                                  # editable install with dev tools
pytest optimalportfolios/                                # run the test suite
pytest optimalportfolios/optimization/tests/constraints_test.py -v
ruff check optimalportfolios/                            # lint (paper_code is excluded)
```

Optional extras: `data`, `reports`, `visualization`, `jupyter`, `dev`, `all`.
Supported Python is >= 3.10; CI runs 3.10 – 3.12.

## Conventions

- Test files are named `*_test.py` and live in a `tests/` directory inside the
  subpackage under test.
- Line length 100 (`ruff`, rules `E`, `F`, `W`, `I`); `paper_code/` is excluded from
  linting on purpose.
- Optimisation problems are expressed with `cvxpy`; `quadprog` is used where a
  dedicated QP solver is faster. Do not introduce a third optimisation backend.
- Enums and dataclasses carry configuration (optimiser type, constraint sets,
  estimation settings) — extend the existing enum rather than passing raw strings.
- Time series are pandas objects with a `DatetimeIndex`; the backtest layer is
  NaN-aware by design, so preserve NaN handling when refactoring.
- Reporting and plotting go through `qis`; do not add a parallel plotting layer here.

## Constraints — do not do these

- Do not reimplement covariance estimation that belongs in `factorlasso`, or analytics
  and factsheets that belong in `qis`. Both are declared dependencies — import them.
- Do not silently change optimiser defaults, constraint semantics, or rebalancing
  conventions: published results depend on them.
- Do not edit `paper_code/` to make linting pass; it is excluded deliberately.
- Do not add a hard dependency on Bloomberg data. Examples run on free data.
- Do not commit backtest output, factsheets, or figures.

## Replication contract

`paper_code/` reproduces results from the published papers. If a change alters
optimiser behaviour, covariance estimation, or backtest mechanics, re-run the relevant
scripts in `paper_code/` and confirm the outputs still match the published tables
before proposing the change.

## Release checklist

A release touches three version locations. All three must agree:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the software BibTeX entry in `README.md` (if it pins a version)

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.

## Known issues

Two stale artefacts in this repository, safe to fix if asked:
`[tool.pytest.ini_options] testpaths = ["tests"]` points at a non-existent directory,
so a bare `pytest` collects nothing — always pass an explicit path. The previous
`CLAUDE.md` described version 4.1.1 and a black/isort/flake8/mypy toolchain; the
project has since moved to `ruff` and this file supersedes it.
