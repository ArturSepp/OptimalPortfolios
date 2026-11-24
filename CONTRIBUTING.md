---
myst:
  html_meta:
    description: >-
      Contribution scope, reproducible starter tasks, development checks and review for OptimalPortfolios.
---

# Contributing to OptimalPortfolios

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-21](https://github.com/ArturSepp/OptimalPortfolios/commit/574ba518a801501dd73a515d28672dc72ca1b43f)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Thanks for your interest in `optimalportfolios`. `optimalportfolios` is the reference implementation of the ROSAA framework published in *The Journal of Portfolio Management*, so published results constrain what can change.

## Scope

In scope:

- Bug fixes in optimisation, constraints, covariance estimation, or the backtest engine
- New optimisation objectives or constraint types with a reference for the formulation
- Numerical robustness improvements, with a test that demonstrates the failure case
- Documentation, examples, and tests

Out of scope — these will be declined, so please open an issue to discuss before
writing code:

- A third optimisation backend. The package uses `cvxpy` and `quadprog`
- Reimplementations of analytics or plotting owned by
  [`qis`](https://github.com/ArturSepp/QuantInvestStrats), or factor estimation owned by
  FactorModels and [`factorlasso`](https://github.com/ArturSepp/factorlasso). The OP library
  consumes QIS and supplied risk inputs, without importing either factor package
- Silent changes to optimiser defaults, constraint semantics, or rebalancing conventions
- Changes to `papers/`, which accompanies the published papers
- Examples that require a paid data subscription to run

## Starter tasks

These are proposed, unassigned tasks, not reports of confirmed numerical defects. Check for
existing or completed work and agree a bounded scope before starting. Follow the
[development setup](#development-setup) and [pull request guidelines](#pull-requests)
below for setup, review and validation. Reuse this
repository's [feature request form](.github/ISSUE_TEMPLATE/feature_request.yml), or its
[bug report form](.github/ISSUE_TEMPLATE/bug_report.yml) for an observed defect.
No starter label implies that numerical redesign is included.

Commands below use PowerShell's `& $py`, where `$py` is the selected development interpreter.
Run them from the repository source root (a C-local source export on the maintainer host).
Follow [development setup](#development-setup) first; for this Windows/OneDrive checkout,
use the external environment and generated-state helper described in [AGENTS.md](AGENTS.md)
before running Python or uv. It gives the required interpreter and generated-state paths.
Each command is a focused check, not a substitute for the package's documentation and PR gates.

### OP-1: Add a short example-classification rehearsal

**Owner and code home:** OptimalPortfolios; [CONTRIBUTING.md](CONTRIBUTING.md).
**Read first:** [.github/scripts/run_examples.py](.github/scripts/run_examples.py), [docs/examples_readme.md](docs/examples_readme.md).

**Current gap:** The contributor guide separates tests and runners but does not walk through classifying a prospective example before running it.

**Reproduce:** Run the classifier with --list. It prints the complete inventory without executing examples; --lane does not filter this listing.

**Acceptance and scope:** Add three steps using an existing listed example: locate its imports, explain its actual offline/network classification, and identify the appropriate existing runner. Explain transitive yfinance imports and the *_local.py exclusion. Link the examples catalogue; do not hardcode moving lane counts or run network examples as a required PR check.

**Useful verification:**

```powershell
& $py .github/scripts/run_examples.py --list
```

### OP-2: Document recovery from a quickstart notebook mismatch

**Owner and code home:** OptimalPortfolios; [CONTRIBUTING.md](CONTRIBUTING.md).
**Read first:** [.github/scripts/check_quickstart_notebook.py](.github/scripts/check_quickstart_notebook.py), [examples/getting_started/production_quickstart.py](examples/getting_started/production_quickstart.py), [examples/getting_started/production_quickstart.ipynb](examples/getting_started/production_quickstart.ipynb).

**Current gap:** The parity checker explains drift in its output, but the contributor guide has no small recovery recipe.

**Reproduce:** Run the parity checker unchanged. In a disposable C-local source export, alter one character only in the tagged mirror cell, observe the checker fail, then restore the original cell and rerun.

**Acceptance and scope:** Document canonical-script-first editing, verbatim tagged-cell synchronisation, removal of execution outputs and the existing parity command. Preserve install/version cells, numerical parameters and notebook tags. Record the deliberate failure and restored success; do not replace the checker or maintain a second analytical workflow.

**Useful verification:**

```powershell
& $py .github/scripts/check_quickstart_notebook.py
```

### OP-3: Show contributors where each quickstart layer belongs

**Owner and code home:** OptimalPortfolios; [CONTRIBUTING.md](CONTRIBUTING.md).
**Read first:** [examples/getting_started/production_quickstart.py](examples/getting_started/production_quickstart.py), [src/optimalportfolios/tests/quickstart_documentation_test.py](src/optimalportfolios/tests/quickstart_documentation_test.py).

**Current gap:** The guide states package boundaries but has no call-by-call ownership map for its runnable first workflow.

**Reproduce:** Open the canonical quickstart imports and main function, then run its existing documentation contract tests.

**Acceptance and scope:** Add a compact map: packaged fixture and EwmaCovarEstimator/Constraints/compute_rolling_optimal_weights belong to OP; TimePeriod/backtest_model_portfolio and realised holdings analytics belong to qis. Link actual source locations and the existing quickstart tests. Keep estimation versus implementation timing explicit. Do not copy a backtester, covariance engine or performance statistic into a new module.

**Useful verification:**

```powershell
& $py -m pytest src/optimalportfolios/tests/quickstart_documentation_test.py -q
```

## Reporting a bug

Open an issue using the bug report template. A report needs the `optimalportfolios` version, your
Python version, a minimal self-contained reproducer, and the full traceback or the
incorrect numbers. Reproducers that depend on proprietary or licensed data cannot be
run, so please use generated or public data.

## Asking a question

Open an issue and describe what you are trying to do. Questions about methodology are
welcome; where a question is really about the published papers, please say which paper
and section you are reading.

Support is best-effort and has no guaranteed response time. The maintainer decision model,
maintenance expectations, release policy, and private route for sensitive security or conduct
reports are documented in [GOVERNANCE.md](GOVERNANCE.md).

## Development setup

For the maintainer's Windows/OneDrive checkout, first run the setup helper named in
`AGENTS.md` and select the registered C:\Python interpreter. Set
`UV_PROJECT_ENVIRONMENT` to that external environment before any uv project operation;
never use the unqualified uv examples below against a OneDrive-local environment.
Keep generated output and pure verification source exports on C: as described in the
[repository workspace policy](AGENTS.md#temporary-workspace-hygiene).

```bash
git clone https://github.com/ArturSepp/OptimalPortfolios.git
cd OptimalPortfolios
uv sync --locked --group test                            # editable install, versions from uv.lock
uv run --no-sync pytest                                  # the full suite; a few minutes
uv run --locked --only-group lint ruff check --select TID251,TID253,ICN,F src/optimalportfolios/
uv run --locked --only-group lint interrogate -v         # docstring coverage, must stay at 100%
uv run --locked --only-group lint deptry src             # declared dependencies against imports
uv run --no-sync pytest --cov=optimalportfolios --cov-report=term-missing  # floor is fail_under = 100
```

The three lint commands are the exact invocations `static.yml` gates with. The coverage command is
what the ubuntu/3.12 cell of `ci.yml` runs, with `--locked` added — that cell enforces the lock on
its `uv sync` step instead, so the effect is the same.

Automated tests and development diagnostics use separate names and commands:

```bash
uv run --no-sync pytest src/optimalportfolios/optimization/constraints/tests/constraints_test.py -v
uv run --no-sync python -m optimalportfolios.optimization.general.run_local.quadratic_run
```

Files ending in `*_test.py` are offline pytest modules and are included in the wheel. Component
development runners live in the nearest source-adjacent `run_local/` directory and end in
`*_run.py`; each exposes `Locals` and `run_local(local=...)`. They may plot, print, download data,
or use local files, so pytest ignores them and built distributions exclude them. Reserve
`examples/` for broader analytical workflows.

`--locked` fails rather than re-resolving if `uv.lock` has drifted from `pyproject.toml`. That is
the point: a dependency edit that has not been re-locked fails here, on your machine, instead of
on the pinned CI cell. If you are deliberately changing dependencies, run `uv lock` first (or drop
the flag until you do).

`ruff` and `interrogate` are reached through `--only-group lint` rather than from the `test`
dependency group, and that is deliberate: they are declared once in the `lint` dependency-group, which is
also where the workflow takes its versions from, so a local `ruff` and CI's `ruff` cannot
disagree about the same file. `--only-group` installs that group alone — not the project, not the
compiled scientific stack. The `test` group gives you `pytest` and `pytest-cov` only; it will
**not** put `ruff` or `interrogate` on your path. PEP 735 groups are a contributor environment
contract rather than installable package features, so use `uv sync --group test` instead of a pip
extra.

Note that `ruff check` is run with an explicit `--select`. Running it bare applies the `E`/`W`
families configured in `pyproject.toml`, which report a deliberate backlog of ~215 `E501`
line-length findings in the older modules. Fix only the lines your change touches; a
repository-wide reflow is not wanted.

The dependency audit is not a source gate — its answer depends on the advisory database rather
than on your diff — so it runs daily in `audit.yml` and on pull requests only when
`pyproject.toml` or `uv.lock` change. If your change touches either, run the same two-tree form
the workflow uses rather than a bare `pip-audit .`, which covers the core tree alone and silently
omits every optional extra:

```bash
uv pip compile --all-extras --python-version 3.12 --quiet pyproject.toml -o /tmp/requirements-fresh.txt
uv run --locked --only-group audit --python 3.12 pip-audit -r /tmp/requirements-fresh.txt
uv export --locked --all-extras --no-emit-project --quiet --format requirements-txt -o /tmp/requirements-locked.txt
uv run --locked --only-group audit --python 3.12 pip-audit -r /tmp/requirements-locked.txt
```

The same four commands under Windows PowerShell:

```powershell
uv pip compile --all-extras --python-version 3.12 --quiet pyproject.toml -o "$env:TEMP\requirements-fresh.txt"
uv run --locked --only-group audit --python 3.12 pip-audit -r "$env:TEMP\requirements-fresh.txt"
uv export --locked --all-extras --no-emit-project --quiet --format requirements-txt -o "$env:TEMP\requirements-locked.txt"
uv run --locked --only-group audit --python 3.12 pip-audit -r "$env:TEMP\requirements-locked.txt"
```

The two trees answer different questions, which is why the workflow runs both. The `uv pip compile`
tree is what a *fresh* install resolves to today from the floors in `pyproject.toml`; the
`uv export --locked` tree is the exact pinned set `uv sync --locked` installs, and a pin can sit on
a vulnerable version long after the floor would resolve past it. `--no-emit-project` drops the
local package, which has no release for `pip-audit` to look up.

Two differences from the workflow, both because your machine is not the runner. `--python 3.12` is
spelled out on the `pip-audit` calls: the requirements files are resolved for CPython 3.12, and
`pip-audit` resolves them again against the interpreter it is running on, so on any other version
it fails on a pin it cannot satisfy rather than reporting a finding. The runner is already 3.12,
so `audit.yml` does not need the flag. And `--only-group` reinstalls the project virtualenv with
that group alone; the next `uv run --locked pytest` syncs the development environment back, but do
not be surprised by the reinstall.

To verify a built or downloaded wheel in a clean environment, install the wheel and `pytest`,
then run the supported post-install check:

```bash
python -m pytest --pyargs optimalportfolios
```

`AGENTS.md` in this repository documents the layout, commands, conventions, and
constraints in more detail — it is written for AI coding agents but is equally useful
to human contributors.

## Pull requests

- One topic per pull request. Unrelated changes in the same PR make review slower and
  are likely to be asked to split.
- Add or update tests for behaviour you change. A bug fix should come with a test that
  fails before the fix.
- Run the documented CI command set before submitting.
- Do not bump the version in `pyproject.toml` or `CITATION.cff`; releases are cut
  separately.
- Do not commit generated output: figures, factsheets, backtest results, or data files.
- Keep the public API stable. If a change alters a public signature or default, say so
  explicitly in the PR description.

## Replication

`papers/` reproduces results from the published papers. If your change alters
optimiser behaviour, covariance estimation, or backtest mechanics, please re-run the
relevant scripts and confirm the published tables still reproduce. If they do not,
report the difference in the PR rather than updating the expected values.

The public, offline CMA snapshot suite runs in `replication.yml` on every push and pull request.
Run the same gate locally with:

```bash
uv run --no-sync pytest papers/cma_data/tests -q
```

## Conduct

Be civil and assume good faith. Technical disagreement is welcome; personal remarks are
not.

## Licence

This project is MIT licensed. By contributing, you agree that your contributions are licensed under
the MIT licence of this project.
