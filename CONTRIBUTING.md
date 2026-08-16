# Contributing to OptimalPortfolios

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
- Reimplementations of analytics or plotting that belong in
  [`qis`](https://github.com/ArturSepp/QuantInvestStrats), or of factor estimation that
  belongs in [`factorlasso`](https://github.com/ArturSepp/factorlasso) — both are
  declared dependencies
- Silent changes to optimiser defaults, constraint semantics, or rebalancing conventions
- Changes to `papers/`, which accompanies the published papers
- Examples that require a paid data subscription to run

## Reporting a bug

Open an issue using the bug report template. A report needs the `optimalportfolios` version, your
Python version, a minimal self-contained reproducer, and the full traceback or the
incorrect numbers. Reproducers that depend on proprietary or licensed data cannot be
run, so please use generated or public data.

## Asking a question

Open an issue and describe what you are trying to do. Questions about methodology are
welcome; where a question is really about the published papers, please say which paper
and section you are reading.

## Development setup

```bash
git clone https://github.com/ArturSepp/OptimalPortfolios.git
cd OptimalPortfolios
uv sync --extra dev                                      # editable install, versions from uv.lock
uv run --locked pytest                                   # the full suite; a few minutes
uv run --locked --only-group lint ruff check --select TID251,TID253,ICN,F src/optimalportfolios/
uv run --locked --only-group lint interrogate -v         # docstring coverage, must stay at 100%
uv run --locked pytest --cov=optimalportfolios --cov-report=term-missing   # floor is fail_under = 100
```

The two lint commands are the exact invocations `static.yml` gates with. The coverage command is
what the ubuntu/3.12 cell of `ci.yml` runs, with `--locked` added — that cell enforces the lock on
its `uv sync` step instead, so the effect is the same.

`--locked` fails rather than re-resolving if `uv.lock` has drifted from `pyproject.toml`. That is
the point: a dependency edit that has not been re-locked fails here, on your machine, instead of
on the pinned CI cell. If you are deliberately changing dependencies, run `uv lock` first (or drop
the flag until you do).

`ruff` and `interrogate` are reached through `--only-group lint` rather than from the `dev`
extra, and that is deliberate: they are declared once in the `lint` dependency-group, which is
also where the workflow takes its versions from, so a local `ruff` and CI's `ruff` cannot
disagree about the same file. `--only-group` installs that group alone — not the project, not the
compiled scientific stack. A plain `pip install -e ".[dev]"` gives you `pytest` and `pytest-cov`
only; it will **not** put `ruff` or `interrogate` on your path, because a pip extra does not
install a PEP 735 dependency-group. It also resolves fresh rather than from `uv.lock`, so it is
not what CI gates the pinned cell against.

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

The public, offline CMA snapshot and manuscript-parity suites run in `replication.yml` on every
push and pull request. Run the same gate locally with:

```bash
uv run --no-sync pytest papers/cma_data/tests papers/matf_cma_jpm_2026/replication/tests -q
```

## Conduct

Be civil and assume good faith. Technical disagreement is welcome; personal remarks are
not.

## Licence

This project is MIT licensed. By contributing, you agree that your contributions are licensed under
the MIT licence of this project.
