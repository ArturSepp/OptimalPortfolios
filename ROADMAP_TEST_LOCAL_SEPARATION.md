# Roadmap: pytest and source-adjacent development runners

**Status:** Completed

**Approved:** 2026-08-21

**Scope:** Repository layout, naming, package discovery, documentation, and verification. No
optimiser, covariance, signal, universe, or backtest behaviour changes.

## Execution contracts

| Purpose | Location | Filename | Entrypoint |
| --- | --- | --- | --- |
| Automated package test | nearest `tests/` | `<subject>_test.py` | pytest |
| Component development runner | nearest `run_local/` | `<subject>_run.py` | `run_local(local=Locals.<SCENARIO>)` |
| Shared runner helper | `optimalportfolios/run_local/` | descriptive `.py` | imported only by runners |
| Broader analytical example | `examples/<domain>/` | descriptive `.py` | `python -m examples...` |
| Local-data analytical example | `examples/<domain>/` | `<subject>_local.py` | explicit; excluded from unattended lanes |

Tests and runners are colocated with the code they support. `examples/` is reserved for larger
workflows rather than component-development utilities.

## Target map

```text
src/optimalportfolios/
  alphas/
    tests/
    signals/
      tests/
      run_local/
        signals_run.py
  covar_estimation/
    tests/
    run_local/
      ewma_covar_estimator_run.py
      factor_covar_estimator_run.py
  optimization/
    tests/
    run_local/
      constraints_run.py
    general/run_local/
    saa/run_local/
    taa/run_local/
    risk_allocation/run_local/
  universe/
    tests/
    run_local/
      universe_data_run.py
  utils/
    tests/
    run_local/
  run_local/data/
    etf_prices.py
    etf_prices_run.py

examples/
  <domain>/
    <broader workflow>.py
```

## Naming contract

Every executable development runner follows this shape:

```python
class Locals(Enum):
    """Available local development scenarios."""


def run_local(local: Locals) -> None:
    """Run the selected local development scenario."""


if __name__ == "__main__":
    run_local(local=Locals.DEFAULT)
```

The old names `LocalTests`, `run_local_test`, and `local_test` are not used in component runners.
Runner modules are never imported from production modules or public `__init__.py` files.

## Migration

- Move signal diagnostics to `alphas/signals/run_local/signals_run.py`.
- Move covariance diagnostics to `covar_estimation/run_local/`.
- Move constraint diagnostics to `optimization/run_local/`.
- Move solver diagnostics to their owning `general`, `saa`, `taa`, or `risk_allocation`
  `run_local/` directory.
- Move universe and utility diagnostics to their owning `run_local/` directories.
- Split the shared ETF data module into `run_local/data/etf_prices.py` and
  `run_local/data/etf_prices_run.py`.
- Keep the three broader local-data workflows under `examples/`.

## Packaging and discovery

- Pytest collects package modules ending in `_test.py`; paper replication retains its existing
  explicitly invoked `test_*.py` files.
- Coverage omits `run_local/` just as it omits tests and examples.
- Setuptools package discovery excludes every `run_local` package from wheels.
- The wheel archive check fails if a `run_local` directory or `_run.py` module is shipped.
- The unattended examples classifier continues to exclude broader `*_local.py` workflows.
- The layout test enforces runner location, suffix, `Locals`, `run_local`, `__main__`, and absence
  of pytest definitions.

## Verification

```powershell
uv run pytest src/optimalportfolios/tests/test_module_layout_test.py -q
uv run pytest
uv run pytest --cov=optimalportfolios --cov-report=term-missing
uv run --only-group lint ruff check --select TID251,TID253,ICN,F src/optimalportfolios/
uv run --only-group lint interrogate -v
uv run python .github/scripts/run_examples.py --list
uv build --wheel --out-dir tmp/test-run-local/dist
```

The wheel must retain package pytest modules and the offline fixture while containing no
development runner. Run its shipped tests outside the checkout and remove task-specific temporary
artifacts afterward.

## Acceptance criteria

- [x] Every component runner is in the nearest source-adjacent `run_local/` directory.
- [x] Every executable runner ends in `_run.py` and exposes `Locals` plus `run_local(local=...)`.
- [x] Shared development helpers live under `optimalportfolios.run_local` and are runner-only.
- [x] Package pytest files remain `_test.py` and contain no executable runners.
- [x] `examples/` contains broader workflows rather than component-development runners.
- [x] The wheel contains tests and fixtures but no `run_local` package or `_run.py` module.
- [x] Dependencies, public API, and numerical behaviour are unchanged.
- [x] Full tests, coverage, Ruff, interrogate, examples, replication, and wheel checks pass.

## Completion evidence

- 17 executable source-adjacent runners and one shared ETF-data helper follow the new contract.
- The source suite passes 1,407 tests at 100% line coverage; Ruff and interrogate are green.
- All 6 offline examples and all 14 public CMA replication tests pass.
- The wheel retains 68 pytest modules, two support modules, and the offline fixture, ships no
  development runner, and passes 1,397 installed-package tests with 10 expected skips.

## Out of scope

- Moving package pytest modules to a top-level `tests/` tree.
- Refactoring the scenario selection used by broader examples.
- Making local-data runners unattended.
- Changing numerical inputs, expected results, solver settings, or data conventions.
- Version bumps or dependency changes.
