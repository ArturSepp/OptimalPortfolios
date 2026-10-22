---
myst:
  html_meta:
    description: >-
      Install OptimalPortfolios, select supported extras, verify the active interpreter,
      run the offline quickstart and distinguish released packages from contributor environments.
---

# Installation

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-09](https://github.com/ArturSepp/OptimalPortfolios/commit/254505981ed43e0dbc12a19c98c034d351b8d059)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

OptimalPortfolios optimizes portfolios from supplied expected returns and covariances and
supports rolling allocation workflows. Its core dependencies include
[QIS](https://github.com/ArturSepp/QuantInvestStrats) for analytics and backtesting.
The core package supports an offline workflow; optional integrations add data downloads
and report backends. Factor estimation belongs to separately installed upstream packages.

This guide follows the checkout's [packaging metadata](../pyproject.toml). A PyPI installation
uses the metadata of the released version it selects; that version can differ from the current
development checkout.

## Install the released package

The declared Python requirement is **Python 3.11 or later** (`requires-python = ">=3.11"`).
Binary dependencies also determine which interpreter/platform combinations can be installed.
The current [CI matrix](../.github/workflows/ci.yml) exercises Python 3.11–3.14 on Linux,
macOS and Windows. The optional risk-budgeting backend is checked separately on Python 3.12;
its binary-wheel availability can differ from the base package.
The metadata requirement alone does not certify every future Python release.

Use an environment dedicated to your project. In the commands below, `python` means that
environment's interpreter; `python -m pip` installs into the interpreter being invoked.

```console
python -m pip install optimalportfolios
```

A release install includes the numerical solvers, covariance integration, QIS backtesting and
ordinary analytics/reporting dependencies. It does not require a Bloomberg terminal, a Yahoo
account, or notebook software to import the core package and run the offline quickstart.

For this repository's Windows/OneDrive checkout, use the external environment and generated-state
setup in [AGENTS.md](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md) before
running Python tools. Its interpreter is `C:\Python\OptimalPortfolios312\Scripts\python.exe`.
Do not create or use a Python environment under OneDrive.

## Choose optional integrations

The checkout declares these five package extras. The `riskbudget` split is part of
unreleased 8.0.0 development; older published versions still install quadprog as a core dependency.

| Extra | Adds | When to select it |
|---|---|---|
| `exact` | MOSEK Python interface | Optional prepared-execution integer solver, with a separately available solver licence. |
| `riskbudget` | `quadprog` | Risk-budgeting projection and solver; the backend retains GPLv2+ terms. |
| `data` | `yfinance` | Examples or loaders that download Yahoo Finance data. |
| `reports` | `pybloqs` | The dedicated HTML/PDF report backend. |
| `docs` | Sphinx, Furo, MyST and the Mermaid extension | Building the documentation from a checkout. |

Install the runtime features needed by the workflow with quoted package requirements:

```console
python -m pip install "optimalportfolios[riskbudget]"
python -m pip install "optimalportfolios[exact]"
python -m pip install "optimalportfolios[data]"
python -m pip install "optimalportfolios[reports]"
python -m pip install "optimalportfolios[data,reports]"
```

These are alternatives; choose the integrations needed by the workflow. Installing `reports`
adds the Python backend, not a guarantee that every external PDF-rendering prerequisite is
available on the machine.

The prepared [execution research API](execution_v2.md) is explicitly imported as
`optimalportfolios.execution.v2`. Its ranked solver uses the ordinary numerical
dependencies. The `exact` extra adds the optional native integer solver, whose
start adapter checks its qualified canonicalisation versions. It does not switch
the existing production execution interface to v2. The paper's frozen software
bundle retains its own research identity independently of this development checkout.

OP's own source licence is MIT. Dependencies keep their own licences: installing
`riskbudget` adds the GPLv2+ quadprog backend. The reviewed Windows/Python 3.12 base
also retains GCC runtime exceptions in NumPy/SciPy, Microsoft runtime terms,
Numba's NVIDIA header terms, and third-party font notices; it is not an all-MIT
binary environment. The packaged [supplemental notices](https://github.com/ArturSepp/OptimalPortfolios/blob/2ed63d8c53a14540c17bbb71f8f7651255705643/THIRD_PARTY_NOTICES.md)
preserve the full OpenBLAS/LAPACK text for the exact SCS/SparseDiffPy binaries reviewed.
Other versions and platforms need their own review.

Without `riskbudget`, public imports, covariance utilities and unrelated optimisers
remain available. Calling the risk-budgeting solver raises an installation message;
it does not silently select another algorithm or return a fallback portfolio.

**`all` is a contributor dependency group, not a package extra.** The current metadata does not
publish `optimalportfolios[all]`. Use `optimalportfolios[data,reports]` for both runtime
integrations. Package extras and dependency groups have different installation semantics;
[PyPA's specification](https://packaging.python.org/en/latest/specifications/dependency-groups/)
explains why groups are not included as package metadata in distributions.

The historical `all` contributor group continues to select `data,reports`. It deliberately
does not select the GPL-licensed risk-budgeting backend; add `--extra riskbudget` explicitly.

There is no `jupyter`, `clustering`, `test` or `dev` package extra. Install local notebook tooling
separately. The repository's Colab quickstart uses its hosted notebook runtime. Risk-lineage
matching uses the existing NumPy/SciPy path.

Composed factor workflows use
[FactorModels](supplied_risk_inputs.md#factormodels-availability) to construct
factor models and covariances, then supply those estimates to OptimalPortfolios.
Install the separately published [factormodels distribution](https://pypi.org/project/factormodels/)
with `pip install factormodels`; composed CI verifies `factormodels==0.1.0`.
Its source repository and method guides retain their existing access requirements.
It includes
[FactorLasso](https://github.com/ArturSepp/FactorLasso) for estimation and
cluster-lineage analytics.

For a workflow that uses FactorLasso's estimator directly, install `factorlasso` separately.
Neither package is required by the base OP installation or the offline quickstart below.

## Verify the interpreter and imports

Run this block with the interpreter used for installation, for example in its Python console
or saved as a short script:

```python
import sys
from importlib.metadata import metadata, version
import optimalportfolios as opt
import qis

versions = {
    name: version(name)
    for name in ("optimalportfolios", "qis")
}
available_extras = sorted(metadata("optimalportfolios").get_all("Provides-Extra") or [])
print("Python:", sys.executable)
print("Versions:", versions)
print("OptimalPortfolios source:", opt.__file__)
print("Available extras:", available_extras)
assert callable(opt.compute_rolling_optimal_weights)
assert callable(qis.backtest_model_portfolio)
```

The source path helps identify an editable checkout or an unexpected installation. Version metadata
and the import path are complementary: an editable source change does not automatically update
installed distribution metadata. `Provides-Extra` lists extras the installed distribution
**offers**; it does not say which optional dependencies are currently installed.

For an environment dependency check:

```console
python -m pip check
```

A successful import or `pip check` is a useful installation check, not numerical validation of
every solver, report backend or strategy.

## Run the first offline workflow

The [production quickstart](../examples/getting_started/production_quickstart.py) uses the
monthly multi-asset fixture shipped with the package. From a checkout, run:

```console
python examples/getting_started/production_quickstart.py
```

A wheel installation does not include the repository's `examples/` directory. Open the linked
script and save a copy before running it outside a checkout, or follow the [quickstart guide](quickstart.md).
The script requires only the core package, writes no files, and prints input dates, rolling-weight
dimensions, final weights, final NAV and elapsed time. Its fixed sample is a teaching workflow,
not an updated market-data report.

Only the examples that fetch Yahoo data need `data`. The `composed` examples additionally need
the separately installed factor packages; `offline` examples use the base package. Local
diagnostics have explicit data prerequisites. From a checkout, inspect the classifier without
running any examples:

```console
python .github/scripts/run_examples.py --list
```

See the [examples guide](examples_readme.md) for the workflow map. Source-adjacent `run_local`
diagnostics are development tools and are excluded from the distribution.

## Work from a source checkout

An editable installation points imports at the local source; install it from the repository root
using the selected external interpreter:

```console
python -m pip install -e .
```

For documentation work with pip:

```console
python -m pip install -e ".[docs]"
```

Pip resolves the declared requirements; it does not consume `uv.lock`, so these commands do not
reproduce the locked environment.

### Contributor groups and the lockfile

The checkout's PEP 735 groups are:

| Group | Purpose |
|---|---|
| `test` | `pytest` and `pytest-cov` for automated checks. |
| `lint` | Pinned `ruff`, `interrogate` and `deptry` for source gates. |
| `audit` | Pinned `pip-audit` for dependency-advisory checks. |
| `all` | Requests `optimalportfolios[data,reports]` for contributor environments. |

Groups are selected from a checkout, not through a bracketed PyPI extra. The `lint` and
`audit` groups are separate from `test` and are not selected by the basic test command.

Before a uv project operation, configure the project environment location. For this
Windows/OneDrive repository, run the setup required by AGENTS.md, then use this PowerShell form:

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'C:\Python\OptimalPortfolios312'
uv sync --locked --group test
uv run --no-sync python -m pytest
```

On other platforms, set `UV_PROJECT_ENVIRONMENT` to the intended absolute environment path
before the project commands. The [uv environment-path documentation](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path)
describes the setting; without it, uv uses a repository-local environment by default.

`--locked` requires the lockfile to remain current and unchanged. `--no-sync` runs against the
existing environment and does not install missing requirements or prove that it matches the lock.
Only run the test command after a successful sync. These behaviors are documented in
[uv's locking and syncing guide](https://docs.astral.sh/uv/concepts/projects/sync/).

To include Yahoo examples, append `--extra data` to the sync command. To prepare tests, static
tools and the documentation toolchain together, keep the same environment setting and use:

```console
uv sync --locked --group test --group lint --extra docs --extra riskbudget
```

Sync selects the requested environment contents; retain the extras and groups the workflow needs
when changing that selection. Follow AGENTS.md and the owning CI workflow for the actual checks.
Documentation builds run from a C-local source export because autosummary generates source files.

Base tests skip only explicitly marked backend cases when quadprog is absent. Run those
cases with `uv run --no-sync pytest -m "riskbudget and not factor_integration"` after
installing the extra; the separate composed lane supplies FM for tests needing both. CI requires
the dedicated source and installed-wheel lanes to execute rather than silently skip them.

<a id="current-development-verification-boundary"></a>

### Locked and resolved versions

`uv.lock` pins one exact version of every dependency. CI tests pull requests in that locked
environment, and Read the Docs builds this site in it; the footer of every page names the
versions of its build. Scheduled CI runs also test the newest and the lowest versions the
declared ranges allow. A pip installation does not read the lockfile: it resolves the ranges
declared in `pyproject.toml` and can select newer releases. When a result differs between the
two, compare the versions printed by the import check above with the footer.

## Troubleshooting and next checks

| Symptom | First check |
|---|---|
| `ModuleNotFoundError` after installing | Compare `sys.executable` with the interpreter used by `python -m pip`. |
| A requested extra is missing | Inspect `Provides-Extra` for the installed version; use the current documented extra names. |
| A copied example cannot find `examples/...` | Obtain the repository-only script; installing the wheel does not create the repository tree. |
| Yahoo downloads fail | Confirm `data` is installed, then distinguish provider/network failures from core-package import failures. |
| `uv sync --locked` fails | Check TOML/lock consistency and dependency availability; keep a lock update separate and review it. |
| Native dependency installation fails | Check the Python/platform combination against the dependency's wheels and the project's CI matrix. |

The distribution includes pytest modules and the offline fixture, but pytest itself is contributor
tooling. For an installed-package test run, install pytest in that same environment and run outside
the checkout:

```console
python -m pip install pytest
python -m pytest --pyargs optimalportfolios
```

Repository-integrity tests skip when no checkout is available; that run is intentionally narrower
than the source suite. For supported workflows, continue to the [quickstart](quickstart.md),
[solver guide](optimization_module_readme.md), or [documentation standard](documentation_standard.md).

## References

- [OptimalPortfolios packaging metadata](../pyproject.toml) and [CI workflow](../.github/workflows/ci.yml).
- [pip installation reference](https://pip.pypa.io/en/stable/cli/pip_install/).
- [PyPA dependency-group specification](https://packaging.python.org/en/latest/specifications/dependency-groups/).
- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
