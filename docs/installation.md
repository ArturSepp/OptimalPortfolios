---
icon: material/package-down
---

# Installation

OptimalPortfolios supports Python 3.10 and later. Install the core package from PyPI:

```bash
pip install optimalportfolios
```

The core install includes covariance estimation, portfolio optimisation, and backtesting.
It has no dependency on data providers, plotting backends or a Bloomberg terminal — the
test suite is green on a core install by design.

## Optional extras

| Extra | What it adds |
|---|---|
| `data` | Free-data loaders backed by `yfinance` and `pandas-datareader`. |
| `clustering` | The default minimum-cost-flow matcher used to keep risk-cluster labels stable through time. Install it when using the `mcf` cluster matcher; the `hungarian` matcher remains available in the core install. |
| `reports` | The `pybloqs` report backend. |
| `visualization` | Plotly charts. |
| `jupyter` | Notebook tooling. |
| `docs` | The MkDocs toolchain used to build this book. |
| `dev` | The test, coverage, lint, and docstring-quality tools used by contributors. It also includes the `data` and `clustering` extras so the complete test suite can run. |
| `all` | All runtime integrations. |

Install an extra by placing its name in square brackets:

```bash
pip install "optimalportfolios[clustering]"
pip install -e ".[dev]"
```

## Building the documentation

Documentation contributors install the MkDocs toolchain and serve the book locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

`mkdocs build --strict` is what CI runs — it fails on a broken internal link, a nav entry
without a file, or an API reference target that cannot be resolved.
