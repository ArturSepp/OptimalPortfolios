# Paper Code

This directory contains reproduction code for published research papers that use the `optimalportfolios` package. Each subdirectory corresponds to one paper and contains the complete pipeline that generates the published exhibits.

Each subdirectory is **self-contained**: it ships its own `README.md` (with full methodology), `requirements.txt` (with version-pinned dependencies), reproduction scripts, and tests. Running the scripts in isolation reproduces the paper's headline numbers within Monte Carlo noise.

## Index

| Subdirectory | Paper | Citation |
|---|---|---|
| [`matf_cma_2026/`](matf_cma_jpm_2026/) | Capital Market Assumptions Using Multi-Asset Tradable Factors: The MATF-CMA Framework | Sepp, Hansen, Kastenholz (2026), *JPM* forthcoming |

## Conventions

- Each per-paper directory is **frozen at the time of paper acceptance** and pinned to a specific version of `optimalportfolios` via its `requirements.txt`.
- Reproduction scripts are designed to be run as standalone Python scripts from within their directory: `cd matf_cma_2026 && python run_bootstrap.py`.
- Per-paper subdirectories are not Python packages and do not have `__init__.py`. They are not installed by `pip install optimalportfolios`.
- Production input data files (proprietary CSV / xlsx pipeline outputs) are not committed to the repository. The methodology in each per-paper README is fully self-contained and reproducible against any equivalent pipeline.
