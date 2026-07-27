# Paper Code

This directory contains reproduction code for published research papers that use the `optimalportfolios` package. Each subdirectory corresponds to one paper and contains the complete pipeline that generates the published exhibits.

Each subdirectory is **self-contained**: it ships its own `README.md` (with full methodology), reproduction scripts, and tests. Running the scripts in isolation reproduces the paper's headline numbers within Monte Carlo noise.

## Index

| Subdirectory | Paper | Citation |
|---|---|---|
| [`matf_cma_jpm_2026/`](matf_cma_jpm_2026/) | Capital Market Assumptions Using Multi-Asset Tradable Factors: The MATF-CMA Framework | Sepp, Hansen, Kastenholz (2026), *JPM*, under review (JPM-093244) |

## Conventions

- A per-paper directory either tracks the current package code or states the `optimalportfolios` and `qis` versions it shipped with, in its own README. A directory is **frozen at the time of paper acceptance**, and the version statement is what makes the frozen result reproducible. No directory carries such a statement yet.
- Reproduction scripts are designed to be run as standalone Python scripts from within their directory: `cd matf_cma_jpm_2026/replication && python run_bootstrap.py`.
- Per-paper subdirectories are not Python packages and do not have `__init__.py`. They are not installed by `pip install optimalportfolios`.
- Production input data files (proprietary CSV / xlsx pipeline outputs) are not committed to the repository. The methodology in each per-paper README is fully self-contained and reproducible against any equivalent pipeline.
