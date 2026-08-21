# Paper Code

This directory contains research code associated with papers that use `optimalportfolios`.
The folders do not all provide the same level of reproduction: the table below states what a
public checkout can run and records package versions only where the repository preserves them.

## Index and public reproduction status

| Subdirectory | Paper | What a public checkout reproduces | Recorded package versions |
|---|---|---|---|
| [`crypto_allocation_risk_2023/`](crypto_allocation_risk_2023/) | Sepp (2023), *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*, *Risk* | Historical CSV files are committed, but the scripts also use live `yfinance` data, one report imports the optional `pybloqs` backend, and no frozen environment is recorded. The repository therefore does not promise exact headline-number reproduction. | Not recorded |
| [`robust_optimisation_jpm_2026/`](robust_optimisation_jpm_2026/) | Sepp, Ossa and Kastenholz (2026), *Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios*, *The Journal of Portfolio Management* 52(4), 86–120 | The folder demonstrates the published HCGL and risk-budgeting workflow, but downloads its ETF panel from `yfinance` and carries neither frozen inputs nor an environment pin. It is a methodological example, not an exact exhibit rebuild. | Not recorded |

`cma_data/` is the shared, manifest-verified CMA data layer; it is not a third paper. Its public
snapshot includes the configuration tables used by local paper workspaces and deliberately omits
licensed index, factor-history and provider panels. The MATF-CMA manuscript workspace is local and
gitignored rather than part of the public repository.

## Conventions

- Each paper folder documents its own run commands and data requirements. A missing input is a
  declared limitation, not something to replace with synthetic or live data silently.
- Paper folders are repository-only research artifacts and are not installed by
  `pip install optimalportfolios`.
- Frozen package versions are quoted from a committed manifest when one exists. The older
  folders have no recorded environment, so no version is inferred after the fact.
