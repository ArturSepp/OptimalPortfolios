# Paper Code

This directory contains research code associated with papers that use `optimalportfolios`.
The folders do not all provide the same level of reproduction: the table below states what a
public checkout can run and records package versions only where the repository preserves them.

## Index and public reproduction status

| Subdirectory | Paper | What a public checkout reproduces | Recorded package versions |
|---|---|---|---|
| [`crypto_allocation_risk_2023/`](crypto_allocation_risk_2023/) | Sepp (2023), *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*, *Risk* | Historical CSV files are committed, but the scripts also use live `yfinance` data, one report imports the optional `pybloqs` backend, and no frozen environment is recorded. The repository therefore does not promise exact headline-number reproduction. | Not recorded |
| [`robust_optimisation_jpm_2026/`](robust_optimisation_jpm_2026/) | Sepp, Ossa and Kastenholz (2026), *Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios*, *The Journal of Portfolio Management* 52(4), 86–120 | The folder demonstrates the published HCGL and risk-budgeting workflow, but downloads its ETF panel from `yfinance` and carries neither frozen inputs nor an environment pin. It is a methodological example, not an exact exhibit rebuild. | Not recorded |
| [`matf_cma_jpm_2026/`](matf_cma_jpm_2026/) | Sepp, Hansen and Kastenholz (2026), *Capital Market Assumptions Using Multi-Asset Tradable Factors: The MATF-CMA Framework*, under review | The committed 2026-Q2 snapshot runs 8 of 12 replication scripts in full. Four scripts need licensed return panels that are not committed; the exact files and affected outputs are listed in the [replication README](matf_cma_jpm_2026/replication/README.md#what-runs-on-a-public-checkout). | Snapshot manifest: `optimalportfolios` 6.6.0, `qis` 5.0.5, `factorlasso` 0.10.1 |

`cma_data/` is the shared, manifest-verified data layer used by the MATF-CMA replication; it is
not a fourth paper. Its public snapshot includes the configuration tables that the public
scripts need and deliberately omits licensed index, factor-history and provider panels.

## Conventions

- Each paper folder documents its own run commands and data requirements. A missing input is a
  declared limitation, not something to replace with synthetic or live data silently.
- Paper folders are repository-only research artifacts and are not installed by
  `pip install optimalportfolios`.
- Frozen package versions are quoted from a committed manifest when one exists. The older
  folders have no recorded environment, so no version is inferred after the fact.
