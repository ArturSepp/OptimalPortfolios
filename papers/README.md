# Paper Code

This directory contains research code associated with papers that use `optimalportfolios`.
The folders do not all provide the same level of reproduction: the table below states what a
public checkout can run and records package versions only where the repository preserves them.

## Index and public reproduction status

| Subdirectory | Paper | What a public checkout reproduces | Recorded package versions |
|---|---|---|---|
| [`crypto_allocation_risk_2023/`](crypto_allocation_risk_2023/) | Sepp (2023), *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*, *Risk* | Historical CSV files are committed, but the scripts also use live `yfinance` data, one report imports the optional `pybloqs` backend, and no frozen environment is recorded. The repository therefore does not promise exact headline-number reproduction. | Not recorded |
| [`robust_optimisation_jpm_2026/`](robust_optimisation_jpm_2026/) | Sepp, Ossa and Kastenholz (2026), *Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios*, *The Journal of Portfolio Management* 52(4), 86–120 | The folder demonstrates the published HCGL and risk-budgeting workflow, but downloads its ETF panel from `yfinance` and carries neither frozen inputs nor an environment pin. It is a methodological example, not an exact exhibit rebuild. | Not recorded |
| [`smart_diversification_joim_2026/`](smart_diversification_joim_2026/) | Sepp and Kastenholz (2026), *The Convexity Premium of Portfolio Overlays*, *Journal of Investment Management*, forthcoming | A public introduction and an offline synthetic 60/40 overlay example using OP optimization and qis illustrations. The manuscript, empirical replication and licensed data remain local. | Not recorded; the README states the minimum and last-tested optimalportfolios and qis versions |

`cma_data/` is the shared, manifest-verified CMA data layer; it is not a paper. Its public
snapshot includes the configuration tables used by local paper workspaces and deliberately omits
licensed index, factor-history and provider panels. The MATF-CMA manuscript workspace is local and
gitignored rather than part of the public repository.

`prior_targets_2026/` is an entirely local workspace migrated from FactorLasso. It holds
the prior-selection manuscript and replication, with MATF versus Bloomberg MAC3 stress
and exposure comparisons retained as working evidence for the developing paper. Its final
scope is undecided. The migration does not authorise redistribution of the manuscript,
replication code or licensed inputs; no exact public reproduction is claimed.

## Conventions

- [AGENTS.md](AGENTS.md) defines the six-section paper-workspace contract, publication
  boundaries and the commands that check them. Shared `cma_data/` remains a separate module.
- The existing crypto TeX is tracked, but its PDF is local and its referenced manuscript
  figures are absent. The ROSAA folder publishes only the existing methodological example
  under `replication/`. Neither is presented as a self-contained current manuscript bundle.
- MATF-CMA and the other fully excluded research workspaces remain local. JOIM exposes
  only its approved introduction and synthetic example; its empirical replication remains
  local. Folder maintenance does not change publication rights.
- `drafts/`, `private/` and per-paper `agents/` are always ignored. `paper/` and
  `presentations/` require exact approved exceptions. Code and paper-specific tests belong
  in `replication/`, except the JOIM companion's `run_overlay_example.py`, which the
  manuscript cites at the folder root; approved static inputs in `replication/data/`, restricted inputs in
  `replication/data/local/`. Runtime outputs stay in the prescribed C-local workspace.
- Each paper folder documents its own run commands and data requirements. A missing input is a
  declared limitation, not something to replace with synthetic or live data silently.
- Paper folders are repository-only research artifacts and are not installed by
  `pip install optimalportfolios`.
- Frozen package versions are quoted from a committed manifest when one exists. The older
  folders have no recorded environment, so no version is inferred after the fact.
