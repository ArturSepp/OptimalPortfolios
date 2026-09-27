---
myst:
  html_meta:
    description: >-
      The research papers behind optimalportfolios: the ROSAA framework in The Journal of
      Portfolio Management, cryptocurrency allocation in Risk and multi-asset capital market
      assumptions, with what each paper contributes and what a public checkout reproduces.
---

# Research papers and replication

*Author: [Artur Sepp](https://github.com/ArturSepp)*

The methods of [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios) are described
in the papers below. This page lists each paper once, with the citation used everywhere on the
site, what it contributes to the package, and what the repository's
[paper folders](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers) let a public
checkout reproduce.

Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## How the papers are used

Pages cite a paper by section and equation. A result from a paper is quoted with its study
design and is never restated as a general performance claim. Exhibits on this site are either
regenerated from code and data tracked in the repository or synthetic analogues drawn from a
page's own script; figures of the papers themselves are not reproduced here. Research that has not
been published is neither cited nor displayed.

## Papers

### Robust optimization of strategic and tactical asset allocation

Sepp, A., Ossa, I. and Kastenholz, M. (2026). *Robust Optimization of Strategic and Tactical
Asset Allocation for Multi-Asset Portfolios*. The Journal of Portfolio Management, 52(4),
86–120. [Publisher page](https://www.pm-research.com/content/iijpormgmt/52/4/86);
[DOI 10.3905/jpm.2025.1.806](https://doi.org/10.3905/jpm.2025.1.806);
[author-shared copy](https://eprints.pm-research.com/17511/143431/index.html).

The ROSAA framework, of which optimalportfolios is the reference implementation. It contributes:

- the covariance estimated with a hierarchical-clustering group LASSO factor model, assembled as
  $\Sigma = \beta \Sigma_F \beta^{\top} + D$ (see [factor covariance with HCGL](factor_covariance_hcgl.md)
  and FactorLasso);
- strategic allocation by constrained risk budgeting (see [risk budgeting](risk_budgeting.md));
- tactical allocation as alpha over a tracking-error budget against the strategic benchmark (see
  [tactical allocation](alpha_over_tracking_error.md)).

The [ROSAA case study](app_rosaa_multi_asset_allocation.md) reports its study design and
results and runs the same configuration offline.

### Optimal allocation to cryptocurrencies

Sepp, A. (2023). *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*. Risk,
October 2023. [Risk](https://www.risk.net/cutting-edge/7957914/optimal-allocation-to-cryptocurrencies-in-diversified-portfolios);
[SSRN 4217841](https://ssrn.com/abstract=4217841).

The paper compares four allocation methods for a diversified portfolio with a
cryptocurrency: equal risk contributions, maximum diversification, maximum Sharpe ratio and
CARA utility under a Gaussian mixture fitted to returns. Each is an objective of the package (see
[choosing an objective](optimization_module_readme.md), [mean-variance objectives](mean_variance_objectives.md) and
[CARA utility under Gaussian mixtures](cara_gaussian_mixture.md)); the manuscript source is tracked in the
repository.

### Capital market assumptions from multi-asset tradable factors

Sepp, A., Hansen, E. and Kastenholz, M. (2026). *Capital Market Assumptions and Strategic Asset
Allocation Using Multi-Asset Tradable Factors*. Working paper,
[SSRN 6785958](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958).

The paper derives capital market assumptions from multi-asset tradable factors and uses them for
strategic asset allocation. On this site it is cited only in its public SSRN version.

## Reproducing the papers

The paper folders are repository-only research code: they are not installed by
`pip install optimalportfolios`, and each folder's README states its own commands and data
requirements. The table summarises what a public checkout can run.

| Paper | Folder | What a public checkout reproduces |
|---|---|---|
| ROSAA | [`robust_optimisation_jpm_2026`](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/robust_optimisation_jpm_2026) | A methodological example of the HCGL covariance and risk-budgeted strategic allocation. It downloads its ETF panel with `yfinance`, carries no frozen inputs and records no environment, so it is not an exact rebuild of the published exhibits. |
| Cryptocurrencies | [`crypto_allocation_risk_2023`](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/crypto_allocation_risk_2023) | The manuscript source, the analysis code, historical price files and offline replication tests, which CI runs on every push. The full update route needs licensed Bloomberg data, so the headline numbers are not promised to reproduce exactly. |
| Capital market assumptions | [`cma_data`](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/cma_data) | A manifest-verified snapshot of the configuration tables behind the capital market assumptions, with tests that CI runs on every push. Licensed index, factor-history and provider panels are omitted. |

Frozen package versions are quoted only from a committed manifest; where a folder records no
environment, none is inferred.

## How to cite

Cite a paper for its method, and the software records for the packages an implementation uses:
[optimalportfolios](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff),
[qis](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff) and
[FactorLasso](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff). FactorLasso's own
papers are listed on its
[research papers page](https://factorlasso.readthedocs.io/en/latest/scientific-replication.html).

## See also

- [Documentation home](index.md)
- [Conventions, notation and glossary](conventions.md)
- [Paper folders and their policy](https://github.com/ArturSepp/OptimalPortfolios/blob/main/papers/README.md)
