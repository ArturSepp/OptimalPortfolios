---
icon: material/chart-timeline-variant
hide:
  - toc
---

# OptimalPortfolios

**Multi-asset portfolio construction and backtesting in Python** — from alpha signals and
covariance estimation through constrained optimisation to factsheet reporting, in a single
pipeline built for real-world data.

[![PyPI](https://img.shields.io/pypi/v/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/)
[![Python](https://img.shields.io/pypi/pyversions/optimalportfolios?style=flat-square)](https://pypi.org/project/optimalportfolios/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ArturSepp/OptimalPortfolios/blob/main/LICENSE.txt)
[![CI](https://github.com/ArturSepp/OptimalPortfolios/actions/workflows/ci.yml/badge.svg)](https://github.com/ArturSepp/OptimalPortfolios/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/optimalportfolios)](https://pepy.tech/project/optimalportfolios)

---

**Quick links:** [Repository](https://github.com/ArturSepp/OptimalPortfolios) ·
[PyPI](https://pypi.org/project/optimalportfolios/) ·
[Issues](https://github.com/ArturSepp/OptimalPortfolios/issues) ·
[Changelog](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CHANGELOG.md)

---

## Overview

OptimalPortfolios implements the full path from raw prices to a backtested portfolio:

1. **Alpha signals** — momentum, carry, low-beta, residual momentum and reversal, with
   cross-sectional and within-cluster scoring.
2. **Covariance estimation** — EWMA estimators and the HCGL sparse factor model supplied
   by [`factorlasso`](https://github.com/ArturSepp/factorlasso).
3. **Constrained optimisation** — risk budgeting, maximum diversification, maximum Sharpe,
   alpha over tracking error, minimum variance at a target return, and others, expressed
   in `cvxpy` with a shared [`Constraints`][optimalportfolios.optimization.Constraints] object.
4. **Rolling backtest and reporting** — drift-aware rebalancing with transaction costs,
   and factsheets through [`qis`](https://github.com/ArturSepp/QuantInvestStrats).

The package is the reference implementation of the ROSAA framework published in
*The Journal of Portfolio Management* (Sepp, Ossa and Kastenholz, 2026).

## Papers

- Sepp, A. (2023), *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*,
  Risk Magazine — [SSRN 4217841](https://ssrn.com/abstract=4217841)
- Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of Strategic and
  Tactical Asset Allocation for Multi-Asset Portfolios*,
  [The Journal of Portfolio Management, 52(4), 86–120](https://www.pm-research.com/content/iijpormgmt/52/4/86)
- Sepp, A., Hansen, E. and Kastenholz, M. (2026), *Capital Market Assumptions and
  Strategic Asset Allocation Using Multi-Asset Tradable Factors* —
  [SSRN 6785958](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958)

## Where to go next

- [Installation](installation.md) — the core package and its optional extras.
- [Quickstart](quickstart.md) — a rolling backtest that runs offline, on committed data.
- [API Reference](api.md) — every public export, grouped by subsystem.
