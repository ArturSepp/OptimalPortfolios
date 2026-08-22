optimalportfolios: production portfolio construction and rolling backtesting
============================================================================

``optimalportfolios`` provides production multi-asset portfolio construction
and rolling backtesting in Python. It connects point-in-time covariance and
alpha estimation with constrained optimisation, rebalancing, transaction
costs, and performance reporting.

The package is the reference implementation of the ROSAA framework published
in *The Journal of Portfolio Management* (Sepp, Ossa and Kastenholz, 2026).

For a first end-to-end result, run the :doc:`quickstart`: it uses wheel-shipped
data to estimate rolling covariance, compute constrained weights, and backtest
them with transaction costs entirely offline.

Overview
--------

``optimalportfolios`` implements the full path from raw prices to a backtested
portfolio:

#. **Alpha signals** -- risk-adjusted and classic momentum, carry, low-beta,
   residual momentum, and reversal, with cross-sectional and within-cluster scoring.
#. **Covariance estimation** -- EWMA estimators and the HCGL sparse factor model
   supplied by `factorlasso <https://github.com/ArturSepp/factorlasso>`_.
#. **Constrained optimisation** -- risk budgeting, maximum diversification,
   maximum Sharpe, alpha over tracking error, and minimum variance at a target
   return, expressed in cvxpy with a shared ``Constraints`` object.
#. **Rolling backtest and reporting** -- drift-aware rebalancing with transaction
   costs and factsheets through `qis
   <https://github.com/ArturSepp/QuantInvestStrats>`_.

Papers
------

* Sepp, A. (2023), *Optimal Allocation to Cryptocurrencies in Diversified
  Portfolios*, Risk Magazine -- `SSRN 4217841
  <https://ssrn.com/abstract=4217841>`_.
* Sepp, A., Ossa, I. and Kastenholz, M. (2026), *Robust Optimization of
  Strategic and Tactical Asset Allocation for Multi-Asset Portfolios*,
  `The Journal of Portfolio Management, 52(4), 86--120
  <https://www.pm-research.com/content/iijpormgmt/52/4/86>`_.
* Sepp, A., Hansen, E. and Kastenholz, M. (2026), *Capital Market Assumptions
  and Strategic Asset Allocation Using Multi-Asset Tradable Factors* --
  `SSRN 6785958
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6785958>`_.

Project links
-------------

* `PyPI <https://pypi.org/project/optimalportfolios/>`_
* `Source repository <https://github.com/ArturSepp/OptimalPortfolios>`_
* `Issue tracker <https://github.com/ArturSepp/OptimalPortfolios/issues>`_
* `Governance, maintenance, and support
  <https://github.com/ArturSepp/OptimalPortfolios/blob/main/GOVERNANCE.md>`_
* `Changelog
  <https://github.com/ArturSepp/OptimalPortfolios/blob/main/CHANGELOG.md>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   examples_readme
   rolling_backtests
   alphas_module_readme
   optimization_module_readme
   minimum_tracking_error
   risk_budgeting
   turnover_and_transaction_costs
   overlay_tail_floor
   mixed_frequency_data
   incomplete_histories
   covariance_estimators
   software_design
   package_comparison
   api
