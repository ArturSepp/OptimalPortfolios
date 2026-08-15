Quickstart
==========

The authoritative `production quickstart
<https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/getting_started/production_quickstart.py>`_
uses the monthly multi-asset fixture shipped in the wheel, so it needs no
network access, credentials, or repository-local data. After a plain install,
run it from a checkout or copy the script anywhere::

   pip install optimalportfolios
   python examples/getting_started/production_quickstart.py

For a zero-setup trial, `open the same workflow in Colab
<https://colab.research.google.com/github/ArturSepp/OptimalPortfolios/blob/main/examples/getting_started/production_quickstart.ipynb>`_.
The thin notebook installs the latest released package from PyPI and prints
its version before running the cell mechanically checked against the script
below. It carries no saved outputs and adds no Jupyter dependency to the
package.

It selects six assets over a fixed 2010--2022 price window. The first five
years warm up a 24-month EWMA covariance estimator; quarterly portfolio
decisions run from March 2015 through September 2022. The portfolio minimizes
variance subject to long-only and 35% per-asset limits. Each decision is
implemented at the next monthly observation, and the backtest charges 10 basis
points per unit traded.

.. literalinclude:: ../examples/getting_started/production_quickstart.py
   :language: python
   :linenos:

The estimator converts the monthly prices to log returns, estimates an
annualised covariance matrix using only information available at each
rebalance, and never fits on a later holding period. The script prints its
input dates, rolling-weight dimensions, final weights, final NAV, and measured
runtime, and writes no generated output.

What to change first
--------------------

* **The objective.** Swap ``PortfolioObjective.MIN_VARIANCE`` for any
  other ``PortfolioObjective`` member; the rest of the call is unchanged.
* **The constraints.** ``Constraints`` carries long-only, leverage, group
  bounds, turnover, and tracking-error limits in one object shared by every
  solver.
* **The covariance estimator.** Replace ``EwmaCovarEstimator`` with
  ``FactorCovarEstimator`` to use the HCGL sparse factor model instead of an
  EWMA covariance matrix.
* **The rebalance cadence.** Change ``rebalancing_freq="QE"`` to another
  pandas-compatible frequency such as ``"YE"``; retain an implementation lag
  appropriate for the decision and price timestamps.
