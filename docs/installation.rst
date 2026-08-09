Installation
============

OptimalPortfolios supports Python 3.10 and later. Install the core package from
PyPI:

.. code-block:: console

   pip install optimalportfolios

The core install includes covariance estimation, portfolio optimisation, and
backtesting. Optional extras add integrations that are not needed by every
user:

``data``
   Free-data loaders backed by yfinance and pandas-datareader.

``clustering``
   The default minimum-cost-flow matcher used to keep risk-cluster labels
   stable through time. Install it when using the ``mcf`` cluster matcher; the
   ``hungarian`` matcher remains available in the core install.

``dev``
   The test, coverage, lint, and docstring-quality tools used by contributors.
   It also includes the ``data`` and ``clustering`` extras so the complete test
   suite can run.

Install an extra by placing its name in square brackets:

.. code-block:: console

   pip install "optimalportfolios[clustering]"
   pip install -e ".[dev]"

Additional ``reports``, ``visualization``, and ``jupyter`` extras provide the
pybloqs report backend, Plotly charts, and notebook tooling respectively. The
``all`` extra installs all runtime integrations. Documentation contributors can
install the Sphinx toolchain with:

.. code-block:: console

   pip install -e ".[docs]"
