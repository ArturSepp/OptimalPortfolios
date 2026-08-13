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
   The free-data loader backed by yfinance.

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

An additional ``reports`` extra provides the pybloqs report backend. The ``all``
extra installs the ``data`` and ``reports`` runtime integrations. There is no
``jupyter`` extra: nothing in this package imports jupyter, notebook or
jupyterlab, so install a notebook stack alongside it rather than through it.
Documentation contributors can install the Sphinx toolchain with:

.. code-block:: console

   pip install -e ".[docs]"
