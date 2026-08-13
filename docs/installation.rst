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
   The free-data loader backed by yfinance. This is what the example scripts
   under ``optimalportfolios/examples/`` need; the test suite does not.

``reports``
   The pybloqs backend for HTML and PDF report rendering.

``all``
   Both runtime integrations: ``data`` and ``reports``.

``dev``
   Pytest and pytest-cov — the test suite and nothing else. The suite collects
   the same tests with or without the runtime extras, so ``dev`` deliberately
   does not pull them in. To run the examples as well, install ``[dev,data]``.

``docs``
   The Sphinx toolchain used to build this documentation.

Install an extra by placing its name in square brackets:

.. code-block:: console

   pip install "optimalportfolios[all]"
   pip install -e ".[dev]"
   pip install -e ".[docs]"

Every extra names a package this project imports, which is why the list is
short. There is no ``jupyter`` extra: nothing here imports jupyter, notebook or
jupyterlab, so install a notebook stack alongside the package rather than
through it. There is no ``clustering`` extra either — the ``mcf`` risk-lineage
matcher once needed NetworkX, but it now uses a SciPy bipartite assignment and
runs on a core install, and the cluster-lineage analytics themselves live in
``factorlasso``, a core dependency.

The lint tools are not an extra at all. ``ruff`` and ``interrogate`` gate the
repository rather than support the suite, so they live in the ``lint``
dependency-group, which never ships to a user and is not synced by default.
