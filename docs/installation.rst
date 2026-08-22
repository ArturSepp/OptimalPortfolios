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
   under the repository-root ``examples/`` tree need; the test suite does not.

``reports``
   The pybloqs backend for HTML and PDF report rendering.

``all``
   Both runtime integrations: ``data`` and ``reports``.

``docs``
   The Sphinx toolchain used to build this documentation.

Install an extra by placing its name in square brackets:

.. code-block:: console

   pip install "optimalportfolios[all]"
   pip install -e ".[docs]"

Contributor test dependencies use the PEP 735 ``test`` dependency group rather
than a package extra. The suite collects the same tests with or without runtime
extras:

.. code-block:: console

   uv sync --locked --group test
   uv run --no-sync pytest

To run examples as well, add ``--extra data`` to the sync command.

The runtime integration extras, ``data`` and ``reports``, correspond to features
that import their dependencies. There is no ``jupyter`` extra: nothing here
imports jupyter, notebook or jupyterlab. The repository-only Colab quickstart
uses Google's hosted runtime; install notebook tooling separately for local
notebooks. The ``docs`` extra remains a contributor toolchain. There
is no ``clustering`` extra either — the ``mcf``
risk-lineage matcher once needed NetworkX, but it now uses a SciPy bipartite
assignment and runs on a core install, and the cluster-lineage analytics
themselves live in ``factorlasso``, a core dependency.

The lint tools are not an extra at all. ``ruff`` and ``interrogate`` gate the
repository rather than support the suite, so they live in the ``lint``
dependency-group, which never ships to a user and is not synced by default.
