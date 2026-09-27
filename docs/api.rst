API reference
=============

Author: `Artur Sepp <https://github.com/ArturSepp>`_ / First recorded: `2026-08-09 <https://github.com/ArturSepp/OptimalPortfolios/commit/254505981ed43e0dbc12a19c98c034d351b8d059>`_.

Documentation for `OptimalPortfolios <https://github.com/ArturSepp/OptimalPortfolios>`_.
Software citation: `CITATION.cff
<https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff>`_.

.. meta::
   :description: Public OptimalPortfolios API: signatures and docstrings, guide entry points, solver layers, and QIS/FactorLasso ownership.

The public API consists of the non-underscore names importable from the
``optimalportfolios`` package root. Open a symbol in the reference below to read
its signature and docstring. The guides explain the assumptions, units and
timing needed to use those interfaces.

Find a guide
------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Your task
     - Guide
   * - Run an offline construction and backtest
     - :doc:`Quickstart <quickstart>`
   * - Build and score alpha signals
     - :doc:`Alpha signals <alphas_module_readme>`
   * - Prepare EWMA or factor covariance estimates
     - :doc:`Covariance estimators <covariance_estimators>`
   * - Choose an objective and understand solver outputs
     - :doc:`Optimization guide <optimization_module_readme>`
   * - Set limits and understand backend support
     - :doc:`Constraints <constraints>`
   * - Connect target weights to executed holdings
     - :doc:`Rolling backtests <rolling_backtests>`
   * - Build a benchmark-relative allocation
     - :doc:`Minimum tracking error <minimum_tracking_error>`

The :doc:`examples guide <examples_readme>` lists complete workflows and their
offline, network or local-data prerequisites. Ordinary source links are also
available in the `Markdown guide directory
<https://github.com/ArturSepp/OptimalPortfolios/tree/main/docs>`_.

Use the installed API
---------------------

Signatures in this site reflect the source and dependencies used to build it.
Inspect your installed package when checking a call locally:

.. code-block:: python

   import inspect
   import optimalportfolios as op

   print(inspect.signature(op.Constraints))
   print(inspect.signature(op.compute_rolling_optimal_weights))

The optimizer interfaces have different responsibilities:

* ``rolling_*`` functions produce dated target-weight tables. Holdings simulation
  and cash transaction costs are a separate step.
* ``wrapper_*`` functions solve one date and restore instrument labels. Return
  values differ by solver family; some include an ``OptimizationOutcome``.
* ``cvx_*`` and ``opt_*`` functions expose numerical solver interfaces. Check
  their individual input and return contracts before calling them directly.

The :doc:`optimization guide <optimization_module_readme>` explains dispatch,
return shapes, solver acceptance and fallback behavior. The
:doc:`constraints contract <constraints>` remains the reference for feasibility,
alignment and supported conditions.

Implementation ownership
------------------------

The `package-root imports
<https://github.com/ArturSepp/OptimalPortfolios/blob/main/src/optimalportfolios/__init__.py>`_
define the re-export surface. The reference below documents those objects;
it does not define a second public API.

OptimalPortfolios owns portfolio construction and rolling orchestration.
`QIS <https://github.com/ArturSepp/QuantInvestStrats>`_ supplies reusable analytics,
holdings simulation and reporting.
`FactorLasso <https://github.com/ArturSepp/FactorLasso>`_ supplies generic sparse
factor estimation and related containers; selected objects are re-exported here
for compatibility. The :doc:`software design guide <software_design>` maps these
responsibilities to their implementations.

When citing work that uses those packages, include the
`QIS software citation
<https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff>`_ and
`FactorLasso software citation
<https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff>`_ as appropriate,
alongside the OptimalPortfolios citation above.

Public reference
----------------

The reference below is generated when the site is built, from the package itself and
the page ownership recorded in the documentation inventory, so it cannot drift from the
exports. Each object appears under the page that explains it, in the order of the
sidebar. The names re-exported from FactorLasso and qis link to the documentation of the
package that defines them, and the configuration fields of the mapped dataclasses are
listed with their defaults and owning pages.

.. include:: _generated/api_reference.rst

.. automodule:: optimalportfolios
