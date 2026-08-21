Software design and boundaries
==============================

``optimalportfolios`` is organized around a dated portfolio decision rather than
around an isolated optimizer call.  At each decision date, point-in-time estimates,
current holdings, eligibility, dealing windows, and constraints determine a target
portfolio.  The target is then evaluated with an explicit implementation lag and
transaction costs.  This design serves multi-asset research in which the state
between successive decisions is part of the model.

Composition across the open-source stack
----------------------------------------

The package composes two narrower projects instead of copying their code:

* `factorlasso <https://github.com/ArturSepp/factorlasso>`_ owns generic sparse
  factor estimation, HCGL clustering, and factor-covariance data structures.
  ``optimalportfolios`` supplies the finance-specific return frequencies,
  estimation dates, annualisation, and conversion into portfolio inputs.
* `qis <https://github.com/ArturSepp/QuantInvestStrats>`_ owns time-series and
  portfolio analytics, the unit-based portfolio simulator, and reporting.
  ``optimalportfolios`` owns the construction decision and passes dated target
  weights, costs, and lag assumptions to those tested primitives.

Vendoring either dependency was considered and rejected.  A copied factor model or
backtest would create a second convention for the same concept and would let fixes
diverge across projects.  Composition keeps the package boundary reviewable: generic
estimation changes belong in ``factorlasso``; generic analytics and holdings simulation
belong in ``qis``; portfolio-construction state belongs here.

The decision pipeline can be summarized as:

.. code-block:: text

   prices and dated inputs
            |
            v
   point-in-time estimates  <--- factorlasso factor models
            |
            v
   constraints + optimizer ---> dated target weights
            |                         |
            +---- current holdings ---+
                                      |
                                      v
                         qis holdings, costs, and reports

Architectural boundaries
------------------------

Point-in-time estimation
~~~~~~~~~~~~~~~~~~~~~~~~

Covariance matrices and alpha inputs carry decision dates.  A rolling solver receives
the estimate available for that date rather than recomputing an in-sample statistic
inside a single-date optimizer.  The alternative--letting each solver estimate its own
inputs--was rejected because it couples estimation and objective choice, prevents reuse
of identical estimates across solvers, and makes look-ahead harder to detect.

Holdings as state
~~~~~~~~~~~~~~~~~

Previous target weights are not assumed to remain current weights.  Realised returns
drift the holdings before the next decision, so turnover limits and costs act on the
trade actually required.  A stateless sequence of optimizer calls was rejected because
it loses that transition and can measure turnover against a stale target.

Mixed-frequency scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~

Assets can have different return-estimation and rebalancing cadences.  Eligibility and
rebalancing indicators can therefore change by date and freeze positions that cannot be
traded.  Converting every series to one fixed frequency was rejected because it either
discards timely liquid-market observations or invents observations for slower assets.

Shared constraints
~~~~~~~~~~~~~~~~~~

The ``Constraints`` data structure carries weight, exposure, group, tracking-error,
turnover, target-return, volatility, and dealing restrictions across the mathematical,
NaN-aware wrapper, and rolling layers.  Independent keyword sets for every solver were
rejected because filtering an incomplete universe or freezing an asset must transform
the same constraint semantics before any objective is solved.

Optimization backends
---------------------

CVXPY expresses the convex quadratic and conic problems and makes the constraint system
readable across objectives.  ``quadprog`` provides a dedicated path where a quadratic
program has a direct formulation.  Maintaining more backends was
considered and rejected: another backend would multiply constraint translations,
diagnostics, and numerical compatibility work without changing the package's research
scope.  This choice favors a small, testable backend surface over the broadest possible
catalogue of optimization problem classes.  It makes no comparative runtime guarantee.

Core installation and integrations
----------------------------------

The core installation contains the numerical construction pipeline.  Network data
clients and optional reporting renderers stay in the ``data`` and ``reports`` extras.
Making them mandatory was rejected because an offline portfolio construction or an
installed-wheel test does not need live downloads, databases, or document rendering.
Optional imports are kept away from core module import paths so that the numerical API
remains usable without those integrations.

Why this is a separate package
------------------------------

PyPortfolioOpt, Riskfolio-Lib, and skfolio are established open-source portfolio
projects.  The versioned :doc:`package_comparison` documents their current public
interfaces and avoids a universal quality ranking.  Their primary abstractions differ:
PyPortfolioOpt presents compact classical allocation workflows, Riskfolio-Lib develops
a broad catalogue of risk measures and portfolio families, and skfolio integrates
portfolio estimators with scikit-learn model selection.

Contributing one additional objective to one of those projects was considered.  It would
not provide the top-level state transition required here: dated estimates, drifted
holdings, mixed dealing schedules, NaN-aware eligibility, constrained targets,
implementation lag, costs, and reporting form one research workflow.  Keeping that
workflow in a separate package also avoids asking another project to adopt the
``qis``/``factorlasso`` stack boundaries.  Where an adjacent project already owns a
generic capability, ``optimalportfolios`` integrates it rather than reimplementing it.

Intentional exclusions
----------------------

The package does not aim to provide every portfolio model.  Non-quadratic risk measures
such as CVaR, MAD, and drawdown constraints are better served by Riskfolio-Lib or
skfolio.  Generic cross-validation and hyperparameter-search abstractions remain a
skfolio/scikit-learn design concern.  Data licensing, broker execution, order management,
and proprietary production orchestration are outside the public package.  Generic
performance analytics and factor estimation remain in ``qis`` and ``factorlasso``.

These exclusions keep the public contract centered on portfolio construction and its
point-in-time state, while allowing the surrounding components to evolve independently.
