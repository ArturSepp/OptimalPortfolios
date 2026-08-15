Choosing a portfolio optimization library
==========================================

``optimalportfolios``, PyPortfolioOpt, Riskfolio-Lib, and skfolio overlap in
portfolio construction, but they optimize for different workflows.  Choose by
the path from inputs to decisions and evaluation, not by a universal package
ranking.

Version snapshot
----------------

This comparison was checked on **15 August 2026** against the current stable
releases on PyPI.

.. list-table::
   :header-rows: 1
   :widths: 28 20 52

   * - Package
     - Stable version
     - Release source
   * - ``optimalportfolios``
     - 6.18.0
     - `PyPI release record
       <https://pypi.org/project/optimalportfolios/6.18.0/>`_
   * - PyPortfolioOpt
     - 1.6.0
     - `PyPI release record
       <https://pypi.org/project/pyportfolioopt/1.6.0/>`_
   * - Riskfolio-Lib
     - 7.3.0
     - `PyPI release record
       <https://pypi.org/project/riskfolio-lib/7.3.0/>`_
   * - skfolio
     - 0.20.2
     - `PyPI release record <https://pypi.org/project/skfolio/0.20.2/>`_

Capability matrix
-----------------

``Not assessed`` means the reviewed official documentation did not establish
an integrated capability for this comparison.  It does not mean that a user
cannot assemble the workflow around the package.

.. list-table::
   :header-rows: 1
   :widths: 15 22 21 21 21

   * - Capability
     - ``optimalportfolios`` 6.18.0
     - PyPortfolioOpt 1.6.0
     - Riskfolio-Lib 7.3.0
     - skfolio 0.20.2
   * - Single-date optimization
     - Public wrappers solve and validate individual dates; see
       :doc:`minimum_tracking_error` and :doc:`risk_budgeting`.
     - Efficient-frontier objects support objectives and constraints in the
       `mean-variance guide
       <https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html>`_.
     - ``Portfolio`` covers mean-risk, risk parity, factor, and related convex
       models in the `portfolio API
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html>`_.
     - Estimators expose ``fit`` and ``predict`` through the
       `optimization guide <https://skfolio.org/user_guide/optimization.html>`_.
   * - Rolling point-in-time evaluation
     - Integrated covariance-date solvers, drifted pre-trade weights, lagged
       implementation, costs, and ``qis`` output; see :doc:`rolling_backtests`.
     - **Not assessed as an integrated engine.** The reviewed
       `user guide <https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html>`_
       documents optimizer construction and reusable inputs.
     - Official examples connect optimized weights to Backtrader or vectorbt;
       see `Backtesting
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/examples.html#backtesting>`_.
     - ``WalkForward`` and ``cross_val_predict`` create out-of-sample
       ``MultiPeriodPortfolio`` results; see
       `model selection <https://skfolio.org/user_guide/model_selection.html>`_.
   * - Expected-return and risk estimation
     - Alpha signals plus annualised EWMA and sparse factor/HCGL covariance;
       see :doc:`covariance_estimators` and :doc:`api`.
     - Historical mean, EMA, and CAPM returns plus sample, semi-, EWMA, and
       shrinkage covariance; see `expected returns
       <https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html>`_
       and `risk models
       <https://pyportfolioopt.readthedocs.io/en/latest/RiskModels.html>`_.
     - Historical, Black-Litterman, and factor-model inputs are documented in
       the `Portfolio model parameterization
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html>`_.
     - Composable expected-return, covariance, prior, and factor estimators;
       see `expected returns <https://skfolio.org/user_guide/expected_returns.html>`_,
       `covariance <https://skfolio.org/user_guide/covariance.html>`_, and
       `prior models <https://skfolio.org/user_guide/prior.html>`_.
   * - Risk-measure breadth
     - A focused objective set including variance, diversification, Sharpe,
       and risk budgeting is exposed through :doc:`api`.
     - Mean-variance plus efficient semivariance, CVaR, and CDaR optimizers;
       see `general efficient frontier
       <https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html>`_.
     - The convex portfolio documentation lists 24 risk measures, including
       tail and drawdown families; see the `portfolio API
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html>`_.
     - ``MeanRisk`` and ``RiskBudgeting`` accept variance, downside, tail, and
       drawdown measures; see the `optimization guide
       <https://skfolio.org/user_guide/optimization.html>`_.
   * - Black-Litterman
     - **Not assessed.** It is not part of the documented workflow summarized
       in :doc:`api`.
     - Dedicated model for priors, views, confidence, posterior returns, and
       covariance; see `Black-Litterman allocation
       <https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html>`_.
     - Historical, factor, Bayesian, and augmented Black-Litterman examples
       are indexed in `official examples
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/examples.html#black-litterman-models>`_.
     - A scikit-learn-compatible ``BlackLitterman`` prior estimator is covered
       by the `prior-model guide <https://skfolio.org/user_guide/prior.html>`_.
   * - Tracking error
     - Single-date and rolling minimum covariance tracking error with aligned
       static or time-varying benchmarks; see :doc:`minimum_tracking_error`.
     - Ex-ante and ex-post tracking-error objective functions are documented
       in the `mean-variance guide
       <https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#module-pypfopt.objective_functions>`_.
     - Benchmark-relative tracking-error constraints use ``allowTE`` and
       ``TE`` in the `Portfolio API
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html>`_.
     - Return-based, weight-based, and objective-based approaches are covered
       in `Tracking Error Optimization
       <https://skfolio.org/user_guide/optimization.html#tracking-error-optimization>`_.
   * - Turnover and transaction costs
     - Target turnover uses drifted pre-trade weights; realised costs are
       deducted on traded notional by ``qis``.  See
       :doc:`turnover_and_transaction_costs`.
     - A simple proportional ``transaction_cost`` objective accepts previous
       weights; see `objective functions
       <https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#module-pypfopt.objective_functions>`_.
     - ``allowTO`` and ``turnover`` constrain target deviations; backtesting is
       shown through external engines in `official examples
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/examples.html#backtesting>`_.
     - ``MeanRisk`` supports linear transaction costs, previous weights, and a
       maximum-turnover constraint; see the `MeanRisk API
       <https://skfolio.org/generated/skfolio.optimization.MeanRisk.html>`_.
   * - Mixed-frequency data
     - Per-asset signal/return cadences, cadence-specific spans, separate
       factor cadence, and independent rebalance cadence; see
       :doc:`mixed_frequency_data`.
     - **Not assessed.** Reviewed estimators expose one ``frequency`` argument;
       see `expected returns
       <https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html>`_.
     - **Not assessed.** The reviewed ``Portfolio`` interface consumes one
       returns matrix; see the `Portfolio API
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html>`_.
     - **Not assessed.** Reviewed preprocessing produces one aligned returns
       matrix; see `prices_to_returns
       <https://skfolio.org/generated/skfolio.preprocessing.prices_to_returns.html>`_.
   * - Incomplete histories
     - Late starters, interior gaps, eligibility, frozen positions, and
       unpriced trades have distinct documented behavior; see
       :doc:`incomplete_histories`.
     - **Not assessed.** No incomplete-history workflow claim is made from the
       reviewed `expected-return utilities
       <https://pyportfolioopt.readthedocs.io/en/latest/ExpectedReturns.html>`_.
     - **Not assessed.** No incomplete-history workflow claim is made from the
       reviewed `Portfolio API
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html>`_.
     - ``prices_to_returns`` documents inception-NaN handling, row thresholds,
       and optional forward filling; frozen-position handling was not assessed.
       See the `preprocessing API
       <https://skfolio.org/generated/skfolio.preprocessing.prices_to_returns.html>`_.
   * - Model selection and cross-validation
     - Point-in-time roll-forward evaluation is documented; a generic
       hyperparameter cross-validation framework is **not assessed**.  See
       :doc:`rolling_backtests`.
     - **Not assessed.** No claim is made beyond the reviewed
       `user guide <https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html>`_.
     - **Not assessed.** No claim is made beyond the reviewed
       `examples index
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/examples.html>`_.
     - Native scikit-learn selection plus ``WalkForward``, purged combinatorial
       CV, randomized CV, and online evaluation; see
       `model selection <https://skfolio.org/user_guide/model_selection.html>`_.
   * - Reporting
     - Rolling results are ``qis.PortfolioData`` objects with NAV, realised
       holdings, turnover, and the ``qis`` reporting layer; see
       :doc:`rolling_backtests`.
     - ``portfolio_performance`` and plotting helpers cover optimizer output;
       see `mean-variance performance
       <https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#pypfopt.efficient_frontier.EfficientFrontier.portfolio_performance>`_
       and `plotting <https://pyportfolioopt.readthedocs.io/en/latest/Plotting.html>`_.
     - Jupyter and Excel reports are provided by the
       `reports module
       <https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/reports.html>`_.
     - ``Portfolio`` and ``MultiPeriodPortfolio`` expose summaries, measures,
       composition, and plots; see the
       `portfolio guide <https://skfolio.org/user_guide/portfolio.html>`_.
   * - Primary design emphasis
     - Multi-asset production roll-forward from point-in-time estimates to
       constrained targets, holdings, costs, and reports; see
       :doc:`rolling_backtests`.
     - Compact classical allocation workflows with modular estimates,
       optimizers, and post-processing; see the
       `official overview <https://pyportfolioopt.readthedocs.io/en/latest/>`_.
     - Broad optimization-model and risk-measure exploration, including
       hierarchical and factor portfolios; see the
       `official documentation <https://riskfolio-lib.readthedocs.io/en/latest/>`_.
     - Scikit-learn-compatible portfolio modelling, pipelines, and model
       selection; see the `official user guide
       <https://skfolio.org/user_guide/index.html>`_.

Choose by workflow
------------------

Choose ``optimalportfolios`` when the main problem is a production
roll-forward: mixed-cadence multi-asset estimates, constrained decisions at
each rebalance, drift-aware turnover, frozen or late-starting assets, explicit
implementation lag and costs, and reporting through ``qis``.

Choose PyPortfolioOpt when the workflow calls for a compact classical
allocation toolkit: expected-return and covariance estimators, efficient-frontier methods,
Black-Litterman, downside-risk optimizers, and weight post-processing are
presented through a direct object-oriented interface.

Choose Riskfolio-Lib when research breadth across convex risk measures and
portfolio families is central.  Its documentation spans mean-risk, risk
parity, factor, Black-Litterman, hierarchical, network-constrained, and report
generation workflows.

Choose skfolio when portfolio estimators must participate in scikit-learn
pipelines, hyperparameter selection, walk-forward or purged cross-validation,
and multi-period out-of-sample comparison.

The specialization of ``optimalportfolios`` is the state transition between
successive decisions: estimates are dated, prior targets drift into current
holdings, constraints can reflect eligibility or dealing windows, and the
implemented portfolio incurs costs on actual trades.  PyPortfolioOpt and
Riskfolio-Lib intentionally expose broader catalogues in parts of classical
allocation and risk measures, while skfolio intentionally develops the
scikit-learn model-selection interface more deeply.  Those are different
design centers rather than a quality ranking.

How this comparison was made
----------------------------

Versions came from the official PyPI JSON records on 15 August 2026.
Capabilities were checked manually against the official documentation linked
in each matrix cell.  Competitor packages were not installed or executed, and
no speed, numerical-quality, dependency-size, popularity, or performance
benchmark was attempted.

Documentation can lag a release.  In particular, the PyPortfolioOpt ``latest``
documentation displayed a 1.5.4 page title during this review while PyPI
reported 1.6.0; the comparison therefore limits its claims to capabilities visible in the
linked public documentation and does not infer undocumented 1.6.0 changes.
``Not assessed`` cells are deliberately unresolved rather than negative
feature claims.  Recheck versions and links before using this page for a later
procurement or architecture decision.
