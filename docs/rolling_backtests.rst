Rolling portfolio backtests
===========================

Use a rolling backtest when portfolio inputs must be re-estimated and the
portfolio re-optimised through time.  A rolling workflow has three distinct
layers:

#. a single-date solver, such as ``wrapper_risk_budgeting``;
#. its rolling wrapper, which solves at every covariance date and carries the
   prior portfolio forward; and
#. ``qis.backtest_model_portfolio``, which turns target weights into holdings,
   applies implementation lag and costs, and reports realised performance.

This separation is useful in production: validate one covariance matrix and
one solution first, then run the same contract through time.

Inputs and conventions
----------------------

``prices`` is a total-return price panel with a ``DatetimeIndex`` and one
column per asset.  ``covar_dict`` maps decision dates to covariance matrices;
the matrix index and columns are asset names.  ``Constraints`` uses
dimensionless portfolio weights.  Each rolling solver returns target weights
on the covariance dates.

The covariance estimator defines the return convention, sampling frequency,
and annualisation.  ``EwmaCovarEstimator`` computes log returns at
``returns_freq``, estimates EWMA covariance, and returns annualised matrices.
Its ``span`` is measured in observations at that return frequency.
``rebalancing_freq`` only selects the covariance/decision dates; it does not
change the return sampling frequency.

Minimal offline example
-----------------------

The example uses monthly total-return prices, a six-observation EWMA span,
quarter-end decisions, and next-observation implementation.  The explicit lag
means a weight decided at *t* is not traded at the same observation.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import qis
   import optimalportfolios as opt

   dates = pd.date_range("2020-01-31", periods=24, freq="ME")
   monthly_returns = np.array([
       [0.010, 0.004, 0.002], [0.015, -0.003, 0.003],
       [-0.008, 0.006, 0.002], [0.012, 0.001, -0.001],
   ] * 6)
   prices = pd.DataFrame(
       100.0 * np.cumprod(1.0 + monthly_returns, axis=0),
       index=dates,
       columns=["Equity", "Bonds", "Diversifier"],
   )
   estimator = opt.EwmaCovarEstimator(
       returns_freq="ME", span=6, rebalancing_freq="QE"
   )
   covar_dict = estimator.fit_rolling_covars(
       prices=prices,
       time_period=qis.TimePeriod("31Dec2020", "30Sep2021"),
   )
   constraints = opt.Constraints(
       is_long_only=True,
       max_weights=pd.Series(0.80, index=prices.columns),
   )
   weights = opt.compute_rolling_optimal_weights(
       prices=prices,
       constraints=constraints,
       covar_dict=covar_dict,
       portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
   )
   portfolio = opt.backtest_rolling_optimal_portfolio(
       prices=prices,
       constraints=constraints,
       covar_dict=covar_dict,
       portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
       rebalancing_costs=0.0003,  # 3 bp of traded notional
       weight_implementation_lag=1,
       ticker="Minimum variance",
   )

``weights`` has one row per decision date and one column per asset.
``portfolio`` is a ``qis.PortfolioData`` containing NAV, realised holdings,
turnover, and costs.  Between decisions, ``qis`` holds units rather than
constant weights, so realised weights drift with asset prices.

Point-in-time and drift rules
-----------------------------

Rolling optimisers pass the previous target through
``apply_drift_to_weights_0`` before the next solve.  For an asset return
:math:`r_i`, prior weight :math:`w_i`, and portfolio NAV growth
:math:`1+\sum_j w_jr_j`, the new pre-trade weight is

.. math::

   w_{i,\mathrm{drift}} = \frac{w_i(1+r_i)}{1+\sum_j w_jr_j}.

Turnover constraints therefore compare the new target with the portfolio that
would actually arrive at the decision date, not with the stale prior target.
On the first decision, or when price anchors are unavailable, the helper
leaves the supplied prior weights unchanged.

No-look-ahead is the caller's responsibility across the full input path.
Every covariance and alpha keyed by *t* must use information available no
later than *t*.  In a backtest, set ``weight_implementation_lag=1`` when a
decision observed at *t* should enter on the next price observation.  Do not
use a full-sample demeaned return series as a rolling estimator input.

Failure modes and missing data
------------------------------

Assets with a NaN covariance diagonal can be excluded by the solver and
returned at zero weight; other non-finite entries are invalid.  An entirely
invalid eligible universe produces the
solver's documented fallback rather than a meaningful portfolio.  A missing
price needed only for drift is treated as a flat return; that defensive rule
does not make a bad price history valid for performance measurement.  Check
solver diagnostics, weight sums, constraint residuals, and the first tradable
date before interpreting a backtest.

See also
--------

* :doc:`minimum_tracking_error` and :doc:`risk_budgeting`
* :doc:`api` for ``compute_rolling_optimal_weights``,
  ``backtest_rolling_optimal_portfolio``, and ``apply_drift_to_weights_0``
* `Offline multi-asset rolling example
  <https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/backtests/multiasset_saa.py>`_
