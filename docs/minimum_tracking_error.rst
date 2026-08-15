Minimum tracking error
======================

Use minimum tracking error when a benchmark portfolio is the natural starting
point but mandate constraints make the benchmark infeasible or undesirable.
The solver finds the feasible portfolio closest to the benchmark in covariance
risk:

.. math::

   \min_w\;(w-w_b)^\mathsf{T}\Sigma(w-w_b).

This is not a Euclidean projection: deviations in volatile or highly
correlated assets receive different risk weights.

Inputs, units, and alignment
----------------------------

``pd_covar`` is a square, symmetric covariance DataFrame whose index and
columns match in the same order.  The solver does not resample or annualise it.
If the covariance is annualised, the objective is annual variance and its
square root is annualised tracking-error volatility.  Portfolio and benchmark
weights are dimensionless.

``benchmark_weights`` must be finite and complete after alignment to the price
columns.  In a rolling solve, a Series is a static benchmark; a DataFrame is
forward-filled to covariance dates, so it must contain an observation no later
than the first decision.  Exposure, asset bounds, group bounds, and turnover
are supplied through ``Constraints``; the optimizer does not assume that an
arbitrary benchmark is itself feasible.

Minimal offline example
-----------------------

Here the covariance is already annualised.  The 35% cap on ``A`` forces a
benchmark-relative trade.

.. code-block:: python

   import pandas as pd
   import optimalportfolios as opt

   assets = ["A", "B", "C"]
   covar = pd.DataFrame(
       [[0.040, 0.006, 0.002],
        [0.006, 0.022, 0.003],
        [0.002, 0.003, 0.012]],
       index=assets,
       columns=assets,
   )
   benchmark = pd.Series([0.50, 0.30, 0.20], index=assets)
   constraints = opt.Constraints(
       is_long_only=True,
       min_weights=pd.Series(0.0, index=assets),
       max_weights=pd.Series([0.35, 0.80, 0.80], index=assets),
   )
   weights, outcome = opt.wrapper_minimise_tracking_error(
       pd_covar=covar,
       benchmark_weights=benchmark,
       constraints=constraints,
       weights_0=benchmark,
   )
   active = weights - benchmark
   tracking_error = float((active @ covar @ active) ** 0.5)
   print(outcome.status, weights, tracking_error)

The returned Series is the accepted portfolio in the original asset order;
``outcome`` carries solver status and validation diagnostics.  In this example
``A`` reaches its cap and the displaced weight is allocated to lower-risk
alternatives while the weights remain fully invested.  ``tracking_error`` is
in annual volatility units because the input covariance is annualised.

Single-date versus rolling use
------------------------------

``wrapper_minimise_tracking_error`` solves and validates one date and returns
``(weights, outcome)``.  Use it to inspect feasibility and alignment.
``rolling_minimise_tracking_error`` accepts a covariance dictionary and a
static or time-varying benchmark, drifts the prior portfolio between dates,
and returns a weight DataFrame on covariance dates.  Backtest those targets
with ``qis.backtest_model_portfolio`` and an explicit implementation lag.

Constraints and failure modes
-----------------------------

An asset with a NaN covariance diagonal or an explicit zero inclusion
indicator is removed and returned at zero weight.  The remaining covariance
must be finite and symmetric, so infinities are rejected.  Missing benchmark values, a benchmark
DataFrame that starts after the first decision, inconsistent covariance
labels, or infeasible constraints fail early or produce a rejected diagnostic
outcome.  A turnover cap only binds when a valid ``weights_0`` exists; there is
no previous portfolio on the first rolling date.

See also
--------

* :doc:`rolling_backtests`
* :doc:`api` for ``wrapper_minimise_tracking_error`` and
  ``rolling_minimise_tracking_error``
* `Canonical one-step and rolling example
  <https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/solvers/minimum_tracking_error.py>`_
  (network-data example)
