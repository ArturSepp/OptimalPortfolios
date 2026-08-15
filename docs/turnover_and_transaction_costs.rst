Turnover and transaction costs
==============================

Portfolio turnover enters the pipeline twice.  The optimizer controls a
*target trade* relative to the pre-trade portfolio.  The backtester later
measures the *realised trade* in units and deducts transaction costs.  Keeping
the two roles separate prevents a turnover constraint from being mistaken for
a performance cost model.

Target turnover in optimisation
-------------------------------

For current weights :math:`w_0` and proposed weights :math:`w`, the portfolio
turnover constraint is

.. math::

   \lVert w-w_0\rVert_1 \leq \tau.

Both weights and ``turnover_constraint`` are dimensionless fractions of
portfolio NAV.  For example, ``0.10`` permits ten percentage points of summed
absolute weight change: a five-point sale plus a five-point purchase consumes
the full limit.  If
``turnover_costs`` is supplied, the constrained expression becomes the
weighted L1 norm ``sum(turnover_costs * abs(w - weights_0))``; its limit must
use units consistent with those coefficients.

In a rolling optimizer, ``weights_0`` is normally the prior target drifted by
realised asset returns to the new decision date.  The first decision has no
prior portfolio, so a turnover constraint cannot bind there.  A solver can
also use a turnover utility penalty or group turnover limits through
``Constraints``; these change the target, but do not deduct cash from the
backtest.

.. code-block:: python

   import pandas as pd
   import optimalportfolios as opt

   current = pd.Series({"A": 0.60, "B": 0.40})
   constraints = opt.Constraints(
       is_long_only=True,
       weights_0=current,
       turnover_constraint=0.10,
       turnover_costs=pd.Series({"A": 1.0, "B": 1.0}),
   )

Pass the current or drifted portfolio explicitly to a single-date wrapper.
Rolling wrappers perform that drift step internally.

Realised turnover and costs in the backtest
-------------------------------------------

``qis.backtest_model_portfolio`` converts target weights to units at each
trade and holds those units between trades.  Realised turnover therefore
depends on price drift and on the implemented target, not only on consecutive
target-weight rows.  Proportional cost is deducted from cash as

.. math::

   \mathrm{cost}_{i,t}=c_{i,t}p_{i,t}|\Delta u_{i,t}|.

``rebalancing_costs`` is fractional cost per unit of traded notional:
``0.0010`` is 10 bp.  A scalar applies to all instruments and dates, a Series
is indexed by ticker, and a date-by-ticker DataFrame supplies a time-varying
schedule.  Costs are read on the actual trade date, including any
``weight_implementation_lag``.

.. code-block:: python

   import pandas as pd
   import qis

   prices = pd.DataFrame(
       {"A": [100.0, 102.0, 101.0], "B": [100.0, 99.0, 101.0]},
       index=pd.date_range("2024-01-02", periods=3, freq="B"),
   )
   targets = pd.DataFrame(
       {"A": [0.60, 0.50], "B": [0.40, 0.50]},
       index=prices.index[:2],
   )
   portfolio = qis.backtest_model_portfolio(
       prices=prices,
       weights=targets,
       rebalancing_costs=0.0010,
       weight_implementation_lag=1,
       ticker="Cost-aware backtest",
   )

Interpretation and failure modes
--------------------------------

Report both target-weight changes and realised turnover when diagnosing a
strategy.  Large differences usually reflect drift, a delayed trade, an
unpriced instrument, or a target that was not implemented.  A price missing
on a rebalancing date prevents that instrument from trading; its intended
weight stays in cash rather than being redistributed.  A date-indexed cost
Series is rejected as ambiguous, and a cost DataFrame must contain every price
column.  Always label cost inputs in fractional units or basis points.

See also
--------

* :doc:`rolling_backtests` for drift and implementation timing
* :doc:`incomplete_histories` for unpriced and frozen positions
* :doc:`api` for ``Constraints`` and ``apply_drift_to_weights_0``
* `qis portfolio backtester
  <https://github.com/ArturSepp/QuantInvestStrats/blob/main/qis/portfolio/backtester.py>`_
