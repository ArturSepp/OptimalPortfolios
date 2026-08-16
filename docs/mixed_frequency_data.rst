Mixed-frequency data
====================

Use mixed-frequency inputs when some assets or signals are observed reliably
at different cadences.  Do not force quarterly private-asset observations
onto a monthly statistical horizon merely because liquid assets are monthly.
The pipeline separates three clocks:

* **estimation cadence**: the return observations used by a signal or risk
  model;
* **signal cadence**: when a new alpha value becomes available; and
* **rebalance cadence**: when the portfolio is allowed to choose a new target.

Changing one clock must not silently change the other two.

Per-asset return frequencies
----------------------------

``qis.compute_asset_returns_dict`` and
``UniverseData.get_asset_returns_dict`` accept either one pandas frequency
string or a Series mapping assets to frequencies.  The result is a dictionary
keyed by cadence.  Arithmetic returns are the ``UniverseData`` default; set
``is_log_returns=True`` explicitly when a model expects log returns.

.. code-block:: python

   import pandas as pd
   import qis
   import optimalportfolios as opt

   return_frequencies = pd.Series({
       "Global Equity": "ME",
       "Government Bonds": "ME",
       "Private Assets": "QE",
   })
   returns_by_frequency = qis.compute_asset_returns_dict(
       prices=prices,
       returns_freqs=return_frequencies,
       is_log_returns=True,
       drop_first=False,
       is_first_zero=True,
   )

The dictionary contains separate ``"ME"`` and ``"QE"`` DataFrames.  It is a
model input, not a merged claim that monthly and quarterly observations are
independent contemporaneous samples.

Signal horizons
---------------

Risk-adjusted momentum, classic momentum, low-beta, and residual-momentum signals accept a per-asset
``returns_freq`` Series.  Horizon parameters accept a mapping by cadence, so a
calendar-year momentum horizon can be represented by 12 monthly observations
and four quarterly observations:

.. code-block:: python

   scores, raw_signal = opt.compute_momentum_alpha(
       prices=prices,
       returns_freq=return_frequencies,
       long_span={"ME": 12, "QE": 4},
       vol_span={"ME": 13, "QE": 4},
   )

Classic momentum uses the same cadence mapping but interprets its parameters as
an exact observation count and hard skip rather than EWMA spans:

.. code-block:: python

   scores, raw_signal = opt.compute_classic_momentum_alpha(
       prices=prices,
       returns_freq=return_frequencies,
       lookback_periods={"ME": 12, "QE": 4},
       skip_periods={"ME": 1, "QE": 1},
   )

The per-cadence signals are computed separately, merged in the original asset
order, and forward-filled between their update dates.  Forward-filling makes a
stale signal explicit; it does not create a new observation.

Risk estimation and rebalancing
--------------------------------

``FactorCovarEstimator`` consumes the frequency-keyed asset-return dictionary
while risk-factor returns use the estimator's separate
``factor_returns_freq``.  ``factor_covar_span`` is measured at that factor
frequency.  ``rebalancing_freq`` controls when rolling decompositions and
portfolio decisions are produced.  The resulting asset covariance matrices
are annualised before optimisation.

For homogeneous liquid data, ``EwmaCovarEstimator`` is simpler: it samples all
asset returns at one ``returns_freq`` and emits annualised matrices at the
separate rebalancing cadence.

Failure modes
-------------

Every asset in a per-frequency Series must have a valid mapping, and every
mapping-valued span must contain each cadence used.  Do not compare a monthly
span of 12 with a quarterly span of 12 as if they represented the same
calendar window.  A lagged quarterly publication may also require an explicit
availability lag; resampling alone does not make it point-in-time.  In rolling
work, verify that every input is truncated to the decision date and that
signals are implemented after they become observable.

See also
--------

* :doc:`covariance_estimators` and :doc:`rolling_backtests`
* :doc:`api` for ``UniverseData``, ``compute_momentum_alpha``, and
  ``FactorCovarEstimator``
* `Offline mixed-frequency covariance example
  <https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/covar_estimation/demo_covar_different_estimation_freqs.py>`_
