Covariance estimators
=====================

Choose the covariance estimator from the structure of the investable universe,
not from the optimizer.  Every optimizer consumes the same contract: a square
asset covariance DataFrame for one date, or a dictionary of those matrices
keyed by decision date.

Estimator choice
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Estimator
     - Use when
     - Main trade-off
   * - ``EwmaCovarEstimator``
     - Assets share a return cadence and a direct, responsive covariance is
       appropriate.
     - Simple and transparent, but a large or short-history universe can
       produce a noisy full matrix.
   * - ``FactorCovarEstimator``
     - A sparse factor structure, mixed asset frequencies, or an HCGL risk
       model is part of the investment process.
     - More structured and diagnostic-rich, but requires factor data, a
       configured ``factorlasso.LassoModel``, and sufficient warm-up history.

EWMA covariance
---------------

``EwmaCovarEstimator`` converts total-return prices to log returns at
``returns_freq``.  It estimates an exponentially weighted covariance with
``span`` measured in observations at that frequency and returns annualised
matrices.  ``rebalancing_freq`` chooses matrix dates independently.

.. code-block:: python

   import optimalportfolios as opt

   estimator = opt.EwmaCovarEstimator(
       returns_freq="W-WED",
       span=52,
       rebalancing_freq="QE",
       demean=True,
   )
   current_covar = estimator.fit_current_covar(prices=prices)

Set ``is_apply_vol_normalised_returns=True`` only when the intended model is
the corresponding DCC-like normalized-return estimator.  Missing returns use
the estimator's zero-fill recursion; use explicit universe eligibility and
warm-up checks rather than interpreting that numerical fallback as observed
risk.

Factor and HCGL covariance
--------------------------

``FactorCovarEstimator`` estimates

.. math::

   \Sigma_y = \beta\Sigma_x\beta^\mathsf{T} + D,

where sparse factor loadings and the diagonal residual variance are supplied
through the canonical `factorlasso
<https://github.com/ArturSepp/factorlasso>`_ implementation.  The
``asset_returns_dict`` may contain different return frequencies for different
asset buckets.  Factor returns have their own ``factor_returns_freq`` and
``factor_covar_span``.  Returned asset covariance matrices are annualised.

The plain methods ``fit_current_covar`` and ``fit_rolling_covars`` satisfy the
common optimizer interface.  The factor-specific methods return betas,
residuals, clusters, R-squared, and other decomposition diagnostics.  Rolling
factor fits truncate every input through each estimation date and use an
expanding point-in-time history; an active cluster smoother is also computed
causally on that schedule.

Units and validation
--------------------

Optimizers do not resample or annualise covariance inputs.  With annualised
covariance, portfolio variance is annual variance and its square root is
annual volatility.  A covariance must use identical index and column labels,
be finite after documented asset filtering, and be symmetric.  Inspect
eigenvalues, marginal volatilities, sample availability, and changes through
time before trusting optimized weights.  For rolling work, independently
confirm that a matrix dated *t* uses no observation after *t*.

See also
--------

* :doc:`mixed_frequency_data` and :doc:`incomplete_histories`
* :doc:`api` for ``EwmaCovarEstimator`` and ``FactorCovarEstimator``
* `Offline mixed-frequency covariance example
  <https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/covar_estimation/demo_covar_different_estimation_freqs.py>`_
* `Factor covariance example
  <https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/covar_estimation/lasso_covar_estimation.py>`_
  (network-data example)
