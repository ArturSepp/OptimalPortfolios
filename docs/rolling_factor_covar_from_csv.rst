Rolling factor risk model from CSV
==================================

This guide reconstructs the rolling factor-covariance objects used by
``optimalportfolios`` without importing ROSAA or any private package.  The
canonical example separates network acquisition from estimation:

* ``fetch`` downloads pedagogical Yahoo Finance proxies and writes a portable
  six-file CSV bundle;
* ``load`` starts a fresh CSV-only pipeline, reconstructs ``qis.FactorsData``
  and ``qis.FxRatesData``, estimates rolling HCGL covariance snapshots, and
  adapts them to ``qis.RiskModel``.

:download:`Download the complete executable example
<../examples/covar_estimation/rolling_factor_covar_from_csv.py>`.

.. important::

   Factor NAVs and FX rates are necessary but not sufficient.  The estimator
   also needs native-currency asset prices, each asset's currency, hedge ratio
   and return cadence, a reference currency, and the covariance calibration.
   The six-file bundle makes those inputs explicit.

Pipeline at a glance
--------------------

::

   Yahoo proxies or delivered MATF factor NAVs
                         |
                         v
                  six input CSV files
                         |
             +-----------+------------+
             |                        |
             v                        v
      qis.FactorsData          qis.FxRatesData
                                      |
                         native asset prices + metadata
                                      |
                                      v
                       frequency-keyed reference-currency
                                  asset returns
             |                        |
             +-----------+------------+
                         v
                 FactorCovarEstimator
                         |
                         v
              RollingFactorCovarData
              betas + factor covariance
                 + diagonal residual risk
                         |
                         v
                    qis.RiskModel

The rolling fit is point in time: every snapshot dated *t* uses factor prices
and asset returns only through *t*.  The default example estimates monthly
returns and produces annual snapshots; changing the rebalance cadence does
not change the return cadence.

Install and run
---------------

The script lives in the repository-only ``examples/`` tree, so run these
commands from a source checkout.  The fetch stage needs the optional data
dependency::

   pip install -e ".[data]"
   python -m examples.covar_estimation.rolling_factor_covar_from_csv fetch --data-dir path/to/risk_model_inputs

Run the analysis later, or on a separate machine, from the saved bundle::

   python -m examples.covar_estimation.rolling_factor_covar_from_csv load --data-dir path/to/risk_model_inputs

The load stage never imports or calls ``yfinance``.  A recipient needs only
the core ``qis``, ``factorlasso`` and ``optimalportfolios`` dependencies plus
the script.  If the script is copied out of the checkout, invoke it directly
and always supply the bundle path::

   python rolling_factor_covar_from_csv.py load --data-dir path/to/risk_model_inputs

Running the module with no mode performs both stages and writes its temporary
bundle below the ignored ``tmp/yahoo_factor_risk_model`` directory.  ``fetch``
replaces the six named files, so never run it over a delivered MATF bundle.

The six-file CSV contract
-------------------------

Every time-series CSV uses its first column as a date index.  Dates must parse
as a sorted, unique ``DatetimeIndex``.

.. list-table::
   :header-rows: 1
   :widths: 27 23 50

   * - File
     - Loaded as
     - Contract
   * - ``futures_risk_factors.csv``
     - ``qis.FactorsData``
     - Strictly positive, finite factor price or NAV **levels**, not returns.
       Columns must exactly match the ordered, pipe-delimited ``factor_names``
       setting.  The demo columns are ``Equity``, ``Rates``, ``Credit``,
       ``Commodities`` and ``Fx``.
   * - ``fx_hedging_data_fx_spots.csv``
     - ``qis.FxRatesData.fx_spots``
     - Strictly positive, finite spots in the qis anchor convention: USD per
       one unit of each currency.  Include ``USD=1`` and every asset and
       reference currency.
   * - ``fx_hedging_data_domestic_rates.csv``
     - ``qis.FxRatesData.domestic_rates``
     - Finite annualised short rates as decimal fractions: ``0.05`` means 5%.
       All required currencies must be present.  Rates feed FX hedge and carry
       calculations even when asset returns are not excess returns.
   * - ``asset_prices.csv``
     - ``pandas.DataFrame``
     - Strictly positive, finite adjusted price levels in each asset's native
       currency.  Columns are asset identifiers.
   * - ``asset_metadata.csv``
     - ``pandas.DataFrame``
     - One row per asset-price column, indexed by ``asset``.  Required columns
       are ``currency``, ``hedge_ratio`` and ``return_frequency``.  Hedge
       ratios lie in ``[0, 1]``; ``0`` is unhedged and ``1`` fully hedged.
       Frequencies use pandas aliases such as ``ME``.
   * - ``risk_model_settings.csv``
     - ``RiskModelSettings``
     - Two columns, ``setting`` and ``value``, containing the factor-column
       contract, return convention, HCGL/LASSO calibration and estimation
       dates.  All 15 settings shown below are required.

The default settings are deliberately explicit:

.. code-block:: text

   reference_ccy,CHF
   is_log_returns,True
   is_excess_returns,False
   factor_returns_freq,ME
   factor_names,Equity|Rates|Credit|Commodities|Fx
   factor_covar_span,36
   rebalancing_freq,YE
   lasso_model_type,HIERARCHICAL_CLUSTER_GROUP_LASSO
   reg_lambda,1e-05
   beta_span,36
   warmup_period,36
   demean,True
   solver,CLARABEL
   estimation_start,2019-12-31
   estimation_end,2025-12-31

``factor_covar_span`` and ``beta_span`` are measured in observations at the
monthly factor/asset cadence here.  ``rebalancing_freq=YE`` selects covariance
snapshot dates independently.  ``estimation_end`` cannot exceed the earliest
last date among factors, assets, FX spots and rates.  Separate ``factor_names``
with literal ``|`` characters and no surrounding spaces.

Yahoo demonstration basis
-------------------------

The free-data stage is an executable data-contract example, not a substitute
for MATF.  It downloads adjusted closes over the fixed interval from
2007-12-31 through 2026-01-01 (Yahoo's end date is exclusive) and aligns them
on one forward-filled business-day grid.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Factor
     - Yahoo proxy
     - Construction
   * - Equity
     - ``SPY``
     - Fully FX-hedged from USD into CHF at the monthly factor cadence.
   * - Rates
     - ``TLT``
     - Fully FX-hedged from USD into CHF at the monthly factor cadence.
   * - Credit
     - ``LQD``
     - Fully FX-hedged from USD into CHF at the monthly factor cadence.
   * - Commodities
     - ``GLD``
     - Fully FX-hedged from USD into CHF at the monthly factor cadence.
   * - Fx
     - ``CHF=X``
     - USD held by a CHF investor, including spot and carry.  Yahoo quotes CHF
       per USD; the input spot panel inverts that quote to the qis USD-per-CHF
       convention.

Fully hedging the four asset-class proxies keeps currency risk out of those
factors.  An unhedged USD asset can then load on its economic asset-class
factor and the separate ``Fx`` factor; a fully hedged USD asset need not take
an offsetting FX loading merely because currency risk was embedded in its
asset-class proxy.

The asset panel contains seven USD ETFs.  ``IEF`` and ``HYG`` are fully
hedged; ``QQQ``, ``EFA``, ``EEM``, ``VNQ`` and ``GSG`` are unhedged.  This
mix makes the role of ``FxRatesData`` visible in the estimated betas.

Yahoo provides ``^IRX`` as a percentage yield, so the example divides it by
100 for the USD annual decimal rate.  The CHF curve is deliberately
illustrative: USD minus one percentage point.  Replace it with an appropriate
point-in-time CHF short-rate history for any production analysis.

Fetch and persist
-----------------

The fetch-only entry point is callable independently:

.. code-block:: python

   from pathlib import Path

   from examples.covar_estimation.rolling_factor_covar_from_csv import (
       fetch_and_save_yahoo_csvs,
   )

   fetch_and_save_yahoo_csvs(Path("path/to/risk_model_inputs"))

``yfinance`` is imported inside the downloader rather than at module import
time.  The function builds a ``qis.FxRatesData`` object to convert the factor
proxies, then writes the QIS-native factor and FX filenames with
``qis.save_df_to_csv`` and ``qis.save_df_dict_to_csv``.  It also persists
asset prices, metadata and calibration so no in-memory fetch object crosses
the stage boundary.

Reload every input
------------------

``load_inputs_from_csv`` reconstructs the full input state:

.. code-block:: python

   from pathlib import Path
   import qis

   from examples.covar_estimation.rolling_factor_covar_from_csv import (
       load_inputs_from_csv,
   )

   inputs = load_inputs_from_csv(Path("path/to/risk_model_inputs"))

   assert isinstance(inputs.factors_data, qis.FactorsData)
   assert isinstance(inputs.fx_rates_data, qis.FxRatesData)
   factor_prices = inputs.factors_data.get_prices()
   asset_prices = inputs.asset_prices
   metadata = inputs.asset_metadata
   settings = inputs.settings

The metadata CSV is loaded with ``parse_dates=False`` because its index holds
asset identifiers rather than dates.  The two fixed FX filenames are loaded
together by ``qis.FxRatesData.load``; the factor panel is loaded by
``qis.FactorsData.load``.

Convert asset returns
---------------------

Asset prices remain in native currency in the bundle.  The load stage uses
the currency and hedge metadata to construct returns in the configured
reference currency:

.. code-block:: python

   asset_returns_dict = inputs.fx_rates_data.compute_fx_adjusted_returns(
       prices=inputs.asset_prices,
       hedge_ratios=metadata["hedge_ratio"],
       local_ccys=metadata["currency"].astype(str),
       reference_ccy=settings.reference_ccy,
       freq=metadata["return_frequency"].astype(str),
       is_log_returns=settings.is_log_returns,
       is_excess_returns=settings.is_excess_returns,
   )

The result is a dictionary keyed by return cadence, which lets one estimator
consume monthly and quarterly asset buckets without pretending that their
observations have the same frequency.  This demo produces monthly CHF log
total returns.  If ``is_excess_returns`` is enabled in a different workflow,
qis subtracts the reference-currency short rate.

``FactorCovarEstimator`` computes log returns from factor NAVs.  The example
therefore requires ``is_log_returns=True`` for the asset side as well and
rejects an inconsistent setting.  Factor NAVs and asset returns must also use
the same total-versus-excess and reference-currency basis; column names cannot
validate that economic convention.

Fit the rolling decomposition
-----------------------------

The core estimation path uses only public ``optimalportfolios`` and qis APIs:

.. code-block:: python

   import optimalportfolios as opt
   import qis

   model_type = opt.LassoModelType[settings.lasso_model_type]
   lasso_model = opt.LassoModel(
       model_type=model_type,
       reg_lambda=settings.reg_lambda,
       span=settings.beta_span,
       warmup_period=settings.warmup_period,
       demean=settings.demean,
       solver=settings.solver,
   )
   estimator = opt.FactorCovarEstimator(
       rebalancing_freq=settings.rebalancing_freq,
       lasso_model=lasso_model,
       factor_returns_freq=settings.factor_returns_freq,
       factor_covar_span=settings.factor_covar_span,
       demean=settings.demean,
   )

   rolling = estimator.fit_rolling_factor_covars(
       risk_factor_prices=inputs.factors_data.get_prices(),
       asset_returns_dict=asset_returns_dict,
       assets=inputs.asset_prices.columns,
       time_period=qis.TimePeriod(
           start=settings.estimation_start,
           end=settings.estimation_end,
       ),
   )
   risk_model = opt.build_risk_model(rolling)

``fit_rolling_factor_covars`` slices factor prices and every asset-return
bucket through each annual estimation date before fitting.  The returned
``RollingFactorCovarData`` is a dated collection of
``CurrentFactorCovarData`` snapshots.  For each date it contains:

* ``x_covar`` -- the annualised :math:`M\times M` factor covariance;
* ``y_betas`` -- the :math:`N\times M` asset-by-factor loading matrix;
* ``y_variances`` -- annualised total and residual variances, alpha and
  R-squared diagnostics;
* ``residuals`` -- the fitted residual history;
* HCGL cluster assignments, linkage rows and cutoff metadata.

The rolling accessors expose panels and optimizer-ready covariance matrices:

.. code-block:: python

   latest_date = rolling.dates[-1]
   latest = rolling.get_latest()

   betas = latest.y_betas
   factor_covar = latest.x_covar
   asset_covars = rolling.get_y_covars()
   r_squared = rolling.get_r2()
   residual_variances = rolling.get_residual_vars()

``opt.build_risk_model`` preserves the dated asset covariances, factor
loadings, factor covariances and residual variances.  That makes
``risk_model`` directly usable by qis exposure and risk-attribution
consumers, while ``rolling.get_y_covars()`` is the standard covariance input
for OP rolling optimizers.

Numerical reconstruction check
------------------------------

Every snapshot must satisfy

.. math::

   \Sigma_{asset,t} = B_t\Sigma_{factor,t}B_t^\mathsf{T} + D_t,

where :math:`D_t` is diagonal and contains the annualised residual variances.
The example rebuilds the matrix with NumPy after explicitly aligning factor
and asset labels:

.. code-block:: python

   import numpy as np
   import optimalportfolios as opt

   snapshot = rolling.get_latest()
   betas = snapshot.y_betas
   factor_covar = snapshot.x_covar.reindex(
       index=betas.columns,
       columns=betas.columns,
   )
   residual_vars = snapshot.y_variances[
       opt.VarianceColumns.RESIDUAL_VARS.value
   ].reindex(betas.index)
   expected = (
       betas.to_numpy()
       @ factor_covar.to_numpy()
       @ betas.to_numpy().T
       + np.diag(residual_vars.to_numpy())
   )
   actual = snapshot.get_y_covar().reindex(
       index=betas.index,
       columns=betas.index,
   )
   np.testing.assert_allclose(
       actual.to_numpy(), expected, rtol=1.0e-12, atol=1.0e-14
   )

It also rejects non-finite covariance entries and any minimum eigenvalue below
``-1e-10``.  The console reports the maximum reconstruction error, latest
betas, annualised factor covariance, annualised residual volatilities and
equal-weight portfolio factor exposures.

Replace Yahoo with delivered MATF data
--------------------------------------

Use the Yahoo bundle as a schema template, not as production factor data:

#. Preserve the bundle and replace ``futures_risk_factors.csv`` with delivered
   MATF **NAV levels**, not a return panel.
#. Set ``factor_names`` in ``risk_model_settings.csv`` to the exact ordered
   factor headers.  No Python enum or ROSAA import is required.
#. Confirm the MATF NAVs already have the same factor-currency construction as
   the asset returns.  If the reference currency or hedge basis changes,
   transform the economics and update the FX, rates, metadata and settings
   coherently; renaming a column is not a currency conversion.
#. Replace ``asset_prices.csv`` and ``asset_metadata.csv`` with the intended
   universe.  Keep native-currency price levels and record each asset's
   currency, hedge ratio and reliable observation cadence.
#. Supply point-in-time FX spots and domestic rates for every local currency
   plus the reference currency.  Preserve the qis spot and annual-decimal rate
   conventions.
#. Set the spans, solver, rebalance schedule and estimation dates in
   ``risk_model_settings.csv``.  Retain enough history before the first
   estimation date for warm-up.
#. Run only ``load``.  The recipient needs neither Yahoo access nor ROSAA.

Structural validation catches missing files, malformed booleans, factor-order
mismatches, duplicate or unsorted dates, non-finite inputs, non-positive NAVs,
prices or spots, asset/metadata mismatches, invalid frequencies, hedge ratios
outside ``[0, 1]``, missing currencies, unknown model types and an estimation
end beyond the common sample.  It cannot infer whether a numerically valid
MATF panel uses the intended economic currency basis; that remains part of
the delivery specification.

Persistence boundary
--------------------

``RollingFactorCovarData`` has no native CSV loader.  The portable contract is
the six source CSVs; the rolling object is deliberately reconstructed by
estimation from those point-in-time inputs.  The programmatic convenience
entry point returns both downstream objects and writes no model artifact:

.. code-block:: python

   from pathlib import Path

   from examples.covar_estimation.rolling_factor_covar_from_csv import (
       fit_rolling_risk_model_from_csv,
   )

   rolling, risk_model = fit_rolling_risk_model_from_csv(
       Path("path/to/risk_model_inputs")
   )

The Yahoo proxies, their fixed history and the illustrative CHF rate are for
education only.  Network data can be revised or unavailable, and applicable
provider terms must be reviewed before redistribution.  Generated data stays
under the ignored ``tmp/`` tree by default and should not be committed.

See also
--------

* :doc:`covariance_estimators` for estimator selection and covariance units.
* :doc:`mixed_frequency_data` for the three independent model clocks.
* :doc:`rolling_backtests` for consuming dated covariance matrices in a
  portfolio backtest.
* :doc:`api` for the exported estimator, covariance-container and risk-model
  adapter APIs.
