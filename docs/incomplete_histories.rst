Incomplete histories and frozen positions
==========================================

Late starters, isolated missing observations, and positions that cannot trade
are different states.  Treating all three as NaN handling can produce a
portfolio that runs without representing the intended mandate.

Missing observations
--------------------

A **late starter** has leading NaNs before its first valid price.  A delisted
or discontinued series may have trailing NaNs.  Those boundaries describe
universe availability and should drive explicit eligibility rules.

An **interior gap** occurs between valid observations.  It is normally a data
quality problem, not an investability signal.  The ``qis`` backtester warns,
holds existing units through the gap, and omits the unpriced leg from NAV on
the missing date.  Repair or exclude the source series before interpreting
performance across such a gap.

Covariance estimators and optimizers have their own defensive rules.
``EwmaCovarEstimator`` feeds zero for a missing return into its recursive
state.  Solver wrappers remove assets whose covariance diagonal is NaN and
reject remaining non-finite matrices; near-zero finite variances are floored
for numerical stability.  Neither rule
is a substitute for an eligibility policy.  In particular, do not let a late
starter enter merely because a finite near-zero variance was produced during
its warm-up period.

Eligibility versus freezing
----------------------------

An **ineligible** asset is excluded from the solve and normally receives zero
target weight.  Depending on the solver, eligibility is expressed through an
``inclusion_indicators`` panel, a zero risk budget, or bounds.

A **frozen** asset remains owned but cannot rebalance.  Rolling solvers that
accept ``rebalancing_indicators`` interpret zero as hold: its current drifted
weight is fixed while tradable assets fill the remaining allocation.  This is
appropriate for lock-ups or staggered dealing windows; it is not the same as
selling the asset to zero.

.. code-block:: python

   import pandas as pd

   decision_dates = pd.to_datetime(["2024-03-28", "2024-06-28"])
   eligibility = pd.DataFrame(
       {"Liquid": [1.0, 1.0], "Late Starter": [0.0, 1.0]},
       index=decision_dates,
   )
   can_rebalance = pd.DataFrame(
       {"Liquid": [1.0, 1.0], "Locked Fund": [0.0, 0.0]},
       index=decision_dates,
   )

These panels encode different economics even if both contain zeros.  At the
first rolling date there is no prior position to freeze.  Missing indicator
rows are aligned differently by different wrappers, so construct a complete
panel on the covariance dates rather than relying on fill defaults.

Price gaps at implementation
----------------------------

When ``qis.backtest_model_portfolio`` cannot price an instrument on its trade
date, that leg is not traded and the intended allocation remains cash.  It is
not automatically redistributed.  Separately, the optimizer's drift helper
treats an unavailable drift price as a flat return and retains the prior
weight entry.  This conservative fallback keeps the rolling solve defined but
does not validate the underlying data.

Operational checks
------------------

Before a rolling run, inspect first-valid and last-valid dates, interior gaps,
eligibility at every decision, and whether a held asset is tradable.  After the
run, reconcile target weights, realised weights, cash, and warnings.  Group
bounds or desired risk shares may become infeasible when a frozen position
drifts through a limit; such an event needs an explicit policy decision, not
silent NaN filling.

See also
--------

* :doc:`rolling_backtests` and :doc:`turnover_and_transaction_costs`
* :doc:`minimum_tracking_error` for explicit inclusion indicators
* :doc:`risk_budgeting` for zero budgets and frozen positions
* :doc:`api` for ``filter_covar_and_vectors_for_nans`` and
  ``apply_drift_to_weights_0``
