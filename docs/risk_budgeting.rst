Risk budgeting
==============

Use risk budgeting when the allocation policy is expressed as shares of total
portfolio risk rather than capital weights or expected returns.  For
covariance :math:`\Sigma`, portfolio volatility :math:`\sigma_p`, and target
budget :math:`b_i`, the solver seeks

.. math::

   \frac{w_i(\Sigma w)_i}{\sigma_p} = b_i\sigma_p.

Thus a 30% risk budget is not a 30% portfolio weight.  Correlations and asset
volatilities determine the capital allocation required to deliver it.

Inputs and conventions
----------------------

``pd_covar`` is a covariance DataFrame in caller-supplied variance units; the
solver performs no frequency conversion.  Weights and risk budgets are
dimensionless.  Positive budgets are normalised internally, so they may be
entered as shares or proportional scores.  A zero or negative budget makes an
asset ineligible and it receives zero weight.

Use ``Constraints`` for long-only status, asset bounds, total exposure, and
group allocation bounds.  Group constraints act on loading-weighted capital
allocations, not on risk contributions.  Feasible bounds remain the caller's
responsibility.

Minimal offline example
-----------------------

The covariance below is annualised, although scale does not alter the capital
weights when every covariance entry is multiplied by the same positive
constant.

.. code-block:: python

   import pandas as pd
   import qis
   import optimalportfolios as opt

   assets = ["Equity", "Bonds", "Diversifier"]
   covar = pd.DataFrame(
       [[0.040, 0.004, 0.002],
        [0.004, 0.010, 0.001],
        [0.002, 0.001, 0.022]],
       index=assets,
       columns=assets,
   )
   budgets = pd.Series([0.50, 0.30, 0.20], index=assets)
   constraints = opt.Constraints(
       is_long_only=True,
       max_weights=pd.Series(0.80, index=assets),
   )
   weights = opt.wrapper_risk_budgeting(
       pd_covar=covar,
       constraints=constraints,
       risk_budget=budgets,
   )
   realised_budgets = qis.compute_portfolio_risk_contribution_ratios(
       weights=weights, covar=covar,
   )
   print(pd.concat([weights.rename("weight"),
                    realised_budgets.rename("risk share")], axis=1))

The weights sum to the exposure required by ``Constraints``.  In an
unconstrained feasible case, ``risk share`` is close to the normalised input
budgets; active bounds or group constraints can prevent exact matching.

Single-date versus rolling use
------------------------------

``wrapper_risk_budgeting`` handles one covariance matrix and is the right
place to inspect achieved risk shares.  ``rolling_risk_budgeting`` repeats the
solve on the dates in ``covar_dict``.  It aligns the covariance to the budget
order, drifts prior weights using realised prices, and optionally freezes
assets whose rebalancing indicator is zero.  ``risk_budget`` may be one static
asset-indexed Series or a date-by-asset DataFrame.  With a DataFrame, the
budget row is selected point in time at each covariance date.  The generic
``compute_rolling_optimal_weights`` dispatcher selects this path with
``PortfolioObjective.EQUAL_RISK_CONTRIBUTION``.

Group risk budgets
------------------

``compute_group_risk_budgets`` converts any complete or partially classified
partition into asset-level risk budgets.  For group size :math:`n_g` and
exponent :math:`\alpha`, the aggregate and within-group budgets are

.. math::

   B_g = \frac{n_g^\alpha}{\sum_h n_h^\alpha}, \qquad
   b_i = \frac{B_g}{n_g}.

``group_size_exponent=0`` gives every available group equal aggregate risk;
``1`` reproduces equal asset risk budgets; and ``0.5`` is the square-root-size
compromise.  Group labels may be statistical clusters, sectors, or asset
classes.  A membership DataFrame is transformed row by row without using
future classifications.

Hierarchical risk parity
------------------------

``compute_hierarchical_risk_parity_weights`` implements canonical recursive
bisection for a labelled covariance matrix and a caller-supplied SciPy
linkage.  Tree estimation deliberately remains outside OptimalPortfolios:
FactorLasso can construct the linkage, while this package converts the tree
into portfolio weights.  QIS owns covariance-implied attribution:
``qis.compute_group_portfolio_risk_contribution_ratios`` aggregates the
normalised Euler contributions of the resulting or any other portfolio over
the supplied groups.

Missing data, frozen assets, and feasibility
--------------------------------------------

Assets with non-positive budgets, explicit exclusion, or a NaN covariance
diagonal are removed and returned at zero; near-zero finite diagonal values
are floored for numerical stability.  If nothing remains,
the wrapper warns and returns zero weights.  A frozen position retains its
drifted pre-trade weight; tradable solved weights are scaled into the residual
allocation.  Frozen holdings still affect portfolio risk and can make desired
group allocations or risk shares infeasible.  Always compare achieved risk
shares with budgets after constraints have been applied.

See also
--------

* :doc:`rolling_backtests`
* :doc:`api` for ``wrapper_risk_budgeting``, ``rolling_risk_budgeting``, and
  the group-risk and HRP functions
* `Canonical risk-budgeting example
  <https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/solvers/risk_budgeting.py>`_
  (network-data example)
* `Offline solver comparison
  <https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/comparisons/risk_budgeting_ccd_vs_scipy.py>`_
