"""
Comparison of two rolling backtests under the two ``weights_0`` policies in
``OptimiserConfig``:

    (A) legacy:   ``use_drifted_weights_0=False``
        — w_0 at each rebalance equals the previous *target* weights, with
        no adjustment for realised drift over the holding period. Turnover
        constraints and transaction-cost penalties are measured against a
        stale baseline.

    (B) drift-aware (production default): ``use_drifted_weights_0=True``
        — w_0 at each rebalance equals the previous target weights drifted
        to the current date using realised price returns. The turnover
        constraint then measures actual trades; transaction costs
        accumulated in the NAV simulator and the turnover budget seen by
        the optimiser refer to the same quantity.

Important: ``rolling_maximise_diversification`` uses SciPy's SLSQP, and
``set_scipy_constraints`` does NOT include the L1 turnover constraint
(only ``is_long_only``, exposure bounds, and group lower/upper bounds).
For SLSQP-based solvers the drift policy therefore only affects the
warm-start, and on this well-conditioned convex problem SLSQP converges
to the same local optimum from either start — so the two backtests come
out numerically identical.

To observe the policy difference we use ``rolling_quadratic_optimisation``
with ``PortfolioObjective.MIN_VARIANCE``, which is CVXPY-based and goes
through ``set_cvx_all_constraints`` — the path that actually enforces
``turnover_constraint``.

The (B) → (A) toggle is governed entirely by the
``OptimiserConfig.use_drifted_weights_0`` field added in the drift patch.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from enum import Enum
from typing import Dict, Tuple

from optimalportfolios import (Constraints, GroupLowerUpperConstraints,
                               EwmaCovarEstimator, OptimiserConfig,
                               PortfolioObjective,
                               rolling_quadratic_optimisation)

from optimalportfolios.examples.data.universe import fetch_benchmark_universe_data


# -----------------------------------------------------------------------------
# diagnostic helpers
# -----------------------------------------------------------------------------

def compute_realised_turnover(weights: pd.DataFrame,
                              prices: pd.DataFrame) -> pd.Series:
    """L1 distance between each rebalance's target and the *drifted* previous
    target — i.e. the trades actually executed in the simulator, in the
    self-financing convention used by ``apply_drift_to_weights_0``.

    Returns one number per rebalance date (first date is NaN).
    """
    realised = pd.Series(index=weights.index, dtype=float, name='realised_turnover')
    prev_date = None
    prev_w = None
    for date, w_new in weights.iterrows():
        if prev_w is None or prev_date is None:
            prev_date, prev_w = date, w_new
            continue
        p0 = prices.loc[:prev_date].ffill().iloc[-1].reindex(w_new.index)
        p1 = prices.loc[:date].ffill().iloc[-1].reindex(w_new.index)
        ratio = (p1 / p0.where(p0 > 0.0)).replace(
            [np.inf, -np.inf], np.nan).fillna(1.0)
        asset_returns = ratio - 1.0
        nav_growth = 1.0 + float((prev_w * asset_returns).sum())
        if not np.isfinite(nav_growth) or nav_growth < 1e-12:
            drifted = prev_w
        else:
            drifted = prev_w * ratio / nav_growth
        realised.loc[date] = float((w_new - drifted).abs().sum())
        prev_date, prev_w = date, w_new
    return realised


def compute_apparent_turnover(weights: pd.DataFrame) -> pd.Series:
    """L1 distance between consecutive target weights — what the optimiser
    believes the turnover to be under policy (A)."""
    diffs = weights.diff().abs().sum(axis=1)
    diffs.iloc[0] = np.nan
    return diffs.rename('apparent_turnover')


# -----------------------------------------------------------------------------
# core backtest under one drift policy
# -----------------------------------------------------------------------------

def run_backtest_under_policy(prices: pd.DataFrame,
                              constraints: Constraints,
                              covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                              use_drifted_weights_0: bool,
                              ticker: str,
                              group_data: pd.Series,
                              rebalancing_costs: float = 0.0003,
                              ) -> Tuple[pd.DataFrame, qis.PortfolioData]:
    cfg = OptimiserConfig(use_drifted_weights_0=use_drifted_weights_0)
    weights = rolling_quadratic_optimisation(
        prices=prices,
        constraints=constraints,
        covar_dict=covar_dict,
        portfolio_objective=PortfolioObjective.MIN_VARIANCE,
        optimiser_config=cfg,
    )
    portfolio_data = qis.backtest_model_portfolio(
        prices=prices,
        weights=weights,
        rebalancing_costs=rebalancing_costs,
        weight_implementation_lag=1,
        ticker=ticker,
    )
    portfolio_data.set_group_data(group_data=group_data)
    return weights, portfolio_data


# -----------------------------------------------------------------------------
# main runner
# -----------------------------------------------------------------------------

class LocalTests(Enum):
    DRIFT_POLICY_COMPARISON = 1


def run_local_test(local_test: LocalTests):
    """Compare drift-on (B) and drift-off (A) rolling backtests of a
    min-variance portfolio with a binding per-rebalance L1 turnover budget.
    """

    import optimalportfolios.local_path as lp

    prices, benchmark_prices, ac_loadings, benchmark_weights, group_data, _ = \
        fetch_benchmark_universe_data()

    # --- constraints ---
    # Group-allocation bounds plus a per-rebalance L1 turnover budget. The
    # budget is the lever that makes the two policies diverge: without it,
    # weights_0 only affects the warm-start.
    group_min_allocation = pd.Series(0.1, index=ac_loadings.columns)
    group_max_allocation = pd.Series(0.3, index=ac_loadings.columns)
    group_lower_upper_constraints = GroupLowerUpperConstraints(
        group_loadings=ac_loadings,
        group_min_allocation=group_min_allocation,
        group_max_allocation=group_max_allocation,
    )
    # turnover_constraint: hard L1 budget per rebalance. Pick a value tight
    # enough to bind: too loose and the difference between (A) and (B) is
    # invisible because the optimiser never trades against the budget.
    # 0.08 per quarter is below the unconstrained turnover (~0.11/qtr on
    # this universe), so the budget binds at most rebalances and exposes
    # the policy difference cleanly.
    constraints = Constraints(
        is_long_only=True,
        min_weights=0.0 * benchmark_weights,
        max_weights=3.0 * benchmark_weights,
        weights_0=benchmark_weights,
        turnover_constraint=0.08,
        group_lower_upper_constraints=group_lower_upper_constraints,
    )

    if local_test == LocalTests.DRIFT_POLICY_COMPARISON:
        # --- shared covar (single estimation, identical inputs to both runs) ---
        time_period = qis.TimePeriod('31Jan2007', '17Apr2025')
        rebalancing_costs = 0.0003
        covar_estimator = EwmaCovarEstimator()
        covar_dict = covar_estimator.fit_rolling_covars(
            prices=prices, time_period=time_period)

        # --- two backtests, same constraints, different drift policy ---
        weights_drift_on, pd_drift_on = run_backtest_under_policy(
            prices=prices, constraints=constraints, covar_dict=covar_dict,
            use_drifted_weights_0=True,
            ticker='MinVar DriftOn (B)',
            group_data=group_data,
            rebalancing_costs=rebalancing_costs,
        )
        weights_drift_off, pd_drift_off = run_backtest_under_policy(
            prices=prices, constraints=constraints, covar_dict=covar_dict,
            use_drifted_weights_0=False,
            ticker='MinVar DriftOff (A)',
            group_data=group_data,
            rebalancing_costs=rebalancing_costs,
        )

        # --- diagnostic: apparent vs realised turnover per rebalance ---
        # Under (A) the optimiser caps ||w - w_prev_target||_1 at 0.20 but
        # the actual trade is ||w - w_drift||_1 which is generally larger.
        # Under (B) the optimiser caps ||w - w_drift||_1 directly, so the
        # two columns should coincide.
        diag_off = pd.concat([
            compute_apparent_turnover(weights_drift_off).rename('apparent (constrained)'),
            compute_realised_turnover(weights_drift_off, prices).rename('realised (actual trades)'),
        ], axis=1)
        diag_on = pd.concat([
            compute_apparent_turnover(weights_drift_on).rename('apparent (constrained)'),
            compute_realised_turnover(weights_drift_on, prices).rename('realised (actual trades)'),
        ], axis=1)

        # --- console summary ---
        print('\n=== TURNOVER per rebalance (annualised averages) ===')
        n_per_year = 4.0  # quarterly rebalance schedule
        summary = pd.DataFrame({
            'Policy A (drift off)': [
                diag_off['apparent (constrained)'].mean() * n_per_year,
                diag_off['realised (actual trades)'].mean() * n_per_year,
                (diag_off['realised (actual trades)'].mean() /
                 diag_off['apparent (constrained)'].mean()
                 if diag_off['apparent (constrained)'].mean() > 0 else np.nan),
            ],
            'Policy B (drift on)': [
                diag_on['apparent (constrained)'].mean() * n_per_year,
                diag_on['realised (actual trades)'].mean() * n_per_year,
                (diag_on['realised (actual trades)'].mean() /
                 diag_on['apparent (constrained)'].mean()
                 if diag_on['apparent (constrained)'].mean() > 0 else np.nan),
            ],
        }, index=['apparent turnover (ann.)',
                  'realised turnover (ann.)',
                  'realised / apparent'])
        print(summary.round(4).to_string())

        print('\n=== TRANSACTION COST DRAG (cumulative, bps of NAV) ===')
        cost_drag = pd.Series({
            'Policy A (drift off)': diag_off['realised (actual trades)'].sum()
                                    * rebalancing_costs * 1e4,
            'Policy B (drift on)':  diag_on['realised (actual trades)'].sum()
                                    * rebalancing_costs * 1e4,
        })
        print(cost_drag.round(1).to_string())

        # --- factsheet: both policies as portfolios in one MultiPortfolioData ---
        multi_portfolio_data = qis.MultiPortfolioData(
            [pd_drift_on, pd_drift_off],
            benchmark_prices=benchmark_prices,
        )
        kwargs = qis.fetch_default_report_kwargs(
            time_period=time_period, add_rates_data=True)
        figs = qis.generate_strategy_benchmark_factsheet_plt(
            multi_portfolio_data=multi_portfolio_data,
            time_period=time_period,
            add_strategy_factsheet=True,
            add_grouped_exposures=False,
            add_grouped_cum_pnl=False,
            **kwargs,
        )

        # --- diagnostic chart: apparent vs realised turnover, both policies ---
        fig_diag, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        diag_off.plot(ax=axes[0],
                      title='Policy A (drift off): apparent vs realised turnover')
        axes[0].axhline(0.08, color='red', linestyle='--', linewidth=0.8,
                        label='turnover_constraint = 0.08')
        axes[0].legend(loc='upper right')
        axes[0].set_ylabel('L1 turnover per rebalance')
        diag_on.plot(ax=axes[1],
                     title='Policy B (drift on): apparent vs realised turnover')
        axes[1].axhline(0.08, color='red', linestyle='--', linewidth=0.8,
                        label='turnover_constraint = 0.08')
        axes[1].legend(loc='upper right')
        axes[1].set_ylabel('L1 turnover per rebalance')
        fig_diag.tight_layout()
        figs = list(figs) + [fig_diag]

        qis.save_figs_to_pdf(
            figs=figs,
            file_name='drift_policy_comparison_min_variance',
            orientation='landscape',
            local_path=lp.get_output_path(),
        )

        plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.DRIFT_POLICY_COMPARISON)
