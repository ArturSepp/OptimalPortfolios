"""
Why scipy SLSQP fails for risk budgeting with negative correlations.

This example demonstrates the fundamental difference between the naive
scipy-based risk budgeting formulation and the pyrb convex reformulation.

Problem setup:
    Three assets with negative correlations and risk budget [45%, 45%, 10%].
    The negative correlations create a hedging opportunity that dramatically
    reduces portfolio volatility — but only if the solver finds the global
    optimum.

The scipy approach minimises the sum of squared deviations between
actual and target risk contributions:

    min_w  Σ_i (RC_i(w) - b_i σ_p(w))²

This objective is non-convex because both RC_i and σ_p depend non-linearly
on w. SLSQP is a local solver: starting from equal weights, it converges
to a local minimum where asset 3 (the hedging asset with negative
correlations) is dropped entirely. It never explores the region where
asset 3 gets ~48% weight — which is the global optimum.

The pyrb package uses the Spinu (2013) convex reformulation. Instead of
minimising RC deviations directly, it applies a change of variables
y = log(w) that transforms the risk budgeting equations into:

    min_y  (1/2) exp(y)' Σ exp(y)  -  Σ_i b_i y_i

This is convex in y with a unique global minimum, regardless of the
correlation structure or starting point.

The result is striking:
    - scipy:  vol ≈ 14.8%, drops the hedging asset, budget MAE ≈ 0.067
    - pyrb:   vol ≈  6.9%, exploits negative correlations, budget MAE ≈ 0.000

This is not a numerical precision issue — it is a fundamental difference
between solving a non-convex problem with a local solver vs solving a
convex reformulation with a global guarantee.

Reference:
    Spinu F. (2013), "An Algorithm for Computing Risk Parity Weights",
    SSRN Working Paper.

    Richard J.-C. and Roncalli T. (2019), "Constrained Risk Budgeting
    Portfolios: Theory, Algorithms, Applications & Puzzles",
    Available at https://arxiv.org/abs/1902.05710

    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation
    for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
"""
import numpy as np
import pandas as pd

from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.general.risk_budgeting import (
    opt_risk_budgeting,
    opt_risk_budgeting_scipy,
)
from optimalportfolios.utils.portfolio_funcs import (
    compute_portfolio_risk_contributions,
    compute_portfolio_variance,
)


def print_solution(label: str,
                   weights: np.ndarray,
                   covar: np.ndarray,
                   budget: np.ndarray,
                   tickers: list) -> None:
    """Print weights, risk contributions, vol, and budget tracking error."""
    rc = compute_portfolio_risk_contributions(weights, covar)
    rc_norm = rc / np.nansum(rc) if np.nansum(rc) > 0 else rc
    port_vol = np.sqrt(compute_portfolio_variance(weights, covar))
    budget_mae = np.mean(np.abs(rc_norm - budget))

    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    for i, t in enumerate(tickers):
        print(f"  {t:10s}  weight={weights[i]:7.4f}  RC={rc_norm[i]:7.4f}  target={budget[i]:5.2f}")
    print(f"  Portfolio vol:   {port_vol:.4%}")
    print(f"  Budget MAE:      {budget_mae:.6f}")
    print(f"  Weight sum:      {np.sum(weights):.4f}")


def run_example():
    # ── Universe setup ──────────────────────────────────────────────────
    # Three assets: two positively correlated, one negatively correlated
    # with both (the "hedging" asset). This creates a non-trivial
    # diversification opportunity that tests whether the solver can find it.
    tickers = ['Asset_1', 'Asset_2', 'Hedge']
    vols = np.array([0.20, 0.15, 0.10])
    corr = np.array([[ 1.0,   0.5,  -0.5  ],
                      [ 0.5,   1.0,  -0.333],
                      [-0.5,  -0.333, 1.0  ]])
    covar = np.outer(vols, vols) * corr

    # risk budget: 45% / 45% / 10%
    risk_budget = np.array([0.45, 0.45, 0.10])

    print("=" * 50)
    print("  Risk Budgeting: scipy vs pyrb comparison")
    print("=" * 50)
    print(f"\nAsset vols:    {dict(zip(tickers, [f'{v:.1%}' for v in vols]))}")
    print(f"Correlations:\n{np.array2string(corr, precision=3)}")
    print(f"Risk budget:   {dict(zip(tickers, risk_budget))}")

    constraints = Constraints(is_long_only=True)

    # ── Solver 1: scipy SLSQP (non-convex, local optimum) ──────────────
    #
    # Starting from x0 = [1/3, 1/3, 1/3], SLSQP finds a local minimum
    # where the hedge asset is dropped. This happens because:
    #
    # 1. From equal weights, reducing the hedge weight and redistributing
    #    to assets 1 and 2 decreases the RC deviation for those two assets
    #    (they approach 50/50 risk split).
    #
    # 2. The gradient at equal weights points toward this local basin,
    #    and SLSQP follows it without global exploration.
    #
    # 3. The global optimum (hedge ≈ 48%) sits in a different basin that
    #    requires increasing the hedge weight substantially — against the
    #    local gradient direction.
    #
    w_scipy = opt_risk_budgeting_scipy(covar=covar,
                                       constraints=constraints,
                                       risk_budget=risk_budget)
    print_solution("Scipy SLSQP (local solver, non-convex formulation)",
                   w_scipy, covar, risk_budget, tickers)

    # ── Solver 2: pyrb (convex reformulation, global optimum) ──────────
    #
    # pyrb uses the Spinu (2013) change of variables y = log(w):
    #
    #     min_y  (1/2) exp(y)' Σ exp(y)  -  Σ_i b_i y_i
    #
    # This is convex in y because:
    #   - exp(y)' Σ exp(y) is convex when Σ is PSD (composition of
    #     convex quadratic with componentwise convex exp)
    #   - Σ_i b_i y_i is linear
    #
    # The unique global minimum satisfies the risk budgeting conditions
    # exactly: RC_i(w*) = b_i σ_p(w*) for all i.
    #
    w_pyrb = opt_risk_budgeting(covar=covar,
                                constraints=constraints,
                                risk_budget=risk_budget)
    print_solution("pyrb (convex reformulation, global optimum)",
                   w_pyrb, covar, risk_budget, tickers)

    # ── Solver 3: pyrb with weight cap ─────────────────────────────────
    #
    # Adding a 40% max weight constraint. pyrb handles linear constraints
    # natively via ADMM. Note that the unconstrained solution already
    # satisfies the 40% cap for assets 1 and 2, but the hedge gets ~48%.
    # With the cap, the hedge is reduced to 40% and the remaining risk
    # budget is redistributed — still globally optimal within the
    # feasible set.
    #
    constraints_bounded = Constraints(
        is_long_only=True,
        max_weights=pd.Series(0.4, index=tickers)
    )
    w_pyrb_bounded = opt_risk_budgeting(covar=covar,
                                        constraints=constraints_bounded,
                                        risk_budget=risk_budget)
    print_solution("pyrb with max 40% per asset",
                   w_pyrb_bounded, covar, risk_budget, tickers)

    # ── Summary ─────────────────────────────────────────────────────────
    vol_scipy = np.sqrt(compute_portfolio_variance(w_scipy, covar))
    vol_pyrb = np.sqrt(compute_portfolio_variance(w_pyrb, covar))
    print(f"\n{'=' * 50}")
    print(f"  Summary")
    print(f"{'=' * 50}")
    print(f"  scipy portfolio vol:  {vol_scipy:.4%}")
    print(f"  pyrb portfolio vol:   {vol_pyrb:.4%}")
    print(f"  Vol reduction:        {1 - vol_pyrb / vol_scipy:.1%}")
    print(f"")
    print(f"  The {1 - vol_pyrb / vol_scipy:.0%} vol reduction comes entirely from")
    print(f"  exploiting the negative correlations via the hedge asset.")
    print(f"  scipy misses this because its non-convex objective has")
    print(f"  a local minimum at hedge weight ≈ 0.")
    print(f"")
    print(f"  Lesson: always use a convex reformulation (pyrb) for risk")
    print(f"  budgeting. The naive SSE formulation is unreliable whenever")
    print(f"  the correlation structure creates multiple local minima —")
    print(f"  which is common in multi-asset portfolios with hedging assets.")


if __name__ == '__main__':
    run_example()