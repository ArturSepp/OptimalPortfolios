# Overlay optimisation with a fixed core and linear side constraints

This note documents a `Constraints` pattern for a common institutional problem:
a core portfolio is held at a fixed weight, a sleeve of overlays is budgeted on
top, and the allocation must satisfy a linear tail or exposure condition in
addition to the usual bounds. The pattern needs no changes to the package: the
whole problem passes through `Constraints` into `cvx_maximize_portfolio_sharpe`.

## The problem

Maximise the Sharpe ratio of `core + sum_i w_i * overlay_i` subject to:

1. the core weight is fixed at 100%,
2. overlay weights are long-only and sum to a fixed budget (for example 100%,
   the "return stacking" configuration),
3. a linear floor `a' w >= b0`, where `a` collects per-asset contributions to a
   tail statistic (for example each asset's annualised volatility times its
   bear-regime Sharpe contribution, so that `a' w` is the portfolio's
   bear-regime return contribution).

## The encoding

**Fixed core.** Set `min_weights = max_weights = 1.0` for the core asset. Under
the Charnes-Cooper transformation used by `cvx_maximize_portfolio_sharpe`,
fixed-weight bounds scale correctly because they are homogeneous in the
transformed variable.

**Sleeve budget.** With one core at weight 1 and a sleeve budget of 1, set
`min_exposure = max_exposure = 2.0`. The equality of exposures is also what
routes the solver to the convex Charnes-Cooper path rather than the SLSQP
fallback. For several sleeves with separate budgets, use
`group_lower_upper_constraints` instead.

**Linear floor.** `Constraints(asset_returns=a, target_return=b0)` generates
the CVXPY constraint `a @ w >= b0`. Under Charnes-Cooper the variable is
`y = k * w` with `k > 0`, so the generated constraint `a @ y >= b0` is exact
if and only if `b0 = 0` (a homogeneous inequality). Two consequences:

- a floor of zero can be passed directly: `target_return = 0.0`;
- any nonzero floor can be made homogeneous whenever total exposure is fixed
  at `E`: since `e' w = E`, the constraint `a' w >= b0` is equivalent to
  `(a - (b0 / E) * e)' w >= 0`. Fold the floor into the coefficient vector and
  keep `target_return = 0.0`.

Do not pass a nonzero `target_return` into the Charnes-Cooper path directly:
the right-hand side would not be scaled by `k`. With variable net exposure
(`max_exposure != min_exposure`) the solver takes the SLSQP path, where a
nonzero `target_return` is handled as written.

## Verification

After solving, `constraints.check_constraints_violation(weights)` confirms the
bounds, and the floor can be checked directly as `a @ w >= b0`. Infeasible
floors are reported by the solver ("status=infeasible") with a fallback to
`weights_0` or zeros; a quick feasibility bound for a long-only sleeve with
budget `W` is `max(a_sleeve) * W + a_core >= b0`.

See `examples/solvers/overlay_tail_floor.py` for a runnable synthetic example.
