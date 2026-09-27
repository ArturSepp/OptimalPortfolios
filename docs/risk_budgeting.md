---
myst:
  html_meta:
    description: >-
      Risk budgeting in OptimalPortfolios: Euler risk contributions, normalized budgets,
      constrained allocation with a binding bound, and offline examples.
---

# Risk budgeting

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/OptimalPortfolios/commit/fb8848d327c0585eaf0933dba6137ec6b8338bbf)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Risk budgeting allocates capital to target shares of portfolio risk. A 30% risk budget is a
target contribution to total risk; it does not specify a 30% capital weight. Asset volatilities,
correlations and binding constraints determine the resulting allocation.

## Overview

Use risk budgeting when an allocation policy is expressed through risk shares rather than
expected returns. Equal risk contribution (ERC) is the special case of equal asset budgets.
This article covers the volatility-based risk-budgeting solver. Group-to-asset budget design and
the separate hierarchical risk parity allocation are described in
[HRP and cluster risk budgets](hierarchical_risk_parity_and_cluster_budgets.md).

OptimalPortfolios owns portfolio construction. [qis](https://github.com/ArturSepp/QuantInvestStrats)
provides the risk-contribution analytics used below and the holdings simulation/reporting layer;
see the [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

<a id="inputs-and-conventions"></a>

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | None; the solvers take a covariance matrix and sample no returns. `rolling_risk_budgeting` uses prices only to drift the previous weights and to order its output columns |
| Estimation grid | None; the examples supply fixed synthetic covariance matrices, and covariance estimation is outside these functions |
| Rebalancing grid | One decision date per `wrapper_risk_budgeting` call; `rolling_risk_budgeting` solves at each key of `covar_dict`, in order |
| Covariance units | Annual, fractional return squared in the examples (0.040 is a 20% volatility); consumed in caller units, and the wrapper raises positive variances below `0.001**2` to that floor |
| Expected returns | None; risk budgeting uses only the covariance and the budgets |
| Weight state | Long-only, fully invested target weights. The rolling path passes each date's weights, drifted by prices, as the next `weights_0`, which serves freezing and the fallback |
| Solver | In-house, no CVXPY: cyclical coordinate descent (CCD) with the default box and no group rows, otherwise ADMM with a CCD step and a `quadprog` projection; the script's reference solve uses CVXPY with CLARABEL |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning and units |
|---|---|
| $w$ | $n$-asset capital-weight vector; fractions of NAV |
| $\Sigma$ | Covariance matrix of size $n \times n$ in caller-supplied variance units |
| $\sigma_p$ | Portfolio volatility in the square-root units of $\Sigma$ |
| $\mathrm{RC}_i$ | Asset $i$'s Euler contribution to portfolio volatility |
| $r_i$ | Asset $i$'s dimensionless fraction of portfolio volatility |
| $b_i$ | Normalized target risk share; positive on the eligible free universe |

Supply a symmetric covariance DataFrame with unique, identically ordered asset labels on both
axes. The model assumes a valid positive-semidefinite covariance and positive portfolio variance;
a positive-definite matrix with positive asset variances avoids degenerate risk-budget problems.
Covariance, budgets and tradability information must be available at the decision date. These
functions do not estimate covariance or convert its frequency.

Finite positive input budgets are normalized over the surviving assets, so proportional scores
such as `[50, 30, 20]` and fractional shares `[0.50, 0.30, 0.20]` define the same target.
The ordinary wrapper path excludes zero, negative or missing budgets from the solve.
Freezing and the single-asset rolling shortcut have qualifications described under
[limitations](#missing-data-frozen-assets-and-feasibility).

Use `Constraints` for asset boxes and loading-weighted group allocation bounds. This risk-budget
backend is long-only and **fully invested**: its solver requires weights to sum to one.
Arbitrary exposure bands, return floors, volatility limits, turnover limits, tracking-error
limits and beta bounds are not compiled by this backend. Setting `is_long_only=False` does not
turn its logarithmic formulation into a short-selling optimizer. Consult the
[backend capability matrix](constraints.md#backend-capability-matrix) before selecting a mandate.

## Methodology

### Euler contributions and target shares

For positive portfolio volatility, the Euler decomposition gives:

$$
\begin{aligned}
\sigma_p(w) &= \sqrt{w^\top \Sigma w},\\
\mathrm{RC}_i(w)
  &= w_i\frac{\partial \sigma_p}{\partial w_i}
   = \frac{w_i(\Sigma w)_i}{\sigma_p},\\
\sum_i \mathrm{RC}_i(w) &= \sigma_p.
\end{aligned}
$$

The derivative is marginal risk; multiplying it by the capital weight gives the asset's
contribution. The general allocation principle is discussed by
[Tasche (2007, revised 2008)](https://arxiv.org/abs/0708.2542).

Fractional contributions and the risk-budgeting target are:

$$
\begin{aligned}
r_i(w) &= \frac{\mathrm{RC}_i(w)}{\sigma_p}
       = \frac{w_i(\Sigma w)_i}{w^\top\Sigma w},\\
r_i(w) &= b_i
\quad\Longleftrightarrow\quad
\mathrm{RC}_i(w)=b_i\sigma_p,\\
\sum_i b_i &= 1.
\end{aligned}
$$

A contribution is not `weight × standalone asset volatility`: the covariance with the entire
portfolio matters. Contributions of an arbitrary constrained or hedged portfolio can be
negative, even when weights are nonnegative. Target shares need not equal achieved shares
after a bound binds or frozen positions are restored.

### Constrained allocation and normalization

On the eligible free universe, let $\ell_i,u_i$ be per-asset bounds, $L_g$ the asset-loading
vector for group $g$, and $\ell_g,u_g$ bounds on that group's capital allocation. The current
implementation minimizes the following objective over the fully invested feasible set, with
positive weights for positive free budgets:

$$
\begin{aligned}
\min_w\quad
  &\log \sigma_p(w)-\sum_i b_i\log w_i,\\
\text{subject to}\quad
  &\sum_i w_i=1,\qquad \ell_i\le w_i\le u_i,\\
  &\ell_g\le L_g^\top w\le u_g.
\end{aligned}
$$

A zero-pinned asset is handled outside the positive logarithmic domain. Group capital bounds constrain $L_g^\top w$;
they are not bounds on the sum of risk contributions.

The solver works with unnormalized positions $y$ and weights $w = y/\sum_j y_j$. On $y$ every
instrument and group bound is homogeneous, and the solver minimizes
$\tfrac12 y^\top Q y - \lambda\sum_i b_i\log y_i$, where $Q$ is the covariance divided by its
largest diagonal entry and $\lambda$ is a positive numerical scale that does not change $w$.
With the default $[0, 1]$ box and no group rows, exact cyclical coordinate descent (CCD) solves
it: each coordinate update is the positive root of a quadratic. Any other bound, even a slack
one, routes the solve through the alternating direction method of multipliers (ADMM), which
alternates a CCD proximal step with a `quadprog` projection onto the homogeneous constraint
faces. Only a converged, feasible solution is returned; the solver has no SciPy fallback.

Instrument and group bounds apply to the normalized portfolio together with full investment.
It does not clip an unconstrained solution and then renormalize it. Exact target shares are
recovered when no additional bound binds; otherwise budgets express preferences in the
constrained objective.

[Richard and Roncalli (2019)](https://arxiv.org/abs/1902.05710) discuss constrained risk budgeting
and the scaling-compatibility problem. The formula above states this package's current
normalized-weight contract. The paper's constrained absolute-bound table values are not
numerical references for this homogeneous formulation.

### Group risk budgets

Budgets for this solver can be set per group rather than per asset. `compute_group_risk_budgets`
turns group or cluster labels into asset-level budgets to pass as `risk_budget`; its formula,
exponents and point-in-time panels are described in
[HRP and cluster risk budgets](hierarchical_risk_parity_and_cluster_budgets.md#group-risk-budgets).

### Hierarchical risk parity

Hierarchical risk parity is a separate allocation rule that takes a covariance and a linkage, and
no budgets or `Constraints`. [HRP and cluster risk budgets](hierarchical_risk_parity_and_cluster_budgets.md#recursive-bisection)
describes it and compares it with this solver under cluster budgets.

## Worked example

The three Python blocks below run in order and need no download, data file or random seed. They
are excerpts of the canonical script
[`examples/docs/risk_budgeting.py`](../examples/docs/risk_budgeting.py), which runs them and
asserts every number and property on this page against a reference computed a different way:
an independent conic solve of the same objective, the closed-form diagonal solution and the
optimality conditions of a binding cap:

```console
python -m examples.docs.risk_budgeting
```

### Minimal offline example

This fixed synthetic example has three assets, an annualized covariance and target budgets of
50%, 30% and 20%. The 80% per-asset cap is slack at the solution: without it, the
solve takes the pure CCD route and returns the same weights.

```python
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
```

The approximate result, in fractions, is:

| Asset | Capital weight | Target risk share | Achieved risk share |
|---|---:|---:|---:|
| Equity | 0.301339 | 0.50 | 0.50 |
| Bonds | 0.441151 | 0.30 | 0.30 |
| Diversifier | 0.257511 | 0.20 | 0.20 |

Unrounded weights sum to one. The higher-volatility Equity asset needs about 30.1% of capital
to contribute 50% of risk in this covariance model. This is a calculation example, not an
empirical performance result.

### When a weight bound binds

Cap Equity at 25%, below its 30.1% solution, and keep the other inputs of the first example:

```python
capped_constraints = opt.Constraints(
    is_long_only=True,
    max_weights=pd.Series([0.25, 0.80, 0.80], index=assets),
)
capped_weights = opt.wrapper_risk_budgeting(
    pd_covar=covar,
    constraints=capped_constraints,
    risk_budget=budgets,
)
capped_shares = qis.compute_portfolio_risk_contribution_ratios(
    weights=capped_weights, covar=covar,
)
print(pd.concat([capped_weights.rename("weight"),
                 capped_shares.rename("risk share")], axis=1))
```

The result, rounded to three decimals, is:

| Asset | Capital weight | Target risk share | Achieved risk share |
|---|---:|---:|---:|
| Equity | 0.250 | 0.50 | 0.394 |
| Bonds | 0.479 | 0.30 | 0.368 |
| Diversifier | 0.271 | 0.20 | 0.238 |

The cap binds and no asset meets its budget. In this example, stationarity of the objective
under full investment gives $r_i = b_i + c w_i$ for both assets below their caps, with one
multiplier $c \ge 0$, and the capped asset absorbs the difference: $r_k = b_k - c(1 - w_k)$.
The solve is not the first solution clipped at 25% with the released capital spread pro rata.

![Left: risk shares of the three assets against their target budgets of 50%, 30% and 20%.
Without a binding bound each share equals its budget; with Equity capped at 25%, Equity carries
39% of the risk, Bonds 37% and Diversifier 24%. Right: capital weights; the cap moves Equity
from 30% to 25% and raises Bonds from 44% to 48% and Diversifier from 26% to
27%.](images/risk_budgeting_contributions.png)

*Figure: target budgets and achieved risk shares of the worked example with the 80% caps slack
and with Equity capped at 25%, and the capital weights of both solves. Drawn by the `exhibit`
function of the canonical script; the [analytics gallery](analytics_gallery.md) lists its
provenance.*

> **Insight.** When a cap binds, no asset meets its budget. Each free asset overshoots by the
> same multiple of its capital weight, 0.141 here, and the capped asset falls short by their
> total excess. Bonds therefore takes 1.77 times the excess of Diversifier, the ratio of their
> weights, not the 1.5 ratio of their budgets.

### Independent diagonal-covariance check

For uncorrelated assets with volatilities $\sigma_i \gt 0$, positive budgets, no binding bounds
and no active variance floor, the target equations reduce to:

$$
w_i=
\frac{\sqrt{b_i}/\sigma_i}
     {\sum_j \sqrt{b_j}/\sigma_j}.
$$

For volatilities `[0.20, 0.10]` and budgets `[0.80, 0.20]`, both numerators are equal.
The capital weights are therefore `[0.50, 0.50]`, while the risk shares are `[0.80, 0.20]`.
Equal budgets instead give inverse-volatility weights; inverse variance is a different rule.

Run this block after the first example:

```python
diagonal_assets = ["Growth", "Defensive"]
diagonal_covar = pd.DataFrame(
    [[0.04, 0.0], [0.0, 0.01]],
    index=diagonal_assets,
    columns=diagonal_assets,
)
diagonal_budgets = pd.Series([0.80, 0.20], index=diagonal_assets)
diagonal_weights = opt.wrapper_risk_budgeting(
    pd_covar=diagonal_covar,
    constraints=opt.Constraints(is_long_only=True),
    risk_budget=diagonal_budgets,
)
diagonal_shares = qis.compute_portfolio_risk_contribution_ratios(
    weights=diagonal_weights, covar=diagonal_covar,
)
print(diagonal_weights.round(6).tolist())  # [0.5, 0.5]
print(diagonal_shares.round(6).tolist())   # [0.8, 0.2]
```

The canonical script checks this closed-form result independently of the solver and compares
both correlated examples with an independent conic solve of the same objective. Risk
contributions are reported by qis; the script recomputes every reported share as
$w_i(\Sigma w)_i / w^\top\Sigma w$.

### Partially classified groups

The group-budget example, with one Growth asset, two Defensive assets and an unclassified asset
that receives a zero budget, is worked on the page
[HRP and cluster risk budgets](hierarchical_risk_parity_and_cluster_budgets.md#partially-classified-groups),
together with a comparison of cluster budgets, ERC and HRP.

## Implementation in optimalportfolios

### Single-date versus rolling use

| Entry point | Input and output contract |
|---|---|
| `wrapper_risk_budgeting` | One covariance DataFrame; optional asset-indexed Series/dict budgets; returns a weight Series by default |
| `rolling_risk_budgeting` | Price panel, covariance-date dictionary, and static Series, date-by-asset DataFrame or `None` budgets; returns dated target weights |
| `opt_risk_budgeting` | NumPy covariance, `Constraints` and budget array, with no filtering, variance floor or freezing; returns a weight array, or the fallback after a failed solve |
| `compute_group_risk_budgets`, `compute_hierarchical_risk_parity_weights` | Group labels to budgets for `risk_budget`, and the separate HRP allocation; see [HRP and cluster risk budgets](hierarchical_risk_parity_and_cluster_budgets.md) |

For rolling allocations, each covariance date must have an exact budget row when budgets are a
DataFrame: missing dates raise `ValueError` rather than selecting a future or nearest observation.
Supply the covariance dictionary in chronological order. The routine aligns covariance to the
budget order and, by default, drifts prior weights with observed prices before the next solve.
`OptimiserConfig.use_drifted_weights_0=False` disables that drift adjustment.

Rebalancing indicators are applied after a prior allocation exists. Missing indicator dates
are filled with zeros in the rolling path, which can freeze that date's holdings; supply an
explicit schedule. `PortfolioObjective.EQUAL_RISK_CONTRIBUTION` selects this path through
`compute_rolling_optimal_weights`. Resulting rows are target weights, not a holdings backtest:
[rolling implementation timing](rolling_backtests.md) and qis determine their application.

The [risk-allocation sources](https://github.com/ArturSepp/OptimalPortfolios/tree/main/src/optimalportfolios/optimization/risk_allocation)
and [API reference](api.rst) describe the public entry points. The
[canonical script](../examples/docs/risk_budgeting.py) runs the worked example and checks its
weights against an independent conic solve, the closed-form diagonal solution and the
optimality conditions of the binding cap, together with the filtering, freezing, fallback and
rolling statements under the limitations below. The test suite runs it, and so does the offline
examples lane of CI.

## Interpretation and limitations

### Missing data, frozen assets, and feasibility

- **Filtering and variance units.** The ordinary wrapper removes non-positive/missing budgets
  and assets with a zero, negative or NaN covariance diagonal. Smaller positive diagonals are
  raised to `0.001**2`; off-diagonals are unchanged. Infinite values and invalid off-diagonals
  are not repaired by that filtering rule. Supply a valid finite covariance.
- **Scale qualification.** Multiplying every covariance entry by the same positive constant
  leaves the mathematical allocation unchanged. The wrapper's absolute variance floor uses
  caller units, so changing units can activate the floor and change its output. The examples
  stay above the floor.
- **Reduced-universe bounds.** By default, `apply_total_to_good_ratio=True` can scale per-name
  maxima when eligible names are lost to covariance filtering. Maxima close to one are retained.
  The surviving budgets are normalized. Inspect the aligned mandate rather than assuming all
  original limits were carried unchanged.
- **Frozen holdings.** In the ordinary wrapper path with previous weights and binary indicators,
  a zero indicator saves the pre-trade weight and removes that asset from the free solve.
  Tradable weights are solved on the reduced covariance, then scaled to the capital left after
  frozen holdings. Cross-covariances with frozen assets do not enter that reduced solve, although
  they affect the risk of the final portfolio. Recompute final risk shares using qis and audit
  final capital bounds: restoration/scaling need not preserve exact budgets or every free-solve
  group/box limit. Supply a valid long-only frozen book with total frozen weight at most one.
- **No remaining free assets.** If filtering leaves an empty solve universe, the current wrapper
  warns and returns an all-zero Series before restoring frozen positions. An all-frozen book
  therefore does not pass through as unchanged holdings. Treat that result explicitly.
- **Single-asset rolling shortcut.** A static budget Series containing one asset returns 100%
  in that asset at each covariance date before the ordinary solver/filtering path. This shortcut
  is not evidence that its budget, covariance, bounds or freezing controls were checked.
- **Acceptance and diagnostics.** The internal solver checks convergence and the joint fully
  invested feasible set. The public wrapper can return a fallback after solver rejection;
  a nonempty result is not proof of successful optimization. Its return is a Series or
  diagnostic DataFrame, not an `OptimizationOutcome`. The separate SciPy entry point is not an
  automatic retry. Check logs, achieved contributions and final constraints.
- **Detailed output.** `detailed_output=True` uses the cleaned solve covariance for contribution
  diagnostics. For a portfolio containing restored frozen holdings, compute full-portfolio
  shares explicitly from the complete covariance rather than treating reduced diagnostics as
  a full risk decomposition.

> **Pitfall.** An infeasible mandate does not raise. With a 20% cap on each of the three example
> assets the caps sum to 60%, the solver fails, and `wrapper_risk_budgeting` logs a warning and
> returns all-zero weights, or `weights_0` when one is supplied. Check that the weights sum to
> one and recompute the risk shares before using a result.

Budgets describe the supplied risk model, not future realized contributions. Binding constraints,
estimation error and changes in correlation can all create a gap between target and achieved risk.

## See also

- [HRP and cluster risk budgets](hierarchical_risk_parity_and_cluster_budgets.md)
- [Portfolio constraints](constraints.md)
- [Rolling backtests](rolling_backtests.md)
- [Covariance estimators](covariance_estimators.md)
- [API reference](api.rst)
- [Risk-budgeting example on market data](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/solvers/risk_budgeting.py)
  (needs a network connection)
- [Offline solver comparison](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/comparisons/risk_budgeting_ccd_vs_scipy.py)
- [Rendered risk-budgeting guide](https://optimalportfolios.readthedocs.io/en/latest/risk_budgeting.html)

## References

- Tasche, D. (2007; revised 2008).
  [Capital Allocation to Business Units and Sub-Portfolios: the Euler Principle](https://arxiv.org/abs/0708.2542).
  arXiv:0708.2542.
- Richard, J.-C., and Roncalli, T. (2019).
  [Constrained Risk Budgeting Portfolios: Theory, Algorithms, Applications & Puzzles](https://arxiv.org/abs/1902.05710).
  arXiv:1902.05710. The package contract above distinguishes its normalized formulation.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
