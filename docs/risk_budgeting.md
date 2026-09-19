---
myst:
  html_meta:
    description: >-
      Risk budgeting in OptimalPortfolios: Euler risk contributions, normalized budgets,
      constrained allocation, group budgets, hierarchical risk parity, and offline examples.
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
This article covers the volatility-based risk-budgeting solver, group-to-asset budget design,
and the separate hierarchical risk parity allocation function.

OptimalPortfolios owns portfolio construction. [qis](https://github.com/ArturSepp/QuantInvestStrats)
provides the risk-contribution analytics used below and the holdings simulation/reporting layer;
see the [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

<a id="inputs-and-conventions"></a>

## Inputs, notation, and assumptions

| Symbol | Meaning and units |
|---|---|
| $w$ | $n$-asset capital-weight vector; fractions of NAV |
| $\Sigma$ | $n$-by-$n$ covariance matrix in caller-supplied variance units |
| $\sigma_p$ | Portfolio volatility in the square-root units of $\Sigma$ |
| $\mathrm{RC}_i$ | Asset $i$'s Euler contribution to portfolio volatility |
| $r_i$ | Asset $i$'s dimensionless fraction of portfolio volatility |
| $b_i$ | Normalized target risk share; positive on the eligible free universe |
| $G_g,n_g$ | Classified asset set for group $g$ and its member count |
| $B_g,\alpha$ | Aggregate group risk budget and group-size exponent |

Supply a symmetric covariance DataFrame with unique, identically ordered asset labels on both
axes. The model assumes a valid positive-semidefinite covariance and positive portfolio variance;
a positive-definite matrix with positive asset variances avoids degenerate risk-budget problems.
Covariance, budgets, group labels and tradability information must be available at the decision
date. These functions do not estimate covariance or convert its frequency.

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

The implementation uses a homogeneous auxiliary-variable formulation with cyclical coordinate
descent (CCD) and the alternating direction method of multipliers (ADMM) for constrained
problems. Instrument and group bounds apply to the
normalized portfolio together with full investment. It does not clip an unconstrained solution
and then renormalize it. Exact target shares are recovered when no additional bound binds;
otherwise budgets express preferences in the constrained objective.

[Richard and Roncalli (2019)](https://arxiv.org/abs/1902.05710) discuss constrained risk budgeting
and the scaling-compatibility problem. The formula above states this package's current
normalized-weight contract. The paper's constrained absolute-bound table values are not
numerical references for this homogeneous formulation.

### Group risk budgets

`compute_group_risk_budgets` converts a complete or partially classified partition into
asset-level targets. For the nonempty groups at the current observation:

$$
B_g=\frac{n_g^\alpha}{\sum_h n_h^\alpha},
\qquad
b_i=\frac{B_g}{n_g}\quad\text{for }i\in G_g.
$$

| `group_size_exponent` | Allocation of target risk |
|---|---|
| `0` | Equal aggregate budget for each available group |
| `1` | Equal budget for each classified asset |
| `0.5` | Aggregate group budget proportional to the square root of group size |

Each asset has one group label. Missing labels receive zero budget; an observation with no
classified assets raises `ValueError`. A membership DataFrame is transformed row by row,
without using future classifications. Labels can represent statistical clusters, sectors or
asset classes. This is a budget-design convention; matching these targets still depends on
the covariance, constraints and tradable universe.

### Hierarchical risk parity

`compute_hierarchical_risk_parity_weights` implements the recursive-bisection allocation
described by [López de Prado (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678).
It orders assets using a supplied SciPy linkage and splits ordered blocks recursively. Each
split allocates capital inversely to the two block variances, measured using within-block
inverse-variance portfolios.

Tree estimation remains outside OptimalPortfolios. [factorlasso](https://github.com/ArturSepp/FactorLasso)
can construct the linkage; this function consumes it and the labelled covariance matrix.
See the [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).

HRP does not take a `Constraints` object or a target risk-budget vector. Its weights need not
equal the ERC or prescribed-budget solution. Use
`qis.compute_group_portfolio_risk_contribution_ratios` to aggregate the resulting portfolio's
Euler risk shares over groups.

## Worked example

### Minimal offline example

This fixed synthetic example retains the original three-asset covariance and target budgets.
The covariance is annualized. The 80% per-asset cap is slack at the solution.

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

### Independent diagonal-covariance check

For uncorrelated assets with volatilities $\sigma_i>0$, positive budgets, no binding bounds
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

The repository test checks this closed-form result independently of the solver. It also compares
the original correlated example with the existing independent conic reference used by the
solver tests. Risk-contribution reporting continues to use qis.

### Partially classified groups

With one Growth asset and two Defensive assets, equal group budgets assign `0.50` to Growth
and `0.25` to each Defensive member. An unclassified asset receives zero:

```python
memberships = pd.Series({
    "Equity": "Growth",
    "Bonds": "Defensive",
    "Diversifier": "Defensive",
    "Unclassified": None,
})
group_budgets = opt.compute_group_risk_budgets(
    groups=memberships, group_size_exponent=0.0,
)
print(group_budgets.tolist())  # [0.5, 0.25, 0.25, 0.0]
```

## Implementation in optimalportfolios

### Single-date versus rolling use

| Entry point | Input and output contract |
|---|---|
| `wrapper_risk_budgeting` | One covariance DataFrame; optional asset-indexed Series/dict budgets; returns a weight Series by default |
| `rolling_risk_budgeting` | Price panel, covariance-date dictionary, and static Series, date-by-asset DataFrame or `None` budgets; returns dated target weights |
| `compute_group_risk_budgets` | Group Series or date-by-asset membership DataFrame; returns budgets with matching labels/shape |
| `compute_hierarchical_risk_parity_weights` | Covariance DataFrame and linkage array; returns long-only, fully invested weights |

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
and [API reference](api.rst) describe the public entry points. The example blocks in this article
are canonical and executed in document order by:

```console
python -m pytest src/optimalportfolios/tests/risk_budgeting_documentation_test.py -q
```

Use the repository's prescribed external Python and C-local setup for contributor checks.
Examples were verified on 2026-09-13 against the working source declaring OptimalPortfolios
7.6.0, with qis 5.26.0. Rounded weights are illustrative and may vary slightly with numerical
dependencies. This does not certify a published 7.6.0 artifact.

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

Budgets describe the supplied risk model, not future realized contributions. Binding constraints,
estimation error and changes in correlation can all create a gap between target and achieved risk.

## See also

- [Portfolio constraints](constraints.md)
- [Rolling backtests](rolling_backtests.md)
- [Covariance estimators](covariance_estimators.md)
- [API reference](api.rst)
- [Canonical risk-budgeting example](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/solvers/risk_budgeting.py)
  (network-data example)
- [Offline solver comparison](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/comparisons/risk_budgeting_ccd_vs_scipy.py)
- [Rendered risk-budgeting guide](https://optimalportfolios.readthedocs.io/en/latest/risk_budgeting.html)

## References

- Tasche, D. (2007; revised 2008).
  [Capital Allocation to Business Units and Sub-Portfolios: the Euler Principle](https://arxiv.org/abs/0708.2542).
  arXiv:0708.2542.
- Richard, J.-C., and Roncalli, T. (2019).
  [Constrained Risk Budgeting Portfolios: Theory, Algorithms, Applications & Puzzles](https://arxiv.org/abs/1902.05710).
  arXiv:1902.05710. The package contract above distinguishes its normalized formulation.
- López de Prado, M. (2016).
  [Building Diversified Portfolios that Outperform Out-of-Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678).
  The Journal of Portfolio Management. DOI: 10.3905/jpm.2016.42.4.059.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
