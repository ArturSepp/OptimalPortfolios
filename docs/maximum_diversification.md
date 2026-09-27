---
myst:
  html_meta:
    description: >-
      Maximum diversification portfolio in Python with optimalportfolios: the diversification
      ratio, why the solution is the minimum-variance portfolio of the correlation matrix
      rescaled by inverse volatility, the equal-correlation property, constraints and the SLSQP
      solver, with a verified offline example.
---

# Maximum diversification

*Author: [Artur Sepp](https://github.com/ArturSepp)*

The maximum diversification portfolio is implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

The maximum diversification portfolio chooses the weights that maximise the diversification
ratio: the weighted average volatility of the assets divided by the volatility of the portfolio.
It uses only the covariance matrix, so it needs no expected returns, and it is the default
objective of the rolling dispatcher.

This page proves two results that explain what the portfolio holds. Under long-only full
investment, it is the minimum-variance portfolio of the correlation matrix, rescaled by inverse
volatility, so volatilities matter only through that rescaling. And every asset it holds has the
same correlation with the portfolio, equal to the inverse of the maximal ratio, while every asset
it excludes is more correlated than that. The worked example checks both results against an
independent computation.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Log returns in the EWMA covariance of the rolling example; any basis otherwise |
| Estimation grid | Weekly returns (`'W-WED'`) with span 52 in the rolling example |
| Rebalancing grid | Quarter ends in the rolling example; the keys of `covar_dict` in general |
| Covariance units | Any consistent units: the ratio does not depend on the scale of the covariance |
| Expected returns | None |
| Weight state | Target weights |
| Solver | SciPy SLSQP; CVXPY only for the independent reference in the example |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $\sigma$ | Vector of asset volatilities $\sigma_i = \sqrt{\Sigma_{ii}}$ |
| $C$ | Correlation matrix, $\Sigma = \mathrm{diag}(\sigma) \, C \, \mathrm{diag}(\sigma)$ |
| $\mathrm{DR}(w)$ | Diversification ratio of the weights $w$ |
| $y$ | Volatility-weighted shares, $y_i = w_i \sigma_i / \sigma^{\top} w$ |
| $\rho_i(w)$ | Correlation of asset $i$ with the portfolio $w$ |

The results below assume long-only, fully invested weights and no other constraint. The last
subsection of the methodology states what changes when constraints are added. Every asset has a
positive volatility; assets with a zero or missing variance are removed before the solve, as
described on the [conventions page](conventions.md#missing-data-and-eligibility).

## Methodology

**Definition.** The diversification ratio of weights $w$ is

$$
\mathrm{DR}(w) = \frac{\sigma^{\top} w}{\sigma(w)} = \frac{\sum_i w_i \sigma_i}{\sqrt{w^{\top} \Sigma w}} .
$$

It does not change when $w$ is multiplied by a positive number, so the budget
$\sum_i w_i = 1$ only fixes the scale. For long-only weights, $\mathrm{DR}(w) \geq 1$, with
equality when all held assets are perfectly correlated; the ratio grows as correlations fall.

**Proof.** For $w \geq 0$, the triangle inequality for the norm induced by $\Sigma$ gives
$\sigma(w) \leq \sum_i w_i \sigma_i$, with equality exactly when the held assets are perfectly
positively correlated. $\square$

The maximum diversification portfolio solves

$$
\max_{w} \mathrm{DR}(w) \quad \text{subject to} \quad \sum_i w_i = 1, \quad w \geq 0 .
$$

**Proposition 1 (correlation form).** With $C$ the correlation matrix and
$y_i = w_i \sigma_i / \sigma^{\top} w$,

$$
\mathrm{DR}(w) = \frac{1}{\sqrt{y^{\top} C y}} .
$$

Hence the maximum diversification portfolio is $w_i \propto y^{\star}_i / \sigma_i$, where
$y^{\star}$ is the minimum-variance portfolio of $C$ on the simplex
$\lbrace y \geq 0, \sum_i y_i = 1 \rbrace$.

**Proof.** Write $\Sigma_{ij} = \sigma_i \sigma_j C_{ij}$. Then
$w^{\top} \Sigma w = \sum_{i,j} (w_i \sigma_i)(w_j \sigma_j) C_{ij} = (\sigma^{\top} w)^2 \, y^{\top} C y$,
so $\mathrm{DR}(w) = 1 / \sqrt{y^{\top} C y}$. The map from long-only $w$ to $y$ is onto the
simplex and inverted, up to scale, by $w_i \propto y_i / \sigma_i$. Maximising the ratio is
therefore minimising the convex quadratic $y^{\top} C y$ on the simplex. $\square$

> **Insight.** Which assets are held depends only on the correlation matrix. The volatilities
> rescale the weights of the held assets but do not change which assets are held.

**Proposition 2 (equal correlations).** Let $w^{\star}$ be the maximum diversification
portfolio and $\mathrm{DR}^{\star} = \mathrm{DR}(w^{\star})$. Every held asset has the same
correlation with the portfolio, and every excluded asset has at least that correlation:

$$
\rho_i(w^{\star}) = \frac{1}{\mathrm{DR}^{\star}} \quad \text{if } w^{\star}_i > 0,
\qquad
\rho_i(w^{\star}) \geq \frac{1}{\mathrm{DR}^{\star}} \quad \text{if } w^{\star}_i = 0 .
$$

**Proof.** Because the ratio is scale-invariant, the budget constraint does not bind at the
optimum, and the first-order conditions for $\max \mathrm{DR}(w)$ over $w \geq 0$ are
$\partial \mathrm{DR} / \partial w_i \leq 0$, with equality where $w_i > 0$. The derivative is

$$
\frac{\partial \mathrm{DR}}{\partial w_i} = \frac{\sigma_i}{\sigma(w)} - \frac{(\sigma^{\top} w) (\Sigma w)_i}{\sigma(w)^3} .
$$

It is non-positive exactly when
$(\Sigma w)_i / (\sigma_i \sigma(w)) \geq \sigma(w) / \sigma^{\top} w$. The left side is
$\rho_i(w)$ and the right side is $1 / \mathrm{DR}(w)$. $\square$

**Two corollaries.** With two assets of correlation $\rho$, the correlation matrix is symmetric
in the two assets, so $y^{\star} = (1/2, 1/2)$: the portfolio holds the assets in inverse
proportion to their volatilities, and $\mathrm{DR}^{\star} = \sqrt{2 / (1 + \rho)}$. The same
argument gives inverse-volatility weights for $N$ assets with a common correlation $\rho$, and
$\mathrm{DR}^{\star} = \sqrt{N / (1 + (N - 1) \rho)}$.

**With further constraints.** Caps, group limits, a tracking-error or turnover limit add
multipliers to the first-order conditions. Proposition 1 no longer reduces the problem to the
correlation matrix, and Proposition 2 holds only among the assets whose constraints do not bind.
The package then solves the ratio directly under all supported constraints.

## Worked example

The canonical script of this page,
[`examples/docs/maximum_diversification.py`](../examples/docs/maximum_diversification.py),
runs offline and asserts every number quoted here:

```console
python -m examples.docs.maximum_diversification
```

With two assets of volatility 5% and 20% and correlation 0.5, the solver returns the
inverse-volatility weights 80% and 20%, and the ratio $\sqrt{2/1.5} \approx 1.155$:

```python
two = covariance(np.array([0.05, 0.20]), np.array([[1.0, 0.5], [0.5, 1.0]]), ['A', 'B'])
weights = op.wrapper_maximise_diversification(
    pd_covar=two, constraints=op.Constraints(is_long_only=True))
inverse_vol = np.array([1 / 0.05, 1 / 0.20])
assert np.allclose(weights, inverse_vol / inverse_vol.sum(), atol=1e-4)
ratio = op.calculate_diversification_ratio(w=weights.to_numpy(), covar=two.to_numpy())
assert abs(ratio - np.sqrt(2.0 / (1.0 + 0.5))) < 1e-6
```

The six-asset universe of the script has government bonds, credit, three equity markets and
commodities, with volatilities from 5% to 24%. Credit is correlated 0.5 with government bonds and
0.4 to 0.45 with the equity markets. The SLSQP solution of the package agrees, within
$10^{-4}$, with the minimum-variance portfolio of the correlation matrix computed by CVXPY and
rescaled by inverse volatility, as Proposition 1 predicts. It holds government bonds at 72% and
does not hold credit:

```python
covar = covariance(VOLS, CORR, TICKERS)
mdp = op.wrapper_maximise_diversification(
    pd_covar=covar, constraints=op.Constraints(is_long_only=True))
reference = correlation_minimum_variance(CORR, VOLS)
assert np.abs(mdp.to_numpy() - reference).max() < 1e-4
assert round(mdp['Govt'], 2) == 0.72 and mdp['Credit'] < HELD
```

The maximal ratio is 1.75. Each of the five held assets has correlation $1/1.75 = 0.570$ with
the portfolio, and credit, which is excluded, has 0.70, as Proposition 2 requires:

```python
ratio = op.calculate_diversification_ratio(w=mdp.to_numpy(), covar=covar.to_numpy())
rho = correlations_with_portfolio(covar.to_numpy(), mdp.to_numpy())
held = mdp.to_numpy() > HELD
assert np.allclose(rho[held], 1.0 / ratio, atol=1e-4)
assert (rho[~held] > 1.0 / ratio).all()
assert round(ratio, 2) == 1.75 and round(1.0 / ratio, 3) == 0.570
assert round(rho[TICKERS.index('Credit')], 2) == 0.70
```

![Left: the SLSQP weights and the CVXPY minimum-variance weights of the correlation matrix,
rescaled by inverse volatility, agree for all six assets, with government bonds at 72% and credit
not held. Right: the five held assets each have correlation 0.570 with the portfolio, on the
dashed 1/DR line, while excluded credit has 0.70.](images/max_diversification_identity.png)

*Figure: the two routes to the maximum diversification portfolio, and the correlation of each
asset with it, for the six-asset universe of the example. Drawn by the `exhibit` function of the
canonical script; the [analytics gallery](analytics_gallery.md) lists its provenance.*

A 30% cap on every asset binds on government bonds and brings credit into the portfolio. The
capped assets no longer share the correlation of the others, so the identity does not survive the
constraint:

```python
caps = pd.Series(0.30, index=TICKERS)
capped = op.wrapper_maximise_diversification(
    pd_covar=covar, constraints=op.Constraints(is_long_only=True, max_weights=caps))
assert abs(capped['Govt'] - 0.30) < 1e-4 and capped['Credit'] > 0.05
rho_capped = correlations_with_portfolio(covar.to_numpy(), capped.to_numpy())
held_capped = rho_capped[capped.to_numpy() > HELD]
assert held_capped.max() - held_capped.min() > 0.05
```

In a rolling allocation, each quarter-end EWMA estimate of simulated prices gives its own
portfolio. Proposition 2 holds at every date for the covariance used at that date:

```python
prices = simulated_prices(covar, seed=SEED)
covar_dict = op.EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE') \
    .fit_rolling_covars(prices=prices, time_period=qis.TimePeriod('2017-12-31', '2025-12-31'))
rolling = op.compute_rolling_optimal_weights(
    prices=prices, constraints=op.Constraints(is_long_only=True), covar_dict=covar_dict,
    portfolio_objective=op.PortfolioObjective.MAX_DIVERSIFICATION)
assert np.allclose(rolling.sum(axis=1), 1.0, atol=1e-6) and (rolling >= 0.0).all().all()
for date, estimate in covar_dict.items():
    w = rolling.loc[date].to_numpy()
    rho = correlations_with_portfolio(estimate.to_numpy(), w)
    dr = op.calculate_diversification_ratio(w=w, covar=estimate.to_numpy())
    assert np.allclose(rho[w > HELD], 1.0 / dr, atol=2e-3)
```

## Implementation in optimalportfolios

The objective has the three layers of every solver family, described in
[choosing an objective](optimization_module_readme.md):

- `opt_maximise_diversification(covar, constraints)` works on a NumPy covariance. It minimises
  $-\mathrm{DR}(w)$ with SciPy SLSQP from equal weights, with function tolerance $10^{-8}$ and at
  most 500 iterations, under the linear constraints and bounds that `Constraints` compiles for
  SciPy. SLSQP status 8, a line search that cannot improve, occurs next to the optimum when
  rounding swamps the finite-difference gradient; since 7.8.1.dev1 such a point is validated
  for feasibility like a converged one instead of being replaced by the fallback. Tiny negative
  weights of a long-only solution are clipped before validation, and a rejected solution falls
  back as described on the [conventions page](conventions.md#solvers-and-outcomes).
- `wrapper_maximise_diversification(pd_covar, constraints, weights_0=None, ...)` removes assets
  with a zero or missing variance, restricts the constraints to the remaining assets, solves, and
  returns weights on the original labels, with zeros for removed assets. Its
  `apply_total_to_good_ratio` defaults to `True`.
- `rolling_maximise_diversification(prices, constraints, covar_dict, ...)` solves at each key of
  `covar_dict`, passing the previous weights drifted to the date as `weights_0`, and returns a
  date-by-asset table of target weights.

The rolling dispatcher `compute_rolling_optimal_weights` routes
`PortfolioObjective.MAX_DIVERSIFICATION`, its default objective, to the rolling function.
`calculate_diversification_ratio(w, covar)` returns $\mathrm{DR}(w)$ for any weights. The
[constraints page](constraints.md) lists which limits the SciPy path supports.

> **Pitfall.** Maximum diversification is the default objective of
> `compute_rolling_optimal_weights`. A call that omits `portfolio_objective` returns this
> portfolio, not the minimum-variance portfolio.

## Interpretation and limitations

- The package takes the objective from Choueifaty and Coignard (2008). The covariance estimate,
  the rebalancing grid and any constraints are the caller's, not those of the paper's empirical
  studies.
- Low-volatility assets with low correlations to the rest receive large weights: government
  bonds hold 72% in the example. Caps are the usual remedy, at the price of the two propositions.
- Assets correlated with the rest of the universe more than $1/\mathrm{DR}^{\star}$ are excluded
  outright, which can surprise when a familiar asset class disappears. The equal-correlation
  property makes the reason visible.
- The weights inherit the estimation error of the correlation matrix. With an EWMA span of 52
  weekly returns they move from quarter to quarter; see the span comparison in the
  [analytics gallery](analytics_gallery.md) and the cost of those moves in
  [turnover and transaction costs](turnover_and_transaction_costs.md).
- The solver is a local method on a ratio. Under long-only full investment the problem is
  equivalent to the convex problem of Proposition 1, which the example uses as an independent
  check; with further constraints the package reports the outcome of the direct solve.
- Maximum diversification is not risk budgeting: the risk contributions of held assets are not
  equal in general. See [risk budgeting](risk_budgeting.md) for that objective.

## See also

- [Choosing an objective](optimization_module_readme.md)
- [Risk budgeting](risk_budgeting.md)
- [Portfolio constraints](constraints.md)
- [Conventions, notation and glossary](conventions.md)
- [Research papers and replication](research_papers.md)

## References

- Choueifaty, Y. and Coignard, Y. (2008). *Toward Maximum Diversification*. The Journal of
  Portfolio Management, 35(1), 40–51.
  [DOI 10.3905/jpm.2008.35.1.40](https://doi.org/10.3905/jpm.2008.35.1.40).
- Choueifaty, Y., Froidure, T. and Reynier, J. (2013). *Properties of the Most Diversified
  Portfolio*. Journal of Investment Strategies, 2(2), 49–70.
  [DOI 10.21314/JOIS.2013.033](https://doi.org/10.21314/JOIS.2013.033). Further properties of
  the ratio and of the portfolio, and invariance properties of portfolio construction.
- Sepp, A. (2023). *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*. Risk,
  October 2023. [SSRN 4217841](https://ssrn.com/abstract=4217841). Section 4.2 applies maximum
  diversification to portfolios with a cryptocurrency.
- Diamond, S. and Boyd, S. (2016). *CVXPY: A Python-Embedded Modeling Language for Convex
  Optimization*. Journal of Machine Learning Research, 17(83), 1–5.
  [JMLR](https://www.jmlr.org/papers/v17/15-408.html).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
