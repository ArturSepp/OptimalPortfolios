---
myst:
  html_meta:
    description: >-
      Minimum variance, quadratic utility and maximum Sharpe portfolios in Python with
      optimalportfolios: what each wrapper solves, the closed forms and the frontier they share,
      the Charnes–Cooper and SLSQP routes of maximum Sharpe, estimated means and their error, with
      a verified offline example.
---

# Minimum variance, quadratic utility and maximum Sharpe

*Author: [Artur Sepp](https://github.com/ArturSepp)*

The three mean-variance objectives are implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

Markowitz (1952) proposed choosing portfolios by their expected return and variance. The package
solves three objectives of this family. Minimum variance uses only the covariance matrix.
Quadratic utility trades expected return against variance at a risk aversion $\gamma$. Maximum
Sharpe maximises the ratio of expected excess return to volatility. `wrapper_quadratic_optimisation`
solves minimum variance by default and quadratic utility on request;
`wrapper_maximize_portfolio_sharpe` solves maximum Sharpe, through the Charnes–Cooper
transformation when the net exposure is fixed and through SciPy SLSQP when it may move within a
band.

This page proves that, when no bound binds, the three solutions are closed forms in
$\Sigma^{-1} \mathbf{1}$ and $\Sigma^{-1} \mu$ that lie on one frontier. Every fully invested
utility portfolio mixes the minimum-variance and the maximum-Sharpe portfolios, and the risk
aversion $\gamma = \mathbf{1}^{\top} \Sigma^{-1} \mu$ returns the maximum-Sharpe portfolio itself.
It states what the Charnes–Cooper route solves and which constraint rows it rescales, and how the
dispatcher estimates expected returns. The worked example checks each statement against an
independent computation.

Expected returns are the weak input. The literature finds that errors in means cost far more
than errors in variances and covariances (Chopra and Ziemba 1993), which is why the package's
default objectives use the covariance matrix only.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | None in the solvers, which take $\mu$ and $\Sigma$ as given; `estimate_rolling_ewma_means`, and so the dispatcher, uses log returns; the rolling functions drift the previous weights with simple price ratios |
| Estimation grid | None for supplied inputs. The dispatcher's means are an expanding EWMA of Wednesday-to-Wednesday log returns (`returns_freq='W-WED'`, `span=52`); the rolling example simulates business-daily prices from 2015 to 2024 |
| Rebalancing grid | One solve per call; the rolling functions solve at the keys of `covar_dict`, the four quarter ends of 2024 in the rolling example |
| Covariance units | The caller's units, never annualised; $\mu$ must share them and $\gamma$ is in their inverse. Annual decimals in the examples |
| Expected returns | Required by utility and maximum Sharpe, as `means` or `expected_returns`. The Sharpe ratio subtracts no cash rate, so supply excess returns; the dispatcher's means are annualised EWMA means of log returns |
| Weight state | Target weights; the rolling functions pass the previous targets, drifted to the date, as `weights_0`, which feeds turnover rows and the fallback |
| Solver | CVXPY with CLARABEL for minimum variance, utility and maximum Sharpe at a fixed exposure (Charnes–Cooper); SciPy SLSQP for maximum Sharpe with an exposure band |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $\mathbf{1}$ | Vector of ones |
| $A$, $B$, $C$ | The scalars $\mathbf{1}^{\top} \Sigma^{-1} \mathbf{1}$, $\mathbf{1}^{\top} \Sigma^{-1} \mu$ and $\mu^{\top} \Sigma^{-1} \mu$ |
| $\mathrm{SR}(w)$ | Sharpe ratio $\mu^{\top} w / \sigma(w)$ of the weights $w$ |
| $w^{\mathrm{mv}}$, $w^{\mathrm{tan}}$ | Minimum-variance and tangency (maximum-Sharpe) portfolios |
| $w(\gamma)$, $\theta$ | Fully invested utility portfolio at risk aversion $\gamma$, and the ratio $\theta = B / \gamma$ |
| $v(m)$ | Frontier variance: the smallest variance of a fully invested portfolio with expected return $m$ |
| $E$ | Fixed net exposure, when `min_exposure` equals `max_exposure` |
| $y$, $k$ | Charnes–Cooper variables: the scaled weights $y = k w$ and the scale $k$ |
| $\hat\mu_t$ | EWMA estimate of the mean return at date $t$ |

The covariance is positive definite and $\mu$ is not proportional to $\mathbf{1}$, so that
$A C - B^2 \gt 0$ by the Cauchy–Schwarz inequality. The closed forms describe fully invested
portfolios with no binding bound; the worked example chooses inputs for which every long-only
bound is slack. Weights are fractions of net asset value. Before each solve the wrappers remove
assets with a missing or non-positive variance or a non-finite mean, as described on the
[solver-outcomes page](solver_numerics_and_outcomes.md#filtering-the-universe).

## Methodology

### What each wrapper solves

With $\gamma$ passed as `carra`, `cvx_quadratic_optimisation` maximises one of

$$
-w^{\top} \Sigma w \quad \text{(minimum variance)}, \qquad \mu^{\top} w - \frac{\gamma}{2} w^{\top} \Sigma w \quad \text{(quadratic utility)},
$$

and `cvx_maximize_portfolio_sharpe` maximises $\mathrm{SR}(w) = \mu^{\top} w / \sigma(w)$, each
under the rows of `Constraints`.

- Minimum variance is `PortfolioObjective.MIN_VARIANCE`, the default `portfolio_objective` of
  `wrapper_quadratic_optimisation` and `rolling_quadratic_optimisation`; utility is
  `PortfolioObjective.QUADRATIC_UTILITY`. `cvx_quadratic_optimisation` takes the objective as its
  first argument and raises `ValueError` for any other member of `PortfolioObjective`, and for
  utility without `means`.
- The risk aversion `carra` defaults to 1.0 in the three quadratic functions. The dispatcher
  passes `carra=0.5`.
- The Sharpe ratio has no cash rate. Given total returns, the solver maximises total return over
  volatility, which selects the tangency portfolio of a zero cash rate.

**Units.** Multiplying $\mu$ and $\Sigma$ by the same $c \gt 0$, for example to annualise weekly
estimates, multiplies the utility by $c$ and the Sharpe ratio by $\sqrt{c}$; neither solution
changes and $\gamma$ keeps its value. Quoting returns in percent multiplies $\mu$ by 100 and
$\Sigma$ by $100^2$, so the same portfolio needs $\gamma / 100$. The Sharpe ratio is also
unchanged when the weights are multiplied by $c \gt 0$.

### Closed forms when no bound binds

Write $\langle x, z \rangle = x^{\top} \Sigma z$ for the inner product that the covariance
defines, so that $\sigma(w)^2 = \langle w, w \rangle$.

**Proposition 1 (minimum variance).** Every fully invested portfolio has
$\sigma(w)^2 \geq 1 / A$, with equality only at $w^{\mathrm{mv}} = \Sigma^{-1} \mathbf{1} / A$,
whose expected return is $B / A$.

**Proof.** Since $\mathbf{1}^{\top} w = \langle \Sigma^{-1} \mathbf{1}, w \rangle$ and
$\langle \Sigma^{-1} \mathbf{1}, \Sigma^{-1} \mathbf{1} \rangle = A$, the Cauchy–Schwarz
inequality gives $1 = \langle \Sigma^{-1} \mathbf{1}, w \rangle^2 \leq A \sigma(w)^2$. Equality
holds only when $w$ is a multiple of $\Sigma^{-1} \mathbf{1}$, and the budget fixes the multiple at
$1 / A$. $\square$

**Proposition 2 (maximum Sharpe).** Every $w$ with $\sigma(w) \gt 0$ has
$\mathrm{SR}(w) \leq \sqrt{C}$, with equality only when $w = c \Sigma^{-1} \mu$ for some
$c \gt 0$. If $B \gt 0$, the fully invested maximiser is the tangency portfolio
$w^{\mathrm{tan}} = \Sigma^{-1} \mu / B$, with expected return $C / B$.

**Proof.** By the Cauchy–Schwarz inequality,
$\mu^{\top} w = \langle \Sigma^{-1} \mu, w \rangle \leq \sqrt{C} \sigma(w)$, with equality only when
$w$ is a non-negative multiple of $\Sigma^{-1} \mu$. The budget fixes the multiple at $1 / B$,
which is positive when $B \gt 0$, and the expected return is
$\mu^{\top} \Sigma^{-1} \mu / B = C / B$. $\square$

**Proposition 3 (quadratic utility).** Without a budget, the utility is maximised at
$\Sigma^{-1} \mu / \gamma = \theta w^{\mathrm{tan}}$, a net exposure of $\theta$. Under the budget
$\mathbf{1}^{\top} w = 1$, it is maximised at

$$
w(\gamma) = \frac{1}{\gamma} \Sigma^{-1} (\mu - \nu \mathbf{1}), \qquad \nu = \frac{B - \gamma}{A}, \qquad w(\gamma) = (1 - \theta) w^{\mathrm{mv}} + \theta w^{\mathrm{tan}} .
$$

**Proof.** The utility is strictly concave. Without the budget its gradient
$\mu - \gamma \Sigma w$ vanishes at $\Sigma^{-1} \mu / \gamma$, and
$\Sigma^{-1} \mu = B w^{\mathrm{tan}}$. With the budget and its multiplier $\nu$, stationarity
reads $\mu - \gamma \Sigma w - \nu \mathbf{1} = 0$, so $w = \Sigma^{-1} (\mu - \nu \mathbf{1}) / \gamma$,
and the budget $(B - \nu A) / \gamma = 1$ gives $\nu$. Substituting
$\Sigma^{-1} \mathbf{1} = A w^{\mathrm{mv}}$ gives the mix. $\square$

`solve_analytic_log_opt` evaluates these closed forms, for a general budget
$a^{\top} w = a_0$ passed as `exposure_budget_eq=(a, a0)`:
$w = \Sigma^{-1} (\mu - \nu a) / \gamma$ with
$\nu = (a^{\top} \Sigma^{-1} \mu - \gamma a_0) / a^{\top} \Sigma^{-1} a$. Its name refers to
logarithmic utility: to second order, the expected logarithm of one plus a portfolio return is its
mean less half its second moment, which is the utility at $\gamma = 1$ when the squared mean is
negligible.

**Proposition 4 (one frontier).** The smallest variance of a fully invested portfolio with
expected return $m$ is

$$
v(m) = \frac{A m^2 - 2 B m + C}{A C - B^2} ,
$$

attained by exactly one portfolio, a combination of $\Sigma^{-1} \mathbf{1}$ and
$\Sigma^{-1} \mu$. Minimum variance, the tangency portfolio and every $w(\gamma)$ are such
combinations, so each lies on the frontier: its variance is $v$ at its own expected return.

**Proof.** The problem of minimising $w^{\top} \Sigma w$ subject to $\mathbf{1}^{\top} w = 1$ and
$\mu^{\top} w = m$ is strictly convex. Stationarity makes $\Sigma w$ a combination of
$\mathbf{1}$ and $\mu$, so $w = \eta \Sigma^{-1} \mathbf{1} + \zeta \Sigma^{-1} \mu$, and the two
constraints read $\eta A + \zeta B = 1$ and $\eta B + \zeta C = m$. Hence
$\eta = (C - B m) / (A C - B^2)$, $\zeta = (A m - B) / (A C - B^2)$ and
$w^{\top} \Sigma w = \eta \mathbf{1}^{\top} w + \zeta \mu^{\top} w = \eta + \zeta m = v(m)$.
Conversely, a fully invested combination of the two vectors with expected return $m$ satisfies
the stationarity condition and both constraints, so it is the unique minimiser. Propositions 1 to
3 give such combinations. $\square$

The frontier in volatility and expected return is the upper branch of this curve above
$w^{\mathrm{mv}}$. By Proposition 2, the line from the origin with slope $\sqrt{C}$ touches it at
$w^{\mathrm{tan}}$ and nowhere lies below it.

> **Insight.** The three objectives are one family. With a full-investment budget, the utility
> portfolio is $w(\gamma) = (1 - \theta) w^{\mathrm{mv}} + \theta w^{\mathrm{tan}}$ with
> $\theta = B / \gamma$. Minimum variance is the limit of infinite risk aversion, maximum Sharpe
> is the risk aversion $\gamma = B$, 6.96 in the example, and a lower $\gamma$ moves past the
> tangency portfolio along the frontier.

### Maximum Sharpe at a fixed exposure: the Charnes–Cooper route

When `min_exposure` equals `max_exposure`, with value $E$, `cvx_maximize_portfolio_sharpe`
solves, in the variables $y$ and $k$,

$$
\min_{y, k} y^{\top} \Sigma y \quad \text{subject to} \quad \mu^{\top} y = E, \quad \mathbf{1}^{\top} y = k E, \quad k \geq 0, \quad \text{and the rows of the constraints},
$$

and returns $w = y / k$. With the default `factorize_covar=True`, the variance uses the factorised
covariance of the
[solver-outcomes page](solver_numerics_and_outcomes.md#how-a-solve-uses-the-factor). The rows
are compiled with `exposure_scaler=k`, which multiplies the per-asset bounds `min_weights` and
`max_weights` and the group-allocation bounds by $k$; a long-only row $y \geq 0$ needs no scale.

**Proposition 5 (the transformation).** Let $E \gt 0$ and let the constraints contain only
long-only, per-asset and group-allocation bounds. Then the pairs with $k \gt 0$ correspond one to
one to the feasible portfolios with a positive expected return, and minimising $y^{\top} \Sigma y$
maximises their Sharpe ratio.

**Proof.** For a feasible $w$ with $\mu^{\top} w \gt 0$, put $k = E / \mu^{\top} w$ and
$y = k w$. Then $\mu^{\top} y = E$, $\mathbf{1}^{\top} y = k E$, each rescaled row is a row of $w$
multiplied by $k \gt 0$, and $y^{\top} \Sigma y = E^2 / \mathrm{SR}(w)^2$. Conversely, a feasible
pair with $k \gt 0$ gives $w = y / k$, which meets every row, has net exposure $E$,
$\mu^{\top} w = E / k \gt 0$ and $\mathrm{SR}(w)^2 = E^2 / y^{\top} \Sigma y$. $\square$

The code imposes $k \geq 0$ rather than $k \gt 0$. A long-only book excludes $k = 0$: then
$y \geq 0$ and $\mathbf{1}^{\top} y = 0$ force $y = 0$, which contradicts $\mu^{\top} y = E$. When
no long-only portfolio has a positive expected return, the program is infeasible and the outcome
is rejected, as in the worked example with negated means. A long-short book is discussed under
the limitations below.

Charnes and Cooper (1962) introduced this change of variables for ratios of linear functions on
a polyhedron, fixing the denominator at one to obtain a linear program. The implementation
applies it to a ratio whose denominator is a volatility: it fixes the numerator at $E$ and
minimises the squared denominator, a convex quadratic program. It rescales only the exposure,
per-asset and group-allocation rows; every other row of `Constraints` is compiled on $y$, not on
$w$, as the pitfall below shows.

### Maximum Sharpe with an exposure band: the SLSQP route

When `min_exposure` differs from `max_exposure`, `cvx_maximize_portfolio_sharpe` minimises
$-\mathrm{SR}(w)$ with SciPy SLSQP from equal weights $1/N$, with function tolerance $10^{-10}$
and at most 500 iterations. It compiles the SciPy rows of `Constraints`: the exposure band as
two inequalities, a long-only inequality when `is_long_only` holds and `min_weights` is `None`,
per-asset bounds, from 0 to 1 for a long-only book without explicit bounds, and group-allocation
bounds. It compiles no return, volatility, tracking-error, turnover, deviation or beta row; see
the [capability matrix](constraints.md#backend-capability-matrix). The objective is zero where
the volatility is below $10^{-12}$, and the raw covariance is used without factorisation. A
successful SLSQP run is reported as `optimal`, any other as `solver_error`, and the result is
validated like every outcome
([acceptance and fallback](solver_numerics_and_outcomes.md#acceptance-and-fallback)).

Because the ratio does not change when the weights are scaled, the band does not decide the
exposure: every point of the optimal ray inside the band is optimal, and the route returns
whichever point its search reaches. SLSQP is a local method; the worked example checks the
direction it finds against Proposition 2.

### Estimated expected returns

`estimate_rolling_ewma_means` computes log returns on the grid `returns_freq`, dropping the
first row, and runs the EWMA recursion from the first return,

$$
\hat\mu_1 = r_1, \qquad \hat\mu_t = \lambda \hat\mu_{t-1} + (1 - \lambda) r_t, \qquad \lambda = 1 - \frac{2}{s + 1} .
$$

With `annualize=True` it multiplies by the annualisation factor $\mathrm{af}$ inferred from the
return dates, 52 for weekly returns. It reads the estimate at each rebalancing date, or at the
last return date before it, so an estimate uses no later return.

**Standard error.** For independent returns with variance $\sigma_i^2 / \mathrm{af}$ per period
and a long history, the weights $(1 - \lambda) \lambda^j$ of the recursion have squares that sum
to $(1 - \lambda) / (1 + \lambda) = 1 / s$. The annualised estimate therefore has the standard
error $\mathrm{af} \sigma_i / \sqrt{\mathrm{af} s} = \sigma_i \sqrt{\mathrm{af} / s}$. With the
dispatcher's weekly returns and span 52, $\mathrm{af} = s$, and the standard error of an
estimated mean equals the asset's volatility: 15% a year for an asset whose mean is 4.5%.

The literature finds this input decisive. Chopra and Ziemba (1993) measured the
certainty-equivalent loss from perturbing each input of a mean-variance optimisation of ten Dow
Jones stocks: at a risk tolerance of 50, errors in means cost about eleven times as much as errors
in variances, which cost about twice as much as errors in covariances, and the weight of errors
in means grows with risk tolerance. DeMiguel, Garlappi and Uppal (2009) found that none of 14
sample-based mean-variance models beat equal weights consistently out of sample across seven
datasets, and estimated that about 3000 months of data would be needed for 25 assets. These are
the papers' findings, not results of this page. They are the reason the package's defaults are
risk-based: the dispatcher's default objective is
[maximum diversification](maximum_diversification.md), and the quadratic wrapper's is minimum
variance.

## Worked example

The canonical script of this page,
[`examples/docs/mean_variance_objectives.py`](../examples/docs/mean_variance_objectives.py),
runs offline and asserts every number and property quoted here against an independent
computation. Two steps deliberately log a rejected solve:

```console
python -m examples.docs.mean_variance_objectives
```

The universe has government bonds, credit, US and emerging-market equities and gold, with annual
expected excess returns from 0.5% to 5.5%, volatilities from 5% to 19% and correlations from −0.2
to 0.5 (the constants `MEANS`, `VOLS` and `CORR`). The references are two linear solves and the
frontier formula of Proposition 4:

```python
def closed_forms(covar: np.ndarray, means: np.ndarray) -> tuple:
    """Return A, B, C and the minimum-variance and tangency portfolios by linear solves."""
    ones = np.ones(len(means))
    inv_ones = np.linalg.solve(covar, ones)
    inv_means = np.linalg.solve(covar, means)
    a, b, c = ones @ inv_ones, ones @ inv_means, means @ inv_means
    return a, b, c, inv_ones / a, inv_means / b


def frontier_variance(m: float, a: float, b: float, c: float) -> float:
    """Minimum variance of a fully invested portfolio with expected return m."""
    return (a * m * m - 2.0 * b * m + c) / (a * c - b * b)
```

```python
covar = covariance(VOLS, CORR, TICKERS)
means = pd.Series(MEANS, index=TICKERS)
sigma, mu = covar.to_numpy(), means.to_numpy()
config = op.OptimiserConfig(apply_total_to_good_ratio=False)
long_only = op.Constraints(is_long_only=True)
A, B, C, w_mv, w_tan = closed_forms(sigma, mu)
print(round(A, 1), round(B, 2), round(C, 4), round(np.sqrt(C), 3))
```

It prints `559.8 6.96 0.1729 0.416`: the largest Sharpe ratio is 0.416. The minimum-variance
portfolio has a volatility of 4.23%, an expected excess return of 1.24% and a ratio of 0.294; the
tangency portfolio has 5.97%, 2.48% and 0.416. Every weight of both is positive, so the long-only
bounds do not bind. The script also checks Propositions 1 and 2 on 4000 random fully invested
portfolios, half long-only and half long-short: none has a lower variance or a higher ratio.

The quadratic wrapper solves minimum variance when no objective is named:

```python
min_var, min_var_outcome = op.wrapper_quadratic_optimisation(
    pd_covar=covar, constraints=long_only, optimiser_config=config)
assert min_var_outcome.accepted and np.abs(min_var.to_numpy() - w_mv).max() < 1e-4
print(min_var.round(3).tolist())
```

It prints `[0.711, 0.13, 0.104, 0.008, 0.048]`, the closed form within $10^{-4}$, and its
variance matches $1 / A$ to a relative $10^{-6}$.

Maximum Sharpe runs through both routes: the default fixed exposure of one, and a band from 80%
to 100%:

```python
fixed, fixed_outcome = op.wrapper_maximize_portfolio_sharpe(
    pd_covar=covar, means=means, constraints=long_only, optimiser_config=config)
band = op.Constraints(is_long_only=True, min_exposure=0.8, max_exposure=1.0)
banded, banded_outcome = op.wrapper_maximize_portfolio_sharpe(
    pd_covar=covar, means=means, constraints=band, optimiser_config=config)
print(fixed_outcome.solver, banded_outcome.solver)
print(fixed.round(3).tolist())
assert np.abs(fixed.to_numpy() - w_tan).max() < 1e-6
assert np.abs(banded.to_numpy() / banded.sum() - w_tan).max() < 1e-4
```

It prints `CLARABEL SLSQP` and `[0.285, 0.304, 0.199, 0.102, 0.11]`. The Charnes–Cooper route
returns the tangency portfolio within $10^{-6}$. The SLSQP route returns the same direction within
$10^{-4}$ at an exposure inside the band, with the ratio 0.416. A fixed exposure of 0.5 keeps the
Charnes–Cooper route and returns half the tangency portfolio, and weekly means and covariance
return the same tangency portfolio as annual ones.

The utility grid solves the risk aversions 5, 10, 20 and 50, and then $\gamma = B$:

```python
utility_weights = {}
for gamma in GAMMAS:
    utility, outcome = op.wrapper_quadratic_optimisation(
        pd_covar=covar, constraints=long_only,
        portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
        means=means, carra=gamma, optimiser_config=config)
    w, theta = utility.to_numpy(), B / gamma
    assert np.abs(w - ((1.0 - theta) * w_mv + theta * w_tan)).max() < 1e-5
    assert abs(w @ sigma @ w / frontier_variance(mu @ w, A, B, C) - 1.0) < 1e-6
    assert abs(frontier_by_cvxpy(sigma, mu, mu @ w) / (w @ sigma @ w) - 1.0) < 1e-6
    utility_weights[f'gamma {gamma:g}'] = utility
grid = pd.DataFrame(utility_weights)
at_b, _ = op.wrapper_quadratic_optimisation(
    pd_covar=covar, constraints=long_only,
    portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
    means=means, carra=B, optimiser_config=config)
assert np.abs(at_b.to_numpy() - fixed.to_numpy()).max() < 1e-6
```

Each utility portfolio is the mix of Proposition 3 within $10^{-5}$. Its variance equals the
frontier variance at its expected return to a relative $10^{-6}$, both from the formula and from
a separate CVXPY solve of the frontier problem, `frontier_by_cvxpy`. At $\gamma = B$ the utility
solver returns the tangency portfolio within $10^{-6}$. At $\gamma = 5$, $\theta = 1.39$ and the
portfolio lies past the tangency: it holds `[0.117, 0.373, 0.236, 0.14, 0.134]`, with a
volatility of 7.24% and an expected excess return of 2.97%. The script also solves it with weekly
means and covariance at the same $\gamma$, and in percent units at $\gamma / 100$, and gets the
same weights.

![Left: expected excess return against volatility. The frontier rises from the minimum-variance
portfolio at 4.2% volatility and 1.24% return; the utility portfolios for risk aversions 50, 20,
10 and 5 lie on it at increasing volatility, and the maximum-Sharpe portfolio at 6.0% volatility
and 2.48% return is where the dotted line from the origin with slope 0.416 touches it; below the
minimum-variance portfolio the inefficient branch is dashed. Right: weights of the three
portfolios; government bonds fall from 71% for minimum variance to 28% for maximum Sharpe and 12%
for utility at risk aversion 5, while credit and the equities
rise.](images/efficient_frontier_objectives.png)

*Figure: where minimum variance, quadratic utility and maximum Sharpe sit on the frontier of the
five-asset example, and what they hold. Drawn by the `exhibit` function of the canonical script;
the [analytics gallery](analytics_gallery.md) lists its provenance.*

`solve_analytic_log_opt` gives the same portfolio in closed form; `shown` is the utility
portfolio of the grid at `SHOWN_GAMMA`, 5. Without the budget it returns
$\theta w^{\mathrm{tan}}$, and the solver agrees when the exposure band is wide:

```python
ones = np.ones(len(TICKERS))
budgeted = op.solve_analytic_log_opt(sigma, mu, exposure_budget_eq=(ones, 1.0),
                                     gamma=SHOWN_GAMMA)
unbudgeted = op.solve_analytic_log_opt(sigma, mu, gamma=SHOWN_GAMMA)
wide = op.Constraints(is_long_only=False, min_exposure=-10.0, max_exposure=10.0)
free, _ = op.wrapper_quadratic_optimisation(
    pd_covar=covar, constraints=wide,
    portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
    means=means, carra=SHOWN_GAMMA, optimiser_config=config)
print(round(unbudgeted.sum(), 3), round(free.sum(), 3))
assert np.abs(budgeted - shown).max() < 1e-5
assert np.abs(unbudgeted - (B / SHOWN_GAMMA) * w_tan).max() < 1e-12
```

It prints `1.393 1.393`. With excess returns, the investor without a budget holds 139.3% in the
tangency portfolio and finances the rest at the cash rate; the budget replaces that financing by
a short position in the minimum-variance portfolio.

Supplying total returns instead of excess returns changes the answer:

```python
total, _ = op.wrapper_maximize_portfolio_sharpe(
    pd_covar=covar, means=means + RISK_FREE, constraints=long_only, optimiser_config=config)
print(total.round(3).tolist())
```

It prints `[0.547, 0.197, 0.14, 0.044, 0.072]`. With a cash rate of 2% added to every mean, the
solver returns $\Sigma^{-1} (\mu + 0.02 \cdot \mathbf{1})$ normalised, the frontier portfolio
where a line from an excess return of −2% at zero volatility touches the frontier. It lies between
minimum variance and the tangency portfolio, holds 54.7% in government bonds instead of 28.5%,
and its excess-return Sharpe ratio is 0.38 instead of 0.416.

A volatility cap on the Charnes–Cooper route shows the rows that the transformation does not
rescale. The cap of 12% is twice the tangency volatility:

```python
capped = op.Constraints(is_long_only=True, max_target_portfolio_vol_an=VOL_CAP)
cap_weights, cap_outcome = op.wrapper_maximize_portfolio_sharpe(
    pd_covar=covar, means=means, constraints=capped, optimiser_config=config)
print(cap_outcome.accepted, cap_outcome.status, cap_outcome.fallback_source)
```

It prints `False infeasible zeros`, and the returned weights are zero. The script also checks
that negated means, for which no long-only portfolio has a positive expected return, make the
program infeasible, and that passing `means` to the quadratic wrapper without naming the utility
objective returns the minimum-variance portfolio.

The dispatcher estimates the means itself. On business-daily prices simulated with the log drifts
`MEANS` and the covariance above, from 2015 to 2024, the estimates at the quarter ends of 2024
are:

```python
prices = simulated_prices(covar, np.asarray(MEANS), seed=SEED)
dates = [pd.Timestamp(date) for date in REBALANCING]
covar_dict = {date: covar for date in dates}
estimated = op.estimate_rolling_ewma_means(prices=prices, rebalancing_dates=dates)
print(estimated.round(3))
```

| Date | Govt | Credit | US eq | EM eq | Gold |
|---|---|---|---|---|---|
| 2024-03-31 | 0.001 | −0.061 | 0.103 | −0.079 | −0.105 |
| 2024-06-30 | −0.016 | 0.005 | 0.245 | 0.092 | 0.060 |
| 2024-09-30 | −0.034 | −0.089 | 0.230 | 0.208 | −0.231 |
| 2024-12-31 | −0.081 | 0.007 | 0.310 | 0.325 | −0.185 |

The values equal a hand-written EWMA recursion on Wednesday-to-Wednesday log returns within
$10^{-12}$. The true means are 0.5% to 5.5%; in December the estimates are 31.0% for US equities,
32.5% for emerging-market equities and −8.1% for government bonds. Each error is between one and
two standard errors, which the formula above puts at the asset's volatility. The script also
checks that the squared EWMA weights of `qis.compute_ewm` with span 52 sum to $1/52$.

The dispatcher routes utility, with its `carra=0.5`, and maximum Sharpe to the rolling functions:

```python
dispatched = op.compute_rolling_optimal_weights(
    prices=prices, constraints=long_only, covar_dict=covar_dict,
    portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY, optimiser_config=config)
direct = op.rolling_quadratic_optimisation(
    prices=prices, constraints=long_only, covar_dict=covar_dict,
    portfolio_objective=op.PortfolioObjective.QUADRATIC_UTILITY,
    expected_returns=estimated, carra=0.5, optimiser_config=config)
tangency_path = op.rolling_maximize_portfolio_sharpe(
    prices=prices, expected_returns=estimated, constraints=long_only,
    covar_dict=covar_dict, optimiser_config=config)
print(dispatched.round(2))
print(tangency_path.round(2))
assert np.abs(dispatched.to_numpy() - direct.to_numpy()).max() < 1e-8
```

| Date | Utility, $\gamma = 0.5$ | Maximum Sharpe |
|---|---|---|
| 2024-03-31 | 100% US eq | 40% Govt, 60% US eq |
| 2024-06-30 | 100% US eq | 80% US eq, 20% Gold |
| 2024-09-30 | 100% US eq | 79% US eq, 21% EM eq |
| 2024-12-31 | 100% EM eq | 69% US eq, 31% EM eq |

The dispatcher's weights equal the direct calls within $10^{-8}$ for both objectives, and each
date matches a hand-written CVXPY program within $10^{-5}$. With estimated means, the utility
portfolio holds a single asset at every date, the one with the highest estimated mean, and the
maximum-Sharpe portfolio never holds credit, although the model's true tangency portfolio holds
30.4%.

`plot_efficient_frontier` draws portfolios that have been solved. Here it receives the four
utility portfolios, with minimum variance as their benchmark:

```python
result = op.PortfolioOptimisationResult(
    weights=grid, benchmark_weights=min_var.rename('minimum variance'),
    covar_data=identity_factor_model(covar), group_attributions={}, expected_return=means)
figure = op.plot_efficient_frontier(result, profiles={'utility': list(grid.columns)})
points, _ = result.compute_efficient_frontier_data(profiles={'utility': list(grid.columns)})
```

It returns a matplotlib figure, and the points it draws are the volatility and expected return of
each portfolio under the result's risk model, here the covariance above.

## Implementation in optimalportfolios

Both solver families have the three layers described in
[choosing an objective](optimization_module_readme.md#three-layer-solver-pattern):

- `cvx_quadratic_optimisation(portfolio_objective, covar, constraints, means=None,
  verbose=False, solver='CLARABEL', carra=1.0, context='', factorize_covar=True)` builds the
  CVXPY problem on NumPy inputs, with every row that `Constraints` compiles for CVXPY, and returns
  an `OptimizationOutcome`. A `SolverError` from the backend is recorded as `solver_error`.
- `wrapper_quadratic_optimisation(pd_covar, constraints, inclusion_indicators=None,
  portfolio_objective=PortfolioObjective.MIN_VARIANCE, means=None, weights_0=None, carra=1.0,
  optimiser_config=OptimiserConfig(apply_total_to_good_ratio=True), context='')` filters the
  universe, aligns the constraints and returns a pair: the weights on the original labels, with
  zeros for removed assets, and the outcome. Given `means`, it drops assets with a non-finite
  mean even for minimum variance, where the means play no other part.
- `rolling_quadratic_optimisation(prices, constraints, covar_dict, inclusion_indicators=None,
  portfolio_objective=PortfolioObjective.MIN_VARIANCE, expected_returns=None, carra=1.0,
  optimiser_config=OptimiserConfig(apply_total_to_good_ratio=True))` solves at each key of
  `covar_dict`, in the dictionary's order, forward-fills `expected_returns` and
  `inclusion_indicators` to those dates, passes the previous weights drifted to each date as
  `weights_0`, and returns a date-by-asset table on the columns of `prices`. It raises
  `ValueError` for utility without `expected_returns`.
- `cvx_maximize_portfolio_sharpe(covar, means, constraints, verbose=False, solver='CLARABEL',
  context='', factorize_covar=True)` routes on `min_exposure == max_exposure`: equal, to the
  Charnes–Cooper program; different, to SLSQP, which ignores `solver` and `factorize_covar` and
  records `SLSQP` as the outcome's `solver`. It returns an `OptimizationOutcome` on both routes.
- `wrapper_maximize_portfolio_sharpe(pd_covar, means, constraints, weights_0=None,
  optimiser_config=OptimiserConfig(apply_total_to_good_ratio=True), context='')` filters on the
  variances and the means and returns the same pair as the quadratic wrapper.
- `rolling_maximize_portfolio_sharpe` is the rolling layer, with the same schedule, forward fill
  and drift; it takes `(prices, expected_returns, constraints, covar_dict,
  optimiser_config=OptimiserConfig(apply_total_to_good_ratio=True))` and no inclusion indicators.
- `solve_analytic_log_opt(covar, means, exposure_budget_eq=None, gamma=1.0)` returns the closed
  form of Proposition 3 as a NumPy array. It inverts the covariance and ignores every other
  constraint.

The code is in [`quadratic.py`](../src/optimalportfolios/optimization/general/quadratic.py) and
[`max_sharpe.py`](../src/optimalportfolios/optimization/general/max_sharpe.py). What the
outcome records, and the fallback of a rejected solve, are on the
[solver-outcomes page](solver_numerics_and_outcomes.md#acceptance-and-fallback); the
`Constraints` fields are on the [constraints page](constraints.md).

The dispatcher `compute_rolling_optimal_weights` routes `PortfolioObjective.MIN_VARIANCE` to
`rolling_quadratic_optimisation` without means. For `QUADRATIC_UTILITY` and
`MAXIMUM_SHARPE_RATIO` it first calls `estimate_rolling_ewma_means` with its own `returns_freq`
and `span`, by default `'W-WED'` and 52, and `annualize=True`, and it passes `carra=0.5` to the
utility route. It accepts no forecast panel; call the rolling functions directly to supply
capital market assumptions. The [dispatch flow](optimization_module_readme.md#dispatch-flow)
lists every route.

`estimate_rolling_ewma_means(prices, rebalancing_dates, returns_freq='W-WED', span=52,
annualize=True)`, in `optimalportfolios.alphas` and exported at the package root, returns a
DataFrame indexed by `rebalancing_dates`; a date before the first return is missing. Its code is
in [`rolling_ewma_mean.py`](../src/optimalportfolios/alphas/signals/rolling_ewma_mean.py), and
the returns and the recursion are computed by qis.

`plot_efficient_frontier(result, profiles, order=3, markersize=40, xvar_format='{:.1%}',
yvar_format='{:.1%}', title=None, drop_duplicated_annotations=False, ax=None, **kwargs)`, in
`optimalportfolios.reports` and exported at the package root, takes a
`PortfolioOptimisationResult` and a mapping from profile names to portfolio names. It reads the
volatility and the expected return of each named portfolio and of its benchmark from the result
and draws them with `qis.plot_scatter`: one colour per profile, solid for portfolios, dotted for
benchmarks, joined by a polynomial fit of order `order`. It returns the matplotlib figure. It
plots the portfolios it is given and computes no frontier; its code is in
[`portfolio_result_plots.py`](../src/optimalportfolios/reports/portfolio_result_plots.py).

> **Pitfall.** On the Charnes–Cooper route only the exposure, per-asset and group-allocation rows
> are rescaled by $k$. A volatility cap, a return floor, a turnover or a tracking-error limit is
> compiled on $y = k w$, not on $w$. In the example a volatility cap of 12%, twice the tangency
> volatility, makes the program infeasible, and the wrapper returns the fallback, here zeros. A
> return floor can be encoded homogeneously as on the
> [overlay page](overlay_tail_floor.md#the-encoding).

## Interpretation and limitations

- Markowitz (1952) states the choice between efficient combinations of expected return and
  variance for given beliefs. The package solves one period at a time for the $\mu$ and $\Sigma$
  it is given and adds the rows of `Constraints`; the dispatcher's only estimator of the means is
  the EWMA above.
- The closed forms hold only while no bound binds. With binding bounds, the solutions lie on the
  constrained frontier, the two-fund mix of Proposition 3 no longer holds, and
  `solve_analytic_log_opt`, which ignores bounds, gives a different portfolio.
- The Sharpe ratio of the objective is a model quantity on the supplied means, not a realised
  statistic, and it subtracts no cash rate. With total returns the solver returns a portfolio
  nearer minimum variance: in the example, 54.7% instead of 28.5% in government bonds.
- The risk aversion has the units of the inverse of the returns. The same $\gamma$ serves annual
  and weekly inputs, but returns in percent need $\gamma / 100$. The default `carra=1.0` of the
  direct functions and the dispatcher's 0.5 are aggressive for annual inputs: in the example,
  $\gamma = 5$ already leads past the tangency portfolio.
- The standard error of the dispatcher's annualised mean equals the asset's volatility. A utility
  portfolio at $\gamma = 0.5$ then holds whichever asset has the highest estimate, and a
  maximum-Sharpe portfolio concentrates on the assets with the luckiest samples, as in the rolling
  example. Chopra and Ziemba (1993) and DeMiguel, Garlappi and Uppal (2009) report the
  consequence for out-of-sample performance.
- Section 4.3 of Sepp (2023) states the maximum-Sharpe problem as equation (7): the ratio of
  expected return to volatility over long-only, fully invested weights no larger than one, which
  `wrapper_maximize_portfolio_sharpe` solves with `Constraints(is_long_only=True)` on the
  Charnes–Cooper route. The paper estimated means and covariances from a six-year window of
  monthly returns at quarterly rebalancing; the dispatcher's defaults are an EWMA of weekly
  returns with span 52. A fixed weight such as the paper's 75% in a balanced sleeve is a
  per-asset bound, which the transformation rescales.
- The SLSQP route is a local method on a non-convex ratio and compiles fewer rows than CVXPY; the
  band leaves the exposure undecided within it. Prefer a fixed exposure unless the band is needed.
- The Charnes–Cooper route needs a feasible portfolio with a positive expected return. In a
  long-only book without one, the program is infeasible and the outcome falls back to the
  pre-trade weights, the benchmark or zeros. A long-short book without per-asset bounds needs
  $B \gt 0$ (Proposition 2): otherwise no fully invested portfolio attains the supremum of the
  ratio, the program pushes $k$ towards zero and the recovered weights are unbounded. The script
  checks that the wrapper then returns no usable portfolio: weights above 100 times capital, or a
  rejected outcome.

## See also

- [Choosing an objective](optimization_module_readme.md)
- [Strategic allocation: target return and target volatility](strategic_allocation_targets.md)
- [CARA utility under Gaussian mixtures](cara_gaussian_mixture.md)
- [Maximum diversification](maximum_diversification.md), the default, risk-based objective
- [Risk budgeting](risk_budgeting.md)
- [Portfolio constraints](constraints.md)
- [Covariance factorisation, solver outcomes and constraint residuals](solver_numerics_and_outcomes.md)
- [Overlay optimisation with a fixed core and linear side constraints](overlay_tail_floor.md)
- [Alpha signals](alphas_module_readme.md)
- [Conventions, notation and glossary](conventions.md)
- [Cryptocurrencies in diversified portfolios](app_crypto_allocation.md)
- [Research papers and replication](research_papers.md)

## References

- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
  [DOI 10.1111/j.1540-6261.1952.tb01525.x](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).
- Charnes, A. and Cooper, W. W. (1962). *Programming with Linear Fractional Functionals*. Naval
  Research Logistics Quarterly, 9(3–4), 181–186.
  [DOI 10.1002/nav.3800090303](https://doi.org/10.1002/nav.3800090303).
- Chopra, V. K. and Ziemba, W. T. (1993). *The Effect of Errors in Means, Variances, and
  Covariances on Optimal Portfolio Choice*. The Journal of Portfolio Management, 19(2), 6–11.
  [DOI 10.3905/jpm.1993.409440](https://doi.org/10.3905/jpm.1993.409440).
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). *Optimal Versus Naive Diversification: How
  Inefficient is the 1/N Portfolio Strategy?* The Review of Financial Studies, 22(5), 1915–1953.
  [DOI 10.1093/rfs/hhm075](https://doi.org/10.1093/rfs/hhm075).
- Sepp, A. (2023). *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*. Risk,
  October 2023. [SSRN 4217841](https://ssrn.com/abstract=4217841). Section 4.3, equation (7),
  states the maximum-Sharpe problem.
- Diamond, S. and Boyd, S. (2016). *CVXPY: A Python-Embedded Modeling Language for Convex
  Optimization*. Journal of Machine Learning Research, 17(83), 1–5.
  [JMLR](https://www.jmlr.org/papers/v17/15-408.html).
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
