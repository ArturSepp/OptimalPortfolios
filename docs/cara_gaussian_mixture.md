---
myst:
  html_meta:
    description: >-
      CARA utility under a Gaussian mixture in Python with optimalportfolios: the closed form
      from the normal moment-generating function, convexity, the mean-variance case with one
      component, the optimum as mean-variance under utility-tilted component probabilities, the
      in-house EM fit and what annualising its components assumes, with a verified offline
      example.
---

# CARA utility under Gaussian mixtures

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Expected CARA utility under a Gaussian mixture is implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

An investor with constant absolute risk aversion (CARA) ranks portfolios by the expected value of
$-e^{-\gamma R}$, where $R$ is the portfolio return and $\gamma$ the risk aversion. When asset
returns follow a mixture of $K$ Gaussian components, that expectation has a closed form, so the
fat tails and the regime-dependent correlations of a mixture enter the objective without
simulation. The package maximises it with SciPy's SLSQP. Its rolling function fits its own
mixture to rolling windows of log returns and does not use a covariance matrix.

This page derives the closed form and proves that the problem is convex. With one component the
objective is mean-variance utility, with its familiar closed form. With several, the optimum is a
mean-variance portfolio in which each component is reweighted by the marginal utility it carries,
which is why a crash component lowers the allocation to the asset that crashes. The page then
shows what the fit keeps from the data and what annualising the fitted components assumes. The
worked example checks each statement against an independent computation.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Log returns in the rolling path; the objective treats $w^{\top} r$ as the portfolio return |
| Estimation grid | Weekly returns (`'W-WED'`) in windows of 312 returns, six years, in the rolling function; fixed annual parameters in the single-date example |
| Rebalancing grid | The first return date on or after each quarter end (`'QE'`); the keys of `covar_dict` are not used |
| Covariance units | Annual: the rolling path multiplies each fitted component's mean and covariance by 52 for weekly returns; the single-date functions use the units they receive |
| Expected returns | The component means $\mu_k$, fitted in the rolling path and supplied to the single-date functions |
| Weight state | Target weights; each rolling solve starts from, and falls back to, the previous targets drifted to the date |
| Solver | SciPy SLSQP; CVXPY only for the independent reference in the example |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $K$ | Number of mixture components |
| $p_k$, $\mu_k$, $\Sigma_k$ | Probability, mean vector and covariance matrix of component $k$ |
| $r$, $R(w)$ | Asset returns over the horizon of the parameters; the portfolio return $w^{\top} r$ |
| $q_k(w)$ | Component exponent $-\gamma \mu_k^{\top} w + \frac{1}{2} \gamma^2 w^{\top} \Sigma_k w$ |
| $\mathrm{CE}(w)$ | Certainty equivalent $-\frac{1}{\gamma} \ln \mathbb{E}[e^{-\gamma R(w)}]$ |
| $\pi_k(w)$ | Tilted probability $p_k e^{q_k(w)} / \sum_j p_j e^{q_j(w)}$ |
| $\bar x$, $S$ | Sample mean and sample covariance, with divisor $n$, of a window of $n$ returns |
| $B$ | Covariance of the component means, $\sum_k p_k (\mu_k - \bar\mu)(\mu_k - \bar\mu)^{\top}$ with $\bar\mu = \sum_k p_k \mu_k$ |
| $\mathbf{1}$ | Vector of ones |

The probabilities are positive and sum to one, and each $\Sigma_k$ is positive semi-definite.
The parameters describe returns over one horizon, a year when they are annual. The propositions
assume full investment, $\mathbf{1}^{\top} w = 1$, and Proposition 4 also assumes that no weight
bound binds. The code calls $\gamma$ `carra` and $K$ `n_components`.

## Methodology

CARA utility of a return $x$ is $u(x) = -e^{-\gamma x}$. Its absolute risk aversion
$-u''(x) / u'(x)$, the measure of Pratt (1964), equals $\gamma$ at every level of $x$, which
gives the utility its name. The investor solves

$$
\max_w \mathbb{E}\left[-e^{-\gamma R(w)}\right] \quad \text{subject to} \quad \mathbf{1}^{\top} w = 1 \text{ and the limits of the mandate.}
$$

### Expected utility in closed form

**Proposition 1 (closed form).** If $r$ follows the mixture, then for every $w$

$$
\mathbb{E}\left[-e^{-\gamma R(w)}\right] = -\sum_{k=1}^{K} p_k \exp\left(-\gamma \mu_k^{\top} w + \frac{1}{2} \gamma^2 w^{\top} \Sigma_k w\right) = -\sum_{k=1}^{K} p_k e^{q_k(w)} .
$$

**Proof.** Given component $k$, the portfolio return $R(w) = w^{\top} r$ is normal with mean
$\mu_k^{\top} w$ and variance $w^{\top} \Sigma_k w$, because a linear map of a Gaussian vector is
Gaussian. A normal variable $X$ with mean $m$ and variance $s^2$ has the moment-generating
function $\mathbb{E}[e^{tX}] = e^{tm + t^2 s^2 / 2}$ for every real $t$. At $t = -\gamma$ this
gives $\mathbb{E}[e^{-\gamma R(w)} \mid k] = e^{q_k(w)}$, and the law of total expectation
weights the components by their probabilities $p_k$. $\square$

The expectation is finite for every $w$, because every component is Gaussian and has all
exponential moments. A distribution with power tails, such as Student's t, has none, and its
expected CARA utility is minus infinity: a mixture adds fat tails that CARA utility can still
price. The certainty equivalent
$\mathrm{CE}(w) = -\frac{1}{\gamma} \ln \sum_k p_k e^{q_k(w)}$ is the sure return with the same
utility. It falls as the expected disutility rises, so both objectives give the same weights.

**Proposition 2 (convexity).** The expected disutility $f(w) = \sum_k p_k e^{q_k(w)}$ is convex,
and strictly convex when some $\Sigma_k$ is positive definite. Under linear constraints, every
point that satisfies the Karush–Kuhn–Tucker conditions is therefore a global optimum, and in the
strictly convex case the only one.

**Proof.** Each $q_k$ is a convex quadratic because $\Sigma_k$ is positive semi-definite. The
exponential is convex and increasing, so $e^{q_k}$ is convex, and a sum with positive weights
stays convex. When $\Sigma_k$ is positive definite, $q_k$ is strictly convex, and composing it
with the strictly increasing exponential keeps it strict. For a convex objective and linear
constraints, the Karush–Kuhn–Tucker conditions are sufficient for a global minimum, and necessary
because linear constraints need no further qualification. $\square$

Sepp (2023, Section 4.4) notes that this objective is convex; Proposition 2 supplies the
argument. A local method such as SLSQP therefore needs only to reach a point that satisfies the
optimality conditions.

### One component: mean-variance utility

**Proposition 3 (one component).** With $K = 1$, maximising expected CARA utility is maximising
$\mu^{\top} w - \frac{\gamma}{2} w^{\top} \Sigma w$. For positive definite $\Sigma$ the
unconstrained optimum is $w = \Sigma^{-1} \mu / \gamma$. Under the budget the optimum is

$$
w^{\star} = \frac{1}{\gamma} \Sigma^{-1} (\mu - \eta \mathbf{1}), \qquad \eta = \frac{\mathbf{1}^{\top} \Sigma^{-1} \mu - \gamma}{\mathbf{1}^{\top} \Sigma^{-1} \mathbf{1}} ,
$$

and the budget does not bind, $\eta = 0$, exactly when $\gamma = \mathbf{1}^{\top} \Sigma^{-1} \mu$.

**Proof.** The exponent is $q(w) = -\gamma (\mu^{\top} w - \frac{\gamma}{2} w^{\top} \Sigma w)$, and
$e^{q}$ is a decreasing function of the mean-variance objective, so both have the same maximiser.
The first-order condition of the budget-constrained problem is
$\mu - \gamma \Sigma w = \eta \mathbf{1}$ for a multiplier $\eta$, which gives $w^{\star}$; the
budget $\mathbf{1}^{\top} w^{\star} = 1$ fixes $\eta$. $\square$

The budget moves the unconstrained optimum by a multiple of $\Sigma^{-1} \mathbf{1}$, the direction
of the minimum-variance portfolio, until the weights sum to one. With long-only bounds the formula
holds while every weight is positive; otherwise a bound binds. The single-Gaussian case is the
quadratic utility of [minimum variance, quadratic utility and maximum Sharpe](mean_variance_objectives.md),
with the same `carra` convention.

### Several components: tilted mean-variance

**Proposition 4 (tilted first-order condition).** Let $w^{\star}$ maximise the mixture utility
under the budget, with no bound binding, and let $\pi_k = \pi_k(w^{\star})$. Then $w^{\star}$ is the
one-component optimum of Proposition 3 for the tilted mean and covariance

$$
\mu_{\pi} = \sum_{k=1}^{K} \pi_k \mu_k, \qquad \Sigma_{\pi} = \sum_{k=1}^{K} \pi_k \Sigma_k, \qquad w^{\star} = \frac{1}{\gamma} \Sigma_{\pi}^{-1} (\mu_{\pi} - \eta \mathbf{1}) .
$$

**Proof.** The gradient of $f$ is
$\nabla f(w) = \sum_k p_k e^{q_k(w)} (\gamma^2 \Sigma_k w - \gamma \mu_k) = \gamma f(w) \sum_k \pi_k(w) (\gamma \Sigma_k w - \mu_k)$.
At an optimum with only the budget active, $\nabla f(w^{\star}) = \nu \mathbf{1}$ for a
multiplier $\nu$. Dividing by $-\gamma f(w^{\star})$ gives
$\mu_{\pi} - \gamma \Sigma_{\pi} w^{\star} = \eta \mathbf{1}$ with
$\eta = -\nu / (\gamma f(w^{\star}))$, the condition of Proposition 3. $\square$

The tilted probability is $\pi_k = p_k \mathbb{E}[e^{-\gamma R} \mid k] / \mathbb{E}[e^{-\gamma R}]$:
the probability of component $k$ weighted by the marginal utility $\gamma e^{-\gamma R}$ of the
portfolio in it. A component in which the portfolio has a low mean or a high variance gets more
than its probability, and a crash component pulls the tilted mean of the crashing asset down and
its tilted variance up. The spread of the component means enters only through the tilt, and
because $\pi$ depends on $w^{\star}$ the condition is a fixed point, not a closed form. For
$K \geq 2$ the certainty equivalent is no longer a function of the portfolio's mean and variance
alone: skewness and fat tails move it. Ang, Morris and Savi (2023) rely on this effect to model
investors who seek positive skewness.

> **Insight.** The mixture optimum is a mean-variance portfolio seen through the investor's
> marginal utility. At $\gamma = 5$ in the worked example, the crash component, with probability
> 5%, carries a tilted weight of 10.1%, twice its probability. The mean-variance portfolio of the
> tilted moments holds 6.3% in crypto, against 7.2% for the Gaussian with the mixture's own mean
> and covariance.

### Fitting the mixture

`fit_gaussian_mixture(x, n_components=2, an_factor=1.0, idx=None)` fits a mixture with full
covariance matrices by the expectation-maximisation (EM) algorithm of Dempster, Laird and Rubin
(1977), implemented in the package without scikit-learn. It starts from the k-means clusters of
`scipy.cluster.vq.kmeans2`, seeded with a fixed `RandomState(3)`, so the same data always give the
same fit. It runs at most 100 iterations, stops when the total log-likelihood changes by less than
$10^{-6}$, and adds $10^{-6}$ to the diagonal of each component covariance. It returns a `Params`
record with the lists `means` and `covars`, each multiplied by `an_factor`, and the array
`probs`; `idx` sorts the components by the mean of one column. The number of components is given,
not selected.

**Proposition 5 (moments of the fit).** After every M-step, and so for the returned fit before
scaling,

$$
\sum_{k=1}^{K} p_k \mu_k = \bar x, \qquad \sum_{k=1}^{K} p_k \Sigma_k + B = S + 10^{-6} I ,
$$

whenever no component is empty. The rolling path multiplies each $\mu_k$ and $\Sigma_k$ by the
annualisation factor $\mathrm{AN}$, 52 for weekly returns. The scaled mixture has mean
$\mathrm{AN} \bar x$ and covariance

$$
\mathrm{AN} (S + 10^{-6} I) + \mathrm{AN} (\mathrm{AN} - 1) B .
$$

**Proof.** With responsibilities $\rho_{ik}$ that sum to one over $k$, the M-step sets
$n_k = \sum_i \rho_{ik}$, $p_k = n_k / n$, $\mu_k = \sum_i \rho_{ik} x_i / n_k$ and
$\Sigma_k = \sum_i \rho_{ik} (x_i - \mu_k)(x_i - \mu_k)^{\top} / n_k + 10^{-6} I$. Summing the
means gives $\sum_k p_k \mu_k = \sum_i x_i / n = \bar x$. Expanding each scatter around $\bar x$,
$\sum_i \rho_{ik} (x_i - \mu_k)(x_i - \mu_k)^{\top} = \sum_i \rho_{ik} (x_i - \bar x)(x_i - \bar x)^{\top} - n_k (\mu_k - \bar x)(\mu_k - \bar x)^{\top}$,
and summing over $k$ and dividing by $n$ gives the second identity. Scaling multiplies the
within-component term $\sum_k p_k \Sigma_k$ by $\mathrm{AN}$ and the spread of the means $B$ by
$\mathrm{AN}^2$, which gives the annual covariance. $\square$

A one-component fit is therefore the sample mean and covariance of the window, plus the
$10^{-6}$ ridge, and fitting more components at the data frequency changes only the higher
moments. Annualising by scaling does more: it counts the spread of the component means
$\mathrm{AN}^2$ times instead of $\mathrm{AN}$ times.

### What annualising by scaling assumes

Multiplying a component's mean and covariance by $\mathrm{AN}$ multiplies its exponent by
$\mathrm{AN}$, so the annual expected disutility is $\sum_k p_k x_k^{\mathrm{AN}}$ with
$x_k = e^{q_k(w)}$ evaluated at the data frequency. That is the disutility of a year spent entirely
in one component, drawn once with probability $p_k$. If instead each week drew its component
independently, the disutility of the year would be $(\sum_k p_k x_k)^{\mathrm{AN}}$.

**Proposition 6 (persistent against independent regimes).** For every $w$,
$\sum_k p_k x_k^{\mathrm{AN}} \geq (\sum_k p_k x_k)^{\mathrm{AN}}$, with equality when $K = 1$ or
all $x_k$ are equal. The scaled mixture is therefore at least as averse to every portfolio as the
model with independent weekly regimes.

**Proof.** The function $x \mapsto x^{\mathrm{AN}}$ is convex on the positive numbers for
$\mathrm{AN} \geq 1$, and strictly convex for $\mathrm{AN} \gt 1$; Jensen's inequality gives the
result. $\square$

By the central limit theorem, the sum of 52 independent weekly draws is close to Gaussian, so
under independent regimes the higher moments that motivate the mixture would largely vanish over
a year. Scaling keeps them. So does equation `eq:l2` of the crypto paper, which models the log
return over the horizon directly as a mixture.

## Worked example

The canonical script of this page,
[`examples/docs/cara_gaussian_mixture.py`](../examples/docs/cara_gaussian_mixture.py),
runs offline and asserts every number quoted here against an independent computation. One step
deliberately makes a solve fail and logs its rejection:

```console
python -m examples.docs.cara_gaussian_mixture
```

Three assets, bonds, equities and crypto, follow a stylised annual mixture of three components.
The correlations are the same in every component: $-0.2$ between bonds and equities, 0 between
bonds and crypto and 0.3 between equities and crypto. Means are annual log returns.

| Component | Probability | Mean: bonds | equities | crypto | Volatility: bonds | equities | crypto |
|---|---|---|---|---|---|---|---|
| Calm | 80% | 2% | 11% | 50% | 5% | 13% | 60% |
| Stress | 15% | 4% | −6% | −20% | 7% | 20% | 80% |
| Crash | 5% | 6% | −30% | −150% | 8% | 30% | 70% |

The mixture has means of 2.5%, 6.4% and 29.5% and volatilities of 5.6%, 18.6% and 80.0%, which
the script confirms on 200,000 draws. The one-component model is the Gaussian with this mean and
covariance. At $\gamma = 5$ the closed form of Proposition 3 holds 76.0% in bonds, 16.8% in
equities and 7.2% in crypto, and three solver routes agree with it within $10^{-4}$: the quadratic
and the exponential objective of `opt_maximize_cara`, and `opt_maximize_cara_mixture` with one
component.

```python
mean, covar = mixture_moments(PROBS, MEANS, component_covariances())
closed_form = budget_mean_variance(mean, covar, GAMMA)
quadratic = op.opt_maximize_cara(means=mean, covar=covar, carra=GAMMA)
exponential = op.opt_maximize_cara(means=mean, covar=covar, carra=GAMMA, is_exp=True)
one = op.opt_maximize_cara_mixture(means=[mean], covars=[covar], probs=np.array([1.0]),
                                   constraints=op.Constraints(is_long_only=True),
                                   carra=GAMMA)
for weights in (quadratic, exponential, one):
    assert np.abs(weights - closed_form).max() < 1e-4
assert quoted(closed_form, 3) == [0.760, 0.168, 0.072]
```

At $\gamma^{\star} = \mathbf{1}^{\top} \Sigma^{-1} \mu = 12.41$ the budget does not bind, and the
solver returns the unconstrained optimum $\Sigma^{-1} \mu / \gamma^{\star}$, which holds 81.6%,
16.1% and 2.4%:

```python
inverse_mean = np.linalg.solve(covar, mean)
gamma_star = inverse_mean.sum()
free = op.opt_maximize_cara(means=mean, covar=covar, carra=gamma_star)
assert np.abs(free - inverse_mean / gamma_star).max() < 1e-4
assert round(gamma_star, 2) == 12.41
```

With the three components themselves, SLSQP agrees with an independent CVXPY solve on the
exponential cone within $10^{-4}$. The portfolio holds 79.8% in bonds, 13.9% in equities and 6.3%
in crypto:

```python
three = op.opt_maximize_cara_mixture(
    means=list(MEANS), covars=component_covariances(), probs=PROBS,
    constraints=op.Constraints(is_long_only=True), carra=GAMMA)
assert np.abs(three - cvxpy_mixture_optimum(GAMMA)).max() < 1e-4
assert quoted(three, 3) == [0.798, 0.139, 0.063]
```

The closed form of Proposition 1 matches the mean of $e^{-\gamma R}$ over the 200,000 draws within
1%. Both models give the one-component portfolio the same mean and variance, yet its certainty
equivalent is 3.41% under the Gaussian and 3.19% under the mixture; the three-component optimum
raises the mixture's certainty equivalent to 3.24%.

The tilted probabilities of Proposition 4 at this optimum are 72% calm, 18% stress and 10.1%
crash. The one-component formula with the tilted mean and covariance reproduces the solver's
weights within $10^{-4}$, and the tilt agrees with the conditional means of $e^{-\gamma R}$ over
the draws of each component:

```python
tilt = tilted_probabilities(three, GAMMA)
tilted_mean = tilt @ MEANS
tilted_covar = sum(p * c for p, c in zip(tilt, component_covariances()))
assert np.abs(three - budget_mean_variance(tilted_mean, tilted_covar, GAMMA)).max() < 1e-4
assert quoted(tilt, 3)[CRASH] == 0.101
```

At every risk aversion from 0.5 to 6 the crash component lowers the weight of crypto, and under
both models the weight falls as risk aversion rises. The three-component solutions agree with the
CVXPY reference at every point, and the one-component solutions with the closed form wherever it
is long-only. It is not at $\gamma = 0.5$, the value of the paper's CARA-3 configuration: the
closed form would short bonds, the long-only bound binds, and the one-component portfolio holds
80.8% in crypto and 19.2% in equities. The three-component portfolio holds 72.4% in crypto there.

![Left: the weight of crypto against risk aversion from 0.5 to 6 on a log scale, for the Gaussian
with the mixture's mean and covariance and for the three-component mixture; both fall from about
81% and 72% at 0.5 to about 6% and 5% at 6, and the three-component line lies below the
one-component line throughout. Right: the weights of bonds, equities and crypto at risk aversion 5,
76.0%, 16.8% and 7.2% with one component and 79.8%, 13.9% and 6.3% with three
components.](images/cara_mixture_allocation.png)

*Figure: how risk aversion and a crash component change the allocation, for the stylised
three-asset mixture of the example. Drawn by the `exhibit` function of the canonical script; the
[analytics gallery](analytics_gallery.md) lists its provenance.*

A fit to data behaves as Proposition 5 states. The script draws eight years of weekly returns,
each week from a component chosen independently, with the annual moments divided by 52. The
three-component fit, scaled by 52, keeps 52 times the sample mean exactly; its covariance is 52
times the sample covariance plus the ridge, plus $52 \times 51$ times the spread of the fitted
means:

```python
returns = simulated_weekly_returns(SEED)
fitted = op.fit_gaussian_mixture(x=returns.to_numpy(), n_components=3, an_factor=52.0)
fitted_mean, fitted_covar = mixture_moments(fitted.probs, fitted.means, fitted.covars)
sample = returns.to_numpy()
centred = sample - sample.mean(axis=0)
persistence = 52.0 * 51.0 * between_covariance(fitted.probs, np.array(fitted.means) / 52.0)
assert np.allclose(fitted_mean, 52.0 * sample.mean(axis=0), atol=1e-12)
assert np.allclose(fitted_covar, 52.0 * (centred.T @ centred / len(sample)
                                         + 1e-6 * np.eye(3)) + persistence, atol=1e-10)
```

The term is not small. The fitted components differ most in the weekly mean of crypto, and
scaling turns each into a year: the annual volatility of crypto under the fitted mixture is about
290%, against 68% for 52 times the sample variance. A one-component fit returns 52 times the
sample covariance plus the ridge, and refitting the same data returns the same parameters.

The rolling function runs these steps at each rebalancing date. Called through the dispatcher
with an empty `covar_dict`, the same settings give the same weights:

```python
prices = prices_from_returns(returns)
rolling = op.rolling_maximize_cara_mixture(
    prices=prices, constraints=op.Constraints(is_long_only=True), time_period=None)
routed = op.compute_rolling_optimal_weights(
    prices=prices, constraints=op.Constraints(is_long_only=True), covar_dict={},
    portfolio_objective=op.PortfolioObjective.MAX_CARA_MIXTURE, roll_window=312,
    n_mixures=3)
assert np.allclose(rolling, routed, atol=1e-12)
```

The weights are dated on the weekly grid: the first, on 5 January 2022, is the first Wednesday
after the quarter end once 312 returns are available, and the eight dates follow the same rule.
Each date is one fit and one solve on its window, which the script rebuilds:

```python
weekly_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                freq='W-WED')
assert list(rolling.index) == expected_rebalancing_dates(weekly_returns.index, 312)
first = rolling.index[0]
window = weekly_returns.loc[:first].iloc[-312:]
params = op.fit_gaussian_mixture(x=window.to_numpy(), n_components=3, an_factor=52.0)
direct = op.opt_maximize_cara_mixture(means=params.means, covars=params.covars,
                                      probs=params.probs,
                                      constraints=op.Constraints(is_long_only=True),
                                      carra=0.5)
assert np.allclose(rolling.loc[first].to_numpy(), direct, atol=1e-10)
```

With one missing equity price inside the window, equities receive zero weight at that date and
the other two assets receive the solution of a two-asset fit.

## Implementation in optimalportfolios

The objective has the three layers of every solver family, described in
[choosing an objective](optimization_module_readme.md), and the public spellings `carra`,
`MaxCarraMixture` and `n_mixures` are the exact spellings to use.

- `opt_maximize_cara_mixture` in
  [`carra_mixture.py`](../src/optimalportfolios/optimization/general/carra_mixture.py) takes
  `means` and `covars`, lists of component means and covariances, `probs`, `constraints`, and
  `carra=0.5`, and returns a weight array. It minimises $f(w)$ of Proposition 2 with SLSQP,
  function tolerance $10^{-8}$ and SciPy's default of 100 iterations, from `constraints.weights_0`
  or, without it, equal weights. `Constraints` compiles for SciPy the long-only bound, the
  exposure band as two inequalities, the box bounds and the group allocations; the
  [backend capability matrix](constraints.md#backend-capability-matrix) lists what it does not
  compile. `validate_scipy_solution` then checks the result as described in
  [solver numerics and outcomes](solver_numerics_and_outcomes.md): a solve that SciPy reports as
  not converged, non-finite weights, or a budget, box or group breach is rejected and replaced by
  `weights_0`, else the benchmark weights, else zeros. Caps of 30% on three assets cannot hold a
  fully invested portfolio, and without `weights_0` the example returns zeros.
- `wrapper_maximize_cara_mixture` takes the same inputs plus `tickers`, and an `optimiser_config`
  that defaults to `OptimiserConfig(apply_total_to_good_ratio=True)`. It calls the solver and
  returns a `pd.Series` indexed by `tickers`. It reads only `verbose` from the configuration and
  removes no assets.
- `rolling_maximize_cara_mixture` takes `prices`, `constraints` and `time_period`, which has no
  default, with the defaults `rebalancing_freq='QE'`, `roll_window=312`, `returns_freq='W-WED'`,
  `carra=0.5` and `n_components=3`. It computes log returns at `returns_freq` with
  `qis.to_returns`, and at the first return date on or after each `rebalancing_freq` period end
  fits `fit_gaussian_mixture` to the last `roll_window` returns, 312 weekly returns or six years by
  default. It drops the assets with any missing return in the window, annualises with the factor
  of `returns_freq`, drifts the previous weights to the date when `use_drifted_weights_0` is set,
  and solves through the wrapper. It always rescales the per-asset caps below one by the ratio of
  all to valid assets, whatever `apply_total_to_good_ratio` says: in the example, caps of 45%
  become 67.5% at a date where equities drop out, with the flag on or off. Dropped assets get zero
  weight, and `time_period` trims the result; pass `None` to keep every date. It returns a
  date-by-asset table of target weights.
- `opt_maximize_cara` takes `means`, `covar`, `carra=0.5`, optional `min_weights` and
  `max_weights`, and `is_exp=False`, and solves the one-component problem without a `Constraints`
  object: always long-only and fully invested, with the quadratic objective of Proposition 3 or,
  with `is_exp=True`, the exponential one, and function tolerance $10^{-12}$. It returns a weight
  array. When SLSQP does not converge it logs a warning and returns the equal-weight start,
  without validation.
- `fit_gaussian_mixture` in
  [`gaussian_mixture.py`](../src/optimalportfolios/utils/gaussian_mixture.py) is the fit described
  under the methodology. Its default `n_components=2` differs from the three of the rolling
  function.

The rolling dispatcher `compute_rolling_optimal_weights` routes
`PortfolioObjective.MAX_CARA_MIXTURE`, whose value is `'MaxCarraMixture'`, to
`rolling_maximize_cara_mixture`. It passes `time_period`, `returns_freq`, `rebalancing_freq`,
`carra`, `roll_window` and the configuration, and forwards `n_mixures`, default 3, as
`n_components`. It does not pass `covar_dict`, which the route ignores. Its own `roll_window`
defaults to 20, which the rolling function reads as 20 weekly returns, under five months: with
the dispatcher's defaults the first weights in the example appear on 6 July 2016, half a year
after the data start, instead of after six years. `backtest_rolling_optimal_portfolio` defaults
`roll_window` to 312. Pass it explicitly to the dispatcher.

The solver minimises the expected disutility itself, not its logarithm, and SLSQP's stopping
tolerance `ftol` is absolute. The scale of the objective grows like $e^{q_k(w)}$: at
$\gamma = 10$ in the example it is 844 at the equal-weight start and 0.81 at the optimum. `validate_scipy_solution` checks
feasibility, not optimality, and accepts even the unchanged equal-weight start when SciPy reports
success. Starting from the one-component closed form through `weights_0`, as the rolling function
starts from the drifted previous weights, the solve at $\gamma = 10$ returns the CVXPY optimum.

> **Pitfall.** Annualising turns weekly components into one-year regimes. The rolling function
> fits weekly returns and multiplies each component by 52, which adds the spread of the component
> means $52 \times 51$ times, 2,652 times, to the annual covariance (Proposition 5); with the
> monthly returns of the paper the factor is $12 \times 11 = 132$. In the example the fitted
> mixture gives crypto an annual volatility of about 290% against 68% from the same returns. Read
> as independent weekly regimes, the three-component mixture of the example would give crypto an
> annual volatility of 64% instead of 80%, and a weight of about 11% instead of 6.3% at
> $\gamma = 5$.

## Interpretation and limitations

- The package takes from Sepp (2023, Section 4.4) the objective: CARA utility of the portfolio
  value (equation `eq:l1`), the portfolio log return as a Gaussian mixture (`eq:l2`), and the
  closed form under the budget and bounds from zero to one (`eq:l3`). The defaults of the rolling
  function match the paper's CARA-3 configuration, three components, $\gamma = 0.5$, a six-year
  window and quarterly rebalancing, except the return frequency: the paper fits monthly returns
  and the package defaults to weekly returns. It does not inherit the paper's scikit-learn fit, its
  cross-validated choice of three components, the fixed 75% weight of its blended portfolios, which
  minimum and maximum weights can express, or its empirical results.
- The paper follows Ang, Morris and Savi (2023), who model Bitcoin with a two-state mixture and,
  as the paper reports, use $\gamma = 0.5$ because it reproduces a 60/40 equity-bond portfolio.
  With annual parameters, a value of this size leaves much of the work to the budget and the
  bounds: at $\gamma = 0.5$ the one-component portfolio of the example is held at the long-only
  bound on bonds, with 80.8% in crypto.
- Applied to a log return, CARA utility is a power utility of the gross return $G$:
  $-e^{-\gamma \ln G} = -G^{-\gamma}$, whose relative risk aversion, in Pratt's terms, is
  $1 + \gamma$. The rolling path takes $w^{\top} r$, the weighted sum of asset log returns, as the
  portfolio log return. For long-only, fully invested weights it is a lower bound, by the
  concavity of the logarithm.
- Propositions 5 and 6 make the return frequency a modelling choice. The same simulated prices
  fitted at monthly frequency give crypto an annual volatility of about 190%, against about 290%
  from the weekly fit.
- A mixture with $K$ full-covariance components in $N$ assets has
  $K (N + N (N + 1) / 2) + K - 1$ parameters, 29 for three components and three assets. EM finds a
  local maximum from one fixed start, and the order of the components carries no meaning unless
  `idx` sorts them.
- The route ignores `covar_dict` and dates its weights on the return grid, so its weights are not
  aligned with those of the covariance-based objectives, which solve at the keys of `covar_dict`.
- SLSQP supports only part of the `Constraints` policy; tracking-error, turnover and volatility
  limits are not compiled. See the [constraints page](constraints.md#backend-capability-matrix).

## See also

- [Choosing an objective](optimization_module_readme.md)
- [Minimum variance, quadratic utility and maximum Sharpe](mean_variance_objectives.md)
- [Strategic allocation: target return and target volatility](strategic_allocation_targets.md)
- [Portfolio constraints](constraints.md)
- [Covariance factorisation, solver outcomes and constraint residuals](solver_numerics_and_outcomes.md)
- [Conventions, notation and glossary](conventions.md)
- [Research papers and replication](research_papers.md)

## References

- Pratt, J. W. (1964). *Risk Aversion in the Small and in the Large*. Econometrica, 32(1/2),
  122–136. [DOI 10.2307/1913738](https://doi.org/10.2307/1913738). The absolute and relative risk
  aversion measures.
- Sepp, A. (2023). *Optimal Allocation to Cryptocurrencies in Diversified Portfolios*. Risk,
  October 2023. [SSRN 4217841](https://ssrn.com/abstract=4217841). Section 4.4 sets out the CARA
  objective under a Gaussian mixture and its CARA-3 configuration.
- Ang, A., Morris, T. and Savi, R. (2023). *Asset Allocation with Crypto: Application of
  Preferences for Positive Skewness*. The Journal of Alternative Investments, 25(4), 7–28.
  [DOI 10.3905/jai.2023.1.185](https://doi.org/10.3905/jai.2023.1.185).
- Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977). *Maximum Likelihood from Incomplete Data
  via the EM Algorithm*. Journal of the Royal Statistical Society: Series B, 39(1), 1–22.
  [DOI 10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x).
- Diamond, S. and Boyd, S. (2016). *CVXPY: A Python-Embedded Modeling Language for Convex
  Optimization*. Journal of Machine Learning Research, 17(83), 1–5.
  [JMLR](https://www.jmlr.org/papers/v17/15-408.html).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
