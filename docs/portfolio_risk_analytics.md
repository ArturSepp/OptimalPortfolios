---
myst:
  html_meta:
    description: >-
      Ex-ante portfolio risk in Python with optimalportfolios: portfolio volatility, Euler risk
      contributions with a proof, risk shares as weights times betas, benchmark-beta loadings
      as regression coefficients, ex-ante beta through time, and the hand-off to qis.RiskModel
      for tracking error and factor exposures, with a verified offline example.
---

# Ex-ante risk contributions, betas and the qis risk model

*Author: [Artur Sepp](https://github.com/ArturSepp)*

The ex-ante risk analytics are implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

Ex-ante risk is the risk of a set of weights under a covariance model, measured before any
return is realised. This page covers the risk analytics that `optimalportfolios` exports and
the ones it delegates to [qis](https://github.com/ArturSepp/QuantInvestStrats): the variance and
volatility of a portfolio, the Euler risk contribution of each asset, benchmark-beta loadings
and the ex-ante beta of a portfolio through time, and `qis.RiskModel`, which owns ex-ante
tracking error, factor exposures and marginal tracking-error contributions.

The page proves three results. Euler's identity: the risk contributions add up to the
volatility, because volatility is homogeneous of degree one in the weights. A risk share is the
capital weight times the asset's beta to the portfolio itself, which explains the risk shares of
the minimum-variance, equal-risk-contribution and maximum-diversification portfolios. And a
benchmark-beta loading is the slope of the regression of the asset's return on the benchmark
return. The worked example checks each result against a computation made a different way,
including a regression on simulated returns.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | None; the functions take a covariance and weights and sample no returns. The regression check of the example draws Gaussian weekly returns with the example covariance |
| Estimation grid | None; the covariance and factor data are inputs, fixed synthetic values in the examples, and no function estimates, resamples or annualises them |
| Rebalancing grid | The keys of the covariance dictionary: `build_risk_model` and `compute_benchmark_beta_loadings_ts` keep them, and `compute_ex_ante_beta_ts` carries each row of loadings forward to later weight dates |
| Covariance units | Any consistent units, used as supplied; volatility, contributions and tracking error are in its square-root units, annual in the examples. Risk shares and betas are unit-free |
| Expected returns | None |
| Weight state | Weights supplied by the caller as fractions of NAV, usually target weights; benchmark weights are fractions that sum to one |
| Solver | None for the analytics; the exhibit's portfolios use CVXPY with CLARABEL (minimum variance), cyclical coordinate descent (equal risk contribution) and SciPy SLSQP (maximum diversification) |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $\mathrm{MR}_i$ | Marginal risk of asset $i$, the derivative of $\sigma(w)$ with respect to $w_i$ |
| $\kappa_i$ | Risk share of asset $i$, $\mathrm{RC}_i / \sigma(w)$ |
| $c(v)$ | Betas of the assets to the portfolio $v$, $c(v) = \Sigma v / (v^{\top} \Sigma v)$ |
| $c$ | Benchmark-beta loadings $c(w^{\mathrm{bm}})$; the [constraints page](constraints.md#benchmark-beta) writes them $h$ |
| $r^{\mathrm{bm}}$ | Benchmark return $w^{\mathrm{bm}\top} r$ |
| $\beta^{\mathrm{bm}}$, $\nu$ | Factor loadings and residual variance of a benchmark index in a factor model |
| $\rho_i(w)$, $\mathrm{DR}$ | Correlation of asset $i$ with the portfolio; diversification ratio |

The covariance is a symmetric positive semi-definite matrix, labelled by the same assets on both
axes. Risk shares and betas divide by a portfolio variance, which must be positive. The results
hold for any weights; the corollaries on the risk-based portfolios assume long-only, fully
invested weights and no other constraint. The covariance and the weights are those available at
the decision date.

## Methodology

### Variance and volatility

The variance and volatility of weights $w$ are

$$
\sigma^2(w) = w^{\top} \Sigma w = \sum_{i=1}^{N} \sum_{j=1}^{N} w_i w_j \Sigma_{ij},
\qquad
\sigma(w) = \sqrt{w^{\top} \Sigma w} .
$$

`compute_portfolio_variance` returns the first and `compute_portfolio_vol` the second, in the
units of the covariance they receive: an annual covariance gives an annual volatility.

### Euler's identity for risk contributions

The marginal risk of asset $i$ is the derivative of the volatility with respect to its weight.
The gradient of $w^{\top} \Sigma w$ is $2 \Sigma w$, so

$$
\mathrm{MR}_i(w) = \frac{\partial \sigma(w)}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma(w)} .
$$

The risk contribution multiplies the marginal risk by the weight,
$\mathrm{RC}_i = w_i \mathrm{MR}_i$, and the risk share divides the contribution by the
volatility, $\kappa_i = \mathrm{RC}_i / \sigma(w)$.

**Proposition 1 (Euler's identity).** For weights with $\sigma(w) \gt 0$, the risk
contributions add up to the volatility and the risk shares add up to one:

$$
\sigma(w) = \sum_{i=1}^{N} w_i \frac{\partial \sigma(w)}{\partial w_i} = \sum_{i=1}^{N} \frac{w_i (\Sigma w)_i}{\sigma(w)},
\qquad
\sum_{i=1}^{N} \kappa_i = 1 .
$$

**Proof.** Volatility is positively homogeneous of degree one: for $\theta \gt 0$,
$\sigma(\theta w) = \sqrt{\theta^2 w^{\top} \Sigma w} = \theta \sigma(w)$. Differentiate both
sides with respect to $\theta$ at $\theta = 1$. By the chain rule the left side gives
$\sum_i w_i \partial \sigma(w) / \partial w_i$ and the right side gives $\sigma(w)$. Dividing
by $\sigma(w)$ gives the shares. $\square$

The proof uses nothing but homogeneity, so it holds for every risk measure that is homogeneous
of degree one; this is the Euler allocation principle (Tasche, 2007, revised 2008). For
volatility the identity can also be read off directly, since
$\sum_i w_i (\Sigma w)_i = \sigma^2(w)$.

Litterman (1996) reads the contributions as a map of the portfolio: the largest ones are its
hot spots. A contribution is negative when the asset's covariance with the portfolio is
negative, even for a positive weight; the position is then a hedge, and adding to it lowers
the volatility at the margin. A contribution is not the weight times the asset's own
volatility: for long-only weights those products add up to at least $\sigma(w)$, and the
difference is the diversification.

### Risk shares are weights times betas

For weights $v$ with $v^{\top} \Sigma v \gt 0$ define

$$
c(v) = \frac{\Sigma v}{v^{\top} \Sigma v} .
$$

Proposition 3 below shows that $c_i(v)$ is the beta of asset $i$ to the portfolio $v$.

**Proposition 2 (risk share as weight times beta).** The risk share of asset $i$ is its
capital weight times its beta to the portfolio itself:

$$
\kappa_i(w) = w_i c_i(w),
\qquad
\sum_{i=1}^{N} w_i c_i(w) = 1 .
$$

**Proof.** By Proposition 1, $\kappa_i = w_i (\Sigma w)_i / \sigma^2(w)$, and the ratio after
$w_i$ is the $i$-th entry of $c(w)$. The sum is Proposition 1 again. $\square$

For a positive weight, an asset takes a larger share of risk than of capital exactly when its
beta to the portfolio exceeds one. Three risk-based portfolios therefore have simple risk
shares, under long-only full investment and no other constraint:

- **Equal risk contribution.** Its risk shares are $1/N$ by construction, so
  $w_i = 1 / (N c_i(w))$: each weight is inversely proportional to the asset's beta to the
  portfolio (Roncalli, 2013).
- **Minimum variance.** Every held asset has beta one to the portfolio and every excluded asset
  at least one, so the risk shares equal the capital weights. *Proof.* The optimality
  conditions of minimising half the variance subject to $\sum_i w_i = 1$ and $w \geq 0$ are
  $(\Sigma w)_i = \eta + \xi_i$ with multipliers $\xi_i \geq 0$ and $\xi_i w_i = 0$. Multiplying
  by $w_i$ and summing gives $\eta = \sigma^2(w)$, so $c_i(w) = 1 + \xi_i / \sigma^2(w)$, which is
  one wherever $w_i \gt 0$. By Proposition 2 the risk share is then $w_i$. $\square$
- **Maximum diversification.** The risk share of each asset is its share of the weighted
  volatility, $w_i \sigma_i / \sum_j w_j \sigma_j$. *Proof.* Proposition 2 of the
  [maximum diversification page](maximum_diversification.md) gives every held asset the
  correlation $\rho_i(w) = 1 / \mathrm{DR}$ with the portfolio, where
  $\mathrm{DR} = \sum_j w_j \sigma_j / \sigma(w)$. Since $c_i(w) = \rho_i(w) \sigma_i / \sigma(w)$,
  the beta of a held asset is $\sigma_i / \sum_j w_j \sigma_j$; apply Proposition 2. $\square$

### Benchmark beta as a regression coefficient

Let $r$ be a random vector of asset returns with covariance $\Sigma$, and
$r^{\mathrm{bm}} = w^{\mathrm{bm}\top} r$ the return of the benchmark. The benchmark-beta
loadings are $c = c(w^{\mathrm{bm}})$, one per asset, and the ex-ante beta of a portfolio is
linear in its weights:

$$
c_i = \frac{(\Sigma w^{\mathrm{bm}})_i}{w^{\mathrm{bm}\top} \Sigma w^{\mathrm{bm}}},
\qquad
\text{beta of } w = c^{\top} w .
$$

**Proposition 3 (regression coefficient).** The loading $c_i$ is the slope of the
least-squares regression of $r_i$ on $r^{\mathrm{bm}}$ with an intercept. The slope for the
portfolio return $w^{\top} r$ is $c^{\top} w$, and the benchmark has beta one to itself,
$c^{\top} w^{\mathrm{bm}} = 1$.

**Proof.** The least-squares slope of a return $x$ on $r^{\mathrm{bm}}$ is the $q$ that, with an
intercept $p$, minimises $\mathrm{E}[(x - p - q r^{\mathrm{bm}})^2]$. Setting both derivatives
to zero gives $q = \mathrm{Cov}(x, r^{\mathrm{bm}}) / \mathrm{Var}(r^{\mathrm{bm}})$. Covariance is
bilinear, so $\mathrm{Cov}(r, w^{\mathrm{bm}\top} r) = \Sigma w^{\mathrm{bm}}$ and
$\mathrm{Var}(r^{\mathrm{bm}}) = w^{\mathrm{bm}\top} \Sigma w^{\mathrm{bm}}$; with $x = r_i$ the
slope is $c_i$. With $x = w^{\top} r$ the numerator is $w^{\top} \Sigma w^{\mathrm{bm}}$, so
the slope is $c^{\top} w$, which is one for $w = w^{\mathrm{bm}}$. $\square$

The proof needs finite second moments and no distributional assumption. It also shows that the
loadings refer to the benchmark return as given: multiplying $w^{\mathrm{bm}}$ by $k \gt 0$
multiplies the benchmark return by $k$ and divides every loading by $k$.

**Joint covariance.** When the benchmark constituents $C$ and the portfolio assets $A$ are
labels of one covariance matrix, the loadings are a slice of it:

$$
c_A = \frac{\Sigma_{A,C} w^{\mathrm{bm}}_C}{(w^{\mathrm{bm}}_C)^{\top} \Sigma_{C,C} w^{\mathrm{bm}}_C} .
$$

`compute_benchmark_beta_loadings_from_covar` computes this slice. The beta that an optimiser
constrains then comes from the same matrix as its tracking-error terms.

**Factor model.** With asset loadings $\beta$ of size $N \times M$, factor covariance
$\Sigma_F$, and a benchmark index with factor loadings $\beta^{\mathrm{bm}}$ and residual
variance $\nu$, `compute_benchmark_beta_loadings` returns

$$
c = \frac{\beta \Sigma_F \beta^{\mathrm{bm}}}{\beta^{\mathrm{bm}\top} \Sigma_F \beta^{\mathrm{bm}} + \nu} .
$$

This is Proposition 3 for the model covariance $\beta \Sigma_F \beta^{\top} + D$ when the
index's residual is uncorrelated with every asset residual. For a benchmark that holds the
portfolio assets that assumption fails: the covariance of the assets with the benchmark also
contains $D w^{\mathrm{bm}}$, which the formula omits, and the joint-covariance slice is the
consistent choice.

### Loadings through time

`compute_benchmark_beta_loadings_ts` computes the loadings at each covariance date $t_k$ and
stacks them by date. `compute_ex_ante_beta_ts` evaluates the beta of dated weights $w_t$ with
the loadings of the last covariance date on or before $t$:

$$
\text{beta at } t = c_{t_k}^{\top} w_t,
\qquad
t_k = \max \lbrace t_j : t_j \leq t \rbrace .
$$

Loadings are carried forward, never interpolated, so no weight date uses a later covariance.

### Ex-ante tracking error and the qis risk model

With active weights $d = w - w^{\mathrm{bm}}$, the ex-ante tracking error is
$\mathrm{TE}(w) = \sqrt{d^{\top} \Sigma d}$. Under a factor model
$\Sigma = \beta \Sigma_F \beta^{\top} + D$, the factor exposures of the portfolio are
$\beta^{\top} w$, and the tracking error splits into a factor and a residual part that add in
squares:

$$
\mathrm{TE}^2(w) = (\beta^{\top} d)^{\top} \Sigma_F (\beta^{\top} d) + d^{\top} D d .
$$

Tracking error is homogeneous of degree one in $d$, so Proposition 1 applies to active risk:
the marginal contributions $d_i (\Sigma d)_i / \mathrm{TE}(w)$ add up to the tracking error, and
each splits into a systematic and a residual part when $\Sigma d$ is split into
$\beta \Sigma_F \beta^{\top} d$ and $D d$.

These quantities belong to `qis.RiskModel`. The stack rule is stated in the repository's
[contributor guidance](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md):
"Never hand-roll `d' Σ d`, a beta ratio, or a TE decomposition." Ex-ante tracking error, factor
exposures, benchmark beta and marginal tracking error come from `qis.RiskModel`, built from
covariance-estimation output by `optimalportfolios.build_risk_model`, and a plain
`{date: covar}` dictionary gives a covariance-only model. Realised tracking error is
`qis.compute_ewma_realised_tracking_error`, and whole-sample tracking error and information
ratio are `qis.compute_te_ir_errors`. The canonical script of this page computes the explicit
products only as independent references for its assertions.

## Worked example

The canonical script of this page,
[`examples/docs/portfolio_risk_analytics.py`](../examples/docs/portfolio_risk_analytics.py),
runs offline and asserts every number quoted here against explicit matrix products, finite
differences, a regression on simulated returns and the optimality conditions of the risk-based
portfolios:

```console
python -m examples.docs.portfolio_risk_analytics
```

### A factor-model snapshot

Five assets load on three factors with annual volatilities of 6%, 16% and 20%, and carry
annual residual volatilities from 1% to 10%. The portfolio holds 30% in government bonds and
US equity, 15% in credit and emerging-market equity and 10% in gold. The benchmark holds 40%
in government bonds and 60% in US equity:

```python
TICKERS = ['Govt', 'Credit', 'US eq', 'EM eq', 'Gold']
FACTORS = ['Rates', 'Equity', 'Commodities']
FACTOR_VOLS = [0.06, 0.16, 0.20]  # annual
FACTOR_CORR = [[1.0, -0.2, 0.0],
               [-0.2, 1.0, 0.3],
               [0.0, 0.3, 1.0]]
LOADINGS = [[1.0, 0.0, 0.0],
            [0.8, 0.3, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.2, 0.3],
            [0.4, 0.1, 0.6]]
RESIDUAL_VOLS = [0.01, 0.03, 0.05, 0.10, 0.10]  # annual
PORTFOLIO = [0.30, 0.15, 0.30, 0.15, 0.10]
BENCHMARK = {'Govt': 0.40, 'US eq': 0.60}
```

The snapshot is a FactorLasso container, the same type that factor-covariance estimation
returns. Its covariance $\beta \Sigma_F \beta^{\top} + D$ gives asset volatilities from 6.1%
for government bonds to 24.0% for emerging-market equity:

```python
def factor_snapshot(factor_corr) -> CurrentFactorCovarData:
    """Return the factor-model snapshot of the page with the given factor correlation."""
    factor_covar = pd.DataFrame(np.outer(FACTOR_VOLS, FACTOR_VOLS) * np.array(factor_corr),
                                index=FACTORS, columns=FACTORS)
    return CurrentFactorCovarData(
        x_covar=factor_covar,
        y_betas=pd.DataFrame(LOADINGS, index=TICKERS, columns=FACTORS),
        y_variances=pd.DataFrame({'residual_var': np.square(RESIDUAL_VOLS)}, index=TICKERS))
```

### Volatility and Euler contributions

```python
snapshot = factor_snapshot(FACTOR_CORR)
covar = snapshot.get_y_covar()
weights = pd.Series(PORTFOLIO, index=TICKERS)
variance = op.compute_portfolio_variance(w=weights.to_numpy(), covar=covar.to_numpy())
vol = op.compute_portfolio_vol(covar=covar, weights=weights)
table = op.compute_portfolio_risk_contribution_outputs(weights=weights, clean_covar=covar)
print(table.round(4))
assert np.isclose(table['risk contribution'].sum(), vol, rtol=1e-14, atol=0.0)
```

The portfolio volatility is 9.61%. The contributions add up to it to floating-point precision,
as Proposition 1 requires, and the script checks the marginal risks against central finite
differences of the volatility. The last column anticipates the next block:

| Asset | Weight | Contribution (%) | Risk share | Beta to the portfolio |
|---|---:|---:|---:|---:|
| Govt | 0.30 | 0.21 | 0.021 | 0.07 |
| Credit | 0.15 | 0.75 | 0.078 | 0.52 |
| US eq | 0.30 | 4.55 | 0.473 | 1.58 |
| EM eq | 0.15 | 3.26 | 0.339 | 2.26 |
| Gold | 0.10 | 0.86 | 0.089 | 0.89 |

Government bonds hold 30% of the capital and carry 2.1% of the risk; US equity holds the same
capital and carries 47.3%. The weights times the asset volatilities add up to more than 1.3
times the portfolio volatility, so they are not the contributions.

### Risk shares as weights times betas

The package's benchmark-beta helper, given the portfolio itself as the benchmark, returns the
betas of the assets to the portfolio. Multiplied by the weights they give the risk shares, as
Proposition 2 states:

```python
beta_to_portfolio = op.compute_benchmark_beta_loadings_from_covar(
    covar=covar, benchmark_weights=weights, asset_tickers=TICKERS)
assert np.allclose(table['asset_rc_ratio'], weights * beta_to_portfolio, rtol=1e-12)
```

> **Insight.** A risk share is a capital weight times a beta to the portfolio itself. US
> equity has beta 1.58 to the portfolio, so its 30% of capital becomes 47.3% of risk;
> emerging-market equity has beta 2.26 and turns 15% into 33.9%. An asset carries more risk
> than capital exactly when that beta exceeds one.

### Four risk-based portfolios

The same covariance gives four long-only portfolios through the package's solvers. The risk
shares come from the package's risk table:

```python
long_only = op.Constraints(is_long_only=True)
minimum_variance, _ = op.wrapper_quadratic_optimisation(pd_covar=covar, constraints=long_only)
portfolios = {
    'Equal weight': pd.Series(1.0 / len(TICKERS), index=TICKERS),
    'Minimum variance': minimum_variance,
    'Equal risk contribution': op.wrapper_risk_budgeting(pd_covar=covar,
                                                         constraints=long_only),
    'Maximum diversification': op.wrapper_maximise_diversification(pd_covar=covar,
                                                                   constraints=long_only),
}
risk_shares = {name: op.compute_portfolio_risk_contribution_outputs(
    weights=portfolio, clean_covar=covar)['asset_rc_ratio']
    for name, portfolio in portfolios.items()}
```

Equal weights give emerging-market equity 42% of the risk and government bonds 1%. The
minimum-variance portfolio holds government bonds at 83%, US equity at 15% and gold at 2%, and
excludes credit and emerging-market equity, whose betas to the portfolio exceed one; its risk
shares equal its capital weights. Equal risk contribution gives each asset 20% of the risk with
45% of the capital in government bonds and 8% in emerging-market equity. Maximum
diversification holds government bonds at 67% of the capital and 41% of the risk, their share of
the weighted volatility. The last three properties are the corollaries of Proposition 2.

![Left: stacked capital weights of four portfolios on one five-asset covariance. Equal weight
holds 20% in each asset, minimum variance holds 83% in government bonds, equal risk
contribution 45% and maximum diversification 67%. Right: the risk shares of the same portfolios.
Equal weight has 42% of its risk in emerging-market equity; minimum variance has risk shares
equal to its weights; equal risk contribution has 20% of risk in each asset; maximum
diversification has 41% of its risk in government bonds.](images/risk_contributions_vs_weights.png)

*Figure: capital weights and Euler risk shares of the equal-weight, minimum-variance,
equal-risk-contribution and maximum-diversification portfolios of the example covariance.
Drawn by the `exhibit` function of the canonical script; the
[analytics gallery](analytics_gallery.md) lists its provenance.*

### Benchmark-beta loadings

```python
benchmark = pd.Series(BENCHMARK)
beta_loadings = op.compute_benchmark_beta_loadings_from_covar(
    covar=covar, benchmark_weights=benchmark, asset_tickers=TICKERS)
portfolio_beta = float(beta_loadings @ weights)
print(beta_loadings.round(3).to_dict(), round(portfolio_beta, 3))
```

The loadings equal the explicit slice of the covariance, and the benchmark's own beta is one.
The portfolio's beta is 0.92. Government bonds are 40% of the benchmark but have a beta of 0.03
to it: US equity dominates the benchmark's 9.9% volatility.

### A regression check

Proposition 3 says that each loading is a regression slope. The script draws 10,000 weekly
Gaussian returns, with a fixed seed, whose annual covariance is the example covariance, and
regresses each asset's return on the benchmark return by least squares with an intercept.
Betas do not depend on the scale of the covariance, so the weekly scale does not matter:

```python
returns = simulated_returns(covar, n_draws=N_DRAWS, seed=SEED)
benchmark_returns = returns[benchmark.index] @ benchmark
slopes, errors = ols_slopes(returns, benchmark_returns)
assert (np.abs(slopes - beta_loadings) < 4.0 * errors).all()
```

Every slope is within four standard errors of its loading, and no gap exceeds 0.01:

| Asset | Loading | Regression slope | Standard error |
|---|---:|---:|---:|
| Govt | 0.03 | 0.03 | 0.006 |
| Credit | 0.47 | 0.47 | 0.005 |
| US eq | 1.64 | 1.65 | 0.004 |
| EM eq | 1.97 | 1.97 | 0.014 |
| Gold | 0.51 | 0.50 | 0.015 |

The regression of the portfolio's return on the benchmark return gives the portfolio beta within
its own sampling error, and its slope is the weighted sum of the asset slopes.

### An index described by a factor model

For an external index known only through its factor loadings, 0.4 on rates and 0.6 on equity,
and a residual volatility of 2%:

```python
factor_covar = snapshot.x_covar
index_loadings = pd.Series(INDEX_LOADINGS, index=FACTORS)
index_beta = op.compute_benchmark_beta_loadings(
    asset_betas=snapshot.y_betas, benchmark_betas=index_loadings,
    factor_covar=factor_covar, benchmark_idio_var=INDEX_RESIDUAL_VOL ** 2)
```

The loadings equal, within $10^{-12}$, those of the joint-covariance slice when the index is
added to the model as a sixth asset with an independent residual. Applied instead to the
example benchmark, which holds government bonds and US equity, the factor variant omits their
residual covariance with the benchmark and understates their loadings; US equity's by 0.153.

### Loadings through time and ex-ante beta

Two quarter-end covariances differ in one factor correlation: in the second quarter rates and
equities move together, with correlation 0.3 instead of -0.2. The portfolio weights are the
same at six month ends:

```python
dates = pd.to_datetime(COVAR_DATES)
stressed = factor_snapshot(STRESSED_FACTOR_CORR).get_y_covar()
covar_dict = {dates[0]: covar, dates[1]: stressed}
loadings_ts = op.compute_benchmark_beta_loadings_ts(
    covar_dict=covar_dict, benchmark_weights=benchmark, asset_tickers=TICKERS)
monthly = pd.DataFrame([PORTFOLIO] * 6, columns=TICKERS,
                       index=pd.date_range('2024-02-29', periods=6, freq='ME'))
ex_ante_beta = op.compute_ex_ante_beta_ts(weights=monthly, beta_loadings=loadings_ts)
print(ex_ante_beta.round(3))
```

| Weight date | Loadings used | Ex-ante beta |
|---|---|---:|
| 2024-02-29 | None yet | 0.000 |
| 2024-03-31 to 2024-05-31 | 2024-03-29 | 0.920 |
| 2024-06-30 to 2024-07-31 | 2024-06-28 | 0.939 |

The positive rates-equity correlation raises the loading of government bonds from 0.03 to 0.27
and the portfolio's beta from 0.92 to 0.94. February precedes the first covariance date and
is reported as zero.

### Tracking error and exposures in qis

`build_risk_model` turns the snapshot into a `qis.RiskModel` with the factor block:

```python
date = dates[0]
risk_model = op.build_risk_model({date: snapshot})
tracking_error = risk_model.compute_tre_at_date(
    benchmark_weights=benchmark, portfolio_weights=weights, date=date)
exposures = risk_model.compute_exposures_at_date(portfolio_weights=weights, date=date)
split = risk_model.compute_tre_decomposition_at_date(
    benchmark_weights=benchmark, portfolio_weights=weights, date=date)
marginal = risk_model.compute_marginal_tre_at_date(
    benchmark_weights=benchmark, portfolio_weights=weights, date=date)
assert isinstance(risk_model, qis.RiskModel)
```

The tracking error is 3.19%, equal to $\sqrt{d^{\top} \Sigma d}$ computed explicitly. Its factor
part is 2.11% and its residual part 2.39%, and their squares add up to its square. The factor
exposures are 0.46 to rates, 0.535 to equity and 0.105 to commodities, the product
$\beta^{\top} w$. The marginal contributions add up to the tracking error: US equity
contributes the most, 1.43%, and the underweight in government bonds contributes a negative
amount, a hedge of the active risk. The risk model's benchmark beta is the 0.92 of the package
helper and its loadings are the same. A covariance-only model built from `{date: covar}` gives
the same tracking error and raises `ValueError` when asked for exposures.

## Implementation in optimalportfolios

The analytics are functions of the package root:

- `compute_portfolio_variance(w, covar)` returns $w^{\top} \Sigma w$ for NumPy arrays.
- `compute_portfolio_vol(covar, weights)` converts a DataFrame or Series to arrays and returns
  the square root of the variance. It matches weights to the covariance by position, not by
  label.
- `compute_portfolio_risk_contribution_outputs(weights, clean_covar, risk_budget=None)` selects
  the weights of the covariance's assets by label, takes the contributions from
  `qis.compute_portfolio_risk_contributions`, divides them by their sum, and returns the columns
  `weights`, `risk contribution`, `Risk Budget` and `asset_rc_ratio`. The budget column is zero
  when no `risk_budget` is given. `wrapper_risk_budgeting` returns this table when called with
  `detailed_output=True`.
- `compute_benchmark_beta_loadings_from_covar(covar, benchmark_weights, asset_tickers)` returns
  the joint-covariance slice, indexed by `asset_tickers`. It raises `KeyError` when a constituent
  is not in `covar`, and `ValueError` when the benchmark variance is not finite and positive or
  a loading is not finite.
- `compute_benchmark_beta_loadings(asset_betas, benchmark_betas, factor_covar, benchmark_idio_var=0.0)`
  returns the factor-model loadings. A factor of `factor_covar` missing from either set of
  loadings counts as a zero loading, and the variance and loadings are validated as for the
  slice.
- `compute_benchmark_beta_loadings_ts(covar_dict, benchmark_weights, asset_tickers)` applies the
  slice to each covariance of a dictionary and returns a table with one row per date, in date
  order.
- `compute_ex_ante_beta_ts(weights, beta_loadings)` aligns the loadings to the weight dates by
  carrying each row forward, treats missing weights as zero, and returns the series
  `ex_ante_beta`. It raises `ValueError` for non-finite loadings and for a weight column without
  loadings.
- `build_risk_model(covar_data)` returns a `qis.RiskModel`. From `RollingFactorCovarData` or a
  dictionary of `CurrentFactorCovarData` it passes the asset covariance with the full residual
  variance, the factor loadings, the factor covariance and the residual variances; from a
  dictionary of covariance DataFrames it builds a covariance-only model. Any other input raises
  `ValueError`.

The two loading helpers are also exported by `optimalportfolios.optimization.constraints`, where
`BenchmarkBetaConstraint.with_loadings` takes their result; see
[benchmark beta](constraints.md#benchmark-beta) on the constraints page. On a `qis.RiskModel`,
`compute_tre_at_date`, `compute_tre_history`, `compute_exposures_at_date`,
`compute_tre_decomposition_at_date`, `compute_marginal_tre_at_date`,
`compute_benchmark_beta_at_date`, `compute_benchmark_beta_history` and
`compute_benchmark_beta_loadings_at_date` provide the risk-model quantities of the methodology;
they use the covariance of an exact grid date and apply no annualisation.

> **Pitfall.** Pass benchmark weights as fractions that sum to one. Multiplying them by $k$
> divides every loading by $k$, so weights in percent turn the portfolio beta of 0.92 into
> 0.0092, while the benchmark's own beta stays one. The docstring of
> `compute_benchmark_beta_loadings_from_covar` says the weights need not sum to one because the
> ratio normalises; that holds only for the benchmark's beta to itself.

## Interpretation and limitations

- The quantities are ex-ante: they describe the supplied covariance, not realised risk. Realised
  contributions, betas and tracking error differ by estimation error and by changes in the
  covariance; realised tracking error is a qis calculation on returns.
- The package takes the language of hot spots and hedges from Litterman (1996), but it computes
  only volatility-based Euler contributions from the supplied covariance: it does not compute
  value at risk, best hedges, implied views or trade recommendations. From Roncalli (2013) it
  takes the volatility risk measure only; risk budgeting with other risk measures, such as
  expected shortfall, is not implemented.
- `compute_portfolio_vol` aligns by position. Reindex the weights to the covariance's labels
  before calling it; the script shows that reversed labels give a different number.
- `compute_portfolio_risk_contribution_outputs` drops, without a warning, weights of assets that
  are not in the covariance, raises `KeyError` when an asset of the covariance has no weight, and
  returns undefined (NaN) shares for a portfolio without risk.
- Risk shares can be negative, and larger than one, for hedged or long-short portfolios, even
  with nonnegative weights.
- The factor-model loadings assume that the benchmark's residual is independent of the asset
  residuals. For a benchmark that holds the assets, use the joint-covariance slice.
- `compute_ex_ante_beta_ts` reports a beta of zero, not a missing value, for weight dates before
  the first covariance date, and it carries each row of loadings forward until the next one.
- `qis.RiskModel` requires an exact covariance date and raises `KeyError` for any other date. Its
  history methods select dated weights as of each covariance date and give zero weights before
  the first weight date.

## See also

- [Risk budgeting](risk_budgeting.md)
- [Maximum diversification](maximum_diversification.md)
- [Minimum tracking error](minimum_tracking_error.md)
- [Portfolio constraints](constraints.md)
- [Rolling factor covariance from CSV](rolling_factor_covar_from_csv.md)
- [Conventions, notation and glossary](conventions.md)
- [qis: portfolio risk and Euler contributions](https://quantinveststrats.readthedocs.io/en/latest/risk_contributions.html)
- [qis: tracking error and benchmark-relative risk](https://quantinveststrats.readthedocs.io/en/latest/tracking_error_and_risk.html)

## References

- Litterman, R. (1996). *Hot Spots and Hedges*. The Journal of Portfolio Management, 23(5),
  special issue, 52–75. [DOI 10.3905/jpm.1996.052](https://doi.org/10.3905/jpm.1996.052). First
  issued in the Goldman Sachs Risk Management Series, October 1996.
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting*. Chapman and Hall/CRC
  Financial Mathematics Series. [DOI 10.1201/b15151](https://doi.org/10.1201/b15151).
- Tasche, D. (2007; revised 2008). *Capital Allocation to Business Units and Sub-Portfolios:
  the Euler Principle*. [arXiv:0708.2542](https://arxiv.org/abs/0708.2542).
- Choueifaty, Y. and Coignard, Y. (2008). *Toward Maximum Diversification*. The Journal of
  Portfolio Management, 35(1), 40–51.
  [DOI 10.3905/jpm.2008.35.1.40](https://doi.org/10.3905/jpm.2008.35.1.40).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
