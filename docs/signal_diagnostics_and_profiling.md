---
myst:
  html_meta:
    description: >-
      Signal diagnostics and alpha-rank portfolios in Python with optimalportfolios: the rank
      information coefficient and its stability, quantile portfolios and their spread,
      per-component diagnostics of AlphasData, what the package adds to the qis diagnostics and
      backtester, and why ranking power is not value in an optimiser, with a verified offline
      example.
---

# Signal diagnostics and alpha-rank portfolios

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Signal diagnostics and alpha-rank portfolios are implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

A score panel ranks assets at each date; the [alpha signals](alphas_module_readme.md) page
builds such panels. Before a score enters an optimiser as an alpha, two questions can be answered
without any optimiser. Does the score order the returns that follow? The rank information
coefficient (IC), the rank correlation of the score with the next return across assets, answers
it date by date, and its time series shows how stable the answer is. Does the order survive in
portfolios? Quantile portfolios, equal-weight baskets of the assets in each slice of the ranking,
such as each fifth, answer it in return units, and the spread between the top and bottom baskets
summarises it.

The package provides both evaluations as thin layers over
[qis](https://github.com/ArturSepp/QuantInvestStrats):

- The **rank profiler** forms equal-weight targets from the highest scores with
  `compute_top_quantile_equal_weights`, simulates them with the qis holdings backtester in
  `backtest_alpha_rank_portfolio`, next to an equal-weight benchmark, and reports them with
  `compute_alpha_rank_analysis_table` and `generate_alpha_profile_report`.
- The **diagnostics adapters** pass a score panel, or the score fields of an `AlphasData`, to
  `qis.estimate_signal_diagnostics`. qis owns the statistics: the pairing of each score with the
  return that follows, the per-date normalisation of returns, the pooled regression and the ICs.
  The [qis signal-diagnostics page](https://quantinveststrats.readthedocs.io/en/stable/signal_diagnostics.html)
  documents them.

This page states what the package adds on top of qis, proves the rank IC and the quantile returns
of a Gaussian score, and checks both on a synthetic score whose IC is known. It ends with the
difference between ranking power and value in an optimiser: a rank diagnostic cannot see the
scale of a score, and an optimiser responds to nothing else.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Diagnostics: the returns supplied in `asset_returns_dict`, log returns by default (`is_log_returns=True`), so a horizon of $h$ periods sums $h$ returns. Rank portfolios: NAVs from simple price ratios in the qis backtester. The example uses monthly log returns |
| Estimation grid | Each asset's native grid, the key of its frame in `asset_returns_dict`; qis resamples the score to it with the last value of each period. Month ends (`'ME'`) in the example: 240 returns from January 2006 to December 2025 |
| Rebalancing grid | Rank portfolios: score dates sampled with `asfreq(rebalancing_freq, method='ffill')`, default quarter ends (`'QE'`); month ends in most of the example. Diagnostics: one pair every $h$ native periods, without overlap |
| Covariance units | None in the diagnostics and the rank portfolios. The optimiser comparison of the example uses an annual diagonal covariance of 20% volatilities with a tracking-error cap in the same units |
| Expected returns | None: a score is a ranking characteristic, not an expected return; the methodology derives the scale that turns it into one |
| Weight state | Target weights: equal weights on the top-ranked assets at each rebalancing date, traded by qis at that date's prices without lag or costs by default, then held as units, so realised weights drift |
| Solver | None; CVXPY with CLARABEL in the optimiser comparison of the example |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $z_{i,t}$ | Score of asset $i$ known at date $t$ |
| $r_{i,t}$ | Log return of asset $i$ over $(t-1, t]$, the period after the score $z_{i,t-1}$ |
| $n_t$ | Number of assets with a finite score and return at date $t$ |
| $\mathrm{IC}_t$ | Rank information coefficient at date $t$ |
| $\overline{\mathrm{IC}}$, $s(\mathrm{IC})$, $T$ | Mean and sample standard deviation of the per-date IC over $T$ dates |
| $\mathrm{IC}^{\mathrm{P}}$ | Pearson information coefficient: the correlation of score and return |
| $q$, $K$ | Fraction held by the top-quantile rule (`quantile`); number of quantile baskets |
| $Q_k$ | The $k$-th quantile basket, counted from the top |
| $\rho$, $\varepsilon_{i,t}$ | Correlation of the score with the next return's shock $\varepsilon_{i,t}$ in the Gaussian model |
| $\Phi$, $\varphi$ | Standard normal distribution and density functions |

A score is dated when it becomes known, its formation date, and a higher score means a more
attractive asset. Scores, prices and returns carry the same asset labels. The diagnostics use
realised future returns: they evaluate a score after the fact and are not inputs available at
formation.

## Methodology

### Pairing a score with the return that follows

Both tools judge the score dated $t-1$ by the return over $(t-1, t]$, through different
mechanisms.

- The diagnostics pair $z_{i,t-1}$ with $r_{i,t}$, or at horizon $h$ with the sum of the $h$
  returns from $r_{i,t}$ on, and keep one pair every $h$ native periods. qis applies the
  one-period lag itself, so the caller passes scores dated at formation and does not shift them.
- The rank profiler does not lag. The target formed from the scores dated $t-1$ carries that date,
  and qis trades it at the first price on or after $t-1$, with `weight_implementation_lag` left at
  zero, and holds the units until the next target. The profile therefore assumes a trade at the
  price that the score could already use; [rolling backtests](rolling_backtests.md#information-and-implementation-clocks)
  describe how a lag moves the trade.

### The rank information coefficient

At date $t$ the rank IC is the Spearman correlation across the $n_t$ assets of the score and the
return that follows,

$$
\mathrm{IC}_t = \operatorname{corr}\left(\operatorname{rank} z_{\cdot,t-1}, \operatorname{rank} r_{\cdot,t}\right) .
$$

qis computes it on returns demeaned across assets and divided by their cross-sectional standard
deviation at each date. That map is increasing and affine, so it changes no rank, and a return
common to all assets drops out; the
[qis page](https://quantinveststrats.readthedocs.io/en/stable/signal_diagnostics.html#information-coefficients)
proves the invariance. The rank IC depends on the score only through its order: any strictly
increasing transform of the scores gives the same IC at every date.

Over $T$ dates, `qis.estimate_ic_ir` reports the mean $\overline{\mathrm{IC}}$ (`mean_IC`), the
standard deviation $s(\mathrm{IC})$ (`std_IC`), the t-statistic
$\overline{\mathrm{IC}} \sqrt{T} / s(\mathrm{IC})$ (`t_stat`) and the share of dates with a
positive IC (`hit_rate`). The standard error of the mean IC is $s(\mathrm{IC}) / \sqrt{T}$.
Stability is the size of $s(\mathrm{IC})$ against $1/\sqrt{n_t - 1}$, the dispersion that
sampling alone gives a small, constant per-date IC measured on $n_t$ assets; qis derives it, and the
[IC information ratio](https://quantinveststrats.readthedocs.io/en/stable/signal_diagnostics.html#the-ic-information-ratio)
built on it. A larger dispersion means that the skill of the score changes over time. qis also
pools all pairs into one regression slope (`beta`) and pooled Pearson and Spearman ICs
(`IC_pearson`, `IC_spearman`); its page states their degrees of freedom and why the pooled and
the per-date summaries can differ.

**Proposition 1 (rank IC of a Gaussian score).** Let the log returns at date $t$ be
$r_{i,t} = m_t + \sigma \varepsilon_{i,t}$ with a term $m_t$ common to all assets and
$\sigma \gt 0$, and let the pairs $(\varepsilon_{i,t}, z_{i,t-1})$ be independent across assets
and standard bivariate normal with correlation $\rho$. Then the correlation of score and return
across assets is $\mathrm{IC}^{\mathrm{P}} = \rho$, and their rank correlation is

$$
\rho_S = \frac{6}{\pi} \arcsin\frac{\rho}{2} .
$$

For $\rho = 0.1$, $\rho_S = 0.0955$.

**Proof.** The common term and the positive scale change no rank, so take $r = \varepsilon$. The
rank correlation of two continuous variables is the Pearson correlation of their distribution
functions, here $\Phi(\varepsilon)$ and $\Phi(z)$. Both are uniform, with mean $1/2$ and variance
$1/12$, so $\rho_S = 12 \mathbb{E}[\Phi(\varepsilon) \Phi(z)] - 3$. Let $\xi$ and $\zeta$ be
standard normal, independent of each other and of the pair. Then $\Phi(\varepsilon) \Phi(z)$ is the
conditional probability that $\xi \leq \varepsilon$ and $\zeta \leq z$, and its expectation is the
probability that $\varepsilon - \xi$ and $z - \zeta$ are both non-negative. These two are bivariate
normal with variances 2 and covariance $\rho$, so their correlation is $\rho/2$, and the quadrant
probability of a centred bivariate normal pair with correlation $\kappa$ is
$1/4 + \arcsin(\kappa)/(2\pi)$. Substituting gives $\rho_S$. $\square$

Two consequences follow from the same formula. At horizon $h$, a score that predicts only the
next period's shock has correlation $\rho/\sqrt{h}$ with the sum of $h$ shocks, so its rank IC is
$(6/\pi) \arcsin(\rho/(2\sqrt{h}))$, and the horizon of a signal matters; see
[signal horizons](mixed_frequency_data.md#signal-horizons) for spans and cadences. And the average
$(z + \eta)/\sqrt{2}$ of the score with an independent standard normal noise score $\eta$ has
correlation $\rho/\sqrt{2}$, so it dilutes the rank IC to $(6/\pi) \arcsin(\rho/(2\sqrt{2}))$.

### Quantile portfolios

At each score date, `compute_top_quantile_equal_weights` takes the assets with a non-missing
score and a non-missing price, holds the $\lceil q n_t \rceil$ highest scores with equal weights,
and gives the others zero; ties are broken by the order of the price columns, and a date with no
eligible asset has all weights zero. The package forms only this top basket. With $K$ quantiles,
the $k$-th basket from the top is the top $k/K$ minus the top $(k-1)/K$, and when $K$ divides
$n_t$ each basket holds $n_t / K$ assets. The spread is the return of $Q_1$ minus that of $Q_K$.

In the qis backtest a target is traded at its date and then held as units. If a basket $S$ of
$\lvert S \rvert$ assets is bought with equal weights at $\tau$ and held without costs until
$\tau'$, the units are $u_i = V_{\tau} / (\lvert S \rvert P_{i,\tau})$, no cash remains, and the NAV
grows by the mean price ratio of the basket, $V_{\tau'} / V_{\tau} = \sum_{i \in S} P_{i,\tau'} / (\lvert S \rvert P_{i,\tau})$.
In between, the realised weights $u_i P_{i,t} / V_t$ drift with prices. With monthly rebalancing
and monthly prices, a basket's return in each month is therefore the equal-weight mean simple
return of its members.

**Proposition 2 (quantile returns of a Gaussian score).** In the model of Proposition 1, let
$c_k = \Phi^{-1}(1 - k/K)$ for $k = 1, \dots, K - 1$, with $c_0 = +\infty$ and
$c_K = -\infty$. An asset whose score lies in the $k$-th bucket from the top,
$c_k \lt z_{i,t-1} \leq c_{k-1}$, has the expected log return

$$
\mathbb{E}\left[r_{i,t} \mid c_k \lt z_{i,t-1} \leq c_{k-1}\right] = m_t + \rho \sigma K \left(\varphi(c_k) - \varphi(c_{k-1})\right) ,
$$

which falls strictly from the top bucket to the bottom one. The expected spread between the top and
bottom buckets is $2 \rho \sigma K \varphi(c_1)$. For $K = 5$ the bucket means of the score are
1.400, 0.532, 0, −0.532 and −1.400, and the spread is $2.80 \rho \sigma$.

**Proof.** Write $\varepsilon = \rho z + \sqrt{1 - \rho^2} \xi$ with $\xi$ standard normal and
independent of $z$, so that $\mathbb{E}[r \mid z] = m_t + \rho \sigma z$. The mean of a standard
normal variable restricted to the interval between $c_k$ and $c_{k-1}$ is the difference of the
densities at the endpoints divided by the probability $1/K$ of the interval, which gives the
formula. Each bucket mean lies inside its own interval, and the intervals are ordered, so the means
fall strictly with $k$. The top and bottom buckets are mirror images, which gives the spread.
$\square$

A sample basket is formed from ranks, not from the population cutoffs, so the proposition holds
for its expected return only approximately; the example checks it within two standard errors.

### Ranking power and value in an optimiser

The IC and the quantile portfolios use the order of the scores. An optimiser uses their size.
For a score standardised across assets and returns with cross-sectional volatility $\sigma$, the
least-squares forecast of the return from the score has the slope
$\operatorname{cov}(r, z) / \operatorname{var}(z) = \mathrm{IC}^{\mathrm{P}} \sigma$. In the
Gaussian model this forecast is the conditional mean,

$$
\mathbb{E}\left[r_{i,t} \mid z_{i,t-1}\right] - m_t = \mathrm{IC}^{\mathrm{P}} \sigma z_{i,t-1} .
$$

This is the rule of Grinold and Kahn (2000) that an alpha is volatility times IC times score. It
is the Pearson IC, not the rank IC, that sets the scale, and the Pearson IC changes under an
increasing transform that the rank IC ignores. For the cube of a standard normal score,
$\mathbb{E}[z^3 \varepsilon] = \rho \mathbb{E}[z^4] = 3\rho$ and $\mathbb{E}[z^6] = 15$, so the
correlation of $z^3$ with the shock is $3\rho/\sqrt{15} \approx 0.775 \rho$ and the slope of the
shock on $z^3$ is $3\rho/15 = \rho/5$.

The optimiser then weighs the alphas against risk. The tactical objective of
`wrapper_maximise_alpha_over_tre` maximises $\alpha^{\top} d$ over the active weights $d$ under a
hard tracking-error cap $\mathrm{TE}(w) \leq \theta$ and full investment, so that the active
weights sum to zero. For a diagonal covariance $\sigma_0^2 I$ and no binding weight bound, subtract
the average alpha $\bar\alpha$ from every entry, which leaves $\alpha^{\top} d$ unchanged; the
Cauchy–Schwarz inequality then gives

$$
\alpha^{\top} d = (\alpha - \bar\alpha)^{\top} d \leq \lVert \alpha - \bar\alpha \rVert \frac{\theta}{\sigma_0} ,
$$

with equality at $d^{\star} = \theta (\alpha - \bar\alpha) / (\sigma_0 \lVert \alpha - \bar\alpha \rVert)$.
The active weights are proportional to the demeaned alphas. They do not change when the alphas
are multiplied by a positive number, and they do change under a nonlinear increasing transform,
which leaves every rank and every quantile basket as it was. With correlated assets the
covariance enters too: assets with high scores that move together share one risk budget, which
no rank statistic measures.

### Per-component diagnostics

`AlphasData` holds a combined score in `alpha_scores` next to optional component scores.
`signal_diagnostics_panel` lists the populated score fields, `run_signal_diagnostics_per_component`
runs the qis diagnostics once per field, and `compare_signal_diagnostics` stacks their pooled
rows into one table. A combined score's IC does not show which component carries the
information; the component rows do.

## Worked example

The canonical script of this page,
[`examples/docs/signal_diagnostics_and_profiling.py`](../examples/docs/signal_diagnostics_and_profiling.py),
runs offline and asserts every number and property stated here against a reference computed a
different way:

```console
python -m examples.docs.signal_diagnostics_and_profiling
```

### A score with a known information coefficient

The script's `synthetic_panel` draws 500 assets over 240 months, January 2006 to December 2025.
Each monthly log return is a drift of 0.5%, a market factor with 4% volatility common to all
assets, and an asset-specific shock with volatility $\sigma = 0.05$. The score of each asset at a
month end is standard normal and correlated $\rho = 0.1$ with the shock of the following month.
Each pair is drawn with the Cholesky factor of its correlation matrix, so the draws are the same
on every platform:

```python
rng = np.random.default_rng(SEED)
dates = pd.date_range(FIRST_DATE, periods=MONTHS + 1, freq='ME')
tickers = [f'A{i:03d}' for i in range(N_ASSETS)]
# Each (shock, score) pair is standard normal with correlation RHO: the Cholesky rule.
pairs = rng.standard_normal(((MONTHS + 1) * N_ASSETS, 2)) @ np.linalg.cholesky(
    np.array([[1.0, RHO], [RHO, 1.0]])).T
shocks, signal = (pairs[:, j].reshape(MONTHS + 1, N_ASSETS) for j in (0, 1))
market = MARKET_VOL * rng.standard_normal((MONTHS, 1))
# The score dated t is paired with the shock of the return over the following month;
# the score at the last date belongs to a month beyond the sample.
log_returns = pd.DataFrame(MU + market + VOL * shocks[:-1], index=dates[1:],
                           columns=tickers)
prices = 100.0 * np.exp(pd.concat([pd.DataFrame(0.0, index=dates[:1], columns=tickers),
                                   log_returns]).cumsum())
scores = pd.DataFrame(signal, index=dates, columns=tickers)
```

The scores are dated at formation and never shifted: the diagnostics apply the lag.

### The rank IC and its stability

`run_signal_diagnostics` passes the panel to qis at horizons of one and three months:

```python
diagnostics = alphas.run_signal_diagnostics(
    asset_returns_dict={'ME': log_returns}, signal=scores, horizons=(1, 3))
ic_summary = qis.estimate_ic_ir(diagnostics)
print(diagnostics.pooled_universe[['n', 'beta', 'IC_pearson', 'IC_spearman']].round(4))
print(ic_summary[['n_dates', 'mean_IC', 'std_IC', 't_stat', 'hit_rate']].round(4))
```

Each pair holds the score of one month end and the log return over the following month, as the
script checks for all 120,000 pairs. Over the 240 months the mean rank IC is 0.0966, less than half
a standard error of 0.0029 from the value 0.0955 of Proposition 1, with a t-statistic above 30. Its
standard deviation is 0.0457, close to the sampling dispersion $1/\sqrt{499} = 0.0448$ of a
per-date IC on 500 assets: the skill of this score does not change over time. The IC is negative
in 2 of the 240 months, a hit rate of 99%, and its 12-month mean stays between 0.06 and 0.14.

At three months the IC falls to 0.051, against $(6/\pi) \arcsin(\rho/(2\sqrt{3})) = 0.055$: the
score predicts one month, and the two further months only add noise. The pooled Pearson IC and
the pooled slope are both 0.102, near $\rho$, because the score and the normalised returns both
have unit dispersion. The script also checks that each per-date IC of `qis.compute_ic_timeseries`
equals a Spearman correlation computed from pandas ranks, before and after demeaning the returns
across assets, and that qis called directly returns the same tables.

### Quintile portfolios

The profiler runs the top quintile, `quantile=0.2`, against the equal-weight benchmark with a
rebalancing at every month end. The window of target dates ends at the penultimate month end, so
the last target is formed on 30 November 2025 and held through December:

```python
window = qis.TimePeriod(prices.index[0], prices.index[-2])
top = alphas.backtest_alpha_rank_portfolio(
    prices=prices, alpha_scores={'Top quintile': scores}, quantile=0.2,
    rebalancing_freq='ME', time_period=window)
print(alphas.compute_alpha_rank_analysis_table(top).round(3))
```

The top quintile earns about 19% a year against 9% for the benchmark. Held as units and
rebalanced monthly, its NAV return in every month equals the equal-weight simple return of the
100 assets with the highest scores at the previous month end, which the script recomputes from
prices. Its two-sided turnover of about 19 a year is close to $12 \times 2 \times (1 - 0.2) = 19.2$:
the scores are independent from month to month, so about four fifths of the basket is sold and
replaced at every rebalancing.

The five quintile baskets come from the same rule. The top $k/5$ minus the top $(k-1)/5$ gives
the $k$-th basket, and each basket enters the profiler as a score panel masked to its members,
with `quantile=1.0`:

```python
tops = [alphas.compute_top_quantile_equal_weights(scores, prices, quantile=k / QUANTILES) > 0
        for k in range(1, QUANTILES + 1)]
panels = {'Q1': scores.where(tops[0])}
panels.update({f'Q{k + 1}': scores.where(tops[k] & ~tops[k - 1])
               for k in range(1, QUANTILES)})
quintiles = alphas.backtest_alpha_rank_portfolio(
    prices=prices, alpha_scores=panels, quantile=1.0, rebalancing_freq='ME',
    time_period=window)
navs = pd.concat([leg.get_portfolio_nav() for leg in quintiles.portfolio_datas], axis=1)
print(navs.iloc[-1].round(1))
```

Each basket holds 100 assets at every date, and the Q1 leg is the top-quintile leg above. From a
start of 100 the NAVs end in order, from about 3,060 for Q1 to 103 for Q5, and the mean monthly
returns fall from Q1 to Q5 with every adjacent gap larger than five standard errors. In log
returns the spread of Q1 over Q5 averages 1.41% a month, against 1.40% from Proposition 2, and
each basket's return in excess of the cross-sectional mean lies within two standard errors of
$\rho \sigma$ times its bucket mean.

![Left: NAVs on a log scale of five quintile portfolios sorted by a synthetic score from 2006 to
2025; they fan out in order from a start of 100, the top quintile ending near 3,060 and the bottom
quintile near 103. Right: the monthly rank IC of the score with next-month returns as grey bars,
negative in only two months, with its 12-month mean moving between 0.06 and 0.14 around the dashed
population value 0.0955.](images/alpha_rank_quantiles.png)

*Figure: quintile portfolios and the rank IC of a score correlated 0.1 with next-month return
shocks, for 500 synthetic assets over 240 months. Drawn by the `exhibit` function of the canonical
script; the [analytics gallery](analytics_gallery.md) lists its provenance.*

### A quarterly profile through an adapter

The five `profile_*` adapters build a score with their signal constructor and call the core with
their own defaults, which include quarter-end rebalancing. Classic momentum on the same prices,
with the top fifth:

```python
quarterly = alphas.profile_classic_momentum(prices=prices, quantile=0.2)
realised = quarterly.portfolio_datas[0].weights
print(realised.loc['2024-12-31':'2025-03-31'].max(axis=1).round(5))
```

The leg is named `classic_momentum`, the value of `ProfileSignal.CLASSIC_MOMENTUM`, and equals the
core run on the score of `compute_classic_momentum_alpha`. Trades happen only at quarter ends. The
largest realised weight is 1% after the trade on 31 December 2024, rises above 1.1% and then 1.2%
at the next two month ends while the units are held, and is 1% again after the trade on
31 March 2025. Over each quarter the NAV grows by the mean price ratio of the selected assets. The
first 13 month ends have no momentum score, so the leg holds cash at 100 until the first quarter
end with one, 31 March 2007.

### Per-component diagnostics of AlphasData

The combined `alpha_scores` below averages the informative score with an independent standard
normal panel. The two components sit in the `momentum_score` and `beta_score` fields, used here
only as labels:

```python
noise = pd.DataFrame(np.random.default_rng(NOISE_SEED).standard_normal(scores.shape),
                     index=scores.index, columns=scores.columns)
data = alphas.AlphasData(alpha_scores=(scores + noise) / np.sqrt(2.0),
                         momentum_score=scores, beta_score=noise)
components = alphas.run_signal_diagnostics_per_component(
    asset_returns_dict={'ME': log_returns}, alphas_data=data, horizons=(1,))
comparison = alphas.compare_signal_diagnostics(components, horizon='1')
print(comparison[['n', 'beta', 'IC_pearson', 'IC_spearman']].round(4))
```

`signal_diagnostics_panel` lists the three populated fields in the module's fixed order, and each
component row equals a `run_signal_diagnostics` call on that field. The informative component
keeps the pooled row of the score, with the pooled rank IC 0.0968. The combined score has a
pooled rank IC of 0.069, against $(6/\pi) \arcsin(\rho/(2\sqrt{2})) = 0.068$, and the noise
component's mean IC is within two standard errors of zero. The combined IC alone would not
reveal that one component carries all the information.

### Ranking power against value in an optimiser

Cubing the score keeps every rank:

```python
cubed = alphas.run_signal_diagnostics(
    asset_returns_dict={'ME': log_returns}, signal=scores ** 3, horizons=(1,))
print(pd.concat({'score': diagnostics.pooled_universe.loc['1'],
                 'cubed score': cubed.pooled_universe.loc['1']}, axis=1)
      .loc[['beta', 'IC_pearson', 'IC_spearman']].round(4))
```

Every monthly rank IC and every quintile basket is unchanged. The pooled Pearson IC falls from
0.10 to 0.08, near $3\rho/\sqrt{15} = 0.077$, and the pooled slope from 0.10 to 0.02, within two
standard errors of $3\rho/15$. An optimiser sees the difference. For eight assets with 20%
volatility and no correlation, an equal-weight benchmark, a hard tracking-error cap of 2% and the
scores 1.5 down to −1.5, `wrapper_maximise_alpha_over_tre` gives:

```python
probe = pd.Series([1.5, 1.0, 0.5, 0.2, -0.2, -0.5, -1.0, -1.5], index=list('ABCDEFGH'))
covar = pd.DataFrame(np.diag(np.full(8, 0.2 ** 2)), index=probe.index, columns=probe.index)
benchmark = pd.Series(1.0 / 8, index=probe.index)
te_cap = op.Constraints(is_long_only=True, benchmark_weights=benchmark,
                        tracking_err_vol_constraint=0.02)
views = {'score': probe, '10 x score': 10.0 * probe, 'cubed score': probe ** 3}
active = pd.DataFrame({name: op.wrapper_maximise_alpha_over_tre(
    covar, alpha, benchmark, te_cap)[0] - benchmark for name, alpha in views.items()})
print(active.round(4))
```

The active weights equal the Cauchy–Schwarz solution $d^{\star}$ within $10^{-5}$ and use the whole
2% budget. Ten times the score gives the same weights. The top-quantile targets of all three
alphas are the same two assets, but the cube puts 75% of the absolute active weight on the two
extreme assets instead of 47%.

> **Insight.** A rank diagnostic cannot tell a score from an increasing transform of it. Cubing
> the score leaves every monthly rank IC and every quintile basket unchanged, yet it cuts the
> pooled slope from 0.10 to 0.02, and under a tracking-error cap it moves the share of active
> weight on the two extreme assets from 47% to 75%.

## Implementation in optimalportfolios

### The rank profiler

- `compute_top_quantile_equal_weights(alpha_scores, prices, quantile=1/3)` returns a date-by-asset
  DataFrame of target weights on the score dates. It reindexes the score columns to the price
  columns, dropping extra columns, and the prices to the score dates without forward filling. It
  selects $\lceil q n_t \rceil$ assets by `rank(method='first')`, so ties follow the column order,
  and it raises `ValueError` for a `quantile` outside $(0, 1]$. Eligibility is a non-missing score
  and price: an infinite score or a negative price is not excluded.
- `backtest_alpha_rank_portfolio(prices, alpha_scores, quantile=1/3, rebalancing_freq='QE',
  time_period=None, rebalancing_costs=None, instruments_carry=None,
  strategy_ticker='Top-quantile', benchmark_ticker='Equal Weight')` accepts one score panel or a
  dictionary of named panels. For each, it forms the targets, keeps the rows inside `time_period`,
  samples them with `asfreq(rebalancing_freq, method='ffill')` and runs
  `qis.backtest_model_portfolio` on the full price panel. The benchmark, from
  `qis.df_to_equal_weight_allocation`, holds equal weights over the assets with a price, whether
  or not they have a score, sampled and simulated the same way. It returns a `qis.MultiPortfolioData`
  with the strategy legs in input order, the benchmark last, and the benchmark NAV as
  `benchmark_prices`; an empty dictionary gives the benchmark alone.
- `compute_alpha_rank_analysis_table(multi_portfolio_data, time_period=None, perf_params=None)`
  returns, per leg, `Return p.a.`, `Vol`, `Sharpe` (the qis zero-rate Sharpe ratio), `Max DD` and
  `Turnover p.a.`, from `qis.compute_ra_perf_table` with monthly performance statistics by
  default. Performance uses each leg's full NAV; `time_period` filters only the turnover. The
  turnover is two-sided: the value of the units bought and sold at each date over the NAV, summed
  and divided by the years between the first and the last date.
- `generate_alpha_profile_report(multi_portfolio_data, time_period=None, perf_params=None,
  regime_benchmark=None, group_data=None, backtest_name='Alpha Signal Profile',
  file_name='alpha_profile_report', local_path=None, add_current_date=True)` renders
  `qis.generate_multi_portfolio_factsheet`, saves the figures as a landscape PDF with
  `qis.save_figs_to_pdf` and returns the list of figures. The regime benchmark defaults to the
  last leg, the equal-weight benchmark. `local_path=None` writes to the current working
  directory, so pass an explicit output directory.

The adapters compute a score and call `backtest_alpha_rank_portfolio`. `ProfileSignal` is a string
enum of five labels, `MOMENTUM`, `CLASSIC_MOMENTUM`, `LOW_BETA`, `RESIDUAL_MOMENTUM` and `CARRY`;
it labels legs and does not dispatch anything. `profile_momentum`, `profile_classic_momentum`,
`profile_low_beta`, `profile_residual_momentum` and `profile_carry` call their signal constructors
on monthly returns by default and select the top third at quarter ends, with no costs; each labels
its leg with the value of its `ProfileSignal` member. The raw signal and the score are not
returned. `profile_carry` uses the carry panel only for the score and passes no carry to the
backtester, and the `group_data` of `profile_classic_momentum` and `profile_carry` scores within
groups while the selection still ranks the whole universe. `profile_alpha_signals` backtests a
nonempty dictionary of precomputed panels and raises `ValueError` for an empty one. The code is in
[`profile/core.py`](../src/optimalportfolios/alphas/profile/core.py) and
[`profile/signal_profilers.py`](../src/optimalportfolios/alphas/profile/signal_profilers.py).

### The diagnostics adapters

- `signal_diagnostics_panel(alphas_data, components=None)` returns a dictionary of the populated
  score fields of an `AlphasData`, in the fixed order `alpha_scores`, `momentum_score`,
  `momentum_cluster_score`, `beta_score`, `beta_cluster_score`, `residual_momentum_score`,
  `residual_momentum_cluster_score`, `managers_scores`, or of the fields named in `components`.
- `run_signal_diagnostics(asset_returns_dict, signal, group_data=None, horizons=(1, 2, 3, 6),
  signal_attribute='alpha_scores', group_order=None, is_log_returns=True)` accepts a DataFrame or
  an `AlphasData`, whose field `signal_attribute` it reads. An unknown field raises
  `AttributeError` and an unpopulated one `ValueError`. It returns the
  `qis.SignalDiagnosticsResult` of `qis.estimate_signal_diagnostics`.
- `run_signal_diagnostics_per_component` runs the same call for each field of
  `signal_diagnostics_panel` and returns a dictionary of results by field name.
- `compare_signal_diagnostics(results, horizon=None)` stacks the `pooled_universe` rows of several
  results, indexed by signal and horizon, or by signal alone for one `horizon` label; no results
  give an empty DataFrame.

The module also defines `compare_signal_ic_ir`, which stacks the qis IC information ratios of
several results, and `build_signal_diagnostics_table`, which joins them to the pooled rows;
`optimalportfolios.alphas` does not export either.
The code is in [`signal_diagnostics.py`](../src/optimalportfolios/alphas/signal_diagnostics.py).

### What the package adds to qis

| Step | qis | optimalportfolios |
|---|---|---|
| Inputs | A score DataFrame and per-cadence returns | Also an `AlphasData` field, chosen by `signal_attribute`, and a loop over all populated fields |
| Return grid | The caller's `asset_returns_dict`, one frame per native cadence | None computed: the caller supplies the returns, for example `qis.to_returns(prices, freq='ME', is_log_returns=True, drop_first=True)`, since `qis.to_returns` defaults to simple returns |
| Alignment and lag | Resamples the score to each native grid, lags it one period, drops assets without a score column | Nothing added |
| Horizons and options | Default horizons `(1, 3, 6)`; `fit_intercept`, `is_vol_normalised`, `min_obs_per_date`, `min_obs_per_group` | Default horizons `(1, 2, 3, 6)`; the four options are not exposed and keep their qis defaults `False`, `True`, 5 and 10 |
| Quantiles | None in the diagnostics | The top quantile only, $\lceil q n_t \rceil$ assets with equal weights; other baskets by masking |
| Holdings | `backtest_model_portfolio` trades dated targets at the first price on or after their date and holds units | Target rows sampled to `rebalancing_freq`, no implementation lag, no costs unless `rebalancing_costs` is given, and an equal-weight benchmark leg |

> **Pitfall.** qis labels horizons with strings. `compare_signal_diagnostics(results, horizon=1)`
> finds no horizon `1` in any result, logs a warning for each signal and returns an empty
> DataFrame. Pass the label, `horizon='1'`, as listed in `horizon_labels` of each result.

The repository workflow [`profile_alpha_signals.py`](../examples/alphas/profile_alpha_signals.py)
profiles carry, low-beta and momentum scores on a bond ETF universe and writes a qis report; it
needs a network connection. The
[module guide](https://github.com/ArturSepp/OptimalPortfolios/blob/main/src/optimalportfolios/alphas/README.md)
gives an offline profiling workflow on the test fixture.

## Interpretation and limitations

- The diagnostics measure a score after the fact, on realised returns. They do not produce inputs
  known at formation, and a good in-sample IC can reflect choices made while looking at the same
  data.
- The standard error of the mean IC is about $1/\sqrt{(n_t - 1) T}$ for a small, constant IC. An
  IC of 0.02 on 100 assets over ten years of months has a t-statistic of about 2.2; the example
  uses 0.1 on 500 assets so that its conclusions hold with a wide margin.
- The IC ignores persistence. The example's scores are independent from month to month, so its top
  quintile trades about 19 times its value a year, and the profiler charges no costs by default;
  see [turnover and transaction costs](turnover_and_transaction_costs.md).
- The profiler ranks and equal-weights the selected assets. It does not use a covariance, control
  factor exposures or size positions by conviction, and ties follow the column order. Its
  eligibility check is only that score and price are present; validate finiteness and positive
  prices before profiling.
- `time_period` selects target rows; it does not end the price history, so the NAV continues after
  its end. In `compute_alpha_rank_analysis_table` it filters only the turnover. Choose the price
  sample explicitly.
- Grinold and Kahn (2000) evaluate forecasts of residual returns with a risk model, turn scores
  into alphas with volatility and the IC, and judge them by the information ratio of an optimised
  active portfolio. The package inherits none of that: the diagnostics normalise total returns
  across assets at each date, which removes a common return but no factor exposure, and the
  profiler is a long-only, equal-weight selection measured against an equal-weight benchmark.
- A high rank IC does not by itself give value in an optimiser. The optimiser needs a calibrated
  scale, and it discounts high scores that share risk; the
  [fundamental law of active management](https://quantinveststrats.readthedocs.io/en/stable/signal_diagnostics.html#relation-to-the-fundamental-law-of-active-management)
  on the qis page states the assumptions under which the IC translates into an information ratio.

## See also

- [Alpha signals](alphas_module_readme.md): the score constructors and `AlphasData`.
- [Mixed-frequency data](mixed_frequency_data.md): native cadences and signal horizons.
- [Rolling backtests](rolling_backtests.md): how targets become holdings, lags and drift.
- [Turnover and transaction costs](turnover_and_transaction_costs.md).
- [Choosing an objective](optimization_module_readme.md): the objectives that take alphas.
- [Conventions, notation and glossary](conventions.md).
- [qis signal diagnostics](https://quantinveststrats.readthedocs.io/en/stable/signal_diagnostics.html).

## References

- Grinold, R. C. and Kahn, R. N. (2000). *Active Portfolio Management: A Quantitative Approach for
  Producing Superior Returns and Controlling Risk*. 2nd edition. McGraw-Hill, New York.
  ISBN 0-07-024882-6.
- qis documentation. [Signal diagnostics: information coefficient and information
  ratio](https://quantinveststrats.readthedocs.io/en/stable/signal_diagnostics.html).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
