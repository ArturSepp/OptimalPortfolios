---
myst:
  html_meta:
    description: >-
      Factor covariance with HCGL in Python with optimalportfolios: sparse factor loadings fitted
      by FactorLasso on monthly and quarterly assets, the assembly of beta, factor covariance and
      residual covariance in annual units, orthogonal and empirical residuals, the residual weight,
      cadence penalties, factors as clustering references and point-in-time rolling fits, with a
      verified offline example.
---

# Factor covariance with HCGL

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Factor covariance estimation is implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

A factor covariance explains the joint variation of $N$ assets through $M$ factors. Each asset's
return is a combination of the factor returns, with loadings $\beta$, plus a residual, so the
covariance splits into a systematic part $\beta \Sigma_F \beta^{\top}$ and a residual part $D$.
The split of risk into systematic and specific parts goes back to Rosenberg and McKibben (1973).

`FactorCovarEstimator` builds this covariance for a multi-asset universe observed at more than one
return frequency. [FactorLasso](https://github.com/ArturSepp/FactorLasso) fits sparse loadings on
each frequency's assets, here with the hierarchical clustering group LASSO (HCGL);
[qis](https://github.com/ArturSepp/QuantInvestStrats) estimates the factor covariance; this package
aligns the inputs, annualises each component and assembles one annual covariance matrix, or one
per rebalancing date. The output has the same form as that of the EWMA estimator, described with
the shared estimator contract on the [covariance estimators](covariance_estimators.md) page. Sepp,
Ossa and Kastenholz (2026) use this covariance in the ROSAA framework; the
[ROSAA case study](app_rosaa_multi_asset_allocation.md) maps that article to the package.

This page states what the estimator computes: the assembly and its units, the two residual types
and the residual weight, the penalties by return frequency, the factors as clustering references,
and the point-in-time contract of rolling fits. The penalties and the clustering are FactorLasso's;
its articles, linked below, derive them.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Log returns: the caller supplies asset log returns for each frequency bucket; factor log returns are log differences of the factor prices on each bucket's dates for the loadings, and at `factor_returns_freq` for the factor covariance |
| Estimation grid | Each bucket's own dates for the loadings, with the span of `LassoModel.span_freq_dict` (36 months and 12 quarters in the example); `factor_returns_freq` with `factor_covar_span` for the factor covariance (month ends and span 36 in the example; the defaults are `'W-WED'` and 52) |
| Rebalancing grid | Calendar dates of `rebalancing_freq` (default quarter ends, `'QE'`) within the rolling `time_period`; the example's rolling fits are at the five quarter ends from 31 December 2023 to 31 December 2024 |
| Covariance units | Annual, fractional log-return squared: the factor covariance times the periods per year of `factor_returns_freq`, each residual variance times those of its bucket (12 monthly, 4 quarterly); a supplied `x_covar` is used without scaling |
| Expected returns | None |
| Weight state | None; the page estimates covariance matrices, not weights |
| Solver | CVXPY with CLARABEL, the `LassoModel` default, inside FactorLasso for the penalised regressions |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $\Sigma_y$ | The assembled annual asset covariance, the $\Sigma$ of the conventions page; FactorLasso calls it `y_covar` |
| $\Sigma_F$ | Annual factor covariance, stored as `x_covar` |
| $f_t$ | Vector of the $M$ factor log returns over the period ending at $t$ |
| $c$, $\mathrm{AN}_c$ | A return frequency, the key of a bucket such as `'ME'` or `'QE'`, and its periods per year |
| $v_i$ | Annual residual variance of asset $i$ |
| $S$, $R$ | The diagonal matrix of residual volatilities $\sqrt{v_i}$, and a residual correlation matrix |
| $\rho$, $\kappa$ | The residual correlation weight `residual_corr_weight` and the residual weight `residual_var_weight` |

**Inputs.** `risk_factor_prices` holds total-return factor prices by date. `asset_returns_dict`
maps each frequency code to the log returns of the assets observed at that frequency, for example
`{'ME': monthly_returns, 'QE': quarterly_returns}`. Each asset belongs to exactly one bucket, and
the estimator does not convert simple returns to log returns. The inputs must be the data as known
at each date; publication lags and revisions are the data producer's responsibility.

**Configuration.** `lasso_model` is a FactorLasso `LassoModel`: its model type, its penalty
`reg_lambda` and its spans, where `span_freq_dict` gives each frequency its own span in that
frequency's observations. The ROSAA article uses 36 months for liquid and 12 quarters for illiquid
instruments. The factor covariance has its own `factor_returns_freq` and `factor_covar_span`, and
`rebalancing_freq` sets the dates of rolling fits. The three clocks are separate; see
[mixed-frequency data](mixed_frequency_data.md).

**Assumptions.** Factor returns and residuals are uncorrelated, under both residual types. The
annualisation multiplies per-period variances by periods per year, which assumes returns without
serial correlation; smoothed returns, such as appraisal-based private-asset returns, must be
unsmoothed before estimation.

## Methodology

### The factor model and its covariance

For asset $i$ in bucket $c$, the model on that bucket's dates is

$$
r_{i,t} = \beta_{i,0} + \beta_i^{\top} f_t + \varepsilon_{i,t},
$$

where $f_t$ holds the log differences of the factor prices sampled on the same dates, the last
earlier price standing in for a missing date. A quarterly asset is therefore regressed on quarterly factor returns. Log
returns add over time, so if the relation holds each month with loadings $\beta_i$, it holds for
quarterly sums with the same loadings: one loading matrix serves both frequencies.

**Proposition 1 (assembly).** If the factor returns are uncorrelated with the residuals, the
covariance of the asset returns is

$$
\Sigma_y = \beta \Sigma_F \beta^{\top} + D,
$$

with $\beta$ the $N \times M$ loadings, $\Sigma_F$ the $M \times M$ factor covariance and $D$ the
$N \times N$ residual covariance.

**Proof.** The return vector is a constant plus $\beta f_t + \varepsilon_t$. Its covariance is
$\beta \mathrm{Cov}(f) \beta^{\top} + \mathrm{Cov}(\varepsilon)$ plus the two cross terms
$\beta \mathrm{Cov}(f, \varepsilon)$ and its transpose, which vanish by assumption. $\square$

The package estimates each term in annual units:

- $\Sigma_F$ is the EWMA covariance of demeaned factor log returns sampled at
  `factor_returns_freq`, with span `factor_covar_span`, times the periods per year of that
  frequency, computed by the qis kernel of the [EWMA estimator](covariance_estimators.md#ewma-covariance).
  `is_apply_vol_normalised_returns=True` selects the normalised-return kernel of qis for this
  estimate. A supplied `x_covar` replaces the estimate and is used as given: it must already be
  annual and ordered like the factor columns.
- $\beta$ comes from the FactorLasso fit of each bucket, described in the next subsection.
- $v_i$, the diagonal of $D$, is the EWMA-weighted mean squared residual of the fit, times the
  periods per year of the bucket:

$$
v_i = \mathrm{AN}_c \sum_{t} \omega_t \left( \tilde r_{i,t} - \beta_i^{\top} \tilde f_t \right)^2,
\qquad
\omega_t = \frac{\lambda^{T-t}}{\sum_{u} \lambda^{T-u}},
$$

where the tilde marks returns net of their EWMA mean and $\lambda$ is the decay of the bucket's
span. The loadings are dimensionless, so the three terms are in the same annual units.

### Sparse loadings, HCGL and cadence penalties

FactorLasso fits each bucket with the configured `LassoModel`. With
`LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO`, it first clusters the bucket's assets by a Ward
linkage of their EWMA correlations, then penalises the loadings with a group penalty weighted by
the clusters. [FactorLasso's article on group penalties](https://factorlasso.readthedocs.io/en/latest/group_penalties_hcgl_fcgl.html)
states the objective and [its article on cluster discovery](https://factorlasso.readthedocs.io/en/latest/cluster_discovery.html)
the partition. Each bucket is clustered and fitted separately, so a cluster never mixes
frequencies, and its label carries the frequency, as in `'QE:1'`.

The model's scalar `reg_lambda` is one penalty for every bucket. A quarterly bucket with a span of
12 quarters rests on far fewer effective observations than a monthly one with a span of 36 months,
so the same penalty can be too weak for it. `reg_lambda_freq_dict` maps each frequency to its own
fixed penalty, which overrides the scalar for that bucket's fit. Every fitted frequency needs an
entry, values must be finite and non-negative, and the model's own `reg_lambda` is restored after
the fit.

### Factors as clustering references

With `include_factors_in_clustering=True`, the factor log returns join the bucket's asset returns
as extra series in the cluster discovery only. They are then removed from the cluster assignments
and the reported tree. They are never responses: they have no loadings, no residuals and no row in
the covariance. What changes is the partition, and through it the group penalty and the loadings.
`factor_clustering_freqs` lists the buckets that receive the references; `None` means all of them.
The option requires the HCGL or FCGL model type, and it makes a current fit truncate its inputs at
`estimation_date`.

### Residual covariance: orthogonal and empirical

`residual_type='orthogonal'`, the default, takes the residuals of different assets as
uncorrelated: $D = \mathrm{diag}(v_1, \dots, v_N)$. `residual_type='empirical'` adds their
correlation:

$$
D = S \left[ (1 - \rho) I + \rho R \right] S ,
$$

where $R$ is a residual correlation matrix and $\rho$ is `residual_corr_weight`, in $[0, 1]$ with
default 1; the weight applies only to empirical residuals.

**Proposition 2 (variances are kept).** For every $\rho$ in $[0, 1]$, the diagonal of $D$ is
$v_1, \dots, v_N$; $\rho = 0$ gives the orthogonal matrix; and $D$ is positive semidefinite.
Hence both residual types give every asset the same variance, and they differ only in the
covariances between assets.

**Proof.** $R$ has a unit diagonal, so the bracket has a unit diagonal too, and
$D_{ii} = \sqrt{v_i} \cdot 1 \cdot \sqrt{v_i} = v_i$; at $\rho = 0$ the bracket is the identity.
The bracket is a convex combination of two positive semidefinite matrices, so it is positive
semidefinite, and so is its congruence by the diagonal $S$. $\square$

The residuals of different buckets live on different dates, so the package estimates $R$ on a
common grid. It divides the stored residuals by their annual multiplier $\mathrm{AN}_c$, sums
complete native periods into periods of `residual_covar_freq`, subtracts a causal EWMA mean and
takes the EWMA correlation with span `residual_covar_span`. By default the grid is the lowest
native frequency, quarterly in the example, and the span is that bucket's loading span, 12
quarters. An explicit span counts periods of the common grid. An explicitly coarser grid converts
the default decay to the new period length: with $\mathrm{AN}_c$ the periods per year of the lowest
native frequency and $\mathrm{AN}$ those of the grid, the decay becomes

$$
\lambda^{\mathrm{AN}_c / \mathrm{AN}} .
$$

A leading or trailing incomplete period is left out and an internal gap is an error; nothing is
prorated or interpolated. FactorLasso's
[article on empirical residual correlation](https://factorlasso.readthedocs.io/en/latest/empirical_residual_correlation.html)
gives the aggregation rules.

### The residual weight

The plain-matrix methods and the getters take a residual weight $\kappa$, `residual_var_weight`,
with default 1:

$$
\Sigma_y(\kappa) = \beta \Sigma_F \beta^{\top} + \kappa D .
$$

It scales the whole residual block, off-diagonal entries included, and $\kappa = 0$ leaves exactly
the systematic part. For active weights $d$, the squared tracking error is
$d^{\top} \beta \Sigma_F \beta^{\top} d + \kappa d^{\top} D d$, so a weight below one understates
it. The weight is an option of the package, not part of the ROSAA method: the article measures
tracking error with the full covariance, residuals included, as the
[case study's Pitfall](app_rosaa_multi_asset_allocation.md#configuration) records.

### The point-in-time contract

A rolling fit generates the calendar dates $t_k$ of `rebalancing_freq` in its `time_period`,
slices the factor prices and every bucket through $t_k$ and fits on that expanding history.

**Proposition 3 (no look-ahead).** The rolling estimate dated $t_k$ equals a current fit on the
inputs sliced through $t_k$. Hence changing any input after $t_k$ changes no estimate dated $t_k$
or earlier.

**Proof.** Before each fit, every input is replaced by its slice through $t_k$, and the fit is a
deterministic function of its inputs. An input after $t_k$ is not in any of these slices. $\square$

A current fit is point in time only if its inputs are. In the default orthogonal fit without
references, `estimation_date` is a label: it does not truncate the inputs, so a historical fit
needs inputs sliced by the caller. An empirical fit, or one with factor references, truncates every
input at `estimation_date`. A supplied `x_covar` is never truncated and must respect the date.

An empirical rolling fit records, with each residual correlation, the last complete common period
it observes and the date it becomes available, and it keeps the last correlation between complete
periods while the loadings and residual variances update. A correlation is never used before its
availability date, and an as-of query before the first fit is refused.

## Worked example

The Python blocks below run in order on a synthetic panel drawn from a fixed seed and need no
download or data file. They are excerpts of the canonical script
[`examples/docs/factor_covariance_hcgl.py`](../examples/docs/factor_covariance_hcgl.py), which
asserts every number on this page, and its main properties, against a reference computed a
different way:

```console
python -m examples.docs.factor_covariance_hcgl
```

The script imports `numpy` as `np`, `pandas` as `pd`, `qis`, `optimalportfolios` as `op`, and
`LassoModel` and `LassoModelType` from `factorlasso`.

### Construct factor prices and asset returns

Four factors, Equity, Rates, Credit and Commodities, drive eight assets through the known loadings
`LOADINGS` of the script. Five liquid assets are observed monthly; three private assets, Private
equity, Real estate and Infrastructure, are observed quarterly, their quarterly log returns being
the sums of three monthly ones. The private assets also share a residual shock that no factor
spans. The panel covers month ends from December 2004 to December 2024:

```python
dates = pd.date_range('2004-12-31', '2024-12-31', freq='ME')
rng = np.random.default_rng(seed)
factor_covar = np.outer(FACTOR_VOLS, FACTOR_VOLS) * np.array(FACTOR_CORR)
# The draws of Generator.multivariate_normal, whose SVD factor has signs that differ between
# LAPACK builds, with each singular vector's largest entry given its sign in SVD_SIGNS so that
# the same seed gives the same returns on every platform.
_, singular_values, vh = np.linalg.svd(factor_covar)
vh = vh * (SVD_SIGNS * np.sign(vh[np.arange(4), np.abs(vh).argmax(axis=1)]))[:, None]
shocks = rng.standard_normal((len(dates) - 1, 4))
factors = 0.004 + shocks @ (np.sqrt(singular_values)[:, None] * vh)
residuals = rng.normal(0.0, RESIDUAL_VOLS, size=(len(dates) - 1, len(ASSETS)))
residuals[:, 5:] += rng.normal(0.0, PRIVATE_SHOCK_VOL, size=(len(dates) - 1, 1))
monthly = pd.DataFrame(factors @ np.array(LOADINGS).T + residuals,
                       index=dates[1:], columns=ASSETS)
factor_prices = pd.DataFrame(
    100.0 * np.exp(np.vstack([np.zeros(4), np.cumsum(factors, axis=0)])),
    index=dates, columns=FACTORS)
buckets = {'ME': monthly[MONTHLY], 'QE': monthly[QUARTERLY].resample('QE').sum()}
```

These lines are the body of `simulated_panel(seed)`, which the script calls with `SEED = 5`. The
result is 241 month-end factor prices, 240 monthly returns of the liquid assets and 80 quarterly
returns of the private ones.

### Fit HCGL at an explicit cutoff

Every estimator below gets a fresh model, because a fit stores its state on the `LassoModel` it
is given. The model is HCGL with the article's penalty of $10^{-5}$ and its spans:

```python
def hcgl_model(reg_lambda: float = 1e-5) -> LassoModel:
    """Return a fresh HCGL model: the article's penalty, 36 months and 12 quarters of span."""
    return LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                      reg_lambda=reg_lambda, span_freq_dict={'ME': 36, 'QE': 12})
```

The fit at 31 December 2023 uses inputs sliced through that date:

```python
estimator = op.FactorCovarEstimator(
    lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
    rebalancing_freq='QE')
as_of = pd.Timestamp('2023-12-31')
history = {'risk_factor_prices': factor_prices.loc[:as_of],
           'asset_returns_dict': {freq: r.loc[:as_of] for freq, r in buckets.items()},
           'estimation_date': as_of}
data = estimator.fit_current_factor_covars(**history)
covar = data.get_y_covar()
factor_only = data.get_y_covar(residual_var_weight=0.0)
```

`data` holds eight asset rows of loadings on four factor columns, with clusters labelled `'ME:'`
and `'QE:'`. The script recomputes each component independently. $\Sigma_F$ equals twelve times an
explicit EWMA weighted sum of the demeaned monthly factor log returns at span 36. Each $v_i$ equals
the formula of the methodology from the fitted loadings, with span 36 and multiplier 12 for the
liquid assets and span 12 and multiplier 4 for the private ones. `covar` equals
$\beta \Sigma_F \beta^{\top} + D$ built from these references, to rounding, and is positive
definite; `factor_only` is $\beta \Sigma_F \beta^{\top}$, and their difference is exactly $D$.

### Orthogonal and empirical residuals

The same fit with empirical residuals and half the correlation retained:

```python
empirical = op.FactorCovarEstimator(
    lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
    residual_type='empirical', residual_corr_weight=0.5)
empirical_data = empirical.fit_current_factor_covars(**history)
blended = empirical.fit_current_covar(**history)
```

The correlation is estimated on quarter ends with span 12, observed through 31 December 2023 and
available from that date. It equals the EWMA correlation that the script builds from quarterly
sums of the stored residuals. The loadings and $v_i$ are those of the orthogonal fit, and
`blended` equals $\beta \Sigma_F \beta^{\top} + S [0.5 I + 0.5 R] S$. As Proposition 2 states, its
diagonal is that of `covar` at every $\rho$, $\rho = 0$ reproduces `covar`, and a residual weight
of zero leaves the systematic part.

> **Insight.** Empirical residuals change portfolio risk, not asset risk. The three private
> assets share a residual shock that the factors do not span. Both residual types give each of
> them the same variance, but an equal-weight portfolio of the three has a residual variance of
> 0.00019 with orthogonal residuals and 0.00040 with empirical residuals at $\rho = 1$, more
> than twice as much.

![Left: stacked bars of systematic and residual annual variance for the eight assets, identical
under orthogonal and empirical residuals; the residual part is visible for EM equity and
Commodities. Right: residual variance of equal weights in the three private assets, 0.00019 with
orthogonal residuals, rising to 0.00030 at rho 0.5 and 0.00040 at rho 1 as residual covariances
are added.](images/factor_covariance_variance_split.png)

*Figure: the variance split of Proposition 2 on the example's fit at 31 December 2023. The residual
types agree on every asset's variance and differ in the residual covariances that a portfolio
collects. Drawn by the `exhibit` function of the canonical script; the
[analytics gallery](analytics_gallery.md) lists its provenance.*

### Cadence penalties

A penalty of $10^{-4}$ for the quarterly bucket, keeping $10^{-5}$ for the monthly one:

```python
penalised = op.FactorCovarEstimator(
    lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
    reg_lambda_freq_dict={'ME': 1e-5, 'QE': 1e-4})
penalised_data = penalised.fit_current_factor_covars(**history)
```

The monthly loadings are those of the scalar fit, and the quarterly ones equal those of a separate
fit of the quarterly bucket with `reg_lambda=1e-4`, made with `estimate_lasso_factor_covar_data`.
The larger penalty shrinks the quarterly loadings by up to 0.30: Real estate's loading on Rates
falls from 0.46 to 0.16. The model's `reg_lambda` is still $10^{-5}$ after the fit, and a map
without the quarterly bucket raises `KeyError`.

### Factors as clustering references

The four factors as references in the quarterly bucket only:

```python
referenced = op.FactorCovarEstimator(
    lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
    include_factors_in_clustering=True, factor_clustering_freqs=['QE'])
referenced_data = referenced.fit_current_factor_covars(**history)
```

Loadings, clusters, residuals and covariance still cover the eight assets only, and the monthly
bucket is unchanged. In the quarterly bucket, the three private assets form one cluster instead of
three, and their loadings move by up to 0.09. A `LASSO` model or an empty
`factor_clustering_freqs` is refused at construction.

### Rolling fits are point in time

```python
period = qis.TimePeriod('2023-12-31', '2024-12-31')
rolling = estimator.fit_rolling_factor_covars(
    risk_factor_prices=factor_prices, asset_returns_dict=buckets, time_period=period)
covars = rolling.get_y_covars()
```

The keys are the five quarter ends from 31 December 2023 to 31 December 2024, and each matrix
equals a current fit on the inputs sliced through its date, as Proposition 3 states. When the
Equity factor price doubles after 30 June 2024, and the later log returns of DM equity and Real
estate rise by 0.10, the first three matrices are unchanged and the last two move. An empirical
rolling fit at month ends holds each quarterly correlation for the two months until the next
complete quarter, while the loadings change every month; its five correlation vintages are dated
at the quarter ends.

> **Pitfall.** A date label is not a cutoff. An orthogonal current fit on the whole history with
> `estimation_date` set to 31 December 2023 gives Commodities a volatility of 23.9%, against 20.4%
> from the inputs sliced through that date: the label does not remove the year of later data.

## Implementation in optimalportfolios

| Entry point | Inputs and output |
|---|---|
| `FactorCovarEstimator.fit_current_covar` | Factor prices and return buckets to one annual matrix $\Sigma_y(\kappa)$, with the configured residual type and `residual_var_weight`. |
| `FactorCovarEstimator.fit_rolling_covars` | The same inputs and a `time_period` to a dictionary of annual matrices keyed by calendar rebalancing dates. |
| `fit_current_factor_covars` / `fit_rolling_factor_covars` | The factor-specific methods: a FactorLasso `CurrentFactorCovarData`, or a `RollingFactorCovarData` of them by date, with every component. |
| `estimate_lasso_factor_covar_data` | The function behind the current fit: alignment, one FactorLasso fit per bucket, annualisation and assembly of the container. |
| `plot_current_covar_data` | One snapshot to four figures: factor correlations, asset correlations, the cluster dendrograms, and the loadings with their fit statistics. |
| `plot_hcgl_covar_data` | The same four figures from the components passed one by one. |
| `run_rolling_covar_report` | Refits the rolling decompositions and returns the figures, if `is_plot`, and one snapshot table of loadings and statistics per date. |

The container holds `y_betas` (assets by factors), `x_covar`, `y_variances` (with the residual
variances $v_i$, the R-squared and the in-sample alpha), `clusters`, `linkages` and `cutoffs` by
frequency, and `residuals`. The residual panel is multiplied by each bucket's periods per year and
is formed without subtracting the fitted intercept; it is not the per-period series, so do not
recompute $v_i$ as its sample variance. `get_y_covar` and `get_residual_covar` take
`residual_var_weight`, `residual_type` and `residual_corr_weight` and assemble $\Sigma_y(\kappa)$
and $\kappa D$ from a snapshot without refitting. These getters default to orthogonal residuals,
while the estimator's plain-matrix methods use its configured `residual_type`. The rolling
`RollingFactorCovarData.get_y_covars` takes optional `dates` and answers each with the latest
snapshot available then. Empirical residuals need FactorLasso 0.19.0 or newer; older releases
serve the orthogonal default.

Configuration fields not covered above:

- `demean` is stored but not read. The internal factor covariance always subtracts its EWMA mean,
  and the regression's demeaning is set by `LassoModel.demean`; the example's fit with
  `demean=False` has the same factor covariance and loadings.
- `residual_covar_freq` and `residual_covar_span` configure the empirical grid and span as in the
  methodology; both default to `None`, the derived choice.
- The fit mutates `lasso_model`: it leaves the last bucket's fitted state on the supplied model,
  here the quarterly one. Read results from the returned containers.
- The optional `assets` argument reindexes the output universe and fills an asset without history
  with zero loadings and zero residual variance.

## Interpretation and limitations

- The package takes the covariance of the ROSAA article (Sepp, Ossa and Kastenholz, 2026): HCGL
  loadings on multi-asset factors with spans per frequency. The article assumes uncorrelated
  residuals, so the empirical residual type and the residual weight are extensions of the
  package, and the article unsmooths private-asset returns before estimation, which the estimator
  does not do.
- It inherits from Rosenberg and McKibben (1973) the split of risk into systematic and specific
  parts, not their prediction of loadings from firm characteristics. Of the three types of factor
  model that Connor (1995) compares, it is closest to the macroeconomic type: the factor returns
  are observed series and each asset's loadings come from a time-series regression, not from
  security attributes or a statistical factor analysis.
- Orthogonal residuals understate the risk of a portfolio concentrated in assets with a common
  residual shock, as the Insight shows. The empirical correlation on quarterly data with span 12
  rests on few effective observations and is itself noisy; $\rho$ below one shrinks it towards
  zero.
- In-sample residual variances on a short effective history tend to be too low, because the
  loadings are fitted to the same observations. In the example, each private asset's $v_i$ is
  below the residual variance of the generating model.
- An asset filled with zeros through `assets` looks riskless; eligibility and warm-up policy are
  separate from assembly. The rolling wrapper raises when a bucket has fewer rows than the model's
  `warmup_period` at a fit date, which checks the bucket, not each asset.
- A residual weight below one understates tracking error, so a tactical allocation takes larger
  tilts than its budget intends.

## See also

- [Covariance estimators](covariance_estimators.md): the EWMA estimator and the shared contract.
- [Mixed-frequency data](mixed_frequency_data.md) and [incomplete histories](incomplete_histories.md).
- [Rolling factor covariance from CSV](rolling_factor_covar_from_csv.md): the same estimator on a
  saved bundle.
- [Strategic and tactical allocation with HCGL covariance (ROSAA)](app_rosaa_multi_asset_allocation.md)
  and [from capital market assumptions to strategic allocation](app_cma_strategic_allocation.md).
- [Risk budgeting](risk_budgeting.md) and [conventions](conventions.md).
- FactorLasso: [factor covariance assembly](https://factorlasso.readthedocs.io/en/latest/factor_covariance_assembly.html),
  [group penalties, HCGL and FCGL](https://factorlasso.readthedocs.io/en/latest/group_penalties_hcgl_fcgl.html),
  [cluster discovery](https://factorlasso.readthedocs.io/en/latest/cluster_discovery.html) and
  [empirical residual correlation](https://factorlasso.readthedocs.io/en/latest/empirical_residual_correlation.html).

## References

- Rosenberg, B. and McKibben, W. (1973). *The Prediction of Systematic and Specific Risk in Common
  Stocks*. Journal of Financial and Quantitative Analysis, 8(2), 317–333.
  [DOI 10.2307/2330027](https://doi.org/10.2307/2330027).
- Connor, G. (1995). *The Three Types of Factor Models: A Comparison of Their Explanatory Power*.
  Financial Analysts Journal, 51(3), 42–46.
  [DOI 10.2469/faj.v51.n3.1904](https://doi.org/10.2469/faj.v51.n3.1904).
- Sepp, A., Ossa, I. and Kastenholz, M. (2026). *Robust Optimization of Strategic and Tactical
  Asset Allocation for Multi-Asset Portfolios*. The Journal of Portfolio Management, 52(4),
  86–120. [DOI 10.3905/jpm.2025.1.806](https://doi.org/10.3905/jpm.2025.1.806);
  [author-shared copy](https://eprints.pm-research.com/17511/143431/index.html).
- FactorLasso documentation: [Factor covariance assembly](https://factorlasso.readthedocs.io/en/latest/factor_covariance_assembly.html),
  [Group penalties: HCGL, sparse-group and FCGL](https://factorlasso.readthedocs.io/en/latest/group_penalties_hcgl_fcgl.html),
  [Cluster discovery](https://factorlasso.readthedocs.io/en/latest/cluster_discovery.html) and
  [Empirical residual correlation](https://factorlasso.readthedocs.io/en/latest/empirical_residual_correlation.html).
- OptimalPortfolios: [factor estimator](../src/optimalportfolios/covar_estimation/factor_covar_estimator.py)
  and [covariance reporting](../src/optimalportfolios/covar_estimation/covar_reporting.py).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
