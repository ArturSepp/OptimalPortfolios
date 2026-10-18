---
myst:
  html_meta:
    description: >-
      Factor covariance with HCGL in TradableFactors for OptimalPortfolios: sparse factor loadings fitted
      by FactorLasso on monthly and quarterly assets, the assembly of beta, factor covariance and
      residual covariance in annual units, orthogonal and empirical residuals, the residual weight,
      cadence penalties, factors as clustering references and point-in-time rolling fits, with a
      verified offline example.
---

# Factor covariance with HCGL

*Author: [Artur Sepp](https://github.com/ArturSepp)*

TradableFactors supplies factor covariance estimates consumed by
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

A factor covariance explains the joint variation of $N$ assets through $M$ factors. Each asset's
return is a combination of the factor returns, with loadings $\beta$, plus a residual, so the
covariance splits into a systematic part $\beta \Sigma_F \beta^{\top}$ and a residual part $D$.
The split of risk into systematic and specific parts goes back to Rosenberg and McKibben (1973).

`tradablefactors.FactorCovarEstimator` builds this covariance for a multi-asset universe observed at more than one
return frequency. [FactorLasso](https://github.com/ArturSepp/FactorLasso) fits sparse loadings on
each frequency's assets, here with the hierarchical clustering group LASSO (HCGL);
[qis](https://github.com/ArturSepp/QuantInvestStrats) estimates the factor covariance; TradableFactors
aligns the inputs, annualises each component and assembles one annual covariance matrix, or one
per rebalancing date. The output has the same form as that of the EWMA estimator, described with
the covariance input contract on the [covariance estimators](covariance_estimators.md) page. Sepp,
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
| Covariance units | Annual, fractional log-return squared: the factor covariance times the periods per year of `factor_returns_freq`, each residual variance times those of its bucket (12 monthly, 4 quarterly); a supplied `DatedFactorCovariance` is used without scaling |
| Expected returns | None |
| Weight state | None; the page estimates covariance matrices, not weights |
| Solver | CVXPY with CLARABEL, the `LassoModel` default, inside FactorLasso for the penalised regressions |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $\Sigma_y$ | The assembled annual asset covariance, the $\Sigma$ of the conventions page; FactorLasso calls it `y_covar` |
| $\Sigma_F$ | Annual factor covariance, stored as `x_covar` |
| $f_t$ | Vector of the $M$ factor log returns over the period ending at $t$ |
| $c$, $\mathrm{af}_c$ | A return frequency, the key of a bucket such as `'ME'` or `'QE'`, and its periods per year |
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
  estimate. A supplied `DatedFactorCovariance` replaces the estimate and is used as given: it must already be
  annual and ordered like the factor columns.
- $\beta$ comes from the FactorLasso fit of each bucket, described in the next subsection.
- $v_i$, the diagonal of $D$, is the EWMA-weighted mean squared residual of the fit, times the
  periods per year of the bucket:

$$
v_i = \mathrm{af}_c \sum_{t} \omega_t \left( \tilde r_{i,t} - \beta_i^{\top} \tilde f_t \right)^2,
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
the explicit `cutoff`.

### Residual covariance: orthogonal and empirical

`residual_type='orthogonal'`, the default, takes the residuals of different assets as
uncorrelated: $D = \mathrm{diag}(v_1, \dots, v_N)$. `residual_type='empirical'` adds their
correlation:

$$
D = S \left[ (1 - \rho) I + \rho R \right] S ,
$$

where $R$ is a residual correlation matrix and $\rho$ is `residual_corr_weight`, in $[0, 1]$ with
default 1; the weight applies to each nonorthogonal residual choice.

The four choices are `orthogonal`, `empirical`, `exposure_cluster`, and `residual_cluster`.
The cluster choices replace $R$ by a block target: each block uses the signed arithmetic
mean of its off-diagonal residual correlations, the diagonal is one, and correlations
between blocks are zero. `exposure_cluster` uses the fitted exposure-cluster labels.
`residual_cluster` clusters the prepared residual correlation with Ward linkage,
distance `one_minus_rho`, and cutoff fraction 0.6. These are covariance assembly choices;
they do not change the fitted factor exposures or the clusters used for alpha estimation.
Both use the same causal, pairwise-available EWMA preparation as `empirical`.
FactorLasso preserves the original universe's targets when selecting a portfolio subset.

**Proposition 2 (variances are kept).** For every $\rho$ in $[0, 1]$, the diagonal of $D$ is
$v_1, \dots, v_N$; $\rho = 0$ gives the orthogonal matrix; and $D$ is positive semidefinite.
Hence all four residual types give every asset the same variance, and they differ only in the
covariances between assets.

**Proof.** $R$ has a unit diagonal, so the bracket has a unit diagonal too, and
$D_{ii} = \sqrt{v_i} \cdot 1 \cdot \sqrt{v_i} = v_i$; at $\rho = 0$ the bracket is the identity.
The bracket is a convex combination of two positive semidefinite matrices, so it is positive
semidefinite, and so is its congruence by the diagonal $S$. $\square$

The residuals of different buckets live on different dates, so the package estimates $R$ on a
common grid. It divides the stored residuals by their annual multiplier $\mathrm{af}_c$, sums
complete native periods into periods of `residual_covar_freq`, subtracts a causal EWMA mean and
takes the EWMA correlation with span `residual_covar_span`. By default the grid is the lowest
native frequency, quarterly in the example, and the span is that bucket's loading span, 12
quarters. An explicit span counts periods of the common grid. An explicitly coarser grid converts
the default decay to the new period length: with $\mathrm{af}_c$ the periods per year of the lowest
native frequency and $\mathrm{af}$ those of the grid, the decay becomes

$$
\lambda^{\mathrm{af}_c / \mathrm{af}} .
$$

An incomplete period remains NaN for the affected asset. Each pair uses its available history
through FactorLasso's EWMA NaN handling; nothing is prorated or interpolated. FactorLasso's
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

Every TF current fit truncates factor prices and all response histories at its
explicit `cutoff`, including the orthogonal residual policy. Supplied factor
covariance and external cluster assignments need availability metadata no later
than that cutoff. Input indices represent availability: the caller must account
for publication delays and restatements when preparing the observations.
An empirical rolling fit records, with each residual correlation, the last complete common period
it observes and the date it becomes available, and it keeps the last correlation between complete
periods while the loadings and residual variances update. A correlation is never used before its
availability date, and an as-of query before the first fit is refused.

## Worked example

### Construct factor prices and asset returns

The [offline script](../examples/docs/factor_covariance_hcgl.py) uses the original
fixed-seed synthetic panel: four factors, five monthly assets and three quarterly
assets. Install TradableFactors in the composed OP example environment and run:

```console
python -m examples.docs.factor_covariance_hcgl
```

The script independently checks explicit EWMA weighted sums, covariance assembly,
selected residual risk and chronological estimation. It does not reproduce the
article's empirical results.

### Fit HCGL at an explicit cutoff

The example supplies a configured FactorLasso estimator and keeps fitted state in
the returned `FactorModelSnapshot`:

```python
estimator = FactorCovarEstimator(
    lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
    rebalancing_freq='QE')
snapshot = estimator.fit_current(**history)
data, covariance = snapshot.data, snapshot.asset_covariance
```

#### A completed model at an explicit cutoff

`history` contains factor prices, the monthly/quarterly return dictionary and
`cutoff`. The fit returns the loadings, factor covariance, residual diagnostics,
selected asset covariance and provenance. `data` exposes the complete FactorLasso
container for additional statistical inspection; downstream optimisation consumes
`covariance` or the QIS view from `snapshot.to_risk_model()`.

The independent factor-covariance reference is an explicit EWMA weighted sum of
factor log returns. A second weighted-sum reference checks each cadence's residual
variances, and direct matrix multiplication verifies the assembled asset covariance.

### Residual policy belongs to the completed object

```python
empirical = replace(estimator, residual_type='empirical', residual_corr_weight=.5)
correlated = empirical.fit_current(**history)
fitted = correlated.data
```

The selected covariance retains half of the empirical residual correlation. The
example independently aggregates stored residuals to complete quarters, estimates
the common-grid correlation and reconstructs the full residual block. Export to
QIS preserves that selected total covariance. Its default factor decomposition
is withheld when QIS's diagonal residual representation cannot express the selected
full residual covariance; an explicitly labelled diagonal reference is a separate view.

The existing synthetic exhibit is retained below. It uses the same model inputs;
this migration does not regenerate or republish its historical image.

![The fixed synthetic example's systematic and residual asset variances and the
residual variance of an equal-weight private-asset portfolio at different
residual-correlation retention weights.](images/factor_covariance_variance_split.png)

### Supplied factor covariance has availability metadata

```python
supplied = DatedFactorCovariance(data.x_covar * 1.7, as_of, as_of)
external = estimator.fit_current(**history, factor_covariance=supplied)
pd.testing.assert_frame_equal(external.data.x_covar, data.x_covar * 1.7)
```

The supplied covariance is annual and is not scaled a second time. Its observation
and availability dates must be no later than the cutoff, and its factor coordinates
must match the model. The caller remains responsible for accurate availability metadata.

### Rolling fits are point-in-time

```python
period = qis.TimePeriod(AS_OF, '2024-12-31')
rolling = estimator.fit_rolling(factor_prices, buckets, time_period=period)
assert list(rolling.snapshots) == list(pd.to_datetime(ROLLING_KEYS))
```

#### Rolling fits retain the same completed-object contract

`rolling.snapshots` contains completed dated models; `rolling.covariances` is the
selected covariance dictionary ready for OP. The example compares each entry with
a current fit at the same cutoff. Shocking factor prices and asset returns after
30 June 2024 leaves earlier entries unchanged and changes later ones.

## Implementation in optimalportfolios

OP consumes prepared expected returns and risk inputs. Factor construction,
estimation and method-specific reports are supplied by TradableFactors; the old
OP factor-estimator and report exports are removed.

| TF entry point | Completed output and use |
|---|---|
| `FactorCovarEstimator.fit_current(..., cutoff=...)` | `FactorModelSnapshot`: FL diagnostics in `.data`, selected full `.asset_covariance`, `.residual_covariance`, cutoff, return coordinates, policy and provenance. |
| `FactorCovarEstimator.fit_rolling(..., time_period=...)` | `RollingFactorModel`: dated `.snapshots`, selected `.covariances` for OP and `.data` for FL statistical diagnostics. |
| `snapshot.to_risk_model()` / `rolling.to_risk_model()` | QIS interchange retaining selected covariance and supported factor metadata. It does not replace the complete fitted result. |
| `DatedFactorCovariance` | Supplied annual factor covariance with observation/availability dates and coordinate types. |
| `tradablefactors.reports.covariance.plot_covariance_snapshot` | Plots a completed snapshot using its selected covariance and existing QIS plotting primitives. |
| `tradablefactors.reports.covariance.run_rolling_covar_report` | Fits the configured rolling model and returns figures and dated diagnostic tables. |

Use `dataclasses.replace` to derive another configuration. A `LassoModel` is
required explicitly; each fit uses independent fitting state. Regression demeaning
is configured on that FL model, while factor covariance retains the QIS EWMA
contract. There is no unused estimator-level `demean` setting.

The selected residual type and weights live in the completed TF object. Calling a
raw FL getter with its own defaults can request a different assembly, so downstream
risk and optimisation should use the completed object's selected covariance.
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
  residual shock, as the retained synthetic exhibit shows. The empirical correlation on quarterly data with span 12
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

- [GLS inference of factor risk premia](gls_factor_premia.md): use the historical
  loadings and residual risk in TradableFactors' separate projection of provider forecasts.
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
- TradableFactors: `FactorCovarEstimator`, completed model objects and covariance reports.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [FactorLasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
