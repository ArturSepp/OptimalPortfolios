---
myst:
  html_meta:
    description: >-
      Mixed-frequency returns, signal horizons and factor covariance in OptimalPortfolios,
      with an offline monthly and quarterly example and explicit information timing.
---

# Mixed-frequency data

*[author / affiliation / date — placeholder]*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

**Mixed-frequency data** are observations whose reliable measurement intervals differ across
assets or model inputs. OptimalPortfolios keeps separate return panels for each cadence,
computes signals or factor exposures within those panels, and combines the resulting model
outputs. This does not turn quarterly observations into monthly statistical observations.

## Overview

The pipeline separates three clocks:

| Clock | Meaning | Example |
|---|---|---|
| Estimation cadence | Observations used to estimate a signal, exposure or risk model | Monthly liquid-asset returns; quarterly private-asset returns |
| Signal cadence | Dates when a newly computed signal is available | Monthly or quarterly, subject to publication delay |
| Rebalance cadence | Dates when a portfolio may select a target | Quarterly decisions using the latest available inputs |

Changing one clock must not silently change the others. A monthly covariance output can
reuse quarterly exposure estimates without adding a quarterly observation. A signal
calculation does not itself trade: implementation timing belongs to the
[rolling backtest](rolling_backtests.md).

Return sampling and signal primitives are delegated to
[qis](https://github.com/ArturSepp/QuantInvestStrats)
([citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)).
Sparse factor regression is delegated to
[factorlasso](https://github.com/ArturSepp/FactorLasso)
([citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff)).

## Inputs, notation, and assumptions

Use finite positive total-return prices or NAVs in a consistent currency, with unique asset
labels and an ordered, unique `DatetimeIndex`. A price-only series omits distributions;
a return convention cannot repair that omission.

| Input or symbol | Contract |
|---|---|
| $P_{i,k}^{(f)}$ | Asset $i$'s observed price at endpoint $k$ on cadence $f$ |
| $r_{i,k}^{(f)}$ | Fractional log return between consecutive sampled endpoints |
| `returns_freq` / `returns_freqs` | A frequency string or Series mapping every selected asset to a frequency |
| `ME`, `QE` | Calendar month end and calendar quarter end; use the exact keys in parameter mappings |
| $L_f$, $s_f$ | Included and skipped observation counts for classic momentum |
| $S_f$ | EWMA span measured in observations at cadence $f$, not years |
| $a_f$ | Annualization multiplier: 12 for monthly and 4 for quarterly variances |
| Information cutoff | Date through which an input was actually known, distinct from its measurement date |

The worked example has no publication delay, revisions, costs or backtest. Its dates are
fixed synthetic sample dates, not a market-data as-of date. Private-asset NAVs exist only at
quarter ends; intervening monthly rows are missing. This is a sampling example, not an
appraisal-unsmoothing model or a claim that those NAVs were tradable at the displayed values.

## Methodology

### Per-asset return frequencies

For each cadence, sample prices at its endpoints and compute returns within that bucket:

$$
r_{i,k}^{(f)} = \log\!\left(\frac{P_{i,k}^{(f)}}{P_{i,k-1}^{(f)}}\right),
\qquad
R_{i,k}^{(f)} = \exp\!\left(r_{i,k}^{(f)}\right)-1.
$$

Here $R_{i,k}^{(f)}$ is the arithmetic return over the same interval. Neither return is
annualized. When intermediate prices are observed, a quarterly log return equals the
sum of its three monthly log returns; a quarterly-only NAV does not reveal that monthly path.

`qis.compute_asset_returns_dict` returns a dictionary keyed by cadence.
`UniverseData.get_asset_returns_dict` delegates to it and defaults to arithmetic returns.
Set `is_log_returns=True` for the factor and momentum examples here. The retained first zero
is an initialization convention, not a measured return before the sample starts. The signal
functions sample prices independently and retain a missing initial return.

QIS normally forward-fills missing prices while sampling. A missing endpoint can therefore
become a stale price and a zero return, rather than a newly observed value. Validate endpoints
and information availability before sampling; consult [incomplete histories](incomplete_histories.md).
Pandas distinguishes sampling frequency from bin labels; see its
[time-series guide](https://pandas.pydata.org/docs/user_guide/timeseries.html#resampling).

### Signal horizons

Risk-adjusted momentum, classic momentum, low-beta and residual momentum accept a
per-asset `returns_freq` Series. Their relevant horizon parameters accept mappings keyed
by cadence. Those parameters have different meanings.

For an EWMA span $S_f>1$, the decay $\lambda_f$ and half-life $h_f$ satisfy:

$$
\lambda_f = 1-\frac{2}{S_f+1},
\qquad
h_f = \frac{\log(1/2)}{\log(\lambda_f)}.
$$

The half-life is measured in observations. `long_span={"ME": 12, "QE": 4}` scales the
smoothing horizon with cadence; it is not an exact one-year window or an exactly matched
calendar half-life. An EWMA retains older information with decaying weight.
See the [pandas decay definitions](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html).

Classic momentum sums exactly $L_f$ completed log returns after excluding the latest
$s_f$ observations:

$$
m_{i,k}^{(f)} = \sum_{j=0}^{L_f-1} r_{i,k-s_f-j}^{(f)}.
$$

Twelve monthly returns and four quarterly returns each cover one year. But
`skip_periods={"ME": 1, "QE": 1}` excludes one month for the monthly bucket and one
quarter for the quarterly bucket. The included calendar windows therefore end on
different dates, even at a shared formation date. The result is a cumulative log return,
not an annualized statistic or a volatility-adjusted EWMA signal.

For the two standard momentum functions below, scores are standardized **within each
cadence bucket**, and within each fixed group if `group_data` is supplied. They are not
a ranking across the full mixed universe. A bucket with one asset has no cross-sectional
dispersion and produces missing scores. Raw signals and scores are merged in the
original asset order and forward-filled between update dates. This can also carry a prior
valid value across a missing update; it does not provide an explicit staleness flag.

### Risk estimation and rebalancing

`FactorCovarEstimator` consumes the frequency-keyed asset-return dictionary. To estimate
each asset's factor exposure, it aligns factor prices to that bucket's endpoints and computes
factor log returns over the same intervals. Separately, `factor_returns_freq` controls the
factor covariance observations, and `factor_covar_span` counts those observations.

Beta-estimation spans belong in `LassoModel.span_freq_dict`; for example, `{"ME": 24, "QE": 8}`.
For clustering models, `cluster_correlation_span_freq_dict` can configure that separate
estimation span. `warmup_period` is one observation count shared across buckets, so eight
observations imply different calendar histories.

The factor covariance is scaled at its own cadence. Each asset's residual variance is
scaled at its asset-return cadence:

$$
\Sigma_{\mathrm{annual}}
= B\left(a_x\Sigma_x^{(f_x)}\right)B^\mathsf{T}
+ \operatorname{diag}\!\left(a_{f_i}v_{\epsilon,i}^{(f_i)}\right).
$$

$B$ is the dimensionless asset-by-factor loading matrix, $f_x$ the factor covariance cadence,
and $v_{\epsilon,i}^{(f_i)}$ the residual variance estimate in asset $i$'s return units.
The default assembly includes the full residual diagonal. The result has annual
fractional-return-squared units. Different sampling intervals can change the estimate even
after annualization; scale conversion does not make them statistically equivalent.
See [covariance estimators](covariance_estimators.md) for the complete decomposition contract.

`rebalancing_freq` controls rolling covariance output dates. Portfolio decisions must use
the intended schedule and subsequent implementation convention. For homogeneous data,
`EwmaCovarEstimator` uses one asset `returns_freq` and a separate rebalance cadence.

## Worked example

Run the following six Python blocks in order. They need no download, data file or random seed.

### Create monthly prices and quarterly NAV observations

The deterministic panel contains 73 month-end rows from 31 December 2018 through
31 December 2024, including 25 observed quarterly NAVs. Asset names are teaching labels.

```python
import numpy as np
import pandas as pd

dates = pd.date_range("2018-12-31", "2024-12-31", freq="ME")
t = np.arange(len(dates), dtype=float)
prices = pd.DataFrame({
    "Global Equity": 100 * np.exp(0.01*t + 0.02*np.sin(t/3)),
    "Government Bonds": 100 * np.exp(0.003*t + 0.01*(np.cos(t/4)-1)),
    "Private Assets": 100 * np.exp(0.007*t + 0.015*np.sin(t/5)),
}, index=dates)
prices.loc[~dates.is_quarter_end, "Private Assets"] = np.nan
```

### Separate the return buckets

```python
import pandas as pd
import qis
import optimalportfolios as opt

return_frequencies = pd.Series({
    "Global Equity": "ME",
    "Government Bonds": "ME",
    "Private Assets": "QE",
})
returns_by_frequency = qis.compute_asset_returns_dict(
    prices=prices,
    returns_freqs=return_frequencies,
    is_log_returns=True,
    drop_first=False,
    is_first_zero=True,
)
```

`"ME"` contains 73 rows and two assets; `"QE"` contains 25 rows and one asset. Each
starts with a zero. Do not fill the quarterly **return panel** onto the monthly grid
and treat repeated values as independent monthly observations.

### Compute EWMA momentum at each cadence

```python
scores, raw_signal = opt.compute_momentum_alpha(
    prices=prices,
    returns_freq=return_frequencies,
    long_span={"ME": 12, "QE": 4},
    vol_span={"ME": 13, "QE": 4},
)
ewma_scores, ewma_raw = scores, raw_signal
```

With these defaults there is no benchmark subtraction, the short EWMA leg is disabled,
and the volatility mean-adjustment setting is `qis.MeanAdjType.NONE`. The long span also
controls the leading warmup. The wrapper uses `weight_lag=0`: outputs are formation-date
signals, so a backtest must arrange implementation after the inputs are known.

### Compute classic momentum with an explicit skip

```python
scores, raw_signal = opt.compute_classic_momentum_alpha(
    prices=prices,
    returns_freq=return_frequencies,
    lookback_periods={"ME": 12, "QE": 4},
    skip_periods={"ME": 1, "QE": 1},
)
classic_scores, classic_raw = scores, raw_signal
```

The private-asset raw signal is carried unchanged in January and February, then updates
in March. Values below are cumulative log returns rounded to six decimals.

| Formation date | Global Equity | Government Bonds | Private Assets |
|---|---:|---:|---:|
| 2023-12-31 | 0.133758 | 0.023399 | 0.064028 |
| 2024-01-31 | 0.144017 | 0.019965 | 0.064028 |
| 2024-02-29 | 0.151632 | 0.017527 | 0.064028 |
| 2024-03-31 | 0.155765 | 0.016237 | 0.078566 |

At 31 December 2023 the monthly window runs from 30 November 2022 to 30 November
2023. The quarterly window runs from 30 September 2022 to 30 September 2023.
Both include a year of returns but exclude different latest intervals.

Both momentum calls return missing `Private Assets` **scores** because it is the only
quarterly asset. Its finite raw signal is not evidence of a usable cross-sectional score.

### Check the universe wrapper's return convention

```python
metadata = pd.DataFrame({
    "name": prices.columns,
    "asset_class": ["Equity", "Bonds", "Private"],
    "currency": "USD",
}, index=prices.columns)
universe = opt.UniverseData(prices=prices, metadata=metadata)
arithmetic_by_frequency = universe.get_asset_returns_dict(
    returns_freqs=return_frequencies,
)
log_by_frequency = universe.get_asset_returns_dict(
    returns_freqs=return_frequencies, is_log_returns=True,
)
```

The default arithmetic panels equal `np.expm1` of the log panels for these sampled
intervals. The log panels match `returns_by_frequency`. The wrapper currently annotates
`returns_freqs` as `str`, but passes a Series through to QIS at runtime.

### Estimate mixed-frequency factor covariance

For a small reproducible model, the two liquid series also serve as factor prices.
This deliberate overlap simplifies the fixture; it is not an independent empirical factor study.
The regression uses LASSO with penalty 1e-5, demeaning and CLARABEL. Volatility normalization,
clustering and fallback solvers are disabled by the selected defaults.

```python
factor_prices = prices[["Global Equity", "Government Bonds"]].rename(columns={
    "Global Equity": "Market", "Government Bonds": "Rates",
})
factor_estimator = opt.FactorCovarEstimator(
    lasso_model=opt.LassoModel(
        model_type=opt.LassoModelType.LASSO,
        reg_lambda=1e-5,
        span_freq_dict={"ME": 24, "QE": 8},
        warmup_period=8,
        demean=True,
        solver="CLARABEL",
    ),
    factor_returns_freq="ME",
    factor_covar_span=24,
    rebalancing_freq="QE",
)
cutoff = pd.Timestamp("2023-12-31")
factor_data = factor_estimator.fit_current_factor_covars(
    risk_factor_prices=factor_prices.loc[:cutoff],
    asset_returns_dict={
        freq: returns.loc[:cutoff] for freq, returns in returns_by_frequency.items()
    },
    assets=prices.columns,
    estimation_date=cutoff,
)
annual_covar = factor_data.y_covar
rolling_data = factor_estimator.fit_rolling_factor_covars(
    risk_factor_prices=factor_prices,
    asset_returns_dict=returns_by_frequency,
    assets=prices.columns,
    time_period=qis.TimePeriod("2023-06-30", "2023-12-31"),
)
rolling_covars = rolling_data.get_y_covars()
```

The current covariance has three assets in the original order. The rolling result has
three quarterly keys: 30 June, 30 September and 31 December 2023. It truncates input
histories at each date. The current call slices **both** factor prices and every asset
return bucket: `estimation_date` alone does not impose a general historical cutoff.
Calendar dates also do not certify that a delayed NAV was already published.

## Implementation in optimalportfolios

| Entry point | Input and result |
|---|---|
| `qis.compute_asset_returns_dict` | Price panel and cadence mapping to separate return panels |
| `opt.UniverseData.get_asset_returns_dict` | Same sampling via the universe container; arithmetic by default |
| `opt.compute_momentum_alpha` | Prices to cadence/group scores and raw risk-adjusted EWMA signals |
| `opt.compute_classic_momentum_alpha` | Prices to cadence/group scores and raw fixed-window log returns |
| `opt.FactorCovarEstimator` | Factor prices and asset-return buckets to annual covariance decompositions |

Ordinary sources: [universe adapter](../src/optimalportfolios/universe/universe_data.py),
[EWMA momentum](../src/optimalportfolios/alphas/signals/momentum.py),
[classic momentum](../src/optimalportfolios/alphas/signals/classic_momentum.py), and
[factor covariance](../src/optimalportfolios/covar_estimation/factor_covar_estimator.py).
Generated signatures are in the [API reference](api.rst). QIS owns the
[return sampler](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/perfstats/returns.py).

The [article verification](../src/optimalportfolios/tests/mixed_frequency_data_documentation_test.py)
executes these blocks and checks returns, observation windows, per-cadence variance units,
scoring, stale updates and historical input cutoffs. From a configured contributor environment:

```console
python -m pytest src/optimalportfolios/tests/mixed_frequency_data_documentation_test.py -q
```

Locally verified on 14 September 2026 with OptimalPortfolios 7.6.0 working source,
qis 5.26.0, factorlasso 0.18.0 and Python 3.12.14. The supplied environment differs from
the existing lockfile; this is not a claim that the locked installation was tested.
GitHub and VS Code preview checks remain pending.

## Interpretation and limitations

### Failure modes

- **Incomplete mappings:** every selected asset needs a unique valid cadence, and every
  mapping-valued horizon must cover each cadence. Signal functions select the price columns
  from the Series; the QIS dictionary helper follows the supplied mapping and can omit an
  unmapped price column. Validate coverage before calling either API.
- **Stale observations:** QIS price filling and signal filling do not track publication
  dates, revisions or elapsed time since the last genuine observation. Retain those records
  separately and enforce an explicit eligibility policy.
- **Delayed publication:** a NAV measured on 31 March but first published on 15 May is
  unavailable for an April decision. Keep its measurement interval for estimation, but
  exclude it from historical inputs until publication. Relabelling it to May and treating
  it as a regular quarterly endpoint would change the return-interval contract.
- **False horizon equivalence:** spans are exponential smoothing parameters; classic
  lookbacks are exact counts; skip counts and warmup counts depend on cadence.
- **Small scoring buckets:** one asset or equal raw signals yield zero dispersion. A
  forward-filled score is not proof of a fresh, informative estimate. Cluster-scored
  variants have their own grouping contract; do not infer it from the standard calls here.
- **Annualization assumptions:** multiplying variances by observations per year does not
  correct serial dependence, appraisal smoothing, stale marks or asynchronous exposure changes.
  The factor residual diagonal omits cross-asset residual covariance.
- **Timing and configuration limits:** the
  [covariance article](covariance_estimators.md#interpretation-and-limitations) records the
  normalized direct-EWMA initialization limitation and the unwired top-level factor
  `demean` field. The current example explicitly slices histories and uses unnormalized
  covariance; production numerical behavior is unchanged.

## See also

- [Covariance estimators](covariance_estimators.md) and [rolling backtests](rolling_backtests.md).
- [Incomplete histories](incomplete_histories.md) and [constraints](constraints.md).
- [Rolling factor covariance from CSV](rolling_factor_covar_from_csv.md).
- [Offline mixed-frequency covariance example](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/covar_estimation/demo_covar_different_estimation_freqs.py).
- [API reference](api.rst).

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
- [Pandas: exponentially weighted calculations and decay parameters](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html).
- [Pandas: time-series sampling and endpoint labels](https://pandas.pydata.org/docs/user_guide/timeseries.html#resampling).

