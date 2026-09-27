---
myst:
  html_meta:
    description: >-
      EWMA covariance and the shared covariance-estimator contract in OptimalPortfolios: return
      conventions, the EWMA recursion, annualization, point-in-time inputs and executable
      examples; the sparse factor estimator has its own page.
---

# Covariance estimators

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/OptimalPortfolios/commit/fb8848d327c0585eaf0933dba6137ec6b8338bbf)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

A covariance estimator summarizes the scale and joint variation of asset returns.
OptimalPortfolios provides direct EWMA and sparse factor estimators with a shared output:
one labeled asset covariance matrix, or a dictionary of matrices keyed by decision date.
This page covers the EWMA estimator and that shared contract; the factor estimator is described
in [Factor covariance with HCGL](factor_covariance_hcgl.md).

## Overview

### Estimator choice

Choose a model from the available history, return cadence and intended risk structure.
A common covariance interface does not mean every optimizer needs only covariance:
some objectives also require expected returns, signals, benchmarks or other inputs.

| Estimator | Appropriate model | Main qualification |
|---|---|---|
| `EwmaCovarEstimator` | A direct estimate for assets sampled at one return frequency. | A short history or wide universe can produce a noisy or singular matrix. |
| `FactorCovarEstimator` | Sparse factor exposures, frequency-specific asset buckets, or an HCGL risk model; see [Factor covariance with HCGL](factor_covariance_hcgl.md). | Requires factor prices, a configured `LassoModel` and sufficient history in every bucket. |

OptimalPortfolios orchestrates return alignment, fitting dates and annualization.
[qis](https://github.com/ArturSepp/QuantInvestStrats) supplies return conversion, EWMA kernels and
calendar utilities ([citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)).
[factorlasso](https://github.com/ArturSepp/FactorLasso) supplies the sparse regression, clustering
and factor decomposition containers of the factor estimator
([citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff)).

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Log returns: `EwmaCovarEstimator` takes log differences of prices sampled at `returns_freq` |
| Estimation grid | Weekly Wednesdays (`"W-WED"`, the default `returns_freq`) with span 52 in the direct example; month ends (`"ME"`) with span 3 in the three-observation example |
| Rebalancing grid | Quarter ends (`"QE"`, the default `rebalancing_freq`): the rolling wrapper reports the first weekly observation on or after each quarter end |
| Covariance units | Annual, fractional log-return squared: the EWMA state times the observations per year of the sampled returns (52 weekly, 12 monthly) |
| Expected returns | None |
| Weight state | None; the page estimates covariance matrices, not weights |
| Solver | None; the EWMA estimators solve nothing |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol or input | Meaning |
|---|---|
| $P_{i,t}$, $r_{i,t}$ | Positive total-return price and log return of asset $i$ at observation $t$. |
| $m_t$, $u_t$ | EWMA mean vector and returns after the configured mean adjustment. |
| $s$, $\lambda$ | Span in return observations and corresponding decay factor. |
| $a$, $S_t$ | Observations per year and unannualized EWMA covariance state. |
| $\Sigma_y$ | Annual asset covariance, in fractional return-squared units. |
| `rebalancing_freq` | Frequency for reported estimation dates; it does not set return sampling frequency. |

Inputs use ordered `DatetimeIndex` rows and unique, consistently ordered instrument labels.
The price panel should represent the intended total-return convention; these estimators do not
add omitted distributions. The direct estimator constructs $r_{i,t}=\log(P_{i,t}/P_{i,t-1})$
after qis sampling at `returns_freq`.

The factor estimator takes factor prices and already constructed asset log returns by frequency
bucket instead; its inputs and their clocks are described in
[Factor covariance with HCGL](factor_covariance_hcgl.md#inputs-notation-and-assumptions) and
[mixed-frequency data](mixed_frequency_data.md).

## Methodology

### EWMA covariance

For a span $s \gt 1$, qis uses the following decay and implied half-life $h$, measured in observations:

$$
\lambda = 1-\frac{2}{s+1},
\qquad
h = \frac{\log(1/2)}{\log(\lambda)}.
$$

Thus `span=52` is approximately an 18-observation half-life, not a 52-observation half-life.

> **Insight.** A span is neither a half-life nor a window. The weights decay geometrically but
> never reach zero, so the span imposes no hard lookback cutoff: in the worked example below,
> changing the first of 160 weekly prices still moves the current matrix.

With `demean=True`, the adapter subtracts the current EWMA mean:

$$
m_t = \lambda m_{t-1} + (1-\lambda)r_t,
\qquad
u_t = r_t-m_t.
$$

For finite data, qis seeds the mean from the first available return. The adapter drops the
first price-difference row, then drops the first mean-adjusted return because its deviation
is zero. With `demean=False` it retains the raw log returns after the first difference and
uses $u_t=r_t$.

The ordinary covariance kernel starts from a zero matrix and applies:

$$
S_t = \lambda S_{t-1} + (1-\lambda)u_tu_t^{\mathsf T},
\qquad
\Sigma_{y,t} = aS_t.
$$

This is a recursive exponentially weighted second moment of the configured adjusted returns.
It is not a sample covariance with a degrees-of-freedom correction, and its finite-history
weights are not renormalized to sum to one. The class infers the annualization factor from
the sampled return index; `rebalancing_freq` only selects output dates.

`NanBackfill.ZERO_FILL` resets non-finite covariance updates to zero. This does not generally
mean replacing a missing input return with zero before recursion. Sampling may already have
handled a missing source price. See [incomplete histories](incomplete_histories.md).

The optional `is_apply_vol_normalised_returns=True` selects a different, DCC-like normalized-return
kernel. It is not an identity-shrinkage option. From qis 5.31.0, which this package requires, that
kernel seeds each volatility with the column's first finite squared return, so the direct rolling
EWMA path is point in time with either kernel: each rolling matrix equals a current fit on the
prices through its date, and later prices change none of them. Earlier qis releases seeded that
volatility from the mean squared return of the entire supplied array, so later observations could
move earlier matrices even when `time_period.end` was earlier.

### Factor and HCGL covariance

The factor estimator assembles $\Sigma_y = \beta \Sigma_F \beta^{\top} + D$ from FactorLasso
loadings, a qis factor covariance and residual variances, each in annual units. Its methodology
has moved to
[Factor covariance with HCGL](factor_covariance_hcgl.md#the-factor-model-and-its-covariance).

### Common-frequency empirical residual covariance

The empirical residual covariance of the factor estimator, which estimates residual correlation on
a common return grid, is described in
[Factor covariance with HCGL](factor_covariance_hcgl.md#residual-covariance-orthogonal-and-empirical).

### Current fits and rolling dates

A current EWMA fit uses the complete supplied price panel; slice the prices through the intended
date for a historical fit. The direct EWMA wrapper computes a tensor on the return grid and
selects scheduled observations, so its keys are return-grid dates.

The factor wrapper instead fits at calendar dates, slicing every input through each date, and its
ordinary current fit treats `estimation_date` as a label, not a cutoff; see
[the point-in-time contract](factor_covariance_hcgl.md#the-point-in-time-contract). Consequently,
the two wrappers need not produce identical keys for the same frequency string. Neither output
date independently guarantees that the upstream observations were actually available then.

## Worked example

The four Python blocks below run in order on fixed synthetic data and need no download, data file
or random seed. They are excerpts of the canonical script
[`examples/docs/covariance_estimators.py`](../examples/docs/covariance_estimators.py), which runs
them and asserts every number on this page, and its main properties, against a reference
computed a different way:

```console
python -m examples.docs.covariance_estimators
```

The small EWMA example gives an exact arithmetic reference. The factor example is on the
[factor covariance page](factor_covariance_hcgl.md#worked-example).

### Create weekly total-return prices

```python
import numpy as np
import pandas as pd

dates = pd.date_range("2021-01-06", periods=160, freq="W-WED")
steps = np.arange(len(dates), dtype=float)
prices = pd.DataFrame(
    {"Equity": 100.0 * np.exp(0.002 * steps + 0.05 * np.sin(steps / 5)),
     "Bonds": 100.0 * np.exp(0.0007 * steps + 0.02 * np.cos(steps / 7))},
    index=dates,
)
```

Estimate current covariance from the synthetic `prices` panel:

```python
import optimalportfolios as opt

estimator = opt.EwmaCovarEstimator(
    returns_freq="W-WED",
    span=52,
    rebalancing_freq="QE",
    demean=True,
)
current_covar = estimator.fit_current_covar(prices=prices)
```

`current_covar` is a 2-by-2 annualized covariance matrix with `Equity` and `Bonds` on both axes.
Changing `rebalancing_freq` would not change this current fit.

### Three observations with an exact result

```python
small_returns = np.array([[0.01, 0.02], [-0.02, 0.01], [0.03, -0.01]])
small_prices = pd.DataFrame(
    100.0 * np.exp(np.vstack([np.zeros(2), np.cumsum(small_returns, axis=0)])),
    index=pd.date_range("2023-12-31", periods=4, freq="ME"),
    columns=["A", "B"],
)
small_estimator = opt.EwmaCovarEstimator(returns_freq="ME", span=3, demean=False)
small_covar = small_estimator.fit_current_covar(prices=small_prices)
```

The three monthly log-return vectors receive weights **1/8, 1/4 and 1/2**, in chronological
order. Their sum is 7/8 because the recursion starts from zero. Multiplying the weighted
second moments by 12 gives:

| Annual covariance | A | B |
|---|---:|---:|
| A | 0.006750 | -0.002100 |
| B | -0.002100 | 0.001500 |

The numbers are fractional return-squared units, not volatility percentages. The example turns
demeaning off to expose the recursion; it does not change the class default.

### Construct factor prices and asset returns

The factor example now builds four factors and eight assets on a monthly and a quarterly grid; see
[Factor covariance with HCGL](factor_covariance_hcgl.md#construct-factor-prices-and-asset-returns).

### Fit HCGL at an explicit cutoff

The HCGL fit at an explicit cutoff, with each component checked against an independent reference,
is on the [factor covariance page](factor_covariance_hcgl.md#fit-hcgl-at-an-explicit-cutoff).

### Obtain rolling matrices

```python
import qis

ewma_period = qis.TimePeriod(dates[60], dates[100])
rolling_covars = estimator.fit_rolling_covars(prices=prices, time_period=ewma_period)
```

In this fixture, the EWMA keys are **6 April, 6 July and 5 October 2022**, on the weekly
observation grid. Rolling factor matrices are keyed by calendar quarter ends instead; see
[rolling fits](factor_covariance_hcgl.md#rolling-fits-are-point-in-time).

> **Pitfall.** Rolling EWMA keys are return-grid dates, not calendar quarter ends. With weekly
> Wednesday returns and quarter-end rebalancing, the example's keys are 6 April, 6 July and
> 5 October 2022, so a lookup of 31 March 2022 finds no matrix.

## Implementation in optimalportfolios

| Entry point | Inputs and output |
|---|---|
| `EwmaCovarEstimator.fit_current_covar` | Price panel to one annual covariance matrix. |
| `EwmaCovarEstimator.fit_rolling_covars` | Price panel and reporting period to a date-keyed dictionary. |
| `estimate_current_ewma_covar` | The function behind `fit_current_covar`; `apply_an_factor=False` keeps per-observation units. |

`FactorCovarEstimator` has the same two methods with its own inputs, and factor-specific methods
that keep every component; see
[its implementation](factor_covariance_hcgl.md#implementation-in-optimalportfolios). The input
signatures differ even though the output contract is shared. The direct class has no
separate shrinkage-to-identity parameter. The legacy exported `estimate_rolling_ewma_covar`
function is a QIS re-export; do not assume its defaults and date semantics are identical to this
class.

### Reproduction and verification context

The [canonical script](../examples/docs/covariance_estimators.py) runs the worked example and
checks the EWMA weighted sums, the exact three-observation weights, the rolling key dates and the
future-price timing properties of both EWMA kernels. The test suite runs it, and so does the
offline examples lane of CI.

## Interpretation and limitations

### Units and validation

Optimizer covariance inputs are not automatically resampled or annualized. Supply annual units
when using annual volatility or tracking-error limits. Check identical row/column labels, finite
values after the chosen asset policy, symmetry, eigenvalues and marginal volatilities.
Some solver paths factorize or regularize the supplied matrix; inspect their diagnostics rather
than treating solver acceptance as a certificate of data quality.

For historical use, preserve data as it was known on each date. Both direct EWMA kernels, the
ordinary one and the optional normalized-return one, pass a future-price perturbation check in
the worked example, and each of their rolling matrices equals a current fit on the prices
through its date. A long warm-up may reduce initialization effects but does not prove that
look-ahead is absent.

The factor estimator has its own limitations, among them zero-filled assets that look riskless,
the residual weight, and a top-level `demean` field that it stores but does not read; set
regression demeaning through `LassoModel.demean`. See
[Factor covariance with HCGL](factor_covariance_hcgl.md#interpretation-and-limitations).

## See also

- [Factor covariance with HCGL](factor_covariance_hcgl.md) for the sparse factor estimator,
  its residual types and its point-in-time contract.
- [Rolling factor covariance from CSV](rolling_factor_covar_from_csv.md) for the complete
  Yahoo-to-CSV and CSV-to-HCGL workflow, including the ROSAA-free MATF handoff.
- [Mixed-frequency data](mixed_frequency_data.md) and [incomplete histories](incomplete_histories.md).
- [API reference](api.rst) for the estimator classes.
- [Offline mixed-frequency covariance example](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/covar_estimation/demo_covar_different_estimation_freqs.py).
- [Factor covariance example](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/covar_estimation/lasso_covar_estimation.py)
  (requires network data).
- [Analytics gallery](analytics_gallery.md#covariance-estimators) for a fixed offline exhibit
  that feeds six covariance estimates to one minimum-variance construction on a simulation with
  known covariance; the [documentation standard](documentation_standard.md#covariance-comparison-family-preview)
  records its provenance.

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
- OptimalPortfolios: [EWMA estimator](../src/optimalportfolios/covar_estimation/ewma_covar_estimator.py),
  [return preparation](../src/optimalportfolios/covar_estimation/utils.py) and
  [factor estimator](../src/optimalportfolios/covar_estimation/factor_covar_estimator.py).
- qis: [EWMA kernels](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ewm.py).
