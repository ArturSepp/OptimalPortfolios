---
myst:
  html_meta:
    description: >-
      EWMA and sparse factor covariance estimators in OptimalPortfolios:
      return conventions, annualization, point-in-time inputs and executable examples.
---

# Covariance estimators

*[author / affiliation / date — placeholder]*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

A covariance estimator summarizes the scale and joint variation of asset returns.
OptimalPortfolios provides direct EWMA and sparse factor estimators with a shared output:
one labeled asset covariance matrix, or a dictionary of matrices keyed by decision date.

## Overview

### Estimator choice

Choose a model from the available history, return cadence and intended risk structure.
A common covariance interface does not mean every optimizer needs only covariance:
some objectives also require expected returns, signals, benchmarks or other inputs.

| Estimator | Appropriate model | Main qualification |
|---|---|---|
| `EwmaCovarEstimator` | A direct estimate for assets sampled at one return frequency. | A short history or wide universe can produce a noisy or singular matrix. |
| `FactorCovarEstimator` | Sparse factor exposures, frequency-specific asset buckets, or an HCGL risk model. | Requires factor prices, a configured `LassoModel` and sufficient history in every bucket. |

OptimalPortfolios orchestrates return alignment, fitting dates and annualization.
[qis](https://github.com/ArturSepp/QuantInvestStrats) supplies return conversion, EWMA kernels and
calendar utilities ([citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)).
[factorlasso](https://github.com/ArturSepp/FactorLasso) supplies sparse regression, clustering and
factor decomposition containers
([citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff)).

## Inputs, notation, and assumptions

| Symbol or input | Meaning |
|---|---|
| $P_{i,t}$, $r_{i,t}$ | Positive total-return price and log return of asset $i$ at observation $t$. |
| $m_t$, $u_t$ | EWMA mean vector and returns after the configured mean adjustment. |
| $s$, $\lambda$ | Span in return observations and corresponding decay factor. |
| $a$, $S_t$ | Observations per year and unannualized EWMA covariance state. |
| $\Sigma_y$, $\Sigma_x$ | Annual asset covariance and annual factor covariance, in fractional return-squared units. |
| $\beta$, $D$ | Dimensionless asset-by-factor loadings and diagonal annual residual variances. |
| `rebalancing_freq` | Frequency for reported estimation dates; it does not set return sampling frequency. |

Inputs use ordered `DatetimeIndex` rows and unique, consistently ordered instrument labels.
The price panel should represent the intended total-return convention; these estimators do not
add omitted distributions. The direct estimator constructs $r_{i,t}=\log(P_{i,t}/P_{i,t-1})$
after qis sampling at `returns_freq`.

For factor estimation, `risk_factor_prices` contains factor prices. `asset_returns_dict` contains
already constructed asset **returns**, keyed by their frequency codes, such as `"ME"` and `"QE"`.
Each asset belongs in exactly one bucket. Supply compatible log-return conventions: the adapter
constructs log factor returns on each bucket's observation dates and does not convert supplied
asset returns from simple to log form.

Factor covariance has its own `factor_returns_freq` and `factor_covar_span`. Regression and
clustering have their own `LassoModel` configuration, including frequency-specific span maps.
A span of 24 monthly observations and a span of 24 quarterly observations represent different
calendar histories. See [mixed-frequency data](mixed_frequency_data.md).

## Methodology

### EWMA covariance

For a span $s>1$, qis uses the following decay and implied half-life $h$, measured in observations:

$$
\lambda = 1-\frac{2}{s+1},
\qquad
h = \frac{\log(1/2)}{\log(\lambda)}.
$$

Thus `span=52` is approximately an 18-observation half-life, not a 52-observation half-life.
The span does not impose a hard lookback cutoff.

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
kernel. It is not an identity-shrinkage option. In the checked qis 5.26.0 implementation, that
kernel seeds volatility from the mean squared returns over the entire supplied array.
**Its use in the direct rolling EWMA path is therefore not point-in-time safe.** Later observations
can affect earlier matrices even when `time_period.end` is earlier. This qualification is
verified below and recorded separately from the ordinary EWMA formulas.

### Factor and HCGL covariance

Hierarchical cluster group LASSO (HCGL) groups assets for joint sparse factor selection.
The default factor covariance assembly is:

$$
\Sigma_y = \beta\Sigma_x\beta^{\mathsf T} + D.
$$

For $N$ assets and $M$ factors, loadings have shape $N\times M$, factor covariance is
$M\times M$, and the result is $N\times N$. The residual term is diagonal: this model does not
estimate an unrestricted cross-asset residual covariance.

`residual_var_weight` scales only the residual diagonal. Its default is 1.0; zero gives the
factor-only component. Very small values that the assembly treats as numerically close to zero
also omit that term. Lowering residual weight changes the model rather than merely its display.

For each asset-return bucket, factor prices are aligned to that bucket's dates with historical
forward-filling, then converted to log returns. FactorLasso fits the configured LASSO or group
model, including `HIERARCHICAL_CLUSTER_GROUP_LASSO` for HCGL. Betas and residual fit statistics
come from that fit; the adapter annualizes residual variances with the bucket's frequency code.

When `x_covar` is omitted, the adapter estimates factor covariance at `factor_returns_freq` and
annualizes it using the frequency conversion to a year. A supplied `x_covar` is used as provided:
it must already be annualized and ordered consistently with factor labels. Factor covariance
and residual variance must be in compatible units before assembly.

The estimator's top-level `demean` field currently does not control the factor covariance fit:
that call uses `demean=True` internally. Regression demeaning is controlled separately by
`LassoModel.demean`. This is a verified implementation qualification, not an interchangeable
configuration choice.

### Current fits and rolling dates

A current EWMA fit uses the complete supplied price panel. In the ordinary factor current-fit
path, `estimation_date` can be only a metadata label; it does not automatically truncate all
inputs. Slice factor prices and every return bucket through the intended date before calling
a current fit, including when supplying an external factor covariance.

`fit_rolling_factor_covars` explicitly slices each input through every scheduled date and uses
an expanding history. An active cluster smoother is delegated to FactorLasso over the causal
estimation schedule. The rolling wrapper checks each bucket's row count against `warmup_period`;
that is not a per-asset completeness or investability check.

The direct EWMA wrapper computes a tensor on the return grid and selects scheduled observations.
The factor wrapper generates calendar estimation dates and fits at those dates. Consequently,
the two wrappers need not produce identical keys for the same frequency string. Neither output
date independently guarantees that the upstream observations were actually available then.

## Worked example

All six blocks are canonical sequential examples using fixed synthetic data, without downloads.
The small EWMA example gives an exact arithmetic reference; the HCGL example demonstrates an
actual fit and component assembly.

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

```python
monthly_dates = pd.date_range("2018-12-31", periods=73, freq="ME")
m = np.arange(72, dtype=float)
factor_returns = np.column_stack([
    0.003 + 0.025 * np.sin(m / 3),
    0.001 + 0.018 * np.cos(m / 5),
])
factor_prices = pd.DataFrame(
    100.0 * np.exp(np.vstack([np.zeros(2), np.cumsum(factor_returns, axis=0)])),
    index=monthly_dates, columns=["Growth", "Rates"],
)
asset_values = (
    factor_returns @ np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.4]]).T
    + 0.004 * np.column_stack([np.cos(m / 2), np.sin(m / 4), np.cos(m / 6)])
)
asset_returns = pd.DataFrame(
    asset_values, index=monthly_dates[1:], columns=["Equity", "Bonds", "Balanced"],
)
```

These monthly asset returns already use the same log convention as the factor-price differences.
Their generating loadings are inputs to this teaching simulation; the fitted penalized loadings
need not recover them exactly.

### Fit HCGL at an explicit cutoff

```python
from factorlasso import LassoModel, LassoModelType

factor_estimator = opt.FactorCovarEstimator(
    lasso_model=LassoModel(
        model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-5, span=24, warmup_period=12, n_clusters=2,
    ),
    factor_returns_freq="ME", factor_covar_span=24, rebalancing_freq="QE",
)
as_of = monthly_dates[48]
factor_data = factor_estimator.fit_current_factor_covars(
    risk_factor_prices=factor_prices.loc[:as_of],
    asset_returns_dict={"ME": asset_returns.loc[:as_of]},
    estimation_date=as_of,
)
factor_covar = factor_data.get_y_covar()
factor_only = factor_data.get_y_covar(residual_var_weight=0.0)
scaled_residual_covar = factor_data.get_y_covar(residual_var_weight=0.35)
```

The cutoff is **31 December 2022**. The fitted object contains three asset rows and two factor
columns. All three assembled outputs are 3-by-3 matrices in annual units. Moving residual weight
from 1.0 to 0.35 changes only the diagonal; it does not refit betas or factor covariance.

### Obtain rolling matrices

```python
import qis

ewma_period = qis.TimePeriod(dates[60], dates[100])
rolling_covars = estimator.fit_rolling_covars(prices=prices, time_period=ewma_period)
factor_period = qis.TimePeriod(monthly_dates[47], monthly_dates[54])
rolling_factor_covars = factor_estimator.fit_rolling_covars(
    risk_factor_prices=factor_prices,
    asset_returns_dict={"ME": asset_returns},
    time_period=factor_period,
)
```

In these fixtures, the EWMA keys are **6 April, 6 July and 5 October 2022**, on the weekly
observation grid. The factor keys are **31 December 2022, 31 March and 30 June 2023**, on calendar
quarter ends. The two examples use different input histories and periods; they illustrate the
date contracts rather than compare investment performance.

## Implementation in optimalportfolios

| Entry point | Inputs and output |
|---|---|
| `EwmaCovarEstimator.fit_current_covar` | Price panel to one annual covariance matrix. |
| `EwmaCovarEstimator.fit_rolling_covars` | Price panel and reporting period to a date-keyed dictionary. |
| `FactorCovarEstimator.fit_current_covar` | Factor prices and asset-return buckets to one assembled annual matrix. |
| `FactorCovarEstimator.fit_rolling_covars` | The factor inputs and period to a dictionary of assembled matrices. |
| `fit_current_factor_covars` / `fit_rolling_factor_covars` | Factor-specific methods retaining betas, residual statistics, clustering and decomposition metadata. |

The input signatures differ even though the output contract is shared. The direct class has no
separate shrinkage-to-identity parameter. The legacy exported `estimate_rolling_ewma_covar`
function is a QIS re-export; do not assume its defaults and date semantics are identical to this
class.

Factor decomposition fields include `y_betas`, `x_covar`, `y_variances` and, where applicable,
`clusters`, `linkages` and `cutoffs`. Cluster identifiers are prefixed by their frequency bucket.
The reported `residuals` panel is scaled by each bucket's annualization factor, and is constructed
without subtracting a fitted intercept. It is not the original per-observation residual series;
do not recompute $D$ as its ordinary sample variance.

The fitting helper mutates the supplied `LassoModel` with the final bucket's fitted state.
Use returned decomposition objects for combined results. Reusing a configuration object does
not make its attached last-fit state a record of all buckets or dates.

### Reproduction and verification context

[Executable article tests](../src/optimalportfolios/tests/covariance_estimators_documentation_test.py)
run all six blocks, verify independent weighted-sum and factor-component references, and perturb
future observations to check the documented timing distinctions:

```console
python -m pytest src/optimalportfolios/tests/covariance_estimators_documentation_test.py
```

Local verification used the OptimalPortfolios 7.6.0 working source with Python 3.12.14,
qis 5.26.0, factorlasso 0.18.0, CVXPY 1.9.2 and CLARABEL 0.11.1. The existing lockfile records
qis 5.22.3; these results characterize the inspected local environment, not a locked installation.

## Interpretation and limitations

### Units and validation

Optimizer covariance inputs are not automatically resampled or annualized. Supply annual units
when using annual volatility or tracking-error limits. Check identical row/column labels, finite
values after the chosen asset policy, symmetry, eigenvalues and marginal volatilities.
Some solver paths factorize or regularize the supplied matrix; inspect their diagnostics rather
than treating solver acceptance as a certificate of data quality.

Zero-filled missing betas or residual variances can make an absent-history asset look riskless.
Eligibility and warm-up policy remain separate from matrix assembly. Sparse factor structure,
cluster choices, residual scaling and return normalization all change the risk model.

For historical use, preserve data as it was known on each date. Direct ordinary EWMA passed a
future-price perturbation check in the included fixture; the optional normalized-return rolling
path did not, owing to full-array volatility initialization. Rolling factor fitting explicitly
truncates inputs, while a current fit requires the caller's cutoff. A long warm-up may reduce
initialization effects but does not prove that look-ahead is absent.

The factor adapter's demeaning-field limitation and the direct normalized-return timing
limitation are documented here without numerical changes. Use the checked causal configuration
and inspect the implementation when changing either option.

## See also

- [Rolling factor covariance from CSV](rolling_factor_covar_from_csv.md) for the complete
  Yahoo-to-CSV and CSV-to-HCGL workflow, including the ROSAA-free MATF handoff.
- [Mixed-frequency data](mixed_frequency_data.md) and [incomplete histories](incomplete_histories.md).
- [API reference](api.rst) for the estimator classes.
- [Offline mixed-frequency covariance example](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/covar_estimation/demo_covar_different_estimation_freqs.py).
- [Factor covariance example](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/covar_estimation/lasso_covar_estimation.py)
  (requires network data).
- [Documentation analytics](documentation_standard.md#covariance-comparison-family-preview)
  for the fixed offline estimator-comparison exhibit and its provenance.

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
- OptimalPortfolios: [EWMA estimator](../src/optimalportfolios/covar_estimation/ewma_covar_estimator.py),
  [return preparation](../src/optimalportfolios/covar_estimation/utils.py) and
  [factor estimator](../src/optimalportfolios/covar_estimation/factor_covar_estimator.py).
- qis: [EWMA kernels](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/models/linear/ewm.py).
