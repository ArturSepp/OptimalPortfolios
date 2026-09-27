---
myst:
  html_meta:
    description: >-
      UniverseData in optimalportfolios: the validated container of prices, metadata, two levels
      of group loadings and asset-class identifiers, how its loadings feed group constraints and
      risk budgets, and how an appraisal-smoothed private-equity series is unsmoothed before
      estimation, with a verified offline example.
---

# Universe data and appraisal unsmoothing

*Author: [Artur Sepp](https://github.com/ArturSepp)*

The universe container and its unsmoothing transform are implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

`UniverseData` holds one investment universe: a price panel, one metadata row per asset, up to
two levels of group loadings, and the labels that identify cash, equity, bond and private-equity
holdings. It is a frozen dataclass that checks on construction that these parts describe the same
assets. Its parts, not the container, are then passed to the covariance estimators, to group
constraints and to group risk budgets, as the worked example shows.

Private-equity and real-estate valuations are appraisals, which move slowly and carry over part
of the previous valuation. Their reported returns are smoothed: positively autocorrelated, with a
volatility well below that of the asset itself. `copy_universe_data_with_unsmoothed_prices`
returns a copy of the universe in which the flagged price columns are unsmoothed by
[qis](https://github.com/ArturSepp/QuantInvestStrats) and everything else is unchanged.
Estimation then samples returns from the copy with `compute_returns_from_prices`. The
[ROSAA case study](app_rosaa_multi_asset_allocation.md#study-design-and-data) also unsmooths its
private-asset returns before estimating the covariance.

This page derives what smoothing does to volatility and autocorrelation, states what the
package's copy changes, and shows on a simulated four-asset universe that unsmoothing lowers the
autocorrelation of the private-equity series, restores its volatility and changes a
risk-budgeting allocation.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Log returns: the unsmoothing regression (`is_log_returns=True`), `compute_returns_from_prices` (default `is_log_returns=True`) and every statistic on the page; the container stores prices |
| Estimation grid | Quarter ends: the unsmoothing default `freq='QE'`, with `unsmooth_span=40` quarters and `warmup_period=8`; the example also samples `compute_returns_from_prices`, whose default is `returns_freq='ME'`, at `'QE'` |
| Rebalancing grid | None for the container; the example solves once, at the last quarter end of the sample |
| Covariance units | Annual: `estimate_current_ewma_covar` multiplies the quarterly estimate by 4, and volatilities on the page are quarterly standard deviations times 2 |
| Expected returns | None |
| Weight state | None for the container; the example's risk-budgeting weights are target weights |
| Solver | None for the container and the unsmoothing, which are closed-form rolling regressions in qis; the example's risk budgeting uses the package's in-house solver |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $x_t$ | Reported quarterly log return of an appraisal-valued asset |
| $r_t$ | Its true quarterly log return, unobserved in practice and simulated in the example |
| $\phi$ | Smoothing weight of the previous reported return, $0 \leq \phi \lt 1$ |
| $\rho_1$ | Lag-1 autocorrelation |
| $L^{(1)}$, $L^{(2)}$ | Group loadings of levels 1 and 2: one row per asset, one column per group |
| $G$, $n_g$ | Number of groups; number of assets in group $g$ |
| $\ell_g$, $u_g$ | Lower and upper limits on the allocation to group $g$ |
| $y_t$, $m_t$ | Any sampled log return and its EWMA mean, with the decay $\lambda$ of a span $s$ |

Prices are positive levels on a `DatetimeIndex`, one column per asset. The smoothing results
assume that the true returns $r_t$ are independent over time, with mean $\mu$ and variance
$\sigma^2$. The example's panel is synthetic and drawn from a fixed seed; it is a teaching sample,
not an estimate for any private-equity index.

## Methodology

### The container and what construction checks

A `UniverseData` has ten fields:

| Field | Default | Content |
|---|---|---|
| `prices` | required | Price panel: a `DatetimeIndex` by one column per asset |
| `metadata` | required | One row per asset, indexed by the asset names |
| `metadata_fields` | `MetadataField` | Enum whose values name the metadata columns that must be present and complete |
| `group_loadings_level1` | `None` | Loadings $L^{(1)}$: one row per asset, one column per group |
| `group_loadings_level2` | `None` | Loadings $L^{(2)}$ of a second grouping |
| `liquidity_ac_id` | `'Liquidity'` | Value of the asset-class column that marks cash-like assets |
| `equity_ac_id` | `'Equities'` | Value of the asset-class column that marks equities |
| `bond_ac_id` | `'Bonds'` | Value of the asset-class column that marks bonds |
| `pe_asset_id` | `None` | Name of the appraisal-valued private-equity asset |
| `validate_on_init` | `True` | Whether construction calls `validate()` |

With `validate_on_init=True`, construction runs `validate()`, which raises `ValueError` at the
first failed check, in this order:

1. **Alignment.** The set of price columns equals the set of metadata rows; the message starts
   `Asset mismatch` and names the assets found on one side only.
2. **Required fields.** Every value of `metadata_fields` is a metadata column.
3. **Duplicates.** No asset name appears twice among the price columns or the metadata rows.
4. **Nulls.** No required metadata column holds a missing value.
5. **Group loadings.** When a level is given, the set of its row labels equals the set of price
   columns.

With `validate_on_init=False` the instance is built unchecked, and `validate()` can be called
later. The checks compare labels, not values or order: the metadata rows may come in another order
than the price columns, and missing or negative prices, fractional loadings, and identifiers that
name no asset or class all pass.

### Metadata fields

`MetadataField` is a string enum with the members `NAME = 'name'`,
`ASSET_CLASS = 'asset_class'` and `CURRENCY = 'currency'`; a member compares equal to its value.
With the default `metadata_fields=MetadataField`, the metadata needs the three columns `name`,
`asset_class` and `currency`, without missing values; further columns are allowed and unchecked.
Another enum sets another schema: its values become the required columns. The properties `name`,
`asset_class` and `currency` of the container read the default column names whatever the enum.

### Group loadings, constraints and budgets

A loading matrix $L$ has one row per asset and one column per group, and $L^{\top} w$ is the
group exposure of the weights $w$. With one-hot loadings, which `qis.set_group_loadings` builds
from a label column, each asset belongs to one group and a group's exposure is the sum of its
members' weights. The container holds two such matrices. Which grouping is level 1 and which is
level 2 is the caller's convention; the example uses asset classes for level 1 and liquidity for
level 2. The container checks only their row labels.

The loadings reach the optimisers through the caller:

- **Group constraints.** `GroupLowerUpperConstraints` takes a loading matrix as
  `group_loadings`, with `group_min_allocation` and `group_max_allocation`, and imposes
  $\ell_g \leq \sum_i L_{ig} w_i \leq u_g$ on every group with a limit.
  `GroupTrackingErrorConstraint` and `GroupTurnoverConstraint` take the same `group_loadings`.
  The [constraints page](constraints.md) describes each.
- **Group risk budgets.** `compute_group_risk_budgets` takes group labels, which one-hot loadings
  give back as `idxmax(axis=1)`. With its default `group_size_exponent=0`, every nonempty group
  receives the same share of risk and splits it equally among its members, as
  [HRP and cluster risk budgets](hierarchical_risk_parity_and_cluster_budgets.md#group-risk-budgets)
  derives:

$$
b_i = \frac{1}{G n_g} \quad \text{for an asset } i \text{ in group } g .
$$

### Asset-class identifiers

`liquidity_ac_id`, `equity_ac_id` and `bond_ac_id` are values of the `asset_class` column, with
the defaults `'Liquidity'`, `'Equities'` and `'Bonds'`. They let code select the cash, equity or
bond assets of a universe by class, as `get_hedge_ratio(hedged_acs)` does for any list of
classes. `pe_asset_id` is different: it names one asset, not a class, and marks the series whose
prices are appraisals. The container stores the four labels and checks none of them.

### What appraisal smoothing does

In the first-order smoothing model behind the reverse filter of Geltner (1993), each reported
return carries a fixed share $\phi$ of the previous reported return:

$$
x_t = \phi x_{t-1} + (1 - \phi) r_t .
$$

**Proposition 1 (smoothing).** If the true returns are independent with mean $\mu$ and variance
$\sigma^2$, and $0 \leq \phi \lt 1$, the stationary reported returns have

$$
\mathrm{E}[x_t] = \mu, \qquad \mathrm{Var}(x_t) = \frac{1 - \phi}{1 + \phi} \sigma^2, \qquad \rho_1(x) = \phi .
$$

**Proof.** Unrolling the recursion gives $x_t = (1 - \phi) \sum_{k \geq 0} \phi^k r_{t-k}$. The
weights $(1 - \phi) \phi^k$ sum to one, so the mean is $\mu$. The variance is
$(1 - \phi)^2 \sigma^2 \sum_k \phi^{2k} = (1 - \phi)^2 \sigma^2 / (1 - \phi^2)$, which simplifies
as stated. Since $r_t$ is independent of $x_{t-1}$, the recursion gives
$\mathrm{Cov}(x_t, x_{t-1}) = \phi \mathrm{Var}(x_{t-1})$, and the two variances are equal in the
stationary state. $\square$

**Proposition 2 (reverse filter).** With $\phi$ known, the true return is recovered exactly from
two consecutive reported returns:

$$
r_t = \frac{x_t - \phi x_{t-1}}{1 - \phi} .
$$

**Proof.** Solve the recursion for $r_t$. $\square$

In practice $\phi$ is unknown and can change over time. qis estimates the smoothing from the
reported series itself, with rolling EWMA regressions of each return on its previous returns,
which also allows smoothing over several lags as in Getmansky, Lo and Makarov (2004), and inverts
it in the manner of Proposition 2. The estimator, its bounds and its warmup are derived
on the qis page
[Private-asset unsmoothing and de-levering](https://quantinveststrats.readthedocs.io/en/latest/private_asset_unsmoothing.html);
the package does not change them.

### What the unsmoothing copy changes

`copy_universe_data_with_unsmoothed_prices(universe_data, assets_for_unsmoothing, ...)` takes a
boolean Series `assets_for_unsmoothing` whose index equals `universe_data.metadata.index`, in the
same order, and applies `qis.compute_ar_unsmoothed_prices` to the columns flagged `True`. It
passes six of its arguments to qis:

| Argument | Default | Passed to qis as |
|---|---|---|
| `freq` | `'QE'` | `freq`: one frequency string, or a Series with one per asset |
| `unsmooth_span` | `40` | `span`, in periods of `freq` |
| `mean_adj_type` | `qis.MeanAdjType.EWMA` | `mean_adj_type` |
| `warmup_period` | `8` | `warmup_period` |
| `max_value_for_beta` | `0.75` | `max_value_for_beta`, the upper bound of the coefficient sum |
| `is_log_returns` | `True` | `is_log_returns` |

Every other qis argument keeps its qis default, among them `ar_order=2` and
`min_value_for_beta=-0.25`.

- **Flagged columns.** Each is replaced by the NAV that qis returns: missing until the first
  identified coefficient, 1.0 on the last date before the first unsmoothed return, then compounded
  from the unsmoothed returns, and forward-filled onto the universe's dates. The original price
  level is not kept.
- **Everything else.** The other price columns are unchanged bit for bit, and the copy is built
  with the same `metadata`, `metadata_fields`, both group loadings, `equity_ac_id`,
  `bond_ac_id`, `pe_asset_id` and `validate_on_init`. `liquidity_ac_id` is not passed and returns
  to its default.
- **Nothing flagged.** The input universe itself is returned.
- **Errors.** A `ValueError` is raised when the flag's index, or the index of a `freq` Series,
  differs from the metadata index.

The function does not read `pe_asset_id`; the caller builds the flag, typically from it.

### Returns for estimation

`compute_returns_from_prices` takes `prices` and has the defaults `returns_freq='ME'`,
`demean=True`, `drop_first=True`, `is_first_zero=False`, `is_log_returns=True` and `span=52`. It
samples log returns at `returns_freq` through `qis.to_returns`, without the first, empty row. With
`demean=True` it subtracts the EWMA mean of span `span`, which includes the current return.
`estimate_current_ewma_covar` and `EwmaCovarEstimator` build their returns with it, so this is
where an unsmoothed copy enters estimation.

**Proposition 3 (demeaned returns).** With $m_1 = y_1$ and
$m_t = \lambda m_{t-1} + (1 - \lambda) y_t$, the demeaned return is

$$
y_t - m_t = \lambda (y_t - m_{t-1}), \qquad t \geq 2,
$$

and zero at $t = 1$, which is why, with `drop_first=True`, the function drops that row as well.

**Proof.** Substituting $m_t$ gives $y_t - \lambda m_{t-1} - (1 - \lambda) y_t$, which is
$\lambda (y_t - m_{t-1})$. $\square$

Each demeaned return is therefore the surprise against the previous mean, scaled down by
$\lambda$: by $39/41 \approx 0.951$ at a span of 40 quarters.

## Worked example

The canonical script of this page,
[`examples/docs/universe_data_and_unsmoothing.py`](../examples/docs/universe_data_and_unsmoothing.py),
runs offline and asserts every number quoted here, including the three propositions:

```console
python -m examples.docs.universe_data_and_unsmoothing
```

The universe has four assets with quarterly prices from 31 December 1989 to 31 December 2024:
government bonds, credit, listed equity and private equity. The true log returns are drawn from a
fixed seed, with annual volatilities of 5%, 8%, 16% and 20%, and the private-equity price is
built from returns smoothed with $\phi = 0.6$. The level-1 loadings group the assets by asset
class, the level-2 loadings by liquidity:

```python
prices, true_returns = simulated_panel(seed=SEED)
metadata = pd.DataFrame({'name': NAMES, 'asset_class': ASSET_CLASSES, 'currency': 'USD'},
                        index=TICKERS)
level1 = qis.set_group_loadings(group_data=metadata['asset_class'])
level2 = qis.set_group_loadings(group_data=pd.Series(LIQUIDITY, index=TICKERS))
universe = op.UniverseData(prices=prices, metadata=metadata, group_loadings_level1=level1,
                           group_loadings_level2=level2, pe_asset_id='PE')
```

Each of the five inputs below fails one check on construction:

```python
invalid = {
    'metadata row missing': dict(prices=prices, metadata=metadata.drop(index='PE')),
    'required column missing': dict(prices=prices, metadata=metadata.drop(columns='name')),
    'duplicate price column': dict(prices=prices[TICKERS + ['PE']], metadata=metadata),
    'null in required column': dict(
        prices=prices, metadata=metadata.assign(currency=['USD', 'USD', 'USD', None])),
    'loadings on other assets': dict(prices=prices, metadata=metadata,
                                     group_loadings_level2=level2.drop(index='PE')),
}
errors = {case: construction_error(**arguments) for case, arguments in invalid.items()}
```

| Input | The `ValueError` message starts |
|---|---|
| Metadata without the `PE` row | `Asset mismatch` |
| Metadata without the `name` column | `Metadata missing required columns` |
| `PE` twice among the price columns | `Duplicate asset names in prices` |
| A missing currency | `Null values in required metadata columns` |
| Level-2 loadings without the `PE` row | `group_loadings_level2 index doesn't match price columns` |

The flag for unsmoothing marks the asset that `pe_asset_id` names:

```python
flag = pd.Series(universe.metadata.index == universe.pe_asset_id,
                 index=universe.metadata.index)
unsmoothed = op.copy_universe_data_with_unsmoothed_prices(universe_data=universe,
                                                          assets_for_unsmoothing=flag)
```

The government-bond, credit and equity prices of the copy equal the originals bit for bit, and
the copy holds the same metadata and loading objects. The private-equity column is missing for the
first 17 quarters, equals 1.0 on 31 March 1994 and compounds the unsmoothed returns from
30 June 1994. Over the sample, the coefficient sum that qis estimates stays between 0.45 and
0.67, inside its bounds.

Over the 123 quarters from June 1994, the reported series has a lag-1 autocorrelation of 0.61
and a volatility of 10.2%. The unsmoothed series has an autocorrelation of −0.04 and a volatility
of 20.5%, close to the −0.02 and 20.4% of the simulated true returns:

```python
before = op.compute_returns_from_prices(universe.prices, returns_freq='QE', demean=False)
after = op.compute_returns_from_prices(unsmoothed.prices, returns_freq='QE', demean=False)
window = after['PE'].dropna().index
stats = pd.DataFrame({
    'lag-1 autocorrelation': [before.loc[window, 'PE'].autocorr(lag=1),
                              after.loc[window, 'PE'].autocorr(lag=1)],
    'annualised volatility': [before.loc[window, 'PE'].std() * 2.0,
                              after.loc[window, 'PE'].std() * 2.0]},
    index=['smoothed', 'unsmoothed'])
```

![Left: the cumulative log returns of private equity from March 1994; the reported path is visibly
smoother than the unsmoothed path, which tracks the simulated true path closely. Middle: the lag-1
autocorrelation falls from 0.61 reported to -0.04 unsmoothed, against -0.02 for the true returns.
Right: the annualised volatility rises from 10.2% to 20.5%, against 20.4% for the true
returns.](images/unsmoothing_autocorrelation.png)

*Figure: the reported, unsmoothed and simulated true returns of the private-equity column of the
example, as cumulative paths and as lag-1 autocorrelation and annualised volatility. Drawn by the
`exhibit` function of the canonical script; the [analytics gallery](analytics_gallery.md) lists
its provenance.*

> **Insight.** With $\phi = 0.6$, the volatility factor $\sqrt{(1 - \phi)/(1 + \phi)}$ of
> Proposition 1 is exactly one half, and the sample agrees: 10.2% reported against 20.4% true. A
> covariance estimated on the reported returns understates the variance of the private asset about
> fourfold, although smoothing leaves its expected return unchanged.

The loadings then feed an allocation. The level-1 loadings give two asset classes of two assets
each, so equal group budgets give every asset a quarter of the risk. The level-2 loadings cap the
illiquid group at 15% of capital. The same risk-budgeting problem is solved on the EWMA
covariance of each universe, estimated over the common sample from March 1994:

```python
budgets = op.compute_group_risk_budgets(groups=universe.group_loadings_level1.idxmax(axis=1))
illiquid_cap = op.GroupLowerUpperConstraints(
    group_loadings=universe.group_loadings_level2, group_min_allocation=None,
    group_max_allocation=pd.Series({'Liquid': 1.0, 'Illiquid': ILLIQUID_CAP}))
constraints = op.Constraints(is_long_only=True, group_lower_upper_constraints=illiquid_cap)
start = unsmoothed.prices['PE'].first_valid_index()
covars, weights = {}, {}
for label, data in {'smoothed': universe, 'unsmoothed': unsmoothed}.items():
    covars[label] = op.estimate_current_ewma_covar(prices=data.prices.loc[start:],
                                                   returns_freq='QE', span=SPAN)
    weights[label] = op.wrapper_risk_budgeting(pd_covar=covars[label],
                                               constraints=constraints, risk_budget=budgets)
```

On the reported prices, the estimated volatility of private equity is less than half its
unsmoothed value. Without the cap, risk budgeting would put 22% of capital in it for its quarter
of the risk; with the cap, it holds exactly 15%. On the unsmoothed prices, private equity takes
10% of capital, the cap is slack, and every asset carries a quarter of the risk.

## Implementation in optimalportfolios

The four objects are exported at the package root:

- `UniverseData`, in
  [`universe/universe_data.py`](../src/optimalportfolios/universe/universe_data.py): the frozen
  container and `validate()`. `from_selection(prices, metadata, assets, ...)` subsets the prices,
  metadata and loadings to the listed assets, in that order. `save(file_name, local_path)` writes
  the parts to CSV files through qis, and `load(file_name, local_path, ...)` reads them back;
  without `metadata_fields`, `load` makes every metadata column it reads a required field.
  `rename_index()` relabels the assets by their `name` column. The properties `name`,
  `asset_class`, `currency`, `assets`, `n_assets` and `date_range` read the parts, and
  `get_hedge_ratio(hedged_acs)` returns 1.0 for the assets of the listed classes and 0.0 for the
  others. `get_asset_returns_dict` is described with [mixed-frequency data](mixed_frequency_data.md).
- `MetadataField`, in the same module: the default schema of the metadata.
- `copy_universe_data_with_unsmoothed_prices`, in
  [`universe/universe_transforms.py`](../src/optimalportfolios/universe/universe_transforms.py):
  the unsmoothing copy described above.
- `compute_returns_from_prices`, in
  [`covar_estimation/utils.py`](../src/optimalportfolios/covar_estimation/utils.py): the return
  construction of the EWMA covariance estimators.

> **Pitfall.** The docstring of `copy_universe_data_with_unsmoothed_prices` promises AR(1)
> unsmoothing, but the function does not pass `ar_order`, and `qis.compute_ar_unsmoothed_prices`
> defaults to `ar_order=2`: the copy is unsmoothed with a rolling AR(2) filter. The example checks
> that its column equals the qis AR(2) result and differs from the AR(1) result. For AR(1), call
> `qis.compute_ar_unsmoothed_prices` with `ar_order=1` and build the universe from its NAVs.

## Interpretation and limitations

- **What Geltner's model does not bring.** The package takes from Geltner (1993) the first-order
  smoothing model that the example simulates and the reverse filter of Proposition 2. It does not
  inherit the paper's derivation of the smoothing from appraisal behaviour, which avoids assuming
  that market returns are uncorrelated over time, nor its corrections for temporal aggregation and
  seasonal reappraisal. qis estimates the smoothing from the autocorrelation of the reported
  series, so all serial correlation is treated as smoothing, including any in the true returns.
- **Estimation noise.** The coefficients come from a rolling EWMA regression with a span of 40
  quarters. The example's unsmoothed volatility is close to the true one, but other samples
  differ: with seed 3 instead of 19, the unsmoothed volatility is 30.7% against a true 19.5%.
- **Shorter history.** The unsmoothed column is missing for the first 17 quarters of the example,
  and the example's covariance starts after them. Its level starts at 1.0, so only its returns are
  comparable with the original.
- **Finer grids.** On a monthly panel with the default `freq='QE'`, the unsmoothed column changes
  only at quarter ends. A monthly estimator then sees two zero returns in every quarter; keep such
  an asset in a quarterly bucket, as described with [mixed-frequency data](mixed_frequency_data.md).
- **Labels, not values.** Construction accepts missing or negative prices, fractional loadings and
  identifiers that name nothing; [incomplete histories](incomplete_histories.md) describes how the
  solvers treat missing data.
- **Identifiers lost in transforms.** `from_selection` and `rename_index` return a universe with
  the default identifiers and `pe_asset_id=None`, and the unsmoothing copy resets
  `liquidity_ac_id`. Pass the identifiers again where they are needed.

## See also

- [Mixed-frequency data](mixed_frequency_data.md)
- [Incomplete histories and frozen positions](incomplete_histories.md)
- [Covariance estimators](covariance_estimators.md)
- [Risk budgeting](risk_budgeting.md)
- [Portfolio constraints](constraints.md)
- [Strategic and tactical allocation with HCGL covariance (ROSAA)](app_rosaa_multi_asset_allocation.md)
- [Conventions, notation and glossary](conventions.md)
- [qis: private-asset unsmoothing and de-levering](https://quantinveststrats.readthedocs.io/en/latest/private_asset_unsmoothing.html)

## References

- Geltner, D. (1993). *Estimating Market Values from Appraised Values without Assuming an
  Efficient Market*. Journal of Real Estate Research, 8(3), 325–345.
  [DOI 10.1080/10835547.1993.12090713](https://doi.org/10.1080/10835547.1993.12090713).
- Getmansky, M., Lo, A. W. and Makarov, I. (2004). *An Econometric Model of Serial Correlation
  and Illiquidity in Hedge Fund Returns*. Journal of Financial Economics, 74(3), 529–609.
  [DOI 10.1016/j.jfineco.2004.04.001](https://doi.org/10.1016/j.jfineco.2004.04.001). Smoothing
  over several lags, the model that qis cites for its filters.
- Sepp, A., Ossa, I. and Kastenholz, M. (2026). *Robust Optimization of Strategic and Tactical
  Asset Allocation for Multi-Asset Portfolios*. The Journal of Portfolio Management, 52(4),
  86–120. [DOI 10.3905/jpm.2025.1.806](https://doi.org/10.3905/jpm.2025.1.806). Unsmooths its
  private-asset returns before estimation.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
