---
myst:
  html_meta:
    description: >-
      Implied risk budgets from target weights in Python with optimalportfolios: the closed-form
      budgets of one covariance, why they reproduce the weights only when every held asset adds
      risk, the hold rule for hedging assets, fitting one budget vector to a rolling path with
      simple or EWMA averaging, and a verified offline example.
---

# Implied risk budgets from target weights

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implied risk budgets are implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

[Risk budgeting](risk_budgeting.md) turns risk budgets into weights. This page runs the map the
other way: given target weights, such as a strategic 40/45/15 mix of equity, bonds and gold,
which risk budgets produce them? The answer lets an existing allocation be restated as risk
budgets and then run forward as a risk-budgeting strategy.

On one covariance matrix the answer is closed form: the budgets are the target's own risk shares.
This page proves that they reproduce the weights exactly when every held asset has a positive
marginal risk, and that no non-negative budget reproduces a target that holds a hedge. With a
covariance that changes from date to date, `solve_for_risk_budgets_from_given_weights` fits one
budget vector whose risk-budgeting weights match the target on average over the rebalance dates,
with the average taken by `average_rolling_weights`. It holds hedging assets, and any asset named
in `fixed_weight_assets`, at their target weights with a zero budget. The worked example checks
each statement against an independent computation.

The risk contributions come from [qis](https://github.com/ArturSepp/QuantInvestStrats); the
fitting and the forward solves are OptimalPortfolios code.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | None; no returns are sampled. `prices` fixes the asset columns, and `rolling_risk_budgeting` drifts its previous weights with simple price ratios, which matters only for its fallback. The examples use constant prices |
| Estimation grid | None; the covariance matrices are supplied in `covar_dict` and estimated elsewhere |
| Rebalancing grid | The keys of `covar_dict`, in chronological order; the fit averages over these dates. Twelve quarter ends from 2023-03-31 in the rolling example |
| Covariance units | Caller units: implied budgets do not depend on the scale, and each forward solve raises a positive variance below `0.001**2` to that floor. Annual, in fractional return squared, in the examples |
| Expected returns | None |
| Weight state | Target weights: `given_weights`, the forward weight path and its average are target weights on rebalance dates, not drifted holdings |
| Solver | Closed form on one date. The fit: a bounded multiplicative fixed point around the CCD/ADMM risk-budgeting solve, then SciPy SLSQP on the mean absolute weight gap, then boundary probing |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $\pi_i(w)$ | Risk share of asset $i$ in the weights $w$, the risk contribution $\mathrm{RC}_i$ divided by $\sigma(w)$ |
| $b(w)$ | Implied budgets of the weights $w$: the vector of their risk shares |
| $S$ | Support of the target: the assets with $w_i \gt 0$ |
| $H$, $F$ | Held assets, kept at their target weights with zero budget, and the free assets of $S$ |
| $x_t(b)$ | Forward risk-budgeting weights at rebalance date $t$ for the budgets $b$ |
| $\bar x(b)$ | Average of the forward weights over the rebalance dates |
| $T$, $m_t$ | Number of rebalance dates; state of the EWMA recursion |

The target is long-only and fully invested: `given_weights` is aligned to the columns of `prices`
and must be finite, non-negative and sum to one. The covariance is positive definite on the
support of the target. The forward solves are long-only and fully invested, with no constraint
other than the equal bounds of the held assets.

## Methodology

### The inverse problem

Risk budgeting takes non-negative budgets $b$ that sum to one and returns long-only, fully
invested weights whose risk shares equal the budgets when no bound binds. The risk share of asset
$i$ is

$$
\pi_i(w) = \frac{w_i (\Sigma w)_i}{w^{\top} \Sigma w}, \qquad \sum_i \pi_i(w) = 1 ,
$$

where the sum is Euler's decomposition of the variance, $\sum_i w_i (\Sigma w)_i = w^{\top} \Sigma w$.
The inverse problem asks for budgets whose risk-budgeting portfolio is a given target $w$.

**Definition.** The implied budgets of the target $w$ are its risk shares, $b(w) = \pi(w)$. They
sum to one and do not change when $\Sigma$ is multiplied by a positive number.

**Proposition 1 (reproduction).** Let $\Sigma$ be positive definite and $w$ long-only and fully
invested with support $S$. Some budget vector that is positive on $S$ and zero elsewhere has $w$
as its risk-budgeting portfolio if and only if $(\Sigma w)_i \gt 0$ for every asset $i$ in $S$.
That budget vector is then unique and equal to $b(w)$.

**Proof.** Assets outside $S$ have zero budget and zero weight, so restrict $\Sigma$ to $S$ and
write $\Sigma_S$. For budgets $b$ that are positive on $S$, the function

$$
f(y) = \frac{1}{2} y^{\top} \Sigma_S y - \sum_{i \in S} b_i \ln y_i , \qquad y \gt 0 ,
$$

is strictly convex and grows without bound at the boundary of the positive orthant and at
infinity, so it has exactly one minimiser $y^{\star}$, the solution of $y_i (\Sigma_S y)_i = b_i$
for all $i$ in $S$. For positive, fully invested weights $x$ on $S$, the vector
$y = x / \sigma(x)$ has
$y_i (\Sigma_S y)_i = \pi_i(x)$, so $\pi(x) = b$ exactly when $y$ solves those equations. The
risk-budgeting portfolio of $b$ is therefore unique, $x = y^{\star} / \sum_i y^{\star}_i$.

If $w$ is that portfolio, then $w_i (\Sigma w)_i = b_i \sigma(w)^2 \gt 0$ with $w_i \gt 0$, so
$(\Sigma w)_i \gt 0$. Conversely, if $(\Sigma w)_i \gt 0$ on $S$, then $b(w)$ is positive on $S$,
sums to one and satisfies $\pi(w) = b(w)$, so $w$ is the unique portfolio of $b(w)$; any budget
vector that reproduces $w$ equals $\pi(w)$. $\square$

The logarithmic form of the risk-budgeting problem is the characterisation of Maillard, Roncalli
and Teïletche (2010) for equal budgets, extended to general budgets by Roncalli (2013); the
[risk budgeting](risk_budgeting.md#constrained-allocation-and-normalization) page states the
normalised form that the package's solver uses.

**A hedge cannot be budgeted.** For a held asset with $(\Sigma w)_k \lt 0$, a small increase in
its weight reduces the variance of the target: the asset hedges the portfolio. Its implied budget
is negative, and by
Proposition 1 no non-negative budget reproduces the target; every risk-budgeting portfolio with
positive budgets gives each asset a positive marginal risk. With $(\Sigma w)_k = 0$ the implied
budget is zero, and the wrapper excludes the asset.

### Holding an asset at its target weight

The package keeps such an asset in the portfolio by holding it: in every forward solve its lower
and upper bounds equal its target weight, its budget is zero, and it stays in the full-covariance
solve. With the held assets $H$ and the free assets $F$ of $S$, the forward problem is the
risk-budgeting objective with the budgets on $F$ only, under full investment and the equal bounds
on $H$.

**Proposition 2 (budgets around held assets).** Write $\pi_H$ and $w_H$ for the total risk share
and the total weight of the held assets. If the forward problem returns the target $w$, with every
free weight strictly inside its bounds, then for every free asset $i$

$$
b_i = \pi_i(w) + \pi_H(w) \frac{w_i}{1 - w_H} .
$$

**Proof.** The objective is $\ln \sigma(x) - \sum_{i \in F} b_i \ln x_i$. Its stationarity in a
free weight, with the multiplier $\nu$ of full investment, is
$(\Sigma x)_i / \sigma(x)^2 - b_i / x_i = \nu$. At $x = w$, multiplying by $w_i$ gives
$\pi_i(w) = b_i + \nu w_i$. Summing over the free assets, whose budgets sum to one, whose risk
shares sum to $1 - \pi_H$ and whose weights sum to $1 - w_H$, gives $\nu = -\pi_H / (1 - w_H)$.
$\square$

A held hedge has $\pi_H \lt 0$, so each free budget is its risk share less a slice of the hedge's
negative share, in proportion to its capital weight.

The package holds an asset in two cases, exactly as follows.

- **The hold rule.** Before fitting, it computes the risk shares of the target at every date of
  `covar_dict`, from `qis.compute_portfolio_risk_contributions`, and averages them over the dates
  with `average_rolling_weights` and the caller's `ewma_span`. Every asset with a positive target
  weight and an averaged share that is zero or negative is held, and a `UserWarning` names each one
  with its weight and averaged share. The target weights are not changed and nothing is
  redistributed. A non-finite averaged share, such as one from a missing variance, raises
  `ValueError`.
- **`fixed_weight_assets`.** Each named asset is held in the same way, even when its averaged share
  is positive. The labels must be a sequence of distinct columns of `prices` with positive target
  weights: a single string raises `TypeError`, anything else `ValueError`.

A held asset reports a zero budget and has its budget bounds set to zero. When one free asset
remains, it receives the whole budget without a search; when none remains, the call raises
`ValueError`. A free asset whose share is negative on at least half of the dates is only warned
about, because the averaged path may still reach its target.

### Fitting one budget vector to a rolling path

With covariances $\Sigma_1, \dots, \Sigma_T$ on the rebalance dates, the implied budgets change
from date to date, and in general no single budget vector reproduces the target on every date.
`solve_for_risk_budgets_from_given_weights` fits one vector $b$ whose forward path matches the
target on average:

$$
\bar x(b) \approx w, \qquad \frac{1}{N} \sum_i \lvert \bar x_i(b) - w_i \rvert \leq 10^{-4}, \qquad \max_i \lvert \bar x_i(b) - w_i \rvert \leq 10^{-3} .
$$

The forward path is `rolling_risk_budgeting` over `covar_dict` with
`Constraints(is_long_only=True)` and the equal bounds of the held assets.

**Averaging.** `average_rolling_weights(weights, ewma_span)` averages a path in the order of its
index, so the path must be sorted by date. With `ewma_span=None`, the default of the fit, it is the
simple mean over the dates. With a span $s$ and the decay $\lambda = 1 - 2/(s + 1)$ it is the last
state of the recursion seeded at the first date,

$$
m_1 = x_1, \qquad m_t = \lambda m_{t-1} + (1 - \lambda) x_t, \qquad \bar x = m_T ,
$$

so date $t \geq 2$ carries the weight $(1 - \lambda) \lambda^{T - t}$ and the first date the
remainder $\lambda^{T-1}$. The implementation is `qis.compute_ewm` with `qis.InitType.X0`. A
missing row holds the state, as if the row were absent; a path with no value returns missing
values; a span that is not a finite positive number raises `ValueError`.

`INVERSE_EWMA_SPAN = 12` is a constant of the module
`optimalportfolios.optimization.risk_allocation.risk_budgeting`, not exported at the package root.
It was the default span of the fit in 7.7.0. Since 7.8.0 the default is the simple mean, and the
constant remains for callers that choose the former span explicitly.

**The fit.** The search runs on the box simplex: free budgets lie between `min_risk_budget` and
`max_risk_budget` (defaults $10^{-4}$ and 0.99), held assets and assets with zero target weight
have zero budget, and the budgets sum to one.

1. The seed is the averaged implied budgets of the hold rule, mapped onto the box simplex by $P$,
   which multiplies the budgets by one common factor and clips them at their bounds.
2. A fixed point runs for at most 50 iterations. Each evaluates $\bar x(b)$ and keeps the budgets
   with the smallest mean gap. It stops when both tolerances hold or the average is not finite,
   and otherwise updates $b \leftarrow P(b/2 + P(b \circ q^2)/2)$, where $q_i = w_i / \bar x_i(b)$
   is clipped to $[10^{-3}, 10^{3}]$, and is one for an asset with zero target weight.
3. If the fixed point misses a tolerance, SciPy SLSQP minimises the mean absolute gap on the box
   simplex, from the best fixed point, with tolerance $10^{-8}$ and at most 100 iterations. Its
   result is returned if SLSQP succeeds and the result meets both tolerances.
4. Otherwise, since 7.8.0, the fit probes the free assets whose average weight exceeds the target
   by more than $10^{-3}$: first those whose budget sits at its floor, then the others, each group
   by its gap. An asset not already at its floor is probed with its budget forced to the floor and
   the other budgets rescaled, and is skipped if the other caps cannot then sum to one. If its
   average weight still exceeds the target by more than $10^{-3}$, the whole fit is rerun with the
   asset added to `fixed_weight_assets`, and the result is accepted, with a `UserWarning`, only if
   its forward path meets both tolerances.
5. If no probe succeeds, the call raises `RuntimeError` with the errors of both searches, the
   largest weight gaps, the rejected probes and the assets with a negative share on some dates. It
   never returns a zero-budget fallback.

For a diagonal covariance the forward weights are proportional to $\sqrt{b_i} / \sigma_i$, so the
undamped update $b_i q_i^2$ reaches the target in one step; the package averages it with the
current budgets.

Two cases skip the search. A single column in `prices` receives the budget one, and naming that
column in `fixed_weight_assets` raises `ValueError`. A target with one positive weight gives that
asset the budget one, unless some asset is named in `fixed_weight_assets`.

## Worked example

The canonical script of this page,
[`examples/docs/implied_risk_budgets.py`](../examples/docs/implied_risk_budgets.py),
runs offline and asserts every number and property stated here:

```console
python -m examples.docs.implied_risk_budgets
```

The universe has Equity, Bonds and Gold with annual volatilities of 16%, 7% and 15%, and the target
weights are 40%, 45% and 15%. The script's `covariance` builds the covariance for a given
stock-bond correlation, and `implied_budgets` computes $b(w)$ with NumPy. At a stock-bond
correlation of 0.2, Equity carries 66.6% of the risk with 40% of the capital, Bonds 22.0% with 45%
and Gold 11.4% with 15%. Passed to `wrapper_risk_budgeting`, these budgets return the target within
$10^{-8}$:

```python
covar = covariance(STOCK_BOND)
target = pd.Series(TARGET_WEIGHTS, index=ASSETS)
budgets = implied_budgets(target, covar)
weights = op.wrapper_risk_budgeting(pd_covar=covar,
                                    constraints=op.Constraints(is_long_only=True),
                                    risk_budget=budgets)
print(pd.concat([target.rename('target weight'), budgets.rename('implied budget'),
                 weights.rename('forward weight')], axis=1).round(4))
```

The script also checks the budgets against the risk shares that qis computes, and checks that they
are the same for the covariance multiplied by 12. The round trip also runs the other way: the
weights of the budgets 50%, 30% and 20% have exactly these risk shares, within $10^{-8}$:

```python
start = pd.Series([0.50, 0.30, 0.20], index=ASSETS)
forward = op.wrapper_risk_budgeting(pd_covar=covar,
                                    constraints=op.Constraints(is_long_only=True),
                                    risk_budget=start)
print(implied_budgets(forward, covar).round(6).tolist())  # [0.5, 0.3, 0.2]
```

With the same covariance on each of twelve quarter ends, the package's fit returns the closed-form
budgets within $10^{-8}$. Its seed is the average of the implied budgets, and the first forward
solve already meets both tolerances:

```python
dates = rebalance_dates()
prices = pd.DataFrame(100.0, index=dates, columns=ASSETS)
fitted = op.solve_for_risk_budgets_from_given_weights(
    prices=prices, given_weights=target, covar_dict={date: covar for date in dates})
print(fitted.round(4).tolist())  # the implied budgets
```

At a stock-bond correlation of −0.7, Bonds hedge the target: the Bonds entry of $\Sigma w$, their
covariance with the portfolio, is −0.00062, and their implied budget is −8.3%. The hold rule
warns, keeps Bonds at 45% with a zero budget, and fits 79% to Equity and 21% to Gold:

```python
hedge = covariance(HEDGE_STOCK_BOND)
print((hedge @ target).round(5).tolist())  # the marginal risk of Bonds is negative
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    held = op.solve_for_risk_budgets_from_given_weights(
        prices=prices, given_weights=target, covar_dict={date: hedge for date in dates})
print(held.round(2).tolist())  # [0.79, 0.0, 0.21]
print(str(caught[0].message)[:88])
```

Proposition 2 predicts these budgets: $0.846 - 0.083 \times 0.40 / 0.55 = 0.786$ for Equity and
0.214 for Gold. The script checks the fit against them within $10^{-3}$, and checks that the
predicted budgets, with Bonds held, return the target through `wrapper_risk_budgeting` within
$10^{-8}$. The fitted budgets reproduce the target only with Bonds held at 45%. Without that bound
the wrapper excludes the zero-budget asset, and the portfolio holds no bonds:

```python
lower = pd.Series({'Equity': 0.0, 'Bonds': target['Bonds'], 'Gold': 0.0})
upper = pd.Series({'Equity': 1.0, 'Bonds': target['Bonds'], 'Gold': 1.0})
pin = op.Constraints(is_long_only=True, min_weights=lower, max_weights=upper)
reused = op.wrapper_risk_budgeting(pd_covar=hedge, constraints=pin, risk_budget=held)
dropped = op.wrapper_risk_budgeting(pd_covar=hedge,
                                    constraints=op.Constraints(is_long_only=True),
                                    risk_budget=held)
print(pd.concat([reused.rename('with the pin'), dropped.rename('without it')],
                axis=1).round(3))
```

With the bound the weights are the target within $10^{-3}$; without it they are 66% Equity, no
Bonds and 34% Gold. An asset that carries risk can be held too. Naming Gold in
`fixed_weight_assets`, at the stock-bond correlation of 0.2, gives Gold a zero budget and fits 72%
to Equity and 28% to Bonds, again as Proposition 2 predicts:

```python
gold_held = op.solve_for_risk_budgets_from_given_weights(
    prices=prices, given_weights=target, covar_dict={date: covar for date in dates},
    fixed_weight_assets=['Gold'])
print(gold_held.round(2).tolist())  # [0.72, 0.28, 0.0]
```

In the rolling example the stock-bond correlation drifts linearly from −0.7 to 0.4 over the twelve
quarter ends. The implied budget of Bonds is negative on the first two dates and positive after
them. On each of the other ten dates, that date's implied budgets return the target within
$10^{-8}$, one date at a time through `wrapper_risk_budgeting` and all at once as a date-by-asset
budget panel through `rolling_risk_budgeting`. On the two hedging dates the wrapper excludes Bonds.
No single budget vector reproduces the target on every date, so the fit finds one whose forward
weights match it on average:

```python
covar_dict = rolling_covariances(dates)
implied_path = pd.DataFrame({date: implied_budgets(target, c)
                             for date, c in covar_dict.items()}).T
fitted_path = op.solve_for_risk_budgets_from_given_weights(
    prices=prices, given_weights=target, covar_dict=covar_dict)
path = op.rolling_risk_budgeting(prices=prices,
                                 constraints=op.Constraints(is_long_only=True),
                                 risk_budget=fitted_path, covar_dict=covar_dict)
average = op.average_rolling_weights(path)
print(pd.concat([target.rename('target weight'), fitted_path.rename('fitted budget'),
                 average.rename('average weight')], axis=1).round(4))
```

The average weights match the target within $10^{-3}$, and the average is the simple mean of the
path. The forward weight of Bonds falls from 59% to 32% along the path. Bonds hedge on two of the
twelve dates, fewer than half, so the fit issues no warning; when they hedge on every other date,
it warns and proceeds. Every forward weight has a positive marginal risk on every date, including
the two hedging dates, as Proposition 1 requires of positive budgets.

![Left: the implied risk budgets of the 40/45/15 target at each quarter end. The Bonds budget rises
from −8% to 25% and is negative on the first two dates, shaded, while Equity falls from 85% to 65%
and Gold from 24% to 10%; dashed lines mark the fitted static budgets of about 70.6%, 14.5% and
14.9%.
Right: the forward weights under the fitted budgets move away from the target, Bonds from 59% to
32%, while their running averages end on the target weights of 40%, 45% and
15%.](images/implied_budgets_round_trip.png)

*Figure: implied budgets of the target and weights under the fitted static budgets through twelve
quarter ends with a drifting stock-bond correlation. Drawn by the `exhibit` function of the
canonical script; the [analytics gallery](analytics_gallery.md) lists its provenance.*

> **Insight.** The seed of the fit, the average of the implied budgets, is not the answer. Run
> forward, the average of its weights misses the target by up to two percentage points. The
> fitted Bonds budget is about 14.5%, two points above the 12.5% average of its implied budgets.

The EWMA average of the same path with `INVERSE_EWMA_SPAN` equals an explicit run of the recursion,
and also the closed-form date weights applied to the path. On twelve dates the first rebalance
carries 15.9% of the weight, more than the latest, which carries $2/13$ or 15.4%:

```python
from optimalportfolios.optimization.risk_allocation.risk_budgeting import INVERSE_EWMA_SPAN
recent = op.average_rolling_weights(path, ewma_span=INVERSE_EWMA_SPAN)
print(INVERSE_EWMA_SPAN, recent.round(4).tolist())
```

A fit with this span matches the EWMA average of its own forward path within $10^{-3}$, while the
simple mean of that path misses the target by more than $5 \times 10^{-3}$. Its budgets differ from
those of the default fit by more than one percentage point:

```python
recent_fit = op.solve_for_risk_budgets_from_given_weights(
    prices=prices, given_weights=target, covar_dict=covar_dict,
    ewma_span=INVERSE_EWMA_SPAN)
recent_path = op.rolling_risk_budgeting(prices=prices,
                                        constraints=op.Constraints(is_long_only=True),
                                        risk_budget=recent_fit, covar_dict=covar_dict)
print(op.average_rolling_weights(recent_path, ewma_span=INVERSE_EWMA_SPAN).round(4).tolist())
```

The last example is the four-sleeve case of the repository workflow
[`inverse_risk_budget_bonds.py`](../examples/solvers/inverse_risk_budget_bonds.py): Equity 65%, a
sleeve of other assets 19.9%, Bond A 12.5% and Bond B 2.6%, over three covariance dates. Bond A
hedges on average and the hold rule holds it. Bond B carries risk on average and hedges on one
date only, yet a budget of $10^{-4}$, the default floor, next to 91% for Equity and 9% for the
other sleeve and with Bond A held, still leaves its average weight more than ten percentage points
above its target.
The fixed point and SLSQP miss the tolerance, and the probe holds Bond B as well:

```python
boundary_prices, boundary_target, boundary_covars = boundary_inputs()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    probed = op.solve_for_risk_budgets_from_given_weights(
        prices=boundary_prices, given_weights=boundary_target, covar_dict=boundary_covars)
print(probed.round(4).tolist())
print([str(item.message)[:72] for item in caught])
```

Both bonds receive zero budgets and the fit issues two warnings, one for each rule. With both held
at their target weights, the fitted budgets reproduce the average target within $10^{-3}$.

## Implementation in optimalportfolios

- `solve_for_risk_budgets_from_given_weights` takes `prices`, `given_weights`, `covar_dict`,
  `min_risk_budget=1e-4`, `max_risk_budget=0.99`, `ewma_span=None` and
  `fixed_weight_assets=None`, and returns the fitted budgets as a Series on the columns of
  `prices`. They sum to one, and they are zero for held assets and for assets with zero target
  weight. It raises `ValueError` for an invalid target,
  span or pin, and `RuntimeError` when no candidate meets the tolerances. Each evaluation runs a
  complete `rolling_risk_budgeting` path, so the cost grows with the number of dates.
- `average_rolling_weights(weights, ewma_span=None)` returns the average of a date-by-asset path,
  indexed by its columns. It is exported so that a caller can report the same average that the fit
  used.
- `INVERSE_EWMA_SPAN` is the former default span, 12 rebalances; import it from
  `optimalportfolios.optimization.risk_allocation.risk_budgeting`.

The forward solves are `rolling_risk_budgeting` and `wrapper_risk_budgeting`, described on the
[risk budgeting](risk_budgeting.md) page; the risk shares come from
`qis.compute_portfolio_risk_contributions`. The code is in
[`risk_budgeting.py`](../src/optimalportfolios/optimization/risk_allocation/risk_budgeting.py).
Two repository workflows use the fit: the offline
[`inverse_risk_budget_bonds.py`](../examples/solvers/inverse_risk_budget_bonds.py) runs the
four-sleeve example: the floor probe, automatic holding, and an explicit `fixed_weight_assets`
that agrees with it; and
[`balanced_risk_budgets.py`](../examples/backtests/balanced_risk_budgets.py) fits the budgets of a
static equity, bond and gold mix and reports the risk-budgeted portfolio against it; it needs a
network connection.

> **Pitfall.** A zero fitted budget does not mean a zero weight. To run the budgets forward, hold
> every asset that has a positive target weight and a zero budget with equal lower and upper
> bounds at its target weight. Without them `wrapper_risk_budgeting` excludes the asset: in the
> example, Bonds fall from 45% to zero and Equity rises to 66%.

## Interpretation and limitations

- Maillard, Roncalli and Teïletche (2010) and Roncalli (2013) study the forward problem on one
  covariance matrix, where the portfolio of positive budgets exists and is unique. The package
  uses the closed form of Proposition 1 only as the seed of its fit. The rolling fit is a
  tolerance search with no uniqueness guarantee, and it holds hedging assets at their weights
  instead of admitting negative budgets.
- The fitted budgets reproduce the target on average, not on each date: in the example the
  forward weight of Bonds moves from 59% to 32% around its 45% target.
- The fit's forward model is long-only with only the equal bounds of held assets. Running the
  budgets with caps, group limits or frozen positions gives different weights; see
  [risk budgeting](risk_budgeting.md#missing-data-frozen-assets-and-feasibility).
- A held asset still carries risk. Its zero budget means that it is excluded from the fit, not
  that its risk contribution is zero: the held Bonds have a risk share of −8.3%.
- The budgets are known only to the tolerance of the fit, a gap of $10^{-4}$ on average and
  $10^{-3}$ at most in weight. For a target weight of 2.6%, a gap of $10^{-3}$ is about 4% of
  the weight.
- With `ewma_span` set, the first rebalance of a short path carries the residual weight
  $\lambda^{T-1}$, which on twelve dates with span 12 exceeds the weight of the latest date.
- Supply `covar_dict` in chronological order. The forward path and its average follow the order
  of the dictionary, while the hold rule sorts the dates.

## See also

- [Risk budgeting](risk_budgeting.md)
- [Portfolio constraints](constraints.md)
- [Rolling backtests](rolling_backtests.md)
- [Conventions, notation and glossary](conventions.md)
- [Examples](examples_readme.md)

## References

- Maillard, S., Roncalli, T. and Teïletche, J. (2010). *The Properties of Equally Weighted Risk
  Contribution Portfolios*. The Journal of Portfolio Management, 36(4), 60–70.
  [DOI 10.3905/jpm.2010.36.4.060](https://doi.org/10.3905/jpm.2010.36.4.060).
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting*. Chapman and Hall/CRC Financial
  Mathematics Series. [DOI 10.1201/b15151](https://doi.org/10.1201/b15151).
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
