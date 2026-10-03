---
myst:
  html_meta:
    description: >-
      Stress a stock and short-option portfolio with OptimalPortfolios risk models and QIS
      repricing: an offline three-stock book reproduces the scenario and repricing equations,
      and a manual runner applies FCGL clusters and VOP prices to five stocks and ten options.
---

# Stress testing with options and FCGL clusters

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-14](https://github.com/ArturSepp/OptimalPortfolios/commit/bdcbc350a2699eb6cf8769f9af7c15aba38c545a)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

This example applies a clustered sparse factor model to an existing option portfolio.
OptimalPortfolios estimates the model, FactorLasso supplies the FCGL fit and its cluster
topology, VOP prices each option, and QIS computes and reports portfolio stress and risk.
No portfolio optimisation or backtest changes the supplied positions.

## Overview

The portfolio is the same teaching book as the
[QIS options example](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/stress_testing_with_options.md):
five long stocks, five short calls and five short puts, with synthetic Bloomberg-style IDs.
The stocks are **AAPL, MSFT, AMZN, GOOGL and NVDA**. Factors remain **SPY, TLT, GLD and USO**.

The difference is the response model. Instead of unpenalised EWMA betas, this example selects
`LassoModelType.FACTOR_CLUSTER_GROUP_LASSO` (**FCGL**). It penalises the vector of a cluster's
asset loadings on each factor jointly. This differs from HCGL's row-wise grouping over an
individual asset's factors. See the [FactorLasso implementation](https://github.com/ArturSepp/factorlasso).

The report displays the **actual fitted dendrogram, cut distance, memberships and descriptive
labels**, followed by cluster contributions to stress, factor exposure and annual risk.
The option payoff remains nonlinear under either risk model; FCGL does not replace full option
repricing with a linear approximation.

The [worked example](#worked-example) first reproduces the scenario construction and the option
repricing offline, on a synthetic three-stock book with fixed inputs, and then reports the
frozen market-data run.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Log returns: weekly `W-WED` log returns of adjusted closes for estimation, log-return loadings and factor log shocks; a simple-return anchor $a$ enters as $\log(1+a)$ |
| Estimation grid | Weekly `W-WED` returns with a 52-observation EWMA span for the FCGL fit, the clustering and the factor covariance; the offline example estimates nothing and supplies fixed loadings, factor covariance and residual variances |
| Rebalancing grid | None: one valuation date (2025-12-31 by default) and one risk date, the last completed weekly observation at or before it; the positions are fixed and never rebalanced |
| Covariance units | Annual log-return squared: the weekly EWMA factor covariance and the residual variances times 52; the risk bands scale it by the one-month horizon $h=1/12$ |
| Expected returns | None: conditional completion assumes zero-mean factors, and `demean=False` adds no intercept or drift to a scenario |
| Weight state | Fixed signed holdings, share counts and option contracts marked in USD; percentages divide by the fixed denominator $N$, the sum of the signed marks |
| Solver | CVXPY with CLARABEL for the FCGL fit; option pricing, scenario completion and repricing are closed form, and the offline example runs no solver |

The notation follows the [conventions page](conventions.md#notation).

The manual market-data runner uses the inputs below; the offline worked example replaces the
data and the fit with the fixed inputs it states.

| Input | Default and interpretation |
|---|---|
| Valuation cutoff | 2025-12-31; configurable with `--as-of`. |
| Historical prices | yfinance, 2015-01-01 through the cutoff; explicit raw and adjusted closes. |
| Stock positions | Largest 100-share lot count within USD 2m per stock. |
| Calls | Sell one contract for every held 100-share lot. |
| Puts | Sell floor(1.5 times the stock lot count) contracts; additional downside obligations. |
| Factors | SPY, TLT, GLD, USO; model inputs, not four additional holdings. |
| Return convention | Weekly W-WED log returns from adjusted prices; no volatility targeting. |
| Estimation span | 52 weekly observations for FCGL weights, clustering and factor covariance. |
| FCGL settings | `reg_lambda=1e-5`, `group_penalty="normalized"`, `l1_weight=0`, CLARABEL. |
| Priors and signs | Zero-centred penalty; no supplied priors or sign restrictions; automatic signs disabled. |
| Clustering | Ward linkage, `ONE_MINUS_RHO` distance, target at most two clusters; no smoothing. |
| Cluster sample | Five underlying stock return histories; options and ETF proxies are not clustering leaves. |
| Reference currency | USD; no currency conversion is required for this book. |
| Reporting denominator | Sum of all fifteen signed current marks, USD 8,726,094 in the frozen run. |
| Risk-band horizon | One month; covariance scales with time, option maturity does not advance. |

Let $Y$ be the $T\times5$ stock-return panel, $X$ the $T\times4$ factor-return panel,
$B$ the $5\times4$ loading matrix, and $g$ a fitted cluster with $n_g$ members.
$\Sigma_F$ and $d_i$ are annual factor covariance and stock residual variance.
$N$ is the fixed reporting denominator and $J_i$ the aggregated dollar sensitivity to
stock $i$'s log return, including its stock, call and put positions.

The downloader explicitly uses `auto_adjust=False` and the next calendar day as Yahoo's
exclusive end date. Raw closes set option spots; adjusted closes supply estimation returns.
Only complete, positive common price rows are retained. No missing return is invented as zero.
The initial return boundary is missing and masked out of the FCGL loss.

A hashed CSV cache freezes the Yahoo response. Later runs verify and reuse its bytes;
`--refresh` explicitly replaces it. Yahoo can revise history, so a date cutoff alone does not
guarantee identical future downloads. The sample is observed market history; option terms and
implied volatilities are synthetic assumptions, not downloaded exchange quotes.

## Methodology

### 1. Fit FCGL with its discovered partition

FactorLasso derives a dependence matrix from the stock histories under the configured
52-week weighting and `demean=False` convention. The distance is $1-\rho_{ik}$ and Ward
linkage produces the tree. `n_clusters=2` requests at most two groups; tied merge heights can
produce fewer. This is a transparent teaching setting, not an optimised cluster count.

For the complete sample, let $\eta=1-2/(52+1)$ and let $m_{ti}$ mark an available observation.
The configured FCGL objective is

$$
\widehat B=\arg\min_B
\left\lbrace
\frac{1}{T}\sum_{t=1}^{T}\sum_{i=1}^{5}
m_{ti}\eta^{T-t}(y_{ti}-B_i x_t)^2
+\lambda\sum_{g=1}^{G}\sqrt{\frac{n_g}{G}}
\sum_{j=1}^{4}\lVert B_{g,j}\rVert_2
\right\rbrace,\qquad \lambda=10^{-5}.
$$

Here $B_{g,j}$ collects all member stocks' loadings on factor $j$. A whole cluster-factor
block can shrink towards zero, while nonzero member loadings can differ. There is no
elementwise L1 term, no prior other than zero, and no sign constraint in this example.
The loss scale includes $1/T$; the same numerical penalty need not have the same strength
under another sample length or a differently normalised regression implementation.

`demean=False` leaves returns uncentred in the regression and adds no intercept or alpha to
scenario valuation. The model is fitted only once, at the last completed weekly observation
at or before valuation. No future returns or option scenario outcomes determine the partition.

### 2. Assemble the assigned risk model through OP

To isolate the change in response estimation, factor covariance uses the same zero-mean
EWMA convention as the QIS example:

$$
C_t=\eta C_{t-1}+(1-\eta)x_tx_t^\top,\qquad C_0=0,
\qquad \Sigma_F=52C_T.
$$

The example obtains this matrix from `qis.compute_ewm_covar` and passes it explicitly as
the **annual** `x_covar` argument to `fit_current_factor_covars`. OP's internally computed
factor covariance applies demeaning; supplying the matrix is therefore deliberate.

FactorLasso residual diagnostics normalise the observation weights over each valid history:

$$
\bar w_{ti}=\frac{m_{ti}\eta^{T-t}}{\sum_s m_{si}\eta^{T-s}},
\qquad d_i=52\sum_t\bar w_{ti}(y_{ti}-\widehat B_i x_t)^2.
$$

The OP snapshot retains $\widehat B$, $\Sigma_F$, $d$, fit diagnostics and cluster metadata.
`optimalportfolios.build_risk_model({date: snapshot})` creates the QIS `RiskModel`, with

$$
\Sigma_Y=\widehat B\Sigma_F\widehat B^\top+\operatorname{diag}(d).
$$

FactorLasso's reported $R^2$ compares the weighted squared residual to weighted dispersion
around the response's weighted mean; this diagnostic denominator is centred even though the
configured regression is not. OP clips negative reported $R^2$ to zero. It differs from the
non-centred $R^2$ shown by the QIS EWMA example and is not an out-of-sample measure.

### 3. Reprice the unchanged option book

Call strikes target 103%, 105%, 107%, 104% and 108% of stock spot; put strikes target
97%, 95%, 93%, 96% and 92%, all rounded to USD 5. Maturities are the third Fridays two,
three, four, five and eight months after the valuation month. For example,
`AAPL US 02/20/26 C280 Equity` is a synthetic identifier using Bloomberg syntax.

The VOP adapter prices European options with a fixed continuous 4% rate, zero dividend yield,
and assumed IV. Call IV is max(15%, 1.15 times 63-day EWMA annual realised volatility);
put IV adds four volatility percentage points. One contract represents 100 shares.

For a factor log-shock vector $z$, the response and option P&L are

$$
S_i(z)=S_i(0)\exp(B_i z),\qquad
\Delta V_\ell(z)=100n_\ell
\left[v_\ell(S_i(z))-v_\ell(S_i(0))\right],
\qquad R_p(z)=\frac{\sum_\ell\Delta V_\ell(z)}{N}.
$$

Signed contracts $n_\ell \lt 0$ create negative option gamma. Stocks use shares times the spot
change. Current source marks equal VOP prices, so zero shock preserves every mark and
produces zero P&L. Premiums are not added again as a separate cash asset.
The [QIS options guide](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/stress_testing_with_options.md)
details BSM pricing and VOP's forward-to-spot Greek conversion.

### 4. Complete correlated shocks and conditional risk bands

For a simple anchor $a_j$, QIS first computes $z_j=\log(1+a_j)$. For anchored factors $A$
and remaining factors $F$, joint completion and conditional covariance are

$$
z_F=\Sigma_{FA}\Sigma_{AA}^{-1}z_A,\qquad
\Sigma_{F\mid A}=\Sigma_{FF}-\Sigma_{FA}\Sigma_{AA}^{-1}\Sigma_{AF}.
$$

An explicit zero remains an anchor; omitted factors are free. Historical monthly scenarios
already contain all four observed factor returns and are replayed without reconditioning.

For each stock, QIS aggregates the stock and both options' dollar deltas before computing
systematic or idiosyncratic risk. With $q_i(z)=J_i(z)/N$ and $e(z)=B^\top q(z)$,

$$
v(z)=h\left[e_F(z)^\top\Sigma_{F\mid A}e_F(z)+\sum_iq_i(z)^2d_i\right],
\qquad h=\frac{1}{12}.
$$

The report plots $R_p(z)\pm\sqrt{v(z)}$ and $R_p(z)\pm2\sqrt{v(z)}$.
The exported 95% bounds use approximately 1.96 standard deviations. Deltas change with
the stressed spots, but these are local Gaussian bands, not nonlinear option P&L quantiles.

### 5. Display the fitted clusters and their contributions

Use `factorlasso.get_clusters_by_freq`, `get_linkages_by_freq` and `snapshot.cutoffs` to
populate `StressReportConfig.cluster_memberships`, `cluster_linkages` and `cluster_cutoffs`.
Never recluster the fitted beta table merely to obtain a dendrogram.

`factorlasso.cluster_lineage.analyze_cluster_lineage` supplies descriptive factor/volatility
labels from an equal-member fingerprint of this snapshot. With only one date, these labels
do not establish persistence or stability over time. Display labels leave raw memberships
unchanged; different clusters may receive the same economic description.

Each stock, call and put belongs to its underlying's fitted cluster. The membership page
shows **response weight** $J_i/N$, including option delta; this is not a derivative MTM
allocation. The contribution page uses the same fixed denominator throughout:

| Display | Computation and meaning |
|---|---|
| Cluster scenario P&L | Sum full holding repricing P&L within the cluster, divided by $N$. |
| Three largest contributors | Rank absolute holding P&L under that cluster's worst requested correlated scenario; retain signed contributions and show the scenario. |
| Factor-exposure heatmap | Sum each member's current dollar response sensitivity times its factor loading, divided by $N$. |
| Factor-risk bars | Cluster allocations of the portfolio's five largest absolute factor Euler contributions; only four factors exist here, so all four are eligible. |
| Annual model-volatility bars | Signed systematic Euler contributions plus shared-response residual Euler contributions. |

For cluster exposure $e_{g,j}=\sum_{i\in g}q_iB_{ij}$ and portfolio annual volatility
$\sigma_p$, the factor and residual Euler contributions are

$$
RC_{g,j}=\frac{e_{g,j}(\Sigma_F e)_j}{\sigma_p},\qquad
RC_g^{\epsilon}=\frac{\sum_{i\in g}q_i^2d_i}{\sigma_p}.
$$

Contributions sum to portfolio volatility across clusters and factors. They are allocations
of total portfolio risk, not standalone cluster volatilities. Negative factor contributions
can reflect hedging. Cluster P&L and risk add across the original holdings without creating
three independent residual risks for a stock and its two options.

## Worked example

The worked example has two parts. An offline synthetic book reproduces the scenario
construction and the repricing equations of the methodology; the frozen market-data run of the
manual runner follows.

The four Python blocks below run in order and need no download, data file or random seed. They
are excerpts of the canonical script
[`examples/docs/stress_testing_with_options.py`](../examples/docs/stress_testing_with_options.py),
which runs them and asserts every equation they exercise and every number stated about them
against a reference computed a different way: option values integrated over the lognormal law,
deltas by finite differences, put-call parity, and explicit solves and sums for the completion,
the bands and the risk contributions:

```console
python -m examples.docs.stress_testing_with_options
```

### An offline risk model

Fixed teaching inputs replace the FCGL fit and the EWMA factor covariance: three factors with
annual volatilities of 16%, 7% and 25%, three stocks with log-return loadings on them, and
annual residual volatilities of 20%, 15% and 25%. `optimalportfolios.build_risk_model` turns
the snapshot into the QIS `RiskModel` with the covariance of section 2.

```python
import numpy as np
import pandas as pd
import qis
from factorlasso import CurrentFactorCovarData
import optimalportfolios as opt

factors = ["Equity", "Rates", "Commodities"]
vols = np.array([0.16, 0.07, 0.25])
correlation = np.array([[1.0, -0.3, 0.3],
                        [-0.3, 1.0, -0.1],
                        [0.3, -0.1, 1.0]])
stocks = ["Stock A", "Stock B", "Stock C"]
loadings = pd.DataFrame([[1.2, -0.3, 0.0],
                         [0.9, 0.2, 0.1],
                         [1.4, -0.5, 0.2]], index=stocks, columns=factors)
snapshot = CurrentFactorCovarData(
    x_covar=pd.DataFrame(np.outer(vols, vols) * correlation,
                         index=factors, columns=factors),
    y_betas=loadings,
    y_variances=pd.DataFrame({"residual_var": [0.20**2, 0.15**2, 0.25**2]},
                             index=stocks),
)
date = pd.Timestamp("2025-12-31")  # Synthetic snapshot label.
risk_model = opt.build_risk_model({date: snapshot})
```

### European options without an external pricer

The market-data runner prices options with VanillaOptionPricers, which is not a dependency of
OptimalPortfolios and which the repository's lint rules admit only in that optional runner.
This block prices the same European model with a small Black-Scholes-Merton helper, with no
dividends. `EuropeanOption` implements the public
`qis.HoldingPayoff` protocol: `evaluate` reprices the position at the scenario spots,
`response_jacobian` returns its current dollar delta per unit log move of the stock, and
`scenario_response_jacobian` returns that delta at a stressed spot for the conditional bands.

```python
from dataclasses import dataclass

from scipy.stats import norm

def bsm(spot, strike, ttm, vol, rate, kind):
    """Black-Scholes-Merton value and spot delta per share, no dividends."""
    spot = np.asarray(spot, dtype=float)
    stdev = vol * np.sqrt(ttm)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * ttm) / stdev
    d2 = d1 - stdev
    pv_strike = strike * np.exp(-rate * ttm)
    if kind == "C":
        return spot * norm.cdf(d1) - pv_strike * norm.cdf(d2), norm.cdf(d1)
    return pv_strike * norm.cdf(-d2) - spot * norm.cdf(-d1), norm.cdf(d1) - 1.0

@dataclass(frozen=True)
class EuropeanOption:
    """Signed European option contracts on one stock, at fixed volatility."""

    underlying: str
    kind: str
    strike: float
    ttm: float
    vol: float
    contracts: int
    rate: float = 0.04
    multiplier: int = 100
    implementation_id = "docs.bsm_european_option.v1"
    coverage = "European BSM value; volatility, rate and maturity held fixed."
    boundary_policy = "Smooth BSM delta; no expiry crossing."

    def evaluate(self, context):
        """Signed position value in each scenario, in USD."""
        spots = context.quotes[self.underlying]
        value, _ = bsm(spots.to_numpy(), self.strike, self.ttm, self.vol,
                       self.rate, self.kind)
        return pd.Series(self.contracts * self.multiplier * value, index=spots.index)

    def dollar_delta(self, context, spot):
        """Dollar sensitivity to each shared log response at one spot."""
        _, delta = bsm(spot, self.strike, self.ttm, self.vol, self.rate, self.kind)
        dollars = self.contracts * self.multiplier * float(delta) * spot
        return dollars * context.quote_response_jacobian.loc[self.underlying]

    def response_jacobian(self, context):
        """Current dollar delta, for QIS exposures and risk."""
        return self.dollar_delta(context, context.baseline_quotes[self.underlying])

    def scenario_response_jacobian(self, context):
        """Dollar delta at the stressed spot, for the conditional bands."""
        return self.dollar_delta(context, context.quotes.iloc[0][self.underlying])
```

The canonical script checks every value against the discounted expected payoff integrated over
the lognormal law, put-call parity at every strike, and every delta against a central difference.

### The book

As in the market-data runner, each stock position is the largest 100-share lot count within a
budget, here USD 1m, with one short call per lot and floor(1.5 times the lots) short puts.
Strikes are rounded to USD 5, and put volatility is call volatility plus four points.
`share_delta` accumulates each stock's net share-equivalent delta: its shares plus 100 times
contracts times option delta.

```python
spots = pd.Series([152.0, 83.0, 236.0], index=stocks)
call_moneyness = pd.Series([1.05, 1.04, 1.06], index=stocks)
put_moneyness = pd.Series([0.95, 0.96, 0.94], index=stocks)
maturities = pd.Series([0.25, 0.40, 0.60], index=stocks)  # years
call_vols = pd.Series([0.28, 0.24, 0.36], index=stocks)

holdings, share_delta = [], pd.Series(0.0, index=stocks)
for stock in stocks:
    spot = spots[stock]
    lots = int(1_000_000 / (100 * spot))
    holdings.append(qis.PortfolioHolding(
        stock, stock, 100 * lots * spot,
        (qis.InstrumentLeg(qis.InstrumentType.DELTA_1, stock, 100 * lots),),
    ))
    share_delta[stock] += 100 * lots
    for kind, moneyness, contracts, vol in (
        ("C", call_moneyness[stock], -lots, call_vols[stock]),
        ("P", put_moneyness[stock], -int(1.5 * lots), call_vols[stock] + 0.04),
    ):
        strike = float(5 * np.round(spot * moneyness / 5))
        option = EuropeanOption(stock, kind, strike, maturities[stock], vol, contracts)
        value, delta = bsm(spot, strike, option.ttm, vol, option.rate, kind)
        holding_id = f"{stock} {kind}{strike:g}"
        holdings.append(qis.PortfolioHolding(
            holding_id, holding_id, contracts * 100 * float(value), payoff=option,
        ))
        share_delta[stock] += contracts * 100 * float(delta)

portfolio = qis.InstrumentPortfolio(
    holdings=tuple(holdings),
    underlyings={stock: qis.Underlying(stock, spots[stock], "USD", stock,
                                       qis.ResponseBasis.LOCAL) for stock in stocks},
    risk_model=risk_model,
    risk_date=date,
    valuation_date=date,
    reference_currency="USD",
    reporting_denominator=sum(holding.observed_mtm for holding in holdings),
    denominator_label="Net marked portfolio value",
)
print(pd.Series({h.holding_id: h.observed_mtm for h in holdings}).round(0))
print(f"N = USD {portfolio.reporting_denominator:,.0f}")
```

The book holds 6,500, 12,000 and 4,200 shares. It is short 65, 120 and 42 calls struck at
160, 85 and 250, and 97, 180 and 63 puts struck at 145, 80 and 220. The nine signed marks sum
to $N$ = USD 2,544,055, the denominator of every percentage below. Each option mark is its
model price, so the zero shock returns every mark and zero P&L.

### Scenarios and the delta-gamma split

Seven Equity scenarios from -30% to +30% are simple-return anchors. QIS converts each to
$z=\log(1+a)$ and completes Rates and Commodities conditionally, as in section 4; at Equity
-20% the completed moves are Rates +2.97% and Commodities -9.93%. The block then splits the
full-repricing P&L into a delta part, linear in the spot moves, and the gamma part that
repricing adds:

$$
P(z)=\sum_i D_i\left[S_i(z)-S_i(0)\right]+G(z),\qquad
D_i=m_i+100\sum_{\ell\in i}n_\ell\delta_\ell,
$$

$$
G(z)=100\sum_\ell n_\ell\left[v_\ell(S_i(z))-v_\ell(S_i(0))-\delta_\ell\left(S_i(z)-S_i(0)\right)\right].
$$

Here $m_i$ is the share count of stock $i$, $\delta_\ell$ the current spot delta of option
$\ell$ on it, and `share_delta` holds $D_i$. Stocks are linear in their spot, so $G$ comes
from the options alone. Each European option value is convex in spot and every
$n_\ell \lt 0$, so $G(z)\le 0$ in every scenario.

```python
equity_moves = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
scenarios = qis.StressScenarios(
    pd.DataFrame({"Equity": equity_moves},
                 index=[f"Equity {move:+.0%}" for move in equity_moves]),
    mode=qis.ScenarioMode.CONDITIONAL,
    convention=qis.ShockConvention.SIMPLE,
)
result = qis.run_portfolio_stress_test(
    portfolio, scenarios, factor_grids={"Equity": scenarios},
    config=qis.StressTestConfig(horizon_years=1 / 12),
)
valuation = result.valuations["requested"]
spot_moves = np.exp(valuation.factor_log_shocks @ loadings.T) * spots - spots
delta_pnl = (spot_moves * share_delta).sum(axis=1)
total_pnl = valuation.portfolio_pnl
split = pd.DataFrame({
    "Delta": delta_pnl,
    "Gamma": total_pnl - delta_pnl,
    "Total": total_pnl,
}) / portfolio.reporting_denominator
print(split.map("{:.2%}".format))
```

In percent of $N$:

| Equity move | Delta part | Gamma part | Total |
|---|---:|---:|---:|
| -30% | -42.38% | -33.41% | -75.79% |
| -20% | -28.73% | -16.56% | -45.28% |
| -10% | -14.59% | -4.37% | -18.96% |
| 0% | 0.00% | 0.00% | 0.00% |
| +10% | +15.02% | -3.96% | +11.06% |
| +20% | +30.44% | -14.02% | +16.42% |
| +30% | +46.26% | -27.54% | +18.72% |

The same call returns one-month conditional bands in `grid_summaries`, computed from the deltas
at the stressed spots. Their standard deviation is 4.15% of $N$ at no move and 6.47% at Equity
-20%, where the in-the-money short puts at least double each stock's delta. It falls to 0.61%
at +30%, where the in-the-money short calls offset 87% or more of that delta.

![Left: for Equity moves of -30%, -20%, -10%, +10%, +20% and +30%, the delta part of the book's
P&L is -42.38%, -28.73%, -14.59%, +15.02%, +30.44% and +46.26% of N, the gamma part is -33.41%,
-16.56%, -4.37%, -3.96%, -14.02% and -27.54%, and the full-repricing total is -75.79%, -45.28%,
-18.96%, +11.06%, +16.42% and +18.72%. Right: the short options' full-repricing P&L against the
Equity move is an inverted parabola, falling to about -34% of N at -30% and -27% at +30%, while
their delta line stays within 1% of zero; the shaded gap between them is the gamma
part.](images/option_stress_scenarios.png)

*Figure: the delta and gamma parts of the offline book's stress P&L by Equity scenario, and the
short options' P&L against the Equity move with their delta line. Drawn by the `exhibit`
function of the canonical script; the [analytics gallery](analytics_gallery.md) lists its
provenance.*

> **Insight.** Short gamma costs in both directions. At Equity -20% the gamma part adds
> -16.56% of $N$ to a delta loss of -28.73%; at +20% it gives back 14.02 points of a 30.44-point
> delta gain, almost half.

> **Pitfall.** A zero in a scenario table is an anchor, not a blank. Filling the free Rates and
> Commodities cells of the Equity -20% row with zeros, as `ScenarioMode.INDEPENDENT` does,
> discards their completed moves of +2.97% and -9.93% and reports -42.41% of $N$ instead of
> -45.28%. Leave the factors you do not anchor as NaN.

### Frozen market-data run

The manual runner applies the same equations to the five-stock, ten-option book with Yahoo
prices and the FCGL model. Its frozen sample yields two fitted groups:

| Raw fitted ID | Underlying members | Holdings assigned |
|---|---|---:|
| W-WED:1 | MSFT, NVDA | 6 |
| W-WED:2 | AAPL, AMZN, GOOGL | 9 |

All stock quantities, ten option strikes/maturities/quantities/IVs, current marks and the
USD 8,726,094 reporting denominator match the QIS example. These are percentage returns
under **correlated SPY shocks**, with other factors following the shared covariance:

| SPY move | FCGL full repricing | FCGL current log-delta | QIS EWMA full repricing |
|---|---:|---:|---:|
| -30% | -70.74% | -48.84% | -84.15% |
| -20% | -42.38% | -30.55% | -51.54% |
| -10% | -17.74% | -14.43% | -21.70% |
| 0% | 0.00% | 0.00% | 0.00% |
| +10% | +10.03% | +13.05% | +11.27% |
| +20% | +14.70% | +24.96% | +15.73% |
| +30% | +16.68% | +35.92% | +17.26% |

The current log-delta column multiplies the completed factor log shocks by the current
`factor_exposures` of the QIS result, so it is linear in the log shocks for the stocks as well
as the options. On the offline book it gives -52.81% at Equity -30%, against -75.79% from full
repricing.

At SPY -20%, the FCGL result combines stock P&L of -25.86%, short calls +5.56% and short
puts -22.08%. At SPY +20%, short calls absorb 19.13 percentage points of the stock gain.
The negative convexity remains visible after changing the risk estimator. Lower losses
under FCGL do not prove greater accuracy: shrinkage changes the extrapolated stock responses.

These market-data numbers are outputs of the manual runner on its frozen cache, not of the
canonical script. The runner records the package versions and source hashes of each run in its
provenance file; the frozen run used VOP 2.2.0 and FactorLasso 0.18.0, and its observed-price
cache SHA-256 is
`1e366949dd117c5ef2a62e810811598ada619f33fbf1e5dbbb4add7154e0f98b`.

## Implementation in optimalportfolios

The [self-contained manual runner](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/reports/stress_testing_with_options_local.py)
requires OP with its QIS dependency (`qis>=5.33.1`, which includes the QIS instrument stress
reports), FactorLasso 0.18 or later, yfinance and VOP. The `_local.py` suffix keeps this
optional pricing/reporting workflow outside the core-only unattended example lanes. It has been
run end to end with the frozen Yahoo cache. It does not require a separate QIS examples checkout
or Bloomberg access.

In your chosen environment, install the example-only prerequisites and run from an OP checkout:

```console
python -m pip install -e ".[data]" "factorlasso>=0.18.0" "vanilla-option-pricers==2.2.0"
python -m examples.reports.stress_testing_with_options_local --cache-dir /absolute/cache/options_stress --output-dir /absolute/output/fcgl_options_stress
```

Use a fresh output directory. Omit `--output-dir` to calculate and verify without exporting.
Changing `--as-of` rebuilds contract terms consistently. Keep the cache for exact replay;
`--refresh` intentionally downloads a new price panel.

The central OP integration is included from the actual runner; the ordinary
[source link](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/reports/stress_testing_with_options_local.py)
also works in Markdown viewers.

```{literalinclude} ../examples/reports/stress_testing_with_options_local.py
:language: python
:start-at: def fit_risk_model(prices):
:end-before: def cluster_report_inputs(snapshot):
```

Outputs include the standard QIS PDF/workbook/manifest, a separate convexity PDF and PNG,
full-precision scenario curves, option inventory, fit diagnostics, factor covariance and
loadings, raw fitted clusters/linkages/cutoffs, descriptive labels, and source provenance.
All plotting and stress analytics remain in QIS; generic sparse estimation remains in FactorLasso.

Every run checks FCGL block optimality independently, residual annualisation, OP-to-QIS
snapshot preservation, the dendrogram cut's membership equivalence, option Greeks against
finite differences, put-call parity, zero-shock marks, factor dollar deltas and P&L attribution.
The checks compare partitions up to arbitrary cluster-ID numbering. They do not require
penalised coefficients to equal unpenalised least squares.

The [canonical script](../examples/docs/stress_testing_with_options.py) runs the offline worked
example and checks the risk-model assembly, the option values, parity and deltas, conditional
completion and explicit zero anchors, stressed repricing and the delta-gamma split, the QIS
attribution and current log-delta line, the conditional bands with stressed deltas, historical
replay without reconditioning, and the Euler identity of the cluster contributions on a fixed
two-group partition. The test suite runs it, and so does the offline examples lane of CI.

## Interpretation and limitations

- Five technology/growth stocks provide a small, concentrated teaching universe. Two clusters
  are a display/regularisation choice, not evidence of two stable market regimes.
- The offline book's loadings, covariances, spots and volatilities are fixed teaching inputs;
  its numbers illustrate the equations, not a market.
- Options are fixed-IV European BSM proxies for US equity contracts. American exercise,
  discrete dividends, volatility-surface shocks, margin, transaction costs and path-dependent
  knockout states are not modelled.
- FCGL extrapolates fitted stock loadings across large shocks. The penalty is not calibrated
  here by cross-validation, and historical scenario ranks are not scenario probabilities.
- Residual dependence between different stocks is omitted. Residual risk for holdings on the
  same stock is shared and aggregated before risk calculation.
- Conditional bands use stressed local deltas but omit gamma/vega dispersion, jumps and
  parameter uncertainty. The central full-repricing curve and the band approximation have
  different accuracy claims.
- The QIS and FCGL fits use different residual and $R^2$ conventions described above. Current
  option inventory, factor covariance and stress mechanics are held fixed for comparison.

## See also

- [Factor covariance with HCGL](factor_covariance_hcgl.md): OP factor estimation and annual covariance.
- [Rolling factor covariance from prices](rolling_factor_covar_from_csv.md): return preparation and rolling interfaces.
- [QIS options stress guide](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/stress_testing_with_options.md): the matched EWMA example and VOP pricing formulas.
- [QIS instrument stress interface](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/portfolio_stress.md): custom payoff and reporting contracts.

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [FactorLasso software and methodological references](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
  The [FCGL implementation](https://github.com/ArturSepp/FactorLasso/blob/main/src/factorlasso/linear_model/_solvers/group_lasso.py)
  defines the cluster-factor penalty, observation weights and residual diagnostics.
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers): BSM values and forward Greeks.
- Black, F., and Scholes, M. (1973). The pricing of options and corporate liabilities.
  *Journal of Political Economy*, 81(3), 637-654. The European model of the offline helper.
- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html): observed stock/ETF data and request conventions.
