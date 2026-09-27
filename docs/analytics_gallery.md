---
myst:
  html_meta:
    description: >-
      Reproducible OptimalPortfolios analytics: six previews of a synthetic portfolio and a
      teaching exhibit for each methodology article and case study, with their provenance.
---

# Analytics gallery

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-14](https://github.com/ArturSepp/OptimalPortfolios/commit/19cae02086d4fbfcf4377796bfef2c576d989134)*

Examples from [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

The gallery has six previews and 25 teaching exhibits. The previews connect
portfolio construction to risk and performance analytics on one synthetic portfolio:
OptimalPortfolios estimates risk and constructs targets,
[qis](https://github.com/ArturSepp/QuantInvestStrats) simulates holdings and computes analytics,
and [factorlasso](https://github.com/ArturSepp/FactorLasso) fits the sparse factor models. Each
teaching exhibit is drawn by the canonical script of its article and illustrates one result of
it.

All inputs are synthetic or fixed, except the cryptocurrency exhibit, which is derived from the
tracked price panel of its paper. The figures illustrate calculations and implementation
conventions; they do not establish historical performance or recommend an allocation or
estimator. The previews' [shared provenance record](../examples/figures/analytics_manifest.json)
identifies the effective source, inputs, parameters, actual software environment, generation
time and visual review; the teaching exhibits have their own
[manifest](images/analytics_manifest.json). A fixed sample ending in 2025 is separate from the
date the image was generated.

## Choose an exhibit

The six previews simulate one synthetic portfolio end to end:

| Question | Exhibit |
|---|---|
| How do growth and drawdowns compare with a synthetic benchmark? | [Performance](#portfolio-performance) |
| What are the final targets and their estimated risk contributions? | [Allocation and risk](#allocation-and-risk) |
| How do decided allocations and realized trading costs evolve? | [Allocation through time](#allocation-through-time) |
| How does covariance smoothing affect this example? | [Span sensitivity](#covariance-span-sensitivity) |
| How do three objectives behave with common risk inputs? | [Objective comparison](#portfolio-objectives) |
| How do estimates compare with a known simulated covariance? | [Covariance estimators](#covariance-estimators) |

The 25 teaching exhibits each illustrate one article, in the order of the sidebar:

| Question | Exhibit |
|---|---|
| Which observations enter each asset's classic momentum when monthly and quarterly assets share a formation date, and how is a quarterly signal carried between quarter ends? | [Mixed-frequency data](#mixed-frequency-data) |
| What happens to NAV and holdings when a price is missing between rebalancings, on a rebalance date, and at the opening trade? | [Incomplete histories](#incomplete-histories) |
| How much autocorrelation and volatility does unsmoothing restore to an appraisal-smoothed private-asset series? | [Universe data and unsmoothing](#universe-data-and-unsmoothing) |
| Do both residual types give each asset the same variance, and how much residual variance does a portfolio of assets sharing an unspanned shock collect under each? | [Factor covariance with HCGL](#factor-covariance-with-hcgl) |
| How do capital weights and risk shares differ across equal weight, minimum variance, equal risk contribution and maximum diversification on one covariance? | [Ex-ante risk contributions](#ex-ante-risk-contributions) |
| How does scoring one signal within clusters change the ranking of assets compared with scoring it across the whole cross-section? | [Alpha signals](#alpha-signals) |
| Does the signal rank returns, and how stable is its IC? | [Signal diagnostics](#signal-diagnostics) |
| Are the risk budgets met when a weight bound binds? | [Risk budgeting](#risk-budgeting) |
| What static risk budgets reproduce a target allocation on average, and how far do the weights they imply drift from it date by date? | [Implied risk budgets](#implied-risk-budgets) |
| How do HRP, equal risk contribution and equal cluster budgets divide capital and risk among the blocks of one universe? | [Hierarchical risk parity](#hierarchical-risk-parity) |
| Is the maximum diversification portfolio the minimum-variance portfolio of the correlation matrix, rescaled by inverse volatility, and do its held assets share one correlation with it? | [Maximum diversification](#maximum-diversification) |
| Where do minimum variance, utility and maximum Sharpe sit on the frontier? | [Mean-variance objectives](#mean-variance-objectives) |
| Do the target-return and target-volatility solvers trace the same frontier, hard and soft? | [Target return and target volatility](#target-return-and-target-volatility) |
| How do risk aversion and a crash component change the allocation? | [CARA utility under Gaussian mixtures](#cara-utility-under-gaussian-mixtures) |
| What does each added constraint cost in ex-ante tracking error against the benchmark? | [Minimum tracking error](#minimum-tracking-error) |
| Do active weights follow the closed form, and how does IR scale with the TE budget? | [Alpha over tracking error](#alpha-over-tracking-error) |
| How do the overlay sleeve and the portfolio's model risk and return change as the linear tail floor tightens from non-binding to near its reachable maximum? | [Overlay tail floor](#overlay-tail-floor) |
| What changes when a tracking-error limit becomes a penalty: how far does the solve exceed the limit, and what active return does that buy, as the penalty weight grows? | [Portfolio constraints](#portfolio-constraints) |
| What does the eigenvalue floor change as two proxies become collinear? | [Solver numerics and outcomes](#solver-numerics-and-outcomes) |
| How far do held weights drift from target weights between quarterly rebalancings? | [Rolling backtests](#rolling-backtests) |
| How does raising `turnover_utility_weight` trade turnover against ex-ante tracking error, and at what weight does trading stop? | [Turnover and transaction costs](#turnover-and-transaction-costs) |
| How do risk budgets translate into strategic weights, and do the tactical tilts spend the tracking-error budget in the direction of the alphas? | [ROSAA layers](#rosaa-layers) |
| How much does each method allocate to the crypto asset? | [Cryptocurrency allocation](#cryptocurrency-allocation) |
| How do factor premia and residual adjustments build the CMAs, and what allocation do they imply? | [Capital market assumptions](#capital-market-assumptions) |
| How do factor shocks and option repricing combine in a portfolio stress test? | [Stress testing with options](#stress-testing-with-options) |

Select a preview to open its full-resolution image. The
[offline quickstart](quickstart.md) provides a smaller executable introduction, and the
[examples guide](examples_readme.md) maps broader workflows and their data requirements.

## Samples and conventions

The first five previews use six assets from the fixed
[qis synthetic-universe generator](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/datasets/synthetic.py):
US and European equity, Treasuries, investment-grade bonds, gold and commodities.
Seed 20260725 and clean mode (`apply_quirks=False`) select complete business-day observations
from 4 January 2010 to 31 December 2025. Early history supplies estimation warmup;
the displayed performance window is 1 April 2015 to 31 December 2025.

The sixth preview, covariance estimators, uses the repository's
[known-factor simulator](../examples/covar_estimation/simulate_factor_returns.py), seed 42:
four factors, eight assets and 783 business-day observations from 2 January 2023 to
31 December 2025. Its displayed backtest starts on 3 January 2024 after weekly estimation warmup.

| Convention | Applied in the six previews |
|---|---|
| Estimation returns | Weekly Wednesday log returns, with trailing EWMA demeaning |
| Covariance units | Annualized using 52 weekly observations per year |
| Construction | Fully invested, long-only target weights; 35% cap per asset |
| Decision and execution | Quarterly schedules mapped to observed Wednesdays; targets trade one business-day observation later |
| Holdings | qis holds units between trades, so realized weights drift |
| Trading costs | 10 basis points of gross traded notional, including entry |
| Other costs | No funding or management fees |
| Displayed growth | Portfolio NAV after costs; the 60/40 reference in the first panel is explicitly gross |

The last target decision is 1 October 2025. That target snapshot differs from holdings at the
sample end. Cost bars sum daily cash cost divided by that day's NAV, expressed in basis points;
they measure cost incidence rather than compounded performance drag.
The [refresh specification](documentation_standard.md#analytical-conventions-and-figures)
records detailed conventions, solver acceptance and independent numerical checks.

## Portfolio performance

The maximum-diversification portfolio is compared with the fixture's daily-rebalanced gross
60/40 benchmark. Both series are displayed as growth of 100; drawdown measures the fall from
each series' running peak over the report window.

[![Synthetic maximum-diversification growth and drawdowns against a gross 60/40 benchmark](../examples/figures/example_portfolio_factsheet1.PNG)](../examples/figures/example_portfolio_factsheet1.PNG)

**Sample:** 1 April 2015–31 December 2025. The portfolio is net of 10 bp trading costs;
the benchmark is gross. This illustrative comparison does not isolate an allocation advantage.
**Producer:** [portfolio reports](../tools/docs_analytics/portfolio_reports.py).
**Related methodology:** [rolling backtests](rolling_backtests.md).

## Allocation and risk

The upper panel shows decided maximum-diversification weights. The lower panel shows each
asset's contribution to estimated annualized portfolio volatility using the same decision's
covariance. Contributions are measured in volatility percentage points, not portfolio weights.

[![Synthetic target weights and contributions to annualized portfolio volatility](../examples/figures/example_portfolio_factsheet2.PNG)](../examples/figures/example_portfolio_factsheet2.PNG)

**Decision:** 1 October 2025. Weekly log returns, EWMA span 52, annualization factor 52.
The 35% asset cap can bind. The estimate describes targets and ex-ante risk, rather than
end-of-sample holdings or realized future volatility.
**Producer:** [portfolio reports](../tools/docs_analytics/portfolio_reports.py).
**Related methodology:** [covariance estimators](covariance_estimators.md).

## Allocation through time

The stacked panel extends each decided target until the following decision for display.
It does not plot drifted holdings. The lower panel aggregates realized trading costs by
calendar quarter, including the initial allocation.

[![Synthetic decided allocations through time and quarterly sums of trading costs](../examples/figures/example_customised_report.PNG)](../examples/figures/example_customised_report.PNG)

**Sample:** 1 April 2015–31 December 2025. Extending the last target to the report end adds
neither a new decision nor a trade. Quarterly cost sums use each day's NAV denominator.
**Producer:** [portfolio reports](../tools/docs_analytics/portfolio_reports.py).
**Related methodology:** [turnover and transaction costs](turnover_and_transaction_costs.md).

## Covariance span sensitivity

Five maximum-diversification backtests vary the EWMA span across **5, 13, 26, 52 and 104**
weekly observations. Each span affects both trailing demeaning and covariance smoothing.
A span is an EWMA parameter, not a half-life or a finite estimation window.

[![Synthetic net maximum-diversification growth and total trading costs across five EWMA spans](../examples/figures/max_diversification_span.PNG)](../examples/figures/max_diversification_span.PNG)

**Sample:** 1 April 2015–31 December 2025. Assets, constraints, costs and implementation dates
are shared. The lower panel sums daily cost/NAV over that common report period. This fixed
path illustrates parameter sensitivity; it does not select a preferred span.
**Producer:** [span sensitivity](../tools/docs_analytics/span_sensitivity.py).

## Portfolio objectives

Minimum variance, maximum diversification and equal risk budgets use the same covariance,
assets, weight constraints and trade dates. Expected-return estimation is outside this
comparison; it covers three objectives driven by covariance.

[![Synthetic net growth and trading costs for minimum variance, maximum diversification and equal risk budgets](../examples/figures/multi_optimisers_backtest.PNG)](../examples/figures/multi_optimisers_backtest.PNG)

**Sample:** 1 April 2015–31 December 2025. The 35% cap can prevent equal target risk budgets
from producing equal risk contributions at the decision date. Cost sums include entry. This single
synthetic path is not evidence of an objective's expected market performance.
**Producer:** [optimiser comparison](../tools/docs_analytics/optimiser_comparison.py).
**Related methodology:** [optimization guide](optimization_module_readme.md) and
[risk budgeting](risk_budgeting.md).

## Covariance estimators

Six estimates feed a common minimum-variance construction: EWMA, Lasso and Group Lasso,
each with its specified volatility-normalized variant. EWMA normalization acts on asset-return
covariance; factor variants normalize factor covariance only. Group Lasso uses four fixed
asset pairs, not clusters inferred from the known loadings.

[![Synthetic minimum-variance backtests and last-decision covariance errors for six estimators](../examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)](../examples/figures/MinVariance_multi_covar_estimator_backtest.PNG)

**Backtest:** 3 January 2024–31 December 2025. **Error snapshot:** 1 October 2025.
The lower panel measures relative Frobenius error against the known simulated covariance:
the Euclidean size of all estimation errors divided by the size of the true matrix.
Estimated weekly covariance is annualized by 52; simulated daily truth by 260.
Known truth is used for descriptive evaluation and never supplied to portfolio estimation.

All cases fit only data available at their shared decision dates. The simulation has constant
loadings/covariance, no return premium and one fixed path. Closely overlapping performance
curves and one-date covariance errors do not establish an estimator ranking.
**Producer:** [covariance comparison](../tools/docs_analytics/covariance_comparison.py).
**Configuration:** [analytics registry](../tools/docs_analytics/registry.json).

## Mixed-frequency data

A teaching exhibit of the [mixed-frequency data](mixed_frequency_data.md) page. At one
formation date, the monthly assets' classic momentum uses 12 monthly returns ending one month
earlier, and the quarterly asset uses 4 quarterly returns ending one quarter earlier; the
quarterly signal is carried unchanged between quarter ends.

![Left: the observations in each asset's classic momentum window at 31 December 2023. Right: raw classic momentum, with the quarterly asset flat between quarter ends.](images/mixed_frequency_grids.png)

**Sample:** deterministic synthetic panel of two monthly series and quarter-end NAVs, December
2018 to December 2024; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/mixed_frequency_data.py`](../examples/docs/mixed_frequency_data.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Incomplete histories

A teaching exhibit of the [incomplete histories](incomplete_histories.md) page. A price missing
between rebalancings lowers NAV only until the price returns; the same gap on a rebalance date
clears the held units without proceeds; a missing opening price leaves that allocation in cash.

![Left: NAV of the hold-through-gap, rebalance-on-gap and missing-opening-price paths over four business days. Right: the rebalance-on-gap path's holdings, with the gapped position replaced by cash.](images/incomplete_histories_price_gaps.png)

**Sample:** two synthetic assets over four business days in January 2024, with one missing
price; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/incomplete_histories.py`](../examples/docs/incomplete_histories.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Universe data and unsmoothing

A teaching exhibit of the [universe data and unsmoothing](universe_data_and_unsmoothing.md) page.
Appraisal smoothing with a weight of 0.6 on the previous reported return halves the reported
volatility of private equity and makes its returns autocorrelated. Unsmoothing lowers the lag-1
autocorrelation from 0.61 to -0.04, against -0.02 for the true returns, and raises the annualised
volatility from 10.2% to 20.5%, against 20.4%.

![Left: cumulative log returns of reported, unsmoothed and simulated true private equity. Middle:
their lag-1 autocorrelations. Right: their annualised volatilities.](images/unsmoothing_autocorrelation.png)

**Sample:** a seeded synthetic quarterly panel of four assets from December 1989 to December 2024,
seed 19, whose private-equity column is appraisal-smoothed; the simulated true returns are known.
**Producer:** the `exhibit` function of
[`examples/docs/universe_data_and_unsmoothing.py`](../examples/docs/universe_data_and_unsmoothing.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Factor covariance with HCGL

A teaching exhibit of the [factor covariance with HCGL](factor_covariance_hcgl.md) page. Orthogonal
and empirical residuals give every asset the same systematic and residual variance; they differ
in the residual covariances. Three private assets share a shock that the factors do not span, and
an equal-weight portfolio of them has a residual variance of 0.00019 with orthogonal residuals
and 0.00040 when the empirical residual correlations enter with full weight.

![Left: stacked bars of systematic and residual annual variance for eight assets, identical under
orthogonal and empirical residuals. Right: residual variance of equal weights in the three private
assets, rising from 0.00019 with orthogonal residuals to 0.00040 as residual covariances are
added.](images/factor_covariance_variance_split.png)

**Sample:** synthetic month ends from December 2004 to December 2024, seed 5: four factors, five
monthly and three quarterly assets with known loadings and a shock common to the private assets;
HCGL fit at 31 December 2023. **Producer:** the `exhibit` function of
[`examples/docs/factor_covariance_hcgl.py`](../examples/docs/factor_covariance_hcgl.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Ex-ante risk contributions

A teaching exhibit of the [ex-ante risk contributions](portfolio_risk_analytics.md) page. On one
covariance, equal weights put 42% of the risk in emerging-market equity; minimum variance has
risk shares equal to its weights; equal risk contribution gives each asset 20% of the risk with
45% of the capital in government bonds; maximum diversification holds 67% of the capital in
government bonds and 41% of the risk.

![Left: stacked capital weights of equal weight, minimum variance, equal risk contribution and
maximum diversification on one five-asset covariance. Right: the Euler risk shares of the same
portfolios.](images/risk_contributions_vs_weights.png)

**Sample:** the page's fixed five-asset, three-factor covariance snapshot; no simulation.
**Producer:** the `exhibit` function of
[`examples/docs/portfolio_risk_analytics.py`](../examples/docs/portfolio_risk_analytics.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Alpha signals

A teaching exhibit of the [alpha signals](alphas_module_readme.md) page. Scored across all nine
assets, twelve-month momentum ranks the equity-like assets first; scored within each cluster,
each group is centred on its own mean, so the strongest bond-like asset rises from fifth to
second and the weakest equity-like asset falls to last.

![Left: cross-sectional momentum scores of nine assets, with each cluster's mean. Right: the same signal scored within each cluster, with the rank of each asset.](images/alpha_scoring_cross_section_vs_cluster.png)

**Sample:** nine synthetic month-end price paths with constant log drifts, five equity-like and
four bond-like, October 2023 to December 2024; no simulation. **Producer:** the `exhibit`
function of [`examples/docs/alphas_module_readme.py`](../examples/docs/alphas_module_readme.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Signal diagnostics

A teaching exhibit of the [signal diagnostics and alpha-rank portfolios](signal_diagnostics_and_profiling.md) page. A
synthetic score whose population rank information coefficient is 0.0955 sorts 500 assets into
quintile portfolios whose values fan out in rank order, while the monthly rank IC scatters widely
around its population value.

![Left: the values of the five quintile portfolios sorted by the score, on a log scale. Right: the
monthly rank IC with its 12-month mean and the population value.](images/alpha_rank_quantiles.png)

**Sample:** 500 synthetic assets with 240 month-end returns from January 2006 to December 2025,
seed 11. **Producer:** the `exhibit` function of
[`examples/docs/signal_diagnostics_and_profiling.py`](../examples/docs/signal_diagnostics_and_profiling.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Risk budgeting

A teaching exhibit of the [risk budgeting](risk_budgeting.md) page. With slack caps, each
asset's share of portfolio risk equals its budget of 50%, 30% or 20%; capping Equity at 25%
leaves no asset at its budget, and the released risk goes to the free assets in proportion to
their capital weights, not their budgets.

![Left: target budgets and achieved risk shares with slack caps and with Equity capped at 25%. Right: the capital weights of both solves.](images/risk_budgeting_contributions.png)

**Sample:** the page's fixed three-asset covariance, solved with slack 80% caps and with Equity
capped at 25%; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/risk_budgeting.py`](../examples/docs/risk_budgeting.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Implied risk budgets

A teaching exhibit of the [implied risk budgets](implied_risk_budgets.md) page. As the stock-bond
correlation drifts from -0.7 to 0.4 over twelve quarter ends, the implied risk budget of Bonds in
a 40/45/15 target rises from -8% to 25%; on the first two dates Bonds hedge the portfolio and no
non-negative budget reproduces the target. One static budget vector of about 70.6%, 14.5% and
14.9% reproduces the target only on average: its forward weight in Bonds moves from 59% to 32%.

![Left: the implied risk budgets of the target at each quarter end, with the fitted static budgets
dashed and the dates of negative Bonds budgets shaded. Right: the forward weights under the fitted
budgets, their running averages and the target weights.](images/implied_budgets_round_trip.png)

**Sample:** three assets with fixed volatilities and a stock-bond correlation drifting from -0.7
to 0.4 over twelve quarter ends from 31 March 2023; no simulation. **Producer:** the `exhibit`
function of [`examples/docs/implied_risk_budgets.py`](../examples/docs/implied_risk_budgets.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Hierarchical risk parity

A teaching exhibit of the
[hierarchical risk parity and cluster risk budgets](hierarchical_risk_parity_and_cluster_budgets.md)
page. On a universe of four government bonds, two equity markets and two real assets, HRP puts
91% of the capital and 90% of the risk in the bonds; equal risk contribution puts 76% of the
capital there for 50% of the risk, and equal cluster budgets 69% of the capital for a third.

![Left: the single-linkage tree of the eight assets, with the first HRP split between the bonds
and the rest. Right: capital and risk shares by block for HRP, equal risk contribution and equal
cluster budgets.](images/hrp_vs_erc_weights.png)

**Sample:** a fixed eight-asset universe of four government bonds, two equity markets and two real
assets with block correlations; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/hierarchical_risk_parity_and_cluster_budgets.py`](../examples/docs/hierarchical_risk_parity_and_cluster_budgets.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Maximum diversification

A teaching exhibit of the [maximum diversification](maximum_diversification.md) page. The
SLSQP weights of the package and the minimum-variance weights of the correlation matrix,
rescaled by inverse volatility and computed by CVXPY, agree for a stylised six-asset universe;
every held asset has correlation 0.570 with the portfolio, the inverse of its diversification
ratio, and the excluded credit asset has 0.70.

![Left: SLSQP and CVXPY routes give the same weights for six assets, with government bonds at 72% and credit not held. Right: held assets have correlation 0.570 with the portfolio; excluded credit has 0.70.](images/max_diversification_identity.png)

**Sample:** fixed volatilities and correlations; no simulation. **Producer:** the `exhibit`
function of [`examples/docs/maximum_diversification.py`](../examples/docs/maximum_diversification.py),
run by [the teaching-exhibit tool](../tools/docs_analytics/teaching.py). **Configuration:**
[teaching registry](../tools/docs_analytics/teaching.json).

## Mean-variance objectives

A teaching exhibit of the [mean-variance objectives](mean_variance_objectives.md) page. With a
full-investment budget, minimum variance, quadratic utility and maximum Sharpe lie on one
frontier: the utility portfolio mixes the minimum-variance and tangency portfolios, and a risk
aversion of 6.96 returns the tangency portfolio, whose Sharpe ratio is 0.416.

![Left: expected excess return against volatility, with the frontier, the minimum-variance and
maximum-Sharpe portfolios and utility portfolios at four risk aversions. Right: the weights of the
minimum-variance, maximum-Sharpe and utility portfolios.](images/efficient_frontier_objectives.png)

**Sample:** fixed expected excess returns, volatilities and correlations of five assets; no
simulation. **Producer:** the `exhibit` function of
[`examples/docs/mean_variance_objectives.py`](../examples/docs/mean_variance_objectives.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Target return and target volatility

A teaching exhibit of the [strategic allocation](strategic_allocation_targets.md) page. Minimum
variance at a target return, maximum return at a target volatility and the utility form at a
penalty weight trace one frontier: a 5% volatility target, a 4.81% return target and a penalty
weight of 4.46 return the same portfolio.

![Left: expected return against volatility, with the frontier and the solutions of the three
routes. Right: the volatility of the utility solution against the penalty weight, crossing the 5%
target at the shadow price.](images/saa_target_duality.png)

**Sample:** the packaged 19-instrument monthly fixture, with an EWMA covariance (span 36) at
31 December 2025 and equal-Sharpe expected returns; no simulation. **Producer:** the `exhibit`
function of
[`examples/docs/strategic_allocation_targets.py`](../examples/docs/strategic_allocation_targets.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## CARA utility under Gaussian mixtures

A teaching exhibit of the [CARA utility under Gaussian mixtures](cara_gaussian_mixture.md) page.
A crash component lowers the allocation to crypto at every risk aversion: at a risk aversion of
5, the three-component mixture holds 6.3% in crypto against 7.2% for the Gaussian with the
mixture's own mean and covariance.

![Left: the weight of crypto against risk aversion for one and three mixture components. Right:
the weights of bonds, equities and crypto at a risk aversion of 5.](images/cara_mixture_allocation.png)

**Sample:** fixed annual parameters of a stylised three-asset, three-component mixture; no
simulation. **Producer:** the `exhibit` function of
[`examples/docs/cara_gaussian_mixture.py`](../examples/docs/cara_gaussian_mixture.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Minimum tracking error

A teaching exhibit of the [minimum tracking error](minimum_tracking_error.md) page. Starting
from the benchmark itself, excluding one asset, capping another, limiting a group and limiting
turnover each add ex-ante tracking error, from 0% to 2.06% a year; the right panel shows where
the covariance sends the displaced weight at two of the steps.

![Left: ex-ante tracking error after each cumulative constraint. Right: active weights at the exclusion step and the turnover step.](images/tracking_error_constraint_cost.png)

**Sample:** fixed volatilities and correlations of a stylised five-asset universe with a 60/40
benchmark and fixed current holdings; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/minimum_tracking_error.py`](../examples/docs/minimum_tracking_error.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Alpha over tracking error

A teaching exhibit of the [alpha over tracking error](alpha_over_tracking_error.md) page. Under a 2%
tracking-error budget the long-only active weights equal the closed form, and the information
ratio stays at 0.336 until a weight bound binds at a 3.82% budget; without weight bounds it does
not fall.

![Left: long-only active weights at a 2% budget against the closed form. Right: the information
ratio against the tracking-error budget, with and without long-only bounds.](images/alpha_over_te_active_weights.png)

**Sample:** fixed volatilities, correlations, benchmark and alphas of a stylised six-asset
universe; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/alpha_over_tracking_error.py`](../examples/docs/alpha_over_tracking_error.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Overlay tail floor

A teaching exhibit of the [overlay tail floor](overlay_tail_floor.md) page. A floor changes
nothing until it exceeds the unconstrained sleeve's floor exposure; tighter floors bind at
equality, move the sleeve from the carry overlays to the defensive ones, and near the reachable
maximum leave it all in one defensive overlay, at a lower model return.

![Left: the overlay sleeve by floor level, stacked by overlay. Right: the portfolio's model volatility and expected excess return against the floor.](images/overlay_floor_allocation.png)

**Sample:** the page's synthetic core and four overlays, solved at 90 floors from -0.08 to
0.009; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/overlay_tail_floor.py`](../examples/docs/overlay_tail_floor.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Portfolio constraints

A teaching exhibit of the [portfolio constraints](constraints.md) page. A hard 2% tracking-error
row holds the solve at the limit; the same limit as a penalty lets tracking error exceed it by an
amount that shrinks as the penalty weight grows, and only the row's shadow price, 3.33 here,
reproduces the forced solve.

![Left: ex-ante tracking error against the 2% limit for six penalty weights and for the hard row. Right: the active expected return of the same solves.](images/constraint_enforcement_hard_vs_soft.png)

**Sample:** the page's synthetic three-asset example with a 45/40/15 benchmark, solved with a
hard tracking-error row and at six penalty weights; no simulation. **Producer:** the `exhibit`
function of [`examples/docs/constraints.py`](../examples/docs/constraints.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Solver numerics and outcomes

A teaching exhibit of the [solver numerics and outcomes](solver_numerics_and_outcomes.md) page.
As two private proxies approach collinearity, the smallest eigenvalue of the covariance falls
toward zero and the condition number grows as the inverse of the correlation gap. The eigenvalue
floor raises only the smallest eigenvalue, to 1e-10, and caps the condition number at the
largest eigenvalue over the floor; the other five eigenvalues are unchanged.

![Left: the six eigenvalues of the covariance with the private pair nearly collinear, on a log
scale, with the smallest raised to the floor. Right: the condition number against one minus the
correlation of the pair, before and after the floor.](images/covariance_conditioning.png)

**Sample:** fixed volatilities and correlations of a six-asset universe whose private pair
approaches a duplicate proxy; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/solver_numerics_and_outcomes.py`](../examples/docs/solver_numerics_and_outcomes.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Rolling backtests

A teaching exhibit of the [rolling backtests](rolling_backtests.md) page. Between quarterly
trades, held weights drift with prices away from the targets in force; each trade, one
observation after its decision date, resets them, and the gap peaks the day before a trade.

![Left: target and held weights of three assets through 2020 and 2021, with the quarterly trades. Right: the sum of absolute differences between held and target weights.](images/target_vs_drifted_weights.png)

**Sample:** a seeded synthetic business-day panel of three assets with weekly EWMA covariances,
minimum-variance targets capped at 60% and eight quarterly trades. **Producer:** the `exhibit`
function of [`examples/docs/rolling_backtests.py`](../examples/docs/rolling_backtests.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Turnover and transaction costs

A teaching exhibit of the [turnover and transaction costs](turnover_and_transaction_costs.md)
page. Raising `turnover_utility_weight` moves the solution along a frontier from the benchmark,
with 30% turnover, to the current holdings, with none; trading stops at a finite weight, 0.3851
in this example, given by the spread of the marginal active risks.

![Left: full L1 turnover against ex-ante tracking error, one point per labelled penalty weight. Right: turnover against the penalty weight, with the no-trade threshold.](images/turnover_penalty_tradeoff.png)

**Sample:** a synthetic four-asset problem with a 50/20/25/5 benchmark and 40/25/20/15
holdings, solved at 51 penalty weights; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/turnover_and_transaction_costs.py`](../examples/docs/turnover_and_transaction_costs.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## ROSAA layers

A teaching exhibit of the [ROSAA case study](app_rosaa_multi_asset_allocation.md). On a
synthetic panel with known factor loadings, the strategic allocation meets its risk budgets
with capital weights that differ from them, and the tactical tilts against it follow the
alphas at a 3% ex-ante tracking error at every quarter end.

![Left: risk budgets and the strategic weights that meet them for eight asset classes. Right: tactical active weights against alpha scores at 43 quarter ends.](images/rosaa_saa_taa_layers.png)

**Sample:** synthetic monthly panel of three factors and eight asset classes, December 2004 to
June 2025, seed 11; quarter ends from December 2014. **Producer:** the `exhibit` function of
[`examples/docs/app_rosaa_multi_asset_allocation.py`](../examples/docs/app_rosaa_multi_asset_allocation.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Cryptocurrency allocation

A paper-derived exhibit of the [cryptocurrencies in diversified portfolios](app_crypto_allocation.md) case
study, run with the current API on the frozen 2023 panel of the paper. In the all-alternatives
template with BTC, the median BTC weight is 5.3% under equal risk contribution, 6.3% under
maximum diversification and 10.9% under maximum Sharpe, and CARA utility with three mixture
components holds the most.

![Two panels of BTC weights at each quarter end from March 2016 to June 2023 for four methods,
with the median marked, in the all-alternatives and the balanced templates.](images/crypto_allocation_by_method.png)

**Sample:** the tracked price panel of the cryptocurrency paper, ETF-derived columns only (BTC,
private equity, real estate, commodities, gold and a 60/40 proxy), monthly log returns and 30
quarter ends from 31 March 2016 to 30 June 2023. **Producer:** the `exhibit` function of
[`examples/docs/app_crypto_allocation.py`](../examples/docs/app_crypto_allocation.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Capital market assumptions

A teaching exhibit of the [capital market assumptions to strategic allocation](app_cma_strategic_allocation.md)
case study. One loading matrix gives the factor-implied part of each CMA; two one-point residual
adjustments, small beside the factor parts, turn a sold-out hedge-fund position into a 3.4-point
overweight at a 1% tracking-error budget.

![Left: the factor-implied and residual parts of eight synthetic CMAs. Right: active weights against
the benchmark with and without the residual adjustments.](images/cma_decomposition_saa.png)

**Sample:** fixed synthetic loadings, factor premia and residual adjustments of eight asset classes
and four factors; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/app_cma_strategic_allocation.py`](../examples/docs/app_cma_strategic_allocation.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Stress testing with options

A teaching exhibit of the [stress testing with options](stress_testing_with_options.md) page.
For a book of stocks with short calls and puts, each Equity scenario's P&L splits into a delta
part, linear in the moves, and a gamma part from full option repricing; the short options lose
against their delta line in both directions.

![Left: delta part, gamma part and total P&L of the book for Equity moves of -30% to +30%. Right: the short options' full-repricing P&L against their delta line, with the gamma gap shaded.](images/option_stress_scenarios.png)

**Sample:** an offline three-stock book with a fixed three-factor model and conditionally
completed Equity moves; no simulation. **Producer:** the `exhibit` function of
[`examples/docs/stress_testing_with_options.py`](../examples/docs/stress_testing_with_options.py).
**Configuration:** [teaching registry](../tools/docs_analytics/teaching.json).

## Reproduce and update

From a source checkout with the documented contributor environment and C-local setup, one
command regenerates all six previews and their supporting tables:

~~~console
python -m tools.docs_analytics.run --all --output-root <new-C-local-bundle>
~~~

Use a new directory below `AGENT_LOCAL_ROOT`, outside OneDrive and the source tree.
The runner executes offline and records the imported environment and effective source bytes.
Regeneration retains the fixed sample and seed; it does not fetch current market prices.
Distribution versions alone cannot identify uncommitted source edits.

Follow the [generation and publication workflow](documentation_standard.md#analytical-conventions-and-figures)
to compare repeated runs, validate the bundle, review each preview at full resolution and
article width, and publish the six images with their shared provenance record. Full tables
and intermediate output remain C-local. Preview paths are stable; the
[provenance file](../examples/figures/analytics_manifest.json) identifies their reviewed generation.

Teaching exhibits are drawn by the canonical scripts of their pages and have their own
[manifest](images/analytics_manifest.json), which records the script, parameters, checks and
software versions of each image:

~~~console
python -m tools.docs_analytics.teaching --all --output-root <new-C-local-bundle>
python -m tools.docs_analytics.teaching --verify
~~~

## See also

- [Documentation standard](documentation_standard.md): conventions, producer details and refresh workflow.
- [Constraints](constraints.md): units, alignment and backend support.
- [Software design](software_design.md): construction, estimation and analytics ownership.

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff),
  for simulation, performance and risk analytics.
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff),
  for sparse factor-model fitting.
