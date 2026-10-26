---
myst:
  html_meta:
    description: Yahoo factors, FM/FL risk and an OP active ETF backtest against a fixed benchmark.
---

# From market data to portfolio: the Sepp OSS stack

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

This repository-only example prepares a public USD multi-asset study: 40 Yahoo price series,
36 investable ETFs, 11 volatility-targeted excess-return factors, and a fixed monthly benchmark.
FM owns factor construction and rolling covariance orchestration; FL estimates loadings,
clusters and residual dependence; QIS supplies return, risk and holdings-accounting kernels.
OP generates public momentum alpha scores and maximises active alpha under a tracking-error
budget. QIS executes the portfolio and fixed benchmark, including traded-notional costs, and
produces performance, allocation, turnover and tracking-error factsheets.

## Run

Run from the OP checkout in a separate environment containing `optimalportfolios[data]>=9.1.1` and
`factormodels>=0.3.0`, `factorlasso>=1.0.6` and `qis>=5.44.3`.
The example is excluded from the OP wheel. Production fitting uses MOSEK and requires its
Python package and a valid license. The public alternative explicitly selects CLARABEL,
which is supplied with CVXPY; there is no automatic solver fallback.
Use an explicit exclusive end date for a reproducible study cut:

```console
python -m examples.oss_portfolio_pipeline --end 2026-10-24
python -m examples.oss_portfolio_pipeline --end 2026-10-24 --solver CLARABEL
```

On the maintainer's Windows host, use the clean external environment rather than changing the
frozen historical OP environment. Before manual Python commands, enter the governed workspace:

```powershell
. "$env:USERPROFILE\OneDrive\analytics\my_github\ArturSepp\scripts\repo_governance\Enter-AgentRepo.ps1"
$pipelinePython = 'C:\Python\OptimalPortfoliosOssPipeline312\Scripts\python.exe'
& $pipelinePython -m examples.oss_portfolio_pipeline --end 2026-10-24
```

`--cache-dir`, `--output-dir` and `--refresh` control the external local data bundle. Defaults
honour `AGENT_LOCAL_ROOT`; otherwise they use the operating system's user cache location.
Checkout and OneDrive output paths are rejected, and Windows generated state stays on C:.
No raw Yahoo data or generated analyses belong in this example directory.
Use `--inputs-only` to run just the download, factor construction and benchmark stage.
Use `--risk-only` to stop after rolling covariance estimation. The default runs all four
stages, ending with QIS reporting and model-layer attribution. `--no-reports` saves the numerical
tables without rendering figures; `--no-attribution` skips the standalone layer portfolios.
`--tracking-error 0.03`, `--transaction-cost-bps 5` and `--implementation-lag 0`
expose the principal backtest assumptions; `BacktestConfig` carries the complete public policy.

The download uses explicit `auto_adjust=True`, daily prices, no automatic repairs, and an
exclusive end. Missing, nonpositive or nonfinite quotes fail with a ticker/date message;
there is no backward filling or implicit redistribution of missing factor legs. The external
cache records request options, retrieval time, package version and a content checksum. Reusing
a request reuses its validated snapshot; `--refresh` deliberately obtains a new snapshot.

## Factor construction

The categories are adapted from Figure 4 of Deutsche Bank Research, *Keep the Alpha, Cut the
Risk: Macro-Factor and PC Neutralization in Equity Portfolios* (Quantcraft, 9 October 2026).
The basket definitions below are this example's public ETF adaptation, not DB's original indices
or a reproduction of the paper's results. They do not use a proprietary factor catalogue.

| Factor | Signed strategic holdings | Annual volatility target |
|---|---|---:|
| Energy | +1 DBE, -1 BIL | 15% |
| Agriculture | +1 DBA, -1 BIL | 15% |
| Base metals | +1 DBB, -1 BIL | 15% |
| Precious metals | +1 DBP, -1 BIL | 15% |
| DM equity | +1 URTH, -1 BIL | 15% |
| EM vs DM equity | +1 EEM, -1 URTH | 15% |
| Nominal vs TIPS | +1 GOVT, -1 TIP | 7.5% |
| Global government bonds | +0.5 GOVT, +0.5 IGOV, -1 BIL | 7.5% |
| USD vs DM | +1 UUP, -1 BIL | 7.5% |
| EM FX stress vs DM | +2 BIL, -1 CEW, -1 UUP | 7.5% |
| HY vs IG credit | +1 HYG, -1 LQD | 5% |

Every basket has zero net capital exposure at each rebalance. A long-only risk exposure is
financed by short BIL; a long/short spread finances itself. The two BIL units in the EM FX
basket finance both shorts. These are excess-P&L portfolios with a zero-interest bookkeeping
cash balance. **Do not subtract BIL again from the output factor returns.** BIL price levels
are not annual cash-rate quotes and are never passed to a rate-based excess-return function.

FM estimates the basket's risk from weekly Wednesday log returns, with a 156-week EWMA span
and a causal 13-observation initial covariance sample. There is no exposure during warm-up.
The latest weekly target known by the actual month-end trading session determines that month's
holdings. QIS holds units until the following rebalance. New holdings affect only subsequent
price intervals. No realised-volatility equality is promised: these are ex-ante targets, and
there is no leverage cap in this factor construction.

`compute_factor_nav` labels targets at calendar month-end, which can trade next month when
the date is a holiday. To implement the study's actual-session convention, this example calls
FM's public `compute_volatility_targeted_portfolio`, selects its latest available target at
each observed month-end, and passes dated weights to `qis.backtest_model_portfolio`. There is
no duplicate volatility estimator or custom backtesting engine.

The output daily returns are **log returns of the financed, held-unit factor NAVs**. They are
not weighted sums of constituent log returns. Rebalancing and financing are therefore explicit
rather than hidden inside a statistical return contrast.

Construction starts in December 2014. Displayed factors start at the last observation on or
before 31 December 2017 (29 December for the Yahoo panel), rebased to 100. The first displayed
return is missing because it is the valuation anchor, not an earned return. A partial final
month does not create an extra rebalance.

## Universe and benchmark

| Sleeve | Investable Yahoo tickers |
|---|---|
| Equity markets and regions | SPY, QQQ, IWM, EFA, VGK, EWJ, EWA, EEM, AAXJ, MCHI, INDA |
| Equity styles | QUAL, MTUM, USMV, VLUE |
| Government bonds and TIPS | SHY, IEF, TLT, GOVT, IGOV, TIP |
| Corporate credit and EM debt | VCSH, LQD, HYG, SHYG, EMB, EMLC |
| Listed property and infrastructure | VNQ, VNQI, IGF |
| Commodities | DBC, DBE, DBA, DBB, GLD |
| Cash | BIL |

URTH, DBP, UUP and CEW are additional factor sources. Broad funds and regional/style alternatives
overlap intentionally. Listed property and infrastructure have a separate reporting sleeve but
are classified as equity-like in `economic_group` for later constraint construction.

| Benchmark sleeve | Constant target holdings | Weight |
|---|---|---:|
| Equities | SPY 25%, EFA 15%, EEM 10% | 50% |
| Government and inflation-linked bonds | GOVT 15%, IGOV 5%, TIP 5% | 25% |
| Credit and EM debt | LQD 5%, HYG 3%, EMB 2% | 10% |
| Commodities | GLD 5%, DBC 5% | 10% |
| Cash | BIL 5% | 5% |

The benchmark begins at the 31 December 2019 close and resets to these targets on the last US
trading session of each completed calendar month. Its weights drift between rebalances. The
other 24 investable ETFs have explicit zero benchmark weights. Benchmark and factor NAVs are
gross of trading and borrowing costs; ETF fees are already embedded in the adjusted histories.

## Rolling factor covariance

`covariance.estimate_covariances(inputs, config)` calls FM's `FactorCovarEstimator.fit_rolling`.
The numerical configuration follows the ROSAA **TAA** production specification:

| Setting | Value |
|---|---|
| Update schedule | Completed calendar ME, starting 31 December 2019 |
| Loading estimator | FL factor-cluster group lasso (FCGL), normalized group penalty |
| Beta and clustering EWMA span | 36 monthly observations; expanding input history |
| Regularisation | 0.00001; pure group penalty, L1 weight zero; weight-sum loss |
| Regression warm-up and centring | 12 observations; causal EWMA demeaning in FL |
| Clusters | Ward, Pearson 1-minus-correlation distance, cutoff fraction 0.60 |
| Cluster smoothing / factor references | None / disabled |
| Signs | Automatic; t threshold 1.0; adaptive weights with floor 0.5; date variance |
| Priors | Highest-R-squared OLS selection; one-standard-error bound setting; HAC 3 |
| Factor covariance | Wednesday weekly log returns; EWMA span 52; annualisation 52 |
| Residual covariance | Empirical correlations, full weight; monthly EWMA span 36 |
| Residual annualisation | 12; full residual variance weight |
| Production solver | MOSEK; explicit CLARABEL override available |

The full parameter dictionaries, including the retained 12-quarter span and HAC-1 settings,
are saved in `covariance_manifest.json`. The 35 risky ETFs use the monthly bucket in this study.
Automatic priors and signs are estimated at each cutoff from these public factors. No MATF
expert selections, sign tables or private module imports enter the example; matching the
production estimator configuration does not reproduce the proprietary universe's estimates.

Monthly asset observations are QIS log returns of **adjusted ETF price divided by adjusted
BIL price**, equivalent to subtracting the matching BIL log return once. BIL therefore has
zero excess risk. It is excluded from pooled-prior regression, and FM restores its zero-risk
row in the complete 36-asset output universe. The factor NAVs are already financed excess
portfolios and receive no additional cash subtraction. Regressions start with the December
2017 level anchor and use
24 monthly returns at the first December 2019 fit; 36 months is an EWMA span, not a minimum
fixed window. There is no full-sample demeaning or additional volatility normalization.

FM keeps its production factor-covariance convention: sample completed Wednesdays, remove
the first price-difference row, subtract the contemporaneous EWMA mean, remove the first
zero deviation, and accumulate the zero-seeded covariance. Residual assembly remains in FM/FL.
Covariances are annualised log-return covariances. The singular cash row is intentional;
the example does not add a ridge or alter the estimator to force positive definiteness.

The native FM model uses **calendar month-end** labels so FL's empirical residual grid is
complete. Monthly prices are taken from the last observed trading session on or before each
label. `result.schedule` records both dates; `result.covariances` exposes the asset matrices
under the corresponding trading-session dates for OP. For example, the 29 February 2020 model
uses the 28 February close. Such a matrix may determine holdings for subsequent returns only.
The native FM model and its `to_risk_model()` retain calendar labels; use the explicit mapping
when aligning that model with an actual-session optimisation schedule. A partial final month
does not generate an extra estimate.

## Active portfolio backtest

`backtest.run_backtest(inputs, risk, config)` maximises the alpha score of the active holdings
subject to a hard annual tracking-error budget against the fixed strategic target. The OP API
name is "maximise alpha over TRE": here this means a **constrained alpha maximisation**,
not a numerical maximisation of an alpha/TE ratio.

The public signal is OP's 12-month EWMA risk-adjusted momentum with a 13-month volatility span,
no short filter and `MeanAdjType.NONE`. QIS reconstructs monthly NAVs from BIL-relative excess
log returns using the complete downloaded ETF history, beginning in December 2014. This common
signal warm-up supports both 12- and 24-month spans at inception; factor/covariance estimation
still begins with the December 2017 anchor. FM extracts each cutoff's clusters and delegates scoring
to FL. Clusters with at most five members use global cross-sectional moments; larger clusters
use their own moments. There is no stability pooling. BIL is excluded from signal scoring and
assigned zero alpha. Scores are dimensionless rankings, not calibrated expected annual returns.
They are an illustrative public signal choice, not the private ROSAA product signal routing.

| Public example policy | Default |
|---|---|
| Inception / decisions | 31 December 2019 / actual month-end trading sessions |
| Annual ex-ante tracking-error cap | 3% |
| Net investment / shorting | 100% / long-only |
| Maximum allocation to one ETF | 30% |
| Economic sleeve allocation | Fixed benchmark sleeve weight ±10 percentage points, clipped to 0–100% |
| Turnover cap after inception | 50% two-way L1 weight change; approximately 25% one-way |
| Transaction cost | 5 bp per unit of bought or sold notional, for both portfolios |
| Execution lag | Zero observations: decision and trade at the same model close |

The economic sleeves are Equities (including styles and listed real assets), Government bonds,
Credit and EM debt, Commodities, and Cash. Initial investment from cash is exempt from the
turnover cap. Later turnover uses the actual QIS holdings before the new trade, including drift
and previous costs. These policy values are configurable assumptions fixed before evaluation;
they have not been selected by optimising this backtest's performance.

The driver is `op.backtest_with_qis_state`, with `op.cvx_maximise_alpha_over_tre` as its decision
callback. This keeps BIL in the optimisation even though its excess covariance row is zero.
The ordinary rolling wrapper filters zero-variance assets, so it would remove this cash asset.
No covariance ridge is inserted to bypass that filtering. Each accepted solution retains its
OP constraint residuals and status. A rejected or noncompliant solve stops the example with
its decision date and reason; fallback holdings are not reported as a successful solve.
For QIS's stricter execution budget, solver round-off is reconciled by clipping only tiny
negative values and normalising the weight sum. This correction is limited to one part per
million in total weight change, recorded in diagnostics, and followed by a fresh OP constraint
audit at a tolerance of one part per million. Larger corrections or failed audits stop the run.

At zero lag, weights decided using the month-end close earn only **subsequent** price returns.
This is a model-close execution assumption; the `--implementation-lag 1` sensitivity trades at
the next observed close for both portfolios. Risk constraints apply at the decision date and
are not promises about realised TE or risk at a later execution price. A terminal model date
without a subsequent post-execution observation produces no backtested trade.

QIS holds units between trades, credits total adjusted ETF returns and charges costs on actual
traded notional, including the initial purchase. Costs debit its currency cash balance; fully
invested targets can therefore leave a small negative cash balance. No additional borrowing
fee or cash funding rate is applied in this example. Fund expenses are embedded in the adjusted
price histories. The saved gross paths replay the **same targets** with zero transaction costs;
they do not reoptimise the gross policy. The benchmark retains the agreed constant ME targets
and uses the same transaction-cost and implementation-lag assumptions as the active portfolio.

Ex-ante TE is calculated with `qis.RiskModel`. Realised TE uses QIS's 36-month EWMA of completed
monthly simple-return differences. Whole-sample TE and information ratio use the same completed
monthly simple-return convention; fewer than three monthly observations report missing summary
statistics. Partial final months remain in daily NAV output but are excluded from these monthly
risk statistics. These risk measures use different conventions from the model's annualised
log-return covariance and need not equal the optimisation cap. Opening trade costs appear in
the inception NAV; return statistics measured from that NAV exclude the opening debit.

## QIS reporting

`reporting.save_reports(inputs, result, output_dir)` adds `reports/portfolio_report.pdf`,
five standalone PNGs and the numerical CSV tables behind the panels. The five pages show:

1. Active and benchmark NAVs, net and same-target gross, net drawdowns, and performance statistics.
2. Daily realised economic-sleeve allocations and the latest allocation snapshot.
3. All 36 ETFs: latest active targets, fixed benchmark targets, and both portfolios' latest holdings.
4. Monthly and trailing 12-month two-way turnover, plus monthly trading cost drag.
5. Ex-ante TE for targets versus fixed weights and holdings versus actual benchmark holdings,
   alongside realised net EWMA TE and whole-sample net/gross TE and information ratios.

All analytics and chart primitives come from QIS. `prepare_report_tables` exposes the tables
without rendering; `create_report_figures` returns named Matplotlib figures for interactive
use, which callers should close after use. Reporting copies display metadata and never changes
the backtest's names, holdings or results. No additional reporting extra or commercial plotting
service is needed.

Performance risk statistics use monthly **simple** returns; drawdowns use the daily NAV path.
The displayed Sharpe is QIS `SHARPE_RF0`: compound annual return divided by annualised volatility,
with a zero risk-free rate. Its numerator and denominator use completed monthly boundaries.
Displayed total and annualised returns instead use native observed NAV endpoints, including a
partial final month. Opening costs reduce inception NAV; return statistics start at that NAV.

Allocations retain QIS's actual, drifting instrument weights. A separate **Funding cash** row
completes the NAV accounting identity, including negative balances from costs; the **Cash**
sleeve is the BIL ETF and does not include that currency balance. Both actual and target weights
are displayed, with their different dates. ETF holdings are not renormalised for presentation.

Turnover is QIS `EXECUTED_NOTIONAL_NAV`: bought plus sold notional divided by same-day NAV.
The initial purchase is excluded at any execution lag. The trailing series requires twelve
complete months after the initial funding month. Cost tables include the initial purchase and
sum daily cost/NAV fractions; these sums are not compounded performance losses. Monthly trading
panels exclude the partial final month; the daily source tables retain it. Raw backtest turnover
CSVs keep QIS's default initial-row convention; the reporting tables apply the stated funding
exclusion explicitly, including when the first execution is delayed.

QIS `RiskModel` calculates both ex-ante series using the annualised log-return model. Realised
TE retains the backtest's completed-month simple-return differential and 36-month EWMA warm-up.
Observed Friday decisions and Saturday/Sunday month-end labels are joined within their own
calendar month, without forward shifting or filling missing realised estimates. The overlay is
a comparison of contemporaneous model risk with trailing realised risk, not a forecast-error
test. The target cap does not constrain realised TE or subsequent drifting holdings.

`reports/report_manifest.json` hashes every PDF, PNG and CSV, records dates, conventions and
QIS/plotting versions, and links to the exact `backtest_manifest.json` by SHA-256. Reports and
downloaded data remain in the external output bundle; only example source is intended for Git.

## Native factsheets and investment diagnostics

The reporting stage also calls QIS's three native generators:

- `generate_strategy_factsheet`: the baseline active portfolio, with economic-sleeve detail.
- `generate_strategy_benchmark_factsheet_plt`: the active portfolio versus the fixed benchmark.
- `generate_multi_portfolio_factsheet`: all supplied active configurations plus the benchmark.

All compared NAVs are net of actual trading costs. Native factsheets include their own monthly
benchmark-quantile regime panels, so no separate regime report is generated. Performance risk
uses monthly simple returns, compound zero-rate Sharpe and daily drawdowns. Rolling panels use
QIS's monthly presets; their titles state their windows. The recent comparison table covers up
to the last three years. Native turnover includes initial funding, unlike the funding-excluded
custom five-page report above. The displayed native instrument count can also differ from the
explicit non-cash, 1bp-threshold breadth convention below. These conventions are recorded in the
factsheet manifest. No rate download or alternative risk-model fit is requested by reporting.

Use the existing four-configuration flag for the complete comparison:

```console
python -m examples.oss_portfolio_pipeline --end 2026-10-24 --feature-attribution --stress-report
```

This compares 12- and 24-month momentum at 3% and 4.5% annual TE. Without feature attribution,
the native multi-strategy sheet compares the baseline active portfolio and the benchmark;
signal diagnostics then contain only the baseline momentum span. `--no-attribution` still
produces all three factsheet types and all three investment reports. `--no-reports` suppresses
their generation together with the existing PDF reports.

Three additional PDFs explain the investment decisions:

1. **Signal diagnostics:** raw momentum versus the cluster-adjusted score, for each supplied
   momentum span. QIS pairs the score known at the previous ME with subsequent 1-, 3- and 6-month
   BIL-relative log returns. Non-overlapping windows start with January 2020; incomplete windows
   are omitted. Score dates remain actual trading sessions, and exported pairs include their
   observed score date and forward-window end. The report shows rank-IC stability and horizon
   summaries, within-sleeve pooled ICs, predictive-return boxplots and per-asset regression
   slopes. Per-asset slope panels require at least twelve non-overlapping observations; shorter
   histories show an explicit insufficient-history panel. The pooled regression uses classical
   independent-pair standard errors; its significance
   labels and the IC t-statistic are descriptive, without HAC or multiple-testing correction.
2. **Portfolio breadth:** active targets versus benchmark targets at every decision date. QIS
   consumes the exact full FM asset covariance rather than estimating another covariance.
   BIL is explicitly excluded from every count; material positions have absolute weight above
   1bp. The report compares invested positions, effective capital positions, effective risk
   contributors, allocation efficiency and current concentration curves. Absolute Euler risk
   shares measure concentration and do not preserve the sign of hedging contributions.
3. **Brinson attribution:** economic-sleeve allocation, selection and interaction using daily
   prior holdings and QIS's BHB convention. Frongello linking reconciles to the difference of
   compounded strategy and benchmark returns. The gross page separates all three effects.
   The net page uses QIS's canonical selection-plus-interaction merge: costs assigned to
   near-empty sleeves otherwise produce unstable offsetting selection and interaction terms.
   Unmerged net effects remain in audit CSVs. A same-target gross/net bridge shows the differential
   trading-cost effect; it is not the sum of daily cost ratios. The first shared NAV is the
   baseline, so the initial purchase debit is already reflected in it.

The owning APIs are documented in [QIS signal diagnostics](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/signal_diagnostics.md),
[portfolio breadth](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/portfolio_breadth.md)
and [Brinson attribution](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/brinson_attribution.md).

Existing completed results can be reported without downloading, optimising or refitting:

```python +SKIP
from examples.oss_portfolio_pipeline.factsheets import save_native_factsheets
from examples.oss_portfolio_pipeline.diagnostics import (
    prepare_investment_diagnostics, save_investment_reports,
)

alternatives = [run.full for run in study.scenarios.values()]  # completed feature study
factsheets = save_native_factsheets(inputs, result, output_dir, alternatives)
diagnostics = prepare_investment_diagnostics(inputs, risk, result, alternatives)
investment_reports = save_investment_reports(diagnostics, output_dir)
```

Fresh timestamped folders under `factsheets/` and `investment_diagnostics/` contain the six
PDFs, page previews, numerical CSVs and SHA-256 manifests linked to the baseline backtest and
covariance bundle. Full signal pairs, original scores, breadth weights/covariance dates, linked
Brinson effects and gross/net reconciliations remain available for audit. Compact group names
are display aliases only; exported tables retain the original economic-sleeve names.

## Attribution by model layer and feature

The pipeline now supplies actual backtested layer NAVs to QIS
[`model_layer.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/attribution/model_layer.py)
and optionally
[`model_feature.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/attribution/model_feature.py).
Model layers describe portfolio-construction stages, rather than exposures to the eleven macro
factors. The distinction matters: QIS takes completed counterfactual portfolios as inputs; it
does not construct the standalone portfolios or infer feature effects from one full-model NAV.

| Input layer | Public construction |
|---|---|
| Benchmark | The agreed constant-weight ME benchmark, gross of trading costs |
| Risk | OP minimum **excess-return variance**, with the full model's covariance, long-only/investment limits, sleeve bands, ETF caps, TE budget, turnover budget and execution assumptions |
| Signal | OP `compute_top_quantile_equal_weights`: equal weight in the top 25% of the 35 non-BIL momentum-score ranks; nine holdings with this complete universe |
| Full gross | The existing max-alpha-under-TE portfolio, replaying the accepted net-policy targets without transaction costs |
| Full net | The original portfolio with its configured transaction costs |

The risk-only solve has an explicit minimum-variance objective. Setting all alpha scores to zero
in this example's linear max-alpha objective would instead leave every feasible portfolio
equally optimal. This is a public analytical layer definition, not an import of private ROSAA
research code. Its risk criterion uses the same BIL-relative covariance as the full model.

Risk-layer decisions use actual net pretrade holdings and the same cost schedule as the full
model. Their gross NAV replays those targets without costs, matching the full model's gross
convention. The signal portfolio has no TE, sleeve or turnover constraint, and is gross of costs;
it is a diagnostic portfolio, not a mandate-compliant alternative. It ranks signed scores as
supplied, without an extra positivity screen, and breaks ties by the fixed universe order.
Every layer uses the full model's exact decision dates and execution lag. No future prices or
regression estimates enter those decisions.

The default run adds `attribution/layers/model_attribution.pdf`, two PNG pages and source CSVs.
QIS fits full-sample benchmark OLS regressions on completed **ME log returns**, with 95% intervals
from Bartlett HAC with three monthly lags. Its exact additive decomposition is:

```text
Full gross log return = systematic return + risk alpha + signal alpha + integration alpha
Full net log return   = full gross log return + trading-cost drag
```

Systematic return uses the full model's benchmark beta. Risk and signal alpha paths remove each
standalone layer's own benchmark beta; integration is the residual reconciling them with the full
portfolio. Integration is not a separately investable sleeve. Annualised mean log contributions
and cumulative log contributions add exactly; they are not CAGRs or simple-return contributions.
The coefficients are descriptive, estimated after the backtest, and never used as alpha forecasts.
Opening costs are already in inception NAV and are outside the measured return intervals.
At least three completed monthly returns are required; use `--no-attribution` for shorter runs.

The optional feature experiment requires the QIS zero-effect HAC fix in
`qis/utils/regression.py`, validated here using a local QIS source build. For exactly zero alpha
with zero HAC standard error, QIS reports p-value one and a zero-width confidence interval;
R-squared remains undefined for a constant response. Older builds can raise
`OLS HAC estimation produced non-finite outputs` on unchanged layer effects. The example uses
QIS inference directly; no returns are perturbed. The source fix has not been published by this task.

To demonstrate **feature attribution across layers**, run a complete two-feature experiment:

```console
python -m examples.oss_portfolio_pipeline --end 2026-10-24 --feature-attribution
python -m examples.oss_portfolio_pipeline --end 2026-10-24 --solver CLARABEL --feature-attribution
```

The four coalitions use the same downloaded prices, fitted covariance matrices, benchmark and
costs. They differ only in the enabled settings:

| Coalition | Momentum span | Annual TE budget |
|---|---:|---:|
| Empty: original baseline | 12 months | 3.0% |
| Momentum feature | 24 months | 3.0% |
| TE-budget feature | 12 months | 4.5% |
| Both features | 24 months | 4.5% |

These are predetermined sensitivities, not tuned parameter recommendations. If the baseline
settings are overridden, the features double its momentum span and multiply its TE budget by 1.5.
All four coalitions use the complete downloaded ETF history for signal warm-up and start on
31 December 2019. Sufficient signal history is required for the longer span; incomplete warm-up
raises rather than moving inception. Each coalition reruns the full and
standalone layer portfolios. A momentum change cannot alter the risk-only layer; a TE-budget
change cannot alter the unconstrained signal layer. The tests check both controls.

`qis.ModelLayerNavs` is the bundle defined in `model_feature.py`. The in-memory example is:

```python +SKIP
# inputs, risk, config and result are the completed pipeline objects above.
import qis
from examples.oss_portfolio_pipeline.attribution import run_attribution, save_attribution

study = run_attribution(inputs, risk, config, result, include_features=True)
coalitions = {enabled: run.navs for enabled, run in study.scenarios.items()}

# Each enabled key is a frozenset of feature names; frozenset() is the baseline.
# Each value carries benchmark, risk-layer, signal-layer, full gross and full net NAVs.
effects = qis.compute_model_feature_alpha_beta_attribution(
    scenario_layer_navs=coalitions, freq='ME', hac_lags=3, confidence_level=0.95)
print(effects.summary.loc['Shapley'])
print(effects.identity_errors)
save_attribution(study, output_dir)  # existing external backtest bundle
```

QIS constructs factorial (Harsanyi) effects, including the interaction, and order-independent
Shapley effects from **NAV ratios**. It fits each effect path directly: confidence intervals must
not be made by subtracting standalone regression intervals. It requires all four coalitions and
identical benchmark paths. Both the factorial and Shapley return paths must reconstruct the joint
feature effect at every date; the layer alpha components must also sum to full-model alpha.

The optional experiment writes a separate `attribution/features/model_attribution.pdf`, with a
third page showing feature effects through risk, signal, integration, full alpha and net return,
including their HAC intervals. CSVs retain all scenario NAVs, target weights, alpha scores,
constraint audits, regressions, exact component returns, factorial/Shapley paths, joint effects
and identity errors. Each bundle has an `attribution_manifest.json` with settings, QIS versions,
file hashes and a link to the baseline backtest manifest. Layer-only and feature bundles use
separate directories so their outputs cannot be mistaken for the same experiment.

## Stress risk report at the last rebalance

Add `--stress-report` to build QIS's native instrument-portfolio stress report for the baseline
optimal portfolio's **last accepted target allocation**. The example finds the FM snapshot
assigned to that decision; it does not substitute a later covariance or the final daily holdings.
For the study ending 24 October 2026, the selected rebalance is 30 September 2026.

```console
python -m examples.oss_portfolio_pipeline --end 2026-10-24 --stress-report --no-attribution
```

The example remains in `stress.py`. Reuse completed pipeline objects without downloading or
refitting anything:

```python +SKIP
from examples.oss_portfolio_pipeline.stress import build_stress_study, save_stress_report

stress = build_stress_study(inputs, risk, result, notional=1_000_000.0)
print(stress.result.risk)
print(stress.result.summaries['conditional'])
artifacts = save_stress_report(stress, output_dir)
print(artifacts.pdf_path)
```

This calls QIS `InstrumentPortfolio`, `run_portfolio_stress_test` and
`generate_portfolio_stress_report`. Every positive target weight becomes a funded `DELTA_1`
holding on a hypothetical USD 1 million book. Adjusted-price units are synthetic index units,
not executable share counts. Zero allocations remain in the source audit and 36-response risk
model. Prices and risk are taken at the decision close; calendar weekend labels are mapped to
the actual observed session. Costs, carry, later price drift and subsequent rebalancing are absent.

The report contains requested independent shocks, conditional co-moves, the ten worst historical
factor-vector replays, exposures, risk contributions, the fitted FM/FL clusters, six sensitivity
grids, fit diagnostics and a production-risk audit. Shocks are simple percentage moves of the
**volatility-targeted excess factor NAVs**, converted by QIS to model log coordinates. A global
bond factor move is not a yield move; relative equity, nominal/TIPS and HY/IG factors retain
their basket definitions. Historical replay uses completed factor returns available by the
snapshot, with its current weights and loadings. It is a stress exercise, not realised performance.

**Correlated-residual boundary:** FM's `snapshot.to_risk_model(diagonal_reference=True)` preserves
the exact ROSAA asset covariance for total volatility, response-level risk and target tracking
error. QIS's current factor-decomposition interface supports diagonal response residuals, so the
factor-risk decomposition, cluster stacks and conditional bands explicitly use that reference.
The report identifies both volatility totals and exports the full selected residual covariance.
Reference bands use a one-month horizon and are not full-ROSAA prediction intervals. No residual
correlations are removed from the optimisation model or substituted into its total-risk calculation.

QIS writes its PDF, complete CSV tables, page previews and hash manifest into a fresh timestamped
directory under `stress/<decision-date>/`. `pipeline_manifest.json` links the report to the saved
backtest and covariance manifests. The native report limits displayed rows, while CSVs retain all
responses and scenarios. The numerical workbook is disabled to keep this example's output to
PDF/CSV. `--stress-report` requires the full backtest with reporting enabled.
Compact factor aliases keep chart headings readable; the source manifest maps them to the
unchanged full factor identifiers used in all numerical tables.

See [QIS instrument portfolio stress](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/portfolio_stress.md)
and [FM's risk-model adapter](https://github.com/ArturSepp/FactorModels/blob/main/src/factormodels/analytics/risk.py)
for the owning APIs and reference-decomposition contract.

## Outputs and verification

`pipeline.build_inputs(prices, config)` accepts a supplied complete adjusted-price panel for
offline reuse. `pipeline.run_pipeline(config)` adds the Yahoo download. The result exposes
asset prices, factor NAVs and returns, actual factor target schedules, universe metadata,
benchmark targets and the QIS benchmark `PortfolioData`.

`save_inputs` writes these tables, the full 40-source adjusted-price panel, and a manifest
recording the source download metadata, output checksums, conventions, exact factor definitions,
settings and installed package versions. Its inputs retain full asset estimation history while
the benchmark starts in December 2019. A saved bundle remains identifiable after a Yahoo refresh.

`save_covariances` adds monthly asset excess returns, the date mapping, factor covariance,
loadings, empirical residual covariance, selected asset covariance, fit statistics, clusters
and derived signs. The covariance manifest records the complete configuration, fit provenance,
package versions and table checksums, and links to the input manifest by SHA-256. In memory,
`result.model` is FM's completed `RollingFactorModel`, ready for subsequent risk analytics.

`save_backtest` adds active and benchmark daily NAVs (net and gross), target and realised
weights, units, raw and scored alphas, actual trade costs and turnover, pretrade holdings,
execution events, solver statuses, constraint audits and ex-ante/realised tracking-error data.
The backtest manifest links to the exact covariance manifest by SHA-256 and records the policy,
return conventions, solver and package versions. `result.executed.portfolio` and the benchmark
are QIS `PortfolioData` objects. The gross benchmark in the original input bundle remains
available; the new benchmark output applies the selected execution and cost assumptions.

Focused offline checks are explicitly invoked; the `_local.py` suffix keeps the check module
out of the unattended examples runner:

```console
python -m pytest examples/oss_portfolio_pipeline/validation_local.py examples/oss_portfolio_pipeline/covariance_validation_local.py examples/oss_portfolio_pipeline/backtest_validation_local.py
python -m pytest examples/oss_portfolio_pipeline/reporting_validation_local.py
python -m pytest examples/oss_portfolio_pipeline/attribution_validation_local.py
python -m pytest examples/oss_portfolio_pipeline/stress_validation_local.py
python -m pytest examples/oss_portfolio_pipeline/diagnostics_validation_local.py
python -m ruff check examples/oss_portfolio_pipeline
```

On the governed host, run those commands with `$pipelinePython` after the helper above; pytest's
cache must be directed to `$env:AGENT_LOCAL_ROOT\cache\pytest`. Checks cover cash financing,
volatility settings, future-data independence, trading calendars, holdings drift, input gaps
and cache integrity. Covariance checks use real FM/FL fits with CLARABEL, independent weighted
moment and covariance-assembly references, weekend ME mapping, and future-perturbation/prefix
invariance. Portfolio checks cover a closed-form cash/risky-asset optimum, direct constraint and
turnover references, held-unit NAV and costs, failure propagation, execution lag, and causal
alpha/portfolio prefixes. Reporting checks compare turnover with executed unit changes, costs
with cash debits, funding balances with NAV minus holdings, and TE with independent quadratic
forms; they also cover weekend labels, warm-up gaps, delayed initial funding, partial months,
input immutability and PDF/table provenance. Attribution checks cover risk-only feasibility and
minimum variance, independent signal ranks and OLS estimates, exact return bridges, all four
feature coalitions, independent two-ordering Shapley calculations and saved report provenance.
Stress checks cover exact snapshot selection, full-covariance risk, independent scenario P&L,
no future historical rows, weekend month-ends, notional scaling and native QIS output hashes.
Investment diagnostics check forward returns against independent price endpoints and score dates,
direct rank correlations, prefix invariance, full-covariance Euler breadth, an independent daily
Brinson recurrence and all six PDF exports. The timing check must fail under a one-month
look-ahead mutation before its passing result is trusted.
The full cached Yahoo/MOSEK run and rendered PDF review are separate integration checks.

## Interpretation and sources

The nominal/TIPS factor retains the paper's sign; it is not a pure inflation shock. The global
bond basket includes currency exposure, and the credit spread has unequal duration. CEW
contains currency carry. Commodity funds contain futures roll effects, and Invesco changed
their index methodology on 10 November 2025. These qualifications should accompany subsequent
reports. A fixed present-day ETF selection also retains survivorship selection.

- [FM](https://github.com/ArturSepp/FactorModels): factor construction and rolling model estimation.
- [QIS](https://github.com/ArturSepp/QuantInvestStrats): returns, risk kernels and backtesting.
- [FactorLasso](https://github.com/ArturSepp/FactorLasso): FCGL loadings, priors, signs and residual risk.
- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
- [Invesco commodity products and methodology notice](https://www.invesco.com/us/en/solutions/invesco-etfs/commodity-investing.html).
- [WisdomTree CEW](https://www.wisdomtree.com/us/products/currency/cew).

Yahoo data access and redistribution follow the provider's terms; the public example publishes
the downloader and definitions, while downloaded snapshots remain local.
