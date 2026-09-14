---
myst:
  html_meta:
    description: >-
      Stress a five-stock, ten-option portfolio using OptimalPortfolios FCGL estimation,
      fitted FactorLasso clusters, VOP option pricing and QIS portfolio reports.
---

# Stress testing with options and FCGL clusters

*[author / affiliation / date — placeholder]*

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

## Inputs, notation, and assumptions

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
\left\{
\frac{1}{T}\sum_{t=1}^{T}\sum_{i=1}^{5}
m_{ti}\eta^{T-t}(y_{ti}-B_i x_t)^2
+\lambda\sum_{g=1}^{G}\sqrt{\frac{n_g}{G}}
\sum_{j=1}^{4}\lVert B_{g,j}\rVert_2
\right\},\qquad \lambda=10^{-5}.
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

Signed contracts $n_\ell<0$ create negative option gamma. Stocks use shares times the spot
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

The frozen sample yields two fitted groups:

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

At SPY -20%, the FCGL result combines stock P&L of -25.86%, short calls +5.56% and short
puts -22.08%. At SPY +20%, short calls absorb 19.13 percentage points of the stock gain.
The negative convexity remains visible after changing the risk estimator. Lower losses
under FCGL do not prove greater accuracy: shrinkage changes the extrapolated stock responses.

The run uses VOP 2.2.0, FactorLasso 0.18.0, and the OP/QIS source checkouts on 2026-09-14.
The observed-price cache SHA-256 is
`1e366949dd117c5ef2a62e810811598ada619f33fbf1e5dbbb4add7154e0f98b`.
The provenance records source hashes separately from installed distribution versions.

## Implementation in optimalportfolios

The [self-contained manual runner](https://github.com/ArturSepp/OptimalPortfolios/blob/main/examples/reports/stress_testing_with_options_local.py)
requires current OP, FactorLasso, QIS instrument stress reports (`qis>=5.30.2`), yfinance and
VOP. The `_local.py` suffix keeps this optional pricing/reporting workflow outside the
core-only unattended example lanes. It has been run end to end with the frozen Yahoo cache.
It does not require a separate QIS examples checkout or Bloomberg access.

In your chosen environment, install the example-only prerequisites and run from an OP checkout:

```console
python -m pip install -e ".[data]" "qis>=5.30.2" "factorlasso>=0.18.0" "vanilla-option-pricers==2.2.0"
python -m examples.reports.stress_testing_with_options_local --cache-dir /absolute/cache/options_stress --output-dir /absolute/output/fcgl_options_stress
```

On the maintainer's Windows host, use the external interpreter and C-local generated state.
The verification run selected the current QIS source through a process-local `PYTHONPATH`;
the OP environment's older installed QIS package was not replaced.

```powershell
. '..\ArturSepp\scripts\repo_governance\Enter-AgentRepo.ps1'
$env:PYTHONPATH = 'C:\Users\artur\OneDrive\analytics\my_github\QuantInvestStrats\src'
& 'C:\Python\OptimalPortfolios312\Scripts\python.exe' -m examples.reports.stress_testing_with_options_local --cache-dir "$env:AGENT_LOCAL_ROOT\data\options_stress_20251231" --output-dir "$env:AGENT_LOCAL_ROOT\runs\fcgl_options_stress_new"
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

## Interpretation and limitations

- Five technology/growth stocks provide a small, concentrated teaching universe. Two clusters
  are a display/regularisation choice, not evidence of two stable market regimes.
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

- [Covariance estimators](covariance_estimators.md): OP factor estimation and annual covariance.
- [Rolling factor covariance from prices](rolling_factor_covar_from_csv.md): return preparation and rolling interfaces.
- [QIS options stress guide](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/stress_testing_with_options.md): the matched EWMA example and VOP pricing formulas.
- [QIS instrument stress interface](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/portfolio_stress.md): custom payoff and reporting contracts.

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [FactorLasso software and methodological references](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff).
  The [FCGL implementation](https://github.com/ArturSepp/factorlasso/blob/main/src/factorlasso/lasso_estimator.py)
  defines the cluster-factor penalty, observation weights and residual diagnostics.
- [QIS software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers): BSM values and forward Greeks.
- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html): observed stock/ETF data and request conventions.
