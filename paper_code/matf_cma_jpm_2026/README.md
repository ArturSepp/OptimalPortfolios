# MATF-CMA: Paper Code Reproduction

Reproduction code for the paper figures, bootstrap exhibits, and SAA optimisation of:

> Sepp, Hansen, Kastenholz (2026). **Capital Market Assumptions Using Multi-Asset Tradable Factors: The MATF-CMA Framework.** *Journal of Portfolio Management*, under revision.

This directory contains the full pipeline that produces the published paper figures, the bootstrap frontier-fan exhibit (Exhibit 1) and the alpha-beta consistency-violation diagnostic (Exhibit 2), and the SAA optimisation results (Exhibit 20). The methodology summary below documents every modelling and design choice; running the included scripts reproduces the published numbers within Monte Carlo noise.

---

## Quick start

```bash
cd paper_code/matf_cma_jpm_2026
pip install -r requirements.txt
```

Three pipelines are available:

```bash
# (1) Bootstrap exhibit (Exhibits 1 & 2)
python run_bootstrap.py

# (2) Per-figure paper exhibits
python figures.py --all --output-path figures/

# (3) SAA optimisation across the 8 mandates (Exhibit 20)
python run_optimisation.py --output-path figures/
```

Expected output:

- `figures/bootstrap_frontier.png` — Exhibit 1 (frontier fan: raw vs MATF factor-structured)
- `figures/consistency_violation.png` — Exhibit 2 (alpha-beta consistency violation)
- `figures/all_factor_cmas.PNG`, `figures/factor_attribution.PNG`, `figures/cma_scenarios.PNG`,
  `figures/risk_factors_perf.PNG`, `figures/risk_factors_annual.PNG`, `figures/risk_factors_corr.PNG`,
  `figures/universe_cmas.PNG`
- `figures/equity_factor_cmas.PNG`, `figures/rates_factor_cmas.PNG` — only when `paper_inputs.xlsx` was built with the rosaa-side parser; see "Input data" below
- `figures/efficient_frontier.PNG` — Exhibit 20

To run the reproducibility tests:

```bash
pytest tests/
```

The tests verify that with `seed=42`, the published headline numbers reproduce within tolerance:
- Reduction ratio at the Balanced-mandate vol target shows a clear MATF advantage
- GK median ‖Δ‖₂ is non-zero (Exhibit 2)
- MATF Δ ≡ 0 by construction (Exhibit 2)

---

## Files

| File | Purpose |
|---|---|
| `universe.py` | `load_paper_assets_short()` — the 17-asset paper universe definition. |
| `build_paper_inputs.py` | LGT-side parser. Reads production CMA pipeline xlsx + futures NAV CSV, produces `paper_inputs.xlsx`. Runs once at LGT inside the `rosaa` environment. |
| `paper_inputs.py` | `PaperInputs` container. Loads `paper_inputs.xlsx` and exposes the small subset of `MatfCmaData` / `CmaDataReport` calls used by figures and optimisation, with no `rosaa` dependency. |
| `figures.py` | Seven paper-figure functions wrapped in a `LocalTests` enum. Each saves via `qis.save_fig`. |
| `run_optimisation.py` | Solves the 8 mandate SAAs with tracking-error and box constraints, saves `efficient_frontier.PNG`, optionally produces `cma_portfolios.pdf`. |
| `run_bootstrap.py` | Bootstrap exhibit driver (Exhibits 1, 2). Self-contained; reads the production xlsx + factor NAV CSV directly without going through `paper_inputs.xlsx`. |
| `run_sensitivity.py` | σ_SR sensitivity sweep — sweeps the SR-prior dispersion across {0.05, 0.075, 0.10, 0.125, 0.15, 0.20} and prints the reduction-ratio table. Calls `run_bootstrap` six times, ~3 minutes runtime. |
| `bootstrap_frontier_analytics.py` | Three primitives: `stationary_block_indices`, `solve_long_only_frontier`, `min_variance_vol`. Reused across paper exhibits. |
| `requirements.txt` | Version-pinned dependencies for reproducibility. |
| `tests/test_reproducibility.py` | Reproducibility test suite (pytest). |
| `data/paper_inputs.xlsx` | Self-contained inputs for figures and optimisation (committed). |
| `data/global_saa_universe_data_cmas_usd_2026q1_jpm.xlsx` | Production CMA pipeline xlsx (must be supplied — see below). |
| `data/futures_risk_factors.csv` | Daily factor NAV history (must be supplied — see below). |
| `figures/` | Generated figures (gitignored). |

### Two-stage data flow

```
   Production CMA pipeline xlsx     ┐
   futures_risk_factors.csv         ├──► run_bootstrap.py ──► Exhibits 1, 2
                                    ┘                         (no paper_inputs.xlsx needed)

   Production CMA pipeline xlsx     ┐
   futures_risk_factors.csv         ├──► build_paper_inputs.py ──► paper_inputs.xlsx
   universe_snapshot.xlsx           ┘   (LGT-side, requires rosaa)

                                       paper_inputs.xlsx ──► figures.py
                                                          ──► run_optimisation.py
                                       (replicator-side, qis + optimalportfolios only)
```

### Input data

Three files must be placed in `data/` before running:

- `global_saa_universe_data_cmas_usd_2026q1_jpm.xlsx` — production CMA pipeline output for the USD investor at 2026-Q1. Required sheets: `asset_returns`, `residuals`, `y_betas`, `x_covar`, `y_variances`, `cma_metadata`, `factor_cmas`, `factor_attribution`.
- `futures_risk_factors.csv` — daily NAV history for the 9 MATF factors (Equity, Rates, Credit, Carry, Inflation, Commodities, Private Equity, Rates Vol, Fx). Index = trading date; columns in the same order as `y_betas` columns; `NAV_0 = 100`. The loader handles `Fx` vs `FX` case-insensitively.
- `universe_snapshot.xlsx` — universe-snapshot xlsx with a `universe weight` tab giving the 8 mandate weights (Income/Low/Balanced/Growth × {with, w/o} Alts).

These files contain proprietary inputs and are not included in the repository. The methodology below is fully self-contained and reproducible against any equivalent pipeline that exposes the same artefacts.

`paper_inputs.xlsx` is the parser output — committed in `data/`. It is **not** required by `run_bootstrap.py`, only by `figures.py` and `run_optimisation.py`. Two of the seven figure tests (`EQUITY_FACTOR_CMAS`, `RATES_FACTOR_CMAS`) need the regional CMA tables that are computed inside `rosaa.CmaDataReport`; if `paper_inputs.xlsx` was built with `--no-rosaa`, those two tests skip cleanly with an informative message.

To rebuild `paper_inputs.xlsx` from updated production data:

```bash
# At LGT, inside the rosaa environment:
python build_paper_inputs.py \
    --production-xlsx data/global_saa_universe_data_cmas_usd_2026q1_jpm.xlsx \
    --factor-navs-csv data/futures_risk_factors.csv \
    --universe-xlsx   data/universe_snapshot.xlsx \
    --output-xlsx     data/paper_inputs.xlsx
```

---

## Purpose of the bootstrap exhibits

The bootstrap quantifies two distinct properties of the MATF-CMA framework on the 17-asset paper universe:

1. **Estimation uncertainty** — the long-only efficient frontier's sampling distribution under two estimators:
   - **Raw asset-level**: NaN-aware sample mean and pairwise sample covariance on the asset-return panel. Mathematically equivalent to a per-asset Grinold-Kroner construction whose μ is the historical sample mean.
   - **MATF factor-structured**: equal-weighted factor covariance Σ_F on the resampled factor panel, equal-weighted residual variance D on the resampled residual panel, and λ drawn from a structural Sharpe-ratio prior.
2. **Alpha-beta consistency** — the property that μ lies in the column space of the factor loading matrix β. MATF satisfies this exactly by construction; per-asset building-block constructions (Grinold-Kroner, qualitative top-down) generally violate it by 50–200 bp per asset.

The frontier-fan comparison isolates *bandwidth*, not *level*: the point-estimate frontiers are visually aligned via per-asset μ recentering (described below) so the figure speaks unambiguously about sampling-distribution width.

---

## Universe and data

### Frontier universe

The frontier is computed on the 17-asset illustrative universe of the paper (snapshot `universe_20260503_1152`), comprising seven fixed-income, five equity, and five alternative-asset indices. The fixed-income sleeve is anchored on Global Government bonds (`LGTRTRUH`, hedged USD) — replacing the US-Treasury-only `LUATTRUU` of earlier drafts — and is supplemented by Global IG, Global HY, EM HC, Other Fixed Income, Global Inflation-Linked (`LF94TRUH`), and Global Convertibles (`H24641US`). Five alternative assets (Private Equity, Private Credit, Hedge Funds, Real Assets, Insurance-Linked) report quarterly (QE); the remaining twelve report monthly (ME). Tickers and display names are listed in `PAPER_TICKERS` in `run_bootstrap.py`. Benchmark mandate weights match the `universe weight` tab of `universe_snapshot.xlsx` (May 2026 update).

### Pipeline outputs

All inputs come from a single CMA pipeline artefact for the USD investor as of 2026-Q1:

- (i) monthly grid asset-return panel **Y** ∈ R<sup>T × N</sup> with NaN entries for QE assets on non-quarter months,
- (ii) HCGL factor loadings **β** ∈ R<sup>N × 9</sup>,
- (iii) EWMA factor covariance Σ<sub>F</sub><sup>EWMA</sup>,
- (iv) annualised residual variances **D**<sup>pipe</sup>,
- (v) annualised residual time series **E**<sup>ann</sup>.

Native-frequency residuals are computed as ε<sub>t,i</sub> = E<sup>ann</sup><sub>t,i</sub> / s<sub>i</sub> with s<sub>i</sub> = 12 (ME) or 4 (QE).

### Factor returns from production NAVs

Factor returns are loaded directly from the production daily NAV file `futures_risk_factors.csv`, which contains the price index (NAV<sub>0</sub> = 100) for each of the 9 MATF factors over Dec 1998 – Mar 2026 at daily frequency.

To produce a monthly factor-return panel aligned with `asset_returns`, we re-index the NAV series to the asset-returns calendar dates (forward-filling where a calendar date falls on a non-trading day) and compute log returns:

x<sub>t,j</sub> = log(NAV<sub>t,j</sub> / NAV<sub>t-1,j</sub>)

The resulting factor-return panel **X̂** ∈ R<sup>(T-1) × 9</sup> is fully observed (no NaNs) for *all* 9 factors including Private Equity. The first calendar period is dropped because the log return at t = 0 requires NAV at t = -1, giving a paired sample length of T = 316 months (Jan 2000 – Mar 2026) before the window trim described below.

This is a substantive improvement over an OLS recovery x̂<sub>t</sub> = (β<sup>T</sup>β)<sup>-1</sup> β<sup>T</sup>(Y<sub>t</sub> - ε<sub>t</sub>) from the asset panel: (i) the PE factor is properly observed rather than held at the pipeline prior; (ii) several factors with low realised diagonal weight on the ME asset universe (Carry, Inflation, Rates Vol) no longer suffer from OLS-projection inflation; (iii) the algorithm matches what the production pipeline itself uses.

### Late-start asset backfill and bootstrap window

Two assets in the 17-asset universe begin trading after January 2000 and require backfill before the bootstrap window starts: Global Convertibles (`H24641US`, first observation January 2009) and Insurance-Linked (`LGT_ILS`, first observation December 2002). A third asset, Global HY (`H23059US`), begins in March 2001 and sets the bootstrap window's left edge. The backfill rules are encoded as `BACKFILL_RULES` in `run_bootstrap.py` and are applied to both `asset_returns` and the native-frequency `residuals` panel:

| Target | Proxy | Scale | Backfill window | Justification |
|---|---|---:|---|---|
| Global Convertibles (`H24641US`) | Global HY (`H23059US`) | 1.30× | Apr 2001 – Dec 2008 (93 ME obs) | Vol-scaling on the post-2009 sample where both series are observed: realised ann vol Convertibles = 10.1%, HY = 7.9%, ratio ≈ 1.28×. Reproduces the 2008 Bloomberg US Convertibles drawdown (≈ −36%) within historical bounds (1.3× HY 2008 = −34%). |
| Insurance-Linked (`LGT_ILS`) | Hedge Funds (`HFRXGL`) | 1.0× | Apr 2001 – Sep 2002 (6 QE obs) | HFRXGL compounded to quarterly returns; backfilled at QE-grid dates only, with ME-grid months remaining NaN per the QE convention. The HFRXGL drawdown in this specific window is mild (max DD ≈ −1.7% Apr 2001 – Sep 2002), so the proxy distortion is small even though HFRXGL and ILS diverge sharply during 2008. |

**Bootstrap window** runs from April 2001 to March 2026 (T = 300 monthly observations). After backfill and trim:

- All twelve ME-frequency assets are fully observed within the window.
- The five QE-frequency assets retain the structural QE pattern (NaN on non-quarter months) which the bootstrap handles via the `min_block = 3` floor described below.
- The factor-return panel **X̂** is trimmed to the same window before bootstrap sampling.

The decision to backfill rather than truncate to the latest first-print date (Jan 2009 for Convertibles) preserves T from 207 to 300 monthly observations — a 45% increase in effective sample size for the raw-asset comparator estimator. The decision to scale the proxy series (1.3× for Convertibles, 1.0× for ILS) rather than use the raw proxy is calibrated empirically: the 1.3× factor for Convertibles brings 2008 stress-period behaviour into line with documented benchmark drawdowns; the 1.0× factor for ILS reflects the absence of a 2008-stress mismatch in the specific Apr 2001 – Sep 2002 backfill window.

### AR(1) unsmoothing for appraisal-based assets

Three assets in the paper universe are reported with appraisal-based smoothing — Private Equity (`MP503001`), Private Credit (`MP503008`), and Insurance-Linked (`LGT_ILS`) — and carry `Unsmoothing = True` in the production xlsx's `cma_metadata` sheet. The AR(1) Geltner-style unsmoothing is applied **upstream by the production CMA pipeline** (`rosaa.market_data.cmas.estimate_global_saa_cma` calls `copy_universe_data_with_unsmoothed_prices` with PM-aligned defaults `span=20, max_value_for_beta=0.5, warmup_period=8, mean_adj_type=EWMA, is_log_returns=True` before `estimate_asset_universe_cma_data` populates the xlsx sheets). The `asset_returns` and `residuals` sheets stored in the production xlsx — and read directly by the bootstrap loader — are therefore the unsmoothed series, identical to those that feed the production CMA estimation.

Empirical confirmation that the production xlsx contains unsmoothed series: lag-1 autocorrelation of `MP503001` (Private Equity) = +0.10, `MP503008` (Private Credit) = +0.07, `LGT_ILS` (Insurance-Linked) = +0.02 — all well below the +0.35 to +0.50 range typical of raw appraisal-based series. By comparison, raw `HFRXGL` (Hedge Funds, not flagged for unsmoothing) shows lag-1 autocorrelation of +0.28 in the same panel.

The `_apply_unsmoothing()` helper is retained in `run_bootstrap.py` for documentation and as a fallback for any future code path that reads the raw (smoothed) `universe_data` excel directly rather than the post-CMA-pipeline xlsx; it is currently a no-op on production xlsx loads.

---

## Block bootstrap design

We use the **stationary block bootstrap** of Politis and Romano (1994) with mean block length L = 12 months. To preserve the integrity of quarterly observations, we impose a *minimum* block length L<sub>min</sub> = 3:

L<sup>(b)</sup> = max(L<sub>min</sub>, Geom(1/L))

With L<sub>min</sub> = 3, every block contains at least one full quarter, so every QE column receives at least one non-NaN observation per block.

For each draw b = 1, …, B (B = 500), we sample a single index vector **i**<sub>b</sub> = (i<sub>b,1</sub>, …, i<sub>b,T</sub>) and apply it to all three panels in **paired** fashion:

- **Y**<sup>(b)</sup> = **Y**[**i**<sub>b</sub>] (asset returns, T × 17)
- **X̂**<sup>(b)</sup> = **X̂**[**i**<sub>b</sub>] (factor returns, T × 9)
- **ε**<sup>(b)</sup> = **ε**[**i**<sub>b</sub>] (native-freq residuals, T × 17)

This pairing ensures any joint dependence (e.g. residual heteroskedasticity correlated with factor stress regimes) is preserved.

---

## Raw asset-level estimator

For each draw b, NaN-aware sample moments on **Y**<sup>(b)</sup>:

- μ̂<sup>(b)</sup><sub>i</sub> = mean(**Y**<sup>(b)</sup><sub>·,i</sub> \ NaN) · s<sub>i</sub>
- Σ̂<sup>(b)</sup><sub>ij</sub> = √s<sub>i</sub> √s<sub>j</sub> · cov<sub>pair-complete</sub>(**Y**<sup>(b)</sup><sub>·,i</sub>, **Y**<sup>(b)</sup><sub>·,j</sub>)

The covariance scaling rests on iid annualisation: a series at native frequency s<sub>i</sub> has annualised variance s<sub>i</sub> · var(**Y**<sub>·,i</sub>). For pairwise covariance with mixed frequencies, scale each series by √s<sub>i</sub> before computing the standard pairwise-complete covariance. Σ̂<sup>(b)</sup> is symmetrised and projected to the nearest PSD matrix by clipping eigenvalues at 10<sup>-10</sup>.

---

## MATF factor-structured estimator

### Factor covariance per draw

Σ<sub>F</sub><sup>(b)</sup> = cov(**X̂**<sup>(b)</sup>) · 12

covering all 9 factors. Symmetrised and PSD-projected.

### Residual variance per draw

D̂<sup>(b)</sup><sub>i</sub> = var(**ε**<sup>(b)</sup><sub>·,i</sub>) · s<sub>i</sub>

### Structural Sharpe-ratio prior for λ

The factor risk premia are reparameterised as Sharpe ratios λ<sub>j</sub> = SR<sub>j</sub> · σ<sub>F,j</sub>, with a structural multivariate normal prior:

SR<sup>(b)</sup> ~ N(**m**<sub>SR</sub>, Σ<sub>SR</sub>)

where

**m**<sub>SR</sub> = (0.30, 0.30, 0.25, 0.30, 0.10, 0.10, 0.50, 0.30, 0.00)

in the order (Equity, Rates, Credit, Carry, Inflation, Commodities, PE, Rates Vol, FX). These are the production MATF Sharpe-ratio priors used throughout the paper: equilibrium structural values of 0.30 for the directional risk premia (Equity, Rates, Carry, Rates Volatility), 0.25 for Credit, 0.10 for Inflation and Commodities, 0.50 for Private Equity reflecting its illiquidity premium, and 0.00 for FX which has no long-run drift.

The covariance is Σ<sub>SR</sub> = σ<sub>SR</sub><sup>2</sup> · ρ with σ<sub>SR</sub> = 0.10, giving a 95% range of ±0.20 around each factor's prior mean. This is approximately 4× tighter than the empirical sample-mean noise scale 1/√T<sub>eff</sub> ≈ 1/√25 ≈ 0.20, reflecting strong structural conviction beyond what the historical sample alone delivers. The correlation matrix ρ is derived from the production factor covariance Σ<sub>F</sub> stored in the `x_covar` sheet of the production xlsx, matching the EWMA correlation matrix displayed in the paper's factor-correlation exhibit. Loaded once at data-load time and stored in `RealData.Sigma_SR` so the bootstrap reuses a single consistent correlation structure across draws.

The factor risk premium per draw is then λ<sup>(b)</sup><sub>j</sub> = SR<sup>(b)</sup><sub>j</sub> · σ<sup>(b)</sup><sub>F,j</sub>.

### Asset moments per draw

- μ<sup>(b)</sup> = r<sub>f</sub> + β λ<sup>(b)</sup>
- Σ<sup>(b)</sup> = β Σ<sub>F</sub><sup>(b)</sup> β<sup>T</sup> + diag(D̂<sup>(b)</sup>)

---

## Per-asset μ recentering for visual alignment

The raw and MATF estimators in their natural form produce different per-asset μ̂ vectors: realised sample means deviate from the structural-prior implied means by ±4% for several assets (most notably Insurance-Linked, Private Credit, and EM HC Bonds, all with favourable post-2005 sample histories). Optimised long-only frontiers depend on the full μ vector, so the two point-estimate frontiers diverge at low vol unless aligned.

To isolate the bandwidth comparison from level disagreement, we apply a constant per-period offset to each asset:

Y<sup>centered</sup><sub>t,i</sub> = Y<sub>t,i</sub> + Δ<sub>i</sub>,    Δ<sub>i</sub> = (μ<sup>matf</sup><sub>i</sub> - μ<sup>raw</sup><sub>i</sub>) / s<sub>i</sub>

where μ<sup>matf</sup><sub>i</sub> = r<sub>f</sub> + β<sub>i</sub><sup>T</sup>(**m**<sub>SR</sub> ⊙ σ<sub>F</sub><sup>full</sup>) is the MATF baseline at the SR prior mean. The shift preserves variances, covariances, serial correlation, distributional shape, and NaN positions; only the sample mean changes. The recentered panel is used in the raw bootstrap; the MATF baseline is unchanged. After this transformation, the two point-estimate frontiers overlay each other across the vol grid within Monte Carlo noise.

---

## Frontier optimisation

For each draw b and a vol-target grid {v<sub>1</sub>, …, v<sub>G</sub>} with G = 24:

max<sub>w ≥ 0</sub> μ<sup>T</sup>w   s.t.   w<sup>T</sup>Σw ≤ v<sub>k</sub><sup>2</sup>,   1<sup>T</sup>w = 1

Each instance is a quadratically-constrained linear program solved with CVXPY's CLARABEL backend. Σ is wrapped via `cp.psd_wrap` with a 10<sup>-10</sup>·I regularizer. Infeasible solves return NaN.

The vol grid is a linear span between 1.02 · min(σ<sub>min,raw</sub>, σ<sub>min,matf</sub>) and 0.99 · min(σ<sub>argmax μ,raw</sub>, σ<sub>argmax μ,matf</sub>), capped at 0.15 (covers all 8 benchmark mandates).

---

## Numerical configuration

| Parameter | Value |
|---|---|
| Number of draws B | 500 |
| Mean block length L | 12 months |
| Minimum block length L<sub>min</sub> | 3 months (QE-safe) |
| Time series length T | 300 months (Apr 2001 – Mar 2026, post backfill + window trim) |
| Effective sample size T<sub>eff</sub> | T/L ≈ 25 |
| Number of frontier assets N | 17 |
| Number of factors M | 9 (all bootstrapped from CSV NAVs) |
| Vol-grid points G | 24 |
| SR mean (PE) | 0.50 |
| SR mean (Equity, Rates, Carry, Rates Vol) | 0.30 |
| SR mean (Credit) | 0.25 |
| SR mean (Inflation, Commodities) | 0.10 |
| SR mean (FX) | 0.00 |
| SR std σ<sub>SR</sub> | 0.10 |
| Random seed | 42 |

---

## Standard error of the three CMA estimators

Each of the three CMA methodologies — historical sample mean, Grinold-Kroner building blocks, and MATF-CMA — writes the asset-level expected return as a linear function of an estimated input vector **θ̂**. The standard error of R̂<sub>i</sub> is therefore a quadratic form Var(R̂<sub>i</sub>) = **a**<sub>i</sub><sup>T</sup> Cov(**θ̂**) **a**<sub>i</sub>, with three structural differences across estimators: (i) the dimension of **θ̂**, (ii) the source of dispersion in Cov(**θ̂**), and (iii) whether the variance shrinks with sample size or with cross-asset pooling.

### Sample-mean estimator

Loads T<sub>eff</sub> resampled returns at coefficient 1/T<sub>eff</sub>. The textbook standard error is σ<sub>i</sub> / √T<sub>eff</sub>. With T = 300 months, mean block length L = 12, and T<sub>eff</sub> = T/L = 25, this gives:

SE(R̂<sub>i</sub><sup>SM</sup>) = σ<sub>i</sub> / √T<sub>eff</sub>

For US equity at σ ≈ 17%: SE ≈ 17% / √25 ≈ **3.3%**.

### Grinold-Kroner estimator

Combines K<sub>GK</sub> = 4 per-asset component forecasts (D/P̂, ĝ, π̂, ΔP/Ê) at unit loadings. Under uniform pairwise correlation ρ between component forecast errors:

SE(R̂<sub>i</sub><sup>GK</sup>) = √(**a**<sub>i</sub><sup>T</sup> Σ<sub>GK</sub> **a**<sub>i</sub>)

Crucially, the variance does *not* shrink with T: each component is a forecast of a separate underlying process, each carrying its own dispersion. With ρ = 0.5 and the input dispersions of Section "Grinold-Kroner Building-Block Method", the standard error for US equity is approximately **3.6%**.

### MATF-CMA estimator

The MATF construction R̂<sub>i</sub> = r<sub>f</sub> + **β̂**<sub>i</sub><sup>T</sup> **λ̂** with **λ** = SR ⊙ σ<sub>F</sub> places the remaining randomness in the M-dimensional Sharpe-ratio prior:

SR ~ N(**m**<sub>SR</sub>, σ<sub>SR</sub><sup>2</sup> **ρ**<sub>SR</sub>)

The factor loadings **β̂**<sub>i</sub> and factor volatilities **σ**<sub>F</sub> are an order of magnitude better identified than **λ** (Chopra-Ziemba ordering) and are treated as fixed at leading order. The variance of R̂<sub>i</sub><sup>MATF</sup> is then a quadratic form in the loadings β̂<sub>i,j</sub> σ<sub>F,j</sub>:

SE(R̂<sub>i</sub><sup>MATF</sup>) = σ<sub>SR</sub> · √(**β̂**<sub>i</sub><sup>T</sup> diag(**σ**<sub>F</sub>) **ρ**<sub>SR</sub> diag(**σ**<sub>F</sub>) **β̂**<sub>i</sub>)

For an asset whose factor exposure is concentrated on a single dominant factor f<sup>*</sup> (US equity, with β̂<sub>Eq</sub> ≈ 1 and other loadings small), the formula reduces to:

SE(R̂<sub>i</sub><sup>MATF</sup>) ≈ |β̂<sub>i, f<sup>*</sup></sub>| · σ<sub>SR</sub> · σ<sub>F, f<sup>*</sup></sub>

Evaluated for US equity using the EWMA factor volatility from `x_covar` (matching `Exhibit tb:risk_factors_corr` in the paper):

SE ≈ 1.00 × 0.10 × 12.9% = **1.29%**

Using a long-run realised equity vol of 16% (closer to historical MSCI World over 2005-2026Q1) instead of the EWMA estimate gives 1.6%; the two values bracket the SE for US equity. The internally consistent choice for the SE table is the EWMA value 12.9%, since that is the same σ<sub>F</sub> that defines the Σ<sub>F</sub> matrix used elsewhere in the framework.

### Comparison

| Estimator | Inputs | Sample size | US equity SE |
|---|---|---|---:|
| Sample mean | T<sub>eff</sub> asset returns | T/L scales noise | 3.32% |
| Grinold-Kroner | K<sub>GK</sub> = 4 forecasts (per asset) | fixed; no T scaling | 3.64% |
| MATF-CMA | M = 9 SR values (universe-wide) | cross-asset pooling via β̂ | 1.29% |

The variance ratio is approximately 5 : 8 : 1, corresponding to a standard-error ratio of approximately 2.6 : 2.8 : 1. The MATF formula carries two structural sources of variance reduction not available to the other estimators:

1. **Universe-wide pooling**: the input dimension M = 9 is fixed across the entire N-asset universe, so every asset that loads on a factor contributes to estimating that factor's risk premium. The Grinold-Kroner construction estimates K<sub>GK</sub> = 4 quantities *per asset* with no cross-asset pooling.

2. **Structural prior tightness**: σ<sub>SR</sub> = 0.10 is calibrated against decades of cross-asset Sharpe-ratio evidence and equilibrium pricing arguments, not against the historical record of any single asset. By contrast, the sample-mean SE depends solely on the noisy asset history.

The bootstrap exhibit below confirms this analytical ordering empirically.

---

## Result: Exhibit 1 (frontier fan)

90% bandwidth of the bootstrap distribution of frontier expected return at selected volatility targets, 17-asset universe, T = 300 monthly observations (Apr 2001 – Mar 2026), seed = 42:

| Vol target | Raw 90% width | MATF 90% width | Reduction |
|---:|---:|---:|---:|
| 3.1% | 2.82% | 1.11% | 2.53× |
| 4.1% | 2.91% | 1.41% | 2.06× |
| 5.2% | 3.36% | 1.66% | 2.03× |
| 6.2% | 3.91% | 1.88% | 2.08× |
| 7.2% | 4.41% | 2.02% | 2.19× |
| 8.2% | 5.06% | 2.21% | 2.29× |
| 9.2% | 5.68% | 2.37% | 2.40× |
| 10.3% | 6.21% | 2.53% | 2.45× |
| 11.3% | 6.77% | 2.75% | 2.46× |
| 12.3% | 7.15% | 2.86% | 2.50× |
| 13.3% | 7.50% | 3.08% | 2.43× |
| 14.4% | 7.82% | 3.22% | 2.43× |

**The MATF framework reduces SAA estimation uncertainty by approximately 2.0–2.2× across the vol grid (3–15%)**, covering all eight benchmark mandates. Equivalently, the structural prior brings information equivalent to roughly 4–5× the sample size in pure historical data.

---

## Sensitivity to the SR-prior dispersion σ_SR

The reduction ratio depends on a single calibration choice: the per-factor standard deviation σ_SR of the structural Sharpe-ratio prior. The paper uses **σ_SR = 0.10**. This is informative but not extreme — at the effective sample size T_eff ≈ 25, the asymptotic standard deviation of the sample-mean Sharpe-ratio estimator is 1/√T_eff ≈ 0.20 (Lo 2002), so σ_SR = 0.10 encodes a structural prior **twice as tight as the historical record**. Because the variance of an estimator scales as σ_SR², halving the standard deviation is informationally equivalent to **four times the sample size**.

The sweep below confirms the calibration is the right side of the line that distinguishes informative from useless: the qualitative claim — *factor structure tightens the frontier fan* — survives across σ_SR ∈ [0.05, 0.15], but the quantitative 2× reduction is specific to σ_SR = 0.10.

| σ_SR | Reduction at 4.1% vol | at 6.2% | at 8.2% | at 10.3% | at 12.3% | at 14.4% | T_eff equiv |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 3.74× | 3.73× | 4.34× | 4.72× | 4.59× | 4.37× | 14× |
| 0.075 | 2.68× | 2.71× | 3.04× | 3.23× | 3.21× | 3.16× | 7× |
| **0.10** | **2.06×** | **2.08×** | **2.29×** | **2.45×** | **2.50×** | **2.43×** | **4× ← baseline** |
| 0.125 | 1.68× | 1.71× | 1.87× | 1.95× | 2.01× | 1.96× | 3× |
| 0.15 | 1.43× | 1.45× | 1.58× | 1.63× | 1.68× | 1.69× | 2× |
| 0.20 | 1.10× | 1.11× | 1.21× | 1.25× | 1.27× | 1.26× | 1× |

The mechanics are clean. The MATF bandwidth is approximately linear in σ_SR (the SR prior's standard deviation propagates linearly through λ = SR · σ_F into μ = β·λ, and the bootstrap fan width is dominated by the spread of μ across draws). So the reduction ratio scales approximately as 1/σ_SR. The right-most column is the variance-equivalent sample-size multiplier, computed as the squared reduction ratio at the Balanced-mandate vol target (6.2%). At σ_SR = 0.20 the prior matches the empirical SR-noise scale and the framework adds essentially no information; at σ_SR = 0.05 the prior implies extreme structural conviction, equivalent to roughly 14 times the historical sample, which is hard to defend without additional cross-asset disciplining.

The 95% prior interval implied by σ_SR = 0.10 — that is, ±0.20 around each factor's prior mean — spans the dispersion of point estimates across published cross-asset Sharpe-ratio surveys. For Equity at m_SR = 0.30 the [0.10, 0.50] range covers the 4–8% equity-premium debate (Welch 2000; Ilmanen 2011); for PE at m_SR = 0.50 the [0.30, 0.70] range covers the illiquidity-premium debate (Phalippou versus Harris-Jenkinson). σ_SR = 0.10 is therefore the tightest defensible calibration that does not contradict published cross-sectional dispersion in any single factor.

To reproduce the sweep:

```bash
python run_sensitivity.py
```

The script monkey-patches `run_bootstrap.SR_STD` for each grid value and re-runs the bootstrap with `seed = 42`. Total runtime is approximately 3 minutes for six σ_SR values × 500 draws each.

---

## Result: Exhibit 2 (alpha-beta consistency violation)

For any μ vector and HCGL factor loading matrix β, the *consistency residual* is

Δ = (I − β(β<sup>T</sup>β)<sup>-1</sup> β<sup>T</sup>)(μ − r<sub>f</sub>**1**)

i.e. the orthogonal projection of μ − r<sub>f</sub> onto the null space of β<sup>T</sup>. A framework is *alpha-beta consistent* if Δ ≡ 0. MATF satisfies this by construction: μ<sup>MATF</sup> = r<sub>f</sub> + βλ where λ is a free 9-vector and β is the same matrix used in Σ.

A per-asset building-block construction (Grinold-Kroner, qualitative top-down) violates this in general: each μ<sub>i</sub> is constructed independently from μ<sub>j</sub>, and the resulting 17-vector lies on a generic point in R<sup>17</sup>, not on the 9-dimensional column space of β.

For the diagnostic we use a sample-mean proxy μ<sup>GK</sup><sub>b</sub> = μ̂<sub>b</sub>, since a sampling distribution is required and the sample mean uses no exogenous inputs. The choice is incidental: **the consistency residual depends on the μ vector itself, not on how it was constructed.** Whether μ comes from forward-looking forecasts, sub-block estimation, or sample means, a μ vector that lies in R<sup>17</sup> but not in the 9-dimensional column space of β has a non-zero residual.

> Per-draw and per-asset tables below are from the production run on the 17-asset universe with backfill (T = 300, seed = 42).

| Statistic | GK |
|---|---:|
| Median ‖Δ<sub>b</sub>‖<sub>2</sub> | 2.67% |
| Mean ‖Δ<sub>b</sub>‖<sub>2</sub> | 2.81% |
| 95th percentile ‖Δ<sub>b</sub>‖<sub>2</sub> | 4.44% |
| **MATF (all statistics)** | **0.00% (by construction)** |

Per-asset median |Δ<sub>i</sub>| under GK, top 8 by violation magnitude:

| Asset | Median \|Δ<sub>i</sub>\| | 5th pct | 95th pct |
|---|---:|---:|---:|
| MSCI Europe | 0.68% | 0.09% | 1.99% |
| Hedge Funds | 0.66% | 0.07% | 1.72% |
| Private Credit | 0.65% | 0.07% | 1.69% |
| Global Convertibles | 0.57% | 0.06% | 1.75% |
| Other Fixed Income | 0.55% | 0.04% | 1.50% |
| Global Inflation-Linked | 0.53% | 0.07% | 1.50% |
| EM HC Bonds | 0.53% | 0.05% | 1.42% |
| Insurance-Linked | 0.52% | 0.04% | 1.59% |

A non-zero Δ<sub>i</sub> means: *there is no factor exposure profile λ such that the model-implied expected return of asset i equals its building-block expected return*. Under GK, the optimiser receives a μ that implies a different λ from the one used to build Σ; the optimal portfolio inherits residual factor exposures the practitioner did not intend. Under MATF, μ and Σ share the same λ; risk attribution and CMA decomposition are internally coherent.

---

## Why each design choice matters

- **Stationary block bootstrap (vs iid resampling).** Factor returns exhibit substantial monthly serial dependence, particularly during 2008 and 2022 stress regimes. iid resampling would break these clusters and understate fan width.

- **Minimum block length 3.** Avoids resampling artefacts for QE assets. Without this, blocks of length 1 or 2 would frequently exclude quarterly observations, producing all-NaN columns and undefined sample covariances.

- **Paired (factor, residual, asset) resampling.** Ensures joint distributional properties are preserved. The factor and residual panels are coupled in real markets (e.g. residual variance spikes during equity-factor stress).

- **Equal-weighted span-T covariance (vs EWMA).** The bootstrap should sample uncertainty around the *same statistic* that anchors the point estimate. Using EWMA inside the loop while showing equal-weighted point estimates (or vice versa) creates a regime mismatch in which the bootstrap median deviates systematically from the point estimate by ~100 bp.

- **Structural SR prior (vs sample-mean λ̂) — the key design choice.** A bootstrap estimator of λ from the historical mean has noise scale σ<sub>F</sub>/√T<sub>eff</sub> ≈ 0.20·σ<sub>F</sub>, precisely the sample-mean uncertainty embedded in the raw estimator. Using such a λ̂ for MATF would yield identical fan widths to the raw method — the framework would add no information. The structural prior with σ<sub>SR</sub> = 0.10 is approximately 4× tighter, reflecting decades of equilibrium-Sharpe-ratio literature, cross-asset evidence, and the no-arbitrage relationships embedded in factor construction.

- **Per-asset recentering.** The level disagreement between the raw sample mean and the MATF structural prior is a substantive output of the framework, but *within the bootstrap exhibit* it muddles the bandwidth comparison. Recentering separates the two effects so the figure speaks unambiguously about uncertainty width.

- **Production-aligned SR means.** The mean vector matches the production MATF Sharpe-ratio priors used throughout the paper: 0.30 for the four directional risk premia (Equity, Rates, Carry, Rates Vol), 0.25 for Credit, 0.10 for Inflation and Commodities, 0.50 for Private Equity, and 0.00 for FX. These are conservative equilibrium structural values, not calibrated to the realised sample mean. As a consequence, the MATF baseline expected-return frontier sits below the raw historical-mean frontier; the per-asset recentering step described above isolates the bandwidth comparison from this level disagreement.

- **Long-only fully-invested constraint.** Matches the SAA use case the paper targets. Allowing short positions would dramatically widen both fans (frontier expected return is unbounded under leveraged positions) and obscure the comparison.

- **Direct factor returns from production NAVs.** In earlier versions, factor returns were recovered via per-row OLS projection from the asset panel. That over-fitted the noise of the asset panel into factor estimates for low-loading factors (Carry, Inflation, Rates Vol), producing factor vols 2–3× larger than the production EWMA estimates. Loading factor returns directly gives clean, fully-observed factor returns for all 9 factors including PE.

- **Late-start asset backfill.** Two of the 17 universe assets begin trading after the bootstrap window starts (Convertibles in 2009, ILS in 2002). Backfilling with vol-scaled proxy series (1.3× HY for Convertibles; 1× HFRXGL for ILS) preserves T = 300 months for the raw-asset comparator estimator instead of truncating to T = 207 (Convertibles' first print). The 1.3× scaling brings backfilled 2008 Convertibles behaviour into line with documented Bloomberg US Convertibles index drawdown of ~−36%. The 1× scaling for ILS is justified by HFRXGL's mild −1.7% drawdown in the specific Apr 2001 – Sep 2002 backfill window.

- **AR(1) unsmoothing for PE / PC / ILS — applied upstream.** The production CMA pipeline (`rosaa.market_data.cmas.estimate_global_saa_cma` → `copy_universe_data_with_unsmoothed_prices`) unsmooths the price panel before `estimate_asset_universe_cma_data` writes the `asset_returns` and `residuals` sheets. The bootstrap reads those sheets directly, so it inherits the unsmoothed series without any additional preprocessing. The empirical lag-1 autocorrelations of the PE/PC/ILS columns confirm this (+0.10, +0.07, +0.02 — well below the +0.35 to +0.50 typical of raw appraisal data).

---

## Anchors in the literature

The exhibit's findings can be anchored to four well-established threads of the portfolio-optimisation literature.

**Estimation error in mean–variance optimisation.** Chopra and Ziemba (1993) established that errors in means dominate errors in variances and covariances, with relative damage approximately 11× and 21× respectively. Decomposing the bootstrap fan into λ-only and Σ-only contributions reproduces this ordering on real MATF factor data: holding λ fixed and resampling only Σ produces fans of ~0.5% width; holding Σ fixed and shocking λ via the SR prior produces fans of ~3% width.

**Resampled efficient frontiers.** The original Michaud (1989) "optimisation enigma" argument — that mean–variance optimisation behaves as an "estimation error maximiser" — underpins the bootstrap-as-diagnostic approach. Michaud (1998) introduced the resampled efficient frontier methodology of which our exhibit is a particular instance. Markowitz and Usmen (2003) compared resampled efficiency to Bayesian shrinkage and found the two empirically similar; our SR-based prior is closer to the Bayesian arm of that comparison.

**Out-of-sample failure of the sample-mean estimator.** DeMiguel, Garlappi and Uppal (2009) document that the historical estimation window required for sample-based mean–variance to beat 1/N is approximately **3,000 months for 25 assets and 6,000 months for 50 assets**, calibrated to U.S. equities. Our 25-year (300-month) sample is more than two orders of magnitude smaller than this threshold, confirming directly that the raw asset-level estimator in Panel A is severely estimation-error-limited.

**Constraints as implicit shrinkage.** Jagannathan and Ma (2003) showed that long-only constraints in mean–variance act as implicit shrinkage of the covariance matrix. Our long-only fully-invested constraint gives both methods the same structural advantage; the reduction ratio measures the marginal value of factor structure *conditional on* long-only regularisation already being in place.

---

## What the exhibit does NOT claim

1. **It does not validate the SR prior.** σ<sub>SR</sub> = 0.10 is a modelling input, not a derived statistic. A weaker prior (σ<sub>SR</sub> = 0.20) would shrink the reduction ratio toward 1×; a stronger prior (σ<sub>SR</sub> = 0.05) would push it beyond 4×.

2. **It does not claim that MATF is more accurate.** The two methods are sampling distributions of *different* estimators. Whether MATF's tighter bands correspond to lower out-of-sample error is an empirical question requiring a separate backtest.

3. **It does not address β uncertainty.** The HCGL factor loadings are held fixed at their full-sample point estimate. Bootstrapping β would widen the MATF fan, probably to the 1.3×–1.5× reduction range.

4. **It does not address regime change.** Equal-weighted full-sample bootstrap implicitly assumes stationarity. EWMA-based variants would respond more rapidly to recent regimes but at the cost of ignoring valuable historical stress observations.

---

## Implementation summary

For each of B = 500 draws:

1. Sample **i**<sub>b</sub> via stationary block with L = 12, L<sub>min</sub> = 3.
2. **Y**<sup>(b)</sup> ← **Y**<sup>centered</sup>[**i**<sub>b</sub>], **X̂**<sup>(b)</sup> ← **X̂**[**i**<sub>b</sub>], **ε**<sup>(b)</sup> ← **ε**[**i**<sub>b</sub>].
3. **Raw**: compute (μ̂<sup>(b)</sup>, Σ̂<sup>(b)</sup>) from **Y**<sup>(b)</sup> via NaN-aware annualised moments.
4. **Raw frontier**: r<sub>k</sub><sup>(b),raw</sup> ← max<sub>w</sub> μ̂<sup>(b)T</sup>w s.t. w<sup>T</sup>Σ̂<sup>(b)</sup>w ≤ v<sub>k</sub><sup>2</sup>, w ≥ 0, 1<sup>T</sup>w = 1, for k = 1, …, G.
5. **MATF**: compute Σ<sub>F</sub><sup>(b)</sup> from **X̂**<sup>(b)</sup>, D̂<sup>(b)</sup> from **ε**<sup>(b)</sup>.
6. Sample SR<sup>(b)</sup> ~ N(**m**<sub>SR</sub>, Σ<sub>SR</sub>).
7. λ<sup>(b)</sup> ← SR<sup>(b)</sup> ⊙ √diag(Σ<sub>F</sub><sup>(b)</sup>).
8. μ<sup>(b)</sup> ← r<sub>f</sub> + β λ<sup>(b)</sup>; Σ<sup>(b)</sup> ← β Σ<sub>F</sub><sup>(b)</sup> β<sup>T</sup> + diag(D̂<sup>(b)</sup>).
9. **MATF frontier**: r<sub>k</sub><sup>(b),matf</sup> ← solve as in step 4.
10. **Consistency residual**: Δ<sub>b</sub> ← (I − β(β<sup>T</sup>β)<sup>-1</sup>β<sup>T</sup>)(μ̂<sup>(b)</sup> − r<sub>f</sub>**1**).

After B draws, per-vol-target percentiles produce the two fans of Exhibit 1; per-draw consistency residuals produce the distribution of Exhibit 2.

---

## References

- Chopra, V.K. and Ziemba, W.T. (1993). The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice. *Journal of Portfolio Management* 19(2): 6–11.
- DeMiguel, V., Garlappi, L., and Uppal, R. (2009). Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy? *Review of Financial Studies* 22(5): 1915–1953.
- Grinold, R. and Kroner, K. (2002). The Equity Risk Premium: Analyzing the Long-Run Prospects for the Stock Market. *Investment Insights* 5(3): 7–33.
- Jagannathan, R. and Ma, T. (2003). Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps. *Journal of Finance* 58(4): 1651–1683.
- Markowitz, H.M. and Usmen, N. (2003). Resampled Frontiers Versus Diffuse Bayes: An Experiment. *Journal of Investment Management* 1(4): 9–25.
- Michaud, R. (1989). The Markowitz Optimization Enigma: Is "Optimized" Optimal? *Financial Analysts Journal* 45(1): 31–42.
- Michaud, R. (1998). *Efficient Asset Management: A Practical Guide to Stock Portfolio Optimization and Asset Allocation.* Harvard Business School Press.
- Politis, D.N. and Romano, J.P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association* 89(428): 1303–1313.
