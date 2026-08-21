# De-PC1 MAC constrained TRE/turnover sweep execution report

> **Superseded reporting convention:** the beta, alpha, and p-alpha columns in this raw-return
> report are retained only as an audit record. The corrected production-compatible excess-return
> estimates and artifacts are documented in
> `2026-08-16_sol_dePC1_MAC_TRE_sweep_excess_alpha_report.md`. All seven non-regression metric
> columns are exactly unchanged across all 32 cells.

**Date:** 2026-08-16  
**Executor:** sol  
**Status:** COMPLETE — 32/32 cells produced; one non-fatal solver fallback disclosed below  
**Scope:** owner-requested TRE grid using the frozen De-PC1 MAC production model  
**Repository actions:** no staging, commit, push, tag, publication, or release

## Outcome

The existing ROSAA TRE/turnover sweep completed on the frozen `MAC_CONSTRAINED_BATCH` input with
the local FactorLasso development source and
`ClusterCorrelationTransform.REMOVE_PC1`. Covariance, SAA, benchmark weights, and alpha scores
were fitted once and reused by all 32 TAA cells; the sweep did not re-estimate the risk model per
cell.

The risk-adjusted optimum on the requested grid is **TRE utility 1 / turnover utility 0.6**:
7.12% annual return, 6.15% volatility, 1.157 Sharpe (rf=0), -18.0% maximum drawdown, 3.47%
ex-ante total TRE versus SAA, and 56.0% annual turnover. The frozen production point
**50 / 0.4** delivers more return, 7.93%, but at 7.26% volatility, 1.093 Sharpe, -23.7%
maximum drawdown, 2.95% ex-ante TRE, and 108.0% turnover.

This is a trade-off rather than a cell that dominates production on every dimension:

- five cells improve both Sharpe and turnover versus production, but all five give up annual
  return;
- no cell improves annual return, Sharpe, and turnover simultaneously;
- at the same turnover-utility setting as production, **1 / 0.4** is the strongest
  risk-adjusted alternative: Sharpe 1.146, drawdown -20.3%, and turnover 98.0%, at the cost of
  54 bp less annual return and 43 bp more ex-ante TRE;
- **25 / 0.6** is the best intermediate compromise: 7.59% return, 1.123 Sharpe, -19.9%
  drawdown, 3.12% ex-ante TRE, and 58.3% turnover;
- forcing the portfolio close to SAA with TRE utilities of 250–1,000 pushes beta toward one and
  reduces TRE, but generally lowers Sharpe below 1.0;
- the maximum-return cell, **100 / 0.2**, earns 8.24%, but its Sharpe is 1.084 and turnover is
  188.5%.

If the selection objective is purely risk-adjusted performance, the grid selects **1 / 0.6**.
If retaining the existing turnover-penalty setting matters, **1 / 0.4** is the cleaner candidate.
The single fallback in the maximum-Sharpe cell should be acknowledged before either is promoted
to a production setting.

## Frozen run specification

| item | value |
|---|---|
| production batch | `MAC_CONSTRAINED_BATCH` |
| mandate | `MAC` |
| constrained | `True` |
| signal | `PROD_MOM_BETA_CLUSTER` |
| returns input | `20260810_APAC_ROSAA_Fund_and_Index_Data` |
| factor model | `MATF_CUSTOM` |
| only numerical model-field change | `cluster_correlation_transform: none -> remove_pc1` |
| production dates | `FUND_BACKTEST_DATES` |
| reporting window | 2004-12-31 through 2026-07-31 |
| reporting frequency | quarterly (`QE`) |
| TRE utility grid | 1, 10, 25, 50, 100, 250, 500, 1,000 |
| turnover utility grid | 0.2, 0.4, 0.6, 0.8 |
| production grid point | 50 / 0.4 |
| grid size | 8 × 4 = 32 cells |
| covariance/SAA computation | once, shared in process across all cells |
| multi-portfolio factsheet | disabled; the standard three-page sweep PDF was emitted |
| execution Excel | not requested and not emitted |
| resource directory | `C:/Users/artur/OneDrive/analytics/my_github/OptimalPortfolios/rosaa/resources/` |
| output directory | `C:/Users/artur/OneDrive/analytics/outputs/depc1_mac_tre_sweep_20260816/` |
| cache directory | none; shared fitted inputs were held in memory |
| run id | `20260816_depc1_tre_grid` |

Runner: `../../../rosaa/products/funds/analysis/run_tre_sweep_depc1.py`  
Sanctioned sweep entry point: `rosaa.core.calibration.run_tre_range_sweep`

Command:

```powershell
$env:PYTHONPATH='C:\Users\artur\OneDrive\analytics\my_github\OptimalPortfolios\src;C:\Users\artur\OneDrive\analytics\my_github\FactorLasso;C:\Users\artur\OneDrive\analytics\my_github\QuantInvestStrats\src;C:\Users\artur\OneDrive\analytics\my_github\BloombergFetch'
$env:FACTORLASSO_DEV_ROOT='C:\Users\artur\OneDrive\analytics\my_github\FactorLasso'
$env:MPLBACKEND='Agg'
C:\Users\artur\OneDrive\analytics\my_github\OptimalPortfolios\.venv\Scripts\python.exe -m rosaa.products.funds.run_tre_sweep_depc1
```

## Metric conventions

- Return, volatility, Sharpe, and maximum drawdown use the sweep's quarterly `PerfParams` over
  the TAA window. The reported Sharpe is the rf=0 convention.
- Beta, annual alpha, and alpha p-value are regressions against the static-weight product
  benchmark, net of its management fee, sampled quarterly.
- The sweep calls `fetch_default_report_kwargs(..., add_rates_data=False)`. Its alpha is therefore
  a raw-return intercept. The ordinary production factsheet calls the same helper with
  `add_rates_data=True` and reports excess-return alpha. The sweep alpha column is internally
  comparable across its 32 cells, but it is **not directly comparable to the production
  factsheet alpha**. This convention difference does not affect the return, volatility, rf=0
  Sharpe, drawdown, TRE, or turnover ranking reported here.
- Ex-ante TRE is the mean TAA-versus-SAA risk implied by the total covariance, with
  `residual_var_weight=1.0`, even though the optimiser uses its frozen production residual-risk
  setting.
- Ex-post TRE is the mean annualised EWMA volatility of monthly TAA-minus-SAA returns, span 36.
- Turnover is the mean rolling 12-period aggregate returned by the production portfolio object.

The production sweep cell reconciles with the previously generated De-PC1 production factsheet
at display precision for annual return (7.9%), QE volatility (7.3%), rf=0 Sharpe (1.09), maximum
drawdown (-24%), beta (0.83), and turnover (108%). Its sweep alpha is 3.26%, whereas the production
factsheet reports 3.0%, for the rates-data convention reason above.

## Selected cells

| selection | TRE | TO | return | vol | Sharpe | max DD | beta | raw alpha | ex-ante TRE | ex-post TRE | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| production | 50 | 0.4 | 7.93% | 7.26% | 1.093 | -23.7% | 0.835 | 3.26% | 2.95% | 3.03% | 108.0% |
| maximum Sharpe | 1 | 0.6 | 7.12% | 6.15% | 1.157 | -18.0% | 0.699 | 3.18% | 3.47% | 3.53% | 56.0% |
| best Sharpe at TO=0.4 | 1 | 0.4 | 7.39% | 6.45% | 1.146 | -20.3% | 0.724 | 3.33% | 3.37% | 3.61% | 98.0% |
| intermediate compromise | 25 | 0.6 | 7.59% | 6.76% | 1.123 | -19.9% | 0.781 | 3.20% | 3.12% | 3.13% | 58.3% |
| maximum return | 100 | 0.2 | 8.24% | 7.60% | 1.084 | -24.0% | 0.890 | 3.31% | 2.79% | 2.79% | 188.5% |
| minimum ex-ante TRE | 1,000 | 0.8 | 7.65% | 8.27% | 0.924 | -26.3% | 0.986 | 2.19% | 2.22% | 1.86% | 25.8% |
| minimum turnover | 500 | 0.8 | 7.44% | 8.25% | 0.901 | -25.9% | 0.983 | 2.00% | 2.25% | 1.82% | 22.1% |

### Deltas versus production

| candidate | return | vol | Sharpe | max-DD improvement | beta | raw alpha | ex-ante TRE | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / 0.4 | -54.4 bp | -81.0 bp | +0.0529 | +3.39 pp | -0.111 | +7.1 bp | +42.9 bp | -10.0 pp |
| 1 / 0.6 | -81.6 bp | -110.7 bp | +0.0641 | +5.68 pp | -0.136 | -7.6 bp | +52.4 bp | -52.1 pp |
| 25 / 0.6 | -34.5 bp | -49.6 bp | +0.0292 | +3.80 pp | -0.054 | -6.0 bp | +17.4 bp | -49.7 pp |
| 100 / 0.2 | +30.7 bp | +34.7 bp | -0.0096 | -0.32 pp | +0.055 | +5.3 bp | -15.3 bp | +80.5 pp |

## Full 32-cell grid

`TO` is turnover utility weight. Alpha is the raw-return regression intercept described above.

| TRE | TO | Return | Vol | Sharpe | Max DD | Beta | Alpha | Ex-ante TRE | Ex-post TRE | Turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2 | 7.44% | 6.70% | 1.110 | -21.5% | 0.757 | 3.20% | 3.42% | 3.57% | 178.1% |
| 10 | 0.2 | 7.37% | 7.08% | 1.040 | -23.6% | 0.808 | 2.86% | 3.34% | 3.38% | 183.9% |
| 25 | 0.2 | 7.69% | 7.19% | 1.071 | -23.1% | 0.823 | 3.12% | 3.19% | 3.25% | 185.7% |
| 50 | 0.2 | 7.98% | 7.35% | 1.085 | -23.5% | 0.848 | 3.27% | 3.03% | 3.10% | 185.7% |
| 100 | 0.2 | 8.24% | 7.60% | 1.084 | -24.0% | 0.890 | 3.31% | 2.79% | 2.79% | 188.5% |
| 250 | 0.2 | 8.04% | 8.16% | 0.985 | -27.2% | 0.958 | 2.80% | 2.53% | 2.47% | 200.2% |
| 500 | 0.2 | 7.99% | 8.50% | 0.940 | -27.7% | 1.000 | 2.54% | 2.40% | 2.28% | 204.4% |
| 1,000 | 0.2 | 7.87% | 8.68% | 0.907 | -28.4% | 1.021 | 2.32% | 2.32% | 2.16% | 209.2% |
| 1 | 0.4 | 7.39% | 6.45% | 1.146 | -20.3% | 0.724 | 3.33% | 3.37% | 3.61% | 98.0% |
| 10 | 0.4 | 7.32% | 6.66% | 1.099 | -21.2% | 0.754 | 3.11% | 3.29% | 3.45% | 103.7% |
| 25 | 0.4 | 7.69% | 7.08% | 1.086 | -23.4% | 0.808 | 3.17% | 3.09% | 3.21% | 106.1% |
| 50 | 0.4 | 7.93% | 7.26% | 1.093 | -23.7% | 0.835 | 3.26% | 2.95% | 3.03% | 108.0% |
| 100 | 0.4 | 7.94% | 7.68% | 1.034 | -24.6% | 0.890 | 2.98% | 2.74% | 2.72% | 110.6% |
| 250 | 0.4 | 8.08% | 8.11% | 0.996 | -25.0% | 0.953 | 2.81% | 2.49% | 2.37% | 111.5% |
| 500 | 0.4 | 7.96% | 8.36% | 0.952 | -26.9% | 0.986 | 2.54% | 2.38% | 2.19% | 113.0% |
| 1,000 | 0.4 | 7.99% | 8.48% | 0.943 | -27.4% | 1.001 | 2.49% | 2.30% | 2.11% | 111.4% |
| 1 | 0.6 | 7.12% | 6.15% | 1.157 | -18.0% | 0.699 | 3.18% | 3.47% | 3.53% | 56.0% |
| 10 | 0.6 | 7.13% | 6.40% | 1.114 | -18.4% | 0.727 | 3.05% | 3.40% | 3.46% | 59.2% |
| 25 | 0.6 | 7.59% | 6.76% | 1.123 | -19.9% | 0.781 | 3.20% | 3.12% | 3.13% | 58.3% |
| 50 | 0.6 | 7.85% | 7.32% | 1.073 | -22.3% | 0.850 | 3.09% | 2.89% | 2.84% | 60.4% |
| 100 | 0.6 | 7.79% | 7.85% | 0.992 | -24.5% | 0.921 | 2.67% | 2.65% | 2.44% | 58.0% |
| 250 | 0.6 | 8.12% | 8.18% | 0.992 | -25.0% | 0.963 | 2.77% | 2.44% | 2.21% | 59.4% |
| 500 | 0.6 | 8.02% | 8.31% | 0.965 | -26.1% | 0.982 | 2.58% | 2.34% | 2.05% | 58.4% |
| 1,000 | 0.6 | 7.99% | 8.43% | 0.948 | -27.1% | 1.001 | 2.47% | 2.27% | 2.02% | 59.6% |
| 1 | 0.8 | 6.50% | 6.08% | 1.068 | -17.7% | 0.683 | 2.69% | 3.49% | 3.52% | 23.1% |
| 10 | 0.8 | 6.89% | 6.41% | 1.074 | -18.4% | 0.732 | 2.81% | 3.29% | 3.24% | 23.2% |
| 25 | 0.8 | 7.13% | 6.59% | 1.081 | -18.7% | 0.760 | 2.89% | 3.14% | 3.04% | 22.1% |
| 50 | 0.8 | 7.28% | 7.19% | 1.013 | -20.7% | 0.842 | 2.59% | 2.82% | 2.61% | 22.2% |
| 100 | 0.8 | 7.34% | 7.78% | 0.943 | -23.7% | 0.920 | 2.25% | 2.50% | 2.18% | 22.6% |
| 250 | 0.8 | 7.19% | 8.15% | 0.882 | -25.5% | 0.970 | 1.84% | 2.32% | 2.06% | 23.0% |
| 500 | 0.8 | 7.44% | 8.25% | 0.901 | -25.9% | 0.983 | 2.00% | 2.25% | 1.82% | 22.1% |
| 1,000 | 0.8 | 7.65% | 8.27% | 0.924 | -26.3% | 0.986 | 2.19% | 2.22% | 1.86% | 25.8% |

All 32 raw alpha p-values are below `6.42e-5` under the sweep's quarterly raw-return regression
convention. The PDF rounds them to 0.00; the CSV retains full precision.

## Solver and input diagnostics

The run did not silently treat successful artifact generation as a clean numerical log:

- 9,183 solver calls completed; one was rejected as a warning-level infeasible fallback and
  zero were rejected for a numerical blow-up. The exact fallback rate is about 0.0109% (the
  logger displays 0.0%).
- The rejected solve occurred on 2013-05-31. The displayed feasibility repair was at numerical
  tolerance: Insurance-Linked group cap 0.0800 and one minimum weight differed below displayed
  four-decimal precision. The optimiser returned the prior weights (`weights_0`) for that date.
- From the sweep's deterministic nested-loop order, the rejection belongs to cell 17,
  **TRE 1 / turnover 0.6**, the maximum-Sharpe cell. There were 16 completed cell-end warnings
  before the rejection and the 17th cell is the first cell in the turnover-0.6 row.
- 303 of 9,088 TAA rebalances applied a group-bound relaxation: Fixed Income group maximum on
  192 and Insurance-Linked group maximum on 111; the largest single relaxation was 0.0084.
- The raw covariance was diagnosed as ill-conditioned on 9,088/9,088 TAA solves. The worst raw
  minimum eigenvalue was `-5.63e-15`, numerical-zero scale. The most frequent reported
  collinear pair was `BNSGWUH ID Equity` / `LGGOBIU LE Equity` on 2,816 solves.
- The SAA benchmark lay outside the pointwise TAA box on 9,088/9,088 rebalances; index 152 was
  below its floor on 9,024 and index 5 exceeded its cap on 7,200.
- Group reachability failed on one of 9,088 rebalances, the same Insurance-Linked cap event.
- Ninety-five aligned constraint sets dropped a zero-loading Liquidity group.
- The log captured 555 warnings: 522 expected FactorLasso warmup-zeroing messages, 32 expected
  early-history unpriced-instrument messages (those weights remained in cash), and one
  early-history unpriced static-benchmark message.

The pervasive conditioning, benchmark-box, and group-relaxation findings are the same broad
production input/constraint geometry documented in the prior De-PC1 production report. No causal
claim is made that they are introduced by De-PC1; attribution would require a matched same-process
raw-correlation sweep.

## Acceptance and verification

| check | measured | tolerance | status |
|---|---:|---:|---|
| runner lint | 0 findings | 0 | PASS |
| focused TRE-sweep regression | 1 passed | all pass | PASS |
| local FactorLasso import | development checkout | exact local root | PASS |
| only `LassoModel` field changed | 1: `cluster_correlation_transform` | exactly 1 | PASS |
| pipeline exit code | 0 | 0 | PASS |
| requested grid rows | 32/32 | 32 | PASS |
| non-finite numeric values | 0 | 0 | PASS |
| sweep PDF count | 1 non-empty PDF | exactly 1 | PASS |
| sweep PDF size | 83,046 bytes | > 0 | PASS |
| sweep PDF pages | 3 | 3 | PASS |
| rendered pages inspected | 3/3 | 100% | PASS |
| clipping, overdraw, or unreadable tables | 0 pages | 0 | PASS |
| execution Excel files | 0 | 0 requested | PASS |
| wall-clock runtime | 20 min 17 sec | reported | PASS |
| numerical solver blow-ups | 0 | 0 | PASS |
| warning-level fallback | 1/9,183 | disclosed, not silent | PASS WITH DEVIATION |
| production alpha convention | sweep uses raw; factsheet uses excess | must be labelled | PASS WITH DEVIATION |

Verification commands:

```powershell
C:\Users\artur\OneDrive\analytics\my_github\OptimalPortfolios\.venv\Scripts\ruff.exe check --isolated --select E,F,W --line-length 100 rosaa\products\funds\run_tre_sweep_depc1.py
$env:PYTHONPATH='C:\Users\artur\OneDrive\analytics\my_github\OptimalPortfolios\src;C:\Users\artur\OneDrive\analytics\my_github\FactorLasso;C:\Users\artur\OneDrive\analytics\my_github\QuantInvestStrats\src;C:\Users\artur\OneDrive\analytics\my_github\BloombergFetch'
C:\Users\artur\OneDrive\analytics\my_github\OptimalPortfolios\.venv\Scripts\python.exe -m pytest rosaa\tests\minimal_mandate_test.py -q -k tre_sweep
```

The final focused pytest invocation completed with one passing test. A bare-shell verification
attempt without the local development `PYTHONPATH` was discarded at collection with
`ModuleNotFoundError: qis`; repeating under the exact pipeline environment produced the pass
shown above. The earlier pre-run invocation also passed, with only existing Pandas-4 deprecation
warnings beyond the pass marker.

PDF verification used Poppler `pdfinfo` followed by a full three-page PNG render at 140 DPI.
The PDF is unencrypted, PDF 1.4, landscape 1152 × 864 points, and contains no forms, JavaScript,
or suspect objects. All three rendered pages were inspected at original resolution.

## Deliverables and provenance

| artifact | SHA-256 |
|---|---|
| `mac_depc1_constraint_tre_turnover_table_20260816_0925.pdf` | `2f2e06c84ad46697c97a227adcc850cd4dc6f7e0a2a12a11b7015582cd2f7c6e` |
| `tre_grid_metrics.csv` | `c554ab9eb68c3ac34e97fb860837ff0e049ec988ef3a33ab30cc3bac0ef43ec5` |
| `selected_cells.csv` | `360921a3acf6ce20294662f8892032d91376d983a92ccdbafc7c49429132a92a` |
| `run_manifest.csv` | `53ce297ff9ef3d07cdfe492fed7f72fae587a4fd53835c564ac9b8eeaa53682a` |
| `sweep_run_20260816_depc1_tre_grid.log` | `234b23ad700564809d372b0032a1be608be21a14d171240eeff8c3702dfc4cb4` |
| FactorLasso `cluster_utils.py` | `86bf04e5965a787ab6d2b5bf6a8f914127c723575686c42cd902ccde24f350b3` |
| `run_tre_sweep_depc1.py` | `b24213f01411440328e1a1f06030cb83bb715147c75116bce2cc97a3b2ce6536` |

Output directory:
`C:/Users/artur/OneDrive/analytics/outputs/depc1_mac_tre_sweep_20260816/`

The `OptimalPortfolios` checkout remains on `main` with no tracked changes from this run because
`rosaa/` and `papers/cluster_lineage_2026/` are deliberately ignored. During final verification,
an unrelated concurrent process created the untracked root file `.codex_msg_reader.py`; it was
not created, edited, deleted, staged, or committed by this run. The pre-existing, unstaged QIS
report-only changes remain on the QIS `main` checkout; this run staged or committed nothing.

## Open items

1. A production decision between 1/0.6, 1/0.4, 25/0.6, and the current 50/0.4 is an owner choice;
   the grid establishes the trade-off but does not silently change production.
2. If production-comparable alpha is required for every cell, re-emit only the metric layer with
   rates data enabled or persist the per-cell NAVs for post-processing. No covariance/model refit
   is conceptually required, but the current runner did not persist those NAVs after process exit.
3. If the exact maximum-Sharpe cell is to be promoted, repeat that cell with an explicitly
   recorded feasibility-rescue policy or a matched tolerance sensitivity, because one of its 284
   historical rebalances held prior weights after a marginal infeasibility.
