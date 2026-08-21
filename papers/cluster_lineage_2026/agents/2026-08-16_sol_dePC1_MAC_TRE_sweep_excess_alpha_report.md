# De-PC1 MAC TRE sweep excess-alpha correction report

**Date:** 2026-08-16  
**Executor:** sol  
**Status:** COMPLETE - production alpha convention reconciled exactly; corrected 32-cell sweep emitted  
**Scope:** owner-requested ROSAA sweep-reporting correction for the frozen De-PC1 MAC grid  
**Repository actions:** no staging, commit, push, tag, publication, or release

## Outcome

`rosaa.core.calibration.run_tre_range_sweep` now constructs its performance parameters with
`add_rates_data=True`, exactly as `rosaa.core.report.reporting.generate_report` does. Sweep beta,
annual alpha, and alpha p-value therefore use the production excess-return regression:

`TAA - risk-free = alpha + beta * (static benchmark - risk-free) + residual`.

The correction is reporting-only. It does not enter covariance estimation, clustering, alpha
scores, constraints, optimization, weights, returns, or turnover. A full end-to-end rerun proves
this numerically: all seven non-regression columns are exactly equal, with maximum absolute
difference `0.0`, in every one of the 32 old-versus-corrected cells.

At the frozen production grid point, TRE utility 50 and turnover utility 0.4, the corrected sweep
now agrees exactly with an independent recomputation from the saved production NAVs:

| statistic vs static benchmark | prior raw-return sweep | corrected excess-return sweep | independent production-NAV calculation |
|---|---:|---:|---:|
| beta | 0.8350469418110487 | 0.8257674037704847 | 0.8257674037704847 |
| annual alpha | 3.25752869111248% | 3.01162056544189% | 3.01162056544189% |
| p-alpha | 1.54127078554456e-07 | 3.52516984521047e-07 | 3.52516984521047e-07 |

Thus the sweep now reports the same underlying value as the production factsheet's displayed
3.0% alpha and 0.83 beta. The production-NAV check used the saved `TAA` and `Benchmark` columns,
the production window 2004-12-31 through 2026-07-31, rates-aware `PerfParams`, and quarterly
(`QE`) regression frequency. The regression contained 86 quarterly observations.

The grid selection is unchanged because Sharpe, return, volatility, drawdown, tracking error, and
turnover are unchanged. The maximum-Sharpe cell remains TRE 1 / turnover 0.6 with Sharpe
1.157484; the production cell remains Sharpe 1.093388.

## Implementation

Changed files:

- `rosaa/core/calibration.py`: the TRE sweep wrapper now passes `add_rates_data=True`; its
  docstring states that alpha is the rates-aware excess-return Jensen alpha versus the static
  benchmark.
- `rosaa/tests/minimal_mandate_test.py`: added
  `test_tre_sweep_uses_the_production_factsheet_perf_params`, which captures the arguments used by
  the sweep and the production factsheet and requires exact equality plus
  `add_rates_data is True`.
- `../../../rosaa/products/funds/analysis/run_tre_sweep_depc1.py`: moved corrected artifacts to a separate output
  directory and recorded `reporting_frequency=QE`, `alpha_return_convention=excess`, and
  `reporting_add_rates_data=True` in the manifest.

The regression test was proven before the fix: it failed with
`{'add_rates_data': False} != {'add_rates_data': True}`. After the one-line numerical correction,
the focused TRE-sweep test set passed.

## Frozen rerun specification

| item | value |
|---|---|
| production batch | `MAC_CONSTRAINED_BATCH` |
| mandate | `MAC`, constrained |
| signal | `PROD_MOM_BETA_CLUSTER` |
| returns input | `20260810_APAC_ROSAA_Fund_and_Index_Data` |
| factor model | `MATF_CUSTOM` |
| cluster transform | `ClusterCorrelationTransform.REMOVE_PC1` |
| model-field change relative to frozen MAC production | only `cluster_correlation_transform` |
| reporting window | 2004-12-31 through 2026-07-31 |
| alpha/beta regression frequency | quarterly (`QE`) |
| alpha convention | excess return, with rates data |
| TRE grid | 1, 10, 25, 50, 100, 250, 500, 1,000 |
| turnover grid | 0.2, 0.4, 0.6, 0.8 |
| grid size | 32 cells |
| covariance and SAA | fitted once and shared by all cells |
| run id | `20260816_depc1_tre_grid_excess_alpha` |
| runner | `../../../rosaa/products/funds/analysis/run_tre_sweep_depc1.py` |
| output directory | `C:/Users/artur/OneDrive/analytics/outputs/depc1_mac_tre_sweep_excess_alpha_20260816/` |
| cache directory | none; shared fitted inputs held in memory |
| execution Excel | not requested and not emitted |

## Corrected regression grid

The final column is corrected minus the retained raw-return alpha. All 32 regression rows changed
as intended. All p-values remain below `5.76e-05`; the PDF rounds them to 0.00, while the CSV
retains full precision.

| TRE | TO | Beta | Excess alpha | p-alpha | correction vs raw |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2 | 0.747510 | 2.8278% | 1.633e-06 | -37.66 bp |
| 10 | 0.2 | 0.798175 | 2.5715% | 7.052e-06 | -29.17 bp |
| 25 | 0.2 | 0.815090 | 2.8467% | 1.257e-06 | -27.05 bp |
| 50 | 0.2 | 0.839972 | 3.0373% | 2.209e-07 | -22.86 bp |
| 100 | 0.2 | 0.882376 | 3.1498% | 1.875e-08 | -16.08 bp |
| 250 | 0.2 | 0.949984 | 2.7579% | 6.352e-07 | -3.92 bp |
| 500 | 0.2 | 0.991571 | 2.5790% | 3.295e-06 | +3.41 bp |
| 1,000 | 0.2 | 1.013274 | 2.3881% | 2.689e-05 | +6.42 bp |
| 1 | 0.4 | 0.714483 | 2.8958% | 1.288e-06 | -43.31 bp |
| 10 | 0.4 | 0.746533 | 2.7244% | 3.842e-06 | -38.96 bp |
| 25 | 0.4 | 0.799633 | 2.8770% | 1.724e-06 | -29.50 bp |
| 50 | 0.4 | 0.825767 | 3.0116% | 3.525e-07 | -24.59 bp |
| 100 | 0.4 | 0.881517 | 2.8295% | 1.501e-06 | -15.52 bp |
| 250 | 0.4 | 0.945456 | 2.7557% | 6.395e-07 | -4.97 bp |
| 500 | 0.4 | 0.979208 | 2.5362% | 1.987e-06 | +0.07 bp |
| 1,000 | 0.4 | 0.994795 | 2.5142% | 1.323e-06 | +2.54 bp |
| 1 | 0.6 | 0.687256 | 2.7103% | 4.119e-07 | -47.10 bp |
| 10 | 0.6 | 0.716855 | 2.6231% | 2.143e-06 | -42.77 bp |
| 25 | 0.6 | 0.771904 | 2.8599% | 7.549e-08 | -33.75 bp |
| 50 | 0.6 | 0.840197 | 2.8688% | 1.553e-07 | -22.02 bp |
| 100 | 0.6 | 0.912599 | 2.5662% | 7.255e-07 | -10.20 bp |
| 250 | 0.6 | 0.955162 | 2.7400% | 1.666e-07 | -3.09 bp |
| 500 | 0.6 | 0.976897 | 2.5725% | 3.249e-07 | -1.08 bp |
| 1,000 | 0.6 | 0.996371 | 2.4906% | 2.085e-07 | +2.01 bp |
| 1 | 0.8 | 0.675003 | 2.1824% | 5.752e-05 | -51.11 bp |
| 10 | 0.8 | 0.724364 | 2.3774% | 8.195e-06 | -42.92 bp |
| 25 | 0.8 | 0.752184 | 2.5070% | 1.783e-06 | -38.24 bp |
| 50 | 0.8 | 0.837059 | 2.3417% | 1.662e-06 | -25.06 bp |
| 100 | 0.8 | 0.916444 | 2.1288% | 4.565e-06 | -12.18 bp |
| 250 | 0.8 | 0.970429 | 1.7896% | 2.632e-05 | -5.26 bp |
| 500 | 0.8 | 0.980087 | 1.9819% | 3.094e-06 | -1.82 bp |
| 1,000 | 0.8 | 0.983431 | 2.1762% | 1.135e-06 | -1.56 bp |

## Acceptance and verification

| check | measured | tolerance | status |
|---|---:|---:|---|
| pre-fix regression proof | failed on False vs True | must fail before fix | PASS |
| focused TRE-sweep tests after fix | 2 passed | all pass | PASS |
| dedicated runner lint | 0 findings | 0 | PASS |
| pipeline completion | exit code 0 | 0 | PASS |
| corrected grid rows | 32/32 | 32 | PASS |
| non-finite numeric values | 0 | 0 | PASS |
| non-regression columns unchanged | 7/7 columns, 32/32 cells exact | 100% exact | PASS |
| maximum non-regression absolute difference | 0.0 | 0.0 | PASS |
| production alpha reconciliation | exact to stored float | 0 difference | PASS |
| production beta reconciliation | exact to stored float | 0 difference | PASS |
| production p-alpha reconciliation | exact to stored float | 0 difference | PASS |
| corrected regression cells | 32/32 changed | all 32 | PASS |
| corrected PDF count | 1 non-empty PDF | exactly 1 | PASS |
| corrected PDF size | 83,065 bytes | > 0 | PASS |
| corrected PDF pages | 3 | 3 | PASS |
| rendered pages inspected | 3/3 at 140 DPI | 100% | PASS |
| visual defects | 0 | 0 | PASS |
| execution Excel files | 0 | 0 requested | PASS |
| wall-clock runtime | 21 min 09 sec | reported | PASS |
| numerical solver blow-ups | 0 | 0 | PASS |
| warning-level solver fallback | 1/9,183 | disclosed | PASS WITH DEVIATION |

Focused verification output:

```text
..                                                                       [100%]
============================== warnings summary ===============================
rosaa/tests/minimal_mandate_test.py::test_tre_sweep_reuses_the_three_universe_computed_inputs
  Pandas4Warning: 'future.no_silent_downcasting' is deprecated
```

The warning is an existing Pandas-4 deprecation emitted by the pre-existing mocked-input test;
the new performance-parameter regression test emitted no warning.

The independent production reconciliation called
`qis.compute_ra_perf_table_with_benchmark` on the saved production `TAA` and `Benchmark` NAVs
using a fresh call to the same rates-aware production report parameters. It did not read the new
sweep metric row when calculating the reference values.

PDF QA used Poppler `pdfinfo` and a full 140-DPI render. The final PDF is unencrypted PDF 1.4,
landscape 1152 x 864 points, with three pages, no forms, no JavaScript, and no suspect objects.
Every page was inspected at original rendered resolution; tables and labels are legible with no
clipping or overlap.

## Solver and data diagnostics

The correction did not change the prior run's diagnostic pattern:

- 9,183 solves; one warning-level infeasible fallback and zero numerical blow-ups;
- the same marginal fallback on 2013-05-31 in the TRE 1 / turnover 0.6 cell, returning prior
  weights for that rebalance;
- 303 group-bound relaxations, with maximum single relaxation 0.0084;
- 9,088/9,088 raw covariance checks diagnosed as ill-conditioned, with worst minimum eigenvalue
  `-5.63e-15`, at numerical-zero scale;
- 555 captured warnings: 522 frozen FactorLasso warmup-zeroing warnings, 32 early-history
  unpriced-instrument warnings, and one unpriced static-benchmark warning.

These are retained production input and constraint diagnostics. None was introduced by the
rates-aware reporting parameters.

## Deliverables and provenance

| artifact | SHA-256 |
|---|---|
| `mac_depc1_constraint_tre_turnover_table_20260816_1009.pdf` | `00aa23da5400a5d9fba2eecf9c73e2a9081bdc7549f6cd4a5ec8731481bd493c` |
| `tre_grid_metrics.csv` | `cc24b0f2526fbc0c980b711511425ce2662b354c5b3e6a1527506cf2307ae864` |
| `selected_cells.csv` | `3b130a1e8f6138999e21c0a875336e9c8ad3fe6e732cffa8eda59ebb945928c6` |
| `run_manifest.csv` | `17672c66816714e55e207c058f607e2535ed741aedbe5fb06de1f4138ff22ab0` |
| `sweep_run_20260816_depc1_tre_grid_excess_alpha.log` | `108588d1ec4f1f842165b65070cd4e2cb81e0513291e1f46ea12a9861d7e6e3f` |
| `rosaa/core/calibration.py` | `16066aa1fa35d1d7256630020760dbed3acd30c42b3ffb4916b40002e1be0f79` |
| `rosaa/tests/minimal_mandate_test.py` | `6eb5a08524913791e73f893d5b7453a1401a44d4535204517bdf9c7977a5a044` |
| `run_tre_sweep_depc1.py` | `aab2b4b8d639805dcdcc194973f556130eef17348b8ee88bbc4ec11a191d2b0d` |

The original raw-alpha output directory remains untouched at
`C:/Users/artur/OneDrive/analytics/outputs/depc1_mac_tre_sweep_20260816/`. The corrected directory
is
`C:/Users/artur/OneDrive/analytics/outputs/depc1_mac_tre_sweep_excess_alpha_20260816/`.

No files were staged or committed. `rosaa/` and `papers/cluster_lineage_2026/` remain deliberately
gitignored, as required by the owner. No production setting was changed.
