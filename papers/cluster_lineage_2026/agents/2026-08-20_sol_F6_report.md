# F6 report — bootstrap confidence intervals for the applications

**Date:** 2026-08-20  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v2  
**Status:** COMPLETE WITH PROVENANCE DEVIATION FOR JOINT F5/F6 GATE

## Execution and outputs

Runner: `papers/cluster_lineage_2026/replication/run_f6_bootstrap.py`.
Focused test: `replication/f6_bootstrap_test.py`.

Output directory:
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/f6/`.

The two manuscript artifacts are `signal_cis.csv` and `risk_cis.csv`. Supporting files are
`point_regression.csv`, `source_manifest.csv`, `acceptance.csv`, and `determinism.csv`.
The method is the frozen joint circular moving-block bootstrap: block length 6, 2,000 draws,
seed 20260813, percentile 95% intervals. Within each comparison the same resampled monthly
indices are applied to both legs. No CI is computed against EW-all.

## Signal intervals

| Comparison | Metric | Point | 95% CI | Excludes zero |
|---|---|---:|---:|---|
| U1 cluster - global | annual net return | +0.003227 | [-0.020441, +0.023258] | No |
| | annual volatility | -0.040019 | [-0.051600, -0.027369] | Yes |
| | RF=0 Sharpe | -0.126946 | [-0.298045, +0.056014] | No |
| U1 cluster - BICS sector | annual net return | -0.001004 | [-0.020288, +0.015665] | No |
| | annual volatility | -0.014115 | [-0.025323, -0.004016] | Yes |
| | RF=0 Sharpe | -0.075833 | [-0.278744, +0.126600] | No |
| U2 cluster - global | annual net return | +0.000200 | [-0.011931, +0.011570] | No |
| | annual volatility | -0.006762 | [-0.011945, -0.001219] | Yes |
| | RF=0 Sharpe | +0.028939 | [-0.093163, +0.135718] | No |
| U3 cluster - global | annual net return | -0.005961 | [-0.030084, +0.019172] | No |
| | annual volatility | -0.024198 | [-0.032652, -0.016462] | Yes |
| | RF=0 Sharpe | +0.090821 | [-0.126888, +0.299028] | No |

The common robust signal result is lower volatility: all four candidate-minus-control
volatility intervals exclude zero in the negative direction. The return and Sharpe
intervals do not exclude zero.

## Risk-allocation intervals

| Comparison | Metric | Point | 95% CI | Excludes zero |
|---|---|---:|---:|---|
| U1 Rolling-Ward HRP - flat ERC | annual net return | +0.001701 | [-0.006925, +0.010111] | No |
| | annual volatility | -0.003958 | [-0.008211, -0.000695] | Yes |
| | RF=0 Sharpe | +0.031123 | [-0.039944, +0.128673] | No |
| U1 Rolling-Ward HRP - single-link HRP | annual net return | +0.000977 | [-0.003361, +0.005765] | No |
| | annual volatility | -0.001176 | [-0.002790, +0.000124] | No |
| | RF=0 Sharpe | +0.013372 | [-0.024717, +0.065627] | No |
| U3 equal-cluster RB - flat ERC | annual net return | +0.003927 | [+0.000386, +0.007725] | Yes |
| | annual volatility | +0.001451 | [+0.000141, +0.002950] | Yes |
| | RF=0 Sharpe | +0.174855 | [-0.045545, +0.367221] | No |

U1 Rolling-Ward HRP's volatility reduction versus flat ERC excludes zero. U3 equal-cluster
risk budgeting has a positive return interval, accompanied by a positive volatility
interval; its Sharpe interval includes zero. The U2 risk limitation remains descriptive and
has no CI row, as required.

## Rerun consequence and sample provenance

The owner directed that missing frozen series be rerun. F0 therefore regenerated the absent
U1/U2/U3 signal NAVs and U1/U3 risk NAVs from the recorded runners and caches before this
stage. F6 matches those regenerated `performance.csv` tables exactly, but one material
historical discrepancy must be gated rather than hidden:

1. The regenerated U1 signal point estimates differ from the 2026-08-17 narrative. The
   current cluster-minus-global deltas are +32.3 bp return, -4.002 pp volatility, and
   -0.1269 Sharpe; cluster-minus-BICS is -10.0 bp, -1.411 pp, and -0.0758. The August 17
   report quoted +69.1 bp/-1.92 pp/+0.0003 and +26.8 bp/+0.67 pp/+0.051, respectively.
   The earlier NAV file was missing and cannot serve as the bootstrap source; the rerun is
   the only executable artifact available under the owner's instruction.
2. The regenerated U1 signal and risk NAV files span 2006-08-02 through 2026-08-05 and the
   frozen performance helper includes that full NAV range, including flat warmup/out-of-
   headline observations, despite the rows being labelled
   `headline_20090831_20260630`. This is why F6 reports 240 monthly observations for U1.
   U2/U3 signal rows use 202 monthly observations from 2009-08-26 through 2026-06-24;
   the explicitly windowed U3 risk row uses 201 observations from 2009-09-02 through
   2026-06-24.

F6 follows its binding acceptance requirement by reproducing the recorded source deltas;
it does not manufacture a different point estimate by silently trimming the series. The
joint F5/F6 gate must decide whether the regenerated U1 vintage supersedes the old narrative
and whether the U1 performance-window defect is accepted as frozen or corrected by an
explicitly authorised rerun.

## Acceptance checks

| Check | Measured | Tolerance | Result |
|---|---:|---:|---|
| F0 sources resolved once | 10/10 | 10/10 | PASS |
| Signal CI rows | 12 | 12 | PASS |
| Risk CI rows | 9 | 9 | PASS |
| Total CI rows | 21 | exactly 21 | PASS |
| Maximum frozen-point regression error | `7.399636e-14` | `<=1e-10` | PASS |
| Block length | 6 | 6 | PASS |
| Bootstrap draws | 2,000 | 2,000 | PASS |
| Seed | 20260813 | 20260813 | PASS |
| NaNs across CI tables | 0 | 0 | PASS |
| Deterministic artifacts | 5/5 byte-identical | 5/5 | PASS |
| Focused pytest | 3 passed | all pass | PASS |
| Isolated Ruff E/F/W | 0 findings | 0 | PASS |

No backtest or estimator was run in F6 itself; it consumed the F0-regenerated NAVs. No git
staging or push occurred.

## GATE REQUEST

Deferred for the required joint F5/F6 gate after F5 is complete. The gate must explicitly
rule on the two U1 provenance deviations above before manuscript integration.
