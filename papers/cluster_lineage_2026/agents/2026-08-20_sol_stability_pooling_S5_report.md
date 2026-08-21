# Stability-pooled z-score S5 robustness report

**Date:** 2026-08-20  
**Executor:** Sol  
**Proposer:** Ben (Monday TAA meeting)  
**Status:** COMPLETE — turnover robust; Sharpe robustness fails

## Outcome

V3/36 was selected mechanically as the highest full-sample Sharpe among turnover-reducing grid
cells. It is the roadmap's mean-plus-variance comparison arm and breaks strict asset-class
neutrality. Its turnover reduction survives both half-windows and the bootstrap, but its Sharpe
improvement does not: the evaluation-half Sharpe delta is -0.099338 and the full-sample 95% CI is
[-0.036563, 0.164376].

The optional U1/U3 generalisation arm was not run. The roadmap permits it only after a positive
MAC result; the selected MAC cell fails the split-window and Sharpe-CI adoption tests.

## Split-window results

The production covariance and cluster fit remains shared. Portfolio holdings are reset at each
half-window boundary and V0 and V3/36 are sent through the existing optimiser/backtester.

| window | dates | cell | Sharpe | ex-post TE | ex-ante TRE | annual turnover | max DD | Sharpe delta | turnover delta |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| selection | 2004-12-31 to 2015-09-30 | V0 | 0.863005 | 0.033803 | 0.027144 | 0.961114 | -0.249967 | 0.000000 | 0.000000 |
| selection | 2004-12-31 to 2015-09-30 | V3_1x | 1.021084 | 0.041135 | 0.030883 | 0.811521 | -0.211537 | +0.158079 | -0.149593 |
| evaluation | 2015-10-31 to 2026-07-31 | V0 | 1.320608 | 0.031711 | 0.026976 | 0.963485 | -0.076863 | 0.000000 | 0.000000 |
| evaluation | 2015-10-31 to 2026-07-31 | V3_1x | 1.221270 | 0.030606 | 0.027300 | 0.832032 | -0.086522 | -0.099338 | -0.131453 |

The split runner retains the complete cached monthly computed-input schedule. Outside each scored
half, the deliberately truncated benchmark bounds cause rejected diagnostic attempts: 130 dates
from 2015-10 through 2026-07 in each selection run, and 2005-01 in each evaluation run. These lie
strictly outside the corresponding reported window. Rejected/infeasible fallbacks inside the two
active scored windows are 0.

## Moving-block bootstrap

The paired circular moving-block bootstrap uses block length 6, 2,000 draws, and seed 20260813.
Intervals are percentile draws centered on the production-convention point delta.

| contrast | metric | estimate | 95% CI | excludes zero |
|---|---|---:|---:|---|
| V3_1x minus V0 | TAA Sharpe | +0.069920 | [-0.036563, +0.164376] | no |
| V3_1x minus V0 | annual turnover | -0.158152 | [-0.246517, -0.073797] | yes |

## Acceptance

| check | measured | tolerance | status |
|---|---:|---:|---|
| split windows | 2 | 2 | PASS |
| active-window rejected/infeasible fallbacks | 0 | 0 | PASS |
| turnover delta in selection half | -0.149593 | below 0 | PASS |
| turnover delta in evaluation half | -0.131453 | below 0 | PASS |
| Sharpe delta in selection half | +0.158079 | at or above 0 | PASS |
| Sharpe delta in evaluation half | -0.099338 | at or above 0 | FAIL |
| Sharpe bootstrap lower bound | -0.036563 | at or above 0 | FAIL |
| turnover bootstrap upper bound | -0.073797 | below 0 | PASS |
| bootstrap parameters | block 6; 2,000 draws; seed 20260813 | exact | PASS |
| optional generalisation gate | MAC not robustly positive | positive MAC required | NOT TRIGGERED |

Evidence files are `split_window.csv` and `bootstrap.csv` under
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/stability_pooling/mac/`.
No files were staged, committed, pushed, tagged, or released.
