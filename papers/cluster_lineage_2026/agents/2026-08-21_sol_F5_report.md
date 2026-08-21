# F5 report — P1--P7 theory scorecard and joint F5/F6 gate request

**Date:** 2026-08-21  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v2  
**Status:** COMPLETE; JOINT OWNER GATE F5/F6 PENDING

## Execution and outputs

Runner: `papers/cluster_lineage_2026/replication/run_f5_scorecard.py`.  
Focused test: `replication/f5_scorecard_test.py`.

Output directory:
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/f5/`.

The central artifact is `theory_scorecard.csv`, with nine fully sourced rows covering
P1--P7. Supporting artifacts are `p7_turnover_bootstrap.csv`, `source_manifest.csv`,
`run_parameters.csv`, `acceptance.csv`, and `determinism.csv`. Every scorecard row contains
the exact source path and SHA-256 digest of every contributing artifact.

The only new inference in F5 is the permitted P7 component bootstrap: circular moving
blocks of length 6, 2,000 draws, seed 20260813, with the same sampled dates applied jointly
to reassignment, signal, total, and signed interaction turnover.

## Scorecard verdicts

| Prediction | Statistic and principal measured value | Verdict |
|---|---|---|
| P1 empirical | Predicted/realised churn correlation: equity full 0.863096, equity headline 0.871747, futures 0.864768, funds 0.911033 | SUPPORTED |
| P1 synthetic | Gaussian flat correlation 0.979820; numerical-delta monotonicity violations 0; Ward correlation 0.974863, MAE 0.032827 | SUPPORTED; Ward descriptive |
| P2 | Level-calibration knee in 3/4 frontiers; equity headline knee is fixed delta 0.05 rather than calibrated 0.0866 | SUPPORTED with stated headline/full difference |
| P3 | Absorbed constants c range 0.808221--2.153557 | SUPPORTED-REVISED |
| P4 original | Full-panel mean realised-minus-predicted ratio gaps: equity 0.200858, futures 0.418458 | REJECTED |
| P4 revised | Cross-config correlations: equity full 0.991636, equity headline 0.992536, futures 0.938516; funds 0.926193 descriptive | SUPPORTED-REVISED |
| P5 | Max relative Frobenius 0.057222; max relative entry 0.064093; max residual-diagonality change 0.013975 | SUPPORTED |
| P6 | Smoothed consecutive-ARI subsample ranges exceed their baseline ranges on all panels | SUPPORTED |
| P7 | Reassignment falls monotonically, but fund signal turnover also changes significantly and U1 net return decreases at the point estimate | REJECTED |

P1's 95% block intervals are [0.850941, 0.871491] for equity full,
[0.864663, 0.878447] for equity headline, [0.814225, 0.903031] for futures,
and [0.885480, 0.931156] for funds.

P6's consecutive-ARI ranges across halves and thirds are 0.922--0.970 versus
0.545--0.673 baseline for equities, 0.977--0.992 versus 0.891--0.951 for futures,
and 0.964--0.979 versus 0.791--0.831 for funds. Crisis rows remain in the F1 source and
are not treated as independent samples.

## P7 decomposition

All values below are adopted-smoothed minus baseline-cluster turnover per rebalance.

| Panel | Component | Delta | 95% block CI | Excludes zero |
|---|---|---:|---:|---|
| Equity | reassignment | -0.117590 | [-0.126787, -0.109330] | Yes |
| | signal | -0.001498 | [-0.004980, +0.001836] | No |
| | total | -0.072877 | [-0.079807, -0.066893] | Yes |
| | signed interaction | +0.046212 | [+0.041301, +0.051598] | Yes |
| Futures | reassignment | -0.055340 | [-0.065512, -0.046841] | Yes |
| | signal | +0.005225 | [-0.000598, +0.011336] | No |
| | total | -0.030535 | [-0.040410, -0.021888] | Yes |
| | signed interaction | +0.019580 | [+0.014605, +0.025439] | Yes |
| Funds | reassignment | -0.087273 | [-0.100060, -0.073903] | Yes |
| | signal | -0.008852 | [-0.016593, -0.001238] | Yes |
| | total | -0.053125 | [-0.064407, -0.042158] | Yes |
| | signed interaction | +0.043000 | [+0.033250, +0.052905] | Yes |

Reassignment turnover is monotone over the available M1 delta grid in all three panels,
and its adopted-minus-baseline interval excludes zero in the negative direction everywhere.
The signal-invariance clause fails on funds. The net-performance clause also fails as a
literal point-ordering statement because equity annual net return changes by -0.002822
([-0.010391, +0.003782]); futures changes by +0.000935 and funds by +0.004442. The full
P7 conjunction is therefore rejected even though its reassignment mechanism is strongly
supported. The residual is retained as the signed trade-interaction term under the owner
ruling; no retired residual guard is applied.

## Acceptance checks

| Check | Measured | Tolerance | Result |
|---|---:|---:|---|
| Predictions represented | 7 | 7 | PASS |
| Scorecard rows | 9 | 9 | PASS |
| Named source artifacts present | 17/17 | 17/17 | PASS |
| P1 0.863/0.872 precise regression error | `1.221245e-15` | `<=1e-12` | PASS |
| P3 minimum c, rounded | 0.81 | 0.81 | PASS |
| P3 maximum c, rounded | 2.15 | 2.15 | PASS |
| P4 0.20/0.42 rounded regression error | 0 | 0 | PASS |
| P5 residual-diagonality maximum, rounded | 0.014 | 0.014 | PASS |
| P6 panel/config ARI ranges | 6 | 6 | PASS |
| Bootstrap block length | 6 | 6 | PASS |
| Bootstrap draws | 2,000 | 2,000 | PASS |
| Bootstrap seed | 20260813 | 20260813 | PASS |
| Deterministic artifacts | 5/5 byte-identical | 5/5 | PASS |
| Focused pytest | 3 passed | all pass | PASS |
| Isolated Ruff E/F/W | 0 findings | 0 | PASS |

No estimator was fit on market data, no empirical search was performed, and no git staging
or push occurred.

## GATE REQUEST — F5 and F6 jointly

Please rule jointly on the F5 scorecard verdicts above and the 21 frozen application CIs in
`agents/2026-08-20_sol_F6_report.md`.

Two U1 provenance items require an explicit ruling before F8 freezes manuscript exhibits:

1. F0 reran the missing U1 signal NAV/weight series under the owner instruction to rerun
   missing data. The executable regenerated result differs from the August 17 narrative:
   cluster-minus-global is +32.3 bp return, -4.002 pp volatility, and -0.1269 Sharpe;
   cluster-minus-BICS is -10.0 bp, -1.411 pp, and -0.0758. The missing earlier NAV cannot
   support bootstrap inference.
2. The regenerated U1 signal and risk helpers label their result as the headline window but
   calculate performance over the full 2006-08-02--2026-08-05 NAV, including flat warmup
   and out-of-headline observations (240 monthly observations). Please rule whether this
   vintage is frozen as regenerated or whether an explicitly windowed rerun must supersede
   it before F8. U2/U3 signal rows use the intended 2009-08-26--2026-06-24 window; U3 risk
   is explicitly windowed to 2009-09-02--2026-06-24.

Under the dispatch, F7 source reconstruction and F10 replication-statement drafting may
continue while this joint gate is pending. F8 and F9 remain held.
