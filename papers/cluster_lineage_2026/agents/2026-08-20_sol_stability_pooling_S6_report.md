# Stability-pooled z-score S6 consolidated report

**REJECT.** Ben's proposed stability pooling reduces annual turnover by 0.158152 in the strongest
full-sample cell, but its Sharpe delta of +0.069920 has a 95% CI of [-0.036563, +0.164376], and
the evaluation-half Sharpe delta is -0.099338. The turnover effect is real; the required
risk-adjusted-performance robustness is not.

**Date:** 2026-08-20  
**Executor:** Sol  
**Proposer:** Ben (Monday TAA meeting)  
**Status:** S0-S6 COMPLETE

## Decision basis

The strongest full-sample row is V3/36: Sharpe 1.138897 versus 1.068978, turnover 0.834140
versus 0.992292, and maximum drawdown -0.211537 versus -0.249967. It is nevertheless a comparison
arm rather than the primary proposal: pooling the mean raises ex-post TE by 0.003102 and ex-ante
TRE by 0.002303 and breaks strict asset-class neutrality.

The variance-only arms are encouraging but not decisive. V1/36, Ben's direct proposal, changes
Sharpe by +0.029652 and annual turnover by -0.088383; V2/36 changes them by +0.018823 and
-0.107604 while leaving both tracking-risk measures essentially unchanged. These full-sample
rows do not override the predeclared robustness gate applied to the strongest cell.

The S4 mechanism evidence is directional rather than clean. For V3/36, annualized reassignment
turnover falls by 0.264602 and signal turnover by 0.206420: reassignment is the larger direct
component, but signal still represents 43.8% of the two absolute reductions. Boundary evidence
is strong—at 36 months, bottom-stability-quartile assets reassign 31.74% of the time versus 4.91%
in the top quartile—and the size-versus-stability confound flag is false.

Robustness decides the result. Turnover falls in both halves and its bootstrap interval excludes
zero. Sharpe rises by 0.158079 in the selection half but falls by 0.099338 in the evaluation half;
its full-sample interval crosses zero. Thus the observed turnover gain costs Sharpe in the
evaluation half at approximately matched tracking risk, meeting the roadmap's explicit rejection
condition.

## Stage record

| stage | outcome | report |
|---|---|---|
| S0 | call sites mapped; definitions feasible | `2026-08-20_sol_stability_pooling_S0_report.md` |
| S1 | additive FactorLasso API and isolated MAC harness | `2026-08-20_sol_stability_pooling_S1_report.md` |
| S2 | fail-before-pass proof; focused and full suites green | `2026-08-20_sol_stability_pooling_S2_report.md` |
| S3 | seven-cell frozen MAC grid complete | `2026-08-20_sol_stability_pooling_S3_report.md` |
| S4 | boundary channel confirmed; turnover channel mixed | `2026-08-20_sol_stability_pooling_S4_report.md` |
| S5 | turnover robust; Sharpe robustness fails | `2026-08-20_sol_stability_pooling_S5_report.md` |

## Baseline and provenance

V0 uses `ClusterCorrelationTransform.NONE` and exactly replays the current scoring path. Its
1.068978 Sharpe matches the 19 August raw-production factsheet's displayed 1.07, along with 447%
cumulative return, 99% turnover, and -25% drawdown. The roadmap's 1.15 reference belongs to the
12 August production vintage; the -0.081022 difference is disclosed in S3. Experimental deltas
remain same-process comparisons against V0.

- Runner: `papers/cluster_lineage_2026/replication/run_stability_pooling_mac.py`.
- Output root:
  `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/stability_pooling/mac/`.
- Manifest: `source_manifest.csv`; parameters: `run_parameters.csv`; acceptance:
  `acceptance.csv`; complete run log: `run_20260820_stability_pooling_mac.log`.
- Runtime FactorLasso:
  `C:/Users/artur/OneDrive/analytics/my_github/FactorLasso/src/factorlasso/__init__.py`.
- Bootstrap: paired circular moving blocks, block 6, 2,000 draws, seed 20260813.
- Optional U1/U3 arm: not triggered because MAC did not pass the positive-result gate.

## Final acceptance

| check | measured | adoption tolerance | status |
|---|---:|---:|---|
| full-sample turnover delta | -0.158152 | below 0 | PASS |
| full-sample Sharpe delta | +0.069920 | at or above 0 | PASS |
| ex-post TE delta | +0.003102 | matched risk | FAIL for the V3 comparison arm |
| mechanism | 56.2% reassignment / 43.8% signal | reassignment-led | PASS WITH CAVEAT |
| turnover split-window | negative in 2/2 halves | 2/2 | PASS |
| Sharpe split-window | nonnegative in 1/2 halves | 2/2 | FAIL |
| turnover 95% CI | [-0.246517, -0.073797] | entirely below 0 | PASS |
| Sharpe 95% CI | [-0.036563, +0.164376] | lower bound at or above 0 | FAIL |
| persistent size-w confound | false at 36 and 72 months | false | PASS |
| production/default changes | 0 | 0 | PASS |
| ROSAA source edits | 0 | 0 | PASS |
| release/tag/version actions | 0 | 0 | PASS |

The opt-in implementation and evidence remain in the local development trees for review; no
production configuration is changed. No files were staged, committed, pushed, tagged, or
released.
