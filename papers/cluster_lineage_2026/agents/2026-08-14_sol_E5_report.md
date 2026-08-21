# Stage E5 execution report - U2/U3 momentum backtest arm

**Date:** 2026-08-14  
**Roadmap:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`  
**Owner dispatch:** `2026-08-14_owner_E3_gate.md`  
**Status:** COMPLETE with one explicit acceptance deviation

## Execution surface

Runner: `papers/cluster_lineage_2026/replication/run_backtests.py`  
Independent validator: `papers/cluster_lineage_2026/replication/validate_e4_e5.py`  
Cache root: `C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\<universe>\<config>\`  
Evidence roots: `C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\backtests\futures\`
and `...\backtests\mac\`

Each evidence directory contains nine deterministic CSV tables and one alpha-profile PDF.
The U2 profile contains EW-all, both yardsticks, and all eight in-band cluster configurations.
The U3 profile contains EW-all, both yardsticks, and all seven in-band cluster configurations;
rejected M1-star is excluded as ruled. The frozen best pairs are U2 M1-star and U3 M1-0.05.

All legs use the same causal raw log-return-sum score panel, percentile ranking within the
specified groups, `rank >= 1-q`, equal weight across selected assets, and one-period
implementation lag through `qis.backtest_model_portfolio`. The primary q is 0.20; q=1/3 and
span-13 volatility-adjusted scores are the robustness passes. U2 costs are 20 bp and U3 costs
50 bp. U3's quarterly sleeve selection is frozen between quarter dates inside the monthly
schedule. The final 2026-07-31 U3 weights cannot trade with lag one because no later price is
present, so qis drops that one terminal weight date and reports 2026-06-30 as the last traded
weight date.

## Primary payoff results

All return, volatility, alpha, and Sharpe figures below are annualized; Sharpe is the explicit
zero-risk-free convention. Turnover is annualized one-way turnover and cost drag is bp/year.

| Universe | Leg | net return | vol | Sharpe | alpha vs EW | beta vs EW | turnover | cost drag |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| U2 | global | 3.0639% | 8.6470% | 0.3935 | 0.3697% | 1.3054 | 0.9890 | 40.51 |
| U2 | taxonomy | 2.4574% | 6.0685% | 0.4315 | 0.2206% | 1.0321 | 1.0766 | 43.96 |
| U2 | cluster baseline | 2.3097% | 5.7071% | 0.4299 | 0.0239% | 1.0456 | 1.0809 | 44.07 |
| U2 | cluster M1-star | 2.4032% | 5.7407% | 0.4438 | 0.1075% | 1.0501 | 1.0085 | 41.15 |
| U2 | EW-all | 2.2078% | 5.1869% | 0.4480 | 0.0000% | 1.0000 | 0.0952 | 3.87 |
| U3 | global | 5.4850% | 11.9329% | 0.5085 | -0.8274% | 1.1049 | 2.2551 | 239.06 |
| U3 | taxonomy | 4.8987% | 10.0650% | 0.5270 | -1.0325% | 1.0154 | 2.0965 | 221.32 |
| U3 | cluster baseline | 3.7433% | 8.8887% | 0.4592 | -1.7430% | 0.9329 | 2.7836 | 291.72 |
| U3 | cluster M1-0.05 | 4.1875% | 8.8456% | 0.5095 | -1.2973% | 0.9298 | 2.3730 | 249.15 |
| U3 | EW-all | 5.9790% | 9.0894% | 0.6863 | 0.0000% | 1.0000 | 0.1597 | 16.88 |

Relative to the baseline cluster leg, the frozen best improves net annualized return by 9 bp,
Sharpe by 0.014, and turnover by 0.072 in U2. In U3 it improves return by 44 bp, Sharpe by
0.050, and turnover by 0.411. The best cluster leg beats EW-all in U2 return but not Sharpe;
in U3 it remains below EW-all on return and Sharpe. It also remains below the taxonomy leg's
Sharpe in both universes, narrowly in U3.

## Turnover decomposition and acceptance deviation

Metric 11 is retained exactly as frozen: reassignment and signal components form a
triangle-inequality bound, with signed residual `total - reassignment - signal`.

| Universe | Leg | reassignment | signal | total | residual | absolute residual / total | guard |
|---|---|---:|---:|---:|---:|---:|---|
| U2 | baseline | 0.059960 | 0.214024 | 0.245018 | -0.028965 | 11.82% | FAIL |
| U2 | M1-star | 0.020748 | 0.218139 | 0.228612 | -0.010274 | 4.49% | PASS |
| U3 | baseline | 0.149694 | 0.208895 | 0.270903 | -0.087686 | 32.37% | FAIL |
| U3 | M1-0.05 | 0.068282 | 0.203964 | 0.230840 | -0.041405 | 17.94% | FAIL |

U2 passes on seven of eight cluster legs; only baseline misses the 10% tolerance. U3 passes
on zero of seven cluster legs, with absolute residual shares from 13.37% to 32.37%. This is a
genuine rejection of the guard, not a missing category or accounting error: the validator
recomputes the frozen signed identity to 1e-14. A pre-report diagnostic removed an incorrect
fill of prior-partition entrants with current labels; the definition-correct rerun made the
reported residuals slightly larger and confirmed the rejection. No component was redefined.

The direction of Prediction 7's attribution mechanism is supported. Across
`baseline/0.02/0.05/0.10`, reassignment turnover falls monotonically
`0.05996/0.03733/0.02163/0.01521` in U2 and
`0.14969/0.09159/0.06828/0.04684` in U3. Signal turnover stays in the narrow ranges
0.2140-0.2198 and 0.2040-0.2121 respectively. Net performance is non-decreasing across that
delta sequence in U3, but not U2: U2 M1-0.05 is below M1-0.02 before recovering at 0.10.

## Robustness and crises

The best-vs-baseline improvement survives q=1/3 in both universes: U2 Sharpe 0.4635 versus
0.4459 and U3 0.5294 versus 0.4815. Under volatility-adjusted momentum, U3 also survives
(0.5216 versus 0.4743), while U2 reverses (0.4248 versus 0.4377).

During the GFC, best-minus-baseline total return is +0.29 percentage points in U2 and +0.89
points in U3. During COVID it is -1.01 points in U2 and -0.25 points in U3. During 2022 it is
-1.25 points in U2 and +0.27 points in U3. The complete crisis table contains all profile
legs, not only the frozen pairs.

## Acceptance and verification

- Profile completeness: PASS, 11 U2 legs and 10 U3 legs, including EW-all and both yardsticks.
- Shared scores: PASS, sampled maximum absolute raw-score difference 0.0.
- U3 quarterly selection freeze: PASS on every leg/date after eligibility exits are applied.
- Robustness: PASS, 2 variants x 4 headline legs = 8 rows per universe.
- Crisis windows: PASS, GFC/COVID/2022 present for every leg.
- PDFs: PASS, both alpha-profile reports created.
- Determinism: PASS, byte-identical rerun of all 9 CSV files per universe.
- Turnover residual tolerance: REJECTED as measured above.

Independent validation output:

```text
E4: PASS (4 runs; vocabulary present; 3 coverage>=0.70 cases per universe)
E5 futures: PASS evidence; residual guard 7/8 rows PASS, max=0.118217
E5 mac: PASS evidence; residual guard 0/7 rows PASS, max=0.323679
E5 futures deterministic CSV rerun: PASS (9 files)
E5 mac deterministic CSV rerun: PASS (9 files)
```

No files were staged or pushed.

## GATE REQUEST

The owner must rule on:

1. Whether the payoff verdicts are accepted: modest U2 and material U3 improvements of the
   frozen best cluster leg over the baseline cluster leg, but no consistent dominance over
   both yardsticks and a U2 reversal under volatility-adjusted momentum.
2. Whether the frozen metric-11 results may proceed as an explicitly reported empirical
   rejection of the 10% residual guard, or whether the owner wants a separately specified
   alternative turnover attribution before E6.
