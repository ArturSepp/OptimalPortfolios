# U1 covariance-frequency and EWMA-span grid report

**Date:** 2026-08-15  
**Status:** COMPLETE  
**Runner:** `papers/cluster_lineage_2026/replication/run_u1_covar_grid.py`  
**Validator:** `papers/cluster_lineage_2026/replication/validate_u1_covar_grid.py`

## Outcome

The strongest raw cluster cell is **ME returns, EWMA span 36, q=0.25**. In the U1
headline window it produces 6.2513% annualised net return, 0.501599 Sharpe, and 4.5539
annual one-way turns. It beats the same-q global leg by 7.7 bp/year and 0.026496 Sharpe,
but adds 2.0312 annual turns and 43.0 bp/year of cost drag.

It does **not** beat the best global leg. Global q=0.30 produces 6.6409% net return,
0.512965 Sharpe, and 2.3212 turns. Relative to that benchmark, ME/36/q=0.25 is lower by
39.0 bp/year and 0.011366 Sharpe, with turnover higher by 2.2327 turns. No one of the 140
headline cluster rows beats best-global q=0.30 on both net return and Sharpe.

All four rows that beat their same-q global leg on both headline metrics come from ME/36
(q=0.25, 0.20, 0.15, and 0.10). The result is exploratory: alternative covariance cells
have not been tested against the frozen E3b taxonomy-fidelity band, and selecting the best
of 140 rows is not a statistical or multiple-testing-adjusted claim.

## Frozen design and varied object

The original point-in-time U1 eligibility mask is fixed across every cell, so neither the
cluster leg nor global benchmark receives a different security universe. The following are
also frozen: ME decision dates, 48-week momentum with four-week skip, q grid 0.30/0.25/0.20/
0.15/0.10, primary `group_equal` construction, 10 bp trading cost, implementation lag 1,
and headline/full-panel window separation.

Only the EWMA asset covariance/correlation matrix used to form the unsmoothed baseline
partition changes. The FF6 factor panel and factor-covariance span are not refitted because
they do not determine the cluster-ranking groups in this test. The grid is:

- B and W-MON, W-TUE, W-WED, W-THU, W-FRI: spans 24, 36, 52, 156;
- ME: spans 12, 24, 36, 52.

Spans are native observation counts, not calendar-normalised horizons. Consequently B/156
is roughly seven months, W-*/156 is roughly three years, and ME/36 is three years. The
current result therefore does not isolate sampling cadence from calendar memory. The fact
that ME/36 and W-WED/156 are the two strongest cadence families is consistent with a
roughly three-year useful covariance horizon, but this is only a descriptive reading.

The compact 28-cell membership cache is stored under
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/covariance_frequency_span_grid/msci_us/partitions/`.
It contains 70.284 MiB rather than full factor-model pickles. The complete output root is
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/covariance_frequency_span_grid/msci_us/`.

## Acceptance

| Check | Measured | Tolerance | Verdict |
|:--|--:|--:|:--|
| grid coverage | 28 cells x 5 q x 2 windows = 280 cluster rows | 280 | PASS |
| reconstructed W-WED returns | max absolute error 4.163336e-16; identical NaN mask | <= 1e-15 | PASS |
| fixed eligibility | exact accepted U1 mask | exact | PASS |
| canonical partition regression | W-WED/156 matches 238/238 accepted baseline dates | 238/238 | PASS |
| canonical payoff regression | maximum metric error 4.263256e-13 | <= 1e-12 | PASS |
| construction acceptance | 280/280 | 280/280 | PASS |
| maximum weight-sum error | 2.220446e-16 | <= 1e-12 | PASS |
| maximum group-budget error | 1.387779e-17 | <= 1e-15 | PASS |
| taxonomy legs or fields | 0 | 0 | PASS |
| payoff deltas versus EW-all | 0 | 0 | PASS |
| deterministic replay | 10/10 numerical artifacts byte-identical | 10/10 | PASS |

Independent validator output:

```text
U1 covariance frequency/span grid independent validation: PASS
grid: 28 cells x 5 q x 2 windows = 280 rows
construction: 280/280 PASS; max weight error=2.220e-16; max group-budget error=1.388e-17
headline raw winner: ME span 36 at q=0.25; Sharpe=0.501599
headline rows beating same-q global on return and Sharpe: 4
headline rows beating best global on return and Sharpe: 0
determinism: 10/10 numerical artifacts byte-identical
```

Cold partition calculation took 193.76-193.83 seconds per daily cell, 51.56-52.44
seconds per weekly cell, and 22.20-27.96 seconds per monthly cell; four cells ran in
parallel. A complete cached portfolio replay took 992.12 seconds.

## Headline global benchmark

| q | net return | Sharpe | turnover | cost drag, bp/year |
|--:|--:|--:|--:|--:|
| 0.30 | 0.066409 | 0.512965 | 2.321201 | 49.061742 |
| 0.25 | 0.061740 | 0.475102 | 2.522758 | 53.126564 |
| 0.20 | 0.058203 | 0.440462 | 2.726819 | 57.234791 |
| 0.15 | 0.058275 | 0.425976 | 2.964027 | 62.254802 |
| 0.10 | 0.051694 | 0.367697 | 3.171192 | 66.221809 |

## Best headline q in every covariance cell

Return and Sharpe deltas below are versus the same-q global leg. Annualised returns are in
decimal units.

| frequency | span | best q | net return | Sharpe | turnover | return delta | Sharpe delta | beats same-q global on both |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| B | 24 | 0.25 | 0.025185 | 0.251644 | 5.794670 | -0.036555 | -0.223458 | no |
| B | 36 | 0.30 | 0.032419 | 0.303518 | 5.467457 | -0.033990 | -0.209446 | no |
| B | 52 | 0.25 | 0.024569 | 0.246890 | 5.378262 | -0.037170 | -0.228213 | no |
| B | 156 | 0.25 | 0.022340 | 0.231052 | 4.528357 | -0.039399 | -0.244051 | no |
| ME | 12 | 0.25 | 0.055306 | 0.450012 | 4.852456 | -0.006434 | -0.025090 | no |
| ME | 24 | 0.30 | 0.057249 | 0.471179 | 4.567833 | -0.009160 | -0.041786 | no |
| ME | 36 | 0.25 | 0.062513 | 0.501599 | 4.553935 | +0.000773 | +0.026496 | yes |
| ME | 52 | 0.30 | 0.055113 | 0.453911 | 4.323077 | -0.011296 | -0.059054 | no |
| W-FRI | 24 | 0.25 | 0.037493 | 0.336613 | 5.112550 | -0.024247 | -0.138489 | no |
| W-FRI | 36 | 0.30 | 0.036550 | 0.327467 | 4.819589 | -0.029858 | -0.185497 | no |
| W-FRI | 52 | 0.30 | 0.040771 | 0.360147 | 4.598770 | -0.025638 | -0.152818 | no |
| W-FRI | 156 | 0.30 | 0.046739 | 0.401446 | 3.805843 | -0.019670 | -0.111519 | no |
| W-MON | 24 | 0.30 | 0.020007 | 0.212039 | 4.996683 | -0.046402 | -0.300926 | no |
| W-MON | 36 | 0.30 | 0.031286 | 0.292222 | 4.798501 | -0.035123 | -0.220743 | no |
| W-MON | 52 | 0.30 | 0.033427 | 0.305929 | 4.652529 | -0.032982 | -0.207036 | no |
| W-MON | 156 | 0.30 | 0.052623 | 0.443771 | 3.807407 | -0.013786 | -0.069193 | no |
| W-THU | 24 | 0.15 | 0.029796 | 0.280246 | 5.766340 | -0.028479 | -0.145730 | no |
| W-THU | 36 | 0.25 | 0.028857 | 0.277208 | 4.901745 | -0.032883 | -0.197894 | no |
| W-THU | 52 | 0.30 | 0.028699 | 0.277033 | 4.594175 | -0.037709 | -0.235932 | no |
| W-THU | 156 | 0.30 | 0.052161 | 0.433516 | 3.835503 | -0.014248 | -0.079449 | no |
| W-TUE | 24 | 0.30 | 0.032442 | 0.301523 | 5.014421 | -0.033966 | -0.211442 | no |
| W-TUE | 36 | 0.30 | 0.034320 | 0.316097 | 4.783164 | -0.032088 | -0.196868 | no |
| W-TUE | 52 | 0.30 | 0.038545 | 0.345260 | 4.640537 | -0.027864 | -0.167705 | no |
| W-TUE | 156 | 0.30 | 0.054586 | 0.455219 | 3.833853 | -0.011823 | -0.057746 | no |
| W-WED | 24 | 0.10 | 0.032211 | 0.294675 | 6.033262 | -0.019483 | -0.073023 | no |
| W-WED | 36 | 0.10 | 0.033798 | 0.309211 | 5.773154 | -0.017896 | -0.058487 | no |
| W-WED | 52 | 0.30 | 0.034097 | 0.319239 | 4.606235 | -0.032312 | -0.193726 | no |
| W-WED | 156 | 0.20 | 0.056295 | 0.470865 | 4.070638 | -0.001908 | +0.030403 | no |

## Frequency leaders

| frequency | span | q | net return | Sharpe | turnover | interpretation |
|:--|--:|--:|--:|--:|--:|:--|
| ME | 36 | 0.25 | 0.062513 | 0.501599 | 4.553935 | overall cluster winner |
| W-WED | 156 | 0.20 | 0.056295 | 0.470865 | 4.070638 | accepted baseline control |
| W-TUE | 156 | 0.30 | 0.054586 | 0.455219 | 3.833853 | second-best weekly endpoint |
| W-MON | 156 | 0.30 | 0.052623 | 0.443771 | 3.807407 | below global |
| W-THU | 156 | 0.30 | 0.052161 | 0.433516 | 3.835503 | below global |
| W-FRI | 156 | 0.30 | 0.046739 | 0.401446 | 3.805843 | weakest long-span weekly endpoint |
| B | 36 | 0.30 | 0.032419 | 0.303518 | 5.467457 | daily native spans are short in calendar time |

## Monthly-span detail

ME/36 is not a generic monthly-frequency result; the other monthly spans do not beat their
same-q global legs on both metrics.

| span | best q | net return | Sharpe | turnover | return delta | Sharpe delta |
|--:|--:|--:|--:|--:|--:|--:|
| 12 | 0.25 | 0.055306 | 0.450012 | 4.852456 | -0.006434 | -0.025090 |
| 24 | 0.30 | 0.057249 | 0.471179 | 4.567833 | -0.009160 | -0.041786 |
| 36 | 0.25 | 0.062513 | 0.501599 | 4.553935 | +0.000773 | +0.026496 |
| 52 | 0.30 | 0.055113 | 0.453911 | 4.323077 | -0.011296 | -0.059054 |

ME/36 has 57.61 clusters on average versus 78.84 for canonical W-WED/156, although both
have identical point-in-time member counts (523/619/641 min/median/max). The ME/36 winner
uses 60.40 available groups and 189.99 selected assets on average in the headline window.
This is descriptively consistent with coarser monthly covariance producing broader groups;
whether that is desirable consolidation or taxonomy degradation requires the frozen E3
fidelity diagnostics and is not decided by this payoff grid.

## Full-panel robustness

The full-panel best row is also ME/36/q=0.25: 3.3380% net return, 0.286357 Sharpe, and
5.3396 turns. It beats same-q global by 48.8 bp/year and 0.035200 Sharpe. Against best-global
q=0.30 it has 5.5 bp/year lower return but 0.003505 higher Sharpe, so it again does not beat
best global on both metrics. The headline window remains primary.

## Interpretation and limits

1. ME/36 is the only covariance cell with a repeated same-q global advantage, and the
   advantage survives 10 bp trading costs. However, at its best q the net-return edge is
   only 7.7 bp/year while cost drag is 96.1 bp/year versus 53.1 bp/year for global.
2. No cluster row displaces global q=0.30. The current decision remains global as the
   benchmark and payoff leader.
3. Daily cells are not a fair same-horizon test of daily sampling because the largest daily
   span requested, 156 observations, is much shorter in calendar time than W-WED/156 or
   ME/36. A calendar-matched follow-up would be needed to isolate cadence itself.
4. The grid uses baseline unsmoothed partitions. M1/M2 smoothing was intentionally excluded
   so temporal smoothing did not confound covariance cadence and span.
5. Alternative covariance cells have no frozen fidelity verdict, bootstrap confidence
   interval, or multiple-testing correction. ME/36 is the best raw research candidate, not
   an accepted replacement for the canonical estimator.

## Deliverables

- `performance.csv`: 280 cluster rows plus 10 invariant global rows.
- `comparison_vs_global.csv`: same-q and best-global deltas for every cluster row.
- `rankings.csv`: all 140 rows ranked separately within each analysis window.
- `cell_summary.csv`: best q for each of 28 cells and each window.
- `frequency_summary.csv`: best span/q within each cadence.
- `partition_diagnostics.csv` and `partition_summary.csv`: membership and cluster counts.
- `construction_diagnostics.csv` and `acceptance.csv`: group counts and numerical checks.
- `regression.csv`: return, eligibility, canonical-partition, and canonical-payoff controls.
- `runtime.csv` and `determinism.csv`: timing, hashes, and replay evidence.

All code, reports, caches, and outputs remain local. Nothing was staged or pushed.
