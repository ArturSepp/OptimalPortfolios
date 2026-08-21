# U2 BlackRock broad-sleeve weight-grid execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE; 70/20/10 long-only is the recommended research candidate  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Decision

The broad-sleeve construction is the right way to continue, because it removes the
strategic asset-allocation mismatch that invalidated the original heterogeneous-ETF
comparison. The global and cluster legs now receive the **same** Equity, Fixed-Income,
and Rest budgets. The cluster treatment is tested only below those common budgets.

The recommended candidate is:

- **70% Equity / 20% Fixed Income / 10% Rest**;
- long-only;
- exact ROSAA production momentum, q = 0.25;
- ME covariance returns, EWMA span 12;
- group-equal budgets across available clusters within each broad sleeve;
- monthly decisions, W-WED performance, lag 1, and 10 bp costs.

On the untouched 2018-01-31 through 2026-06-30 evaluation window, this candidate earns
5.8558% net annualised, versus 5.5512% for the identically budgeted sleeve-global rank
and 5.0125% for the original unconstrained global rank. The respective Sharpe ratios are
0.4802, 0.4633, and 0.4101. Thus the measured cluster edge is **+30.46 bp/year against
the fair same-budget global control** and +84.33 bp/year against the original global
portfolio.

This is a research candidate, not yet an unconditional paper claim. Before 2018, every
allocation in the cluster grid loses to its same-budget global control; after 2018, every
allocation wins. The full-window cluster edge is therefore negative, and the ME/span-36
transfer robustness does not reproduce the win. The next defensible step is a frozen
walk-forward and inference pass, not further in-sample weight searching.

The long-short construction is not recommended. Its best evaluation result remains
negative after costs.

## Design

The current-vintage BlackRock universe and all signal/backtest conventions are unchanged
from the accepted U2 grid. The experiment uses 480 current U.S. iShares ETFs and maps the
official Aladdin asset classes as follows:

| broad sleeve | source classification | current funds |
|---|---|---:|
| Equity | Equity | 288 |
| Fixed Income | Fixed Income | 154 |
| Rest | Cash, Commodity, Digital Assets, Multi Asset, Real Estate | 38 |

Eight allocations were declared in ten-point increments before inspecting their results:

| allocation id | Equity | Fixed Income | Rest |
|---|---:|---:|---:|
| E40_F30_R30 | 40% | 30% | 30% |
| E40_F40_R20 | 40% | 40% | 20% |
| E50_F20_R30 | 50% | 20% | 30% |
| E50_F30_R20 | 50% | 30% | 20% |
| E50_F40_R10 | 50% | 40% | 10% |
| E60_F20_R20 | 60% | 20% | 20% |
| E60_F30_R10 | 60% | 30% | 10% |
| E70_F20_R10 | 70% | 20% | 10% |

At every date, both legs receive the same top-level budgets. The sleeve-global leg ranks
within each broad sleeve and splits its sleeve budget equally across selected assets. The
sleeve-cluster leg ranks within correlation clusters split by broad sleeve, gives equal
budgets to available within-sleeve clusters, and then splits each cluster budget equally
across selected assets. A sleeve with no valid score would be excluded, but every sleeve
is available on every evaluated decision date.

For long-short portfolios, the stated allocation is applied independently to the long and
short sides. Each broad sleeve is therefore net neutral. Long/short overlap is removed and
each signed sleeve is renormalised to its exact target.

The three analysis windows are fixed:

| label | dates | monthly decisions | role |
|---|---|---:|---|
| selection | 2009-08-31 to 2017-12-31 | 101 | allocation selection only |
| evaluation | 2018-01-31 to 2026-06-30 | 102 | untouched primary result |
| headline/full | 2009-08-31 to 2026-06-30 | 203 | labelled descriptive result |

The previous U2 grid supplies the primary covariance cells without a new covariance
search: ME/span 12 for long-only and W-THU/span 156 for long-short. The frozen U1
ME/span-36 specification is retained as a separately labelled transfer robustness.

EW-all is not a ranking yardstick or payoff comparison. It remains reference-only for
alpha and beta columns in the machine-readable performance table.

## Evaluation results

### Long-only grid

All eight cluster allocations beat their identically budgeted sleeve-global controls
out-of-sample. Three also beat the original unconstrained global rank on net return.

| allocation | cluster net | sleeve-global net | delta vs same-budget global | delta vs original global | cluster Sharpe |
|---|---:|---:|---:|---:|---:|
| 70/20/10 | **5.8558%** | 5.5512% | **+30.46 bp** | **+84.33 bp** | **0.4802** |
| 60/20/20 | 5.4300% | 5.1644% | +26.56 bp | +41.75 bp | 0.4614 |
| 60/30/10 | 5.0342% | 4.7963% | +23.79 bp | +2.17 bp | 0.4505 |
| 50/20/30 | 4.9982% | 4.7651% | +23.31 bp | -1.43 bp | 0.4397 |
| 50/30/20 | 4.6064% | 4.4034% | +20.29 bp | -40.61 bp | 0.4287 |
| 50/40/10 | 4.2065% | 4.0280% | +17.85 bp | -80.60 bp | 0.4145 |
| 40/30/30 | 4.1725% | 3.9981% | +17.44 bp | -84.00 bp | 0.4035 |
| 40/40/20 | 3.7766% | 3.6291% | +14.75 bp | -123.59 bp | 0.3889 |

The original global reference is 5.0125% net, 14.5457% volatility, and 0.4101 Sharpe.
The 70/20/10 cluster candidate has 13.9116% volatility. Its 5.5855 annualised one-way
turnover produces 118.38 bp/year cost drag, versus 68.77 bp for its sleeve-global control
and 70.51 bp for the original global. The cluster still wins net: its gross edge against
the same-budget control is about 80.07 bp/year, of which about 49.61 bp is consumed by
additional costs.

The owner's proposed 50/30/20 is useful as a balanced robustness allocation. It beats the
fair same-budget global control by 20.29 bp/year and raises Sharpe from 0.4194 to 0.4287,
but it does not stay competitive with the original global return: the gap is -40.61
bp/year. For the stated objective, 70/20/10 is preferable.

### Long-short grid

All eight primary cluster cells beat their same-budget sleeve-global controls by 14.06 to
29.54 bp/year in the evaluation window, and three marginally beat the original global
spread. None is investable on the present evidence because every absolute net return and
Sharpe is negative.

The training-selected 50/40/10 cluster spread earns -1.7243% net with Sharpe -0.4718,
versus -1.8649% for its same-budget global control and -1.8205% for the original global
spread. Its small +9.62 bp/year advantage against original global is overwhelmed by the
negative absolute payoff and 166.60 bp/year cost drag. Long-short should not be the
headline strategy.

## Stability and robustness

The absolute allocation ranking is stable: the Spearman correlation of training and
evaluation cluster net-return ranks is 1.000 for long-only and 0.976 for long-short.
Accordingly, 70/20/10 is the highest-return long-only allocation in both periods; it was
selected using only the earlier period.

The **cluster edge**, however, is not temporally stable:

| window | strategy | cells beating same-budget global | edge range |
|---|---|---:|---:|
| 2009-08-31 to 2017-12-31 | long-only | 0/8 | -185.68 to -118.95 bp/year |
| 2018-01-31 to 2026-06-30 | long-only | 8/8 | +14.75 to +30.46 bp/year |
| full 2009-08-31 to 2026-06-30 | long-only | 0/8 | -62.81 to -53.46 bp/year |
| 2009-08-31 to 2017-12-31 | long-short | 0/8 | -191.24 to -132.68 bp/year |
| 2018-01-31 to 2026-06-30 | long-short | 8/8 | +14.06 to +29.54 bp/year |
| full 2009-08-31 to 2026-06-30 | long-short | 0/8 | -91.79 to -69.29 bp/year |

For 70/20/10 long-only, the full-window cluster return is 6.3996%, effectively tying the
original global return of 6.4100% (-1.04 bp/year), but losing to the same-budget global
return of 6.9342% by 53.46 bp/year. The training-period same-budget gap is -185.68
bp/year. The positive evaluation result is therefore a genuine out-of-sample observation,
but it is also a regime reversal rather than a stable full-history premium.

The covariance result is also specification-sensitive. Replacing primary ME/span 12 with
the transferred U1 ME/span 36 at 70/20/10 gives 4.1985% net in evaluation, losing 135.26
bp/year to the same-budget global and 81.40 bp/year to original global. This robustness
failure must accompany any use of the ME/span-12 result.

## Acceptance record

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| strategic allocations | 8, including 50/30/20 | exact | PASS |
| current funds classified into broad sleeves | 480/480 | exact | PASS |
| selection/evaluation/full decisions | 101 / 102 / 203 | exact | PASS |
| portfolio rows | 150 | 150 | PASS |
| cluster-to-control comparison rows | 96 | 96 | PASS |
| selection rows / selected-window rows | 8 / 24 | 8 / 24 | PASS |
| portfolio acceptance rows | 150/150 PASS | 100% | PASS |
| max weight/net-sum error | 9.021e-16 | <= 1e-12 | PASS |
| max top-level sleeve-budget error | 1.221e-15 | <= 1e-12 | PASS |
| max within-sleeve group-budget error | 1.110e-16 | <= 1e-15 | PASS |
| max long exposure error | 3.997e-15 | <= 1e-12 | PASS |
| max short exposure error | 3.997e-15 | <= 1e-12 | PASS |
| max net exposure error | 9.021e-16 | <= 1e-12 | PASS |
| max gross exposure error | 1.177e-14 | <= 1e-12 | PASS |
| original-global accepted-run regression | 2/2 PASS | <= 1e-12 | PASS |
| deterministic numerical artifacts | 9/9 byte-identical | 100% | PASS |
| focused pytest | 6/6 passed | all pass | PASS |
| independent payoff reconstruction | 2/2 selected OOS legs | <= 5e-12 | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |
| EW ranking/performance leg | 0 | 0 | PASS |

One complete cache-hit pass took 399.04 seconds: 107.78 seconds for selection, 112.71
seconds for evaluation, and 172.73 seconds for the full window. The complete pass was
replayed and all nine non-timing CSV artifacts were byte-identical.

Verbatim focused pytest output:

```text
......                                                                   [100%]
```

Verbatim independent-validation output:

```text
BlackRock broad-sleeve independent validation: PASS (8 weights, 150 portfolios, 96 comparisons, 2 reconstructed payoffs, 9 hashes)
```

## Code and deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_u2_blackrock_sleeve_grid.py`

Checks:

- `papers/cluster_lineage_2026/replication/validate_u2_blackrock_sleeve_grid.py`
- `papers/cluster_lineage_2026/replication/u2_blackrock_sleeve_grid_test.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\covariance_frequency_span_grid\blackrock_us_etfs\broad_sleeve_weight_grid
```

Machine-readable artifacts:

- `weight_grid.csv`
- `performance.csv`
- `comparison.csv`
- `selection.csv`
- `selected_evaluation.csv`
- `rank_stability.csv`
- `allocation_diagnostics.csv`
- `acceptance.csv`
- `regression.csv`
- `runtime.csv`
- `determinism.csv`

The BlackRock input remains a 2026-08-15 current-vintage survivor cohort, not a
survivorship-free historical fund census. No covariance cache was re-estimated, no EW-all
payoff comparison was introduced, and nothing was staged or pushed.

## Recommended next experiment

Freeze 70/20/10 and ME/span 12 before any further payoff inspection, then run rolling
walk-forward origins and a moving-block bootstrap on the cluster-minus-same-budget-global
return. Report the fraction of test origins won, the confidence interval for the net
return and Sharpe deltas, and the cost break-even. Keep 50/30/20 as the balanced robustness
allocation and ME/span 36 as the negative covariance robustness. Do not expand the weight
grid merely because the current winner lies on its 70% Equity boundary.
