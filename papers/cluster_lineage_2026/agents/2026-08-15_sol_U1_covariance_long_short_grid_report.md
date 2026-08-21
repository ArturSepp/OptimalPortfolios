# U1 covariance-frequency and EWMA-span long-short grid report

**Date:** 2026-08-15  
**Status:** COMPLETE  
**Runner:** `papers/cluster_lineage_2026/replication/run_u1_covar_grid_long_short.py`  
**Validator:** `papers/cluster_lineage_2026/replication/validate_u1_covar_grid_long_short.py`

## Outcome

The cluster-neutral long-short result is broad across covariance specifications. In the U1
headline window, **17 of 28 cells** deliver both higher annual net return and lower
volatility than the matched global q=0.25 spread. All 28 cells reduce volatility. In the
full panel, 26 of 28 cells mean-variance dominate global and three also have a higher
numerical Sharpe.

The headline leader is **W-THU returns with EWMA span 156**: -2.4326% annual net return
versus -4.0009% global, an improvement of **156.82 bp/year**, with 6.5531% volatility
versus 12.8460% global. Its cumulative net return is -38.91% versus -55.82% global. Before
costs it returns -0.8795% per year versus -2.9967% global, a 211.72 bp/year improvement.

ME/36 is effectively tied: -2.4406% net return and 6.8764% volatility, improving on global
by 156.03 bp/year while reducing volatility by 596.96 bp/year. W-THU/156 exceeds ME/36 by
only 0.80 bp/year, which is economically negligible. The robust result is not the precise
weekday winner; it is the repeated cluster-neutral advantage, especially with long
covariance memory.

## Frozen design

- q is fixed at 0.25. This run varies covariance cadence and EWMA span only.
- Both portfolios target +1 long and -1 short: gross exposure two and net exposure zero.
- U1 point-in-time index membership and eligibility, 48-week momentum with four-week skip,
  ME decisions, implementation lag one, and 10 bp costs are unchanged.
- Cluster legs rank within the unsmoothed baseline partition and apply group-equal budgets
  separately to the top and bottom books. The global leg ranks across the eligible
  universe and is invariant across covariance cells.
- Global rank is the sole payoff benchmark. EW-all is used only as the market reference
  for the reported beta and alpha fields, never as a ranking-performance yardstick.
- Existing accepted partition pickles are consumed cache-first. No covariance model,
  cluster model, or momentum backtest input is re-estimated.

The 28 cells are B and W-MON through W-FRI at native spans 24, 36, 52, and 156, plus ME at
spans 12, 24, 36, and 52. Spans are observation counts at the stated cadence rather than
calendar-normalised horizons.

## Headline results

Global q=0.25 returns -4.0009% net per year with 12.8460% volatility, -0.251495 Sharpe,
2.5960 annual one-way turns, and 100.42 bp/year cost drag. Rows are ordered by net-return
improvement over global. Returns and volatility are annualised.

| frequency | span | pre-cost return | net return | return delta | volatility | vol delta | turnover | cost bp/year | mean-variance win |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| W-THU | 156 | -0.8795% | -2.4326% | +1.5682 pp | 6.5531% | -6.2929 pp | 3.9554 | 155.32 | yes |
| ME | 36 | -0.6360% | -2.4406% | +1.5603 pp | 6.8764% | -5.9696 pp | 4.5894 | 180.45 | yes |
| W-TUE | 156 | -0.9046% | -2.4609% | +1.5400 pp | 6.0414% | -6.8046 pp | 3.9641 | 155.63 | yes |
| W-FRI | 156 | -0.9708% | -2.5133% | +1.4875 pp | 6.1441% | -6.7019 pp | 3.9327 | 154.26 | yes |
| ME | 24 | -0.6801% | -2.5405% | +1.4604 pp | 8.1940% | -4.6520 pp | 4.7368 | 186.04 | yes |
| W-WED | 156 | -1.0317% | -2.5878% | +1.4131 pp | 6.1015% | -6.7445 pp | 3.9689 | 155.61 | yes |
| B | 156 | -1.3445% | -3.1403% | +0.8605 pp | 6.1106% | -6.7354 pp | 4.6005 | 179.58 | yes |
| W-MON | 156 | -1.6166% | -3.1693% | +0.8315 pp | 6.4609% | -6.3851 pp | 3.9825 | 155.27 | yes |
| ME | 12 | -1.3876% | -3.3004% | +0.7004 pp | 8.5864% | -4.2596 pp | 4.9035 | 191.28 | yes |
| W-TUE | 52 | -1.5376% | -3.3982% | +0.6027 pp | 6.8515% | -5.9944 pp | 4.7777 | 186.06 | yes |
| W-FRI | 52 | -1.5767% | -3.4246% | +0.5763 pp | 6.9901% | -5.8558 pp | 4.7447 | 184.79 | yes |
| W-WED | 24 | -1.5000% | -3.4884% | +0.5125 pp | 7.4412% | -5.4048 pp | 5.1087 | 198.84 | yes |
| W-WED | 36 | -1.8284% | -3.7416% | +0.2593 pp | 7.1582% | -5.6877 pp | 4.9277 | 191.32 | yes |
| W-MON | 52 | -1.9089% | -3.7609% | +0.2400 pp | 7.0448% | -5.8011 pp | 4.7723 | 185.20 | yes |
| ME | 52 | -2.1341% | -3.8640% | +0.1368 pp | 6.9101% | -5.9359 pp | 4.4597 | 173.00 | yes |
| W-TUE | 24 | -1.8587% | -3.8728% | +0.1281 pp | 7.5635% | -5.2825 pp | 5.1947 | 201.40 | yes |
| W-TUE | 36 | -1.9865% | -3.9015% | +0.0993 pp | 7.1865% | -5.6595 pp | 4.9388 | 191.50 | yes |
| W-MON | 36 | -2.1252% | -4.0366% | -0.0357 pp | 7.2999% | -5.5461 pp | 4.9362 | 191.14 | no |
| W-WED | 52 | -2.2043% | -4.0472% | -0.0463 pp | 6.7206% | -6.1253 pp | 4.7628 | 184.29 | no |
| W-FRI | 24 | -2.2171% | -4.2047% | -0.2038 pp | 6.9072% | -5.9388 pp | 5.1410 | 198.76 | no |
| W-THU | 24 | -2.2481% | -4.2262% | -0.2253 pp | 7.4883% | -5.3577 pp | 5.1190 | 197.81 | no |
| W-THU | 52 | -2.5012% | -4.3254% | -0.3246 pp | 6.3934% | -6.4526 pp | 4.7237 | 182.43 | no |
| W-FRI | 36 | -2.5298% | -4.4424% | -0.4416 pp | 6.8928% | -5.9532 pp | 4.9579 | 191.27 | no |
| B | 52 | -2.4312% | -4.5088% | -0.5080 pp | 6.3589% | -6.4871 pp | 5.3868 | 207.77 | no |
| B | 24 | -2.5015% | -4.7297% | -0.7288 pp | 7.1158% | -5.7302 pp | 5.7906 | 222.82 | no |
| B | 36 | -2.7355% | -4.8817% | -0.8809 pp | 6.5633% | -6.2826 pp | 5.5830 | 214.62 | no |
| W-THU | 36 | -3.1616% | -5.0556% | -1.0548 pp | 6.9003% | -5.9457 pp | 4.9360 | 189.41 | no |
| W-MON | 24 | -4.0151% | -5.9726% | -1.9718 pp | 7.8206% | -5.0254 pp | 5.1475 | 195.75 | no |

All headline cells have lower volatility than global. Their numerical Sharpe ratios remain
below global because every return is negative: dividing a smaller loss by materially lower
volatility can produce a more-negative ratio. The economic comparison is unambiguous for
the 17 mean-variance winners because each has both higher return and lower risk.

## Frequency and span pattern

| frequency | cells | headline return wins | headline mean-variance wins | best return delta | median return delta |
|:--|--:|--:|--:|--:|--:|
| B | 4 | 1 | 1 | +0.8605 pp | -0.6184 pp |
| ME | 4 | 4 | 4 | +1.5603 pp | +1.0804 pp |
| W-FRI | 4 | 2 | 2 | +1.4875 pp | +0.1862 pp |
| W-MON | 4 | 2 | 2 | +0.8315 pp | +0.1021 pp |
| W-THU | 4 | 1 | 1 | +1.5682 pp | -0.2750 pp |
| W-TUE | 4 | 4 | 4 | +1.5400 pp | +0.3654 pp |
| W-WED | 4 | 3 | 3 | +1.4131 pp | +0.3859 pp |

Every available 156-observation cell beats global. All four ME cells and all four W-TUE
cells also win. Short native spans are less reliable, especially B/24-52 and W-MON/24.
Because native spans represent different calendar horizons, this is evidence for a stable,
long-memory covariance partition rather than evidence for one privileged weekday.

## Full-panel robustness

The full panel is stronger: 26/28 cells have higher return and lower volatility than global;
three also have higher numerical Sharpe. W-THU/156 remains the leader at -3.6482% annual
net return versus -7.5875% global, a +3.9393 pp improvement, with 8.8411% volatility versus
16.1774%. Its Sharpe is -0.375251 versus -0.401400 global.

Annual net-return deltas versus global, in percentage points:

| frequency | 12 | 24 | 36 | 52 | 156 |
|:--|--:|--:|--:|--:|--:|
| B | -- | +0.234 | +0.740 | +1.305 | +3.365 |
| ME | +2.491 | +2.826 | +3.605 | +1.869 | -- |
| W-FRI | -- | +0.713 | +0.628 | +2.083 | +3.256 |
| W-MON | -- | -0.736 | +0.918 | +0.598 | +1.763 |
| W-THU | -- | +0.950 | -0.245 | +0.875 | +3.939 |
| W-TUE | -- | +1.358 | +1.609 | +1.999 | +3.168 |
| W-WED | -- | +1.673 | +1.688 | +1.672 | +2.945 |

## Why volatility falls

The result is not driven by leverage or a larger effective asset count. Every portfolio has
gross exposure two and net exposure zero. At W-THU/156, the cluster portfolio's maximum
post-net group L1 exposure is numerical zero, whereas the global portfolio's mean L1 net
exposure across the same clusters is 1.4344. The cluster and global books have 139.18 and
153.15 effective names per side, respectively. Thus the lower cluster volatility comes
from matching long and short systematic group exposures, not simply holding more names.

The winner's market beta is -0.1126 versus -0.3245 global. It therefore removes both the
explicit cross-cluster bets and much of the unintended market exposure left by a global
top-minus-bottom sort.

## Acceptance

| check | measured | tolerance | verdict |
|:--|--:|--:|:--|
| grid coverage | 28 cells x 2 windows = 56 comparisons | 56 | PASS |
| performance rows | 56 cluster + 2 global | 58 | PASS |
| exact exposure rows | 58/58 | 58/58 | PASS |
| maximum long-exposure error | 6.106227e-15 | <= 1e-12 | PASS |
| maximum short-exposure error | 6.106227e-15 | <= 1e-12 | PASS |
| maximum net-exposure error | 1.492730e-15 | <= 1e-12 | PASS |
| maximum gross-exposure error | 1.953993e-14 | <= 1e-12 | PASS |
| maximum pre-net weight error | 2.220446e-16 | <= 1e-12 | PASS |
| maximum pre-net group-budget error | 1.110223e-16 | <= 1e-15 | PASS |
| maximum post-net cluster L1 exposure | 1.670105e-15 | <= 1e-12 | PASS |
| ME/36 and global frozen regression | maximum error 4.405365e-13 | <= 1e-12 | PASS |
| deterministic replay | 6/6 artifacts byte-identical | 6/6 | PASS |
| global-only payoff benchmark | all 56 comparisons | all | PASS |

One cached pass took 286.28 seconds; deterministic verification ran two passes. Focused
construction tests passed 4/4 and ruff E/F/W checks passed.

Independent validator output:

```text
U1 q=0.25 covariance long-short grid independent validation: PASS
grid: 28 cells x 2 windows = 56 comparisons
exposures: 58/58 PASS; max error=1.954e-14
headline winner: W-THU span 156; net return delta=+0.015682; volatility delta=-0.062929
headline cells: return wins=17/28; mean-variance wins=17/28; return-and-Sharpe wins=0/28
determinism: 6/6 artifacts byte-identical
```

## Interpretation and limits

1. The global-relative improvement is robust to covariance cadence and span: a majority of
   headline cells and nearly every full-panel cell mean-variance dominate global.
2. Long covariance memory is the clearest pattern. W-THU/156, ME/36, W-TUE/156,
   W-FRI/156, and W-WED/156 form the leading group; their exact ordering should not be
   over-interpreted.
3. The spread remains negative before and after costs, so the supported claim is relative
   outperformance and risk cancellation, not positive standalone momentum alpha.
4. These alternative covariance partitions have not been passed through the frozen E3b
   taxonomy-fidelity band or an inference/multiple-testing layer. W-THU/156 is a raw grid
   leader, not yet a replacement for a frozen production configuration.
5. The full numerical tables are persisted under
   `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/covariance_frequency_span_grid/msci_us/long_short_grid_q_025/`.

The runner, validator, test, report, and output artifacts remain local and ignored by git.
Nothing was staged or pushed.
