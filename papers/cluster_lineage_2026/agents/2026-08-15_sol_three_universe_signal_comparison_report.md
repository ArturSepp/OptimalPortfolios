# Three-universe ROSAA versus classic momentum execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Runner:** `papers/cluster_lineage_2026/replication/run_three_universe_signal_comparison.py`  
**Focused tests:** `papers/cluster_lineage_2026/replication/three_universe_signal_comparison_test.py`  
**Repository scope:** ignored `papers/cluster_lineage_2026/` tree only; no staging or push

## Outcome

The matched long-short comparison was completed for both classic 12m-ex-1m and
ROSAA risk-adjusted momentum on all three universes. Every portfolio uses q=25%,
+1/-1 exposure, one implementation-period lag, and the common 203-decision window
from 2009-08-31 through 2026-06-30.

The cluster construction beats every requested U1 ranking benchmark on net return
under both signals. It also reproduces the owner-frozen futures ROSAA result, beating
the same-budget global rank by 4.60 bp/year net. It does **not** beat global for funds
under either signal, and it does not beat futures global under classic momentum.

Thus the net-return outperformance result is not universal:

- U1: cluster wins 4/4 signal-by-benchmark comparisons;
- BlackRock funds: cluster wins 0/2 comparisons;
- futures: cluster wins 1/2 comparisons;
- total: 5/8 net-return comparisons won.

Clustering lowers volatility in all 8/8 comparisons, but improves RF=0 Sharpe in only
1/8: U1 ROSAA versus BICS sectors. No cluster leg beats its matched global rank on
Sharpe. This remains primarily a volatility and systematic-exposure compression result.

## Frozen matched design

| universe | cluster treatment | requested ranking benchmark(s) | signed sleeve weights | cost |
|:--|:--|:--|:--|--:|
| U1 equities | M1-star, delta 0.0866 | BICS sector and global | not applicable | 10 bp |
| U2 BlackRock funds | W-THU covariance, EWMA span 156 | same-budget global | 50% Equity / 30% Fixed Income / 20% Rest | 20 bp |
| U3 futures | M1-star, delta 0.0691 | same-budget global | 30% Equity / 30% Fixed Income / 30% Commodities / 10% FX | 10 bp |

The stated sleeve weights apply independently to both the long and short sides. The
fund global benchmark ranks within Equity, Fixed Income, and Rest before applying
50/30/20. The futures global benchmark ranks within its four sleeves before applying
30/30/30/10. These are fair same-budget global controls, not unconstrained global books.

U1 uses the matched BICS-classified universe: stocks missing BICS metadata are excluded
from cluster, sector, and global legs alike. Futures apply all seven owner-frozen
low-liquidity exclusions. Point-in-time eligibility and changing cross-sectional sample
sizes are respected on every decision date.

The two signal definitions are:

- ROSAA: monthly benchmark-relative returns, long EWMA span 12, no short/reversal
  span, volatility span 13, `MeanAdjType.EWMA`, and the point-in-time eligible EW
  market return as benchmark;
- classic: exactly 12 completed monthly log returns after a hard one-month skip,
  with no benchmark subtraction, volatility scaling, mean adjustment, or EWMA filter.

Classic momentum is computed through the public
`optimalportfolios.alphas.compute_classic_momentum_from_returns` API added in v6.20.0.
No covariance or cluster model was refit.

## Full performance table

| universe | signal | leg | net/year | gross/year | vol/year | Sharpe | turnover/year | cost drag |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| U1 | ROSAA | M1-star cluster | -2.2512% | -1.0687% | 6.3513% | -0.3263 | 3.0113 | 118.25 bp |
| U1 | ROSAA | BICS sector | -3.5830% | -2.3867% | 9.5981% | -0.3312 | 3.0815 | 119.63 bp |
| U1 | ROSAA | global | -4.0568% | -2.8349% | 12.8129% | -0.2572 | 3.1609 | 122.19 bp |
| U1 | classic | M1-star cluster | -2.6816% | -1.6623% | 6.0983% | -0.4147 | 2.6080 | 101.94 bp |
| U1 | classic | BICS sector | -3.8221% | -2.8551% | 9.3630% | -0.3681 | 2.4986 | 96.70 bp |
| U1 | classic | global | -4.0000% | -3.0323% | 12.7709% | -0.2536 | 2.5042 | 96.77 bp |
| funds | ROSAA | W-THU/156 cluster | -2.9999% | +0.1003% | 3.9274% | -0.7552 | 3.9271 | 310.03 bp |
| funds | ROSAA | 50/30/20 global | -1.7460% | +0.7933% | 7.1198% | -0.2117 | 3.1907 | 253.92 bp |
| funds | classic | W-THU/156 cluster | -3.1254% | -0.3000% | 4.2602% | -0.7233 | 3.5870 | 282.54 bp |
| funds | classic | 50/30/20 global | -1.6863% | +0.3614% | 7.6894% | -0.1827 | 2.5754 | 204.77 bp |
| futures | ROSAA | M1-star cluster | +0.0297% | +1.2501% | 4.4196% | 0.0179 | 3.0330 | 122.05 bp |
| futures | ROSAA | 30/30/30/10 global | -0.0163% | +1.3683% | 8.2125% | 0.0296 | 3.4414 | 138.46 bp |
| futures | classic | M1-star cluster | +0.2148% | +1.2799% | 4.4152% | 0.0680 | 2.6448 | 106.51 bp |
| futures | classic | 30/30/30/10 global | +1.7014% | +2.8827% | 8.8453% | 0.2285 | 2.8920 | 118.13 bp |

Net-return and Sharpe columns reflect the stated one-way trading costs; gross return is
the pre-cost result. Sharpe is the frozen RF=0 convention. EW-all appears only in
alpha/beta calculations and is not a ranking or payoff yardstick.

## Cluster versus requested benchmarks

| universe | signal | benchmark | net-return delta | gross-return delta | vol delta | Sharpe delta | turnover delta |
|:--|:--|:--|--:|--:|--:|--:|--:|
| U1 | ROSAA | BICS sector | **+133.19 bp** | +131.80 bp | -3.2467 pp | **+0.0048** | -0.0702 |
| U1 | ROSAA | global | **+180.56 bp** | +176.62 bp | -6.4616 pp | -0.0691 | -0.1496 |
| U1 | classic | BICS sector | **+114.05 bp** | +119.28 bp | -3.2647 pp | -0.0466 | +0.1094 |
| U1 | classic | global | **+131.84 bp** | +137.01 bp | -6.6726 pp | -0.1611 | +0.1038 |
| funds | ROSAA | 50/30/20 global | **-125.40 bp** | -69.29 bp | -3.1924 pp | -0.5435 | +0.7365 |
| funds | classic | 50/30/20 global | **-143.91 bp** | -66.14 bp | -3.4292 pp | -0.5406 | +1.0117 |
| futures | ROSAA | 30/30/30/10 global | **+4.60 bp** | -11.82 bp | -3.7929 pp | -0.0117 | -0.4084 |
| futures | classic | 30/30/30/10 global | **-148.66 bp** | -160.28 bp | -4.4300 pp | -0.1605 | -0.2472 |

### U1

U1 is the robust relative-return result. Clustering loses less than both BICS-sector and
global ranks under both signals and cuts volatility by 3.25--6.67 percentage points.
ROSAA is preferable for the cluster leg: it improves net return by 43.04 bp/year versus
classic, though every U1 book remains negative gross and net.

### BlackRock funds

At the requested 20 bp cost, clustering is not competitive with the fair 50/30/20
global control. The gross deficits are 69.29 bp/year under ROSAA and 66.14 bp/year under
classic; higher cluster turnover then widens the net deficits to 125.40 and 143.91 bp.
The cluster books halve volatility approximately, but that does not compensate for the
payoff and cost shortfall. This current-vintage 480-fund cohort also remains subject to
the previously disclosed survivorship limitation.

### Futures

The ROSAA result reproduces the owner-frozen selected cell exactly. M1-star gives up
11.82 bp/year gross but saves 16.41 bp/year in cost drag, creating a small +4.60 bp/year
net edge. This is a turnover-driven crossover, not a gross-alpha win, and its Sharpe is
slightly lower than global.

Classic momentum is much stronger in absolute futures performance, especially for the
global rank, but it reverses the relative conclusion: global earns 1.7014% net versus
0.2148% for M1-star. The cluster book still halves volatility. The result shows that the
futures cluster-outperformance claim is signal-dependent; it holds narrowly for the
selected ROSAA signal and fails for classic 12m-ex-1m.

## Signal comparison under unchanged portfolio legs

Classic minus ROSAA annual net-return deltas are:

| universe | cluster | sector/global controls |
|:--|--:|:--|
| U1 | -43.04 bp | sector -23.90 bp; global +5.67 bp |
| funds | -12.54 bp | 50/30/20 global +5.97 bp |
| futures | +18.51 bp | 30/30/30/10 global +171.77 bp |

There is no single signal winner across universes. ROSAA is better for the U1 and fund
cluster legs, while classic is better for both futures legs. The large futures global
improvement is the reason classic weakens the cluster-relative result despite slightly
improving the cluster itself.

## Acceptance and verification

| acceptance line | measured | tolerance | verdict |
|:--|:--|:--|:--|
| common analysis dates | 203 per universe, 2009-08-31 through 2026-06-30 | exact | PASS |
| performance rows | 14 | 14 | PASS |
| cluster-versus-benchmark rows | 8 | 8 | PASS |
| classic-minus-ROSAA rows | 7 | 7 | PASS |
| costs | U1 10 bp / funds 20 bp / futures 10 bp | exact | PASS |
| sleeve weights | funds 50/30/20; futures 30/30/30/10 on each side | exact | PASS |
| all construction acceptance rows | 58/58 `PASS` | 100% | PASS |
| maximum new fund/futures weight or exposure error | `7.994e-15` | `<= 1e-12` | PASS |
| maximum new within-sleeve group-budget error | `1.110e-16` | `<= 1e-15` | PASS |
| maximum signal look-ahead | 0 days | `<= 0` | PASS |
| maximum ROSAA source reconstruction error | `3.109e-14` | `<= 1e-12` | PASS |
| maximum classic score reconstruction error | `2.665e-15` | `<= 1e-14` | PASS |
| futures frozen-liquidity exclusions in weights/scores | 0 | 0 | PASS |
| owner-frozen futures ROSAA performance regression | 14/14 metrics | `<= 1e-12` | PASS |
| deterministic non-timing artifacts | 6/6 byte-identical | 100% | PASS |
| focused pytest | 6/6 passed | all pass | PASS |
| isolated Ruff E/F/W | no findings | no findings | PASS |
| EW performance comparison | 0 | 0 | PASS |

The classic signal is independently reconstructed by explicit date-by-date history
slicing. ROSAA price conversion is independently round-tripped to the monthly log-return
source. The hashes in `determinism.csv` were also rechecked against the final files after
the focused test run; all 6/6 still match. One complete cache-first pass took 187.31
seconds.

Final focused pytest output was:

```text
......                                                                   [100%]
```

Final isolated Ruff output was:

```text
All checks passed!
```

## Deliverables

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\three_universe_rosaa_vs_classic_20260815
```

Artifacts:

- `performance.csv` -- all 14 matched portfolios;
- `benchmark_comparison.csv` -- the eight permitted cluster comparisons;
- `signal_comparison.csv` -- classic-minus-ROSAA deltas under unchanged legs;
- `signal_preflight.csv` -- timing, source, mask, and reconstruction checks;
- `acceptance.csv` -- all construction checks;
- `design.csv`, `runtime.csv`, and `determinism.csv`.

No covariance cache or cluster cache was changed, no model was refit, and nothing was
staged or pushed.
