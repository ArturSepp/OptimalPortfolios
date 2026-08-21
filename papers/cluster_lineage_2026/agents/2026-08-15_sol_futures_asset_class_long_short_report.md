# Futures standalone asset-class long-short execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

Four independent +1/-1 books were run for Equity, Fixed Income, Commodities, and FX.
Each class has its own within-class global rank, baseline-cluster rank, and M1-star-cluster
rank. The corrected U1 calendar window is used throughout.

The result is unambiguous: **no cluster book beats its own global control on both net
return and Sharpe in any of the 16 class-by-q-by-cluster comparisons**.

After the owner excluded `CUA1 Comdty` from futures investability, the primary q=20%
failure remains driven mainly by Commodities, but the global edge is substantially smaller:

- global commodity long-short earns +0.5938% net/year with 0.1479 Sharpe;
- baseline clustering earns -1.6283% with -0.1148 Sharpe;
- M1-star clustering earns -1.6710% with -0.1161 Sharpe.

This is not primarily a cost effect. Commodity cluster gross returns are only +0.6302%
for baseline and +0.2609% for M1-star, versus +3.0981% for global. Ranking separately
inside approximately ten commodity clusters expands each side from 7 contracts to about
12 and removes the profitable cross-cluster ordering.

Equity global is mildly negative and clusters make it more negative. FX global is near
flat and clusters make it materially negative. Fixed Income is the only partial exception:
q=20% M1-star improves net return by 7.84 bp/year and reduces volatility by 0.6881
percentage points, but its Sharpe is still 0.0086 lower than global. It is the sole
mean-variance-dominance row, not a return-and-Sharpe win.

The q=25% robustness strengthens the negative conclusion. Global Commodities improves
to +2.7507% net/year and 0.2356 Sharpe, while both commodity cluster books remain negative.
No cluster treatment wins jointly in any other class.

## Frozen design

| component | value |
|---|---|
| universe | 95 source series; 94 ever eligible after excluding CUA1 |
| point-in-time eligible count | 88 to 94 futures; 31 to 33 Commodities |
| standalone books | Equity, Fixed Income, Commodities, FX |
| exposure per book | +1 long / -1 short; gross 2, net 0 |
| signal | 48-week production momentum, latest four weeks skipped |
| primary selection | q = 0.20 |
| robustness | q = 0.25 |
| decisions | 203 ME dates, 2009-08-31 through 2026-06-30 |
| measured NAV | W-WED, 2009-09-02 through 2026-06-24 |
| implementation lag | one W-WED observation |
| costs | 20 bp |
| cluster treatments | accepted baseline and M1-star caches |

The global method ranks all eligible contracts within one asset class. Cluster methods
rank within the accepted correlation clusters contained in that class, give available
clusters equal side budgets, and split a cluster budget equally among selected contracts.
Long and short overlap is removed before each side is renormalised to one.

Each standalone book contains zero exposure outside its stated class. Recombining the four
books with 30% Equity, 30% Fixed Income, 30% Commodities, and 10% FX reproduces the prior
combined-book target weights within 5.551e-17. The decomposition is therefore exact at the
decision-weight level.

EW-all is used only as the market reference for beta and alpha columns. It is not a payoff
leg or comparison benchmark.

## Equity long-short

### Primary q=20%

| method | gross return | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| global | **2.0224%** | **-0.9004%** | 13.0798% | **-0.0081** | 3.6310 | 292.28 bp |
| baseline cluster | 1.7299% | -1.5518% | **8.7036%** | -0.1458 | 4.0936 | 328.16 bp |
| M1-star cluster | 1.2076% | -1.5744% | 8.9942% | -0.1407 | **3.4804** | **278.20 bp** |

Baseline and M1-star reduce volatility by 4.3762 and 4.0856 percentage points, but lose
65.13 and 67.40 bp/year of net return and reduce Sharpe by 0.1378 and 0.1327.

### q=25% robustness

| method | net return | volatility | RF=0 Sharpe |
|---|---:|---:|---:|
| global | **-0.3966%** | 10.9445% | **0.0172** |
| baseline cluster | -1.3390% | **7.9624%** | -0.1385 |
| M1-star cluster | -1.2195% | 8.0756% | -0.1204 |

Equity verdict: clustering reduces risk but worsens an already weak spread payoff.

## Fixed Income long-short

### Primary q=20%

| method | gross return | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| global | **1.1817%** | -1.2720% | 7.4260% | **-0.1410** | 3.0690 | 245.37 bp |
| baseline cluster | 0.7368% | -1.8527% | 6.9178% | -0.2381 | 3.2525 | 258.95 bp |
| M1-star cluster | 1.0360% | **-1.1936%** | **6.7379%** | -0.1496 | **2.7904** | **222.97 bp** |

M1-star gains 7.84 bp/year net and lowers volatility and turnover versus global. Its gross
return is nevertheless 14.57 bp/year lower, and its monthly-return Sharpe is 0.0086 lower.
The improvement comes from lower trading cost rather than stronger gross signal.

### q=25% robustness

| method | net return | volatility | RF=0 Sharpe |
|---|---:|---:|---:|
| global | **-1.1268%** | 6.8243% | **-0.1364** |
| baseline cluster | -1.7958% | 6.3042% | -0.2566 |
| M1-star cluster | -1.2988% | **6.1250%** | -0.1884 |

Fixed Income verdict: only q=20% M1-star offers a small cost-led net-return improvement;
all variants have negative returns and Sharpes.

## Commodities long-short

### Primary q=20%

| method | gross return | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| global | **3.0981%** | **0.5938%** | 24.1161% | **0.1479** | 3.0724 | 250.43 bp |
| baseline cluster | 0.6302% | -1.6283% | 9.5589% | -0.1148 | 2.8280 | 225.85 bp |
| M1-star cluster | 0.2609% | -1.6710% | **9.1832%** | -0.1161 | **2.4254** | **193.19 bp** |

Relative to global, baseline loses 222.21 bp/year net and 0.2627 Sharpe; M1-star loses
226.48 bp/year and 0.2639 Sharpe. Lower risk and cost cannot offset the gross-signal loss.

### q=25% robustness

| method | net return | volatility | RF=0 Sharpe |
|---|---:|---:|---:|
| global | **2.7507%** | 21.4423% | **0.2356** |
| baseline cluster | -0.6580% | 9.0834% | -0.0156 |
| M1-star cluster | -1.8441% | **8.7777%** | -0.1451 |

Commodities verdict: global ranking is the only robustly positive asset-class payoff and
the largest source of the combined global spread's advantage.

## FX long-short

### Primary q=20%

| method | gross return | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| global | **2.4300%** | **0.0252%** | 16.9942% | **0.0851** | **2.9567** | **240.48 bp** |
| baseline cluster | 1.2336% | -1.8667% | **7.1521%** | -0.2252 | 3.8802 | 310.03 bp |
| M1-star cluster | 0.8597% | -1.9213% | 7.5616% | -0.2166 | 3.4928 | 278.10 bp |

The cluster-minus-global net-return deficits are 189.19 bp/year for baseline and
194.65 bp/year for M1-star. Both treatments turn a near-flat global spread materially
negative.

### q=25% robustness

| method | net return | volatility | RF=0 Sharpe |
|---|---:|---:|---:|
| global | **0.0252%** | 16.9942% | **0.0851** |
| baseline cluster | -2.0209% | **5.6576%** | -0.3550 |
| M1-star cluster | -1.5268% | 6.3773% | -0.2294 |

The global q=20% and q=25% rows coincide because the small 11-contract class produces
the same selected sets at both thresholds.

FX verdict: cluster risk reduction is accompanied by substantially worse gross and net
payoff.

## Selection-channel diagnosis

Primary q=20% average construction sizes:

| class | method | available groups | long assets | short assets |
|---|---|---:|---:|---:|
| Equity | global | 1.00 | 6.00 | 6.00 |
| Equity | baseline / M1-star | 2.83 / 2.79 | 7.52 / 7.51 | 7.52 / 7.51 |
| Fixed Income | global | 1.00 | 4.47 | 4.47 |
| Fixed Income | baseline / M1-star | 2.52 / 2.52 | 5.10 / 5.07 | 5.10 / 5.07 |
| Commodities | global | 1.00 | 7.00 | 7.00 |
| Commodities | baseline / M1-star | 10.70 / 10.39 | 11.58 / 11.72 | 11.58 / 11.72 |
| FX | global | 1.00 | 3.00 | 3.00 |
| FX | baseline / M1-star | 3.12 / 3.41 | 2.38 / 2.22 | 2.38 / 2.22 |

The evidence points to a selection problem rather than a covariance-estimation problem.
Within-cluster ranking forces representation from many clusters even when entire clusters
have weak or oppositely signed momentum. The effect is strongest in Commodities, where
the method increases selected breadth by about 70% and discards the cross-cluster
component of the signal.

For the three-universe long-short programme, the next cluster-aware construction should
retain **global ranking for selection** and use clusters only for risk budgets, caps, or
covariance-aware scaling. That tests whether clustering can improve implementation without
removing the return source it is meant to diversify.

## Acceptance and verification

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| asset classes | 4/4 | exact | PASS |
| contracts classified | 95/95 | exact | PASS |
| CUA1 eligible dates / non-zero weights | 0 / 0 | exact | PASS |
| point-in-time eligible futures | 88 to 94 | dynamic | PASS |
| point-in-time eligible Commodities | 31 to 33 | dynamic | PASS |
| portfolio rows | 24 | 24 | PASS |
| cluster comparison rows | 16 | 16 | PASS |
| cluster return-and-Sharpe wins | 0/16 | reported honestly | PASS |
| exact +1/-1 construction rows | 24/24 | 100% | PASS |
| max pre-scale weight-sum error | 2.220e-16 | <= 1e-12 | PASS |
| max group-budget error | 0.000e+00 | <= 1e-15 | PASS |
| max long/short exposure error | 4.441e-16 | <= 1e-12 | PASS |
| max net exposure error | 6.939e-16 | <= 1e-12 | PASS |
| max gross exposure error | 1.332e-15 | <= 1e-12 | PASS |
| max cross-class weight leakage | 0.000e+00 | 0 | PASS |
| combined-book weight identities | 6/6 | <= 1e-12 | PASS |
| max combined-weight error | 5.551e-17 | <= 1e-12 | PASS |
| pre/post-window NAV rows | 0 / 0 | 0 / 0 | PASS |
| deterministic numerical artifacts | 7/7 byte-identical | 100% | PASS |
| focused pytest | 8/8 passed | all pass | PASS |
| independent payoff reconstruction | 12/12 primary rows | <= 5e-12 | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |
| EW payoff leg/comparison | 0 / 0 | 0 / 0 | PASS |

One pass over the 24 portfolios took 40.11 seconds and was replayed in full.

The fail-before-pass test was observed before implementation:

```text
ModuleNotFoundError: No module named
'papers.cluster_lineage_2026.replication.run_futures_asset_class_long_short'
```

Final verification output:

```text
........                                                                 [100%]
Futures asset-class long-short independent validation: PASS
(4 classes, 24 portfolios, 16 comparisons, 12 reconstructed payoffs,
6 combined-weight identities, 7 hashes)
All checks passed!
```

## Deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_futures_asset_class_long_short.py`

Checks:

- `papers/cluster_lineage_2026/replication/futures_asset_class_long_short_test.py`
- `papers/cluster_lineage_2026/replication/validate_futures_asset_class_long_short.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\futures_asset_class_long_short_u1_window
```

Machine-readable artifacts:

- `design.csv`
- `performance.csv`
- `comparison.csv`
- `construction_diagnostics.csv`
- `acceptance.csv`
- `horizon_diagnostic.csv`
- `combined_weight_reconstruction.csv`
- `runtime.csv`
- `determinism.csv`

No cluster cache was altered, no EW-all payoff comparison was introduced, and no file
was staged or pushed.
