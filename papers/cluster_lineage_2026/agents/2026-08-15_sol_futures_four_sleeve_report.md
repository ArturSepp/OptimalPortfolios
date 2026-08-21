# Futures global-rank and four-sleeve execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE; equal-sleeve follow-up triggered and executed  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

The accepted futures global momentum rank is materially concentrated, so the owner's
conditional four-sleeve follow-up was necessary. Its average long-only allocation is
48.65% Commodities, 37.21% Equity, 10.14% Fixed Income, and 4.00% FX. Fixed Income is
absent on 194 of 295 decision dates and FX on 164. This breaches the predeclared maximum
ten-percentage-point deviation from an equal 25% budget.

The fair follow-up gives exactly 25% to Equity, Fixed Income, Commodities, and FX in both
global and cluster portfolios. Under the primary q=20% construction, M1-star clustering
does **not** beat the identically budgeted global rank on absolute return:

- M1-star cluster: 1.9529% net annual return;
- equal-sleeve global: 2.0642%;
- cluster-minus-global return: **-11.13 bp/year**.

It does improve risk-adjusted performance. M1-star reduces annualised volatility from
5.9210% to 4.9362%, raises RF=0 Sharpe from 0.3756 to 0.4173, reduces annualised one-way
turnover from 1.1252 to 0.8897, and reduces cost drag from 45.80 to 36.18 bp/year. Thus the
Sharpe difference is **+0.0417**, even though the return difference is negative.

The q=25% robustness is consistent: M1-star trails equal-sleeve global by 8.00 bp/year
but raises Sharpe by 0.0447. Long-short clustering has negative net return and Sharpe at
both quantiles and is rejected as a payoff strategy.

The appropriate conclusion is therefore:

- retain unconstrained global ranks if absolute return is the sole objective;
- use M1-star equal-sleeve clustering if the objective is lower volatility, lower turnover,
  and higher risk-adjusted performance under controlled strategic exposures;
- do not claim futures return outperformance from the equal-sleeve test.

## Frozen design

No covariance estimation or cluster search was run. The accepted futures caches were used
unchanged.

| component | value |
|---|---|
| universe | 95 global futures contracts |
| signal | 48-week log-return sum, latest four weeks skipped |
| primary selection | q = 0.20 |
| robustness | q = 0.25 |
| decisions | ME, 2002-01-31 through 2026-07-31, 295 dates |
| performance returns | W-WED, non-excess |
| implementation lag | one W-WED observation |
| costs | 20 bp |
| cluster treatments | baseline and calibrated M1-star only |
| long-only exposure | +1 |
| long-short exposure | +1 / -1, gross 2, net 0 |
| equal-sleeve budgets | 25% Equity / 25% Fixed Income / 25% Commodities / 25% FX |

The seven accepted taxonomy classes map completely into four broad sleeves:

| broad sleeve | accepted asset classes | contracts |
|---|---|---:|
| Equity | Equities | 29 |
| Fixed Income | Bonds, STIR | 21 |
| Commodities | Agriculture, Energy, Metals | 34 |
| FX | FX | 11 |

The original global leg ranks over the whole universe and weights selected contracts
equally. The equal-sleeve global control ranks independently within each broad sleeve and
splits each 25% sleeve budget equally over selected contracts. Cluster legs rank within
estimated correlation clusters split by broad sleeve; within each sleeve, available
clusters receive equal budgets and selected contracts split their cluster budget equally.

For long-short portfolios, the 25% budget is imposed independently on both signed sides.
Assets selected on both sides in singleton or tied groups are removed and each sleeve is
renormalised, leaving every sleeve net neutral. EW-all is reference-only for alpha and
beta columns and is never a ranking or payoff yardstick.

## Global-rank exposure diagnostic

| sleeve | mean weight | standard deviation | minimum | maximum | empty dates |
|---|---:|---:|---:|---:|---:|
| Commodities | 48.65% | 23.44% | 5.26% | 100.00% | 0 |
| Equity | 37.21% | 25.59% | 0.00% | 94.12% | 43 |
| Fixed Income | 10.14% | 20.22% | 0.00% | 88.89% | 194 |
| FX | 4.00% | 6.68% | 0.00% | 41.18% | 164 |

The largest mean deviation from 25% is 23.65 percentage points, against the predeclared
10-point trigger. Equal-sleeve testing was therefore activated before its payoffs were
read.

## Long-only results

### Primary q=20%

| method | gross return | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| original global | 3.4691% | **3.0639%** | 8.6470% | 0.3935 | 0.9890 | 40.51 bp |
| equal-sleeve global | 2.5222% | 2.0642% | 5.9210% | 0.3756 | 1.1252 | 45.80 bp |
| equal-sleeve baseline cluster | 2.3397% | 1.9133% | 5.0311% | 0.4026 | 1.0481 | 42.64 bp |
| equal-sleeve M1-star cluster | 2.3147% | 1.9529% | **4.9362%** | **0.4173** | **0.8897** | **36.18 bp** |

M1-star versus the fair equal-sleeve global control:

| metric | measured difference |
|---|---:|
| net annual return | -11.13 bp |
| gross annual return | -20.75 bp |
| annualised volatility | -0.9848 percentage points |
| RF=0 Sharpe | +0.0417 |
| annualised one-way turnover | -0.2355 |
| annual cost drag | -9.62 bp |

M1-star versus original global has -111.11 bp/year lower net return but +0.0238 higher
Sharpe and 3.7108 percentage points lower volatility. The original rank's larger absolute
return is therefore inseparable from its time-varying commodity/equity concentration.

### q=25% robustness

| method | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|
| original global | 2.9370% | 8.1083% | 0.3990 | 0.9228 | 37.79 bp |
| equal-sleeve global | 2.0849% | 5.7633% | 0.3879 | 1.0583 | 43.06 bp |
| equal-sleeve baseline cluster | 1.9845% | 4.9495% | 0.4225 | 0.9592 | 39.02 bp |
| equal-sleeve M1-star cluster | 2.0049% | **4.8734%** | **0.4325** | **0.8142** | **33.11 bp** |

M1-star's net-return difference against equal-sleeve global is -8.00 bp/year and its
Sharpe difference is +0.0447. Quantile choice does not change the verdict.

## Long-short results

| q | method | net return | volatility | RF=0 Sharpe | cost drag |
|---:|---|---:|---:|---:|---:|
| 0.20 | original global | 0.1694% | 9.7983% | 0.0667 | 86.90 bp |
| 0.20 | equal-sleeve global | -0.1684% | 5.4262% | -0.0040 | 90.86 bp |
| 0.20 | equal-sleeve M1-star cluster | -0.4771% | 2.6477% | -0.1673 | 87.60 bp |
| 0.25 | original global | 0.2662% | 9.0181% | 0.0750 | 80.92 bp |
| 0.25 | equal-sleeve global | 0.0404% | 5.0039% | 0.0330 | 84.15 bp |
| 0.25 | equal-sleeve M1-star cluster | -0.4368% | 2.3728% | -0.1726 | 78.51 bp |

The cluster spread's volatility reduction does not compensate for its lost payoff. No
long-short cluster treatment beats either global control on return or Sharpe.

## Acceptance record

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| futures contracts classified | 95/95 | exact | PASS |
| decision dates | 295 | 295 | PASS |
| portfolio rows | 16 | 16 | PASS |
| cluster comparison rows | 8 | 8 | PASS |
| portfolio construction rows | 16/16 PASS | 100% | PASS |
| max pre-scale weight-sum error | 4.441e-16 | <= 1e-12 | PASS |
| max within-sleeve group-budget error | 0.000e+00 | <= 1e-15 | PASS |
| max final weight/net-sum error | 4.441e-16 | <= 1e-12 | PASS |
| max top-level sleeve-budget error | 1.110e-16 | <= 1e-12 | PASS |
| max long exposure error | 1.110e-15 | <= 1e-12 | PASS |
| max short exposure error | 1.110e-15 | <= 1e-12 | PASS |
| max gross exposure error | 2.665e-15 | <= 1e-12 | PASS |
| accepted global weight regression | 4.857e-17 | <= 1e-12 | PASS |
| accepted global payoff regression | 4.974e-14 | <= 1e-12 | PASS |
| deterministic numerical artifacts | 7/7 byte-identical | 100% | PASS |
| focused pytest | 6/6 passed | all pass | PASS |
| independent payoff reconstruction | 2/2 primary long-only legs | <= 5e-12 | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |
| EW ranking/performance leg | 0 | 0 | PASS |

One complete pass over the 16 portfolios took 45.88 seconds and was replayed in full.

The required fail-before-pass test was observed before implementation:

```text
ModuleNotFoundError: No module named 'papers.cluster_lineage_2026.replication.run_futures_sleeve_grid'
```

Final focused pytest output:

```text
......                                                                   [100%]
```

Independent validator output:

```text
Futures four-sleeve independent validation: PASS (95 contracts, 16 portfolios, 8 comparisons, 2 reconstructed payoffs, 7 hashes)
```

## Code and deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_futures_sleeve_grid.py`

Checks:

- `papers/cluster_lineage_2026/replication/futures_sleeve_grid_test.py`
- `papers/cluster_lineage_2026/replication/validate_futures_sleeve_grid.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\futures_four_sleeve
```

Machine-readable artifacts:

- `design.csv`
- `global_exposure_diagnostic.csv`
- `performance.csv`
- `comparison.csv`
- `allocation_diagnostics.csv`
- `acceptance.csv`
- `global_regression.csv`
- `runtime.csv`
- `determinism.csv`

No futures cache was altered, no EW-all payoff comparison was introduced, and no file was
staged or pushed.

## Recommended next step

If absolute-return competitiveness remains the objective, freeze a small strategic-budget
grid before inspecting its payoffs and apply each budget identically to global and cluster
legs. The present diagnostic suggests centring that grid around rounded versions of the
global rank's observed allocation while retaining the equal 25/25/25/25 control. Use
separate early and recent evaluation windows so a weight is not selected and judged on the
same sample. The equal-sleeve result should remain the primary fair risk-control exhibit,
not be discarded because its return verdict is negative.
