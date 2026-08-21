# Futures 30/30/30/10 — U1-window execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE; earlier full-price-path futures performance superseded  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

The futures experiment now uses the exact U1 headline decision window: 203 monthly dates
from 2009-08-31 through 2026-06-30. Performance is measured only on W-WED observations
inside that calendar interval, from 2009-09-02 through 2026-06-24. Long-only and
long-short results are emitted and interpreted separately.

The corrected result does **not** support a primary q=20% cluster-outperformance claim.
Against the fair 30/30/30/10 global-within-sleeve control:

- baseline clustering earns 5.7376% versus 6.8665% net/year and has effectively identical
  Sharpe, 0.7885 versus 0.7886;
- M1-star earns 5.3919% net/year with 0.7505 Sharpe, below the global control on both;
- both clusters reduce volatility by about 1.45--1.49 percentage points.

At q=25%, baseline clustering produces the highest long-only Sharpe, 0.8318 versus
0.8102 for the same-budget global control, while still giving up 79.80 bp/year of net
return. M1-star again trails on return and Sharpe. No cluster portfolio beats its fair
global control on both return and Sharpe at either quantile.

Long-short is decisively negative for clustering. The same-budget global spreads earn
positive net returns and positive Sharpes at both q values; both cluster spreads have
negative net returns and negative Sharpes. The cluster long-short construction is rejected.

The unconstrained original global rank remains the return leader in both strategies.
It is included as an external reference, not as the clean clustering counterfactual,
because its strategic sleeve exposures are uncontrolled.

## Material horizon correction

The preceding futures report passed the complete historical price panel to the backtester.
Although decisions began in 2002, its NAV statistics had an implied measurement horizon of
67.0965 years because decades of pre-strategy cash observations remained in the NAV. Those
return, volatility, Sharpe, turnover, and cost statistics are superseded. The prior report
has been marked accordingly; caches, scores, partitions, and portfolio decisions were not
the defect.

The replacement uses one real W-WED mark from 2009-08-26 solely so qis can align the
2009-08-31 decision with implementation lag one. The performance view then removes that
alignment observation and every other observation outside the stated calendar window:

| horizon check | measured | required | result |
|---|---:|---:|---|
| monthly decisions | 203 | 203 | PASS |
| first decision | 2009-08-31 | exact | PASS |
| last decision | 2026-06-30 | exact | PASS |
| first measured NAV | 2009-09-02 | first W-WED in window | PASS |
| last measured NAV | 2026-06-24 | last W-WED in window | PASS |
| measured NAV years | 16.8077 | calendar-bounded | PASS |
| pre-window measured NAV rows | 0 | 0 | PASS |
| post-window measured NAV rows | 0 | 0 | PASS |

The accepted U1 headline artifact itself contains an analogous boundary issue: its NAV
starts on 2006-08-02, first becomes active on 2009-09-09, and ends on 2026-08-05. Its
reported annual returns imply 20.0082 years rather than the stated headline interval.
This task did not alter accepted U1 artifacts. Consequently, the new futures run matches
the **stated U1 calendar window**, but previously reported U1 payoff statistics must be
remeasured before making a numerical cross-universe performance comparison.

## Frozen construction

| component | value |
|---|---|
| universe | 95 global futures contracts |
| strategic budget | 30% Equity / 30% Commodities / 30% Fixed Income / 10% FX |
| signal | 48-week production momentum, latest four weeks skipped |
| primary selection | q = 0.20 |
| robustness | q = 0.25 |
| decisions | monthly |
| performance returns | W-WED, simple non-excess NAV returns |
| implementation lag | one W-WED observation |
| costs | 20 bp |
| cluster treatments | accepted baseline and M1-star caches |
| long-only exposure | +1 |
| long-short exposure | +1 / -1, gross 2, net 0 |

The 30/30/30/10 targets are imposed identically on the global-within-sleeve and cluster
legs, and independently on both signed sides in long-short portfolios. EW-all remains
reference-only for alpha and beta columns; it is not a ranking leg or payoff yardstick.

## Long-only results

### Primary q=20%

| method | gross return | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| original global | **11.4283%** | **10.2173%** | 13.3069% | **0.7977** | 2.7587 | 121.10 bp |
| 30/30/30/10 global | 8.2410% | 6.8665% | 8.8077% | 0.7886 | 3.2150 | 137.45 bp |
| 30/30/30/10 baseline cluster | 7.0782% | 5.7376% | 7.3616% | 0.7885 | 3.1646 | 134.06 bp |
| 30/30/30/10 M1-star cluster | 6.5197% | 5.3919% | **7.3153%** | 0.7505 | **2.6727** | **112.77 bp** |

Fair cluster-minus-global differences:

| cluster | net-return difference | volatility difference | Sharpe difference | turnover difference |
|---|---:|---:|---:|---:|
| baseline | -112.89 bp/year | -1.4462 pp | -0.0001 | -0.0504 |
| M1-star | -147.46 bp/year | -1.4924 pp | -0.0381 | -0.5423 |

Baseline's risk reduction is almost exactly offset by its lower return under the Sharpe
metric. M1-star's additional stability and turnover reduction do not compensate for its
larger payoff loss.

### q=25% robustness

| method | gross return | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| original global | **10.2116%** | **9.0780%** | 12.6288% | 0.7482 | 2.6060 | 113.36 bp |
| 30/30/30/10 global | 8.0296% | 6.7694% | 8.4261% | 0.8102 | 2.9523 | 126.02 bp |
| 30/30/30/10 baseline cluster | 7.2108% | 5.9714% | 7.2220% | **0.8318** | 2.9227 | 123.94 bp |
| 30/30/30/10 M1-star cluster | 6.4686% | 5.4414% | **7.2088%** | 0.7645 | **2.4346** | **102.72 bp** |

Baseline versus the same-budget global control has -79.80 bp/year net return,
-1.2042 percentage points volatility, and +0.0216 Sharpe. It also has +0.0836 Sharpe
versus the unconstrained original global rank, but 310.66 bp/year lower net return.

Long-only verdict: q=25% baseline offers the strongest risk-adjusted portfolio, but the
global rank remains the absolute-return winner. There is no joint return-and-Sharpe win.

## Long-short results

### Primary q=20%

| method | gross return | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| original global | **5.8040%** | **3.3149%** | 15.7129% | **0.2891** | 2.9826 | 248.91 bp |
| 30/30/30/10 global | 3.5699% | 0.9572% | 9.4743% | 0.1465 | 3.1995 | 261.27 bp |
| 30/30/30/10 baseline cluster | 1.3554% | -1.4078% | 4.6092% | -0.2875 | 3.4523 | 276.32 bp |
| 30/30/30/10 M1-star cluster | 1.0678% | -1.2977% | **4.5928%** | -0.2601 | **2.9582** | **236.55 bp** |

Baseline and M1-star trail the same-budget global spread by respectively 236.49 and
225.49 bp/year. Their Sharpe deficits are 0.4340 and 0.4066. Clustering lowers risk but
destroys the spread payoff before and after costs.

### q=25% robustness

| method | gross return | net return | volatility | RF=0 Sharpe | turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| original global | **4.5937%** | **2.2818%** | 14.5387% | **0.2261** | 2.7974 | 231.18 bp |
| 30/30/30/10 global | 3.6448% | 1.2775% | 8.3363% | 0.1931 | 2.8938 | 236.73 bp |
| 30/30/30/10 baseline cluster | 1.3423% | -1.2164% | 4.1763% | -0.2793 | 3.1948 | 255.88 bp |
| 30/30/30/10 M1-star cluster | 0.7481% | -1.4091% | **4.1470%** | -0.3224 | **2.7030** | **215.72 bp** |

The cluster-minus-global net-return deficits are 249.39 bp/year for baseline and
268.66 bp/year for M1-star. The long-short rejection is robust to q.

## Acceptance and verification

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| exact U1 decision dates | 203 | 203 | PASS |
| portfolio rows | 8 long-only + 8 long-short | exact separation | PASS |
| comparison rows | 4 long-only + 4 long-short | exact separation | PASS |
| portfolio construction rows | 16/16 PASS | 100% | PASS |
| max pre-scale weight-sum error | 4.441e-16 | <= 1e-12 | PASS |
| max within-sleeve group-budget error | 0.000e+00 | <= 1e-15 | PASS |
| max final weight/net-sum error | 4.441e-16 | <= 1e-12 | PASS |
| max top-level sleeve-budget error | 1.665e-16 | <= 1e-12 | PASS |
| max long/short exposure error | 1.110e-15 | <= 1e-12 | PASS |
| max gross exposure error | 2.665e-15 | <= 1e-12 | PASS |
| accepted q=20% global decision regression | 4.857e-17 | <= 1e-12 | PASS |
| pre/post-window measured NAV rows | 0 / 0 | 0 / 0 | PASS |
| deterministic numerical artifacts | 11/11 byte-identical | 100% | PASS |
| focused pytest | 6/6 passed | all pass | PASS |
| independent payoff reconstruction | 4/4 primary legs | <= 5e-12 | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |
| EW ranking/performance leg | 0 | 0 | PASS |

One pass over the 16 portfolios took 32.58 seconds and was replayed in full.

The fail-before-pass test was observed before implementation:

```text
ModuleNotFoundError: No module named
'papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window'
```

Final verification output:

```text
......                                                                   [100%]
Futures 30/30/30/10 U1-window independent validation: PASS
(203 decisions, 16 portfolios, 8 comparisons, 4 reconstructed payoffs, 11 hashes)
All checks passed!
```

## Deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_futures_weight_30303010_u1_window.py`

Checks:

- `papers/cluster_lineage_2026/replication/futures_weight_30303010_u1_window_test.py`
- `papers/cluster_lineage_2026/replication/validate_futures_weight_30303010_u1_window.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\futures_weight_30_30_30_10_u1_window
```

Primary strategy files are deliberately separate:

- `performance_long_only.csv`
- `comparison_long_only.csv`
- `performance_long_short.csv`
- `comparison_long_short.csv`

Additional machine-readable audit files:

- `design.csv`
- `allocation_diagnostics.csv`
- `acceptance.csv`
- `horizon_diagnostic.csv`
- `global_weight_regression.csv`
- `legacy_horizon_diagnostic.csv`
- `u1_reference_horizon_diagnostic.csv`
- `runtime.csv`
- `determinism.csv`

No cluster cache was altered, no EW-all payoff comparison was introduced, and no file
was staged or pushed.
