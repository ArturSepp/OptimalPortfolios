# U2 BlackRock funds: ROSAA versus classic 12-minus-1 -- execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Specification

The U1 long-short comparison design is transferred to the 480-fund BlackRock
current-vintage universe. The previously selected fund covariance cell is retained:
W-THU returns, EWMA span 156, cluster fallback 5. The headline window is
2009-08-31 through 2026-06-30 (203 monthly decisions), q=25%, +1/-1 exposure,
one-period implementation lag, and 10 bp one-way costs.

The three matched ranking legs are:

- cluster: equal budget across available statistical clusters, then equal weight
  across selected funds within each cluster;
- asset class: equal budget across available official BlackRock Aladdin
  `asset_class` groups, then equal weight across selected funds within each class;
- global: asset-equal top and bottom quartiles across all eligible scored funds.

The two signals change no portfolio-construction setting. ROSAA uses monthly returns,
long span 12, no short span, volatility span 13, and EWMA mean adjustment. Classic
12-minus-1 is the sum of exactly 12 completed monthly log returns after excluding the
most recent month. All seven official asset classes are mapped for all 480 funds.

## Results

| signal | leg | gross return/year | net return/year | volatility/year | RF=0 Sharpe | turnover/year | cost drag/year |
|---|---|---:|---:|---:|---:|---:|---:|
| ROSAA | cluster | 0.4643% | -1.1816% | **4.3010%** | -0.2549 | 4.1297 | 164.59 bp |
| ROSAA | equal asset class | **2.1662%** | **1.0441%** | 7.1416% | **0.1810** | 2.7648 | 112.21 bp |
| ROSAA | global | 0.4215% | -0.9067% | 9.8523% | -0.0430 | 3.3280 | 132.82 bp |
| classic 12m ex 1m | cluster | -0.0082% | -1.5271% | **4.6033%** | -0.3111 | 3.8234 | 151.89 bp |
| classic 12m ex 1m | equal asset class | **0.8328%** | **-0.1300%** | 7.7309% | **0.0216** | 2.3980 | 96.29 bp |
| classic 12m ex 1m | global | -0.1410% | -1.2198% | 10.4671% | -0.0648 | 2.7121 | 107.88 bp |

The cluster construction is again a strong volatility compressor. Under ROSAA it
cuts volatility by **39.8%** versus equal asset-class ranks and **56.3%** versus
global ranks. Under classic momentum the reductions are **40.5%** and **56.0%**.

It does not outperform the ranking benchmarks on net return in this funds sample.
The ROSAA cluster leg trails global by 27.49 bp/year and equal asset class by
222.57 bp/year. The classic cluster leg trails global by 30.73 bp/year and equal
asset class by 139.71 bp/year. Thus clustering supplies risk compression here, not
an absolute or relative return win.

ROSAA is preferable to classic for every leg. It improves annualised cluster net
return by **34.55 bp**, lowers cluster volatility by **30.23 bp**, and improves the
asset-class and global net returns by 117.42 bp and 31.31 bp respectively. The only
positive net-return book is the ROSAA equal-asset-class leg at 1.0441% per year.

## Coverage and interpretation

| quantity over headline dates | minimum | median | maximum |
|---|---:|---:|---:|
| point-in-time eligible funds | 162 | 283 | 474 |
| eligible funds with either valid signal | 154 | 271 | 446 |
| available official asset classes | 4 | 5 | 7 |

The official current-vintage classification counts are Equity 288, Fixed Income
154, Multi Asset 17, Real Estate 8, Commodity 7, Digital Assets 4, and Cash 2.
Groups without a valid scored fund are omitted from the per-date group count.

This is not a survivorship-free historical fund universe: the input is today's
BlackRock cohort with point-in-time availability inferred from return histories.
The changing eligible count is respected at every rebalance, but funds that closed
before the current product snapshot are absent. Return conclusions therefore remain
research diagnostics rather than publication-ready evidence until historical product
membership is available.

## Acceptance and independent verification

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| official classifications present | 480 | 480 | PASS |
| eligible cluster memberships missing | 0 | 0 | PASS |
| performance rows | 6 | 6 | PASS |
| ROSAA signal reconstruction error | 5.482e-16 | <= 1e-12 | PASS |
| classic signal reconstruction error | 4.441e-16 | <= 1e-14 | PASS |
| signal look-ahead, both signals | 0 days | <= 0 days | PASS |
| valid scored assets in headline window, both signals | minimum 154 | > 0 | PASS |
| warmup valid-asset minimum, reported separately | 0 | expected pre-headline warmup | PASS |
| deterministic artifacts | 9 / 9 byte-identical | 100% | PASS |

Every signal-by-leg construction line also passed:

| signal | leg | max weight outside eligibility | max exposure error | max group-budget error | max selected-fund weight range | tolerance |
|---|---|---:|---:|---:|---:|---:|
| ROSAA | cluster | 0 | 5.551e-15 | 1.665e-16 | 0 | 1e-12 |
| ROSAA | asset class | 0 | 8.438e-15 | 6.106e-16 | 0 | 1e-12 |
| ROSAA | global | 0 | 1.177e-14 | 2.331e-15 | 0 | 1e-12 |
| classic | cluster | 0 | 6.439e-15 | 1.943e-16 | 0 | 1e-12 |
| classic | asset class | 0 | 8.438e-15 | 6.106e-16 | 0 | 1e-12 |
| classic | global | 0 | 1.177e-14 | 2.331e-15 | 0 | 1e-12 |

The classic signal was independently reconstructed through explicit monthly history
slices, including an exact NaN-mask comparison. ROSAA prices were independently
round-tripped to monthly log returns. The deterministic check executed two full QIS
backtest passes; the recorded final pass took 24.7 seconds.

## Deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_u2_blackrock_signal_comparison.py`

Focused regression tests:

- `papers/cluster_lineage_2026/replication/u2_blackrock_signal_comparison_test.py`

Verified partition cache:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\covariance_frequency_span_grid\blackrock_us_etfs\partitions\W_THU_span_156.pkl
```

External outputs:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\covariance_frequency_span_grid\blackrock_us_etfs\equal_asset_class_rosaa_vs_classic_20260815
```

No covariance or cluster model was refit, no cache was changed, and no file was
staged or pushed.
