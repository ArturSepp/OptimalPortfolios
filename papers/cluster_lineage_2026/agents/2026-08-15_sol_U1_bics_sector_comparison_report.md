# U1 Bloomberg BICS sector comparison -- execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Owner specification executed

The U1 comparison transfers the owner-selected futures method without another
signal or method search:

- analysis window: 2009-08-31 through 2026-06-30 (203 decisions);
- strategy: +1/-1 long-short, q=25%, one-period implementation lag;
- signal: ROSAA production momentum, ME frequency, long span 12, no short span,
  volatility span 13, `MeanAdjType.EWMA`;
- cluster leg: cached U1 `M1_star` partition, U1 delta 0.0866, fallback 5;
- BICS leg: rank inside `bbg_bics_sector`; each available sector receives 1/G
  separately on the long and short sides and selected stocks split the sector
  budget equally;
- global leg: same signal and eligibility, ranked across the whole cross-section;
- transaction costs: 10 bp one-way;
- EW-all: market reference for beta and alpha only, never a payoff yardstick.

This is not the earlier U1 ME/span-36 classic-momentum cell. It intentionally
transfers the method and signal specification selected for futures, while using the
U1-specific M1-star calibration and cached U1 partitions.

## BICS coverage and matched-universe rule

The new metadata field contains 11 BICS sectors. Of 1,358 metadata rows, 1,333 are
classified and 25 are missing. Every missing row is listed in
`missing_bics_assets.csv`; nothing is silently dropped.

The primary comparison excludes a missing-BICS stock from **all three** legs on a
date when it is otherwise eligible. This keeps cluster, sector, and global books on
the identical investable cross-section. Full-U1 cluster/global results are retained
as a separate sensitivity.

| coverage diagnostic | measured |
|---|---:|
| metadata rows | 1,358 |
| BICS-classified metadata rows | 1,333 |
| missing BICS rows | 25 |
| missing rows active at least once in headline window | 13 |
| missing eligible stocks/date, min / median / max | 0 / 1 / 11 |
| BICS coverage share/date, min / mean / median / max | 98.17% / 99.52% / 99.84% / 100.00% |
| available BICS sectors/date | 11 on every date |

The labels are static security-level metadata combined with point-in-time index
eligibility; they are not a point-in-time history of BICS reclassifications.

## Primary matched-universe performance: volatility is the main result

| metric | M1-star cluster | BICS sector | global |
|---|---:|---:|---:|
| gross return/year | **-1.0687%** | -2.3867% | -2.8349% |
| net return/year | **-2.2512%** | -3.5830% | -4.0568% |
| RF=0 Sharpe | **-0.3263** | -0.3312 | -0.2572 |
| volatility/year | **6.3513%** | 9.5981% | 12.8129% |
| one-way turnover/year | **3.0113** | 3.0815 | 3.1609 |
| cost drag/year | **118.25 bp** | 119.63 bp | 122.19 bp |
| net total return | **-36.5910%** | -51.8119% | -56.3343% |

The central result is risk compression. The cluster leg reduces volatility by
**3.25 percentage points (33.8%)** relative to BICS-sector ranks and by **6.46
percentage points (50.4%)** relative to global ranks.

The cluster leg beats BICS-sector ranks by **133.19 bp/year net** and 131.80
bp/year gross. It also improves RF=0 Sharpe by 0.0048 and reduces turnover by
0.0702/year.

The cluster leg beats global rank by **180.56 bp/year net** and 176.62 bp/year
gross and reduces turnover by 0.1496/year, but its Sharpe is 0.0691 lower because
both returns are negative and the global leg has much higher volatility.

The finding is relative outperformance and volatility control only. All three legs
lose money both gross and net under the transferred futures signal specification,
consistent with long-short momentum being weak in this U1 implementation. This run
does not establish a profitable U1 momentum strategy.

## Missing-classification sensitivity

| leg | matched classified net/year | full U1 net/year | full minus matched |
|---|---:|---:|---:|
| M1-star cluster | -2.2512% | -2.4154% | -16.42 bp |
| global | -4.0568% | -4.0884% | -3.16 bp |

The relative ranking is not created by the BICS coverage filter. Reintroducing the
25 unclassified metadata names makes both available full-universe legs slightly
worse.

## Acceptance and verification

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| signal look-ahead | 0 days | <= 0 | PASS |
| monthly return round-trip error | 3.109e-14 | <= 1e-12 | PASS |
| missing BICS rows explicitly reported | 25 / 25 | exact | PASS |
| eligible M1-star memberships missing | 0 | 0 | PASS |
| primary/robustness performance rows | 5 / 5 | exact | PASS |
| maximum weight outside eligibility | 0 | <= 1e-12 | PASS |
| maximum long/short exposure error | 1.732e-14 | <= 1e-12 | PASS |
| maximum post-net equal-group budget error | 3.997e-15 | <= 1e-12 | PASS |
| maximum within-group selected-stock weight range | 0 | <= 1e-12 | PASS |
| deterministic artifacts | 10 / 10 byte-identical | 100% | PASS |

The independent post-construction check found exactly 11 sector groups on both
sides on all 203 dates, equal 1/11 sector budgets to floating-point precision, and
identical weights among selected stocks within each sector. One complete pass took
65.4 seconds; the deterministic verification executed two complete passes.

## Deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_u1_bics_sector_comparison.py`

Focused regression tests:

- `papers/cluster_lineage_2026/replication/u1_bics_sector_comparison_test.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\u1_bics_sector_vs_m1_star_owner_20260815
```

Primary artifacts are `performance.csv`, `comparison.csv`, `coverage_per_date.csv`,
`missing_bics_assets.csv`, `group_budget_diagnostics.csv`,
`exposure_diagnostics.csv`, `side_budget_diagnostics.csv`, `signal_preflight.csv`,
`acceptance.csv`, `design.csv`, and `determinism.csv`.

No covariance or cluster cache was changed, no model was refit, and no file was
staged or pushed.
