# U2 ROSAA long-only USD50m primary-cutoff ruling

Date: 2026-08-16  
Status: complete; owner selection recorded from the completed sensitivity

## Decision

The owner selected a point-in-time AUM cutoff strictly greater than USD50 million as the
primary eligibility rule for the U2 signal-ranking experiment. Other AUM cutoffs remain
labelled sensitivity rows and their outputs are preserved unchanged.

This selection was made after observing the sensitivity grid. It is therefore an empirical
operating-point choice, not an ex-ante or cross-validated optimum.

## Frozen U2 signal specification

- Universe: all BlackRock funds with sufficient return history and complete Bloomberg AUM.
- AUM statistic: arithmetic mean of the latest 12 completed calendar month-end observations.
- Eligibility: rolling AUM strictly greater than USD50m; missing or incomplete histories are
  ineligible.
- Eligibility is imposed before clustering, ROSAA benchmark construction, and ranking.
- Signal: ROSAA risk-adjusted momentum, monthly long span 12, short span 3, volatility span 13,
  and EWMA mean adjustment.
- Cluster-score fallback: minimum cluster size 10.
- Selection: top quartile, `q=0.25`.
- Construction: long-only and equal weight across every selected eligible fund; asset classes
  do not receive construction budgets.
- Rebalancing: every two months with implementation lag 1.
- Costs: 20 bp one way.
- Headline window: 2009-08-31 through 2026-06-30.

The frozen eligibility object is
`papers/cluster_lineage_2026/replication/empirical_specs.py` —
`U2_SIGNAL_PRIMARY_AUM_SPEC`.

## Primary measured result

| Method | Cumulative net | Annual net | Volatility | Sharpe | Annual turnover |
|---|---:|---:|---:|---:|---:|
| Global ROSAA rank | 274.348% | 8.161% | 14.772% | 0.607 | 1.536 |
| Cluster ROSAA rank | 220.529% | 7.168% | 13.290% | 0.589 | 1.876 |
| Cluster minus global | -53.819 pp | -0.993 pp | -1.483 pp | -0.018 | +0.341 |

The AUM50 row has the smallest cluster-versus-global Sharpe deficit in the six-cutoff grid.
Clustering lowers annualised volatility by 1.483 percentage points but does not exceed global
ranking in return or Sharpe.

Eligible breadth is 403 funds ever eligible, with 133 minimum, 249.916 mean, and 400 maximum
eligible funds per headline date. The average partition contains 14.946 clusters.

## Sensitivity context

| AUM cutoff | Cluster annual net | Cluster Sharpe | Cluster minus global Sharpe |
|---|---:|---:|---:|
| None | 7.089% | 0.580 | -0.027 |
| USD25m | 7.112% | 0.583 | -0.020 |
| **USD50m primary** | **7.168%** | **0.589** | **-0.018** |
| USD100m | 7.048% | 0.582 | -0.032 |
| USD250m | 7.264% | 0.595 | -0.023 |
| USD500m | 6.948% | 0.560 | -0.064 |

USD250m has the highest standalone cluster Sharpe, while USD50m is the most competitive matched
comparison against global ranking. USD500m removes too much breadth and is materially worse.

## Acceptance and reproduction

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Point-in-time AUM threshold | > USD50m | > USD50m | PASS |
| Missing eligible cluster memberships | 0 | 0 | PASS |
| Maximum weight error | 2.22e-15 | 1e-12 | PASS |
| Maximum signal look-ahead days | 0 | 0 | PASS |
| AUM50 accounting error | 7.18e-13 | 1e-10 | PASS |
| Sensitivity deterministic artifacts | 54 / 54 | 54 / 54 | PASS |

Runner:
`papers/cluster_lineage_2026/replication/run_u2_rosaa_long_only_aum_sensitivity.py`

Primary detailed output:
`papers/cluster_lineage_2026/local_outputs/e5b/u2_rosaa_short3_min10_equal_fund_long_only_aum_sensitivity_20260816/aum_50m/`

No files were staged or pushed. The paper tree remains gitignored.
