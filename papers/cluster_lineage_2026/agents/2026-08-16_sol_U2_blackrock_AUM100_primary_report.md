# U2 BlackRock USD 100m primary-cutoff report

Date: 2026-08-16  
Status: complete; owner selection materialized and all acceptance checks pass

## Decision

The owner selected a strictly greater-than USD 100 million point-in-time AUM cutoff as the
primary U2 BlackRock fund eligibility rule after reviewing the completed AUM sensitivity.
USD 100m supersedes USD 50m for future primary U2 analysis. The historical USD 50m code,
caches, and report remain unchanged for auditability and become a labelled robustness case.

Selection provenance is explicit: USD 100m was chosen after observing the threshold-sensitivity
table. It must not be described as an ex-ante cutoff or as a cross-validated optimum. Future
article exhibits should show the threshold sensitivity alongside the primary result.

## Canonical primary specification

- Bloomberg field: `FUND_TOTAL_ASSETS`, converted and audited in USD millions.
- AUM statistic: arithmetic average of the latest 12 completed calendar month-end values
  available before the decision date.
- Eligibility: rolling AUM strictly greater than USD 100m; missing or incomplete histories are
  ineligible.
- The AUM filter is applied before partition fitting and before both global and cluster ranks.
- Signal: ROSAA production risk-adjusted momentum.
- Clusters: W-THU returns, EWMA span 156.
- Selection: q = 25%.
- Gross sleeve budgets on each side: Equity 50%, Fixed Income 30%, Rest 20%.
- Cluster construction: group-equal.
- Primary payoff: global-rank long / cluster-rank short.
- Rebalancing: every two months; costs 20 bp one way.

## Primary measured result

Headline window: 2009-08-31 through 2026-06-30.

| Portfolio | Net return | Volatility | Sharpe |
|---|---:|---:|---:|
| Matched global rank | -0.771% | 7.561% | -0.065 |
| Primary hybrid | +0.192% | 6.715% | 0.062 |
| Hybrid minus global | +96.33 bp/year | -0.846 pp | +0.1266 |

Eligible breadth under USD 100m is 118 funds at headline start, 229 at the median headline
date, and 370 at headline end. The hybrid's gross return is 2.187%/year, annual one-way turnover
is 2.468, and cost drag is 199.4 bp/year.

The independent split-window net-return/Sharpe differences versus global are -21.29 bp/-0.034
for 2009-08-31..2017-12-31 and +95.14 bp/+0.130 for
2018-01-31..2026-06-30. These independent robustness windows reset holdings and their
two-month rebalance phase at the window start and therefore do not splice into the uninterrupted
headline result.

## Acceptance

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Primary threshold | USD 100m | USD 100m | PASS |
| Selected performance rows | 9 | 9 | PASS |
| Selected comparison rows | 6 | 6 | PASS |
| Source sensitivity acceptance passes | 8 | 8 | PASS |
| Source deterministic artifacts | 11 | 11 | PASS |
| Maximum signal lookahead days | 0 | 0 | PASS |
| Maximum weight/exposure error | 7.99360577730113e-15 | 1e-12 | PASS |
| Full-window hybrid net-return delta versus global | 0.00963286015945408 | > 0 | PASS |
| Full-window hybrid Sharpe delta versus global | 0.126587784079802 | > 0 | PASS |

The canonical materialization produced 13/13 byte-identical deterministic artifacts over two
replays. The six focused primary and sensitivity tests passed, and Ruff reported `All checks
passed!` for the new specification, runner, and tests. The source sensitivity payoff replay was
cache-first and took 90.7254 seconds; canonical row selection and writing took 0.0159 seconds.
An independent read-back comparison confirmed exact numerical equality between the primary and
source USD 100m performance, comparison, eligibility, and partition tables (4/4 PASS). The only
storage-level difference was harmless CSV inference of the constant threshold column as integer
`100` in the primary-only tables versus float in the multi-threshold source.

## Reproduction

Canonical runner:

`papers/cluster_lineage_2026/replication/run_u2_blackrock_primary.py`

Frozen specification:

`papers/cluster_lineage_2026/replication/empirical_specs.py` —
`U2_BLACKROCK_PRIMARY_AUM_SPEC`

Focused test:

`papers/cluster_lineage_2026/replication/u2_blackrock_primary_test.py`

External primary output directory:

`C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\covariance_frequency_span_grid\blackrock_us_etfs\aum100_primary_20260816\`

The directory contains the primary specification and selection record, eligibility and
partition diagnostics, selected performance and comparison tables, the source manifest,
acceptance table, runtime, and deterministic replay hashes.

No files were staged or pushed. The complete `papers/cluster_lineage_2026/` tree remains
gitignored as owner instructed.
