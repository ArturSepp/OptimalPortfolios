# U2 BlackRock AUM-cutoff sensitivity report

Date: 2026-08-16  
Status: complete; all declared acceptance checks pass

## Outcome

The fixed selected fund model was rerun with no AUM cutoff and with point-in-time cutoffs of
USD 25m, 50m, 100m, 250m, and 500m. USD 50m was the primary eligibility rule when this
sensitivity was run. After reviewing the completed table, the owner selected USD 100m as the
new primary cutoff. That decision is recorded as post-sensitivity selection and must not be
described as ex ante.

Over the full 2009-08-31 through 2026-06-30 headline window, the selected hybrid beats its
matched global-rank book in net return and Sharpe at every cutoff. The largest descriptive
full-window edge occurs at USD 100m: +96.33 bp/year net return and +0.1266 Sharpe versus
global. Its own net return is +0.192%/year with Sharpe 0.062.

The result is not monotone in AUM and is not stable across both subperiods. Every cutoff loses
to global in the independent 2009-08-31 through 2017-12-31 selection-window backtest. USD 50m
and higher win in the independent 2018-01-31 through 2026-06-30 evaluation-window backtest,
while no AUM filter and USD 25m do not. The owner-directed USD 100m choice therefore remains
paired with the complete threshold sweep; its full-window advantage is not presented as a
cross-validated cutoff optimization.

The pure cluster book underperforms global at every cutoff. The useful construction remains the
selected hybrid: global-rank long side and cluster-rank short side.

## Frozen model and sensitivity rule

- Signal: ROSAA production risk-adjusted momentum.
- Clusters: W-THU returns, EWMA span 156.
- Selection: q = 25% on each long and short side.
- Broad-sleeve gross budgets: Equity 50%, Fixed Income 30%, Rest 20% on each side.
- Cluster construction: group-equal.
- Selected hybrid: global long / cluster short.
- Rebalancing: every two months, implementation lag unchanged.
- Costs: 20 bp one way.
- AUM variable: Bloomberg ETF fund total assets in USD millions.
- AUM observation: arithmetic mean of the 12 completed calendar month-end observations
  available before each decision date.
- Eligibility: rolling AUM strictly greater than the cutoff; missing or incomplete AUM is
  ineligible. The same eligibility panel is applied to global and cluster books, and clusters
  are refit after filtering.

## Full-window results

Returns, volatility, and cost drag are annualized. Sharpe is the frozen `sharpe_rf0` measure.
Breadth is eligible funds at headline start / median headline date / headline end.

| AUM cutoff | Breadth | Global net | Global Sharpe | Pure cluster net | Pure cluster Sharpe | Hybrid net | Hybrid Sharpe | Hybrid - global net | Hybrid - global Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| History only | 162 / 283 / 474 | -0.656% | -0.054 | -1.316% | -0.327 | -0.126% | 0.010 | +52.97 bp | +0.0637 |
| > USD 25m | 140 / 262 / 419 | -0.605% | -0.048 | -2.391% | -0.620 | -0.157% | 0.004 | +44.75 bp | +0.0516 |
| > USD 50m | 133 / 245 / 400 | -0.873% | -0.081 | -2.430% | -0.598 | -0.534% | -0.050 | +33.85 bp | +0.0312 |
| > USD 100m | 118 / 229 / 370 | -0.771% | -0.065 | -2.144% | -0.528 | +0.192% | 0.062 | +96.33 bp | +0.1266 |
| > USD 250m | 87 / 194 / 318 | -0.689% | -0.053 | -2.556% | -0.580 | -0.579% | -0.052 | +10.97 bp | +0.0014 |
| > USD 500m | 62 / 157 / 275 | -0.752% | -0.062 | -1.407% | -0.260 | +0.036% | 0.039 | +78.81 bp | +0.1011 |

The USD 100m improvement is primarily a gross-payoff change, not a cost reduction. Its hybrid
gross return is 2.187%/year versus 1.025% for matched global; hybrid cost drag is 199.4 bp/year
versus 179.5 bp/year. Across the sweep, hybrid annual one-way turnover remains 2.468 to 2.556
and cost drag remains 198.8 to 205.1 bp/year.

## Split-window robustness

Each cell reports hybrid minus matched global as annualized net-return bp / Sharpe difference.

| AUM cutoff | Selection 2009-08-31..2017-12-31 | Evaluation 2018-01-31..2026-06-30 | Full 2009-08-31..2026-06-30 |
|---|---:|---:|---:|
| History only | -21.60 / -0.0255 | -29.64 / -0.0767 | +52.97 / +0.0637 |
| > USD 25m | -15.79 / -0.0120 | -15.62 / -0.0410 | +44.75 / +0.0516 |
| > USD 50m | -53.69 / -0.0840 | +19.36 / +0.0150 | +33.85 / +0.0312 |
| > USD 100m | -21.29 / -0.0340 | +95.14 / +0.1300 | +96.33 / +0.1266 |
| > USD 250m | -186.42 / -0.2840 | +65.83 / +0.0840 | +10.97 / +0.0014 |
| > USD 500m | -66.63 / -0.0970 | +124.85 / +0.1690 | +78.81 / +0.1011 |

These subperiods are independent window-start robustness backtests. Each initializes holdings
and includes its own first date in the every-two-month rebalance grid. The evaluation grid is
therefore phase-shifted relative to the uninterrupted full-window grid, so split-window returns
are not intended to splice algebraically into the full-window return.

## Acceptance

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| History-only partition membership mismatches versus frozen unfiltered cache | 0 | 0 | PASS |
| USD 50m partition membership mismatches versus frozen AUM50 cache | 0 | 0 | PASS |
| Eligible memberships missing from partitions | 0 | 0 | PASS |
| Maximum weight/exposure error | 9.10382880192628e-15 | 1e-12 | PASS |
| Maximum signal lookahead days | 0 | 0 | PASS |
| Declared performance rows | 54 | 54 | PASS |
| Declared comparison rows | 36 | 36 | PASS |
| USD 50m prior-run numerical regression error | 4.54747350886464e-13 | 1e-12 | PASS |

The cache-first replay processed six filter-specific partitions and 54 performance rows in
67.0947 seconds. Eleven deterministic artifacts were hashed after two backtest replays; 11/11
were byte-identical.

## Verification

Focused tests:

```text
........                                                                 [100%]
```

Ruff on the sensitivity runner and its tests:

```text
All checks passed!
```

## Reproduction

Runner:

`papers/cluster_lineage_2026/replication/run_u2_blackrock_aum_sensitivity.py`

Focused test:

`papers/cluster_lineage_2026/replication/u2_blackrock_aum_sensitivity_test.py`

External cache and output directory:

`C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\covariance_frequency_span_grid\blackrock_us_etfs\aum50_filter_20260816\threshold_sensitivity\`

Key artifacts: `partitions.pkl`, `specification.csv`, `eligibility_by_date.csv`,
`eligibility_summary.csv`, `partition_diagnostics.csv`, `signal_diagnostics.csv`,
`weight_diagnostics.csv`, `performance.csv`, `comparison_vs_global.csv`,
`sensitivity_vs_50m.csv`, `full_window_summary.csv`, `acceptance.csv`, `runtime.csv`, and
`determinism.csv`.

No files were staged or pushed. The complete `papers/cluster_lineage_2026/` tree remains
gitignored as owner instructed.
