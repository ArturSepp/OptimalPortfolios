# U1 specification freeze and U2 BlackRock ETF grid execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE; primary global-outperformance objective NOT MET on U2  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

The selected U1 operating point is now frozen in
`replication/empirical_specs.py` as `U1_OPTIMAL_SPEC`. The BlackRock U2 experiment
transfers that specification without looking at U2 payoffs, then runs the full 28-cell
covariance grid for both long-only and +100%/-100% long-short portfolios.

The headline result is negative and should be recorded as such:

- **Long-only:** 0/28 cluster cells beat the same-signal global rank on net annual return;
  0/28 beat it on Sharpe. The least-bad cluster cell is ME/span 12: 3.9188% net versus
  6.4100% global, a -249.12 bp/year gap.
- **Long-short:** 0/28 cluster cells beat global net of 10 bp costs; 0/28 beat global
  Sharpe. W-THU/span 156 is a near tie: -1.1894% net versus -1.1697% global, a
  **-1.97 bp/year** gap, while cutting annualised volatility from 9.8183% to 4.3397%.
- The transferred U1 ME/span-36 cell does not travel: its net gaps are -319.10 bp/year
  long-only and -103.73 bp/year long-short.

This is not a hidden data or construction failure. Every source, no-look-ahead,
weight-budget, exposure, replay, and independent reconstruction check passed. The main
economic mechanism is exposure: global long-only rank averages 87.36% equity, while
group-equal clustering deliberately deconcentrates the duplicate-heavy ETF set into
equity, rates, commodity, and other correlation groups. The long-short global leg also
carries an average +30.14% equity / -29.61% fixed-income macro bet; the closest cluster
leg is nearly asset-class neutral.

## Frozen U1 operating specification

| component | frozen value |
|---|---:|
| covariance winner | ME returns, EWMA span 36 |
| clustering | Pearson, one-minus-rho, Ward, cutoff 0.60, demeaned |
| signal | exact ROSAA production monthly momentum |
| momentum spans | long 12, volatility 13, no short/reversal span |
| mean adjustment | `NONE` |
| cluster-score fallback | production minimum cluster size 5 |
| selection | q = 0.25, top within group; bottom within group for short book |
| cluster construction | `group_equal` |
| global construction | `asset_equal` |
| decision schedule | ME |
| payoff frequency | W-WED |
| implementation lag | one W-WED observation |
| costs | 10 bp |
| long-only exposure | +1 |
| long-short exposure | +1 / -1, gross 2, net 0 |

The machine-readable table is `specification.csv` in the output directory. The U1 cell is
marked by `is_transferred_u1_cell` in every U2 comparison row, so it cannot be confused
with an exploratory U2 grid winner.

## U2 data and eligibility

Inputs are the 2026-08-15 official U.S. iShares/BlackRock screener vintage and Bloomberg
dividend/capital-change-adjusted price histories created by
`data/fetch_blackrock_etf_data.py`:

| input | shape / size | SHA-256 |
|---|---:|---|
| `blackrock_us_etfs.csv` | 480 rows; 106,773 bytes | `63ee1002562092f6c98fd5d2bf5915a7b14313482bc72b971fa7ff3bf66af65d` |
| `blackrock_etf_metadata.csv` | 480 rows; 180,650 bytes | `d50899a2df282f0d48b844ab4c4c4a06482f42f4d553fc2b8625b7699a935fa6` |
| `blackrock_etf_adjusted_prices.csv` | 5,041 x 480; 11,920,803 bytes | `e367d8f990e5fbfe83338c43cc8033737c54be5249f343085e9d9d381dc665af` |
| `blackrock_etf_excess_log_returns.csv` | 5,041 x 480; 28,630,945 bytes | `89a62327e3fa35f9ecc0105edd80cf44ebcd0a6c1474ec757555628c7c0c2020` |

The experiment truncates returns at the last complete month, 2026-07-31, leaving 5,031
daily rows from 2006-08-01. Metadata, price, and return tickers match exactly. The return
availability mask exactly equals `price.notna() & price.shift().notna()` (0 mismatches).
All six Aladdin classification fields are complete: seven asset classes and 26 sub-asset
classes.

Eligibility begins after 12 valid W-WED returns and therefore uses only observed history.
There are 0 eligible ETFs on the first truncated estimation date because the supplied
history itself starts in August 2006, 162 on the headline start, a median 261.5, and 476
on 2026-07-31. The production signal becomes available on 2007-10-31 with 117 valid
scores; it has 154 valid scores at the headline start and 447 at the final date.

### Binding limitation

The screener is a **current-vintage survivor cohort**. Actual history start controls
entry, so no fund is backfilled before an observed return, but the dataset cannot include
BlackRock ETFs liquidated before 2026-08-15. Results must not be described as a
survivorship-free historical fund census.

Four products have legal inception dates more than one year before their available
Bloomberg history and enter only at actual history start: `SECU` and `MBBA` (2026-01-26),
`HIMU` (2025-02-10), and `BIDD` (2024-11-18). Four shorter 32-71 day gaps are also listed
in `coverage_anomalies.csv`. Nothing is silently dropped.

## Experiment design

The treatment grid is identical to the U1 grid:

- B and W-MON, W-TUE, W-WED, W-THU, W-FRI: spans 24, 36, 52, 156.
- ME: spans 12, 24, 36, 52.
- 240 point-in-time partition dates, 2006-08-31 through 2026-07-31.
- Primary headline window: 203 dates, 2009-08-31 through 2026-06-30.
- Labelled available-history robustness: 226 dates, 2007-10-31 through 2026-07-31.

At every cell, only the correlation partition changes. The production momentum input,
eligibility, q, rebalance dates, costs, and global leg are fixed. The cluster score is the
production `score_within_clusters` output with the five-name fallback. Cluster long-only
and each signed side receive equal budgets across available correlation groups; the global
leg is one asset-equal group. EW-all is not a payoff yardstick and never appears as a
performance leg. It is used only for the alpha and beta columns.

The payoff window retains exactly the last W-WED mark not after the first ME decision,
then applies `weight_implementation_lag=1`. This provides the required lag mark without
including years of pre-window flat NAV.

## Headline payoff results

### Global benchmarks

| strategy | gross return | net return | vol | Sharpe | one-way turnover | cost drag |
|---|---:|---:|---:|---:|---:|---:|
| long-only global | 7.1364% | 6.4100% | 13.1265% | 0.5411 | 3.4241 | 72.64 bp/yr |
| long-short global | 0.2366% | -1.1697% | 9.8183% | -0.0705 | 3.5299 | 140.63 bp/yr |

### Long-only leaders versus global

| rank | covariance cell | cluster net | delta vs global | cluster vol | cluster Sharpe | turnover |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ME / 12 | 3.9188% | -249.12 bp | 10.4853% | 0.4207 | 5.7284 |
| 2 | ME / 52 | 3.8832% | -252.68 bp | 9.9655% | 0.4337 | 4.6071 |
| 3 | W-MON / 24 | 3.8385% | -257.15 bp | 9.3973% | 0.4494 | 5.8488 |
| 4 | ME / 24 | 3.6142% | -279.58 bp | 10.1101% | 0.4033 | 5.1003 |
| 5 | W-THU / 156 | 3.4085% | -300.15 bp | 8.5554% | 0.4361 | 3.8991 |
| transferred U1 | ME / 36 | 3.2190% | -319.10 bp | 10.0876% | 0.3661 | 4.7591 |

All 28 cluster cells reduce volatility, but none offsets the directional return lost when
the 87%-equity global selection is deconcentrated.

### Long-short leaders versus global

| rank | covariance cell | cluster gross | gross delta | cluster net | net delta | cluster vol | turnover |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | W-THU / 156 | 0.5126% | +27.60 bp | -1.1894% | **-1.97 bp** | 4.3397% | 4.2693 |
| 2 | W-THU / 52 | 0.7912% | +55.46 bp | -1.2035% | -3.38 bp | 4.5930% | 4.9974 |
| 3 | W-WED / 156 | 0.3054% | +6.89 bp | -1.3782% | -20.85 bp | 4.4359% | 4.2308 |
| 4 | W-WED / 52 | 0.4497% | +21.32 bp | -1.5544% | -38.47 bp | 4.3352% | 5.0360 |
| 5 | W-MON / 24 | 0.6467% | +41.01 bp | -1.6999% | -53.02 bp | 5.1300% | 5.8919 |
| transferred U1 | ME / 36 | -0.2655% | -50.21 bp | -2.2070% | -103.73 bp | 5.0546% | 4.8524 |

Ten of 28 long-short cells beat global **before costs**, but none beats after the frozen
10 bp cost. W-THU/156 earns +27.60 bp/year more gross and pays +29.57 bp/year more cost
drag, producing the -1.97 bp/year net miss. A linear break-even calculation is 9.33 bp;
this is diagnostic only and does not alter the frozen 10 bp result.

### Available-history robustness

Over 2007-10-31 through 2026-07-31, long-only remains 0/28. Long-short has 4/28 positive
net deltas: W-WED/156 (+47.54 bp/year), W-MON/24 (+24.13 bp), W-TUE/156 (+13.35 bp),
and ME/52 (+9.45 bp). All four cluster spreads and the global spread still have negative
absolute net returns and Sharpe ratios, so this labelled robustness does not overturn the
headline conclusion.

## Exposure mechanism

Average headline decision-date exposures show that the global comparison embeds a much
larger asset-class allocation channel:

| leg | strategy | equity long | fixed-income long | equity short | fixed-income short | equity net | fixed-income net |
|---|---|---:|---:|---:|---:|---:|---:|
| global | long-only | 87.36% | 6.90% | - | - | 87.36% | 6.90% |
| ME/12 best long-only | long-only | 61.24% | 29.57% | - | - | 61.24% | 29.57% |
| ME/36 transferred | long-only | 57.08% | 33.50% | - | - | 57.08% | 33.50% |
| global | long-short | 87.36% | 6.90% | 57.22% | 36.51% | +30.14% | -29.61% |
| W-THU/156 closest | long-short | 50.02% | 40.17% | 45.57% | 43.47% | +4.44% | -3.30% |
| ME/36 transferred | long-short | 59.11% | 31.14% | 51.35% | 38.62% | +7.76% | -7.48% |

Thus the cluster portfolios are doing what group-equal construction requests: neutralising
duplicate product counts and macro group imbalance. On this heterogeneous ETF universe,
that behaviour reduces volatility in 28/28 cells but does not beat the global rank's
directional equity/rates exposure on net payoff.

## Acceptance record

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| ticker identity across four inputs | exact, 480/480/480/480 | exact | PASS |
| price/return availability mismatches | 0 | 0 | PASS |
| missing six-field Aladdin classifications | 0 | 0 | PASS |
| duplicate tickers | 0 | 0 | PASS |
| estimation dates | 240 | 240 | PASS |
| headline dates | 203 | 203 | PASS |
| covariance cells | 28 | 28 | PASS |
| transferred U1 marker | ME/36 | ME/36 | PASS |
| maximum signal look-ahead | 0 days | <= 0 | PASS |
| signal return round-trip error | 5.482e-16 | <= 1e-12 | PASS |
| partition caches | 28 x 240 dates | exact | PASS |
| portfolio construction rows | 116/116 PASS | 100% | PASS |
| max long-only weight-sum error | 2.220e-16 | <= 1e-12 | PASS |
| max pre-net group-budget error | 1.110e-16 | <= 1e-15 | PASS |
| max long exposure error | 3.997e-15 | <= 1e-12 | PASS |
| max short exposure error | 3.997e-15 | <= 1e-12 | PASS |
| max net exposure error | 2.019e-15 | <= 1e-12 | PASS |
| max gross exposure error | 1.177e-14 | <= 1e-12 | PASS |
| max post-net cluster L1 exposure | 1.837e-15 | <= 1e-12 | PASS |
| asset-class exposure rows | 812; every leg sums to target | exact within 5e-12 | PASS |
| deterministic numerical artifacts | 16/16 byte-identical | 100% | PASS |
| focused pytest | 6 passed | all pass | PASS |
| independent reconstruction | 28 caches, 116 constructions, 4 transfer payoffs, 16 hashes | all pass | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |
| EW used as payoff yardstick | 0 rows | 0 | PASS |

Cold partition construction used four workers. Observed per-cell wall times were about
25.1 seconds for B, 5.7-7.2 seconds for weekly cells, and 2.2-3.0 seconds for ME cells.
The 28 pickle caches occupy 26,159,840 bytes (24.95 MiB). A cache-hit payoff/replay pass
took 185.4 seconds; each cell's two windows and two strategies took 5.6-6.9 seconds.

## Code, commands, and deliverables

Runner and checks:

- `papers/cluster_lineage_2026/replication/empirical_specs.py`
- `papers/cluster_lineage_2026/replication/run_u2_blackrock_etf_grid.py`
- `papers/cluster_lineage_2026/replication/validate_u2_blackrock_etf_grid.py`
- `papers/cluster_lineage_2026/replication/u2_blackrock_etf_grid_test.py`

External cache/output directory:

`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/covariance_frequency_span_grid/blackrock_us_etfs/`

Primary persisted tables are `comparison_vs_global.csv`, `performance.csv`,
`win_summary.csv`, `u1_transfer_cell.csv`, `allocation_diagnostics.csv`,
`partition_summary.csv`, `acceptance.csv`, and `determinism.csv`. Partition pickles live
under its `partitions/` child directory.

Executed commands (with `CLUSTER_LINEAGE_OUTPUT_DIR` set to the external output root and
the local `qis` source checkout on `PYTHONPATH`):

```text
python -m papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid
python -m pytest papers/cluster_lineage_2026/replication/u2_blackrock_etf_grid_test.py -q
python -m papers.cluster_lineage_2026.replication.validate_u2_blackrock_etf_grid
ruff check --isolated --select E,F,W --line-length 100 <four changed Python files>
```

Verbatim terminal conclusions:

```text
BlackRock U2 long-only/long-short grid: PASS (16/16 deterministic)
......                                                                   [100%]
BlackRock U2 independent validation: PASS (28 caches, 116 constructions, 4 transfer payoffs, 16 hashes)
All checks passed!
```

## Deviations and interpretation boundary

1. The existing roadmap still labels U2 as global futures. This owner instruction replaces
   U2 with the BlackRock ETF cohort for this follow-up, but the historical accepted futures
   artifacts were not deleted or overwritten. A later roadmap replacement should make the
   universe change explicit.
2. A first payoff smoke test stopped before producing results because the cropped W-WED
   price panel began after the ME decision date. The slicer was corrected to retain exactly
   one prior W-WED mark. Partition estimates were already cached and were not refitted.
3. The local editable `qis` mapping pointed to its pre-`src/` path after main-branch work in
   the sibling checkout. Execution used the existing local `QuantInvestStrats/src` on
   `PYTHONPATH`; no dependency was installed or modified.
4. These are grid-search point estimates, not multiplicity-adjusted claims. The U2
   covariance winner is exploratory. The U1 ME/36 row is the only pre-specified transfer.

## Research conclusion

The exact U1 specification does **not** demonstrate global-rank outperformance on the
full heterogeneous BlackRock ETF cohort. The most promising follow-up is not another
unsmoothed covariance span: W-THU/156 already has a positive gross edge and misses net by
only 1.97 bp/year. The testable next lever is turnover control (for example, the accepted
partition smoother or a slower holding schedule) while keeping the signal and global
benchmark frozen. Long-only requires a separate design decision because group-equal and
global asset-equal portfolios intentionally carry very different strategic asset-class
budgets.

