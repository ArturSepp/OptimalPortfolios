# Futures owner-exclusion recomputation — execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

The futures eligibility mask now excludes these seven actual source tickers on every
decision date. The owner classifies the screen as low-liquidity and freezes this as
the current eligible futures universe as of 2026-08-15:

- `BMR1 Curncy`
- `CUA1 Comdty`
- `IJ1 Comdty`
- `KC1 Comdty`
- `KM1 Index`
- `MES1 Index`
- `RS1 Comdty`

The owner-specified `MMR1 Curncy` is absent from both the return panel and metadata. It
was resolved explicitly to `BMR1 Curncy`, the plotted `BTC MINI` contract. The mapping
`MMR1 Curncy -> BMR1 Curncy` is frozen in the code and outputs; no requested name was
silently ignored.

The fixed best-relative specification was recomputed without searching a new grid:
M1-star versus matched global rank, q=25%, ROSAA monthly signal with long span 12,
short span None, volatility span 13, EWMA mean adjustment, fallback 5, +1/-1
long-short exposure, 30/30/30/10 sleeve budgets, one W-WED implementation lag, 10 bp
one-way costs, and the U1 headline calendar. Existing M1-star cluster partitions were
reused without refitting; the signal benchmark, valid ranks, available groups, weights,
turnover, costs, and payoff paths were recomputed under the updated eligibility mask.
The M1-star cell and the matched global comparator are now recorded as the selected
method specification, not candidates for another grid search.

## Recomputed performance

| metric | M1-star cluster | matched global | cluster minus global |
|---|---:|---:|---:|
| net return/year | **0.0297%** | -0.0163% | **+4.60 bp** |
| RF=0 Sharpe | 0.0179 | **0.0296** | -0.0117 |
| volatility/year | **4.4196%** | 8.2125% | -3.7929 pp |
| one-way turnover/year | **3.0330** | 3.4414 | -0.4084 |
| cost drag/year | **122.05 bp** | 138.46 bp | -16.41 bp |
| gross return/year | 1.2501% | **1.3683%** | -11.82 bp |
| net total return | **0.5001%** | -0.2733% | +0.7734 pp |

The cluster strategy now beats global on net annual return by 4.60 bp, but not on
Sharpe. The net-return crossover is entirely a cost/turnover result: global retains an
11.82 bp gross-return advantage, while cluster saves 16.41 bp/year of transaction-cost
drag. Both net returns are economically near zero.

Relative to the pre-exclusion result, cluster annual return fell by 33.73 bp and global
fell by 60.75 bp. The cluster-minus-global annual-return spread moved from -22.42 bp to
+4.60 bp, a +27.02 bp swing.

## Instrument attribution

| diagnostic | measured |
|---|---:|
| instruments eligible at least once | 88 |
| instruments with higher cluster contribution | 44 |
| instruments with higher global contribution | 44 |
| contribution correlation | 0.6466 |
| attributed cluster total return | 0.5001% |
| attributed global total return | -0.2733% |

Instrument P&L is exact prior-units-times-price-change less realised instrument cost,
normalised by beginning NAV. It reconciles exactly to each portfolio's NAV change.

## Acceptance and verification

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| actual source exclusions represented | 7 / 7 | exact | PASS |
| eligible-universe freeze and low-liquidity reason | owner-frozen 2026-08-15 | exact | PASS |
| selected-method specification freeze | owner-frozen 2026-08-15 | exact | PASS |
| excluded eligible observations | 0 | 0 | PASS |
| excluded rows in attribution | 0 | 0 | PASS |
| maximum excluded portfolio weight | 0 | 0 | PASS |
| cluster maximum step P&L error | 2.295e-14 | <= 1e-10 | PASS |
| global maximum step P&L error | 2.531e-14 | <= 1e-10 | PASS |
| cluster cumulative P&L error | 9.504e-14 | <= 1e-10 | PASS |
| global cumulative P&L error | 7.105e-15 | <= 1e-10 | PASS |
| frozen performance regression | 14 / 14 metrics | <= 1e-12 | PASS |
| deterministic numerical/Plotly artifacts | 7 / 7 byte-identical | 100% | PASS |

## Owner freeze ruling and research record

The binding futures specification from 2026-08-15 is:

- eligible universe: the point-in-time futures panel less the seven frozen
  low-liquidity exclusions above;
- selected cluster method: M1-star, q=25%, ROSAA ME 12-month signal, no short span,
  volatility span 13, EWMA mean adjustment, fallback 5;
- portfolio: +1/-1 long-short with 30/30/30/10 Equity / Fixed Income / Commodities / FX
  budgets per side, one-period implementation lag, and 10 bp one-way costs;
- comparator: the same-signal global rank under the identical eligibility, sleeve,
  timing, and cost conventions;
- analysis window: U1 headline dates, 2009-08-31 through 2026-06-30.

No further performance-driven universe edits or method selection are part of this
specification. The historical sequence remains disclosed: the owner supplied the
low-liquidity rationale after the exploratory contribution diagnostic and then froze
the universe and method. For publication-grade reproducibility, the data appendix
should attach the available objective liquidity evidence or threshold for these
contracts; subsequent runs use the frozen list regardless of realised P&L.

## Deliverables

Updated eligibility hook:

- `papers/cluster_lineage_2026/replication/run_backtests.py`

Recomputation and Plotly attribution runner:

- `papers/cluster_lineage_2026/replication/run_futures_best_relative_pnl_scatter.py`

Focused tests:

- `papers/cluster_lineage_2026/replication/futures_best_relative_pnl_scatter_test.py`
- `papers/cluster_lineage_2026/replication/futures_commodity_pnl_attribution_test.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\futures_prod_signal_grid_30_30_30_10_10bp_u1_window\best_relative_instrument_pnl_owner_exclusions_20260815
```

Primary files are `performance.csv`, `performance_comparison.csv`,
`performance_regression.csv`, `instrument_pnl.csv`, `reconciliation.csv`,
`design.csv`, `determinism.csv`, and
`best_relative_cluster_vs_global_instrument_pnl.html`.

No cache was altered, no cluster model was refit, and no file was staged or pushed.
