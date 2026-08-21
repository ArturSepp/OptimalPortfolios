# Futures global 30/30/30/10 at 10 bp — execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Outcome

The global-rank futures long-short portfolio is profitable after reducing the assumed cost
from 20 bp to 10 bp per one-way traded notional. Each signed side receives the strategic
budgets 30% Equity / 30% Fixed Income / 30% Commodities / 10% FX. Ranks are global within
each broad sleeve on its point-in-time eligible cross-section.

| selection | gross return/year | net return/year | volatility/year | RF=0 Sharpe | one-way turnover/year | cost drag/year | net total return |
|---|---:|---:|---:|---:|---:|---:|---:|
| q=20% primary | 3.0640% | 1.7548% | 9.4861% | 0.2285 | 3.2029 | 130.92 bp | 33.9609% |
| q=25% robustness | 3.4944% | 2.2994% | 8.4368% | 0.3102 | 2.9099 | 119.50 bp | 46.5349% |

The q=25% robustness setting is stronger on every reported payoff dimension: it has higher
net return and Sharpe, and lower volatility, turnover, and cost drag. It remains labelled a
robustness setting unless the owner explicitly promotes it over the frozen q=20% primary.

## Matched cost sensitivity

The 10 bp and 20 bp rows below use exactly the same ranks, selections, weights, decisions,
implementation lag, and performance window. Only the transaction-cost rate changes.

| q | net return/year at 10 bp | net return/year at 20 bp | 10 bp improvement | Sharpe at 10 bp | Sharpe at 20 bp | cost drag at 10 bp | cost drag at 20 bp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | 1.7548% | 0.4591% | +1.2957 pp | 0.2285 | 0.0941 | 130.92 bp | 260.49 bp |
| 25% | 2.2994% | 1.1155% | +1.1839 pp | 0.3102 | 0.1731 | 119.50 bp | 237.89 bp |

The drag is not exactly linear in the reported geometric annual return because the costs
compound through the NAV path. The underlying cost convention is 10 bp for each unit of
one-way traded notional; a direct +1 to -1 flip trades two units and therefore costs 20 bp.

## Construction and data conventions

| component | value |
|---|---|
| universe | futures, with `CUA1 Comdty` owner-excluded |
| analysis window | U1 headline calendar window |
| decision dates | 203 monthly dates, 2009-08-31 through 2026-06-30 |
| measured NAV | 878 W-WED rows, 2009-09-02 through 2026-06-24 |
| signal | production 48-week log-return sum, latest four weeks skipped |
| signal history rule | `sum(min_count=1)`; full 48-week history is not required |
| implementation lag | one W-WED observation |
| portfolio | +1 long / -1 short; gross 2, net 0 |
| per-side strategic budgets | 30% Equity / 30% Fixed Income / 30% Commodities / 10% FX |
| selection | q=20% primary; q=25% robustness |
| primary cost | 10 bp per one-way traded notional |
| comparison benchmark | global ranks within each sleeve; no EW-all payoff comparison |

The eligible cross-section is recomputed at every decision date. It ranges from 88 to 94
contracts after the owner exclusion: Equity 29, Fixed Income 18--21, Commodities 31--33,
and FX 10--11. Later-starting contracts therefore enter only when their scores become
available. The established partial-history rule is unchanged: an asset may receive a score
with fewer than 48 weekly observations because the signal sum uses `min_count=1`.

## Acceptance and verification

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| decision dates | 203 | 203 | PASS |
| pre/post-window measured NAV rows | 0 / 0 | 0 / 0 | PASS |
| primary cost | 10 bp one-way | exact | PASS |
| owner-excluded `CUA1 Comdty` maximum absolute weight | 0 | 0 | PASS |
| maximum top-level sleeve-budget error | 5.551e-17 | <= 1e-12 | PASS |
| maximum net-exposure error | 1.804e-16 | <= 1e-12 | PASS |
| maximum gross-exposure error | 2.665e-15 | <= 1e-12 | PASS |
| weighted standalone-sleeve reconstruction error | 1.388e-17 | <= 1e-12 | PASS |
| deterministic numerical artifacts | 8/8 byte-identical | 100% | PASS |
| focused pytest | 3/3 passed | all pass | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |
| EW-all payoff comparison | 0 | 0 | PASS |

The standalone reconstruction independently builds one global long-short book for each
broad sleeve, weights them 30/30/30/10, and reproduces the combined portfolio weights to
`1.388e-17` at both q values. One complete pass took 13.20 seconds and was replayed in full.

The numerical-defect test was also exercised by temporarily restoring 20 bp: the design
regression failed (`20.0 != 10.0`) before the requested 10 bp value was restored.

Final verification output:

```text
...                                                                      [100%]
All checks passed!
Futures global 30/30/30/10 at 10 bp: PASS (8/8 deterministic)
```

## Deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_futures_global_30303010_10bp.py`

Focused regression checks:

- `papers/cluster_lineage_2026/replication/futures_global_30303010_10bp_test.py`

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\futures_global_30_30_30_10_10bp_u1_window
```

The machine-readable output contains `performance.csv`, `cost_sensitivity.csv`,
`allocation_diagnostics.csv`, `availability_by_date.csv`, `acceptance.csv`,
`horizon_diagnostic.csv`, `standalone_weight_reconstruction.csv`, `design.csv`,
`runtime.csv`, and `determinism.csv`.

No cluster cache was altered, no EW-all payoff comparison was introduced, and no file was
staged or pushed.
