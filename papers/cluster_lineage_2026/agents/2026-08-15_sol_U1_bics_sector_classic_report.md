# U1 BICS comparison with classic 12-minus-1 momentum -- execution report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Repository scope:** `papers/cluster_lineage_2026/` only; no staging or push

## Specification

Only the signal changes from the preceding U1 BICS comparison. The classic score is
the sum of exactly 12 completed monthly log returns after shifting the monthly panel
by one period. The most recent month is therefore excluded.

The following remain frozen: the 2009-08-31 through 2026-06-30 headline window,
point-in-time eligibility, matched BICS-classified universe, q=25%, cached U1
M1-star partitions (delta 0.0866), fallback 5, equal group and selected-stock
budgets, +1/-1 long-short exposure, one-period implementation lag, and 10 bp
one-way costs.

## Primary result: volatility compression

| metric | M1-star cluster | BICS sector | global |
|---|---:|---:|---:|
| volatility/year | **6.0983%** | 9.3630% | 12.7709% |
| gross return/year | **-1.6623%** | -2.8551% | -3.0323% |
| net return/year | **-2.6816%** | -3.8221% | -4.0000% |
| RF=0 Sharpe | -0.4147 | -0.3681 | **-0.2536** |
| one-way turnover/year | 2.6080 | **2.4986** | 2.5042 |
| cost drag/year | 101.94 bp | **96.70 bp** | 96.77 bp |
| net total return | **-41.9500%** | -54.1467% | -55.8148% |

The cluster construction cuts volatility by **3.26 percentage points (34.9%)**
relative to BICS-sector ranks and by **6.67 percentage points (52.2%)** relative to
global ranks. This is the clearest result of the comparison.

The cluster leg also loses less: it beats BICS by **114.05 bp/year net** and global
by **131.84 bp/year net**. It does not beat either on Sharpe. With a negative return
numerator, lower volatility makes the conventional RF=0 Sharpe more negative, so the
Sharpe ordering should not be read as evidence against the observed risk compression.

All three gross and net returns remain negative. The classic signal therefore
confirms that this U1 long-short momentum implementation is weak in absolute terms;
clustering acts primarily as a volatility and loss-compression layer here.

## Classic versus ROSAA production signal

| leg | classic net/year | production net/year | classic minus production | classic vol | production vol |
|---|---:|---:|---:|---:|---:|
| M1-star cluster | -2.6816% | **-2.2512%** | -43.04 bp | **6.0983%** | 6.3513% |
| BICS sector | -3.8221% | **-3.5830%** | -23.90 bp | **9.3630%** | 9.5981% |
| global | **-4.0000%** | -4.0568% | +5.67 bp | **12.7709%** | 12.8129% |

Classic 12-minus-1 lowers volatility for every leg and materially lowers turnover,
but it does not rescue performance. It is worse for cluster and BICS returns and
only 5.67 bp/year better for global rank. The production signal remains preferable
for the cluster-vs-sector comparison under this fixed configuration.

## Coverage robustness

The primary universe rule remains matched: missing BICS stocks are excluded from all
three legs and all 25 metadata gaps remain itemised. Reintroducing those names in the
available full-U1 robustness rows produces cluster net return of -2.8147% and global
net return of -3.9734% per year. The main volatility ordering is unchanged.

## Acceptance and independent verification

| acceptance line | measured | tolerance | result |
|---|---:|---:|---|
| classic score reconstruction error | 2.665e-15 | <= 1e-14 | PASS |
| score NaN mask | identical | exact | PASS |
| latest included monthly return | one month before formation | exact | PASS |
| eligible M1-star memberships missing | 0 | 0 | PASS |
| maximum weight outside eligibility | 0 | <= 1e-12 | PASS |
| maximum long/short exposure error | 1.732e-14 | <= 1e-12 | PASS |
| maximum post-net group-budget error | 3.997e-15 | <= 1e-12 | PASS |
| maximum within-group selected-stock weight range | 0 | <= 1e-12 | PASS |
| full-U1 global payoff versus validated classic grid | exact | <= 1e-12 | PASS |
| deterministic artifacts | 12 / 12 byte-identical | 100% | PASS |

The score was independently reconstructed by explicit history slicing. The new
full-U1 global leg also reproduces the previously validated classic-grid global
payoff exactly, providing an independent end-to-end check of the backtest path. One
complete pass took 63.3 seconds; deterministic verification ran two passes.

## Deliverables

Runner:

- `papers/cluster_lineage_2026/replication/run_u1_bics_sector_comparison_classic.py`

Focused regression tests:

- `papers/cluster_lineage_2026/replication/u1_bics_sector_comparison_classic_test.py`

External outputs:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\u1_bics_sector_vs_m1_star_classic_12m_skip1_20260815
```

No covariance or cluster cache was changed, no model was refit, and no file was
staged or pushed.
