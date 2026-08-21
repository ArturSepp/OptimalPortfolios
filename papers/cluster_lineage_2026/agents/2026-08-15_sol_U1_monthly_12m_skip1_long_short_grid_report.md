# U1 classic monthly 12m-ex-1m long-short grid report

**Date:** 2026-08-15  
**Status:** COMPLETE  
**Runner:** `papers/cluster_lineage_2026/replication/run_u1_covar_grid_long_short_monthly.py`  
**Validator:** `papers/cluster_lineage_2026/replication/validate_u1_covar_grid_long_short_monthly.py`  
**Focused tests:** `papers/cluster_lineage_2026/replication/u1_covar_grid_long_short_monthly_test.py`

## Outcome

The U1 E5/covariance grid just tested does **not** use the ROSAA production momentum
definition. It uses the paper's frozen finite raw-log-return signal: 48 weekly returns
after a four-week skip. ROSAA production instead uses benchmark-relative returns, EWMA
filtering and volatility normalisation before cross-sectional scoring. The distinction is
material and is documented below.

The requested classic monthly alternative was run as exactly 12 completed monthly log
returns with the most recent completed month excluded. At q=0.25, the headline leader is
**W-FRI covariance returns with EWMA span 156**. It returns -2.2710% net per year versus
-3.9734% for the matched global long-short rank, an improvement of **170.24 bp/year**.
Volatility is 6.3018% versus 12.8396%, a reduction of **653.77 bp/year**.

The monthly signal improves the best raw cell but is less broad than the frozen weekly
signal. It mean-variance dominates global in 13/28 headline cells, compared with 17/28 for
the weekly 48-minus-4 signal. The monthly signal improves the global-relative return delta
in only 5/28 matched cells and lowers that delta by 31.16 bp/year on average. The evidence
therefore supports the monthly signal as a useful robustness variant and makes W-FRI/156
the new raw-grid leader; it does not support replacing the frozen weekly signal yet.

## Momentum-definition audit

### Paper grid currently executed

The accepted E5/E5b and covariance-grid path calls the paper-local
`_raw_momentum_scores`, whose explicit contract is a finite log-return sum without
look-ahead. For U1 it consumes the frozen `UniverseSpec` settings:

- frequency W-WED;
- 48 included weekly observations;
- four skipped weekly observations;
- no volatility adjustment in the primary run;
- raw scores ranked globally or within the applicable taxonomy/cluster groups.

The paper implementation deliberately does not call `compute_momentum_alpha`, because that
function implements the production EWMA risk-adjusted momentum rather than the roadmap's
finite log-return sum. The paper harness continues to contain no ROSAA imports.

### ROSAA production definition

The production definition in `rosaa/alphas/alpha_aggregator.py` calls
`optimalportfolios.alphas.signals.momentum.compute_momentum_alpha` or its rolling-cluster
variant. For monthly assets, the production parameters and mechanics are:

- benchmark-relative monthly log returns;
- EWMA long filter with span 12;
- no short/reversal filter;
- EWMA volatility normalisation with span 13;
- cross-sectional scoring globally, within a supplied classification, or within rolling
  statistical clusters;
- global-score fallback for production clusters of size at most five.

The broader ROSAA default Beta signal group also combines momentum with low-beta before
the aggregator's final score transform. This study arm is momentum-only. Consequently,
neither the frozen weekly paper score nor the new classic monthly score is a ROSAA
production-alpha backtest.

A separate E8b **U3M** `S_prod` arm already reproduces this production momentum
primitive without importing ROSAA. That arm does not change the signal used by this U1
grid, and no U1 production-momentum grid has been run here.

## Classic monthly construction

The new score is

```text
score(t, i) = sum of r_ME(t-1, i), ..., r_ME(t-12, i)
```

where `r_ME(t, i)`, the latest completed monthly return at the formation date, is excluded.
Operationally this is `monthly_log_returns.shift(1).rolling(12, min_periods=12).sum()`.
Monthly returns are sums of the frozen daily U1 excess log returns. The excess-return
adjustment is common across assets and therefore does not change their cross-sectional
ordering.

Everything else is held fixed:

- q=0.25, +1 long and -1 short, gross exposure two and net exposure zero;
- U1 point-in-time index membership and eligibility;
- ME decision dates, one-period implementation lag, and 10 bp trading cost;
- cluster-side group-equal budgets and the asset-equal global long-short benchmark;
- unsmoothed cached baseline partitions for the same 28 covariance frequency/span cells;
- EW-all used only for beta/alpha measurement, never as a payoff yardstick.

No covariance or clustering model was re-estimated. The existing 28 partition caches were
consumed and only the signal, weights, and backtests were recomputed.

The frozen daily panel begins too late to supply 13 completed monthly observations at the
first ten estimation dates. Those dates, 2006-10-31 through 2007-07-31, remain explicitly
scoreless; nothing was filled or silently dropped. Results are reported for:

- primary headline window: 203 dates, 2009-08-31 through 2026-06-30;
- secondary monthly-available window: 228 dates, 2007-08-31 through 2026-07-31.

Monthly-versus-weekly comparisons use only the common headline window.

## Headline results

The monthly global q=0.25 spread returns -3.9734% net per year with 12.8396% volatility,
-0.249423 Sharpe, 2.5034 annual one-way turns, and 96.75 bp/year cost drag. Rows below are
ordered by annual net-return improvement over global.

| frequency | span | pre-cost return | net return | return delta | volatility | vol delta | Sharpe | turnover | cost bp/year |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| W-FRI | 156 | -0.7455% | -2.2710% | +1.7024 pp | 6.3018% | -6.5377 pp | -0.332740 | 3.8807 | 152.56 |
| W-TUE | 156 | -0.7558% | -2.3107% | +1.6627 pp | 5.8058% | -7.0338 pp | -0.373246 | 3.9557 | 155.49 |
| ME | 36 | -0.5797% | -2.3657% | +1.6078 pp | 6.9924% | -5.8472 pp | -0.307011 | 4.5411 | 178.60 |
| W-THU | 156 | -0.9898% | -2.5367% | +1.4367 pp | 6.6281% | -6.2115 pp | -0.354037 | 3.9428 | 154.69 |
| ME | 24 | -1.0782% | -2.9066% | +1.0668 pp | 8.2251% | -4.6145 pp | -0.316624 | 4.6723 | 182.84 |
| W-WED | 156 | -1.5524% | -3.0922% | +0.8812 pp | 5.9928% | -6.8468 pp | -0.493515 | 3.9466 | 153.99 |
| W-MON | 156 | -1.6149% | -3.1537% | +0.8197 pp | 6.3126% | -6.5270 pp | -0.475191 | 3.9462 | 153.88 |
| W-TUE | 52 | -1.5244% | -3.3795% | +0.5940 pp | 6.6620% | -6.1776 pp | -0.482055 | 4.7635 | 185.51 |
| ME | 12 | -1.5518% | -3.4434% | +0.5301 pp | 8.7887% | -4.0509 pp | -0.353798 | 4.8578 | 189.15 |
| B | 156 | -1.6908% | -3.4737% | +0.4997 pp | 6.0191% | -6.8205 pp | -0.556519 | 4.5817 | 178.29 |

All 28 headline cells have lower volatility than global. Thirteen also have higher net
return and therefore mean-variance dominate it. As in the weekly run, numerical Sharpe can
be worse even when both loss and risk are smaller because both strategies have negative
returns. The supported result is relative outperformance and systematic-risk cancellation,
not positive standalone momentum alpha.

Annual net-return deltas versus global, in percentage points:

| frequency | 12 | 24 | 36 | 52 | 156 |
|:--|--:|--:|--:|--:|--:|
| B | -- | -1.301 | -0.946 | -0.758 | +0.500 |
| ME | +0.530 | +1.067 | +1.608 | -0.207 | -- |
| W-FRI | -- | -0.266 | -1.256 | -0.130 | +1.702 |
| W-MON | -- | -2.233 | -0.739 | -1.095 | +0.820 |
| W-THU | -- | -0.329 | -1.031 | -0.676 | +1.437 |
| W-TUE | -- | +0.018 | +0.153 | +0.594 | +1.663 |
| W-WED | -- | +0.047 | -0.401 | -0.819 | +0.881 |

All available 156-observation cells beat global. ME and W-TUE each win in three and four
of four spans, respectively; the remaining frequency win counts are B 1/4, W-FRI 1/4,
W-MON 1/4, W-THU 1/4, and W-WED 2/4. The secondary monthly-available window is stronger:
26/28 cells mean-variance dominate global and three also have a higher numerical Sharpe.

## Monthly versus frozen weekly signal

The covariance partitions, eligibility, q, costs, and headline dates are identical in this
comparison. Only the raw momentum score changes.

| comparison | measured result |
|:--|--:|
| monthly cells with larger global-relative return delta | 5/28 |
| monthly-minus-weekly cluster net return, mean | -28.41 bp/year |
| monthly-minus-weekly cluster net return, median | -22.81 bp/year |
| monthly-minus-weekly global-relative delta, mean | -31.16 bp/year |
| monthly-minus-weekly global-relative delta, median | -25.55 bp/year |
| monthly-minus-weekly turnover, mean | -0.0247 annual turns |

At W-FRI/156, monthly improves the same cell's cluster net return by 24.23 bp/year and its
global-relative delta by 21.49 bp/year. At ME/36, the improvements are 7.49 and 4.75
bp/year, respectively. The monthly winner's -2.2710% return is also 16.16 bp/year above the
previous weekly-grid leader, W-THU/156, but that cross-cell comparison should not be read as
a pure signal effect.

Thus monthly 12-minus-1 sharpens the top result, while weekly 48-minus-4 is more robust
across covariance specifications.

## Why cluster volatility remains lower

At W-FRI/156 the cluster book's mean post-net group L1 exposure is 2.78e-16 and its maximum
is 1.07e-15. The global book's mean and maximum exposures to the same groups are 1.4494 and
1.7987. Both books have gross exposure two and net exposure zero. The cluster and global
books have 139.70 and 152.18 effective names per side, respectively, so the volatility
reduction is not a leverage or diversification-count artifact. It comes from neutralising
the long and short books within every covariance cluster.

The winner's beta versus EW-all is -0.1275, compared with -0.3124 for global. EW-all is
reported only as the market reference for this beta, in accordance with the binding
yardstick ruling.

## Acceptance

| check | measured | tolerance | verdict |
|:--|--:|--:|:--|
| grid coverage | 28 cells x 2 windows = 56 comparisons | 56 | PASS |
| performance rows | 56 cluster + 2 global | 58 | PASS |
| exact exposure rows | 58/58 | 58/58 | PASS |
| maximum long-exposure error | 6.106227e-15 | <= 1e-12 | PASS |
| maximum short-exposure error | 6.106227e-15 | <= 1e-12 | PASS |
| maximum net-exposure error | 1.514414e-15 | <= 1e-12 | PASS |
| maximum gross-exposure error | 1.909584e-14 | <= 1e-12 | PASS |
| maximum pre-net weight error | 2.220446e-16 | <= 1e-12 | PASS |
| maximum pre-net group-budget error | 1.387779e-17 | <= 1e-15 | PASS |
| maximum post-net cluster L1 exposure | 1.894752e-15 | <= 1e-12 | PASS |
| independent score reconstruction | 2.664535e-15 | <= 1e-14 | PASS |
| score NaN-mask agreement | exact | exact | PASS |
| warm-up-empty dates | 10, explicitly excluded only from secondary window | 10 | PASS |
| first score-available date | 2007-08-31 | 2007-08-31 | PASS |
| global-only payoff benchmark | all 56 comparisons | all | PASS |
| deterministic replay | 8/8 artifacts byte-identical | 8/8 | PASS |
| focused no-look-ahead/construction tests | 4/4 | 4/4 | PASS |
| ruff E/F/W on three new files | no findings | no findings | PASS |

The independent score reconstruction uses explicit date-by-date history slices, a separate
implementation from the vectorised shift-and-roll path. Its 2.665e-15 maximum difference
is floating-point accumulation order; the NaN mask is identical. The focused causality test
also perturbs the skipped month and all future returns and confirms that the formation-date
score is unchanged.

One cached pass took 309.09 seconds; deterministic verification ran two complete passes.
Final focused pytest output was:

```text
....                                                                     [100%]
```

Ruff output was:

```text
All checks passed!
```

Independent validator output:

```text
U1 monthly 12m-skip-1 covariance long-short grid validation: PASS
grid: 28 cells x 2 windows = 56 comparisons
signal: 12 included ME returns, skip 1; independent max error=2.665e-15
exposures: 58/58 PASS; max error=1.910e-14
headline winner: W-FRI span 156; net return delta=+0.017024; volatility delta=-0.065377
headline cells: global-relative return wins=13/28; mean-variance wins=13/28; monthly delta beats weekly delta=5/28
determinism: 8/8 artifacts byte-identical
```

## Interpretation and limits

1. Classic monthly 12-minus-1 modestly improves the strongest raw result, especially at
   W-FRI/156 and ME/36, but the frozen weekly signal wins across more covariance cells.
2. The repeated long-span advantage remains the more robust result than any weekday label.
   Every available 156-observation cell beats global under both signal variants.
3. W-FRI/156 has not passed the frozen taxonomy-fidelity band, bootstrap inference, or a
   multiple-testing correction. It is a research candidate, not an accepted production
   replacement.
4. This is not a ROSAA production-momentum result. A faithful U1 ROSAA arm would require
   the benchmark-relative EWMA risk-adjusted score, production cluster fallback, and the
   surrounding production signal-combination rules to be specified and tested separately.
5. Both strategies remain negative before and after costs. The supported empirical claim is
   that clustering beats the global rank on return and risk in the winning cells, not that
   this standalone momentum spread earns positive alpha.

## Deliverables

The output root is
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/covariance_frequency_span_grid/msci_us/long_short_grid_q_025_monthly_12m_skip1/`.
It contains:

- `performance.csv`, `comparison_vs_global.csv`, and `rankings.csv`;
- `comparison_vs_weekly_signal.csv`;
- `risk_diagnostics.csv` and `score_diagnostics.csv`;
- `signal_regression.csv` and `acceptance.csv`;
- `runtime.csv` and `determinism.csv`.

The runner, validator, test, report, and output artifacts remain local and ignored by git.
Nothing was staged or pushed.
