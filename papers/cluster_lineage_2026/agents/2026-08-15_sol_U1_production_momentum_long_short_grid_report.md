# U1 production-momentum covariance long-short grid report

**Date:** 2026-08-15  
**Status:** COMPLETE  
**Runner:** papers/cluster_lineage_2026/replication/run_u1_covar_grid_long_short_prod.py  
**Validator:** papers/cluster_lineage_2026/replication/validate_u1_covar_grid_long_short_prod.py  
**Focused tests:** papers/cluster_lineage_2026/replication/u1_covar_grid_long_short_prod_test.py

## Outcome

Production momentum is decisively better than classic monthly momentum for the stated
criterion: beating the corresponding global rank in the largest number of U1 covariance
cells.

| signal | same-signal global return wins | mean-variance wins | best cell | best net-return delta |
|:--|--:|--:|:--|--:|
| ROSAA production, exact monthly 12/13 | **27/28** | **27/28** | ME/36 | **+276.92 bp/year** |
| production mechanics, calendar-scaled 12 months | **26/28** | **26/28** | ME/36 | **+276.92 bp/year** |
| paper raw weekly 48w skip 4w | 17/28 | 17/28 | W-THU/156 | +156.82 bp/year |
| classic monthly 12m skip 1m | 13/28 | 13/28 | W-FRI/156 | +170.24 bp/year |

The faithful production setting is therefore the best operating point for breadth. The
user-requested cadence-scaled extension is nearly as broad, but loses one additional cell.
Production's relative-return advantage exceeds classic monthly's in 27/28 matched cells
and exceeds the frozen weekly signal's in 27/28.

The improvement is not solely a weaker production global benchmark. The exact-production
global spread is 41.62 bp/year worse than the classic global spread, but the production
cluster portfolios themselves improve on their classic counterparts in 23/28 cells and by
88.17 bp/year on average. They improve on the raw-weekly cluster portfolios in 22/28 cells
and by 59.76 bp/year on average.

## What "production momentum" means here

The exact-monthly arm consumes the validated E8b production momentum primitive:

- point-in-time eligible equal-weight benchmark-relative log returns;
- monthly return cadence;
- EWMA long span 12;
- EWMA volatility span 13;
- no short/reversal filter;
- MeanAdjType.NONE;
- cross-sectional production scoring globally or within the current statistical cluster;
- global-score fallback for clusters of size at most five.

The portfolio layer remains the frozen E5 long-short experiment: q=0.25, group-equal
budgets on the cluster side, asset-equal global rank, +1 long and -1 short, ME decisions,
implementation lag one, and 10 bp costs. Thus this isolates the production momentum
primitive inside the accepted paper construction. It is not the complete ROSAA Beta stack,
which also combines low-beta and applies the aggregator's final transformation.

The calendar-scaled arm retains the production mechanics but changes observation counts so
the momentum horizon is always twelve months:

| signal cadence | long span | long horizon | volatility span | approximate vol horizon |
|:--|--:|--:|--:|--:|
| B | 252 | 12 months | 273 | 13 months |
| W-MON through W-FRI | 52 | 12 months | 56 | 12.92 months |
| ME | 12 | 12 months | 13 | 13 months |

This calendar-scaled arm is a controlled extension, not a claim that ROSAA production
currently runs daily or weekly momentum. The exact arm remains ME 12/13 for every
covariance cell. Exact and scaled ME outputs coincide with maximum error 0.0.

Every covariance cell is compared only with a global portfolio formed from the same
production variant and signal cadence. EW-all remains solely the beta/alpha market
reference and is never a payoff comparator.

## Frozen backtest design

- The 28 unsmoothed covariance partitions are unchanged and loaded cache-first.
- Covariance cells are B and W-MON through W-FRI at spans 24, 36, 52, and 156, plus
  ME at spans 12, 24, 36, and 52.
- U1 point-in-time index membership and investability are unchanged.
- q=0.25, group-equal cluster budgets, gross exposure two, and net exposure zero are
  unchanged.
- Rebalancing is ME with implementation lag one and 10 bp trading cost.
- The headline window is 203 dates, 2009-08-31 through 2026-06-30.
- The common production-available robustness window is 225 dates, 2007-11-30 through
  2026-07-31. It starts only when every daily, weekly, and monthly signal has completed
  warm-up.
- No covariance model or cluster model was refit.

For B and weekly production scores, each ME decision consumes only the latest signal
timestamp not after the decision date. The maximum measured timestamp look-ahead is zero
days.

## Best exact-production result

ME covariance returns with EWMA span 36 is the leader under both production variants.

| metric | ME/36 production cluster | same-signal global | cluster minus global |
|:--|--:|--:|--:|
| pre-cost annual return | +0.2985% | -3.1189% | +3.4174 pp |
| net annual return | -1.6204% | -4.3897% | **+2.7692 pp** |
| volatility | 6.8166% | 12.8415% | **-6.0249 pp** |
| Sharpe, rf=0 | -0.205318 | -0.283345 | +0.078026 |
| one-way annual turnover | 4.8427 | 3.2970 | +1.5457 |
| cost drag | 191.89 bp/year | 127.07 bp/year | +64.82 bp/year |
| beta versus EW-all | -0.1244 | -0.3795 | +0.2551 |

Production ME/36 improves the same cell's cluster net return by 74.52 bp/year relative to
classic monthly and raises its global-relative edge from 160.78 to 276.92 bp/year. Its
turnover and cost drag are higher, but its pre-cost result becomes marginally positive.
The net long-short result remains negative.

The leading exact-production cells are:

| covariance frequency | span | cluster net return | global delta | volatility | turnover | cost bp/year |
|:--|--:|--:|--:|--:|--:|--:|
| ME | 36 | -1.6204% | +2.7692 pp | 6.8166% | 4.8427 | 191.89 |
| W-FRI | 156 | -1.8893% | +2.5004 pp | 6.0905% | 4.2168 | 166.44 |
| ME | 24 | -2.1223% | +2.2673 pp | 8.1202% | 4.9507 | 195.25 |
| W-THU | 156 | -2.3365% | +2.0532 pp | 6.5167% | 4.2897 | 168.68 |
| W-WED | 52 | -2.3368% | +2.0528 pp | 7.0018% | 5.0152 | 197.27 |
| W-WED | 156 | -2.4660% | +1.9236 pp | 6.5040% | 4.2950 | 168.73 |
| B | 156 | -2.6311% | +1.7586 pp | 5.9376% | 4.8823 | 191.55 |
| W-FRI | 52 | -2.6476% | +1.7421 pp | 7.2217% | 4.9382 | 193.69 |

## Breadth across the grid

Exact-production annual net-return deltas versus its common monthly global rank, in
percentage points:

| frequency | 12 | 24 | 36 | 52 | 156 |
|:--|--:|--:|--:|--:|--:|
| B | -- | +1.149 | +1.072 | +1.262 | +1.759 |
| ME | +0.852 | +2.267 | +2.769 | +1.730 | -- |
| W-FRI | -- | +0.851 | -0.117 | +1.742 | +2.500 |
| W-MON | -- | +0.257 | +0.469 | +0.740 | +1.165 |
| W-THU | -- | +1.184 | +0.990 | +1.222 | +2.053 |
| W-TUE | -- | +0.213 | +0.785 | +0.918 | +1.016 |
| W-WED | -- | +1.508 | +0.841 | +2.053 | +1.924 |

Calendar-scaled production annual net-return deltas versus its cadence-matched global rank:

| frequency | 12 | 24 | 36 | 52 | 156 |
|:--|--:|--:|--:|--:|--:|
| B | -- | +1.943 | +2.352 | +2.139 | +2.614 |
| ME | +0.852 | +2.267 | +2.769 | +1.730 | -- |
| W-FRI | -- | +0.290 | -0.452 | +0.981 | +1.822 |
| W-MON | -- | -0.584 | +0.131 | +0.020 | +1.231 |
| W-THU | -- | +2.046 | +0.974 | +1.406 | +2.676 |
| W-TUE | -- | +0.125 | +0.528 | +1.232 | +1.398 |
| W-WED | -- | +1.798 | +1.154 | +1.691 | +2.435 |

All 28 cells have lower volatility than their matched globals under both production
variants. Exact production has higher numerical Sharpe in 3/28 cells and scaled production
in 4/28; negative returns make Sharpe ordering less informative than the joint return/risk
comparison. In the common production-available window, both production variants
mean-variance dominate global in 28/28 cells.

The only headline failures are:

| variant | cell | cluster return | global return | return delta | cluster vol | global vol |
|:--|:--|--:|--:|--:|--:|--:|
| exact monthly | W-FRI/36 | -4.5064% | -4.3897% | -11.68 bp | 7.2067% | 12.8415% |
| calendar-scaled | W-FRI/36 | -4.5255% | -4.0734% | -45.21 bp | 7.4687% | 13.1885% |
| calendar-scaled | W-MON/24 | -5.0028% | -4.4190% | -58.38 bp | 8.6467% | 13.0237% |

These cells still reduce volatility, but they do not mean-variance dominate because their
returns are lower.

## Why the relative improvement is credible

The exact-production global rank is weaker than classic monthly by 41.62 bp/year, so
global deterioration contributes to the larger relative delta. It does not explain the
whole result:

- exact production's cluster-minus-global delta exceeds classic monthly's in 27/28 cells,
  by 129.79 bp/year on average;
- exact production cluster returns exceed classic cluster returns in 23/28 cells, by
  88.17 bp/year on average;
- exact production cluster returns exceed raw-weekly cluster returns in 22/28 cells, by
  59.76 bp/year on average.

At the ME/36 winner, the cluster book's mean post-net cluster L1 exposure is 2.46e-16,
versus 1.3772 for the global book against the same groups. The cluster and global books
have 139.82 and 152.18 effective names per side, respectively. The risk reduction is
therefore still explained by within-cluster long-short neutralisation, not by leverage or
a larger effective name count.

## Acceptance

| check | measured | tolerance | verdict |
|:--|--:|--:|:--|
| grid coverage | 28 cells x 2 variants x 2 windows = 112 comparisons | 112 | PASS |
| performance rows | 112 cluster + 16 matched globals = 128 | 128 | PASS |
| signal parameter rows | 14 | 14 | PASS |
| exact long horizons | 12.0 months for every scaled cadence | 12.0 | PASS |
| exact production control | ME long 12 / vol 13 / short None / mean NONE | frozen | PASS |
| production cluster fallback | 5 | 5 | PASS |
| maximum signal timestamp look-ahead | 0 days | <= 0 | PASS |
| maximum return-to-NAV round-trip error | 1.154632e-13 | <= 1e-12 | PASS |
| exact-production versus scaled-ME regression | maximum error 0.0 | <= 1e-12 | PASS |
| exact exposure rows | 128/128 | 128/128 | PASS |
| maximum long-exposure error | 6.106227e-15 | <= 1e-12 | PASS |
| maximum short-exposure error | 6.106227e-15 | <= 1e-12 | PASS |
| maximum net-exposure error | 1.660130e-15 | <= 1e-12 | PASS |
| maximum gross-exposure error | 1.865175e-14 | <= 1e-12 | PASS |
| maximum pre-net weight error | 2.220446e-16 | <= 1e-12 | PASS |
| maximum pre-net group-budget error | 1.387779e-17 | <= 1e-15 | PASS |
| maximum post-net cluster L1 exposure | 1.583586e-15 | <= 1e-12 | PASS |
| same-variant/same-cadence global matching | 112/112 | 112/112 | PASS |
| EW-all payoff comparisons | 0 | 0 | PASS |
| deterministic numerical artifacts | 10/10 byte-identical | 10/10 | PASS |
| focused production/classic/construction tests | 9/9 | 9/9 | PASS |
| ruff E/F/W on new files | no findings | no findings | PASS |

One corrected replay took 1,027.97 seconds. The first determinism attempt found that
regression.csv mistakenly contained per-run timing: the other 9/10 artifacts were already
byte-identical, while that diagnostic could not be. Timing was removed from regression.csv
and retained only in runtime.csv. No score, weight, return, comparison, or acceptance
number changed. The corrected replay passed 10/10.

Final focused pytest output:

    .........                                                                [100%]

Final ruff output:

    All checks passed!

Independent validator output:

    U1 production-momentum covariance long-short grid validation: PASS
    grid: 28 cells x 2 production variants x 2 windows = 112 comparisons
    horizons: scaled B=252, weekly=52, ME=12; exact control=ME 12/13; all long horizons=12 months
    causality: maximum lookahead=0 days; return roundtrip max error=1.155e-13
    exposures: 128/128 PASS; max error=1.865e-14
    headline global-relative return wins: exact=27/28; scaled=26/28
    winner prod_calendar_scaled_12m: ME span 36; delta=+0.027692
    winner prod_exact_monthly_12m: ME span 36; delta=+0.027692
    determinism: 10/10 artifacts byte-identical

## Interpretation and limits

1. Use exact monthly production momentum if the objective is to beat global in the most
   covariance cells. Its 27/28 breadth is substantially stronger than classic monthly's
   13/28 and raw weekly's 17/28.
2. The cadence-scaled twelve-month extension is also strong at 26/28, but it does not
   improve breadth over exact production. There is no empirical reason here to replace the
   production monthly cadence.
3. ME/36 is the common leader and is stronger than the earlier raw-grid leaders, but the
   absolute post-cost long-short return remains negative. The supported claim is relative
   performance and risk cancellation, not positive standalone alpha.
4. The 28 alternative covariance cells have not received the frozen taxonomy-fidelity,
   bootstrap, or multiple-testing treatment. ME/36 remains a research candidate rather
   than a production covariance replacement.
5. This test isolates production momentum. It does not test the full production Beta
   signal combination with low-beta.

## Deliverables

The output root is:

C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/covariance_frequency_span_grid/msci_us/long_short_grid_q_025_prod_12m/

It contains signal_parameters.csv, performance.csv, comparison_vs_global.csv,
comparison_vs_other_signals.csv, signal_breadth_summary.csv, rankings.csv,
risk_diagnostics.csv, score_diagnostics.csv, regression.csv, acceptance.csv, runtime.csv,
and determinism.csv.

The runner, validator, tests, report, and numerical outputs remain local and ignored by
git. Nothing was staged or pushed.
