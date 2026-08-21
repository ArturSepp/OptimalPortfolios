# U2 BlackRock funds long-short improvement search

> **SUPERSEDED WINDOW DIAGNOSTIC (2026-08-16):** the search harness used
> `run_u2_blackrock_etf_grid._window_prices`, which always extended performance prices to
> 2026-07-31.  Consequently the labelled 2009-2017 selection backtests continued after
> their declared end while holding the last selected units, and the labelled headline
> window included July 2026.  The specification-grid arithmetic remains a legacy
> diagnostic, but its split-sample selection, stability conclusions, and headline payoff
> conclusions must not be cited.  Corrected closed-window results are in
> `2026-08-16_sol_U2_blackrock_AUM_eligibility_report.md`.

**Date:** 2026-08-16  
**Executor:** sol  
**Status:** COMPLETE; a full-sample cluster-assisted win exists, but no stable pure-cluster
premium was found  
**Runner:**
`papers/cluster_lineage_2026/replication/run_u2_blackrock_long_short_search.py`  
**Focused tests:**
`papers/cluster_lineage_2026/replication/u2_blackrock_long_short_search_test.py`  
**Repository scope:** ignored `papers/cluster_lineage_2026/` tree only; no staging or push

## Decision

The original group-equal cluster long-short portfolio underperforms for two distinct
reasons. Over the full 2009-08-31 through 2026-06-30 window, it gives up **69.29 bp/year
gross** to the matched global rank and incurs **56.10 bp/year more cost drag**, producing
the accepted **-125.40 bp/year net gap**.

The search improves the portfolio materially but does not support an unconditional
cluster-outperformance claim for funds:

- **Asset-equal cluster budgeting** is superior to group-equal budgeting. For the owner
  base signal it reduces the full-window deficit from -125.40 bp to -56.17 bp and changes
  the 2018-2026 edge from +2.81 bp to **+118.21 bp**. It still loses before 2018 and over
  the full window, and it does not beat global on Sharpe.
- A **global-long / cluster-short hybrid** under the owner base specification earns
  -1.3228% net versus -1.7460% for global over the continuous full window: a
  **+42.32 bp/year net edge**, with Sharpe -0.1825 versus -0.2117. It also wins by
  +113.10 bp and +0.112 Sharpe in 2018-2026. It loses by -59.14 bp before 2018, so this
  is a regime-dependent exploratory overlay, not a stable result.
- The three-month short/reversal signal is the strongest absolute signal in every
  window. Quarterly global earns **0.8262% net** with 0.1509 Sharpe over the full sample.
  The best cluster-assisted quarterly variant earns 0.7456% with 0.1604 Sharpe: slightly
  better risk-adjusted performance but **8.07 bp lower return**. In 2018-2026 it earns
  0.4714% versus 0.2950% for global, a +17.64 bp and +0.032 Sharpe win, but it loses by
  62.48 bp before 2018.

Therefore the implementable research choices are different depending on the objective:

1. To show a measured full-window net-return win against global, use the owner-base
   **global-long / cluster-short hybrid**, labelled exploratory and accompanied by its
   failed split-sample stability result.
2. To maximize positive absolute fund performance, use the **global** three-month-reversal
   signal with quarterly holding. Clustering is a recent-period Sharpe enhancement, not
   the full-window return winner.
3. A paper claim that clustering robustly beats global for funds is **not supported** by
   the current-vintage BlackRock panel.

## Frozen comparison and search design

Every primary comparison is fair: cluster and global use the same signal, quantile,
Equity/Fixed-Income/Rest budgets, dates, +1/-1 gross exposure, one-period implementation
lag, and 20 bp one-way costs. Global ranks within the three broad sleeves. Cluster ranks
within the intersection of the broad sleeve and statistical cluster.

The base is the requested 50/30/20 allocation, q=25%, W-THU covariance returns with EWMA
span 156, monthly 12-month ROSAA signal with no short span, volatility span 13 and EWMA
mean adjustment. Partitions are read from the frozen 28-cell cache; nothing is refit.

The predeclared marginal grids were:

| dimension | values |
|:--|:--|
| signal | 24 ROSAA cells: short span None/1/2/3, vol span 13/26/52, mean adjustment NONE/EWMA; plus classic 12m-ex-1m |
| covariance | B and W-MON through W-FRI at spans 24/36/52/156; ME at spans 12/24/36/52 |
| quantile | 10%, 15%, 20%, 25%, 30% |
| sleeve allocation | the eight accepted 40-70% Equity / 20-40% Fixed Income / 10-30% Rest cells |
| cluster budget | group-equal, square-root selected-group-size, asset-equal |

This union produced 65 marginal candidates. The top two values per dimension on
2009-08-31 through 2017-12-31 were crossed, retaining all three constructions, producing
48 interactions and **113 unique candidates**. Evaluation is 2018-01-31 through
2026-06-30; the full window is descriptive.

The production fallback grid 5/7/10 was not repeated. Earlier verification showed that
fallback changes normalized scores but not percentile selections or portfolio weights
under within-cluster ranking.

## Grid verdict

| window | pure-cluster candidates | net-return wins vs matched global | Sharpe wins | wins on both |
|:--|--:|--:|--:|--:|
| pre-2018 selection | 113 | **0** | 0 | 0 |
| 2018-2026 evaluation | 113 | **21** | 0 | 0 |
| full 2009-2026 | 113 | **0** | 0 | 0 |

No pure-cluster candidate has a positive net return in either the evaluation or full
window at 20 bp. Seven matched global specifications are positive over the full window,
while neither family is positive in 2018-2026 under monthly implementation.

The honest pre-2018-selected pure-cluster candidate is classic 12m-ex-1m, W-WED/span
156, q=30%, 40/30/30 and asset-equal cluster budgeting. Its pre-2018 net edge is
-0.02 bp/year, but it loses by 34.85 bp in evaluation and 121.74 bp over the full window.

The best descriptive full-window pure-cluster row is monthly ROSAA with no short span,
no mean adjustment, W-WED/span 156, q=25%, 40/30/30 and asset-equal budgeting. It still
loses by 37.05 bp/year net and 0.457 Sharpe.

Train/evaluation rank stability is weak: Pearson correlation of the 113 net edges is
0.376 and Spearman correlation is 0.321. There is no candidate with a positive net edge
in both halves.

## What drives the original underperformance

### Gross signal versus costs

| window | gross cluster-global | extra cluster cost | net cluster-global |
|:--|--:|--:|--:|
| pre-2018 | -186.91 bp | 24.91 bp | **-211.81 bp** |
| 2018-2026 | +65.32 bp | 62.51 bp | **+2.81 bp** |
| full | -69.29 bp | 56.10 bp | **-125.40 bp** |

Full-window annual turnover is 3.927 for cluster versus 3.191 for global. At 20 bp this
produces 310.03 bp versus 253.92 bp annual cost drag. Cost reduction cannot repair the
base case by itself because the gross full-window edge is already negative.

### Equal-cluster budgets are too aggressive

The group-equal base assigns, on average, **46.84% of gross capital to hierarchical
groups containing five or fewer eligible funds**. It has only 39.2 effective long
positions despite selecting 79.3 long names on average. Asset-equal budgeting raises
effective long breadth to 63.2, lowers small-group gross capital to 27.74%, lowers
turnover, and improves both gross and net results.

| base construction | pre-2018 edge | evaluation edge | full edge |
|:--|--:|--:|--:|
| group-equal | -211.81 bp | +2.81 bp | -125.40 bp |
| square-root group size | -185.36 bp | +65.17 bp | -88.98 bp |
| asset-equal | **-147.70 bp** | **+118.21 bp** | **-56.17 bp** |

This is the clearest mechanical improvement found. Equal capital per statistical cluster
overweights small, noisy peer groups in this 480-fund universe.

### Clustering helps shorts but hurts longs

The standalone component backtests show the source of the hybrid result. Over the full
window, the cluster long side earns 5.10% gross versus 6.72% for global, approximately a
161 bp sacrifice. The cluster short side loses 5.68% versus 6.39% for global,
approximately a 71 bp improvement. In 2018-2026 the approximate long-side sacrifice is
104 bp and the short-side improvement is 154 bp.

Thus clustering acts as a useful short-side risk filter in the recent regime but dilutes
the strongest long-side winners. The owner-base global-long/cluster-short hybrid captures
that asymmetry:

| window | hybrid net | global net | net edge | hybrid/global Sharpe |
|:--|--:|--:|--:|--:|
| pre-2018 | +0.3714% | +0.9629% | **-59.14 bp** | 0.0868 / 0.1805 |
| 2018-2026 | -2.0133% | -3.1443% | **+113.10 bp** | -0.2908 / -0.4029 |
| full | -1.3228% | -1.7460% | **+42.32 bp** | -0.1825 / -0.2117 |

The first naïve hybrid replay was rejected because global-long and cluster-short names
could overlap, reducing gross exposure below two. The accepted replay removes overlap
and renormalizes every broad sleeve separately. All 226 hybrid exposure rows then pass
with maximum error 1.021e-14 against 1e-12.

The hybrid chosen exclusively on pre-2018 data instead uses classic momentum and the
cluster-long/global-short orientation. It wins by 60.35 bp before 2018, then loses by
54.00 bp in evaluation and 34.73 bp over the full window. No evaluated hybrid has a
positive edge in both halves.

### Holding period and stronger absolute signal

Slower implementation reduces costs but does not create a stable cluster premium for
the owner base. The continuous full-window global-long/cluster-short hybrid improves from
-1.3228% monthly to -0.4123% every two months, versus -0.9263% for its global control.
However, independently initialized pre-2018 and evaluation books both lose to their
controls. This full-window edge is therefore not treated as split-sample confirmation.

The three-month short/reversal signal is a much stronger absolute long-short signal:

| short-3 quarterly result | net return | Sharpe | delta vs global |
|:--|--:|--:|--:|
| full global | **+0.8262%** | 0.1509 | reference |
| full global-long / sqrt-cluster-short | +0.7456% | **0.1604** | -8.07 bp; +0.009 Sharpe |
| evaluation global | +0.2950% | 0.0766 | reference |
| evaluation global-long / sqrt-cluster-short | **+0.4714%** | **0.1083** | +17.64 bp; +0.032 Sharpe |

The same hybrid loses 62.48 bp to global before 2018. This supports a candidate recent-
regime robustness exhibit, not a full-history cluster-return claim.

## Risk, universe, and interpretation

The base cluster portfolio remains a strong risk compressor: full-window volatility is
3.927% versus 7.120% for global. Its EW-market beta is -0.094 versus -0.162. The lower
volatility does not rescue Sharpe because net alpha is more negative: -2.278% versus
-0.368% annualized against the EW market reference.

The current-vintage BlackRock cohort has rapidly changing historical coverage: eligible
funds rise from 162 to 474, with median 283 and mean 287 over the headline dates. Funds
that closed before the 2026 screener vintage are absent. The sharp pre/post-2018 change
may therefore combine a genuine regime change with current-cohort survivorship and
coverage effects. It is not safe to tune away using this panel.

EW-all is used only for beta and alpha diagnostics. No ranking-leg performance conclusion
is stated against EW-all.

## Acceptance and verification

| acceptance line | measured | tolerance | result |
|:--|--:|--:|:--|
| unique candidate specifications | 113 | exact | PASS |
| candidate/window QIS performance rows | 678 | exact | PASS |
| signal timing/reconstruction rows | 25/25 | 100% | PASS |
| maximum candidate weight/exposure error | 1.288e-14 | <= 1e-12 | PASS |
| owner-base regression to accepted three-universe run | 1.705e-13 | <= 1e-12 | PASS |
| corrected hybrid exposure rows | 226/226 | 100% | PASS |
| maximum corrected hybrid exposure error | 1.021e-14 | <= 1e-12 | PASS |
| monthly holding-period base regression | 1.705e-13 | <= 1e-12 | PASS |
| short-3 monthly grid regression | 4.547e-13 | <= 1e-12 | PASS |
| focused pytest | 5/5 passed | all pass | PASS |
| isolated E/F/W lint | no findings | no findings | PASS |

The initial 113-candidate grid took 918.49 seconds. The corrected 226-row hybrid training
screen took 883.64 seconds. The monthly base portfolio is independently replayed by the
holding-period follow-up and matches the accepted prior run within 1e-12.

Verbatim focused pytest output:

```text
.....                                                                    [100%]
```

Verbatim focused lint output:

```text
All checks passed!
```

## Deliverables

External output directory:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\covariance_frequency_span_grid\blackrock_us_etfs\long_short_spec_search_20260815
```

Principal artifacts:

- `comparison.csv`, `performance.csv`: all 113 candidates in three windows;
- `marginal_grid_tags.csv`, `marginal_finalist_selection.csv`, `selection.csv`;
- `driver_decomposition.csv`, `weight_diagnostics.csv`, `component_attribution.csv`;
- `cost_sensitivity.csv`;
- `hybrid_comparison.csv`, `hybrid_selection.csv`, `hybrid_cost_sensitivity.csv`,
  `hybrid_acceptance.csv`;
- `holding_period_performance.csv`, `holding_period_comparison.csv`;
- `short3_performance.csv`, `short3_comparison.csv`;
- `acceptance.csv`, `runtime.csv`.

No covariance cache or input dataset was modified. No file was staged or pushed.

## Recommended treatment

For the article, retain group-equal as the frozen primary construction but report
asset-equal as the economically important funds robustness. Present the owner-base
global-long/cluster-short full-window win only as a diagnosed asymmetric overlay, with the
failed pre/post stability table beside it. If a positive absolute funds illustration is
needed, use the short-3 quarterly global result and show the recent cluster-short Sharpe
enhancement separately.

Do not claim stable fund cluster outperformance until a point-in-time historical BlackRock
universe is available or the side-specific rule is frozen and tested on a genuinely new
period.
