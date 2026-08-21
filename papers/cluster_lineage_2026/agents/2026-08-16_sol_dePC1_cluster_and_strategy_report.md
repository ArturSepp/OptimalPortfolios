# de-PC1 cluster and strategy execution report

**Date:** 2026-08-16  
**Executor:** sol  
**Status:** COMPLETE  
**Stages:** D4-D6, following the completed D0-D3 implementation gates  
**Roadmap:** `agents/ROADMAP_factorlasso_depc1_clustering_and_backtest.md`  
**Output root:** `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/depc1`  
**Repository actions:** no staging, commit, tag, push, build publication, or release

## Outcome

The opt-in FactorLasso de-PC1 transform was run at the three frozen operating points without
retuning a signal, quantile, cost, universe, cutoff, span, or smoother parameter. All 47 D4-D6
mechanics acceptance rows passed, every deterministic numerical artifact replayed byte for byte,
and all 25 exhibit artifacts replayed byte for byte.

The empirical result does **not** support replacing the raw-correlation cluster specification:

- de-PC1 created more, smaller clusters and reduced annualized lineage churn in all three
  universes;
- it reduced annualized strategy volatility in all three universes;
- it reduced taxonomy ARI in all three universes;
- it reduced gross and net payoff relative to the frozen raw-cluster strategy in all three
  universes;
- it failed the full-window global-rank comparison in U2 and U3; in U1 it retained a net-return
  advantage over global and sector ranks, but lost to the raw cluster leg and had worse Sharpe
  than both comparators.

The recommended classification is therefore **robustness specification**, not primary. It is a
useful common-mode sensitivity diagnostic because it changes the correlation geometry exactly as
intended and improves several stability/risk measures, but its economic and taxonomy evidence is
inferior at the frozen operating points.

There is no performance acceptance gate in the roadmap. The negative performance result is a
valid, untuned experiment rather than a software failure.

## Implementation provenance

The D1-D3 implementation is documented separately in
`2026-08-16_sol_dePC1_factorlasso_implementation_report.md`. FactorLasso now exposes the
default-off `ClusterCorrelationTransform.REMOVE_PC1` path and performs

```text
Q       = R - lambda_1 v_1 v_1'
R_dePC1 = diag(Q)^(-1/2) Q diag(Q)^(-1/2)
```

after exact point-in-time eligibility restriction and before smoothing/linkage. No covariance-
basis removal, multi-PC removal, RMT bulk filter, PSD projection, or nearest-correlation repair was
introduced. `NONE` remains an exact bypass.

The local FactorLasso checkout remained first on `PYTHONPATH`. Final source hashes were:

| source | SHA-256 |
|---|---|
| `factorlasso/cluster_utils.py` | `86bf04e5965a787ab6d2b5bf6a8f914127c723575686c42cd902ccde24f350b3` |
| `factorlasso/cluster_smoothing.py` | `dc184d0231f6c509f7cd6885d8352dd73e14e1604fa4efc21b6d725fb3b081e7` |
| `factorlasso/lasso_estimator.py` | `14d8af3ccc967aeaaa4d2e9f0caa3f75bd6817a00dfd02bbb96de1ea0ca4c186` |

Paper-local runner hashes were:

| runner | SHA-256 |
|---|---|
| `run_depc1_cluster_comparison.py` | `9c58358898294900c6bfe88f1c56ee3dccdad83bd24634843364b8b955a269b1` |
| `run_depc1_strategy_backtests.py` | `731d10d1c00815450651cb621ba929f644c106133a6dbbc71c8d6b3993a25a40` |
| `build_depc1_exhibits.py` | `ab408d4f70449b695b95170296b62d53553911d39d9735f0137ac769268018cc` |

Input hashes, eligibility/date fingerprints, transform, model settings, partition hashes, and
source hashes are recorded in each universe's `source_manifest.csv` and cache payload. Existing
E2/E3/E5 caches were not written or changed.

## Frozen empirical specifications

| universe | cluster cell | frozen strategy | comparator(s) | cost |
|---|---|---|---|---:|
| U1 MSCI US | ME, EWMA span 36, Ward, `1-rho`, Pearson | exact `U1_OPTIMAL_SPEC`, q=25%, group-equal | global and BICS equal-sector ranks | 10 bp |
| U2 BlackRock funds | W-THU, EWMA span 156, Ward, `1-rho`, Pearson | AUM100, global long / group-equal cluster short, q=25%, E/FI/Rest 50/30/20 | matched global rank | 20 bp |
| U3 futures | W-WED, EWMA span 156, M1-star delta 0.0691 | ROSAA production, q=25%, E/FI/C/FX 30/30/30/10 | matched within-sleeve global rank | 10 bp |

U1 uses the exact frozen monthly production signal: long span 12, no short span, volatility span
13, and `MeanAdjType.NONE`. U3 uses the seven owner-frozen exclusions, including the documented
alias resolution `MMR1 Curncy -> BMR1 Curncy`. U2 applies the strict 12-completed-month rolling
average AUM rule `> USD 100m`.

EW-all is not a ranking yardstick. It remains only the market reference for beta/alpha, which is
not used for any comparison conclusion below.

## D4: correlation geometry and partitions

All values in the following table are medians across active snapshots. The taxonomy columns are
the frozen U1 BICS sector, U2 asset class/sub-asset class, and U3 asset class.

| universe | active dates | eligible assets | PC1 trace share | median raw corr | median residual corr | raw -> de-PC1 clusters | fixed-cut ARI | matched-count ARI | raw -> de-PC1 taxonomy ARI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| U2 funds | 238/240 | 211 | 60.23% | 0.5964 | -0.0146 | 12 -> 28 | 0.2874 | 0.3850 | asset 0.2089 -> 0.0554; sub-asset 0.1673 -> 0.1072 |
| U3 futures | 295/295 | 84 | 28.34% | 0.1007 | 0.0614 | 16 -> 19 | 0.5812 | 0.5883 | 0.5346 -> 0.3295 |
| U1 stocks | 203/203 | 619 | 35.59% | 0.3345 | -0.0062 | 60 -> 104 | 0.3825 | 0.4035 | BICS 0.1403 -> 0.0959 |

Median pairwise Rand was 0.8351 for U2, 0.9380 for U3, and 0.9787 for U1. The high U1 Rand is
consistent with the measure being dominated by the many asset pairs that remain in different
clusters; the lower ARI is the more informative partition-agreement statistic here.

### Geometry versus granularity

PC1 removal had the intended geometric effect. In U1, it removes an equity market mode; in U2
and U3, the more neutral interpretation is the dominant common mode. The strongest common mode
was in funds, where the median PC1 share was 60.23% and the median correlation moved from 0.5964
to -0.0146.

Finer cuts explain only part of the partition change. Matching the de-PC1 cut to each raw
date-specific cluster count increased raw/de-PC1 ARI by 0.0976 in U2, 0.0071 in U3, and 0.0209 in
U1. Thus granularity matters visibly for U2, but most U2 disagreement and nearly all U1/U3
disagreement remains after controlling for cluster count. Matched-count partitions are diagnostic
only and never enter strategy weights.

The taxonomy evidence runs against the motivating sector-revelation hypothesis at these operating
points: every reported taxonomy ARI declined. In particular, removing the U1 market mode did not
reveal a partition closer to BICS sectors under the fixed Ward/cutoff specification.

## D4: lineage and stability

| universe | transform | tracks | tracks/asset | annualized churn | fragmentation | births | deaths | splits | merges |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U2 | raw | 77 | 9.0079 | 1.6356 | 6.4167 | 19 | 33 | 58 | 44 |
| U2 | de-PC1 | 119 | 7.8136 | 1.5319 | 4.2500 | 13 | 42 | 106 | 77 |
| U3 | raw | 55 | 4.3864 | 0.2024 | 3.4375 | 26 | 20 | 29 | 35 |
| U3 | de-PC1 | 65 | 3.8977 | 0.1864 | 3.4211 | 29 | 18 | 36 | 47 |
| U1 | raw | 273 | 28.5794 | 5.3840 | 4.5500 | 50 | 51 | 223 | 222 |
| U1 | de-PC1 | 391 | 24.7052 | 4.3834 | 3.7596 | 83 | 74 | 308 | 317 |

Annualized churn fell by 6.3% in U2, 7.9% in U3, and 18.6% in U1. The absolute number of tracks
increased because de-PC1 created finer partitions, while distinct tracks visited per asset and
fragmentation fell. This is a genuine stability benefit, but it did not translate into better
frozen-strategy payoff.

## D5: full-window strategy evidence

Returns and volatility are annualized; Sharpe is RF=0; turnover is annualized one-way turnover.
U1 and U3 cover 2009-08-31 through 2026-06-30. U2's headline covers the same endpoints, with its
native weekly implementation series. U1 is restricted to assets with available BICS so the
cluster, global, and sector legs use an identical universe.

| universe / leg | gross return | net return | volatility | Sharpe | turnover | cost drag bp/year | total net return |
|---|---:|---:|---:|---:|---:|---:|---:|
| U2 raw hybrid | 2.1866% | 0.1925% | 6.7155% | 0.0620 | 2.4681 | 199.41 | 3.2889% |
| U2 de-PC1 hybrid | 1.0968% | -0.9333% | 6.0755% | -0.1240 | 2.5359 | 203.01 | -14.5967% |
| U2 global | 1.0245% | -0.7708% | 7.5614% | -0.0646 | 2.2432 | 179.53 | -12.2082% |
| U3 raw cluster | 1.2501% | 0.0297% | 4.4196% | 0.0179 | 3.0330 | 122.05 | 0.5001% |
| U3 de-PC1 cluster | 1.0568% | -0.1184% | 4.0630% | 0.0009 | 2.9252 | 117.53 | -1.9720% |
| U3 global | 1.3683% | -0.0163% | 8.2125% | 0.0296 | 3.4414 | 138.46 | -0.2733% |
| U1 raw cluster | 0.3050% | -1.6153% | 6.7886% | -0.2057 | 4.8456 | 192.04 | -27.8080% |
| U1 de-PC1 cluster | -1.1087% | -2.8575% | 5.5040% | -0.4986 | 4.4619 | 174.88 | -44.0137% |
| U1 sector | -2.2893% | -3.5331% | 9.4860% | -0.3308 | 3.2023 | 124.38 | -51.3102% |
| U1 global | -3.1318% | -4.4019% | 12.7914% | -0.2860 | 3.2955 | 127.01 | -59.3722% |

### Frozen contrasts

| contrast | net-return delta | volatility delta | Sharpe delta | turnover delta | cost-drag delta |
|---|---:|---:|---:|---:|---:|
| U2 de-PC1 minus raw | -1.1258 pp | -0.6400 pp | -0.1859 | +0.0678 | +3.60 bp/year |
| U2 de-PC1 minus global | -0.1625 pp | -1.4859 pp | -0.0594 | +0.2928 | +23.48 bp/year |
| U2 raw minus global | +0.9633 pp | -0.8459 pp | +0.1266 | +0.2249 | +19.88 bp/year |
| U3 de-PC1 minus raw | -0.1481 pp | -0.3566 pp | -0.0170 | -0.1078 | -4.52 bp/year |
| U3 de-PC1 minus global | -0.1021 pp | -4.1496 pp | -0.0287 | -0.5162 | -20.93 bp/year |
| U3 raw minus global | +0.0460 pp | -3.7929 pp | -0.0117 | -0.4084 | -16.41 bp/year |
| U1 de-PC1 minus raw | -1.2422 pp | -1.2846 pp | -0.2929 | -0.3837 | -17.15 bp/year |
| U1 de-PC1 minus global | +1.5444 pp | -7.2874 pp | -0.2127 | +1.1664 | +47.87 bp/year |
| U1 raw minus global | +2.7866 pp | -6.0027 pp | +0.0803 | +1.5501 | +65.03 bp/year |
| U1 de-PC1 minus sector | +0.6756 pp | -3.9820 pp | -0.1678 | +1.2596 | +50.50 bp/year |
| U1 raw minus sector | +1.9178 pp | -2.6973 pp | +0.1251 | +1.6434 | +67.65 bp/year |

For U2, the primary portfolio holds the global long leg fixed, so the raw/de-PC1 difference is
isolated to cluster-defined short selection and its trading. The pure-cluster diagnostic moved
from -2.1437% to -1.1644% net annual return under de-PC1, but both remained negative and this
diagnostic is not the frozen primary portfolio.

## Gross payoff versus turnover and cost

The performance deterioration is not a cost artifact:

- U1 de-PC1 saved 17.15 bp/year of cost but lost 141.37 bp/year of gross return, leaving a
  124.22 bp/year net loss versus raw.
- U3 de-PC1 saved 4.52 bp/year of cost but lost 19.33 bp/year of gross return, leaving a
  14.81 bp/year net loss versus raw.
- U2 de-PC1 increased cost by only 3.60 bp/year while losing 108.98 bp/year of gross return,
  leaving a 112.58 bp/year net loss versus raw.

Thus the common conclusion is lower gross payoff from the changed groups. De-PC1 lowered
turnover in U1 and U3, but **raised** it slightly in U2. It lowered volatility in every universe;
return fell by more than risk, so Sharpe also deteriorated everywhere relative to raw.

## U2 chronological split robustness

U2 has the predeclared 2009-2017 selection slice and 2018-2026 evaluation slice. This is a
chronological robustness diagnostic; the AUM100 strategy provenance remains post-sensitivity and
was not reselected for this experiment.

| slice / leg | net annual return | volatility | Sharpe |
|---|---:|---:|---:|
| 2009-2017 raw | -0.0420% | 6.5055% | 0.0257 |
| 2009-2017 de-PC1 | -1.4047% | 5.8038% | -0.2149 |
| 2009-2017 global | 0.1709% | 7.2999% | 0.0595 |
| 2018-2026 raw | 0.7436% | 6.9442% | 0.1410 |
| 2018-2026 de-PC1 | 0.2956% | 6.2663% | 0.0780 |
| 2018-2026 global | -0.2078% | 7.7180% | 0.0114 |

De-PC1 beats global in the evaluation slice by 50.34 bp/year net return and 0.0665 Sharpe, but
still trails raw by 44.81 bp/year and 0.0630 Sharpe. This supports retaining de-PC1 as a robustness
case, not replacing the raw primary.

## Acceptance: U2 BlackRock funds

| stage | check | measured | tolerance | status |
|---|---|---:|---:|---|
| D4 | raw and de-PC1 schedule identity | 1 | 1 | PASS |
| D4 | raw exact eligibility membership | 1 | 1 | PASS |
| D4 | de-PC1 exact eligibility membership | 1 | 1 | PASS |
| D4 | raw frozen partition match share | 1 | 1 | PASS |
| D4 | de-PC1 injected/fitted partition match share | 1 | 1 | PASS |
| D4 | active snapshot count | 238 | 238 | PASS |
| D5 | maximum signal lookahead days | 0 | 0 | PASS |
| D5 | maximum weight and sleeve exposure error | `7.994e-15` | `<=1e-12` | PASS |
| D5 | weight outside point-in-time eligibility | 0 | `<=1e-12` | PASS |
| D5 | AUM <= USD100m eligible observations | 0 | 0 | PASS |
| D5 | global score arm difference | 0 | `<=1e-12` | PASS |
| D5 | global weight arm difference | 0 | `<=1e-12` | PASS |
| D5 | global portfolio arm difference | 0 | `<=1e-12` | PASS |
| D5 | instrument P&L reconciliation error | `1.030e-13` | `<=1e-10` | PASS |
| D5 | one-way transaction cost | 20 bp | 20 bp | PASS |
| D6 | required traced exhibits emitted | 5 | 5 | PASS |

## Acceptance: U3 futures

| stage | check | measured | tolerance | status |
|---|---|---:|---:|---|
| D4 | raw and de-PC1 schedule identity | 1 | 1 | PASS |
| D4 | raw exact eligibility membership | 1 | 1 | PASS |
| D4 | de-PC1 exact eligibility membership | 1 | 1 | PASS |
| D4 | raw frozen partition match share | 1 | 1 | PASS |
| D4 | de-PC1 injected/fitted partition match share | 1 | 1 | PASS |
| D4 | active snapshot count | 295 | 295 | PASS |
| D5 | maximum signal lookahead days | 0 | 0 | PASS |
| D5 | maximum weight and sleeve exposure error | `2.665e-15` | `<=1e-12` | PASS |
| D5 | weight outside point-in-time eligibility | 0 | `<=1e-12` | PASS |
| D5 | raw partition match to owner-frozen M1-star | 1 | 1 | PASS |
| D5 | owner-excluded eligible observations | 0 | 0 | PASS |
| D5 | maximum owner-excluded weight | 0 | `<=1e-12` | PASS |
| D5 | owner exclusion set size | 7 | 7 | PASS |
| D5 | global comparator arm difference | 0 | `<=1e-12` | PASS |
| D5 | instrument P&L reconciliation error | `9.504e-14` | `<=1e-10` | PASS |
| D5 | one-way transaction cost | 10 bp | 10 bp | PASS |
| D6 | required traced exhibits emitted | 5 | 5 | PASS |

## Acceptance: U1 MSCI US

| stage | check | measured | tolerance | status |
|---|---|---:|---:|---|
| D4 | raw and de-PC1 schedule identity | 1 | 1 | PASS |
| D4 | raw exact eligibility membership | 1 | 1 | PASS |
| D4 | de-PC1 exact eligibility membership | 1 | 1 | PASS |
| D4 | raw frozen partition match share | 1 | 1 | PASS |
| D4 | de-PC1 injected/fitted partition match share | 1 | 1 | PASS |
| D4 | active snapshot count | 203 | 203 | PASS |
| D5 | maximum signal lookahead days | 0 | 0 | PASS |
| D5 | maximum weight and group exposure error | `2.331e-14` | `<=1e-12` | PASS |
| D5 | weight outside matched BICS eligibility | 0 | `<=1e-12` | PASS |
| D5 | eligible cluster memberships missing | 0 | 0 | PASS |
| D5 | global comparator arm difference | 0 | `<=1e-12` | PASS |
| D5 | instrument P&L reconciliation error | `1.208e-13` | `<=1e-10` | PASS |
| D5 | one-way transaction cost | 10 bp | 10 bp | PASS |
| D6 | required traced exhibits emitted | 5 | 5 | PASS |

## Runtime, cache, and determinism

| universe | schedule / active | initial D4 miss runtime | latest D4 hit runtime | latest D5 runtime | raw cache bytes | de-PC1 cache bytes | combined D4-D5 replay |
|---|---:|---:|---:|---:|---:|---:|---:|
| U2 | 240 / 238 | 116.52 s | 14.21 s | 30.03 s | 2,800,455 | 2,800,935 | 13/13 byte-identical |
| U3 | 295 / 295 | 30.77 s | 15.82 s | 20.90 s | 2,019,271 | 1,839,199 | 13/13 byte-identical |
| U1 | 203 / 203 | 462.92 s | 110.66 s | 73.07 s | 8,623,213 | 8,623,619 | 13/13 byte-identical |

The standalone D4 replay was also byte-identical for all 13 artifacts per universe. D6 regenerated
all exhibits twice and obtained 25/25 byte-identical files. One cumulative-NAV exhibit per
universe was opened and read back programmatically; a legend-overlap issue found in the first
render was corrected before the final deterministic replay.

Each universe emits the required diagnostics, partition, lineage, performance, turnover/cost,
instrument-P&L, acceptance, runtime, determinism, and manifest CSVs. Each `exhibits/` directory
contains the PC1-share time series, cluster-count/taxonomy-ARI time series, final partition heat
map, cumulative net NAV, and raw-versus-de-PC1 instrument P&L scatter plus its source table.

## Verification

Final commands and measured results:

```text
FactorLasso: pytest --cov=factorlasso --cov-report=term -q
FactorLasso: ruff check factorlasso/ tests/
Paper harness: pytest depc1_cluster_comparison_test.py
                     depc1_strategy_backtests_test.py
                     depc1_exhibits_test.py -q
Paper harness: ruff check --isolated --select E,F,W --line-length 100 <six D4-D6 files>
```

| check | measured | tolerance | status |
|---|---:|---:|---|
| FactorLasso full suite | all passed; 9 expected skips | all pass | PASS |
| FactorLasso coverage | 92.67% | >=90% | PASS |
| FactorLasso Ruff findings | 0 | 0 | PASS |
| base-import banned modules | 0 | 0 | PASS |
| FactorLasso local-source assertion | 1 | 1 | PASS |
| paper D4-D6 focused tests | 13 passed | all pass | PASS |
| paper isolated Ruff findings | 0 | 0 | PASS |
| per-universe D4-D5 deterministic artifacts | 13/13 each | all identical | PASS |
| D6 deterministic artifacts | 25/25 | all identical | PASS |
| D4-D6 mechanics rows | 47/47 | all pass | PASS |

## Deviations and limitations

1. **U1 full warmup panel not emitted.** The primary 203-date headline window is complete. On
   sparse pre-headline warmup dates, the accepted pairwise raw correlation can be materially
   indefinite; after exact PC1 deflation and restandardization, a residual off-diagonal value
   reached approximately 1.0137. The FactorLasso material-bound guard correctly raised. The
   roadmap forbids silent clipping, PSD projection, or nearest-correlation repair, so no full
   de-PC1 warmup series was substituted. The roadmap makes the separate full schedule optional
   (“if emitted”); all U1 conclusions here are explicitly headline-window conclusions.
2. **U3 frozen-versus-exact-universe history.** The accepted U3 M1-star partitions were fitted
   before the later seven low-liquidity exclusions and then restricted for eligibility and
   weights. An exact-universe raw refit matched the frozen partition on 57/295 dates (19.32%). To
   satisfy the binding frozen-strategy comparison, the raw arm preserves the owner-frozen
   partition and has zero membership/weight for excluded assets. The de-PC1 arm estimates PC1 and
   clusters only the exact current eligible set. This diagnostic is reported rather than silently
   replacing the raw strategy.
3. **U1 BICS matching.** The frozen D0 raw result was -1.6204% net annual return. The D5 raw row is
   -1.6153% because all four U1 legs are evaluated on the identical BICS-covered universe; a few
   delisted stocks lack BICS. This is a classification-coverage restriction applied equally to
   all arms, not an effect of de-PC1.
4. **No retuning and no inference layer.** These are fixed-cell comparisons. They do not search
   for a de-PC1-specific cutoff or smoother delta and make no new statistical-significance claim.

## Interpretation and owner gate

The experiment isolates a meaningful trade-off. Common-mode removal gives finer residual
communities, lower lineage churn, and lower long-short volatility, but the removed component also
contains economically useful grouping information for the frozen momentum constructions. The
matched-count diagnostic shows that this is not merely a cluster-count effect. The gross-payoff
loss dominates the small turnover/cost benefit, and taxonomy ARI does not improve.

**Executor recommendation: ROBUSTNESS.** Retain raw correlation as the primary specification;
retain de-PC1 as a labelled common-mode sensitivity exhibit. Do not use the de-PC1 partitions in
the production/article primary backtests without a new owner-authorized hypothesis and roadmap.

**OWNER GATE REQUEST:** classify the de-PC1 clustering specification as exactly one of
`PRIMARY`, `ROBUSTNESS`, or `REJECTED`.
