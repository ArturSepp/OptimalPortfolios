# Stability-pooled z-score S5b execution report

**Date:** 2026-08-20  
**Executor:** Sol  
**Proposer:** Ben (Monday TAA meeting)  
**Status:** COMPLETE — V1/V2 FAMILY REJECT confirmed

## Conclusion

The variance-only family is **REJECTED** under the predeclared rule. The three decisive
numbers are the V1 evaluation-half Sharpe delta of **-0.052292**, the V2 evaluation-half
Sharpe delta of **-0.086375**, and the least-favourable turnover-CI upper bound of
**-0.024010**. Turnover reduction is robust, but each negative evaluation-half Sharpe delta
independently rejects its cell; no adoption or owner-selection gate remains open.

Both cells otherwise pass the operational case. V1 reduces full-sample annual turnover by
0.089777 with a 95% CI of [-0.150473, -0.024010]; V2 reduces it by 0.103962 with a 95% CI
of [-0.167407, -0.042561]. Both hold full-sample ex-post TE and ex-ante TRE within 0.001 of
V0, and the reassignment shares of direct turnover reductions are 53.30% and 52.49%.
The evidence therefore confirms a real turnover effect, but not held Sharpe across regimes.

## Owner rulings implemented

- V3 was removed from the unreleased public enum and every executable scorer/harness surface.
  The only remaining source-text occurrences are negative test/harness assertions proving
  absence; there is no V3 enum member, scoring branch, or runnable cell.
- Stability is EWMA co-association with the explicit pinned map `{'ME': 36, 'QE': 18}`,
  `alpha = 2 / (span + 1)`, observed partition dates through `t`, and `adjust=True`.
- The MAC operating partition panel is monthly, so ME span 36 binds. QE span 18 is retained as
  an owner-pinned configuration entry and deliberately differs from the production lasso QE
  span 12; it does not bind in this run.
- `ClusterStabilityStatistics` is constructed once per cell beside the shared covariance object.
  Momentum-cluster and low-beta-cluster consume the same object and the same `w_i` panel.
- The 1x/2x flat-window sweep is retired. Only V1/EWMA-36 and V2/EWMA-36 were run.
- No ROSAA source file, production configuration, frozen U1/U2/U3 specification, release
  metadata, tag, or version was changed.

## FactorLasso implementation

The new `factorlasso.cluster_statistics` module provides:

- `compute_cluster_stability_statistics(partitions, span_by_freq, min_history=12)`;
- immutable run metadata plus the per-asset `w_i`, per-cluster `w_g`, and coverage panels;
- boundary/reassignment statistics, size-versus-w correlation, and within-cluster asset-w
  dispersion as methods on `ClusterStabilityStatistics`;
- reuse of the public `compute_co_association_panel` accessor for both flat and optional EWMA
  co-association; the flat default remains unchanged.

The EWMA path is vectorised across historical peer assignments. A second, independently written
direct pandas implementation matches it on changing peer sets and missing assets to `1e-15`.
The output is clipped only to the mathematical `[0, 1]` bounds to remove floating roundoff such
as `1.0000000000000002`; persisted MAC weights range from 0.054055 to exactly 1.0.

## Fail-before-pass and verification

The new contract tests were run before implementation and failed as intended: six failures
(five missing `compute_cluster_stability_statistics` imports and one still-present V3 member).
After implementation and the numerical second pass:

| check | measured | tolerance | status |
|---|---:|---:|---|
| hand EWMA reference | boundary `w = 5/7` exactly | explicit adjust-True arithmetic | PASS |
| independent vectorised reference | maximum tolerance `1e-15` | `<= 1e-15` | PASS |
| ME/QE span resolution | ME 36; QE 18 | pinned map | PASS |
| EWMA causality | panel through cutoff byte-identical | exact | PASS |
| short-history fallback | first 11 dates at `w=1` | fewer than 12 dates | PASS |
| V0 scorer identity | byte-identical | exact | PASS |
| V1/V2 `w=1` endpoints | byte-identical to V0 | exact | PASS |
| V1/V2 `w=0` endpoints | explicit local-mean/global-variance result | exact to `1e-14` | PASS |
| small-cluster fallback precedence | global fallback | exact | PASS |
| FactorLasso full suite | 510 passed; 9 intentional skips | 0 failures/errors | PASS |
| OptimalPortfolios full suite | 1,399 passed | 0 failures/errors | PASS |
| FactorLasso ruff | all checks passed | 0 findings | PASS |
| changed FactorLasso modules docstrings | 100.0% | 100.0% | PASS |

The final full-suite passes used `MPLBACKEND=Agg` and the OneDrive FactorLasso checkout on
`PYTHONPATH`. An initial FactorLasso pass without `Agg` had two Tk backend errors; the same suite
passed completely under the repository's non-interactive test convention. That was an
environment correction, not a numerical or API failure.

## Compute-once wiring

| cell | statistics constructions | signal legs | shared object | active frequency/span |
|---|---:|---:|---|---|
| V1_ewma36 | 1 | 2 | true | ME / 36 |
| V2_ewma36 | 1 | 2 | true | ME / 36 |

The exact assertions are recorded in `s5b_wiring.csv` and in the run log. The two separately
constructed cell objects also produce byte-identical `w_i` panels, as expected from their common
operating partitions and pinned span. Signals never construct stability statistics.

For the Metric-11 prior-partition counterfactual, the operating partition and the already
precomputed stability state are both shifted by one partition date. This preserves the pinned
"prior partition and prior stability state" convention without constructing a second statistics
object.

## Full-sample results

Production costs remain 0.0, so Sharpe deltas contain no return benefit from reduced turnover.
All deltas are versus the same-process V0.

| cell | TAA Sharpe | delta | annual turnover | delta | ex-post TE | delta | ex-ante TRE | delta | max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 1.068978 | 0.000000 | 0.992292 | 0.000000 | 0.031291 | 0.000000 | 0.028320 | 0.000000 | -0.249967 |
| V1_ewma36 | 1.082074 | +0.013096 | 0.902515 | -0.089777 | 0.031162 | -0.000128 | 0.028158 | -0.000163 | -0.230313 |
| V2_ewma36 | 1.076197 | +0.007219 | 0.888330 | -0.103962 | 0.030659 | -0.000631 | 0.028058 | -0.000262 | -0.230777 |

V0 reproduces the accepted S3 baseline exactly across all five persisted metrics. Both EWMA cells
pass matched risk: absolute TE/TRE deltas are 0.000128/0.000163 for V1 and
0.000631/0.000262 for V2, all below 0.001.

## Transition from flat-36 weights

| cell | Sharpe transition | turnover transition | TE transition | TRE transition | abs Sharpe >0.02? |
|---|---:|---:|---:|---:|---|
| V1 EWMA minus V1 flat | -0.016556 | -0.001394 | -0.000543 | -0.000127 | no |
| V2 EWMA minus V2 flat | -0.011605 | +0.003643 | -0.000607 | -0.000131 | no |

Neither cell triggers the predeclared prominent-transition threshold. EWMA slightly lowers Sharpe
relative to flat 36 in both variants; it marginally lowers V1 turnover and marginally raises V2
turnover. Risk is slightly lower in every transition measure.

## Split-window robustness

| window | dates | cell | Sharpe | Sharpe delta | annual turnover | turnover delta |
|---|---|---|---:|---:|---:|---:|
| selection | 2004-12-31 to 2015-09-30 | V0 | 0.863005 | 0.000000 | 0.961114 | 0.000000 |
| selection | 2004-12-31 to 2015-09-30 | V1_ewma36 | 0.892154 | +0.029149 | 0.902672 | -0.058442 |
| selection | 2004-12-31 to 2015-09-30 | V2_ewma36 | 0.890493 | +0.027487 | 0.888203 | -0.072911 |
| evaluation | 2015-10-31 to 2026-07-31 | V0 | 1.320608 | 0.000000 | 0.963485 | 0.000000 |
| evaluation | 2015-10-31 to 2026-07-31 | V1_ewma36 | 1.268317 | **-0.052292** | 0.868428 | -0.095057 |
| evaluation | 2015-10-31 to 2026-07-31 | V2_ewma36 | 1.234234 | **-0.086375** | 0.875510 | -0.087975 |

Both cells improve Sharpe in the selection half and reverse in the evaluation half. The decision
rule states that any negative half-window Sharpe delta rejects the cell, so both verdicts are
REJECT without relying on bootstrap Sharpe significance.

Solver diagnostics reproduce the accepted S5 truncation behavior. There are 338 rejected solver
attempts in total: the three selection runs have 130, 106, and 99 rejections, all dated
2015-10-31 or later and therefore outside the selection window; the three evaluation runs each
have one rejection at 2005-01-31, before the evaluation window. Full-sample and prior-partition
runs have zero rejected attempts. Active scored-window rejected attempts are **0**, tolerance 0.

## Paired moving-block bootstrap

The frozen method is the paired circular moving-block bootstrap with block length 6, 2,000 draws,
and seed 20260813. Intervals are percentile draws centered on the production-convention point
delta.

| contrast | metric | estimate | 95% CI | excludes zero? |
|---|---|---:|---:|---|
| V1 EWMA minus V0 | TAA Sharpe | +0.013096 | [-0.063061, +0.075705] | no |
| V1 EWMA minus V0 | annual turnover | -0.089777 | [-0.150473, -0.024010] | yes |
| V2 EWMA minus V0 | TAA Sharpe | +0.007219 | [-0.074212, +0.076069] | no |
| V2 EWMA minus V0 | annual turnover | -0.103962 | [-0.167407, -0.042561] | yes |

The full-sample Sharpe intervals crossing zero are allowed by S5b but provide no rescue from the
negative evaluation halves. The turnover intervals satisfy the adoption criterion for both cells.

## Mechanism and stability diagnostics

V1 has the higher full-sample Sharpe and is the predeclared "better cell." The Metric-11
decomposition was also computed for V2 so that the cell-level decision rule could be evaluated
symmetrically; this adds no refit and does not select V2.

| cell | reassignment delta/year | signal delta/year | total delta/year | trade-interaction delta/year | reassignment share |
|---|---:|---:|---:|---:|---:|
| V1_ewma36 | -0.145947 | -0.127879 | -0.048003 | +0.225823 | 53.30% |
| V2_ewma36 | -0.230044 | -0.208250 | -0.050800 | +0.387493 | 52.49% |

The reassignment share exceeds 50% for both cells, but remains a directional triangle-bound
decomposition: the signed trade-interaction term offsets part of the two direct reductions.

EWMA stability strengthens the accepted boundary diagnostic:

| diagnostic | measured |
|---|---:|
| mean w, reassigned assets | 0.526283 |
| mean w, stable assets | 0.733121 |
| reassignment rate, bottom-w quartile | 33.48% |
| reassignment rate, top-w quartile | 4.21% |
| mean within-cluster asset-w standard deviation | 0.090825 |
| median size-versus-w correlation | -0.170931 |
| dates with abs(size-versus-w corr)>0.5 | 14.65% |
| persistent size-confound flag | false |

Coverage is 100% on every one of the 284 operating-partition dates. The first 11 dates use the
mandated unit-weight fallback; estimated pooling begins on date 12.

## Decision table

| cell | both half Sharpe deltas >=0 | turnover CI below 0 | matched risk | reassignment share >50% | verdict |
|---|---|---|---|---|---|
| V1_ewma36 | **no** | yes | yes | yes | **REJECT** |
| V2_ewma36 | **no** | yes | yes | yes | **REJECT** |

Because both cells reject, the variance-only family closes as **REJECT**. V3 remains separately
closed under S6; no stability-pooling production adoption is recommended.

## Acceptance checklist

| check | measured | tolerance | status |
|---|---|---|---|
| V3 public members/runnable cells | 0 | 0 | PASS |
| EWMA hand reference | exact `5/7` boundary weight | explicit arithmetic | PASS |
| span map | `{'ME': 36, 'QE': 18}` | exact | PASS |
| V0 byte identity | exact scorer and empirical baseline replay | exact | PASS |
| EWMA causality | byte-identical through cutoff | exact | PASS |
| stability constructions | V1=1; V2=1 | exactly 1 per cell | PASS |
| shared signal consumers | 2 legs per cell, one object | exact | PASS |
| ROSAA source edits | 0 | 0 | PASS |
| full suites | 0 failures/errors | 0 | PASS |
| split windows | 2 per cell | 2 | PASS |
| active-window rejected solves | 0 | 0 | PASS |
| bootstrap | block 6; 2,000 draws; seed 20260813 | exact | PASS |
| matched risk, V1 | abs TE 0.000128; abs TRE 0.000163 | each `<=0.001` | PASS |
| matched risk, V2 | abs TE 0.000631; abs TRE 0.000262 | each `<=0.001` | PASS |
| mechanism, V1 | 53.30% reassignment share | `>50%` | PASS |
| mechanism, V2 | 52.49% reassignment share | `>50%` | PASS |
| half-window Sharpe, V1 | evaluation -0.052292 | both halves `>=0` | **FAIL / REJECT** |
| half-window Sharpe, V2 | evaluation -0.086375 | both halves `>=0` | **FAIL / REJECT** |
| release/tag/version actions | 0 | 0 | PASS |

## Evidence and reproducibility

The unchanged shared cache is:

`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/stability_pooling/mac/shared_pipeline_inputs.pkl`

Twenty `s5b_*.csv` files and the run log are under the same output directory. The key files are
`s5b_metrics.csv`, `s5b_transition.csv`, `s5b_split_window.csv`, `s5b_bootstrap.csv`,
`s5b_turnover_decomposition.csv`, `s5b_decision.csv`, `s5b_wiring.csv`,
`s5b_acceptance.csv`, and `s5b_source_manifest.csv`. The manifest contains six source hashes;
all six were re-read and verified after the run. The artifact audit also confirmed exact V0
replay, all eight harness acceptance rows PASS, stability weights within `[0,1]`, minimum
coverage 1.0, and exactly 11 short-history fallback dates.

Run command:

```powershell
$env:CLUSTER_LINEAGE_OUTPUT_DIR='C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026'
$env:PYTHONPATH='C:\Users\artur\OneDrive\analytics\my_github\FactorLasso\src;C:\Users\artur\OneDrive\analytics\my_github\OptimalPortfolios\src;C:\Users\artur\OneDrive\analytics\my_github\OptimalPortfolios'
$env:MPLBACKEND='Agg'
C:\Users\artur\OneDrive\analytics\my_github\OptimalPortfolios\.venv\Scripts\python.exe -m papers.cluster_lineage_2026.replication.run_stability_pooling_s5b_mac
```

Measured runtime was 343.0 seconds with the shared covariance/SAA cache. No file was staged,
committed, pushed, tagged, or released. The paper tree remains ignored by the repository's
existing `/papers/cluster_lineage_2026/` rule.
