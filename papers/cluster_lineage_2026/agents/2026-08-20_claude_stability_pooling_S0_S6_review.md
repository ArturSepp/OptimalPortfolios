# Claude review: stability-pooled z-score workstream (S0-S6)

**Date:** 2026-08-20
**Reviewer:** Claude
**Scope:** Sol reports S0-S6 of 2026-08-20 and `ROADMAP_stability_pooled_zscore.md`
**Proposer of the method:** Ben (Monday TAA meeting)

## Summary

Sol completed all seven stages in one day and recommends REJECT. The three decisive numbers are: full-sample annual turnover falls by 0.158, the full-sample Sharpe delta of +0.070 carries a 95% CI of [-0.037, +0.164], and the evaluation-half Sharpe delta is -0.099. The turnover effect is real and survives both half-windows and the bootstrap. The Sharpe effect does not survive either robustness test. Under the roadmap's predeclared bar, a turnover gain that costs Sharpe at matched risk is a REJECT, and Sol applied the bar correctly.

I agree with the verdict as stated, but it is narrower than it reads. The robustness gate was applied only to V3/36, the mean-plus-variance comparison arm that breaks asset-class neutrality. Ben's actual proposal, the variance-only arms V1 and V2, never reached S5. The workstream has therefore rejected the comparison arm and left the primary proposal untested at the robustness gate. I recommend one incremental run before closing.

## What the record establishes

The engineering discipline is clean throughout. The fail-before-pass checkpoint was genuine (12 failures plus one collection error before the APIs existed). V0 byte-identity, the w=1 and w=0 endpoint identities, causality, and the short-history fallback all pass exactly. No production path, ROSAA source file, or default was changed, and nothing was committed or released. The baseline provenance question was handled well: V0 at 1.069 matches the 19 August production factsheet, and the roadmap's 1.15 reference is a superseded 12 August vintage. All deltas are same-process comparisons, so the vintage drift does not touch the verdict. The roadmap reference should still be corrected if the workstream continues.

## Findings

### 1. The REJECT binds V3, not Ben's proposal

The S5 selection rule mechanically picked the highest full-sample Sharpe among turnover-reducing cells, which is V3/36. Sol followed the roadmap as written. But V3 pools the mean as well as the variance, raises ex-post TE by 0.0031 and ex-ante TRE by 0.0023, and was declared a comparison arm from the outset. The variance-only arms hold tracking risk essentially flat and were the production-eligible candidates. Their full-sample rows are positive on both axes:

| cell | Sharpe delta | turnover delta | TE delta | TRE delta |
|---|---:|---:|---:|---:|
| V1/36 (Ben's proposal) | +0.0297 | -0.0884 | +0.0004 | -0.0000 |
| V2/36 (per-asset) | +0.0188 | -0.1076 | -0.0000 | -0.0001 |

Neither row has a split-window or bootstrap result. The honest current status of the variance-only family is ROBUSTNESS (undetermined), not REJECT.

### 2. Grid structure: 36 dominates 72, and V2/36 is the cleanest matched-risk cell

The 1x window beats the 2x window in every variant pair, and V2/72 is the only cell with a negative full-sample Sharpe delta. The 72-month window dilutes the stability signal rather than stabilising it. V2/36 is the only cell that reduces turnover, reduces both risk measures, and raises Sharpe simultaneously. The S4 within-cluster asset-level w dispersion of 0.10 supports V2's per-asset granularity: V1 smears exactly that information across the cluster.

### 3. Mechanism: boundary channel confirmed, turnover channel mixed

The boundary diagnostic is the strongest single result in the workstream. At the 36-month window, reassigned assets carry mean stability 0.506 against 0.697 for stable assets, and bottom-stability-quartile assets reassign 31.7% of the time against 4.9% in the top quartile. Trailing co-association identifies boundary assets well. This result stands regardless of the adoption decision and is worth keeping as a descriptive exhibit.

The turnover decomposition is weaker than the headline split suggests. The reassignment share of the direct reductions is 56.2%, so 43.8% of the effect sits in the signal component. The two direct legs (1.43 and 1.65 per year for V0) are roughly three times the counterfactual total (0.56), with a trade-interaction residual of -2.52 absorbing the difference. A triangle bound of this shape limits how much weight the 56/44 split can carry. The mechanism evidence is directional, as Sol says plainly.

### 4. The split asymmetry is the substantive negative result

V3/36's entire gain sits in the 2004-2015 half (+0.158 against a baseline Sharpe of 0.86) and reverses in the 2015-2026 half (-0.099 against a baseline of 1.32 with a -0.077 max drawdown). Two readings are consistent with this pattern. Either the pooling helps when clustering is unstable and markets are stressed and hurts in the calm trending regime after 2015, or the full-sample +0.070 is selection noise on a seven-cell grid. The bootstrap CI covering zero is consistent with both. Either way, the full-sample Sharpe rows in S3 should not be quoted without the split.

### 5. The turnover gain is operational, not a performance argument

MAC production rebalancing cost is 0.0 in the harness, so the measured Sharpe deltas contain no contribution from turnover savings. At a hypothetical 10 bp one-way cost, V1/36's turnover reduction of 0.088 is worth under 0.1 bp of annual return. The case for pooling is fewer reassignment-driven trades and cleaner implementation, never net-of-cost performance. Any adoption argument should be framed that way.

## Recommendation

Run the S5 protocol on V1/36 and V2/36 before closing the workstream. The shared fitted-input cache exists and the S3 runtime was dominated by the one-off covariance/SAA fit, so two additional split-window and bootstrap passes are cheap. The decision rule differs from V3's: these arms hold tracking risk matched, so the adoption bar reduces to (a) evaluation-half Sharpe delta at or above zero and (b) turnover CI excluding zero. Expect the full-sample Sharpe CI to cover zero, since V3's larger delta did. A flat-Sharpe, robust-turnover outcome for V2/36 would meet the letter of the roadmap's adoption bar. A negative evaluation-half Sharpe delta would confirm REJECT for the whole family and the workstream closes.

One design note for Ben's attention. The S4 evidence shows w is a good boundary detector but variance pooling is an indirect instrument: it shrinks every denominator toward the global variance and therefore touches all scores, which is why 44% of the turnover reduction leaks into the signal channel. If the goal is reassignment turnover specifically, a more surgical instrument would act on the assignment itself, for example gating reassignment on w falling below a threshold, or weighting the optimizer's turnover penalty by 1-w. Either would use the same co-association panel already promoted to a public accessor in S1. This is a candidate follow-on workstream, not an amendment to this one.

Nothing here touches the frozen U1/U2/U3 paper specs, and the S6 constraint record confirms none were touched. If a variance-only arm survives its robustness pass, the natural paper home is a short section in the cluster-lineage manuscript with the boundary diagnostic (31.7% against 4.9%) as the motivating exhibit.

## Stage-record cross-check

| stage | Sol status | review status |
|---|---|---|
| S0 | definitions confirmed, call-site map complete | consistent, transport clarification is sound |
| S1 | additive API, isolated harness | consistent, no scope creep found |
| S2 | fail-before-pass, suites green | consistent |
| S3 | seven cells, baseline deviation disclosed | consistent, vintage drift handled correctly |
| S4 | boundary confirmed, mechanism mixed | consistent, triangle-bound caveat added above |
| S5 | turnover robust, Sharpe fails | consistent, but run on V3/36 only |
| S6 | REJECT | agree for V3, variance-only arms remain undetermined |
