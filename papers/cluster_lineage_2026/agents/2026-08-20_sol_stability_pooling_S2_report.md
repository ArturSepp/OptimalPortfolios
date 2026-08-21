# Stability-pooled z-score S2 verification report

**Date:** 2026-08-20  
**Executor:** Sol  
**Proposer:** Ben (Monday TAA meeting)  
**Status:** COMPLETE — fail-before-pass checkpoint and regression suites green

## Outcome

All frozen arithmetic, identity, fallback, and causality tests pass. The fail-before-pass
checkpoint was genuine: before implementation the new FactorLasso file produced 12 failures from
the missing APIs, and the OptimalPortfolios identity suite failed during collection on the same
missing imports. The APIs were then implemented without changing the tests.

## Frozen tests

| requirement | measured result |
|---|---|
| hand-computed V1/V2/V3 | exact to explicit arithmetic (`atol=1e-14`) |
| V0 identity | exact pandas frame equality and identical NumPy bytes |
| `w=1` identity | exact frame equality and identical bytes for V1/V2/V3 |
| `w=0` endpoint | exact cluster-demeaned/global-variance arithmetic |
| minimum-cluster fallback | exact global-score fallback under V1/V2/V3 |
| causality | future partition perturbations leave the panel through `t` byte-identical |
| short history | first 11 dates equal one; estimated pooling begins on date 12 |
| public/private panel regression | public default equals the pre-existing private panel exactly |

## Verification output

Focused FactorLasso tests:

```text
............................                                             [100%]
```

Focused OptimalPortfolios scoring tests:

```text
..........................................                               [100%]
```

Full FactorLasso suite, with `MPLBACKEND=Agg` and an explicit workspace temp directory:

```text
........................................................................ [ 13%]
........................................................................ [ 27%]
........................................................................ [ 41%]
.................................................sssssssss.............. [ 55%]
........................................................................ [ 69%]
........................................................................ [ 83%]
........................................................................ [ 97%]
...........                                                              [100%]
```

Full OptimalPortfolios suite:

```text
........................................................................ [  5%]
........................................................................ [ 10%]
........................................................................ [ 15%]
........................................................................ [ 20%]
........................................................................ [ 25%]
........................................................................ [ 30%]
........................................................................ [ 36%]
........................................................................ [ 41%]
........................................................................ [ 46%]
........................................................................ [ 51%]
........................................................................ [ 56%]
........................................................................ [ 61%]
........................................................................ [ 66%]
........................................................................ [ 72%]
........................................................................ [ 77%]
........................................................................ [ 82%]
........................................................................ [ 87%]
........................................................................ [ 92%]
........................................................................ [ 97%]
................................                                         [100%]
```

The first full FactorLasso invocation found two Tk-backend failures and one denied system-temp
fixture, all caused by the execution sandbox. Repeating the identical suite with the mandated
non-interactive backend and `tmp/pytest/stability-factor` made every collected test green. The two
remaining warnings are pre-existing warmup/insufficient-observation disclosures.

## Acceptance

| check | measured | tolerance | status |
|---|---:|---:|---|
| fail-before-pass proof | 12 failures + 1 collection error before APIs existed | must fail | PASS |
| focused FactorLasso | 28/28 passed | all | PASS |
| focused OptimalPortfolios | 42/42 passed | all | PASS |
| full FactorLasso | 100% completed; 9 skips; 0 failures/errors | 0 failures/errors | PASS |
| full OptimalPortfolios | 100% completed; 0 failures/errors | 0 failures/errors | PASS |
| FactorLasso lint | 0 findings | 0 | PASS |
| OptimalPortfolios changed-file lint | 0 gated `F/W` findings | 0 | PASS |
| MAC harness lint | 0 findings | 0 | PASS |

No files were staged, committed, pushed, tagged, or released.
