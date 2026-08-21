# F2 report — revised-form P4 re-evaluation

**Date:** 2026-08-20  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v2  
**Status:** COMPLETE

## Execution and outputs

Runner: `papers/cluster_lineage_2026/replication/run_f2_p4_revised.py`.
Focused test: `replication/f2_p4_revised_test.py`.

Output directory:
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/f2/`.

The principal artifact is `p4_revised.csv`; `source_manifest.csv`, `acceptance.csv`, and
`determinism.csv` record input provenance and execution checks. All 32 panel/window/config
rows trace to F0-inventoried E3b margin, kurtosis, predicted-realised, and corrected
frequency-scaling tables. No estimator was refit.

## Revised construction

For each panel/window, the Gaussian crossing calculation first removes the cached
elliptical multiplier `sqrt(1 + kappa_hat)` from sigma. One proportionality constant
`c = realised / Gaussian-predicted annualised churn` is then calibrated on the baseline
configuration at the native estimation frequency and held fixed across all eight
configurations. The quarterly prediction evaluates sigma at the quarterly step and applies
the same `c`; it is never re-estimated per configuration.

| Panel/window | Baseline constant c | Cross-config revised correlation | Mean realised-minus-predicted quarterly gap | Verdict |
|---|---:|---:|---:|---|
| Equity, full | 1.118558 | 0.991636 | 0.169071 | Supported revised ordering; level equality rejected |
| Equity, headline | 1.032973 | 0.992536 | 0.131371 | Supported revised ordering; level equality rejected |
| Futures, full | 0.808221 | 0.938516 | 0.077345 | Supported revised ordering; level equality rejected |
| Fund, full | 1.321226 | 0.926193 | 0.229830 | Descriptive: different ME/QE sleeves and spans |

The revised statement is therefore supported as a cross-configuration ordering result for
the equity and futures panels, not as an equality in levels. Every equity/futures quarterly
gap is positive. The fund-panel correlation is reported for completeness but excluded from
the P4 verdict under the binding different-sleeve/different-span ruling.

Selected level checks illustrate the remaining gap. Full-panel equity baseline predicted
1.252364 against realised 1.436503; calibrated smoothing predicted 0.196288 against realised
0.451840. Futures baseline predicted 0.239636 against realised 0.382598; calibrated
smoothing predicted 0.085246 against realised 0.142160. These are annualised quarterly-
schedule churn quantities in the frozen E3b units.

## Acceptance checks

| Check | Measured | Tolerance | Result |
|---|---:|---:|---|
| F0 source paths resolved once | 32/32 | 32/32 | PASS |
| Unrevised E3b ratio regression error | `5.551115e-16` | `<=1e-9` | PASS |
| Baseline native c-calibration error | `1.376677e-14` | `<=1e-12` | PASS |
| Constants per panel/window | 1 | 1 | PASS |
| Constant-calibration configurations | baseline only | baseline only | PASS |
| Revised P4 rows | 32 | 32 | PASS |
| Fund rows entering the P4 verdict | 0 | 0 | PASS |
| NaNs in `p4_revised.csv` | 0 | 0 | PASS |
| Deterministic artifacts | 3/3 byte-identical | 3/3 | PASS |
| Focused pytest | 2 passed | all pass | PASS |
| Isolated Ruff E/F/W | 0 findings | 0 | PASS |

No cache was modified. No git staging or push occurred.
