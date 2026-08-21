# Stage E0 execution report — paths, metric library, and config registry

**Date:** 2026-08-13  
**Roadmap:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`  
**Stage:** E0  
**Status:** COMPLETE; OWNER GATE E0 pending

## Outcome

Stage E0 is implemented and its acceptance checks pass. The paper harness now has one
environment-controlled output-root helper, one frozen universe/config registry, and one
canonical paper-local metric library. No file under `optimalportfolios/`, `factorlasso`,
`qis`, or `rosaa/` was changed by this execution.

The existing S&P 500 baseline calculations for raw churn, lineage churn, matcher churn,
taxonomy ARI, rank stability, partition equality, and residual diagonality were moved into
the metric library without changing their numerical outputs. The lineage calculation now
calls canonical `factorlasso.analyze_cluster_lineage`, rather than the deprecated
`optimalportfolios.covar_estimation.risk_labelling` compatibility shim.

## Deliverables

- `papers/cluster_lineage_2026/replication/local_path.py`
  - Reads `CLUSTER_LINEAGE_OUTPUT_DIR`.
  - Defaults to `~/OneDrive/analytics/outputs/cluster_lineage_2026/`.
  - Creates output/cache directories only when explicitly requested.
- `papers/cluster_lineage_2026/replication/configs.py`
  - Defines enum/dataclass registries for U1 MSCI US, U2 futures, and U3 MAC.
  - Defines all six fixed configurations and owner-calibration slots `M1_star`/`M2_star`.
  - An unresolved calibrated slot raises rather than silently selecting a value.
- `papers/cluster_lineage_2026/replication/metrics.py`
  - Implements the frozen E0 metric inventory and states formulas/aggregation conventions
    in its module and function docstrings.
  - Consumes FactorLasso for lineage matching and residual diagonality.
- `papers/cluster_lineage_2026/replication/metrics_test.py`
  - Formula identities, frozen S&P 500 regression, and repeated-run byte determinism.
- `papers/cluster_lineage_2026/replication/validate_e0.py`
  - Reproducible acceptance runner with machine-readable measured values.
- `papers/cluster_lineage_2026/replication/validate_e0_independent.py`
  - Independent ARI check through scikit-learn and direct pair-count lineage check.

The following existing runners were refactored onto the paper-local path/metric helpers:

- `papers/cluster_lineage_2026/replication/sp500_baseline.py`
- `papers/cluster_lineage_2026/replication/methods.py`
- `papers/cluster_lineage_2026/replication/run_sweep.py`
- `papers/cluster_lineage_2026/replication/lineage_matching_validation.py`

The roadmap named the first three as the existing `rosaa` import sites. Audit found a fourth
in `lineage_matching_validation.py`; it was also removed so the binding global constraint
"no rosaa imports anywhere in the paper harness" holds literally.

## Runners and cache directories

Primary acceptance runner:

```text
papers/cluster_lineage_2026/replication/validate_e0.py
```

Independent numerical runner:

```text
papers/cluster_lineage_2026/replication/validate_e0_independent.py
```

Regression-test runner:

```text
papers/cluster_lineage_2026/replication/metrics_test.py
```

Frozen cache read by all three:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_smoothing\sp500_baseline\baseline\
```

The E0 commands set
`CLUSTER_LINEAGE_OUTPUT_DIR=C:\Users\artur\OneDrive\analytics\outputs` only to address this
pre-roadmap frozen cache in place. The new helper's default remains the roadmap-mandated
`C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\`. E0 wrote no cache files.

## Acceptance checks

| Check | Measured | Target / tolerance | Result |
|---|---:|---:|---|
| Frozen snapshots | 60 | 60 | PASS |
| Lineage churn | 3.2114693774 | 3.2115 ± 0.0001; error 0.0000306226 | PASS |
| Derived tracks | 216 | 216 exactly | PASS |
| Matcher-attributable churn | 0.4857547101 | 0.486 ± 0.005; error 0.0002452899 | PASS |
| Median sector ARI | 0.2029670733 | frozen 0.202967 ± 0.005; error 0.0000000733 | PASS |
| Median industry-group ARI | 0.2970123965 | frozen 0.297012 ± 0.005; error 0.0000003965 | PASS |
| Median industry ARI | 0.3319349338 | frozen 0.331935 ± 0.005; error 0.0000000662 | PASS |
| Repeated metric bytes | 884 bytes = 884 bytes | byte-identical | PASS |
| Independent ARI maximum difference | 1.110e-16 | ≤ 1e-12 | PASS |
| Independent lineage pair/panel difference | 0.000e+00 | ≤ 1e-12 | PASS |
| Python files importing `rosaa` in paper replication | 0 | 0 | PASS |
| Refactored-module import/compile check | 4/4 | 4/4 | PASS |
| New-file Ruff E/F/W audit | 0 findings | 0 | PASS |

The frozen ARI targets above are the values in
`C:\Users\artur\OneDrive\analytics\outputs\sp500_cluster_smoothing_sweep_20260811.xlsx`,
sheet `tier2_grid`, baseline row. This workbook was read independently before the new metric
runner was evaluated.

## Verification commands and results

Formula/import smoke test:

```powershell
$env:CLUSTER_LINEAGE_OUTPUT_DIR='C:\Users\artur\OneDrive\analytics\outputs'
python -m compileall -q papers\cluster_lineage_2026\replication
python -c "from papers.cluster_lineage_2026.replication import configs, local_path, metrics, methods, run_sweep, sp500_baseline, lineage_matching_validation; ..."
```

Result: `compile/import/formula smoke: PASS`; three universe specs and eight smoother slots.

Acceptance runner:

```powershell
python -m papers.cluster_lineage_2026.replication.validate_e0
```

Result: all six frozen numerical checks PASS; repeated serialisation byte-identical.

Regression test:

```powershell
python -m pytest papers\cluster_lineage_2026\replication\metrics_test.py -q
```

The required red/green proof was performed. With the lineage target deliberately changed
from 3.2115 to 3.0000, the test failed with measured error 0.2114693774. The correct frozen
target was restored and the final result was `2 passed`.

Independent numerical pass:

```powershell
python -m papers.cluster_lineage_2026.replication.validate_e0_independent
```

Result: scikit-learn ARI differs from the paper implementation by at most `1.110e-16`;
direct pair-count and wide membership-panel lineage churn differ by `0.000e+00`.

Focused lint:

```powershell
ruff check --isolated --select E,F,W --line-length 100 \
  papers\cluster_lineage_2026\replication\local_path.py \
  papers\cluster_lineage_2026\replication\configs.py \
  papers\cluster_lineage_2026\replication\metrics.py \
  papers\cluster_lineage_2026\replication\metrics_test.py \
  papers\cluster_lineage_2026\replication\validate_e0.py \
  papers\cluster_lineage_2026\replication\validate_e0_independent.py
```

Result: `All checks passed!`

## Deviations and open items

- Deviation from the three-file refactor inventory: the additional pre-existing
  `lineage_matching_validation.py` `rosaa` import was removed to satisfy the stronger global
  no-`rosaa` constraint. Its MAC input is now an explicit paper-local pickle path and it
  fails with instructions when that cache is absent; it no longer estimates through rosaa.
- The calibrated `M1_star` and `M2_star` registry slots are present but intentionally unset,
  pending the owner values specified by the roadmap.
- `.gitignore` line 185 excludes the entire `papers/cluster_lineage_2026` tree. These
  deliverables exist in the working tree but do not appear in `git status` and cannot be
  staged without an explicit owner decision about that ignore rule. E0 did not change the
  repository ignore policy.
- E1 and all later stages have not started. This is required by the stage gate.

## GATE REQUEST

The owner must rule whether to approve the E0 metric definitions, aggregation conventions,
registry, and no-`rosaa` path refactor as frozen, thereby authorising Stage E1. If not
approved, the owner must identify the metric or convention to revise before E1 begins.
