# Stability-pooled z-score S0 environment and read-in report

**Date:** 2026-08-20  
**Executor:** Sol  
**Proposer:** Ben (Monday TAA meeting)  
**Status:** COMPLETE — definitions feasible; implementation gate open

## Outcome

All five pinned definitions are implementable without changing a ROSAA production path. The
diagnostic will enter only at the OptimalPortfolios cluster-scoring seam and will be activated by
the isolated research harness. The frozen U1/U2/U3 paper specifications and exhibits remain out of
scope.

The required checkout-first assertion passed:

```text
factorlasso runtime= C:\Users\artur\OneDrive\analytics\my_github\FactorLasso\src\factorlasso\__init__.py
checkout-first assertion: PASS
```

The FactorLasso checkout was clean on `main...origin/main` before implementation.

## Three-way usage map

### FactorLasso

- `src/factorlasso/cluster_smoothing.py:253` defines the private
  `_co_association_panel`.
- `src/factorlasso/cluster_smoothing.py:441` is its only call: the optional confidence panel
  attached to `RollingClusterData`.
- FactorLasso has no `score_within_clusters` implementation or call site before S1.

### OptimalPortfolios

- Canonical scoring seam: `src/optimalportfolios/alphas/signals/utils.py:113`,
  `score_within_clusters`.
- Public re-exports: `optimalportfolios.alphas` and
  `optimalportfolios.alphas.signals`.
- Runtime callers: momentum, classic momentum, low beta, carry, residual momentum, and residual
  reversal cluster signals.
- Tests: `alphas/tests/cluster_scoring_test.py` and the classic-momentum signal tests.
- OptimalPortfolios has no `_co_association_panel` call.

### ROSAA

- ROSAA has no direct call to either symbol. `AlphaAggregator.compute_alphas` routes the
  `MOMENTUM_CLUSTER` and `LOW_BETA_CLUSTER` legs through the OptimalPortfolios signal functions,
  which then call `score_within_clusters`.
- The only textual `score_within_clusters` occurrence is an explanatory comment in
  `products/funds/run_sweep_backtests.py`.
- The MAC production stack is `Signal.PROD_MOM_BETA_CLUSTER`, so both required treatment signals
  cross the same OptimalPortfolios scoring seam.

## MAC runtime configuration

The production input was constructed without fitting or backtesting. The measured configuration
is:

| item | measured |
|---|---|
| product | `mac` |
| TAA estimation/rebalance schedule | `ME` |
| base Lasso span | 36 |
| per-frequency spans | `ME: 36`, `QE: 12` |
| cluster smoother | `NONE` |
| recluster frequency | none |
| minimum cluster size | 5 |

The ME and QE spans represent the same three-year calendar horizon. The co-association panel is
indexed by the monthly covariance snapshots, so the S3 windows are 36 and 72 monthly partition
dates. Counting only 12 monthly rows for the QE-labelled assets would shorten their horizon to one
year and would not implement the production span.

## Pinned definitions

| definition | S0 ruling |
|---|---|
| trailing window | CONFIRMED with a transport clarification: resolve the span from the runtime `PipelineInputData.covar_estimator.lasso_model`, because `ProductConfig` itself has no covariance-span field. For MAC this yields 36 monthly partition dates; the sweep is 36/72. No value is hardcoded in the scorer. |
| partitions feeding `w` | CONFIRMED. Use raw operating partitions. MAC uses `ClusterSmootherType.NONE`, so raw and operating partitions coincide. A future smoothed diagnostic must pass raw partitions explicitly. |
| point in time | CONFIRMED. The panel at date `t` uses partition dates through `t` only. Existing one-period signal implementation lag remains downstream and unchanged. |
| short history | CONFIRMED. The first 11 available partition dates use `w=1`; pooling begins on the twelfth. Coverage is recorded per date. |
| small-cluster bias | CONFIRMED. Compute the cross-sectional Pearson correlation between current cluster size and per-cluster `w` on every eligible date; report the share of finite dates with `abs(corr)>0.5`. |

The transport clarification preserves the owner's numerical intent and requires no owner ruling.

## Proposed wiring

1. FactorLasso exposes a documented public co-association accessor while leaving the existing
   private call used by `compute_rolling_smoothed_clusters` unchanged.
2. FactorLasso owns the additive stability-pooling standardisation implementation and enum. Its
   default mode is an exact bypass.
3. OptimalPortfolios keeps the existing default `score_within_clusters` branch untouched and
   exposes optional stability inputs that delegate only when explicitly selected.
4. The MAC diagnostic is an isolated research runner under
   `papers/cluster_lineage_2026/replication/`. It patches the two imported scoring references in
   the momentum and low-beta modules for the duration of a run, derives the 36/72 windows from the
   runtime covariance model, and restores the references afterward.
5. No ROSAA source file or production configuration is edited.

## Acceptance

| check | measured | tolerance | status |
|---|---:|---:|---|
| required background documents read | 4/4 | 4 | PASS |
| FactorLasso checkout-first assertion | local `src/factorlasso/__init__.py` | local checkout | PASS |
| repositories searched | 3/3 | 3 | PASS |
| direct ROSAA production call sites | 0 | 0 | PASS |
| pinned definitions feasible | 5/5 | 5 | PASS |
| production configuration changes | 0 | 0 | PASS |

No files were staged, committed, pushed, tagged, or released.
