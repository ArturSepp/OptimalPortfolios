---
myst:
  html_meta:
    description: >-
      Exact Execution, Ranked Execution and audited execution policy: shared lifecycle
      inputs, hard trade rules, production integration and migration from numbered APIs.
---

# Execution integration

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-05](https://github.com/ArturSepp/OptimalPortfolios/commit/13ae5d43aa4d44f61787adfa59b18f4204e851ac)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Scope

The public engines use descriptive names and the same declared execution domain:

| Module | Contract |
|---|---|
| `optimalportfolios.execution.ranked_execution` | Ranked Execution: bounded support search with joint sizing. |
| `optimalportfolios.execution.exact_execution` | Exact Execution: the matched mixed-integer formulation; a time-limited result need not be optimal. |
| `optimalportfolios.execution.policy` | Exact primary with independently audited Ranked fallback. |
| `optimalportfolios.execution.legacy_miqp` | Older operational solver with its separate alpha, position and repair policy. |

Shared lifecycle, preparation, domain, sizing and audit modules sit outside either
engine. Native solver imports remain optional. Preparation can itself need a
mixed-integer backend, so fallback does not protect against every backend outage.
The [scientific article and replication](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/ranked_execution_2026)
compare declared methods and retain the exact software identity of their experiments.
The [prepared-input reference](execution_v2.md) gives the numerical contract.

<a id="documentation-and-method-ownership"></a>
<a id="resolved-inputs"></a>

## Consumer inputs

The consumer supplies current and raw target weights, covariance, calendar eligibility,
settlement cash, explicit instruction intervals, instrument bounds and original
fractional group exposures. Workbook reading, group-membership reconciliation,
asset taxonomy, dates and output formatting remain consumer responsibilities.
Missing group membership is not a zero-exposure declaration. Never synthesize it
from an asset-class label or silently discard an active row.

`MandatoryTrades`, `merge_mandatory_trades` and `build_execution_target` implement
shared lifecycle and instruction mechanics. `ExecutionPolicyConfig` is the common
lifecycle configuration; `ExecutionLifecycleConfig` adds `constraint_set`.
Use `RankedExecutionPolicyConfig` or `MatchedExecutionPolicyConfig` for execution.
`ExecutionRankingConfig`, `score_execution_trades` and
`resolve_effective_max_trades` retain diagnostic score and ticket-count conventions;
the modern engines select from executable states, not the diagnostic selected flag.
`compute_group_bound_bridges` reports signed interval feasibility.

Model support uses `MODEL_WEIGHT_ZERO_TOLERANCE` = 1e-6 NAV (0.01 bp), without
changing raw model weights in the risk objective. Actual movement uses a separate
1e-8 zero threshold. The original-unit feasibility audit remains 1e-7 NAV.
A descriptive objective comparison is not permission to weaken a hard row.

<a id="ranking-and-sizing-contract"></a>

## Implementation in optimalportfolios

`resolve_production_problem` compiles the resolved lifecycle table into an
`ExecutionProblem` and its original execution constraints. `prepare_common_execution`
reserves compulsory repairs and supplies a common audited base. Both engines use
this same preparation, objective and audit.

```python
import numpy as np
from optimalportfolios.execution import (
    ExecutionProblem, ExecutionRules, RankedConfig, audit_execution,
    wrapper_ranked_execution,
)

problem = ExecutionProblem(
    names=("Asset", "Cash"), cash=1,
    current=np.array([0.4, 0.6]), base=np.array([0.4, 0.6]),
    target=np.array([0.5, 0.5]), lower=np.array([0.4, 0.0]),
    upper=np.array([0.5, 1.0]), full_lower=np.zeros(2), full_upper=np.ones(2),
    mandatory=np.array([False, False]), categories=("", ""),
    eligible=np.array([True, True]), frozen=np.array([False, False]),
    groups=np.zeros((0, 2)), group_lower=np.zeros(0), group_upper=np.zeros(0),
    rules=ExecutionRules(minimum_ticket=0.01, holding_floor=0.01, ticket_cap=1),
    covariance=np.diag([0.04, 0.0]),
)
solved = wrapper_ranked_execution(problem, RankedConfig(qp_solver="CLARABEL"))
assert audit_execution(problem, solved.weights)["passed"]
```

`wrapper_ranked_execution` returns a candidate and `audit_execution` checks it
against the original inputs. `wrapper_exact_execution` uses the same prepared problem with an explicit
`CORRIDOR` or `FULL` domain. These are different economic domains; Full results
are not matched algorithm comparisons against Corridor results.
`compile_states` constructs executable intervals. `compute_perspective_lower_bound`
provides relaxation guidance. `ExecutionResult` carries weights, status and diagnostics.
`ResolvedExecutionProblem` and `from_resolved_problem` provide an explicit adapter
for resolved numerical tables plus separately supplied, qualified preparation.
They do not reconstruct lifecycle history or silently repair the inputs.

<a id="acceptance-and-operational-use"></a>

## Production policy and failures

`solve_ranked_execution_portfolio` resolves, prepares and runs Ranked Execution.
It has no V1 callback or V1 escape. An absent audited result raises with the dated
trade table and diagnostics. It must not become an executable display-model fallback.

`wrapper_execution_policy` attempts Exact first on one immutable prepared problem.
An independently audited native witness and consistent bound are required for
primary acceptance. Callback failure, native unavailability, absent witnesses,
invalid weights or inconsistent bounds trigger lazy Ranked fallback on that same
domain. Invalid fallback raises. The policy does not run both engines to pick the
better objective. Native time-limited incumbents remain eligible when all gates pass.
`solve_execution_portfolio` adds lifecycle compilation and common preparation.

`ExecutionOptimizationResult` contains weights, a trade table and an
`OptimizationOutcome`; both `accepted` and `compliant` must pass.
`ExecutionSolverInfeasibility` retains structural diagnostics where applicable.
Modern audit metadata is stored at `trade_table.attrs["execution"]`.
Engine labels are `exact_execution`, `ranked_execution` and
`ranked_audited_fallback`. Numerical trade-table column values are preserved.

<a id="opt-in-scoring-and-support-search"></a>
<a id="guarded-proposals-and-compression"></a>
<a id="compression-of-independent-ranking-branches"></a>
<a id="bounded-support-swaps"></a>
<a id="reusing-identical-resolved-projections"></a>
<a id="reusing-compatible-projection-programs"></a>
<a id="delivering-eligible-incumbents-during-search"></a>

## Migration from numbered interfaces

| Previous interface | Current interface |
|---|---|
| `execution.v2.ranked` | `execution.ranked_execution` |
| `execution.v2.exact` | `execution.exact_execution` |
| `execution.v2.primary` | `execution.policy` |
| `V2ExecutionPolicyConfig` | `RankedExecutionPolicyConfig` |
| `solve_v2_execution_portfolio` | `solve_ranked_execution_portfolio` |
| `wrapper_matched_execution` | `wrapper_execution_policy` |
| `solve_matched_execution_portfolio` | `solve_execution_portfolio` |
| `execution.MIQP.solve_exact_execution_portfolio` | `execution.legacy_miqp.solve_legacy_miqp_portfolio` |

The V1 projection/search, guarded and branched entry points are retired. The former
`solve_ranked_execution(ResolvedExecutionProblem)` is not silently repurposed.
The `allow_v1_fallback` keyword and `fallback` callback are removed; passing them
to the new policy raises. This is a deliberate breaking policy-interface change.
Shared target resolution, schemas, funding conventions and audit tolerances survive.

Historical experiments and serialized records continue in the retained OP 9.2.0
cohort. Do not rewrite their labels, pickles, hashes or numerical expectations.
Export explicit numerical tables/configuration under their owning runtime before
comparing with this API. A namespace migration is not evidence of solver superiority.

Rosaa workbook aliases can retain `ranked_v2` and `matched_miqp_v2` only with an
explicit migration notice and unchanged routing. Old `exact` and `exact_soft`
mean the legacy operational model, never the matched engine. Retired `ranked`
and `risk_budget_ranked` require an explicit replacement choice.

## Configuration and result fields

The following lists name the complete owned dataclass fields. Economic inputs and
numerical controls retain their existing units and defaults. Shared legacy scoring
fields remain for lifecycle and diagnostic compatibility; the modern policy rejects
controls that would change its full-covariance objective or state search.

`ExecutionProblem`: `names`, `cash`, `current`, `base`, `target`, `lower`, `upper`, `full_lower`, `full_upper`, `groups`, `group_lower`, `group_upper`, `mandatory`, `categories`, `eligible`, `frozen`, `rules`, `covariance`, `factor_loadings`, `factor_covariance`, `residual_variance`, `investment`, `annualisation`, `provenance`, `preparation_seconds`.

`ExecutionRules`: `minimum_ticket`, `holding_floor`, `ticket_cap`, `mandatory_count_in_cap`, `exemption_mask`, `discretionary_allowance`, `zero_tolerance`, `audit_tolerance`.

`RankedConfig`: `drop`, `exchange`, `compute_bound`, `rescue_trials`, `mandatory_state_trials`, `objective_tolerance`, `qp_solver`, `improvement_passes`.

`ExecutionResult`: `weights`, `status`, `diagnostics`.

`ExecutionRankingConfig`: `tre_weight_by_asset_class`, `max_trades`, `add_mandatory_trades_to_max_trades`, `minimum_trade_score`, `sequential_greedy_selection`.

`ResolvedExecutionProblem`: `target`, `covariance`, `alphas`, `asset_classes`, `constraints`, `ranking_config`, `optimiser_config`, `asset_class_tre_weights`, `partition_groups`, `expand_for_feasibility`, `sign_directed_rescue`, `allow_corridor_relaxation`, `preserve_model_directions`, `context`.

`ExecutionOptimizationResult`: `weights`, `trade_table`, `outcome`.

`ExecutionPolicyConfig`: `small_trade_cutoff`, `small_trade_cutoff_by_asset_class`, `sell_down_dust_threshold`, `material_trade_threshold`, `max_trades`, `expand_for_feasibility`, `minimum_trade_score`, `sequential_greedy_selection`, `sign_directed_rescue`, `allow_corridor_relaxation`, `tre_weight_by_asset_class`, `use_weight_drift`, `mandatory_trades`, `output_by_ticker`, `instruction_correction_tolerance`, `exact_execution`, `add_mandatory_trades_to_max_trades`.

`ExecutionLifecycleConfig`: `small_trade_cutoff`, `small_trade_cutoff_by_asset_class`, `sell_down_dust_threshold`, `material_trade_threshold`, `max_trades`, `expand_for_feasibility`, `minimum_trade_score`, `sequential_greedy_selection`, `sign_directed_rescue`, `allow_corridor_relaxation`, `tre_weight_by_asset_class`, `use_weight_drift`, `mandatory_trades`, `output_by_ticker`, `instruction_correction_tolerance`, `exact_execution`, `add_mandatory_trades_to_max_trades`, `constraint_set`.

`RankedExecutionPolicyConfig`: `small_trade_cutoff`, `small_trade_cutoff_by_asset_class`, `sell_down_dust_threshold`, `material_trade_threshold`, `max_trades`, `expand_for_feasibility`, `minimum_trade_score`, `sequential_greedy_selection`, `sign_directed_rescue`, `allow_corridor_relaxation`, `tre_weight_by_asset_class`, `use_weight_drift`, `mandatory_trades`, `output_by_ticker`, `instruction_correction_tolerance`, `exact_execution`, `add_mandatory_trades_to_max_trades`, `constraint_set`, `ranked_config`, `preparation_time_limit_s`, `preparation_solver`.

`MatchedExecutionPolicyConfig`: `small_trade_cutoff`, `small_trade_cutoff_by_asset_class`, `sell_down_dust_threshold`, `material_trade_threshold`, `max_trades`, `expand_for_feasibility`, `minimum_trade_score`, `sequential_greedy_selection`, `sign_directed_rescue`, `allow_corridor_relaxation`, `tre_weight_by_asset_class`, `use_weight_drift`, `mandatory_trades`, `output_by_ticker`, `instruction_correction_tolerance`, `exact_execution`, `add_mandatory_trades_to_max_trades`, `constraint_set`, `ranked_config`, `preparation_time_limit_s`, `preparation_solver`, `primary_time_limit_s`, `primary_threads`, `primary_gap_rel`.

`MandatoryTrades`: `hard_sell`, `hard_buy_min`, `hard_max`, `hard_target`.

<a id="production-preparation-and-operational-exact-controls"></a>

## Legacy operational model

`ExactExecutionConfig` and `solve_legacy_miqp_portfolio` belong only to
`optimalportfolios.execution.legacy_miqp`. Its optional alpha term, position cap,
repair-on-touch rules and soft sub-asset penalties are distinct from matched
Exact Execution. Naming changes do not substitute its objective or constraint policy.

`ExactExecutionConfig`: `max_positions`, `repair_on_touch`, `alpha_tradeoff`, `solver`, `relative_gap_tolerance`, `absolute_gap_tolerance`, `max_branches`, `max_solve_seconds`, `accept_audited_incumbent`, `seed`, `sub_asset_class_penalty`, `constraint_set`.

<a id="see-also"></a>

## Verification and references

Run the package execution tests with the repository's prescribed external runtime.
Qualification compares delivered weights, trade selection, mandatory categories,
hard bounds, residuals and fallback acceptance, separately from metadata names.
[QIS](https://github.com/ArturSepp/QuantInvestStrats) owns the surrounding backtest
and risk-reporting calculations. See [constraints](constraints.md),
[prepared execution](execution_v2.md) and the
[public manuscript](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/ranked_execution_2026).
