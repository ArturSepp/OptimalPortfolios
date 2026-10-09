---
myst:
  html_meta:
    description: >-
      Ranked execution from resolved portfolio decisions: copied inputs, funded
      trade ranking, corridor sizing, feasibility rescue and acceptance diagnostics.
---

# Ranked execution integration

*Author: [Artur Sepp](https://github.com/ArturSepp)*

This guide describes [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Scope

`optimalportfolios.execution` ranks and sizes one execution decision whose product
policy has already been resolved by the caller. It keeps the current portfolio,
raw model, effective target and funded post-mandatory base distinct. The main
entry point, `solve_ranked_execution`, accepts a `ResolvedExecutionProblem` and
returns an `ExecutionOptimizationResult`.

The caller owns lifecycle rules, desk instructions, cadence calendars, cash
bootstrap and construction of the resolved table. This interface does not parse
workbooks, infer a named production preset or switch a live execution engine.
Parity of this numerical stage does not establish parity of the complete
production decision pipeline. Exact execution and target-band risk-budget
execution are outside this interface.

## Resolved inputs

`ResolvedExecutionProblem` copies and aligns its mutable inputs at construction.
Its pandas objects can still be edited; `solve_ranked_execution` creates and
validates a fresh snapshot for each call. Keep the original inputs and effective
configuration with the decision record when comparing engines.

| Input | Meaning |
|---|---|
| `target` | Instrument-indexed resolved table, with unique rows and columns. Use the labels in `optimalportfolios.execution.schema`. |
| `covariance` | Finite symmetric covariance covering every execution instrument; aligned to the target rows. |
| `alphas` | Labeled selection scores. Missing values retain the legacy zero substitution; preserve missing-alpha provenance in the caller's resolved table. |
| `asset_classes` | Complete caller-defined class labels; each label needs a configured ranking coefficient. |
| `constraints` | Existing OP `Constraints`, copied before execution-specific construction. |
| `ranking_config` | `ExecutionRankingConfig` with the effective settings for this decision. |
| `optimiser_config` | Existing `OptimiserConfig`; numerical and covariance-factorization settings. |
| `asset_class_tre_weights` | Optional complete nonnegative multipliers for the configured classes. |
| `partition_groups` | Optional group names identifying an exhaustive, disjoint membership partition for bridge tightening. |
| `expand_for_feasibility` | Allow the solver to admit additional eligible trades after the requested solve fails. |
| `sign_directed_rescue` | Enable the legacy signed-residual rescue filter where no numeric interval bridge is available. |
| `allow_corridor_relaxation` | Allow an explicitly flagged retry using wider selected-trade bounds. |
| `context` | Decision label passed to diagnostics. |

The required numeric table columns are `current_weight`, `raw_model_weight`,
`effective_model_weight`, `base_weight`, `desired_reweight`, `mandatory_funding`,
`policy_min_weight` and `policy_max_weight`. Required flags are `settlement_cash`,
`mandatory_trade`, `rule4_trade`, `rebalance_cadence_eligible`, `trade_candidate`
and `material_trade`. Exactly one row identifies settlement cash. A discretionary
candidate must be a material, cadence-eligible, noncash Rule 4 row with a nonzero
desired move and no mandatory-trade flag.

Additional instruction, entry, exit and cutoff provenance columns affect the
legacy corridor rules. Preserve the relevant columns from the resolved decision;
the required-column check alone does not reconstruct these product semantics.
The caller remains responsible for consistency between the resolved weights,
funding amounts, flags and instruction provenance.

Weights are fractions of NAV. Covariance is used in the supplied variance units;
there is no estimation, resampling or annualization in this interface. State the
return convention and covariance frequency in the calling workflow. PSD handling
remains that of the existing OP factorization and solver; this interface does not
introduce a separate covariance repair policy.

Sizing also inherits the minimum-TRE wrapper's covariance filtering: instruments
with nonpositive diagonal variance are excluded before its solve. This includes
an exactly riskless cash row. Such a model is not supported as a retained cash
instrument by that wrapper; do not add an arbitrary variance to conceal the
limitation. See [input filtering](minimum_tracking_error.md#input-filtering-and-benchmark-coverage)
and inspect the final full-table funding and constraint audit.

## Ranking and sizing contract

`ExecutionRankingConfig` contains `tre_weight_by_asset_class`, `max_trades`,
`add_mandatory_trades_to_max_trades`, `minimum_trade_score` and
`sequential_greedy_selection`. A missing `max_trades` means unrestricted requested
selection. `resolve_effective_max_trades` calculates the effective allowance with
the declared mandatory-ticket treatment.

`score_execution_trades` scores cash-funded candidate moves using the legacy
alpha contribution and incremental tracking-error contribution. Optional
sequential selection updates the active portfolio between picks. Class labels
and their coefficients are caller inputs; no four-class product taxonomy is
required. The legacy alpha/TRE score is preserved, including its units and
exchange-rate limitations; it is not presented as a new pure-risk objective.

The sizing step minimizes tracking variance to `raw_model_weight`. Selected
discretionary rows normally remain between their base and effective-model
weights. Unselected rows remain fixed under the resolved policy. Mandatory pins
and one-sided repair corridors keep their separate meanings, and settlement
cash funds noncash movements inside its applicable bounds.

The semantic identifiers on `ResolvedExecutionProblem` are `contract_version`
(`1.0`), `ranking_objective` (`legacy_funded_alpha_tre`) and
`projection_objective` (`minimum_raw_model_tracking_variance`). These identify
the implemented contract; they are not objective-selection switches.

`build_selected_execution_constraints` constructs the execution-local constraints.
It preserves the legacy filtering of inherited turnover and tracking-error
controls. It also retains the small group-bridge normalization and instruction
rules from the numerical execution stage. Consequently, acceptance refers to
the constructed execution constraints, not automatically to every field on the
unmodified input constraints. Review the returned diagnostics and the
[general constraints contract](constraints.md) together.

`compute_group_bound_bridges` reports interval-capacity diagnostics.
`solve_selected_execution_portfolio` sizes an already selected table, while
When corridor relaxation is explicitly enabled, constraint-directed rescue uses
the buy and sale capacity of each eligible instrument's intersected hard bounds.
An originally model-directed buy can therefore supply a necessary sale in this
flagged retry. Desk pins, cadence and mandatory intervals remain intact; ordinary
strict-corridor rescue retains the original desired direction.

`solve_feasible_execution_portfolio` also performs the configured rescue and
retry process. The typed entry point combines scoring and that feasible-solve
orchestration. Full-investment bridge tightening uses only an explicitly supplied
`partition_groups` partition.

The typed entry point inherits the existing `OptimiserConfig` default, CLARABEL.
The lower-level selected and feasible solver functions retain their legacy
MOSEK default. Supply `OptimiserConfig(solver="CLARABEL")` explicitly for portable
calls to either lower-level function. MOSEK is optional and must already be
available when selected; importing the execution package does not require it.

## Acceptance and operational use

The result contains `weights`, an enriched `trade_table` and the OP solver
`outcome`. Require both `accepted` and `compliant` before treating the result as
an accepted execution proposal. Retain the outcome status, reason, residuals and
fallback source. A returned portfolio alone is not proof that the numerical
solver accepted it or that it satisfies the applicable execution constraints.

`ExecutionSolverInfeasibility` carries bridge diagnostics and the last attempted
trade table for structural corridor failures. Such a failure applies to the
attempted policy intervals and admitted set; it is not a certificate that every
unrestricted portfolio problem is infeasible. Numerical failure and rejection
must remain distinguishable from structural interval failure.

The requested trade allowance governs selection rather than a universal hard
ticket cap. Mandatory and rescue trades can increase the realized count.
Materiality screening is based on desired movement; convex sizing can produce
smaller realized trades. Report requested, mandatory, rescue and realized tickets
separately. This interface does not guarantee hard cardinality, a minimum size
for every realized trade or feasibility for every mandate.

For a read-only engine comparison, resolve the decision once in the consumer,
pass a copied snapshot to each numerical engine, and compare weights, selected
and rescue sets, acceptance, residuals and runtime. Keep the existing production
engine authoritative until the consumer's independent end-to-end parity and
adoption checks pass. The library call does not place orders or write a workbook.

Use `build_risk_model` and `qis.RiskModel` for reported ex-ante tracking error and
qis for realized analytics and holdings backtests. The incremental covariance
calculations inside ranking are algorithmic evaluations, not an alternative
reporting layer.

## Opt-in scoring and support search

`score_execution_candidates(problem, method)` selects from the same resolved
candidate mask. `ExecutionScoreMethod.LEGACY` returns the existing ranking table
without additional columns or changed controls. The other methods require
`minimum_trade_score=None` and `sequential_greedy_selection=False` in the supplied
ranking configuration; a cutoff in old score units cannot be reused implicitly.

| Method | Score and selection |
|---|---|
| `NO_ALPHA` | Remove the alpha term while retaining effective class coefficients and the funded finite tracking-error change. |
| `FULL_RISK` | Rank full desired cash-funded moves by tracking-variance reduction, with uniform coefficients. |
| `PARTIAL_RISK` | Maximize funded variance reduction within each optional coordinate's actual execution interval, including position limits. |
| `SEQUENTIAL_FULL_RISK` | Recompute full-move gains after each virtual funded selection; joint sizing still follows selection. |

Partial scoring includes eligible submaterial rescue rows even though they cannot
enter ordinary requested selection. Each scored optional interval must contain
zero. Coupled cash and group constraints remain in the joint sizing problem;
a positive single-coordinate score does not certify that the move is feasible.
For zero directional curvature, the best endpoint is used unless the whole
interval ties, in which case zero is preferred. Full and partial scores are in
the supplied covariance's variance units, while `NO_ALPHA` retains weighted
tracking-error units. `execution_score_method` and `scored_displacement` identify
the calculation. `selection_score` records the score when selected;
`selection_marginal_tre` is left missing for research methods rather than mixing
variance and tracking-error units. Legacy alpha and class-coefficient columns
retain their input provenance; the chosen research method determines whether
they enter the score.

The optional keyword-only `tie_priority` argument supplies a finite, unique
priority for every instrument; lower values win exact score/efficiency ties.
It applies at each sequential selection step as well. The covariance, table
rows and numerical summation order remain unchanged. The default retains the
existing input-order tie behavior. Legacy scoring rejects an override.

`improve_ranked_execution(problem, method, config)` first obtains an audited
incumbent with the existing sizing and rescue process. It then tries optional
trade removals and one-for-one exchanges. Every trial sizes all admitted trades
jointly and disables rescue, so a removed coordinate cannot be readmitted by a
retry. Mandatory rows, cadence pins and corridor rules retain their meanings.
Failed baselines are returned without a search; structural exceptions retain
their diagnostics. A relaxed-corridor incumbent is unsupported by this search.

`ExecutionSearchConfig` controls the work and acceptance criteria:

| Field | Meaning and default |
|---|---|
| `max_removal_trials` | At most 100 removal projections; zero disables removal. |
| `max_exchange_trials` | At most 100 exchange projections; zero disables exchanges. |
| `te_allowance_bp` | Maximum total tracking-error increase from the initial incumbent during compression, default 1 bp. It is not a fresh allowance for each deletion. |
| `min_exchange_improvement_bp` | Minimum tracking-error reduction versus the current incumbent for an exchange, default 0.0001 bp and strictly positive. |
| `exchange_candidate_limit` | Consider at most the first 20 unselected eligible candidates in score order per pass. |

Both stages require the existing hard audit to pass and actual noncash tickets
not to increase. Actual tickets compare final weights to original pre-trade
holdings, exclude settlement cash, and use the existing weight tolerance of
`1e-8`. Removing an admitted coordinate need not save an actual ticket. Report
both measures rather than treating selected support size as executed count.
Reported risk comes from the existing QIS risk adapter. Basis-point allowances
assume an annualized covariance supplied by the caller.

The `ExecutionSearchResult` contains the final audited `result`, an `attempts`
table and a `summary` with initial/final risk and actual ticket counts, trial
counts and budget flags. The enriched table preserves original ranking requests
in `requested_trade`, accepted baseline membership in `initial_selected_trade`
and `initial_rescue_trade`, and final changes in `improvement_added` and
`improvement_removed`. Trial budgets include structural failures and bound
projection counts, not elapsed time. A caller needing a wall-clock deadline
must supervise the solve in a separate process. These bounded searches provide
no global minimum-ticket or complete-neighbourhood optimality guarantee.

`diagnose_execution_corridors(problem)` opens the full saved rescue-eligible
domain and solves the continuous corridor problem without widening any corridor.
Its `ExecutionCorridorDiagnostic` contains `status`, `bridges`, optional `result`
and `reason`. Status distinguishes `accepted_continuous`, `interval_infeasible`,
`solver_reported_infeasible` and `unresolved_or_rejected`. Acceptance is not a
hard-cardinality or minimum-ticket-size certificate. An interval conflict is
evidence about this declared domain, not about a mandate with different pins.

## Guarded proposals and compression

`solve_guarded_execution(problem, config)` starts from the audited legacy result,
evaluates a fixed bank of alternative rankings and then tries bounded removals.
The original legacy portfolio anchors every tracking-error, ticket and turnover
limit, including after a proposal replaces it. The legacy solve retains its
original configuration. Alternative scores explicitly disable the legacy score
cutoff and saved sequential flag, because their score units and selection method
are declared separately.

`ExecutionGuardConfig` defines this opt-in research policy:

| Field | Meaning and default |
|---|---|
| `methods` | Fixed proposal order, initially partial-risk then sequential full-risk. Unique nonlegacy methods only; an empty tuple disables proposals. |
| `canonical_ties` | True uses instrument string-label order to resolve exact ties; False preserves input order. String labels must be distinct. |
| `te_allowance_bp` | One total allowance of 1 bp above original legacy TE, usable only with fewer registered tickets. |
| `min_te_improvement_bp` | Minimum TE improvement of 0.0001 bp when no registered ticket is saved. |
| `ticket_size_bp` | Count noncash changes strictly above 1 bp NAV as sized tickets. This is a counting threshold, not order rounding. |
| `max_sized_ticket_increase` | Zero additional sized tickets versus the original baseline; None disables this extra guard. |
| `max_turnover_increase_bp` | Zero additional gross noncash turnover in bp NAV versus the original baseline; None disables this extra guard. A comparison tolerance of 0.00000001 bp handles numerical noise. |
| `max_removal_trials` | At most 50 removal projections, including structural failures; zero disables compression. |
| `max_search_seconds` | Optional cooperative time budget after the baseline, checked between solves. None disables it; zero returns the baseline without search. An in-flight solver cannot be interrupted by this control. |
| `max_projection_calls` | Optional ceiling on all post-baseline projection attempts, including proposal rescues and structural failures. None preserves the previous uncapped proposal/rescue behavior. Zero retains the baseline. |

Every eligible candidate must pass the existing hard audit, use strict corridors
and have no more registered noncash tickets than original legacy execution.
It must either improve TE by the minimum amount or save a registered ticket
within the single allowance. Sized-ticket and turnover limits apply to both
routes. Gross noncash turnover sums absolute changes from original holdings;
there is no division by two and settlement cash is excluded.

Qualifying portfolios are compared lexicographically: registered tickets first,
then TE, sized tickets and gross noncash turnover. Exact ties retain the
incumbent. This declares an execution-burden preference, rather than choosing the
lowest observed TE after the experiment. Each removal must save at least one
of the two ticket counts, improve that ordering and pass all original-baseline
guards. Joint sizing and the hard audit run for every projection, with rescue
disabled during compression so deleted coordinates cannot be readmitted.

The returned `ExecutionSearchResult` keeps the accepted final result, every
trial's metrics and reason in `attempts`, and original/final metrics, the selected
method, work counts and timing scope in `summary`. Failed or relaxed legacy
results return without improvement; structural baseline exceptions retain their
existing behavior. Candidate structural failures and audit rejections retain
the incumbent. Unexpected programming or invalid-metric errors propagate.

These thresholds do not impose executable minimum sizes, lot rounding or
transaction costs. A process supervisor is still needed for a hard deadline;
the cooperative timer only stops further work between solves. The method is
an opt-in bounded search, with no global support-optimality guarantee. Validate
it on the consumer's resolved inputs before changing a production engine.

## Compression of independent ranking branches

`solve_branched_execution(problem, config)` keeps a legacy-seeded branch and a
separate branch for every strict, hard-audited proposal. It compresses these
branches before choosing the final eligible portfolio. An exploratory seed may
exceed optional size-count or turnover caps; it is never returned until a saved
state satisfies every original-baseline return guard.

`ExecutionBranchConfig` inherits all fields and return limits from
`ExecutionGuardConfig`. It defaults `max_projection_calls` and
`max_removal_trials` to 52. A finite, nonnegative projection ceiling is required.
Both limits are shared across the complete search. The unchanged production
baseline is outside the ceiling and remains the fallback. Projection attempts
include structural failures and feasibility-rescue projections. The counter
limits attempts through the execution projection entry point, not diagnostic
optimizations or elapsed time.

After creating seeds in the fixed `methods` order, the search schedules one
removal attempt per active branch per round: legacy first, then proposal order.
Within a branch, it tries optional trades from smallest actual weight change to
largest, resolving equal sizes by instrument label. Accepted removal advances
reset that branch's deletion list; rejected attempts move to its next candidate.
Exhausted branches yield their turns. All branches retain the original holdings,
mandatory instructions, cadence and strict corridor rules.

An exploratory removal must save at least one of the two ticket counts, improve
its own branch's registered-ticket/TE/size-count/turnover ordering, and keep TE at
most `te_allowance_bp` above original legacy TE. Rescue is disabled for removals.
Every replacement of the returned incumbent independently passes the complete
original-baseline guard, including the minimum TE gain or strict registered-ticket
saving rule and the operational caps. An internal branch advance is therefore
distinct from an eligible replacement of the returned portfolio. Exact return
preference ties keep the incumbent.

`ExecutionBranchSearchResult` carries the usual `result`, `attempts` and `summary`,
plus `checkpoints`, a mapping from checkpoint identifiers to audited execution
results. It includes the baseline and every retained proposal/removal state.
Internal checkpoints are research states, not execution orders. Trial rows record
the branch, parent checkpoint, saved checkpoint, separate branch/return decisions,
metrics and projection count. The summary identifies the selected checkpoint,
original/final metrics, final branch states, work counts and stop reason.

Failed or relaxed baselines return unchanged, while structural baseline failures
and unexpected errors propagate. Exhausting the shared ceiling keeps the last
eligible incumbent. The cooperative `max_search_seconds` still checks between
solves and cannot interrupt an in-flight solve. Keeping a legacy-seeded branch
with a shared budget does not guarantee the result dominates a separate legacy
compression run that spends its entire budget on that single branch.

Two independent controls extend this search without changing its defaults:

| Field | Meaning and default |
|---|---|
| `allow_support_plateaus` | False. True permits an exact deletion of one optional eligible selected coordinate when both registered and sized trade counts stay unchanged. TE must remain within the original allowance. |
| `preserve_guarded_path` | False. True adds a continuation from the best eligible seed after proposals. It receives the first removal attempt in each round, followed by the legacy and proposal branches. All paths share the same work ceilings. |

### Bounded support swaps

`ExecutionBranchConfig.max_swap_trials` defaults to zero. A positive value enables
one-for-one optional-support exchanges after deletion work, seeded from the best
eligible incumbent. With `reserve_swap_budget=True`, it reserves that many
positions of the existing shared
projection ceiling from deletions. Proposal/rescue work still uses the global
ceiling, so the reservation does not guarantee that many swap solves. It neither
enlarges the ceiling nor recycles cached calls into uncharged work.

| Field | Meaning and default |
|---|---|
| `max_swap_trials` | Zero disables swaps. A nonnegative integer no larger than `max_projection_calls` bounds attempted strict swap projections. |
| `swap_candidate_limit` | Eight incoming names per outgoing name, ordered by conditional funded variance gain. |
| `swap_depth` | One continuation level. A positive integer allows successive swaps through a bounded frontier. |
| `swap_beam_width` | One retained path per level. A positive integer bounds the frontier of provisional strict states. |
| `reserve_swap_budget` | True preserves the original reservation. False completes deletion work first and uses only unused shared projections for swaps. |

With `reserve_swap_budget=False` and `max_search_seconds=None`, the complete
swap-disabled path is preserved before any exchange. Subsequent updates can only
improve its returned preference. If deletion exhausts the shared ceiling there
are no swaps. This is a fixed-work property, not a wall-clock dominance promise;
extra bookkeeping and a different stopping time can change deadline delivery.

The score uses the existing OP funded-risk kernel after releasing an outgoing
trade and adjusting settlement cash. Incoming displacements use the original
strict solver intervals. This is a candidate-ordering heuristic: joint resizing,
cash limits and all other hard constraints still require the existing projection
and audit. Mandatory, cash, cadence-ineligible and non-Rule-4 coordinates cannot
be edited. Rescue is disabled and exactly the proposed selected-support edit
must survive the solve. Every attempted projection, including structural failure,
is charged to the same budget.

When the original turnover cap is enabled, swap sizing appends a signed linear
row through the existing constraint compiler, preserving other linear policies.
On directional intervals this row is exact L1 noncash turnover. Intervals that
straddle original current holdings use a conservative secant upper envelope;
fixed changes are constants and cash is excluded. A positive cap reserves
`1e-4` bp internally for numerical accuracy. This tightens sizing and never
weakens the unchanged independent return guard.

Swap states never increase their parent's actual registered count and remain
within the original risk allowance and original registered/sized/turnover caps.
They may be worse than the eligible incumbent, allowing a short continuation to
cross an intermediate support. They do not reset any reference or qualify as
execution orders. Only the unchanged original-baseline return guard and complete
return preference can replace the incumbent or emit an incumbent callback.

Each level receives a share of the remaining trial allowance; the best bounded
children form the next frontier. Already visited selected supports are skipped.
Optional trace fields record `added` and `swap_depth`; summary fields report
swap trials, candidate pairs, levels, seed, final frontier and deletion-phase
projection limit. Default-off searches preserve prior trace columns and paths.
Under a fixed ceiling, reserved swaps can displace useful deletion work. There
is no dominance or global support-optimality guarantee; qualify configurations
on the consumer's resolved inputs before production adoption.

A plateau changes selected support even when executed trade counts stay flat.
Its internal TE or turnover may worsen within the branch rules; it cannot replace
an already preferred returned portfolio. Exactly one eligible selected coordinate
must disappear, so retained deletions make finite support progress and cannot
readmit deleted instruments. Count increases are not plateau moves.

The additional guarded path skips ordinary deletions that fail original return
qualification. If plateau handling is also enabled, a flat-count step may advance
without a qualifying improvement, but it must satisfy the original registered,
sized-ticket and turnover caps and the original TE allowance. It remains an
internal state until it passes the unchanged final return guard. Its seed is
recorded as `guarded_seed_checkpoint`; retained plateau rows report
`support_plateau`. Exploratory branches remain available for seeds that can
become eligible after compression.

The extra path consumes real projections. Under a fixed shared ceiling it may
reduce exploration elsewhere; these controls guarantee neither dominance over
separate compression nor dominance over the previous branch policy. They do not
memoize apparently identical portfolios or recycle unused work into a larger
budget.

### Reusing identical resolved projections

`ExecutionBranchConfig.reuse_projection_solves` defaults to False. When enabled,
one search can reuse a previously successful, compliant, `optimal` numerical
projection with exactly the same resolved wrapper arguments. The cache key
serializes the complete argument dictionary: ordered covariance, raw benchmark,
current holdings, resolved constraints, optimiser configuration and context.
It compares full serialized bytes and the projection callable. Equal byte blocks
share storage while preserving every original key byte. Metadata differences
may cause conservative misses; inputs that cannot be serialized use a fresh
solve. Keys are never deserialized or persisted, and the cache ends with this
search.

Numerical calls within this scope reuse covariance preparation only when the
filtered numerical matrix, asset order and factorization callable match exactly.
Filtering and constraint alignment still run for each request. The original
factorization and PSD-repair policy supply every prepared result, including its
conditioning diagnostics. The cache shares immutable covariance and factor
arrays; writing to them or enabling writeability raises an error. Copy an array
explicitly when a consumer needs mutable local storage. Other outcome fields,
constraints and weights remain detached. Changed matrices or filtered universes
require new preparation, and failed factorizations are retried.

Each hit still consumes one logical projection attempt. The original
`projection_calls`, `search_projection_calls`, removal limit, scheduling and
return guards remain unchanged; saved work does not buy extra search trials.
The common baseline is always solved outside the cache. Constraint resolution,
execution-table construction, dust cleanup and cash/funding audits still run
for every request. Cached weights, outcomes and constraints are copied before
use, preserving branch-specific metadata and avoiding shared mutable results.
Rejected, inaccurate, nonfinite and exceptional solver outcomes are not cached.

With reuse enabled, trial rows additionally record `numerical_projection_calls`
and `projection_cache_hits`; the summary records
`search_numerical_projection_calls` and `search_projection_cache_hits`.
The summary also records `search_covariance_factorizations` and
`search_covariance_cache_hits` for actual preparation and reused preparation,
respectively. All four search counters exclude the common baseline.
Numerical calls count invocations of the minimum-tracking-error wrapper,
excluding the common baseline and diagnostic optimisations. Structural failures
can consume logical work without invoking that wrapper, so logical attempts need
not equal numerical calls plus hits.

Reuse requires `max_search_seconds=None`: faster solves could otherwise change
which trials fit before the cooperative limit. Exact decision parity must be
validated with a fixed solver/runtime; identical inputs do not guarantee that
an externally nondeterministic solver would return identical fresh solutions.
The cache is bounded by the finite logical projection ceiling, with memory
depending on the distinct resolved inputs and saved outcomes.

### Reusing compatible projection programs

`ExecutionBranchConfig.reuse_projection_programs` defaults to False and requires
`reuse_projection_solves=True`. It can reuse a CVXPY problem whose lower/upper
weight bounds change while its filtered risk geometry and all other policy inputs
match exactly. The current implementation supports the base `Constraints` class,
CLARABEL or MOSEK, and immutable covariance preparation owned by the same search.

The complete existing constraint compiler supplies every row. Lower and upper
boxes become parameters; fixed benchmark/current weights, group/risk/trading
constraints and the builder/solver remain part of exact identity. Changed bound
presence, covariance, asset order or fixed policy creates a new program.
Unsupported or unserializable requests construct a fresh problem. The cache
verifies DPP compliance and that every intended box parameter enters the graph.
[CVXPY's DPP guide](https://www.cvxpy.org/tutorial/dpp/index.html) explains its
cached parameter-to-problem-data compilation.

Each actual numerical request still runs the native solver with warm-start
disabled, clears preceding primal values, and performs fresh original constraint
and funding audits. Fixed graph data have an independent deep copy; reported
weights, constraints and diagnostic context belong to the current request.
Rejected or erroneous solves evict their graph. Whole resolved-projection hits
continue to avoid numerical solving under their existing exact identity.

The summary adds `search_program_builds`, `search_program_cache_hits` and
`search_program_bypasses`. Builds include fresh fallback constructions; a rejected
parameter graph can cost an additional build before the fresh fallback. Counters
exclude the common baseline and its rescue calls. Program storage also ends with
this search and can add memory beyond the projection/preparation cache.
Logical projections, branch order and original return guards retain their
existing meaning. Validate outputs and complete paths under the consumer's fixed
runtime before adopting this mode.

### Delivering eligible incumbents during search

`solve_branched_execution(problem, config, on_incumbent=callback)` accepts an
optional keyword-only callback. Its default is `None`. The synchronous call
`callback(checkpoint_id, result)` receives a deep copy of the strict accepted
baseline, identified by `baseline`, and each subsequent replacement of the
eligible returned incumbent. Those replacements satisfy the original-baseline
return guards. Exploratory branch states and unqualified support plateaus are
never delivered; rejected or relaxed baselines produce no callback.

The consumer can retain or persist each result without mutating the search.
Callback exceptions propagate, including persistence errors. Callback execution
consumes elapsed time but no logical projection allowance; use a short callback
when elapsed time matters. With a deterministic solver and fixed projection
budget, enabling a passive callback preserves the final decision and search
order. With a cooperative wall-clock limit, callback overhead can affect which
trials finish. The existing restriction on combining projection reuse with
`max_search_seconds` still applies.

This hook does not interrupt numerical work or enforce a hard deadline. An
external supervisor can stop a worker and recover its latest completely written,
independently audited checkpoint. The supervisor owns atomic storage, input/run
identity, final audit, time accounting and interruption status. If the baseline
has not completed, no feasible fallback is promised. A saved portfolio may be
available even though search ended by interruption or error; retain both facts.
No market deadline, ten-second production requirement, order transmission or
strict minimum-ticket sizing guarantee is introduced.

## See also

- [Constraints and solver contracts](constraints.md)
- [Minimum tracking error](minimum_tracking_error.md)
- [Rolling backtests](rolling_backtests.md)
- [API reference](api.rst)
- [Execution source](https://github.com/ArturSepp/OptimalPortfolios/tree/main/src/optimalportfolios/execution)
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
