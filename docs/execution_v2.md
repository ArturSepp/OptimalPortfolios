---
myst:
  html_meta:
    description: >-
      Prepared cash-funded execution with hard ticket rules, ranked selection,
      matched conic integer formulations and independently audited delivery.
---

# Prepared execution optimisation

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-21](https://github.com/ArturSepp/OptimalPortfolios/commit/42f38ef9eeb27156b2f8997b701677ed4c1836e4)*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Purpose

The [scientific manuscript and portable benchmark](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/ranked_execution_2026)
examine execution modelling, corridor-policy costs, search and numerical reliability
using retained ETF decisions and frozen synthetic/external-risk studies. The manuscript is a public-facing draft; the input
archive's persistent deposit remains pending. Production policy qualification
and internal research records remain separate.

The namespace `optimalportfolios.execution` consumes a final prepared
portfolio decision. `wrapper_ranked_execution` selects actual executable states
and jointly sizes them. `wrapper_exact_execution` solves the matching CORRIDOR or
FULL conic integer formulation. Both retain an independently audited common
incumbent. The [production interface and migration guide](execution.md) use the same domain.

The caller owns lifecycle policy. `prepare_common_execution` supplies the reusable
compulsory preparation shared by production adapters and comparative experiments.
Preparation may require mixed-integer optimisation and must be charged to every arm. An accepted
common incumbent establishes conditional delivery, not universal feasibility or
an intrinsic feasibility advantage over Exact.

## Relationship to operational execution

Use **Ranked Execution** and **Exact Execution** for the matched methods.
The [operational reference](execution.md) distinguishes the separate legacy
operational MIQP, which has its own alpha, position-cap and repair-on-touch
semantics. The old V1 projection and its fallback are retired from the current API.
Historical V1 comparisons remain reproducible in their original software cohort.

A scientific comparison must fix the same objective, prepared domain, exemptions,
incumbent and audit. Record a supplied proposal's own-policy validity, common-domain
admission and final delivery separately. Rolling comparisons instead include
policy differences and each strategy's path-dependent holdings.

The model-support classifier used by lifecycle preparation is separate from the
actual-ticket zero threshold. Historical group-endpoint normalization is not
permission to alter a prepared domain. Preserve frozen experiment arrays and
identify changed preparation as a new instance. Software qualification alone does
not establish speed, certified optimality or investment-performance advantage.

## Inputs and units

### Production preparation and dispatch

`RankedExecutionPolicyConfig` is an explicit opt-in policy. `resolve_production_problem`
converts a lifecycle-resolved trade table, annual covariance and execution constraints
to the immutable domain below. It reuses OP's instrument-bound compiler and preserves
the original group rows. `solve_ranked_execution_portfolio` then calls common preparation,
the same `wrapper_ranked_execution` used by the manuscript, and independent acceptance.
It returns the standard production weights, constraint residuals and trade audit table.
Rosaa supplies workbook rules and histories; QIS supplies drift, backtesting and reporting.

Preparation first seeks a floor-compliant funded book inside the corridor. If necessary,
it searches the eligible hard domain, minimising corridor excursion before gross noncash
intervention and additional repair tickets. All objectives are risk-neutral. Among
equally admissible minimum-count supports, repair identities are selected in
lexicographic order of the supplied instrument names, independent of column order.
Any remaining zero/held choices prefer the supplied base's holding state, again in identity
order. These tie-breaks retain all preceding objective caps and require an optimal
native result within the same shared preparation deadline; an unproven choice fails
closed. A strictly convex, covariance-free tie break then sizes the selected repairs.
Renaming identifiers can change a tied choice, so consumers must keep them stable.
If a native repair-identity stage reports infeasibility after accepting a discrete
incumbent, preparation can re-prove the repair count and canonical identities
using the existing SciPy/HiGHS route with presolve disabled and tighter native
precision. This bounded numerical retry retains the exact earlier excursion and
turnover caps, instrument and group bounds, no-repair intervals and shared deadline.
It cannot enlarge the policy ticket cap or weaken the final audit. The discarded
native count is re-optimised, not accepted as a feasible integer certificate;
both attempts are retained in diagnostics. A failed or unproved retry remains a
failed preparation.
If an optional base lies just outside a legal interval or holding-state endpoint,
the no-repair branch may use the nearest legal endpoint only when its distance
from both base and original holdings is within the existing zero-trade convention.
If the entire legal corridor is no wider than the existing zero-trade convention
and every endpoint lies within that convention of both holdings, the no-repair
branch may retain that numerical interval. It cannot open the wider material
repair interval. The portfolio must still satisfy the coupled constrained solve
and unchanged return audit, and any numerical movement is pinned before risk search.
This prevents tiny necessary endpoint corrections from opening material repair
choices or inflating the repair-count objective.
New compulsory endpoints
are pinned and declared before risk search; original mandates and calendar restrictions
remain in force. A fixed total cap is never enlarged. An explicit discretionary allowance
remains additional to the final mandatory reservation. The audit tolerance stays unchanged.
Native boundary rounding can trigger one constrained endpoint re-solve, never clipping of
delivered free weights. Genuine hard conflicts return a recorded preparation failure.

| Policy fields | Meaning in Ranked Execution |
|---|---|
| `small_trade_cutoff`, `small_trade_cutoff_by_asset_class` | Resolved nonzero holding floors, with cash excluded. |
| `material_trade_threshold` | Actual non-exempt ticket minimum measured from the original current book. |
| `max_trades`, `add_mandatory_trades_to_max_trades` | Hard total cap, or discretionary allowance plus mandatory reservation. |
| `ranked_config` | OP-owned bounded state-search and QP controls. |
| `preparation_time_limit_s`, `preparation_solver` | One shared preparation deadline and integer backend. |
| `constraint_set` | Consumer selection of workbook execution group bands. |
| `mandatory_trades`, `instruction_correction_tolerance` | Consumer-resolved instructions and their pre-existing correction convention. |
| `sell_down_dust_threshold` | Retained lifecycle setting; it cannot weaken the terminal holding floor. |
| `use_weight_drift`, `output_by_ticker` | Consumer replay and display settings. |
| `expand_for_feasibility` | Must be enabled for common preparation. |
| `minimum_trade_score`, `sequential_greedy_selection`, `sign_directed_rescue`, `allow_corridor_relaxation` | Incompatible V1 search controls are rejected. |
| `tre_weight_by_asset_class` | Must be neutral: Ranked minimises full-covariance risk to the raw model. |
| `exact_execution` | Must be absent; operational MIQP remains a distinct method. |

The existing ranked execution scope excludes inherited model turnover and tracking-error
controls. Other hard rules outside the supported long-only funding, instrument and group
domain raise an error; they are not discarded. Standalone Ranked has no V1 fallback;
an absent audited delivery remains an explicit failure. The result records the engine used. An audited retained common base is a feasible incumbent, not a certificate
of optimality or proof that additional discretionary trading was impossible.

### Prepared arrays

`ExecutionProblem` copies arrays into immutable buffers. `names` and `cash`
identify instruments and settlement cash. `current` is the original book,
`base` the prepared common incumbent and `target` the raw model. All weights and
`investment` are fractions of NAV. `annualisation` is explicitly `annual`.
`preparation_seconds` charges the caller's shared work and `provenance` records
its immutable source identities, policy and currency conventions.

`lower` and `upper` describe the prepared corridors. `full_lower` and
`full_upper` describe the containing eligible hard domain, retaining all pins
and mandatory intervals. `eligible` and `frozen` describe the final policy
restrictions. `mandatory` and `categories` are declared before optimisation.
`groups`, `group_lower` and `group_upper` retain fractional exposures, including
balanced instruments, without treating them as disjoint partitions.

`covariance` is the authoritative annual covariance. Alternatively,
`factor_loadings`, `factor_covariance` and `residual_variance` supply an exact
factor plus diagonal representation. Supplying both requires reconstruction
agreement. Correlated empirical residuals cannot be substituted by a diagonal.
No covariance jitter or risk-model refitting occurs in this namespace.

`rules` is an `ExecutionRules` object. `minimum_ticket` is measured from original
current holdings. `holding_floor` applies to every nonzero final noncash holding.
`exemption_mask` removes only minimum ticket size for explicitly mandatory names.
`mandatory_count_in_cap` must be true. `ticket_cap` is the total unique noncash
ticket limit. An optional `discretionary_allowance` requires the total to equal
the mandatory reservation plus that allowance. `None` denotes unlimited.
`zero_tolerance` defaults to 1e-8 and `audit_tolerance` to 1e-7 in weight units.
An optional no-ticket branch retains the exact supplied base when its difference
from current is within the zero tolerance.

Native group rows use `group_lower - audit_tolerance` and
`group_upper + audit_tolerance`, rounded one floating-point step toward the
original limits. This explicitly represents the existing per-row exposure
allowance in NAV units. Fixed-state sizing, both Exact formulations and continuous
relaxations use the same envelope on every group row, including already feasible
ones; small sizing changes can therefore occur outside conflicting cases.
It is neither a relative tolerance nor a soft
penalty, and does not expand instrument bounds, funding, states, floors or caps.
The original input arrays remain immutable. The independent return audit still
checks them and rejects every group residual beyond the configured tolerance.
Inward endpoint rounding cannot guarantee that native solutions remain inside
that limit after original-space arithmetic. Exact can independently resize one
group-rejected candidate's fixed states within its remaining budget, as described
below. A rejected final return may still leave the supplied incumbent in place
even after a native optimal-status report.
Result diagnostics identify this policy under `group_feasibility`.

For example, a group pinned at 0.05 minus 6e-10 can be sized under an original
0.05 group minimum and the default 1e-7 audit. A shortfall of 1.01e-7 remains
infeasible and fails the audit. This numerical execution domain differs from the
literal mathematical domain; retain both interpretations in experiment records.

`from_resolved_problem` adapts resolved numerical inputs with an explicit `prepared`
declaration and containing full bounds. It never rebuilds lifecycle decisions.
Unsupported extra hard constraints cause rejection. The caller must retain all
original group bands and eligibility restrictions in its qualified declaration.

## Selection, sizing and delivery

`compile_states` intersects endpoint bounds with the holding disjunction and
current-relative minimum ticket. Alternative states share one instrument ticket.
An exit is available only when zero belongs to the declared interval.
`audit_execution` checks original weights without relying on the compiler.

`RankedConfig` defaults to one `drop` and one `exchange` pass, followed by bounded
coordinated improvement when `exchange` is enabled. Each pass considers one
or two instrument-state changes around the incumbent after the earlier passes,
including admissions into unused ticket slots. A two-dimensional cash-funded
quadratic ranks pairs even when neither individual trade has a positive gain.
Full joint sizing and the unchanged audit decide whether a trial can be retained.

`rescue_trials` caps initial support rescue and, separately, the coordinated
trials (24 supports for each stage by default). Previously visited supports do not
consume the new shortlist. `improvement_passes` permits up to three anchors by
default, sharing that total trial allowance. If a pass improves the incumbent
and trials remain, the next pass rebuilds priorities around the new sized weights.
A pass without improvement stops search; this does not certify local optimality.
When available optional candidates exceed the positive allowance, an additional
scarcity stage retains the completed original search and tries an alternative
support rounded from the existing continuous perspective model. Fractional
occupancies rank names and states; fractional weights are never delivered. The
rounded support must pass the unchanged fixed-state sizing and original-unit audit.
An audited but initially inferior seed may be searched independently, with the
original incumbent restored unless the completed branch strictly improves it.

At each of up to `improvement_passes` scarcity anchors, all one-for-one optional
replacements are jointly sized. This matters when retained trades must resize:
a score that freezes them can miss the useful replacement. The extended shortlist
also permits two admissions coupled to one or two removals, so a funded pair can
replace another pair at a full ticket budget. Cash/group-constrained two-coordinate
priorities use polygon edge and interior minima. If that slice is empty, the box
priority is retained because other selected coordinates can still move in joint sizing.
This is search guidance, never a constraint or acceptance change. The scarcity stage
has a separate `rescue_trials` allowance for its rounded seed and shortlisted supports;
complete replacement sweeps are counted separately in `qp_counts.exchange_sweep`.
An improvement can trigger an early restart in that stage. Failure or exhaustion
cannot certify optimality. `mandatory_state_trials` bounds
compulsory holding alternatives. `objective_tolerance` controls strictly improving acceptance.
`qp_solver` defaults to CLARABEL. `compute_bound` optionally requests separately
timed relaxation certificates. State truncation and exhausted search are reported
explicitly. They never establish infeasibility or authorise a wider corridor.

The scarcity stage then runs complete replacement sweeps from its completed
incumbent and unrestricted guided seed. It also tries up to `rescue_trials`
selected names in identifier order, pinning one name at its no-ticket state only
in a temporary continuous guidance model. Each distinct rounded support gets up
to `improvement_passes` sweeps. If the seed cannot be sized, the first sweep still
tries its one-for-one neighbours. Every executable candidate uses the original
problem; the excluded name can re-enter. No Exact incumbent guides this search.
Each branch retains the earlier completed incumbent unless it strictly improves
it. A per-call support cache reuses only original-domain sizing outcomes over
the unchanged mandatory assignments. Separate branch visitation permits cached
audited supports as intermediate points. Failed guidance or sizing cannot erase
an audited incumbent. `restarts`, `qp_counts.restart`, `qp_counts.restart_guidance`
and `sizing_cache_hits` expose this work and its failure paths.

An additional bounded phase conditions on both inclusion and exclusion of ticket
states in the union of the completed incumbent and unrestricted relaxed seed.
This supplies alternatives even when no optional incumbent exists and the rounded
seed is infeasible. Decisions use stable instrument/state order, inclusion before
exclusion, with at most `rescue_trials` additional continuous solves. A decision
adds a selector equality only to that guidance graph; its objective, original
rows and numerical tolerances remain unchanged. Rounding respects the decision.
The temporary condition does not pass to executable sizing or its audit.

Each distinct seed receives the existing replacement restart, with at most
`improvement_passes` sweeps. A final restart revisits the completed incumbent.
Earlier audited results remain retained, and the same per-call sizing cache
avoids repeated QPs. `conditional_guidance`, `conditional_restarts` and
`qp_counts.conditional_guidance` report every attempt and its outcome. This is a
finite heuristic neighbourhood, not recursive branch-and-bound or an optimality
certificate. Its extra work must be included in comparative timings.

A subsequent scarce-ticket phase examines up to `rescue_trials ** 2` additional
coordinated supports (576 by default), shared across at most `improvement_passes`
anchors. It retains the complete single-replacement sweep at each anchor and
skips all cached supports. A conservative interval screen rejects only supports
that cannot satisfy funding or a group row. Selected coordinates and cash span
their original intervals; mandatory coordinates span all their possible states.
Signed group coefficients use the appropriate lower or upper endpoint. Existing
coordinate and row audit allowances make this an outer screen; survivors still
require the unchanged sizing model and audit. No executable constraint is widened.

This wider search addresses a limitation of slice priorities: they hold retained
positions fixed, while joint sizing can resize them. A low priority therefore
cannot prove that a pair replacement is unhelpful. `completion_passes`,
`completion_screened` and `completion_trial_limit` expose the additional work.
The finite budget can still truncate the neighbourhood; it does not certify
global optimality. The one-anchor ablation can also stop before a useful sequence
of pair admissions and replacements is completed.

Search then revisits up to `rescue_trials` audited alternatives already in the
sizing cache, ordered by objective and instrument/state identifiers. Each seeds
a separate bounded replacement branch, allowing an inferior intermediate support
without discarding the incumbent. At the first strict improvement, one additional
pair-completion phase uses the same squared trial budget and anchor limit, then
search stops. `bridge_restarts` records seed, incumbent and completed objective
values; `bridge_completion_passes` records the extra completion work. No reference
portfolio enters this search. This can reach a three-position replacement missed
by improving-only descent, but remains a finite heuristic.

In a fixed-state sizing QP, substitute the fixed coordinates and funding first,
then omit a bound/group interval only if all remaining coefficients are exactly
zero and its constant value satisfies both unchanged endpoints. Keep every
nonconstant row, including arbitrarily small coefficients, and every constant
contradiction. No tolerance is used to classify a coefficient as zero.
Removing these satisfied constant rows avoids interior-point degeneracy without
changing the feasible set, objective or original-space audit. Native integer
hulls and continuous relaxations retain their original graph.

After a CLARABEL numerical `SolverError`, sizing makes at most one fresh native
attempt with automatic equilibration disabled. The graph, objective, hard rows
and absolute/relative gap and feasibility tolerances (1e-10) remain identical.
Reported infeasibility does not trigger a retry; other backends retain their
existing error handling. Every returned vector still requires the original-space
audit. Each trial's `sizing_solver` diagnostics record `attempts`,
`equilibration_retry` and any `initial_error`. QP counts refer to candidate sizing
problems; sum their attempt counts to include numerical retries. Sizing/search
timings include both attempts.

The dense selection stage uses one covariance multiplication and funded column
updates, costing O(n squared + K n). Exact structured inputs use factor products.
Joint sizing, mandatory state enumeration, rescue, improvement passes and bound
computation are separate work and appear in diagnostic counts and timings.
With s alternative state changes, coordinated scoring enumerates O(s squared)
pairs and retains only the configured number of best priorities. Dense pair
curvatures use constant work; exact factor inputs use factor products. The
shortlist bounds its QP calls, not total scoring work. With B selected names, the
scarcity extension can enumerate O(B squared s squared) pair replacements per anchor,
and each complete single-replacement sweep tries at most B s supports. It also solves
one unrestricted and at most min(B, `rescue_trials`) restricted continuous
perspective models. Up to 2 + min(B, `rescue_trials`) distinct starting branches
each add at most `improvement_passes` sweeps of B s supports; cached supports
avoid repeated sizing. These costs are additional to the original search;
the two shortlists each have their own 24-support default allowance. Diagnostics
also report the screened completion phase, whose coordinated budget is 576 supports
by default plus at most three complete single-replacement sweeps. Diagnostics
also retain up to 24 cached-alternative restart branches, each with at most three
replacement sweeps, and at most one further completion phase after improvement.
Every restart and completion call contributes to the reported total runtime.
Diagnostics separately report `coordinated_passes`, `scarce_passes`, seed trials, guidance status,
candidate counts, truncation and all sizing counts. No such extension runs when
the optional candidate count does not exceed the allowance.

`ExecutionResult` contains immutable accepted `weights`, delivery `status` and
`diagnostics`. Diagnostics record audits, unique mandatory/discretionary tickets,
exemption identity, cap convention, gross noncash turnover, objective variance,
annual tracking error in basis points, candidate count and all solve counts.
An unaudited vector is never delivered and native output is never silently repaired.

## Exact starts and certificates

The Exact wrapper accepts `formulation`, `perspective`, `mip_start`,
`time_limit_s`, `threads`, `solver` and `gap_rel`. The default backend is MOSEK.
The optional `optimalportfolios[exact]` extra supplies its Python interface.
Running MOSEK also requires an appropriate solver licence.

The qualified native adapter sets integer feasibility to `1e-9`. The earlier
`1e-4` setting could terminate with a false optimal status and a bound above a
known feasible portfolio on an overlapping-group instance. This tighter internal
setting changes neither the compiled mathematical constraints nor the original-
unit audit. That change alone left smaller inconsistencies. Integer models now
write the quadratic cone in percentage-point displacement coordinates, an
algebraically identical quadratic with better scaling. Linear rows and native
start mapping still use basis points. Fixed-state sizing and continuous
relaxation graphs are unchanged. Integer distance, relative/absolute MIP gap and
conic controls remain unchanged. The full-weight fixed-support regression
verifies the feasible witness independently of the funded-coordinate graph.

`native_bound_check` records consistency with the delivered incumbent using the
existing default objective comparison scale (`1e-12` variance). It preserves the
raw native bound and status; it never clips a bound to an incumbent or promotes
a consistent diagnostic to an optimality certificate. Cross-arm comparisons
also check native bounds against every independently audited delivered witness.
Missing optional solvers are reported while retaining the audited common incumbent.
The qualified native start adapter is specific to CVXPY 1.9.2. It fixes original
weights and selectors in a continuous conic lift, supplies every native variable,
then queries actual solver receipt. CVXPY warm-start flags alone do not establish
receipt. Canonicalisation and lift costs consume the declared solve budget.

After native execution, Exact may make one fixed-state convex refinement of the
best raw candidate rejected **only** by the group audit. It proceeds only when
that candidate's objective improves the retained audited incumbent. For each
noncash instrument it selects the nearest admissible state interval, requiring
distance within the existing audit tolerance; no alternative states are searched.
The fresh QP retains the original objective, group envelope, instrument and cash
bounds, funding, minima, holding floors and caps. It neither rounds weights nor
widens the audit. A genuinely infeasible fixed-state problem stays infeasible.

Graph construction and this single CLARABEL attempt consume the remaining Exact
budget. CLARABEL receives the remaining time and the existing sizing tolerances
of 1e-10; there is no retry or new time allowance. If time expires, the state
cannot be matched, the solve fails or its vector fails the original-space audit,
the previous audited incumbent remains available. In-progress solver work may
overrun the deadline. This conditional sizing step does not establish optimality
over other states or repair the mathematical literal-domain contradictions.

`boundary_refinement` diagnostics preserve the raw rejected candidate, rejection
audit, selected states, status, elapsed time and output audit when available.
`delivered_source` attributes a winning refinement explicitly. The original
native statuses and events remain unchanged, and refined portfolios do not count
as native integer incumbents or first native audited feasibility. Comparisons
must report refinement, native success and retained starts separately.

An audited supplied start is retained externally even if native lifting or
receipt fails. Equality between Ranked and Ranked-started Exact can therefore reflect the
same fallback, not successful optimisation. Comparative studies should report
native outcomes separately and declare all independent, pooled and common-base
starts under equal native budgets. Historical V1 starts retain their original cohort. Charge the generation costs
of every algorithm used to choose a start; do not label pooled incumbents as a
single native run or infer global optimality from their delivery.

Perspective terms use one funded displacement per movable instrument, with
closed rotated cones. The complete constant and affine objective terms remain.
The split is verified on the declared funded domain without modifying covariance.

`compute_perspective_lower_bound` solves plain and perspective relaxations of the
same CORRIDOR model. It reports primal values separately from lower bounds. A
supporting-plane LP dual includes an explicit bounded-variable residual correction
and floating-point guard. If unavailable, the certificate is explicitly limited
to nonnegative variance. Native integer bounds remain diagnostics until their
certificate contract is validated. No conic primal value is labelled a lower bound.

## Qualification and limits

The tests compare fixed-state QPs to independent active-set KKT enumeration,
exercise real optional native starts, and check independent return gates.
Coordinated-search regressions include a feasible zero-variance pair that scalar
ranking misses, fixed-cash funding, the trial cap, rejected solver output and
independent convex-QP verification of pair priorities. Private
data qualification and main computational experiments remain distinct stages.
Licence-free native protocol tests check start submission versus receipt,
callback inversion, charged deadlines and failure retention in ordinary CI.
They complement the separately licensed numerical formulation tests.
Algorithm timings on one host do not define a portable correctness deadline.
The production adapter and research solver share preparation and acceptance rules.
Qualification includes mandate-specific own-book rolling replay, separately
assembled constraint checks, prefix replay checks and independent fixed-state
QP references. A bounded heuristic can still miss an improving support; neither
these checks nor an accepted portfolio establish global optimality or universal
tracking-error improvement. Native-bound certification remains separate work.
The current public namespace uses the descriptive APIs in the migration guide;
consumers choose an explicit engine or execution policy.
Frozen paper bundles retain their own source and wheel, so replication does not
depend on the current checkout or a sibling development worktree.

## Exact primary and audited Ranked fallback

`MatchedExecutionPolicyConfig`, `wrapper_execution_policy` and
`solve_execution_portfolio` provide an explicit `exact_primary_ranked_fallback` policy.
The production adapter resolves workbook inputs and prepares the common domain
once. Exact Execution uses that Corridor domain first; Ranked is invoked lazily on the
same prepared problem when the primary cannot supply an acceptable native
candidate. This policy never invokes V1. Standalone Ranked and legacy operational
MIQP retain their separate interfaces and acceptance behavior.

The controls `primary_time_limit_s`, `primary_threads` and `primary_gap_rel`
default to 10 seconds, one thread and 1e-4 respectively. The relative gap is
measured in original tracking variance, including the canonical objective
constant. A time limit does not by itself reject a valid audited native
incumbent or certify optimality. Missing native witnesses, solver and callback
errors, inconsistent bounds and independently invalid weights trigger fallback.
A Ranked fallback that fails the unchanged audit is rejected; genuine compulsory
repairs and infeasible mandates remain visible. The retired `allow_v1_fallback`
keyword is not accepted by either new policy configuration.

Diagnostics retain the primary status, native gap and bound, independent audits,
search time, delivered engine and fallback reason. A successful primary delivery
has no fallback attribution. Both the compiled-domain and original hard
constraint audits must pass before a production portfolio is returned.
