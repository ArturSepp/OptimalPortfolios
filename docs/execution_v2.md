---
myst:
  html_meta:
    description: >-
      Prepared cash-funded execution with hard ticket rules, ranked selection,
      matched conic integer formulations and independently audited delivery.
---

# Prepared execution optimisation

*Author: [Artur Sepp](https://github.com/ArturSepp)*

This guide describes [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Purpose

The research namespace `optimalportfolios.execution.v2` consumes a final prepared
portfolio decision. `wrapper_ranked_execution` selects actual executable states
and jointly sizes them. `wrapper_exact_execution` solves the matching CORRIDOR or
FULL conic integer formulation. Both retain an independently audited common
incumbent. The existing [production interface](execution.md) remains unchanged.

The caller owns lifecycle policy and compulsory preparation. Preparation may
require mixed-integer optimisation and must be charged to every arm. An accepted
common incumbent establishes conditional delivery, not universal feasibility or
an intrinsic feasibility advantage over Exact.

## Inputs and units

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

`from_resolved_problem` adapts saved v1 inputs with an explicit `prepared`
declaration and containing full bounds. It never rebuilds lifecycle decisions.
Unsupported extra hard constraints cause rejection. The caller must retain all
original group bands and eligibility restrictions in its qualified declaration.

## Selection, sizing and delivery

`compile_states` intersects endpoint bounds with the holding disjunction and
current-relative minimum ticket. Alternative states share one instrument ticket.
An exit is available only when zero belongs to the declared interval.
`audit_execution` checks original weights without relying on the compiler.

`RankedConfig` defaults to one `drop` and one `exchange` pass. `rescue_trials`
bounds support rescue and `mandatory_state_trials` bounds compulsory holding
alternatives. `objective_tolerance` controls strictly improving acceptance.
`qp_solver` defaults to CLARABEL. `compute_bound` optionally requests separately
timed relaxation certificates. State truncation and exhausted search are reported
explicitly. They never establish infeasibility or authorise a wider corridor.

The dense selection stage uses one covariance multiplication and funded column
updates, costing O(n squared + K n). Exact structured inputs use factor products.
Joint sizing, mandatory state enumeration, rescue, improvement passes and bound
computation are separate work and appear in diagnostic counts and timings.

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
Missing optional solvers are reported while retaining the audited common incumbent.
The qualified native start adapter is specific to CVXPY 1.9.2. It fixes original
weights and selectors in a continuous conic lift, supplies every native variable,
then queries actual solver receipt. CVXPY warm-start flags alone do not establish
receipt. Canonicalisation and lift costs consume the declared solve budget.

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
exercise real optional native starts, and check independent return gates. Private
data qualification and main computational experiments remain distinct stages.
Licence-free native protocol tests check start submission versus receipt,
callback inversion, charged deadlines and failure retention in ordinary CI.
They complement the separately licensed numerical formulation tests.
Algorithm timings on one host do not define a portable correctness deadline.
The implementation has passed its research qualification gate. Full comparative
experiments and native-bound certification remain separate work. Importing this
namespace does not replace the existing production entry points or defaults.
Frozen paper bundles retain their own source and wheel, so replication does not
depend on the current checkout or a sibling development worktree.
