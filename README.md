# optimalportfolios — solver hardening drop-in

Adds unconditional post-solve feasibility validation to **every** optimiser in
`optimalportfolios`, threads the rebalance date through for diagnostics, and
removes the deprecated pandas `future.no_silent_downcasting` option from
`constraints.py`.

## Latest additions — pre-solve input contract + observability

This drop also adds (see `CHANGELOG_input_contract_and_observability.md` for the
symbol-level detail):

1. **Pre-solve input contract** (`validate_solver_inputs` in
   `solver_diagnostics.py`, wired into the TRE wrapper, gated by
   `OptimiserConfig.validate_inputs`, default True). Validates the covariance
   (finite / symmetric / right dimension), flags ill-conditioning (reuses
   `check_covar_conditioning`), checks constraint self-consistency (box caps
   reach full investment, group bounds reachable given the box, benchmark within
   bounds), and notes dropped assets — at entry, before the solve. An opt-in
   `deep_feasibility=True` runs the elastic min-violation LP
   (`diagnose_infeasibility`) as a definitive pre-flight. Findings are emitted as
   one structured `InputContractRecord` per solve (ERROR for hard covariance
   problems; conditioning / benchmark / group-reachability folded into a single
   INFO line) and aggregated by `InputContractSummary` into one line per category
   — so a run firing on every rebalance reports "ill-conditioned on N/M; most
   frequent pair …" rather than N log lines.
2. **Relaxation tally** — the per-rebalance frozen-overhang relaxation moved off
   `warnings.warn` onto the constraints logger at INFO (file, not console spam),
   carrying a structured `RelaxationRecord`; a `RelaxationSummary` handler
   aggregates them into one run-level line.
3. **Relaxation magnitude bound** — `OptimiserConfig.max_constraint_relaxation`
   (default None); a single relaxation above it escalates the log to ERROR.
4. **Environment banner** — `log_environment()` logs python + clarabel / cvxpy /
   numpy / pandas / scipy versions once per run for reproducibility.
5. **`RunDiagnostics` bundle** — `configure_run_logging(attach_summary=True)` now
   returns a bundle holding both the rejection and relaxation summaries;
   `.summary()` combines them and `.check_fallback_gate(...)` delegates (existing
   call sites keep working).

> The 20 other `warnings.warn` sites in `constraints.py` (rare edge / API-misuse
> cases, not per-rebalance spam) are intentionally left as warnings; migrating
> them for channel purity is a separate follow-up.

## How to apply

**Option A — overlay (simplest).** Copy the `optimalportfolios/` tree in this
archive over your package root; it overwrites the 10 modified files and adds the
3 new ones:

```bash
cp -r optimalportfolios/ /path/to/your/repo/
```

**Option B — patch.** From your package root (the dir that contains
`optimalportfolios/optimization/`), apply the unified diff. Note `solver_diagnostics.py`
and the two test files are new, so copy those three from this archive first, then:

```bash
git apply wiring.diff      # the 10 modified files; review first with: git apply --stat wiring.diff
```

> **Reviewing `constraints.py`:** its diff is large (~150 lines) but almost
> entirely whitespace — removing the `with pd.option_context(...)` wrapper
> de-indents its body by one level. Review the real change with `git diff -w`,
> which collapses it to the relaxation-logger lines plus the removed `with`.

## File manifest

```
optimalportfolios/optimization/
  solver_diagnostics.py                         [MOD]  validators + conditioning + infeasibility diagnosis + input contract + run-level summaries/gate/banner
  constraints.py                                [MOD]  relaxation -> structured logger record + magnitude bound; pandas option removed
  config.py                                     [MOD]  diagnose_infeasibility, validate_inputs, max_constraint_relaxation fields
  general/
    quadratic.py                                [MOD]  validate_solution
    max_sharpe.py                               [MOD]  validate_solution (Charnes-Cooper + scipy fallback)
    risk_budgeting.py                           [MOD]  validate_pyrb_solution + validate_scipy_solution + inline guard
    carra_mixture.py                            [MOD]  validate_scipy_solution + inline guard
    max_diversification.py                      [MOD]  validate_scipy_solution
  saa/
    max_return_target_vol.py                    [MOD]  validate_solution x2
    min_variance_target_return.py               [MOD]  validate_solution x2
  taa/
    maximise_alpha_over_tre.py                  [MOD]  validate_solution x2 + pre-solve input contract + relaxation magnitude wiring
    maximise_alpha_with_target_yield.py         [MOD]  validate_solution
  tests/
    solver_diagnostics_test.py                  [MOD]  validators + records + gate + input contract + relaxation tally + banner
    maximise_alpha_over_tre_hardening_test.py   [NEW]  end-to-end integration tests on the wired TRE solver
    frozen_overshoot_relaxation_test.py         [MOD]  relaxation now asserted via caplog + magnitude-bound test
```

## What `solver_diagnostics.py` provides

- `validate_solution(weights, problem_status, constraints, n, *, solver, context, ...)`
  — CVXPY backends. Rejects None / hard-status / non-finite / budget /
  box-or-long-only violations; accepts `optimal_inaccurate` iff feasible.
- `validate_scipy_solution(weights, res, constraints, n, *, solver, context, ...)`
  — scipy backends. Reads `res.success`/`status`/`message`.
- `validate_pyrb_solution(weights, constraints, n, *, solver, context, converged, ...)`
  — pyrb `ConstrainedRiskBudgeting` (validates `.x`).
- `check_covar_conditioning(pd_covar, ...)` — diagnostic only; never modifies Σ.

All three share one feasibility core, so accept/reject logic is identical across
backends. On rejection each returns a feasible fallback (drifted `weights_0` →
benchmark → zeros), logs via `logging.getLogger("optimalportfolios.optimization.solver_diagnostics")`,
and emits `warnings.warn` on hard rejects. Module dependencies: numpy / pandas
only.

## Coverage

17/17 solve sites guarded (9 CVXPY + 1 scipy-Sharpe via `validate_solution`;
3 scipy via `validate_scipy_solution`; 1 pyrb via `validate_pyrb_solution`;
2 non-portfolio/array helpers via inline `res.success` guards).

## Tests

```bash
python -m pytest optimalportfolios/optimization/tests/ -q
# 36 passed   (also clean under -W error::FutureWarning, confirming the pandas fix)
```
