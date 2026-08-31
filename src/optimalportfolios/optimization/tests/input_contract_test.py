"""
the pre-solve input contract and the post-rejection infeasibility diagnosis.

Two layers of ``solver_diagnostics`` sit either side of a solve and neither is reached by a
solve that works:

* ``validate_solver_inputs`` runs *before* the solve and answers "can this problem be
  satisfied at all?" — box caps that cannot reach full investment, floors that overshoot it,
  group bounds unreachable given the box, a benchmark outside its own constraints, a
  covariance that is not finite / square / symmetric.
* ``diagnose_infeasibility`` runs *after* a rejection and answers "which constraints must
  give, and by how much?" via an elastic LP, turning the word "infeasible" into a list of
  bounds and required relaxations.

Both exist to convert a silent bad rebalance into a legible one, so a bug in them costs
exactly the explanation they were added to provide — and nothing else fails. Every case here
constructs a constraint set that is deliberately impossible in one specific way and asserts
the contract names *that* way, not merely that something was reported.

``evaluate_constraint_residuals`` is covered alongside them: it is the audit that decides
whether a returned weight vector is compliant, and it distinguishes hard constraints from
soft utility terms that are reported but do not gate.
"""
# packages
import logging
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
from optimalportfolios import Constraints
from optimalportfolios.optimization.constraints import (
    BenchmarkDeviationConstraints, ConstraintEnforcementType, GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint, GroupTurnoverConstraint)
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.solver_diagnostics import (
    check_covar_conditioning,
    diagnose_infeasibility,
    diagnose_solver_failure,
    OptimizationOutcome,
    evaluate_constraint_residuals,
    validate_solution,
    validate_solver_inputs,
)

TICKERS = ['growth', 'balanced', 'defensive']
VOLS = np.array([0.22, 0.14, 0.06])
CORR = np.array([[1.00, 0.45, 0.15],
                 [0.45, 1.00, 0.25],
                 [0.15, 0.25, 1.00]])
COVAR = np.outer(VOLS, VOLS) * CORR
COVAR_DF = pd.DataFrame(COVAR, index=TICKERS, columns=TICKERS)


def make_constraints(**overrides) -> Constraints:
    """A satisfiable long-only, fully invested constraint set, with overrides."""
    kwargs = dict(is_long_only=True,
                  min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS))
    kwargs.update(overrides)
    return Constraints(**kwargs)


def issues_text(result) -> str:
    """All reported issues joined, for substring assertions."""
    return ' | '.join(result.issues).lower()


# --------------------------------------------------------------------------- #
# the input contract: covariance integrity
# --------------------------------------------------------------------------- #
def test_a_satisfiable_problem_passes_the_contract_cleanly() -> None:
    """the happy path reports ok with no issues, which is what makes issues meaningful"""
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=make_constraints(),
                                    context='clean')
    assert result.ok is True
    assert result.issues == []
    assert result.n_assets == len(TICKERS)
    assert result.n_dropped == 0


def test_a_non_finite_covariance_is_a_hard_failure() -> None:
    """a NaN in the covariance makes the solve produce garbage, so ok goes False"""
    covar = COVAR_DF.copy()
    covar.iloc[0, 0] = np.nan
    result = validate_solver_inputs(pd_covar=covar, constraints=make_constraints())
    assert result.ok is False
    assert 'finite' in issues_text(result) or 'nan' in issues_text(result)


def test_an_absent_covariance_is_a_hard_failure() -> None:
    """no covariance at all is reported as such, and stops the contract there

    Every later check reads the matrix. Without the early return they would each raise on
    ``None`` and the caller would see an ``AttributeError`` instead of the one fact that
    matters.
    """
    result = validate_solver_inputs(pd_covar=None, constraints=make_constraints())
    assert result.ok is False
    assert 'covariance is none' in issues_text(result)


def test_a_non_square_covariance_is_a_hard_failure() -> None:
    """a rectangular matrix is not a covariance, whatever its entries are"""
    rectangular = pd.DataFrame(np.ones((3, 2)) * 0.04, index=TICKERS, columns=TICKERS[:2])
    result = validate_solver_inputs(pd_covar=rectangular, constraints=make_constraints())
    assert result.ok is False
    assert 'not square' in issues_text(result)


def test_a_covariance_of_the_wrong_size_is_a_hard_failure() -> None:
    """a covariance that does not match the constraints cannot be solved against them"""
    smaller = COVAR_DF.iloc[:2, :2]
    result = validate_solver_inputs(pd_covar=smaller, constraints=make_constraints())
    assert result.ok is False


def test_a_nested_only_constraint_index_sets_the_expected_covariance_size() -> None:
    """nested loadings determine asset order even without top-level indexed fields"""
    loadings = pd.DataFrame({'all': 1.0}, index=TICKERS)
    constraints = Constraints(
        is_long_only=True,
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=pd.Series({'all': 0.0}),
            group_max_allocation=pd.Series({'all': 1.0}),
        ),
    )

    result = validate_solver_inputs(
        pd_covar=COVAR_DF.iloc[:2, :2], constraints=constraints)

    assert result.ok is False
    assert 'covariance dim 2 != n_constraints 3' in issues_text(result)


def test_an_asymmetric_covariance_is_reported() -> None:
    """asymmetry means the geometry the solver enforces is not the one supplied"""
    covar = COVAR_DF.copy()
    covar.iloc[0, 1] = covar.iloc[1, 0] + 0.05
    result = validate_solver_inputs(pd_covar=covar, constraints=make_constraints())
    assert 'symmetr' in issues_text(result)


def test_dropped_assets_are_recorded_without_failing_the_contract() -> None:
    """assets removed for NaN or zero variance are counted, not treated as an error

    The count travels on the result so a caller can tally it across a run; it is
    deliberately not an "issue", because dropping a stale asset is normal.
    """
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=make_constraints(),
                                    n_dropped=4)
    assert result.n_dropped == 4
    assert result.ok is True
    assert result.issues == []


# --------------------------------------------------------------------------- #
# the input contract: structural feasibility
# --------------------------------------------------------------------------- #
def test_box_caps_that_cannot_reach_full_investment_are_reported() -> None:
    """caps summing below the budget make full investment impossible before any solve"""
    constraints = make_constraints(max_weights=pd.Series(0.2, index=TICKERS))  # sums to 0.6
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints)
    assert issues_text(result) != ''
    # structural infeasibility degrades the solve rather than crashing it, so ok stays True
    assert result.ok is True


def test_box_floors_that_overshoot_the_budget_are_reported() -> None:
    """floors summing above the budget are equally impossible, in the other direction"""
    constraints = make_constraints(min_weights=pd.Series(0.5, index=TICKERS))  # sums to 1.5
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints)
    assert issues_text(result) != ''


def test_a_benchmark_outside_its_own_box_is_reported() -> None:
    """a benchmark the constraints forbid makes every active quantity meaningless"""
    constraints = make_constraints(
        max_weights=pd.Series([0.4, 0.4, 0.4], index=TICKERS),
        benchmark_weights=pd.Series([0.9, 0.05, 0.05], index=TICKERS))
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints)
    assert 'benchmark' in issues_text(result)


def test_an_unreachable_group_bound_is_rejected_at_construction() -> None:
    """a group floor the box cannot supply never reaches the solver at all

    This one is caught earlier than the input contract: ``Constraints.__post_init__``
    validates group bounds against the box and raises, naming the group, the two numbers and
    both ways to fix it. The contract is the second line of defence, not the first.
    """
    group_loadings = pd.DataFrame({'risky': [1.0, 0.0, 0.0], 'safe': [0.0, 1.0, 1.0]},
                                  index=TICKERS)
    with pytest.raises(ValueError, match="Group 'risky'"):
        make_constraints(
            max_weights=pd.Series([0.1, 1.0, 1.0], index=TICKERS),
            group_lower_upper_constraints=GroupLowerUpperConstraints(
                group_loadings=group_loadings,
                group_min_allocation=pd.Series({'risky': 0.5, 'safe': 0.0}),
                group_max_allocation=pd.Series({'risky': 1.0, 'safe': 1.0})))


def test_a_reachable_group_bound_constructs_and_passes_the_contract() -> None:
    """the mirror case, so the check above is not passing for the wrong reason"""
    group_loadings = pd.DataFrame({'risky': [1.0, 0.0, 0.0], 'safe': [0.0, 1.0, 1.0]},
                                  index=TICKERS)
    constraints = make_constraints(
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=group_loadings,
            group_min_allocation=pd.Series({'risky': 0.1, 'safe': 0.1}),
            group_max_allocation=pd.Series({'risky': 0.6, 'safe': 0.9})))
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints)
    assert result.ok is True


def make_unvalidated_constraints(**mutations):
    """A valid constraint set, mutated *after* construction to defeat ``__post_init__``.

    ``Constraints.__post_init__`` rejects an unreachable group bound outright, so a set that
    reaches the input contract in that state cannot be constructed — but one can arrive there,
    because ``update_with_valid_tickers`` rebuilds the box for the surviving universe on every
    rebalancing date and a group whose members were dropped is then unreachable against the
    new box. The contract is the second line of defence for exactly that, so the state is
    reproduced here by assigning the fields rather than passing them.
    """
    constraints = make_constraints()
    for field, value in mutations.items():
        object.__setattr__(constraints, field, value)   # the dataclass is frozen
    return constraints


def test_a_group_floor_the_box_cannot_supply_is_reported() -> None:
    """the contract restates the construction-time check, on a set that got past it"""
    group_loadings = pd.DataFrame({'risky': [1.0, 0.0, 0.0], 'safe': [0.0, 1.0, 1.0]},
                                  index=TICKERS)
    constraints = make_unvalidated_constraints(
        max_weights=pd.Series([0.1, 1.0, 1.0], index=TICKERS),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=group_loadings,
            group_min_allocation=pd.Series({'risky': 0.5, 'safe': 0.0}),
            group_max_allocation=pd.Series({'risky': 1.0, 'safe': 1.0})))
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints)
    # reported, not fatal: only a covariance-integrity problem clears ``ok``
    assert "group 'risky' floor" in issues_text(result)
    assert 'max reachable' in issues_text(result)


def test_a_group_cap_below_what_the_box_forces_is_reported() -> None:
    """the mirror case: box floors that already exceed the group's own ceiling"""
    group_loadings = pd.DataFrame({'risky': [1.0, 0.0, 0.0], 'safe': [0.0, 1.0, 1.0]},
                                  index=TICKERS)
    constraints = make_unvalidated_constraints(
        min_weights=pd.Series([0.4, 0.0, 0.0], index=TICKERS),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=group_loadings,
            group_min_allocation=pd.Series({'risky': 0.0, 'safe': 0.0}),
            group_max_allocation=pd.Series({'risky': 0.1, 'safe': 1.0})))
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints)
    assert "group 'risky' cap" in issues_text(result)
    assert 'min forced' in issues_text(result)


def group_constraints_over_a_wider_universe() -> GroupLowerUpperConstraints:
    """A group block describing an asset the box no longer covers.

    ``GroupLowerUpperConstraints`` drops an all-zero loadings column at construction, so a
    group with no members cannot be *built*. It can still arrive: the box is rebuilt for the
    surviving universe at every rebalancing date, and a group whose only member was dropped
    then reindexes to a column of zeros. That is the state reproduced here — the loadings
    describe four assets and the box describes three.
    """
    loadings = pd.DataFrame({'alts': [0.0, 0.0, 0.0, 1.0], 'core': [1.0, 1.0, 1.0, 0.0]},
                            index=TICKERS + ['private'])
    return GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series({'alts': 0.5, 'core': 0.0}),
        group_max_allocation=pd.Series({'alts': 1.0, 'core': 1.0}))


def test_a_group_with_no_members_left_is_not_checked() -> None:
    """a group that reindexes to nothing constrains nothing, so it cannot be unreachable

    Testing its 0.5 floor against a reachable total of zero would report an infeasibility on
    every rebalancing date, for a limit that no longer applies to any held asset.
    """
    constraints = make_unvalidated_constraints(
        group_lower_upper_constraints=group_constraints_over_a_wider_universe())
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints)
    assert 'alts' not in issues_text(result)
    assert result.ok is True


def test_feasibility_checking_can_be_switched_off() -> None:
    """the structural pass is optional, for callers paying for it per rebalance"""
    constraints = make_constraints(max_weights=pd.Series(0.2, index=TICKERS))
    unchecked = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints,
                                       check_feasibility=False)
    checked = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints,
                                     check_feasibility=True)
    assert len(unchecked.issues) < len(checked.issues)


def test_conditioning_checking_can_be_switched_off() -> None:
    """likewise the conditioning pass, which is the more expensive of the two"""
    singular = pd.DataFrame(np.ones((3, 3)) * 0.04, index=TICKERS, columns=TICKERS)
    unchecked = validate_solver_inputs(pd_covar=singular, constraints=make_constraints(),
                                       check_conditioning=False)
    checked = validate_solver_inputs(pd_covar=singular, constraints=make_constraints(),
                                     check_conditioning=True)
    assert len(checked.issues) >= len(unchecked.issues)


def test_deep_feasibility_runs_an_actual_lp() -> None:
    """the heavyweight check solves the problem rather than testing necessary conditions"""
    constraints = make_constraints(max_weights=pd.Series(0.2, index=TICKERS))
    result = validate_solver_inputs(pd_covar=COVAR_DF, constraints=constraints,
                                    deep_feasibility=True)
    assert issues_text(result) != ''


# --------------------------------------------------------------------------- #
# covariance conditioning
# --------------------------------------------------------------------------- #
def test_conditioning_flags_a_singular_covariance(caplog) -> None:
    """a rank-deficient covariance is exactly the numerical blow-up case"""
    singular = pd.DataFrame(np.ones((3, 3)) * 0.04, index=TICKERS, columns=TICKERS)
    with caplog.at_level(logging.WARNING):
        check_covar_conditioning(singular, context='singular')
    assert caplog.text != ''


def test_the_contract_reads_conditioning_off_a_factorisation_when_one_is_given() -> None:
    """with a factorisation in hand the contract reports the raw *and* stabilised numbers

    The solve runs on the stabilised covariance, so reporting only the raw condition number
    would overstate the problem and reporting only the stabilised one would hide it. Both are
    named, together with how many eigenvalues had to be floored to get there — which is the
    number that says whether the stabilisation was cosmetic or structural.
    """
    singular = pd.DataFrame(np.ones((3, 3)) * 0.04, index=TICKERS, columns=TICKERS)
    factorization = factorize_covariance(singular.to_numpy())
    result = validate_solver_inputs(pd_covar=singular, constraints=make_constraints(),
                                    covar_factorization=factorization)
    text = issues_text(result)
    assert 'raw covariance ill-conditioned' in text
    assert 'stabilized with' in text
    assert 'eigenvalue(s) floored' in text
    # the collinear pair is named too, so the report points at the assets to fix
    assert "pair 'growth'" in text or "pair 'balanced'" in text


def test_a_healthy_covariance_with_a_factorisation_reports_nothing() -> None:
    """the mirror case, so the branch above is not passing for the wrong reason"""
    result = validate_solver_inputs(
        pd_covar=COVAR_DF, constraints=make_constraints(),
        covar_factorization=factorize_covariance(COVAR))
    assert result.ok is True
    assert 'ill-conditioned' not in issues_text(result)


def test_conditioning_is_quiet_on_a_well_behaved_covariance(caplog) -> None:
    """a healthy covariance must not produce noise, or the warning means nothing"""
    with caplog.at_level(logging.WARNING):
        check_covar_conditioning(COVAR_DF, context='healthy')
    assert 'ill-conditioned' not in caplog.text.lower()


# --------------------------------------------------------------------------- #
# infeasibility diagnosis
# --------------------------------------------------------------------------- #
def test_elastic_diagnosis_names_the_bound_that_must_give() -> None:
    """caps summing to 0.6 must relax by 0.4 in total to allow full investment"""
    constraints = make_constraints(max_weights=pd.Series(0.2, index=TICKERS))
    breaches = diagnose_infeasibility(constraints, covar=COVAR, context='capped')
    assert breaches, "an impossible box produced no diagnosis"
    assert sum(breaches.values()) == pytest.approx(0.4, abs=1e-3)


def test_elastic_diagnosis_returns_nothing_when_its_subset_is_satisfiable(caplog) -> None:
    """a feasible box/group subset has no slack without claiming the whole policy passes"""
    with caplog.at_level(logging.WARNING):
        result = diagnose_infeasibility(make_constraints(), covar=COVAR, context='fine')
    assert result == {}
    assert 'box/group subset is satisfiable' in caplog.text


def test_elastic_diagnosis_skips_constraints_with_no_indexed_bounds(caplog) -> None:
    """without an asset index there is nothing to relax, so it says so and stops"""
    with caplog.at_level(logging.WARNING):
        result = diagnose_infeasibility(Constraints(is_long_only=True), context='bare')
    assert result == {}
    assert 'skipped' in caplog.text


def test_elastic_diagnosis_reports_the_group_cap_that_must_widen() -> None:
    """a group ceiling below what the box forces is named with the widening it needs

    Box bounds are per asset and a group bound is a row over several of them. The elastic LP
    reports both on the same footing, so the answer to "which limit is the binding one" does
    not depend on which kind of limit it happens to be.
    """
    loadings = pd.DataFrame({'risky': [1.0, 1.0, 0.0], 'safe': [0.0, 0.0, 1.0]},
                            index=TICKERS)
    constraints = make_unvalidated_constraints(
        min_weights=pd.Series([0.4, 0.4, 0.0], index=TICKERS),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=pd.Series({'risky': 0.0, 'safe': 0.0}),
            group_max_allocation=pd.Series({'risky': 0.5, 'safe': 1.0})))
    breaches = diagnose_infeasibility(constraints, covar=COVAR, context='grouped')
    assert 'group_max:risky' in breaches
    # the floors force 0.8 into 'risky' against a 0.5 cap, a gap of 0.3. The LP minimises
    # *total* slack, so it is free to split that between widening the cap and lowering the
    # box floors — what is pinned is the total, and that the group row is named at all
    assert sum(breaches.values()) == pytest.approx(0.3, abs=1e-3)


def test_elastic_diagnosis_ignores_a_group_with_no_members() -> None:
    """a group that reindexes to nothing has no slack to add, so it is left out entirely

    Giving it one would put an unsatisfiable row into the elastic LP and make every
    diagnosis report that group as the binding constraint, whatever actually went wrong.
    """
    constraints = make_unvalidated_constraints(
        max_weights=pd.Series(0.2, index=TICKERS),
        group_lower_upper_constraints=group_constraints_over_a_wider_universe())
    breaches = diagnose_infeasibility(constraints, covar=COVAR, context='empty-group')
    assert not any(key.endswith(':alts') for key in breaches)
    assert breaches, 'the box breach itself should still be diagnosed'


def test_elastic_diagnosis_stops_when_there_is_nothing_to_relax(caplog) -> None:
    """with an asset index but no box or group bounds the LP has no slack variables

    The diagnosis exists to name a bound. A constraint set carrying only a benchmark and a
    budget has none to name, and an LP minimising an empty sum would report "satisfiable" —
    which reads as "the constraints were fine" rather than "there were no constraints".
    """
    constraints = Constraints(is_long_only=True, min_exposure=1.0, max_exposure=1.0,
                              benchmark_weights=pd.Series(1 / 3, index=TICKERS))
    with caplog.at_level(logging.WARNING):
        result = diagnose_infeasibility(constraints, covar=COVAR, context='no-bounds')
    assert result == {}
    assert 'box/group subset has no bounds to test' in caplog.text


def test_elastic_diagnosis_that_will_not_solve_reports_and_returns(caplog,
                                                                   monkeypatch) -> None:
    """the diagnosis must never be the thing that ends a run

    It runs *after* a solve was already rejected, as an explanation. A backend that raises on
    the elastic LP too would otherwise turn a logged bad rebalance into a crashed backtest.
    """
    import cvxpy as cvx

    def _raise(self, *args, **kwargs):
        """Stand in for a backend that cannot solve the elastic LP either."""
        raise cvx.error.SolverError('elastic LP failed')

    monkeypatch.setattr(cvx.Problem, 'solve', _raise)
    constraints = make_constraints(max_weights=pd.Series(0.2, index=TICKERS))
    with caplog.at_level(logging.WARNING):
        assert diagnose_infeasibility(constraints, covar=COVAR, context='broken') == {}
    assert 'failed to solve' in caplog.text


def test_an_inconclusive_elastic_solve_says_so_rather_than_reporting_nothing(
        caplog, monkeypatch) -> None:
    """an elastic LP that did not reach optimality proves nothing either way

    Returning an empty dict silently would be read as "the constraints were satisfiable",
    which is the opposite of what an unsolved diagnosis establishes.
    """
    import cvxpy as cvx

    def _do_nothing(self, *args, **kwargs):
        """Leave the problem unsolved: no status, no slack values."""
        return None

    monkeypatch.setattr(cvx.Problem, 'solve', _do_nothing)
    constraints = make_constraints(max_weights=pd.Series(0.2, index=TICKERS))
    with caplog.at_level(logging.WARNING):
        assert diagnose_infeasibility(constraints, covar=COVAR, context='unsolved') == {}
    assert 'inconclusive' in caplog.text


def test_a_floor_that_overshoots_the_budget_is_diagnosed() -> None:
    """the elastic LP handles the min side as well as the max side"""
    constraints = make_constraints(min_weights=pd.Series(0.5, index=TICKERS))
    breaches = diagnose_infeasibility(constraints, covar=COVAR, context='floored')
    assert breaches
    assert sum(breaches.values()) == pytest.approx(0.5, abs=1e-3)


def test_infeasible_status_routes_to_the_elastic_diagnosis(caplog) -> None:
    """the router sends an infeasible solve to the constraint analysis"""
    constraints = make_constraints(max_weights=pd.Series(0.2, index=TICKERS))
    with caplog.at_level(logging.WARNING):
        diagnose_solver_failure('infeasible', constraints, covar=COVAR, context='r1')
    assert caplog.text != ''


def test_a_numerical_rejection_routes_to_conditioning_instead(caplog) -> None:
    """a solve the solver called optimal but we rejected had a non-empty feasible region

    So the useful question is about the covariance, not the constraints — the router must
    not run the elastic LP, which would report no breaches and explain nothing.
    """
    with caplog.at_level(logging.WARNING):
        diagnose_solver_failure('optimal', make_constraints(), covar=COVAR, context='r2')
    assert 'conditioning' in caplog.text.lower()


def test_the_router_does_nothing_without_a_covariance() -> None:
    """with neither an infeasible status nor a covariance there is nothing to diagnose"""
    diagnose_solver_failure('numerical_error', make_constraints(), covar=None)


# --------------------------------------------------------------------------- #
# constraint residuals
# --------------------------------------------------------------------------- #
def test_residuals_pass_for_a_compliant_weight_vector() -> None:
    """a portfolio inside every bound produces residuals that all pass"""
    weights = np.array([0.4, 0.35, 0.25])
    residuals = evaluate_constraint_residuals(weights=weights,
                                              constraints=make_constraints(), covar=COVAR)
    assert len(residuals) > 0
    assert all(r.passed for r in residuals if r.hard)


def test_budget_and_box_residual_order_is_stable() -> None:
    """the analytics extraction preserves the report-facing residual row order"""
    residuals = evaluate_constraint_residuals(
        weights=np.array([0.4, 0.35, 0.25]),
        constraints=make_constraints(),
        covar=COVAR,
    )

    assert [
        (residual.constraint_type, residual.name, residual.lower, residual.upper)
        for residual in residuals
    ] == [
        ('exposure', 'total', 1.0, 1.0),
        ('long_only', 'minimum_weight', 0.0, None),
        ('instrument_weight', 'growth', 0.0, None),
        ('instrument_weight', 'growth', None, 1.0),
        ('instrument_weight', 'balanced', 0.0, None),
        ('instrument_weight', 'balanced', None, 1.0),
        ('instrument_weight', 'defensive', 0.0, None),
        ('instrument_weight', 'defensive', None, 1.0),
    ]


def test_residuals_flag_a_breached_box_bound() -> None:
    """a weight above its cap is reported with the size of the breach"""
    constraints = make_constraints(max_weights=pd.Series([0.3, 1.0, 1.0], index=TICKERS))
    residuals = evaluate_constraint_residuals(weights=np.array([0.8, 0.1, 0.1]),
                                              constraints=constraints, covar=COVAR)
    failed = [r for r in residuals if not r.passed]
    assert failed
    assert max(r.violation for r in failed) == pytest.approx(0.5, abs=1e-6)


def test_residuals_flag_a_breached_budget() -> None:
    """a portfolio that is not fully invested breaches the exposure constraint"""
    residuals = evaluate_constraint_residuals(weights=np.array([0.2, 0.2, 0.2]),
                                              constraints=make_constraints(), covar=COVAR)
    assert any(not r.passed for r in residuals)


def test_utility_terms_are_reported_but_do_not_gate_compliance() -> None:
    """soft penalties appear as residuals with hard=False, so they never fail an audit"""
    constraints = Constraints(
        is_long_only=True, min_weights=pd.Series(0.0, index=TICKERS),
        max_weights=pd.Series(1.0, index=TICKERS),
        benchmark_weights=pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS),
        tracking_err_vol_constraint=0.01,
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    residuals = evaluate_constraint_residuals(weights=np.array([1.0, 0.0, 0.0]),
                                              constraints=constraints, covar=COVAR)
    soft = [r for r in residuals if not r.hard]
    assert soft, "the utility formulation reported no soft terms"
    # the tracking error is wildly breached, yet every hard residual still passes
    assert all(r.passed for r in residuals if r.hard)


def test_residual_records_carry_their_bounds_and_tolerance() -> None:
    """each record is self-describing, because it is what an audit report prints"""
    residuals = evaluate_constraint_residuals(weights=np.array([0.4, 0.35, 0.25]),
                                              constraints=make_constraints(), covar=COVAR)
    record = residuals[0]
    assert record.constraint_type
    assert record.name
    assert record.tolerance > 0.0
    assert isinstance(record.passed, (bool, np.bool_))


def test_turnover_is_audited_against_the_prior_portfolio() -> None:
    """the L1 trade against weights_0 is measured, with per-asset costs when given

    Costed turnover is what a mandate actually limits — a percentage point traded in an
    illiquid sleeve is not the same budget as one in cash. Without the cost weighting the
    audit reports a constraint the solver was never asked to respect.
    """
    weights_0 = pd.Series([0.4, 0.35, 0.25], index=TICKERS)
    weights = np.array([0.6, 0.25, 0.15])              # L1 trade of 0.4
    plain = evaluate_constraint_residuals(
        weights=weights, covar=COVAR,
        constraints=make_constraints(weights_0=weights_0, turnover_constraint=0.10))
    turnover = [r for r in plain if r.constraint_type == 'turnover']
    assert len(turnover) == 1
    assert turnover[0].actual == pytest.approx(0.4, abs=1e-8)
    assert not turnover[0].passed

    costed = evaluate_constraint_residuals(
        weights=weights, covar=COVAR,
        constraints=make_constraints(
            weights_0=weights_0, turnover_constraint=0.10,
            turnover_costs=pd.Series([2.0, 1.0, 1.0], index=TICKERS)))
    costed_turnover = [r for r in costed if r.constraint_type == 'turnover'][0]
    # the doubled cost on 'growth' adds its 0.2 trade a second time
    assert costed_turnover.actual == pytest.approx(0.6, abs=1e-8)


def test_group_turnover_is_audited_group_by_group() -> None:
    """a per-group trade limit is a separate residual for each named group"""
    loadings = pd.DataFrame({'risky': [1.0, 1.0, 0.0], 'safe': [0.0, 0.0, 1.0]},
                            index=TICKERS)
    constraints = make_constraints(
        weights_0=pd.Series([0.4, 0.35, 0.25], index=TICKERS),
        group_turnover_constraint=GroupTurnoverConstraint(
            group_loadings=loadings,
            group_max_turnover=pd.Series([0.10, np.nan, 0.20],
                                         index=['risky', 'safe', 'not_a_group'])))
    residuals = evaluate_constraint_residuals(weights=np.array([0.6, 0.25, 0.15]),
                                              constraints=constraints, covar=COVAR)
    by_name = {r.name: r for r in residuals if r.constraint_type == 'group_turnover'}
    # 'safe' carries a NaN limit and 'not_a_group' is absent from the loadings: neither is
    # a stated limit, and inventing one would fail an audit the mandate never asked for
    assert set(by_name) == {'risky'}
    assert by_name['risky'].actual == pytest.approx(0.3, abs=1e-8)
    assert not by_name['risky'].passed


def test_a_group_tracking_error_limit_without_a_stated_vol_is_skipped() -> None:
    """a group named in the loadings but not in the vol table carries no limit"""
    loadings = pd.DataFrame({'risky': [1.0, 1.0, 0.0], 'safe': [0.0, 0.0, 1.0]},
                            index=TICKERS)
    constraints = make_constraints(
        benchmark_weights=pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS),
        group_tracking_error_constraint=GroupTrackingErrorConstraint(
            group_loadings=loadings,
            group_tre_vols=pd.Series([0.02, np.nan], index=['risky', 'safe'])))
    residuals = evaluate_constraint_residuals(weights=np.array([0.8, 0.1, 0.1]),
                                              constraints=constraints, covar=COVAR)
    names = {r.name for r in residuals if r.constraint_type == 'group_tracking_error'}
    assert names == {'risky'}


def test_a_group_bound_stated_as_nan_is_not_a_bound() -> None:
    """NaN is how an unset side of a group range is written, and it must not gate

    Treating NaN as a number gives a bound of NaN, and every comparison against it is False —
    so the residual silently passes whatever the weights are, which is worse than absent.
    """
    loadings = pd.DataFrame({'risky': [1.0, 1.0, 0.0], 'safe': [0.0, 0.0, 1.0]},
                            index=TICKERS)
    constraints = make_constraints(group_lower_upper_constraints=GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series([0.0, np.nan], index=['risky', 'safe']),
        group_max_allocation=pd.Series([np.nan, 1.0], index=['risky', 'safe'])))
    residuals = evaluate_constraint_residuals(weights=np.array([0.4, 0.35, 0.25]),
                                              constraints=constraints, covar=COVAR)
    by_name = {r.name: r for r in residuals if r.constraint_type == 'group_weight'}
    assert set(by_name) == {'risky', 'safe'}
    assert by_name['risky'].upper is None and by_name['risky'].lower == 0.0
    assert by_name['safe'].lower is None and by_name['safe'].upper == 1.0


def test_sector_and_style_deviations_are_audited_as_absolute_gaps() -> None:
    """a deviation limit is two-sided, so the residual is on the magnitude

    An underweight breaches a deviation mandate exactly as an overweight does. Recording the
    signed number would let a large underweight pass a one-sided upper bound.
    """
    loadings = pd.DataFrame({'Tech': [1.0, 0.0, 0.0], 'Value': [0.0, 1.0, 0.0]},
                            index=TICKERS)
    deviations = BenchmarkDeviationConstraints(
        factor_loading_mat=loadings,
        factor_max_deviation=pd.Series([0.05, np.nan], index=['Tech', 'Value']))
    constraints = make_constraints(
        benchmark_weights=pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS),
        sector_deviation_constraints=deviations,
        style_deviation_constraints=deviations)
    residuals = evaluate_constraint_residuals(weights=np.array([0.1, 0.6, 0.3]),
                                              constraints=constraints, covar=COVAR)
    sector = {r.name: r for r in residuals if r.constraint_type == 'sector_deviation'}
    assert set(sector) == {'Tech'}                 # the NaN limit is not a limit
    # the 'Tech' underweight is 0.1 - 1/3, and it is reported as its magnitude
    assert sector['Tech'].actual == pytest.approx(abs(0.1 - 1 / 3), abs=1e-8)
    assert not sector['Tech'].passed
    assert {r.name for r in residuals if r.constraint_type == 'style_deviation'} == {'Tech'}


def test_the_residual_audit_renders_as_a_report_table() -> None:
    """the audit is printed on a factsheet, so it has a frame form with fixed columns"""
    outcome = validate_solution(np.array([0.4, 0.35, 0.25]), 'optimal', make_constraints(),
                                len(TICKERS), covar=COVAR)
    frame = outcome.residuals_frame()
    assert list(frame.columns) == ['constraint_type', 'name', 'actual', 'lower', 'upper',
                                   'violation', 'tolerance', 'hard', 'passed']
    assert len(frame) == len(outcome.constraint_residuals)
    assert outcome.compliant


def test_the_report_table_of_an_unaudited_solve_is_empty_but_shaped() -> None:
    """an outcome carrying no residuals still renders a table with its columns

    A factsheet concatenates these across rebalancing dates. An empty frame with no columns
    would silently drop the block rather than showing an empty one.
    """
    outcome = OptimizationOutcome(
        weights=np.array([0.4, 0.35, 0.25]), accepted=True, solver='CLARABEL',
        status='optimal', context='t', reason='', fallback_source=None,
        constraint_residuals=())
    assert outcome.compliant
    frame = outcome.residuals_frame()
    assert frame.empty
    assert list(frame.columns) == ['constraint_type', 'name', 'actual', 'lower', 'upper',
                                   'violation', 'tolerance', 'hard', 'passed']
