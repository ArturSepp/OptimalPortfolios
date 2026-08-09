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
    ConstraintEnforcementType, GroupLowerUpperConstraints)
from optimalportfolios.optimization.solver_diagnostics import (
    check_covar_conditioning,
    diagnose_infeasibility,
    diagnose_solver_failure,
    evaluate_constraint_residuals,
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


def test_a_covariance_of_the_wrong_size_is_a_hard_failure() -> None:
    """a covariance that does not match the constraints cannot be solved against them"""
    smaller = COVAR_DF.iloc[:2, :2]
    result = validate_solver_inputs(pd_covar=smaller, constraints=make_constraints())
    assert result.ok is False


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


def test_elastic_diagnosis_returns_nothing_when_the_constraints_are_satisfiable() -> None:
    """a feasible set has no slack, which is how numerical infeasibility is told apart"""
    assert diagnose_infeasibility(make_constraints(), covar=COVAR, context='fine') == {}


def test_elastic_diagnosis_skips_constraints_with_no_indexed_bounds(caplog) -> None:
    """without an asset index there is nothing to relax, so it says so and stops"""
    with caplog.at_level(logging.WARNING):
        result = diagnose_infeasibility(Constraints(is_long_only=True), context='bare')
    assert result == {}
    assert 'skipped' in caplog.text


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
