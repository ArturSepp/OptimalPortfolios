"""
the paths through ``Constraints`` that a well-formed problem never takes.

``constraints_test.py`` and ``specialised_constraints_test.py`` cover the constraint classes
on inputs that make sense. What is left is the other half of the module: the guards that fire
when a caller supplies half a constraint, the realignment branches that run only when a
particular optional block is present, and the reporting helpers.

None of these is exotic. A production run reaches most of them within a few rebalancing dates
— an asset drops out of the covariance and every loadings matrix has to follow it; a mandate
states a turnover cap but the first date has no prior portfolio; a frozen position pushes a
group above its own ceiling. What they have in common is that they are *silent*: a
realignment that quietly skips one block leaves a group bound computed over the wrong
membership, and the solve still returns a plausible portfolio.

So each case here states the malformed or partial input and asserts the specific thing the
module does about it — the exception and its message, the warning and the group it names, or
the constraint that did or did not make it into the cvx problem.
"""
# packages
import logging
import numpy as np
import pandas as pd
import pytest
import cvxpy as cvx
# optimalportfolios
from optimalportfolios.optimization.constraints import (
    BenchmarkBetaConstraint,
    BenchmarkDeviationConstraints,
    Constraints,
    GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
    _cvx_factor_risk,
    _reindex_optional_series,
    compute_benchmark_beta_loadings_from_covar,
    merge_group_lower_upper_constraints,
)
from optimalportfolios.optimization.covar_factorization import factorize_covariance

TICKERS = ['growth', 'balanced', 'defensive']
VOLS = np.array([0.22, 0.14, 0.06])
CORR = np.array([[1.00, 0.45, 0.15],
                 [0.45, 1.00, 0.25],
                 [0.15, 0.25, 1.00]])
COVAR = pd.DataFrame(np.outer(VOLS, VOLS) * CORR, index=TICKERS, columns=TICKERS)
GROUP_LOADINGS = pd.DataFrame({'risky': [1.0, 1.0, 0.0], 'safe': [0.0, 0.0, 1.0]},
                              index=TICKERS)
BENCHMARK = pd.Series([0.4, 0.35, 0.25], index=TICKERS)
WEIGHTS_0 = pd.Series([0.3, 0.3, 0.4], index=TICKERS)


def make_constraints(**overrides) -> Constraints:
    """A satisfiable long-only, fully invested constraint set, with overrides."""
    kwargs = dict(is_long_only=True,
                  min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS))
    kwargs.update(overrides)
    return Constraints(**kwargs)


def solve_under(w: cvx.Variable, constraint_list) -> np.ndarray:
    """Minimise variance under the supplied constraints, against the same ``w``."""
    problem = cvx.Problem(cvx.Minimize(cvx.sum_squares(w)), list(constraint_list))
    problem.solve(solver='CLARABEL')
    return w.value


# --------------------------------------------------------------------------- #
# module helpers
# --------------------------------------------------------------------------- #
def test_the_factor_risk_expression_checks_its_own_dimensions() -> None:
    """a factorisation of the wrong universe would otherwise broadcast into nonsense

    The factor is ``k x n``; multiplying it by a weight vector of a different length either
    raises deep inside cvxpy or, worse, broadcasts. Checking here names both shapes.
    """
    factorization = factorize_covariance(COVAR.to_numpy())
    with pytest.raises(ValueError, match='does not match the weight vector'):
        _cvx_factor_risk(cvx.Variable(5), factorization)


def test_reindexing_an_optional_series_passes_absence_through() -> None:
    """the realignment helper treats "not supplied" and "supplied and empty" differently"""
    assert _reindex_optional_series(None, pd.Index(TICKERS)) is None
    reindexed = _reindex_optional_series(pd.Series([1.0], index=['growth']),
                                         pd.Index(TICKERS), fill_value=0.0)
    assert list(reindexed.index) == TICKERS
    assert reindexed['balanced'] == 0.0


# --------------------------------------------------------------------------- #
# GroupLowerUpperConstraints
# --------------------------------------------------------------------------- #
def test_a_group_with_no_stated_bound_is_skipped_with_a_warning() -> None:
    """a group present in the loadings but absent from the bounds constrains nothing

    Silently applying no bound is right — the alternative is inventing one — but it has to be
    said out loud, because a typo in a group name looks exactly like this.
    """
    gluc = GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS,
        group_min_allocation=pd.Series({'risky': 0.1, 'safe': 0.1}),
        group_max_allocation=pd.Series({'risky': 0.9, 'safe': 0.9}))
    # __post_init__ reindexes the bounds onto every group, so a genuinely absent entry is
    # assigned afterwards — which is the state a caller-built block arrives in
    object.__setattr__(gluc, 'group_min_allocation', pd.Series({'risky': 0.1}))
    object.__setattr__(gluc, 'group_max_allocation', pd.Series({'risky': 0.9}))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    with pytest.warns(UserWarning, match='no group=safe in group_min_allocation'):
        with pytest.warns(UserWarning, match='no group=safe in group_max_allocation'):
            constraints = gluc.set_cvx_group_lower_upper_constraints(w=w)
    # only the two 'risky' bounds were emitted
    assert len(constraints) == 2


def test_group_bounds_print_their_three_tables(capsys) -> None:
    """the debugging print names each table, so a mis-set bound is visible at a glance"""
    GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS,
        group_min_allocation=pd.Series({'risky': 0.1, 'safe': 0.1}),
        group_max_allocation=pd.Series({'risky': 0.9, 'safe': 0.9})).print()
    printed = capsys.readouterr().out
    assert 'group_loadings' in printed
    assert 'group_min_allocation' in printed
    assert 'group_max_allocation' in printed


def with_a_duplicated_asset_row() -> GroupLowerUpperConstraints:
    """A group block whose loadings repeat one asset.

    The duplicate cannot survive ``__post_init__`` — its reindex raises — so it is assigned
    afterwards, which is the state a block assembled upstream from a concatenated metadata
    table arrives in.
    """
    block = GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS,
        group_min_allocation=pd.Series({'risky': 0.1, 'safe': 0.1}),
        group_max_allocation=pd.Series({'risky': 0.9, 'safe': 0.9}))
    object.__setattr__(block, 'group_loadings',
                       pd.concat([GROUP_LOADINGS, GROUP_LOADINGS.loc[['growth']]], axis=0))
    return block


def clean_second_block() -> GroupLowerUpperConstraints:
    """A well-formed block over a different grouping, to merge against."""
    return GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({'domestic': [1.0, 0.0, 1.0]}, index=TICKERS),
        group_min_allocation=pd.Series({'domestic': 0.0}),
        group_max_allocation=pd.Series({'domestic': 1.0}))


def test_merging_names_the_input_whose_asset_index_is_duplicated() -> None:
    """the merge warns before it fails, and says which of the two inputs is at fault

    The concat on the group axis cannot align a repeated asset and raises a pandas error that
    names neither input. The warning is what turns that into an actionable message — so it
    has to be emitted for whichever side carries the duplicate, not only the first.
    """
    with pytest.warns(UserWarning, match='group_lower_upper_constraints1'):
        with pytest.raises(pd.errors.InvalidIndexError):
            merge_group_lower_upper_constraints(with_a_duplicated_asset_row(),
                                                clean_second_block())
    with pytest.warns(UserWarning, match='group_lower_upper_constraints2'):
        with pytest.raises(pd.errors.InvalidIndexError):
            merge_group_lower_upper_constraints(clean_second_block(),
                                                with_a_duplicated_asset_row())


def test_merging_keeps_whichever_side_states_a_bound() -> None:
    """one-sided blocks merge without inventing the missing side

    A block stating only floors and one stating only caps are both legitimate; the merge has
    to carry each through rather than dropping the side its partner did not state.
    """
    floors_only = GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS,
        group_min_allocation=pd.Series({'risky': 0.1, 'safe': 0.1}),
        group_max_allocation=None)
    caps_only = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({'domestic': [1.0, 0.0, 1.0]}, index=TICKERS),
        group_min_allocation=None,
        group_max_allocation=pd.Series({'domestic': 0.8}))

    merged = merge_group_lower_upper_constraints(floors_only, caps_only)
    assert merged.group_min_allocation['risky'] == 0.1
    assert merged.group_max_allocation['domestic'] == 0.8
    # the sides the inputs did not state come back as NaN, which reads as "unbounded"
    assert np.isnan(merged.group_max_allocation['risky'])
    assert np.isnan(merged.group_min_allocation['domestic'])

    # and the mirror order, which takes the other arm of each branch
    mirrored = merge_group_lower_upper_constraints(caps_only, floors_only)
    assert mirrored.group_min_allocation['risky'] == 0.1
    assert mirrored.group_max_allocation['domestic'] == 0.8


def test_merging_two_unbounded_blocks_states_no_bounds_at_all() -> None:
    """loadings with no allocations on either side merge to loadings with none"""
    first = GroupLowerUpperConstraints(group_loadings=GROUP_LOADINGS,
                                       group_min_allocation=None,
                                       group_max_allocation=None)
    second = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({'domestic': [1.0, 0.0, 1.0]}, index=TICKERS),
        group_min_allocation=None, group_max_allocation=None)
    merged = merge_group_lower_upper_constraints(first, second)
    assert merged.group_min_allocation is None
    assert merged.group_max_allocation is None
    assert set(merged.group_loadings.columns) == {'risky', 'safe', 'domestic'}


# --------------------------------------------------------------------------- #
# group tracking error and group turnover
# --------------------------------------------------------------------------- #
def test_group_tre_utility_weights_warn_about_a_group_they_do_not_cover() -> None:
    """the utility form is validated exactly like the hard form"""
    with pytest.warns(UserWarning, match='Missing in group_loadings.columns'):
        GroupTrackingErrorConstraint(
            group_loadings=GROUP_LOADINGS,
            group_tre_utility_weights=pd.Series({'risky': 1.0}))


def test_group_tre_utility_with_nothing_to_penalise_warns_and_returns_nothing() -> None:
    """every group weight is NaN, so no term is built and the caller is told

    Returning ``None`` silently would leave the objective without its tracking-error term and
    the solve would maximise alpha unpenalised — a very different portfolio, arrived at with
    no indication that a stated penalty was dropped.
    """
    constraint = GroupTrackingErrorConstraint(
        group_loadings=GROUP_LOADINGS,
        group_tre_utility_weights=pd.Series({'risky': np.nan, 'safe': np.nan}))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    with pytest.warns(UserWarning, match='objective_fun is None'):
        term = constraint.set_cvx_group_tre_utility(
            w=w, benchmark_weights=BENCHMARK, covar=COVAR.to_numpy())
    assert term is None


def test_group_turnover_utility_weights_warn_about_a_group_they_do_not_cover() -> None:
    """the same validation on the turnover side"""
    with pytest.warns(UserWarning, match='Missing in self.group_loadings.columns'):
        GroupTurnoverConstraint(
            group_loadings=GROUP_LOADINGS,
            group_turnover_utility_weights=pd.Series({'risky': 1.0}))


def test_group_turnover_without_a_prior_portfolio_emits_no_constraint(caplog) -> None:
    """turnover is a distance from somewhere, and the first rebalancing has no somewhere

    Skipping is right — there is nothing to trade away from — but the alternative reading is
    that the cap was applied and happened not to bind, so the skip is logged.
    """
    constraint = GroupTurnoverConstraint(
        group_loadings=GROUP_LOADINGS,
        group_max_turnover=pd.Series({'risky': 0.1, 'safe': 0.1}))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    with caplog.at_level(logging.DEBUG,
                         logger='optimalportfolios.optimization.constraints'):
        assert constraint.set_group_turnover_constraints(w=w, weights_0=None) == []
    assert 'weights_0 is absent' in caplog.text


def test_group_turnover_utility_requires_its_weights() -> None:
    """a constraint configured with caps cannot be asked for a penalty instead"""
    constraint = GroupTurnoverConstraint(
        group_loadings=GROUP_LOADINGS,
        group_max_turnover=pd.Series({'risky': 0.1, 'safe': 0.1}))
    with pytest.raises(ValueError, match='group_turnover_utility_weights must be supplied'):
        constraint.set_cvx_group_turnover_utility(w=cvx.Variable(len(TICKERS)),
                                                  weights_0=WEIGHTS_0)


def test_group_turnover_utility_without_a_prior_portfolio_builds_no_term(caplog) -> None:
    """the penalty form skips the same way the hard form does"""
    constraint = GroupTurnoverConstraint(
        group_loadings=GROUP_LOADINGS,
        group_turnover_utility_weights=pd.Series({'risky': 1.0, 'safe': 1.0}))
    with caplog.at_level(logging.DEBUG,
                         logger='optimalportfolios.optimization.constraints'):
        term = constraint.set_cvx_group_turnover_utility(w=cvx.Variable(len(TICKERS)),
                                                         weights_0=None)
    assert term is None
    assert 'weights_0 is absent' in caplog.text


# --------------------------------------------------------------------------- #
# beta loadings from a joint covariance
# --------------------------------------------------------------------------- #
def test_joint_beta_loadings_reject_a_benchmark_the_covariance_does_not_cover() -> None:
    """the whole point of the joint form is one matrix, so a missing constituent is fatal

    The message names the estimation flag that fixes it, because the alternative — silently
    dropping the constituent — changes the benchmark the beta is measured against.
    """
    benchmark = pd.Series([0.5, 0.5], index=['growth', 'not_in_covar'])
    with pytest.raises(KeyError, match='missing from joint covariance'):
        compute_benchmark_beta_loadings_from_covar(
            covar=COVAR, benchmark_weights=benchmark, asset_tickers=TICKERS)


def test_joint_beta_loadings_reject_a_benchmark_with_no_variance() -> None:
    """beta divides by the benchmark variance, so a zero-weight benchmark has no beta"""
    with pytest.raises(ValueError, match='benchmark variance must be positive'):
        compute_benchmark_beta_loadings_from_covar(
            covar=COVAR, benchmark_weights=pd.Series(0.0, index=TICKERS),
            asset_tickers=TICKERS)


# --------------------------------------------------------------------------- #
# realignment
# --------------------------------------------------------------------------- #
def all_blocks(**overrides) -> Constraints:
    """A constraint set carrying every optional block, so realignment touches each one."""
    kwargs = dict(
        benchmark_weights=BENCHMARK,
        weights_0=WEIGHTS_0,
        asset_returns=pd.Series([0.09, 0.06, 0.02], index=TICKERS),
        turnover_constraint=0.5,
        turnover_costs=pd.Series(1.0, index=TICKERS),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=GROUP_LOADINGS,
            group_min_allocation=pd.Series({'risky': 0.0, 'safe': 0.0}),
            group_max_allocation=pd.Series({'risky': 1.0, 'safe': 1.0})),
        group_tracking_error_constraint=GroupTrackingErrorConstraint(
            group_loadings=GROUP_LOADINGS,
            group_tre_vols=pd.Series({'risky': 0.05, 'safe': 0.05})),
        group_turnover_constraint=GroupTurnoverConstraint(
            group_loadings=GROUP_LOADINGS,
            group_max_turnover=pd.Series({'risky': 0.5, 'safe': 0.5})),
        sector_deviation_constraints=BenchmarkDeviationConstraints(
            factor_loading_mat=GROUP_LOADINGS,
            factor_max_deviation=pd.Series({'risky': 0.1, 'safe': 0.1})),
        style_deviation_constraints=BenchmarkDeviationConstraints(
            factor_loading_mat=GROUP_LOADINGS,
            factor_max_deviation=pd.Series({'risky': 0.1, 'safe': 0.1})),
        benchmark_beta_constraint=BenchmarkBetaConstraint(
            beta_max=1.2, beta_loadings=pd.Series([1.2, 1.0, 0.3], index=TICKERS)))
    kwargs.update(overrides)
    return make_constraints(**kwargs)


def test_update_realigns_every_optional_block_together() -> None:
    """one dropped asset has to disappear from all six loadings tables at once

    A block that keeps the dropped asset applies its bound over a membership the solver no
    longer has, and the constraint row then has the wrong length or the wrong sum. Nothing
    raises: the group limit simply stops meaning what the mandate says.
    """
    survivors = ['growth', 'balanced']
    updated = all_blocks().update(valid_tickers=survivors)
    assert list(updated.group_lower_upper_constraints.group_loadings.index) == survivors
    assert list(updated.group_tracking_error_constraint.group_loadings.index) == survivors
    assert list(updated.group_turnover_constraint.group_loadings.index) == survivors
    assert list(updated.sector_deviation_constraints.factor_loading_mat.index) == survivors
    assert list(updated.style_deviation_constraints.factor_loading_mat.index) == survivors
    assert list(updated.benchmark_beta_constraint.beta_loadings.index) == survivors


def test_update_with_valid_tickers_carries_the_stored_universe_forward() -> None:
    """with nothing passed in, the constraint set realigns what it already holds

    The rolling wrappers pass a fresh ``weights_0`` each date but leave the benchmark and the
    expected returns on the constraint set. Those still have to follow the surviving universe,
    or a later Series-indexed operation lines up the wrong assets.
    """
    survivors = ['growth', 'balanced']
    updated = all_blocks().update_with_valid_tickers(valid_tickers=survivors)
    assert list(updated.weights_0.index) == survivors
    assert list(updated.asset_returns.index) == survivors
    assert list(updated.benchmark_weights.index) == survivors
    assert list(updated.turnover_costs.index) == survivors
    assert list(updated.group_turnover_constraint.group_loadings.index) == survivors


def test_update_with_valid_tickers_rescales_the_turnover_budget() -> None:
    """a turnover cap stated over the whole universe is widened for the surviving part

    ``total_to_good_ratio`` is how the constraint set keeps its intent when assets drop: the
    remaining names carry the whole portfolio, so the same trade is a larger share of it.
    """
    updated = all_blocks().update_with_valid_tickers(
        valid_tickers=['growth', 'balanced'], total_to_good_ratio=1.5)
    assert updated.turnover_constraint == pytest.approx(0.75)


def test_update_min_max_weights_realigns_the_new_box_onto_the_old_index() -> None:
    """a replacement box is reindexed onto the universe it is replacing

    A caller supplying a box for a subset would otherwise silently drop every asset it did
    not mention — which reads as "no bound" but is actually "asset removed from the box".
    """
    constraints = make_constraints()
    updated = constraints.update_min_max_weights(
        min_weights=pd.Series({'growth': 0.1}),
        max_weights=pd.Series({'growth': 0.5}))
    assert list(updated.min_weights.index) == TICKERS
    assert updated.min_weights['growth'] == 0.1
    assert updated.min_weights['balanced'] == 0.0       # filled, not dropped
    assert updated.max_weights['growth'] == 0.5


def test_update_min_max_weights_accepts_a_box_where_there_was_none() -> None:
    """with no existing box there is no index to align to, and the input is taken as given"""
    bare = Constraints(is_long_only=True)
    updated = bare.update_min_max_weights(min_weights=pd.Series(0.05, index=TICKERS),
                                          max_weights=pd.Series(0.6, index=TICKERS))
    assert list(updated.min_weights.index) == TICKERS
    assert updated.max_weights['defensive'] == 0.6


def test_a_group_whose_members_all_dropped_is_left_alone_in_the_frozen_overhang_pass():
    """the relaxation loop skips a group with no members rather than dividing by nothing

    The frozen-overhang pass widens a group bound that frozen positions have already
    exceeded. A group that reindexed to an empty membership has no overhang to measure, and
    computing one would relax a bound on the strength of a sum over zero assets.
    """
    loadings = pd.DataFrame({'alts': [0.0, 0.0, 0.0, 1.0], 'core': [1.0, 1.0, 1.0, 0.0]},
                            index=TICKERS + ['private'])
    constraints = make_constraints(
        weights_0=WEIGHTS_0,
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=pd.Series({'alts': 0.0, 'core': 0.0}),
            group_max_allocation=pd.Series({'alts': 1.0, 'core': 1.0})))
    updated = constraints.update_with_valid_tickers(
        valid_tickers=TICKERS,
        rebalancing_indicators=pd.Series([0.0, 1.0, 1.0], index=TICKERS))
    # the group had no members left, so it carries no bound to relax — and no bound at all
    assert 'alts' not in updated.group_lower_upper_constraints.group_loadings.columns
    assert updated.group_lower_upper_constraints.group_max_allocation['core'] == 1.0


# --------------------------------------------------------------------------- #
# building the cvx problem
# --------------------------------------------------------------------------- #
def test_an_exposure_band_survives_the_charnes_cooper_rescaling() -> None:
    """the maximum-Sharpe change of variables must keep both ends of the band

    ``max_sharpe`` solves in ``y = k w`` with a free scale ``k``, so an exposure *band* has to
    become ``k*min <= sum(y) <= k*max``. Collapsing it to the equality that the fully invested
    case uses would silently forbid every portfolio in the interior of the band.
    """
    constraints = make_constraints(min_exposure=0.5, max_exposure=1.0)
    w = cvx.Variable(len(TICKERS), nonneg=True)
    scaler = cvx.Variable(nonneg=True)
    emitted = constraints.set_cvx_exposure_constraints(w=w, exposure_scaler=scaler)
    solved = solve_under(w, list(emitted) + [scaler == 1.0,
                                             cvx.sum(w) == 0.75])   # inside the band
    assert solved is not None
    assert float(np.sum(solved)) == pytest.approx(0.75, abs=1e-6)


def test_a_volatility_cap_without_a_covariance_is_rejected() -> None:
    """the constraint is a quadratic form, so it cannot be built without the matrix"""
    constraints = make_constraints(max_target_portfolio_vol_an=0.12)
    with pytest.raises(ValueError, match='covar must be given'):
        constraints.set_cvx_all_constraints(w=cvx.Variable(len(TICKERS)))


def test_a_volatility_floor_needs_a_covariance_too() -> None:
    """and the minimum side, which has no factorised form to fall back on"""
    constraints = make_constraints(min_target_portfolio_vol_an=0.05)
    with pytest.raises(ValueError, match='covar must be given'):
        constraints.set_cvx_all_constraints(w=cvx.Variable(len(TICKERS)))


def test_a_volatility_floor_is_emitted_as_a_quadratic_lower_bound() -> None:
    """the floor is built, even though no convex solver will take it

    ``w'Sigma w >= v^2`` is not a convex set, so cvxpy rejects the problem at solve time. The
    constraint is emitted all the same — a caller asking for a minimum volatility gets a
    DCP error naming the term, not a silently dropped mandate.
    """
    constraints = make_constraints(min_target_portfolio_vol_an=0.12)
    w = cvx.Variable(len(TICKERS), nonneg=True)
    emitted = constraints.set_cvx_all_constraints(w=w, covar=cvx.psd_wrap(COVAR.to_numpy()))
    assert any('QuadForm' in str(constraint) for constraint in emitted)
    with pytest.raises(cvx.error.DCPError):
        cvx.Problem(cvx.Minimize(cvx.sum_squares(w)), emitted).solve(solver='CLARABEL')


def test_a_group_turnover_block_takes_precedence_over_the_portfolio_cap() -> None:
    """with both stated, the group form is the one built — they are not additive"""
    constraints = make_constraints(
        weights_0=WEIGHTS_0, turnover_constraint=0.01,
        group_turnover_constraint=GroupTurnoverConstraint(
            group_loadings=GROUP_LOADINGS,
            group_max_turnover=pd.Series({'risky': 1.0, 'safe': 1.0})))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    emitted = constraints.set_cvx_all_constraints(w=w, covar=cvx.psd_wrap(COVAR.to_numpy()))
    solved = solve_under(w, emitted)
    # the 1% portfolio cap would have pinned the solve to weights_0; the group caps did not
    assert not np.allclose(solved, WEIGHTS_0.to_numpy(), atol=1e-3)


def test_a_turnover_cap_without_a_prior_portfolio_is_skipped(caplog) -> None:
    """the first rebalancing has nothing to trade from, so the cap does not apply yet"""
    constraints = make_constraints(turnover_constraint=0.01)
    w = cvx.Variable(len(TICKERS), nonneg=True)
    with caplog.at_level(logging.DEBUG,
                         logger='optimalportfolios.optimization.constraints'):
        constraints.set_cvx_all_constraints(w=w, covar=cvx.psd_wrap(COVAR.to_numpy()))
    assert 'turnover constraint skipped' in caplog.text


def test_per_asset_turnover_costs_weight_the_trade_budget() -> None:
    """a costed cap trades less where trading is dearer, at the same nominal budget"""
    w = cvx.Variable(len(TICKERS), nonneg=True)
    costed = make_constraints(
        weights_0=WEIGHTS_0, turnover_constraint=0.10,
        turnover_costs=pd.Series([5.0, 1.0, 1.0], index=TICKERS))
    solved = solve_under(w, costed.set_cvx_all_constraints(
        w=w, covar=cvx.psd_wrap(COVAR.to_numpy())))
    # 'growth' costs five times as much to trade, so it barely moves off weights_0
    assert abs(solved[0] - WEIGHTS_0['growth']) < 0.03


def test_a_tracking_error_cap_without_a_benchmark_is_rejected() -> None:
    """tracking error is measured against something, and there is nothing to measure against"""
    constraints = make_constraints(tracking_err_vol_constraint=0.03)
    with pytest.raises(ValueError, match='benchmark_weights must be given'):
        constraints.set_cvx_all_constraints(w=cvx.Variable(len(TICKERS)),
                                            covar=cvx.psd_wrap(COVAR.to_numpy()))


def test_style_deviation_bounds_reach_the_cvx_problem() -> None:
    """the style block is emitted alongside the sector block, not instead of it"""
    deviations = BenchmarkDeviationConstraints(
        factor_loading_mat=GROUP_LOADINGS,
        factor_max_deviation=pd.Series({'risky': 0.02, 'safe': 0.02}))
    constraints = make_constraints(benchmark_weights=BENCHMARK,
                                   style_deviation_constraints=deviations)
    w = cvx.Variable(len(TICKERS), nonneg=True)
    emitted = constraints.set_cvx_all_constraints(w=w, covar=cvx.psd_wrap(COVAR.to_numpy()))
    solved = solve_under(w, emitted)
    active_risky = float(solved[:2].sum() - BENCHMARK[:2].sum())
    assert abs(active_risky) <= 0.02 + 1e-5


# --------------------------------------------------------------------------- #
# the utility formulation
# --------------------------------------------------------------------------- #
def utility_constraints(**overrides) -> Constraints:
    """A penalised constraint set carrying a benchmark, with overrides."""
    kwargs = dict(benchmark_weights=BENCHMARK)
    kwargs.update(overrides)
    return make_constraints(**kwargs)


def test_the_utility_objective_penalises_group_turnover_when_configured() -> None:
    """a group turnover block becomes a penalty term rather than a hard cap"""
    constraints = utility_constraints(
        weights_0=WEIGHTS_0,
        group_turnover_constraint=GroupTurnoverConstraint(
            group_loadings=GROUP_LOADINGS,
            group_turnover_utility_weights=pd.Series({'risky': 50.0, 'safe': 50.0})))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    objective, emitted = constraints.set_cvx_utility_objective_constraints(
        w=w, alphas=np.array([0.09, 0.06, 0.02]), covar=cvx.psd_wrap(COVAR.to_numpy()))
    problem = cvx.Problem(cvx.Maximize(objective), emitted)
    problem.solve(solver='CLARABEL')
    # the penalty is heavy enough that the solve stays close to where it started
    assert np.abs(w.value - WEIGHTS_0.to_numpy()).sum() < 0.30


def test_the_utility_objective_without_a_prior_portfolio_skips_the_turnover_term(caplog):
    """with no weights_0 there is no trade to penalise, on either turnover form"""
    constraints = utility_constraints(
        group_turnover_constraint=GroupTurnoverConstraint(
            group_loadings=GROUP_LOADINGS,
            group_turnover_utility_weights=pd.Series({'risky': 1.0, 'safe': 1.0})))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    with caplog.at_level(logging.DEBUG,
                         logger='optimalportfolios.optimization.constraints'):
        constraints.set_cvx_utility_objective_constraints(
            w=w, alphas=np.array([0.09, 0.06, 0.02]), covar=cvx.psd_wrap(COVAR.to_numpy()))
    assert 'group turnover utility skipped' in caplog.text


def test_the_utility_objective_applies_per_asset_turnover_costs() -> None:
    """the portfolio-level penalty is cost-weighted the same way the hard cap is"""
    constraints = utility_constraints(
        weights_0=WEIGHTS_0, turnover_utility_weight=5.0,
        turnover_costs=pd.Series([20.0, 1.0, 1.0], index=TICKERS))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    objective, emitted = constraints.set_cvx_utility_objective_constraints(
        w=w, alphas=np.array([0.09, 0.06, 0.02]), covar=cvx.psd_wrap(COVAR.to_numpy()))
    cvx.Problem(cvx.Maximize(objective), emitted).solve(solver='CLARABEL')
    assert abs(w.value[0] - WEIGHTS_0['growth']) < 0.05


def test_the_utility_objective_needs_a_benchmark_for_a_group_tre_penalty() -> None:
    """a tracking-error penalty against nothing is not a penalty"""
    constraints = make_constraints(
        group_tracking_error_constraint=GroupTrackingErrorConstraint(
            group_loadings=GROUP_LOADINGS,
            group_tre_utility_weights=pd.Series({'risky': 1.0, 'safe': 1.0})))
    with pytest.raises(ValueError, match='benchmark_weights must be given'):
        constraints.set_cvx_utility_objective_constraints(
            w=cvx.Variable(len(TICKERS)), covar=cvx.psd_wrap(COVAR.to_numpy()))


def test_the_utility_objective_needs_a_benchmark_for_the_portfolio_tre_penalty() -> None:
    """the same requirement on the portfolio-level penalty, which is the default one"""
    constraints = make_constraints(tre_utility_weight=1.0)
    with pytest.raises(ValueError, match='benchmark_weights must be given'):
        constraints.set_cvx_utility_objective_constraints(
            w=cvx.Variable(len(TICKERS)), covar=cvx.psd_wrap(COVAR.to_numpy()))


def test_the_utility_objective_needs_asset_returns_for_a_return_floor() -> None:
    """the return floor stays hard under the penalty formulation, so it needs its inputs"""
    constraints = utility_constraints(target_return=0.05)
    with pytest.raises(ValueError, match='asset_returns must be given'):
        constraints.set_cvx_utility_objective_constraints(
            w=cvx.Variable(len(TICKERS)), covar=cvx.psd_wrap(COVAR.to_numpy()))


def test_group_bounds_and_the_beta_range_stay_hard_under_the_penalty_formulation() -> None:
    """policy limits are constraints even when risk and turnover are penalties

    The utility formulation exists so a solve always returns *something*; a mandate limit is
    not something to trade off against alpha, so both of these are emitted as constraints.
    """
    constraints = utility_constraints(
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=GROUP_LOADINGS,
            group_min_allocation=pd.Series({'risky': 0.0, 'safe': 0.0}),
            group_max_allocation=pd.Series({'risky': 0.4, 'safe': 1.0})),
        benchmark_beta_constraint=BenchmarkBetaConstraint(
            beta_max=0.8, beta_loadings=pd.Series([1.2, 1.0, 0.3], index=TICKERS)))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    objective, emitted = constraints.set_cvx_utility_objective_constraints(
        w=w, alphas=np.array([0.09, 0.06, 0.02]), covar=cvx.psd_wrap(COVAR.to_numpy()))
    cvx.Problem(cvx.Maximize(objective), emitted).solve(solver='CLARABEL')
    solved = w.value
    assert solved[:2].sum() <= 0.4 + 1e-5, 'the group cap was traded away'
    beta = float(np.array([1.2, 1.0, 0.3]) @ solved)
    assert beta <= 0.8 + 1e-5, 'the beta range was traded away'


# --------------------------------------------------------------------------- #
# the quadprog / pyrb translation
# --------------------------------------------------------------------------- #
def test_pyrb_constraints_report_no_rows_when_no_group_bounds_are_stated() -> None:
    """the risk-budgeting backend takes ``None`` for "no linear rows", not an empty array

    An empty ``(0, n)`` array would be a valid matrix and the solver would build a constraint
    block out of it; ``None`` is what its API reads as "unconstrained".
    """
    bounds, c_rows, c_lhs = make_constraints().set_pyrb_constraints(covar=COVAR.to_numpy())
    assert c_rows is None and c_lhs is None
    assert len(bounds) == len(TICKERS)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
