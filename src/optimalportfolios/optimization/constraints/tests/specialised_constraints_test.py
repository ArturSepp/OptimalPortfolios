"""
the specialised constraint classes: group TRE, group turnover, deviation and beta.

``constraints_test.py`` covers ``Constraints`` and ``GroupLowerUpperConstraints``. The four
classes here — ``GroupTrackingErrorConstraint``, ``GroupTurnoverConstraint``,
``BenchmarkDeviationConstraints`` and ``BenchmarkBetaConstraint`` — had no collected tests
at all.

They share a shape worth testing as one: each validates on construction, each has an
``update(valid_tickers)`` that realigns it when the solver drops assets, and each emits cvx
constraints. The realignment is the part that bites — an asset dropped for a NaN covariance
row has to disappear from every loadings matrix too, and a class that forgets silently
applies a group bound computed over the wrong membership.

Pure benchmark-beta helper analytics live in
``optimalportfolios/utils/tests/benchmark_beta_test.py``. This module uses their canonical
joint-covariance helper only to construct loadings for ``BenchmarkBetaConstraint`` tests.
"""
# packages
import numpy as np
import pandas as pd
import pytest
import cvxpy as cvx
# optimalportfolios
from optimalportfolios.optimization.constraints import (
    BenchmarkBetaConstraint,
    BenchmarkDeviationConstraints,
    GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
    merge_group_lower_upper_constraints,
)
from optimalportfolios.utils.benchmark_beta import compute_benchmark_beta_loadings_from_covar

TICKERS = ['growth', 'balanced', 'defensive']
VOLS = np.array([0.22, 0.14, 0.06])
CORR = np.array([[1.00, 0.45, 0.15],
                 [0.45, 1.00, 0.25],
                 [0.15, 0.25, 1.00]])
COVAR = pd.DataFrame(np.outer(VOLS, VOLS) * CORR, index=TICKERS, columns=TICKERS)
GROUP_LOADINGS = pd.DataFrame({'risky': [1.0, 1.0, 0.0], 'safe': [0.0, 0.0, 1.0]},
                              index=TICKERS)
BENCHMARK = pd.Series([0.4, 0.35, 0.25], index=TICKERS)


def solve_with(w: cvx.Variable, constraint_list, objective=None) -> np.ndarray:
    """Solve a trivial fully invested problem under the supplied constraints.

    ``w`` must be the *same* variable the constraints were built against — a fresh one here
    would leave them referencing a different variable, and the solve would silently ignore
    every constraint under test.
    """
    extra = [cvx.sum(w) == 1.0]
    problem = cvx.Problem(objective(w) if objective else cvx.Minimize(cvx.sum_squares(w)),
                          list(constraint_list) + extra)
    problem.solve(solver='CLARABEL')
    return w.value


# --------------------------------------------------------------------------- #
# GroupTrackingErrorConstraint
# --------------------------------------------------------------------------- #
def test_group_tre_requires_either_vols_or_utility_weights() -> None:
    """the class enforces one of the two enforcement styles rather than defaulting"""
    with pytest.raises(ValueError, match='group_tre_vols or group_tre_utility_weights'):
        GroupTrackingErrorConstraint(group_loadings=GROUP_LOADINGS)


def test_group_tre_warns_when_a_group_has_no_bound() -> None:
    """a loadings column with no matching bound would be silently unconstrained"""
    with pytest.warns(UserWarning, match='Missing in group_loadings.columns'):
        GroupTrackingErrorConstraint(group_loadings=GROUP_LOADINGS,
                                     group_tre_vols=pd.Series({'risky': 0.05}))


def test_group_tre_validates_utility_weights_when_hard_bounds_are_also_present() -> None:
    """both configured enforcement styles must cover every group independently"""
    with pytest.warns(UserWarning, match='Missing in group_loadings.columns'):
        GroupTrackingErrorConstraint(
            group_loadings=GROUP_LOADINGS,
            group_tre_vols=pd.Series({'risky': 0.05, 'safe': 0.03}),
            group_tre_utility_weights=pd.Series({'risky': 1.0}),
        )


def test_group_tre_update_realigns_the_loadings_to_the_surviving_assets() -> None:
    """dropping an asset must drop its row, or the group bound covers the wrong members"""
    constraint = GroupTrackingErrorConstraint(
        group_loadings=GROUP_LOADINGS,
        group_tre_vols=pd.Series({'risky': 0.05, 'safe': 0.03}))
    aligned = constraint.update(valid_tickers=['growth', 'defensive'])
    assert list(aligned.group_loadings.index) == ['growth', 'defensive']
    assert aligned.group_tre_vols.equals(constraint.group_tre_vols)


def test_group_tre_hard_constraints_bind_the_active_risk() -> None:
    """a tight group TRE budget keeps the solve near the benchmark within that group"""
    constraint = GroupTrackingErrorConstraint(
        group_loadings=GROUP_LOADINGS,
        group_tre_vols=pd.Series({'risky': 0.005, 'safe': 0.005}))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    cvx_constraints = constraint.set_cvx_group_tre_constraints(
        w=w, benchmark_weights=BENCHMARK, covar=COVAR.to_numpy())
    assert cvx_constraints
    weights = solve_with(w, cvx_constraints,
                         objective=lambda v: cvx.Maximize(v[0]))  # push away from benchmark
    assert weights is not None
    assert np.abs(weights - BENCHMARK.to_numpy()).max() < 0.3


def test_group_tre_utility_requires_utility_weights() -> None:
    """the soft form needs its weights; asking for it without them raises"""
    constraint = GroupTrackingErrorConstraint(
        group_loadings=GROUP_LOADINGS, group_tre_vols=pd.Series({'risky': 0.05, 'safe': 0.03}))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    with pytest.raises(ValueError, match='group_tre_utility_weights'):
        constraint.set_cvx_group_tre_utility(w=w, benchmark_weights=BENCHMARK,
                                             covar=COVAR.to_numpy())


def test_group_tre_utility_builds_a_penalty_term() -> None:
    """the soft form returns an objective contribution rather than constraints"""
    constraint = GroupTrackingErrorConstraint(
        group_loadings=GROUP_LOADINGS,
        group_tre_utility_weights=pd.Series({'risky': 10.0, 'safe': 5.0}))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    term = constraint.set_cvx_group_tre_utility(w=w, benchmark_weights=BENCHMARK,
                                                covar=COVAR.to_numpy())
    assert term is not None


def test_group_tre_print_names_its_fields(capsys) -> None:
    """the diagnostic print exists for notebooks and must not raise"""
    GroupTrackingErrorConstraint(group_loadings=GROUP_LOADINGS,
                                 group_tre_vols=pd.Series({'risky': 0.05,
                                                           'safe': 0.03})).print()
    assert 'group_tre' in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# GroupTurnoverConstraint
# --------------------------------------------------------------------------- #
def test_group_turnover_requires_either_a_cap_or_utility_weights() -> None:
    """same either/or contract as the group TRE constraint"""
    with pytest.raises(ValueError, match='group_max_turnover or '
                                         'group_turnover_utility_weights'):
        GroupTurnoverConstraint(group_loadings=GROUP_LOADINGS)


def test_group_turnover_warns_when_a_group_has_no_cap() -> None:
    """an unbounded group would trade freely while its neighbours are capped"""
    with pytest.warns(UserWarning, match='Missing in self.group_loadings.columns'):
        GroupTurnoverConstraint(group_loadings=GROUP_LOADINGS,
                                group_max_turnover=pd.Series({'risky': 0.10}))


def test_group_turnover_validates_utility_weights_when_hard_caps_are_also_present() -> None:
    """both configured enforcement styles must cover every group independently"""
    with pytest.warns(UserWarning, match='Missing in self.group_loadings.columns'):
        GroupTurnoverConstraint(
            group_loadings=GROUP_LOADINGS,
            group_max_turnover=pd.Series({'risky': 0.10, 'safe': 0.05}),
            group_turnover_utility_weights=pd.Series({'risky': 1.0}),
        )


def test_group_turnover_update_realigns_the_loadings() -> None:
    """dropped assets leave the loadings, keeping the cap over the right membership"""
    constraint = GroupTurnoverConstraint(
        group_loadings=GROUP_LOADINGS,
        group_max_turnover=pd.Series({'risky': 0.10, 'safe': 0.05}))
    aligned = constraint.update(valid_tickers=['balanced', 'defensive'])
    assert list(aligned.group_loadings.index) == ['balanced', 'defensive']


def test_group_turnover_caps_trading_within_a_group() -> None:
    """the cap binds: the solve cannot move the risky group further than allowed"""
    weights_0 = pd.Series([0.4, 0.35, 0.25], index=TICKERS)
    constraint = GroupTurnoverConstraint(
        group_loadings=GROUP_LOADINGS,
        group_max_turnover=pd.Series({'risky': 0.02, 'safe': 1.0}))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    cvx_constraints = constraint.set_group_turnover_constraints(w=w, weights_0=weights_0)
    weights = solve_with(w, cvx_constraints, objective=lambda v: cvx.Maximize(v[0]))
    assert weights is not None
    risky_turnover = np.abs(weights[:2] - weights_0.to_numpy()[:2]).sum()
    assert risky_turnover <= 0.02 + 1e-5


def test_group_turnover_utility_builds_a_penalty_term() -> None:
    """the soft form returns an objective contribution"""
    constraint = GroupTurnoverConstraint(
        group_loadings=GROUP_LOADINGS,
        group_turnover_utility_weights=pd.Series({'risky': 5.0, 'safe': 2.0}))
    w = cvx.Variable(len(TICKERS), nonneg=True)
    term = constraint.set_cvx_group_turnover_utility(
        w=w, weights_0=pd.Series([0.4, 0.35, 0.25], index=TICKERS))
    assert term is not None


def test_group_turnover_print_names_its_fields(capsys) -> None:
    """the diagnostic print must not raise"""
    GroupTurnoverConstraint(group_loadings=GROUP_LOADINGS,
                            group_max_turnover=pd.Series({'risky': 0.1,
                                                          'safe': 0.1})).print()
    assert capsys.readouterr().out != ''


# --------------------------------------------------------------------------- #
# BenchmarkDeviationConstraints
# --------------------------------------------------------------------------- #
def make_deviation(**overrides) -> BenchmarkDeviationConstraints:
    """A sector-style deviation constraint over the standard loadings."""
    kwargs = dict(factor_loading_mat=GROUP_LOADINGS,
                  factor_max_deviation=pd.Series({'risky': 0.05, 'safe': 0.05}))
    kwargs.update(overrides)
    return BenchmarkDeviationConstraints(**kwargs)


def test_deviation_constraints_require_a_bound() -> None:
    """a deviation constraint with no maximum constrains nothing"""
    with pytest.raises(ValueError, match='factor_max_deviation must be given'):
        BenchmarkDeviationConstraints(factor_loading_mat=GROUP_LOADINGS,
                                      factor_max_deviation=None)


def test_deviation_constraints_warn_on_an_unknown_factor() -> None:
    """a bound naming a factor the loadings do not have would be silently ignored"""
    with pytest.warns(UserWarning, match='not in factor_loading_mat.columns'):
        make_deviation(factor_max_deviation=pd.Series({'risky': 0.05, 'nonsense': 0.05}))


def test_deviation_copy_is_independent() -> None:
    """the copy must not alias, or a per-rebalance edit would leak backwards"""
    original = make_deviation()
    duplicate = original.copy()
    duplicate.factor_loading_mat.iloc[0, 0] = 99.0
    assert original.factor_loading_mat.iloc[0, 0] == 1.0


def test_deviation_print_names_its_fields(capsys) -> None:
    """the diagnostic print must not raise"""
    make_deviation().print()
    assert capsys.readouterr().out != ''


# --------------------------------------------------------------------------- #
# BenchmarkBetaConstraint
# --------------------------------------------------------------------------- #
def test_beta_constraint_requires_a_bound() -> None:
    """a range with neither end constrains nothing"""
    with pytest.raises(ValueError, match='at least one of beta_min / beta_max'):
        BenchmarkBetaConstraint()


def test_beta_constraint_rejects_an_inverted_range() -> None:
    """min above max is always a typo, and would be infeasible at solve time"""
    with pytest.raises(ValueError, match='beta_min=1.2 > beta_max=0.8'):
        BenchmarkBetaConstraint(beta_min=1.2, beta_max=0.8)


def test_beta_constraint_needs_loadings_before_it_can_be_built() -> None:
    """the loadings are per-rebalance state, so building without them must raise"""
    with pytest.raises(ValueError, match='beta_loadings not set'):
        BenchmarkBetaConstraint(beta_min=0.9,
                                beta_max=1.1).set_cvx_beta_constraints(
            w=cvx.Variable(len(TICKERS)))


def test_beta_constraint_with_loadings_keeps_the_bounds() -> None:
    """injecting this rebalance's loadings does not disturb the static spec"""
    spec = BenchmarkBetaConstraint(beta_min=0.9, beta_max=1.1)
    loadings = compute_benchmark_beta_loadings_from_covar(
        covar=COVAR, benchmark_weights=BENCHMARK, asset_tickers=TICKERS)
    live = spec.with_loadings(loadings)
    assert live.beta_min == 0.9 and live.beta_max == 1.1
    assert live.beta_loadings is not None
    assert spec.beta_loadings is None, "with_loadings must not mutate the static spec"


def test_beta_constraint_copy_is_independent() -> None:
    """a copy carries the loadings without aliasing them"""
    loadings = pd.Series([1.0, 0.8, 0.2], index=TICKERS)
    original = BenchmarkBetaConstraint(beta_min=0.9, beta_loadings=loadings)
    duplicate = original.copy()
    duplicate.beta_loadings.iloc[0] = 99.0
    assert original.beta_loadings.iloc[0] == 1.0


def test_beta_constraint_update_zero_fills_dropped_assets() -> None:
    """a dropped asset carries zero weight, so a zero loading is the right alignment"""
    loadings = pd.Series([1.0, 0.8, 0.2], index=TICKERS)
    aligned = BenchmarkBetaConstraint(beta_min=0.9,
                                      beta_loadings=loadings).update(
        valid_tickers=['growth', 'balanced', 'newcomer'])
    assert aligned.beta_loadings['newcomer'] == 0.0
    assert aligned.beta_loadings['growth'] == 1.0


def test_beta_constraint_update_without_loadings_is_a_no_op() -> None:
    """a spec with no loadings yet has nothing to realign"""
    spec = BenchmarkBetaConstraint(beta_min=0.9)
    assert spec.update(valid_tickers=['growth']) is spec


def test_beta_constraint_binds_the_solved_portfolio_beta() -> None:
    """the constraint does what it says: the achieved beta lands inside the range"""
    loadings = compute_benchmark_beta_loadings_from_covar(
        covar=COVAR, benchmark_weights=BENCHMARK, asset_tickers=TICKERS)
    constraint = BenchmarkBetaConstraint(beta_min=0.0, beta_max=0.5,
                                         beta_loadings=loadings)
    w = cvx.Variable(len(TICKERS), nonneg=True)
    weights = solve_with(w, constraint.set_cvx_beta_constraints(w=w),
                         objective=lambda v: cvx.Maximize(v[0]))
    assert weights is not None
    assert float(loadings.to_numpy() @ weights) <= 0.5 + 1e-5


def test_beta_constraint_supports_a_one_sided_range() -> None:
    """only one bound need be given, and only that one is emitted"""
    loadings = pd.Series([1.0, 0.8, 0.2], index=TICKERS)
    w = cvx.Variable(len(TICKERS), nonneg=True)
    lower_only = BenchmarkBetaConstraint(beta_min=0.5, beta_loadings=loadings)
    upper_only = BenchmarkBetaConstraint(beta_max=0.9, beta_loadings=loadings)
    assert len(lower_only.set_cvx_beta_constraints(w=w)) == 1
    assert len(upper_only.set_cvx_beta_constraints(w=w)) == 1


def test_beta_constraint_print_names_its_fields(capsys) -> None:
    """the diagnostic print must not raise"""
    BenchmarkBetaConstraint(beta_min=0.9, beta_max=1.1,
                            beta_loadings=pd.Series([1.0, 0.8, 0.2],
                                                    index=TICKERS)).print()
    assert 'beta range' in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# merging group bounds
# --------------------------------------------------------------------------- #
def test_merging_group_constraints_unions_their_groups() -> None:
    """two constraint sets over different groupings combine into one"""
    first = GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS,
        group_min_allocation=pd.Series({'risky': 0.1, 'safe': 0.1}),
        group_max_allocation=pd.Series({'risky': 0.9, 'safe': 0.9}))
    second_loadings = pd.DataFrame({'domestic': [1.0, 0.0, 1.0], 'foreign': [0.0, 1.0, 0.0]},
                                   index=TICKERS)
    second = GroupLowerUpperConstraints(
        group_loadings=second_loadings,
        group_min_allocation=pd.Series({'domestic': 0.2, 'foreign': 0.0}),
        group_max_allocation=pd.Series({'domestic': 0.8, 'foreign': 1.0}))
    merged = merge_group_lower_upper_constraints(first, second)
    assert set(merged.group_loadings.columns) == {'risky', 'safe', 'domestic', 'foreign'}
    assert merged.group_min_allocation['domestic'] == 0.2
    assert merged.group_max_allocation['risky'] == 0.9
    assert list(merged.group_loadings.index) == TICKERS
