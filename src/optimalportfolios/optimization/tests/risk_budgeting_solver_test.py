"""Regression tests for ``optimization.risk_allocation.risk_budgeting_solver``.

Unconstrained Richard & Roncalli (2019) tables and frozen pyrb weights retain
their original parity bands. The 2026-09-10 full-investment correction replaces
the absolute-bound formulation for constrained portfolios: those old numbers
are retained below as migration comparators, NOT relabelled as new reference
solutions. Constrained cases now match an independent volatility/log conic solve
and improve the normalized homogeneous objective over their historical weights.
KKT tests include the full-investment multiplier; binding bounds no longer imply
equal RC/b on the other assets. Feasibility and inference remain separate tests.

Run with pytest and use ``-k`` to select paper-table, parity, or property cases.
"""
# packages
import numpy as np

# optimalportfolios
from optimalportfolios.optimization.risk_allocation.risk_budgeting_solver import (
    solve_constrained_risk_budgeting,
)
from optimalportfolios.optimization.tests.risk_budgeting_full_investment_test import (
    conic_reference,
)

PAPER_ATOL_PP = 0.01  # paper tables are rounded to 2 dp of percent
PYRB_PARITY_ATOL = 5e-5  # historical pyrb portfolio-weight parity band
GROUP_FEASIBILITY_ATOL = 1e-4  # conservative post-solve band on C x <= d


def _cov_from(vols: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Build a covariance from a vol vector and a correlation matrix."""
    vols = np.asarray(vols)
    return np.outer(vols, vols) * np.asarray(rho)


def _risk_contributions(x: np.ndarray, covar: np.ndarray) -> np.ndarray:
    """Per-asset risk contributions to portfolio vol (they sum to it)."""
    sigma = np.sqrt(x @ covar @ x)
    return x * (covar @ x) / sigma


def _assert_homogeneous_solution(x, covar, bounds=None, budgets=None,
                                 c_rows=None, c_lhs=None, legacy=None):
    """Check a different convex solver and, if supplied, the old formulation's delta."""
    n = len(x)
    budgets = np.ones(n) / n if budgets is None else budgets / budgets.sum()
    bounds = np.column_stack([np.zeros(n), np.ones(n)]) if bounds is None else bounds
    expected = conic_reference(covar, budgets, bounds, c_rows, c_lhs)
    np.testing.assert_allclose(x, expected, atol=2e-6)
    if legacy is not None:
        # Explicit objective comparison is a numerical-method reference, not risk analytics.
        value = 0.5 * np.log(x @ covar @ x) - budgets @ np.log(x)
        old_value = 0.5 * np.log(legacy @ covar @ legacy) - budgets @ np.log(legacy)
        assert value < old_value - 1e-8


def _normalized_gradient(x, covar, budgets):
    """Differentiate log(sigma(w)) - b'log(w) for an independent KKT check."""
    return covar @ x / (x @ covar @ x) - budgets / x


# -----------------------------------------------------------------------------
# problem set: the two covariance matrices of the paper's examples
# -----------------------------------------------------------------------------

# Section 3.1: four assets
COV4 = _cov_from(vols=np.array([0.10, 0.15, 0.20, 0.30]),
                 rho=np.array([[1.00, 0.50, 0.50, 0.50],
                               [0.50, 1.00, 0.50, 0.50],
                               [0.50, 0.50, 1.00, 0.75],
                               [0.50, 0.50, 0.75, 1.00]]))

# Section 3.4.1: five assets (dynamic allocation example)
COV5 = _cov_from(vols=np.array([0.15, 0.20, 0.25, 0.30, 0.10]),
                 rho=np.array([[1.00, 0.10, 0.40, 0.50, 0.50],
                               [0.10, 1.00, 0.70, 0.40, 0.40],
                               [0.40, 0.70, 1.00, 0.80, 0.05],
                               [0.50, 0.40, 0.80, 1.00, 0.10],
                               [0.50, 0.40, 0.05, 0.10, 1.00]]))

# Section 3.4.2: eight assets (multi-asset universe)
COV8 = _cov_from(vols=np.array([0.05, 0.05, 0.07, 0.10, 0.15, 0.15, 0.15, 0.18]),
                 rho=np.array([[1.00, 0.80, 0.60, -0.20, -0.10, -0.20, -0.20, -0.20],
                               [0.80, 1.00, 0.40, -0.20, -0.20, -0.10, -0.20, -0.20],
                               [0.60, 0.40, 1.00, 0.50, 0.30, 0.20, 0.20, 0.30],
                               [-0.20, -0.20, 0.50, 1.00, 0.60, 0.60, 0.50, 0.60],
                               [-0.10, -0.20, 0.30, 0.60, 1.00, 0.90, 0.70, 0.70],
                               [-0.20, -0.10, 0.20, 0.60, 0.90, 1.00, 0.60, 0.70],
                               [-0.20, -0.20, 0.20, 0.50, 0.70, 0.60, 1.00, 0.70],
                               [-0.20, -0.20, 0.30, 0.60, 0.70, 0.70, 0.70, 1.00]]))

# deterministic mixed problem: AR(1) correlation 0.4^|i-j|, vols 6% + 2% * i,
# budgets proportional to (1, ..., 10), box [0, 0.25], group cap x1..x5 <= 0.45
_IDX10 = np.arange(10)
COV10_AR1 = _cov_from(vols=0.06 + 0.02 * _IDX10,
                      rho=0.4 ** np.abs(np.subtract.outer(_IDX10, _IDX10)))
BUDGETS10 = (_IDX10 + 1.0) / np.sum(_IDX10 + 1.0)
BOUNDS10 = np.stack([np.zeros(10), np.full(10, 0.25)], axis=1)
C_ROWS10 = np.array([[1.0] * 5 + [0.0] * 5])
C_LHS10 = np.array([0.45])


def _make_stiff_constrained_case() -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic synthetic analogue of a large mandate solve."""
    n_assets = 16
    variances = np.array([
        8.106130830989492e-06, 6.570935704911838e-05, 3.2710663101885945e-05,
        1.7423290380245302e-02, 1.0699874805650802e-03, 2.1493966410752433e-03,
        2.6515691467230774e-04, 1.3199731201901538e-04, 4.035299180816822e-06,
        3.5000000000000003e-02, 5.326486450603552e-04, 8.673458504981366e-03,
        4.317719603808523e-03, 1.0e-06, 2.008805411386785e-06,
        1.6283639478700946e-05,
    ])
    loadings = np.array([
        [0.2985144698332214, -0.5350014137339273, -0.3070622870057959],
        [1.5080722715721342, -0.5822293778059019, -0.22812380385128453],
        [-0.7245145044162855, -0.5172493374907822, -0.30655788195456973],
        [0.2565516458675206, -0.2938560550257995, -0.3547084649887684],
        [-0.6168151525371501, 0.08958446690608873, -1.3441660008066356],
        [0.05218489930777669, 1.3100257323485311, -0.7669128788338543],
        [-0.04539173803348671, 2.817209016484243, -0.9926744097609564],
        [-1.621141524211868, -0.42680913157992273, 1.8069379803781687],
        [2.035115511784091, -1.2159872631305197, 0.5701405884703373],
        [0.00452143134313492, -0.5883707554353481, -1.9456978173547614],
        [1.0245839797071592, -0.8605383067300788, 0.29900768417420936],
        [0.08189396317057414, 1.8701812688972848, 1.1759947894135871],
        [-1.1553360632608776, -0.2623951558776075, 1.0368475527289776],
        [-1.333014964281339, -0.3922883318669356, 0.22757580070206257],
        [-1.6158261586823717, -1.2354581464367866, -1.7040770041170938],
        [0.28697587878613073, -0.5615144689797945, 0.9286482876454955],
    ])
    raw_correlation = loadings @ loadings.T + 10.0 * np.eye(n_assets)
    scale = np.sqrt(np.diag(raw_correlation))
    correlation = raw_correlation / np.outer(scale, scale)
    covar = np.outer(np.sqrt(variances), np.sqrt(variances)) * correlation

    budgets = np.array([
        1.1192741084681311e-02, 2.7986197728280482e-02, 7.5407772162258985e-03,
        4.0105943252015323e-02, 7.2502984184098394e-03, 1.2501628739332539e-03,
        9.4937018454929563e-06, 6.3287841856353097e-02, 8.2969932106055622e-01,
        6.7078271817916889e-03, 2.7951237403106501e-03, 4.9341332841446406e-08,
        3.9625081388181248e-06, 6.5108196619732058e-04, 2.3601589100096266e-06,
        1.5168179110176573e-03,
    ])
    bounds = np.column_stack([
        np.zeros(n_assets),
        np.array([
            0.20046001523893048, 0.24101779737573187, 0.2074148814254873,
            0.2399211423473894, 0.14861205075992007, 0.2989140849657521,
            0.14590321393747593, 0.24015520365334164, 0.13186385323086525,
            0.1357245066648552, 0.2772654544069495, 0.254936708019428,
            0.1373452698235738, 0.17060060799399707, 0.13263059682588957,
            0.2883972337396115,
        ]),
    ])

    c_rows = np.zeros((4, n_assets))
    memberships = (
        [3, 13, 9, 1, 11, 8, 10],
        [6, 11, 14, 10, 12, 4],
        [13, 6, 15, 11, 0, 12, 4, 3],
        [14, 8, 6, 3, 7, 12, 0, 4, 2],
    )
    for group, members in enumerate(memberships):
        c_rows[group, members] = 1.0
    c_lhs = np.array([
        0.5045899063573196, 0.38947002161405686,
        0.5705833071197635, 0.5738080066562662,
    ])
    return covar, budgets, bounds, c_rows, c_lhs


# -----------------------------------------------------------------------------
# frozen pyrb baselines (full float precision, generated by the removed fork)
# -----------------------------------------------------------------------------

PYRB_ERC4 = np.array([0.41014194629549444, 0.2734276970361358,
                      0.18985815982260562, 0.12657219685264187])
PYRB_RB4 = np.array([0.4505275283205357, 0.300351166413348,
                     0.1466837394085323, 0.10243756586596994])
PYRB_ERC4_BOX30 = np.array([0.3, 0.3, 0.23999997796455802, 0.16000002203677885])
PYRB_ERC8_ROW1 = np.array([0.25786482489349377, 0.27408251405631884,
                           0.09515309231261021, 0.07290469491235481,
                           0.0705796091051367, 0.0771258246017887,
                           0.09226265477328403, 0.06002678533955569])
PYRB_ERC8_ROW2 = np.array([0.24522886989692277, 0.28690165601698575,
                           0.09518134411982815, 0.07267512763193787,
                           0.06970921903781303, 0.07803691352039269,
                           0.0923086443786485, 0.05995822540993676])
PYRB_MIXED10 = np.array([0.06070571719316891, 0.08922865196895383,
                         0.09929677626508025, 0.10137119440421428,
                         0.09939434537668487, 0.11438718663031198,
                         0.10790562444997003, 0.10446651126242266,
                         0.10482175762424321, 0.11842223482126565])


# -----------------------------------------------------------------------------
# 1. paper tables
# -----------------------------------------------------------------------------

def test_paper_table1_erc():
    """Table 1, ERC column: weights and equal risk contributions."""
    x, _ = solve_constrained_risk_budgeting(covar=COV4)
    np.testing.assert_allclose(100.0 * x, [41.01, 27.34, 18.99, 12.66],
                               atol=PAPER_ATOL_PP)
    rc = _risk_contributions(x, COV4)
    np.testing.assert_allclose(rc / np.sum(rc), 0.25 * np.ones(4), atol=1e-6)


def test_paper_table1_rb():
    """Table 1, RB column with budgets (30, 30, 19.5, 20.5)%."""
    budgets = np.array([0.30, 0.30, 0.195, 0.205])
    x, _ = solve_constrained_risk_budgeting(covar=COV4, budgets=budgets)
    np.testing.assert_allclose(100.0 * x, [45.05, 30.04, 14.67, 10.24],
                               atol=PAPER_ATOL_PP)
    rc = _risk_contributions(x, COV4)
    np.testing.assert_allclose(rc / np.sum(rc), budgets, atol=1e-6)


def test_paper_table5_erc():
    """Table 5: ERC over the five-asset dynamic-allocation universe."""
    x, _ = solve_constrained_risk_budgeting(covar=COV5)
    np.testing.assert_allclose(100.0 * x, [22.40, 16.51, 12.03, 10.51, 38.54],
                               atol=PAPER_ATOL_PP)


def test_paper_table6_inputs_with_joint_full_investment():
    """Table 6's input now solves the homogeneous, not absolute-bound, objective."""
    x0 = np.array([0.25, 0.25, 0.10, 0.10, 0.30])
    bounds = np.stack([np.maximum(x0 - 0.05, 0.0), x0 + 0.05], axis=1)
    x, _ = solve_constrained_risk_budgeting(covar=COV5, bounds=bounds)
    _assert_homogeneous_solution(
        x, COV5, bounds=bounds,
        legacy=np.array([22.89, 20.00, 11.69, 10.42, 35.00]) / 100.0)


def test_paper_table9_erc_unconstrained():
    """Table 9 column 1: ERC over the eight-asset multi-asset universe."""
    x, _ = solve_constrained_risk_budgeting(covar=COV8)
    np.testing.assert_allclose(100.0 * x,
                               [26.83, 28.68, 11.41, 9.80, 5.61, 5.90, 6.66, 5.11],
                               atol=PAPER_ATOL_PP)


def test_paper_table9_equity_floor_with_joint_full_investment():
    """Table 9 column 2 inputs retain the 30% floor under the corrected objective."""
    c_rows = -np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    c_lhs = -np.array([0.30])
    x, _ = solve_constrained_risk_budgeting(covar=COV8, c_rows=c_rows, c_lhs=c_lhs)
    _assert_homogeneous_solution(x, COV8, c_rows=c_rows, c_lhs=c_lhs)


def test_paper_table9_two_rows_with_joint_full_investment():
    """Table 9 column 3 inputs retain the floor and euro-vs-US overweight row."""
    c_rows = np.array([[0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
                       [1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]])
    c_lhs = np.array([-0.30, -0.05])
    x, _ = solve_constrained_risk_budgeting(covar=COV8, c_rows=c_rows, c_lhs=c_lhs)
    _assert_homogeneous_solution(x, COV8, c_rows=c_rows, c_lhs=c_lhs)


# -----------------------------------------------------------------------------
# 2. frozen pyrb parity
# -----------------------------------------------------------------------------

def test_pyrb_parity_erc4():
    """Frozen pyrb parity: unconstrained ERC on the four-asset problem."""
    x, _ = solve_constrained_risk_budgeting(covar=COV4)
    np.testing.assert_allclose(x, PYRB_ERC4, atol=PYRB_PARITY_ATOL)


def test_pyrb_parity_rb4():
    """Frozen pyrb parity: tilted budgets on the four-asset problem."""
    x, _ = solve_constrained_risk_budgeting(covar=COV4,
                                            budgets=np.array([0.30, 0.30, 0.195, 0.205]))
    np.testing.assert_allclose(x, PYRB_RB4, atol=PYRB_PARITY_ATOL)


def test_pyrb_migration_erc4_box():
    """Four-asset 30% box caps now satisfy the homogeneous conic reference."""
    x, _ = solve_constrained_risk_budgeting(covar=COV4,
                                            bounds=np.array([[0.0, 0.30]] * 4))
    _assert_homogeneous_solution(x, COV4, bounds=np.array([[0.0, 0.30]] * 4),
                                 legacy=PYRB_ERC4_BOX30)


def test_pyrb_migration_erc8_one_row():
    """Retain the frozen eight-asset baseline as an explicit formulation comparator."""
    c_rows = -np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    c_lhs = -np.array([0.30])
    x, _ = solve_constrained_risk_budgeting(covar=COV8, c_rows=c_rows, c_lhs=c_lhs)
    _assert_homogeneous_solution(x, COV8, c_rows=c_rows, c_lhs=c_lhs,
                                 legacy=PYRB_ERC8_ROW1)


def test_pyrb_migration_erc8_two_rows():
    """Two group rows now describe the normalized portfolio, not an auxiliary scale."""
    c_rows = np.array([[0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
                       [1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]])
    c_lhs = np.array([-0.30, -0.05])
    x, _ = solve_constrained_risk_budgeting(covar=COV8, c_rows=c_rows, c_lhs=c_lhs)
    _assert_homogeneous_solution(x, COV8, c_rows=c_rows, c_lhs=c_lhs,
                                 legacy=PYRB_ERC8_ROW2)


def test_pyrb_migration_mixed10():
    """Tilted budgets, box and group cap match a separately solved convex program."""
    x, lambda_star = solve_constrained_risk_budgeting(covar=COV10_AR1,
                                                      budgets=BUDGETS10,
                                                      bounds=BOUNDS10,
                                                      c_rows=C_ROWS10,
                                                      c_lhs=C_LHS10)
    _assert_homogeneous_solution(x, COV10_AR1, bounds=BOUNDS10, budgets=BUDGETS10,
                                 c_rows=C_ROWS10, c_lhs=C_LHS10, legacy=PYRB_MIXED10)
    # In the homogeneous volatility/log lift the normalized multiplier IS volatility.
    np.testing.assert_allclose(lambda_star, np.sqrt(x @ COV10_AR1 @ x), atol=1e-12)


# -----------------------------------------------------------------------------
# 3. properties: feasibility, KKT, degeneracies, validation
# -----------------------------------------------------------------------------

def test_feasibility_and_kkt_mixed10():
    """Box, group row and full investment hold; free gradients share the budget dual."""
    x, _ = solve_constrained_risk_budgeting(covar=COV10_AR1, budgets=BUDGETS10,
                                            bounds=BOUNDS10,
                                            c_rows=C_ROWS10, c_lhs=C_LHS10)
    assert abs(float(np.sum(x)) - 1.0) < 1e-8
    assert np.all(x >= BOUNDS10[:, 0] - 1e-12)
    assert np.all(x <= BOUNDS10[:, 1] + 1e-12)
    assert float(np.max(C_ROWS10 @ x - C_LHS10)) < GROUP_FEASIBILITY_ATOL
    # Free gradients share the multiplier of sum(w)=1, rather than zero.
    gradient = _normalized_gradient(x, COV10_AR1, BUDGETS10)
    group_slack = float(C_LHS10[0] - C_ROWS10[0] @ x)
    free = (x > BOUNDS10[:, 0] + 1e-6) & (x < BOUNDS10[:, 1] - 1e-6)
    if group_slack < GROUP_FEASIBILITY_ATOL:  # binding: exclude group members
        free = free & (C_ROWS10[0] == 0.0)
    assert np.ptp(gradient[free]) < 1e-6


def test_stiff_constrained_problem_stays_fully_invested():
    """A stiff constrained solve must finish fully invested and feasible.

    This synthetic 16-asset problem has four overlapping group caps, highly tilted budgets,
    and a covariance condition number around 4e4. The old outer Brent solve could jump
    over full investment; the corrected projection represents the joint feasible set.
    """
    covar, budgets, bounds, c_rows, c_lhs = _make_stiff_constrained_case()
    weights, _ = solve_constrained_risk_budgeting(
        covar=covar,
        budgets=budgets,
        bounds=bounds,
        c_rows=c_rows,
        c_lhs=c_lhs,
    )
    assert abs(float(np.sum(weights)) - 1.0) < 1e-8
    assert float(np.max(c_rows @ weights - c_lhs)) <= 1e-7
    assert np.all(weights >= bounds[:, 0] - 1e-12)
    assert np.all(weights <= bounds[:, 1] + 1e-12)


def test_strict_ccd_tolerance_controls_free_asset_kkt_spread():
    """A stiff box solve retains a narrow normalized-gradient KKT spread."""
    covar, budgets, bounds, _c_rows, _c_lhs = _make_stiff_constrained_case()
    weights, _ = solve_constrained_risk_budgeting(
        covar=covar,
        budgets=budgets,
        bounds=bounds,
    )
    free = (
        (weights > bounds[:, 0] + 1e-6)
        & (weights < bounds[:, 1] - 1e-6)
    )
    gradient = _normalized_gradient(weights, covar, budgets)
    assert np.ptp(gradient[free]) < 1e-5


def test_kkt_box_only():
    """Box-constrained free gradients share the full-investment multiplier."""
    x, _ = solve_constrained_risk_budgeting(covar=COV4,
                                            bounds=np.array([[0.0, 0.30]] * 4))
    gradient = _normalized_gradient(x, COV4, np.full(4, 0.25))
    free = (x > 1e-6) & (x < 0.30 - 1e-6)
    assert int(np.sum(free)) == 2  # assets 3 and 4 are off the cap
    np.testing.assert_allclose(gradient[free][0], gradient[free][1], atol=1e-7)


def test_pinned_box_returns_pinned_vector():
    """lo == hi with sum one short-circuits to the pinned weights (no root-finding)."""
    pinned = np.array([0.4, 0.3, 0.2, 0.1])
    bounds = np.stack([pinned, pinned], axis=1)
    x, lambda_star = solve_constrained_risk_budgeting(covar=COV4, bounds=bounds)
    np.testing.assert_allclose(x, pinned, atol=1e-12)
    assert np.isnan(lambda_star)


def test_infeasible_box_raises():
    """sum of upper bounds below one is infeasible and must raise, not return."""
    bounds = np.array([[0.0, 0.10]] * 4)  # sum(hi) = 0.4 < 1
    try:
        solve_constrained_risk_budgeting(covar=COV4, bounds=bounds)
        raise AssertionError("expected ValueError for infeasible box")
    except ValueError:
        pass


def test_invalid_covar_raises():
    """NaN in the covariance must raise with the offending count in the message."""
    covar = COV4.copy()
    covar[0, 1] = np.nan
    try:
        solve_constrained_risk_budgeting(covar=covar)
        raise AssertionError("expected ValueError for NaN covariance")
    except ValueError as exc:
        assert "non-finite" in str(exc)


def test_zero_budget_asset_gets_zero_weight():
    """b_i = 0 drives x_i to its lower bound (paper: zero-budget exclusion)."""
    budgets = np.array([0.5, 0.5, 0.0, 0.0])
    x, _ = solve_constrained_risk_budgeting(covar=COV4, budgets=budgets)
    assert float(x[2]) < 1e-10 and float(x[3]) < 1e-10
    assert abs(float(np.sum(x)) - 1.0) < 1e-8


def test_zero_sum_budgets_raise_cleanly():
    """an all-zero budget raises ValueError naming the sum, with no numpy
    RuntimeWarning from a premature normalisation."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes a failure
        try:
            solve_constrained_risk_budgeting(covar=COV4, budgets=np.zeros(4))
            raise AssertionError("expected ValueError for zero-sum budgets")
        except ValueError as exc:
            assert "positive sum" in str(exc)


def test_negative_budget_raises():
    """A negative budget raises, naming the non-negativity requirement."""
    try:
        solve_constrained_risk_budgeting(covar=COV4,
                                         budgets=np.array([0.5, 0.5, 0.5, -0.5]))
        raise AssertionError("expected ValueError for a negative budget")
    except ValueError as exc:
        assert "non-negative" in str(exc)


def test_malformed_bounds_raise_value_error_not_index_error():
    """bounds of the wrong shape must raise ValueError (the caller's fallback
    contract catches ValueError only), not IndexError from premature slicing."""
    for bad_bounds in [np.array([0.0, 1.0, 0.0, 1.0]),  # (4,) instead of (4, 2)
                       np.array([[0.0, 1.0]] * 3)]:  # (3, 2) for n = 4
        try:
            solve_constrained_risk_budgeting(covar=COV4, bounds=bad_bounds)
            raise AssertionError(f"expected ValueError for bounds {bad_bounds.shape}")
        except ValueError as exc:
            assert "bounds must have shape" in str(exc)
