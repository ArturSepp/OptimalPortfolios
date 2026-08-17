"""Tests for group-aware and hierarchical risk allocation primitives."""

import numpy as np
import pandas as pd
import pytest

import optimalportfolios as opt
from optimalportfolios import Constraints
from optimalportfolios.optimization.risk_allocation import (
    compute_group_risk_budgets,
    compute_hierarchical_risk_parity_weights,
    rolling_risk_budgeting,
)
from optimalportfolios.optimization.general.risk_budgeting import (
    rolling_risk_budgeting as legacy_rolling_risk_budgeting,
)
from optimalportfolios.optimization.general.risk_budgeting_solver import (
    solve_constrained_risk_budgeting as legacy_risk_budgeting_solver,
)
from optimalportfolios.optimization.risk_allocation.risk_budgeting_solver import (
    solve_constrained_risk_budgeting,
)
from optimalportfolios.utils import compute_group_risk_contributions


ASSETS = ["a", "b", "c", "d", "e", "f"]


def test_risk_allocation_primitives_are_available_from_the_package_root() -> None:
    """The new public functions follow the package's root re-export convention."""
    assert opt.compute_group_risk_budgets is compute_group_risk_budgets
    assert opt.compute_group_risk_contributions is compute_group_risk_contributions
    assert (opt.compute_hierarchical_risk_parity_weights
            is compute_hierarchical_risk_parity_weights)


def test_legacy_general_risk_budgeting_imports_remain_compatible() -> None:
    """The canonical namespace move does not break existing direct imports."""
    assert legacy_rolling_risk_budgeting is rolling_risk_budgeting
    assert legacy_risk_budgeting_solver is solve_constrained_risk_budgeting


def test_equal_group_risk_budgets_are_split_equally_inside_each_group() -> None:
    """Exponent zero gives every available group the same aggregate budget."""
    groups = pd.Series(["x", "x", "y", "z", "z", "z"], index=ASSETS)
    budgets = compute_group_risk_budgets(groups, group_size_exponent=0.0)

    expected = pd.Series([1 / 6, 1 / 6, 1 / 3, 1 / 9, 1 / 9, 1 / 9], index=ASSETS,
                         name="risk_budget")
    pd.testing.assert_series_equal(budgets, expected)
    pd.testing.assert_series_equal(
        budgets.groupby(groups, sort=False).sum(),
        pd.Series([1 / 3, 1 / 3, 1 / 3], index=pd.Index(["x", "y", "z"]),
                  name="risk_budget"),
    )


def test_asset_equal_risk_budgets_are_the_exponent_one_endpoint() -> None:
    """Exponent one reproduces equal budgets over every classified asset."""
    groups = pd.Series(["x", "x", "y", "z", "z", np.nan], index=ASSETS)
    budgets = compute_group_risk_budgets(groups, group_size_exponent=1.0)

    expected = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2, 0.0], index=ASSETS,
                         name="risk_budget")
    pd.testing.assert_series_equal(budgets, expected)


def test_group_risk_budgets_are_invariant_to_asset_order() -> None:
    """Permuting assets changes only output order, not their assigned budgets."""
    groups = pd.Series(["x", "x", "y", "z", "z", "z"], index=ASSETS)
    permuted = groups.iloc[[4, 1, 5, 0, 3, 2]]
    original = compute_group_risk_budgets(groups, group_size_exponent=0.5)
    reordered = compute_group_risk_budgets(permuted, group_size_exponent=0.5)

    pd.testing.assert_series_equal(reordered.reindex(ASSETS), original)


def test_group_risk_budget_panel_is_computed_point_in_time() -> None:
    """Each membership-panel row gets budgets from only its available groups."""
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])
    groups = pd.DataFrame(
        [["x", "x", "y"], ["x", "y", "z"]], index=dates, columns=ASSETS[:3]
    )
    budgets = compute_group_risk_budgets(groups, group_size_exponent=0.0)

    expected = pd.DataFrame(
        [[0.25, 0.25, 0.5], [1 / 3, 1 / 3, 1 / 3]],
        index=dates, columns=ASSETS[:3], dtype=float,
    )
    expected.columns.name = groups.columns.name
    pd.testing.assert_frame_equal(budgets, expected)


def test_rolling_risk_budgeting_accepts_date_varying_budgets() -> None:
    """The rolling solver selects the applicable budget row at every date."""
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])
    assets = ASSETS[:2]
    covar = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    covar_dict = {date: covar for date in dates}
    prices = pd.DataFrame(
        100.0, index=pd.date_range("2023-01-02", "2024-02-29", freq="B"), columns=assets
    )
    budgets = pd.DataFrame([[0.8, 0.2], [0.2, 0.8]], index=dates, columns=assets)
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=assets),
        max_weights=pd.Series(1.0, index=assets),
    )

    weights = rolling_risk_budgeting(
        prices=prices, constraints=constraints, risk_budget=budgets, covar_dict=covar_dict
    )
    realised = weights.pow(2).div(weights.pow(2).sum(axis=1), axis=0)
    np.testing.assert_allclose(realised.to_numpy(), budgets.to_numpy(), atol=2e-4)


def test_group_risk_contributions_reconcile_with_asset_contributions() -> None:
    """Grouping normalized Euler contributions preserves their total exactly."""
    assets = ASSETS[:3]
    covar = pd.DataFrame(np.eye(3), index=assets, columns=assets)
    weights = pd.Series([0.5, 0.25, 0.25], index=assets)
    groups = pd.Series(["standalone", "pair", "pair"], index=assets)

    actual = compute_group_risk_contributions(weights, covar, groups)
    expected = pd.Series([2 / 3, 1 / 3], index=pd.Index(["standalone", "pair"]),
                         name="risk_contribution")
    pd.testing.assert_series_equal(actual, expected)


def test_hrp_consumes_an_external_linkage_and_matches_recursive_bisection() -> None:
    """A supplied two-block tree yields the independently calculated HRP weights."""
    assets = ASSETS[:4]
    covar = pd.DataFrame(np.diag([1.0, 4.0, 9.0, 16.0]), index=assets, columns=assets)
    linkage = np.array([
        [0.0, 1.0, 0.1, 2.0],
        [2.0, 3.0, 0.2, 2.0],
        [4.0, 5.0, 0.3, 4.0],
    ])

    weights = compute_hierarchical_risk_parity_weights(covar, linkage)
    left_share = 5.76 / (0.8 + 5.76)
    expected = pd.Series(
        [left_share * 0.8, left_share * 0.2,
         (1.0 - left_share) * 0.64, (1.0 - left_share) * 0.36],
        index=assets, name="weight",
    )
    pd.testing.assert_series_equal(weights, expected, atol=1e-12, rtol=1e-12)


def test_group_risk_budget_validation_rejects_ambiguous_inputs() -> None:
    """Invalid exponents, labels, panels and empty observations fail explicitly."""
    duplicate_assets = pd.Series(["x", "y"], index=["a", "a"])
    no_groups = pd.Series([np.nan, np.nan], index=ASSETS[:2])
    duplicate_dates = pd.DataFrame(
        [["x"], ["y"]], index=[pd.Timestamp("2024-01-31")] * 2, columns=["a"]
    )
    duplicate_columns = pd.DataFrame([["x", "y"]], columns=["a", "a"])
    empty_observation = pd.DataFrame([[np.nan, np.nan]], columns=ASSETS[:2])

    with pytest.raises(ValueError, match="group_size_exponent must be finite"):
        compute_group_risk_budgets(pd.Series(["x"], index=["a"]),
                                   group_size_exponent=np.inf)
    with pytest.raises(TypeError, match="Series or DataFrame"):
        compute_group_risk_budgets(["x", "y"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="asset labels must be unique"):
        compute_group_risk_budgets(duplicate_assets)
    with pytest.raises(ValueError, match="at least one valid"):
        compute_group_risk_budgets(no_groups)
    with pytest.raises(ValueError, match="at least one observation"):
        compute_group_risk_budgets(pd.DataFrame())
    with pytest.raises(ValueError, match="observation labels must be unique"):
        compute_group_risk_budgets(duplicate_dates)
    with pytest.raises(ValueError, match="asset labels must be unique"):
        compute_group_risk_budgets(duplicate_columns)
    with pytest.raises(ValueError, match="invalid groups at observation"):
        compute_group_risk_budgets(empty_observation)
    with pytest.raises(ValueError, match="positive finite sum"):
        compute_group_risk_budgets(
            pd.Series(["x", "x"], index=ASSETS[:2]), group_size_exponent=1e308
        )


def test_rolling_risk_budget_validation_rejects_ambiguous_panels() -> None:
    """Rolling budgets require unique assets and one unambiguous row per covariance date."""
    date = pd.Timestamp("2024-01-31")
    assets = ASSETS[:2]
    covar = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    common = dict(
        prices=pd.DataFrame([[100.0, 100.0]], index=[date], columns=assets),
        constraints=Constraints(
            is_long_only=True,
            min_weights=pd.Series(0.0, index=assets),
            max_weights=pd.Series(1.0, index=assets),
        ),
        covar_dict={date: covar},
    )

    with pytest.raises(TypeError, match="Series, DataFrame, or None"):
        rolling_risk_budgeting(risk_budget=[0.5, 0.5], **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="asset labels must be unique"):
        rolling_risk_budgeting(
            risk_budget=pd.Series([0.5, 0.5], index=["a", "a"]), **common
        )
    with pytest.raises(ValueError, match="observation labels must be unique"):
        rolling_risk_budgeting(
            risk_budget=pd.DataFrame(
                [[0.5, 0.5], [0.5, 0.5]], index=[date, date], columns=assets
            ),
            **common,
        )
    with pytest.raises(ValueError, match="asset labels must be unique"):
        rolling_risk_budgeting(
            risk_budget=pd.DataFrame([[0.5, 0.5]], index=[date], columns=["a", "a"]),
            **common,
        )
    with pytest.raises(ValueError, match="missing covariance dates"):
        rolling_risk_budgeting(
            risk_budget=pd.DataFrame(
                [[0.5, 0.5]], index=[pd.Timestamp("2024-02-29")], columns=assets
            ),
            **common,
        )


def test_group_risk_contribution_validation_requires_a_complete_partition() -> None:
    """Group aggregation rejects malformed risk universes and incomplete classifications."""
    assets = ASSETS[:2]
    covar = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    weights = pd.Series([0.5, 0.5], index=assets)

    with pytest.raises(TypeError, match="groups must be a pandas Series"):
        compute_group_risk_contributions(weights, covar, ["x", "y"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty and square"):
        compute_group_risk_contributions(weights, pd.DataFrame(), pd.Series(dtype=object))
    with pytest.raises(ValueError, match="identical unique"):
        compute_group_risk_contributions(
            weights, covar.rename(columns={"b": "different"}),
            pd.Series(["x", "y"], index=assets),
        )
    with pytest.raises(ValueError, match="asset labels must be unique"):
        compute_group_risk_contributions(
            weights, covar, pd.Series(["x", "y"], index=["a", "a"])
        )
    with pytest.raises(ValueError, match="classify every covariance asset"):
        compute_group_risk_contributions(
            weights, covar, pd.Series(["x"], index=["a"])
        )


def test_hrp_validation_rejects_malformed_covariances_and_linkages() -> None:
    """HRP validates both labelled covariance structure and the external tree."""
    assets = ASSETS[:2]
    covar = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    linkage = np.array([[0.0, 1.0, 0.1, 2.0]])

    with pytest.raises(TypeError, match="pandas DataFrame"):
        compute_hierarchical_risk_parity_weights(np.eye(2), linkage)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty and square"):
        compute_hierarchical_risk_parity_weights(pd.DataFrame(), np.empty((0, 4)))
    with pytest.raises(ValueError, match="identical unique"):
        compute_hierarchical_risk_parity_weights(
            covar.rename(columns={"b": "different"}), linkage
        )
    with pytest.raises(ValueError, match="only finite"):
        compute_hierarchical_risk_parity_weights(
            covar.mask(np.eye(2, dtype=bool), np.nan), linkage
        )
    with pytest.raises(ValueError, match="symmetric"):
        compute_hierarchical_risk_parity_weights(
            pd.DataFrame([[1.0, 0.2], [0.1, 1.0]], index=assets, columns=assets), linkage
        )
    with pytest.raises(ValueError, match="strictly positive"):
        compute_hierarchical_risk_parity_weights(
            pd.DataFrame([[1.0, 0.0], [0.0, 0.0]], index=assets, columns=assets), linkage
        )
    with pytest.raises(ValueError, match="linkage must have shape"):
        compute_hierarchical_risk_parity_weights(covar, np.empty((0, 4)))
    with pytest.raises(ValueError, match="only finite"):
        compute_hierarchical_risk_parity_weights(
            covar, np.array([[0.0, 1.0, np.nan, 2.0]])
        )
    with pytest.raises(ValueError, match="identifiers must be integers"):
        compute_hierarchical_risk_parity_weights(
            covar, np.array([[0.5, 1.0, 0.1, 2.0]])
        )
    with pytest.raises(ValueError, match="unavailable cluster"):
        compute_hierarchical_risk_parity_weights(
            covar, np.array([[0.0, 2.0, 0.1, 2.0]])
        )
    with pytest.raises(ValueError, match="observation count is inconsistent"):
        compute_hierarchical_risk_parity_weights(
            covar, np.array([[0.0, 1.0, 0.1, 3.0]])
        )
    with pytest.raises(ValueError, match="invalid linkage"):
        compute_hierarchical_risk_parity_weights(
            covar, np.array([[0.0, 0.0, 0.1, 2.0]])
        )
    reused_child = np.array([
        [0.0, 1.0, 0.1, 2.0],
        [0.0, 2.0, 0.2, 2.0],
    ])
    with pytest.raises(ValueError, match="missing or reused"):
        compute_hierarchical_risk_parity_weights(
            pd.DataFrame(np.eye(3), index=ASSETS[:3], columns=ASSETS[:3]),
            reused_child,
        )
    with pytest.raises(ValueError, match="invalid linkage"):
        compute_hierarchical_risk_parity_weights(
            covar, np.array([[0.0, 1.0, -0.1, 2.0]])
        )


def test_hrp_handles_one_asset_and_rejects_nonpositive_block_variance() -> None:
    """The trivial tree is total, while an indefinite child block is rejected."""
    singleton = pd.DataFrame([[0.04]], index=["a"], columns=["a"])
    actual = compute_hierarchical_risk_parity_weights(singleton, np.empty((0, 4)))
    pd.testing.assert_series_equal(actual, pd.Series([1.0], index=["a"], name="weight"))

    assets = ASSETS[:3]
    indefinite = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.0, 1.0, -2.0], [0.0, -2.0, 1.0]],
        index=assets, columns=assets,
    )
    linkage = np.array([[1.0, 2.0, 0.1, 2.0], [0.0, 3.0, 0.2, 3.0]])
    with pytest.raises(ValueError, match="cluster variances"):
        compute_hierarchical_risk_parity_weights(indefinite, linkage)
