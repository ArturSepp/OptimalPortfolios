"""Characterize solver translations and public rebalancing-bound semantics."""

import numpy as np
import pandas as pd

from optimalportfolios import (
    Constraints,
    GroupLowerUpperConstraints,
    compute_eligible_rebalancing_bounds,
)


TICKERS = ['growth', 'balanced', 'defensive']
COVAR = np.eye(len(TICKERS))


def _constraints_with_group_bounds() -> Constraints:
    """Return one ordered constraint set shared by the backend contract tests."""
    group_loadings = pd.DataFrame(
        {
            'risky': [1.0, 1.0, 0.0],
            'safe': [0.0, 0.0, 1.0],
        },
        index=TICKERS,
    )
    group_bounds = GroupLowerUpperConstraints(
        group_loadings=group_loadings,
        group_min_allocation=pd.Series({'risky': 0.25, 'safe': 0.10}),
        group_max_allocation=pd.Series({'risky': 0.75, 'safe': 0.40}),
    )
    return Constraints(
        is_long_only=True,
        max_weights=pd.Series([0.60, 0.50, 0.40], index=TICKERS),
        min_exposure=0.80,
        max_exposure=1.10,
        group_lower_upper_constraints=group_bounds,
    )


def test_scipy_translation_preserves_callable_order_values_and_bounds() -> None:
    """SciPy receives long-only, exposure, then per-group min/max callables."""
    translated, bounds = _constraints_with_group_bounds().set_scipy_constraints(
        covar=COVAR)

    assert [item['type'] for item in translated] == ['ineq'] * 7
    probe = np.array([0.20, 0.15, 0.05])
    np.testing.assert_array_equal(translated[0]['fun'](probe), probe)
    np.testing.assert_allclose(
        [item['fun'](probe) for item in translated[1:]],
        [0.70, -0.40, 0.10, 0.40, -0.05, 0.35],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        bounds,
        np.array([[0.0, 0.60], [0.0, 0.50], [0.0, 0.40]]),
    )


def test_pyrb_translation_preserves_group_row_signs_rhs_and_bounds() -> None:
    """PyRB receives each group minimum as a negated row before its maximum."""
    bounds, c_rows, c_rhs = _constraints_with_group_bounds().set_pyrb_constraints(
        covar=COVAR)

    np.testing.assert_array_equal(
        bounds,
        np.array([[0.0, 0.60], [0.0, 0.50], [0.0, 0.40]]),
    )
    np.testing.assert_array_equal(
        c_rows,
        np.array([
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]),
    )
    np.testing.assert_array_equal(c_rhs, np.array([-0.25, 0.75, -0.10, 0.40]))


def test_eligible_bounds_align_every_input_to_the_current_weight_index() -> None:
    """Reordered inputs ignore extra labels and zero-fill labels missing from them."""
    current = pd.Series([0.40, 0.20, 0.0], index=['b', 'a', 'missing'])
    model = pd.Series([0.60, 0.10, 0.90], index=['a', 'b', 'extra'])
    current_min = pd.Series([0.25, 0.0, -1.0], index=['a', 'b', 'extra'])
    current_max = pd.Series([0.90, 0.50, 1.0], index=['b', 'a', 'extra'])

    lower, upper, indicators = compute_eligible_rebalancing_bounds(
        current_weights=current,
        model_weights=model,
        current_min_weights=current_min,
        current_max_weights=current_max,
    )

    pd.testing.assert_series_equal(
        lower, pd.Series([0.10, 0.25, 0.0], index=current.index))
    pd.testing.assert_series_equal(
        upper, pd.Series([0.40, 0.50, 0.0], index=current.index))
    pd.testing.assert_series_equal(
        indicators, pd.Series([1, 1, 0], index=current.index))


def test_rebalancing_materiality_threshold_is_strictly_greater_than_one_e_minus_eight(
) -> None:
    """Weights exactly at the threshold are ineligible; the adjacent float is eligible."""
    positive_above = np.nextafter(1e-8, np.inf)
    negative_above = np.nextafter(-1e-8, -np.inf)
    assets = pd.Index(['current_at', 'current_above', 'model_at', 'model_above'])

    _, _, indicators = compute_eligible_rebalancing_bounds(
        current_weights=pd.Series([1e-8, positive_above, 0.0, 0.0], index=assets),
        model_weights=pd.Series([0.0, 0.0, -1e-8, negative_above], index=assets),
        current_min_weights=pd.Series(-1.0, index=assets),
        current_max_weights=pd.Series(1.0, index=assets),
    )

    pd.testing.assert_series_equal(indicators, pd.Series([0, 1, 0, 1], index=assets))


def test_eligible_corridor_supports_existing_and_model_short_weights() -> None:
    """The corridor uses signed endpoints while materiality remains absolute."""
    assets = pd.Index(['cover_short', 'cross_zero', 'open_short', 'absent'])
    lower, upper, indicators = compute_eligible_rebalancing_bounds(
        current_weights=pd.Series([-0.40, -0.20, 0.0, 0.0], index=assets),
        model_weights=pd.Series([-0.10, 0.10, -0.30, 0.0], index=assets),
        current_min_weights=pd.Series(-1.0, index=assets),
        current_max_weights=pd.Series(1.0, index=assets),
    )

    pd.testing.assert_series_equal(
        lower, pd.Series([-0.40, -0.20, -0.30, 0.0], index=assets))
    pd.testing.assert_series_equal(
        upper, pd.Series([-0.10, 0.10, 0.0, 0.0], index=assets))
    pd.testing.assert_series_equal(indicators, pd.Series([1, 1, 1, 0], index=assets))
