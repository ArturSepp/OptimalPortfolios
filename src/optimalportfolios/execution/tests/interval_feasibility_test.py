"""Independent interval references and malformed execution-bound contracts."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution.feasibility import compute_group_bound_bridges


def _inputs():
    """Return a signed exposure fixture with finite executable intervals."""
    assets = pd.Index(["A", "B", "C"])
    return (
        pd.Series([0.1, 0.2, 0.0], index=assets),
        pd.Series([0.3, 0.4, 0.5], index=assets),
        pd.DataFrame({"Positive": [1.0, 1.0, 0.0], "Signed": [1.0, -2.0, 0.5]}, index=assets),
        pd.Series({"Positive": 0.8, "Signed": 0.2}),
        pd.Series({"Positive": 0.25, "Signed": -0.8}),
    )


def test_signed_group_ranges_match_independent_corner_enumeration():
    """Each reported extreme equals the minimum or maximum over every corner."""
    lower, upper, loadings, minimum, maximum = _inputs()
    corners = np.array(list(product(*zip(lower, upper))))
    exposures = corners @ loadings.to_numpy()
    result = compute_group_bound_bridges(lower, upper, loadings, minimum, maximum)
    np.testing.assert_allclose(result.e_min, exposures.min(axis=0))
    np.testing.assert_allclose(result.e_max, exposures.max(axis=0))
    np.testing.assert_allclose(result.bridge_min, [0.1, 0.05])
    np.testing.assert_allclose(result.bridge_max, [0.05, 0.1])
    assert result.index.name == "group"


def test_unbounded_groups_and_reordered_upper_are_supported():
    """Missing group limits impose no bridge and upper bounds align by label."""
    lower, upper, loadings, _, _ = _inputs()
    result = compute_group_bound_bridges(
        lower, upper.iloc[::-1], loadings, pd.Series(dtype=float), pd.Series(dtype=float)
    )
    assert result[["bridge_min", "bridge_max"]].eq(0).all().all()
    np.testing.assert_allclose(result.e_min, [0.3, -0.7])


@pytest.mark.parametrize("axis", ["lower", "upper", "loading_rows", "loading_columns"])
def test_duplicate_axes_are_rejected(axis):
    """Duplicate labels cannot silently identify different execution inputs."""
    lower, upper, loadings, minimum, maximum = _inputs()
    if axis == "lower":
        lower.index = ["A", "A", "C"]
    elif axis == "upper":
        upper.index = ["A", "A", "C"]
    elif axis == "loading_rows":
        loadings.index = ["A", "A", "C"]
    else:
        loadings.columns = ["Group", "Group"]
    with pytest.raises(ValueError, match="unique"):
        compute_group_bound_bridges(lower, upper, loadings, minimum, maximum)


@pytest.mark.parametrize("field", ["upper", "loadings"])
def test_missing_aligned_instrument_is_rejected(field):
    """Every executable asset requires a complete upper bound and loading row."""
    lower, upper, loadings, minimum, maximum = _inputs()
    if field == "upper":
        upper = upper.drop("B")
    else:
        loadings = loadings.drop("B")
    with pytest.raises(ValueError, match="complete"):
        compute_group_bound_bridges(lower, upper, loadings, minimum, maximum)


@pytest.mark.parametrize("field", ["lower", "upper", "loadings"])
def test_nonfinite_executable_input_is_rejected(field):
    """Unbounded executable intervals do not produce misleading finite bridges."""
    lower, upper, loadings, minimum, maximum = _inputs()
    if field == "lower":
        lower.iloc[0] = -np.inf
    elif field == "upper":
        upper.iloc[0] = np.inf
    else:
        loadings.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        compute_group_bound_bridges(lower, upper, loadings, minimum, maximum)


def test_empty_executable_interval_is_rejected():
    """A lower bound above its upper bound reports the affected asset."""
    lower, upper, loadings, minimum, maximum = _inputs()
    lower.loc["A"] = 0.4
    with pytest.raises(ValueError, match="exceeds upper bound.*A"):
        compute_group_bound_bridges(lower, upper, loadings, minimum, maximum)
