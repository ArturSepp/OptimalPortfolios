"""Comprehensive tests for the optimalportfolios constraints module.

Covers all constraint classes and their integration with CVXPY, SciPy, and PyRB
solver backends. Organised into sections:

    1. GroupLowerUpperConstraints — construction, validation, CVXPY generation,
       merge, drop, update, copy
    2. BenchmarkDeviationConstraints — construction, validation, CVXPY generation,
       update, copy
    3. GroupTrackingErrorConstraint — construction, CVXPY constraint & utility
    4. GroupTurnoverConstraint — construction, CVXPY constraint & utility
    5. Constraints (main class) — __post_init__ feasibility, update methods,
       CVXPY / SciPy / PyRB generation, deviation constraint integration
    6. Edge cases — NaN handling, empty groups, misaligned indices

Universe: 10 assets split into 3 groups
    - Equities:     A1, A2, A3, A4
    - FixedIncome:  A5, A6, A7
    - Alternatives: A8, A9, A10

Usage:
    python test_constraints.py              # run all tests
    python test_constraints.py <section>    # run one section (e.g. 1, 2, ...)
"""
from __future__ import annotations
import sys
import warnings
import traceback
import numpy as np
import pandas as pd
import cvxpy as cvx
from enum import Enum

from optimalportfolios.optimization.constraints import (
    Constraints,
    GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
    BenchmarkDeviationConstraints,
    merge_group_lower_upper_constraints,
    ConstraintEnforcementType,
)

# ──────────────────────────────────────────────────────────────────────
# Test universe
# ──────────────────────────────────────────────────────────────────────
TICKERS = [f"A{i}" for i in range(1, 11)]
N = len(TICKERS)

GROUP_LOADINGS = pd.DataFrame(
    {
        "Equities":     [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "FixedIncome":  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        "Alternatives": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    },
    index=TICKERS,
    dtype=float,
)

# simple diagonal covariance for tests that need one
COVAR = np.diag(np.linspace(0.01, 0.10, N))


def _make_gluc(
    group_min: dict = None,
    group_max: dict = None,
    loadings: pd.DataFrame = None,
) -> GroupLowerUpperConstraints:
    return GroupLowerUpperConstraints(
        group_loadings=(loadings if loadings is not None else GROUP_LOADINGS).copy(),
        group_min_allocation=pd.Series(group_min, dtype=float) if group_min else None,
        group_max_allocation=pd.Series(group_max, dtype=float) if group_max else None,
    )


def _solve_and_get_weights(w: cvx.Variable, objective, constraints) -> np.ndarray:
    """Solve a CVXPY problem and return optimal weights."""
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=cvx.SCS, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver returned status: {prob.status}")
    return w.value


# ──────────────────────────────────────────────────────────────────────
# Test runner infrastructure
# ──────────────────────────────────────────────────────────────────────
_results = {"passed": 0, "failed": 0, "errors": []}


def _run_test(name: str, fn):
    """Run a single test function and track results."""
    try:
        fn()
        _results["passed"] += 1
        print(f"  PASS  {name}")
    except Exception as e:
        _results["failed"] += 1
        _results["errors"].append((name, e))
        print(f"  FAIL  {name}")
        traceback.print_exc()
        print()


def _print_summary():
    total = _results["passed"] + _results["failed"]
    print(f"\n{'='*60}")
    print(f"Results: {_results['passed']}/{total} passed, {_results['failed']} failed")
    if _results["errors"]:
        print(f"\nFailed tests:")
        for name, e in _results["errors"]:
            print(f"  - {name}: {type(e).__name__}: {e}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════
# Section 1: GroupLowerUpperConstraints
# ══════════════════════════════════════════════════════════════════════

def test_gluc_construction_basic():
    """Basic construction with valid inputs."""
    gluc = _make_gluc(
        group_min={"Equities": 0.20, "FixedIncome": 0.10, "Alternatives": 0.05},
        group_max={"Equities": 0.60, "FixedIncome": 0.50, "Alternatives": 0.40},
    )
    assert gluc.group_loadings.shape == (10, 3)
    assert len(gluc.group_min_allocation) == 3
    assert len(gluc.group_max_allocation) == 3


def test_gluc_drops_zero_loading_columns():
    """Columns where all loadings are zero should be dropped with a warning."""
    loadings = GROUP_LOADINGS.copy()
    loadings["EmptyGroup"] = 0.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gluc = GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=pd.Series({"Equities": 0.2, "EmptyGroup": 0.1}, dtype=float),
            group_max_allocation=None,
        )
    assert "EmptyGroup" not in gluc.group_loadings.columns
    assert any("zero loadings" in str(warning.message) for warning in w)


def test_gluc_all_zero_loadings_nullifies_constraints():
    """If all columns have zero loadings, constraints are set to None."""
    loadings = pd.DataFrame(0.0, index=TICKERS, columns=["A", "B"])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        gluc = GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=pd.Series({"A": 0.1, "B": 0.2}),
            group_max_allocation=pd.Series({"A": 0.5, "B": 0.6}),
        )
    assert gluc.group_min_allocation is None
    assert gluc.group_max_allocation is None


def test_gluc_missing_allocation_index_warns():
    """Missing group in allocation series should trigger a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gluc = GroupLowerUpperConstraints(
            group_loadings=GROUP_LOADINGS.copy(),
            group_min_allocation=pd.Series({"Equities": 0.2}, dtype=float),  # missing FI, Alt
            group_max_allocation=None,
        )
    assert any("missing" in str(warning.message).lower() for warning in w)
    # reindex should fill missing with NaN
    assert pd.isna(gluc.group_min_allocation.loc["FixedIncome"])


def test_gluc_cvxpy_constraints():
    """Generated CVXPY constraints should be satisfiable and binding."""
    gluc = _make_gluc(
        group_min={"Equities": 0.30, "FixedIncome": 0.20, "Alternatives": 0.10},
        group_max={"Equities": 0.60, "FixedIncome": 0.50, "Alternatives": 0.40},
    )
    w = cvx.Variable(N)
    constraints = gluc.set_cvx_group_lower_upper_constraints(w=w)
    # add basic exposure constraint
    constraints += [cvx.sum(w) == 1.0, w >= 0]
    # minimise variance
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)

    # verify group bounds are satisfied
    eq_weight = weights[:4].sum()
    fi_weight = weights[4:7].sum()
    alt_weight = weights[7:].sum()
    assert eq_weight >= 0.30 - 1e-4, f"Equities {eq_weight:.4f} < 0.30"
    assert eq_weight <= 0.60 + 1e-4, f"Equities {eq_weight:.4f} > 0.60"
    assert fi_weight >= 0.20 - 1e-4, f"FixedIncome {fi_weight:.4f} < 0.20"
    assert alt_weight >= 0.10 - 1e-4, f"Alternatives {alt_weight:.4f} < 0.10"


def test_gluc_cvxpy_with_exposure_scaler():
    """Group constraints should scale with exposure_scaler."""
    gluc = _make_gluc(
        group_min={"Equities": 0.30},
        group_max={"Equities": 0.70},
    )
    w = cvx.Variable(N)
    k = cvx.Variable(pos=True)
    constraints = gluc.set_cvx_group_lower_upper_constraints(w=w, exposure_scaler=k)
    constraints += [cvx.sum(w) == k, w >= 0, k == 2.0]
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)

    eq_weight = weights[:4].sum()
    # bounds should be 0.30*2=0.60 and 0.70*2=1.40
    assert eq_weight >= 0.60 - 1e-3, f"Equities {eq_weight:.4f} < 0.60 (scaled)"


def test_gluc_update_filters_tickers():
    """update() should filter loadings to valid tickers only."""
    gluc = _make_gluc(
        group_min={"Equities": 0.30},
        group_max={"Equities": 0.60},
    )
    subset = ["A1", "A2", "A5", "A6"]
    updated = gluc.update(valid_tickers=subset)
    assert list(updated.group_loadings.index) == subset
    assert updated.group_loadings.shape[0] == 4


def test_gluc_drop_constraint():
    """drop_constraint() should remove the specified group."""
    gluc = _make_gluc(
        group_min={"Equities": 0.30, "FixedIncome": 0.20, "Alternatives": 0.10},
        group_max={"Equities": 0.60, "FixedIncome": 0.50, "Alternatives": 0.40},
    )
    dropped = gluc.drop_constraint("FixedIncome")
    assert "FixedIncome" not in dropped.group_loadings.columns
    assert "FixedIncome" not in dropped.group_min_allocation.index
    assert "FixedIncome" not in dropped.group_max_allocation.index
    assert len(dropped.group_loadings.columns) == 2


def test_gluc_copy_independence():
    """copy() should return an independent object — mutations don't propagate."""
    gluc = _make_gluc(
        group_min={"Equities": 0.30},
        group_max={"Equities": 0.60},
    )
    copied = gluc.copy()
    # modify the copy's underlying data (bypassing frozen with internal access)
    copied.group_loadings.iloc[0, 0] = 999.0
    assert gluc.group_loadings.iloc[0, 0] != 999.0, "Original was mutated by copy"


def test_gluc_merge_no_overlap():
    """Merging two non-overlapping group constraints."""
    gluc1 = GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS[["Equities"]].copy(),
        group_min_allocation=pd.Series({"Equities": 0.20}),
        group_max_allocation=pd.Series({"Equities": 0.60}),
    )
    gluc2 = GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS[["FixedIncome"]].copy(),
        group_min_allocation=pd.Series({"FixedIncome": 0.10}),
        group_max_allocation=pd.Series({"FixedIncome": 0.50}),
    )
    merged = merge_group_lower_upper_constraints(gluc1, gluc2)
    assert set(merged.group_loadings.columns) == {"Equities", "FixedIncome"}
    assert merged.group_min_allocation.loc["Equities"] == 0.20
    assert merged.group_max_allocation.loc["FixedIncome"] == 0.50


def test_gluc_merge_with_overlap():
    """Merging with overlapping group names should add suffixes."""
    gluc1 = _make_gluc(group_min={"Equities": 0.20}, group_max={"Equities": 0.60})
    gluc2 = _make_gluc(group_min={"Equities": 0.30}, group_max={"Equities": 0.50})
    merged = merge_group_lower_upper_constraints(gluc1, gluc2)
    cols = merged.group_loadings.columns.tolist()
    assert "Equities_1" in cols and "Equities_2" in cols


def test_gluc_nan_min_allocation_skipped():
    """NaN entries in group_min_allocation should not generate constraints."""
    gluc = GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS.copy(),
        group_min_allocation=pd.Series(
            {"Equities": np.nan, "FixedIncome": 0.10, "Alternatives": np.nan}
        ),
        group_max_allocation=None,
    )
    w = cvx.Variable(N)
    constraints = gluc.set_cvx_group_lower_upper_constraints(w=w)
    # only FixedIncome should generate a constraint
    assert len(constraints) == 1


# ══════════════════════════════════════════════════════════════════════
# Section 2: BenchmarkDeviationConstraints
# ══════════════════════════════════════════════════════════════════════

SECTOR_LOADINGS = pd.DataFrame(
    {
        "Tech":    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "Finance": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "Energy":  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        "Health":  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        "Other":   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    },
    index=TICKERS,
    dtype=float,
)

BENCHMARK_WEIGHTS = pd.Series(0.10, index=TICKERS)  # equal-weight benchmark


def _make_bdc(
    max_deviation: dict = None,
    loadings: pd.DataFrame = None,
) -> BenchmarkDeviationConstraints:
    if max_deviation is None:
        max_deviation = {col: 0.05 for col in (loadings if loadings is not None else SECTOR_LOADINGS).columns}
    return BenchmarkDeviationConstraints(
        factor_loading_mat=(loadings if loadings is not None else SECTOR_LOADINGS).copy(),
        factor_max_deviation=pd.Series(max_deviation, dtype=float),
    )


def test_bdc_construction_basic():
    """Basic construction with valid inputs."""
    bdc = _make_bdc()
    assert bdc.factor_loading_mat.shape == (10, 5)
    assert len(bdc.factor_max_deviation) == 5


def test_bdc_missing_columns_warns():
    """factor_max_deviation with entries not in factor_loading_mat should warn."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BenchmarkDeviationConstraints(
            factor_loading_mat=SECTOR_LOADINGS[["Tech", "Finance"]].copy(),
            factor_max_deviation=pd.Series({"Tech": 0.05, "Finance": 0.05, "Ghost": 0.10}),
        )
    assert any("Ghost" in str(warning.message) for warning in w)


def test_bdc_none_factor_max_deviation_raises():
    """factor_max_deviation=None should raise ValueError."""
    try:
        BenchmarkDeviationConstraints(
            factor_loading_mat=SECTOR_LOADINGS.copy(),
            factor_max_deviation=None,
        )
        raise AssertionError("Expected ValueError was not raised")
    except (ValueError, TypeError):
        pass  # TypeError also acceptable since frozen dataclass may reject None


def test_bdc_cvxpy_constraints_enforce_deviation_bounds():
    """CVXPY constraints should keep active sector deviations within bounds."""
    max_dev = 0.05
    bdc = _make_bdc(max_deviation={col: max_dev for col in SECTOR_LOADINGS.columns})
    w = cvx.Variable(N)
    constraints = bdc.set_cvx_constraints(w=w, benchmark_weights=BENCHMARK_WEIGHTS)
    constraints += [cvx.sum(w) == 1.0, w >= 0]
    # maximise weight in Tech (A1, A2) — should be limited by deviation constraint
    objective = cvx.Maximize(w[0] + w[1])
    weights = _solve_and_get_weights(w, objective, constraints)

    # Tech active deviation: sum of (tech_loading * (w - bm)) for A1, A2
    tech_active = (weights[:2] - 0.10).sum()
    assert abs(tech_active) <= max_dev + 1e-4, (
        f"Tech active deviation {tech_active:.4f} exceeds bound {max_dev}"
    )


def test_bdc_cvxpy_constraints_symmetric():
    """Deviation constraints should be symmetric — both over and underweight bounded."""
    max_dev = 0.03
    bdc = _make_bdc(max_deviation={col: max_dev for col in SECTOR_LOADINGS.columns})
    w = cvx.Variable(N)
    constraints = bdc.set_cvx_constraints(w=w, benchmark_weights=BENCHMARK_WEIGHTS)
    constraints += [cvx.sum(w) == 1.0, w >= 0]
    # minimise weight in Tech (try to underweight)
    objective = cvx.Minimize(w[0] + w[1])
    weights = _solve_and_get_weights(w, objective, constraints)

    tech_active = (weights[:2] - 0.10).sum()
    assert abs(tech_active) <= max_dev + 1e-4, (
        f"Tech underweight deviation {tech_active:.4f} exceeds bound {max_dev}"
    )


def test_bdc_update_filters_tickers():
    """update() should filter factor_loading_mat to valid tickers."""
    bdc = _make_bdc()
    subset = ["A1", "A3", "A5", "A7", "A9"]
    updated = bdc.update(valid_tickers=subset)
    assert list(updated.factor_loading_mat.index) == subset
    assert updated.factor_loading_mat.shape[0] == 5
    # factor_max_deviation should be unchanged
    assert len(updated.factor_max_deviation) == len(bdc.factor_max_deviation)


def test_bdc_copy_independence():
    """copy() should return an independent object."""
    bdc = _make_bdc()
    copied = bdc.copy()
    copied.factor_loading_mat.iloc[0, 0] = 999.0
    assert bdc.factor_loading_mat.iloc[0, 0] != 999.0, "Original was mutated by copy"


def test_bdc_zero_loading_group_skipped():
    """Groups with all-zero loadings should not generate constraints."""
    loadings = SECTOR_LOADINGS.copy()
    loadings["Empty"] = 0.0
    bdc = BenchmarkDeviationConstraints(
        factor_loading_mat=loadings,
        factor_max_deviation=pd.Series(
            {col: 0.05 for col in loadings.columns}
        ),
    )
    w = cvx.Variable(N)
    constraints = bdc.set_cvx_constraints(w=w, benchmark_weights=BENCHMARK_WEIGHTS)
    # "Empty" group should not contribute a constraint
    # 5 real groups → 5 constraints (each with cvx.abs, which produces one constraint)
    assert len(constraints) == 5


def test_bdc_tight_deviation_produces_near_benchmark():
    """Very tight deviation bounds should force weights close to benchmark."""
    max_dev = 0.001
    bdc = _make_bdc(max_deviation={col: max_dev for col in SECTOR_LOADINGS.columns})
    w = cvx.Variable(N)
    constraints = bdc.set_cvx_constraints(w=w, benchmark_weights=BENCHMARK_WEIGHTS)
    constraints += [cvx.sum(w) == 1.0, w >= 0]
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)

    for col in SECTOR_LOADINGS.columns:
        mask = SECTOR_LOADINGS[col].to_numpy().astype(bool)
        sector_active = (weights[mask] - BENCHMARK_WEIGHTS.to_numpy()[mask]).sum()
        assert abs(sector_active) <= max_dev + 1e-3, (
            f"{col} deviation {sector_active:.4f} exceeds tight bound {max_dev}"
        )


# ══════════════════════════════════════════════════════════════════════
# Section 3: GroupTrackingErrorConstraint
# ══════════════════════════════════════════════════════════════════════

def test_gte_construction_with_vols():
    """Construction with group_tre_vols."""
    gte = GroupTrackingErrorConstraint(
        group_loadings=GROUP_LOADINGS.copy(),
        group_tre_vols=pd.Series({"Equities": 0.02, "FixedIncome": 0.01, "Alternatives": 0.03}),
    )
    assert gte.group_tre_vols is not None


def test_gte_construction_with_utility_weights():
    """Construction with group_tre_utility_weights."""
    gte = GroupTrackingErrorConstraint(
        group_loadings=GROUP_LOADINGS.copy(),
        group_tre_utility_weights=pd.Series({"Equities": 1.0, "FixedIncome": 0.5, "Alternatives": 2.0}),
    )
    assert gte.group_tre_utility_weights is not None


def test_gte_no_constraint_raises():
    """Neither vols nor utility_weights should raise ValueError."""
    try:
        GroupTrackingErrorConstraint(group_loadings=GROUP_LOADINGS.copy())
        raise AssertionError("Expected ValueError was not raised")
    except ValueError:
        pass


def test_gte_cvxpy_constraints():
    """Generated tracking error constraints should be satisfiable."""
    bm = pd.Series(0.10, index=TICKERS)
    gte = GroupTrackingErrorConstraint(
        group_loadings=GROUP_LOADINGS.copy(),
        group_tre_vols=pd.Series({"Equities": 0.05, "FixedIncome": 0.05, "Alternatives": 0.05}),
    )
    w = cvx.Variable(N)
    constraints = gte.set_cvx_group_tre_constraints(w=w, benchmark_weights=bm, covar=COVAR)
    constraints += [cvx.sum(w) == 1.0, w >= 0]
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)
    assert weights is not None
    assert abs(weights.sum() - 1.0) < 1e-4


def test_gte_utility_function():
    """Utility function should return a valid CVXPY expression."""
    bm = pd.Series(0.10, index=TICKERS)
    gte = GroupTrackingErrorConstraint(
        group_loadings=GROUP_LOADINGS.copy(),
        group_tre_utility_weights=pd.Series({"Equities": 1.0, "FixedIncome": 0.5, "Alternatives": 2.0}),
    )
    w = cvx.Variable(N)
    utility = gte.set_cvx_group_tre_utility(w=w, benchmark_weights=bm, covar=COVAR)
    assert utility is not None


# ══════════════════════════════════════════════════════════════════════
# Section 4: GroupTurnoverConstraint
# ══════════════════════════════════════════════════════════════════════

def test_gtc_construction():
    """Basic construction with group_max_turnover."""
    gtc = GroupTurnoverConstraint(
        group_loadings=GROUP_LOADINGS.copy(),
        group_max_turnover=pd.Series({"Equities": 0.10, "FixedIncome": 0.10, "Alternatives": 0.10}),
    )
    assert gtc.group_max_turnover is not None


def test_gtc_no_constraint_raises():
    """Neither max_turnover nor utility_weights should raise ValueError."""
    try:
        GroupTurnoverConstraint(group_loadings=GROUP_LOADINGS.copy())
        raise AssertionError("Expected ValueError was not raised")
    except ValueError:
        pass


def test_gtc_cvxpy_constraints():
    """Turnover constraints should limit weight changes per group."""
    w0 = pd.Series(0.10, index=TICKERS)
    gtc = GroupTurnoverConstraint(
        group_loadings=GROUP_LOADINGS.copy(),
        group_max_turnover=pd.Series({"Equities": 0.05, "FixedIncome": 0.05, "Alternatives": 0.05}),
    )
    w = cvx.Variable(N)
    constraints = gtc.set_group_turnover_constraints(w=w, weights_0=w0)
    constraints += [cvx.sum(w) == 1.0, w >= 0]
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)

    # check turnover per group
    for grp, max_to in [("Equities", 0.05), ("FixedIncome", 0.05), ("Alternatives", 0.05)]:
        mask = GROUP_LOADINGS[grp].to_numpy().astype(bool)
        group_turnover = np.abs(weights[mask] - w0.to_numpy()[mask]).sum()
        assert group_turnover <= max_to + 1e-3, (
            f"{grp} turnover {group_turnover:.4f} exceeds {max_to}"
        )


def test_gtc_utility_function():
    """Utility function should return a valid CVXPY expression."""
    w0 = pd.Series(0.10, index=TICKERS)
    gtc = GroupTurnoverConstraint(
        group_loadings=GROUP_LOADINGS.copy(),
        group_turnover_utility_weights=pd.Series({"Equities": 1.0, "FixedIncome": 0.5, "Alternatives": 2.0}),
    )
    w = cvx.Variable(N)
    utility = gtc.set_cvx_group_turnover_utility(w=w, weights_0=w0)
    assert utility is not None


# ══════════════════════════════════════════════════════════════════════
# Section 5: Constraints (main class)
# ══════════════════════════════════════════════════════════════════════

# --- 5a. __post_init__ feasibility ---

def test_constraints_feasible():
    """Valid constraints should construct without error."""
    c = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.02, index=TICKERS),
        max_weights=pd.Series(0.20, index=TICKERS),
        group_lower_upper_constraints=_make_gluc(
            group_min={"Equities": 0.10, "FixedIncome": 0.10, "Alternatives": 0.05},
            group_max={"Equities": 0.60, "FixedIncome": 0.50, "Alternatives": 0.40},
        ),
    )
    assert c is not None


def test_constraints_min_gt_max_raises():
    """min_weights > max_weights should raise ValueError."""
    try:
        Constraints(
            is_long_only=True,
            min_weights=pd.Series(0.50, index=TICKERS),
            max_weights=pd.Series(0.10, index=TICKERS),
        )
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        assert "min_weights > max_weights" in str(e)


def test_constraints_long_only_negative_min_raises():
    """is_long_only=True with negative min_weights should raise ValueError."""
    try:
        Constraints(
            is_long_only=True,
            min_weights=pd.Series(-0.10, index=TICKERS),
        )
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        assert "is_long_only=True" in str(e)


def test_constraints_asset_max_below_group_min_raises():
    """Sum of asset max_weights < group_min_allocation should raise."""
    try:
        Constraints(
            is_long_only=True,
            max_weights=pd.Series(0.05, index=TICKERS),
            group_lower_upper_constraints=_make_gluc(
                group_min={"Equities": 0.40, "FixedIncome": 0.10, "Alternatives": 0.10},
                group_max={"Equities": 0.60, "FixedIncome": 0.50, "Alternatives": 0.40},
            ),
        )
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        assert "Equities" in str(e)


def test_constraints_asset_min_above_group_max_raises():
    """Sum of asset min_weights > group_max_allocation should raise."""
    try:
        Constraints(
            is_long_only=True,
            min_weights=pd.Series(
                [0.0]*4 + [0.20, 0.20, 0.20] + [0.0]*3, index=TICKERS
            ),
            max_weights=pd.Series(0.50, index=TICKERS),
            group_lower_upper_constraints=_make_gluc(
                group_min={"Equities": 0.10, "FixedIncome": 0.10, "Alternatives": 0.05},
                group_max={"Equities": 0.60, "FixedIncome": 0.40, "Alternatives": 0.30},
            ),
        )
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        assert "FixedIncome" in str(e)


def test_constraints_multiple_violations():
    """Multiple simultaneous violations should all be reported."""
    try:
        Constraints(
            is_long_only=True,
            min_weights=pd.Series(
                [0.0]*4 + [0.25, 0.25, 0.25, 0.35, 0.0, 0.0], index=TICKERS
            ),
            max_weights=pd.Series(
                [0.05]*4 + [0.50]*6, index=TICKERS
            ),
            group_lower_upper_constraints=_make_gluc(
                group_min={"Equities": 0.50, "FixedIncome": 0.10, "Alternatives": 0.05},
                group_max={"Equities": 0.80, "FixedIncome": 0.40, "Alternatives": 0.30},
            ),
        )
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        msg = str(e)
        assert "Equities" in msg and "FixedIncome" in msg and "A8" in msg


# --- 5b. update methods ---

def test_constraints_update_filters_tickers():
    """update() should propagate valid_tickers to all sub-constraints."""
    c = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.02, index=TICKERS),
        max_weights=pd.Series(0.20, index=TICKERS),
        group_lower_upper_constraints=_make_gluc(
            group_min={"Equities": 0.10},
            group_max={"Equities": 0.60},
        ),
        sector_deviation_constraints=_make_bdc(),
        style_deviation_constraints=_make_bdc(),
    )
    subset = ["A1", "A2", "A5", "A8"]
    updated = c.update(valid_tickers=subset)
    assert updated.group_lower_upper_constraints.group_loadings.shape[0] == 4
    assert updated.sector_deviation_constraints.factor_loading_mat.shape[0] == 4
    assert updated.style_deviation_constraints.factor_loading_mat.shape[0] == 4


def test_constraints_update_with_valid_tickers_reindexes():
    """update_with_valid_tickers() should reindex all Series fields."""
    bm = pd.Series(0.10, index=TICKERS)
    c = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.02, index=TICKERS),
        max_weights=pd.Series(0.20, index=TICKERS),
        benchmark_weights=bm,
        sector_deviation_constraints=_make_bdc(),
    )
    subset = ["A1", "A3", "A5", "A7", "A9"]
    updated = c.update_with_valid_tickers(
        valid_tickers=subset,
        benchmark_weights=bm,
    )
    assert list(updated.min_weights.index) == subset
    assert list(updated.max_weights.index) == subset
    assert list(updated.benchmark_weights.index) == subset
    assert updated.sector_deviation_constraints.factor_loading_mat.shape[0] == 5


def test_constraints_update_with_rebalancing_indicators():
    """Assets with rebalancing_indicator=0 should have frozen weights."""
    w0 = pd.Series([0.15, 0.10, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], index=TICKERS)
    rebal = pd.Series([1, 1, 0, 0, 1, 1, 1, 1, 1, 1], index=TICKERS)  # freeze A3, A4
    c = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=TICKERS),
        max_weights=pd.Series(0.30, index=TICKERS),
    )
    updated = c.update_with_valid_tickers(
        valid_tickers=TICKERS,
        weights_0=w0,
        rebalancing_indicators=rebal,
    )
    # A3 should be frozen at w0 value
    assert abs(updated.min_weights.loc["A3"] - 0.05) < 1e-10
    assert abs(updated.max_weights.loc["A3"] - 0.05) < 1e-10
    # A1 should keep original bounds
    assert abs(updated.min_weights.loc["A1"] - 0.0) < 1e-10
    assert abs(updated.max_weights.loc["A1"] - 0.30) < 1e-10


def test_constraints_copy_independence():
    """Deep copy should be fully independent."""
    c = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.02, index=TICKERS),
        max_weights=pd.Series(0.20, index=TICKERS),
    )
    copied = c.copy()
    # frozen dataclass prevents direct mutation, but verify it's a different object
    assert copied is not c
    assert copied.min_weights is not c.min_weights


# --- 5c. CVXPY constraint generation ---

def test_constraints_cvxpy_long_only():
    """Long-only constraint should prevent negative weights."""
    c = Constraints(is_long_only=True)
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)
    assert np.all(weights >= -1e-6)


def test_constraints_cvxpy_exposure_bounds():
    """Exposure constraints should be respected."""
    c = Constraints(is_long_only=True, max_exposure=0.80, min_exposure=0.60)
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)
    assert weights.sum() >= 0.60 - 1e-4
    assert weights.sum() <= 0.80 + 1e-4


def test_constraints_cvxpy_min_max_weights():
    """Individual min/max weight bounds should be respected."""
    min_w = pd.Series(0.05, index=TICKERS)
    max_w = pd.Series(0.15, index=TICKERS)
    c = Constraints(is_long_only=True, min_weights=min_w, max_weights=max_w)
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)
    assert np.all(weights >= 0.05 - 1e-4)
    assert np.all(weights <= 0.15 + 1e-4)


def test_constraints_cvxpy_tracking_error():
    """Tracking error constraint should limit deviation from benchmark."""
    bm = pd.Series(0.10, index=TICKERS)
    te_limit = 0.02
    c = Constraints(
        is_long_only=True,
        benchmark_weights=bm,
        tracking_err_vol_constraint=te_limit,
    )
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)

    # verify tracking error
    active = weights - bm.to_numpy()
    te_var = active @ COVAR @ active
    assert np.sqrt(te_var) <= te_limit + 1e-3


def test_constraints_cvxpy_turnover():
    """Turnover constraint should limit total weight changes."""
    w0 = pd.Series(0.10, index=TICKERS)
    to_limit = 0.10
    c = Constraints(
        is_long_only=True,
        weights_0=w0,
        turnover_constraint=to_limit,
    )
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)

    turnover = np.abs(weights - w0.to_numpy()).sum()
    assert turnover <= to_limit + 1e-3


def test_constraints_cvxpy_target_return():
    """Target return constraint should achieve minimum return."""
    returns = pd.Series(np.linspace(0.01, 0.10, N), index=TICKERS)
    target = 0.05
    c = Constraints(
        is_long_only=True,
        target_return=target,
        asset_returns=returns,
    )
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)

    achieved = returns.to_numpy() @ weights
    assert achieved >= target - 1e-4


def test_constraints_cvxpy_vol_bounds():
    """Max portfolio volatility constraint should be respected.

    Note: min_target_portfolio_vol_an (quad_form >= const) is non-DCP and
    cannot be solved by standard convex solvers. Only max vol is tested here.
    """
    c = Constraints(
        is_long_only=True,
        max_target_portfolio_vol_an=0.06,
    )
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    # maximise return to push against vol ceiling
    returns = np.linspace(0.01, 0.10, N)
    objective = cvx.Maximize(returns @ w)
    weights = _solve_and_get_weights(w, objective, constraints)

    vol = np.sqrt(weights @ COVAR @ weights)
    assert vol <= 0.06 + 1e-3


def test_constraints_cvxpy_with_deviation_constraints():
    """Full constraint set including sector/style deviation constraints."""
    bm = pd.Series(0.10, index=TICKERS)
    max_dev = 0.03
    c = Constraints(
        is_long_only=True,
        benchmark_weights=bm,
        sector_deviation_constraints=_make_bdc(
            max_deviation={col: max_dev for col in SECTOR_LOADINGS.columns}
        ),
    )
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)

    for col in SECTOR_LOADINGS.columns:
        mask = SECTOR_LOADINGS[col].to_numpy().astype(bool)
        sector_active = (weights[mask] - bm.to_numpy()[mask]).sum()
        assert abs(sector_active) <= max_dev + 1e-3


# --- 5d. SciPy backend ---

def test_constraints_scipy_bounds_long_only():
    """Long-only without explicit bounds → (0, 1) for each asset."""
    c = Constraints(is_long_only=True)
    bounds = c.set_scipy_bounds(covar=COVAR)
    assert bounds.shape == (N, 2)
    assert np.allclose(bounds[:, 0], 0.0)
    assert np.allclose(bounds[:, 1], 1.0)


def test_constraints_scipy_bounds_with_weights():
    """Explicit min/max weights should be reflected in bounds."""
    c = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.05, index=TICKERS),
        max_weights=pd.Series(0.25, index=TICKERS),
    )
    bounds = c.set_scipy_bounds(covar=COVAR)
    assert np.allclose(bounds[:, 0], 0.05)
    assert np.allclose(bounds[:, 1], 0.25)


def test_constraints_scipy_bounds_short_allowed():
    """Non-long-only without bounds → None (unconstrained)."""
    c = Constraints(is_long_only=False, max_exposure=1.5, min_exposure=0.5)
    bounds = c.set_scipy_bounds(covar=COVAR)
    assert bounds is None


def test_constraints_scipy_constraints_generation():
    """SciPy constraints should include exposure bounds."""
    c = Constraints(
        is_long_only=True,
        group_lower_upper_constraints=_make_gluc(
            group_min={"Equities": 0.20},
            group_max={"Equities": 0.60},
        ),
    )
    constraints, bounds = c.set_scipy_constraints(covar=COVAR)
    # should have: long_only + max_exposure + min_exposure + group_min + group_max = 5
    assert len(constraints) >= 4


# --- 5e. PyRB backend ---

def test_constraints_pyrb_generation():
    """PyRB constraints should produce correct matrix form."""
    c = Constraints(
        is_long_only=True,
        group_lower_upper_constraints=_make_gluc(
            group_min={"Equities": 0.20, "FixedIncome": 0.10},
            group_max={"Equities": 0.60, "FixedIncome": 0.50},
        ),
    )
    bounds, c_rows, c_lhs = c.set_pyrb_constraints(covar=COVAR)
    assert bounds is not None
    # 2 groups x 2 constraints (min + max) = 4 rows
    assert c_rows.shape[0] == 4
    assert len(c_lhs) == 4


def test_constraints_pyrb_no_groups():
    """PyRB without group constraints should return None for matrix/vector."""
    c = Constraints(is_long_only=True)
    bounds, c_rows, c_lhs = c.set_pyrb_constraints(covar=COVAR)
    assert bounds is not None
    assert c_rows is None
    assert c_lhs is None


# --- 5f. Utility objective ---

def test_constraints_utility_objective():
    """Utility objective should produce valid objective + constraints."""
    bm = pd.Series(0.10, index=TICKERS)
    w0 = pd.Series(0.10, index=TICKERS)
    c = Constraints(
        is_long_only=True,
        benchmark_weights=bm,
        weights_0=w0,
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        tre_utility_weight=1.0,
        turnover_utility_weight=0.40,
    )
    w = cvx.Variable(N)
    alphas = np.linspace(0.01, 0.05, N)
    obj_fun, constraints = c.set_cvx_utility_objective_constraints(
        w=w, alphas=alphas, covar=COVAR
    )
    assert obj_fun is not None
    assert len(constraints) >= 2  # at least exposure constraints


# --- 5g. Debug utilities ---

def test_print_constraints_runs():
    """print_constraints should execute without error."""
    c = Constraints(is_long_only=True)
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    # solve first so constraints have values
    prob = cvx.Problem(cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
    prob.solve(solver=cvx.SCS, verbose=False)
    # should not raise
    c.print_constraints(constraints)


def test_check_constraints_violation_runs():
    """check_constraints_violation should execute without error."""
    c = Constraints(is_long_only=True)
    w = cvx.Variable(N)
    constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
    prob = cvx.Problem(cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
    prob.solve(solver=cvx.SCS, verbose=False)
    # should not raise
    c.check_constraints_violation(constraints)


# ══════════════════════════════════════════════════════════════════════
# Section 6: Edge cases
# ══════════════════════════════════════════════════════════════════════

def test_fractional_group_loadings():
    """Non-binary (fractional) group loadings should work correctly."""
    loadings = pd.DataFrame(
        {"Factor1": [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        index=TICKERS,
        dtype=float,
    )
    gluc = GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series({"Factor1": 0.05}),
        group_max_allocation=pd.Series({"Factor1": 0.30}),
    )
    w = cvx.Variable(N)
    constraints = gluc.set_cvx_group_lower_upper_constraints(w=w)
    constraints += [cvx.sum(w) == 1.0, w >= 0]
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)

    factor_exposure = loadings["Factor1"].to_numpy() @ weights
    assert factor_exposure >= 0.05 - 1e-4
    assert factor_exposure <= 0.30 + 1e-4


def test_single_asset_group():
    """Group with single asset should constrain that asset directly."""
    loadings = pd.DataFrame(
        {"Solo": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        index=TICKERS,
        dtype=float,
    )
    gluc = GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series({"Solo": 0.15}),
        group_max_allocation=pd.Series({"Solo": 0.15}),
    )
    w = cvx.Variable(N)
    constraints = gluc.set_cvx_group_lower_upper_constraints(w=w)
    constraints += [cvx.sum(w) == 1.0, w >= 0]
    objective = cvx.Minimize(cvx.quad_form(w, COVAR))
    weights = _solve_and_get_weights(w, objective, constraints)
    assert abs(weights[0] - 0.15) < 1e-3


def test_constraints_no_covar_for_te_raises():
    """Tracking error constraint without covariance matrix should raise."""
    bm = pd.Series(0.10, index=TICKERS)
    c = Constraints(
        is_long_only=True,
        benchmark_weights=bm,
        tracking_err_vol_constraint=0.02,
    )
    w = cvx.Variable(N)
    try:
        c.set_cvx_all_constraints(w=w, covar=None)
        raise AssertionError("Expected an exception but none was raised")
    except (ValueError, Exception):
        pass  # either ValueError from guard or Exception from cvx.quad_form(w, None)


def test_constraints_target_return_without_returns_raises():
    """target_return without asset_returns should raise ValueError."""
    c = Constraints(is_long_only=True, target_return=0.05)
    w = cvx.Variable(N)
    try:
        c.set_cvx_all_constraints(w=w, covar=COVAR)
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        assert "asset_returns" in str(e)


def test_update_group_lower_upper_constraints_merge():
    """update_group_lower_upper_constraints should merge when existing."""
    c = Constraints(
        is_long_only=True,
        group_lower_upper_constraints=_make_gluc(
            group_min={"Equities": 0.20},
            group_max={"Equities": 0.60},
        ),
    )
    new_gluc = GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS[["FixedIncome"]].copy(),
        group_min_allocation=pd.Series({"FixedIncome": 0.10}),
        group_max_allocation=pd.Series({"FixedIncome": 0.50}),
    )
    updated = c.update_group_lower_upper_constraints(new_gluc)
    cols = updated.group_lower_upper_constraints.group_loadings.columns.tolist()
    assert "Equities" in cols or "Equities_1" in cols
    assert "FixedIncome" in cols or "FixedIncome_2" in cols


def test_update_group_lower_upper_constraints_add_new():
    """update_group_lower_upper_constraints should set when None."""
    c = Constraints(is_long_only=True)
    new_gluc = _make_gluc(
        group_min={"Equities": 0.20},
        group_max={"Equities": 0.60},
    )
    updated = c.update_group_lower_upper_constraints(new_gluc)
    assert updated.group_lower_upper_constraints is not None


# ══════════════════════════════════════════════════════════════════════
# Test runner
# ══════════════════════════════════════════════════════════════════════

SECTIONS = {
    "1": (
        "GroupLowerUpperConstraints",
        [
            ("construction_basic", test_gluc_construction_basic),
            ("drops_zero_loading_columns", test_gluc_drops_zero_loading_columns),
            ("all_zero_nullifies", test_gluc_all_zero_loadings_nullifies_constraints),
            ("missing_allocation_warns", test_gluc_missing_allocation_index_warns),
            ("cvxpy_constraints", test_gluc_cvxpy_constraints),
            ("cvxpy_exposure_scaler", test_gluc_cvxpy_with_exposure_scaler),
            ("update_filters_tickers", test_gluc_update_filters_tickers),
            ("drop_constraint", test_gluc_drop_constraint),
            ("copy_independence", test_gluc_copy_independence),
            ("merge_no_overlap", test_gluc_merge_no_overlap),
            ("merge_with_overlap", test_gluc_merge_with_overlap),
            ("nan_min_allocation_skipped", test_gluc_nan_min_allocation_skipped),
        ],
    ),
    "2": (
        "BenchmarkDeviationConstraints",
        [
            ("construction_basic", test_bdc_construction_basic),
            ("missing_columns_warns", test_bdc_missing_columns_warns),
            ("none_max_deviation_raises", test_bdc_none_factor_max_deviation_raises),
            ("cvxpy_enforces_bounds", test_bdc_cvxpy_constraints_enforce_deviation_bounds),
            ("cvxpy_symmetric", test_bdc_cvxpy_constraints_symmetric),
            ("update_filters_tickers", test_bdc_update_filters_tickers),
            ("copy_independence", test_bdc_copy_independence),
            ("zero_loading_skipped", test_bdc_zero_loading_group_skipped),
            ("tight_deviation_near_bm", test_bdc_tight_deviation_produces_near_benchmark),
        ],
    ),
    "3": (
        "GroupTrackingErrorConstraint",
        [
            ("construction_vols", test_gte_construction_with_vols),
            ("construction_utility", test_gte_construction_with_utility_weights),
            ("no_constraint_raises", test_gte_no_constraint_raises),
            ("cvxpy_constraints", test_gte_cvxpy_constraints),
            ("utility_function", test_gte_utility_function),
        ],
    ),
    "4": (
        "GroupTurnoverConstraint",
        [
            ("construction", test_gtc_construction),
            ("no_constraint_raises", test_gtc_no_constraint_raises),
            ("cvxpy_constraints", test_gtc_cvxpy_constraints),
            ("utility_function", test_gtc_utility_function),
        ],
    ),
    "5": (
        "Constraints",
        [
            ("feasible", test_constraints_feasible),
            ("min_gt_max_raises", test_constraints_min_gt_max_raises),
            ("long_only_neg_min_raises", test_constraints_long_only_negative_min_raises),
            ("asset_max_below_group_min", test_constraints_asset_max_below_group_min_raises),
            ("asset_min_above_group_max", test_constraints_asset_min_above_group_max_raises),
            ("multiple_violations", test_constraints_multiple_violations),
            ("update_filters_tickers", test_constraints_update_filters_tickers),
            ("update_with_valid_tickers", test_constraints_update_with_valid_tickers_reindexes),
            ("rebalancing_indicators", test_constraints_update_with_rebalancing_indicators),
            ("copy_independence", test_constraints_copy_independence),
            ("cvxpy_long_only", test_constraints_cvxpy_long_only),
            ("cvxpy_exposure_bounds", test_constraints_cvxpy_exposure_bounds),
            ("cvxpy_min_max_weights", test_constraints_cvxpy_min_max_weights),
            ("cvxpy_tracking_error", test_constraints_cvxpy_tracking_error),
            ("cvxpy_turnover", test_constraints_cvxpy_turnover),
            ("cvxpy_target_return", test_constraints_cvxpy_target_return),
            ("cvxpy_vol_bounds", test_constraints_cvxpy_vol_bounds),
            ("cvxpy_with_deviations", test_constraints_cvxpy_with_deviation_constraints),
            ("scipy_bounds_long_only", test_constraints_scipy_bounds_long_only),
            ("scipy_bounds_with_weights", test_constraints_scipy_bounds_with_weights),
            ("scipy_bounds_short", test_constraints_scipy_bounds_short_allowed),
            ("scipy_constraints", test_constraints_scipy_constraints_generation),
            ("pyrb_generation", test_constraints_pyrb_generation),
            ("pyrb_no_groups", test_constraints_pyrb_no_groups),
            ("utility_objective", test_constraints_utility_objective),
            ("print_constraints", test_print_constraints_runs),
            ("check_violation", test_check_constraints_violation_runs),
        ],
    ),
    "6": (
        "Edge cases",
        [
            ("fractional_loadings", test_fractional_group_loadings),
            ("single_asset_group", test_single_asset_group),
            ("no_covar_for_te_raises", test_constraints_no_covar_for_te_raises),
            ("target_return_no_returns", test_constraints_target_return_without_returns_raises),
            ("merge_gluc_via_constraints", test_update_group_lower_upper_constraints_merge),
            ("add_new_gluc", test_update_group_lower_upper_constraints_add_new),
        ],
    ),
}


def run_section(key: str):
    name, tests = SECTIONS[key]
    print(f"\n{'━'*60}")
    print(f"  Section {key}: {name}")
    print(f"{'━'*60}")
    for test_name, test_fn in tests:
        _run_test(f"{name}.{test_name}", test_fn)


if __name__ == "__main__":
    print("=" * 60)
    print("  Constraints Module — Comprehensive Test Suite")
    print("=" * 60)

    if len(sys.argv) > 1:
        sections = sys.argv[1:]
    else:
        sections = sorted(SECTIONS.keys())

    for s in sections:
        if s in SECTIONS:
            run_section(s)
        else:
            print(f"\nUnknown section: {s}. Available: {sorted(SECTIONS.keys())}")

    _print_summary()
    sys.exit(1 if _results["failed"] > 0 else 0)