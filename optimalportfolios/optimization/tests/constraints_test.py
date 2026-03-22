"""Tests for Constraints.__post_init__ feasibility checks between
individual asset bounds and group lower/upper constraints.

Universe: 10 assets split into 3 groups
    - Equities:     A1, A2, A3, A4
    - FixedIncome:  A5, A6, A7
    - Alternatives: A8, A9, A10
"""
import numpy as np
import pandas as pd
from enum import Enum

from optimalportfolios.optimization.constraints import (
    Constraints,
    GroupLowerUpperConstraints,
)

TICKERS = [f"A{i}" for i in range(1, 11)]

GROUP_LOADINGS = pd.DataFrame(
    {
        "Equities":     [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "FixedIncome":  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        "Alternatives": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    },
    index=TICKERS,
    dtype=float,
)


def _make_gluc(
    group_min: dict = None,
    group_max: dict = None,
) -> GroupLowerUpperConstraints:
    return GroupLowerUpperConstraints(
        group_loadings=GROUP_LOADINGS.copy(),
        group_min_allocation=pd.Series(group_min, dtype=float) if group_min else None,
        group_max_allocation=pd.Series(group_max, dtype=float) if group_max else None,
    )


class LocalTests(Enum):
    FEASIBLE = 1
    ASSET_MAX_SUM_BELOW_GROUP_MIN = 2
    ASSET_MIN_SUM_ABOVE_GROUP_MAX = 3
    SINGLE_ASSET_MIN_EXCEEDS_GROUP_MAX = 4
    MULTIPLE_VIOLATIONS = 5


def run_local_test(local_test: LocalTests):

    if local_test == LocalTests.FEASIBLE:
        # all bounds comfortably within group bounds -> no error
        # per-group sum of min_weights: Eq=0.08, FI=0.06, Alt=0.06
        # per-group sum of max_weights: Eq=0.80, FI=0.60, Alt=0.60
        constraints = Constraints(
            is_long_only=True,
            min_weights=pd.Series(0.02, index=TICKERS),
            max_weights=pd.Series(0.20, index=TICKERS),
            group_lower_upper_constraints=_make_gluc(
                group_min={"Equities": 0.30, "FixedIncome": 0.20, "Alternatives": 0.10},
                group_max={"Equities": 0.60, "FixedIncome": 0.50, "Alternatives": 0.40},
            ),
        )
        print(f"PASSED: feasible constraints created successfully")

    elif local_test == LocalTests.ASSET_MAX_SUM_BELOW_GROUP_MIN:
        # Equities: 4 assets x 0.05 max = 0.20 < group_min 0.40 -> can't reach group floor
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
            assert "sum of asset max_weights" in str(e) and "Equities" in str(e)
            print(f"PASSED: caught expected ValueError:\n  {e}")

    elif local_test == LocalTests.ASSET_MIN_SUM_ABOVE_GROUP_MAX:
        # FixedIncome: 3 assets x 0.20 min = 0.60 > group_max 0.40 -> can't stay under ceiling
        try:
            Constraints(
                is_long_only=True,
                min_weights=pd.Series(
                    [0.0, 0.0, 0.0, 0.0, 0.20, 0.20, 0.20, 0.0, 0.0, 0.0],
                    index=TICKERS,
                ),
                max_weights=pd.Series(0.50, index=TICKERS),
                group_lower_upper_constraints=_make_gluc(
                    group_min={"Equities": 0.10, "FixedIncome": 0.10, "Alternatives": 0.05},
                    group_max={"Equities": 0.60, "FixedIncome": 0.40, "Alternatives": 0.30},
                ),
            )
            raise AssertionError("Expected ValueError was not raised")
        except ValueError as e:
            assert "sum of asset min_weights" in str(e) and "FixedIncome" in str(e)
            print(f"PASSED: caught expected ValueError:\n  {e}")

    elif local_test == LocalTests.SINGLE_ASSET_MIN_EXCEEDS_GROUP_MAX:
        # A8 min_weight=0.50 > Alternatives group_max=0.30 -> immediate infeasibility
        try:
            Constraints(
                is_long_only=True,
                min_weights=pd.Series(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.0, 0.0],
                    index=TICKERS,
                ),
                max_weights=pd.Series(1.0, index=TICKERS),
                group_lower_upper_constraints=_make_gluc(
                    group_min={"Equities": 0.10, "FixedIncome": 0.10, "Alternatives": 0.05},
                    group_max={"Equities": 0.60, "FixedIncome": 0.50, "Alternatives": 0.30},
                ),
            )
            raise AssertionError("Expected ValueError was not raised")
        except ValueError as e:
            assert "A8" in str(e) and "Alternatives" in str(e)
            print(f"PASSED: caught expected ValueError:\n  {e}")

    elif local_test == LocalTests.MULTIPLE_VIOLATIONS:
        # three simultaneous violations:
        #   Equities:     4 x 0.05 max = 0.20 < group_min 0.50
        #   FixedIncome:  3 x 0.25 min = 0.75 > group_max 0.40
        #   Alternatives: A8 min 0.35 > group_max 0.30
        try:
            Constraints(
                is_long_only=True,
                min_weights=pd.Series(
                    [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.35, 0.0, 0.0],
                    index=TICKERS,
                ),
                max_weights=pd.Series(
                    [0.05, 0.05, 0.05, 0.05, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
                    index=TICKERS,
                ),
                group_lower_upper_constraints=_make_gluc(
                    group_min={"Equities": 0.50, "FixedIncome": 0.10, "Alternatives": 0.05},
                    group_max={"Equities": 0.80, "FixedIncome": 0.40, "Alternatives": 0.30},
                ),
            )
            raise AssertionError("Expected ValueError was not raised")
        except ValueError as e:
            msg = str(e)
            assert "4 violation(s)" in msg
            assert "Equities" in msg and "FixedIncome" in msg and "A8" in msg
            print(f"PASSED: caught expected ValueError with 3 violations:\n  {e}")


if __name__ == '__main__':

    for test in LocalTests:
        print(f"\n{'='*60}")
        print(f"Running {test.name}")
        print(f"{'='*60}")
        run_local_test(local_test=test)
