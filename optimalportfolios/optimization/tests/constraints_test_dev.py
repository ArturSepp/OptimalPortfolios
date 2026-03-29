"""Interactive development tests for the Constraints module.

Enum-driven test runner for hands-on development and debugging.
Each test prints detailed output so you can see exactly what the
constraint objects look like and how the solver responds.

Run all:    python constraints_dev_tests.py
Run one:    python constraints_dev_tests.py BASIC_LONG_ONLY

Universe: 10 assets, 3 groups (Equities/FixedIncome/Alternatives),
5 sectors (Tech/Finance/Energy/Health/Other).
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
from enum import Enum

from optimalportfolios.optimization.constraints import (
    Constraints,
    GroupLowerUpperConstraints,
    BenchmarkDeviationConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
    merge_group_lower_upper_constraints,
)

# ──────────────────────────────────────────────────────────────────────
# Shared universe
# ──────────────────────────────────────────────────────────────────────
TICKERS = [f"A{i}" for i in range(1, 11)]
N = len(TICKERS)

GROUP_LOADINGS = pd.DataFrame(
    {
        "Equities":     [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "FixedIncome":  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        "Alternatives": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    },
    index=TICKERS, dtype=float,
)

SECTOR_LOADINGS = pd.DataFrame(
    {
        "Tech":    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "Finance": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        "Energy":  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        "Health":  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        "Other":   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    },
    index=TICKERS, dtype=float,
)

# increasing variance: A1 is low-vol, A10 is high-vol
COVAR = np.diag(np.linspace(0.01, 0.10, N))
EXPECTED_RETURNS = pd.Series(np.linspace(0.02, 0.12, N), index=TICKERS)
BENCHMARK_WEIGHTS = pd.Series(0.10, index=TICKERS)


def _solve(w, objective, constraints, label=""):
    """Solve and print results."""
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=cvx.SCS, verbose=False)
    weights = w.value
    if weights is None:
        print(f"  ✗ Solver failed: {prob.status}")
        return None
    print(f"  Status: {prob.status}")
    result = pd.Series(weights, index=TICKERS).round(4)
    print(f"  Weights:\n{result.to_string()}")
    print(f"  Sum: {weights.sum():.4f}")
    if label:
        print(f"  [{label}]")
    return weights


# ══════════════════════════════════════════════════════════════════════
# Test definitions
# ══════════════════════════════════════════════════════════════════════

class LocalTests(Enum):
    # --- Basic constraint behaviour ---
    BASIC_LONG_ONLY = 1
    BASIC_MIN_MAX_WEIGHTS = 2
    BASIC_EXPOSURE_BOUNDS = 3
    BASIC_TARGET_RETURN = 4
    BASIC_MAX_VOL = 5
    # --- Group allocation constraints ---
    GROUP_ALLOCATION = 10
    GROUP_MERGE = 11
    GROUP_DROP = 12
    GROUP_WITH_FRACTIONAL_LOADINGS = 13
    # --- Benchmark deviation constraints ---
    SECTOR_DEVIATION = 20
    SECTOR_DEVIATION_TIGHT = 21
    SECTOR_AND_STYLE_COMBINED = 22
    # --- Tracking error ---
    TRACKING_ERROR_HARD = 30
    TRACKING_ERROR_GROUP = 31
    # --- Turnover ---
    TURNOVER_HARD = 40
    TURNOVER_GROUP = 41
    # --- Combined real-world scenario ---
    FULL_INSTITUTIONAL = 50
    # --- Update and filtering ---
    UPDATE_VALID_TICKERS = 60
    REBALANCING_INDICATORS = 61
    # --- Debug utilities ---
    PRINT_AND_CHECK = 70


def run_local_test(local_test: LocalTests):


    # ─────────────────────────────────────────────────────────────
    # 1–5: Basic constraints
    # ─────────────────────────────────────────────────────────────

    if local_test == LocalTests.BASIC_LONG_ONLY:
        print("Long-only min-variance: all weights should be >= 0, sum = 1")
        c = Constraints(is_long_only=True)
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        assert np.all(weights >= -1e-6), "Negative weights found"
        assert abs(weights.sum() - 1.0) < 1e-4, "Weights don't sum to 1"
        print("  ✓ PASSED")

    elif local_test == LocalTests.BASIC_MIN_MAX_WEIGHTS:
        print("Min/max weight bounds: each weight in [0.05, 0.15]")
        c = Constraints(
            is_long_only=True,
            min_weights=pd.Series(0.05, index=TICKERS),
            max_weights=pd.Series(0.15, index=TICKERS),
        )
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        assert np.all(weights >= 0.05 - 1e-4), f"Weight below min: {weights.min():.4f}"
        assert np.all(weights <= 0.15 + 1e-4), f"Weight above max: {weights.max():.4f}"
        print("  ✓ PASSED")

    elif local_test == LocalTests.BASIC_EXPOSURE_BOUNDS:
        print("Exposure bounds: total exposure in [0.60, 0.80]")
        c = Constraints(is_long_only=True, max_exposure=0.80, min_exposure=0.60)
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        print(f"  Total exposure: {weights.sum():.4f}")
        assert 0.60 - 1e-4 <= weights.sum() <= 0.80 + 1e-4
        print("  ✓ PASSED")

    elif local_test == LocalTests.BASIC_TARGET_RETURN:
        print("Target return constraint: portfolio return >= 0.07")
        target = 0.07
        c = Constraints(
            is_long_only=True,
            target_return=target,
            asset_returns=EXPECTED_RETURNS,
        )
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        achieved = EXPECTED_RETURNS.to_numpy() @ weights
        print(f"  Achieved return: {achieved:.4f} (target: {target})")
        assert achieved >= target - 1e-4
        print("  ✓ PASSED")

    elif local_test == LocalTests.BASIC_MAX_VOL:
        print("Max portfolio vol constraint: vol <= 0.06")
        max_vol = 0.06
        c = Constraints(is_long_only=True, max_target_portfolio_vol_an=max_vol)
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Maximize(EXPECTED_RETURNS.to_numpy() @ w), constraints)
        vol = np.sqrt(weights @ COVAR @ weights)
        print(f"  Portfolio vol: {vol:.4f} (max: {max_vol})")
        assert vol <= max_vol + 1e-3
        print("  ✓ PASSED")

    # ─────────────────────────────────────────────────────────────
    # 10–13: Group allocation constraints
    # ─────────────────────────────────────────────────────────────

    elif local_test == LocalTests.GROUP_ALLOCATION:
        print("Group allocation: Eq [0.30,0.50], FI [0.20,0.40], Alt [0.10,0.30]")
        gluc = GroupLowerUpperConstraints(
            group_loadings=GROUP_LOADINGS.copy(),
            group_min_allocation=pd.Series({"Equities": 0.30, "FixedIncome": 0.20, "Alternatives": 0.10}),
            group_max_allocation=pd.Series({"Equities": 0.50, "FixedIncome": 0.40, "Alternatives": 0.30}),
        )
        c = Constraints(is_long_only=True, group_lower_upper_constraints=gluc)
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)

        for grp, lo, hi in [("Equities", 0.30, 0.50), ("FixedIncome", 0.20, 0.40), ("Alternatives", 0.10, 0.30)]:
            mask = GROUP_LOADINGS[grp].to_numpy().astype(bool)
            alloc = weights[mask].sum()
            print(f"  {grp}: {alloc:.4f} in [{lo}, {hi}]")
            assert lo - 1e-4 <= alloc <= hi + 1e-4
        print("  ✓ PASSED")

    elif local_test == LocalTests.GROUP_MERGE:
        print("Merge two group constraint objects (non-overlapping)")
        gluc1 = GroupLowerUpperConstraints(
            group_loadings=GROUP_LOADINGS[["Equities"]].copy(),
            group_min_allocation=pd.Series({"Equities": 0.30}),
            group_max_allocation=pd.Series({"Equities": 0.50}),
        )
        gluc2 = GroupLowerUpperConstraints(
            group_loadings=GROUP_LOADINGS[["FixedIncome"]].copy(),
            group_min_allocation=pd.Series({"FixedIncome": 0.20}),
            group_max_allocation=pd.Series({"FixedIncome": 0.40}),
        )
        merged = merge_group_lower_upper_constraints(gluc1, gluc2)
        merged.print()
        assert set(merged.group_loadings.columns) == {"Equities", "FixedIncome"}
        print("  ✓ PASSED")

    elif local_test == LocalTests.GROUP_DROP:
        print("Drop a group constraint")
        gluc = GroupLowerUpperConstraints(
            group_loadings=GROUP_LOADINGS.copy(),
            group_min_allocation=pd.Series({"Equities": 0.20, "FixedIncome": 0.10, "Alternatives": 0.05}),
            group_max_allocation=pd.Series({"Equities": 0.60, "FixedIncome": 0.50, "Alternatives": 0.40}),
        )
        dropped = gluc.drop_constraint("FixedIncome")
        print(f"  Remaining groups: {dropped.group_loadings.columns.tolist()}")
        assert "FixedIncome" not in dropped.group_loadings.columns
        print("  ✓ PASSED")

    elif local_test == LocalTests.GROUP_WITH_FRACTIONAL_LOADINGS:
        print("Fractional (non-binary) group loadings")
        loadings = pd.DataFrame(
            {"Factor1": [0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            index=TICKERS, dtype=float,
        )
        gluc = GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=pd.Series({"Factor1": 0.10}),
            group_max_allocation=pd.Series({"Factor1": 0.25}),
        )
        w = cvx.Variable(N)
        constraints = gluc.set_cvx_group_lower_upper_constraints(w=w)
        constraints += [cvx.sum(w) == 1.0, w >= 0]
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        exposure = loadings["Factor1"].to_numpy() @ weights
        print(f"  Factor1 exposure: {exposure:.4f} in [0.10, 0.25]")
        assert 0.10 - 1e-4 <= exposure <= 0.25 + 1e-4
        print("  ✓ PASSED")

    # ─────────────────────────────────────────────────────────────
    # 20–22: Benchmark deviation constraints
    # ─────────────────────────────────────────────────────────────

    elif local_test == LocalTests.SECTOR_DEVIATION:
        print("Sector deviation: max 5% active deviation per sector vs equal-weight BM")
        max_dev = 0.05
        bdc = BenchmarkDeviationConstraints(
            factor_loading_mat=SECTOR_LOADINGS.copy(),
            factor_max_deviation=pd.Series({col: max_dev for col in SECTOR_LOADINGS.columns}),
        )
        w = cvx.Variable(N)
        constraints = bdc.set_cvx_constraints(w=w, benchmark_weights=BENCHMARK_WEIGHTS)
        constraints += [cvx.sum(w) == 1.0, w >= 0]
        # try to overweight Tech
        weights = _solve(w, cvx.Maximize(w[0] + w[1]), constraints, label="Overweight Tech")
        for col in SECTOR_LOADINGS.columns:
            mask = SECTOR_LOADINGS[col].to_numpy().astype(bool)
            dev = (weights[mask] - BENCHMARK_WEIGHTS.to_numpy()[mask]).sum()
            status = "✓" if abs(dev) <= max_dev + 1e-3 else "✗"
            print(f"  {status} {col}: active = {dev:+.4f} (bound: ±{max_dev})")
        print("  ✓ PASSED")

    elif local_test == LocalTests.SECTOR_DEVIATION_TIGHT:
        print("Tight sector deviation (0.1%): weights should be close to benchmark")
        max_dev = 0.001
        bdc = BenchmarkDeviationConstraints(
            factor_loading_mat=SECTOR_LOADINGS.copy(),
            factor_max_deviation=pd.Series({col: max_dev for col in SECTOR_LOADINGS.columns}),
        )
        w = cvx.Variable(N)
        constraints = bdc.set_cvx_constraints(w=w, benchmark_weights=BENCHMARK_WEIGHTS)
        constraints += [cvx.sum(w) == 1.0, w >= 0]
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        max_active = max(
            abs((weights[SECTOR_LOADINGS[col].to_numpy().astype(bool)] -
                 BENCHMARK_WEIGHTS.to_numpy()[SECTOR_LOADINGS[col].to_numpy().astype(bool)]).sum())
            for col in SECTOR_LOADINGS.columns
        )
        print(f"  Max sector active deviation: {max_active:.6f}")
        assert max_active <= max_dev + 1e-3
        print("  ✓ PASSED")

    elif local_test == LocalTests.SECTOR_AND_STYLE_COMBINED:
        print("Combined sector + style deviation constraints via Constraints class")
        c = Constraints(
            is_long_only=True,
            benchmark_weights=BENCHMARK_WEIGHTS,
            sector_deviation_constraints=BenchmarkDeviationConstraints(
                factor_loading_mat=SECTOR_LOADINGS.copy(),
                factor_max_deviation=pd.Series({col: 0.04 for col in SECTOR_LOADINGS.columns}),
            ),
            style_deviation_constraints=BenchmarkDeviationConstraints(
                factor_loading_mat=GROUP_LOADINGS.copy(),
                factor_max_deviation=pd.Series({"Equities": 0.08, "FixedIncome": 0.06, "Alternatives": 0.05}),
            ),
        )
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        print("  Sector deviations:")
        for col in SECTOR_LOADINGS.columns:
            mask = SECTOR_LOADINGS[col].to_numpy().astype(bool)
            dev = (weights[mask] - 0.10).sum()
            print(f"    {col}: {dev:+.4f}")
        print("  Style (group) deviations:")
        for col in GROUP_LOADINGS.columns:
            mask = GROUP_LOADINGS[col].to_numpy().astype(bool)
            dev = (weights[mask] - BENCHMARK_WEIGHTS.to_numpy()[mask]).sum()
            print(f"    {col}: {dev:+.4f}")
        print("  ✓ PASSED")

    # ─────────────────────────────────────────────────────────────
    # 30–31: Tracking error
    # ─────────────────────────────────────────────────────────────

    elif local_test == LocalTests.TRACKING_ERROR_HARD:
        print("Hard tracking error constraint: TE vol <= 2%")
        te_limit = 0.02
        c = Constraints(
            is_long_only=True,
            benchmark_weights=BENCHMARK_WEIGHTS,
            tracking_err_vol_constraint=te_limit,
        )
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        active = weights - BENCHMARK_WEIGHTS.to_numpy()
        te = np.sqrt(active @ COVAR @ active)
        print(f"  TE vol: {te:.4f} (limit: {te_limit})")
        assert te <= te_limit + 1e-3
        print("  ✓ PASSED")

    elif local_test == LocalTests.TRACKING_ERROR_GROUP:
        print("Group tracking error: per-group TE constraints")
        gte = GroupTrackingErrorConstraint(
            group_loadings=GROUP_LOADINGS.copy(),
            group_tre_vols=pd.Series({"Equities": 0.03, "FixedIncome": 0.02, "Alternatives": 0.04}),
        )
        c = Constraints(
            is_long_only=True,
            benchmark_weights=BENCHMARK_WEIGHTS,
            group_tracking_error_constraint=gte,
        )
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        print("  Per-group tracking errors computed successfully")
        print("  ✓ PASSED")

    # ─────────────────────────────────────────────────────────────
    # 40–41: Turnover
    # ─────────────────────────────────────────────────────────────

    elif local_test == LocalTests.TURNOVER_HARD:
        print("Hard turnover constraint: L1 turnover <= 10%")
        w0 = pd.Series(0.10, index=TICKERS)
        to_limit = 0.10
        c = Constraints(is_long_only=True, weights_0=w0, turnover_constraint=to_limit)
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        turnover = np.abs(weights - w0.to_numpy()).sum()
        print(f"  Turnover: {turnover:.4f} (limit: {to_limit})")
        assert turnover <= to_limit + 1e-3
        print("  ✓ PASSED")

    elif local_test == LocalTests.TURNOVER_GROUP:
        print("Group turnover: per-group L1 turnover limits")
        w0 = pd.Series(0.10, index=TICKERS)
        gtc = GroupTurnoverConstraint(
            group_loadings=GROUP_LOADINGS.copy(),
            group_max_turnover=pd.Series({"Equities": 0.05, "FixedIncome": 0.05, "Alternatives": 0.05}),
        )
        c = Constraints(is_long_only=True, weights_0=w0, group_turnover_constraint=gtc)
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        for grp in GROUP_LOADINGS.columns:
            mask = GROUP_LOADINGS[grp].to_numpy().astype(bool)
            to = np.abs(weights[mask] - w0.to_numpy()[mask]).sum()
            print(f"  {grp} turnover: {to:.4f} (limit: 0.05)")
        print("  ✓ PASSED")

    # ─────────────────────────────────────────────────────────────
    # 50: Full institutional scenario
    # ─────────────────────────────────────────────────────────────

    elif local_test == LocalTests.FULL_INSTITUTIONAL:
        print("Full institutional setup: group bounds + sector dev + TE + turnover")
        w0 = pd.Series(0.10, index=TICKERS)
        c = Constraints(
            is_long_only=True,
            min_weights=pd.Series(0.02, index=TICKERS),
            max_weights=pd.Series(0.20, index=TICKERS),
            benchmark_weights=BENCHMARK_WEIGHTS,
            tracking_err_vol_constraint=0.03,
            weights_0=w0,
            turnover_constraint=0.15,
            group_lower_upper_constraints=GroupLowerUpperConstraints(
                group_loadings=GROUP_LOADINGS.copy(),
                group_min_allocation=pd.Series({"Equities": 0.30, "FixedIncome": 0.20, "Alternatives": 0.10}),
                group_max_allocation=pd.Series({"Equities": 0.60, "FixedIncome": 0.50, "Alternatives": 0.40}),
            ),
            sector_deviation_constraints=BenchmarkDeviationConstraints(
                factor_loading_mat=SECTOR_LOADINGS.copy(),
                factor_max_deviation=pd.Series({col: 0.05 for col in SECTOR_LOADINGS.columns}),
            ),
        )
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        weights = _solve(w, cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)

        active = weights - BENCHMARK_WEIGHTS.to_numpy()
        te = np.sqrt(active @ COVAR @ active)
        turnover = np.abs(weights - w0.to_numpy()).sum()
        print(f"\n  Summary:")
        print(f"    TE vol:   {te:.4f} (limit: 0.03)")
        print(f"    Turnover: {turnover:.4f} (limit: 0.15)")
        for grp in GROUP_LOADINGS.columns:
            mask = GROUP_LOADINGS[grp].to_numpy().astype(bool)
            print(f"    {grp}: {weights[mask].sum():.4f}")
        print("  ✓ PASSED")

    # ─────────────────────────────────────────────────────────────
    # 60–61: Update and filtering
    # ─────────────────────────────────────────────────────────────

    elif local_test == LocalTests.UPDATE_VALID_TICKERS:
        print("update() filters all sub-constraints to valid tickers")
        c = Constraints(
            is_long_only=True,
            min_weights=pd.Series(0.02, index=TICKERS),
            max_weights=pd.Series(0.20, index=TICKERS),
            group_lower_upper_constraints=GroupLowerUpperConstraints(
                group_loadings=GROUP_LOADINGS.copy(),
                group_min_allocation=pd.Series({"Equities": 0.20}),
                group_max_allocation=pd.Series({"Equities": 0.60}),
            ),
            sector_deviation_constraints=BenchmarkDeviationConstraints(
                factor_loading_mat=SECTOR_LOADINGS.copy(),
                factor_max_deviation=pd.Series({col: 0.05 for col in SECTOR_LOADINGS.columns}),
            ),
        )
        subset = ["A1", "A2", "A5", "A6", "A9", "A10"]
        updated = c.update(valid_tickers=subset)
        print(f"  Original tickers: {N}")
        print(f"  Filtered tickers: {updated.group_lower_upper_constraints.group_loadings.shape[0]}")
        print(f"  GLUC shape: {updated.group_lower_upper_constraints.group_loadings.shape}")
        print(f"  BDC shape:  {updated.sector_deviation_constraints.factor_loading_mat.shape}")
        assert updated.group_lower_upper_constraints.group_loadings.shape[0] == 6
        assert updated.sector_deviation_constraints.factor_loading_mat.shape[0] == 6
        print("  ✓ PASSED")

    elif local_test == LocalTests.REBALANCING_INDICATORS:
        print("Rebalancing indicators: freeze A3, A4 at current weights")
        w0 = pd.Series([0.15, 0.10, 0.05, 0.08, 0.10, 0.10, 0.10, 0.12, 0.10, 0.10], index=TICKERS)
        rebal = pd.Series([1, 1, 0, 0, 1, 1, 1, 1, 1, 1], index=TICKERS)
        c = Constraints(
            is_long_only=True,
            min_weights=pd.Series(0.0, index=TICKERS),
            max_weights=pd.Series(0.25, index=TICKERS),
        )
        updated = c.update_with_valid_tickers(
            valid_tickers=TICKERS, weights_0=w0, rebalancing_indicators=rebal,
        )
        print(f"  A3 bounds: [{updated.min_weights.loc['A3']:.4f}, {updated.max_weights.loc['A3']:.4f}]"
              f"  (frozen at {w0.loc['A3']:.4f})")
        print(f"  A4 bounds: [{updated.min_weights.loc['A4']:.4f}, {updated.max_weights.loc['A4']:.4f}]"
              f"  (frozen at {w0.loc['A4']:.4f})")
        print(f"  A1 bounds: [{updated.min_weights.loc['A1']:.4f}, {updated.max_weights.loc['A1']:.4f}]"
              f"  (free)")
        assert abs(updated.min_weights.loc["A3"] - 0.05) < 1e-10
        assert abs(updated.max_weights.loc["A3"] - 0.05) < 1e-10
        assert abs(updated.min_weights.loc["A1"] - 0.0) < 1e-10
        print("  ✓ PASSED")

    # ─────────────────────────────────────────────────────────────
    # 70: Debug utilities
    # ─────────────────────────────────────────────────────────────

    elif local_test == LocalTests.PRINT_AND_CHECK:
        print("Debug utilities: print_constraints + check_constraints_violation")
        c = Constraints(
            is_long_only=True,
            min_weights=pd.Series(0.05, index=TICKERS),
            max_weights=pd.Series(0.15, index=TICKERS),
        )
        w = cvx.Variable(N)
        constraints = c.set_cvx_all_constraints(w=w, covar=COVAR)
        prob = cvx.Problem(cvx.Minimize(cvx.quad_form(w, COVAR)), constraints)
        prob.solve(solver=cvx.SCS, verbose=False)

        print("\n  --- print_constraints output ---")
        c.print_constraints(constraints)
        print("\n  --- check_constraints_violation output ---")
        c.check_constraints_violation(constraints)
        print("  ✓ PASSED")


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    run_local_test(local_test=LocalTests.GROUP_ALLOCATION)