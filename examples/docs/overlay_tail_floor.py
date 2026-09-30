"""Canonical script of docs/overlay_tail_floor.md.

The page's eight Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: inputs rebuilt entry by entry from the factor constants, the no-floor optimum by
linear algebra, the binding-floor optimum from its active face with a first-order certificate,
the named-row floors against the homogeneous shifted-coefficient encoding, which also draws the
figure's sweep, risk as an explicit quadratic form, and the compiled CVXPY and SciPy rows
evaluated at known points. Sockets are blocked throughout.

The page imports the repository example ``examples/solvers/overlay_tail_floor.py``, so the
script runs from a source checkout with the core install and needs no data file or random seed:

    python -m examples.docs.overlay_tail_floor

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import contextlib
from dataclasses import replace
import inspect
import socket
from unittest.mock import patch

import cvxpy as cvx
import numpy as np
import pandas as pd

import optimalportfolios as opt
from examples.solvers import overlay_tail_floor as solver_example

ASSETS = ['Core', 'Defensive A', 'Defensive B', 'Carry C', 'Carry D']
# Inputs of the solver example's factory: annual volatilities, Sharpe ratios, betas to the core
# (the single factor) and bear-regime scores.
VOLS = [0.10, 0.15, 0.12, 0.10, 0.08]
SHARPES = [0.60, 0.35, 0.30, 0.90, 1.00]
BETAS = [1.00, -0.30, -0.20, 0.60, 0.40]
BEAR_SCORES = [-0.80, 0.60, 0.35, -0.45, -0.30]
IDIOSYNCRATIC_FLOOR = 1e-6
OVERLAY_BUDGET = 1.0
# The floor sweep of the figure: 90 levels 0.001 apart, from non-binding to near the reachable
# maximum of 0.01.
FLOOR_FIRST = -0.08
FLOOR_LAST = 0.009
FLOOR_COUNT = 90
OFFLINE = AssertionError('The overlay article must execute offline')


@contextlib.contextmanager
def offline():
    """Deny socket connections; used as the decorator of ``main``."""
    with (patch.object(socket, 'create_connection', side_effect=OFFLINE),
          patch.object(socket.socket, 'connect', side_effect=OFFLINE)):
        yield


def factor_model_inputs() -> tuple:
    """Rebuild the factory's means, covariance and coefficients entry by entry from constants."""
    n = len(ASSETS)
    factor_variance = VOLS[0] ** 2
    covar = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            covar[i, j] = BETAS[i] * BETAS[j] * factor_variance
        covar[i, i] += max(VOLS[i] ** 2 - BETAS[i] ** 2 * factor_variance, IDIOSYNCRATIC_FLOOR)
    means = [sharpe * vol for sharpe, vol in zip(SHARPES, VOLS)]
    coefficients = [vol * score for vol, score in zip(VOLS, BEAR_SCORES)]
    return (pd.Series(means, index=ASSETS), pd.DataFrame(covar, index=ASSETS, columns=ASSETS),
            pd.Series(coefficients, index=ASSETS))


def floor_levels() -> np.ndarray:
    """The floors of the figure's sweep."""
    return np.linspace(FLOOR_FIRST, FLOOR_LAST, FLOOR_COUNT)


def fixed_core_constraints(tickers: pd.Index, budget: float) -> opt.Constraints:
    """The page's constraints: core fixed at 1, long-only overlays, total exposure 1 + budget."""
    min_weights = pd.Series(0.0, index=tickers)
    min_weights['Core'] = 1.0
    max_weights = pd.Series(budget, index=tickers)
    max_weights['Core'] = 1.0
    return opt.Constraints(is_long_only=True, min_weights=min_weights, max_weights=max_weights,
                           min_exposure=1.0 + budget, max_exposure=1.0 + budget)


def floor_path(base: opt.Constraints, means: pd.Series, covar: pd.DataFrame, a: pd.Series,
               floors: np.ndarray) -> tuple:
    """Solve the homogeneously encoded overlay problem at each floor; weights and outcomes."""
    weights, outcomes = {}, []
    for floor in floors:
        spec = replace(base, asset_returns=a - floor / base.max_exposure, target_return=0.0)
        outcome = opt.cvx_maximize_portfolio_sharpe(covar=covar.to_numpy(),
                                                    means=means.to_numpy(), constraints=spec)
        weights[floor] = outcome.weights
        outcomes.append(outcome)
    return pd.DataFrame.from_dict(weights, orient='index', columns=means.index), outcomes


def equality_optimum(covar: np.ndarray, means: np.ndarray, budget: float) -> np.ndarray:
    """No-floor optimum from the two equality rows alone: min y'Sy, mu'y = 1 + W, sleeve = W y_c."""
    sleeve_row = np.r_[-budget, np.ones(len(means) - 1)]
    equations = np.vstack([means, sleeve_row])
    inverse_rows = np.linalg.solve(covar, equations.T)
    y = inverse_rows @ np.linalg.solve(equations @ inverse_rows, [1.0 + budget, 0.0])
    return y / y[0]


def defensive_face(a: np.ndarray, floor: float) -> np.ndarray:
    """Weights where the floor binds and only the two defensive overlays share a unit sleeve."""
    fraction = (floor - a[0] - a[2]) / (a[1] - a[2])
    return np.array([1.0, fraction, 1.0 - fraction, 0.0, 0.0])


def supporting_gradient(means: np.ndarray, covar: np.ndarray, a: np.ndarray,
                        w: np.ndarray) -> tuple:
    """Ratio, floor multiplier and overlay supporting gradient of the Sharpe ratio at ``w``."""
    risk = np.sqrt(w @ covar @ w)
    ratio = (means @ w) / risk
    gradient = means - ratio * (covar @ w) / risk
    multiplier = (gradient[2] - gradient[1]) / (a[1] - a[2])
    return ratio, multiplier, gradient[1:] + multiplier * a[1:]


def path_risk(weights: pd.DataFrame, means: pd.Series, covar: pd.DataFrame) -> pd.DataFrame:
    """Expected excess return, volatility and model Sharpe of each row of weights."""
    w = weights.to_numpy()
    expected = w @ means.to_numpy()
    volatility = np.sqrt(np.einsum('fi,ij,fj->f', w, covar.to_numpy(), w))
    return pd.DataFrame({'expected_excess': expected, 'volatility': volatility,
                         'model_sharpe': expected / volatility}, index=weights.index)


@offline()
def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    from dataclasses import replace
    import numpy as np
    import pandas as pd
    import optimalportfolios as opt
    from examples.solvers.overlay_tail_floor import (
        create_synthetic_inputs, solve_overlay_tail_floor,
    )

    means, covar, a = create_synthetic_inputs()
    tickers = covar.index
    assert tickers.equals(covar.columns)
    assert means.index.equals(tickers) and a.index.equals(tickers)
    assert np.isfinite(covar.to_numpy()).all()
    assert np.isfinite(means).all() and np.isfinite(a).all()
    assert np.linalg.eigvalsh(covar).min() > 0.0
    inputs = pd.DataFrame({"Expected excess return": means, "Linear coefficient": a})
    print(inputs.round(4))

    # The factory equals the constants rebuilt entry by entry: means are Sharpe ratio times
    # volatility, coefficients volatility times bear score, and the core is the only factor.
    reference_means, reference_covar, reference_a = factor_model_inputs()
    assert list(tickers) == ASSETS
    np.testing.assert_allclose(means, reference_means, rtol=0, atol=1e-15)
    np.testing.assert_allclose(covar, reference_covar, rtol=0, atol=1e-15)
    np.testing.assert_allclose(a, reference_a, rtol=0, atol=1e-15)
    # The page's input table.
    np.testing.assert_allclose(inputs, [[0.06, -0.08], [0.0525, 0.09], [0.036, 0.042],
                                        [0.09, -0.045], [0.08, -0.024]], rtol=0, atol=1e-14)
    # The factor volatility is the core's 0.10; the 1e-6 floor binds for the core alone, whose
    # diagonal is 0.010001; every feasible allocation has a positive expected excess return.
    assert VOLS[0] == 0.10 and IDIOSYNCRATIC_FLOOR == 1e-6
    raw_residuals = [vol ** 2 - beta ** 2 * VOLS[0] ** 2 for vol, beta in zip(VOLS, BETAS)]
    assert [value < IDIOSYNCRATIC_FLOOR for value in raw_residuals] == [True] + [False] * 4
    np.testing.assert_allclose(np.diag(covar), [0.010001, 0.0225, 0.0144, 0.01, 0.0064],
                               rtol=1e-12)
    np.testing.assert_allclose(covar.to_numpy()[0, 1:], [-0.003, -0.002, 0.006, 0.004],
                               rtol=1e-12)
    assert (means > 0.0).all()

    overlay_budget = 1.0
    total_exposure = 1.0 + overlay_budget
    min_weights = pd.Series(0.0, index=tickers)
    min_weights["Core"] = 1.0
    max_weights = pd.Series(overlay_budget, index=tickers)
    max_weights["Core"] = 1.0
    base = opt.Constraints(
        is_long_only=True, min_weights=min_weights, max_weights=max_weights,
        min_exposure=total_exposure, max_exposure=total_exposure,
    )

    floors = {"No floor": None, "Zero floor": 0.0, "Floor 0.005": 0.005}
    specifications = {}
    outcomes = {}
    for label, floor in floors.items():
        spec = base if floor is None else replace(
            base, linear_constraints=opt.LinearConstraints(
                loadings=a.to_frame("floor"), lower=pd.Series({"floor": floor}),
            ),
        )
        specifications[label] = spec
        outcomes[label] = opt.cvx_maximize_portfolio_sharpe(
            covar=covar.to_numpy(), means=means.to_numpy(), constraints=spec,
            context=f"overlay article: {label}",
        )
        assert outcomes[label].accepted and outcomes[label].compliant

    allocation = pd.DataFrame(
        {label: outcome.weights for label, outcome in outcomes.items()}, index=tickers,
    )
    print(allocation.round(6))

    # Every case keeps the capital mandate: core 1.0, sleeve 1.0, total 2.0, within the bounds,
    # from a CLARABEL solve with no fallback.
    assert total_exposure == 2.0
    for label, outcome in outcomes.items():
        w = allocation[label]
        assert outcome.solver == "CLARABEL" and outcome.fallback_source is None
        assert abs(w["Core"] - 1.0) < 1e-6 and abs(w.drop("Core").sum() - 1.0) < 1e-6
        assert abs(w.sum() - 2.0) < 1e-6
        assert (w >= -1e-6).all() and (w <= 1.0 + 1e-6).all()
    # The page's allocation table; carry weights of the two floors display as zero.
    np.testing.assert_allclose(allocation, [[1.0, 1.0, 1.0],
                                            [0.283145, 0.791667, 0.895833],
                                            [0.213616, 0.208333, 0.104167],
                                            [0.128974, 0.0, 0.0],
                                            [0.374264, 0.0, 0.0]], rtol=0, atol=5.1e-7)
    # No floor: the two equality rows solved by linear algebra give interior overlays, so the
    # bounds are inactive and this is the optimum.
    no_floor = equality_optimum(covar.to_numpy(), means.to_numpy(), overlay_budget)
    assert (no_floor[1:] > 0.0).all() and (no_floor[1:] < 1.0).all()
    np.testing.assert_allclose(allocation["No floor"], no_floor, rtol=0, atol=2e-6)
    # Binding floors: the active two-defensive face, with a positive floor multiplier and a
    # supporting gradient under which the unused carry overlays are strictly worse. Since
    # mu'w - ratio * sqrt(w'Sw) is concave and zero there, this certifies the global maximum.
    for label in ("Zero floor", "Floor 0.005"):
        face = defensive_face(a.to_numpy(), floors[label])
        np.testing.assert_allclose(allocation[label], face, rtol=0, atol=2e-6)
        assert abs(a.to_numpy() @ face - floors[label]) < 1e-14
        ratio, multiplier, supporting = supporting_gradient(
            means.to_numpy(), covar.to_numpy(), a.to_numpy(), face)
        assert ratio > 0.0 and multiplier > 0.0
        assert abs(supporting[0] - supporting[1]) < 1e-12
        assert (supporting[2:] < supporting[0]).all()
    # The positive floor holds more Defensive A; normalising to one would halve the core and the
    # floor contribution.
    defensive_a = allocation.loc["Defensive A"]
    assert defensive_a["Floor 0.005"] > defensive_a["Zero floor"] > defensive_a["No floor"]
    normalised = allocation["Floor 0.005"] / allocation["Floor 0.005"].sum()
    assert abs(normalised["Core"] - 0.5) < 1e-6 and abs(a @ normalised - 0.0025) < 1e-6
    # Each floor is one named row on the unshifted coefficients; no return row is configured.
    for label in ("Zero floor", "Floor 0.005"):
        row = specifications[label].linear_constraints
        pd.testing.assert_series_equal(row.loadings["floor"], a, check_names=False)
        assert row.lower["floor"] == floors[label] and row.upper is None
        assert specifications[label].asset_returns is None
        assert specifications[label].target_return is None
    assert specifications["No floor"].linear_constraints is None
    # The homogeneous cross-check shifts all five coefficients by b0 / E, 0.0025 for the positive
    # floor, the core included, with a zero right side; it reproduces both floored allocations.
    for label in ("Zero floor", "Floor 0.005"):
        shifted = replace(base, asset_returns=a - floors[label] / total_exposure,
                          target_return=0.0)
        cross_check = opt.cvx_maximize_portfolio_sharpe(covar.to_numpy(), means.to_numpy(),
                                                        shifted)
        assert cross_check.accepted and cross_check.compliant
        np.testing.assert_allclose(cross_check.weights, allocation[label], rtol=0, atol=1e-8)
    np.testing.assert_allclose(shifted.asset_returns, a - 0.0025, rtol=0, atol=1e-15)
    # The identity a'w - b0 = a~'w holds on the exposure equality for any budget and scales with
    # k; off the equality it misses by b0 (1 - e'w / E), so E must match the budget.
    coefficients = a.to_numpy()
    for budget, floor in ((0.5, -0.02), (1.0, 0.0), (1.0, 0.005), (2.0, 0.02)):
        exposure = 1.0 + budget
        point = np.r_[1.0, budget * np.array([0.1, 0.2, 0.3, 0.4])]
        encoded = coefficients - floor / exposure
        assert abs(encoded @ point - (coefficients @ point - floor)) < 1e-14
        for scale in (0.25, 3.0, 20.0):
            assert abs(encoded @ (scale * point) - scale * (coefficients @ point - floor)) < 1e-13
        off_budget = point + np.r_[0.0, 0.0, 0.0, 0.0, 0.2]
        gap = encoded @ off_budget - (coefficients @ off_budget - floor)
        assert abs(gap - floor * (1.0 - off_budget.sum() / exposure)) < 1e-14
    # The compiled CVXPY rows scale the core, the exposure and the floor by k: a feasible point
    # stays feasible at every scale, an infeasible one does not, and k = 0 admits only y = 0.
    y, k = cvx.Variable(5), cvx.Variable(nonneg=True)
    rows = specifications["Floor 0.005"].set_cvx_all_constraints(
        w=y, covar=covar.to_numpy(), exposure_scaler=k)
    for scale in (0.2, 2.0, 20.0):
        y.value, k.value = scale * np.array([1.0, 1.0, 0.0, 0.0, 0.0]), scale
        assert all(np.max(row.violation()) < 1e-9 for row in rows)
    y.value, k.value = 2.0 * np.array([1.0, 0.0, 0.0, 0.0, 1.0]), 2.0
    assert any(np.max(row.violation()) > 0.1 for row in rows)
    y.value, k.value = np.array([0.0, 0.1, 0.0, 0.0, 0.0]), 0.0
    assert any(np.max(row.violation()) > 0.05 for row in rows)
    # Several sleeves: disjoint group rows with equal bounds keep the fixed core and the floor.
    groups = opt.GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Defensive": [0.0, 1.0, 1.0, 0.0, 0.0],
                                     "Carry": [0.0, 0.0, 0.0, 1.0, 1.0]}, index=tickers),
        group_min_allocation=pd.Series({"Defensive": 0.95, "Carry": 0.05}),
        group_max_allocation=pd.Series({"Defensive": 0.95, "Carry": 0.05}),
    )
    sleeves = opt.cvx_maximize_portfolio_sharpe(
        covar.to_numpy(), means.to_numpy(),
        replace(specifications["Zero floor"], group_lower_upper_constraints=groups))
    assert sleeves.accepted and sleeves.compliant
    np.testing.assert_allclose(groups.group_loadings.T @ sleeves.weights, [0.95, 0.05],
                               atol=1e-6)
    assert abs(sleeves.weights[0] - 1.0) < 1e-6 and coefficients @ sleeves.weights >= -1e-6
    # Dispatch reads only the exposure fields: equal group rows under an exposure band still
    # route to SLSQP.
    whole_sleeve = opt.GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Sleeve": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=tickers),
        group_min_allocation=pd.Series({"Sleeve": 1.0}),
        group_max_allocation=pd.Series({"Sleeve": 1.0}),
    )
    routed = opt.cvx_maximize_portfolio_sharpe(
        covar.to_numpy(), means.to_numpy(),
        replace(base, min_exposure=1.8, group_lower_upper_constraints=whole_sleeve))
    assert routed.solver == "SLSQP"

    risk_date = pd.Timestamp("2024-12-31")  # Synthetic risk-model key, not a data cutoff.
    risk_model = opt.build_risk_model({risk_date: covar})
    zero_benchmark = pd.Series(0.0, index=tickers)
    metrics = {}
    for label in floors:
        w = allocation[label]
        volatility = risk_model.compute_tre_at_date(
            benchmark_weights=zero_benchmark, portfolio_weights=w, date=risk_date,
        )
        expected_excess = float(means @ w)
        metrics[label] = {
            "Expected excess": expected_excess,
            "Volatility": volatility,
            "Model excess Sharpe": expected_excess / volatility,
            "Linear contribution": float(a @ w),
        }
    summary = pd.DataFrame.from_dict(metrics, orient="index")
    print(summary.round(6))

    # Tracking error against zero is the quadratic-form volatility, and the ratio is the model
    # expected excess return over it.
    for label in floors:
        w = allocation[label].to_numpy()
        risk = np.sqrt(w @ covar.to_numpy() @ w)
        expected = means.to_numpy() @ w
        assert abs(summary.at[label, "Volatility"] - risk) < 1e-12
        assert abs(summary.at[label, "Expected excess"] - expected) < 1e-12
        assert abs(summary.at[label, "Model excess Sharpe"] - expected / risk) < 1e-12
        assert abs(summary.at[label, "Linear contribution"] - coefficients @ w) < 1e-12
    # The page's summary table; the ratio falls as the floor tightens.
    np.testing.assert_allclose(summary, [[0.124104, 0.123441, 1.005372, -0.060331],
                                         [0.109063, 0.139076, 0.784193, 0.0],
                                         [0.110781, 0.150114, 0.737981, 0.005]],
                               rtol=0, atol=5.1e-7)
    sharpe = summary["Model excess Sharpe"]
    assert sharpe["No floor"] > sharpe["Zero floor"] > sharpe["Floor 0.005"]
    # Both floors bind at equality, with Defensive A at (b0 + 0.038) / 0.048: 0.038 is minus the
    # core and Defensive B coefficients, 0.048 the Defensive A minus Defensive B coefficient.
    assert abs(-(a["Core"] + a["Defensive B"]) - 0.038) < 1e-12
    assert abs(a["Defensive A"] - a["Defensive B"] - 0.048) < 1e-12
    for label in ("Zero floor", "Floor 0.005"):
        assert abs(summary.at[label, "Linear contribution"] - floors[label]) < 1e-6
        assert abs(allocation.at["Defensive A", label] - (floors[label] + 0.038) / 0.048) < 2e-6
    # Scaling the covariance, as a wrong annualisation would, scales risk by its square root
    # and leaves the allocation unchanged.
    for scale in (0.25, 4.0):
        scaled = opt.cvx_maximize_portfolio_sharpe(
            covar=scale * covar.to_numpy(), means=means.to_numpy(),
            constraints=specifications["Floor 0.005"])
        assert scaled.accepted and scaled.compliant
        np.testing.assert_allclose(scaled.weights, allocation["Floor 0.005"], atol=3e-6)
        scaled_model = opt.build_risk_model({risk_date: scale * covar})
        scaled_risk = scaled_model.compute_tre_at_date(
            benchmark_weights=zero_benchmark,
            portfolio_weights=pd.Series(scaled.weights, index=tickers), date=risk_date)
        assert abs(scaled_risk - np.sqrt(scale) * summary.at["Floor 0.005", "Volatility"]) < 1e-6

    # The figure's sweep: 90 floors from -0.08 to 0.009. Up to the no-floor contribution
    # -0.0603 the allocation is unchanged; above it every floor binds at equality.
    sweep = floor_levels()
    assert len(sweep) == 90 and sweep[0] == -0.08 and abs(sweep[-1] - 0.009) < 1e-15
    np.testing.assert_allclose(np.diff(sweep), 0.001, rtol=0, atol=1e-12)
    path, path_outcomes = floor_path(base, means, covar, a, sweep)
    assert all(outcome.accepted and outcome.compliant for outcome in path_outcomes)
    contribution = summary.at["No floor", "Linear contribution"]
    assert round(contribution, 4) == -0.0603 and round(contribution, 3) == -0.060
    loose = sweep <= contribution
    np.testing.assert_allclose(path[loose], np.tile(allocation["No floor"], (loose.sum(), 1)),
                               rtol=0, atol=2e-6)
    np.testing.assert_allclose(path[~loose] @ a, sweep[~loose], rtol=0, atol=1e-6)
    # Carry C leaves the sleeve first and then Carry D, each for good; Defensive B is still held
    # at 0.009. At the reachable maximum 0.01, Defensive A has the unique largest overlay
    # coefficient, so the whole sleeve in Defensive A is the only allocation that meets it.
    exits = {}
    for asset in ("Carry C", "Carry D"):
        held = path[asset].to_numpy() > 1e-6
        exit_index = int(np.argmin(held))
        assert held[:exit_index].all() and not held[exit_index:].any()
        exits[asset] = sweep[exit_index]
    assert exits["Carry C"] < exits["Carry D"] < sweep[-1]
    assert 0.01 < path["Defensive B"].iloc[-1] and path["Defensive A"].iloc[-1] > 0.95
    assert (a.drop(["Core", "Defensive A"]) < a["Defensive A"]).all()
    assert abs(a["Core"] + overlay_budget * a["Defensive A"] - 0.01) < 1e-15
    # The model Sharpe falls with every tighter binding floor. Volatility dips, then rises from
    # 12.3% to 16.0%. The expected excess return falls to 10.7% while carry leaves, then
    # recovers to 11.2% as Defensive A replaces Defensive B.
    path_metrics = path_risk(path, means, covar)
    assert (np.diff(path_metrics["model_sharpe"]) < 1e-9).all()
    assert (np.diff(path_metrics["model_sharpe"][~loose]) < 0.0).all()
    volatility = path_metrics["volatility"]
    assert contribution < volatility.idxmin() < sweep[-1]
    assert (round(100 * volatility.iloc[0], 1), round(100 * volatility.iloc[-1], 1)) == (12.3, 16.0)
    expected = path_metrics["expected_excess"]
    trough = expected.idxmin()
    assert exits["Carry D"] - 0.0015 < trough <= exits["Carry D"]
    assert (np.diff(expected[(sweep > contribution) & (sweep <= trough)]) < 0.0).all()
    assert (np.diff(expected[sweep >= trough]) > 0.0).all()
    replaced = path[sweep >= exits["Carry D"]]
    assert (np.diff(replaced["Defensive A"]) > 0.0).all()
    assert (np.diff(replaced["Defensive B"]) < 0.0).all()
    assert (round(100 * expected.min(), 1), round(100 * expected.iloc[-1], 1)) == (10.7, 11.2)

    coverage = 0.40
    bear_coverage = opt.LinearConstraints(
        loadings=a.to_frame("bear_coverage"),
        lower=pd.Series({"bear_coverage": (1.0 - coverage) * a["Core"]}),
    )
    covered = opt.cvx_maximize_portfolio_sharpe(
        covar=covar.to_numpy(), means=means.to_numpy(),
        constraints=replace(base, linear_constraints=bear_coverage),
        context="overlay article: 40% coverage",
    )
    assert covered.accepted and covered.compliant
    covered_weights = pd.Series(covered.weights, index=tickers)
    realised_coverage = 1.0 - float(a @ covered_weights) / a["Core"]
    maximum_coverage = -overlay_budget * a.drop("Core").max() / a["Core"]
    print(covered_weights.round(6))
    print(round(realised_coverage, 6), round(maximum_coverage, 6))
    # 0.4 1.125

    # A 40% coverage is the floor 0.6 * -0.08 = -0.048, a level of the figure's sweep above the
    # no-floor contribution: it binds, covers exactly 40%, and equals the sweep's allocation,
    # which the homogeneous encoding solved.
    assert abs((1.0 - coverage) * a["Core"] + 0.048) < 1e-15
    level = int(np.argmin(np.abs(sweep + 0.048)))
    assert abs(sweep[level] + 0.048) < 1e-12 and sweep[level] > contribution
    np.testing.assert_allclose(covered_weights, path.iloc[level], rtol=0, atol=2e-6)
    assert abs(realised_coverage - 0.40) < 1e-6 and abs(maximum_coverage - 1.125) < 1e-12
    # The page's table. The no-floor allocation covers 24.6%, the zero floor 100% and the floor
    # 0.005 106.25%; the reachable maximum 0.01 is 1 - 0.01 / -0.08 = 112.5%.
    np.testing.assert_allclose(covered_weights, [1.0, 0.350787, 0.260699, 0.056942, 0.331573],
                               rtol=0, atol=5.1e-7)
    covers = {label: 1.0 - float(a @ allocation[label]) / a["Core"] for label in floors}
    assert round(100 * covers["No floor"], 1) == 24.6
    assert abs(covers["Zero floor"] - 1.0) < 1e-6 and abs(covers["Floor 0.005"] - 1.0625) < 1e-6
    reachable = a["Core"] + overlay_budget * a["Defensive A"]
    assert abs(1.0 - reachable / a["Core"] - maximum_coverage) < 1e-12
    # All four overlays stay; weight moves from carry to defensive. Volatility 11.98% is below
    # the no-floor 12.34%, and the model excess Sharpe ratio falls from 1.005 to 0.997.
    no_floor_weights = allocation["No floor"]
    carry, defensive = ["Carry C", "Carry D"], ["Defensive A", "Defensive B"]
    assert (covered_weights.drop("Core") > 1e-3).all()
    assert (covered_weights[carry] < no_floor_weights[carry]).all()
    assert (covered_weights[defensive] > no_floor_weights[defensive]).all()
    covered_risk = float(np.sqrt(covered_weights @ covar @ covered_weights))
    covered_sharpe = float(means @ covered_weights) / covered_risk
    assert round(100 * covered_risk, 2) == 11.98
    assert round(100 * summary.at["No floor", "Volatility"], 2) == 12.34
    assert round(covered_sharpe, 3) == 0.997
    assert round(summary.at["No floor", "Model excess Sharpe"], 3) == 1.005

    selected = outcomes["Floor 0.005"]
    selected_weights = allocation["Floor 0.005"]
    assert float(a @ selected_weights) >= 0.005 - 1e-6
    assert abs(selected_weights["Core"] - 1.0) < 1e-6
    assert abs(selected_weights.drop("Core").sum() - overlay_budget) < 1e-6
    residuals = selected.residuals_frame()
    floor_row = residuals[residuals["constraint_type"] == "linear"].iloc[0]
    assert floor_row["name"] == "floor" and floor_row["lower"] == 0.005
    assert abs(floor_row["actual"] - float(a @ selected_weights)) < 1e-12
    hard_breaches = [r for r in selected.constraint_residuals if r.hard and not r.passed]
    assert not hard_breaches

    # Each floored case stores one hard, passing linear residual named floor in the
    # characteristic's units, and no target_return residual.
    for label in ("Zero floor", "Floor 0.005"):
        outcome = outcomes[label]
        rows = [r for r in outcome.constraint_residuals if r.constraint_type == "linear"]
        assert len(rows) == 1 and rows[0].name == "floor"
        assert abs(rows[0].actual - float(coefficients @ outcome.weights)) < 1e-12
        assert rows[0].lower == floors[label] and rows[0].hard and rows[0].passed
        assert all(r.constraint_type != "target_return" for r in outcome.constraint_residuals)
    # With the homogeneous cross-check the target_return residual measures the shifted
    # coefficients against zero, which equals the original margin on the exposure equality.
    shifted = replace(base, asset_returns=a - 0.005 / total_exposure, target_return=0.0)
    shifted_outcome = opt.cvx_maximize_portfolio_sharpe(covar.to_numpy(), means.to_numpy(),
                                                        shifted)
    shifted_rows = [r for r in shifted_outcome.constraint_residuals
                    if r.constraint_type == "target_return"]
    assert len(shifted_rows) == 1 and shifted_rows[0].lower == 0.0
    shifted_actual = shifted_rows[0].actual
    assert abs(shifted_actual - float(shifted.asset_returns @ shifted_outcome.weights)) < 1e-12
    assert abs(shifted_actual - (float(coefficients @ shifted_outcome.weights) - 0.005)) < 1e-8

    config = opt.OptimiserConfig(apply_total_to_good_ratio=False)
    labelled_weights, labelled_outcome = opt.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=means, constraints=specifications["Zero floor"],
        optimiser_config=config, context="overlay article: labelled call",
    )
    assert labelled_outcome.accepted and labelled_outcome.compliant
    legacy_weights = solve_overlay_tail_floor(
        means=means, covar=covar, bear_contributions=a, floor_b0=0.0,
    )
    assert np.allclose(labelled_weights, allocation["Zero floor"], atol=1e-6)
    assert np.allclose(legacy_weights, allocation["Zero floor"], atol=1e-6)

    # The wrapper's default configuration rescales bounds on a reduced universe; the example
    # turns that off. The helper returns a labelled Series and also reproduces the no-floor case.
    signature = inspect.signature(opt.wrapper_maximize_portfolio_sharpe)
    assert signature.parameters["optimiser_config"].default.apply_total_to_good_ratio
    assert labelled_weights.index.equals(tickers)
    assert isinstance(legacy_weights, pd.Series) and legacy_weights.index.equals(tickers)
    helper_no_floor = solve_overlay_tail_floor(means, covar, a, floor_b0=None)
    np.testing.assert_allclose(helper_no_floor, allocation["No floor"], atol=1e-6)
    # A missing input shrinks the effective universe: the wrapper solves without Carry C and
    # reports a zero weight for it.
    gap_means = means.copy()
    gap_means["Carry C"] = np.nan
    gap_weights, gap_outcome = opt.wrapper_maximize_portfolio_sharpe(
        pd_covar=covar, means=gap_means, constraints=base, optimiser_config=config)
    assert gap_outcome.accepted and gap_outcome.weights.shape == (4,)
    assert gap_weights.index.equals(tickers) and gap_weights["Carry C"] == 0.0

    maximum_linear = float(a["Core"] + overlay_budget * a.drop("Core").max())
    impossible_floor = 0.02
    assert impossible_floor > maximum_linear
    impossible = replace(
        base, linear_constraints=opt.LinearConstraints(
            loadings=a.to_frame("floor"), lower=pd.Series({"floor": impossible_floor}),
        ),
        weights_0=allocation["No floor"],
    )
    rejected = opt.cvx_maximize_portfolio_sharpe(
        covar=covar.to_numpy(), means=means.to_numpy(), constraints=impossible,
        context="overlay article: deliberate infeasibility",
    )
    print(rejected.status, rejected.accepted, rejected.fallback_source, rejected.compliant)
    # infeasible False weights_0 False

    # The bound 0.01 is the best of the four corners of the sleeve simplex. The rejected solve
    # returns the prior no-floor allocation unprojected, which violates the floor.
    assert abs(maximum_linear - 0.01) < 1e-15
    corners = np.column_stack([np.ones(4), np.eye(4)])
    assert abs(np.max(corners @ coefficients) - maximum_linear) < 1e-15
    assert (rejected.status, rejected.accepted, rejected.fallback_source, rejected.compliant) \
        == ("infeasible", False, "weights_0", False)
    np.testing.assert_array_equal(rejected.weights, allocation["No floor"])
    assert coefficients @ rejected.weights < impossible_floor - 1e-6
    # The constructor accepted the row: over the boxes alone, where each overlay may reach its own
    # cap of 1.0, the row reaches -0.08 + 0.09 + 0.042 = 0.052; the sleeve budget is what binds.
    box_maximum = float(a["Core"] + a.drop("Core").clip(lower=0.0).sum())
    assert abs(box_maximum - 0.052) < 1e-12 and impossible_floor < box_maximum
    assert impossible.linear_constraints.lower["floor"] == impossible_floor
    # Fallback order: a finite prior, then a finite benchmark, then zeros.
    benchmark = pd.Series([1.0, 0.25, 0.25, 0.25, 0.25], index=tickers)
    for spec, source, weights in (
            (replace(impossible, weights_0=pd.Series(np.nan, index=tickers),
                     benchmark_weights=benchmark), "benchmark_weights", benchmark),
            (replace(impossible, weights_0=None), "zeros", np.zeros(5))):
        fallback = opt.cvx_maximize_portfolio_sharpe(
            covar=covar.to_numpy(), means=means.to_numpy(), constraints=spec)
        assert fallback.fallback_source == source and not fallback.accepted
        np.testing.assert_array_equal(fallback.weights, weights)

    direct_floor = replace(base, asset_returns=a, target_return=0.005)
    shifted_floor = replace(base, asset_returns=a - 0.005 / total_exposure, target_return=0.0)
    banded = replace(specifications["Zero floor"], min_exposure=1.8)
    cross_checks = {}
    for label, spec in {"Direct floor": direct_floor, "Shifted floor": shifted_floor,
                        "Exposure band": banded}.items():
        candidate = opt.cvx_maximize_portfolio_sharpe(
            covar=covar.to_numpy(), means=means.to_numpy(), constraints=spec,
            context=f"overlay article: floor check {label}",
        )
        cross_checks[label] = candidate
        print(label, candidate.solver, candidate.status, candidate.accepted)
        assert candidate.accepted and candidate.compliant
    # Direct floor CLARABEL optimal True
    # Shifted floor CLARABEL optimal True
    # Exposure band SLSQP optimal True
    for label in ("Direct floor", "Shifted floor"):
        np.testing.assert_allclose(cross_checks[label].weights, allocation["Floor 0.005"],
                                   atol=1e-8)

    # The printed solver and status of each check.
    assert [(c.solver, c.status) for c in cross_checks.values()] == [
        ("CLARABEL", "optimal"), ("CLARABEL", "optimal"), ("SLSQP", "optimal")]
    # The band's SciPy rows include the named zero floor, which the no-floor allocation misses by
    # more than 0.05; SLSQP returns an allocation inside the band that meets it.
    assert banded.linear_constraints.lower["floor"] == 0.0 and banded.min_exposure == 1.8
    scipy_rows, bounds = banded.set_scipy_constraints(covar.to_numpy())
    point = allocation["No floor"].to_numpy()
    assert any(np.min(row["fun"](point)) < -0.05 for row in scipy_rows)
    band = cross_checks["Exposure band"]
    assert coefficients @ band.weights >= -1e-7
    assert 1.8 - 1e-6 <= band.weights.sum() <= 2.0 + 1e-6
    assert "floor" in band.residuals_frame()["name"].tolist()
    print("overlay_tail_floor: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: the overlay sleeve, and its risk and return, as the floor tightens.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    means, covar, a = solver_example.create_synthetic_inputs()
    reference_means, reference_covar, reference_a = factor_model_inputs()
    floors = floor_levels()
    base = fixed_core_constraints(means.index, OVERLAY_BUDGET)
    weights, outcomes = floor_path(base, means, covar, a, floors)
    metrics = path_risk(weights, means, covar)
    no_floor = opt.cvx_maximize_portfolio_sharpe(covar=covar.to_numpy(), means=means.to_numpy(),
                                                 constraints=base).weights
    contribution = float(a.to_numpy() @ no_floor)
    overlays = ASSETS[1:]
    table = pd.concat([weights[overlays], metrics], axis=1).rename_axis('floor')

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = dict(zip(overlays, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    bottom = np.zeros(len(floors))
    for asset in overlays:
        top = bottom + weights[asset].to_numpy()
        left.fill_between(floors, bottom, top, color=colours[asset], linewidth=0)
        left.text(floors[2], (bottom[2] + top[2]) / 2, asset, color=ink, fontsize=9,
                  va='center')
        bottom = top
    left.set_title('Overlay sleeve by floor (sums to 1)', loc='left', color=ink)
    left.set_ylabel('Overlay weight')
    left.set_ylim(0.0, 1.0)
    right.plot(floors, 100 * metrics['volatility'], color=ink, linewidth=2, label='volatility')
    right.plot(floors, 100 * metrics['expected_excess'], color=muted, linewidth=2,
               linestyle='--', label='expected excess return')
    right.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.36, 1.0), fontsize=10,
                 labelcolor=ink)
    right.set_title('Portfolio model risk and return', loc='left', color=ink)
    right.set_ylabel('Annual, %')
    for axis, top in ((left, 0.98), (right, 16.2)):
        axis.axvline(contribution, color=muted, linestyle=':', linewidth=1.2)
        axis.text(contribution + 0.001, top, 'floor\nbinds', color=ink, fontsize=9, va='top')
        axis.set_xlim(floors[0], floors[-1])
        axis.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
        axis.set_xlabel(r'Floor $b_0$ on $a^\top w$')
        axis.set_facecolor(surface)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    sleeve = weights[overlays].sum(axis=1)
    checks = {
        'inputs_match_constants': bool(
            np.allclose(means, reference_means, rtol=0, atol=1e-15)
            and np.allclose(covar, reference_covar, rtol=0, atol=1e-15)
            and np.allclose(a, reference_a, rtol=0, atol=1e-15)),
        'every_level_accepted_and_compliant': all(o.accepted and o.compliant for o in outcomes),
        'floor_met_at_every_level': bool((weights.to_numpy() @ a.to_numpy()
                                          >= floors - 1e-6).all()),
        'core_one_sleeve_one_total_two': bool(
            np.allclose(weights['Core'], 1.0, atol=1e-6)
            and np.allclose(sleeve, OVERLAY_BUDGET, atol=1e-6)
            and np.allclose(weights.sum(axis=1), 1.0 + OVERLAY_BUDGET, atol=1e-6)),
        'weights_long_only': bool((weights.to_numpy() >= -1e-6).all()),
        'looser_floors_keep_the_no_floor_allocation': bool(np.allclose(
            weights[floors <= contribution], no_floor, atol=2e-6)),
        'tighter_floors_bind_at_equality': bool(np.allclose(
            weights[floors > contribution].to_numpy() @ a.to_numpy(),
            floors[floors > contribution], atol=1e-6)),
        'model_sharpe_never_rises': bool((np.diff(metrics['model_sharpe']) < 1e-9).all()),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
