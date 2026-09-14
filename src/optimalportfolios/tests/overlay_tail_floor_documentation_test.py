"""Verify fixed-core overlay examples, homogeneous floors and backend limitations."""

from dataclasses import replace
from pathlib import Path
import re
import runpy

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def article(root: Path) -> str:
    """Read the authoritative article only in a source checkout."""
    return (root / "docs/overlay_tail_floor.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def examples(article: str) -> dict:
    """Execute every published Python block in order with network access prohibited."""
    blocks = re.findall(r"^```python([^\n]*)\n(.*?)^```", article, re.M | re.S)
    assert len(blocks) == 7 and all(not options.strip() for options, _ in blocks)
    state = {"__name__": "__overlay_article__"}

    def reject_network(*args, **kwargs):
        """Reject an example that acquires a network dependency."""
        raise AssertionError("The overlay article must execute offline")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("socket.create_connection", reject_network)
        patch.setattr("socket.socket.connect", reject_network)
        for index, (_, code) in enumerate(blocks):
            exec(compile(code, f"overlay_tail_floor.md (block {index})", "exec"), state)
    return state


def test_article_structure_and_legacy_headings(article, root):
    """Preserve the existing title and three section anchors within the methodology form."""
    checker = runpy.run_path(str(root / "tools/check_docs.py"))
    assert not checker["check_document"](article, methodology=True)
    assert not checker["check_local_links"](article, root / "docs/overlay_tail_floor.md", root)
    headings = set(re.findall(r"^#{1,3} (.+)$", article, re.M))
    assert {
        "Overlay optimisation with a fixed core and linear side constraints",
        "The problem", "The encoding", "Verification",
    } <= headings
    assert "../examples/solvers/overlay_tail_floor.py" in article


def test_original_input_units_and_core_variance_floor(examples):
    """Retain the original annual inputs, coefficient signs and idiosyncratic variance floor."""
    np.testing.assert_allclose(examples["means"], [.06, .0525, .036, .09, .08], atol=1e-14)
    np.testing.assert_allclose(examples["a"], [-.08, .09, .042, -.045, -.024], atol=1e-14)
    covar = examples["covar"].to_numpy()
    np.testing.assert_allclose(np.diag(covar), [.010001, .0225, .0144, .01, .0064])
    np.testing.assert_allclose(covar[0, 1:], [-.003, -.002, .006, .004])
    assert np.linalg.eigvalsh(covar).min() > 0.0


@pytest.mark.parametrize("label", ["No floor", "Zero floor", "Floor 0.005"])
def test_each_allocation_preserves_core_sleeve_and_outcome(examples, label):
    """Check the original-capital mandate instead of merely normalising an output vector."""
    w = examples["allocation"][label]
    outcome = examples["outcomes"][label]
    assert outcome.accepted and outcome.compliant and outcome.solver == "CLARABEL"
    assert outcome.fallback_source is None
    assert w["Core"] == pytest.approx(1.0, abs=1e-6)
    assert w.drop("Core").sum() == pytest.approx(1.0, abs=1e-6)
    assert w.sum() == pytest.approx(2.0, abs=1e-6)
    assert (w >= -1e-6).all() and (w <= 1.0 + 1e-6).all()


def test_unconstrained_floor_case_has_independent_equality_solution(examples):
    """Solve its two equality conditions by linear algebra, independently of CVXPY."""
    covar = examples["covar"].to_numpy()
    means = examples["means"].to_numpy()
    # Independent optimality reference; production risk reporting uses qis.RiskModel.
    equations = np.vstack([means, [-1., 1., 1., 1., 1.]])
    inverse_rows = np.linalg.solve(covar, equations.T)
    rhs = np.array([2., 0.])
    y = inverse_rows @ np.linalg.solve(equations @ inverse_rows, rhs)
    reference = y / y[0]
    assert (reference[1:] > 0).all() and (reference[1:] < 1).all()
    np.testing.assert_allclose(examples["allocation"]["No floor"], reference, atol=2e-6)


@pytest.mark.parametrize("label,floor", [("Zero floor", 0.0), ("Floor 0.005", 0.005)])
def test_binding_floor_has_independent_global_optimality_certificate(examples, label, floor):
    """Verify the active face and supporting-gradient conditions for the positive ratio."""
    a = examples["a"].to_numpy()
    fraction = (floor - a[0] - a[2]) / (a[1] - a[2])
    reference = np.array([1., fraction, 1. - fraction, 0., 0.])
    np.testing.assert_allclose(examples["allocation"][label], reference, atol=2e-6)
    assert a @ reference == pytest.approx(floor, abs=1e-14)
    # Independent reference for the objective, not a production volatility implementation.
    covar = examples["covar"].to_numpy()
    means = examples["means"].to_numpy()
    risk = np.sqrt(reference @ covar @ reference)
    ratio = (means @ reference) / risk
    gradient = means - ratio * (covar @ reference) / risk
    multiplier = (gradient[2] - gradient[1]) / (a[1] - a[2])
    supporting = gradient[1:] + multiplier * a[1:]
    assert ratio > 0.0 and multiplier > 0.0
    assert supporting[0] == pytest.approx(supporting[1], abs=1e-12)
    assert np.all(supporting[2:] < supporting[0])
    # mu'w - ratio*||Lw|| is concave, zero here, with this supporting gradient.
    # The nonnegative floor multiplier and inactive inequalities certify its global maximum.


def test_displayed_input_allocation_and_metric_tables_match_execution(article, examples):
    """Tie every displayed numeric cell to the executed annual-input calculations."""
    for asset in examples["tickers"]:
        rows = re.findall(rf"^\| {re.escape(asset)} \| (.+) \|$", article, re.M)
        assert len(rows) == 2
        input_cells, weight_cells = (
            [float(cell.strip()) for cell in row.split("|")] for row in rows
        )
        np.testing.assert_allclose(input_cells, examples["inputs"].loc[asset], atol=5.1e-5)
        np.testing.assert_allclose(
            weight_cells, examples["allocation"].loc[asset], atol=5.1e-7, rtol=0,
        )
    for label in examples["floors"]:
        row = re.findall(rf"^\| {re.escape(label)} \| (.+) \|$", article, re.M)
        assert len(row) == 1
        cells = [float(cell.strip()) for cell in row[0].split("|")]
        np.testing.assert_allclose(cells, examples["summary"].loc[label], atol=5.1e-7, rtol=0)


def test_qis_risk_and_model_excess_ratio_match_independent_reference(examples):
    """Check the QIS-based risk table against a separate quadratic-form reference."""
    covar = examples["covar"].to_numpy()
    for label in examples["floors"]:
        w = examples["allocation"][label].to_numpy()
        # Independent verification only; the article delegates risk calculation to QIS.
        risk = np.sqrt(w @ covar @ w)
        expected = examples["means"].to_numpy() @ w
        metrics = examples["summary"].loc[label]
        assert metrics["Volatility"] == pytest.approx(risk, abs=1e-12)
        assert metrics["Expected excess"] == pytest.approx(expected, abs=1e-12)
        assert metrics["Model excess Sharpe"] == pytest.approx(expected / risk, abs=1e-12)
    ratios = examples["summary"]["Model excess Sharpe"]
    assert ratios["No floor"] > ratios["Zero floor"] > ratios["Floor 0.005"]


@pytest.mark.parametrize("budget,floor", [(0.5, -.02), (1., 0.), (1., .005), (2., .02)])
def test_floor_conversion_uses_total_exposure_and_every_coefficient(examples, budget, floor):
    """Prove the homogeneous identity on full and partial exposures without solving."""
    a = examples["a"].to_numpy()
    exposure = 1. + budget
    w = np.r_[1., budget * np.array([.1, .2, .3, .4])]
    encoded = a - floor / exposure
    assert encoded @ w == pytest.approx(a @ w - floor, abs=1e-14)
    for scale in [.25, 3., 20.]:
        assert encoded @ (scale * w) == pytest.approx(scale * (a @ w - floor), abs=1e-13)
    # The documented identity needs the equality, not just an exposure maximum.
    off_budget = w.copy()
    off_budget[-1] += .2
    assert encoded @ off_budget - (a @ off_budget - floor) == pytest.approx(
        floor * (1. - off_budget.sum() / exposure), abs=1e-14,
    )


def test_executed_positive_floor_coefficients_preserve_original_margin(examples):
    """Catch an incorrect denominator or omission of the core from the actual example."""
    spec = examples["specifications"]["Floor 0.005"]
    np.testing.assert_allclose(spec.asset_returns, examples["a"] - .0025, atol=1e-14)
    assert spec.target_return == 0.0
    assert examples["original_margin"] == pytest.approx(examples["encoded_margin"], abs=1e-8)
    assert examples["original_margin"] >= -1e-6 and not examples["hard_breaches"]


@pytest.mark.parametrize("label", ["Zero floor", "Floor 0.005"])
def test_return_residual_records_the_encoded_not_nominal_floor(examples, label):
    """Match the residual row to the shifted coefficient vector and exposure equality."""
    outcome = examples["outcomes"][label]
    rows = [r for r in outcome.constraint_residuals if r.constraint_type == "target_return"]
    assert len(rows) == 1
    row = rows[0]
    expected = float(examples["specifications"][label].asset_returns @ outcome.weights)
    assert row.actual == pytest.approx(expected, abs=1e-12)
    assert row.lower == 0.0 and row.hard and row.passed


@pytest.mark.parametrize("scale", [0.25, 4.])
def test_covariance_scaling_changes_risk_but_not_the_fixed_input_allocation(examples, scale):
    """Detect covariance/annualisation errors with unchanged expected returns and floor."""
    opt = examples["opt"]
    spec = examples["specifications"]["Floor 0.005"]
    out = opt.cvx_maximize_portfolio_sharpe(
        covar=scale * examples["covar"].to_numpy(),
        means=examples["means"].to_numpy(), constraints=spec,
    )
    assert out.accepted and out.compliant
    np.testing.assert_allclose(out.weights, examples["allocation"]["Floor 0.005"], atol=3e-6)
    model = opt.build_risk_model({examples["risk_date"]: scale * examples["covar"]})
    risk = model.compute_tre_at_date(
        benchmark_weights=examples["zero_benchmark"],
        portfolio_weights=pd.Series(out.weights, index=examples["tickers"]),
        date=examples["risk_date"],
    )
    assert risk == pytest.approx(
        np.sqrt(scale) * examples["summary"].loc["Floor 0.005", "Volatility"], abs=1e-6,
    )


def test_original_helper_and_labelled_wrapper_agree(examples):
    """Preserve both original helper scenarios while keeping outcome-returning alternatives."""
    assert examples["labelled_outcome"].accepted
    assert examples["labelled_weights"].index.equals(examples["tickers"])
    no_floor = examples["solve_overlay_tail_floor"](
        examples["means"], examples["covar"], examples["a"], floor_b0=None,
    )
    np.testing.assert_allclose(no_floor, examples["allocation"]["No floor"], atol=1e-6)
    np.testing.assert_allclose(
        examples["legacy_weights"], examples["allocation"]["Zero floor"], atol=1e-6,
    )


def test_linear_reachability_and_noncompliant_prior_fallback(examples):
    """An unreachable floor rejects even though a finite prior allocation is available."""
    assert examples["maximum_linear"] == pytest.approx(.01, abs=1e-14)
    a = examples["a"].to_numpy()
    corners = np.column_stack([np.ones(4), np.eye(4)])
    assert np.max(corners @ a) == pytest.approx(examples["maximum_linear"])
    outcome = examples["rejected"]
    assert outcome.status == "infeasible" and not outcome.accepted and not outcome.compliant
    assert outcome.fallback_source == "weights_0"
    np.testing.assert_allclose(outcome.weights, examples["allocation"]["No floor"])
    assert a @ outcome.weights < examples["impossible_floor"] - 1e-6


def test_unscaled_nonzero_row_is_rejected_after_solver_reports_optimal(examples):
    """Document the current transformed-RHS limitation without changing the backend."""
    out = examples["unsupported_outcomes"]["Unscaled floor"]
    assert out.solver == "CLARABEL" and out.status == "optimal"
    assert not out.accepted and not out.compliant and out.fallback_source == "zeros"
    assert "target_return" in out.reason
    np.testing.assert_array_equal(out.weights, 0.0)


def test_cvx_homogeneous_rows_scale_core_budget_and_floor(examples):
    """Evaluate compiled CVXPY rows at known feasible scaled points without a solver."""
    spec = examples["specifications"]["Floor 0.005"]
    w = np.array([1., 1., 0., 0., 0.])
    y, k = cvx.Variable(5), cvx.Variable(nonneg=True)
    rows = spec.set_cvx_all_constraints(
        w=y, covar=examples["covar"].to_numpy(), exposure_scaler=k,
    )
    for scale in [.2, 2., 20.]:
        y.value, k.value = scale * w, scale
        assert all(np.max(row.violation()) < 1e-9 for row in rows)
    y.value = 2. * np.array([1., 0., 0., 0., 1.])
    k.value = 2.
    assert any(np.max(row.violation()) > .1 for row in rows)


def test_scipy_omits_return_row_but_validation_rejects_the_breach(examples):
    """Keep the documented distinction between optimizer support and residual rejection."""
    spec = examples["banded"]
    rows, bounds = spec.set_scipy_constraints(examples["covar"].to_numpy())
    w = examples["allocation"]["No floor"].to_numpy()
    assert all(float(row["fun"](w)) >= -1e-7 for row in rows)
    assert all(low - 1e-7 <= value <= high + 1e-7
               for (low, high), value in zip(bounds, w))
    assert float(examples["a"] @ w) < -.05
    out = examples["unsupported_outcomes"]["Exposure band"]
    assert out.solver == "SLSQP" and out.status == "optimal"
    assert not out.accepted and not out.compliant and "target_return" in out.reason


def test_multiple_sleeve_budget_rows_keep_the_fixed_core(examples):
    """Compile and solve disjoint sleeve equalities with the same homogeneous floor."""
    opt = examples["opt"]
    groups = opt.GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame(
            {"Defensive": [0., 1., 1., 0., 0.], "Carry": [0., 0., 0., 1., 1.]},
            index=examples["tickers"],
        ),
        group_min_allocation=pd.Series({"Defensive": .95, "Carry": .05}),
        group_max_allocation=pd.Series({"Defensive": .95, "Carry": .05}),
    )
    spec = replace(examples["specifications"]["Zero floor"],
                   group_lower_upper_constraints=groups)
    out = opt.cvx_maximize_portfolio_sharpe(
        examples["covar"].to_numpy(), examples["means"].to_numpy(), spec,
    )
    assert out.accepted and out.compliant
    np.testing.assert_allclose(groups.group_loadings.T @ out.weights, [.95, .05], atol=1e-6)
    assert out.weights[0] == pytest.approx(1.0, abs=1e-6)
    assert float(examples["a"] @ out.weights) >= -1e-6

