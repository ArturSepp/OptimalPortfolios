"""Pathological tests for ``optimalportfolios.optimization.solver_diagnostics``.

These pin the two safeguards added in response to the GROWM
(tre_utility_weight=100, turnover_utility_weight=0.2) blow-up, where a
near-collinear private-asset block (two return proxies at corr 1.00) made the
covariance rank-deficient (cond ~5e12), CLARABEL returned a non-optimal iterate
that summed to ~1.5e6, and the only post-solve check (``w.value is None``)
accepted it — corrupting one 2021 quarter and, through it, every second-moment
backtest statistic.

Design note
-----------
We deliberately do NOT try to provoke a real solver failure: modern CLARABEL
stays feasible even at cond ~6e14 on small problems, and ``psd_wrap`` defeats
CVXPY's DCP convexity check, so the failure is data/solver-version specific and
not reliably reproducible in a unit test. Instead we test the *validator*
directly by feeding it the kind of output a failed solve produces (including
the literal 1.5e6 incident vector), which is deterministic and version-robust.
The conditioning tests reproduce the upstream *cause* (the corr-1.00 block) and
assert it is flagged before the solve.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization.constraints import (
    Constraints, GroupLowerUpperConstraints)
from optimalportfolios.optimization.solver_diagnostics import (
    validate_solution,
    validate_scipy_solution,
    validate_rb_solution,
    check_covar_conditioning,
)
from types import SimpleNamespace


# -----------------------------------------------------------------------------
# fixtures
# -----------------------------------------------------------------------------

TICKERS = ["EQ", "BOND", "GOLD", "PE", "HF"]  # PE/HF stand in for the alts block


def _constraints(is_long_only=True, with_weights_0=True, with_benchmark=True,
                 max_w=0.40):
    """A simple, feasible long-only Constraints over 5 assets, fully invested."""
    idx = pd.Index(TICKERS)
    benchmark = pd.Series([0.40, 0.35, 0.15, 0.05, 0.05], index=idx) if with_benchmark else None
    weights_0 = pd.Series([0.38, 0.34, 0.16, 0.06, 0.06], index=idx) if with_weights_0 else None
    return Constraints(
        is_long_only=is_long_only,
        min_weights=pd.Series(0.0, index=idx),
        max_weights=pd.Series(max_w, index=idx),
        max_exposure=1.0,
        min_exposure=1.0,
        benchmark_weights=benchmark,
        weights_0=weights_0,
    )


def _feasible_w():
    """A clean feasible weight vector (sums to 1, within [0, 0.40])."""
    return np.array([0.40, 0.30, 0.15, 0.10, 0.05])


# -----------------------------------------------------------------------------
# validate_solution — acceptance of good solutions
# -----------------------------------------------------------------------------

def test_accepts_clean_feasible_solution():
    """A clean feasible solve is accepted and returned unchanged."""
    c = _constraints()
    w_in = _feasible_w()
    outcome = validate_solution(w_in, "optimal", c, n=5, solver="CLARABEL")
    assert outcome.accepted is True
    np.testing.assert_allclose(outcome.weights, w_in)


# -----------------------------------------------------------------------------
# validate_solution — THE incident: budget blow-up is rejected
# -----------------------------------------------------------------------------

def test_rejects_budget_blowup_1p5e6(caplog):
    """Reproduce the GROWM 2021-04 signature: sum(w) ~ 1.5e6, one weight ~3.3e5.

    The pre-fix code (``if w.value is None``) accepts this finite-but-insane
    vector. The validator must reject it and fall back to weights_0.
    """
    c = _constraints(with_weights_0=True)
    blowup = np.array([3.2672e5, 2.9852e5, 1.5216e5, 4.5e5, 3.0e5])  # sum ≈ 1.52e6
    assert blowup.sum() > 1e6  # sanity: this is the pathology

    with caplog.at_level(logging.ERROR):
        outcome = validate_solution(
            blowup, "optimal_inaccurate", c, n=5,
            solver="CLARABEL", context="GROWM 2021-04-30")

    assert outcome.accepted is False
    # fell back to weights_0
    np.testing.assert_allclose(outcome.weights, c.weights_0.to_numpy())
    assert abs(outcome.weights.sum() - 1.0) < 1e-6
    # the log names the budget violation, the date, and the bad sum
    rec = caplog.text.lower()
    assert "budget" in rec and "rejected" in rec
    assert "growm 2021-04-30" in rec


def test_rejects_budget_blowup_even_when_status_optimal(caplog):
    """A wildly infeasible iterate is rejected on feasibility alone, regardless
    of an 'optimal' status (the status cannot be trusted when the KKT solve
    broke down)."""
    c = _constraints()
    blowup = _feasible_w() * 1.0e5  # sum = 1e5
    with caplog.at_level(logging.ERROR):
        outcome = validate_solution(blowup, "optimal", c, n=5, solver="CLARABEL")
    assert outcome.accepted is False


# -----------------------------------------------------------------------------
# validate_solution — other failure modes
# -----------------------------------------------------------------------------

def test_rejects_none_solution():
    """A ``None`` solution falls back to weights_0."""
    c = _constraints()
    outcome = validate_solution(None, "optimal", c, n=5, solver="CLARABEL")
    assert outcome.accepted is False
    np.testing.assert_allclose(outcome.weights, c.weights_0.to_numpy())


def test_rejects_nonfinite_weights():
    """A non-finite weight is rejected even under an 'optimal' status."""
    c = _constraints()
    bad = _feasible_w().copy()
    bad[3] = np.inf
    outcome = validate_solution(bad, "optimal", c, n=5)
    assert outcome.accepted is False


def test_rejects_box_violation():
    """Sums to 1 but one weight (0.90) exceeds the 0.40 cap."""
    c = _constraints(max_w=0.40)
    bad = np.array([0.90, 0.05, 0.03, 0.01, 0.01])
    outcome = validate_solution(bad, "optimal", c, n=5)
    assert outcome.accepted is False


def test_rejects_negative_weight_long_only():
    """A negative weight is rejected when the mandate is long-only."""
    c = _constraints(is_long_only=True)
    bad = np.array([0.70, 0.40, 0.15, -0.30, 0.05])  # sums to 1 but PE < 0
    outcome = validate_solution(bad, "optimal", c, n=5)
    assert outcome.accepted is False


@pytest.mark.parametrize("status", ["infeasible", "unbounded",
                                     "unbounded_inaccurate", "solver_error", None])
def test_rejects_hard_status_even_if_weights_look_ok(status):
    """A feasible-looking w.value paired with a hard/None status is rejected —
    the solver is telling us it did not actually solve the problem."""
    c = _constraints()
    outcome = validate_solution(_feasible_w(), status, c, n=5)
    assert outcome.accepted is False


# -----------------------------------------------------------------------------
# validate_solution — inaccurate handling
# -----------------------------------------------------------------------------

def test_accepts_inaccurate_but_feasible_with_warning(caplog):
    """An imprecise but feasible solve is accepted with a warning."""
    c = _constraints()
    with caplog.at_level(logging.WARNING):
        outcome = validate_solution(
            _feasible_w(), "optimal_inaccurate", c, n=5, accept_inaccurate=True)
    assert outcome.accepted is True
    assert "imprecise" in caplog.text.lower()


def test_rejects_inaccurate_and_infeasible():
    """An imprecise solve that is also infeasible is rejected."""
    c = _constraints()
    blowup = _feasible_w() * 1e4
    outcome = validate_solution(blowup, "optimal_inaccurate", c, n=5)
    assert outcome.accepted is False


def test_inaccurate_rejected_when_flag_off(caplog):
    """An imprecise solve is rejected when ``accept_inaccurate`` is off."""
    c = _constraints()
    with caplog.at_level(logging.WARNING):
        outcome = validate_solution(
            _feasible_w(), "optimal_inaccurate", c, n=5, accept_inaccurate=False)
    assert outcome.accepted is False


# -----------------------------------------------------------------------------
# validate_solution — fallback chain
# -----------------------------------------------------------------------------

def test_fallback_to_benchmark_when_no_weights_0():
    """Without weights_0 the fallback is the benchmark."""
    c = _constraints(with_weights_0=False, with_benchmark=True)
    outcome = validate_solution(None, "optimal", c, n=5)
    assert outcome.accepted is False
    np.testing.assert_allclose(outcome.weights, c.benchmark_weights.to_numpy())


def test_fallback_to_zeros_when_no_weights_0_or_benchmark():
    """Without weights_0 or a benchmark the fallback is zeros."""
    c = _constraints(with_weights_0=False, with_benchmark=False)
    outcome = validate_solution(None, "optimal", c, n=5)
    assert outcome.accepted is False
    np.testing.assert_allclose(outcome.weights, np.zeros(5))


# -----------------------------------------------------------------------------
# validate_solution — the band budget and the non-long-only floor
# -----------------------------------------------------------------------------

def _band_constraints(min_exposure=0.0, max_exposure=1.0):
    """Constraints stating an exposure *band* rather than full investment.

    A band is how a cash-holding or leverage-capable mandate is written: the budget test then
    has to check two bounds instead of one equality, and both halves of it are separate code.
    """
    idx = pd.Index(TICKERS)
    return Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=idx),
        max_weights=pd.Series(1.0, index=idx),
        min_exposure=min_exposure,
        max_exposure=max_exposure,
        weights_0=pd.Series(0.2, index=idx),
    )


def test_a_weight_vector_outside_the_exposure_band_is_rejected(caplog):
    """a band is two bounds, and breaching either one rejects the solve"""
    c = _band_constraints(min_exposure=0.5, max_exposure=1.0)
    with caplog.at_level(logging.WARNING,
                         logger="optimalportfolios.optimization.solver_diagnostics"):
        over = validate_solution(np.full(5, 0.4), "optimal", c, n=5)     # sums to 2.0
    assert over.accepted is False
    assert any("exposure band violated" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING,
                         logger="optimalportfolios.optimization.solver_diagnostics"):
        under = validate_solution(np.full(5, 0.05), "optimal", c, n=5)   # sums to 0.25
    assert under.accepted is False
    assert any("exposure band violated" in r.getMessage() for r in caplog.records)


def test_a_weight_vector_inside_the_exposure_band_is_accepted():
    """within the band there is no budget residual to report, and the solve stands"""
    c = _band_constraints(min_exposure=0.5, max_exposure=1.0)
    w = np.full(5, 0.15)                                                  # sums to 0.75
    outcome = validate_solution(w, "optimal", c, n=5)
    assert outcome.accepted is True
    np.testing.assert_allclose(outcome.weights, w)


def test_a_long_short_solve_below_its_floor_names_the_asset(caplog):
    """a min-weight breach is reported by index and by how far short it fell

    Under long-only the negative-weight check catches this first; with shorts permitted the
    floor is the only thing standing between the solve and a position the mandate forbids.
    """
    idx = pd.Index(TICKERS)
    c = Constraints(is_long_only=False,
                    min_weights=pd.Series([0.10, 0.0, 0.0, 0.0, 0.0], index=idx),
                    max_weights=pd.Series(1.0, index=idx),
                    min_exposure=1.0, max_exposure=1.0,
                    weights_0=pd.Series(0.2, index=idx))
    with caplog.at_level(logging.WARNING,
                         logger="optimalportfolios.optimization.solver_diagnostics"):
        outcome = validate_solution(np.array([0.0, 0.25, 0.25, 0.25, 0.25]), "optimal",
                                    c, n=5)
    assert outcome.accepted is False
    assert any("min_weight violated" in r.getMessage() for r in caplog.records)
    np.testing.assert_allclose(outcome.weights, c.weights_0.to_numpy())


def test_a_solve_that_breaches_a_group_bound_is_rejected(caplog):
    """the residual audit gates too: in-box and fully invested is not sufficient

    This vector passes every box and budget test — the breach is only visible once the group
    loadings are applied. Without the residual gate it would be accepted, and a mandate limit
    would be quietly exceeded for that rebalancing date.
    """
    idx = pd.Index(TICKERS)
    loadings = pd.DataFrame({'alts': [0.0, 0.0, 0.0, 1.0, 1.0]}, index=idx)
    c = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=idx),
        max_weights=pd.Series(1.0, index=idx),
        min_exposure=1.0, max_exposure=1.0,
        weights_0=pd.Series(0.2, index=idx),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=pd.Series([0.0], index=['alts']),
            group_max_allocation=pd.Series([0.10], index=['alts'])))
    w = np.array([0.20, 0.20, 0.20, 0.20, 0.20])          # 0.40 in 'alts', cap is 0.10
    with caplog.at_level(logging.WARNING,
                         logger="optimalportfolios.optimization.solver_diagnostics"):
        outcome = validate_solution(w, "optimal", c, n=5, covar=np.eye(5) * 0.04)
    assert outcome.accepted is False
    assert any("hard constraint group_weight:alts violated" in r.getMessage()
               for r in caplog.records)
    assert not outcome.compliant


def test_bounds_given_as_plain_arrays_are_honoured():
    """the box may arrive unlabelled, and it is then taken in the tickers' own order

    ``Constraints`` is a plain dataclass with no coercion, so a caller working in numpy can
    hand it arrays. Treating those as absent would drop the box silently.
    """
    idx = pd.Index(TICKERS)
    c = Constraints(is_long_only=True,
                    max_weights=np.array([0.25, 1.0, 1.0, 1.0, 1.0]),
                    min_exposure=1.0, max_exposure=1.0,
                    weights_0=pd.Series(0.2, index=idx))
    breached = validate_solution(np.array([0.6, 0.1, 0.1, 0.1, 0.1]), "optimal", c, n=5)
    assert breached.accepted is False
    respected = validate_solution(np.array([0.2, 0.2, 0.2, 0.2, 0.2]), "optimal", c, n=5)
    assert respected.accepted is True


def test_the_asset_order_can_come_from_a_group_loadings_table():
    """with no labelled box or benchmark, the group loadings state the asset order

    Some constraint sets carry nothing but a group block. The index has to be recovered from
    it, because falling back to 0..n-1 would misalign every Series-valued bound that arrives
    later in the same run.
    """
    idx = pd.Index(TICKERS)
    loadings = pd.DataFrame({'alts': [0.0, 0.0, 0.0, 1.0, 1.0]}, index=idx)
    c = Constraints(
        is_long_only=True, min_exposure=1.0, max_exposure=1.0,
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=pd.Series([0.0], index=['alts']),
            group_max_allocation=pd.Series([1.0], index=['alts'])))
    outcome = validate_solution(np.full(5, 0.2), "optimal", c, n=5,
                                covar=np.eye(5) * 0.04)
    assert outcome.accepted is True
    assert [r.name for r in outcome.constraint_residuals
            if r.constraint_type == 'group_weight'] == ['alts']


# -----------------------------------------------------------------------------
# validate_solution — property: any output yields a feasible vector
# -----------------------------------------------------------------------------

def test_property_output_always_feasible_for_arbitrary_garbage():
    """Whatever the solver returns, the validated weights are finite, and either
    the accepted feasible solution or a feasible fallback. This is the guarantee
    that keeps a bad rebalance out of the NAV."""
    c = _constraints()
    rng = np.random.default_rng(0)
    import warnings as _w
    for _ in range(200):
        scale = 10.0 ** rng.integers(-2, 8)
        w = rng.standard_normal(5) * scale
        status = rng.choice(["optimal", "optimal_inaccurate", "unbounded",
                             "infeasible", "solver_error"])
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            outcome = validate_solution(w, status, c, n=5)
        assert np.all(np.isfinite(outcome.weights))
        if not outcome.accepted:
            # fallback is always feasible: fully invested (or zeros) and in-box
            s = outcome.weights.sum()
            assert abs(s - 1.0) < 1e-6 or np.allclose(outcome.weights, 0.0)
            assert outcome.weights.min() >= -1e-9
            assert (outcome.weights.max() <= 0.40 + 1e-9
                    or np.allclose(outcome.weights, 0.0))


# -----------------------------------------------------------------------------
# check_covar_conditioning — the upstream cause
# -----------------------------------------------------------------------------

def _covar_from_corr(vols, corr):
    """Build a covariance DataFrame over ``TICKERS`` from vols and a correlation matrix."""
    vols = np.asarray(vols, float)
    c = np.outer(vols, vols) * np.asarray(corr, float)
    c = (c + c.T) / 2.0
    return pd.DataFrame(c, index=TICKERS, columns=TICKERS)


def test_flags_collinear_block(caplog):
    """Two assets at corr 1.00 (the Franklin Lexington / PG3 Longreach case)
    → rank-deficient covar flagged, worst pair named, matrix NOT modified."""
    vols = [0.20, 0.05, 0.15, 0.10, 0.10]
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = 0.2
    corr[0, 3] = corr[3, 0] = 0.5
    corr[3, 4] = corr[4, 3] = 1.0    # PE and HF identical → singular
    pd_covar = _covar_from_corr(vols, corr)

    with caplog.at_level(logging.WARNING):
        diag = check_covar_conditioning(pd_covar, cond_warn=1e12,
                                        context="GROWM 2021-04-30")

    assert diag["rank_deficient"] is True
    assert diag["cond"] > 1e12
    assert set(diag["worst_pair"]) == {"PE", "HF"}
    assert "ill-conditioned" in caplog.text.lower()
    # diagnostic must not mutate the input
    assert pd_covar.iloc[3, 4] == pytest.approx(0.10 * 0.10 * 1.0)


def test_flags_indefinite_covar(caplog):
    """A covariance with a negative eigenvalue (as the factor Σ_F note warns)
    is flagged as indefinite — psd_wrap would otherwise hide it."""
    vols = [0.20, 0.05, 0.15, 0.10, 0.10]
    base = _covar_from_corr(vols, np.eye(5)).to_numpy()
    ev, V = np.linalg.eigh(base)
    ev[0] = -1e-4                       # inject a negative eigenvalue
    indef = (V * ev) @ V.T
    pd_covar = pd.DataFrame((indef + indef.T) / 2, index=TICKERS, columns=TICKERS)

    with caplog.at_level(logging.WARNING):
        diag = check_covar_conditioning(pd_covar, context="indef")
    assert diag["min_eig"] < 0
    assert "indefinite" in caplog.text.lower()


def test_clean_covar_no_warning(caplog):
    """A well-conditioned covariance is reported clean and logs nothing."""
    vols = [0.20, 0.05, 0.15, 0.10, 0.12]
    corr = np.eye(5)
    corr[0, 4] = corr[4, 0] = 0.5
    corr[0, 1] = corr[1, 0] = -0.1
    pd_covar = _covar_from_corr(vols, corr)
    with caplog.at_level(logging.WARNING):
        diag = check_covar_conditioning(pd_covar, cond_warn=1e12)
    assert diag["rank_deficient"] is False
    assert np.isfinite(diag["cond"]) and diag["cond"] < 1e6
    assert caplog.text == ""


# -----------------------------------------------------------------------------
# validate_scipy_solution — scipy backends (risk-budgeting SLSQP, CARA, CARA-mixture)
# -----------------------------------------------------------------------------

def _scipy_res(success=True, status=0, message="Optimization terminated successfully"):
    """Stand-in for scipy.optimize.OptimizeResult."""
    return SimpleNamespace(success=success, status=status, message=message)


def test_scipy_accepts_feasible_solution():
    """A clean feasible scipy solve is accepted and returned unchanged."""
    c = _constraints()
    w_out, ok = validate_scipy_solution(_feasible_w(), _scipy_res(), c, n=5)
    assert ok is True
    np.testing.assert_allclose(w_out, _feasible_w())


def test_scipy_rejects_nonconvergence(caplog):
    """A scipy non-convergence falls back and surfaces the solver message."""
    c = _constraints()
    res = _scipy_res(success=False, status=8,
                     message="Positive directional derivative for linesearch")
    with caplog.at_level(logging.WARNING):
        w_out, ok = validate_scipy_solution(_feasible_w(), res, c, n=5)
    assert ok is False
    np.testing.assert_allclose(w_out, c.weights_0.to_numpy())
    assert "non-convergence" in caplog.text.lower()
    assert "linesearch" in caplog.text.lower()      # scipy message surfaced in the log


def test_scipy_rejects_budget_blowup_even_if_success():
    """SLSQP can report success on a point that drifted off the simplex."""
    c = _constraints()
    _, ok = validate_scipy_solution(_feasible_w() * 50.0, _scipy_res(success=True),
                                    c, n=5)
    assert ok is False


def test_scipy_rejects_nonfinite():
    """A non-finite scipy weight is rejected even when scipy reports success."""
    c = _constraints()
    bad = _feasible_w().copy()
    bad[2] = np.nan
    _, ok = validate_scipy_solution(bad, _scipy_res(), c, n=5)
    assert ok is False


# -----------------------------------------------------------------------------
# validate_rb_solution — internal CCD/ADMM risk-budgeting solver (caller passes
# None on solver failure; group inequality rows checked at group_atol)
# -----------------------------------------------------------------------------

def test_rb_accepts_feasible_solution():
    """A clean feasible risk-budgeting solve is accepted and returned unchanged."""
    c = _constraints()
    w_out, ok = validate_rb_solution(_feasible_w(), c, n=5)
    assert ok is True
    np.testing.assert_allclose(w_out, _feasible_w())


def test_rb_rejects_nan_solution(caplog):
    """NaN weights — the validator catches them and falls back."""
    c = _constraints()
    bad = _feasible_w().copy()
    bad[3] = np.nan
    with caplog.at_level(logging.ERROR):
        w_out, ok = validate_rb_solution(bad, c, n=5)
    assert ok is False
    np.testing.assert_allclose(w_out, c.weights_0.to_numpy())


def test_rb_rejects_none_solution():
    """A ``None`` risk-budgeting solution falls back to weights_0."""
    c = _constraints()
    _, ok = validate_rb_solution(None, c, n=5)
    assert ok is False


def test_rb_rejects_nonconvergence_flag():
    """An explicit ``converged=False`` forces a rejection."""
    c = _constraints()
    _, ok = validate_rb_solution(_feasible_w(), c, n=5, converged=False)
    assert ok is False


def test_rb_accepts_group_rows_within_atol():
    """ADMM-level group breach (~1e-5) is within group_atol=1e-4 → accepted."""
    c = _constraints()
    w = _feasible_w()
    c_rows = np.ones((1, 5))          # sum(w) <= 1, satisfied with equality
    c_lhs = np.array([float(np.sum(w)) - 1e-5])   # breach by 1e-5 < 1e-4
    w_out, ok = validate_rb_solution(w, c, n=5, c_rows=c_rows, c_lhs=c_lhs)
    assert ok is True
    np.testing.assert_allclose(w_out, w)


def test_rb_rejects_group_row_violation(caplog):
    """A material group-row breach (> group_atol) is rejected with fallback."""
    c = _constraints()
    w = _feasible_w()
    c_rows = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]])
    c_lhs = np.array([float(w[0] + w[1]) - 1e-3])  # breach by 1e-3 > 1e-4
    with caplog.at_level(logging.ERROR):
        w_out, ok = validate_rb_solution(w, c, n=5, c_rows=c_rows, c_lhs=c_lhs)
    assert ok is False
    np.testing.assert_allclose(w_out, c.weights_0.to_numpy())


# -----------------------------------------------------------------------------
# writability — every validator must return a WRITABLE array (pandas 3.0
# Series.to_numpy() is read-only; downstream wrappers do in-place edits like
# `weights[np.isinf(weights)] = 0.0`, which crash on a read-only array).
# -----------------------------------------------------------------------------

def _assert_writable_and_assignable(w):
    """Assert the validated weights are writable and survive in-place assignment."""
    assert w.flags.writeable, "validator returned a read-only array"
    w[np.isinf(w)] = 0.0          # the exact op that crashed in wrapper_maximise_alpha_over_tre
    w[0] = 0.123                  # general in-place write must succeed


def test_cvxpy_fallback_is_writable():
    """Infeasible solve → fallback to weights_0 must be writable."""
    c = _constraints()            # carries a pd.Series weights_0
    outcome = validate_solution(None, 'infeasible', c, n=5)
    assert outcome.accepted is False
    _assert_writable_and_assignable(outcome.weights)


def test_cvxpy_accepted_solution_is_writable():
    """An accepted CVXPY solution is returned as a writable array."""
    c = _constraints()
    outcome = validate_solution(_feasible_w(), 'optimal', c, n=5)
    assert outcome.accepted is True
    _assert_writable_and_assignable(outcome.weights)


def test_scipy_fallback_is_writable():
    """A scipy fallback is returned as a writable array."""
    c = _constraints()
    w, ok = validate_scipy_solution(None, _scipy_res(success=False), c, n=5)
    assert ok is False
    _assert_writable_and_assignable(w)


def test_rb_fallback_is_writable():
    """A risk-budgeting fallback is returned as a writable array."""
    c = _constraints()
    w, ok = validate_rb_solution(None, c, n=5)
    assert ok is False
    _assert_writable_and_assignable(w)


# -----------------------------------------------------------------------------
# severity + summary (single logger channel: WARNING = benign fallback,
# ERROR = numerical blow-up the solver mislabelled optimal/inaccurate)
# -----------------------------------------------------------------------------

def test_rejection_severity_levels(caplog):
    """Blow-ups log at ERROR while honest no-solution statuses log at WARNING."""
    c = _constraints()
    # optimal status but budget blow-up -> ERROR
    with caplog.at_level(logging.WARNING):
        validate_solution(_feasible_w() * 1e5, "optimal", c, n=5)
    assert any(r.levelno == logging.ERROR and "REJECTED" in r.getMessage()
               for r in caplog.records)
    caplog.clear()
    # no solution -> WARNING
    with caplog.at_level(logging.WARNING):
        validate_solution(None, "infeasible", c, n=5)
    rej = [r for r in caplog.records if "REJECTED" in r.getMessage()]
    assert rej and all(r.levelno == logging.WARNING for r in rej)
    caplog.clear()
    # honest infeasible status with finite weights -> WARNING
    with caplog.at_level(logging.WARNING):
        validate_solution(_feasible_w(), "infeasible", c, n=5)
    assert any(r.levelno == logging.WARNING and "REJECTED" in r.getMessage()
               for r in caplog.records)


def test_rejection_log_distinguishes_attempt_from_final_execution(caplog):
    """A rejected solve says its fallback impact is provisional at this layer."""
    c = _constraints()
    with caplog.at_level(
            logging.WARNING,
            logger="optimalportfolios.optimization.solver_diagnostics",
    ):
        validate_solution(
            None,
            "infeasible_inaccurate",
            c,
            n=5,
            solver="CLARABEL",
            context="growf 2026-07-31",
        )

    message = next(
        record.getMessage() for record in caplog.records
        if "REJECTED solution" in record.getMessage()
    )
    assert "falling back to weights_0" in message
    assert "this solver attempt produced no usable weights" in message
    assert "final portfolio/trade outcome is not decided at this layer" in message
    assert "reported separately as execution_rebalance" in message


def test_rejection_summary_handler():
    """The summary handler tallies one ERROR and one WARNING rejection."""
    from optimalportfolios.optimization.solver_diagnostics import (
        SolverRejectionSummary)
    h = SolverRejectionSummary()
    lg = logging.getLogger("optimalportfolios.optimization.solver_diagnostics")
    lg.addHandler(h)
    try:
        c = _constraints()
        validate_solution(_feasible_w() * 1e5, "optimal", c, n=5)   # ERROR
        validate_solution(None, "infeasible", c, n=5)               # WARNING
        s = h.summary()
        assert "1 ERROR" in s and "1 WARNING" in s
    finally:
        lg.removeHandler(h)


# -----------------------------------------------------------------------------
# infeasibility diagnosis (elastic min-violation re-solve)
# -----------------------------------------------------------------------------

def _group_floor_infeasible_constraints():
    """Constraints whose two group floors jointly overshoot full investment."""
    from optimalportfolios.optimization.constraints import GroupLowerUpperConstraints
    idx = pd.Index(["A", "B", "C", "D", "E"])
    glu = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"G1": [1, 1, 0, 0, 0], "G2": [0, 0, 1, 1, 1]},
                                    index=idx, dtype=float),
        group_min_allocation=pd.Series({"G1": 0.60, "G2": 0.60}),   # sum 1.2 > 1
        group_max_allocation=pd.Series({"G1": 1.0, "G2": 1.0}))
    return Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=idx),
                       max_weights=pd.Series(1.0, index=idx),
                       max_exposure=1.0, min_exposure=1.0,
                       group_lower_upper_constraints=glu)


def test_diagnose_infeasibility_identifies_group_floor_overshoot():
    """The elastic re-solve names both overshooting group floors and their sizes."""
    from optimalportfolios.optimization.solver_diagnostics import diagnose_infeasibility
    out = diagnose_infeasibility(_group_floor_infeasible_constraints(), context="t")
    assert set(out) == {"group_min:G1", "group_min:G2"}
    assert abs(sum(out.values()) - 0.20) < 1e-3   # floors jointly overshoot by 0.20


def test_diagnose_infeasibility_box_cannot_reach_full_investment():
    """The elastic re-solve names the caps that block full investment."""
    from optimalportfolios.optimization.solver_diagnostics import diagnose_infeasibility
    idx = pd.Index(["A", "B", "C", "D", "E"])
    c = Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=idx),
                    max_weights=pd.Series(0.16, index=idx),  # sum 0.8 < 1
                    max_exposure=1.0, min_exposure=1.0)
    out = diagnose_infeasibility(c, context="t")
    assert all(k.startswith("box_max:") for k in out)
    assert abs(sum(out.values()) - 0.20) < 1e-3   # need +0.20 of cap to reach 1.0


def test_diagnose_infeasibility_feasible_returns_empty():
    """A satisfiable constraint set yields no elastic breaches."""
    from optimalportfolios.optimization.solver_diagnostics import diagnose_infeasibility
    idx = pd.Index(["A", "B", "C"])
    c = Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=idx),
                    max_weights=pd.Series(1.0, index=idx),
                    max_exposure=1.0, min_exposure=1.0)
    assert diagnose_infeasibility(c, context="t") == {}


# -----------------------------------------------------------------------------
# structured records + run-level fallback gate
# -----------------------------------------------------------------------------

def test_rejection_carries_structured_record(caplog):
    """A rejection attaches a SolverDiagnostic with the right fields/metrics."""
    from optimalportfolios.optimization.solver_diagnostics import SolverDiagnostic
    c = _constraints()
    with caplog.at_level(logging.WARNING,
                         logger="optimalportfolios.optimization.solver_diagnostics"):
        validate_solution(_feasible_w() * 1e5, "optimal", c, n=5,
                          solver="CLARABEL", context="GROWM 2021-04-30")
    diags = [getattr(r, "solver_diag", None) for r in caplog.records]
    diags = [d for d in diags if isinstance(d, SolverDiagnostic)]
    assert len(diags) == 1
    d = diags[0]
    assert d.accepted is False and d.outcome == "rejected"
    assert d.severity == logging.ERROR            # optimal status + blow-up
    assert d.fallback_source == "weights_0"
    assert d.context == "GROWM 2021-04-30"
    assert d.budget_residual > 1.0                # sum(w) ~ 1e5 over target 1
    assert d.n_assets == 5


def _attach_summary_at_debug():
    """Attach a fresh summary and lift the logger to DEBUG (as configure_run_logging
    does), returning (summary, restore_fn)."""
    from optimalportfolios.optimization.solver_diagnostics import SolverRejectionSummary
    lg = logging.getLogger("optimalportfolios.optimization.solver_diagnostics")
    h = SolverRejectionSummary()
    prev = lg.level
    lg.setLevel(logging.DEBUG)
    lg.addHandler(h)

    def restore():
        """Detach the handler and restore the logger level."""
        lg.removeHandler(h)
        lg.setLevel(prev)
    return h, restore


def test_summary_counts_every_solve_and_fallback_fraction():
    """Accepted and rejected solves are both counted in the fallback fraction."""
    c = _constraints()
    h, restore = _attach_summary_at_debug()
    try:
        # 3 clean accepts (DEBUG) + 1 rejection (WARNING) -> 25% fallback
        for _ in range(3):
            validate_solution(_feasible_w(), "optimal", c, n=5)
        validate_solution(None, "infeasible", c, n=5)
    finally:
        restore()
    assert h.n_total == 4
    assert h.n_accepted == 3
    assert h.n_rejected == 1
    assert h.n_infeasible_fallback == 1 and h.n_blowup == 0
    assert abs(h.fallback_fraction - 0.25) < 1e-9
    assert "fallback rate 25.0%" in h.summary()


def test_fallback_gate_passes_within_threshold(caplog):
    """A fallback fraction at the threshold passes the gate without an ERROR."""
    c = _constraints()
    h, restore = _attach_summary_at_debug()
    try:
        for _ in range(19):
            validate_solution(_feasible_w(), "optimal", c, n=5)   # accepts
        validate_solution(None, "infeasible", c, n=5)             # 1/20 = 5%
        with caplog.at_level(logging.INFO,
                             logger="optimalportfolios.optimization.solver_diagnostics"):
            ok = h.check_fallback_gate(max_fraction=0.05)
    finally:
        restore()
    assert ok is True                                              # 5% is not > 5%
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_fallback_gate_breaches_and_can_raise(caplog):
    """A breached gate logs at ERROR and raises only when asked to."""
    c = _constraints()
    h, restore = _attach_summary_at_debug()
    try:
        validate_solution(_feasible_w(), "optimal", c, n=5)        # 1 accept
        for _ in range(3):
            validate_solution(None, "infeasible", c, n=5)          # 3/4 = 75%
        # logs ERROR, does not raise
        with caplog.at_level(logging.ERROR,
                             logger="optimalportfolios.optimization.solver_diagnostics"):
            ok = h.check_fallback_gate(max_fraction=0.05, raise_on_breach=False)
        assert ok is False
        assert any("FALLBACK GATE breached" in r.getMessage() for r in caplog.records)
        # raises when asked
        with pytest.raises(RuntimeError, match="FALLBACK GATE breached"):
            h.check_fallback_gate(max_fraction=0.05, raise_on_breach=True)
    finally:
        restore()


def test_fallback_gate_noop_when_no_solves():
    """With no solves recorded the gate passes and the fraction is zero."""
    h, restore = _attach_summary_at_debug()
    try:
        assert h.check_fallback_gate(max_fraction=0.05) is True    # nothing recorded
        assert h.fallback_fraction == 0.0
    finally:
        restore()


# -----------------------------------------------------------------------------
# pre-solve input contract
# -----------------------------------------------------------------------------

_SD_LOGGER = "optimalportfolios.optimization.solver_diagnostics"


def _diag_covar(diag=(0.04, 0.03, 0.02, 0.05, 0.05)):
    """A clean diagonal covariance over ``TICKERS`` for input-contract tests."""
    return pd.DataFrame(np.diag(diag), index=TICKERS, columns=TICKERS)


def test_input_contract_accepts_clean_inputs():
    """Clean covariance and constraints pass the contract with no issues."""
    from optimalportfolios.optimization.solver_diagnostics import validate_solver_inputs
    res = validate_solver_inputs(_diag_covar(), _constraints(), context="t")
    assert res.ok is True
    assert res.issues == []
    assert res.n_assets == 5


def test_input_contract_flags_nonfinite_covar():
    """A non-finite covariance entry is a hard contract failure."""
    from optimalportfolios.optimization.solver_diagnostics import validate_solver_inputs
    m = np.diag([0.04, 0.03, 0.02, 0.05, 0.05])
    m[0, 0] = np.nan
    res = validate_solver_inputs(pd.DataFrame(m, index=TICKERS, columns=TICKERS),
                                 _constraints(), context="t")
    assert res.ok is False
    assert any("non-finite" in s for s in res.issues)


def test_input_contract_flags_box_cannot_reach_budget(caplog):
    """A box that cannot reach full investment is flagged but is not a hard failure."""
    from optimalportfolios.optimization.solver_diagnostics import validate_solver_inputs
    idx = pd.Index(TICKERS)
    c = Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=idx),
                    max_weights=pd.Series(0.15, index=idx),   # 5*0.15 = 0.75 < 1.0
                    max_exposure=1.0, min_exposure=1.0)
    with caplog.at_level(logging.INFO, logger=_SD_LOGGER):
        res = validate_solver_inputs(_diag_covar(), c, context="t")
    assert res.ok is True   # structural infeasibility is flagged, not a hard error
    assert any("full investment is infeasible" in s for s in res.issues)
    assert any("input contract" in r.getMessage() for r in caplog.records)


def test_input_contract_flags_benchmark_out_of_bounds():
    """A benchmark weight above its cap is reported as an issue."""
    from optimalportfolios.optimization.solver_diagnostics import validate_solver_inputs
    idx = pd.Index(TICKERS)
    c = Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=idx),
                    max_weights=pd.Series(0.40, index=idx),
                    max_exposure=1.0, min_exposure=1.0,
                    benchmark_weights=pd.Series([0.60, 0.10, 0.10, 0.10, 0.10],
                                                index=idx))   # 0.60 > 0.40 cap
    res = validate_solver_inputs(_diag_covar(), c, context="t")
    assert any("exceeds its cap" in s for s in res.issues)


def test_input_contract_deep_feasibility_runs_elastic(caplog):
    """``deep_feasibility`` runs the elastic pre-flight and reports its finding."""
    from optimalportfolios.optimization.solver_diagnostics import validate_solver_inputs
    idx = pd.Index(TICKERS)
    c = Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=idx),
                    max_weights=pd.Series(0.15, index=idx),   # box cannot reach 1.0
                    max_exposure=1.0, min_exposure=1.0)
    with caplog.at_level(logging.WARNING, logger=_SD_LOGGER):
        res = validate_solver_inputs(_diag_covar(), c, context="t",
                                     deep_feasibility=True)
    assert any("infeasibility diagnosis" in r.getMessage() for r in caplog.records)
    assert any("elastic pre-flight" in s for s in res.issues)


# -----------------------------------------------------------------------------
# relaxation tally + run-diagnostics bundle + environment banner
# -----------------------------------------------------------------------------

def test_relaxation_summary_tallies_records():
    """The relaxation handler tallies records, bound frequency, and the largest relaxation."""
    from optimalportfolios.optimization.solver_diagnostics import RelaxationSummary
    from optimalportfolios.optimization.constraints import RelaxationRecord
    h = RelaxationSummary()
    lg = logging.getLogger("optimalportfolios.optimization.constraints")
    prev = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(h)
    try:
        r1 = RelaxationRecord(context="d1",
                              items=(("PD", "group_min", 0.005, 0.0045),),
                              total_relaxation=0.0005, max_relaxation=0.0005,
                              breached_budget=False, breached_tol=False)
        r2 = RelaxationRecord(context="d2",
                              items=(("PD", "group_min", 0.005, 0.004),
                                     ("EQ", "group_max", 0.30, 0.33)),
                              total_relaxation=0.031, max_relaxation=0.03,
                              breached_budget=False, breached_tol=True)
        lg.info("relax d1", extra={"relaxation": r1})
        lg.info("relax d2", extra={"relaxation": r2})
    finally:
        lg.removeHandler(h)
        lg.setLevel(prev)
    assert h.n_rebalances_relaxed == 2
    assert h.bound_frequency()[("PD", "group_min")] == 2
    assert h.n_breached_tol == 1
    assert abs(h.max_relaxation - 0.03) < 1e-9
    assert "PD group_min (2)" in h.summary()


def test_run_diagnostics_bundle_delegates():
    """The bundle delegates the gate and concatenates the handler summaries."""
    from optimalportfolios.optimization.solver_diagnostics import (
        RunDiagnostics, SolverRejectionSummary, RelaxationSummary)
    d = RunDiagnostics(rejections=SolverRejectionSummary(),
                       relaxations=RelaxationSummary())
    assert d.check_fallback_gate(max_fraction=0.05) is True   # empty -> within gate
    s = d.summary()
    assert "solver outcomes" in s and "relaxations" in s


def test_log_environment_emits_versions(caplog):
    """The environment banner records library versions and the config hash."""
    from optimalportfolios.optimization.solver_diagnostics import log_environment
    with caplog.at_level(logging.INFO, logger=_SD_LOGGER):
        log_environment(config_hash="abc123")
    text = caplog.text
    assert "run environment" in text and "numpy=" in text and "config=abc123" in text


def test_input_contract_summary_tallies():
    """The contract handler normalises pair order and tallies findings per category."""
    from optimalportfolios.optimization.solver_diagnostics import (
        InputContractSummary, InputContractRecord)
    h = InputContractSummary()
    lg = logging.getLogger(_SD_LOGGER)
    prev = lg.level
    lg.setLevel(logging.DEBUG)
    lg.addHandler(h)
    try:
        # two ill-conditioned solves (same pair, reported in either order) with a
        # benchmark-cap breach; the second also has an unreachable group; one clean.
        r1 = InputContractRecord(context="d1", ok=True, ill_conditioned=True,
                                 cond=float("inf"), min_eig=-1e-16,
                                 collinear_pair=("A", "B"),
                                 benchmarks=((5, "cap_exceeded"),))
        r2 = InputContractRecord(context="d2", ok=True, ill_conditioned=True,
                                 cond=float("inf"), min_eig=-2e-16,
                                 collinear_pair=("B", "A"),
                                 groups=(("Private Debt", "floor_unreachable"),),
                                 benchmarks=((5, "cap_exceeded"),))
        r3 = InputContractRecord(context="d3", ok=True)
        for r in (r1, r2, r3):
            lg.debug("rec", extra={"input_contract": r})
    finally:
        lg.removeHandler(h)
        lg.setLevel(prev)
    assert h.n_solves == 3
    assert h.n_ill_conditioned == 2
    assert h.n_benchmark == 2
    assert h.n_group == 1
    assert h.collinear_frequency()[("A", "B")] == 2     # order-normalised
    assert h.benchmark_frequency()[(5, "cap_exceeded")] == 2
    s = h.summary()
    assert "ill-conditioned on 2/3" in s
    assert "index 5 cap_exceeded (2)" in s
    assert "Private Debt floor_unreachable (1)" in s


def test_run_diagnostics_includes_contract():
    """The bundle includes the input-contract section when a contract handler is attached."""
    from optimalportfolios.optimization.solver_diagnostics import (
        RunDiagnostics, SolverRejectionSummary, RelaxationSummary,
        InputContractSummary)
    d = RunDiagnostics(rejections=SolverRejectionSummary(),
                       relaxations=RelaxationSummary(),
                       contract=InputContractSummary())
    s = d.summary()
    assert "solver outcomes" in s and "relaxations" in s and "input contract" in s


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
