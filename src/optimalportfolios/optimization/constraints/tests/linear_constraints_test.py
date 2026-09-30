"""Signed linear mandates, solver enforcement, alignment and diagnostic contracts."""
import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

import optimalportfolios as op


def _rows(lower=-0.048, upper=None):
    """Return a signed core/overlay policy with a removable zero-loaded asset."""
    return op.LinearConstraints(
        pd.DataFrame({'bear_coverage': [-0.08, -0.04, 0.12, 0.0]},
                     index=['core', 'growth', 'hedge', 'unused']),
        lower=None if lower is None else pd.Series({'bear_coverage': lower}),
        upper=None if upper is None else pd.Series({'bear_coverage': upper}))


def test_linear_public_surface_and_defensive_copy():
    """All OP entry points expose the same object, with independent policy copies."""
    from optimalportfolios.optimization import LinearConstraints
    from optimalportfolios.optimization.constraints import LinearConstraints as Canonical
    assert op.LinearConstraints is LinearConstraints is Canonical
    block = _rows()
    copied = block.copy()
    copied.loadings.iloc[0, 0] = 99.0
    copied.lower.iloc[0] = -99.0
    assert block.loadings.iloc[0, 0] == -0.08
    assert block.lower.iloc[0] == -0.048


@pytest.mark.parametrize('band', [False, True])
@pytest.mark.parametrize('floor', [-0.048, 0.0, 0.005])
def test_signed_row_matches_return_floor_and_reports_residual(band, floor):
    """Named rows and return floors impose the same policy in both Sharpe routes."""
    block = _rows(floor).update(['core', 'growth', 'hedge'])
    covar = np.array([[0.01, 0.002, -0.004],
                      [0.002, 0.0225, 0.0], [-0.004, 0.0, 0.01]])
    means = np.array([0.035, 0.06, -0.02])
    base = op.Constraints(min_weights=pd.Series([1., 0., 0.], index=block.loadings.index),
                          max_weights=pd.Series(1., index=block.loadings.index),
                          min_exposure=1.8 if band else 2.0, max_exposure=2.0)
    outcome = op.cvx_maximize_portfolio_sharpe(
        covar, means, base.copy(linear_constraints=block))
    reference = op.cvx_maximize_portfolio_sharpe(
        covar, means, base.copy(asset_returns=block.loadings.iloc[:, 0], target_return=floor))
    assert outcome.accepted and reference.accepted
    np.testing.assert_allclose(outcome.weights, reference.weights, atol=1e-7)
    rows = op.evaluate_constraint_residuals(outcome.weights,
                                           base.copy(linear_constraints=block))
    row = next(row for row in rows if row.constraint_type == 'linear')
    assert row.name == 'bear_coverage' and row.passed and row.hard
    assert row.actual >= floor - 1e-7
    assert 'bear_coverage' in outcome.residuals_frame().to_string()


def test_alignment_reorders_and_rejects_dropped_loaded_assets():
    """Both constraint update paths keep signed rows aligned and reject weakened policy."""
    block = _rows()
    base = op.Constraints(linear_constraints=block)
    for updated in (base.update(['hedge', 'core', 'growth']),
                    base.update_with_valid_tickers(['hedge', 'core', 'growth'])):
        assert updated.linear_constraints.loadings.index.tolist() == ['hedge', 'core', 'growth']
    with pytest.raises(ValueError, match='drop loaded'):
        base.update_with_valid_tickers(['growth', 'hedge', 'unused'])
    with pytest.raises(ValueError, match='Missing linear loadings'):
        block.update(['new'])
    with pytest.raises(ValueError, match='unique'):
        block.update(['core', 'core'])
    assert _rows(None).update(['hedge']).loadings.index.tolist() == ['hedge']


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_signed_box_reachability_checks_both_sides(side):
    """A negative fixed-core contribution cannot be discarded by membership filtering."""
    block = _rows(lower=0.05) if side == 'lower' else _rows(None, upper=-0.13)
    names = block.loadings.index
    with pytest.raises(ValueError, match='bear_coverage'):
        op.Constraints(linear_constraints=block,
                       min_weights=pd.Series([1., 0., 0., 0.], index=names),
                       max_weights=pd.Series([1., 1., 1., 0.], index=names))


def test_reachability_does_not_invent_caps_or_long_short_floors():
    """Unspecified boxes stay unbounded; zero times infinity contributes zero."""
    op.Constraints(linear_constraints=_rows(10.0))
    op.Constraints(is_long_only=False, linear_constraints=_rows(None, -10.0))
    block = op.LinearConstraints(pd.DataFrame({'zero': [0.]}, index=['a']),
                                 lower=pd.Series({'zero': 0.}))
    op.Constraints(is_long_only=False, linear_constraints=block)
    with pytest.raises(ValueError, match='zero'):
        op.Constraints(linear_constraints=block.copy(lower=pd.Series({'zero': 1.})))


@pytest.mark.parametrize('utility', [False, True])
def test_cvx_compilation_scales_both_bounds_and_keeps_hard_rows(utility):
    """Affine bounds scale with k, including under utility enforcement."""
    block = _rows(-0.05, -0.04)
    spec = op.Constraints(linear_constraints=block,
                          tre_utility_weight=None, turnover_utility_weight=None,
                          max_exposure=2., min_exposure=2.)
    y = cvx.Variable(4)
    k = cvx.Parameter(nonneg=True, value=3.0)
    if utility:
        _, rows = spec.set_cvx_utility_objective_constraints(y, exposure_scaler=k)
    else:
        rows = spec.set_cvx_all_constraints(y, exposure_scaler=k)
    y.value = np.array([1., 0.55, 0.45, 0.]) * k.value
    assert max(float(np.max(row.violation())) for row in rows) < 1e-12
    y.value = np.array([1., 0.9, 0.1, 0.]) * k.value
    assert max(float(np.max(row.violation())) for row in rows) > 0.1


def test_upper_and_zero_rows_compile_and_diagnose_without_scaler():
    """SciPy and CVXPY keep upper bounds and detect a violated zero coefficient row."""
    block = op.LinearConstraints(pd.DataFrame({'signed': [-1., 1.], 'zero': [0., 0.]},
                                             index=['a', 'b']),
                                 upper=pd.Series({'signed': 0.2, 'zero': 0.0}))
    spec = op.Constraints(linear_constraints=block)
    y = cvx.Variable(2)
    rows = spec.set_cvx_all_constraints(y)
    callbacks, _ = spec.set_scipy_constraints(np.eye(2))
    y.value = np.array([0.25, 0.75])
    assert any(np.max(row.violation()) > 0.2 for row in rows)
    assert any(np.min(row['fun'](y.value)) < -0.2 for row in callbacks)
    residuals = op.evaluate_constraint_residuals(y.value, spec)
    signed = next(row for row in residuals if row.name == 'signed')
    assert not signed.passed and signed.violation == pytest.approx(0.3)
    assert next(row for row in residuals if row.name == 'zero').passed
    with pytest.raises(ValueError, match='linear_constraints'):
        spec.set_pyrb_constraints(np.eye(2))


@pytest.mark.parametrize('change,match', [
    ({'loadings': pd.DataFrame()}, 'nonempty'),
    ({'loadings': pd.DataFrame({'x': [1., 2.]}, index=['a', 'a'])}, 'unique'),
    ({'loadings': pd.DataFrame([[1., 2.]], columns=['x', 'x'])}, 'unique'),
    ({'loadings': pd.DataFrame({'x': [1.]}, index=[None])}, 'missing'),
    ({'loadings': pd.DataFrame({'x': [np.nan]})}, 'finite'),
    ({'loadings': pd.DataFrame({'x': [np.inf]})}, 'finite'),
    ({'lower': pd.Series([1., 2.], index=['bear_coverage'] * 2)}, 'unique'),
    ({'lower': pd.Series({'typo': 1.})}, 'unknown'),
    ({'lower': pd.Series({'bear_coverage': np.inf})}, 'infinite'),
    ({'upper': pd.Series({'bear_coverage': -np.inf})}, 'infinite'),
    ({'upper': pd.Series({'bear_coverage': -1.})}, 'exceed'),
])
def test_linear_policy_validation(change, match):
    """Malformed rows must fail before a backend can misinterpret their labels."""
    with pytest.raises(ValueError, match=match):
        _rows().copy(**change)


def test_partial_nan_and_infinite_bounds_are_unbounded():
    """Unbounded sides compile no row; bounded zero coefficients are retained."""
    block = op.LinearConstraints(pd.DataFrame({'a': [1.], 'b': [0.], 'c': [-1.]}),
                                 lower=pd.Series({'a': -np.inf, 'b': 0.}),
                                 upper=pd.Series({'a': np.inf, 'c': np.nan}))
    assert [name for name, _, _, _ in block.iter_bounds()] == ['b']
