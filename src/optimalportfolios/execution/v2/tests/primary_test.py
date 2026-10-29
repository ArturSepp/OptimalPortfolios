"""Matched primary/fallback contracts on real synthetic prepared execution domains."""

from dataclasses import fields, replace

import numpy as np
import pytest

from optimalportfolios.execution.v2 import primary
from optimalportfolios.execution.v2 import ExecutionResult, prepare_common_execution
from optimalportfolios.execution.v2.production import resolve_production_problem
from optimalportfolios.execution.v2.tests.production_test import inputs


def production_inputs():
    """Keep the original synthetic workbook controls in the matched policy."""
    table, constraints, covariance, config = inputs()
    values = {item.name: getattr(config, item.name) for item in fields(config)}
    values['allow_v1_fallback'] = False
    return table, constraints, covariance, primary.MatchedExecutionPolicyConfig(**values)


@pytest.fixture
def problem():
    """Prepare a funded synthetic problem using its original constraints."""
    table, constraints, covariance, config = production_inputs()
    raw, _ = resolve_production_problem(table, constraints, covariance, config)
    prepared, _ = prepare_common_execution(raw, solver='SCIPY')
    assert prepared is not None
    return prepared


def native_result(problem, status='solsta.integer_optimal', weights=None,
                  bound=True, witness=True):
    """Attribute a test native result without claiming independent feasibility."""
    return ExecutionResult(
        problem.base.copy() if weights is None else weights, 'audited_incumbent',
        dict(native_status=status, native_bound_check=dict(consistent_with_delivery=bound),
             incumbent_events=[dict(source='native_return', audit=dict(passed=True))]
             if witness else []))


def test_audited_primary_does_not_call_or_warm_start_from_v2(problem, monkeypatch):
    """Preserve qualified budgets and return a primary without spurious fallback labels."""
    def native(p, formulation, **kwargs):
        """Verify the exact controls passed to the real public adapter."""
        assert p is problem and formulation == 'CORRIDOR'
        assert kwargs == dict(time_limit_s=10., threads=1, gap_rel=1e-4)
        return native_result(p)

    def forbidden(*args):
        """Detect accidental eager V2 search or a V2 MIP start."""
        raise AssertionError('V2 called before it was needed')

    monkeypatch.setattr(primary, 'wrapper_exact_execution', native)
    monkeypatch.setattr(primary, 'wrapper_ranked_execution', forbidden)
    solved = primary.wrapper_matched_execution(problem, production_inputs()[3])
    assert solved.diagnostics['engine'] == 'matched_miqp'
    assert solved.diagnostics['fallback_reason'] is None
    assert solved.diagnostics['audit']['passed']


@pytest.mark.parametrize('case,reason', [
    ('unavailable', 'native_unavailable_or_failed'),
    ('exception', 'primary_exception'),
    ('no_native', 'no_audited_native_candidate'),
    ('invalid_weights', 'primary_independent_audit_failed'),
    ('bad_bound', 'native_bound_inconsistent'),
    ('callback', 'native_callback_error'),
])
def test_rejected_primary_uses_same_prepared_domain(problem, monkeypatch, case, reason):
    """Neither a status nor a common prepared incumbent can hide primary failure."""
    calls = []

    def native(p, *args, **kwargs):
        """Inject one independent primary failure at a time."""
        if case == 'exception':
            raise RuntimeError('solver failed')
        candidate = native_result(
            p, status='solver_unavailable' if case == 'unavailable' else 'solsta.integer_optimal',
            weights=p.base+.01 if case == 'invalid_weights' else None,
            bound=case != 'bad_bound', witness=case != 'no_native')
        if case == 'callback':
            candidate.diagnostics['callback_errors'] = ['injected']
        return candidate

    def fallback(p, config):
        """Require immutable input identity and return a genuinely feasible witness."""
        assert p is problem
        calls.append(True)
        return native_result(p)

    monkeypatch.setattr(primary, 'wrapper_exact_execution', native)
    monkeypatch.setattr(primary, 'wrapper_ranked_execution', fallback)
    solved = primary.wrapper_matched_execution(problem, production_inputs()[3])
    assert calls == [True]
    assert solved.diagnostics['engine'] == 'v2_audited_fallback'
    assert solved.diagnostics['fallback_reason'] == reason
    assert solved.diagnostics['v2']['audit']['passed']


def test_invalid_fallback_is_rejected_with_reason(problem, monkeypatch):
    """A genuine funding breach remains a failure, including after native rejection."""
    monkeypatch.setattr(primary, 'wrapper_exact_execution',
                        lambda p, *a, **k: native_result(p, witness=False))
    monkeypatch.setattr(primary, 'wrapper_ranked_execution',
                        lambda p, *a: native_result(p, weights=p.base+.01))
    with pytest.raises(RuntimeError, match='unchanged common audit') as caught:
        primary.wrapper_matched_execution(problem, production_inputs()[3])
    assert caught.value.diagnostics['fallback_reason'] == 'no_audited_native_candidate'
    assert 'funding' in caught.value.diagnostics['v2']['audit']['violations']


def test_time_limited_audited_native_incumbent_remains_eligible(problem, monkeypatch):
    """Retain the qualified policy for a valid native incumbent without optimal status."""
    monkeypatch.setattr(primary, 'wrapper_exact_execution',
                        lambda p, *a, **k: native_result(p, status='solsta.prim_feas'))
    solved = primary.wrapper_matched_execution(problem, production_inputs()[3])
    assert solved.diagnostics['engine'] == 'matched_miqp'


@pytest.mark.parametrize('fallback', [False, True])
def test_production_bridge_retains_original_controls_and_engine(monkeypatch, fallback):
    """Preserve bands, tickets and floor while distinguishing primary from fallback."""
    table, constraints, covariance, config = production_inputs()
    monkeypatch.setattr(primary, 'wrapper_exact_execution',
                        lambda p, *a, **k: native_result(p, witness=not fallback))
    solved = primary.solve_matched_execution_portfolio(
        table, constraints, covariance, config, context='synthetic production')
    assert solved.accepted and solved.compliant
    engine = 'v2_audited_fallback' if fallback else 'matched_miqp'
    assert solved.trade_table.execution_engine.eq(engine).all()
    assert solved.outcome.fallback_source == (engine if fallback else None)
    assert solved.outcome.constraints.group_lower_upper_constraints is (
        constraints.group_lower_upper_constraints)
    assert solved.trade_table.attrs['v2']['audit']['passed']
    assert config.execution_method == 'matched_miqp_v2'


@pytest.mark.parametrize('failure', ['preparation', 'fallback', 'unsupported'])
def test_production_failure_preserves_trade_table_and_diagnostics(monkeypatch, failure):
    """Failed preparation or search cannot become an executable display portfolio."""
    import pandas as pd

    table, constraints, covariance, config = production_inputs()
    if failure == 'preparation':
        monkeypatch.setattr(primary, 'prepare_common_execution',
                            lambda *a, **k: (None, {'status': 'infeasible'}))
    elif failure == 'fallback':
        monkeypatch.setattr(primary, 'wrapper_exact_execution',
                            lambda p, *a, **k: native_result(p, witness=False))
        monkeypatch.setattr(primary, 'wrapper_ranked_execution',
                            lambda p, *a: native_result(p, weights=p.base+.01))
    else:
        constraints = replace(constraints, max_target_portfolio_vol_an=.2)
    with pytest.raises((ValueError, RuntimeError)) as caught:
        primary.solve_matched_execution_portfolio(table, constraints, covariance, config)
    pd.testing.assert_frame_equal(caught.value.trade_table, table)
    if failure == 'fallback':
        assert caught.value.diagnostics['fallback_reason'] == 'no_audited_native_candidate'


@pytest.mark.parametrize('kwargs', [
    {'allow_v1_fallback': True}, {'primary_time_limit_s': 0.},
    {'primary_time_limit_s': np.nan}, {'primary_gap_rel': -1.},
    {'primary_gap_rel': np.inf}, {'primary_threads': True},
    {'primary_threads': 1.5}, {'primary_threads': 0},
])
def test_matched_policy_rejects_invalid_controls(kwargs):
    """Reject a third fallback and invalid budgets without changing audit tolerances."""
    with pytest.raises(ValueError):
        primary.MatchedExecutionPolicyConfig(tre_weight_by_asset_class={'Risk': 1.}, **kwargs)
