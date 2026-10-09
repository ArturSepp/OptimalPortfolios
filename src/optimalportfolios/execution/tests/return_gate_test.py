"""Common-audit gating without solver changes or original-anchor drift."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import schema as s
from optimalportfolios.execution import _return_gate as gate
from optimalportfolios.execution.branches import ExecutionBranchConfig, solve_branched_execution
from optimalportfolios.execution.solver import RELAXED_CORRIDOR_SOLVE
from optimalportfolios.execution.tests.improvement_test import _problem
from optimalportfolios.optimization.constraints import GroupLowerUpperConstraints


def _scripted(monkeypatch, metrics):
    """Supply independent metrics and distinguishable immutable candidate weights."""
    problem = _problem(max_trades=3)
    candidates = {}
    for i, name in enumerate(metrics):
        candidates[name] = SimpleNamespace(accepted=True, compliant=True,
            weights=pd.Series(float(i), index=problem.target.index, name=name),
            trade_table=pd.DataFrame({RELAXED_CORRIDOR_SOLVE: False}, index=problem.target.index))
    monkeypatch.setattr(gate, '_metrics',
                        lambda problem, result, *args: metrics[result.weights.name])
    search = SimpleNamespace(checkpoints=candidates,
                             summary={'selected_checkpoint': list(metrics)[-1]})
    return problem, search


def _values(te=10., tickets=3, sized=2, turnover=100.):
    """Independently declare scalar checkpoint properties in the guard's units."""
    return dict(te_bp=te, tickets=tickets, sized_tickets=sized, gross_turnover_bp=turnover)


def test_rejects_best_invalid_return_and_recovers_earlier_valid_improvement(monkeypatch):
    """A failed final candidate must not displace a valid earlier checkpoint."""
    p, search = _scripted(monkeypatch, dict(baseline=_values(),
        earlier=_values(te=9.), invalid=_values(te=9.5, tickets=2)))
    before = deepcopy(search)
    result = gate._gate_branched_return(p, search, ExecutionBranchConfig(),
                                       lambda weights: weights.name != 'invalid')
    assert result.result is search.checkpoints['earlier']
    assert result.summary['selected_checkpoint'] == 'earlier'
    assert result.summary['common_audit_rejections'] == 1
    for name in search.checkpoints:
        pd.testing.assert_series_equal(search.checkpoints[name].weights,
                                       before.checkpoints[name].weights)


def test_valid_baseline_is_retained_when_every_improvement_fails(monkeypatch):
    """Fallback itself must pass the same audit."""
    p, search = _scripted(monkeypatch, dict(baseline=_values(), invalid=_values(te=9.)))
    result = gate._gate_branched_return(p, search, ExecutionBranchConfig(),
                                       lambda weights: weights.name == 'baseline')
    assert result.result is search.checkpoints['baseline']


def test_no_valid_incumbent_returns_none_not_rejected_baseline(monkeypatch):
    """Known invalid fallback cannot silently escape the gate."""
    p, search = _scripted(monkeypatch, dict(baseline=_values(), invalid=_values(te=9.)))
    result = gate._gate_branched_return(p, search, ExecutionBranchConfig(), lambda weights: False)
    assert result.result is None
    assert result.summary['stop_reason'] == 'no_common_audited_incumbent'


def test_valid_candidate_can_resolve_invalid_baseline_without_resetting_guards(monkeypatch):
    """An invalid baseline remains the guard anchor, not an authorized fallback."""
    p, search = _scripted(monkeypatch, dict(baseline=_values(), good=_values(te=9.)))
    result = gate._gate_branched_return(p, search, ExecutionBranchConfig(),
                                       lambda weights: weights.name == 'good')
    assert result.result is search.checkpoints['good']


@pytest.mark.parametrize('candidate', [
    _values(te=9., tickets=4), _values(te=9., sized=3),
    _values(te=9., turnover=101.), _values(te=11.1, tickets=2),
])
def test_common_feasibility_cannot_override_original_return_limits(monkeypatch, candidate):
    """Feasible exploratory checkpoints still obey original operational caps."""
    p, search = _scripted(monkeypatch, dict(baseline=_values(), ineligible=candidate))
    result = gate._gate_branched_return(p, search, ExecutionBranchConfig(), lambda weights: True)
    assert result.result is search.checkpoints['baseline']


def test_audit_callback_gets_a_copy_and_must_return_boolean(monkeypatch):
    """Caller mutation cannot change a candidate; truthy records are not booleans."""
    p, search = _scripted(monkeypatch, dict(baseline=_values()))
    before = search.checkpoints['baseline'].weights.copy()

    def accept(weights):
        """Deliberately mutate the audit input to prove isolation."""
        weights.iloc[:] = 99.
        return True

    gate._gate_branched_return(p, search, ExecutionBranchConfig(), accept)
    pd.testing.assert_series_equal(before, search.checkpoints['baseline'].weights)
    with pytest.raises(TypeError, match='boolean'):
        gate._gate_branched_return(p, search, ExecutionBranchConfig(),
                                  lambda weights: {'passed': False})


def test_real_search_gate_preserves_solver_work_and_unfiltered_winner_when_valid():
    """Actual projections and a simple independent box/budget audit exercise the seam."""
    p, controls = _problem(max_trades=3), ExecutionBranchConfig()
    search = solve_branched_execution(p, controls)
    before = deepcopy(search.summary)
    result = gate._gate_branched_return(p, search, controls,
        lambda weights: bool(abs(weights.sum()-1.) < 1e-7 and weights.between(-1e-7, 1+1e-7).all()))
    assert result.result is search.result
    assert search.summary == before


def test_native_failed_baseline_stops_before_common_audit(monkeypatch):
    """A common audit cannot rehabilitate a failed native baseline solve."""
    p, search = _scripted(monkeypatch, dict(baseline=_values()))
    search.checkpoints['baseline'].accepted = False
    calls = []
    result = gate._gate_branched_return(p, search, ExecutionBranchConfig(),
                                       lambda weights: calls.append(weights) or True)
    assert result.result is None
    assert result.summary['stop_reason'] == 'baseline_not_accepted'
    assert result.audit.empty and not calls


@pytest.mark.parametrize('reason', ['candidate_not_accepted', 'candidate_relaxed_corridors'])
def test_native_rejected_or_relaxed_candidate_cannot_pass_common_gate(monkeypatch, reason):
    """The callback supplements native admission; it never replaces it."""
    p, search = _scripted(monkeypatch, dict(baseline=_values(), invalid=_values(te=9.)))
    if reason == 'candidate_not_accepted':
        search.checkpoints['invalid'].compliant = False
    else:
        search.checkpoints['invalid'].trade_table.loc[:, RELAXED_CORRIDOR_SOLVE] = True
    calls = []
    result = gate._gate_branched_return(p, search, ExecutionBranchConfig(),
                                       lambda weights: calls.append(weights.name) or True)
    assert result.result is search.checkpoints['baseline']
    assert calls == ['baseline']
    assert result.audit.set_index('checkpoint').loc['invalid', 'reason'] == reason


def _precision_case(gap=1.2e-7):
    """Build a nearly inconsistent A floor and a feasible, suboptimal C holding."""
    p = _problem(max_trades=3)
    table = p.target.copy()
    table.loc['A', s.EFFECTIVE_MODEL_WEIGHT] = .4-gap
    table.loc['C', s.EFFECTIVE_MODEL_WEIGHT] = .22
    table.loc['Cash', s.EFFECTIVE_MODEL_WEIGHT] = .18+gap
    table[s.DESIRED_REWEIGHT] = table[s.EFFECTIVE_MODEL_WEIGHT]-table[s.BASE_WEIGHT]
    groups = GroupLowerUpperConstraints(
        pd.DataFrame({'A floor': [1., 0., 0., 0.]}, index=table.index),
        pd.Series({'A floor': .4}), pd.Series({'A floor': .5}))
    p = replace(p, target=table,
                constraints=p.constraints.copy(group_lower_upper_constraints=groups))
    base = solve_branched_execution(p, ExecutionBranchConfig(max_projection_calls=0)).result
    weights = pd.Series([.4-gap, .2, .215, .185+gap], index=table.index)
    base = replace(base, weights=weights,
                   trade_table=gate._precision_trade_table(base.trade_table, weights),
                   outcome=replace(base.outcome, weights=weights.to_numpy()))
    return p, base


def test_precision_correction_passes_original_audit_and_keeps_return_guards():
    """Analytical A-floor audit requires correction; C permits lower risk and turnover."""
    p, base = _precision_case()
    before = deepcopy(p)
    result = gate._recover_precision_return(p, base, ExecutionBranchConfig(),
        lambda w: bool(w['A'] >= .4-1e-7 and abs(w.sum()-1.) < 1e-7))
    assert result.result is not None
    w = result.result.weights
    assert .4-1e-7 <= w['A'] <= p.target.loc['A', s.EFFECTIVE_MODEL_WEIGHT]+1e-7
    assert (w-base.weights).abs().max() <= 1e-5+1e-10
    assert result.summary['final']['te_bp'] < result.summary['baseline']['te_bp']-1e-4
    assert result.summary['final']['gross_turnover_bp'] <= (
        result.summary['baseline']['gross_turnover_bp']+1e-8)
    np.testing.assert_allclose(result.result.trade_table['executed_trade'],
                               w-p.target[s.CURRENT_WEIGHT], atol=0.)
    pd.testing.assert_frame_equal(p.target, before.target)
    pd.testing.assert_frame_equal(base.trade_table, gate._precision_trade_table(
        base.trade_table, base.weights))


def test_precision_recovery_does_not_inflate_tolerance_for_material_inconsistency():
    """A genuine 0.2 bp shortfall cannot be hidden inside the 0.001 bp audit."""
    p, base = _precision_case(gap=2e-5)
    result = gate._recover_precision_return(p, base, ExecutionBranchConfig(),
                                           lambda w: bool(w['A'] >= .4-1e-7))
    assert result.result is None
    assert result.summary['stop_reason'] == 'no_optimal_correction'


def test_precision_recovery_must_pass_caller_audit_and_improvement_requirement():
    """A converged auxiliary solve alone never authorizes a returned portfolio."""
    p, base = _precision_case()
    denied = gate._recover_precision_return(p, base, ExecutionBranchConfig(), lambda w: False)
    assert denied.result is None and denied.summary['stop_reason'] == 'common_audit_failed'
    guarded = gate._recover_precision_return(p, base,
        ExecutionBranchConfig(min_te_improvement_bp=100.),
        lambda w: bool(w['A'] >= .4-1e-7))
    assert guarded.result is None
    assert guarded.summary['stop_reason'] == 'no_qualifying_improvement'


def test_precision_recovery_leaves_an_audited_incumbent_untouched():
    """Valid returns incur no recovery solve or numerical drift."""
    p, base = _precision_case()
    result = gate._recover_precision_return(p, base, ExecutionBranchConfig(), lambda w: True)
    assert result.result is base and result.summary['projection_calls'] == 0


@pytest.mark.parametrize('reason', ['baseline_not_accepted', 'baseline_relaxed_corridors'])
def test_precision_recovery_does_not_override_native_admission(reason):
    """Neither a native failure nor an intentional relaxation is a precision case."""
    base = SimpleNamespace(accepted=reason != 'baseline_not_accepted', compliant=True,
        trade_table=pd.DataFrame({RELAXED_CORRIDOR_SOLVE: [True]}))
    result = gate._recover_precision_return(None, base, ExecutionBranchConfig(), None)
    assert result.result is None and result.summary['stop_reason'] == reason
    assert result.summary['projection_calls'] == 0


@pytest.mark.parametrize('at_final', [False, True])
def test_precision_recovery_requires_boolean_at_each_audit(at_final):
    """A truthy diagnostic record cannot accidentally authorize a returned correction."""
    p, base = _precision_case()
    replies = iter([False, {'passed': False}] if at_final else [{'passed': False}])
    with pytest.raises(TypeError, match='boolean'):
        gate._recover_precision_return(p, base, ExecutionBranchConfig(), lambda w: next(replies))


def test_precision_recovery_rejects_baseline_outside_correction_region():
    """An accepted-looking but materially inconsistent input is not repaired silently."""
    p, base = _precision_case()
    weights = base.weights.copy()
    weights.loc['A'] = .5
    result = gate._recover_precision_return(p, replace(base, weights=weights),
                                            ExecutionBranchConfig(), lambda w: False)
    assert result.result is None and result.summary['stop_reason'] == 'empty_correction_interval'
    assert result.summary['projection_calls'] == 0


@pytest.mark.parametrize('failure', ['solver_error', 'correction_too_large',
                                    'original_constraint_audit_failed'])
def test_precision_recovery_fails_closed_on_numerical_faults(monkeypatch, failure):
    """Solver exceptions, inaccurate output and failed native audits cannot escape."""
    p, base = _precision_case()
    if failure == 'solver_error':
        def broken(program, **kwargs):
            """Emulate a numerical solver failure."""
            raise gate.cvx.error.SolverError('injected numerical failure')
        monkeypatch.setattr(gate.cvx.Problem, 'solve', broken)
    elif failure == 'correction_too_large':
        def inaccurate(program, **kwargs):
            """Emulate an optimal status accompanied by an excessive correction."""
            program.variables()[0].value = np.full(4, 10.)
            program._status = gate.cvx.OPTIMAL
        monkeypatch.setattr(gate.cvx.Problem, 'solve', inaccurate)
    else:
        monkeypatch.setattr(gate, 'validate_solution', lambda *args, **kwargs:
            SimpleNamespace(accepted=False, compliant=False, residuals_frame=pd.DataFrame))
    result = gate._recover_precision_return(p, base, ExecutionBranchConfig(), lambda w: False)
    assert result.result is None and result.summary['stop_reason'] == failure


def test_precision_recovery_enforces_cash_funding_inside_the_correction_region():
    """A small native funding residual must not be fixed after leaving the trust region."""
    p, base = _precision_case()
    table = base.trade_table.copy()
    table.loc['A', s.MANDATORY_FUNDING] += 1e-9
    result = gate._recover_precision_return(p, replace(base, trade_table=table),
        ExecutionBranchConfig(), lambda w: bool(w['A'] >= .4-1e-7))
    assert result.result is not None
    weights = result.result.weights
    cash = table[s.SETTLEMENT_CASH]
    expected = (table.loc[cash, s.CURRENT_WEIGHT].iloc[0]
        -table.loc[~cash, s.MANDATORY_FUNDING].sum()
        -(weights-table[s.BASE_WEIGHT]).loc[~cash].sum())
    assert weights.loc[cash].iloc[0] == expected
    assert (weights-base.weights).abs().max() <= 1e-5+1e-10
