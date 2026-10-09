"""Independent optional-support, original-guard, beam and shared-budget contracts."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import branches as b, schema as s
from optimalportfolios.execution._budget import _consume_projection
from optimalportfolios.execution._swaps import _exact_swap, _support, _swap_candidates
from optimalportfolios.execution.tests.improvement_test import _problem
from optimalportfolios.execution.tests.branches_test import _values


def _script(monkeypatch, values, pairs, replies=None, fourth=False):
    """Use independently supplied metrics and counted strict projections for path tests."""
    problem=_problem(max_trades=2)
    original=b._solve(problem,b.score_execution_candidates(problem),rescue=True)
    table=original.trade_table.copy()
    if fourth:
        table.loc['D']=table.loc['C'].copy()
    optional=list(table.index[~table[s.SETTLEMENT_CASH]])
    table[s.SELECTED_TRADE]=False
    table.loc[optional[:2],s.SELECTED_TRADE]=True
    baseline=replace(original,trade_table=table)
    metric=iter(values)
    scripted=iter(replies) if replies is not None else None
    calls=[]
    def solve(p,frame,rescue):
        """Charge actual entry calls and copy the prescribed support into each outcome."""
        calls.append((rescue,frame.copy(deep=True)))
        if len(calls)==1:
            return baseline
        _consume_projection()
        if scripted is not None:
            reply=next(scripted)
            if isinstance(reply,Exception):
                raise reply
            if reply is not None:
                return reply
        return replace(baseline,trade_table=frame.copy(deep=True))
    monkeypatch.setattr(b,'_solve',solve)
    monkeypatch.setattr(b,'_metrics',lambda *args:next(metric))
    choices=iter(pairs)
    monkeypatch.setattr(b,'_swap_candidates',lambda *args:next(choices))
    monkeypatch.setattr(b,'_turnover_capped_problem',lambda p,*args:p)
    return problem,baseline,calls,optional


def test_provisional_swap_is_not_returned_but_can_lead_to_improvement(monkeypatch):
    """A worse first level never resets the baseline allowance or replaces the fallback."""
    p,baseline,calls,names=_script(monkeypatch,[_values(tickets=2),
        _values(te=10.8,tickets=2),_values(te=9.,tickets=2)],
        [[(1.,'A','C')],[(1.,'B','A')]])
    delivered=[]
    result=b.solve_branched_execution(p,b.ExecutionBranchConfig(methods=(),max_removal_trials=0,
        max_projection_calls=2,max_swap_trials=2,swap_depth=2),
        on_incumbent=lambda name,result:delivered.append(name))
    assert result.attempts.branch_retained.tolist()==[True,True]
    assert result.attempts.return_retained.tolist()==[False,True]
    assert result.attempts.parent_checkpoint.tolist()==['baseline','checkpoint_00001']
    assert result.summary['final']['te_bp']==9.
    assert result.summary['baseline']['te_bp']==10.
    assert len(calls)==3 and all(not rescue for rescue,_ in calls[1:])
    assert result.summary['swap_trials']==result.summary['search_projection_calls']==2
    assert delivered==['baseline','checkpoint_00002']


@pytest.mark.parametrize('candidate',[
    _values(te=11.01,tickets=2),_values(te=9.,tickets=3),
    _values(te=9.,tickets=2,sized=3),_values(te=9.,tickets=2,turnover=100.0001)])
def test_swap_caps_reject_provisional_and_returned_states(monkeypatch,candidate):
    """Risk, parent ticket count and original operational caps each gate a beam state."""
    p,baseline,_,_=_script(monkeypatch,[_values(tickets=2),candidate],[[(1.,'A','C')]])
    result=b.solve_branched_execution(p,b.ExecutionBranchConfig(methods=(),max_removal_trials=0,
        max_swap_trials=1,max_projection_calls=1))
    assert result.result is baseline
    assert not result.attempts.iloc[0].branch_retained
    assert result.summary['swap_frontier']==[]


def test_provisional_risk_allowance_does_not_accumulate(monkeypatch):
    """Two successive 0.8-bp increases cannot enlarge a one-bp original allowance."""
    p,baseline,_,_=_script(monkeypatch,[_values(tickets=2),_values(te=10.8,tickets=2),
        _values(te=11.6,tickets=2)], [[(1.,'A','C')],[(1.,'B','A')]])
    result=b.solve_branched_execution(p,b.ExecutionBranchConfig(methods=(),max_removal_trials=0,
        max_swap_trials=2,max_projection_calls=2,swap_depth=2))
    assert result.result is baseline
    assert result.attempts.branch_retained.tolist()==[True,False]


def test_swaps_reserve_existing_work_and_charge_structural_failure(monkeypatch):
    """Deletion reservations cannot create extra work and failed swaps consume the same counter."""
    p,baseline,calls,_=_script(monkeypatch,[_values(tickets=2),_values(tickets=2)],
        [[(1.,'A','C')]],replies=[None,b.ExecutionSolverInfeasibility('swap conflict')])
    result=b.solve_branched_execution(p,b.ExecutionBranchConfig(methods=(),max_projection_calls=2,
        max_swap_trials=1,max_removal_trials=10))
    assert result.summary['removal_trials']==1
    assert result.summary['swap_trials']==1
    assert result.summary['search_projection_calls']==2 and len(calls)==3
    assert result.attempts.iloc[-1].status=='interval_infeasible'
    assert result.result is baseline


def test_spare_budget_swaps_do_not_displace_deletion_work(monkeypatch):
    """A useful last-budget deletion remains available when swaps use only spare calls."""
    p, _, calls, _ = _script(monkeypatch,
        [_values(tickets=2), _values(tickets=2), _values(te=10.5, tickets=1)], [])
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(methods=(),
        max_projection_calls=2, max_removal_trials=10, max_swap_trials=1,
        reserve_swap_budget=False))
    assert result.summary['removal_trials'] == 2
    assert result.summary['swap_trials'] == 0
    assert result.summary['final']['tickets'] == 1
    assert result.summary['search_projection_calls'] == 2 and len(calls) == 3
    assert result.summary['removal_phase_projection_limit'] == 2


def test_spare_budget_support_continuation_is_retained(monkeypatch):
    """Finishing deletions does not suppress a productive provisional two-level path."""
    p, _, calls, _ = _script(monkeypatch,
        [_values(tickets=2), _values(te=10.8, tickets=2), _values(te=9., tickets=2)],
        [[(1., 'A', 'C')], [(1., 'B', 'A')]])
    result = b.solve_branched_execution(p, b.ExecutionBranchConfig(methods=(),
        max_removal_trials=0, max_projection_calls=2, max_swap_trials=2, swap_depth=2,
        reserve_swap_budget=np.bool_(False)))
    assert result.summary['final']['te_bp'] == 9.
    assert result.summary['swap_trials'] == 2 and len(calls) == 3
    with pytest.raises(ValueError, match='reserve_swap_budget'):
        b.ExecutionBranchConfig(reserve_swap_budget='no')


def test_visited_supports_do_not_cycle(monkeypatch):
    """Returning to the initial support costs no new projection and cannot form a loop."""
    p,baseline,calls,_=_script(monkeypatch,[_values(tickets=2),_values(te=10.8,tickets=2)],
        [[(1.,'A','C')],[(1.,'C','A')]])
    result=b.solve_branched_execution(p,b.ExecutionBranchConfig(methods=(),max_removal_trials=0,
        max_swap_trials=4,max_projection_calls=4,swap_depth=2))
    assert result.summary['swap_trials']==1 and len(calls)==2
    assert result.result is baseline


def test_swap_change_must_be_exact(monkeypatch):
    """A solver cannot silently retain the outgoing coordinate or alter other support rows."""
    p,baseline,_,_=_script(monkeypatch,[_values(tickets=2),_values(te=9,tickets=2)],
        [[(1.,'A','C')]])
    monkeypatch.setattr(b,'_exact_swap',lambda *args:False)
    result=b.solve_branched_execution(p,b.ExecutionBranchConfig(methods=(),max_removal_trials=0,
        max_swap_trials=1,max_projection_calls=1))
    assert result.result is baseline and result.attempts.iloc[0].branch_reason=='swap_rejected'


@pytest.mark.parametrize('field,value',[
    ('max_swap_trials',True),('max_swap_trials',-1),('max_swap_trials',1.5),
    ('swap_candidate_limit',0),('swap_depth',0),('swap_beam_width',0),
    ('swap_depth',np.bool_(True)),('swap_candidate_limit',1.5),('swap_beam_width',-2)])
def test_swap_control_validation(field,value):
    """Budgets and frontier bounds must be finite integers with explicit positive limits."""
    with pytest.raises(ValueError):
        b.ExecutionBranchConfig(**{field:value})
    with pytest.raises(ValueError,match='cannot exceed'):
        b.ExecutionBranchConfig(max_projection_calls=1,max_swap_trials=2)


def test_exact_edit_protects_mandatory_cash_and_cadence():
    """Every disallowed incoming/outgoing coordinate is excluded from exact support edits."""
    p=_problem(max_trades=2)
    parent=b.score_execution_candidates(p)
    parent[s.SELECTED_TRADE]=False
    parent.at['A',s.SELECTED_TRADE]=True
    child=parent.copy(deep=True)
    child.at['A',s.SELECTED_TRADE]=False
    child.at['C',s.SELECTED_TRADE]=True
    assert _exact_swap(parent,child,'A','C')
    for field,value in [(s.MANDATORY_TRADE,True),(s.REBALANCE_CADENCE_ELIGIBLE,False),
                        (s.RULE4_TRADE,False),(s.SETTLEMENT_CASH,True)]:
        blocked=parent.copy(deep=True)
        blocked.at['C',field]=value
        assert not _exact_swap(blocked,child,'A','C')
    assert not _exact_swap(parent,child,'A','A')
    assert not _exact_swap(parent,child,'missing','C')
    assert not _exact_swap(parent,child.iloc[::-1],'A','C')
    assert not _exact_swap(parent,parent,'A','C')
    assert _support(child)==('C',)


def test_funded_pair_ordering_and_input_ownership():
    """The kernel yields finite deterministic pairs without changing numerical inputs."""
    p=_problem(max_trades=1)
    result=b._solve(p,b.score_execution_candidates(p),rescue=True)
    before=deepcopy((p.target,p.covariance,result.weights,result.trade_table))
    pairs=_swap_candidates(p,result,2,True)
    assert pairs and all(np.isfinite(score) for score,_,_ in pairs)
    assert pairs==_swap_candidates(p,result,2,True)
    assert _swap_candidates(p,result,2,False)
    for actual,expected in zip((p.target,p.covariance,result.weights,result.trade_table),before):
        if isinstance(actual,pd.DataFrame):
            pd.testing.assert_frame_equal(actual,expected)
        else:
            pd.testing.assert_series_equal(actual,expected)
    empty=result.trade_table.copy()
    empty[s.SELECTED_TRADE]=False
    assert _swap_candidates(p,replace(result,trade_table=empty),2,True)==[]
    full=result.trade_table.copy()
    full[s.SELECTED_TRADE]=~full[s.SETTLEMENT_CASH]
    assert _swap_candidates(p,replace(result,trade_table=full),2,True)==[]


def test_pair_gains_against_independent_qis_quadratic_probes():
    """Three QIS scalar probes identify each one-dimensional optimum independently of the kernel."""
    from optimalportfolios.covar_estimation.risk_model_adapter import build_risk_model
    from optimalportfolios.execution.solver import _build_selected_execution_bound_series
    p=_problem(max_trades=1)
    result=b._solve(p,b.score_execution_candidates(p),rescue=True)
    risk=build_risk_model({pd.Timestamp('2000-01-01'):p.covariance})
    def variance(weights):
        """Use the stack's risk calculation, not covariance algebra, as the reference."""
        return float(risk.compute_tre_at_date(p.target[s.RAW_MODEL_WEIGHT],weights,
                                              pd.Timestamp('2000-01-01')))**2
    opened=result.trade_table.copy()
    opened[s.SELECTED_TRADE]=~opened[s.SETTLEMENT_CASH]
    lower,upper=_build_selected_execution_bound_series(p.constraints,opened,False)
    cash=opened.index[opened[s.SETTLEMENT_CASH]][0]
    initial=variance(result.weights)
    for gain,removed,added in _swap_candidates(p,result,10,True):
        weights=result.weights.copy()
        freed=weights.at[removed]-p.target.at[removed,s.BASE_WEIGHT]
        weights.at[removed]-=freed
        weights.at[cash]+=freed
        def at(move):
            """Evaluate a funded incoming displacement while preserving all other coordinates."""
            changed=weights.copy()
            changed.at[added]+=move
            changed.at[cash]-=move
            return variance(changed)
        lo,hi=lower.at[added]-weights.at[added],upper.at[added]-weights.at[added]
        grid=np.array([lo,(lo+hi)/2,hi])
        coefficients=np.polynomial.polynomial.polyfit(grid,[at(x) for x in grid],2)
        optimum=np.clip(-coefficients[1]/(2*coefficients[2]),lo,hi)
        assert gain==pytest.approx(initial-at(optimum),abs=1e-12)


def test_beam_keeps_two_provisional_paths_and_excludes_all_extra_work(monkeypatch):
    """The second-best first level can lead to the winning path under the same finite ceiling."""
    p,_,calls,_=_script(monkeypatch,[_values(tickets=2),_values(te=10.5,tickets=2),
        _values(te=10.8,tickets=2),_values(te=9.,tickets=2)],
        [[(2.,'A','C'),(1.,'B','C')],[],[(1.,'A','D')]],fourth=True)
    result=b.solve_branched_execution(p,b.ExecutionBranchConfig(methods=(),max_removal_trials=0,
        max_swap_trials=4,max_projection_calls=4,swap_depth=2,swap_beam_width=2))
    assert result.summary['final']['te_bp']==9.
    assert result.attempts.iloc[-1].parent_checkpoint=='checkpoint_00002'
    assert result.summary['swap_trials']==3 and len(calls)==4


def test_empty_frontier_and_per_level_limits_stop_without_extra_work(monkeypatch):
    """Candidate and level exhaustion stop exploration while keeping the eligible fallback."""
    p,baseline,calls,_=_script(monkeypatch,[_values(tickets=2)], [[]])
    result=b.solve_branched_execution(p,b.ExecutionBranchConfig(methods=(),max_removal_trials=0,
        max_swap_trials=4,max_projection_calls=4,swap_depth=2))
    assert result.result is baseline and result.summary['swap_trials']==0 and len(calls)==1
    monkeypatch.undo()
    p,_,calls,_=_script(monkeypatch,[_values(tickets=2),_values(te=9,tickets=2)],
        [[(2.,'A','C'),(1.,'B','C')]])
    result=b.solve_branched_execution(p,b.ExecutionBranchConfig(methods=(),max_removal_trials=0,
        max_swap_trials=1,max_projection_calls=1))
    assert result.summary['swap_trials']==1 and len(calls)==2


def test_turnover_row_preserves_original_policy_and_cash_exclusion():
    """Cap noncash L1 change while preserving existing signed linear constraints."""
    from optimalportfolios.execution._swaps import _turnover_capped_problem
    from optimalportfolios.optimization.constraints import LinearConstraints
    p=_problem(max_trades=2)
    table=b.score_execution_candidates(p)
    table[s.SELECTED_TRADE]=~table[s.SETTLEMENT_CASH]
    idx=table.index
    old=LinearConstraints(pd.Series(1.,index=idx).to_frame('__execution_swap_turnover'),
                          lower=pd.Series({'__execution_swap_turnover':.5}))
    p=replace(p,constraints=p.constraints.copy(linear_constraints=old))
    result=_turnover_capped_problem(p,table,{'gross_turnover_bp':100.},b.ExecutionBranchConfig())
    assert result.constraints.linear_constraints.loadings.columns.tolist()==[
        '__execution_swap_turnover','__execution_swap_turnover_']
    assert (result.constraints.linear_constraints.loadings.loc[
        idx[-1],'__execution_swap_turnover_']==0)
    pd.testing.assert_frame_equal(p.constraints.linear_constraints.loadings,old.loadings)
    assert _turnover_capped_problem(
        p,table,{},b.ExecutionBranchConfig(max_turnover_increase_bp=None)) is p


def test_straddling_turnover_envelope_upper_bounds_actual_l1():
    """Signed rows remain conservative when post-mandatory base and original holdings differ."""
    from optimalportfolios.execution._swaps import _turnover_capped_problem
    from optimalportfolios.optimization.constraints import LinearConstraints
    p=_problem(max_trades=2)
    table=b.score_execution_candidates(p)
    table[s.SELECTED_TRADE]=~table[s.SETTLEMENT_CASH]
    table.at['A',s.BASE_WEIGHT]=.1
    table.at['A',s.EFFECTIVE_MODEL_WEIGHT]=.3
    p.target.at['A',s.CURRENT_WEIGHT]=.2
    table.at['A',s.CURRENT_WEIGHT]=.2
    old=LinearConstraints(pd.Series(1.,index=table.index).to_frame('old'),
                          upper=pd.Series({'old':1.}))
    p=replace(p,constraints=p.constraints.copy(linear_constraints=old))
    capped=_turnover_capped_problem(p,table,{'gross_turnover_bp':2000.},b.ExecutionBranchConfig())
    row=capped.constraints.linear_constraints.loadings['__execution_swap_turnover']
    assert row.at['A']==pytest.approx(0.,abs=1e-10)
    assert capped.constraints.linear_constraints.upper.at['old']==1.
    assert capped.constraints.linear_constraints.upper.at['__execution_swap_turnover']<2000.


def test_new_turnover_row_uses_original_cash_free_cap():
    """An ordinary mandate gains a signed row without changing its constraint object."""
    from optimalportfolios.execution._swaps import _turnover_capped_problem
    p=_problem(max_trades=2)
    table=b.score_execution_candidates(p)
    assert p.constraints.linear_constraints is None
    capped=_turnover_capped_problem(p,table,{'gross_turnover_bp':100.},b.ExecutionBranchConfig())
    block=capped.constraints.linear_constraints
    assert block.lower is None and block.loadings.columns.tolist()==['__execution_swap_turnover']
    assert block.upper.iloc[0]<100.+1e4*float(p.target[s.CURRENT_WEIGHT].abs().sum())
    assert p.constraints.linear_constraints is None
