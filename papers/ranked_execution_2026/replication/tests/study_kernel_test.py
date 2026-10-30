"""Kernel adaptation respects the shared problem and declared candidate coverage."""
import numpy as np

from papers.ranked_execution_2026.replication.study_design import generated_problem,original_audit,identity
from papers.ranked_execution_2026.replication.study_kernel import restricted_problem,search_partition,kernel_search


def small_problem():
    """Supply a feasible small portfolio without any market-data dependency."""
    return generated_problem(dict(universe=16,held=8,additions=6,mandatory=2,budget=2,
        groups='overlap',regime='wide',pattern='exit_rich',seed=2026102955))


def test_restriction_preserves_every_hard_row_and_mandatory_action():
    """Only optional search eligibility changes in a restricted subproblem."""
    p=small_problem(); original=identity(p)
    q=restricted_problem(p,{2,8})
    for key in ('current','base','target','full_lower','full_upper','groups','group_lower','group_upper',
                'mandatory','factor_loadings','factor_covariance','residual_variance'):
        np.testing.assert_array_equal(getattr(p,key),getattr(q,key))
    assert q.lower[3]==q.upper[3]==p.base[3]
    assert q.lower[8]==p.lower[8] and q.upper[8]==p.upper[8]
    assert identity(p)==original


def test_partition_covers_candidates_once_without_outcome_tuning():
    """Stable priorities and all remaining buckets are fixed before restricted solves."""
    names=tuple(f'A{i:03}' for i in range(100))
    kernel,buckets=search_partition(list(range(100)),{50:.9,51:.8},10,names)
    assert 50 in kernel and 51 in kernel and len(kernel)==24
    flattened=[i for bucket in buckets for i in bucket]
    assert not kernel.intersection(flattened)
    assert len(flattened)==len(set(flattened))==76
    assert kernel|set(flattened)==set(range(100))


def test_complete_small_kernel_recovers_matched_exact_objective():
    """With every candidate included, the independent comparator solves the same MIQP."""
    from optimalportfolios.execution.v2 import wrapper_exact_execution
    from optimalportfolios.execution.v2.domain import variance
    p=small_problem(); before=identity(p)
    reference=wrapper_exact_execution(p,'CORRIDOR',time_limit_s=10.,threads=1,gap_rel=1e-4)
    actual=kernel_search(p,time_limit_s=10.)
    assert original_audit(p,actual.weights)['passed']
    assert abs(variance(p,actual.weights)-variance(p,reference.weights))<1e-10
    assert identity(p)==before
    assert actual.diagnostics['bounds_scope']=='restricted_subproblems_only'


def test_rejected_native_vectors_cannot_displace_the_base(monkeypatch):
    """A subproblem's status does not bypass the original independent audit."""
    from papers.ranked_execution_2026.replication import study_kernel as module
    p=small_problem()
    def invalid(problem,model,initial,deadline,threads,gap,retain):
        """Emulate a native solver delivering an invalid funded vector."""
        retain(p.base+.01,'native_test')
        return {'native_status':'test'}
    monkeypatch.setattr(module,'native_mosek',invalid)
    actual=kernel_search(p,10.)
    np.testing.assert_array_equal(actual.weights,p.base)
    assert not actual.diagnostics['incumbent_events'][0]['audit']['passed']
