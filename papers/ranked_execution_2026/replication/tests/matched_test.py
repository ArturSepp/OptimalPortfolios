"""Protect released-cohort controls, input isolation and primary/fallback accounting."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from papers.ranked_execution_2026.replication import matched
from papers.ranked_execution_2026.replication.breadth import case_grid, generated_case, fingerprint


def specifications():
    """Load frozen public designs without reading an empirical or private input."""
    spec = json.loads(matched.PROTOCOL.read_text())
    legacy = json.loads(matched.PROTOCOL.with_name('joint_protocol.json').read_text())
    return spec, legacy


def tiny():
    """Return an independently generated small public instance."""
    _, legacy = specifications()
    return generated_case(case_grid(legacy)[-1], legacy)


def test_holdout_is_fresh_and_exit_targets_preserve_hard_constraints():
    """Changing targets must not change funding, hard bounds, groups or audit rules."""
    spec, legacy = specifications()
    design, cases = matched.holdout_grid(spec, legacy)
    assert len(cases) == 68 and sum(c['sample'] == 'tiny' for c in cases) == 4
    assert not set(design['grid']['seed']) & set(legacy['grid']['seed'])
    for case in cases:
        base = generated_case(case, design)
        changed = matched.holdout_case(case, design)
        for name in ('current','base','full_lower','full_upper','groups','group_lower',
                     'group_upper','mandatory','eligible','frozen','covariance'):
            np.testing.assert_array_equal(getattr(base,name),getattr(changed,name))
        assert asdict(changed.rules) == asdict(base.rules)
        assert changed.target.sum() == pytest.approx(base.target.sum())
        assert changed.rules.audit_tolerance == 1e-7 and changed.rules.zero_tolerance == 1e-8
        assert changed.rules.minimum_ticket == .005 and changed.rules.holding_floor == .0025
        if case['target_pattern'] == 'exit_rich':
            assert np.sum(changed.target[2:changed.cash] == 0) == case['optional_assets']//2
        else:
            assert fingerprint(changed) == fingerprint(base)


@pytest.mark.parametrize('index', [0,1,2])
def test_independent_starts_and_costs(monkeypatch, index):
    """Policy and independent MIQP receive no V2 start and every arm pays preparation."""
    spec, _ = specifications()
    problem = tiny()
    ticks = iter(range(100))
    monkeypatch.setattr(matched,'perf_counter',lambda: next(ticks))
    calls = []
    def ranked(p, config):
        """Return the feasible base without a solver dependency."""
        return SimpleNamespace(status='test',weights=p.base.copy(),diagnostics={})
    def exact(p, domain, **kwargs):
        """Mutate only the supplied copy to expose cross-arm aliasing."""
        start = kwargs['mip_start']
        calls.append(None if start is None else start.copy())
        if start is not None:
            start[:] = 0.
        return ranked(p,None)
    monkeypatch.setattr(matched,'wrapper_ranked_execution',ranked)
    monkeypatch.setattr(matched,'wrapper_exact_execution',exact)
    monkeypatch.setattr(matched,'wrapper_matched_execution',ranked)
    records, order = matched.serial_arms(problem,spec,'Legacy',index,3.)
    assert set(records) == set(matched.method_names(spec,'Legacy'))
    assert len(order) == 8
    assert sum(c is None for c in calls) == 3
    for name,item in records.items():
        assert item['end_to_end_seconds'] == (5. if name.startswith('MIQPStartV2') else 4.)
    np.testing.assert_array_equal(records['V2']['weights'],problem.base)


def test_failure_is_retained_without_a_contaminated_start(monkeypatch):
    """A heuristic failure stays visible and cannot be used as an incumbent."""
    spec, _ = specifications()
    def fail(p, config):
        """Represent a failed heuristic call."""
        raise RuntimeError('deliberate')
    def exact(p, *args, **kwargs):
        """Require the common start after the failed V2 arm."""
        assert kwargs['mip_start'] is None
        return SimpleNamespace(status='test',weights=p.base.copy(),diagnostics={})
    monkeypatch.setattr(matched,'wrapper_ranked_execution',fail)
    monkeypatch.setattr(matched,'wrapper_exact_execution',exact)
    monkeypatch.setattr(matched,'wrapper_matched_execution',lambda p,c: exact(p,mip_start=None))
    records,_ = matched.serial_arms(tiny(),spec,'Holdout',0,0.)
    assert len(records) == 4 and not records['V2']['audit']['passed']
    assert records['MIQPStartV2']['supplied_start'] == 'common_base'
    assert records['MIQPStartV2']['start_generation_seconds'] > 0.


def test_changed_shared_domain_aborts(monkeypatch):
    """An arm must not leave subsequent comparators with different inputs."""
    spec, _ = specifications()
    def mutate(p, config):
        """Emulate an invalid in-place change to the common problem."""
        object.__setattr__(p,'target',p.target+.01)
        return SimpleNamespace(status='test',weights=p.base.copy(),diagnostics={})
    monkeypatch.setattr(matched,'wrapper_ranked_execution',mutate)
    with pytest.raises(ValueError,match='mutated'):
        matched.serial_arms(tiny(),spec,'Holdout',0,0.)


@pytest.mark.parametrize('fault', ['native_error','invalid_weights','inconsistent_bound'])
def test_real_policy_rejects_primary_and_audits_fallback(monkeypatch, fault):
    """Exercise the published wrapper with failed primary and independently checked V2."""
    import optimalportfolios.execution.v2.primary as primary
    p = tiny()
    monkeypatch.setattr(primary,'wrapper_exact_execution',lambda p,*a,**k: matched.injected_primary(p,fault))
    monkeypatch.setattr(primary,'wrapper_ranked_execution',lambda p,c: SimpleNamespace(
        status='test',weights=p.base.copy(),diagnostics={}))
    delivered = primary.wrapper_matched_execution(p,matched.policy_config())
    assert delivered.diagnostics['engine'] == 'v2_audited_fallback'
    assert matched.independent_audit(p,delivered.weights)['passed']
    monkeypatch.setattr(primary,'wrapper_ranked_execution',lambda p,c: SimpleNamespace(
        status='test',weights=p.base+.01,diagnostics={}))
    with pytest.raises(RuntimeError,match='fallback rejected'):
        primary.wrapper_matched_execution(p,matched.policy_config())


def test_summary_keeps_failed_cases_and_bound_contradictions():
    """Failing methods and preparation cannot shrink the reported denominator."""
    from papers.ranked_execution_2026.replication.summarize_matched import cohort_summary
    spec,_ = specifications()
    good = dict(audit={'passed':True},te_bp=10.,weights=[.5,.5],
                seconds=1.,end_to_end_seconds=2.,diagnostics={})
    row = dict(id='a',cohort='Holdout',sample='broad',optional_candidates=12,
               preparation_seconds=1.,methods={n:dict(good) for n in spec['methods']})
    row['methods']['MIQP'] = dict(good,te_bp=20.,diagnostics={
        'native_bound':4e-6,'delivered_source':'native_callback','native_gap_relative':.01})
    row['methods']['V2'] = dict(good,audit={'passed':False})
    failed = dict(id='b',cohort='Holdout',sample='broad',methods={},
                  preparation_seconds=1.,preparation={'status':'infeasible'})
    summary = cohort_summary([row,failed],spec)
    assert summary['expected'] == 2 and summary['prepared'] == 1
    assert summary['methods']['V2']['audited'] == 0
    assert summary['methods']['V2']['end_to_end_seconds']['n'] == 1
    assert summary['paired']['V2_minus_MIQP']['n'] == 0
    assert len(summary['bound_contradictions']) == 1
    assert len(summary['failed_cases']) == 2
    assert summary['methods']['MIQP']['delivered_sources'] == {'native_callback':1}
    assert summary['methods']['MIQP']['native_gap_above_target_cases'] == ['a']


def test_summary_loader_refuses_smoke_run(tmp_path):
    """A partially completed experiment cannot generate manuscript tables."""
    from papers.ranked_execution_2026.replication.summarize_matched import load_complete
    (tmp_path/'protocol.json').write_text(json.dumps({'cases':[{'id':'a'}]}))
    (tmp_path/'complete.json').write_text(json.dumps({'full_sample':False,'cases':1}))
    with pytest.raises(ValueError,match='complete'):
        load_complete(tmp_path)
