"""Offline controls for the new generated domains, separate starts and references."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from optimalportfolios.execution.v2 import ExecutionProblem, ExecutionRules, compute_perspective_lower_bound
from papers.ranked_execution_2026.replication import breadth
from papers.ranked_execution_2026.replication.benchmark import independent_audit


def specification():
    """Read the predeclared experiment controls."""
    return json.loads(breadth.PROTOCOL.read_text())


def paired_problem():
    """Create a cash-pinned two-ticket optimum, independent of the random generator."""
    current, target = np.array([.3,.3,.4]), np.array([.4,.2,.4])
    return ExecutionProblem(names=('Asset001','Asset002','Cash'), cash=2,
        current=current, base=current, target=target,
        lower=np.minimum(current,target), upper=np.maximum(current,target),
        full_lower=np.zeros(3), full_upper=np.ones(3), groups=np.array([[0.,0.,1.]]),
        group_lower=np.array([.4]), group_upper=np.array([.4]), mandatory=np.zeros(3,bool),
        categories=('','',''), eligible=np.ones(3,bool), frozen=np.zeros(3,bool), covariance=np.eye(3)*.01,
        rules=ExecutionRules(minimum_ticket=.005, holding_floor=.0025,
                             ticket_cap=2, discretionary_allowance=2))


@pytest.mark.parametrize('n,regime', [(24,'tight'), (72,'wide')])
def test_generator_is_repeatable_and_exposure_coupling_is_isolated(n, regime):
    """A new seed supplies funded, PSD inputs; extra group rows change no risk arrays."""
    case=dict(optional_assets=n,budget=2,groups='disjoint',regime=regime,seed=90001)
    p=breadth.generated_case(case,specification())
    other=breadth.generated_case(dict(case,groups='overlap'),specification())
    assert independent_audit(p,p.base)['passed']
    assert independent_audit(other,other.base)['passed']
    assert breadth.fingerprint(p)==breadth.fingerprint(breadth.generated_case(case,specification()))
    assert breadth.fingerprint(p)!=breadth.fingerprint(other)
    assert p.groups.shape==(3,n+3) and other.groups.shape==(6,n+3)
    for field in ('current','base','target','lower','upper','covariance'):
        np.testing.assert_array_equal(getattr(p,field),getattr(other,field))
    assert np.linalg.eigvalsh(p.covariance).min()>0
    assert np.ptp(p.current[:-1])>0
    assert p.rules.audit_tolerance==1e-7 and p.rules.zero_tolerance==1e-8


def test_grid_declares_all_cases_without_selection():
    """All factorial cells and the separate enumeration subset are retained."""
    cases=breadth.case_grid(specification())
    assert len(cases)==108 and len({c['id'] for c in cases})==108
    assert sum(c['sample']=='broad' for c in cases)==96
    assert {c['seed'] for c in cases}=={
        71001,71002,72001,72002,81001,81002,82001,82002,91001,91002,92001,92002}


def test_search_work_distinguishes_guidance_missing_records_and_tiny_controls():
    """Work counters must preserve missing-case denominators and exclude guidance."""
    from papers.ranked_execution_2026.replication.summarize_combinations import search_work
    diagnostics = dict(qp_counts=dict(sizing=2, restart=5, guidance=1,
        restart_guidance=2, conditional_guidance=3),
        bridge_restarts=[dict(improved=False), dict(improved=True)],
        bridge_completion_passes=[dict(trials=3)])
    result = search_work([
        dict(sample='broad', methods=dict(Ranked=dict(diagnostics=diagnostics))),
        dict(sample='broad', methods={}),
        dict(sample='tiny', methods=dict(Ranked=dict(diagnostics=diagnostics)))])
    assert result['Ranked']['expected'] == 2 and result['Ranked']['recorded'] == 1
    assert result['Ranked']['sizing_attempts']['median'] == 7
    assert result['Ranked']['bridge_improved_cases'] == 1
    assert result['Ranked']['bridge_pair_trials']['median'] == 3
    assert result['OneAnchor']['recorded'] == 0
    assert result['OneAnchor']['sizing_attempts']['median'] is None


@pytest.mark.parametrize('index', [0,1])
def test_separate_starts_and_time_controls_do_not_pool_incumbents(monkeypatch,index):
    """The two heuristic starts are deliberately different and must reach their own calls."""
    p=paired_problem()
    calls=[]
    def ranked(problem,config):
        """Return two distinguishable, feasible incumbents."""
        weights=p.target if config.improvement_passes==3 else p.base
        return SimpleNamespace(weights=weights,status='audited',diagnostics={'config':asdict(config)})
    def exact(problem,formulation,**kwargs):
        """Record the actual supplied weights and unchanged solver controls."""
        calls.append(kwargs)
        return SimpleNamespace(weights=problem.base if kwargs['mip_start'] is None else kwargs['mip_start'],
                               status='audited',diagnostics={})
    monkeypatch.setattr(breadth,'wrapper_ranked_execution',ranked)
    monkeypatch.setattr(breadth,'wrapper_exact_execution',exact)
    results,order=breadth.serial_arms(p,specification(),index,preparation_seconds=.1)
    assert {r['time_limit_s'] for r in calls}=={10.,60.}
    assert all(r['threads']==1 and r['gap_rel']==1e-4 for r in calls)
    assert sum(r['mip_start'] is None for r in calls)==2
    supplied=[r['mip_start'] for r in calls if r['mip_start'] is not None]
    assert any(w is p.target for w in supplied) and any(w is p.base for w in supplied)
    for method,source in [('ExactFull60','Ranked'),('ExactOne60','OneAnchor')]:
        assert results[method]['start_source']==source
        assert results[method]['start_generation_seconds']==results[source]['seconds']
        assert results[method]['end_to_end_seconds']==pytest.approx(
            results[method]['seconds']+.1+results[source]['seconds'])
    assert order[0]==('Ranked' if index==0 else 'OneAnchor')


def test_exhaustive_reference_keeps_the_full_budget_and_original_audit():
    """Omitting the two-ticket support would miss an independently known zero-TE optimum."""
    p=paired_problem()
    reference=breadth.enumerate_tiny(p)
    assert reference['audit']['passed'] and reference['te_bp']<.0001
    assert reference['complete'] and len(reference['trials'])==4
    np.testing.assert_allclose(reference['weights'],p.target,atol=1e-7)
    broken=replace(p,group_lower=np.array([0.]),group_upper=np.array([.4-2e-7]))
    assert not independent_audit(broken,p.base)['passed']
    bounds=compute_perspective_lower_bound(p)
    assert all(b['lower_bound'] <= (reference['te_bp']/1e4)**2+1e-12 for b in bounds.values())


def test_enumeration_refuses_an_unsupported_domain():
    """Positive-target interval enumeration must never claim to cover optional exits."""
    p=paired_problem()
    with pytest.raises(ValueError,match='positive-target'):
        breadth.enumerate_tiny(replace(p,target=np.array([0.,.6,.4])))


def test_diagnosis_distinguishes_unvisited_support_from_failed_sizing(monkeypatch):
    """A known funded pair is feasible even when neither heuristic visited it."""
    from papers.ranked_execution_2026.replication.summarize_breadth import diagnose
    from papers.ranked_execution_2026.replication.benchmark import tracking_error_bp
    p=paired_problem()
    monkeypatch.setattr(breadth,'generated_case',lambda case,spec:p)
    heuristic=dict(weights=p.base,te_bp=tracking_error_bp(p,p.base),diagnostics={'trials':[]})
    case=dict(id='analytic',optional_assets=2,budget=2,groups='overlap',regime='tight',seed=0,
        prepared_fingerprint=breadth.fingerprint(p),methods=dict(
            Ranked=heuristic,OneAnchor=heuristic,ExactBase60=dict(weights=p.target,te_bp=0.)))
    result=diagnose(case,specification())
    assert result['conditional_audit']['passed']
    # The known optimum is zero variance; native QP precision is in the objective,
    # not the square-root TE reporting band used for the nonzero benchmark cases.
    assert (result['conditional_te_bp']/1e4)**2 <= specification()['objective_tolerance']
    assert all(not h['visited'] and h['added']==[0,1] and h['removed']==[]
               for h in result['heuristics'].values())
    case['prepared_fingerprint']='changed input'
    with pytest.raises(ValueError,match='identical prepared input'):
        diagnose(case,specification())


def test_failed_complete_study_keeps_denominators_and_rejects_changed_bytes(tmp_path):
    """Neither failure nor a changed input can silently disappear from headline results."""
    from papers.ranked_execution_2026.replication.summarize_breadth import summarize
    from papers.ranked_execution_2026.replication.benchmark import sha256
    inputs=tmp_path/'inputs'
    inputs.mkdir()
    file=inputs/'breadth_000.npz'
    file.write_bytes(b'fixed test input bytes')
    case=dict(id='breadth_000',sample='broad',input_sha256=sha256(file))
    (tmp_path/'protocol.json').write_text(json.dumps(dict(cases=[case],specification=specification(),versions={})))
    (tmp_path/'complete.json').write_text(json.dumps(dict(cases=1,full_sample=True)))
    (tmp_path/'breadth_000.json').write_text(json.dumps(dict(case,methods={})))
    result=summarize(tmp_path,tmp_path/'summary')
    assert result['broad']['cases']==1 and result['broad']['prepared']==0
    assert result['broad']['methods']['Ranked']['expected']==1
    assert not result['simplification_gate']['passed']
    assert 'failed' in (tmp_path/'summary/breadth.tex').read_text()
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed frozen'):
        summarize(tmp_path,tmp_path/'summary')


@pytest.mark.parametrize('fault', [None,'inputs','source','controls','seeds'])
def test_corrected_holdout_requires_frozen_source_controls_and_disjoint_seeds(fault):
    """A before/after or fresh-holdout claim must fail when its control is changed."""
    import copy
    from papers.ranked_execution_2026.replication.summarize_combinations import validate_designs
    case=dict(id='case',fingerprint='fixed',sample='broad',optional_assets=24,
              budget=2,groups='overlap',regime='tight',seed=71001)
    current=dict(cases=[case],specification=specification(),
        source_sha256={'ranked.py':'frozen'},runner_sha256='runner',audit_sha256='audit',
        ranked_config={'improvement_passes':3},versions={'qis':'same'})
    previous=copy.deepcopy(current)
    fresh=copy.deepcopy(current)
    fresh['specification']['grid']['seed']=[101001,101002]
    fresh['specification']['tiny']['seed']=[102001,102002]
    if fault=='inputs':
        previous['cases'][0]['fingerprint']='changed'
    elif fault=='source':
        fresh['source_sha256']['ranked.py']='retuned'
    elif fault=='controls':
        fresh['specification']['audit_tolerance']=1e-6
    elif fault=='seeds':
        fresh['specification']['grid']['seed']=[71001,71002]
    if fault is None:
        validate_designs(current,previous,fresh)
    else:
        with pytest.raises(ValueError):
            validate_designs(current,previous,fresh)


@pytest.mark.parametrize('fault', [None,'omitted','duplicate','renumbered_input','runtime'])
def test_merged_cohorts_match_inputs_independently_of_ordinal_filenames(fault):
    """Merging earlier cohorts must retain every input and unchanged runtime controls."""
    import copy
    from papers.ranked_execution_2026.replication.summarize_combinations import validate_designs
    current=dict(cases=[dict(c,fingerprint=str(c['seed'])+'-'+c['id'])
                        for c in breadth.case_grid(specification())],specification=specification(),
        source_sha256={'ranked.py':'new'},runner_sha256='runner',audit_sha256='audit',
        ranked_config={'improvement_passes':3},versions={'qis':'same'})
    previous=[]
    for seeds,tiny in [([71001,71002,81001,81002],[72001,72002,82001,82002]),
                       ([91001,91002],[92001,92002])]:
        p=copy.deepcopy(current)
        p['specification']['grid']['seed']=seeds
        p['specification']['tiny']['seed']=tiny
        p['cases']=[dict(c,id=f'old_{i:03}') for i,c in enumerate(current['cases'])
                    if c['seed'] in seeds+tiny]
        p['source_sha256']={'ranked.py':'old'}
        previous.append(p)
    fresh=copy.deepcopy(current)
    fresh['specification']['grid']['seed']=[101001,101002]
    fresh['specification']['tiny']['seed']=[102001,102002]
    if fault=='omitted':
        previous[1]['cases'].pop()
    elif fault=='duplicate':
        previous.append(copy.deepcopy(previous[0]))
    elif fault=='renumbered_input':
        previous[0]['cases'][0]['fingerprint']='different bytes'
    elif fault=='runtime':
        previous[1]['versions']['qis']='changed'
    if fault is None:
        validate_designs(current,previous,fresh)
    else:
        with pytest.raises(ValueError):
            validate_designs(current,previous,fresh)
