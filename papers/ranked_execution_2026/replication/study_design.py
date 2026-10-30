"""Sparse-holdings scaling inputs, separate from the frozen historical generator.

Universe size does not imply the same number of executable corridor candidates.
All thresholds retain the released study's economic meaning and audit units.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from optimalportfolios.execution.v2 import ExecutionProblem, ExecutionRules, compile_states
from .benchmark import ARRAYS, independent_audit, sha256
from .run_benchmark import write_json

STRUCTURED = ('factor_loadings', 'factor_covariance', 'residual_variance')


def generated_problem(case):
    """Build a feasible sparse book and a different supplied long-only target."""
    n, held = case['universe'], case.get('held', 96)
    mandatory_count = case.get('mandatory', 2)
    additions = case.get('additions', 100)
    if (not all(isinstance(x, int) and not isinstance(x, bool) for x in
                (n, held, mandatory_count, additions, case['budget'], case['seed']))
            or n < held+additions or not 2 <= mandatory_count < held
            or held*.006 >= .94 or additions < 1 or case['budget'] < 0):
        raise ValueError('Invalid sparse current-book design')
    if case['groups'] not in ('disjoint', 'overlap') or case['regime'] not in ('wide', 'tight'):
        raise ValueError('Unknown constraint design')
    if case['pattern'] not in ('diffuse', 'exit_rich'):
        raise ValueError('Unknown target design')
    retained = np.arange(mandatory_count, held, 2) if case['pattern']=='diffuse' else np.array([],int)
    selected = np.r_[retained, np.arange(held, held+additions)]
    if .005*len(selected) >= .98:
        raise ValueError('Target support incompatible with retained minimum trade')
    streams = np.random.SeedSequence([case['seed'], n, held, additions]).spawn(3)
    current_rng, target_rng, rng = (np.random.default_rng(s) for s in streams)
    raw = np.exp(current_rng.normal(0., .45, held))
    current = np.zeros(n+1)
    current[:held] = .006+(.94-.006*held)*raw/raw.sum()
    current[n] = .06
    base = current.copy()
    base[n] += base[:mandatory_count].sum()
    base[:mandatory_count] = 0.
    target = np.zeros(n+1)
    relative = np.exp(target_rng.normal(0., .6, len(selected)))
    target[selected] = .005+(.98-.005*len(selected))*relative/relative.sum()
    target[n] = .02
    sector = np.zeros((8,n+1))
    sector[np.arange(n)%8,np.arange(n)] = 1.
    baskets = np.c_[rng.uniform(.05,.95,(8,n)),np.zeros(8)]
    groups = sector if case['groups']=='disjoint' else np.vstack([sector,baskets])
    loadings = np.c_[rng.uniform(.04,.12,n+1), .10*sector.T, rng.normal(0.,.04,(n+1,3))]
    loadings[n] = 0.
    factor_covariance = np.eye(loadings.shape[1])
    residual = np.r_[rng.uniform(.04,.09,n)**2,.001**2]
    lower, upper = np.zeros(n+1), np.full(n+1,.35)
    upper[:mandatory_count] = 0.
    if case['regime']=='tight':
        lower[n],upper[n],width = base[n]-.0005,base[n]+.0005,.001
    else:
        lower[n],upper[n],width = .005,.5,.01
    lo,hi = np.minimum(base,target),np.minimum(np.maximum(base,target),upper)
    lo[n],hi[n] = lower[n],upper[n]
    exposure = groups@base
    mandatory = np.arange(n+1)<mandatory_count
    return ExecutionProblem(names=tuple([f'Asset{i:04}' for i in range(n)]+['Cash']),
        cash=n,current=current,base=base,target=target,lower=lo,upper=hi,
        full_lower=lower,full_upper=upper,groups=groups,
        group_lower=np.maximum(0.,exposure-width),group_upper=exposure+width,
        mandatory=mandatory,categories=('model_exit',)*mandatory_count+('',)*(n+1-mandatory_count),
        eligible=np.ones(n+1,bool),frozen=np.zeros(n+1,bool),
        factor_loadings=loadings,factor_covariance=factor_covariance,residual_variance=residual,
        rules=ExecutionRules(minimum_ticket=.005,holding_floor=.0025,
            ticket_cap=mandatory_count+case['budget'],discretionary_allowance=case['budget'],
            audit_tolerance=1e-7,zero_tolerance=1e-8),
        provenance={'benchmark':'sparse-scaling-1','seed':case['seed']})


def original_audit(problem, weights, domain='CORRIDOR'):
    """Reuse the independent original-weight checker with explicitly selected bounds."""
    if domain=='CORRIDOR':
        return independent_audit(problem,weights)
    if domain!='FULL':
        raise ValueError('Unknown audit domain')
    from dataclasses import replace
    return independent_audit(replace(problem,lower=problem.full_lower,upper=problem.full_upper),weights)


def identity(problem):
    """Hash all authoritative dense/structured inputs without serializing objects."""
    digest = hashlib.sha256()
    for name in ARRAYS+STRUCTURED:
        value = getattr(problem,name)
        digest.update(name.encode())
        if value is None:
            digest.update(b'null')
        else:
            a = np.asarray(value)
            digest.update(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes())
    rules = asdict(problem.rules)
    for key in ('holding_floor','exemption_mask'):
        rules[key] = np.asarray(rules[key]).tolist()
    digest.update(json.dumps(dict(names=problem.names,cash=problem.cash,
        categories=problem.categories,rules=rules,investment=problem.investment,
        annualisation=problem.annualisation),sort_keys=True,allow_nan=False).encode())
    return digest.hexdigest()


def structural_summary(problem):
    """Count actual executable choices and verify the supplied feasible witness."""
    states = compile_states(problem)
    candidates = [i for i,row in enumerate(states) if i!=problem.cash and not problem.mandatory[i]
                  and any(s.ticket for s in row)]
    base_audit = original_audit(problem,problem.base)
    if not base_audit['passed']:
        raise ValueError('Generated base lacks an independent feasibility witness')
    return dict(universe=len(problem.names)-1,current_holdings=int(np.sum(problem.current[:problem.cash]>0)),
        target_holdings=int(np.sum(problem.target[:problem.cash]>0)),
        mandatory=problem.mandatory_count,optional_candidates=len(candidates),
        optional_ticket_states=sum(sum(s.ticket for s in states[i]) for i in candidates),
        total_states=sum(len(s) for s in states),base_audit=base_audit,
        covariance_input_bytes=sum(getattr(problem,k).nbytes for k in ('covariance',)+STRUCTURED
                                   if getattr(problem,k) is not None),
        funded_dense_bytes=8*len(candidates)**2)


def save_problem(path, problem, description):
    """Save numeric arrays and explicit metadata, including structured covariance."""
    rules = asdict(problem.rules)
    arrays = {k:getattr(problem,k) for k in ARRAYS+STRUCTURED if getattr(problem,k) is not None}
    for key in ('holding_floor','exemption_mask'):
        arrays[key] = rules.pop(key)
    np.savez_compressed(path,**arrays)
    return dict(description,array_file=path.name,array_sha256=sha256(path),fingerprint=identity(problem),
        names=problem.names,cash=problem.cash,categories=problem.categories,rules=rules,
        investment=problem.investment,annualisation=problem.annualisation,
        structure=structural_summary(problem))


def load_problem(directory, row):
    """Refuse altered or escaping files and load only non-pickled numerical arrays."""
    root=Path(directory).resolve(); path=(root/row['array_file']).resolve()
    if not path.is_relative_to(root) or sha256(path)!=row['array_sha256']:
        raise ValueError('Unverified instance array')
    with np.load(path,allow_pickle=False) as data:
        arrays={key:data[key] for key in ARRAYS+STRUCTURED if key in data}
        rules=ExecutionRules(**row['rules'],holding_floor=data['holding_floor'],exemption_mask=data['exemption_mask'])
    p=ExecutionProblem(**arrays,names=tuple(row['names']),cash=row['cash'],categories=tuple(row['categories']),
        rules=rules,investment=row['investment'],annualisation=row['annualisation'])
    if identity(p)!=row['fingerprint']:
        raise ValueError('Frozen instance identity changed')
    return p


def freeze_cases(output, cases, specification, source_names):
    """Freeze an entire declared sample before solving; never overwrite an earlier run."""
    output=Path(output).resolve()
    if 'onedrive' in str(output).lower():
        raise ValueError('Research runs belong outside OneDrive')
    if len({r['id'] for r in cases})!=len(cases):
        raise ValueError('Duplicate case identity')
    output.mkdir(parents=True,exist_ok=False); (output/'inputs').mkdir()
    rows=[]
    for case in cases:
        if not case['id'] or Path(case['id']).name!=case['id'] or any(c in case['id'] for c in '/\\:'):
            raise ValueError('Unsafe case identity')
        rows.append(save_problem(output/'inputs'/(case['id']+'.npz'),generated_problem(case),case))
    from .matched import public_sources
    import importlib.metadata
    sources={name:sha256(Path(__file__).with_name(name)) for name in source_names}
    write_json(output/'manifest.json',dict(schema='sparse-scaling-1',specification=specification,
        cases=rows,sources=sources,baseline_sources=public_sources(),
        versions={name:importlib.metadata.version(name) for name in
            ('optimalportfolios','qis','numpy','pandas','scipy','cvxpy','Mosek','clarabel')}))
    return rows
