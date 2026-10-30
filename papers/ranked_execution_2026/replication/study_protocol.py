"""Freeze a prospective declared sample before running any evaluation solver."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import zipfile

import optimalportfolios

from .benchmark import sha256
from .matched import public_sources
from .run_benchmark import write_json
from .study_design import save_problem
from .study_inputs import make_problem

PROTOCOL=Path(__file__).with_suffix('.json')
SOURCES=('study_design.py','study_run.py','study_kernel.py','study_inputs.py',
         'study_protocol.py','study_protocol.json')


def cases(spec):
    """Return all cases and supplementary arms without observing their solutions."""
    result=[]; scaling=spec['scaling']; extra=scaling['additional_arms_at']
    for n in scaling['universes']:
        for budget in scaling['budgets']:
            for profile in scaling['profiles']:
                arms=list(spec['main_arms'])
                if profile['profile']==extra['profile'] and budget==extra['budget']:
                    arms+=extra['arms']
                result.append(dict(profile,id=f'scale_{n}_{budget}_{profile["profile"]}',
                    family='scaling',universe=n,held=scaling['held'],additions=scaling['additions'],
                    budget=budget,arms=arms))
    external=spec['external']
    for source in external['sources']:
        for observations in external['observations']:
            result.append(dict(source,id=f'or_{source["universe"]}_{observations}',family='external',
                observations=observations,seed=external['seed'],groups='overlap',regime='wide',
                pattern='diffuse',mandatory=2,arms=list(spec['main_arms'])))
    result.append(dict(id='analytic_corridor',family='counterexample',
        arms=['Exact60','Ranked','Kernel60','Full60']))
    return result


def freeze(output,references,wheel,protocol=PROTOCOL):
    """Verify the released installation and freeze source, inputs and all denominators."""
    spec=json.loads(Path(protocol).read_text())
    if importlib.metadata.version('optimalportfolios')!=spec['package_version'] or sha256(wheel)!=spec['wheel_sha256']:
        raise ValueError('Released package identity mismatch')
    package=Path(optimalportfolios.__file__).resolve().parent
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('optimalportfolios/') and name.endswith('.py'):
                actual=package/Path(name).relative_to('optimalportfolios')
                if hashlib.sha256(archive.read(name)).hexdigest()!=sha256(actual):
                    raise ValueError('Installed source differs from released wheel: '+name)
    output=Path(output).resolve()
    if 'onedrive' in str(output).lower(): raise ValueError('Use C-local research output')
    declared=cases(spec)
    if len({r['id'] for r in declared})!=len(declared): raise ValueError('Duplicate case')
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('inputs','frozen_sources'): (output/folder).mkdir()
    sources={name:sha256(Path(__file__).with_name(name)) for name in SOURCES}
    for name in SOURCES: shutil.copy2(Path(__file__).with_name(name),output/'frozen_sources'/name)
    rows=[]
    for case in declared:
        problem=make_problem(case,references)
        rows.append(save_problem(output/'inputs'/(case['id']+'.npz'),problem,case))
    write_json(output/'manifest.json',dict(schema='execution-research-1',
        frozen_at=datetime.now(timezone.utc).isoformat(),specification=spec,cases=rows,sources=sources,
        baseline_sources=public_sources(),planned_deliveries=sum(len(r['arms']) for r in rows),
        machine=dict(system=platform.system(),release=platform.release(),processor=platform.processor()),
        versions={name:importlib.metadata.version(name) for name in
            ('optimalportfolios','qis','numpy','pandas','scipy','cvxpy','Mosek','clarabel')}))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--references',required=True,type=Path)
    parser.add_argument('--wheel',required=True,type=Path)
    arguments=parser.parse_args()
    freeze(arguments.output,arguments.references,arguments.wheel)
