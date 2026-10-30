"""Isolated public research workers with explicit process ceilings and failure records.

Process interruption is an experimental resource ceiling, not a modification of
the published Ranked algorithm. A killed worker supplies no delivered portfolio.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from optimalportfolios.execution.v2 import (
    prepare_common_execution,wrapper_exact_execution,wrapper_ranked_execution,RankedConfig,
)
from .benchmark import sha256,tracking_error_bp
from .run_benchmark import write_json
from .study_design import identity,load_problem,save_problem,original_audit


def peak_memory_bytes():
    """Read this worker's peak resident memory, including imports and input construction."""
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            """Windows PROCESS_MEMORY_COUNTERS layout."""
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
        query=ctypes.WinDLL('psapi').GetProcessMemoryInfo
        query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        query.restype=wintypes.BOOL
        if not query(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value*(1 if sys.platform=='darwin' else 1024))


def checked_manifest(root):
    """Reject any change to frozen study sources or released numerical sources."""
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['sources'].items():
        if sha256(Path(__file__).with_name(name))!=expected:
            raise ValueError('Frozen research source changed: '+name)
    from .matched import public_sources
    if public_sources()!=manifest['baseline_sources']:
        raise ValueError('Released source identity changed')
    return manifest


def worker(root,case_id,arm):
    """Prepare once or run one independent arm, always retaining original-unit audit."""
    manifest=checked_manifest(root); spec=manifest['specification']
    row=next(r for r in manifest['cases'] if r['id']==case_id)
    destination=root/f'{case_id}--{arm}.json'
    tick=perf_counter()
    if arm=='prepare':
        raw=load_problem(root/'inputs',row)
        raw_identity=identity(raw)
        start=perf_counter()
        try:
            p,diagnostics=prepare_common_execution(raw,time_limit_s=spec['preparation_seconds'],solver='MOSEK')
            elapsed=perf_counter()-start
            if identity(raw)!=raw_identity:
                raise ValueError('Preparation mutated original inputs')
            record=dict(status='prepared' if p is not None else 'failed',diagnostics=diagnostics,
                        preparation_seconds=elapsed,prepared=None)
            if p is not None:
                record['prepared']=save_problem(root/'prepared'/(case_id+'.npz'),p,{'id':case_id})
        except Exception as error:
            record=dict(status='exception',error=repr(error),preparation_seconds=perf_counter()-start,prepared=None)
    else:
        preparation=json.loads((root/f'{case_id}--prepare.json').read_text())
        if preparation.get('prepared') is None:
            record=dict(status='preparation_failed',audit={'passed':False},weights=None)
        else:
            p=load_problem(root/'prepared',preparation['prepared'])
            control=spec['arms'][arm]
            transform=perf_counter()
            if control.get('representation')=='dense':
                p=replace(p,covariance=p.dense_covariance(),factor_loadings=None,
                          factor_covariance=None,residual_variance=None)
            representation_seconds=perf_counter()-transform
            before=identity(p); start=perf_counter()
            try:
                if control['engine']=='Ranked':
                    result=wrapper_ranked_execution(p,RankedConfig())
                elif control['engine']=='Exact':
                    result=wrapper_exact_execution(p,control.get('domain','CORRIDOR'),
                        time_limit_s=control['seconds'],threads=1,gap_rel=1e-4)
                elif control['engine']=='Kernel':
                    from .study_kernel import kernel_search
                    result=kernel_search(p,time_limit_s=control['seconds'])
                else:
                    raise ValueError('Unknown engine')
                seconds=perf_counter()-start
                if identity(p)!=before:
                    raise ValueError('Solver mutated a frozen problem')
                audit=original_audit(p,result.weights,control.get('domain','CORRIDOR'))
                record=dict(status=result.status,audit=audit,weights=result.weights,
                    diagnostics=result.diagnostics,seconds=seconds,
                    te_bp=tracking_error_bp(p,result.weights) if audit['passed'] else None)
            except Exception as error:
                record=dict(status='exception',error=repr(error),audit={'passed':False},
                            weights=None,seconds=perf_counter()-start)
            record.update(representation_seconds=representation_seconds,
                preparation_seconds=preparation['preparation_seconds'],
                end_to_end_seconds=record['seconds']+representation_seconds+preparation['preparation_seconds'])
    record.update(case=case_id,arm=arm,worker_seconds=perf_counter()-tick,
                  peak_process_memory_bytes=peak_memory_bytes())
    write_json(destination,record)


def isolated_call(root,case_id,arm,ceiling):
    """Stop only the created worker at a declared ceiling, retaining missing deliveries."""
    path=root/f'{case_id}--{arm}.json'
    if path.exists():
        return json.loads(path.read_text())
    command=[sys.executable,'-m','papers.ranked_execution_2026.replication.study_run',
             'worker','--root',str(root),'--case',case_id,'--arm',arm]
    started=perf_counter()
    with (root/'logs'/f'{case_id}--{arm}.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        try:
            code=process.wait(timeout=ceiling)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
            record=dict(case=case_id,arm=arm,status='process_ceiling',weights=None,
                audit={'passed':False,'violations':['no_returned_delivery']},
                process_ceiling_seconds=ceiling,process_seconds=perf_counter()-started)
            # A result written exactly at the boundary is retained explicitly.
            if path.exists(): record['boundary_result']=json.loads(path.read_text())
            write_json(path,record)
            return record
    if not path.exists():
        write_json(path,dict(case=case_id,arm=arm,status='worker_failure',returncode=code,
            weights=None,audit={'passed':False,'violations':['no_returned_delivery']}))
    record=json.loads(path.read_text()); record['process_seconds']=perf_counter()-started
    record['worker_returncode']=code; write_json(path,record)
    return record


def run(root):
    """Run every frozen case serially, with rotating independent-arm order."""
    manifest=checked_manifest(root); spec=manifest['specification']
    for folder in ('logs','prepared'): (root/folder).mkdir(exist_ok=True)
    for index,row in enumerate(manifest['cases']):
        case_id=row['id']
        prepared=isolated_call(root,case_id,'prepare',spec['preparation_ceiling_seconds'])
        arms=row.get('arms',list(spec['arms']))
        offset=index%len(arms); order=arms[offset:]+arms[:offset]
        for arm in order:
            destination=root/f'{case_id}--{arm}.json'
            if prepared.get('prepared') is None:
                if not destination.exists():
                    write_json(destination,dict(case=case_id,arm=arm,status='preparation_failed',
                        weights=None,audit={'passed':False,'violations':['preparation_failed']}))
                record=json.loads(destination.read_text())
            else:
                record=isolated_call(root,case_id,arm,spec['arms'][arm]['process_ceiling_seconds'])
            print(case_id,arm,record['status'],flush=True)
        write_json(root/'progress.json',dict(completed=index+1,total=len(manifest['cases']),last=case_id))
    write_json(root/'complete.json',dict(cases=len(manifest['cases']),full_sample=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','worker'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--case'); parser.add_argument('--arm')
    args=parser.parse_args()
    if args.action=='run': run(args.root)
    else: worker(args.root,args.case,args.arm)
