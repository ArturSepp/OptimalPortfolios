"""Research-only Kernel Search adaptation on the unchanged execution state model.

Framework: Guastaroba and Speranza (2012), doi:10.1016/j.ejor.2011.09.004;
general scheme specified in their 2014 paper, doi:10.1016/j.ejor.2014.04.007,
Section 2, Algorithm 1. This convex-quadratic adaptation uses fractional trade
occupancy instead of LP reduced costs, a kernel of at most max(24, 2B) names,
equally sized buckets and one pass. It is not their published index-tracking MILP.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import cvxpy as cp
import numpy as np

from optimalportfolios.execution.v2 import compile_states
from optimalportfolios.execution.v2.domain import variance
from optimalportfolios.execution.v2.exact import native_mosek
from optimalportfolios.execution.v2.model import build_model,SCALE
from .study_design import original_audit


def restricted_problem(problem,allowed):
    """Pin excluded optional names without changing target, risk or any hard row."""
    lower,upper=problem.lower.copy(),problem.upper.copy()
    for i in range(len(lower)):
        if i!=problem.cash and not problem.mandatory[i] and i not in allowed:
            lower[i]=upper[i]=problem.base[i]
    return replace(problem,lower=lower,upper=upper)


def search_partition(candidates,occupancies,budget,names):
    """Prioritise relaxed trade occupancy with stable name ties and disjoint buckets."""
    order=sorted(candidates,key=lambda i:(-occupancies.get(i,0.),names[i]))
    size=min(len(order),max(24,2*budget))
    return set(order[:size]),[order[i:i+size] for i in range(size,len(order),size)] if size else []


def kernel_search(problem,time_limit_s=60.):
    """Search restricted mixed-integer subproblems with charged guidance and starts."""
    if not np.isfinite(time_limit_s) or time_limit_s<=0:
        raise ValueError('Positive finite Kernel Search allowance required')
    started=perf_counter(); deadline=started+time_limit_s
    best=problem.base.copy() if original_audit(problem,problem.base)['passed'] else None
    value=variance(problem,best) if best is not None else np.inf
    diagnostics=dict(method='Kernel adaptation',bounds_scope='restricted_subproblems_only',
        configuration=dict(kernel_size='min(candidates,max(24,2B))',bucket_size='initial_kernel_size',
                           passes=1,selection='fractional_ticket_occupancy_then_name'),
        guidance={},subproblems=[],incumbent_events=[],first_audited_feasible_seconds=0. if best is not None else None)
    states=compile_states(problem)
    candidates=[i for i,row in enumerate(states) if i!=problem.cash and not problem.mandatory[i]
                and any(s.ticket for s in row)]
    budget=problem.rules.discretionary_allowance
    if budget is None:
        budget=len(candidates) if problem.rules.ticket_cap is None else max(0,problem.rules.ticket_cap-problem.mandatory_count)
    occupancies={}
    guidance_start=perf_counter()
    try:
        relaxation=build_model(problem,perspective=True,continuous=True)
        relaxation.optimisation.solve(solver='CLARABEL',time_limit=max(.001,deadline-perf_counter()),
            tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=200)
        diagnostics['guidance']['status']=relaxation.optimisation.status
        if relaxation.optimisation.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE):
            occupancies={i:float(relaxation.occupancy[i].value) for i in candidates
                         if relaxation.occupancy[i].value is not None}
    except (ValueError,RuntimeError,cp.SolverError) as error:
        diagnostics['guidance']['error']=repr(error)
    diagnostics['guidance']['seconds']=perf_counter()-guidance_start
    kernel,buckets=search_partition(candidates,occupancies,budget,problem.names)
    for stage,bucket in enumerate([[]]+buckets):
        if perf_counter()>=deadline:
            break
        available=kernel|set(bucket)
        restricted=restricted_problem(problem,available)
        model_start=perf_counter(); model=build_model(restricted)
        rows=list(model.optimisation.constraints)
        if bucket:
            rows.append(cp.sum(cp.hstack([model.occupancy[i] for i in bucket]))>=1)
        if np.isfinite(value):
            rows.append(model.optimisation.objective.expr<=SCALE*(value+1e-12))
        model.optimisation=cp.Problem(model.optimisation.objective,rows)
        record=dict(stage=stage,kernel=sorted(kernel),bucket=bucket,
                    construction_seconds=perf_counter()-model_start)
        stage_best=None; stage_value=np.inf
        def retain(weights,source):
            """Independently check original and restricted domains before retention."""
            nonlocal best,value,stage_best,stage_value
            audit=original_audit(problem,weights)
            local=original_audit(restricted,weights)
            bucket_used=not bucket or any(abs(weights[i]-problem.current[i])>problem.rules.zero_tolerance for i in bucket)
            event=dict(source=source,seconds=perf_counter()-started,audit=audit,
                       restricted_audit=local,bucket_used=bucket_used)
            if audit['passed'] and local['passed'] and bucket_used:
                candidate=variance(problem,weights)
                event['variance']=candidate
                if candidate<stage_value:
                    stage_best,stage_value=np.asarray(weights).copy(),candidate
                if candidate<value:
                    best,value=np.asarray(weights).copy(),candidate
                    event['improved']=True
                if diagnostics['first_audited_feasible_seconds'] is None:
                    diagnostics['first_audited_feasible_seconds']=perf_counter()-started
            diagnostics['incumbent_events'].append(event)
        remaining=deadline-perf_counter()
        stages_left=len(buckets)+1-stage
        sub_deadline=min(deadline,perf_counter()+max(.001,remaining/stages_left))
        initial=best if not bucket and best is not None and original_audit(restricted,best)['passed'] else None
        try:
            if model.selectors:
                record['native']=native_mosek(restricted,model,initial,sub_deadline,1,1e-4,retain)
            else:
                model.optimisation.solve(solver='CLARABEL',time_limit=max(.001,sub_deadline-perf_counter()),
                    tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10)
                record['native']={'native_status':model.optimisation.status}
                if model.weights.value is not None: retain(np.asarray(model.weights.value),'continuous_return')
        except (ValueError,RuntimeError,cp.SolverError) as error:
            record['error']=repr(error)
        if stage_best is not None:
            selected={i for i in candidates if abs(stage_best[i]-problem.current[i])>problem.rules.zero_tolerance}
            # Keep the original kernel plus selected bucket names. This one-pass
            # adaptation omits ageing/removal rather than adding untested tuning.
            kernel|=selected
        record['seconds']=perf_counter()-model_start
        diagnostics['subproblems'].append(record)
    diagnostics.update(seconds=perf_counter()-started,variance=value if best is not None else None,
                       completed_buckets=max(0,len(diagnostics['subproblems'])-1),total_buckets=len(buckets))
    return SimpleNamespace(weights=best,status='audited_incumbent' if best is not None else 'abstention',diagnostics=diagnostics)
