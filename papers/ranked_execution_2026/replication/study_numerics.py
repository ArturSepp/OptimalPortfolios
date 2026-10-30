"""Isolated algebraic controls and retained public numerical defect reproductions.

These are declared mechanism examples, not new holdout observations. Research
variants change only equivalent graph representation in this process. Installed
package files, production controls and the prospective comparison are untouched.
"""
import argparse
from dataclasses import replace
import importlib.metadata
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import cvxpy as cp
import numpy as np

from optimalportfolios.execution.v2 import model as formulation
from optimalportfolios.execution.v2.domain import variance
from optimalportfolios.execution.v2.exact import native_mosek
from optimalportfolios.execution.v2.tests.fixed_rows_test import one_trade
from optimalportfolios.execution.v2.tests.formulation_test import independent_kkt
from optimalportfolios.execution.v2.tests.group_audit_test import pinned_group
from optimalportfolios.execution.v2.tests.native_accuracy_test import (
    public_counterexample, independent_witness,
)
from .benchmark import tracking_error_bp,sha256
from .matched import public_sources
from .run_benchmark import write_json
from .study_design import original_audit,identity


def origin_example():
    """Demonstrate a changed stopping decision for the same translated bound pair."""
    incumbent,bound,constant=1e-10,9e-11,1e-3
    absolute=abs(incumbent-bound)
    correct=absolute/max(1e-14,abs(incumbent))
    shifted=absolute/max(1e-14,abs(incumbent-constant))
    return dict(kind='analytical_bound_pair_not_solver_output',incumbent=incumbent,
        bound=bound,removed_constant=constant,absolute_gap=absolute,
        original_relative_gap=correct,translated_relative_gap=shifted,
        absolute_target=1e-12,relative_target=1e-4,
        original_stops=absolute<=1e-12 or correct<=1e-4,
        translated_stops=absolute<=1e-12 or shifted<=1e-4)


def cone_model(problem,balanced):
    """Use the same integer rows with algebraically equivalent quadratic coordinates."""
    model=formulation.build_model(problem,perspective=False)
    x=model.displacement_bp
    # QIS does not expose a CVXPY objective-coordinate compiler. This control
    # reconstructs the same graph expression to isolate conic representation.
    quadratic=(cp.quad_form(x/100.,cp.psd_wrap(model.funded_covariance)) if balanced
               else cp.quad_form(x,cp.psd_wrap(model.funded_covariance))/formulation.SCALE)
    expression=formulation.SCALE*model.constant+2*model.gradient@x+quadratic
    model.optimisation=cp.Problem(cp.Minimize(expression),model.linear_constraints)
    return model


def run_cone(optional,balanced,seconds=10.):
    """Retain all audited native candidates and unmodified raw native diagnostics."""
    problem=public_counterexample(optional)
    started=perf_counter(); model=cone_model(problem,balanced)
    best=problem.base.copy(); value=variance(problem,best); events=[]
    best_source='prepared_base'
    def retain(weights,source):
        """Check a native candidate against independent original-unit rules."""
        nonlocal best,value,best_source
        audit=original_audit(problem,weights)
        event=dict(source=source,audit=audit)
        if audit['passed']:
            candidate=variance(problem,weights); event['variance']=candidate
            if candidate<value:
                best,value=np.asarray(weights).copy(),candidate
                best_source=source
        events.append(event)
    try:
        native=native_mosek(problem,model,problem.base,started+seconds,1,1e-4,retain)
    except (ValueError,RuntimeError,cp.SolverError) as error:
        native=dict(native_status='error',error=repr(error))
    record=dict(optional=optional,balanced=balanced,input_fingerprint=identity(problem),
        seconds=perf_counter()-started,allowance_seconds=seconds,
        weights=best.tolist(),variance=value,te_bp=tracking_error_bp(problem,best),
        audit=original_audit(problem,best),delivered_source=best_source,native=native,events=events)
    if optional==24:
        witness=independent_witness(problem)
        record['independent_reference_variance']=variance(problem,witness)
        record['independent_reference_audit']=original_audit(problem,witness)
    return record


def fixed_row_model(problem,reduce_rows):
    """Restore redundant constant rows locally without changing their coefficients."""
    fixed,states=one_trade(problem)
    original=formulation._interval_rows
    def intervals(values,directions,constant,lower,upper,reduce_fixed):
        """Toggle only exact constant-row reduction in this isolated graph build."""
        return original(values,directions,constant,lower,upper,reduce_fixed and reduce_rows)
    with patch.object(formulation,'_interval_rows',intervals):
        model=formulation.build_model(problem,fixed_states=fixed)
    return model,fixed,states


def run_fixed_rows(side,fraction,reduce_rows):
    """Compare graph delivery with an independent active-set reference or contradiction."""
    problem=pinned_group(side,fraction*1e-7)
    model,fixed,states=fixed_row_model(problem,reduce_rows)
    started=perf_counter(); weights=None; error=None
    try: weights=formulation.solve_qp(model)
    except cp.SolverError as failure: error=str(failure)
    record=dict(side=side,fraction_of_existing_audit_tolerance=fraction,
        reduce_satisfied_constant_rows=reduce_rows,input_fingerprint=identity(problem),
        seconds=perf_counter()-started,status=model.optimisation.status,error=error,
        diagnostics=model.qp_diagnostics,audit=original_audit(problem,weights),
        variance=None if weights is None else variance(problem,weights))
    if fraction<1.:
        reference=replace(problem,group_lower=np.array([-1.]),group_upper=np.array([1.]))
        record['independent_reference_variance']=independent_kkt(reference,{
            i:(states[i][j].lower,states[i][j].upper) for i,j in fixed.items()})
    else:
        record['independent_infeasibility_reason']='Pinned group violates unchanged audit envelope'
    return record


def run(output):
    """Run a separate fixed sequence only on the frozen released package cohort."""
    output=Path(output).resolve()
    if 'onedrive' in str(output).lower() or output.exists():
        raise ValueError('New C-local output directory required')
    if importlib.metadata.version('optimalportfolios')!='9.2.0':
        raise ValueError('Mechanism examples require the frozen OP 9.2.0 wheel')
    output.mkdir(parents=True)
    specification=dict(schema='execution-numerical-mechanisms-1',package='9.2.0',
        interpretation='Historical public defect controls, not untouched evaluation',
        cone_optional_counts=[24,72],cone_allowance_seconds=10.,
        cone_representations=['unbalanced','balanced'],
        fixed_sides=['lower','upper'],fixed_fractions=[1.-1e-8,1.01],
        fixed_reduction=[False,True],origin=origin_example(),
        sources=public_sources(),runner_sha256=sha256(Path(__file__)),
        fixture_sources={fn.__module__:sha256(Path(sys.modules[fn.__module__].__file__))
            for fn in (public_counterexample,independent_witness,one_trade,independent_kkt,pinned_group)})
    write_json(output/'protocol.json',specification)
    cones=[]; rows=[]
    for optional in specification['cone_optional_counts']:
        for balanced in (False,True):
            result=run_cone(optional,balanced); cones.append(result)
            write_json(output/f'cone_{optional}_{balanced}.json',result)
    for side in specification['fixed_sides']:
        for fraction in specification['fixed_fractions']:
            for reduce_rows in specification['fixed_reduction']:
                result=run_fixed_rows(side,fraction,reduce_rows); rows.append(result)
                write_json(output/f'rows_{side}_{fraction}_{reduce_rows}.json',result)
    write_json(output/'summary.json',dict(protocol=specification,cones=cones,fixed_rows=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
