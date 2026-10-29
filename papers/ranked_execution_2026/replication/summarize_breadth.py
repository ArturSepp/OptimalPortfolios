"""Report every broader-case outcome, independent controls and bound limitations."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from .benchmark import sha256
from .breadth import METHODS
from .run_benchmark import write_json
from .summarize_robustness import describe


def diagnose(case, spec):
    """Check missed Exact supports post hoc without altering frozen benchmark outcomes."""
    from optimalportfolios.execution.v2 import compile_states
    from optimalportfolios.execution.v2.model import build_model, solve_qp
    from .breadth import generated_case, fingerprint
    from .benchmark import independent_audit, tracking_error_bp
    problem=generated_case(case,spec)
    if fingerprint(problem)!=case['prepared_fingerprint']:
        raise ValueError('Diagnosis requires the identical prepared input')
    weights=np.asarray(case['methods']['ExactBase60']['weights'])
    fixed,support={},[]
    for i,row in enumerate(compile_states(problem)):
        if i==problem.cash:
            continue
        traded=abs(weights[i]-problem.current[i])>problem.rules.zero_tolerance
        matches=[j for j,state in enumerate(row) if state.ticket==traded
                 and state.lower-problem.rules.audit_tolerance<=weights[i]
                 <=state.upper+problem.rules.audit_tolerance]
        if len(matches)!=1:
            raise ValueError('Ambiguous diagnostic fixed state')
        fixed[i]=matches[0]
        if traded and not problem.mandatory[i]:
            support.append([i,matches[0]])
    model=build_model(problem,fixed_states=fixed)
    sized=solve_qp(model)
    audit=independent_audit(problem,sized)
    heuristics={}
    for method in ('Ranked','OneAnchor'):
        item=case['methods'][method]
        selected=[i for i in range(len(weights)) if i!=problem.cash and not problem.mandatory[i]
                  and abs(item['weights'][i]-problem.current[i])>problem.rules.zero_tolerance]
        target={i for i,_ in support}
        heuristics[method]=dict(support=selected,
            removed=sorted(set(selected)-target),added=sorted(target-set(selected)),
            visited=any(t['support']==support for t in item['diagnostics']['trials']),
            te_bp=item['te_bp'])
    return dict(case=case['id'],optional_assets=case['optional_assets'],budget=case['budget'],
        groups=case['groups'],regime=case['regime'],seed=case['seed'],exact_support=support,
        exact_te_bp=case['methods']['ExactBase60']['te_bp'],conditional_status=model.optimisation.status,
        conditional_audit=audit,conditional_te_bp=tracking_error_bp(problem,sized) if audit['passed'] else None,
        heuristics=heuristics,purpose='Post-hoc support diagnosis only; no retuning')


def comparison(cases, left, right, band):
    """Keep the full expected denominator and report paired differences explicitly."""
    pairs = [(c, c['methods'][left], c['methods'][right]) for c in cases
             if all(c['methods'].get(m, {}).get('audit', {}).get('passed', False) for m in (left,right))]
    values = np.asarray([a['te_bp']-b['te_bp'] for _,a,b in pairs])
    return dict(expected=len(cases), pairs=len(pairs), lower=int((values < -band).sum()),
        equal=int((np.abs(values) <= band).sum()), higher=int((values > band).sum()),
        gap_bp=describe(values), worse_cases=[dict(case=c['id'],gap_bp=float(v))
            for (c,_,_),v in zip(pairs,values) if v > band],
        seconds_ratio=describe([a['seconds']/b['seconds'] for _,a,b in pairs]))


def sample_summary(cases, band):
    """Attribute native and retained starts without interpreting either as certificates."""
    result = dict(cases=len(cases), prepared=sum('base_audit' in c for c in cases),
                  optional_candidates=describe([c['optional_candidates'] for c in cases if 'optional_candidates' in c]),
                  methods={}, comparisons={}, certificates=[])
    for method in METHODS:
        items = [c['methods'][method] for c in cases if method in c['methods']]
        valid = [r for r in items if r['audit']['passed']]
        result['methods'][method] = dict(expected=len(cases),returned=len(items),audited=len(valid),
            te_bp=describe([r['te_bp'] for r in valid]), seconds=describe([r['seconds'] for r in items]),
            end_to_end_seconds=describe([r['end_to_end_seconds'] for r in items]),
            statuses=dict(Counter(r['status'] for r in items)),
            sources=dict(Counter(r.get('diagnostics',{}).get('delivered_source','heuristic') for r in items)),
            native_status=dict(Counter(r.get('diagnostics',{}).get('native_status','not_applicable') for r in items)),
            start_received=sum(r.get('diagnostics',{}).get('initial_start',{}).get('received') is True for r in items))
    for left,right in [('OneAnchor','Ranked'),('Ranked','ExactBase60'),('OneAnchor','ExactBase60'),
                       ('Ranked','ExactFull60'),('OneAnchor','ExactOne60'),('ExactBase60','ExactBase10')]:
        result['comparisons'][left+'_'+right] = comparison(cases,left,right,band)
    for case in cases:
        valid = [r for r in case['methods'].values() if r.get('audit',{}).get('passed')]
        if not valid:
            continue
        upper_te = min(r['te_bp'] for r in valid)
        bound = max([r['lower_bound'] for r in case.get('relaxations',{}).values()]+[0.])
        consistent = bound <= (upper_te/1e4)**2
        result['certificates'].append(dict(case=case['id'], lower_variance=bound,
            consistent=consistent, best_delivered_te_bp=upper_te,
            lower_te_bp=1e4*float(np.sqrt(bound)),
            best_gap_bp=upper_te-1e4*float(np.sqrt(bound)) if consistent else None,
            reasons={name:r['certificate'] for name,r in case.get('relaxations',{}).items()},
            failures={name:r['certificate_error'] for name,r in case.get('relaxations',{}).items()
                      if 'certificate_error' in r}))
    result['bound_gap_bp'] = describe([r['best_gap_bp'] for r in result['certificates'] if r['consistent']])
    result['bound_closed_at_band'] = sum(r['consistent'] and r['best_gap_bp'] <= band for r in result['certificates'])
    result['bound_inconsistencies'] = sum(not r['consistent'] for r in result['certificates'])
    result['enumeration'] = [dict(case=c['id'],complete=c['enumeration']['complete'],
        unresolved=c['enumeration']['unresolved'],trials=len(c['enumeration']['trials']),
        te_bp=c['enumeration']['te_bp'],audit=c['enumeration']['audit'],
        arm_minus_reference_bp={m:r['te_bp']-c['enumeration']['te_bp'] for m,r in c['methods'].items()
            if r.get('audit',{}).get('passed') and c['enumeration']['audit']['passed']})
        for c in cases if 'enumeration' in c]
    return result


def summarize(run, output):
    """Require the complete frozen design and matching record/input identities."""
    protocol = json.loads((run/'protocol.json').read_text())
    completion = json.loads((run/'complete.json').read_text())
    cases = [json.loads(p.read_text()) for p in sorted(run.glob('breadth_*.json'))]
    declared = {c['id']:c for c in protocol['cases']}
    if not completion['full_sample'] or len(cases)!=len(declared) or completion['cases']!=len(cases):
        raise ValueError('Headline results require every frozen case')
    for c in cases:
        expected = declared.pop(c['id'])
        if any(c[k]!=v for k,v in expected.items()):
            raise ValueError('Case identity differs from frozen design')
        if sha256(run/'inputs'/(c['id']+'.npz'))!=c['input_sha256']:
            raise ValueError('Changed frozen input bytes')
    band = protocol['specification']['descriptive_te_equality_bp']
    result = dict(protocol_sha256=sha256(run/'protocol.json'),
        records_sha256={p.name:sha256(p) for p in sorted(run.glob('breadth_*.json'))},
        specification=protocol['specification'], versions=protocol['versions'],
        broad=sample_summary([c for c in cases if c['sample']=='broad'],band),
        tiny=sample_summary([c for c in cases if c['sample']=='tiny'],band))
    broad = result['broad']
    pair = broad['comparisons']['OneAnchor_Ranked']
    one, full = (broad['methods'][m] for m in ('OneAnchor','Ranked'))
    no_failures = one['audited']==full['audited']==broad['cases']
    result['simplification_gate'] = dict(
        all_paired=pair['pairs']==broad['cases'], no_quality_loss=pair['higher']==0,
        no_extra_failures=no_failures,
        paired_median_faster=pair['seconds_ratio']['median'] is not None and pair['seconds_ratio']['median']<1,
        candidate_only=True)
    result['simplification_gate']['passed'] = all(result['simplification_gate'][k] for k in
        ('all_paired','no_quality_loss','no_extra_failures','paired_median_faster'))
    misses={r['case'] for r in broad['comparisons']['Ranked_ExactBase60']['worse_cases']}
    result['miss_diagnosis']=[diagnose(c,protocol['specification']) for c in cases if c['id'] in misses]
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'summary.json',result)
    (output/'breadth.tex').write_text(render_tex(result),encoding='utf-8')
    return result


def render_tex(result):
    """Render measured results, including rejection of simplification when warranted."""
    broad, tiny = result['broad'], result['tiny']
    if any(item['audited'] != sample['cases'] for sample in (broad,tiny)
           for item in sample['methods'].values()):
        return (r'\section{Broader comparison failures}'+'\n'
                'At least one declared portfolio failed to prepare, return or pass its audit. '
                'All records and denominators remain in the accompanying summary; '
                'success-only tables have not been generated.\n')
    pair=broad['comparisons']['OneAnchor_Ranked']
    lines=[r'\section{Broader ablation and longer Exact controls}\label{sec:breadth}',
        'After the earlier studies, we froze both heuristic configurations before a new generated '
        'factorial: 24/72 nominal optional instruments, 2/4 discretionary tickets, three disjoint '
        'sector rows or those rows plus three overlapping fractional baskets, two funding/exposure '
        'regimes, and seeds 71001/71002. Current holdings are heterogeneous and targets are relative '
        'lognormal tilts. Market, sector and random factor loadings plus positive residual variance '
        'generate the dense annual covariance. Two forced exits are shared. Tight group bands are '
        '0.1 NAV percentage points either side of base exposure, with cash within 0.05 percentage '
        'points of its funded base. Wider bands are 1 percentage point, with cash between 0.5 and '
        '50 percent. Cells sharing size and seed share holdings and covariance, so the '
        'factorial cells are dependent and are not treated as independent replications. '
        'These are new synthetic domains, not external market validation.',
        'All arms share each prepared problem, the unchanged objective, a 0.5 percent minimum '
        'ticket, 0.25 percent holding floor, and original audit/zero tolerances. Only the OneAnchor '
        'pass count differs from full Ranked. ExactBase10 and ExactBase60 are separate calls from '
        'the common base. ExactFull60 and ExactOne60 receive only their named heuristic '
        'incumbent; no pooled start is used. The 60-second controls retain the common '
        'thread, seed and gap settings. Heuristic, independent and started-call orders alternate '
        'deterministically across cases. Calls are serial, with preparation and each supplied start '
        'own generation cost recorded separately.',
        f"The broad grid prepared {broad['prepared']}/{broad['cases']} cases, with "
        f"{broad['optional_candidates']['minimum']:.0f}--{broad['optional_candidates']['maximum']:.0f} "
        'effective optional candidates. No failed record is omitted.',
        r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{Broader grid. Annual TE in bp; algorithm call and end-to-end (E2E) times in seconds.}',
        r'\begin{tabular}{lrrrrr}\toprule',
        r'Arm & Audited & Mean TE & Median s & 95th pct. s & Median E2E s\\\midrule']
    for method,item in broad['methods'].items():
        if item['audited']:
            lines.append(f"{method} & {item['audited']}/{item['expected']} & {item['te_bp']['mean']:.3f} & "
                         f"{item['seconds']['median']:.3f} & {item['seconds']['p95']:.3f} & "
                         f"{item['end_to_end_seconds']['median']:.3f}"+r'\\')
        else:
            lines.append(f"{method} & 0/{item['expected']} & -- & -- & -- & --"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}',
        f"OneAnchor has lower TE in {pair['lower']} paired cases, ties in {pair['equal']} and "
        f"higher TE in {pair['higher']}; the maximum disadvantage is "
        f"{max(0.,pair['gap_bp']['maximum'] or 0.):.5f} bp. "
        f"The paired median OneAnchor/Ranked runtime ratio is {pair['seconds_ratio']['median']:.3f}. "
        'Equality uses the unchanged descriptive band of $10^{-4}$ annual bp. '
        + ('The predeclared relative simplification gate passes; this identifies a candidate for '
           'separate production qualification, not approval to change production.' if result['simplification_gate']['passed']
           else 'The predeclared simplification gate fails. Full-search defaults remain unchanged; '
           'the new losses were not tuned away.'),
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Broad-grid left-minus-right TE comparisons; counts retain paired audit denominators.}',
        r'\begin{tabular}{llrrrr}\toprule',
        r'Left & Right & Lower & Equal & Higher & Max loss bp\\\midrule']
    for key,c in broad['comparisons'].items():
        left,right=key.split('_')
        lines.append(f"{left} & {right} & {c['lower']} & {c['equal']} & {c['higher']} & "
                     f"{max(0.,c['gap_bp']['maximum'] or 0.):.5f}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    if result.get('miss_diagnosis'):
        lines += ['Both heuristic configurations miss better independent-Exact portfolios in '
            'the cases below. The diagnostic fixes the recorded Exact states and runs the '
            'unchanged OP sizing QP; it does not modify or rerun a benchmark arm.',
            r'\begin{table}[htbp]\centering\small',
            r'\caption{Post-hoc search diagnosis. Sets count optional instruments; TE loss is Ranked minus ExactBase60.}',
            r'\begin{tabular}{lrrrrr}\toprule',
            r'Case & Nominal assets & Budget & Removed & Added & Loss bp\\\midrule']
        for d in result['miss_diagnosis']:
            h=d['heuristics']['Ranked']
            lines.append(f"{d['case'].removeprefix('breadth_')} & {d['optional_assets']} & {d['budget']} & "
                         f"{len(h['removed'])} & {len(h['added'])} & {h['te_bp']-d['exact_te_bp']:.5f}"+r'\\')
        lines.append(r'\bottomrule\end{tabular}\end{table}')
        verified=[d for d in result['miss_diagnosis'] if d['conditional_audit']['passed']]
        absent=sum(not d['heuristics']['Ranked']['visited'] and not d['heuristics']['OneAnchor']['visited']
                   for d in result['miss_diagnosis'])
        tight=sum(d['regime']=='tight' and d['groups']=='overlap' for d in result['miss_diagnosis'])
        difference=max((abs(d['conditional_te_bp']-d['exact_te_bp']) for d in verified),default=None)
        difference_text=f'{difference:.8f} bp' if difference is not None else 'unavailable'
        lines.append(f"{tight} cases have tight overlapping fractional rows. "
            f"The recorded Exact support was absent from both trial logs in {absent} cases. "
            f"Conditional resizing passes the original audit in {len(verified)} cases; the maximum "
            f"absolute TE difference from Exact is {difference_text}. ")
        if len(verified)==absent==len(result['miss_diagnosis']) and difference < 1e-4:
            lines.append('These observations identify search coverage failures, rather than failed '
                'sizing of the same support. The support changes are reported in the table. '
                'Additional passes alone did not discover them.')
        lines.append('These new cases were not used to change the algorithms.')
    for method in METHODS[2:]:
        item=broad['methods'][method]
        sources=', '.join(f"{k.replace('_',' ')}: {v}" for k,v in item['sources'].items())
        lines.append(f"{method}: {sources}. Verified native start receipt occurs in "
                     f"{item['start_received']}/{item['returned']} calls.")
    lines += ['A native optimal-status report and native global bound remain solver diagnostics. '
        'A retained start or callback reproducing that start is not independent discovery. '
        'The strongest existing plain/perspective supporting-plane LP dual bound is reported '
        'with its box residual correction and floating-point guard; conic primal values are never '
        'substituted for lower bounds. The best delivered upper value is pooled for reporting '
        'only, never for a solver start.',
        f"Conservative bound gaps close within the descriptive band on {broad['bound_closed_at_band']} "
        f"of {broad['cases']} broad cases. Median and maximum best-delivered TE minus bound are "
        f"{broad['bound_gap_bp']['median']:.3f} and {broad['bound_gap_bp']['maximum']:.3f} bp. "
        f"There are {broad['bound_inconsistencies']} upper/lower inconsistencies.",
        'Four additional eight-optional, two-ticket cases use seeds 72001/72002 and both group '
        'structures under tight bands. An independent full-weight QP in basis-point units '
        'enumerates every support using directly constructed monotone intervals, without the OP '
        'state compiler or sizing builder. Solver tolerances and original-unit audits remain fixed. '
        'Unresolved or rejected support solves prevent a completeness claim; these are numerical '
        'references, not exact-arithmetic certificates.']
    for e in tiny['enumeration']:
        gaps=list(e['arm_minus_reference_bp'].values())
        gap_text=f'{max(abs(g) for g in gaps):.2e}' if gaps else 'unavailable'
        lines.append(f"Tiny case {e['case'].removeprefix('breadth_')}: {e['trials']} supports, "
                     f"{e['unresolved']} unresolved; maximum absolute arm/reference TE "
                     f"difference {gap_text} bp.")
    lines.append('These comparisons test two fixed search budgets on a broader but finite synthetic '
        'population. They neither change production nor establish realised performance, universal '
        'dominance or an optimality certificate for time-limited integer outputs.')
    return '\n\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(summarize(args.run,args.output)['simplification_gate'],indent=2))
