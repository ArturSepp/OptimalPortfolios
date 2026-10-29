"""Compare the combination correction, its preserved baseline and a fresh-seed holdout."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .breadth import METHODS
from .run_benchmark import write_json
from .summarize_breadth import comparison, summarize
from .summarize_robustness import describe


def case_key(case):
    """Identify generated inputs independently of filenames renumbered by merged cohorts."""
    return tuple(case[k] for k in ('sample','optional_assets','budget','groups','regime','seed'))


def validate_designs(current, previous, holdout):
    """Reject changed inputs/controls, duplicate archives or reused fresh seeds."""
    previous = [previous] if isinstance(previous,dict) else previous
    old = [c for p in previous for c in p['cases']]
    if len({case_key(c) for c in old}) != len(old):
        raise ValueError('Duplicate archived development cases')
    if len({case_key(c) for c in current['cases']}) != len(current['cases']):
        raise ValueError('Duplicate current development cases')
    if {case_key(c):c['fingerprint'] for c in current['cases']} != {
            case_key(c):c['fingerprint'] for c in old}:
        raise ValueError('Before/after comparison requires identical development inputs')
    for name in ('source_sha256', 'runner_sha256', 'audit_sha256', 'ranked_config', 'versions'):
        if current[name] != holdout[name]:
            raise ValueError('Algorithm and runtime must stay frozen across the holdout')
    for p in previous:
        for name in ('runner_sha256','audit_sha256','ranked_config','versions'):
            if current[name] != p[name]:
                raise ValueError('Before/after runtime and experiment controls changed')
    specs = [copy.deepcopy(p['specification']) for p in (current,holdout,*previous)]
    seeds = [{key:set(s[key].pop('seed')) for key in ('grid','tiny')} for s in specs]
    if any(s != specs[0] for s in specs[1:]):
        raise ValueError('Development or holdout controls changed')
    for key in ('grid','tiny'):
        if seeds[0][key] != set().union(*(s[key] for s in seeds[2:])):
            raise ValueError('Merged development seeds differ from the archived union')
    if (seeds[0]['grid'] | seeds[0]['tiny']) & (seeds[1]['grid'] | seeds[1]['tiny']):
        raise ValueError('Holdout must use disjoint, predeclared seeds')


def records(run):
    """Read each frozen result without selecting by outcome."""
    cases=[json.loads(p.read_text()) for p in sorted(run.glob('breadth_*.json'))]
    return {case_key(c):c for c in cases}


def number(value, digits=5):
    """Format missing values and rounded zero without changing retained numerical records."""
    if value is None:
        return '--'
    return f'{0.0 if round(value,digits)==0 else value:.{digits}f}'


def search_work(cases):
    """Report recorded search work without treating missing diagnostics as zero work."""
    broad = [c for c in cases if c['sample'] == 'broad']
    output = {}
    for method in ('Ranked', 'OneAnchor'):
        diagnostics = [c['methods'].get(method, {}).get('diagnostics', {}) for c in broad]
        recorded = [d for d in diagnostics if 'qp_counts' in d and 'bridge_restarts' in d]
        output[method] = dict(expected=len(broad), recorded=len(recorded),
            sizing_attempts=describe([sum(v for k, v in d['qp_counts'].items()
                if k not in ('guidance', 'restart_guidance', 'conditional_guidance'))
                for d in recorded]),
            bridge_restarts=describe([len(d['bridge_restarts']) for d in recorded]),
            bridge_improved_cases=sum(any(r['improved'] for r in d['bridge_restarts'])
                                      for d in recorded),
            bridge_pair_trials=describe([sum(r['trials'] for r in
                d['bridge_completion_passes']) for d in recorded]))
    return output


def report(run, previous, holdout, output):
    """Retain individual summaries and add controlled before/after comparisons."""
    previous = [previous] if isinstance(previous,Path) else previous
    current_protocol=json.loads((run/'protocol.json').read_text())
    previous_protocols=[json.loads((p/'protocol.json').read_text()) for p in previous]
    holdout_protocol=json.loads((holdout/'protocol.json').read_text())
    validate_designs(current_protocol,previous_protocols,holdout_protocol)
    result = summarize(run, output/'development')
    baselines = [summarize(p,output/'previous'/str(i)) for i,p in enumerate(previous)]
    fresh = summarize(holdout, output/'holdout')
    old={k:c for p in previous for k,c in records(p).items()}
    new=records(run)
    result['search_work'] = search_work(new.values())
    fresh['search_work'] = search_work(records(holdout).values())
    write_json(output/'holdout/summary.json', fresh)
    band = result['specification']['descriptive_te_equality_bp']
    result['before_after'] = {}
    for method in ('Ranked','OneAnchor'):
        paired = [dict(id=c['id'],methods=dict(before=old[k]['methods'].get(method,{}),
                                       after=c['methods'].get(method,{})))
                  for k,c in new.items() if c['sample']=='broad']
        result['before_after'][method] = comparison(paired,'after','before',band)
    result['prior_misses'] = []
    for cohort,(baseline,p) in enumerate(zip(baselines,previous)):
        by_id={c['id']:c for c in records(p).values()}
        for miss in baseline['broad']['comparisons']['Ranked_ExactBase60']['worse_cases']:
            earlier=by_id[miss['case']];k=case_key(earlier)
            reference=earlier['methods']['ExactBase60']['te_bp']
            result['prior_misses'].append(dict(case=earlier['id'],current_case=new[k]['id'],
                cohort=cohort,seed=earlier['seed'],prior_gap_bp=miss['gap_bp'],
                archived_reference_te_bp=reference,
                new_gap_bp={m:new[k]['methods'][m]['te_bp']-reference for m in ('Ranked','OneAnchor')
                            if new[k]['methods'].get(m,{}).get('audit',{}).get('passed')}))
    result['baseline_protocol_sha256'] = [b['protocol_sha256'] for b in baselines]
    result['holdout'] = fresh
    write_json(output/'summary.json',result)
    (output/'breadth.tex').write_text(render(result),encoding='utf-8')
    return result


def render(result):
    """Render failures and independent controls without converting native statuses to proofs."""
    cohorts = [('Development',result),('Holdout',result['holdout'])]
    lines = [r'\section{Broader domains: correction and fresh holdout}\label{sec:breadth}',
        'A broader factorial uses 24/72 nominal optional instruments, 2/4 discretionary '
        'tickets, disjoint sector rows or those rows plus three overlapping fractional '
        'baskets, and tight/wide funding-exposure bands. Holdings are heterogeneous, '
        'targets are lognormal relative tilts, and dense annual covariances combine '
        'market, sector and random factor loadings with positive residual variance. '
        'Two mandatory exits are shared. Tight group half-widths are 0.1 NAV percentage '
        'points and cash half-widths are 0.05 points. Wider group half-widths are '
        '1 percentage point, with cash between 0.5 and 50 percent. All arms retain '
        'the original objective, 0.5 percent minimum ticket, 0.25 percent holding '
        'floor and original audit/zero tolerances.',
        'Three earlier 32-case grids, using seed pairs 71001 and 71002, 81001 and '
        '81002, and 91001 and 91002, '
        'are now one 96-case development cohort. The first exposed four shared search misses, '
        'recovered by conditional guidance. The second exposed five further misses, '
        'including disjoint groups and an empty discretionary incumbent. Unchanged '
        'fixed-state sizing reproduced their better recorded Exact portfolios. '
        'An interval-screened search spends up to 576 pair trials across '
        'three anchors, allowing retained trades to resize. Low slice priorities '
        'cannot reject those supports. Earlier outcomes and source hashes remain '
        'in the deposit. The third grid retained one 0.549 annual bp miss: its better '
        'reference replaced three positions and passed unchanged fixed-state sizing. '
        'The present correction restarts from at most 24 cached audited alternatives, '
        'then completes the first improving branch with one more bounded pair phase. '
        'The reference support is never supplied to this search.',
        'After freezing the correction, a fresh 32-case grid uses seeds 101001 and 101002. '
        'The code is not retuned on its outcomes. Twelve eight-optional, two-ticket '
        'controls accompany development, using seeds 72001, 72002, 82001, 82002, '
        '92001 and 92002; four fresh controls use seeds 102001 and 102002. Cells sharing size and seed '
        'share risk arrays and holdings, so factorial cells are dependent. This is '
        'a same-generator holdout, not independent external validation.',
        'Full Ranked and OneAnchor differ only in the improvement-pass count. '
        'ExactBase10/60 are independent common-base calls with 10/60-second allowances. '
        'ExactFull60 and ExactOne60 receive only their named heuristic incumbent. '
        'Each start generation cost is charged to its own call; starts are never pooled. '
        'All Exact calls retain the same seed, one thread and relative gap setting. '
        'Calls are serial and the heuristic, independent and started arm orders each '
        'alternate deterministically. Timing on one host remains descriptive. '
        'Equality below uses the unchanged $10^{-4}$ annual bp descriptive band.']
    for method,c in result['before_after'].items():
        lines.append(f"On identical inputs, corrected {method} records "
            f"{c['lower']}/{c['equal']}/{c['higher']} lower, equal and higher TE outcomes in "
            f"{c['pairs']}/{c['expected']} audited pairs. "
            f"Median paired runtime ratio:~{number(c['seconds_ratio']['median'],3)}.")
    lines += [
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Recovery of the archived three-position miss. Annual TE minus the archived independent-Exact reference, in bp.}',
        r'\begin{tabular}{lrrr}\toprule',
        r'Case & Earlier Ranked & Corrected Ranked & Corrected OneAnchor\\\midrule']
    for miss in result['prior_misses']:
        values = [number(miss['new_gap_bp'].get(m)) for m in ('Ranked','OneAnchor')]
        lines.append(f"{miss['seed']}/{miss['case'].removeprefix('breadth_')} & {miss['prior_gap_bp']:.5f} & "
                     +' & '.join(values)+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}',
        'In the archived seed-91001 example, the old optional support was '
        'Asset004/022/039/063. The recovered reference retains Asset022 and '
        'admits Asset029/032/049 in place of the other three names. An audited '
        'cached alternative supplies a different search path; the first improving '
        'branch and subsequent pair completion reach this support without a '
        'reference start. The seed has TE 172.310523~bp, above the retained '
        '171.989658~bp incumbent. Its branch reaches 171.518364~bp and pair '
        'completion reaches 171.440647~bp. These are search states; only the '
        'best audited final portfolio is delivered. Mandatory exits, the four-ticket discretionary allowance '
        'and every sizing constraint remain unchanged. A three-name endpoint '
        'difference does not imply that all intermediate portfolios are infeasible.']
    for label,summary in cohorts:
        broad=summary['broad']
        lines += [r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
            '\\caption{'+label+' broad grid: annual TE in bp and times in seconds. E2E charges preparation and the supplied start.}',
            r'\begin{tabular}{lrrrrr}\toprule',
            r'Arm & Audited & Mean TE & Median s & 95th pct. s & Median E2E s\\\midrule']
        for method in METHODS:
            item=broad['methods'][method]
            values=[item['te_bp']['mean'],item['seconds']['median'],item['seconds']['p95'],
                    item['end_to_end_seconds']['median']]
            lines.append(f"{method} & {item['audited']}/{item['expected']} & "
                         +' & '.join('--' if v is None else f'{v:.3f}' for v in values)+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}']
        work_items = []
        for method, work in summary.get('search_work', {}).items():
            work_items.append(f"{method}, {work['recorded']}/{work['expected']} recorded cases, "
                f"median uncached sizing attempts {number(work['sizing_attempts']['median'],1)}, "
                f"median cached-alternative restarts {number(work['bridge_restarts']['median'],1)}, "
                f"cases improved by those restarts {work['bridge_improved_cases']}")
        if work_items:
            lines.append(label+' search work: '+'; '.join(work_items)+'. '
                'Sizing counts exclude continuous guidance, include failed attempts, '
                'and can differ from native solve counts because of internal retries.')
        pair=broad['comparisons']['OneAnchor_Ranked']
        lines.append(f"{label}: OneAnchor has lower/equal/higher TE in "
            f"{pair['lower']}/{pair['equal']}/{pair['higher']} of {pair['pairs']} paired cases. "
            f"The paired median runtime ratio is {number(pair['seconds_ratio']['median'],3)}. "
            f"The relative simplification gate {'passes' if summary['simplification_gate']['passed'] else 'fails'}; "
            'it can identify a research candidate only and never authorizes production adoption.')
        lines += [r'\begin{table}[htbp]\centering\small',
            '\\caption{'+label+' left-minus-right TE outcomes; annual bp, with the same descriptive band.}',
            r'\begin{tabular}{llrrrr}\toprule',
            r'Left & Right & Lower & Equal & Higher & Max loss bp\\\midrule']
        for key,c in broad['comparisons'].items():
            method,reference=key.split('_')
            maximum=c['gap_bp']['maximum']
            maximum=number(None if maximum is None else max(0.,maximum))
            lines.append(f"{method} & {reference} & {c['lower']} & {c['equal']} & {c['higher']} & {maximum}"+r'\\')
        lines += [r'\bottomrule\end{tabular}\end{table}']
        misses=broad['comparisons']['Ranked_ExactBase60']['worse_cases']
        if misses:
            diagnosis={d['case']:d for d in summary['miss_diagnosis']}
            lines += [r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
                '\\caption{'+label+' remaining misses. Loss is Ranked minus independent Exact '
                'annual TE in bp; removals/admissions compare their discretionary supports. '
                'Visited records Full/OneAnchor. Sizing error is the unchanged fixed-state '
                'QP minus Exact TE, also in bp.}',
                r'\begin{tabular}{lrrrrrr}\toprule',
                r'Case & Loss & Remove & Admit & Visited & Audit & Sizing error\\\midrule']
            for miss in misses:
                d=diagnosis[miss['case']]
                full=d['heuristics']['Ranked']
                visited='/'.join('yes' if d['heuristics'][m]['visited'] else 'no'
                                 for m in ('Ranked','OneAnchor'))
                error=None if d['conditional_te_bp'] is None else d['conditional_te_bp']-d['exact_te_bp']
                lines.append(f"{miss['case'].removeprefix('breadth_')} & {miss['gap_bp']:.5f} & "
                    f"{len(full['removed'])} & {len(full['added'])} & {visited} & "
                    f"{'pass' if d['conditional_audit']['passed'] else 'fail'} & {number(error)}"+r'\\')
            lines += [r'\bottomrule\end{tabular}\end{table}',
                'This post-hoc diagnostic fixes each recorded Exact support and applies the '
                'unchanged sizing QP and audit. It does not alter a recorded arm or supply '
                'a reference support to the heuristic. The deposit retains group types, '
                'instrument identities and every tested support.']
            confirmed=[d for d in diagnosis.values() if d['conditional_audit']['passed']
                and abs(d['conditional_te_bp']-d['exact_te_bp']) <= result['specification']['descriptive_te_equality_bp']
                and all(not h['visited'] for h in d['heuristics'].values())]
            lines.append(f"{len(confirmed)}/{len(misses)} misses are unvisited by both variants "
                'and reproduce the better reference within the descriptive band under unchanged '
                'sizing and audit. These are demonstrated search-coverage limits.')
            empty=[d for d in confirmed if not d['heuristics']['Ranked']['support']]
            if empty:
                counts=', '.join(str(len(d['exact_support'])) for d in empty)
                lines.append(f"{len(empty)} cases have empty discretionary incumbents, while "
                    f'the recorded Exact supports contain {counts} trades. This support '
                    'difference does not prove that every intermediate support is infeasible. '
                    'The other diagnosed misses replace multiple discretionary trades.')
            if confirmed and all(d['regime']=='tight' for d in confirmed):
                overlap=sum(d['groups']=='overlap' for d in confirmed)
                lines.append(f"All diagnosed misses use tight bands; {overlap} have overlapping "
                    f"groups and {len(confirmed)-overlap} have disjoint groups. "
                    + ('The limitation is therefore not restricted to overlapping exposure rows.'
                       if overlap < len(confirmed) else
                       'The remaining diagnosed limitation in this cohort involves overlapping rows.'))
        lines.append(f"{label}: conservative bound gaps close within the descriptive band on "
            f"{broad['bound_closed_at_band']}/{broad['cases']} cases; median/maximum gaps are "
            f"{number(broad['bound_gap_bp']['median'],3)}/{number(broad['bound_gap_bp']['maximum'],3)} bp. "
            f"There are {broad['bound_inconsistencies']} upper/lower inconsistencies.")
        sources=[]
        for method in METHODS[2:]:
            item=broad['methods'][method]
            attribution=', '.join(f"{k.replace('_',' ')} {v}" for k,v in item['sources'].items())
            sources.append(f"{method}: {attribution} (native start received "
                           f"{item['start_received']}/{item['returned']})")
        lines.append(label+' delivery attribution: '+'; '.join(sources)+'.')
    combined_gate = all(s['simplification_gate']['passed'] for _,s in cohorts)
    lines.append('Across both cohorts, the relative simplification gate '
        + ('passes' if combined_gate else 'fails')
        + '; a pass on one cohort cannot erase a disadvantage on the other. '
        'This gate concerns the ablation only, not optimality or production qualification.')
    audited=sum(item['audited'] for _,s in cohorts for sample in ('broad','tiny')
                for item in s[sample]['methods'].values())
    expected=sum(item['expected'] for _,s in cohorts for sample in ('broad','tiny')
                 for item in s[sample]['methods'].values())
    references=[e for _,s in cohorts for e in s['tiny']['enumeration']]
    gaps=[abs(g) for e in references for g in e['arm_minus_reference_bp'].values()]
    lines.append(f"The two rerun grids deliver {audited}/{expected} independently audited arm outputs. "
        f"Independent full-weight QPs enumerate {sum(e['trials'] for e in references)} tiny-case "
        f"supports, with {sum(e['unresolved'] for e in references)} unresolved solves and "
        f"{sum(e['complete'] for e in references)}/{len(references)} complete enumerations. "
        f"Maximum absolute arm/reference TE difference is {max(gaps,default=float('nan')):.2e} bp. "
        'The reference constructs original full-weight rows independently of the OP state '
        'compiler and sizing builder; unresolved or unaudited solves preclude completeness.')
    lines.append('Native optimal status, native bounds and retained-start equality remain '
        'diagnostics, not independently validated global certificates. Reported conservative '
        'bounds use the existing plain/perspective supporting-plane LP dual with box residual '
        'correction and floating-point guard; relaxation primal values are never substituted. '
        'The pooled best feasible value is used for reporting only. These experiments do not '
        'change production or establish realised performance or universal dominance.')
    return '\n\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','holdout','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True,action='append')
    args=parser.parse_args()
    result=report(args.run,args.previous,args.holdout,args.output)
    print(json.dumps(result['before_after'],indent=2))
