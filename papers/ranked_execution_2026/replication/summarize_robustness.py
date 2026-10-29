"""Summarise complete robustness runs without dropping failed observations."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import sha256
from .run_benchmark import write_json


def diagnose_support(case):
    """Post-hoc fixed-support sizing check; it cannot change any benchmark outcome."""
    from optimalportfolios.execution.v2 import compile_states
    from optimalportfolios.execution.v2.model import build_model, solve_qp
    from .robustness import synthetic_case, prepare
    from .benchmark import independent_audit, tracking_error_bp
    raw = synthetic_case(case['optional_assets'], case['budget'], case['width'],
                         case['cash_band'], case['seed'])
    problem, _ = prepare(raw)
    exact = np.asarray(case['methods']['ExactRanked']['weights'])
    fixed, support = {}, []
    for i, states in enumerate(compile_states(problem)):
        if i == problem.cash:
            continue
        traded = abs(exact[i]-problem.current[i]) > problem.rules.zero_tolerance
        choices = [j for j, state in enumerate(states) if state.ticket == traded
                   and state.lower-problem.rules.audit_tolerance <= exact[i]
                   <= state.upper+problem.rules.audit_tolerance]
        if len(choices) != 1:
            raise ValueError('Ambiguous diagnostic fixed state')
        fixed[i] = choices[0]
        if traded and not problem.mandatory[i]:
            support.append([i, choices[0]])
    model = build_model(problem, fixed_states=fixed)
    weights = solve_qp(model)
    audit = independent_audit(problem, weights)
    return dict(case=case['id'], purpose='Post-hoc support diagnosis, not benchmark tuning',
        support=support, visited_by_ranked=any(trial['support'] == support
            for trial in case['methods']['Ranked']['diagnostics']['trials']),
        status=model.optimisation.status, audit=audit,
        conditional_te_bp=tracking_error_bp(problem, weights) if audit['passed'] else None,
        exact_ranked_te_bp=case['methods']['ExactRanked']['te_bp'],
        ranked_te_bp=case['methods']['Ranked']['te_bp'])


def describe(values):
    """Return transparent descriptive statistics, with an explicit empty denominator."""
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                minimum=float(x.min()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                p95=float(np.quantile(x, .95)) if len(x) else None,
                maximum=float(x.max()) if len(x) else None)


def compare_previous(cases, previous, band):
    """Pair archived development grids by generator inputs, never ordinal file names."""
    paths = [previous] if isinstance(previous, Path) else list(previous)
    fields = ('optional_assets', 'budget', 'width', 'cash_band', 'seed')
    archived = {}
    for path in paths:
        for file in path.glob('scarcity_*.json'):
            old = json.loads(file.read_text())
            key = tuple(old[k] for k in fields)
            if key in archived:
                raise ValueError('Duplicate archived development inputs')
            archived[key] = old
    deltas, misses, recovered = [], 0, 0
    for case in cases:
        key = tuple(case[k] for k in fields)
        if key not in archived:
            raise ValueError('Missing archived development inputs')
        old = archived.pop(key)
        before, after = old['methods']['Ranked'], case['methods']['Ranked']
        if not before['audit']['passed'] or not after['audit']['passed']:
            raise ValueError('Before/after comparison requires audited paired incumbents')
        deltas.append(before['te_bp']-after['te_bp'])
        reference = old['methods']['ExactRanked']['te_bp']
        if before['te_bp']-reference > band:
            misses += 1
            recovered += after['te_bp']-reference <= band
    if archived or not deltas:
        raise ValueError('Unpaired archived development inputs')
    values = np.asarray(deltas)
    return dict(previous_protocol_sha256=[sha256(p/'protocol.json') for p in paths],
        pairs=len(values), improved=int((values > band).sum()),
        equal=int((np.abs(values) <= band).sum()), worsened=int((values < -band).sum()),
        mean_reduction_bp=float(values.mean()), maximum_reduction_bp=float(values.max()),
        old_misses=misses, old_misses_recovered=int(recovered))


def summarize(run, output, holdout=None, previous=None, compact=False):
    """Keep every declared case, failed preparation and failed audit in the summary."""
    complete = json.loads((run/'complete.json').read_text())
    protocol = json.loads((run/'protocol.json').read_text())
    if not complete['full_sample'] or complete['study'] not in ('all', 'scarcity'):
        raise ValueError('Headline summary requires the complete frozen study')
    spec = protocol['specification']
    sensitivity = [json.loads(p.read_text()) for p in sorted(run.glob('sensitivity_*.json'))]
    scarcity = [json.loads(p.read_text()) for p in sorted(run.glob('scarcity_*.json'))]
    expected = int(np.prod([len(spec['scarcity'][k]) for k in (
        'optional_assets', 'discretionary_allowance', 'group_half_width', 'cash_band', 'seeds')]))
    sensitivity_count = spec['sensitivity']['case_count'] if complete['study'] == 'all' else 0
    if len(sensitivity) != sensitivity_count or len(scarcity) != expected:
        raise ValueError('Missing declared experiment records')
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(protocol_sha256=sha256(run/'protocol.json'),
                   record_sha256={p.name: sha256(p) for p in sorted(run.glob('*.json'))},
                   versions=protocol['versions'], specification=spec, sensitivity={}, scarcity={})
    rows = []
    for case in sensitivity:
        for item in case['variants']:
            row = dict(case=case['id'], domain=item['domain'], variant=item['variant'],
                       level=item['level'], target_half_l1=item['target_half_l1'],
                       audit_passed=item.get('result', {}).get('audit', {}).get('passed', False))
            row.update(item.get('metrics', {}))
            rows.append(row)
    frame = pd.DataFrame(rows).reindex(columns=[
        'case', 'domain', 'variant', 'level', 'target_half_l1', 'audit_passed', 'half_l1',
        'support_symmetric_difference', 'mandatory_added', 'mandatory_removed',
        'common_reference_te_bp', 'baseline_reference_te_bp', 'noncash_gross_turnover_change'])
    frame.to_csv(output/'sensitivity.csv', index=False)
    expected_variants = 2*len(spec['sensitivity']['variants'])*len(spec['sensitivity']['domains'])
    summary['sensitivity']['cases'] = len(sensitivity)
    summary['sensitivity']['baseline_audit_passed'] = sum(
        case.get('baseline', {}).get('audit', {}).get('passed', False) for case in sensitivity)
    summary['sensitivity']['expected_variants'] = len(sensitivity)*expected_variants
    summary['sensitivity']['recorded_variants'] = len(frame)
    summary['sensitivity']['audited_variants'] = int(frame.audit_passed.sum())
    summary['sensitivity']['cells'] = []
    for (domain, variant, level), cell in frame.groupby(['domain', 'variant', 'level']):
        valid = cell[cell.audit_passed & cell.half_l1.notna()]
        summary['sensitivity']['cells'].append(dict(
            domain=domain, variant=variant, level=int(level), expected=len(sensitivity),
            completed=len(cell), audited=int(cell.audit_passed.sum()), paired=len(valid),
            target_half_l1=describe(valid.target_half_l1), half_l1=describe(valid.half_l1),
            support_changed=int((valid.support_symmetric_difference > 0).sum()),
            support_difference=describe(valid.support_symmetric_difference),
            mandatory_added=int(valid.mandatory_added.sum()),
            mandatory_removed=int(valid.mandatory_removed.sum()),
            common_reference_te_change_bp=describe(
                valid.common_reference_te_bp-valid.baseline_reference_te_bp),
            gross_noncash_turnover_change=describe(valid.noncash_gross_turnover_change)))
    sc = summary['scarcity']
    sc['cases'] = len(scarcity)
    sc['prepared'] = sum('base_audit' in case for case in scarcity)
    sc['base_audit_passed'] = sum(case.get('base_audit', {}).get('passed', False) for case in scarcity)
    sc['optional_candidates'] = describe([case['optional_candidates'] for case in scarcity
                                         if 'optional_candidates' in case])
    sc['methods'] = {}
    flat = []
    for method in spec['scarcity']['methods']:
        entries = [case['methods'][method] for case in scarcity if method in case['methods']]
        valid = [item for item in entries if item['audit']['passed']]
        sc['methods'][method] = dict(expected=len(scarcity), returned=len(entries), audited=len(valid),
            te_bp=describe([item['te_bp'] for item in valid]),
            seconds=describe([item['seconds'] for item in entries]),
            end_to_end_seconds=describe([item['end_to_end_seconds'] for item in entries]),
            allowance_binding=sum(item['allowance_binding'] for item in valid),
            delivered_sources=dict(Counter(item['diagnostics'].get('delivered_source', 'ranked_search')
                                           for item in valid)),
            statuses=dict(Counter(item['status'] for item in entries)))
        for case in scarcity:
            item = case['methods'].get(method, {})
            flat.append(dict(case=case['id'], method=method, budget=case['budget'],
                optional_assets=case['optional_assets'], width=case['width'], cash_band=case['cash_band'],
                audited=item.get('audit', {}).get('passed', False), te_bp=item.get('te_bp'),
                tickets=item.get('audit', {}).get('optional_tickets'),
                seconds=item.get('seconds'), source=item.get('diagnostics', {}).get('delivered_source')))
    pd.DataFrame(flat).to_csv(output/'scarcity.csv', index=False)
    sc['ranked_comparisons'] = {}
    for method in spec['scarcity']['methods'][1:]:
        pairs = [(case, case['methods']['Ranked'], case['methods'][method]) for case in scarcity
                 if all(case['methods'].get(m, {}).get('audit', {}).get('passed', False)
                        for m in ('Ranked', method))]
        gap = np.array([ranked['te_bp']-other['te_bp'] for _, ranked, other in pairs])
        band = spec['descriptive_te_equality_bp']
        sc['ranked_comparisons'][method] = dict(expected=len(scarcity), pairs=len(pairs),
            lower=int((gap < -band).sum()), equal=int((np.abs(gap) <= band).sum()),
            higher=int((gap > band).sum()), gap_bp=describe(gap),
            worst_case=None if not pairs else pairs[int(np.argmax(gap))][0]['id'])
    worst = sc['ranked_comparisons']['ExactRanked']['worst_case']
    sc['worst_support_diagnosis'] = (None if worst is None else
        diagnose_support(next(c for c in scarcity if c['id'] == worst)))
    if holdout is not None:
        summary['holdout'] = summarize(holdout, output/'holdout')
    if previous is not None:
        summary['previous_comparison'] = compare_previous(
            scarcity, previous, spec['descriptive_te_equality_bp'])
    write_json(output/'summary.json', summary)
    if complete['study'] == 'all':
        (output/'robustness.tex').write_text(render_tex(summary, compact=compact), encoding='utf-8')
        plot_sensitivity(frame, output/'sensitivity.pdf')
    return summary


def render_tex(summary, compact=False):
    """Generate measured development and holdout results without success-only denominators."""
    scarcity, sensitivity = summary['scarcity'], summary['sensitivity']
    if (sensitivity['audited_variants'] != sensitivity['expected_variants']
            or sensitivity['baseline_audit_passed'] != sensitivity['cases']
            or any(item['audited'] != item['expected'] for item in scarcity['methods'].values())):
        return (r'\section{Robustness experiment failures}'+'\n'
                'At least one preparation or audit failed. Inspect all denominators in summary.json; '
                'success-only manuscript tables have not been generated.\n')
    lines = [r'\section{Controlled sensitivity and scarce tickets}\label{sec:robustness}',
        'These experiments use only public decision arrays and generated portfolios. '
        'The original 32 synthetic problems and the first 32 held-out problems exposed '
        'search misses and were then used for algorithm development. All 64 are now '
        'explicitly development data. After fixing the search, '
        'we froze its source and ran new seeds without further tuning. All arms retain the same '
        'objective, domain, minimum trades, holding floors and audit tolerances.',
        r'\subsection{Input estimates and execution domains}',
        'Twenty-four evenly spaced dates, including both endpoints of the 219-case manifest, '
        'receive target-only, covariance-only and joint shocks. Target shocks have 10/50 bp '
        r'half-$L^1$ NAV distance, preserving cash and zero targets. Covariance shocks use '
        r'$D\Sigma D$, $D_{ii}=\exp(\epsilon u_i)$, $u_i\in[-1,1]$, '
        r'$\epsilon=0.01/0.05$, preserving positive semidefiniteness. Each runs with the '
        'original domain fixed and with optional target corridors rebuilt within unchanged '
        'full bounds. Lifecycle declarations, eligibility, group rows and cash limits stay fixed. '
        'Common preparation is repeated for rebuilt corridors. This is conditional sensitivity, '
        'not upstream estimation, changed lifecycle policy or an own-book return simulation.',
        f"All {sensitivity['baseline_audit_passed']} baseline and "
        f"{sensitivity['audited_variants']} perturbed portfolios pass the independent audit. "
        'Risk changes are also evaluated against the original target and covariance. '
        'No case or failed output is removed.',
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Large-shock allocation changes: half-$L^1$ percent of NAV.}',
        r'\begin{tabular}{llrrr}\toprule',
        r'Input & Domain & 95th pct. & Maximum & Support changed\\\midrule']
    for variant in ('target', 'covariance', 'joint'):
        for domain in ('fixed', 'rebuilt_corridor'):
            cell = next(c for c in sensitivity['cells']
                        if c['variant'] == variant and c['domain'] == domain and c['level'] == 1)
            lines.append(f"{variant.title()} & {'Fixed' if domain == 'fixed' else 'Rebuilt'} & "
                f"{100*cell['half_l1']['p95']:.3f} & {100*cell['half_l1']['maximum']:.3f} & "
                f"{cell['support_changed']}/{cell['expected']}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}',
        'These dates overlap across treatments; they are not independent replications. '
        'The numerical archive retains turnover, common-reference risk and full weights.',
        r'\begin{figure}[htbp]\centering',
        r'\includegraphics[width=\linewidth]{figures/sensitivity.pdf}',
        r'\caption{All conditional allocation shifts. Small/large target shocks are 10/50 NAV bp; '
        r'volatility shocks have maximum absolute log shifts of 1/5 percent.}\end{figure}',
        r'\subsection{Earlier scarce-ticket cohorts}',
        'The factorial generator crosses 12/36 nominal optional instruments, 3/6 discretionary '
        'tickets, group half-widths of 0.5/8 NAV percentage points, narrow/wide cash bands, '
        'and seeds. Every case has two forced exits, three group rows, a 0.5 percent '
        'minimum trade and a 0.25 percent holding floor. Four random factors plus positive '
        'diagonal residual variance generate the annual covariance. Seeds 41001/41002 '
        'and 51001/51002 define the 64 development cases; seeds 61001/61002 formed '
        'the earlier 32-case holdout from the same generator. After subsequent algorithm '
        'development, both are regression replays, not a new holdout. The fresh frozen '
        'comparison is reported in Section~\\ref{sec:breadth}. These are not independent '
        'external validation or a new market population. Exact has ten seconds and one native thread.',
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Scarce-ticket quality and computation. TE is annual bp; times are call seconds. '
        r'Binding counts refer to discretionary allowance.}',
        r'\begin{tabular}{llrrrr}\toprule',
        r'Sample & Method & Mean TE & Median s & 95th pct. s & Binding\\\midrule']
    samples = [('Development', scarcity)]
    if 'holdout' in summary:
        samples.append(('Earlier holdout', summary['holdout']['scarcity']))
    for label, sample in samples:
        for method, item in sample['methods'].items():
            lines.append(f"{label} & {method} & {item['te_bp']['mean']:.3f} & "
                f"{item['seconds']['median']:.3f} & {item['seconds']['p95']:.3f} & "
                f"{item['allowance_binding']}/{item['expected']}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}',
        r'\begin{table}[htbp]\centering\small',
        r'\caption{Ranked minus comparator outcomes; descriptive equality band $10^{-4}$ annual bp.}',
        r'\begin{tabular}{llrrrr}\toprule',
        r'Sample & Comparator & Lower & Equal & Higher & Mean difference\\\midrule']
    for label, sample in samples:
        for method, c in sample['ranked_comparisons'].items():
            lines.append(f"{label} & {method} & {c['lower']} & {c['equal']} & {c['higher']} & "
                         f"{c['gap_bp']['mean']:.5f}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    if all(sample['ranked_comparisons']['OneAnchor']['equal'] == sample['cases']
           for _, sample in samples):
        lines.append('The OneAnchor ablation matches full Ranked on every synthetic case. '
            'It retains the alternative starting supports but uses only one sweep per branch. '
            'The extra passes have no measured quality benefit in these samples; their '
            'additional cost cannot be justified by these outcomes alone. These earlier '
            'cohorts do not replace fresh qualification.')
    for label, sample in samples:
        entries = sample['methods']
        lines.append(f"{label}: all {sum(e['audited'] for e in entries.values())} of "
            f"{sum(e['expected'] for e in entries.values())} outputs pass the independent audit. "
            f"Effective optional candidates range from {sample['optional_candidates']['minimum']:.0f} "
            f"to {sample['optional_candidates']['maximum']:.0f}.")
        c = sample['ranked_comparisons']['ExactRanked']
        lines.append(f"Ranked-started Exact improves {c['higher']} incumbents; its maximum "
                     f"delivered improvement is {max(0., c['gap_bp']['maximum']):.5f} bp.")
        diagnosis = sample['worst_support_diagnosis']
        if c['higher'] and diagnosis['audit']['passed']:
            lines.append(f"In worst case {diagnosis['case'].removeprefix('scarcity_')}, "
                f"post-hoc sizing of Exact's support with the unchanged QP gives "
                f"{diagnosis['conditional_te_bp']:.5f} bp versus Ranked's "
                f"{diagnosis['ranked_te_bp']:.5f} bp. The support "
                f"{'was' if diagnosis['visited_by_ranked'] else 'was not'} in Ranked's trial log. "
                'This diagnosis does not alter benchmark outputs.')
        for method in ('ExactBase', 'ExactRanked'):
            sources = ', '.join(f"{key.replace('_', ' ')}: {value}" for key, value in
                                entries[method]['delivered_sources'].items())
            lines.append(f"{method} delivery attribution is {sources}. "
                         'A callback may reproduce a submitted start; a retained start is not '
                         'independent discovery or an optimality certificate.')
    if 'previous_comparison' in summary:
        old = summary['previous_comparison']
        lines.append(f"Against the archived pre-fix Ranked outputs on identical development inputs, "
            f"the corrected search improves {old['improved']} cases, ties {old['equal']} and worsens "
            f"{old['worsened']} at the descriptive band. Mean TE falls by {old['mean_reduction_bp']:.5f} "
            f"bp; the maximum reduction is {old['maximum_reduction_bp']:.5f} bp. "
            f"All {old['old_misses_recovered']} of the {old['old_misses']} previously recorded "
            'ExactRanked disadvantages are recovered against those archived incumbents. '
            'Fresh Exact calls above, rather than the archived reference, determine the current comparison.')
    lines += ['The extension retains the original completed search, so guidance failure or an inferior '
        'alternate branch cannot degrade that incumbent. It changes support exploration, not sizing '
        'constraints or tolerance. Complete replacement sweeps and multiple perspective solves add cost; '
        'the original 24-trial budget is not presented as the total extension work. Calls in each '
        'reported rerun are serial, with no concurrent verification launched by the study runner. '
        'Uncontrolled workstation activity remains possible. ExactRanked end-to-end records include '
        'Ranked generation. These experiments establish neither a universal speed advantage nor '
        'global optimality, and the holdout does not establish investment performance.']
    if compact:
        stop = lines.index(r'\subsection{Earlier scarce-ticket cohorts}')
        lines = lines[:stop]
        lines[0] = r'\section{Controlled input sensitivity}\label{sec:robustness}'
        lines[1] = ('These retained conditional experiments use only public decision arrays. '
                    'All treatments preserve the objective form, minimum trades, holding floors '
                    'and audit tolerances. Earlier scarce-ticket development results and their '
                    'complete summaries remain in the replication archive.')
    return '\n\n'.join(lines)+'\n'


def plot_sensitivity(frame, path):
    """Plot all paired allocation shifts; points represent decisions, not independent samples."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9., 3.1), sharey=True)
    labels = ['Target', 'Covariance', 'Joint']
    for ax, domain in zip(axes, ('fixed', 'rebuilt_corridor')):
        for level, color, offset in ((0, '#287c9a', -.12), (1, '#c67b32', .12)):
            for position, variant in enumerate(('target', 'covariance', 'joint')):
                values = frame[(frame.domain == domain) & (frame.variant == variant)
                               & (frame.level == level)].half_l1.dropna().to_numpy()*100
                jitter = np.linspace(-.065, .065, len(values))
                ax.scatter(position+offset+jitter, values, s=12, alpha=.7, color=color,
                           label=('Small shock' if level == 0 else 'Large shock') if position == 0 else None)
        ax.set_xticks(range(3), labels)
        ax.set_title('Fixed execution domain' if domain == 'fixed' else 'Rebuilt optional corridors')
        ax.grid(axis='y', alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Half-L1 allocation change (% NAV)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--previous', type=Path, action='append')
    parser.add_argument('--compact', action='store_true', help='Render sensitivity only for the main article')
    args = parser.parse_args()
    result = summarize(args.run, args.output, args.holdout, args.previous, compact=args.compact)
    print(json.dumps(result['scarcity']['ranked_comparisons'], indent=2))
