"""Report complete joint-MIQP comparisons without treating retained starts as proof."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from .benchmark import sha256
from .joint import methods
from .run_benchmark import write_json
from .summarize_robustness import describe


def sample_summary(cases, spec):
    """Keep failures in denominators and separate feasible quality from valid bounds."""
    arms = methods(spec)
    band = spec['descriptive_te_equality_bp']
    result = dict(expected=len(cases), prepared=sum(bool(c['methods']) for c in cases),
                  methods={}, comparisons={}, certificates=[], tiny=[])
    best = {c['id']:min((r['te_bp'] for r in c['methods'].values()
                        if r['audit']['passed']), default=None) for c in cases}
    for arm in arms:
        pairs = [(c, c['methods'][arm]) for c in cases if arm in c['methods']]
        valid = [(c, r) for c, r in pairs if r['audit']['passed']]
        result['methods'][arm] = dict(expected=len(cases), attempted=len(pairs),
            audited=len(valid), te_bp=describe([r['te_bp'] for _, r in valid]),
            gap_to_best_bp=describe([r['te_bp']-best[c['id']] for c, r in valid]),
            seconds=describe([r['seconds'] for _, r in pairs]),
            end_to_end_seconds=describe([r['end_to_end_seconds'] for _, r in pairs]),
            first_native_feasible_seconds=describe([r['diagnostics']['first_native_audited_feasible_seconds']
                for _, r in pairs if r.get('diagnostics', {}).get('first_native_audited_feasible_seconds') is not None]),
            allowance_overruns=sum(r.get('time_limit_seconds') is not None
                and r['seconds'] > r['time_limit_seconds'] for _, r in pairs),
            ticket_binding=sum(r['audit'].get('optional_tickets') == c['budget'] for c, r in valid),
            statuses=dict(Counter(r['status'] for _, r in pairs)),
            native_statuses=dict(Counter(r.get('diagnostics', {}).get('native_status', 'none')
                                         for _, r in pairs)),
            delivered_sources=dict(Counter(r.get('diagnostics', {}).get('delivered_source', 'heuristic')
                                           for _, r in pairs)),
            native_bound_above_best=[dict(case=c['id'], native_bound=r['diagnostics']['native_bound'],
                best_delivered_variance=(best[c['id']]/1e4)**2,
                excess_te_bp=1e4*np.sqrt(r['diagnostics']['native_bound'])-best[c['id']]) for c, r in pairs
                if r.get('diagnostics', {}).get('native_bound') is not None
                and best[c['id']] is not None
                and r['diagnostics']['native_bound'] > (best[c['id']]/1e4)**2 + spec['objective_tolerance']])
        if arm != 'Ranked':
            matched = [(c, r, c['methods']['Ranked']) for c, r in valid
                       if c['methods']['Ranked']['audit']['passed']]
            gaps = [v2['te_bp']-r['te_bp'] for _, r, v2 in matched]
            result['comparisons'][arm] = dict(pairs=len(matched),
                v2_lower=sum(x < -band for x in gaps), equal=sum(abs(x) <= band for x in gaps),
                v2_higher=sum(x > band for x in gaps), v2_minus_mip_bp=describe(gaps),
                worse_cases=[dict(case=c['id'], gap_bp=g) for (c, _, _), g in zip(matched, gaps) if g > band])
    for c in cases:
        if best[c['id']] is not None:
            lower = max([r['lower_bound'] for r in c.get('relaxations', {}).values()]+[0.])
            consistent = lower <= (best[c['id']]/1e4)**2
            gap = best[c['id']]-1e4*float(np.sqrt(lower)) if consistent else None
            result['certificates'].append(dict(case=c['id'], lower_variance=lower,
                consistent=consistent, best_delivered_te_bp=best[c['id']], gap_bp=gap))
        if 'enumeration' in c:
            e = c['enumeration']
            result['tiny'].append(dict(case=c['id'], complete=e['complete'], unresolved=e['unresolved'],
                supports=len(e['trials']), audit=e['audit'], te_bp=e['te_bp'],
                differences_bp={a:r['te_bp']-e['te_bp'] for a,r in c['methods'].items()
                    if r['audit']['passed'] and e['audit']['passed']}))
    result['conservative_gap_bp'] = describe([r['gap_bp'] for r in result['certificates'] if r['consistent']])
    result['bound_inconsistencies'] = sum(not r['consistent'] for r in result['certificates'])
    result['bound_closed_at_band'] = sum(r['consistent'] and r['gap_bp'] <= band for r in result['certificates'])
    result['optional_candidates'] = describe([c['optional_candidates'] for c in cases if c['methods']])
    result['ranked_gap_cases'] = [dict(case=c['id'], nominal=c['optional_assets'],
        gap_bp=c['methods']['Ranked']['te_bp']-best[c['id']]) for c in cases
        if c['methods'].get('Ranked', {}).get('audit', {}).get('passed')
        and c['methods']['Ranked']['te_bp']-best[c['id']] > band]
    result['budget_quality_changes'] = {start:sum(
        max(values)-min(values) > band for c in cases
        if (values := [c['methods'][f'Joint{start}{s:g}']['te_bp'] for s in spec['exact_seconds']
            if c['methods'].get(f'Joint{start}{s:g}', {}).get('audit', {}).get('passed')]))
        for start in ('Base','Ranked')}
    return result


def load_complete(run):
    """Reject missing arms, changed inputs, duplicate identities and smoke samples."""
    protocol = json.loads((run/'protocol.json').read_text())
    completion = json.loads((run/'complete.json').read_text())
    declared = {c['id']:c for c in protocol['cases']}
    if len(declared) != len(protocol['cases']) or protocol['methods'] != methods(protocol['specification']):
        raise ValueError('Duplicate cases or changed comparison arms in frozen protocol')
    paths = sorted(run.glob('joint_*.json'))
    if not completion['full_sample'] or completion['cases'] != len(declared) or len(paths) != len(declared):
        raise ValueError('Every frozen case is required for manuscript results')
    records = []
    for path in paths:
        row = json.loads(path.read_text())
        expected = declared.pop(row['id'], None)
        if expected is None or any(row.get(k) != v for k,v in expected.items()):
            raise ValueError('Frozen case identity changed')
        if sha256(run/'inputs'/(row['id']+'.npz')) != row['input_sha256']:
            raise ValueError('Frozen input bytes changed')
        if row['methods'] and set(row['methods']) != set(protocol['methods']):
            raise ValueError('Prepared case has a missing or undeclared arm')
        records.append(row)
    return protocol, records


def label(arm):
    """Use short reader labels while preserving reproducible machine identifiers."""
    return arm.replace('JointRanked', 'MIP-V').replace('JointBase', 'MIP-B').replace('Ranked', 'V2')


def number(value, digits=3):
    """Show missing outcomes explicitly rather than coercing them to zero."""
    return '--' if value is None else f'{value:.{digits}f}'


def render_tex(result):
    """Keep the headline comparison compact and qualify every certificate claim."""
    b, tiny = result['broad'], result['tiny']
    lines = [r'\section{Full joint mixed-integer benchmark}\label{sec:joint}',
        r'\label{sec:breadth}',
        'A frozen public synthetic grid crosses 24/72/144 nominal optional instruments, '
        '2/6 discretionary tickets, disjoint/overlapping groups and tight/wide cash and group bands. '
        'Seed 111001 supplies 24 paired instances; seed 112001 supplies two eight-instrument controls. '
        'The generator retains two forced exits, a 50 NAV bp minimum ticket and a 25 NAV bp holding floor. '
        'The V2 heuristic and rules remain frozen. This paired rerun corrects the MIQP reference\'s '
        'native feasibility setting and quadratic-cone scaling after this grid exposed a false optimal termination. '
        'It is a numerical regression comparison, not an untouched validation of the corrected reference. '
        'Factorial variants share random draws and are not independent market replications.',
        '',
        'Each joint MIQP optimises all admissible states and sizes simultaneously in the same prepared '
        'CORRIDOR domain. MIP-B starts from the common prepared incumbent; MIP-V receives only V2\'s '
        'audited proposal. Suffixes denote 10/60/600-second wrapper allowances. Every call starts afresh: '
        'incumbents never pass between budgets. Native calls use one thread, seed zero and the same '
        r'$10^{-4}$ relative-gap setting; native integer feasibility is $10^{-9}$. '
        'Budget and start orders alternate deterministically. '
        'The mathematical MIQP is solved in an algebraically equivalent, balanced conic representation.',
        '',
        f"All {b['expected']} broad cases were attempted; {b['prepared']} prepared. "
        f"Effective optional candidates range from {number(b['optional_candidates']['minimum'],0)} to "
        f"{number(b['optional_candidates']['maximum'],0)}. "
        'Table~\\ref{tab:joint} reports audited outputs over the full expected denominator. '
        'Tracking-error means use audited outputs; time summaries include every attempted arm. '
        'End-to-end (E2E) adds preparation to every arm and V2 generation to each MIP-V call.',
        '', r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{Full joint optimisation versus V2. TE is annual bp; E2E time is seconds.}\label{tab:joint}',
        r'\begin{tabular}{lrrrrr}\toprule',
        r'Arm & Audited & Mean TE & Median E2E & 95th pct. & V2 worse\\\midrule']
    for arm, row in b['methods'].items():
        worse = '--' if arm == 'Ranked' else str(b['comparisons'][arm]['v2_higher'])
        lines.append(f"{label(arm)} & {row['audited']}/{row['expected']} & {number(row['te_bp']['mean'])} & "
            f"{number(row['end_to_end_seconds']['median'])} & {number(row['end_to_end_seconds']['p95'])} & {worse}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}', '',
        r'\begin{figure}[htbp]\centering',
        r'\includegraphics[width=\linewidth]{figures/joint_comparison.pdf}',
        r'\caption{Left: median E2E computation at the 600-second wrapper allowance, including preparation '
        r'and V2 start generation. Right: each audited outcome above the best delivered TE on that case; '
        r'this pooled reporting reference is never supplied as a start. Points identify nominal '
        r'optional-universe size; the dashed line is the unchanged descriptive equality band.}\label{fig:joint}',
        r'\end{figure}', '',
        f"V2 exceeds the best audited delivery by more than the $10^{{-4}}$ bp descriptive band "
        f"on {len(b['ranked_gap_cases'])}/{b['expected']} cases; its largest delivered gap is "
        f"{number(b['methods']['Ranked']['gap_to_best_bp']['maximum'],6)} bp. "
        'A worse time-limited MIP delivery does not imply a worse mathematical optimum. '
        'A tie with MIP-V may simply retain V2\'s supplied incumbent. All native/retained/refined '
        'sources, first native audited feasibility times, failed outputs and allowance overruns '
        'are preserved in the numerical summary.',
        '',
        f"Changing the allowance changes delivered TE above the descriptive band on "
        f"{b['budget_quality_changes']['Base']} common-base and {b['budget_quality_changes']['Ranked']} "
        'V2-started cases. These allowances are upper limits, not enforced run durations. '
        f"At 600 seconds, V2 has lower TE than the common-base delivery on "
        f"{b['comparisons']['JointBase600']['v2_lower']}/{b['expected']} cases and ties on "
        f"{b['comparisons']['JointBase600']['equal']}/{b['expected']}; its largest advantage is "
        f"{number(max(0.,-b['comparisons']['JointBase600']['v2_minus_mip_bp']['minimum']),6)} bp. "
        f"MIP-V600 retains the supplied V2 start on "
        f"{b['methods']['JointRanked600']['delivered_sources'].get('requested_mip_start',0)} cases.",
        '',
        f"Across all six MIQP arms, {sum(len(r['native_bound_above_best']) for r in b['methods'].values())} "
        'native bounds exceed a better audited cross-arm objective at the unchanged '
        r'$10^{-12}$ variance comparison scale. Raw bounds and statuses are retained, '
        'not clipped to the incumbent. Consistency with delivered witnesses is necessary '
        'but does not independently certify the native branch-and-bound result.',
        '',
        f"Separate conservative relaxation bounds close within the descriptive band on "
        f"{b['bound_closed_at_band']}/{b['expected']} cases. Their median/maximum best-delivery gaps "
        f"are {number(b['conservative_gap_bp']['median'])}/{number(b['conservative_gap_bp']['maximum'])} bp; "
        f"there are {b['bound_inconsistencies']} upper/lower inconsistencies. Native optimal status "
        'and solver-reported MIP gaps remain diagnostics, not independent global certificates.',
        '',
        f"Independent full-weight QP enumeration covers {sum(r['supports'] for r in tiny['tiny'])} "
        f"supports across {len(tiny['tiny'])} tiny cases; "
        f"{sum(r['complete'] for r in tiny['tiny'])} enumerations are complete and "
        f"{sum(r['unresolved'] for r in tiny['tiny'])} supports remain unresolved. "
        f"The maximum absolute delivered difference across all tiny arms is "
        f"{max(abs(v) for r in tiny['tiny'] for v in r['differences_bp'].values()):.2e} bp. "
        'These are floating-point numerical references, not exact-arithmetic proofs.',
        '',
        'The earlier development, holdout and search-ablation records remain in the replication '
        'package. This rerun changes only the reference\'s internal numerical setting, conic scaling and '
        'diagnostic checks, preserving the heuristic and production policy. Computational '
        'cost is measured rather than assumed: the experiment establishes neither a universal '
        'speed advantage nor realised investment performance.']
    return '\n'.join(lines)+'\n'


def plot_comparison(cases, spec, path):
    """Plot scientific runtime/quality observations from the retained result records."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10., 3.5))
    sizes = sorted({c['optional_assets'] for c in cases})
    for arm, text, color, marker in [('Ranked','V2','#172b4d','o'),
            ('JointBase600','Joint MIQP, base start','#b36b25','s'),
            ('JointRanked600','Joint MIQP, V2 start','#287c9a','^')]:
        medians = [np.median([c['methods'][arm]['end_to_end_seconds'] for c in cases
            if c['optional_assets'] == n and arm in c['methods']]) for n in sizes]
        axes[0].plot(sizes, medians, marker=marker, color=color, label=text)
    axes[0].set(xlabel='Nominal optional instruments', ylabel='Median E2E seconds', yscale='log')
    axes[0].set_xticks(sizes)
    axes[0].legend(frameon=False, fontsize=8)
    arms = methods(spec)
    for n, color in zip(sizes, ['#287c9a','#b36b25','#8663a6']):
        for position, arm in enumerate(arms):
            values = []
            for c in cases:
                row = c['methods'].get(arm, {})
                if c['optional_assets'] == n and row.get('audit', {}).get('passed'):
                    best = min(r['te_bp'] for r in c['methods'].values() if r['audit']['passed'])
                    values.append(row['te_bp']-best)
            axes[1].scatter(position+np.linspace(-.14,.14,len(values)), values,
                s=14, alpha=.75, color=color, label=str(n) if position == 0 else None)
    axes[1].axhline(spec['descriptive_te_equality_bp'], ls='--', color='#888888', lw=.8)
    axes[1].set_xticks(range(len(arms)), [label(a).replace('MIP-','') for a in arms], rotation=35)
    axes[1].set_ylabel('Annual TE above best delivered (bp)')
    axes[1].legend(title='Nominal size', frameon=False, fontsize=8, title_fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(run, output):
    """Validate all records, then publish the compact table, plot and complete summary."""
    protocol, cases = load_complete(run)
    spec = protocol['specification']
    broad = [c for c in cases if c['sample'] == 'broad']
    result = dict(protocol_sha256=sha256(run/'protocol.json'), specification=spec,
        records_sha256={p.name:sha256(p) for p in sorted(run.glob('joint_*.json'))},
        broad=sample_summary(broad,spec),
        tiny=sample_summary([c for c in cases if c['sample'] == 'tiny'],spec))
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'summary.json',result)
    (output/'joint_results.tex').write_text(render_tex(result),encoding='utf-8')
    plot_comparison(broad,spec,output/'joint_comparison.pdf')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.run,args.output)
