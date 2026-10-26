"""Regenerate the manuscript's descriptive tables from a completed benchmark."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, median


METHODS = ('Ranked', 'Greedy', 'OneAnchor', 'ExactBase', 'ExactRanked')
COMPARISON_BP = 1e-4


def quantile(values, fraction):
    """Interpolate the sample quantile with the usual linear convention."""
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    lower = int(position)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower])


def aggregate(records):
    """Keep failed decisions in denominators and use common pairs for differences."""
    prepared = [r for r in records if r['methods']]
    if not prepared:
        raise ValueError('No prepared cases; there are no comparative results')
    complete = [r for r in prepared if all(
        r['methods'][m]['audit']['passed'] for m in METHODS)]
    if not complete:
        raise ValueError('No common audited cases')
    summary = dict(cases=len(records), prepared=len(prepared), common=len(complete),
        preparation_failures=[r['id'] for r in records if not r['methods']],
        comparison_tolerance_bp=COMPARISON_BP,
        preparation_seconds=dict(mean=mean(r['preparation_wall_seconds'] for r in records),
            median=median(r['preparation_wall_seconds'] for r in records)),
        base_mean_te_bp=mean(r['base_te_bp'] for r in complete),
        preparation_modes=dict(Counter(r['preparation'].get('mode', 'none') for r in records)),
        arms={}, comparisons={})
    for name in ('optional_candidates', 'mandatory', 'cap'):
        values = [r[name] for r in prepared]
        summary[name] = dict(min=min(values), median=median(values), max=max(values),
                             zero=sum(v == 0 for v in values))
    for method in METHODS:
        all_items = [r['methods'][method] for r in prepared]
        items = [r['methods'][method] for r in complete]
        times = [r['seconds'] for r in items]
        summary['arms'][method] = dict(valid=sum(r['audit']['passed'] for r in all_items),
            mean_te_bp=mean(r['te_bp'] for r in items),
            median_te_bp=median(r['te_bp'] for r in items),
            mean_seconds=mean(times), median_seconds=median(times),
            p95_seconds=quantile(times, .95), maximum_seconds=max(times),
            median_end_to_end_seconds=median(r['end_to_end_seconds'] for r in items),
            mean_tickets=mean(r['audit']['tickets'] for r in items),
            mean_gross_noncash_turnover=mean(r['audit']['turnover'] for r in items),
            statuses=dict(Counter(r['status'] for r in all_items)),
            delivered_sources=dict(Counter(r['diagnostics'].get('delivered_source', 'heuristic')
                                           for r in all_items)),
            native_audited=sum(r['diagnostics'].get('first_native_audited_feasible_seconds')
                               is not None for r in all_items),
            native_statuses=dict(Counter(r['diagnostics'].get('native_status', 'none')
                                         for r in all_items)))
    for other in METHODS[1:]:
        pairs = [(r['id'], r['methods']['Ranked']['te_bp']-r['methods'][other]['te_bp'])
                 for r in prepared if r['methods']['Ranked']['audit']['passed']
                 and r['methods'][other]['audit']['passed']]
        values = [v for _, v in pairs]
        summary['comparisons'][other] = dict(n=len(pairs),
            ranked_lower=sum(v < -COMPARISON_BP for v in values),
            tied=sum(abs(v) <= COMPARISON_BP for v in values),
            ranked_higher=sum(v > COMPARISON_BP for v in values),
            mean_ranked_minus_other_bp=mean(values), maximum_ranked_loss_bp=max(values),
            worst_case=max(pairs, key=lambda row: row[1])[0])
    worst = next(r for r in prepared
                 if r['id'] == summary['comparisons']['ExactRanked']['worst_case'])
    summary['worst_ranked_disadvantage'] = dict(id=worst['id'],
        optional_candidates=worst['optional_candidates'], mandatory=worst['mandatory'],
        base_te_bp=worst['base_te_bp'],
        ranked_te_bp=worst['methods']['Ranked']['te_bp'],
        exact_te_bp=worst['methods']['ExactRanked']['te_bp'],
        ranked_sizing_statuses=[t.get('native_status') for t in
            worst['methods']['Ranked']['diagnostics'].get('trials', []) if t.get('phase') == 'sizing'])
    return summary


def manuscript_results(summary):
    """Format only measured statistics, with denominators and bounded claims."""
    s = summary
    lines = [
        f"All {s['cases']} decisions were attempted. Common preparation succeeded on "
        f"{s['prepared']}; all five arms supplied independently audited portfolios on "
        f"{s['common']} decisions. The archive retains every attempted case, "
        "including any failed preparation. Comparisons use paired audited outputs.",
        "",
        "Table~\\ref{tab:results} uses the common audited subset. Tracking error is "
        "annual basis points; computation is elapsed seconds. The prepared base has "
        f"mean tracking error {s['base_mean_te_bp']:.3f} basis points. Median common "
        f"preparation takes {s['preparation_seconds']['median']:.3f} seconds.",
        "",
        r"\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Conditional execution quality and measured computation. "
        r"End-to-end (E2E) includes common preparation and, for ExactRanked, "
        r"Ranked start generation.}\label{tab:results}",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"Arm & Mean TE & Median TE & Median s & 95th pct. s & Median E2E s\\\midrule",
    ]
    for method in METHODS:
        a = s['arms'][method]
        lines.append(f"{method} & {a['mean_te_bp']:.3f} & {a['median_te_bp']:.3f} & "
                     f"{a['median_seconds']:.3f} & {a['p95_seconds']:.3f} & "
                     f"{a['median_end_to_end_seconds']:.3f}"+r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}", "",
        "Pairwise differences use a descriptive equality band of "
        r"$10^{-4}$ basis points, distinct from the portfolio feasibility "
        "tolerance. Negative differences favour Ranked. No independence-based "
        "significance test is applied to these serially related decisions.",
        r"\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Ranked relative to each comparator on paired audited decisions. "
        r"The last column is mean Ranked minus comparator TE, in basis points.}"
        r"\label{tab:pairs}",
        r"\begin{tabular}{lrrrrr}\toprule",
        r"Comparator & Pairs & Ranked lower & Equal & Ranked higher & Mean difference\\\midrule"]
    for other, c in s['comparisons'].items():
        lines.append(f"{other} & {c['n']} & {c['ranked_lower']} & {c['tied']} & "
                     f"{c['ranked_higher']} & {c['mean_ranked_minus_other_bp']:.5f}"+r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\end{table}", ""]
    o, m = s['optional_candidates'], s['mandatory']
    lines += [
        f"Optional candidate counts range from {o['min']} to {o['max']} "
        f"(median {o['median']:g}); {o['zero']} prepared cases have no optional "
        f"candidate. Mandatory reservations range from {m['min']} to {m['max']} "
        f"(median {m['median']:g}). These are the effective search dimensions, "
        "not the nominal 302-asset universe.",
        "",
    ]
    modes = s['preparation_modes']
    lines += [
        f"Preparation retains {modes.get('none', 0)} already audited bases, "
        f"repairs {modes.get('CORRIDOR', 0)} within the ordinary corridor, and "
        f"uses FULL preparation on {modes.get('FULL', 0)} decisions. The resulting "
        "pins and corridor exceptions are shared by every arm.",
        "",
        "The optional ten-ticket allowance never binds in this sample: every "
        "prepared case has at most five optional candidates. Mandatory "
        "interventions dominate the ticket count. Thus these data do not test "
        "scarce-ticket selection among a large optional candidate set.",
        "",
        f"Ranked improves on Greedy in {s['comparisons']['Greedy']['ranked_lower']} "
        f"cases and is equal in {s['comparisons']['Greedy']['tied']}. "
        f"OneAnchor and the full search have equal delivered objectives on "
        f"{s['comparisons']['OneAnchor']['tied']} paired cases. "
        f"ExactBase median call time is {s['arms']['ExactBase']['median_seconds']:.3f} "
        f"seconds versus Ranked's {s['arms']['Ranked']['median_seconds']:.3f}. "
        "The measured computation does not support a general heuristic speed advantage.",
        "",
    ]
    for method in ('ExactBase', 'ExactRanked'):
        a = s['arms'][method]
        sources = ', '.join(r"\emph{"+name.replace('_', ' ')+f"}}: {count}"
                            for name, count in sorted(a['delivered_sources'].items()))
        lines.append(f"{method} delivered-source attribution is {sources}. "
                     f"It recorded a native audited candidate on {a['native_audited']} "
                     f"of {s['prepared']} prepared cases. A callback or native return "
                     "may reproduce a submitted start; this count alone is not evidence "
                     "of an independently discovered improvement.")
        lines.append("")
    c = s['comparisons']['ExactRanked']
    lines += [
        f"The largest delivered TE disadvantage of Ranked against ExactRanked is "
        f"{max(0., c['maximum_ranked_loss_bp']):.5f} basis points "
        f"(case {c['worst_case'].removeprefix('case_')}). Delivered differences compare "
        "feasible incumbents, not certified global gaps. Full weights, numerical "
        "audits, native statuses and timing are retained per decision.",
        "",
    ]
    if c['maximum_ranked_loss_bp'] > COMPARISON_BP:
        w = s['worst_ranked_disadvantage']
        statuses = ', '.join(str(x).replace('_', ' ') for x in w['ranked_sizing_statuses'])
        lines += [
            f"That case has {w['mandatory']} mandatory reservations and "
            f"{w['optional_candidates']} optional candidates. The prepared base, "
            f"Ranked and ExactRanked have tracking errors {w['base_te_bp']:.5f}, "
            f"{w['ranked_te_bp']:.5f} and {w['exact_te_bp']:.5f} basis points, respectively. "
            f"Recorded Ranked sizing statuses are {statuses or 'unavailable'}. "
            "These diagnostics distinguish search scope, numerical outcomes and "
            "incumbent retention. A delivered objective gap alone does not "
            "identify its cause and must not be hidden by pooling incumbents.",
            "",
        ]
    lines += [
        "This frozen benchmark tests a distinct public-input cohort, but its "
        "low effective candidate counts limit conclusions about dense optional "
        "search. The comparison establishes the reported conditional outcomes; "
        "it does not establish universal superiority or realised investment gains.",
    ]
    return '\n'.join(lines)+'\n'


def summarize(run, output):
    """Require a completed full run and export machine-readable and TeX summaries."""
    completion = json.loads((run/'complete.json').read_text())
    if not completion['full_sample']:
        raise ValueError('A smoke run cannot supply manuscript results')
    records = [json.loads(p.read_text()) for p in sorted(run.glob('case_*.json'))]
    if len(records) != completion['cases']:
        raise ValueError('Missing completed case records')
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(records)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output/'results.tex').write_text(manuscript_results(summary), encoding='utf-8')
    with (output/'case_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['case', 'method', 'status', 'valid', 'te_bp', 'seconds', 'end_to_end_seconds',
                  'tickets', 'optional_tickets', 'turnover', 'delivered_source']
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for record in records:
            for method, item in record['methods'].items():
                writer.writerow(dict(case=record['id'], method=method, status=item['status'],
                    valid=item['audit']['passed'], te_bp=item.get('te_bp'),
                    seconds=item['seconds'], end_to_end_seconds=item['end_to_end_seconds'],
                    tickets=item['audit'].get('tickets'),
                    optional_tickets=item['audit'].get('optional_tickets'),
                    turnover=item['audit'].get('turnover'),
                    delivered_source=item['diagnostics'].get('delivered_source', 'heuristic')))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.output), indent=2))
