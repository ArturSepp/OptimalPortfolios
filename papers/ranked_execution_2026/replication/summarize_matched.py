"""Summarise complete public MIQP/V2 runs without hiding failures or start costs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .benchmark import sha256, verify_bundle
from .matched import method_names
from .run_benchmark import write_json


def distribution(values):
    """Keep the observed denominator and tail rather than silently filling failures."""
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, median=None, p95=None, maximum=None, minimum=None)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.quantile(a,.95)), maximum=float(a.max()), minimum=float(a.min()))


def native_diagnostics(name, item):
    """Preserve primary diagnostics separately from any delivered fallback."""
    diagnostics = item.get('diagnostics',{})
    return diagnostics.get('primary',{}).get('diagnostics',{}) if name == 'Policy' else diagnostics


def load_complete(root):
    """Refuse smoke runs, missing methods, altered inputs and inconsistent identities."""
    protocol = json.loads((root/'protocol.json').read_text())
    completion = json.loads((root/'complete.json').read_text())
    if not completion.get('full_sample') or completion['cases'] != len(protocol['cases']):
        raise ValueError('A complete frozen sample is required')
    if sha256(root/'inputs/manifest.json') != protocol['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    manifest = verify_bundle(root/'inputs')
    declared = protocol['cases']
    if len({r['id'] for r in declared}) != len(declared):
        raise ValueError('Duplicate case identity')
    if [{key:r[key] for key in ('id','cohort','sample','fingerprint')} for r in manifest['cases']] != declared:
        raise ValueError('Manifest identities changed')
    rows = []
    for entry in declared:
        record = json.loads((root/(entry['id']+'.json')).read_text())
        if any(record.get(k) != v for k,v in entry.items()):
            raise ValueError('Case identity changed')
        if record.get('prepared_fingerprint') and set(record['methods']) != set(method_names(protocol['specification'],record['cohort'])):
            raise ValueError('Missing method from prepared case')
        rows.append(record)
    return protocol, rows


def cohort_summary(rows, spec):
    """Report every expected case; comparisons use explicitly paired audited outputs."""
    names = method_names(spec, rows[0]['cohort'])
    broad = [r for r in rows if r['sample'] != 'tiny']
    result = dict(expected=len(rows), broad=len(broad), tiny=len(rows)-len(broad),
        prepared=sum(bool(r['methods']) for r in rows),
        optional_candidates=distribution([r['optional_candidates'] for r in broad if 'optional_candidates' in r]),
        preparation_seconds=distribution([r['preparation_seconds'] for r in rows]),
        methods={}, paired={}, bound_contradictions=[], failed_cases=[], enumerations=[])
    for name in names:
        attempted = [r['methods'][name] for r in rows if name in r['methods']]
        good = [m for m in attempted if m['audit']['passed']]
        natives = [native_diagnostics(name,m) for m in attempted]
        result['methods'][name] = dict(expected=len(rows), attempted=len(attempted), audited=len(good),
            te_bp=distribution([m['te_bp'] for m in good]),
            end_to_end_seconds=distribution([m['end_to_end_seconds'] for m in attempted]),
            seconds=distribution([m['seconds'] for m in attempted]),
            native_statuses=dict(Counter(str(d.get('native_status')) for d in natives)),
            native_gap_relative=distribution([d['native_gap_relative'] for d in natives if d.get('native_gap_relative') is not None]),
            delivered_sources=dict(Counter(str(d.get('delivered_source')) for d in natives)),
            native_absolute_gap_variance=distribution([d['native_incumbent']-d['native_bound']
                for d in natives if d.get('native_incumbent') is not None and d.get('native_bound') is not None]),
            native_gap_above_target_cases=[r['id'] for r in rows if name in r['methods'] and
                (native_diagnostics(name,r['methods'][name]).get('native_gap_relative') or 0.) > spec['gap_rel']],
            engines=dict(Counter(str(m.get('diagnostics',{}).get('engine')) for m in attempted)),
            fallback_reasons=dict(Counter(str(m.get('diagnostics',{}).get('fallback_reason')) for m in attempted)))
    band, objective = spec['descriptive_te_equality_bp'], spec['objective_tolerance']
    for left,right in [('V2','MIQP'),('V2','Policy'),('MIQP','Policy'),('V2','MIQPStartV2')]:
        paired = [(r['id'],r['methods'][left],r['methods'][right]) for r in rows
                  if all(r['methods'].get(m,{}).get('audit',{}).get('passed') for m in (left,right))]
        differences = [a['te_bp']-b['te_bp'] for _,a,b in paired]
        result['paired'][left+'_minus_'+right] = dict(n=len(paired),
            left_worse=sum(x>band for x in differences), ties=sum(abs(x)<=band for x in differences),
            left_better=sum(x < -band for x in differences), gap_bp=distribution(differences),
            max_weight_difference=distribution([float(np.max(np.abs(np.array(a['weights'])-b['weights']))) for _,a,b in paired]),
            cases=[dict(id=case, gap_bp=a['te_bp']-b['te_bp']) for case,a,b in paired if abs(a['te_bp']-b['te_bp'])>band])
    for row in rows:
        valid = [m for m in row['methods'].values() if m['audit']['passed']]
        best = min(((m['te_bp']/1e4)**2 for m in valid), default=None)
        if not row['methods']:
            result['failed_cases'].append(dict(id=row['id'],method='preparation',detail=row['preparation']))
        for name,item in row['methods'].items():
            if not item['audit']['passed']:
                result['failed_cases'].append(dict(id=row['id'],method=name,audit=item['audit'],error=item.get('error')))
            native = native_diagnostics(name,item)
            if best is not None and native.get('native_bound') is not None and native['native_bound']>best+objective:
                result['bound_contradictions'].append(dict(id=row['id'],method=name,bound=native['native_bound'],best=best))
        if 'enumeration' in row:
            enumeration = row['enumeration']
            gaps = {name:item['te_bp']-enumeration['te_bp'] for name,item in row['methods'].items()
                    if item['audit']['passed'] and enumeration['te_bp'] is not None}
            result['enumerations'].append(dict(id=row['id'],complete=enumeration['complete'],
                unresolved=enumeration['unresolved'],supports=len(enumeration['trials']),gaps_bp=gaps))
    return result


def render_tex(summary):
    """Generate the concise headline tables directly from complete retained records."""
    lines = [r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{Released OP 9.2.0 on unchanged public inputs and a fresh holdout. Synthetic rows include the tiny controls. TE is annual basis points; E2E time includes common preparation.}\label{tab:matched}\label{tab:joint}',
        r'\begin{tabular}{llrrrr}\toprule',
        r'Sample & Method & Audited & Mean TE & Median s & 95th pct. s\\\midrule']
    for cohort,label in [('ETF','ETF'),('Legacy','Existing synthetic'),('Holdout','Fresh holdout')]:
        group = summary['cohorts'][cohort]
        for method,display in [('MIQP','MIQP'),('V2','V2'),('Policy','Policy')]:
            item = group['methods'][method]
            fmt = lambda x: '--' if x is None else f'{x:.3f}'
            lines.append(f"{label} & {display} & {item['audited']}/{group['expected']} & "
                f"{fmt(item['te_bp']['mean'])} & {fmt(item['end_to_end_seconds']['median'])} & "
                f"{fmt(item['end_to_end_seconds']['p95'])}"+r'\\')
        lines.append(r'\addlinespace')
    lines.extend([r'\bottomrule\end{tabular}\end{table}',
        r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{Paired V2 minus independent MIQP tracking error. Positive gaps favour MIQP. The descriptive equality band is $10^{-4}$ annual bp, not a feasibility tolerance.}\label{tab:paired}',
        r'\begin{tabular}{lrrrrr}\toprule',r'Sample & Paired & V2 lower & Ties & V2 higher & Max gap bp\\\midrule'])
    for cohort,label in [('ETF','ETF'),('Legacy','Existing synthetic'),('Holdout','Fresh holdout')]:
        p = summary['cohorts'][cohort]['paired']['V2_minus_MIQP']
        lines.append(f"{label} & {p['n']} & {p['left_better']} & {p['ties']} & {p['left_worse']} & {p['gap_bp']['maximum']:.6f}"+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{table}'])
    return '\n'.join(lines)+'\n'


def plot_comparison(summary, rows, path):
    """Use QIS to display timing and every fresh broad-case quality difference."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import qis
    names = {'ETF':'ETF','Legacy':'Existing','Holdout':'Fresh'}
    timing = pd.DataFrame({m:[summary['cohorts'][c]['methods'][m]['end_to_end_seconds']['median']
        for c in names] for m in ('MIQP','V2','Policy')},index=list(names.values()))
    observations = [dict(candidates=r['optional_candidates'],
        gap=r['methods']['V2']['te_bp']-r['methods']['MIQP']['te_bp'],pattern=r['target_pattern'])
        for r in rows if r['cohort']=='Holdout' and r['sample']=='broad' and
        all(r['methods'].get(m,{}).get('audit',{}).get('passed') for m in ('V2','MIQP'))]
    fig,axes = plt.subplots(1,2,figsize=(8,3.8))
    qis.plot_bars(timing,stacked=False,x_rotation=0,ax=axes[0],fontsize=10,
        ylabel='Median E2E seconds (log scale)',legend_loc='upper left',
        colors=['#287c9a','#b36b25','#172b4d'],yvar_format='{:,.2f}')
    axes[0].set_yscale('log')
    axes[0].set_ylim(.05,10.)
    qis.plot_scatter(pd.DataFrame(observations),x='candidates',y='gap',hue='pattern',
        order=0,full_sample_order=None,add_universe_model_label=False,
        add_hue_model_label=False,xvar_format='{:,.0f}',yvar_format='{:,.3f}',
        xlabel='Effective optional candidates',ylabel='V2 minus MIQP annual TE (bp)',
        markersize=24,fontsize=10,legend_loc='upper left',ax=axes[1])
    axes[1].axhline(0.,color='#888888',linewidth=.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(root, output):
    """Publish summaries only after the full run and separate failure controls finish."""
    protocol, rows = load_complete(root)
    faults = json.loads((root/'fault_controls.json').read_text())
    spec = protocol['specification']
    expected_faults = {(next(r['id'] for r in rows if r['cohort']==c),f)
                       for c in ('ETF','Legacy','Holdout') for f in spec['faults']}
    if {(r['case'],r['fault']) for r in faults['records']} != expected_faults or len(faults['records']) != len(expected_faults):
        raise ValueError('Incomplete or duplicate injected controls')
    summary = dict(schema='matched-summary-920-1',protocol_sha256=sha256(root/'protocol.json'),
        summary_source_sha256=sha256(Path(__file__)),
        expected=len(rows),versions=protocol['versions'],
        cohorts={c:cohort_summary([r for r in rows if r['cohort']==c],spec) for c in ('ETF','Legacy','Holdout')},
        injected_controls=dict(expected=len(expected_faults),
          audited=sum(r['audit']['passed'] for r in faults['records']),
          engines=dict(Counter(r.get('engine','failed') for r in faults['records'])),
          reasons=dict(Counter(r.get('reason','failed') for r in faults['records']))))
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'summary.json',summary)
    (output/'results.tex').write_text(render_tex(summary),encoding='utf-8')
    plot_comparison(summary,rows,output/'joint_comparison.pdf')
    print(json.dumps({c:dict(expected=s['expected'],prepared=s['prepared'],
        failures=len(s['failed_cases']),paired=s['paired']['V2_minus_MIQP'],
        candidates=s['optional_candidates']) for c,s in summary['cohorts'].items()},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    summarize(args.run,args.output)
