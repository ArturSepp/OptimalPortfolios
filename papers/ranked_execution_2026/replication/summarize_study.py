"""Summarise complete prospective comparisons, preserving failures and open gaps."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from optimalportfolios.execution.v2.domain import variance,compile_states
from .benchmark import sha256
from .run_benchmark import write_json
from .study_design import load_problem,original_audit
from .study_run import checked_manifest
from .summarize_matched import distribution


def paired(rows,left,right,band):
    """Use paired audited deliveries only and retain both signs and missing pairs."""
    eligible=[r for r in rows if left in r['methods'] and right in r['methods']]
    complete=[r for r in eligible if all(r['methods'][arm]['audit']['passed'] for arm in (left,right))]
    gaps=[r['methods'][left]['te_bp']-r['methods'][right]['te_bp'] for r in complete]
    return dict(expected=len(eligible),paired=len(complete),missing=len(eligible)-len(complete),
        left_better=sum(x < -band for x in gaps),ties=sum(abs(x)<=band for x in gaps),
        left_worse=sum(x>band for x in gaps),gap_bp=distribution(gaps),
        cases=[dict(id=r['id'],gap_bp=x) for r,x in zip(complete,gaps)])


def method_summary(items):
    """Keep timing, successful deliveries, native bounds and source attribution separate."""
    valid=[m for m in items if m['audit']['passed']]
    diagnostics=[m.get('diagnostics',{}) for m in items]
    def observed(key):
        """Describe observed measurements without inventing values for killed workers."""
        return distribution([m[key] for m in items if m.get(key) is not None])
    return dict(expected=len(items),audited=len(valid),
        statuses=dict(Counter(m['status'] for m in items)),
        te_bp=distribution([m['te_bp'] for m in valid]),
        tickets=distribution([m['audit']['tickets'] for m in valid if 'tickets' in m['audit']]),
        optional_tickets=distribution([m['audit']['optional_tickets'] for m in valid
                                       if 'optional_tickets' in m['audit']]),
        seconds=observed('seconds'),end_to_end_seconds=observed('end_to_end_seconds'),
        process_seconds=observed('process_seconds'),
        research_pipeline_seconds=observed('research_pipeline_seconds'),
        peak_process_memory_bytes=observed('peak_process_memory_bytes'),
        native_statuses=dict(Counter(str(d.get('native_status')) for d in diagnostics)),
        delivered_sources=dict(Counter(str(d.get('delivered_source')) for d in diagnostics)),
        native_gap_relative=distribution([d['native_gap_relative'] for d in diagnostics
                                         if d.get('native_gap_relative') is not None]),
        native_absolute_gap_variance=distribution([d['native_incumbent']-d['native_bound']
            for d in diagnostics if d.get('native_incumbent') is not None and d.get('native_bound') is not None]),
        delivery_minus_native_bound=distribution([d['variance']-d['native_bound'] for d in diagnostics
            if d.get('variance') is not None and d.get('native_bound') is not None]),
        first_native_audited_seconds=distribution([d['first_native_audited_feasible_seconds']
            for d in diagnostics if d.get('first_native_audited_feasible_seconds') is not None]),
        native_witnesses=sum(any(e.get('audit',{}).get('passed') and
            (e.get('source','').startswith('native') or e.get('source')=='boundary_refinement')
            for e in d.get('incumbent_events',[])) for d in diagnostics),
        rejected_native_events=sum(sum(not e.get('audit',{}).get('passed',False)
            for e in d.get('incumbent_events',[]) if e.get('source','').startswith('native')) for d in diagnostics))


def research_pipeline_seconds(method,preparation):
    """Charge both observed worker processes, including their two import phases."""
    times=[item.get('process_seconds') for item in (method,preparation)]
    return None if any(value is None for value in times) else sum(times)


def load_complete(root):
    """Require every frozen arm and independently recheck weights and recorded risks."""
    root=Path(root); manifest=checked_manifest(root)
    completion=json.loads((root/'complete.json').read_text())
    if not completion.get('full_sample') or completion['cases']!=len(manifest['cases']):
        raise ValueError('Complete prospective sample required')
    rows=[]; identities={}; contradictions=[]
    tolerance=manifest['specification']['controls']['variance_comparison_tolerance']
    for case in manifest['cases']:
        raw=load_problem(root/'inputs',case)
        prep_path=root/f'{case["id"]}--prepare.json'
        preparation=json.loads(prep_path.read_text()); identities[prep_path.name]=sha256(prep_path)
        prepared=preparation.get('prepared')
        problem=load_problem(root/'prepared',prepared) if prepared else None
        methods={}; domains={}; structure={}
        for arm in case['arms']:
            path=root/f'{case["id"]}--{arm}.json'
            item=json.loads(path.read_text()); identities[path.name]=sha256(path)
            if item['case']!=case['id'] or item['arm']!=arm:
                raise ValueError('Changed result identity')
            # Derived attribution only; the hashed raw record is never rewritten.
            item['research_pipeline_seconds']=research_pipeline_seconds(item,preparation)
            domain=manifest['specification']['arms'][arm].get('domain','CORRIDOR')
            domains[arm]=domain
            if domain not in structure:
                states=compile_states(problem if problem is not None else raw,domain)
                selected=problem if problem is not None else raw
                optional=[row for i,row in enumerate(states)
                    if i!=selected.cash and not selected.mandatory[i]]
                structure[domain]=dict(optional_candidates=sum(any(s.ticket for s in row) for row in optional),
                    ticket_states=sum(sum(s.ticket for s in row) for row in optional))
            if item['audit']['passed']:
                if problem is None: raise ValueError('Delivery without common preparation')
                domain=manifest['specification']['arms'][arm].get('domain','CORRIDOR')
                actual=original_audit(problem,item['weights'],domain)
                if not actual['passed']: raise ValueError('Retained delivery failed original audit')
                value=variance(problem,item['weights'])
                if item['te_bp']<0 or abs((item['te_bp']/1e4)**2-value)>1e-12:
                    raise ValueError('Recorded risk differs from authoritative covariance')
            methods[arm]=item
        # Never compare a CORRIDOR bound with a FULL-only feasible witness.
        for arm,item in methods.items():
            bound=item.get('diagnostics',{}).get('native_bound')
            if bound is None or problem is None: continue
            domain=manifest['specification']['arms'][arm].get('domain','CORRIDOR')
            candidates=[(name,variance(problem,other['weights'])) for name,other in methods.items()
                if other['audit']['passed'] and original_audit(problem,other['weights'],domain)['passed']]
            if candidates:
                witness,value=min(candidates,key=lambda entry:entry[1])
                if bound>value+tolerance:
                    contradictions.append(dict(case=case['id'],arm=arm,domain=domain,
                        native_bound=bound,witness=witness,witness_variance=value,excess=bound-value))
        rows.append(dict(id=case['id'],family=case['family'],structure=case['structure'],
            design={k:case[k] for k in ('universe','budget','profile') if k in case},
            method_domains=domains,domain_structure=structure,
            preparation=preparation,methods=methods,raw_fingerprint=case['fingerprint']))
    if sum(len(r['methods']) for r in rows)!=manifest['planned_deliveries']:
        raise ValueError('Changed denominator')
    return manifest,rows,identities,contradictions


def corridor_reference(row):
    """Use all audited Corridor arms, excluding different-policy Full comparisons."""
    values={arm:item['te_bp'] for arm,item in row['methods'].items()
        if item['audit']['passed'] and row['method_domains'][arm]=='CORRIDOR'}
    if not values: return dict(te_bp=None,methods=[])
    best=min(values.values())
    return dict(te_bp=best,methods=[arm for arm,value in values.items() if value==best])


def summarize(root):
    """Generate cohort and paired statistics only after the complete-sample gate."""
    manifest,rows,identities,contradictions=load_complete(root)
    band=manifest['specification']['controls']['descriptive_te_band_bp']
    summary=dict(schema='execution-research-1',manifest_sha256=sha256(Path(root)/'manifest.json'),
        expected_cases=len(rows),expected_deliveries=manifest['planned_deliveries'],
        summary_source_sha256=sha256(Path(__file__)),
        result_hashes=identities,bound_contradictions=contradictions,cohorts={},failures=[],cases=[])
    for family in ('scaling','external','counterexample'):
        selected=[r for r in rows if r['family']==family]
        names=sorted({arm for r in selected for arm in r['methods']})
        summary['cohorts'][family]=dict(expected=len(selected),
            prepared=sum(r['preparation'].get('prepared') is not None for r in selected),
            preparation_seconds=distribution([r['preparation']['preparation_seconds'] for r in selected
                if r['preparation'].get('preparation_seconds') is not None]),
            preparation_process_seconds=distribution([r['preparation']['process_seconds'] for r in selected
                if r['preparation'].get('process_seconds') is not None]),
            optional_candidates=distribution([r['structure']['optional_candidates'] for r in selected]),
            methods={name:method_summary([r['methods'][name] for r in selected if name in r['methods']]) for name in names},
            paired={left+'_minus_'+right:paired(selected,left,right,band) for left,right in
                [('Ranked','Exact60'),('Kernel60','Exact60'),('Ranked','Kernel60'),
                 ('Exact10','Exact600'),('Exact60','Exact600'),('Dense10','Exact10'),('Full60','Exact60')]})
    for row in rows:
        summary['cases'].append(dict(id=row['id'],family=row['family'],design=row['design'],
            structure=row['structure'],domain_structure=row['domain_structure'],
            best_observed_corridor=corridor_reference(row),
            methods={arm:dict({k:item.get(k) for k in ('status','audit','te_bp','seconds',
                'process_seconds','research_pipeline_seconds','peak_process_memory_bytes')},
                native={k:item.get('diagnostics',{}).get(k) for k in ('native_status',
                    'native_bound','native_incumbent','native_gap_relative','variance',
                    'delivered_source','first_native_audited_feasible_seconds')})
                for arm,item in row['methods'].items()}))
        for arm,item in row['methods'].items():
            if not item['audit']['passed']:
                summary['failures'].append(dict(case=row['id'],arm=arm,status=item['status'],
                    audit=item['audit'],error=item.get('error')))
    return summary,rows


def render_tex(summary):
    """Produce concise comparison tables without treating missing deliveries as ties."""
    def fmt(value):
        """Render only observed finite statistics."""
        rendered='--' if value is None else f'{value:.3f}'
        return '0.000' if rendered=='-0.000' else rendered
    lines=[r'\begin{table}[htbp]\centering\small',
        r'\caption{Delivery and process cost in the prospective study. Exact and Kernel have 60-second internal allowances; Ranked retains released defaults with a 180-second process ceiling. Process time includes imports and reporting. Memory is the largest observed worker peak, with its observation count in parentheses; killed workers have no peak measurement.}\label{tab:scalingtime}',
        r'\begin{tabular}{llrrrr}\toprule',
        r'Sample & Method & Audited & Median s & 95th pct. s & Observed MiB\\\midrule']
    for cohort,label in [('scaling','Scaling'),('external','External risk')]:
        for arm,display in [('Exact60','Exact'),('Ranked','Ranked'),('Kernel60','Kernel')]:
            item=summary['cohorts'][cohort]['methods'][arm]
            peak=item['peak_process_memory_bytes']['maximum']
            lines.append(f'{label} & {display} & {item["audited"]}/{item["expected"]} & '
                f'{fmt(item["process_seconds"]["median"])} & {fmt(item["process_seconds"]["p95"])} & '
                f'{fmt(None if peak is None else peak/2**20)} '
                f'({item["peak_process_memory_bytes"]["n"]})'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{table}',r'\begin{table}[htbp]\centering\small',
        r'\caption{Paired audited TE differences, left method minus Exact60, in annual bp. Negative differences favour the left method. Open Exact gaps mean these are incumbent comparisons. The equality band is $10^{-4}$ bp.}\label{tab:scalingquality}',
        r'\begin{tabular}{llrrrrrr}\toprule',
        r'Sample & Left & Paired & Lower & Ties & Higher & Min bp & Max bp\\\midrule'])
    for cohort,label in [('scaling','Scaling'),('external','External risk')]:
        for left,display in [('Ranked','Ranked'),('Kernel60','Kernel')]:
            item=summary['cohorts'][cohort]['paired'][left+'_minus_Exact60']
            lines.append(f'{label} & {display} & {item["paired"]}/{item["expected"]} & '
                f'{item["left_better"]} & {item["ties"]} & {item["left_worse"]} & '
                f'{fmt(item["gap_bp"]["minimum"])} & {fmt(item["gap_bp"]["maximum"])}'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{table}'])
    return '\n'.join(lines)+'\n'


def render_historical(summary):
    """Relabel only the presentation of immutable earlier results, retaining both signs."""
    lines=[r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{Retained 313-case cohort: Ranked minus independent Exact TE, in annual bp. Negative differences favour Ranked. The equality band is $10^{-4}$ bp; the smaller ETF differences remain visible.}\label{tab:historical}',
        r'\begin{tabular}{lrrrrrr}\toprule',
        r'Sample & Paired & Ranked lower & Ties & Ranked higher & Minimum & Maximum\\\midrule']
    for cohort,label in [('ETF','ETF'),('Legacy','Existing synthetic'),('Holdout','Fresh holdout')]:
        pair=summary['cohorts'][cohort]['paired']['V2_minus_MIQP']
        minimum=f'{pair["gap_bp"]["minimum"]:.6f}'.replace('-0.000000','0.000000')
        maximum=f'{pair["gap_bp"]["maximum"]:.6f}'.replace('-0.000000','0.000000')
        lines.append(f'{label} & {pair["n"]} & {pair["left_better"]} & {pair["ties"]} & '
            f'{pair["left_worse"]} & {minimum} & {maximum}'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{table}'])
    return '\n'.join(lines)+'\n'


def render_controls(summary):
    """Display all prespecified long-budget, representation and domain controls."""
    rows=[r for r in summary['cases'] if 'Exact600' in r['methods']]
    def te(item):
        """Show missing portfolios explicitly rather than as a zero objective."""
        return f'{item["te_bp"]:.3f}' if item['audit']['passed'] else '--'
    lines=[r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
        r'\caption{All prespecified 30-ticket, tight-profile controls. TE is annual bp. Full60 changes the policy domain; its difference is not search error on the Corridor problem. The final column reports the native relative gap for Exact600, which can differ from the gap of the audited delivery.}\label{tab:studycontrols}',
        r'\begin{tabular}{rrrrrrr}\toprule',
        r'Names & Exact10 & Exact60 & Exact600 & Dense10 & Full60 & Native gap \%\\\midrule']
    for row in rows:
        methods=row['methods']; gap=methods['Exact600']['native'].get('native_gap_relative')
        values=[str(row['design']['universe'])]+[te(methods[arm]) for arm in
            ('Exact10','Exact60','Exact600','Dense10','Full60')]
        values.append('--' if gap is None else f'{100*gap:.3f}')
        lines.append(' & '.join(values)+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{table}'])
    return '\n'.join(lines)+'\n'


def numerical_bounds(summary):
    """Compare native bounds with audited witnesses on identical mechanism inputs."""
    rows=[]
    for item in summary['cones']:
        same=[other for other in summary['cones'] if other['optional']==item['optional']]
        if any(other['input_fingerprint']!=item['input_fingerprint'] for other in same):
            raise ValueError('Equivalent-representation witnesses require identical inputs')
        witnesses=[other['variance'] for other in same if other['audit']['passed']]
        bound=item['native'].get('native_bound')
        best=min(witnesses) if witnesses else None
        rows.append(dict(optional=item['optional'],balanced=item['balanced'],
            variance=item['variance'],native_bound=bound,best_audited_variance=best,
            bound_minus_best=None if bound is None or best is None else bound-best))
    return rows


def render_numerics(summary):
    """Expose original-unit native bound discrepancies, including unsuccessful controls."""
    lines=[r'\begin{table}[htbp]\centering\small',
        r'\caption{Equivalent cone controls under unchanged native settings. Variances are in original annual units. Excess is the native bound minus the best audited witness from the two equivalent graphs, in units of $10^{-12}$ variance. A native optimal label does not remove a contradictory witness.}\label{tab:numericcones}',
        r'\begin{tabular}{rlrrr}\toprule',
        r'Optional & Graph & Delivered variance & Native bound & Excess\\\midrule']
    for row in numerical_bounds(summary):
        bound='--' if row['native_bound'] is None else f'{row["native_bound"]:.12f}'
        excess='--' if row['bound_minus_best'] is None else f'{row["bound_minus_best"]/1e-12:.3f}'
        lines.append(f'{row["optional"]} & {"Balanced" if row["balanced"] else "Unbalanced"} & '
            f'{row["variance"]:.12f} & {bound} & {excess}'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{table}'])
    return '\n'.join(lines)+'\n'


def plot_difficulty(summary,path):
    """Show incumbent quality by declared cell, retaining missing-delivery counts."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd
    import qis
    arms=[('Exact60','Exact','#3563a6'),('Ranked','Ranked','#b14b3c'),('Kernel60','Kernel','#29845b')]
    rows=[r for r in summary['cases'] if r['family']=='scaling']
    fig,axes=plt.subplots(2,3,figsize=(7.5,6),sharey=True)
    for a,profile in enumerate(('wide','tight')):
        for b,n in enumerate((300,1000,3000)):
            selected=[r for r in rows if r['design']['profile']==profile and r['design']['universe']==n]
            points=[]; colors=[]; counts=[]
            for arm,label,color in arms:
                passed=0
                for row in selected:
                    item=row['methods'][arm]
                    if item['audit']['passed']:
                        passed+=1
                        points.append(dict(B=row['design']['budget'],method=label,
                            gap=item['te_bp']-row['best_observed_corridor']['te_bp']))
                counts.append(f'{label} {passed}/{len(selected)}')
                if passed: colors.append(color)
            ax=axes[a,b]
            if points:
                qis.plot_scatter(pd.DataFrame(points),x='B',y='gap',hue='method',
                    colors=colors,order=0,full_sample_order=None,add_universe_model_label=False,
                    add_hue_model_label=False,xvar_format='{:,.0f}',yvar_format='{:,.1f}',
                    xlabel='Optional ticket allowance',ylabel='TE above best observed (bp)' if b==0 else '',
                    title=None,markersize=32,fontsize=9.5,
                    legend_loc=None,ax=ax)
            ax.set_title(f'{n:,} names; {profile}\n'+', '.join(counts),fontsize=8.5)
            ax.set_xticks([10,30,100])
            ax.axhline(0.,color='#888888',linewidth=.6)
    # QIS formats each panel before later shared-axis limits expand. Set final
    # common positions and labels together, so early-panel ticks do not bunch.
    largest=max(item['te_bp']-row['best_observed_corridor']['te_bp']
        for row in rows for arm,_,_ in arms for item in [row['methods'][arm]]
        if item['audit']['passed'])
    upper=max(1.,np.ceil(largest/25.)*25.)
    ticks=np.linspace(0.,upper,5)
    for a in range(2):
        for b in range(3):
            ax=axes[a,b]
            ax.set_ylim(-.03*upper,1.05*upper)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f'{value:g}' for value in ticks],fontsize=9)
            ax.tick_params(axis='y',labelleft=(b==0))
    # A global legend retains methods missing from the first panel.
    handles=[Line2D([],[],marker='o',linestyle='',color=color,label=label)
             for _,label,color in arms]
    fig.legend(handles=handles,loc='upper center',ncol=len(arms),frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(path); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--numerics',type=Path,help='Completed separate mechanism-control directory')
    args=parser.parse_args()
    summary,rows=summarize(args.root)
    write_json(args.root/'summary.json',summary)
    (args.root/'study_results.tex').write_text(render_tex(summary),encoding='utf-8')
    (args.root/'study_controls.tex').write_text(render_controls(summary),encoding='utf-8')
    historical=Path(__file__).with_name('matched_expected_summary.json')
    if sha256(historical)!='810425d7c6504a78d041227bc59f2effa1749c947d78fcd070c822b4d49ef3a4':
        raise ValueError('Retained historical summary has changed')
    (args.root/'historical_results.tex').write_text(
        render_historical(json.loads(historical.read_text())),encoding='utf-8')
    plot_difficulty(summary,args.root/'difficulty_comparison.pdf')
    if args.numerics is not None:
        mechanisms=json.loads((args.numerics/'summary.json').read_text())
        (args.root/'numerical_results.tex').write_text(render_numerics(mechanisms),encoding='utf-8')
