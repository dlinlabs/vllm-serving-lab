"""Build fixed development summaries and figures; no fitting or model selection."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def aggregate(report, split, horizon, subset='all'):
    rows = [r for r in report['scores'] if r['split'] == split and r['horizon_s'] == horizon
            and r['subset'] == subset and r['target'] == 'cohort_ttft_met' and r['models']['direct_ridge']['n']]
    n = sum(r['models']['direct_ridge']['n'] for r in rows)
    return dict(split=split, horizon_s=horizon, subset=subset, n=n,
        rate_mae={m: sum(r['models'][m]['mae'] * r['models'][m]['n'] for r in rows) / n / horizon
                  for m in rows[0]['models']} if n else {})


def summarize(report, dataset):
    modes = ('chronological', 'fixed_to_scheduled', 'scheduled_to_fixed')
    result = dict(protocol=report['protocol'], acceptance=None, c_used=False, runtime=report['runtime'],
        aggregates=[aggregate(report, mode, h) for mode in modes for h in (1, 2, 5)],
        near_changes=[aggregate(report, 'chronological', h, 'near_cap_change') for h in (1, 2, 5)],
        per_run_five_seconds=[dict(run=r['run'], rate_mae={m:v['mae']/5 for m,v in r['models'].items()}) for r in report['scores']
            if r['split']=='chronological' and r['horizon_s']==5 and r['subset']=='all' and r['target']=='cohort_ttft_met'],
        predicted_best_candidates={m:dict(Counter(','.join(r['best_candidates'][m]) for r in report['cap_responses']))
                                  for m in report['cap_responses'][0]['best_candidates']},
        source_runs=[dict(name=e['meta']['name'], context=e['meta']['service_context'], table_sha256=e['sha256']) for e in report['source_runs']],
        note=report['note'])
    matrix = Counter()
    wrapper = json.loads(gzip.decompress(dataset.read_bytes()))
    manifest=json.loads(wrapper['manifest_text'])
    durations={e['file']:e['meta']['duration_s'] for e in manifest['runs']}
    for name,text in wrapper['tables'].items():
        for row in map(json.loads, text.splitlines()):
            label = row['labels']['5']
            if (row['label_window_complete']['5'] and label is not None and
                    max(row['t_s']+5, label['cohort_resolved_s'])<=durations[name]*2/3 and len(set(row['planned_caps'][:5]))==1):
                matrix[row['features']['budget'],row['planned_caps'][0]]+=1
    result['training_applied_to_planned_hold']=[dict(applied=a, planned=b, rows=n) for (a,b),n in sorted(matrix.items())]
    return result


def figure(summary, path):
    labels=dict(direct_ridge='V1 direct ridge',context_ridge='+ Context',gauge_ridge='+ Gauges',counter_ridge='+ Counters',
                service_quadratic='+ Cap interactions',service_no_plan='No planned cap')
    colors=['#366a56','#7199c1','#a1ba89','#edb965','#9b65a5','#ce8598']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    chronological=[r for r in summary['aggregates'] if r['split']=='chronological']
    for (m,label),color in zip(labels.items(),colors):
        axes[0,0].plot([r['horizon_s'] for r in chronological],[r['rate_mae'][m] for r in chronological],marker='o',label=label,color=color)
    axes[0,0].set(title='Chronological SLO-goodput rate MAE',xlabel='Horizon (s)',ylabel='Error (requests/s)',xticks=[1,2,5])
    axes[0,0].grid(alpha=.2)
    transfers=[r for r in summary['aggregates'] if r['horizon_s']==5]
    x=np.arange(len(transfers)); width=.13
    for i,((m,label),color) in enumerate(zip(labels.items(),colors)):
        axes[0,1].bar(x+(i-2.5)*width,[r['rate_mae'][m] for r in transfers],width,color=color,label=label)
    axes[0,1].set(title='Five-second development transfer',ylabel='MAE (requests/s)',xticks=x,
                  xticklabels=['Chronological','Fixed → A/B','A/B → fixed*'])
    axes[0,1].text(.02,.98,'* Includes cap extrapolation and host/runtime shifts',transform=axes[0,1].transAxes,va='top',fontsize=8)
    # These are model outputs only; no observed counterfactual outcomes exist.
    m='service_quadratic'; counts=summary['predicted_best_candidates'][m]
    caps=[16.,32.,64.,96.]
    axes[1,0].bar([str(int(c)) for c in caps],[counts.get(str(c),0) for c in caps],color=colors[4])
    axes[1,0].set_ylim(0,max(counts.values())*1.35)
    axes[1,0].set(title='Predicted best cap: interaction model ONLY',xlabel='Candidate cap',ylabel='Origins (not successful requests)')
    axes[1,0].text(.02,.98,'Diagnostic ranking — NOT measured policy gains\nNo-plan model ties all candidates at every origin',
                  transform=axes[1,0].transAxes,va='top',fontsize=8)
    levels=[16.,32.,48.,64.,80.,96.]; matrix=np.zeros((6,6))
    for r in summary['training_applied_to_planned_hold']:
        matrix[levels.index(r['applied']),levels.index(r['planned'])]=r['rows']
    axes[1,1].imshow(np.log1p(matrix),cmap='Blues',aspect='auto')
    for i in range(6):
        for j in range(6):
            axes[1,1].text(j,i,str(int(matrix[i,j])),ha='center',va='center',color='white' if matrix[i,j]>100 else 'black',fontsize=9)
    axes[1,1].set(title='Training origins: applied cap → planned hold',xlabel='Planned cap for five seconds',ylabel='Observed applied cap',
                  xticks=range(6),yticks=range(6),xticklabels=[int(c) for c in levels],yticklabels=[int(c) for c in levels])
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Service-context development: better forecasting ≠ validated cap decisions\nSame eight qualified captures; no C, controller, or new GPU collection',fontsize=13)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Choose fresh output')
    raw=args.report.read_bytes();report=json.loads(raw)
    summary=summarize(report,args.dataset)
    args.output_dir.mkdir(parents=True)
    compressed=gzip.compress(raw,mtime=0)
    (args.output_dir/'full-report.json.gz').write_bytes(compressed)
    summary.update(full_report_sha256=hashlib.sha256(raw).hexdigest(),compressed_report_sha256=hashlib.sha256(compressed).hexdigest(),
        dataset_pack_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),dataset_sha256=report['dataset_sha256'])
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['split','run','horizon_s','subset','target','model','n','mae','p99_abs','optimistic_p99','skill_vs_v1_direct','extrapolation_rows'])
    for r in report['scores']:
        for m,v in r['models'].items():
            writer.writerow([r['split'],r['run'],r['horizon_s'],r['subset'],r['target'],m,v['n'],v.get('mae'),v.get('p99_abs'),v.get('optimistic_p99'),r['skill_vs_direct'].get(m),r['diagnostic_extrapolation_rows']])
    (args.output_dir/'metrics.csv').write_text(stream.getvalue())
    figure(summary,args.output_dir/'forecast-and-support.png')
    print(json.dumps(summary['aggregates'],indent=2))


if __name__=='__main__':main()
