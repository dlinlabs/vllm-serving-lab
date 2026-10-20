"""Summarize the fixed combined review; no fitting, tuning or model selection."""
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


def digest(data):
    return hashlib.sha256(data).hexdigest()


def aggregate(report, split, horizon, subset='all'):
    rows = [r for r in report['scores'] if r['split'] == split and r['horizon_s'] == horizon
            and r['subset'] == subset and r['target'] == 'cohort_ttft_met' and r['models']['direct_ridge']['n']]
    n = sum(r['models']['direct_ridge']['n'] for r in rows)
    return dict(split=split, horizon_s=horizon, subset=subset, n=n,
        extrapolated_rows=sum(r['diagnostic_extrapolation_rows'] for r in rows),
        unseen_action_rows=sum(r['diagnostic_unseen_action_rows'] for r in rows),
        rate_mae={model: sum(r['models'][model]['mae'] * r['models'][model]['n'] for r in rows) / n / horizon
                  for model in rows[0]['models']} if n else {})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Choose a fresh output directory')
    raw=args.report.read_bytes();report=json.loads(raw)
    modes=['chronological','fixed_to_scheduled','scheduled_to_fixed']
    modes += sorted({r['split'] for r in report['scores'] if r['split'].startswith('exclude_cap:')},key=lambda s:float(s.split(':')[1]))
    summary=dict(protocol=report['protocol'],acceptance=None,c_used=False,runtime=report['runtime'],
        aggregates=[aggregate(report,split,h) for split in modes for h in (1,2,5)],
        chronological_support=[aggregate(report,'chronological',5,subset) for subset in ('seen_action_pattern','unseen_action_pattern')],
        note=report['note'],code_sha256=report['code_sha256'],dataset_sha256=report['dataset_sha256'],
        source_runs=[dict(name=e['meta']['name'],context=e['meta']['service_context'],table_sha256=e['sha256']) for e in report['source_runs']],
        unsupported_candidate_counts=dict(Counter(cap for row in report['cap_responses'] for cap in row['unsupported_candidates'])),
        candidate_origins=len(report['cap_responses']),
        predicted_best_candidates={m:dict(Counter(','.join(r['best_candidates'][m]) for r in report['cap_responses']))
                                  for m in report['cap_responses'][0]['best_candidates']})
    compressed=gzip.compress(raw,mtime=0)
    manifest_text=(args.dataset/'dataset.json').read_text()
    manifest=json.loads(manifest_text)
    wrapper=dict(format='audited-service-tables-v2',manifest_text=manifest_text,
                 tables={e['file']:(args.dataset/e['file']).read_text() for e in manifest['runs']})
    packed=gzip.compress(json.dumps(wrapper,ensure_ascii=False,separators=(',',':')).encode(),mtime=0)
    summary.update(full_report_sha256=digest(raw),compressed_report_sha256=digest(compressed),dataset_pack_sha256=digest(packed))
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'full-report.json.gz').write_bytes(compressed)
    (args.output_dir/'dataset.json.gz').write_bytes(packed)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['split','run','horizon_s','subset','target','model','n','mae','p99_abs','optimistic_p99','skill_vs_direct','extrapolation_rows','unseen_action_rows'])
    for r in report['scores']:
        for model,v in r['models'].items():
            writer.writerow([r['split'],r['run'],r['horizon_s'],r['subset'],r['target'],model,v['n'],v.get('mae'),v.get('p99_abs'),v.get('optimistic_p99'),r['skill_vs_direct'].get(model),r['diagnostic_extrapolation_rows'],r['diagnostic_unseen_action_rows']])
    (args.output_dir/'metrics.csv').write_text(stream.getvalue())
    figure(summary,args.output_dir/'review-errors-and-support.png')
    print(json.dumps(summary['aggregates'],indent=2))


def figure(summary,path):
    labels={'direct_ridge':'V1 direct','gauge_ridge':'+ Context / gauges','counter_ridge':'+ Counters',
            'service_quadratic':'+ Cap interactions','service_sequence':'Ordered cap sequence','service_no_plan':'No planned cap'}
    colors=['#366a56','#7199c1','#edb965','#9b65a5','#cf655a','#75827b']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    chronological=[r for r in summary['aggregates'] if r['split']=='chronological']
    for (model,label),color in zip(labels.items(),colors):
        axes[0,0].plot([r['horizon_s'] for r in chronological],[r['rate_mae'][model] for r in chronological],marker='o',label=label,color=color)
    axes[0,0].set(title='Chronological development forecast error',xlabel='Horizon (s)',ylabel='SLO-goodput rate MAE (req/s)',xticks=[1,2,5])
    groups=[r for r in summary['aggregates'] if r['split'].startswith('exclude_cap:') and r['horizon_s']==5]
    x=np.arange(len(groups));width=.14
    for i,((model,label),color) in enumerate(zip(labels.items(),colors)):
        axes[0,1].bar(x+(i-2.5)*width,[r['rate_mae'][model] for r in groups],width,color=color)
    axes[0,1].set(title='Exclude applied/planned cap group, 5 s',xlabel='Held-out cap (all repeats)',ylabel='MAE (req/s)',
        xticks=x,xticklabels=[r['split'].split(':')[1].split('.')[0] for r in groups])
    caps=['16.0','32.0','64.0','96.0'];n=summary['candidate_origins']
    missing=[summary['unsupported_candidate_counts'].get(cap,0) for cap in caps]
    axes[1,0].bar(caps,[n-m for m in missing],label='Seen applied-cap / plan pattern',color='#7199c1')
    axes[1,0].bar(caps,missing,bottom=[n-m for m in missing],label='Unseen pattern: diagnostic override',color='#d69a81')
    axes[1,0].set(title='Candidate support at chronological test origins',xlabel='Candidate held cap',ylabel='Origins',ylim=(0,n*1.15))
    axes[1,0].legend(fontsize=8)
    axes[1,0].text(.02,.97,'Seen pattern does not certify state coverage or safety',transform=axes[1,0].transAxes,va='top',fontsize=8)
    support=summary['chronological_support']
    for i,(model,color) in enumerate(zip(labels,colors)):
        axes[1,1].bar(np.arange(2)+(i-2.5)*width,[r['rate_mae'].get(model,0) for r in support],width,color=color)
    axes[1,1].set(title='Realized-plan forecast error by pattern coverage',ylabel='Five-second MAE (req/s)',
        xticks=[0,1],xticklabels=['Seen pattern\nn='+str(support[0]['n']), 'Unseen pattern\nn='+str(support[1]['n'])])
    for ax in axes.flat:ax.grid(axis='y',alpha=.15)
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Combined offline review: development forecasts, not validated MPC\nSame eight captures; grouped exclusions are not causal or unseen-workload tests',fontsize=13)
    fig.savefig(path,dpi=160);plt.close(fig)


if __name__=='__main__':main()
