"""Archive frozen-model diagnostic evidence and render explicitly nondeployable oracles."""
import argparse
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


def find(report, model, mode, subset='all', step=5):
    return next(s for s in report['scores'] if (s['model'],s['mode'],s['subset'],s['step'])==(model,mode,subset,step))


def figure(report,path):
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    modes=['free','state_reset','arrival_history_refresh','activity_history_refresh','activity_outcome_refresh','all_observations_reset']
    labels=['Original','True Q/D/R','True arrival\nhistory','True activity\nhistory','True activity +\noutcome history','All observed\ninputs']
    x=np.arange(len(modes));width=.38
    for i,(model,color,label) in enumerate([('flow_ridge','#366a56','Flow'),('flow_ridge_engine','#7398bd','Flow + engine')]):
        axes[0,0].bar(x+(i-.5)*width,[find(report,model,m)['goodput_rate_mae'] for m in modes],width,color=color,label=label)
    axes[0,0].set(title='Five-second goodput error: frozen weights',ylabel='MAE (requests/s)',xticks=x,xticklabels=labels)
    axes[0,0].tick_params(axis='x',labelsize=8);axes[0,0].legend()
    for mode,label,color in [('free','Original','#366a56'),('state_reset','True Q/D/R each step','#b47882'),
                             ('history_refresh','True rolling history','#d7a448'),('all_observations_reset','All observed inputs','#7398bd')]:
        axes[0,1].plot(range(1,6),[find(report,'flow_ridge',mode,step=h)['scores']['awaiting_first_content']['mae'] for h in range(1,6)],marker='o',label=label,color=color)
    axes[0,1].set(title='Waiting-count error: same 768 origins',xlabel='Step (s)',ylabel='MAE (requests)',xticks=range(1,6))
    axes[0,1].legend(fontsize=8)
    subsets=['near_cap_change','away_from_cap_change'];x=np.arange(2);width=.24
    for i,(mode,label,color) in enumerate([('free','Original','#366a56'),('history_refresh','True rolling history','#d7a448'),('all_observations_reset','All observed inputs','#7398bd')]):
        axes[1,0].bar(x+(i-1)*width,[find(report,'flow_ridge',mode,s)['goodput_rate_mae'] for s in subsets],width,label=label,color=color)
    axes[1,0].set(title='Issue persists away from cap changes',ylabel='Five-second MAE (requests/s)',xticks=x,
        xticklabels=['Near change (n=90)','Away from change (n=678)'])
    axes[1,0].legend(fontsize=8)
    keys=['admitted','first','exit_awaiting','exit_streaming','cost_in','cost_out']
    matrix=np.array([[report['observed_input_flow_residual_correlations']['flow_ridge'][a][b] for b in keys] for a in keys],dtype=float)
    axes[1,1].imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
    for i in range(6):
        for j in range(6):
            axes[1,1].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>.65 else 'black')
    axes[1,1].set(title='One-step flow residual correlations (800 unique times)',xticks=range(6),yticks=range(6),xticklabels=keys,yticklabels=keys)
    axes[1,1].tick_params(axis='x',rotation=35,labelsize=8);axes[1,1].tick_params(axis='y',labelsize=8)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.grid(axis='y',alpha=.15)
    fig.suptitle('Model diagnosis: future-observation oracles are NOT deployable improvements\nOriginal coefficients, original captures; no new GPU data or MPC policy test',fontsize=14)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Choose a fresh output')
    raw=args.report.read_bytes();report=json.loads(raw);compressed=gzip.compress(raw,mtime=0)
    summary={k:report[k] for k in ('protocol','acceptance','c_used','models_unchanged','archived_coefficients_match','verified_free_rollouts','one_step_flow_errors_with_observed_inputs','observed_input_flow_residual_correlations','origin_only_repeat_goodput','limitations','code_sha256','dataset_sha256','archived_report_sha256')}
    summary.update(prediction_records=len(report['predictions']),score_groups=len(report['scores']),
        five_second_all=[s for s in report['scores'] if s['subset']=='all' and s['step']==5],
        full_report_sha256=hashlib.sha256(raw).hexdigest(),compressed_report_sha256=hashlib.sha256(compressed).hexdigest())
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'full-report.json.gz').write_bytes(compressed)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    stream=io.StringIO();writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['model','mode','subset','step','target','n','mae','bias','p99_abs','optimistic_p99','goodput_rate_mae'])
    for row in report['scores']:
        for target,score in row['scores'].items():
            writer.writerow([row['model'],row['mode'],row['subset'],row['step'],target,score['n'],score['mae'],score['bias'],score['p99_abs'],score.get('optimistic_p99'),row['goodput_rate_mae'] if target=='cohort_ttft_met' else ''])
    (args.output_dir/'metrics.csv').write_text(stream.getvalue())
    figure(report,args.output_dir/'oracle-diagnosis.png')


if __name__=='__main__':main()
