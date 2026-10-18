"""Scientific figures for the archived A/B development captures; no C input."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex='col')
    for col,side in enumerate(('a','b')):
        name=f'shared-client-{side}-r1'
        rows=[json.loads(s) for s in (args.analysis_root/'features'/f'{name}-features.jsonl').read_text().splitlines()]
        raw=args.raw_root/f'shared-{side}'/'results'/name
        manifest=json.loads((raw/'manifest.json').read_text());start=manifest['measurement_start']
        clients=json.loads((raw/'client.json').read_text())['requests']
        t=[r['t_s'] for r in rows]
        axes[0,col].step(t,[r['features']['budget'] for r in rows],where='post',color='#222222',label='Applied cap')
        axes[0,col].plot(t,[r['features']['reserved_cost'] for r in rows],color='#2563a6',lw=1,label='Reserved cost')
        axes[0,col].set_title(f'{side.upper()}: '+('forward schedule' if side=='a' else 'reversed schedule'))
        good=[]
        for begin in range(0,300,10):
            cohort=[r for r in clients if begin<=r['scheduled_at']-start<begin+10]
            good.append(sum(r['success'] and r['first_content_at']-r['scheduled_at']<=.5 for r in cohort)/10)
        axes[1,col].bar(np.arange(0,300,10)+5,good,width=9,color='#2f7a60')
        axes[2,col].plot(t,[r['features']['awaiting_first_content'] for r in rows],color='#b86b23',label='Awaiting first content')
        axes[2,col].plot(t,[r['features']['in_flight'] for r in rows],color='#777777',alpha=.6,label='In flight')
        for ax in axes[:,col]:ax.grid(axis='y',alpha=.2);ax.set_xlim(0,300);ax.set_ylim(bottom=0)
        axes[0,col].legend(fontsize=8,loc='lower left');axes[2,col].legend(fontsize=8,loc='upper right')
        axes[2,col].set_xlabel('Seconds since measurement start')
    axes[0,0].set_ylabel('Estimated cost units')
    axes[1,0].set_ylabel('SLO goodput (requests/s)\n10-second arrival cohorts')
    axes[2,0].set_ylabel('Requests')
    fig.suptitle('Shared-client A/B: same arrival trace, different schedules and rental hosts\nTTFT SLO 0.5 s; development data, no causal policy ranking',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(args.output_dir/'ab-overview.png',dpi=160);plt.close(fig)

    full=json.loads((args.analysis_root/'forecast/forecast_report.json').read_text())['evaluation']['results']
    diagnostics=args.analysis_root/'diagnostics'
    if not (diagnostics/'interactions-summary.json').exists():return
    versions=json.loads((diagnostics/'version-ablation-summary.json').read_text())['results']
    nonlinear=json.loads((diagnostics/'nonlinear-summary.json').read_text())['results']
    interaction=json.loads((diagnostics/'interactions-summary.json').read_text())['results']
    phase=json.loads((diagnostics/'summary.json').read_text())['results']['phase_planned']
    targets=('awaiting_first_content','cohort_ttft_met','cohort_admitted_ttft_missed')
    def picked(rows,target):return next(r for r in rows if r['split']=='chronological' and r['scope']=='pooled' and r['horizon_s']==5 and r['target']==target and r['subset']=='all')
    values=[('Persistence',[picked(full,t)['models']['persistence']['mae'] for t in targets]),
        ('Original ridge',[picked(full,t)['models']['ridge']['mae'] for t in targets]),
        ('Phase features',[picked(phase,t)['models']['ridge']['mae'] for t in targets]),
        ('Drop version',[picked(versions['no_version_planned'],t)['models']['ridge']['mae'] for t in targets]),
        ('Nonlinear + history',[picked(nonlinear['tree_phase_history_planned'],t)['mae'] for t in targets]),
        ('State x cap',[picked(interaction,t)['mae'] for t in targets])]
    fig,ax=plt.subplots(figsize=(11.5,5.5));width=.13;colors=('#9ca3af','#3b6fa3','#c48637','#548269','#956da4','#d26a58')
    for k,((name,v),color) in enumerate(zip(values,colors)):
        bars=ax.bar(np.arange(3)+(k-2.5)*width,v,width=width,label=name,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
    ax.set_xticks(np.arange(3),('Awaiting first content','TTFT SLO successes','Admitted TTFT misses'))
    ax.set_ylabel('Mean absolute error (requests; lower is better)');ax.set_ylim(0,10)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('5-second forecasts: chronological A/B development comparison\n382 overlapping evaluation rows; no held-out C or MPC benefit claim')
    ax.legend(ncol=3,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.12));fig.tight_layout()
    fig.savefig(args.output_dir/'forecast-comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
