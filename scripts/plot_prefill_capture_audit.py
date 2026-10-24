"""Scientific static view of measured wall overlap and legacy profile coverage."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from prefill_lab.analysis import rows_from_capture


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='pr32-b04477e-native-audit'
    import matplotlib.pyplot as plt
    report=json.loads((a.output/'analysis.json').read_text());training=report['training_wall_labels'][0]
    c=json.loads((a.capture/(training['trial_id']+'.json')).read_text());rows=rows_from_capture(c)[:8]
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    base=rows[0]['start']
    for i,row in enumerate(rows):
        axes[0].barh(i,(row['end']-row['start'])*1000,left=(row['start']-base)*1000,
                     color='#2563eb' if i%2==0 else '#0891b2',height=.65)
    axes[0].invert_yaxis();axes[0].set_yticks(range(8),[str(i+1) for i in range(8)])
    axes[0].set(xlabel='Wall time from first selected step (ms)',ylabel='Scheduled step',title='Wall spans overlap: two in flight')
    axes[0].spines[['top','right']].set_visible(False);axes[0].grid(axis='x',alpha=.2)
    ax=axes[1]
    for i,deep in enumerate(report['deep_profiles']):
        ax.barh(i,10,left=.5,color='#94a3b8',height=.45)
        ax.barh(i,5,left=10.5,color='#f59e0b',height=.45)
    ax.axvspan(.5,5.5,color='#ef4444',alpha=.15,label='Recorded legacy profile: steps1–5')
    ax.axvline(10.5,color='#b45309',linestyle='--',linewidth=1)
    ax.set_yticks([0,1],['Single7168 / D16','Mixed / D16']);ax.set_xticks([1,5,10,11,15])
    ax.set(xlim=(.5,15.5),ylim=(-.6,1.6),xlabel='Worker step (1-based)',title='Both deep traces miss foreground work')
    ax.text(3,1.4,'Profile window',ha='center',color='#b91c1c',fontsize=9)
    ax.text(13,1.4,'Foreground begins',ha='center',color='#b45309',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle('PR32 native capture b04477e: collection complete, model unqualified',fontsize=13)
    fig.savefig(a.output/'timing-and-profile.png',dpi=170)
    svg=a.output/'timing-and-profile.svg'
    fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
