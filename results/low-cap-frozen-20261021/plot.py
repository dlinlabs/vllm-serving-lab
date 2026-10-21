"""Plot frozen-model factual forecasts from the committed numerical report."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    report = json.loads(gzip.decompress((HERE / 'full-report.json.gz').read_bytes()))
    rows = report['predictions']
    names = list(dict.fromkeys(r['run'] for r in rows))
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        values = [r for r in rows if r['run'] == name]
        t = [r['t'] for r in values]
        ax.plot(t, [r['actual_goodput'] for r in values], color='#202020', lw=1.5, label='Measured completed SLO goodput')
        ax.plot(t, [r['simulator'] for r in values], color='#166aaf', lw=1.3, label='Frozen model + causal bias')
        ax.plot(t, [r['nowcast'] for r in values], color='#c87b16', lw=1.1, alpha=.9, label='Trailing-60 s first-token rate')
        for at in (60, 120, 180, 240):
            if '-a-' in name or '-b-' in name:
                ax.axvline(at, color='#999999', alpha=.35, linestyle='--')
        ax.set_title(name.replace('low-cap-switch-v1-', ''))
        ax.grid(alpha=.15)
        ax.set_ylim(0, 8)
    for ax in axes[-1]:
        ax.set_xlabel('Scheduled arrival cohort start (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('SLO requests / arrival second')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Frozen PR #28 model on new low-cap captures: 5 s factual forecasts\n'
                 'Development transfer diagnosis; no counterfactual policy outcome is observed', fontsize=13)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(HERE / 'frozen-forecasts.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
