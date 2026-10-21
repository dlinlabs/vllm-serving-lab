"""Generate development-only figures and inspectable arithmetic from saved forecasts."""
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/state-model-20261021'


def main():
    report = json.loads(gzip.decompress((OUT / 'full-report.json.gz').read_bytes()))
    scores, folds = report['scores'], report['calibration_folds']
    frozen = json.loads(gzip.decompress((ROOT / 'results/low-cap-frozen-20261021/full-report.json.gz').read_bytes()))
    decomposition = []
    for cap in (16, 32):
        for start in (None, 60, 120, 180, 240):
            rows = [r for r in frozen['predictions'] if r['cap'] == cap and
                    (start is None or start <= r['t'] < start + 60)]
            terms = []
            for r in rows:
                actual_pass = r['actual_goodput'] / r['actual_admission'] if r['actual_admission'] else 0.
                sim_pass = r['simulator'] / r['admission'] if r['admission'] else 0.
                admission = (r['admission'] - r['actual_admission']) * actual_pass
                hit_rate = r['admission'] * (sim_pass - actual_pass)
                assert abs(admission + hit_rate - (r['simulator'] - r['actual_goodput'])) < 1e-10
                terms.append([admission + hit_rate, admission, hit_rate])
            values = np.mean(terms, axis=0)
            decomposition.append(dict(cap=cap, start=start, n=len(rows), goodput_bias=values[0],
                admission_term=values[1], hit_rate_term=values[2],
                simulated_pass_ratio=sum(r['simulator'] for r in rows) / sum(r['admission'] for r in rows),
                actual_pass_ratio=sum(r['actual_goodput'] for r in rows) / sum(r['actual_admission'] for r in rows)))
    probes = {}
    for kind in ('state', 'tree', 'schedule_holdout_state', 'schedule_holdout_tree'):
        ps = [p for f in folds[kind] for p in f['hypothetical_same_state_probes']]
        probes[kind] = dict(origins=len(ps), prefer32=sum(p['delta'] > 0 for p in ps),
            mean_delta=float(np.mean([p['delta'] for p in ps])),
            by_segment=[dict(start=s, mean_delta=float(np.mean([p['delta'] for p in ps if s <= p['t'] < s + 60])))
                        for s in (60, 120, 180, 240)], counterfactual_ground_truth_available=False)
    detail = dict(decomposition=decomposition, shadow_probes=probes)
    (OUT / 'error-decomposition.json').write_text(json.dumps(detail, indent=2) + '\n')
    keys = ['frozen', 'prefix_nnls', 'prefix_shrink', 'quadratic_025', 'calibration_state',
            'calibration_tree', 'schedule_nested_family']
    labels = ['Frozen', 'Prefix NNLS', 'Prefix + prior', 'Quadratic stress', 'State logistic', 'Small tree', 'Schedule + selection']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    ax[0].barh(labels[::-1], [scores[k]['overall']['mae'] for k in keys][::-1], color='#317c9b')
    ax[0].set(xlabel='Five-second goodput MAE (requests/s)', title='Development forecast error')
    for key, label in [('frozen', 'Frozen'), ('calibration_state', 'State logistic'), ('calibration_tree', 'Small tree')]:
        rows = [r for r in scores[key]['by_segment'] if r['cap'] == 32]
        ax[1].plot([r['start'] for r in rows], [r['bias'] for r in rows], 'o-', label=label)
    ax[1].axhspan(-.2, .2, color='#cce5cc', alpha=.5); ax[1].axhline(0, color='grey', lw=.7)
    ax[1].set(xlabel='Segment start (s)', ylabel='Predicted minus observed (req/s)', title='Cap 32: remaining segment bias')
    ax[1].legend(fontsize=8)
    for key, label in [('state', 'State logistic'), ('tree', 'Small tree')]:
        ps = probes[key]['by_segment']
        ax[2].plot([p['start'] for p in ps], [p['mean_delta'] for p in ps], 'o-', label=label)
    ax[2].axhline(0, color='grey', lw=.7); ax[2].legend(fontsize=8)
    ax[2].set(xlabel='Segment start (s)', ylabel='Predicted cap 32 minus cap 16 (req/s)', title='Shadow probes: no observed counterfactual')
    fig.savefig(OUT / 'development-comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
