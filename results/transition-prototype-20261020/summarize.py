"""Summarize an offline transition report; no new scoring or model selection."""
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


def summarize(report):
    selected = [r for r in report['scores'] if r['split'] == 'chronological' and r['subset'] == 'all']
    aggregate = []
    for target in ('awaiting_first_content', 'streaming', 'reserved_cost', 'cohort_ttft_met'):
        for horizon in (1, 2, 5):
            rows = [r for r in selected if r['target'] == target and r['horizon_s'] == horizon]
            models = {}
            for name in rows[0]['models']:
                n = sum(r['models'][name]['n'] for r in rows)
                mae = sum(r['models'][name]['mae'] * r['models'][name]['n'] for r in rows) / n
                models[name] = dict(n=n, mae=mae, rate_mae=mae / horizon if target == 'cohort_ttft_met' else None)
            reference = models['persistence']['mae']
            aggregate.append(dict(target=target, horizon_s=horizon, models=models,
                skill_vs_persistence={m: 1 - v['mae'] / reference if reference else None for m, v in models.items()}))
    by_run = [dict(run=r['run'], horizon_s=r['horizon_s'],
                   models={m: v['mae'] / r['horizon_s'] for m, v in r['models'].items()})
              for r in selected if r['target'] == 'cohort_ttft_met']
    projections = {}
    for name in ('flow_ridge', 'flow_ridge_engine'):
        values = [r['projection_l1'][name] for r in report['chronological_predictions'] if r['horizon_s'] == 5]
        projections[name] = dict(n=len(values), fraction_nonzero=float(np.mean(np.array(values) > 1e-8)),
                                 mean=float(np.mean(values)), max=float(max(values)))
    return dict(protocol=report['protocol'], acceptance=None, c_used=False,
        dataset_sha256=report['dataset_sha256'], source_runs=[dict(name=e['meta']['name'],
            qualified=e['meta']['qualified_for_comparison'], state_valid=e['meta']['state_reconstruction']['valid'],
            conservation_rows=e['meta']['conservation_verified_rows'], table_sha256=e['sha256']) for e in report['source_runs']],
        chronological_aggregate=aggregate, goodput_rate_mae_by_run=by_run,
        projection_five_seconds=projections, limitations=report['limitations'])


def plot(summary, destination):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    targets = ('awaiting_first_content', 'streaming', 'reserved_cost', 'cohort_ttft_met')
    titles = ('Awaiting first content (requests)', 'Streaming (requests)', 'Reserved cost (cost units)', 'Cohort SLO goodput (requests/s)')
    labels = dict(persistence='Persistence', mean='Training mean', direct_ridge='Direct ridge',
                  flow_ridge='Flow rollout', flow_ridge_engine='Flow + engine gauges')
    for ax, target, title in zip(axes.flat, targets, titles):
        rows = [r for r in summary['chronological_aggregate'] if r['target'] == target]
        for name, label in labels.items():
            ax.plot([r['horizon_s'] for r in rows],
                    [r['models'][name]['rate_mae' if target == 'cohort_ttft_met' else 'mae'] for r in rows],
                    marker='o', linewidth=1.6, label=label)
        ax.set(title=title, xlabel='Forecast horizon (s)', ylabel='Mean absolute error', xticks=[1, 2, 5])
        ax.grid(alpha=.25)
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Retrospective chronological development: 8 qualified captures\nPooled by forecast count; overlapping windows are not independent trials', fontsize=13)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Choose a fresh output directory')
    raw = args.report.read_bytes()
    report = json.loads(raw)
    summary = summarize(report)
    args.output_dir.mkdir(parents=True)
    compressed = gzip.compress(raw, mtime=0)
    (args.output_dir / 'full-report.json.gz').write_bytes(compressed)
    summary['full_report_sha256'] = hashlib.sha256(raw).hexdigest()
    summary['compressed_report_sha256'] = hashlib.sha256(compressed).hexdigest()
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['split', 'run', 'horizon_s', 'subset', 'target', 'model', 'n', 'mae', 'rmse', 'p99_abs', 'optimistic_p99', 'skill_vs_persistence', 'extrapolation_rows'])
    for row in report['scores']:
        for name, value in row['models'].items():
            writer.writerow([row['split'], row['run'], row['horizon_s'], row['subset'], row['target'], name,
                value['n'], value.get('mae'), value.get('rmse'), value.get('p99_abs'), value.get('optimistic_p99'),
                row['skill_vs_persistence'].get(name), row['diagnostic_extrapolation_rows']])
    (args.output_dir / 'metrics.csv').write_text(stream.getvalue())
    plot(summary, args.output_dir / 'forecast-errors.png')
    print(json.dumps(summary['chronological_aggregate'], indent=2))


if __name__ == '__main__':
    main()
