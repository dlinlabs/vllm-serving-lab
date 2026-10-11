"""Offline trace diagnosis. Audit raw files in a temporary directory; never rewrite them.

Requires NumPy and Matplotlib. Percentiles use linear interpolation. Stage shares
use the same successful-request cohort, not differences between percentiles.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from analyze_trace_pilot import LIFECYCLE, analyze as audit

RAW = ('manifest.json', 'client.json', 'warmup.json', 'gateway.jsonl', 'trace.json',
       'budget_changes.json', 'state_observations.json', 'telemetry_status.json',
       'report.json', 'timeseries.png')
STAGES = ('ingress', 'admission', 'client_build', 'post_dispatch', 'delivery')
LABELS = ('Client start → handler', 'Handler → accepted', 'Accepted → dispatch',
          'Dispatch → gateway content', 'Gateway → client content')
COLORS = ('#7199c1', '#b0c5dc', '#edb965', '#366a56', '#ce8598')


def hashes(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RAW}


def distribution(values):
    a = np.array(values, dtype=float)
    if not len(a):
        return {'count': 0, 'mean': None, 'p50': None, 'p99': None, 'max': None}
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite measurement')
    return dict(count=len(a), mean=float(a.mean()), p50=float(np.quantile(a, .5)),
                p99=float(np.quantile(a, .99)), max=float(a.max()))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def analyze_run(root, stall_threshold):
    before = hashes(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    original = json.loads((root / 'report.json').read_text())
    # The existing auditor writes report.json even with plotting disabled.
    # Symlink its read-only inputs, but keep its report output separate.
    with tempfile.TemporaryDirectory() as scratch:
        temporary = Path(scratch)
        for name in RAW:
            if name not in {'report.json', 'timeseries.png'}:
                (temporary / name).symlink_to(root / name)
        checked = audit(temporary, plot=False)
    if not checked['valid']:
        raise ValueError(f'{root.name}: integrity audit failed: {checked["issues"]}')
    if checked != original:
        raise ValueError(f'{root.name}: saved report differs from re-audit')
    clients = json.loads((root / 'client.json').read_text())['requests']
    records = [json.loads(line) for line in (root / 'gateway.jsonl').read_text().splitlines() if line.strip()]
    start, end = manifest['measurement_start'], manifest['measurement_end']
    histories = defaultdict(dict)
    lifecycle = []
    for event in records:
        if event['event'] in LIFECYCLE:
            histories[event['request_id']][event['event']] = event
            lifecycle.append(event)
    # Cumulative counts include the three warmup requests; increments do not.
    accepted_times = [e['monotonic'] for e in lifecycle if e['event'] == 'accepted']
    received_times = [e['monotonic'] for e in lifecycle if e['event'] == 'received']
    pairs = list(zip(lifecycle, lifecycle[1:]))
    joined = []
    for client in clients:
        history = histories[client['client_request_id']]
        row = dict(request_id=client['client_request_id'], trace_id=client['trace_id'],
                   scheduled_s=client['scheduled_at']-start, started_s=client['started_at']-start,
                   received_s=history['received']['monotonic']-start,
                   scheduling_lag_s=client['started_at']-client['scheduled_at'],
                   input_tokens=client['expected_input_tokens'], output_tokens=client['output_tokens'],
                   success=client['success'], rejected=client['rejected'], ttft_s=client['ttft'])
        if client['success']:
            times = [client['started_at']] + [history[k]['monotonic'] for k in
                     ('received', 'accepted', 'backend_sent', 'first_content')] + [client['first_content_at']]
            stages = dict(zip(STAGES, np.diff(times).tolist()))
            if min(stages.values()) < -1e-6 or abs(sum(stages.values())-client['ttft']) > 1e-6:
                raise ValueError('Latency decomposition does not reconcile')
            row.update(stages=stages, dispatch_s=times[3]-start, gateway_content_s=times[4]-start)
        joined.append(row)
    successful = [r for r in joined if r['success']]
    if not successful:
        raise ValueError(f'{root.name}: latency decomposition requires a successful request')
    tail = sorted(successful, key=lambda r: r['ttft_s'], reverse=True)[:math.ceil(len(successful)*.01)]
    stalls = []
    for monitor in records:
        if monitor['event'] != 'event_loop_lag' or monitor['lag_seconds'] <= stall_threshold:
            continue
        if not start <= monitor['scheduled_at'] < end:
            continue
        due, resumed = monitor['scheduled_at'], monitor['monotonic']
        # Search the nearby lifecycle gap, which can start before the monitor
        # became due. This gap is not proof of continuous event-loop blocking.
        nearby = [(a, b) for a, b in pairs if a['monotonic'] < resumed and b['monotonic'] > due-.2]
        if not nearby:
            raise ValueError('No lifecycle events near monitor delay')
        a, b = max(nearby, key=lambda pair: pair[1]['monotonic']-pair[0]['monotonic'])
        feature = next((histories[a['request_id']][k] for k in ('accepted', 'rejected')
                        if k in histories[a['request_id']]), {})
        stalls.append(dict(due_s=due-start, resumed_s=resumed-start, lag_s=monitor['lag_seconds'],
                           gap_start_s=a['monotonic']-start, gap_end_s=b['monotonic']-start,
                           gap_s=b['monotonic']-a['monotonic'], before_event=a['event'], after_event=b['event'],
                           same_request=a['request_id']==b['request_id'], before_request_id=a['request_id'],
                           after_request_id=b['request_id'], before_input_tokens=feature.get('estimated_input_tokens'),
                           accepted_count_at_gap_start=bisect_right(accepted_times, a['monotonic']),
                           received_count_at_gap_start=bisect_right(received_times, a['monotonic'])))
    def overlap(row):
        # Union of delayed-monitor intervals: empirical timing association only.
        intervals = sorted((max(row['dispatch_s'], s['due_s']), min(row['gateway_content_s'], s['resumed_s']))
                           for s in stalls)
        total, previous_end = 0., -math.inf
        for left, right in intervals:
            if right > left:
                total += max(0., right-max(left, previous_end))
                previous_end = max(previous_end, right)
        return total
    stage_stats = {k: distribution([r['stages'][k] for r in successful]) for k in STAGES}
    tail_means = {k: float(np.mean([r['stages'][k] for r in tail])) for k in STAGES}
    tail_ttft = float(np.mean([r['ttft_s'] for r in tail]))
    relative = lambda key: [r[key] for r in joined]
    bins = np.arange(0, math.ceil(end-start)+1)
    arrivals = {key: np.histogram(relative(key), bins=bins)[0].tolist()
                for key in ('scheduled_s', 'started_s', 'received_s')}
    expected_counts = {'scheduled_s': len(joined), 'started_s': len(joined),
                       'received_s': sum(0 <= r['received_s'] <= end-start for r in joined)}
    if any(sum(arrivals[k]) != expected_counts[k] for k in arrivals):
        raise ValueError('Arrival bins do not reconcile')
    summary = dict(name=root.name, raw_sha256=before, original_report_reproduced=True,
                   manifest_git_commit=manifest['git_commit'], recorded_git_dirty=manifest['git_dirty'],
                   runtime_code_matches={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==sha
                                         for name, sha in manifest['code_sha256'].items()},
                   trace_sha256=manifest['trace_sha256'], gpu=manifest['gpu'], packages=manifest['packages'],
                   config=manifest['config'], duration_s=end-start, report=checked,
                   scheduling_delayed_count=sum(r['scheduling_lag_s'] > manifest['config']['max_scheduling_lag'] for r in joined),
                   successful_output=distribution([r['output_tokens'] for r in successful]),
                   output_at_cap_count=sum(r['output_tokens']==r['max_output_tokens'] for r in clients if r['success']),
                   stage_seconds=stage_stats,
                   slowest_one_percent=dict(count=len(tail), request_ids=[r['request_id'] for r in tail],
                       mean_ttft_s=tail_ttft, mean_stage_seconds=tail_means,
                       post_dispatch_fraction=tail_means['post_dispatch']/tail_ttft,
                       post_dispatch_monitor_overlap_count=sum(overlap(r)>0 for r in tail)),
                   large_monitor_delays=stalls,
                   delay_due_spacing_s=np.diff([s['due_s'] for s in stalls]).tolist(),
                   delay_accepted_spacing=np.diff([s['accepted_count_at_gap_start'] for s in stalls]).tolist(),
                   arrival_counts_per_second=arrivals,
                   client_late_overlapping_large_monitor_count=sum(any(s['due_s'] < r['started_s'] and
                       s['resumed_s'] > r['scheduled_s'] for s in stalls) for r in joined
                       if r['scheduling_lag_s'] > manifest['config']['max_scheduling_lag']))
    if hashes(root) != before:
        raise ValueError('Raw files changed during analysis')
    return summary, joined, records, start


def figures(output, data, stall_threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(11, 5))
    labels, groups = [], []
    for summary, _, _, _ in data:
        labels.append(summary['name'].replace('trace-experiment-', '').replace('trace-', '')+'\n' +
                      ('load fidelity PASS' if summary['report']['load_fidelity']['valid'] else 'load fidelity FAILED'))
        groups.append(summary['slowest_one_percent'])
    left = np.zeros(len(groups))
    for key, label, color in zip(STAGES, LABELS, COLORS):
        values = np.array([g['mean_stage_seconds'][key] for g in groups])
        ax.barh(np.arange(len(groups)), values, left=left, label=label, color=color)
        left += values
    for i, group in enumerate(groups):
        ax.text(left[i]+.04, i, f'{left[i]:.3f}s; post-dispatch {100*group["post_dispatch_fraction"]:.1f}%\nn={group["count"]}', va='center', fontsize=9)
    ax.set_yticks(np.arange(len(groups)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left)*1.45)
    ax.set_xlabel('Mean client TTFT (seconds), same slowest ceil(1%) successful requests')
    ax.set_title('Trace replay — where tail latency occurs\nPost-dispatch includes transport, engine queue/compute, and gateway scheduling')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3, frameon=False)
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'latency_stages.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    repeats = [d for d in data if d[0]['duration_s'] > 30] or data
    fig, axes = plt.subplots(4, len(repeats), figsize=(14, 11), sharex='col', squeeze=False)
    for column, (summary, joined, records, start) in enumerate(repeats):
        successful = [r for r in joined if r['success']]
        arrivals = summary['arrival_counts_per_second']
        ax = axes[0, column]
        for key, label, color in [('scheduled_s','Planned arrivals','#888888'), ('received_s','Gateway received','#286e99')]:
            ax.step(np.arange(len(arrivals[key])), arrivals[key], where='post', label=label, color=color, alpha=.8)
        ax.set_ylabel('Requests / 1s bin')
        fidelity = 'PASS' if summary['report']['load_fidelity']['valid'] else 'FAILED'
        ax.set_title(summary['name']+'\nIntegrity PASS / load fidelity '+fidelity)
        ax = axes[1, column]
        ax.scatter([r['scheduled_s'] for r in joined], [1000*r['scheduling_lag_s'] for r in joined], s=5, label='Client scheduling lag', color='#286e99')
        loop = [e for e in records if e['event']=='event_loop_lag' and start <= e['scheduled_at'] < start+summary['duration_s']]
        ax.plot([e['scheduled_at']-start for e in loop], [1000*e['lag_seconds'] for e in loop], label='Gateway loop monitor lag', color='#ba5537', linewidth=.7)
        limit_ms = 1000*summary['config']['max_scheduling_lag']
        ax.axhline(limit_ms, ls='--', color='black', linewidth=.6, label=f'Client fidelity limit: {limit_ms:g}ms')
        ax.set_ylabel('Lag (ms)')
        ax = axes[2, column]
        ax.scatter([r['started_s'] for r in successful], [r['stages']['post_dispatch'] for r in successful], s=6, alpha=.4, color='#366a56', label='Post-dispatch stage')
        tail_ids = set(summary['slowest_one_percent']['request_ids'])
        tail = [r for r in successful if r['request_id'] in tail_ids]
        ax.scatter([r['started_s'] for r in tail], [r['stages']['post_dispatch'] for r in tail], s=22, marker='x', color='#b32e32', label='Slowest 1% client TTFT')
        ax.set_ylabel('Seconds')
        ax = axes[3, column]
        snapshots = [e for e in records if e['event']=='snapshot' and start <= e['monotonic'] < start+summary['duration_s']]
        ax.plot([e['monotonic']-start for e in snapshots], [e['current_admitted_cost'] for e in snapshots], label='Reserved cost (gateway proxy)', color='#8662a2')
        ax.axhline(summary['config']['budget'], color='black', ls='--', label='Fixed budget')
        ax.set_ylabel('Cost units')
        ax.set_ylim(0, summary['config']['budget']*1.1)
        ax.set_xlabel('Seconds since measurement start')
        for ax in axes[:, column]:
            for stall in summary['large_monitor_delays']:
                ax.axvspan(stall['due_s'], stall['resumed_s'], color='#ba5537', alpha=.14)
            ax.set_xlim(0, summary['duration_s'])
            ax.grid(alpha=.15)
            ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f'Trace replay timing — shaded bands: monitor delays >{1000*stall_threshold:g}ms\nMonitor delays and event gaps locate symptoms; neither identifies CPU, GC, or GPU causes', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output/'stall_alignment.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stall-threshold', type=float, default=.3,
                        help='Exploratory monitor-delay selection only; never changes fidelity limits')
    parser.add_argument('--all-requests', action='store_true',
                        help='Also export all joined measurement requests (several MB per run)')
    args = parser.parse_args()
    if not np.isfinite(args.stall_threshold) or args.stall_threshold <= 0:
        parser.error('--stall-threshold must be positive and finite')
    roots, output = [p.resolve() for p in args.runs], args.output_dir.resolve()
    if len({p.name for p in roots}) != len(roots):
        parser.error('Run directory names must be unique')
    if any(output == p or p in output.parents for p in roots):
        parser.error('Output must be separate from raw run directories')
    data = [analyze_run(root, args.stall_threshold) for root in roots]
    output.mkdir(parents=True, exist_ok=True)
    dump(output/'summary.json', dict(schema_version=1, analyst_stall_threshold_s=args.stall_threshold,
         analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         runs=[d[0] for d in data],
         repeat_trace_identical=(len({d[0]['trace_sha256'] for d in data if d[0]['duration_s'] > 30})==1
                                if sum(d[0]['duration_s'] > 30 for d in data) >= 2 else None)))
    for summary, joined, _, _ in data:
        tail_ids = set(summary['slowest_one_percent']['request_ids'])
        selected = joined if args.all_requests else [r for r in joined if r['request_id'] in tail_ids or
                    r['scheduling_lag_s'] > summary['config']['max_scheduling_lag']]
        suffix = '-requests.jsonl' if args.all_requests else '-tail-and-late.jsonl'
        with (output/(summary['name']+suffix)).open('w') as stream:
            for row in selected:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    figures(output, data, args.stall_threshold)
    print(json.dumps({s['name']: dict(integrity=s['report']['valid'], fidelity=s['report']['load_fidelity']['valid'],
                       tail_post_dispatch_fraction=s['slowest_one_percent']['post_dispatch_fraction'],
                       large_monitor_delays=len(s['large_monitor_delays'])) for s, _, _, _ in data}, indent=2))


if __name__ == '__main__':
    main()
