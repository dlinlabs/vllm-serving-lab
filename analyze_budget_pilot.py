"""Validate one collected pilot and render its observed time series offline."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def _analyze(output, plot=True):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    clients = json.loads((output / 'client.json').read_text())['requests']
    warmup = json.loads((output / 'warmup.json').read_text())['requests']
    records = [json.loads(line) for line in (output / 'gateway.jsonl').read_text().splitlines() if line.strip()]
    records = [r for r in records if r['run_id'] == manifest.get('run_id')]
    issues = []
    if manifest['status'] != 'collected':
        issues.append('Collection did not finish normally')
    if manifest.get('forced_gateway_kill'):
        issues.append('Gateway was forcibly killed')
    snapshots = [r for r in records if r['event'] == 'snapshot']
    if not snapshots:
        issues.append('No snapshots')
    for r in snapshots:
        if r.get('telemetry_dropped_records', 0) or r.get('telemetry_write_errors', 0) or r.get('tokenizer_fallback_requests', 0):
            issues.append('Telemetry loss/write error or tokenizer fallback')
            break
    if snapshots and (snapshots[-1]['active_requests'] or abs(snapshots[-1]['current_admitted_cost']) > 1e-6):
        issues.append('Final snapshot is not drained')
    terminal_times = [r['monotonic'] for r in records if r['event'] in ('completed', 'failed', 'cancelled', 'rejected')]
    if snapshots and terminal_times and snapshots[-1]['monotonic'] < max(terminal_times):
        issues.append('Missing final snapshot after request termination')
    events = defaultdict(list)
    for r in records:
        if 'request_id' in r:
            events[r['request_id']].append(r)
    ids = [r.get('request_id') for r in clients + warmup]
    if None in ids or len(ids) != len(set(ids)):
        issues.append('Missing or duplicate client request IDs')
    if set(ids) != set(events):
        issues.append('Client/gateway request IDs do not match')
    for client in clients + warmup:
        history = events.get(client.get('request_id'), [])
        counts = Counter(r['event'] for r in history)
        terminal = [r['event'] for r in history if r['event'] in ('completed', 'failed', 'cancelled', 'rejected')]
        expected = 'completed' if client['success'] else 'rejected' if client['rejected'] else 'failed'
        if counts['received'] != 1 or terminal != [expected]:
            issues.append('Lifecycle/client outcome mismatch')
            break
        if expected == 'completed' and (counts['accepted'] != 1 or counts['first_content'] != 1):
            issues.append('Successful stream missing admission or first content')
            break
    start, end = manifest['measurement_start'], manifest['measurement_end']
    changes = [r for r in records if r['event'] == 'budget_updated']
    if [r['budget'] for r in changes] != manifest['config']['budgets']:
        issues.append('Budget update sequence mismatch')
    config = manifest['config']
    import math
    expected_count = math.ceil(config['rps'] * config['step_seconds'] * len(config['budgets']))
    if len(clients) != expected_count:
        issues.append('Unexpected measurement request count')
    ttft = [r['ttft'] for r in clients if r['success'] and r['ttft'] is not None]
    lags = [r['started_at'] - r['scheduled_at'] for r in clients if r['started_at'] is not None]
    # Data integrity is separate from whether this load excites useful dynamics.
    report = {'valid': not issues, 'issues': issues, 'measurement_requests': len(clients),
              'success': sum(r['success'] for r in clients),
              'rejected': sum(r['rejected'] for r in clients),
              'failed': sum(not r['success'] and not r['rejected'] for r in clients),
              'client_p99_ttft': percentile(ttft, .99), 'p99_scheduling_lag': percentile(lags, .99),
              'max_scheduling_lag': max(lags, default=None),
              'note': 'Integrity validation is not evidence of MPC benefit or steady state.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        visible = [r for r in snapshots if start <= r['monotonic'] <= end + 120]
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        x = [r['monotonic'] - start for r in visible]
        axes[0].plot(x, [r['current_admitted_cost'] for r in visible], label='Admitted cost')
        if changes:
            bx = [r['monotonic'] - start for r in changes]
            by = [r['budget'] for r in changes]
            axes[0].step(bx + [max(x, default=end-start)], by + [by[-1]], where='post', label='Actual budget')
        for name in ('awaiting_first_content', 'streaming_requests'):
            axes[1].plot(x, [r[name] for r in visible], label=name)
        for name in ('received', 'completed', 'rejected'):
            axes[2].plot(x, [r['counts'][name] / r['interval_seconds'] if r['monotonic'] - r['interval_seconds'] >= start else float('nan') for r in visible], label=name)
        first = [r for client in clients for r in events.get(client.get('request_id'), []) if r['event'] == 'first_content']
        axes[3].scatter([r['monotonic'] - start for r in first], [r['ttft'] for r in first], s=5, alpha=.3, label='Gateway TTFT at first content')
        axes[3].plot(x, [r['oldest_awaiting_first_seconds'] for r in visible], label='Oldest pending age')
        for ax, label in zip(axes, ('Cost', 'Requests', 'Requests / second', 'Seconds')):
            ax.set_ylabel(label)
            ax.legend(loc='upper left')
            ax.axvline(end-start, color='gray', linestyle=':', label='Arrivals stop')
            ax.grid(alpha=.2)
        axes[-1].set_xlabel('Seconds since measurement start; right of dotted line is drain')
        fig.suptitle('Budget step pilot' + ('' if report['valid'] else ' — INVALID DATA'))
        fig.tight_layout()
        fig.savefig(output / 'timeseries.png', dpi=160)
        plt.close(fig)
    return report


def analyze(output, plot=True):
    try:
        return _analyze(output, plot)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {'valid': False, 'issues': [f'Incomplete or malformed experiment data: {exc}']}
        (Path(output) / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    args = parser.parse_args()
    result = analyze(args.output_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['valid'] else 1)
