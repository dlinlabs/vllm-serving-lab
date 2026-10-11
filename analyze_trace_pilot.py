"""Audit replay accounting separately from offered-load fidelity and performance."""
import argparse
from collections import Counter, defaultdict
import json
import math
import uuid
from pathlib import Path

from analyze_budget_pilot import percentile
from run_budget_pilot import save
from trace_workload import load_trace, sha256_file

TERMINAL = {'completed', 'rejected', 'failed', 'cancelled'}
LIFECYCLE = {'received', 'accepted', 'backend_sent', 'first_content'} | TERMINAL
PRE_HEADER_ERRORS = {'ConnectError', 'ConnectTimeout', 'ReadError', 'ReadTimeout', 'WriteError',
                     'WriteTimeout', 'PoolTimeout', 'RemoteProtocolError', 'LocalProtocolError',
                     'ProxyError', 'UnsupportedProtocol', 'DecodingError', 'CancelledError'}


def stats(rows, duration, slo):
    successful = [r for r in rows if r['success']]
    actual = [r['ttft'] for r in successful]
    scheduled = [r['first_content_at'] - r['scheduled_at'] for r in successful]
    good = sum(r['first_content_at'] - r['scheduled_at'] <= slo for r in successful)
    return dict(attempted=len(rows), success=len(successful), rejected=sum(r['rejected'] for r in rows),
                failed=sum(not r['success'] and not r['rejected'] for r in rows),
                client_p99_ttft=percentile(actual, .99), scheduled_p99_ttft=percentile(scheduled, .99),
                client_p99_e2e=percentile([r['end_to_end'] for r in successful], .99),
                slo_successes=good, slo_goodput_per_arrival_second=good / duration,
                slo_success_fraction_of_offered=good / len(rows) if rows else None,
                slo_definition='stream completed successfully; first content minus scheduled arrival <= TTFT SLO')


def _analyze(output, plot):
    manifest = json.loads((output / 'manifest.json').read_text())
    trace = load_trace(output / 'trace.json')
    clients = json.loads((output / 'client.json').read_text())['requests']
    warmup = json.loads((output / 'warmup.json').read_text())['requests']
    changes = json.loads((output / 'budget_changes.json').read_text())
    observations = json.loads((output / 'state_observations.json').read_text())
    shutdown = json.loads((output / 'telemetry_status.json').read_text())
    records = [json.loads(line) for line in (output / 'gateway.jsonl').read_text().splitlines() if line.strip()]
    issues, fidelity = [], []
    def require(condition, message):
        if not condition and message not in issues:
            issues.append(message)
    require(manifest['status'] == 'collected', 'Collection did not finish normally')
    require(not manifest.get('forced_gateway_kill'), 'Gateway was forcibly killed')
    require(shutdown['run_id'] == manifest['run_id'] and shutdown['status'] == 'closed' and
            shutdown['dropped'] == shutdown['write_errors'] == 0, 'Telemetry shutdown did not confirm successful persistence')
    require(sha256_file(output / 'trace.json') == manifest['trace_sha256'], 'Replay trace hash mismatch')
    require(all(r['run_id'] == manifest['run_id'] for r in records), 'Foreign run IDs in gateway log')
    require(all(math.isfinite(r['monotonic']) for r in records), 'Invalid event timestamps')
    require(all(a['monotonic'] <= b['monotonic'] for a, b in zip(records, records[1:])), 'Gateway log is out of order')
    require(all(r['monotonic'] <= shutdown['closed_at'] for r in records), 'Events appear after telemetry shutdown')
    start, end = manifest['measurement_start'], manifest['measurement_end']
    require(abs(end - start - trace['duration_seconds']) < 1e-6, 'Measurement duration differs from trace')
    require(len(clients) == len(trace['requests']), 'Missing or extra client outcomes')
    expected = {row['trace_id']: row for row in trace['requests']}
    require(Counter(r['trace_id'] for r in clients) == Counter(expected.keys()), 'Missing or duplicate trace rows')
    all_clients = clients + warmup
    require(len(warmup) == manifest['config']['warmup_requests'] and all(r['success'] for r in warmup),
            'Warmup count or outcome mismatch')
    identifiers = [r.get('client_request_id') for r in all_clients]
    require(all(isinstance(i, str) and len(i) == 32 for i in identifiers) and
            len(set(identifiers)) == len(identifiers), 'Missing or duplicate preassigned client IDs')
    events = defaultdict(list)
    for record in records:
        if record['event'] in LIFECYCLE and 'request_id' in record:
            events[record['request_id']].append(record)
    require(set(events) <= set(identifiers), 'Gateway recorded an unknown client ID')
    snapshots = [r for r in records if r['event'] == 'snapshot']
    require(bool(snapshots), 'No gateway snapshots')
    for snapshot in snapshots:
        require(not any(snapshot.get(field, 0) for field in
                        ('telemetry_dropped_records', 'telemetry_write_errors', 'tokenizer_fallback_requests')),
                'Telemetry loss/write error or tokenizer fallback')
    if snapshots:
        final = snapshots[-1]
        require(final['active_requests'] == 0 and final['current_in_flight'] == 0 and
                abs(final['current_admitted_cost']) < 1e-6, 'Final gateway state is not drained')
        totals = Counter(r['event'] for r in records if r['event'] in LIFECYCLE)
        require(all(final['totals'].get(k, 0) == totals[k] for k in
                    ('received', 'accepted', 'completed', 'rejected', 'failed', 'cancelled')),
                'Final cumulative counters disagree with lifecycle records')
        terminals = [r['monotonic'] for r in records if r['event'] in TERMINAL]
        require(not terminals or final['monotonic'] >= max(terminals), 'Missing final snapshot after termination')
    for row in all_clients:
        identifier = row.get('client_request_id')
        history = events.get(identifier, [])
        counts = Counter(r['event'] for r in history)
        terminal = [r for r in history if r['event'] in TERMINAL]
        require(not (row['success'] and row['rejected']), 'Contradictory client outcome')
        require(row.get('error_type') != 'NotLaunched', 'A planned request was not launched')
        if row['status'] is not None or row.get('request_id') is not None:
            require(row['request_id'] == identifier, 'Gateway response ID does not match preassigned ID')
        if not history:
            # A recorded transport failure can genuinely happen before the handler runs.
            require(not row['success'] and not row['rejected'] and row['status'] is None and
                    row.get('error_phase') == 'before_response_headers' and row.get('error_type') in PRE_HEADER_ERRORS,
                    'Missing lifecycle without a recorded pre-header failure')
        else:
            require(counts['received'] == 1 and len(terminal) == 1, 'Missing or duplicate gateway lifecycle')
            require(all(counts[event] <= 1 for event in LIFECYCLE), 'Duplicate lifecycle event')
            if terminal:
                outcome = terminal[0]['event']
                require(history[0]['event'] == 'received' and history[-1]['event'] in TERMINAL,
                        'Lifecycle ordering is invalid')
                order = ['received', 'accepted', 'backend_sent', 'first_content', outcome]
                require([order.index(r['event']) for r in history] ==
                        sorted(order.index(r['event']) for r in history), 'Lifecycle stages are out of order')
                if row['success']:
                    require(outcome == 'completed' and counts['accepted'] == 1 and
                            counts['backend_sent'] == 1 and counts['first_content'] == 1,
                            'Successful client has incomplete gateway lifecycle')
                elif row['rejected']:
                    require(outcome == 'rejected' and counts['accepted'] == 0 and row['status'] == 503,
                            'Client rejection differs from gateway')
                else:
                    # Backend can complete even though client delivery times out. Keep both outcomes.
                    require(outcome in {'completed', 'failed', 'cancelled'}, 'Client failure conflicts with admission rejection')
                require(not counts['first_content'] or counts['backend_sent'] == counts['accepted'] == 1,
                        'First content without upstream admission')
        if row['started_at'] is not None:
            require(all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in
                        ('started_at', 'finished_at', 'end_to_end')), 'Invalid client timing')
            require(row['finished_at'] >= row['started_at'] >= row['scheduled_at'] - 1e-6,
                    'Client timestamps are out of order')
            require(abs(row['end_to_end'] - (row['finished_at'] - row['started_at'])) < 1e-6,
                    'Client duration disagrees with timestamps')
        else:
            require(False, 'Missing client start time')
        if history and row['started_at'] is not None:
            # Runner and gateway share the same host monotonic clock. Terminal
            # cleanup may finish after the client; do not impose that ordering.
            require(history[0]['monotonic'] >= row['started_at'] - 1e-6,
                    'Gateway received precedes client start')
        if row['success']:
            require(row['status'] == 200 and row['stream_complete'] and row.get('first_content_at') is not None,
                    'Success without a complete content stream')
            require(row['started_at'] <= row['first_content_at'] <= row['finished_at'], 'Invalid first content timing')
            require(abs(row['ttft'] - (row['first_content_at'] - row['started_at'])) < 1e-6, 'TTFT timestamp mismatch')
            require(type(row.get('input_tokens')) is int and type(row.get('output_tokens')) is int,
                    'Successful request is missing token usage')
            require(row.get('error_type') is None, 'Success has a client error')
            first = next((r for r in history if r['event'] == 'first_content'), None)
            if first:
                require(first['monotonic'] <= row['first_content_at'] + 1e-6,
                        'Client first content precedes gateway first content')
        if 'trace_id' in row:
            planned = expected[row['trace_id']]
            require(identifier == uuid.uuid5(uuid.UUID(manifest['run_id']), row['trace_id']).hex,
                    'Preassigned ID differs from run and trace row')
            require(abs(row['scheduled_at'] - start - planned['arrival_seconds']) < 1e-6, 'Client schedule differs from trace')
            require(row['max_output_tokens'] == trace['output_cap'], 'Client output cap differs from trace')
            require(row['expected_input_tokens'] == planned['input_tokens'], 'Client target input count differs from trace')
            if row['success']:
                require(row['input_tokens'] == planned['input_tokens'], 'Actual prompt tokens differ from trace')
                require(0 <= row['output_tokens'] <= trace['output_cap'], 'Actual output exceeds fixed cap')
            feature = next((r for r in history if r['event'] in {'accepted', 'rejected'}), None)
            if feature:
                require(feature['estimated_input_tokens'] == planned['input_tokens'] and
                        feature['max_output_tokens'] == trace['output_cap'], 'Gateway request features differ from trace')
    updates = [r for r in records if r['event'] == 'budget_updated']
    require(len(changes) == len(updates) == 1, 'Fixed-budget replay must have one measured budget command')
    for command, applied in zip(changes, updates):
        require(command['budget'] == applied['budget'] == manifest['config']['budget'], 'Budget value mismatch')
        require(command['budget_version'] == applied['budget_version'], 'Budget version mismatch')
        require(command['applied_at'] == applied['applied_at'], 'Budget application timestamp mismatch')
        require(command['sent'] <= applied['applied_at'] <= applied['monotonic'] <= command['acknowledged'],
                'Budget application timing is invalid')
        require(applied['applied_at'] <= start, 'Initial budget was not applied before measurement')
    require(bool(observations), 'No external state observations')
    for observation in observations:
        if 'metrics' in observation:
            require(observation['metrics']['telemetry_run_id'] == manifest['run_id'], 'State observation belongs to another run')
            require(observation['sent_at'] <= observation['metrics']['state_sampled_at'] <= observation['received_at'],
                    'Invalid state observation timing')
            require(abs(observation['state_age_seconds'] -
                        (observation['received_at'] - observation['metrics']['state_sampled_at'])) < 1e-6,
                    'State age mismatch')
    lags = [r['started_at'] - r['scheduled_at'] for r in clients if r['started_at'] is not None]
    if len(lags) != len(trace['requests']):
        fidelity.append('Not all planned requests were launched')
    if max(lags, default=0) > manifest['config']['max_scheduling_lag']:
        fidelity.append('Load generator exceeded predeclared maximum scheduling lag')
    report = dict(valid=not issues, issues=issues,
                  load_fidelity=dict(valid=not fidelity, issues=fidelity, p99_scheduling_lag=percentile(lags, .99),
                                     max_scheduling_lag=max(lags, default=None)),
                  performance=stats(clients, trace['duration_seconds'], manifest['config']['ttft_slo']) if not issues else None,
                  gateway_terminal_counts=dict(Counter(r['event'] for r in records if r['event'] in TERMINAL)),
                  failed_without_gateway_record=sum(not events.get(r.get('client_request_id')) for r in clients),
                  state_poll_errors=sum('error_type' in r for r in observations),
                  cache_policy_self_reported=manifest['config']['cache_policy'],
                  note='Integrity and load fidelity do not establish MPC benefit. Source output lengths are not replayed.')
    if report['performance']:
        groups = defaultdict(list)
        for row in clients:
            count = row['expected_input_tokens']
            groups['short_input' if count <= 128 else 'medium_input' if count <= 512 else 'long_input'].append(row)
        report['performance_by_input_size'] = {k: stats(v, trace['duration_seconds'], manifest['config']['ttft_slo'])
                                               for k, v in groups.items()}
    polls = [r for r in observations if start <= r['received_at'] <= end]
    ages = [r['state_age_seconds'] for r in polls if 'metrics' in r]
    loop = [r['lag_seconds'] for r in records if r['event'] == 'event_loop_lag' and start <= r['monotonic'] <= end]
    report['timing'] = dict(p99_state_age=percentile(ages, .99), max_state_age=max(ages, default=None),
                          p99_metrics_roundtrip=percentile([r['received_at'] - r['sent_at'] for r in polls], .99),
                          max_state_poll_gap=max((b['received_at'] - a['received_at'] for a, b in zip(polls, polls[1:])), default=None),
                          p99_event_loop_lag=percentile(loop, .99), max_event_loop_lag=max(loop, default=None))
    save(output / 'report.json', report)
    if plot:
        render(output, manifest, trace, clients, records, events, report)
    return report


def render(output, manifest, trace, clients, records, events, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start = manifest['measurement_start']
    duration = trace['duration_seconds']
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    times = {
        'Planned arrivals': [r['scheduled_at'] - start for r in clients],
        'Client starts': [r['started_at'] - start for r in clients if r['started_at'] is not None],
        'Gateway received': [e['monotonic'] - start for r in clients
                             for e in events.get(r.get('client_request_id'), []) if e['event'] == 'received'],
    }
    # One-second counts, not an interpolation of missing snapshots.
    for label, values in times.items():
        bins = Counter(math.floor(v) for v in values if v >= 0)
        n = max(math.ceil(duration), max(bins, default=0) + 1)
        axes[0].stairs([bins[i] for i in range(n)], list(range(n + 1)), label=label)
    axes[0].set_ylabel('Requests / 1 s bin')
    snapshots = [r for r in records if r['event'] == 'snapshot' and r['monotonic'] >= start]
    x = [r['monotonic'] - start for r in snapshots]
    axes[1].plot(x, [r['current_admitted_cost'] for r in snapshots], marker='.', label='Reserved cost')
    axes[1].axhline(manifest['config']['budget'], linestyle='--', label='Fixed budget')
    axes[1].set_ylabel('Cost')
    successful = [r for r in clients if r['success'] and r['first_content_at'] is not None]
    axes[2].scatter([r['started_at'] - start for r in successful], [r['ttft'] for r in successful],
                    s=8, alpha=.4, label='Client TTFT, successful only')
    ingress = [(r['started_at'] - start, e['monotonic'] - r['started_at']) for r in clients
               if r['started_at'] is not None for e in events.get(r.get('client_request_id'), []) if e['event'] == 'received']
    axes[2].scatter([p[0] for p in ingress], [p[1] for p in ingress], s=8, alpha=.4, label='Client start to handler received')
    axes[2].set_ylabel('Seconds')
    loop = [r for r in records if r['event'] == 'event_loop_lag' and r['monotonic'] >= start]
    axes[3].plot([r['monotonic'] - start for r in loop], [r['lag_seconds'] for r in loop], label='Event loop lag')
    axes[3].scatter([r['started_at'] - start for r in clients if r['started_at'] is not None],
                    [r['started_at'] - r['scheduled_at'] for r in clients if r['started_at'] is not None],
                    s=5, alpha=.4, label='Client scheduling lag')
    axes[3].set_ylabel('Seconds')
    for axis in axes:
        axis.axvline(duration, color='gray', linestyle=':')
        axis.grid(alpha=.2)
        axis.legend(loc='upper left')
    axes[-1].set_xlabel('Seconds since measurement start; dotted line = end of arrival window')
    fig.suptitle('Trace replay — ' + ('integrity valid' if report['valid'] else 'INVALID DATA') +
                 ('; load fidelity FAILED' if not report['load_fidelity']['valid'] else ''))
    fig.tight_layout()
    fig.savefig(output / 'timeseries.png', dpi=140)
    plt.close(fig)


def analyze(output, plot=True):
    output = Path(output)
    try:
        return _analyze(output, plot)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        report = dict(valid=False, issues=[f'Incomplete or malformed replay data: {exc}'],
                      load_fidelity=dict(valid=False, issues=['Cannot audit load fidelity']), performance=None)
        save(output / 'report.json', report)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    args = parser.parse_args()
    report = analyze(args.output_dir)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['valid'] and report['load_fidelity']['valid'] else 1)
