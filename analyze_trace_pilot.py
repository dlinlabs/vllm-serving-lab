"""Audit replay accounting separately from offered-load fidelity and performance."""
import argparse
from collections import Counter, defaultdict
import json
import math
import uuid
from pathlib import Path

from analyze_budget_pilot import percentile
from run_budget_pilot import save
from trace_control import (RESPONSE_FIELDS, budget_curve, has_control_plan, initial_control_version,
                           read_plan, validate_response)
from trace_workload import finite_number, load_trace, sha256_file

TERMINAL = {'completed', 'rejected', 'failed', 'cancelled'}
LIFECYCLE = {'received', 'accepted', 'backend_sent', 'first_content'} | TERMINAL
PRE_HEADER_ERRORS = {'ConnectError', 'ConnectTimeout', 'ReadError', 'ReadTimeout', 'WriteError',
                     'WriteTimeout', 'PoolTimeout', 'RemoteProtocolError', 'LocalProtocolError',
                     'ProxyError', 'UnsupportedProtocol', 'DecodingError', 'CancelledError'}
DIAGNOSTIC_EVENTS = {'diagnostics_started', 'diagnostics_stopped', 'diagnostic_cpu',
                     'gc_span', 'gc_unmatched', 'request_diagnostics'}


def diagnostic_audit(output, manifest, records, clients, warmup, shutdown):
    """Separate opt-in evidence integrity from the unchanged client/fidelity verdicts."""
    from telemetry import DIAGNOSTICS_SCHEMA, INVESTIGATION_PROTOCOL, REQUEST_PHASES
    issues = []
    result = dict(valid=False, issues=issues, gateway_mode='unknown', external_mode='unknown',
                  gc_coverage_complete=False, request_linked_prefill='unavailable',
                  note='Elapsed spans and aggregate engine metrics are not exclusive CPU or per-request GPU attribution.')

    def require(condition, message):
        if not condition and message not in issues:
            issues.append(message)

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def interval(left, right):
        return number(left) and number(right) and left <= right

    def gc_state(state):
        return (isinstance(state, dict) and type(state.get('enabled')) is bool and
                all(isinstance(state.get(k), (tuple, list)) and len(state[k]) == 3 and
                    all(type(v) is int and v >= 0 for v in state[k]) for k in ('counts', 'thresholds')))

    try:
        require(manifest['investigation_protocol'] == INVESTIGATION_PROTOCOL and
                manifest['diagnostics_schema_version'] == DIAGNOSTICS_SCHEMA, 'Unsupported diagnostic protocol/schema')
        enabled = manifest['gateway_diagnostics_requested']
        external = manifest['sampler']['enabled']
        require(type(enabled) is bool and type(external) is bool, 'Diagnostic modes must be explicit booleans')
        result.update(gateway_mode='enabled' if enabled else 'disabled',
                      external_mode='enabled' if external else 'disabled')
        require(enabled == manifest['config']['gateway_diagnostics'] == manifest['gateway_diagnostics_observed'],
                'Requested and observed diagnostic modes differ')
        require(external == (enabled or manifest['config']['engine_observations']),
                'External sampler mode differs from selected options')
        managed_sampler = any(key in manifest['sampler'] for key in ('mode', 'worker')) or 'sampler_mode' in manifest['config']
        sampler_mode = 'thread'
        if managed_sampler:
            sampler_mode = manifest['sampler']['mode']
            require(sampler_mode in ('thread', 'process') and sampler_mode == manifest['config']['sampler_mode'],
                    'Requested and observed sampler execution modes differ')
            require(external or (sampler_mode == 'thread' and manifest['sampler'].get('worker') is None),
                    'Disabled sampler has an execution worker')
        require(not enabled or external, 'Gateway diagnostics require external observations')
        metric_records = [manifest['initial_metrics'], manifest['post_warmup_metrics'], manifest['final_metrics']]
        metric_records += [r for r in records if r['event'] == 'snapshot']
        require(all(r.get('gateway_diagnostics') is enabled for r in metric_records),
                'Gateway metrics disagree with selected diagnostic mode')
        require(interval(manifest['warmup_start'], manifest['warmup_end']) and
                manifest['warmup_end'] <= manifest['measurement_start'], 'Invalid warmup boundaries')
        require(all(manifest['warmup_start'] <= r['started_at'] <= r['finished_at'] <= manifest['warmup_end']
                    for r in warmup), 'Warmup requests are outside recorded boundaries')
        require(not manifest.get('collector_errors'), 'A diagnostic collector failed')
        fingerprints = manifest['code_sha256']
        required_files = ('gateway.py', 'telemetry.py', 'run_trace_pilot.py', 'run_budget_pilot.py',
                          'sustained_load.py', 'trace_workload.py', 'diagnostic_sampling.py',
                          'analyze_trace_pilot.py', 'analyze_trace_dynamics.py', 'requirements-v1.txt')
        if has_control_plan(output, manifest):
            required_files += ('trace_control.py',)
        require(all(isinstance(fingerprints.get(k), str) and len(fingerprints[k]) == 64 and
                    all(c in '0123456789abcdef' for c in fingerprints[k]) for k in required_files),
                'Incomplete code fingerprints')
        packages = ('httpx', 'httpcore', 'fastapi', 'uvicorn') + (('psutil', 'prometheus-client') if external else ())
        require(all(isinstance(manifest['packages'].get(k), str) for k in packages),
                'Incomplete package fingerprints')
        diagnostic = [r for r in records if r['event'] in DIAGNOSTIC_EVENTS]
        require(all(r.get('diagnostics_schema_version') == DIAGNOSTICS_SCHEMA for r in diagnostic),
                'Diagnostic event schema mismatch')
        require(shutdown['status'] == 'closed' and shutdown['run_id'] == manifest['run_id'] and
                shutdown['dropped'] == shutdown['write_errors'] == 0,
                'Gateway diagnostic persistence is incomplete')
        finish = max([manifest['measurement_end']] +
                     [r['finished_at'] for r in clients + warmup if number(r.get('finished_at'))] +
                     [r['monotonic'] for r in records if r['event'] in TERMINAL])
        if not enabled:
            require(not diagnostic and 'diagnostics' not in shutdown,
                    'Disabled gateway produced diagnostic observations')
        else:
            starts = [r for r in diagnostic if r['event'] == 'diagnostics_started']
            stops = [r for r in diagnostic if r['event'] == 'diagnostics_stopped']
            require(len(starts) == len(stops) == 1, 'Missing or duplicate diagnostic coverage boundaries')
            begin, stop = starts[0], stops[0]
            left, right = begin['coverage_start'], stop['coverage_end']
            result.update(coverage_start=left, coverage_end=right)
            require(interval(left, right) and left <= manifest['warmup_start'] and right >= finish and
                    stop['coverage_start'] == left and right <= stop['monotonic'] <= shutdown['closed_at'],
                    'Diagnostic coverage does not bracket warmup, measurement and drain')
            require(begin['process_id'] == manifest['gateway_launch']['process_id'] and
                    begin['python_implementation'] == 'cpython' and begin['gil_enabled'] is True,
                    'Unsupported diagnostic process/runtime identity')
            require(gc_state(begin.get('gc')) and all(gc_state(r.get('diagnostics', {}).get('gc'))
                                                     for r in metric_records),
                    'Missing GC enabled/threshold/count observations')
            health = shutdown['diagnostics']
            require(shutdown['diagnostics_schema_version'] == DIAGNOSTICS_SCHEMA and health == stop['health'],
                    'Final diagnostic health does not match shutdown')
            require(health['closed'] is True and health['quiesced'] is True and health['complete'] is True and
                    health['registration_missing'] is False and
                    all(health[k] == 0 for k in ('lost_callbacks', 'undelivered_callbacks', 'buffered_callbacks',
                        'inflight_callbacks', 'callback_errors', 'pending_spans', 'unmatched_boundaries')),
                    'GC callback coverage is incomplete')
            spans = [r for r in diagnostic if r['event'] == 'gc_span']
            cpu = [r for r in diagnostic if r['event'] == 'diagnostic_cpu']
            unmatched = [r for r in diagnostic if r['event'] == 'gc_unmatched']
            require(health['callbacks_produced'] == health['callbacks_received'] == 2 * len(spans) + len(unmatched) and
                    health['gc_spans'] == len(spans) and health['unmatched_boundaries'] == len(unmatched) and
                    not unmatched, 'GC edge/span counts do not reconcile')
            sequences = []
            for span in spans:
                require(interval(span['span_start'], span['span_end']) and
                        left <= span['span_start'] <= span['span_end'] <= right and
                        span['span_end'] <= span['monotonic'], 'Invalid GC actual-span/emission timestamps')
                require(type(span['generation']) is int and span['generation'] in (0, 1, 2) and
                        type(span['callback_thread_id']) is int and type(span['callback_native_thread_id']) is int and
                        span['on_loop_thread'] is (span['callback_thread_id'] == begin['loop_thread_id']),
                        'Invalid GC thread/generation identity')
                require(number(span['callback_thread_cpu_seconds']) and span['callback_thread_cpu_seconds'] >= 0,
                        'Invalid GC callback-thread CPU duration')
                require(all(type(span[k]) is int and span[k] >= 0 for k in
                            ('start_sequence', 'stop_sequence', 'collected', 'uncollectable')) and
                        span['start_sequence'] < span['stop_sequence'], 'Invalid GC sequence or collection counts')
                sequences.extend([span['start_sequence'], span['stop_sequence']])
            require(len(set(sequences)) == len(sequences), 'Duplicate GC boundary sequences')
            require(len(cpu) >= 2 and health['cpu_samples'] == len(cpu) and
                    cpu[0]['sampled_at'] <= manifest['warmup_start'] and cpu[-1]['sampled_at'] >= finish,
                    'Gateway CPU observations do not bracket warmup, measurement and drain')
            for row in cpu:
                require(interval(row['sampled_at'], row['monotonic']) and
                        left <= row['sampled_at'] <= stop['monotonic'] and
                        row['loop_thread_id'] == begin['loop_thread_id'] and gc_state(row.get('gc')),
                        'Invalid loop-thread CPU identity/timestamp')
                require(all(number(row[k]) and row[k] >= 0 for k in
                            ('loop_thread_cpu_seconds', 'process_cpu_seconds')), 'Invalid gateway CPU counters')
            for previous, current in zip(cpu, cpu[1:]):
                require(all(previous[k] <= current[k] for k in
                            ('sampled_at', 'loop_thread_cpu_seconds', 'process_cpu_seconds')),
                        'Gateway CPU observations went backwards')
            histories, summaries = defaultdict(list), defaultdict(list)
            for row in records:
                if row['event'] in LIFECYCLE:
                    histories[row['request_id']].append(row)
                elif row['event'] == 'request_diagnostics':
                    summaries[row['request_id']].append(row)
            require(set(summaries) == set(histories) and all(len(v) == 1 for v in summaries.values()),
                    'Missing, extra or duplicate request diagnostic summaries')
            for identifier, values in summaries.items():
                summary = values[0]
                history = histories[identifier]
                terminal = [r for r in history if r['event'] in TERMINAL]
                require(len(terminal) == 1 and summary['outcome'] == terminal[0]['event'] and
                        summary['cleanup_complete'] is True, 'Request diagnostic terminal/cleanup mismatch')
                require(interval(summary['span_start'], summary['span_end']) and
                        left <= history[0]['monotonic'] <= summary['span_start'] <= terminal[0]['monotonic'] <=
                        summary['span_end'] <= summary['monotonic'] <= right,
                        'Request summary was emitted before cleanup or outside coverage')
                phases = summary['phases']
                require(set(phases) == set(REQUEST_PHASES), 'Request phase schema mismatch')
                previous_end = summary['span_start']
                for name in REQUEST_PHASES:
                    phase = phases[name]
                    if phase['status'] == 'not_reached':
                        require(phase['start'] is phase['end'] is None, 'Unreached phase contains timestamps')
                    else:
                        require(phase['status'] in {'completed', 'failed', 'cancelled'} and
                                interval(phase['start'], phase['end']) and previous_end <= phase['start'] and
                                phase['end'] <= summary['span_end'], 'Invalid, incomplete or out-of-order request phase')
                        previous_end = phase['end']
                require(phases['body_read']['status'] != 'not_reached', 'Missing request body-read phase')
                for name in ('first_body_at', 'first_content_at'):
                    if summary[name] is not None:
                        require(number(summary[name]) and summary['span_start'] <= summary[name] <= summary['span_end'] and
                                phases['http_send_headers']['end'] <= summary[name],
                                'Invalid first-body/content milestone')
                first = [r for r in history if r['event'] == 'first_content']
                require(bool(first) == (summary['first_content_at'] is not None), 'Missing or extra first-content milestone')
                if first:
                    require(interval(summary['first_body_at'], summary['first_content_at']) and
                            summary['first_content_at'] <= first[0]['monotonic'], 'Content milestone ordering is invalid')
            result.update(gc_coverage_complete=not issues, gc_spans=len(spans), cpu_samples=len(cpu),
                          request_summaries=sum(map(len, summaries.values())))
        paths = (output / 'engine_observations.jsonl', output / 'engine_observations_status.json')
        if not external:
            require(not any(path.exists() for path in paths), 'Disabled external sampler produced observations')
        else:
            from diagnostic_sampling import metric_inventory, REQUIRED_METRICS
            observations = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
            status = json.loads(paths[1].read_text())
            sampler_pid = manifest['runner_process_id']
            if managed_sampler:
                worker = manifest['sampler']['worker']
                sampler_pid = worker['process_id']
                require(all(type(pid) is int and pid > 0 for pid in
                            (sampler_pid, manifest['runner_process_id'], manifest['gateway_launch']['process_id'])) and
                        (sampler_pid == manifest['runner_process_id'] if sampler_mode == 'thread' else
                         sampler_pid not in (manifest['runner_process_id'], manifest['gateway_launch']['process_id'])),
                        'Invalid sampler execution process identity')
                require(worker['started'] is True and worker['stopped'] is True and
                        worker['forced_termination'] is False and worker['unexpected_exit'] is False and
                        (worker['start_method'] is None and worker['exitcode'] is None if sampler_mode == 'thread' else
                         worker['start_method'] == 'spawn' and type(worker['exitcode']) is int and worker['exitcode'] == 0),
                        'External sampler worker did not stop cleanly')
                require(interval(status['started_at'], worker['ready_at']) and
                        interval(observations[0]['finished_at'], worker['ready_at']) and
                        worker['ready_at'] <= manifest['warmup_start'] and
                        interval(finish, worker['stop_requested_at']) and
                        observations[-1]['sent_at'] >= worker['stop_requested_at'],
                        'Sampler readiness/final sample does not bracket collection')
            sampler_identity = (managed_sampler or 'sampler_process_id' in status or
                                any('sampler_process_id' in row for row in observations))
            if sampler_identity:
                require(type(status.get('sampler_process_id')) is int and
                        type(status.get('runner_process_id')) is int and
                        status['sampler_process_id'] == sampler_pid and
                        status.get('runner_process_id') == manifest['runner_process_id'],
                        'External collector process identity mismatch')
            require(status['run_id'] == manifest['run_id'] and status['status'] == 'closed' and
                    status['diagnostics_schema_version'] == DIAGNOSTICS_SCHEMA and
                    status['samples'] == len(observations) and status['sample_errors'] == status['write_errors'] == 0,
                    'External collector persistence/health is incomplete')
            require(len(observations) >= 2 and observations[0]['finished_at'] <= manifest['warmup_start'] and
                    observations[-1]['sent_at'] >= finish and status['started_at'] <= observations[0]['sent_at'] and
                    observations[-1]['finished_at'] <= status['closed_at'] <= shutdown['closed_at'],
                    'External observations do not bracket warmup, measurement and drain')
            previous_counters, previous_exporter, epoch = {}, None, 0
            identities = {}
            for i, row in enumerate(observations):
                if sampler_identity:
                    require(type(row.get('sampler_process_id')) is int and row['sampler_process_id'] == sampler_pid,
                            'External observation collector process mismatch')
                require(row['run_id'] == manifest['run_id'] and row['diagnostics_schema_version'] == DIAGNOSTICS_SCHEMA and
                        row['host'] == manifest['host'] and row['source'] == observations[0]['source'],
                        'External observation identity/schema mismatch')
                require(row['interval_seconds'] == status['interval_seconds'] == manifest['sampler']['interval_seconds'] and
                        number(row['interval_seconds']) and row['interval_seconds'] > 0,
                        'External sampler cadence mismatch')
                require(interval(row['sent_at'], row['received_at']) and interval(row['received_at'], row['finished_at']) and
                        (i == 0 or observations[i-1]['finished_at'] <= row['sent_at']),
                        'Invalid external observation timestamps')
                require(row['error_type'] is None and not row['unavailable_metrics'],
                        'Required engine observations are unavailable')
                if isinstance(row['raw_metrics'], str) and row['error_type'] is None:
                    inventory, counters, exporter = metric_inventory(row['raw_metrics'])
                    require(row['metrics'] == inventory and set(inventory) == set(REQUIRED_METRICS) and
                            all(v['available'] for v in inventory.values()) and row['exporter_identity'] == exporter,
                            'Engine inventory does not match raw metrics')
                    resets = sorted(k for k, v in counters.items() if k in previous_counters and v < previous_counters[k])
                    disappeared = sorted(previous_counters.keys() - counters.keys())
                    changed = previous_exporter is not None and previous_exporter != exporter
                    epoch += bool(resets or disappeared or changed)
                    require(sorted(row['counter_resets']) == resets and row['disappeared_counters'] == disappeared and
                            row['exporter_identity_changed'] == changed and row['counter_epoch'] == epoch,
                            'Engine counter reset/epoch mismatch')
                    previous_counters, previous_exporter = counters, exporter
                else:
                    require(False, 'Raw engine metrics are missing')
                for name in ('gateway', 'runner', 'host_cpu'):
                    resource = row['resources'][name]
                    require(resource['available'] is True, 'Required current RSS/CPU observations are unavailable')
                    require(interval(resource['sample_start'], resource['sample_end']) and
                            row['received_at'] <= resource['sample_start'] <= resource['sample_end'] <= row['finished_at'],
                            'Invalid resource observation timestamps')
                    if not resource['available']:
                        continue
                    if name != 'host_cpu':
                        expected_pid = (manifest['gateway_launch']['process_id'] if name == 'gateway'
                                        else manifest['runner_process_id'])
                        require(resource['pid'] == expected_pid and number(resource['process_created_unix']) and
                                type(resource['rss_bytes']) is int and resource['rss_bytes'] >= 0 and
                                all(number(resource[k]) and resource[k] >= 0 for k in
                                    ('user_cpu_seconds', 'system_cpu_seconds')),
                                'Invalid current RSS/process CPU identity or counters')
                        identity = (resource['pid'], resource['process_created_unix'])
                        require(name not in identities or identities[name] == identity, 'Observed resource process restarted')
                        identities[name] = identity
                        if i and observations[i-1]['resources'][name]['available']:
                            require(all(observations[i-1]['resources'][name][k] <= resource[k] for k in
                                        ('user_cpu_seconds', 'system_cpu_seconds')), 'Process CPU counters went backwards')
                    else:
                        require(type(resource['logical_cpus']) is int and resource['logical_cpus'] > 0 and
                                bool(resource['cpu_seconds']) and
                                all(number(v) and v >= 0 for v in resource['cpu_seconds'].values()),
                                'Invalid host CPU observations')
            result.update(engine_samples=len(observations), engine_counter_epochs=epoch + 1)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        issues.append(f'Incomplete or malformed diagnostic data: {type(exc).__name__}: {exc}')
    result['valid'] = not issues
    return result


def control_audit(manifest, plan, changes, records, observations):
    issues, fidelity = [], []
    def require(condition, message):
        if not condition and message not in issues:
            issues.append(message)
    start = finite_number(manifest['measurement_start'], 'measurement start')
    end = finite_number(manifest['measurement_end'], 'measurement end')
    base = initial_control_version(manifest, plan)
    schedule = plan['budget_schedule']
    updates = [r for r in records if r['event'] == 'budget_updated']
    require(len(changes) == len(updates) == len(schedule), 'Missing or extra scheduled budget commands/applications')
    sends, applications, roundtrips = [], [], []
    previous_ack = manifest['warmup_end']
    for index, (step, command, applied) in enumerate(zip(schedule, changes, updates)):
        if not isinstance(command, dict) or command.get('status') != 'acknowledged':
            require(False, 'Budget command did not receive a successful acknowledgement')
            continue
        previous_budget = schedule[max(0, index - 1)]['budget']
        validate_response(command, step['budget'], previous_budget, base + index + 1,
                          command['sent'], command['acknowledged'])
        validate_response(applied, step['budget'], previous_budget, base + index + 1,
                          command['sent'], command['acknowledged'])
        require(all(command[name] == applied[name] for name in RESPONSE_FIELDS),
                'Budget acknowledgement differs from the gateway application')
        require(type(command.get('step_index')) is int and command['step_index'] == index,
                'Budget command order differs from the schedule')
        require(finite_number(command['requested_budget'], 'requested budget', True) == step['budget'],
                'Requested budget differs from the schedule')
        target = start + step['at_seconds']
        require(abs(finite_number(command['target'], 'budget target') - target) < 1e-6,
                'Budget target differs from the shared measurement clock')
        require(not command.get('error_type') and not command.get('error'), 'Acknowledged budget command has an error')
        require(previous_ack <= command['sent'] and
                command['sent'] <= applied['applied_at'] <= applied['monotonic'] <= command['acknowledged'],
                'Budget application/acknowledgement ordering is invalid')
        roundtrips.append(command['acknowledged'] - command['sent'])
        if index == 0:
            require(command['acknowledged'] <= start, 'Initial budget was not acknowledged before measurement')
        else:
            require(previous_ack <= target, 'Previous budget acknowledgement crossed the next target')
            require(command['sent'] >= target - 1e-6 and applied['applied_at'] < end,
                    'Scheduled budget was applied early or outside the arrival window')
            sends.append(command['sent'] - target)
            applications.append(applied['applied_at'] - target)
        previous_ack = command['acknowledged']
    final = manifest['final_metrics']
    require(type(final['budget_version']) is int and final['budget_version'] == base + len(schedule) and
            finite_number(final['admission_budget'], 'final budget', True) == schedule[-1]['budget'],
            'Final budget state differs from the declared schedule')
    if not issues:
        metrics = ([final] + [r['metrics'] for r in observations if 'metrics' in r] +
                   [r for r in records if r['event'] == 'snapshot'])
        states = [(r, r['state_sampled_at'], 'admission_budget') for r in metrics]
        # Cost-aware admission emits its decision before yielding the gateway loop.
        states += [(r, r['monotonic'], 'budget') for r in records if r['event'] in {'accepted', 'rejected'}]
        for state, timestamp, budget_field in states:
            sampled = finite_number(timestamp, 'budget state time')
            count = sum(r['applied_at'] <= sampled for r in updates)
            require(type(state['budget_version']) is int and state['budget_version'] == base + count and
                    finite_number(state[budget_field], 'observed budget', True) == schedule[max(0, count - 1)]['budget'],
                    'Observed budget state differs from application history')
    if issues:
        fidelity.append('Cannot qualify an incomplete or inconsistent control ledger')
    if max(sends + applications, default=0) > plan['max_budget_lag']:
        fidelity.append('Budget control exceeded predeclared maximum send/application lag')
    return issues, dict(valid=not fidelity, issues=fidelity, budget_mode=plan['budget_mode'],
                        planned_commands=len(schedule), recorded_commands=len(changes), gateway_applications=len(updates),
                        max_budget_lag_limit=plan['max_budget_lag'],
                        p99_send_lag=percentile(sends, .99), max_send_lag=max(sends, default=None),
                        p99_application_lag=percentile(applications, .99), max_application_lag=max(applications, default=None),
                        p99_acknowledgement_roundtrip=percentile(roundtrips, .99),
                        note='Initial budget is acknowledged before measurement; lag metrics cover subsequent steps only.')


def stats(rows, duration, slo, e2e_slo=None):
    finite_number(slo, 'TTFT SLO', True)
    if e2e_slo is not None:
        finite_number(e2e_slo, 'E2E SLO', True)
    successful = [r for r in rows if r['success']]
    actual = [r['ttft'] for r in successful]
    scheduled = [r['first_content_at'] - r['scheduled_at'] for r in successful]
    good = sum(r['first_content_at'] - r['scheduled_at'] <= slo and
               (e2e_slo is None or r['finished_at'] - r['scheduled_at'] <= e2e_slo) for r in successful)
    result = dict(attempted=len(rows), success=len(successful), rejected=sum(r['rejected'] for r in rows),
                failed=sum(not r['success'] and not r['rejected'] for r in rows),
                client_p99_ttft=percentile(actual, .99), scheduled_p99_ttft=percentile(scheduled, .99),
                client_p99_e2e=percentile([r['end_to_end'] for r in successful], .99),
                slo_successes=good, slo_goodput_per_arrival_second=good / duration,
                slo_success_fraction_of_offered=good / len(rows) if rows else None,
                slo_definition='stream completed successfully; first content minus scheduled arrival <= TTFT SLO')
    if e2e_slo is not None:
        completed = [r['finished_at'] - r['scheduled_at'] for r in successful]
        result.update(scheduled_p99_e2e=percentile(completed, .99),
                      ttft_slo_successes=sum(value <= slo for value in scheduled),
                      e2e_slo_successes=sum(value <= e2e_slo for value in completed),
                      slo_definition=result['slo_definition'] + '; completion minus scheduled arrival <= E2E SLO')
    return result


def _analyze(output, plot):
    manifest = json.loads((output / 'manifest.json').read_text())
    trace = load_trace(output / 'trace.json')
    plan = read_plan(output, manifest, trace)
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
    if plan is None:
        updates = [r for r in records if r['event'] == 'budget_updated']
        require(not any('step_index' in r or 'requested_budget' in r for r in changes),
                'Budget command metadata requires a declared control plan')
        require(len(changes) == len(updates) == 1, 'Fixed-budget replay must have one measured budget command')
        for command, applied in zip(changes, updates):
            require(command['budget'] == applied['budget'] == manifest['config']['budget'], 'Budget value mismatch')
            require(command['budget_version'] == applied['budget_version'], 'Budget version mismatch')
            require(command['applied_at'] == applied['applied_at'], 'Budget application timestamp mismatch')
            require(command['sent'] <= applied['applied_at'] <= applied['monotonic'] <= command['acknowledged'],
                    'Budget application timing is invalid')
            require(applied['applied_at'] <= start, 'Initial budget was not applied before measurement')
    else:
        control_issues, control_fidelity = control_audit(manifest, plan, changes, records, observations)
        issues.extend(control_issues)
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
                  performance=stats(clients, trace['duration_seconds'], manifest['config']['ttft_slo'],
                                    manifest['config'].get('e2e_slo')) if not issues else None,
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
        report['performance_by_input_size'] = {k: stats(v, trace['duration_seconds'], manifest['config']['ttft_slo'],
                                                       manifest['config'].get('e2e_slo'))
                                               for k, v in groups.items()}
    if plan is not None:
        report['control_fidelity'] = control_fidelity
    polls = [r for r in observations if start <= r['received_at'] <= end]
    ages = [r['state_age_seconds'] for r in polls if 'metrics' in r]
    loop = [r['lag_seconds'] for r in records if r['event'] == 'event_loop_lag' and start <= r['monotonic'] <= end]
    report['timing'] = dict(p99_state_age=percentile(ages, .99), max_state_age=max(ages, default=None),
                          p99_metrics_roundtrip=percentile([r['received_at'] - r['sent_at'] for r in polls], .99),
                          max_state_poll_gap=max((b['received_at'] - a['received_at'] for a, b in zip(polls, polls[1:])), default=None),
                          p99_event_loop_lag=percentile(loop, .99), max_event_loop_lag=max(loop, default=None))
    if 'investigation_protocol' in manifest:
        report['diagnostic_integrity'] = diagnostic_audit(output, manifest, records, clients, warmup, shutdown)
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
    if 'control_fidelity' in report:
        unavailable = 'no pre-measurement application'
        try:
            times, budgets = budget_curve(records, start, max([start + duration] + [r['monotonic'] for r in snapshots]))
        except ValueError as exc:
            times, budgets = [], []
            unavailable = str(exc)
        if times:
            axes[1].step(times, budgets, where='post', linestyle='--', label='Applied budget')
        else:
            axes[1].text(.02, .9, 'Applied budget unavailable: ' + unavailable,
                         transform=axes[1].transAxes, fontsize=8)
    else:
        axes[1].axhline(manifest['config']['budget'], linestyle='--', label='Fixed budget')
    axes[1].set_ylabel('Cost')
    successful = [r for r in clients if r['success'] and r['first_content_at'] is not None]
    axes[2].scatter([r['started_at'] - start for r in successful], [r['ttft'] for r in successful],
                    s=8, alpha=.4, label='Client TTFT, successful only')
    if manifest['config'].get('e2e_slo') is not None:
        axes[2].scatter([r['scheduled_at'] - start for r in successful],
                        [r['first_content_at'] - r['scheduled_at'] for r in successful],
                        s=8, alpha=.4, label='Scheduled TTFT')
        axes[2].scatter([r['scheduled_at'] - start for r in successful],
                        [r['finished_at'] - r['scheduled_at'] for r in successful],
                        s=8, alpha=.4, label='Scheduled E2E')
        axes[2].axhline(manifest['config']['ttft_slo'], linestyle=':', label='TTFT SLO')
        axes[2].axhline(manifest['config']['e2e_slo'], linestyle='--', label='E2E SLO')
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
                 ('; load fidelity FAILED' if not report['load_fidelity']['valid'] else '') +
                 ('; control fidelity FAILED' if not report.get('control_fidelity', {'valid': True})['valid'] else ''))
    fig.tight_layout()
    fig.savefig(output / 'timeseries.png', dpi=140)
    plt.close(fig)


def analyze(output, plot=True):
    output = Path(output)
    try:
        return _analyze(output, plot)
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        report = dict(valid=False, issues=[f'Incomplete or malformed replay data: {exc}'],
                      load_fidelity=dict(valid=False, issues=['Cannot audit load fidelity']), performance=None)
        try:
            manifest = json.loads((output / 'manifest.json').read_text())
        except (OSError, ValueError) as metadata_error:
            report['issues'].append(f'Cannot determine diagnostic protocol: {metadata_error}')
        else:
            if isinstance(manifest, dict) and has_control_plan(output, manifest):
                report['control_fidelity'] = dict(valid=False, issues=['Cannot audit declared budget control'])
            if isinstance(manifest, dict) and 'investigation_protocol' in manifest:
                report['diagnostic_integrity'] = dict(valid=False,
                    issues=['Core replay artifacts are incomplete; diagnostic coverage cannot be qualified'],
                    gc_coverage_complete=False, request_linked_prefill='unavailable')
        save(output / 'report.json', report)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    args = parser.parse_args()
    report = analyze(args.output_dir)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['valid'] and report['load_fidelity']['valid'] and
                     report.get('control_fidelity', {'valid': True})['valid'] and
                     report.get('diagnostic_integrity', {'valid': True})['valid'] else 1)
