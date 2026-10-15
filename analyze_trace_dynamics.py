"""Offline trace diagnosis. Audit raw files in a temporary directory; never rewrite them.

Requires NumPy and Matplotlib. Percentiles use linear interpolation. Stage shares
use the same successful-request cohort, not differences between percentiles.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np

from analyze_trace_pilot import LIFECYCLE, TERMINAL, analyze as audit
from trace_control import PLAN_FILE, budget_curve, has_control_plan
from trace_workload import finite_number
from runner_diagnostics import FILES as RUNNER_RAW
from runner_gc_control import FILE as GC_RAW

RAW = ('manifest.json', 'client.json', 'warmup.json', 'gateway.jsonl', 'trace.json',
       'budget_changes.json', 'state_observations.json', 'telemetry_status.json',
       'report.json', 'timeseries.png')
LEGACY_RAW = RAW
SAMPLER_RAW = ('engine_observations.jsonl', 'engine_observations_status.json')
RAW += SAMPLER_RAW + (PLAN_FILE,) + RUNNER_RAW + (GC_RAW,)
STAGES = ('ingress', 'admission', 'client_build', 'post_dispatch', 'delivery')
LABELS = ('Client start → handler', 'Handler → accepted', 'Accepted → dispatch',
          'Dispatch → gateway content', 'Gateway → client content')
COLORS = ('#7199c1', '#b0c5dc', '#edb965', '#366a56', '#ce8598')


def hashes(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in RAW if (root / name).is_file()}


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


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def difference(end, start):
    return end - start if finite(end) and finite(start) else None


def read_jsonl(path):
    records, malformed = [], []
    if path.is_file():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                malformed.append(number)
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                malformed.append(number)
    return records, malformed


def interval_overlap(left, right, spans):
    """Intersect with an interval union, never sum overlapping wall intervals."""
    if not finite(left) or not finite(right) or right < left:
        return None
    intervals = sorted((max(left, a), min(right, b)) for a, b in spans
                       if finite(a) and finite(b) and a <= b)
    total, previous_end = 0., -math.inf
    for a, b in intervals:
        if b > a:
            total += max(0., b - max(a, previous_end))
            previous_end = max(previous_end, b)
    return total


def cpu_intervals(samples, start, host=False):
    """Difference adjacent readings without bridging unavailable or invalid samples."""
    intervals = []
    latest_end = -math.inf
    for index, (previous, current) in enumerate(zip(samples, samples[1:])):
        issues = list(dict.fromkeys(previous['issues'] + current['issues']))
        bounds = [[difference(sample.get(key), start) for key in ('start', 'end')]
                  for sample in (previous, current)]
        bounds = [[value if finite(value) else None for value in pair] for pair in bounds]
        timed = all(finite(value) for pair in bounds for value in pair)
        left = right = elapsed = None
        if timed:
            left, right = [a + (b-a)/2 for a, b in bounds]
            elapsed = right-left
            if (any(b < a for a, b in bounds) or bounds[0][1] > bounds[1][0] or
                    not finite(elapsed) or elapsed <= 0):
                issues.append('invalid_or_overlapping_read_times')
            if finite(left) and left < latest_end:
                issues.append('out_of_order_interval')
            if finite(right):
                latest_end = max(latest_end, right)
        else:
            issues.append('missing_read_times')
        if previous['identity'] != current['identity']:
            issues.append('identity_changed')
        old, new = previous['counters'], current['counters']
        if (not old or old.keys() != new.keys() or
                not all(finite(v) and v >= 0 for values in (old, new) for v in values.values())):
            issues.append('missing_or_invalid_cpu_counters')
        delta = None
        if not issues:
            delta = {key: new[key]-old[key] for key in old}
            if not all(finite(v) and v >= 0 for v in delta.values()):
                issues.append('cpu_counter_decreased_or_nonfinite')
        cores = nonidle = None
        if not issues:
            if host:
                # These fields overlap user/nice (Linux) or system (Windows).
                duplicate = {'guest', 'guest_nice', 'interrupt', 'dpc'}
                if 'dpc' in delta:
                    duplicate.add('irq')
                counted = {key: value for key, value in delta.items() if key not in duplicate}
                total = sum(counted.values())
                if not finite(total) or total <= 0:
                    issues.append('host_cpu_counters_did_not_advance')
                else:
                    nonidle = sum(value for key, value in counted.items() if key not in {'idle', 'iowait'})/total
                    if not finite(nonidle) or not 0 <= nonidle <= 1:
                        issues.append('invalid_host_nonidle_fraction')
            else:
                cores = sum(delta.values())/elapsed
                if not finite(cores):
                    issues.append('nonfinite_cpu_rate')
        intervals.append(dict(previous_sample_index=index, current_sample_index=index+1,
            sample_bounds_s=bounds, start_s=left if finite(left) else None, end_s=right if finite(right) else None,
            elapsed_s=elapsed if finite(elapsed) and elapsed > 0 else None,
            cpu_seconds=delta if not issues else None, cpu_cores=cores if not issues else None,
            host_nonidle_fraction=nonidle if not issues else None, issues=issues))
    valid = sum(not row['issues'] for row in intervals)
    return dict(sample_count=len(samples), valid_interval_count=valid,
                invalid_interval_count=len(intervals)-valid, intervals=intervals,
                status='partial' if valid and valid < len(intervals) else 'available' if valid else 'unavailable',
                reason=('fewer_than_two_samples' if len(samples) < 2 else
                        'invalid_intervals' if valid < len(intervals) else None))


def analyze_cpu(joined, records, observations, start, manifest, report):
    """Offline interval context, not CPU attribution to a delayed launch."""
    integrity = report.get('diagnostic_integrity') or {}
    sampler = manifest.get('sampler', {})
    sampler_mode = sampler.get('mode', 'thread') if sampler.get('enabled') is True else None
    sources = {}
    for name in ('runner', 'gateway', 'host', 'gateway_loop'):
        internal = name == 'gateway_loop'
        enabled = (manifest.get('gateway_diagnostics_requested') is True if internal else
                   manifest.get('sampler', {}).get('enabled') is True)
        if 'investigation_protocol' not in manifest or not enabled or not integrity.get('valid'):
            status = ('legacy_unavailable' if 'investigation_protocol' not in manifest else
                      'disabled' if not enabled else 'unqualified')
            sources[name] = dict(status=status, reason=status, intervals=None,
                                 sample_count=None, valid_interval_count=None, invalid_interval_count=None)
            continue
        samples = []
        rows = ([row for row in records if row.get('event') == 'diagnostic_cpu'] if internal else observations)
        for row in rows:
            issues = []
            if internal:
                sampled = row.get('sampled_at')
                identity = (row.get('run_id'), row.get('loop_thread_id'))
                if (identity[0] != manifest.get('run_id') or type(identity[1]) is not int or identity[1] <= 0):
                    issues.append('invalid_loop_identity')
                samples.append(dict(start=sampled, end=sampled, identity=identity, issues=issues,
                                    counters=dict(loop_thread_cpu_seconds=row.get('loop_thread_cpu_seconds'))))
                continue
            resources = row.get('resources')
            resource = resources.get('host_cpu' if name == 'host' else name) if isinstance(resources, dict) else None
            resource = resource if isinstance(resource, dict) else {}
            if resource.get('available') is not True:
                issues.append(resource.get('error_reason') or resource.get('error_type') or 'resource_unavailable')
            if (row.get('run_id') != manifest.get('run_id') or not row.get('host') or
                    row.get('host') != manifest.get('host')):
                issues.append('invalid_observation_identity')
            identity = (row.get('run_id'), row.get('host'))
            if name == 'host':
                count = resource.get('logical_cpus')
                identity += (count,)
                if type(count) is not int or count <= 0:
                    issues.append('invalid_logical_cpu_count')
                counters = resource.get('cpu_seconds')
                counters = counters if isinstance(counters, dict) else {}
                supported = {'user', 'system', 'idle', 'nice', 'iowait', 'irq', 'softirq',
                             'steal', 'guest', 'guest_nice', 'interrupt', 'dpc'}
                if not {'user', 'system', 'idle'} <= counters.keys() or not counters.keys() <= supported:
                    issues.append('unsupported_host_cpu_fields')
            else:
                pid, created = resource.get('pid'), resource.get('process_created_unix')
                expected = (manifest.get('runner_process_id') if name == 'runner' else
                            manifest.get('gateway_launch', {}).get('process_id'))
                if type(pid) is not int or pid <= 0 or pid != expected or not finite(created) or created <= 0:
                    issues.append('invalid_process_identity')
                identity += (pid, created)
                counters = {key: resource.get(key) for key in ('user_cpu_seconds', 'system_cpu_seconds')}
            samples.append(dict(start=resource.get('sample_start'), end=resource.get('sample_end'),
                                identity=identity, counters=counters, issues=issues))
        sources[name] = cpu_intervals(samples, start, host=name == 'host')
    threshold = manifest['config']['max_scheduling_lag']
    selection_issue = None if finite(threshold) and threshold > 0 else 'invalid_scheduling_lag_limit'
    late = [] if selection_issue is None else None
    for index, row in enumerate(joined if selection_issue is None else []):
        if not finite(row['scheduling_lag_s']) or row['scheduling_lag_s'] <= threshold:
            continue
        if not finite(row['scheduled_s']) or not finite(row['started_s']):
            continue
        overlaps = {}
        for name, source in sources.items():
            if source['intervals'] is None:
                overlaps[name] = None
                continue
            matches = []
            for number, interval in enumerate(source['intervals']):
                a, b = interval['sample_bounds_s'][0][0], interval['sample_bounds_s'][1][1]
                if not finite(a) or not finite(b) or b <= a:
                    continue
                overlap = interval_overlap(row['scheduled_s'], row['started_s'], [(a, b)])
                if overlap > 0:
                    matches.append(dict(interval_index=number, possible_overlap_s=overlap))
            overlaps[name] = matches
        late.append(dict(ledger_index=index, request_id=row['request_id'], trace_id=row['trace_id'],
                         outcome=row['outcome'], scheduled_s=row['scheduled_s'], started_s=row['started_s'],
                         scheduling_lag_s=row['scheduling_lag_s'], cpu_interval_overlaps=overlaps))
    return dict(schema_version=1, sampler_mode=sampler_mode, sources=sources,
                late_launches=late, selection_issue=selection_issue,
                scheduling_lag_limit_s=threshold if finite(threshold) else None,
                launch_timing_unavailable_count=sum(not finite(row['scheduling_lag_s']) for row in joined),
                note='CPU rates are whole-interval averages, not instantaneous peaks or CPU spent during a late launch. '
                     'External rates use resource-read midpoints; original read bounds are retained. Gateway loop CPU '
                     'uses sampled_at, not emission time. Overlaps use outer read bounds and may be uncertain at edges; '
                     'never prorate CPU by overlap duration. Process cores are CPU seconds per elapsed second, not '
                     'normalized by host CPU count, and may exceed one. Host non-idle fraction excludes idle/iowait '
                     'and double-counted fields; it is not container CPU headroom. Runner CPU includes sampler work '
                     'in thread mode, but excludes the separate sampler process in process mode; a lower runner '
                     'CPU rate across modes is not evidence of less total CPU work. '
                     'Gateway process/loop/GC CPU overlap and must not be added. Missing sampler-thread CPU, runner GC, '
                     'per-core load and CPU quota/throttling remain unknown. Timing overlap does not establish causation.')


def join_diagnostics(joined, records, observations, start, manifest, report):
    integrity = report.get('diagnostic_integrity')
    requested = manifest.get('gateway_diagnostics_requested')
    mode = ('legacy_unavailable' if 'investigation_protocol' not in manifest else
            'enabled' if requested is True else 'disabled' if requested is False else 'unknown')
    complete = bool(integrity and integrity.get('gc_coverage_complete'))
    phase_records = defaultdict(list)
    gc_records, cpu_records, phase_spans = [], [], defaultdict(list)
    for record in records:
        if record.get('event') == 'request_diagnostics':
            phase_records[record.get('request_id')].append(record)
            phases = record.get('phases', {})
            if isinstance(phases, dict):
                for name, phase in phases.items():
                    if isinstance(phase, dict):
                        phase_spans[name].append((phase.get('start'), phase.get('end')))
        elif record.get('event') == 'gc_span' and all(finite(record.get(k)) for k in ('span_start', 'span_end')):
            if record['span_start'] <= record['span_end']:
                gc_records.append(dict(start_s=record['span_start']-start, end_s=record['span_end']-start,
                    emission_s=difference(record.get('monotonic'), start),
                    on_loop_thread=record.get('on_loop_thread') if type(record.get('on_loop_thread')) is bool else None,
                    callback_thread_cpu_seconds=record.get('callback_thread_cpu_seconds')
                        if finite(record.get('callback_thread_cpu_seconds')) else None))
        elif record.get('event') == 'diagnostic_cpu' and all(finite(record.get(k)) for k in
                ('sampled_at', 'loop_thread_cpu_seconds', 'process_cpu_seconds')):
            cpu_records.append(record)
    gc_spans = [(r['start_s'], r['end_s']) for r in gc_records]
    loop_gc_spans = [(r['start_s'], r['end_s']) for r in gc_records if r['on_loop_thread']]
    for row in joined:
        summaries = phase_records.get(row['request_id'], [])
        observed = summaries[0] if len(summaries) == 1 else None
        phase_output = {}
        if observed is not None and isinstance(observed.get('phases'), dict):
            for name, phase in observed['phases'].items():
                if not isinstance(phase, dict):
                    continue
                duration = difference(phase.get('end'), phase.get('start'))
                phase_output[name] = dict(start_s=difference(phase.get('start'), start),
                    end_s=difference(phase.get('end'), start),
                    elapsed_s=duration if duration is not None and duration >= 0 else None,
                    status=phase.get('status'))
        row['diagnostics'] = dict(mode=mode, summaries=len(summaries),
            phases=phase_output or None, cleanup_complete=observed.get('cleanup_complete') if observed else None,
            first_body_s=difference(observed.get('first_body_at'), start) if observed else None,
            first_content_s=difference(observed.get('first_content_at'), start) if observed else None,
            post_dispatch_gc_overlap_s=interval_overlap(row['dispatch_s'], row['gateway_content_s'], gc_spans) if complete else None,
            post_dispatch_loop_gc_overlap_s=interval_overlap(row['dispatch_s'], row['gateway_content_s'], loop_gc_spans) if complete else None)
    monitors = []
    for row in records:
        if row.get('event') != 'event_loop_lag' or not all(finite(row.get(k)) for k in ('scheduled_at', 'monotonic')):
            continue
        a, b = row['scheduled_at'], row['monotonic']
        if b < a:
            continue
        monitors.append(dict(due_s=a-start, emission_s=b-start, overdue_s=b-a,
            gc_overlap_s=interval_overlap(a-start, b-start, gc_spans) if complete else None,
            loop_gc_overlap_s=interval_overlap(a-start, b-start, loop_gc_spans) if complete else None,
            observed_phase_overlap_s={name: interval_overlap(a, b, spans) for name, spans in phase_spans.items()}))
    cpu_intervals = []
    for previous, current in zip(cpu_records, cpu_records[1:]):
        a, b = previous['sampled_at'], current['sampled_at']
        loop_cpu = current['loop_thread_cpu_seconds'] - previous['loop_thread_cpu_seconds']
        process_cpu = current['process_cpu_seconds'] - previous['process_cpu_seconds']
        if b <= a or min(loop_cpu, process_cpu) < 0 or previous.get('loop_thread_id') != current.get('loop_thread_id'):
            continue
        cpu_intervals.append(dict(start_s=a-start, end_s=b-start, elapsed_s=b-a,
                                 loop_thread_cpu_seconds=loop_cpu, process_cpu_seconds=process_cpu))
    external_qualified = bool(integrity and integrity.get('valid') and integrity.get('external_mode') == 'enabled')
    engine_points = [dict(sent_s=row['sent_at']-start, received_s=row['received_at']-start,
                         finished_s=row['finished_at']-start, counter_epoch=row['counter_epoch'],
                         counter_resets=row['counter_resets'], disappeared_counters=row['disappeared_counters'],
                         metrics=row['metrics'], resources=row['resources']) for row in observations] if external_qualified else None
    return dict(gateway_mode=mode, integrity=integrity, gc_coverage_complete=complete,
                gc_spans=gc_records if requested is True else None,
                cpu_intervals=cpu_intervals if requested is True else None,
                monitor_overdue_intervals=monitors if requested is True else None,
                engine_observation_count=len(observations), engine_observations=engine_points,
                request_linked_prefill='unavailable',
                note='GC overlap uses actual spans, not emission times. Overlap categories and CPU counters are not additive '
                     'causal attribution. Async phase time is not blocked CPU time. Engine samples are aggregate observations, '
                     'not per-request queue/prefill attribution. Missing or unqualified GC coverage is null, never zero.')


def join_requests(clients, trace, histories, start, slo, run_id, e2e_slo=None):
    """Keep observed outcomes, including invalid/incomplete ones, separate from qualification."""
    if e2e_slo is not None:
        finite_number(e2e_slo, 'E2E SLO', True)
    planned = defaultdict(list)
    for row in trace['requests']:
        planned[row['trace_id']].append(row)
    rows = list(clients)
    recorded = Counter(r.get('trace_id') for r in rows)
    for identifier, plans in planned.items():
        for plan in plans[recorded[identifier]:]:
            rows.append(dict(trace_id=identifier, outcome_missing=True,
                             scheduled_at=start + plan['arrival_seconds'],
                             client_request_id=uuid.uuid5(uuid.UUID(run_id), identifier).hex))
    joined = []
    matched = Counter()
    for client in rows:
        identifier = client.get('client_request_id')
        trace_id = client.get('trace_id')
        plans = planned.get(trace_id, [])
        plan = plans[matched[trace_id]] if matched[trace_id] < len(plans) else {}
        matched[trace_id] += 1
        history = histories.get(identifier, {})
        issues = []
        def event_time(name):
            events = history.get(name, [])
            if len(events) > 1:
                issues.append(f'duplicate {name}')
            return events[0].get('monotonic') if len(events) == 1 else None
        times = [client.get('started_at')] + [event_time(k) for k in
                 ('received', 'accepted', 'backend_sent', 'first_content')] + [client.get('first_content_at')]
        stages = {}
        for name, left, right in zip(STAGES, times, times[1:]):
            value = difference(right, left)
            if value is not None and value < -1e-6:
                issues.append(f'negative {name} duration')
                value = None
            stages[name] = value
        complete_stages = all(value is not None for value in stages.values())
        ttft = client.get('ttft')
        if complete_stages and (not finite(ttft) or abs(sum(stages.values()) - ttft) > 1e-6):
            issues.append('stage sum differs from client TTFT')
        scheduled = client.get('scheduled_at')
        scheduled_ttft = difference(client.get('first_content_at'), scheduled)
        scheduled_e2e = difference(client.get('finished_at'), scheduled)
        if scheduled_ttft is not None and scheduled_ttft < 0:
            issues.append('negative scheduled TTFT')
            scheduled_ttft = None
        if scheduled_e2e is not None and scheduled_e2e < 0:
            issues.append('negative scheduled E2E')
            scheduled_e2e = None
        success, rejected = client.get('success'), client.get('rejected')
        if client.get('outcome_missing'):
            outcome = 'missing_client_outcome'
        elif success and (rejected or client.get('stream_complete') is not True):
            outcome = 'inconsistent_success'
        elif success:
            outcome = ('successful_ttft_unknown' if scheduled_ttft is None else
                       'successful_within_ttft' if scheduled_ttft <= slo else 'successful_over_ttft')
        elif client.get('error_type') == 'NotLaunched' and client.get('started_at') is None:
            outcome = 'never_launched'
        elif rejected:
            outcome = 'rejected'
        elif client.get('error_type') == 'CancelledError':
            outcome = 'cancelled'
        else:
            outcome = 'failed'
        terminal = [e for name in TERMINAL for e in history.get(name, [])]
        deadline = scheduled + slo if finite(scheduled) else None
        if e2e_slo is not None:
            e2e_met = joint_met = None
            if outcome.startswith('successful_') and success is True and rejected is False and not client.get('error_type'):
                first, finished = client.get('first_content_at'), client.get('finished_at')
                ttft_valid = scheduled_ttft is not None
                e2e_valid = scheduled_e2e is not None
                if finite(first) and finite(times[0]) and first < times[0]:
                    issues.append('first content precedes client start')
                    ttft_valid = False
                if finite(finished) and any(finite(t) and t > finished for t in (times[0], first)):
                    issues.append('completion precedes client timing')
                    e2e_valid = False
                e2e_met = scheduled_e2e <= e2e_slo if e2e_valid else None
                ttft_met = scheduled_ttft <= slo if ttft_valid else None
                if e2e_met is False or ttft_met is False:
                    joint_met = False
                elif e2e_met is True and ttft_met is True:
                    joint_met = True
            elif success is False and outcome != 'missing_client_outcome':
                e2e_met = joint_met = False
        row = dict(request_id=identifier, trace_id=client.get('trace_id'), outcome=outcome,
                   planned=bool(plan), client_outcome_present=not client.get('outcome_missing', False),
                   schedule_source='planned' if client.get('outcome_missing') else 'client',
                   planned_scheduled_s=plan.get('arrival_seconds'),
                   scheduled_s=difference(scheduled, start), started_s=difference(times[0], start),
                   received_s=difference(times[1], start), admitted_s=difference(times[2], start),
                   dispatch_s=difference(times[3], start), gateway_content_s=difference(times[4], start),
                   finished_s=difference(client.get('finished_at'), start),
                   scheduling_lag_s=difference(times[0], scheduled),
                   input_tokens=client.get('expected_input_tokens', plan.get('input_tokens')),
                   actual_input_tokens=client.get('input_tokens'), output_tokens=client.get('output_tokens'),
                   success=success is True, rejected=rejected is True, ttft_s=ttft if finite(ttft) else None,
                   end_to_end=client.get('end_to_end') if finite(client.get('end_to_end')) else None, scheduled_ttft=scheduled_ttft,
                   scheduled_e2e=scheduled_e2e, error_type=client.get('error_type'),
                   error_phase=client.get('error_phase'), stream_complete=client.get('stream_complete'),
                   stages=stages, stages_complete=complete_stages and not issues,
                   missing_stages=[k for k, v in stages.items() if v is None],
                   measurement_issues=issues, gateway_events={k: len(v) for k, v in history.items()},
                   gateway_outcome=terminal[0]['event'] if len(terminal) == 1 else None,
                   gateway_terminal_s=difference(terminal[0].get('monotonic'), start) if len(terminal) == 1 else None,
                   deadline_remaining_at_admission_s=difference(deadline, times[2]))
        if e2e_slo is not None:
            e2e_deadline = scheduled + e2e_slo if finite(scheduled) else None
            row.update(e2e_slo_met=e2e_met, joint_slo_met=joint_met,
                       e2e_deadline_remaining_at_admission_s=difference(e2e_deadline, times[2]))
        joined.append(row)
    accounting = dict(planned=len(trace['requests']), unique_planned_ids=len(planned),
                        recorded_client_rows=len(clients), ledger_rows=len(joined),
                        missing_client_outcomes=sum(r['outcome'] == 'missing_client_outcome' for r in joined),
                        extra_client_rows=sum(not r['planned'] for r in joined),
                        duplicate_planned_ids={k: len(v) for k, v in planned.items() if len(v) > 1},
                        duplicate_trace_ids={k: v for k, v in recorded.items() if v > 1},
                        outcomes=dict(Counter(r['outcome'] for r in joined)))
    if e2e_slo is not None:
        accounting['joint_slo'] = dict(Counter('met' if r['joint_slo_met'] is True else
            'missed' if r['joint_slo_met'] is False else 'unknown' for r in joined))
    return joined, accounting


def request_windows(joined, lifecycle, start, end):
    """Half-open one-second windows, including drain; counts are observed, not imputed."""
    identifiers = {r['request_id'] for r in joined if r['request_id'] is not None}
    events = [e for e in lifecycle if e.get('request_id') in identifiers]
    latest = max([end - start] + [r[k] for r in joined for k in ('finished_s', 'started_s', 'scheduled_s') if finite(r[k])] +
                 [e['monotonic'] - start for e in events if finite(e.get('monotonic'))])
    count = max(1, math.floor(latest) + 1)
    windows = [dict(start_s=i, end_s=i + 1, arrival_window=i < end-start,
                    scheduled=0, started=0, received=0, accepted=0, completed=0, rejected=0,
                    failed=0, cancelled=0, admitted_input_tokens_known=0,
                    admitted_input_tokens_unknown_requests=0) for i in range(count)]
    joint_slo = any('joint_slo_met' in row for row in joined)
    if joint_slo:
        for window in windows:
            window.update(cohort_slo_successes=0, cohort_slo_misses=0, cohort_slo_unknown=0)
    unbinned = Counter()
    def bucket(value, field):
        if not finite(value) or value < 0:
            unbinned[field] += 1
            return None
        index = math.floor(value)
        if index >= len(windows):
            unbinned[field] += 1
            return None
        return windows[index]
    for row in joined:
        for field, key in (('scheduled', 'scheduled_s'), ('started', 'started_s')):
            window = bucket(row[key], field)
            if window is not None:
                window[field] += 1
        if joint_slo:
            met = row.get('joint_slo_met')
            field = 'cohort_slo_successes' if met is True else 'cohort_slo_misses' if met is False else 'cohort_slo_unknown'
            window = bucket(row['scheduled_s'], field)
            if window is not None:
                window[field] += 1
    for event in events:
        field = event['event']
        if field not in {'received', 'accepted', *TERMINAL}:
            continue
        window = bucket(difference(event.get('monotonic'), start), field)
        if window is None:
            continue
        window[field] += 1
        if field == 'accepted':
            tokens = event.get('estimated_input_tokens')
            if type(tokens) is int and tokens >= 0:
                window['admitted_input_tokens_known'] += tokens
            else:
                window['admitted_input_tokens_unknown_requests'] += 1
    return dict(width_s=1, interval='[start_s, end_s)', windows=windows, unbinned=dict(unbinned),
                note='Observed gateway admissions/completions; missing lifecycle events are not zero work. '
                     'Input tokens are gateway admission estimates. Drain windows are retained.' +
                     (' Joint SLO outcomes are all-offered arrival cohorts, not completion throughput.' if joint_slo else ''))


def analyze_run(root, stall_threshold):
    before = hashes(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    trace = json.loads((root / 'trace.json').read_text())
    original = json.loads((root / 'report.json').read_text()) if (root / 'report.json').is_file() else None
    # Copies also work on hosts that cannot create symlinks. Never audit in place.
    with tempfile.TemporaryDirectory() as scratch:
        temporary = Path(scratch)
        for name in RAW:
            if name not in {'report.json', 'timeseries.png'} and (root / name).is_file():
                shutil.copyfile(root / name, temporary / name)
        checked = audit(temporary, plot=False)
    clients = json.loads((root / 'client.json').read_text())['requests'] if (root / 'client.json').is_file() else []
    records, malformed_lines = read_jsonl(root / 'gateway.jsonl')
    engine, malformed_engine_lines = read_jsonl(root / 'engine_observations.jsonl')
    start, end = manifest['measurement_start'], manifest['measurement_end']
    if not finite(start) or not finite(end) or end <= start:
        raise ValueError('Investigation requires finite, ordered measurement boundaries')
    histories = defaultdict(lambda: defaultdict(list))
    lifecycle = []
    for event in records:
        if event.get('event') in LIFECYCLE and event.get('request_id') is not None:
            histories[event['request_id']][event['event']].append(event)
            lifecycle.append(event)
    # Cumulative counts include the three warmup requests; increments do not.
    timed_lifecycle = sorted((e for e in lifecycle if finite(e.get('monotonic'))), key=lambda e: e['monotonic'])
    accepted_times = [e['monotonic'] for e in timed_lifecycle if e['event'] == 'accepted']
    received_times = [e['monotonic'] for e in timed_lifecycle if e['event'] == 'received']
    pairs = list(zip(timed_lifecycle, timed_lifecycle[1:]))
    joined, accounting = join_requests(clients, trace, histories, start,
                                       manifest['config']['ttft_slo'], manifest['run_id'],
                                       manifest['config'].get('e2e_slo'))
    successful = [r for r in joined if r['outcome'].startswith('successful_') and finite(r['ttft_s'])]
    tail = sorted(successful, key=lambda r: r['ttft_s'], reverse=True)[:math.ceil(len(successful)*.01)]
    stalls = []
    for monitor in records:
        if (monitor.get('event') != 'event_loop_lag' or not finite(monitor.get('lag_seconds')) or
                monitor['lag_seconds'] <= stall_threshold or not finite(monitor.get('scheduled_at')) or
                not finite(monitor.get('monotonic'))):
            continue
        if not start <= monitor['scheduled_at'] < end:
            continue
        due, resumed = monitor['scheduled_at'], monitor['monotonic']
        # Search the nearby lifecycle gap, which can start before the monitor
        # became due. This gap is not proof of continuous event-loop blocking.
        nearby = [(a, b) for a, b in pairs if a['monotonic'] < resumed and b['monotonic'] > due-.2]
        if not nearby:
            stalls.append(dict(due_s=due-start, resumed_s=resumed-start, lag_s=monitor['lag_seconds'],
                               gap_start_s=None, gap_end_s=None, gap_s=None,
                               accepted_count_at_gap_start=None, received_count_at_gap_start=None))
            continue
        a, b = max(nearby, key=lambda pair: pair[1]['monotonic']-pair[0]['monotonic'])
        feature = next((histories[a['request_id']][k][0] for k in ('accepted', 'rejected')
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
        return interval_overlap(row['dispatch_s'], row['gateway_content_s'],
                                [(s['due_s'], s['resumed_s']) for s in stalls])
    stage_stats = {k: distribution([r['stages'][k] for r in successful if r['stages'][k] is not None]) for k in STAGES}
    complete_tail = [r for r in tail if r['stages_complete']]
    tail_means = {k: distribution([r['stages'][k] for r in complete_tail])['mean'] for k in STAGES}
    tail_ttft = distribution([r['ttft_s'] for r in tail])['mean']
    decomposed_ttft = distribution([r['ttft_s'] for r in complete_tail])['mean']
    relative = lambda key: [r[key] for r in joined if finite(r[key])]
    bins = np.arange(0, math.ceil(end-start)+1)
    arrivals = {key: np.histogram(relative(key), bins=bins)[0].tolist()
                for key in ('scheduled_s', 'started_s', 'received_s')}
    expected_counts = {k: sum(0 <= value <= bins[-1] for value in relative(k)) for k in arrivals}
    if any(sum(arrivals[k]) != expected_counts[k] for k in arrivals):
        raise ValueError('Arrival bins do not reconcile')
    required_raw = LEGACY_RAW + (SAMPLER_RAW if manifest.get('sampler', {}).get('enabled') else ())
    if has_control_plan(root, manifest):
        required_raw += (PLAN_FILE,)
    if manifest.get('config', {}).get('runner_diagnostics'):
        required_raw += RUNNER_RAW
    if manifest.get('config', {}).get('runner_gc_policy', 'automatic') != 'automatic':
        required_raw += (GC_RAW,)
    summary = dict(name=root.name, raw_sha256=before, missing_raw_files=[n for n in required_raw if n not in before],
                   malformed_gateway_lines=malformed_lines,
                   malformed_engine_lines=malformed_engine_lines,
                   original_report_reproduced=checked == original, saved_report=original,
                   qualified_for_comparison=bool(checked['valid'] and checked['load_fidelity']['valid'] and checked == original and
                                                 checked.get('runner_diagnostic_integrity', {'valid': True})['valid'] and
                                                 checked.get('runner_gc_integrity', {'valid': True})['valid'] and
                                                 checked.get('control_fidelity', {'valid': True})['valid'] and
                                                 checked.get('diagnostic_integrity', {'valid': True})['valid']),
                   manifest_git_commit=manifest.get('git_commit'), recorded_git_dirty=manifest.get('git_dirty'),
                   runtime_code_matches={name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==sha
                                         if (Path(__file__).parent/name).is_file() else None
                                         for name, sha in manifest.get('code_sha256', {}).items()},
                   trace_sha256=manifest.get('trace_sha256'), gpu=manifest.get('gpu'), packages=manifest.get('packages'),
                   config=manifest['config'], duration_s=end-start, report=checked,
                   accounting=accounting,
                   outcomes_by_input_size={label: dict(Counter(r['outcome'] for r in joined
                       if ('unknown' if not finite(r['input_tokens']) else 'short_input' if r['input_tokens'] <= 128
                           else 'medium_input' if r['input_tokens'] <= 512 else 'long_input') == label))
                       for label in ('short_input', 'medium_input', 'long_input', 'unknown')},
                   e2e_seconds={outcome: {key: distribution([r[key] for r in joined if r['outcome'] == outcome and finite(r[key])])
                                         for key in ('end_to_end', 'scheduled_e2e')}
                                for outcome in accounting['outcomes']},
                   request_windows=request_windows(joined, lifecycle, start, end),
                   scheduling_delayed_count=sum(r['scheduling_lag_s'] > manifest['config']['max_scheduling_lag']
                                                for r in joined if finite(r['scheduling_lag_s'])),
                   successful_output=distribution([r['output_tokens'] for r in successful if finite(r['output_tokens'])]),
                   output_at_cap_count=sum(r.get('output_tokens') == r.get('max_output_tokens') for r in clients
                                           if r.get('success') and r.get('output_tokens') is not None),
                   stage_seconds=stage_stats,
                   slowest_one_percent=dict(count=len(tail), request_ids=[r['request_id'] for r in tail],
                       mean_ttft_s=tail_ttft, mean_stage_seconds=tail_means,
                       decomposed_count=len(complete_tail), decomposed_mean_ttft_s=decomposed_ttft,
                       post_dispatch_fraction=tail_means['post_dispatch']/decomposed_ttft if decomposed_ttft else None,
                       post_dispatch_monitor_overlap_count=sum(overlap(r)>0 for r in tail if overlap(r) is not None),
                       post_dispatch_monitor_overlap_unknown_count=sum(overlap(r) is None for r in tail)),
                   large_monitor_delays=stalls,
                   delay_due_spacing_s=np.diff([s['due_s'] for s in stalls]).tolist(),
                   delay_accepted_spacing=[difference(b['accepted_count_at_gap_start'], a['accepted_count_at_gap_start'])
                                           for a, b in zip(stalls, stalls[1:])],
                   arrival_counts_per_second=arrivals,
                   client_late_overlapping_large_monitor_count=sum(any(s['due_s'] < r['started_s'] and
                       s['resumed_s'] > r['scheduled_s'] for s in stalls) for r in joined
                       if finite(r['scheduling_lag_s']) and r['scheduling_lag_s'] > manifest['config']['max_scheduling_lag']))
    summary['diagnostics'] = join_diagnostics(joined, records, engine, start, manifest, checked)
    summary['cpu_analysis'] = analyze_cpu(joined, records, engine, start, manifest, checked)
    if 'runner_diagnostic_integrity' in checked:
        verdict = checked['runner_diagnostic_integrity']
        runner_records, malformed = read_jsonl(root / RUNNER_RAW[0])
        late = None
        if verdict['valid'] and not malformed:
            gc = [(r['span_start']-start, r['span_end']-start) for r in runner_records if r['event']=='gc_span']
            overdue = [(r['scheduled_at']-start, r['resumed_at']-start)
                       for r in runner_records if r['event']=='event_loop_lag']
            late = [dict(request_id=r['request_id'], trace_id=r['trace_id'], outcome=r['outcome'],
                         scheduled_s=r['scheduled_s'], started_s=r['started_s'],
                         scheduling_lag_s=r['scheduling_lag_s'],
                         gc_overlap_s=interval_overlap(r['scheduled_s'], r['started_s'], gc),
                         loop_overdue_overlap_s=interval_overlap(r['scheduled_s'], r['started_s'], overdue))
                    for r in joined if finite(r['scheduling_lag_s']) and
                    r['scheduling_lag_s'] > manifest['config']['max_scheduling_lag']]
        summary['runner_diagnostics'] = dict(integrity=verdict, malformed_lines=malformed,
            late_launches=late, note='Overlap is timing association, not exclusive CPU or OS scheduling attribution.')
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
        fidelity = summary['report']['load_fidelity']['valid']
        fidelity_label = 'load fidelity'
        if 'control_fidelity' in summary['report']:
            fidelity = fidelity and summary['report']['control_fidelity']['valid']
            fidelity_label = 'load/control fidelity'
        labels.append(summary['name'].replace('trace-experiment-', '').replace('trace-', '')+'\n' +
                      fidelity_label + (' PASS' if fidelity else ' FAILED'))
        groups.append(summary['slowest_one_percent'])
    left = np.zeros(len(groups))
    for key, label, color in zip(STAGES, LABELS, COLORS):
        values = np.array([g['mean_stage_seconds'][key] or 0 for g in groups])
        ax.barh(np.arange(len(groups)), values, left=left, label=label, color=color)
        left += values
    for i, group in enumerate(groups):
        fraction = group['post_dispatch_fraction']
        text = ('No complete decomposition' if fraction is None else
                f'{left[i]:.3f}s; post-dispatch {100*fraction:.1f}%')
        ax.text(left[i]+.04, i, text+f'\ndecomposed n={group["decomposed_count"]}; tail n={group["count"]}', va='center', fontsize=9)
    ax.set_yticks(np.arange(len(groups)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, max(left)*1.45))
    ax.set_xlabel('Mean client TTFT (seconds), complete decompositions within slowest ceil(1%) successes')
    ax.set_title('Trace replay — where tail latency occurs\nPost-dispatch includes transport, engine queue/compute, and gateway scheduling')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3, frameon=False)
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'latency_stages.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    repeats = [d for d in data if d[0]['duration_s'] > 30] or data
    fig, axes = plt.subplots(4, len(repeats), figsize=(14, 11), sharex='col', squeeze=False)
    for column, (summary, joined, records, start) in enumerate(repeats):
        successful = [r for r in joined if r['outcome'].startswith('successful_') and
                      finite(r['started_s']) and finite(r['stages']['post_dispatch'])]
        arrivals = summary['arrival_counts_per_second']
        ax = axes[0, column]
        for key, label, color in [('scheduled_s','Planned arrivals','#888888'), ('received_s','Gateway received','#286e99')]:
            ax.step(np.arange(len(arrivals[key])), arrivals[key], where='post', label=label, color=color, alpha=.8)
        ax.set_ylabel('Requests / 1s bin')
        fidelity = 'PASS' if summary['report']['load_fidelity']['valid'] else 'FAILED'
        integrity = 'PASS' if summary['report']['valid'] else 'FAILED'
        diagnostic = summary['report'].get('diagnostic_integrity')
        diagnostic_label = (' / diagnostics '+('PASS' if diagnostic['valid'] else 'FAILED')) if diagnostic else ''
        control = summary['report'].get('control_fidelity')
        control_label = (' / control '+('PASS' if control['valid'] else 'FAILED')) if control else ''
        ax.set_title(summary['name']+'\nIntegrity '+integrity+' / load fidelity '+fidelity+control_label+diagnostic_label)
        ax = axes[1, column]
        timed = [r for r in joined if finite(r['scheduling_lag_s'])]
        ax.scatter([r['scheduled_s'] for r in timed], [1000*r['scheduling_lag_s'] for r in timed], s=5, label='Client scheduling lag', color='#286e99')
        loop = [e for e in records if e.get('event')=='event_loop_lag' and finite(e.get('scheduled_at')) and
                finite(e.get('lag_seconds')) and start <= e['scheduled_at'] < start+summary['duration_s']]
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
        snapshots = [e for e in records if e.get('event')=='snapshot' and finite(e.get('monotonic')) and
                     finite(e.get('current_admitted_cost')) and start <= e['monotonic'] < start+summary['duration_s']]
        ax.plot([e['monotonic']-start for e in snapshots], [e['current_admitted_cost'] for e in snapshots], label='Reserved cost (gateway proxy)', color='#8662a2')
        if control is not None:
            unavailable = 'no pre-measurement application'
            try:
                times, budgets = budget_curve(records, start, start + summary['duration_s'])
            except ValueError as exc:
                times, budgets = [], []
                unavailable = str(exc)
            if times:
                ax.step(times, budgets, where='post', color='black', ls='--', label='Applied budget')
            else:
                ax.text(.02, .9, 'Applied budget unavailable: ' + unavailable, transform=ax.transAxes, fontsize=8)
        else:
            budgets = [summary['config']['budget']]
            ax.axhline(budgets[0], color='black', ls='--', label='Fixed budget')
        ax.set_ylabel('Cost units')
        ax.set_ylim(0, max([1.] + budgets + [e['current_admitted_cost'] for e in snapshots])*1.1)
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
    cpu_figure(output, data)


def cpu_figure(output, data):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(data), figsize=(7*len(data), 10), sharex='col', squeeze=False)
    for column, (summary, joined, _, _) in enumerate(data):
        cpu = summary['cpu_analysis']
        runner_label = ('Runner process (includes sampler)' if cpu['sampler_mode'] == 'thread' else
                        'Runner process (sampler separate)' if cpu['sampler_mode'] == 'process' else 'Runner process')
        for name, label, color in (
                ('runner', runner_label, '#286e99'),
                ('gateway', 'Gateway process (external)', '#ba5537'),
                ('gateway_loop', 'Gateway loop thread', '#366a56')):
            source = cpu['sources'][name]
            intervals = [row for row in source['intervals'] or [] if not row['issues']]
            if intervals:
                axes[0, column].hlines([r['cpu_cores'] for r in intervals],
                    [r['start_s'] for r in intervals], [r['end_s'] for r in intervals],
                    color=color, linewidth=1.5, label=label)
            else:
                axes[0, column].plot([], [], color=color, label=f'{label}: {source["status"]}')
        source = cpu['sources']['host']
        intervals = [row for row in source['intervals'] or [] if not row['issues']]
        if intervals:
            axes[1, column].hlines([100*r['host_nonidle_fraction'] for r in intervals],
                [r['start_s'] for r in intervals], [r['end_s'] for r in intervals],
                color='#8662a2', label='Host counter fraction (not container capacity)')
        else:
            axes[1, column].text(.5, .5, f'Host CPU: {source["status"]}',
                                transform=axes[1, column].transAxes, ha='center')
        timed = [row for row in joined if finite(row['scheduling_lag_s']) and finite(row['scheduled_s'])]
        axes[2, column].scatter([r['scheduled_s'] for r in timed],
            [1000*r['scheduling_lag_s'] for r in timed], s=8, color='#286e99', label='Client scheduling lag')
        if cpu['selection_issue'] is None:
            axes[2, column].axhline(1000*cpu['scheduling_lag_limit_s'], color='black',
                                   linestyle='--', label='Original fidelity limit')
        for row in cpu['late_launches'] or []:
            for ax in axes[:, column]:
                ax.axvspan(row['scheduled_s'], row['started_s'], color='#ba5537', alpha=.15)
        axes[0, column].set_title(summary['name']+'\nLoad fidelity '+
                                 ('PASS' if summary['report']['load_fidelity']['valid'] else 'FAILED'))
        axes[0, column].set_ylabel('Average CPU cores')
        axes[1, column].set_ylabel('Host non-idle (%)')
        axes[2, column].set_ylabel('Launch delay (ms)')
        axes[2, column].set_xlabel('Seconds since measurement start')
        for ax in axes[:, column]:
            ax.set_xlim(0, summary['duration_s'])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('CPU interval averages and late launches\n'
                 'Shading = late-launch intervals; CPU averages are not attributed to these intervals',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output/'cpu_scheduling.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stall-threshold', type=float, default=.3,
                        help='Exploratory monitor-delay selection only; never changes fidelity limits')
    parser.add_argument('--all-requests', action='store_true',
                        help='Compatibility option: all offered requests are now always exported')
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
    dump(output/'summary.json', dict(schema_version=2, analyst_stall_threshold_s=args.stall_threshold,
         analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         runs=[d[0] for d in data],
         repeat_trace_identical=(len({d[0]['trace_sha256'] for d in data if d[0]['duration_s'] > 30})==1
                                if sum(d[0]['duration_s'] > 30 for d in data) >= 2 else None)))
    for summary, joined, _, _ in data:
        with (output/(summary['name']+'-requests.jsonl')).open('w') as stream:
            for row in joined:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    figures(output, data, args.stall_threshold)
    print(json.dumps({s['name']: dict(integrity=s['report']['valid'], fidelity=s['report']['load_fidelity']['valid'],
                       runner_diagnostics=s['report'].get('runner_diagnostic_integrity', {}).get('valid'),
                       runner_gc=s['report'].get('runner_gc_integrity', {}).get('valid'),
                       control_fidelity=s['report'].get('control_fidelity', {}).get('valid'),
                       diagnostics=s['report'].get('diagnostic_integrity', {}).get('valid'),
                       tail_post_dispatch_fraction=s['slowest_one_percent']['post_dispatch_fraction'],
                       large_monitor_delays=len(s['large_monitor_delays'])) for s, _, _, _ in data}, indent=2))


if __name__ == '__main__':
    main()
