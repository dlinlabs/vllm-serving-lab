"""Causal gateway-side identification features, stored separately from forecast labels.

Offline forecasting input only: not a model, controller or online feature channel.
Features at decision time t use only gateway records emitted at or before t (budget
applications use their recorded application time) and engine samples fully received
by t. Labels use later events and eventual all-offered outcomes. Control columns are the
realized cap path over (t, t+h] from recorded application times, so they include actuation
timing that is unknown at t: models using them are conditioned on realized control
(system identification), not deployable planned-control forecasts.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from analyze_trace_dynamics import analyze_run, dump, finite, hashes, join_requests
from analyze_trace_pilot import LIFECYCLE, TERMINAL
from experiment_protocol import capture_identity
from trace_control import read_plan

SCHEMA_VERSION = 1
PROTOCOL = 'causal-gateway-features-v1'
CONTROL_CONDITIONING = 'realized_control_conditioned'
CONTROL_LIMITS = ('Control columns use recorded budget application times over (t, t+h], including actuation '
                  'timing unknown at t. They support system identification conditioned on the applied control, '
                  'not a deployable forecast; that needs control inputs built from decision-time information '
                  'plus an explicit actuation-delay assumption, with applications kept as evidence.')
FLOWS = ('received', 'accepted', 'rejected', 'first_content', 'completed', 'failed')
RESOLVED = ('resolved_ttft_met', 'resolved_admitted_ttft_missed')
STATE_LABELS = ('awaiting_first_content', 'oldest_awaiting_age_s', 'reserved_cost', 'streaming', 'in_flight')
UNKNOWN_OUTCOMES = {'successful_ttft_unknown', 'inconsistent_success', 'missing_client_outcome'}
# (snapshot field, reconstructed field, absolute tolerance)
SNAPSHOT_FIELDS = (('current_admitted_cost', 'reserved_cost', 1e-6),
                   ('current_in_flight', 'in_flight', 0),
                   ('awaiting_first_content', 'awaiting_first_content', 0),
                   ('oldest_awaiting_first_seconds', 'oldest_awaiting_age_s', 1e-3),
                   ('streaming_requests', 'streaming', 0),
                   ('active_requests', 'active_requests', 0),
                   ('admission_budget', 'budget', 0),
                   ('budget_version', 'budget_version', 0))
COLUMNS = {
    'features.budget': 'Applied admitted-work cost cap at t (cost units), from budget applications.',
    'features.budget_version': 'Gateway budget version at t.',
    'features.since_budget_application_s': 'Seconds since the latest budget application, including no-op values.',
    'features.since_budget_value_change_s': 'Seconds since an application changed the cap value; null if never.',
    'features.reserved_cost': 'Reserved admitted cost at t; can exceed a reduced cap while work drains.',
    'features.budget_headroom': 'Applied cap minus reserved cost; negative after a reduction below reserved work.',
    'features.in_flight': 'Admitted requests without a terminal gateway event.',
    'features.awaiting_first_content': 'Admitted requests without gateway first content.',
    'features.oldest_awaiting_age_s': 'Oldest awaiting request age since gateway receipt; 0 if none.',
    'features.streaming': 'Requests with first content and no terminal event.',
    'features.active_requests': 'Received requests without a terminal event, including undecided ones.',
    'features.<flow>_<w>s': 'Gateway events in (t-w, t]; failed includes cancelled.',
    'features.accepted_input_tokens_<w>s': 'Gateway input-token estimates of admissions in (t-w, t].',
    'features.offered_cost_<w>s': 'Estimated cost of admitted plus rejected requests decided in (t-w, t].',
    'features.resolved_ttft_met_<w>s': 'Requests whose gateway-receipt TTFT deadline fell in (t-w, t] with '
                                       'first content by that deadline. Receipt-anchored proxy.',
    'features.resolved_admitted_ttft_missed_<w>s': 'Admitted requests whose receipt-anchored TTFT deadline fell '
                                                   'in (t-w, t] without first content by then.',
    'features.resolved_joint_met_<w>s': 'With an E2E SLO: receipt-anchored first content and completion '
                                        'deadlines met, resolved at the E2E deadline in (t-w, t].',
    'features.engine_*': 'Latest engine sample fully received by t. Values are null and engine_missing is '
                         'true when none exists or its send time is older than the maximum age.',
    'control.<h>.budget_mean': 'Time-weighted realized cap over (t, t+h] from recorded application times, '
                               'including actuation timing unknown at t. Control input, not a feature.',
    'control.<h>.budget_end': 'Realized cap at t+h from recorded application times. Control input, not a feature.',
    'labels.<h>.<state>': 'Reconstructed gateway state at t+h.',
    'labels.<h>.<flow>': 'Gateway events in (t, t+h].',
    'labels.<h>.cohort_*': 'Eventual outcomes of requests scheduled in [t, t+h), measured from scheduled arrival. '
                           'Met/missed counts are null when any cohort outcome is unknown.',
    'labels.<h>.cohort_resolved_s': 'Latest time any cohort outcome was observed; the label is not known earlier.',
    'label_window_complete.<h>': 't+h is inside the arrival window; drain-only windows are retained but flagged.',
    'request_labels.*': 'Separate per-request file: outcome, scheduled-arrival TTFT/E2E and resolved_s, the '
                        'earliest time the outcome is known. Label data only.',
}


def require_positive(value, name):
    if not finite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def steps(value, step, name):
    count = round(value / step)
    if count < 1 or abs(count * step - value) > 1e-9:
        raise ValueError(f'{name} must be a positive integer multiple of the step')
    return count


def label_key(value):
    return f'{value:g}'


def budget_applications(records, start, initial_budget, initial_version):
    """Valid consecutive gateway applications as (relative time, budget, version)."""
    issues, rows = [], []
    for index, record in enumerate(records):
        if record.get('event') != 'budget_updated':
            continue
        moment, budget, version = record.get('applied_at'), record.get('budget'), record.get('budget_version')
        if not finite(moment) or not finite(budget) or budget <= 0 or type(version) is not int:
            issues.append(f'invalid budget application at record {index}')
            continue
        rows.append((moment - start, index, float(budget), version))
    rows.sort()
    applications, expected = [], initial_version + 1
    for moment, _, budget, version in rows:
        if version != expected:
            issues.append(f'nonconsecutive budget version {version} at {moment:.6f}s')
            continue
        applications.append((moment, budget, version))
        expected += 1
    return applications, issues


class GatewayReplay:
    """Apply gateway events in emission order; read state only at nondecreasing cutoffs."""

    def __init__(self, records, start, initial_budget, initial_version=0):
        applications, self.issues = budget_applications(records, start, initial_budget, initial_version)
        events = [(moment, -1, 'budget_updated', dict(budget=budget, budget_version=version))
                  for moment, budget, version in applications]
        for index, record in enumerate(records):
            kind = record.get('event')
            if kind not in LIFECYCLE or record.get('request_id') is None:
                continue
            if not finite(record.get('monotonic')):
                self.issues.append(f'nonfinite {kind} time at record {index}')
                continue
            events.append((record['monotonic'] - start, index, kind, record))
        events.sort(key=lambda event: event[:2])
        self.events, self.position, self.cutoff = events, 0, -math.inf
        self.budget, self.version = float(initial_budget), initial_version
        self.applied_at = self.value_changed_at = None
        self.active, self.finished = {}, set()
        self.reserved, self.in_flight = 0., 0

    def _apply(self, moment, kind, record):
        if kind == 'budget_updated':
            if record['budget'] != self.budget:
                self.value_changed_at = moment
            self.budget, self.version, self.applied_at = record['budget'], record['budget_version'], moment
            return
        identifier = record['request_id']
        entry = self.active.get(identifier)
        if kind == 'received':
            if entry is not None or identifier in self.finished:
                self.issues.append(f'duplicate received event for {identifier}')
            else:
                self.active[identifier] = dict(received=moment, admitted=False, cost=0., first=None)
        elif entry is None:
            self.issues.append(f'{kind} event without an active request {identifier}')
        elif kind == 'accepted':
            cost = record.get('request_cost')
            if entry['admitted'] or not finite(cost) or cost < 0:
                self.issues.append(f'invalid accepted event for {identifier}')
                return
            entry.update(admitted=True, cost=cost)
            self.reserved += cost
            self.in_flight += 1
        elif kind == 'first_content':
            if not entry['admitted'] or entry['first'] is not None:
                self.issues.append(f'invalid first_content event for {identifier}')
            else:
                entry['first'] = moment
        elif kind in TERMINAL:
            del self.active[identifier]
            self.finished.add(identifier)
            if entry['admitted']:
                if kind == 'rejected':
                    self.issues.append(f'admitted request {identifier} was later rejected')
                # Same arithmetic and clamping as the gateway's release.
                self.reserved = max(0., self.reserved - entry['cost'])
                self.in_flight = max(0, self.in_flight - 1)

    def state(self, cutoff):
        if cutoff < self.cutoff:
            raise ValueError('Replay cutoffs must not decrease')
        self.cutoff = cutoff
        while self.position < len(self.events) and self.events[self.position][0] <= cutoff:
            moment, _, kind, record = self.events[self.position]
            self.position += 1
            self._apply(moment, kind, record)
        awaiting = [entry['received'] for entry in self.active.values()
                    if entry['admitted'] and entry['first'] is None]
        return dict(budget=self.budget, budget_version=self.version,
                    since_budget_application_s=None if self.applied_at is None else cutoff - self.applied_at,
                    since_budget_value_change_s=None if self.value_changed_at is None else cutoff - self.value_changed_at,
                    reserved_cost=self.reserved, budget_headroom=self.budget - self.reserved,
                    in_flight=self.in_flight, awaiting_first_content=len(awaiting),
                    oldest_awaiting_age_s=max((cutoff - received for received in awaiting), default=0.),
                    streaming=sum(entry['first'] is not None for entry in self.active.values()),
                    active_requests=len(self.active))


def validate_snapshots(records, start, initial_budget, initial_version=0):
    """Compare the replay with every gateway snapshot taken under the gateway state lock.

    Every snapshot needs a finite sample time and a finite value for every compared field; missing,
    null, boolean or nonfinite evidence invalidates the reconstruction instead of being skipped.
    """
    candidates = [r for r in records if r.get('event') == 'snapshot']
    snapshots = sorted((r for r in candidates if finite(r.get('state_sampled_at'))),
                       key=lambda r: r['state_sampled_at'])
    invalid_timestamps = len(candidates) - len(snapshots)
    replay = GatewayReplay(records, start, initial_budget, initial_version)
    compared, missing, mismatches, examples = Counter(), Counter(), Counter(), []
    for snapshot in snapshots:
        moment = snapshot['state_sampled_at'] - start
        state = replay.state(moment)
        for source, target, tolerance in SNAPSHOT_FIELDS:
            value = snapshot.get(source)
            if not finite(value):
                missing[source] += 1
                continue
            compared[source] += 1
            if abs(value - state[target]) > tolerance:
                mismatches[source] += 1
                if len(examples) < 10:
                    examples.append(dict(time_s=moment, field=source, snapshot=value, reconstructed=state[target]))
    replay.state(math.inf)  # surface inconsistent events after the last snapshot too
    times = [snapshot['state_sampled_at'] - start for snapshot in snapshots]
    return dict(snapshots=len(snapshots), invalid_timestamps=invalid_timestamps,
                required_fields=[source for source, _, _ in SNAPSHOT_FIELDS], compared=dict(compared),
                missing=dict(missing), mismatches=dict(mismatches), examples=examples,
                span_s=[times[0], times[-1]] if times else None,
                max_gap_s=max((b - a for a, b in zip(times, times[1:])), default=None),
                replay_issues=replay.issues[:20], replay_issue_count=len(replay.issues),
                valid=bool(snapshots) and not invalid_timestamps and not missing and not mismatches
                and not replay.issues)


class Window:
    """Sorted event times (with optional values) for half-open (a, b] sums."""

    def __init__(self, pairs):
        pairs = sorted(pairs)
        self.times = [moment for moment, _ in pairs]
        self.prefix = [0.]
        for _, value in pairs:
            self.prefix.append(self.prefix[-1] + value)

    def total(self, left, right):
        return self.prefix[bisect_right(self.times, right)] - self.prefix[bisect_right(self.times, left)]


def event_windows(records, start, ttft_slo, e2e_slo=None):
    """Gateway flows and receipt-anchored deadline proxies; proxies resolve at their deadlines."""
    pairs = defaultdict(list)
    history = defaultdict(dict)
    for record in records:
        kind, identifier, moment = record.get('event'), record.get('request_id'), record.get('monotonic')
        if kind not in LIFECYCLE or identifier is None or not finite(moment):
            continue
        moment -= start
        history[identifier].setdefault(kind, moment)
        flow = 'failed' if kind == 'cancelled' else kind
        if flow in FLOWS:
            pairs[flow].append((moment, 1.))
        tokens, cost = record.get('estimated_input_tokens'), record.get('request_cost')
        if kind == 'accepted' and type(tokens) is int and tokens >= 0:
            pairs['accepted_input_tokens'].append((moment, float(tokens)))
        if kind in ('accepted', 'rejected') and finite(cost) and cost >= 0:
            pairs['offered_cost'].append((moment, float(cost)))
    for events in history.values():
        received = events.get('received')
        if received is None:
            continue
        deadline, first = received + ttft_slo, events.get('first_content')
        met = first is not None and first <= deadline
        if met:
            pairs['resolved_ttft_met'].append((deadline, 1.))
        elif events.get('accepted') is not None and events['accepted'] <= deadline:
            pairs['resolved_admitted_ttft_missed'].append((deadline, 1.))
        if e2e_slo is not None:
            completed, e2e_deadline = events.get('completed'), received + e2e_slo
            if met and completed is not None and completed <= e2e_deadline:
                pairs['resolved_joint_met'].append((max(deadline, e2e_deadline), 1.))
    names = FLOWS + ('accepted_input_tokens', 'offered_cost') + RESOLVED
    names += ('resolved_joint_met',) if e2e_slo is not None else ()
    return {name: Window(pairs[name]) for name in names}


def engine_series(points):
    """Successful engine scrapes, ordered by when their body was fully received."""
    series = []
    for point in points or []:
        values = {}
        for key in ('waiting', 'running', 'kv_cache'):
            metric = (point.get('metrics') or {}).get(key)
            samples = metric.get('samples') if isinstance(metric, dict) and metric.get('available') is True else None
            if samples and all(finite(s.get('value')) for s in samples):
                total = sum(s['value'] for s in samples)
                values[key] = total / len(samples) if key == 'kv_cache' else total
            else:
                values[key] = None
        sent, received = point.get('sent_s'), point.get('received_s')
        if (values['waiting'] is None or values['running'] is None or not finite(sent)
                or not finite(received) or received < sent):
            continue
        series.append((received, sent, values))
    series.sort(key=lambda row: row[0])
    return series


def engine_features(series, received_times, moment, max_age):
    index = bisect_right(received_times, moment) - 1
    if index < 0:
        return dict(engine_waiting=None, engine_running=None, engine_kv_cache_usage=None,
                    engine_age_s=None, engine_missing=True)
    _, sent, values = series[index]
    age = moment - sent
    stale = age > max_age
    return dict(engine_waiting=None if stale else values['waiting'],
                engine_running=None if stale else values['running'],
                engine_kv_cache_usage=None if stale else values['kv_cache'],
                engine_age_s=age, engine_missing=stale)


def budget_path(applications, initial_budget):
    times = [moment for moment, _, _ in applications]
    budgets = [float(initial_budget)] + [budget for _, budget, _ in applications]

    def at(moment):
        return budgets[bisect_right(times, moment)]

    def mean(left, right):
        total, cursor = 0., left
        for moment in times[bisect_right(times, left):bisect_right(times, right)]:
            total += at(cursor) * (moment - cursor)
            cursor = moment
        return (total + at(cursor) * (right - cursor)) / (right - left)
    return at, mean


def resolved_at(row):
    """Latest observed time for a request outcome; the outcome is not known before it."""
    return max(value for value in (row.get(key) for key in
               ('finished_s', 'gateway_terminal_s', 'started_s', 'scheduled_s')) if finite(value))


def request_labels(ledger):
    """Per-request all-offered outcomes, latency measured from scheduled arrival (label side only)."""
    keys = ('request_id', 'trace_id', 'outcome', 'scheduled_s', 'received_s', 'admitted_s',
            'gateway_content_s', 'finished_s', 'scheduled_ttft', 'scheduled_e2e', 'input_tokens',
            'output_tokens', 'e2e_slo_met', 'joint_slo_met')
    rows = [dict({key: row.get(key) for key in keys if key in row}, resolved_s=resolved_at(row))
            for row in ledger if finite(row.get('scheduled_s'))]
    return sorted(rows, key=lambda row: row['scheduled_s'])


def cohort_labels(ledger, scheduled, left, right, e2e_slo):
    """All offered requests scheduled in [left, right), including rejected and failed ones."""
    window = ledger[bisect_left(scheduled, left):bisect_left(scheduled, right)]
    unknown = sum(row['outcome'] in UNKNOWN_OUTCOMES for row in window)
    admitted = [row for row in window if finite(row.get('admitted_s'))]
    resolved = [resolved_at(row) for row in window]
    labels = dict(cohort_offered=len(window), cohort_admitted=len(admitted), cohort_unknown=unknown,
                  cohort_ttft_met=None if unknown else sum(row['outcome'] == 'successful_within_ttft' for row in window),
                  cohort_admitted_ttft_missed=None if unknown else
                  sum(row['outcome'] != 'successful_within_ttft' for row in admitted),
                  cohort_resolved_s=max([right] + resolved))
    if e2e_slo is not None:
        joint = [row.get('joint_slo_met') for row in window]
        labels['cohort_joint_met'] = None if any(value is None for value in joint) else sum(joint)
        labels['cohort_joint_unknown'] = sum(value is None for value in joint)
    return labels


def build_table(records, ledger, engine_points, *, start, duration, initial_budget, ttft_slo,
                e2e_slo=None, initial_version=0, step=.5, horizons=(1., 2., 5.), windows=(1., 5.),
                engine_max_age=3., coverage_end=None):
    """Rows at t = 0, step, ... < duration. Features never read events after t."""
    duration, step = require_positive(duration, 'duration'), require_positive(step, 'step')
    ttft_slo = require_positive(ttft_slo, 'TTFT SLO')
    if e2e_slo is not None:
        e2e_slo = require_positive(e2e_slo, 'E2E SLO')
    engine_max_age = require_positive(engine_max_age, 'engine maximum age')
    horizons = sorted({require_positive(h, 'horizon') for h in horizons})
    windows = sorted({require_positive(w, 'window') for w in windows})
    if not horizons or not windows:
        raise ValueError('At least one horizon and one window are required')
    horizon_steps = {h: steps(h, step, 'horizon') for h in horizons}
    for w in windows:
        steps(w, step, 'window')
    if coverage_end is None:
        coverage_end = max((r['monotonic'] for r in records if finite(r.get('monotonic'))), default=start) - start
    count = math.ceil(duration / step - 1e-9)
    replay = GatewayReplay(records, start, initial_budget, initial_version)
    states = [replay.state(round(k * step, 9)) for k in range(count + max(horizon_steps.values()))]
    flows = event_windows(records, start, ttft_slo, e2e_slo)
    series = engine_series(engine_points)
    received_times = [row[0] for row in series]
    applications, _ = budget_applications(records, start, initial_budget, initial_version)
    budget_at, budget_mean = budget_path(applications, initial_budget)
    ledger = sorted((row for row in ledger if finite(row.get('scheduled_s'))), key=lambda row: row['scheduled_s'])
    scheduled = [row['scheduled_s'] for row in ledger]
    rows = []
    for k in range(count):
        t = round(k * step, 9)
        features = dict(states[k])
        for w in windows:
            for name, window in flows.items():
                value = window.total(t - w, t)
                features[f'{name}_{w:g}s'] = value if name in ('accepted_input_tokens', 'offered_cost') else round(value)
        features.update(engine_features(series, received_times, t, engine_max_age))
        control, labels, complete = {}, {}, {}
        for h, n in horizon_steps.items():
            key, end = label_key(h), round(t + h, 9)
            control[key] = dict(budget_mean=budget_mean(t, end), budget_end=budget_at(end))
            complete[key] = end <= duration + 1e-9
            if end > coverage_end:
                labels[key] = None
                continue
            label = {name: states[k + n][name] for name in STATE_LABELS}
            label.update({name: round(flows[name].total(t, end)) for name in FLOWS})
            label.update(cohort_labels(ledger, scheduled, t, end, e2e_slo))
            labels[key] = label
        rows.append(dict(t_s=t, features=features, control=control, labels=labels, label_window_complete=complete))
    return rows, dict(replay_issues=replay.issues[:20], replay_issue_count=len(replay.issues),
                      engine_samples=len(series), coverage_end_s=coverage_end,
                      budget_applications=[dict(time_s=m, budget=b, budget_version=v) for m, b, v in applications],
                      unplaced_ledger_rows=0)


def rebuild_ledger(root, records, start, manifest, e2e_slo):
    """The audited ledger, recomputed only to add analyst-supplied joint E2E labels."""
    clients = json.loads((root / 'client.json').read_text())['requests']
    trace = json.loads((root / 'trace.json').read_text())
    histories = defaultdict(lambda: defaultdict(list))
    for event in records:
        if event.get('event') in LIFECYCLE and event.get('request_id') is not None:
            histories[event['request_id']][event['event']].append(event)
    ledger, _ = join_requests(clients, trace, histories, start, manifest['config']['ttft_slo'],
                              manifest['run_id'], e2e_slo)
    return ledger


def extract(root, *, stall_threshold=.3, e2e_slo=None, allow_unqualified=False, **options):
    summary, ledger, records, start = analyze_run(root, stall_threshold)
    if not summary['qualified_for_comparison'] and not allow_unqualified:
        raise ValueError(f'{root.name} is not qualified for comparison; features would describe a failed capture')
    manifest = json.loads((root / 'manifest.json').read_text())
    declared = manifest['config'].get('e2e_slo')
    source = 'declared' if declared is not None else None
    if e2e_slo is not None and e2e_slo != declared:
        if declared is not None:
            raise ValueError(f'{root.name} declares E2E SLO {declared}; refusing a different analyst value')
        ledger = rebuild_ledger(root, records, start, manifest, e2e_slo)
        source = 'analyst_offline'
        if hashes(root) != summary['raw_sha256']:
            raise ValueError('Raw files changed during analysis')
    effective = declared if e2e_slo is None else e2e_slo
    initial = manifest.get('initial_metrics') or {}
    budget = initial.get('admission_budget', manifest['config'].get('budget'))
    version = initial.get('budget_version', 0)
    duration = manifest['measurement_end'] - manifest['measurement_start']
    engine = summary['diagnostics']['engine_observations']
    rows, info = build_table(records, ledger, engine, start=start, duration=duration,
                             initial_budget=budget, initial_version=version,
                             ttft_slo=manifest['config']['ttft_slo'], e2e_slo=effective, **options)
    info['unplaced_ledger_rows'] = sum(not finite(row.get('scheduled_s')) for row in ledger)
    validation = validate_snapshots(records, start, budget, version)
    trace = json.loads((root / 'trace.json').read_text())
    try:
        plan = read_plan(root, manifest, trace)
    except ValueError:
        plan = None
    capture = capture_identity(manifest['config'], trace, summary['raw_sha256'].get('trace.json'), plan,
                               manifest.get('experiment_protocol'))
    if hashes(root) != summary['raw_sha256']:
        raise ValueError('Raw files changed during analysis')
    keys = rows[0]['labels'].keys() if rows else ()
    coverage = {key: dict(rows=len(rows), labeled=sum(r['labels'][key] is not None for r in rows),
                          complete_window=sum(r['label_window_complete'][key] for r in rows)) for key in keys}
    run = dict(name=root.name, qualified_for_comparison=summary['qualified_for_comparison'],
               original_report_reproduced=summary['original_report_reproduced'],
               raw_sha256=summary['raw_sha256'], manifest_git_commit=summary['manifest_git_commit'],
               recorded_git_dirty=summary['recorded_git_dirty'], trace_sha256=summary['trace_sha256'],
               duration_s=duration, ttft_slo=manifest['config']['ttft_slo'], e2e_slo=effective,
               e2e_slo_source=source, initial_budget=budget, accounting=summary['accounting'], capture=capture,
               engine=dict(status='available' if engine is not None else 'unavailable',
                           usable_samples=info['engine_samples']),
               state_reconstruction=validation, table=info, label_coverage=coverage)
    return run, rows, request_labels(ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--step', type=float, default=.5, help='Table grid, not a validated control period')
    parser.add_argument('--horizons', type=float, nargs='+', default=[1., 2., 5.])
    parser.add_argument('--windows', type=float, nargs='+', default=[1., 5.])
    parser.add_argument('--engine-max-age', type=float, default=3.)
    parser.add_argument('--e2e-slo', type=float,
                        help='Analyst E2E deadline for offline joint labels; recorded as analyst_offline, '
                             'never as a declared experiment SLO')
    parser.add_argument('--stall-threshold', type=float, default=.3,
                        help='Passed to the audit for exploratory monitor selection only')
    parser.add_argument('--allow-unqualified', action='store_true',
                        help='Export failed captures for debugging; they remain marked unqualified')
    args = parser.parse_args()
    roots, output = [p.resolve() for p in args.runs], args.output_dir.resolve()
    if len({p.name for p in roots}) != len(roots):
        parser.error('Run directory names must be unique')
    if any(output == p or p in output.parents for p in roots):
        parser.error('Output must be separate from raw run directories')
    if args.e2e_slo is not None and (not math.isfinite(args.e2e_slo) or args.e2e_slo <= 0):
        parser.error('--e2e-slo must be finite and positive')
    options = dict(step=args.step, horizons=args.horizons, windows=args.windows,
                   engine_max_age=args.engine_max_age)
    try:
        results = [extract(root, stall_threshold=args.stall_threshold, e2e_slo=args.e2e_slo,
                           allow_unqualified=args.allow_unqualified, **options) for root in roots]
    except ValueError as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    dump(output / 'features_summary.json', dict(
        schema_version=SCHEMA_VERSION, protocol=PROTOCOL, parameters=options,
        analysis_code_sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in
                              ('identification_features.py', 'analyze_trace_dynamics.py', 'analyze_trace_pilot.py')},
        runs=[run for run, _, _ in results], columns=COLUMNS,
        control_conditioning=CONTROL_CONDITIONING, control_limits=CONTROL_LIMITS,
        note='Features are causal for a controller co-located with the gateway; an external poller would see '
             'staler state. Grid step is a table resolution, not a validated control period. Labels use '
             'eventual outcomes and must never be used as features. Neighboring rows overlap and are not '
             'independent samples. <run>-request-labels.jsonl holds per-request outcomes with latency from '
             'scheduled arrival so other E2E deadlines can be applied offline; it is label data only.'))
    for run, rows, requests in results:
        for suffix, items in (('features', rows), ('request-labels', requests)):
            with (output / f'{run["name"]}-{suffix}.jsonl').open('w') as stream:
                for item in items:
                    stream.write(json.dumps(item, allow_nan=False) + '\n')
    print(json.dumps({run['name']: dict(rows=len(rows), requests=len(requests),
                                        qualified=run['qualified_for_comparison'],
                                        state_reconstruction=run['state_reconstruction']['valid'],
                                        engine_samples=run['engine']['usable_samples'],
                                        e2e_slo=run['e2e_slo'], e2e_slo_source=run['e2e_slo_source'])
                      for run, rows, requests in results}, indent=2))


if __name__ == '__main__':
    main()
