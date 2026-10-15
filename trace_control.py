"""Predeclared trace controls, independent of immutable request arrivals."""
import json

from trace_workload import finite_number, sha256_file

CONTROL_PROTOCOL = 'trace-identification-v1'
PLAN_FILE = 'replay_plan.json'
RESPONSE_FIELDS = ('previous_budget', 'budget', 'current_admitted_cost', 'budget_version', 'applied_at')
LIMIT_FIELDS = ('ttft_slo', 'e2e_slo', 'max_scheduling_lag', 'max_budget_lag')


def validate_schedule(schedule, duration):
    finite_number(duration, 'duration', True)
    if not isinstance(schedule, list) or not schedule:
        raise ValueError('Budget schedule must be a nonempty JSON array')
    normalized, previous = [], -1.
    for index, step in enumerate(schedule):
        if not isinstance(step, dict) or set(step) != {'at_seconds', 'budget'}:
            raise ValueError('Each budget step needs only at_seconds and budget')
        at = float(finite_number(step['at_seconds'], 'budget offset'))
        budget = float(finite_number(step['budget'], 'budget', True))
        if not 0 <= at < duration or at <= previous or (index == 0 and at != 0):
            raise ValueError('Budget offsets must start at zero, strictly increase, and precede trace end')
        normalized.append(dict(at_seconds=at, budget=budget))
        previous = at
    return normalized


def make_plan(config, trace, trace_hash, schedule):
    for name in LIMIT_FIELDS:
        if name != 'e2e_slo' or config[name] is not None:
            finite_number(config[name], name, True)
    schedule = validate_schedule(schedule, trace['duration_seconds'])
    if finite_number(config['budget'], 'initial budget', True) != schedule[0]['budget']:
        raise ValueError('Initial budget differs from the schedule')
    scheduled = config['budget_schedule'] is not None
    if scheduled and (not isinstance(config['budget_schedule'], str) or not config['budget_schedule']):
        raise ValueError('Scheduled control needs a source filename')
    if not scheduled and len(schedule) != 1:
        raise ValueError('Fixed control must have exactly one initial budget')
    return dict(schema_version=1, trace_sha256=trace_hash, duration_seconds=trace['duration_seconds'],
                budget_mode='scheduled' if scheduled else 'fixed', budget_schedule=schedule,
                **{name: config[name] for name in LIMIT_FIELDS})


def has_control_plan(root, manifest):
    config = manifest.get('config', {})
    return ('control_protocol' in manifest or 'replay_plan_sha256' in manifest or
            (root / PLAN_FILE).exists() or
            (isinstance(config, dict) and
             any(name in config for name in ('budget_schedule', 'e2e_slo', 'max_budget_lag'))))


def read_plan(root, manifest, trace):
    """Only genuinely old captures may use the legacy fixed-budget audit."""
    if not isinstance(manifest, dict):
        raise ValueError('Manifest must be an object')
    if not has_control_plan(root, manifest):
        return None
    if manifest.get('control_protocol') != CONTROL_PROTOCOL:
        raise ValueError('Missing or unsupported trace control protocol')
    if sha256_file(root / PLAN_FILE) != manifest.get('replay_plan_sha256'):
        raise ValueError('Replay plan hash mismatch')
    plan = json.loads((root / PLAN_FILE).read_text())
    if not isinstance(plan, dict) or type(plan.get('schema_version')) is not int or plan['schema_version'] != 1:
        raise ValueError('Unsupported replay plan schema')
    finite_number(plan['duration_seconds'], 'plan duration', True)
    for name in LIMIT_FIELDS:
        if name != 'e2e_slo' or plan[name] is not None:
            finite_number(plan[name], name, True)
    expected = make_plan(manifest['config'], trace, manifest['trace_sha256'], plan['budget_schedule'])
    if plan != expected:
        raise ValueError('Replay plan differs from the declared trace or configuration')
    return plan


def initial_control_version(manifest, plan):
    initial, warmed = manifest['initial_metrics'], manifest['post_warmup_metrics']
    version = initial['budget_version']
    if type(version) is not int or version != 0 or type(warmed['budget_version']) is not int:
        raise ValueError('Invalid initial budget version')
    budget = plan['budget_schedule'][0]['budget']
    if (warmed['budget_version'] != version or
            finite_number(initial['admission_budget'], 'initial budget', True) != budget or
            finite_number(warmed['admission_budget'], 'post-warmup budget', True) != budget):
        raise ValueError('Budget changed outside the declared control schedule')
    return version


def validate_response(response, budget, previous_budget, version, sent, acknowledged):
    if not isinstance(response, dict) or any(name not in response for name in RESPONSE_FIELDS):
        raise ValueError('Incomplete budget acknowledgement')
    for name in ('budget', 'previous_budget', 'current_admitted_cost', 'applied_at'):
        finite_number(response[name], name, name in {'budget', 'previous_budget'})
    finite_number(sent, 'budget send time')
    finite_number(acknowledged, 'budget acknowledgement time')
    if (response['budget'] != budget or response['previous_budget'] != previous_budget or
            type(response['budget_version']) is not int or response['budget_version'] != version):
        raise ValueError('Budget acknowledgement value or version differs from the declared command')
    if response['current_admitted_cost'] < 0 or not sent <= response['applied_at'] <= acknowledged:
        raise ValueError('Invalid budget acknowledgement cost or timing')
    return {name: response[name] for name in RESPONSE_FIELDS}


def budget_curve(records, start, end):
    """Observed applications, not command targets or interpolated snapshots."""
    updates = [r for r in records if r.get('event') == 'budget_updated']
    for row in updates:
        finite_number(row.get('applied_at'), 'budget application time')
        finite_number(row.get('budget'), 'applied budget', True)
    before = [r for r in updates if r['applied_at'] <= start]
    if not before:
        return [], []
    initial = max(before, key=lambda r: r['applied_at'])
    during = sorted((r for r in updates if start < r['applied_at'] < end),
                    key=lambda r: r['applied_at'])
    times = [0.] + [r['applied_at'] - start for r in during] + [end - start]
    budgets = [initial['budget']] + [r['budget'] for r in during]
    return times, budgets + [budgets[-1]]
