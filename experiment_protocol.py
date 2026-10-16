"""Experiment protocols: declared capture identity and collection-time binding.

A protocol (experiments/<name>.json) declares traces, budget schedules, replay settings and runs.
Every capture's identity (trace provenance, normalized replay plan and replay settings) is compared
with its run's declaration. Captures started with run_trace_pilot.py --protocol also record the
protocol hash and acceptance status at collection time, so acceptance can check that a test run was
collected under the exact frozen protocol it is scored against. The binding is written by the runner:
it rules out editing the protocol (including its status) after collection, not repeated captures.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from trace_control import make_plan, validate_schedule

ROOT = Path(__file__).resolve().parent
ROLES = ('development', 'test')
# Not identity settings: the trace and schedule are checked by content, the budget through the
# normalized plan, and the server URL and output path are host-specific.
NOT_REPLAY = frozenset({'trace', 'output_dir', 'budget', 'budget_schedule', 'vllm_url', 'protocol',
                        'protocol_run'})
# Declared trace field -> capture trace identity field.
TRACE_FIELDS = dict(sha256='sha256', source_sha256='source_sha256', model='model',
                    duration_seconds='duration_seconds', output_cap='output_cap',
                    sample_probability='sampling_probability', seed='seed', time_scale='time_scale',
                    preset='preset')
# Locations, not identity; window and start are compared as instants.
TRACE_OTHER = frozenset({'path', 'source', 'output_dir', 'window', 'start'})


def normalized_sha256(path):
    """Hash of LF-normalized bytes, so a CRLF checkout keeps the committed hash."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def instant(value):
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def same(declared, actual):
    if isinstance(declared, bool) or isinstance(actual, bool) or declared is None or actual is None:
        return declared is actual
    if isinstance(declared, (int, float)) and isinstance(actual, (int, float)):
        return float(declared) == float(actual)
    return declared == actual


def validate_declarations(document):
    for section in ('traces', 'schedules', 'replay'):
        if not isinstance(document.get(section), dict) or not document[section]:
            raise ValueError(f'The protocol needs a nonempty {section} section')
    for name, trace in document['traces'].items():
        if set(trace) - set(TRACE_FIELDS) - TRACE_OTHER or not set(trace) & {'sha256', 'source_sha256'}:
            raise ValueError(f'Trace {name} needs sha256 or source_sha256 and only known fields')
        try:
            if 'window' in trace:
                start, end = trace['window']
                if instant(end) <= instant(start):
                    raise ValueError
            if 'start' in trace:
                instant(trace['start'])
        except (TypeError, ValueError):
            raise ValueError(f'Trace {name} needs ISO timestamps and an ordered window') from None
    for name, schedule in document['schedules'].items():
        if ('fixed_budget' in schedule) == ('path' in schedule) or ('path' in schedule) != ('sha256' in schedule):
            raise ValueError(f'Schedule {name} needs fixed_budget, or path with sha256')
    runs = document.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('The protocol needs a nonempty runs list')
    names = [run.get('name') for run in runs]
    if None in names or len(set(names)) != len(names):
        raise ValueError('Protocol runs need unique names')
    for run in runs:
        if (run.get('trace') not in document['traces'] or run.get('schedule') not in document['schedules']
                or run.get('role') not in ROLES):
            raise ValueError(f"Run {run['name']} needs a declared trace, schedule and role")


def load_protocol(path):
    """The validated protocol document and the hash of its LF-normalized bytes."""
    document = json.loads(Path(path).read_bytes())
    validate_declarations(document)
    return document, normalized_sha256(path)


def declared_run(document, name):
    return next((run for run in document['runs'] if run['name'] == name), None)


def capture_identity(config, trace, trace_sha256, plan, binding=None):
    """Identity of one capture: trace provenance, normalized replay plan and replay settings."""
    provenance = trace.get('provenance') or {}
    preset = provenance.get('preset')
    return dict(
        trace=dict(sha256=trace_sha256, source_sha256=provenance.get('source_sha256'),
                   window_start=provenance.get('window_start'), duration_seconds=trace.get('duration_seconds'),
                   model=trace.get('model'), output_cap=trace.get('output_cap'),
                   sampling_probability=provenance.get('sampling_probability'), seed=provenance.get('seed'),
                   time_scale=provenance.get('time_scale'),
                   preset=preset.get('name') if isinstance(preset, dict) else None),
        budget_mode=plan['budget_mode'] if plan else None,
        budget_schedule=plan['budget_schedule'] if plan else None,
        replay={key: value for key, value in config.items() if key not in NOT_REPLAY},
        experiment_protocol=binding)


def expected_schedule(declared, duration):
    if 'fixed_budget' in declared:
        return 'fixed', validate_schedule([dict(at_seconds=0, budget=declared['fixed_budget'])], duration)
    path = Path(declared['path'])
    path = path if path.is_absolute() else ROOT / path
    if normalized_sha256(path) != declared['sha256']:
        raise ValueError(f"{declared['path']} differs from its declared sha256")
    steps = validate_schedule(json.loads(path.read_text()), duration)
    if 'budgets' in declared and [step['budget'] for step in steps] != [float(b) for b in declared['budgets']]:
        raise ValueError(f"{declared['path']} differs from its declared budgets")
    return 'scheduled', steps


def capture_mismatches(document, name, capture):
    """Why a capture differs from its run's declaration; empty when trace, plan and replay match."""
    run = declared_run(document, name)
    if run is None:
        return [f'{name} is not declared in the protocol runs']
    if not isinstance(capture, dict):
        return [f'{name} has no capture provenance; rebuild the features with this version']
    problems, declared, actual = [], document['traces'][run['trace']], capture.get('trace') or {}
    for key, field in TRACE_FIELDS.items():
        if key in declared and not same(declared[key], actual.get(field)):
            problems.append(f'trace {key} is {actual.get(field)!r}, declared {declared[key]!r}')
    start = declared.get('start', (declared.get('window') or [None])[0])
    if start is not None:
        try:
            matches = instant(actual['window_start']) == instant(start)
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            problems.append(f"trace window starts {actual.get('window_start')!r}, declared {start!r}")
    if 'window' in declared:
        span = (instant(declared['window'][1]) - instant(declared['window'][0])).total_seconds()
        if not same(span, actual.get('duration_seconds')):
            problems.append(f"trace lasts {actual.get('duration_seconds')!r} s, declared window {span:g} s")
    try:
        mode, steps = expected_schedule(document['schedules'][run['schedule']], actual.get('duration_seconds'))
    except (OSError, TypeError, ValueError) as exc:
        problems.append(f"schedule {run['schedule']} cannot be checked: {exc}")
    else:
        if capture.get('budget_mode') != mode or capture.get('budget_schedule') != steps:
            problems.append(f"budget plan {capture.get('budget_mode')} {capture.get('budget_schedule')} differs "
                            f"from declared schedule {run['schedule']}")
    replay, expected = capture.get('replay') or {}, document['replay']
    for key in sorted(set(replay) | set(expected)):
        if key not in replay or key not in expected or not same(expected[key], replay[key]):
            problems.append(f"replay {key} is {replay.get(key, 'missing')!r}, declared "
                            f"{expected.get(key, 'nothing')!r}")
    return [f'{name}: {problem}' for problem in problems]


def binding(document, sha256, name):
    """What a capture records about the protocol it was collected under."""
    return dict(protocol=document.get('protocol'), sha256=sha256, run=name, role=declared_run(document, name)['role'],
                acceptance_status=(document.get('acceptance') or {}).get('status'))


def collection_binding(path, name, output_dir, config, trace, trace_sha256, schedule):
    """Refuse a capture that differs from its declaration; return the binding to record in its manifest."""
    document, sha256 = load_protocol(path)
    run = declared_run(document, name)
    if run is None:
        raise ValueError(f'{name} is not declared in {path}')
    if Path(output_dir).name != name:
        raise ValueError('The output directory name must equal --protocol-run')
    if run['role'] == 'test' and (document.get('acceptance') or {}).get('status') != 'frozen':
        raise ValueError('A test run is collected only after the acceptance limits are frozen')
    plan = make_plan(config, trace, trace_sha256, schedule)
    problems = capture_mismatches(document, name, capture_identity(config, trace, trace_sha256, plan))
    if problems:
        raise ValueError('The capture differs from the protocol: ' + '; '.join(problems))
    return dict(path=str(path), **binding(document, sha256, name))


def predeclaration_problems(document, sha256, name, capture):
    """Why a test capture lacks a record of collection under these exact, already frozen limits."""
    recorded = (capture or {}).get('experiment_protocol')
    if not isinstance(recorded, dict):
        return [f'{name} has no collection-time protocol binding (run_trace_pilot.py --protocol)']
    expected = binding(document, sha256, name)
    problems = [f"{name} recorded {key} {recorded.get(key)!r}, now {value!r}" for key, value in expected.items()
                if recorded.get(key) != value]
    if recorded.get('acceptance_status') != 'frozen':
        problems.append(f'{name} was collected before the acceptance limits were frozen')
    return problems
