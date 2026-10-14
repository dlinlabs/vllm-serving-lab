"""Control/protocol regressions; native execution is deferred with the CPU suite."""
import asyncio
from copy import deepcopy
import json
import os
import time
from types import SimpleNamespace

import httpx
import pytest

from analyze_trace_pilot import control_audit, stats
from run_budget_pilot import save
import run_trace_pilot
from test_trace_replay import sse, trace
from trace_control import (CONTROL_PROTOCOL, PLAN_FILE, budget_curve, make_plan, read_plan,
                           validate_schedule)
from trace_workload import sha256_file


def controls():
    return [dict(at_seconds=0, budget=96), dict(at_seconds=1, budget=64),
            dict(at_seconds=2, budget=96)]


def config():
    return dict(budget=96., budget_schedule='steps.json', ttft_slo=.5, e2e_slo=1.,
                max_scheduling_lag=.05, max_budget_lag=.05)


@pytest.fixture
def declared(tmp_path):
    data = trace((0, 1.5, 2.5), 3.)
    save(tmp_path / 'trace.json', data)
    manifest = dict(config=config(), trace_sha256=sha256_file(tmp_path / 'trace.json'),
                    control_protocol=CONTROL_PROTOCOL)
    plan = make_plan(manifest['config'], data, manifest['trace_sha256'], controls())
    save(tmp_path / PLAN_FILE, plan)
    manifest['replay_plan_sha256'] = sha256_file(tmp_path / PLAN_FILE)
    return data, manifest, plan


def test_schedule_is_separate_normalized_and_does_not_modify_arrivals(declared, tmp_path):
    data, manifest, plan = declared
    original = deepcopy(data)
    schedule = controls()
    normalized = validate_schedule(schedule, 3)
    normalized[1]['budget'] = 80
    assert schedule == controls()
    assert read_plan(tmp_path, manifest, data) == plan
    assert data == original
    assert sha256_file(tmp_path / 'trace.json') == manifest['trace_sha256']


@pytest.mark.parametrize('schedule', [
    [], {}, [dict(at_seconds=1, budget=96)],
    [dict(at_seconds=0, budget=96), dict(at_seconds=0, budget=64)],
    [dict(at_seconds=0, budget=96), dict(at_seconds=2, budget=64), dict(at_seconds=1, budget=96)],
    [dict(at_seconds=0, budget=96), dict(at_seconds=3, budget=64)],
    [dict(at_seconds=-1, budget=96)], [dict(at_seconds=False, budget=96)],
    [dict(at_seconds=float('nan'), budget=96)], [dict(at_seconds=float('inf'), budget=96)],
    [dict(at_seconds=0, budget=0)], [dict(at_seconds=0, budget=-1)],
    [dict(at_seconds=0, budget=True)], [dict(at_seconds=0, budget='96')],
    [dict(at_seconds=0, budget=float('nan'))], [dict(at_seconds=0, budget=float('inf'))],
    [dict(at_seconds=0)], [dict(at_seconds=0, budget=96, extra='unsupported')],
])
def test_invalid_schedule_fails_before_collection(schedule):
    with pytest.raises(ValueError):
        validate_schedule(schedule, 3)


def test_fixed_plan_and_legacy_are_distinct(declared, tmp_path):
    data, manifest, _ = declared
    manifest['config']['budget_schedule'] = None
    plan = make_plan(manifest['config'], data, manifest['trace_sha256'], controls()[:1])
    save(tmp_path / PLAN_FILE, plan)
    manifest['replay_plan_sha256'] = sha256_file(tmp_path / PLAN_FILE)
    assert read_plan(tmp_path, manifest, data)['budget_mode'] == 'fixed'
    with pytest.raises(ValueError, match='exactly one'):
        make_plan(manifest['config'], data, manifest['trace_sha256'], controls())
    (tmp_path / PLAN_FILE).unlink()
    assert read_plan(tmp_path, dict(config=dict(budget=96, ttft_slo=.5)), data) is None
    with pytest.raises(FileNotFoundError):
        read_plan(tmp_path, manifest, data)


@pytest.mark.parametrize('damage', ['protocol', 'hash', 'config', 'schema', 'extra', 'threshold', 'boolean', 'trace_binding'])
def test_partial_or_changed_declaration_cannot_be_reinterpreted_as_legacy(declared, tmp_path, damage):
    data, manifest, plan = declared
    if damage == 'protocol':
        del manifest['control_protocol']
    elif damage == 'hash':
        del manifest['replay_plan_sha256']
    elif damage == 'config':
        manifest['config']['e2e_slo'] = 2.
    elif damage == 'schema':
        plan['schema_version'] = 2
    elif damage == 'extra':
        plan['extra'] = 'not declared'
    elif damage == 'threshold':
        plan['max_budget_lag'] = .1
    elif damage == 'boolean':
        plan['e2e_slo'] = True
    else:
        plan['trace_sha256'] = '0' * 64
    if damage not in {'protocol', 'hash', 'config'}:
        save(tmp_path / PLAN_FILE, plan)
        manifest['replay_plan_sha256'] = sha256_file(tmp_path / PLAN_FILE)
    with pytest.raises(ValueError):
        read_plan(tmp_path, manifest, data)


def control_fixture():
    plan = make_plan(config(), dict(duration_seconds=3.), '0' * 64, controls())
    manifest = dict(measurement_start=100., measurement_end=103., warmup_end=99.,
        initial_metrics=dict(admission_budget=96., budget_version=0),
        post_warmup_metrics=dict(admission_budget=96., budget_version=0),
        final_metrics=dict(admission_budget=96., budget_version=3, state_sampled_at=104.))
    changes, records, observations = [], [], []
    for index, step in enumerate(controls()):
        target = 100 + step['at_seconds']
        sent = target + .01 if index else 99.8
        applied = dict(previous_budget=controls()[max(0, index - 1)]['budget'],
            budget=step['budget'], budget_version=index + 1, applied_at=sent + .005,
            current_admitted_cost=90. if index == 1 else 0.)
        changes.append(dict(step_index=index, requested_budget=step['budget'], target=target, sent=sent,
                            acknowledged=sent + .01, status='acknowledged', **applied))
        records.append(dict(event='budget_updated', monotonic=sent + .006, **applied))
        observations.append(dict(metrics=dict(state_sampled_at=target + .5,
                                              admission_budget=step['budget'], budget_version=index + 1)))
    return manifest, plan, changes, records, observations


def test_complete_control_audit_and_actual_application_curve():
    manifest, plan, changes, records, observations = control_fixture()
    issues, fidelity = control_audit(manifest, plan, changes, records, observations)
    assert not issues and fidelity['valid']
    assert fidelity['planned_commands'] == fidelity['gateway_applications'] == 3
    assert fidelity['max_application_lag'] == pytest.approx(.015)
    assert fidelity['max_send_lag'] == pytest.approx(.01)
    # Lowering the cap does not evict the already reserved 90 cost units.
    assert changes[1]['current_admitted_cost'] > changes[1]['budget']
    times, budgets = budget_curve(records, 100., 103.)
    assert times == pytest.approx([0., 1.015, 2.015, 3.])
    assert budgets == [96, 64, 96, 96]
    assert budget_curve([], 100., 103.) == ([], [])


@pytest.mark.parametrize('field,value', [
    ('applied_at', None), ('applied_at', float('nan')), ('budget', 'unknown'), ('budget', True),
])
def test_malformed_application_cannot_be_drawn_as_a_known_budget(field, value):
    _, _, _, records, _ = control_fixture()
    records[1][field] = value
    with pytest.raises(ValueError):
        budget_curve(records, 100., 103.)


@pytest.mark.parametrize('event', ['snapshot', 'accepted', 'rejected'])
def test_gateway_snapshots_and_admission_versions_match_actual_applications(event):
    manifest, plan, changes, records, observations = control_fixture()
    budget_field = 'admission_budget' if event == 'snapshot' else 'budget'
    state = dict(event=event, monotonic=101.5, state_sampled_at=101.5,
                 budget_version=2, **{budget_field: 64.})
    records.append(state)
    records.sort(key=lambda r: r['monotonic'])
    issues, fidelity = control_audit(manifest, plan, changes, records, observations)
    assert not issues and fidelity['valid']
    state[budget_field] = 96.
    issues, fidelity = control_audit(manifest, plan, changes, records, observations)
    assert issues and not fidelity['valid']
    state['budget_version'] = 1
    issues, fidelity = control_audit(manifest, plan, changes, records, observations)
    assert issues and not fidelity['valid']


@pytest.mark.parametrize('metrics,field,value', [
    ('initial_metrics', 'budget_version', 1), ('initial_metrics', 'budget_version', False),
    ('post_warmup_metrics', 'budget_version', 1), ('post_warmup_metrics', 'admission_budget', 80.),
])
def test_control_requires_a_fresh_unchanged_warmup_budget(metrics, field, value):
    manifest, plan, changes, records, observations = control_fixture()
    manifest[metrics][field] = value
    with pytest.raises(ValueError):
        control_audit(manifest, plan, changes, records, observations)


@pytest.mark.parametrize('damage', ['missing', 'extra', 'order', 'target', 'early', 'late_ack', 'unacknowledged', 'state'])
def test_command_count_order_targets_and_observed_state_are_strict(damage):
    manifest, plan, changes, records, observations = control_fixture()
    if damage == 'missing':
        changes.pop()
    elif damage == 'extra':
        records.append(dict(records[-1]))
    elif damage == 'order':
        changes[1]['step_index'] = 0
    elif damage == 'target':
        changes[1]['target'] += .1
    elif damage == 'early':
        changes[1]['sent'] = 100.99
    elif damage == 'late_ack':
        changes[0]['acknowledged'] = 100.01
    elif damage == 'unacknowledged':
        changes[1]['status'] = 'cancelled'
    else:
        observations[1]['metrics']['budget_version'] = 1
    issues, fidelity = control_audit(manifest, plan, changes, records, observations)
    assert issues and not fidelity['valid']


@pytest.mark.parametrize('field,value', [
    ('budget', 65), ('previous_budget', 80), ('budget_version', 4),
    ('budget_version', True), ('applied_at', 99.), ('current_admitted_cost', -1),
])
def test_acknowledgement_is_bound_to_the_command_and_gateway_event(field, value):
    manifest, plan, changes, records, observations = control_fixture()
    changes[1][field] = value
    with pytest.raises(ValueError):
        control_audit(manifest, plan, changes, records, observations)


def test_control_fidelity_is_separate_from_structural_integrity():
    manifest, plan, changes, records, observations = control_fixture()
    changes[1].update(applied_at=101.08, acknowledged=101.09)
    records[1].update(applied_at=101.08, monotonic=101.081)
    issues, fidelity = control_audit(manifest, plan, changes, records, observations)
    assert not issues and not fidelity['valid']
    assert fidelity['max_application_lag'] == pytest.approx(.08)


@pytest.mark.parametrize('failure', ['send_late', 'apply_late', 'prior_ack_late', 'window_ended'])
def test_overdue_control_does_not_catch_up_or_issue_later_steps(monkeypatch, failure):
    manifest, plan, changes, _, _ = control_fixture()
    changes = changes[:1]
    clock = SimpleNamespace(now=100.)
    requested = []
    if failure == 'prior_ack_late':
        changes[0]['acknowledged'] = 101.001
    async def sleep(delay):
        clock.now += delay
        if failure == 'send_late':
            clock.now += .06
        elif failure == 'window_ended':
            clock.now = 103.
    async def send(client, key, budget):
        requested.append(budget)
        if failure == 'apply_late':
            clock.now += .06
        return dict(previous_budget=96., budget=budget, budget_version=2,
                    applied_at=clock.now, current_admitted_cost=90.)
    monkeypatch.setattr(run_trace_pilot, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(run_trace_pilot, 'asyncio', SimpleNamespace(sleep=sleep, CancelledError=asyncio.CancelledError))
    monkeypatch.setattr(run_trace_pilot, 'set_budget', send)
    with pytest.raises(RuntimeError):
        asyncio.run(run_trace_pilot.scheduled_budgets(None, 'unused', plan, 100., 0, changes))
    assert requested == ([64.] if failure == 'apply_late' else [])
    assert len(changes) == 2 and changes[1]['status'] == 'failed'


@pytest.mark.parametrize('failure', ['control', 'traffic'])
def test_failure_cancels_the_other_task_and_persists_all_planned_rows(declared, tmp_path, monkeypatch, failure):
    data, manifest, plan = declared
    if failure == 'traffic':
        data = trace((0, .01, .02), 3.)
        save(tmp_path / 'trace.json', data)
        manifest['trace_sha256'] = sha256_file(tmp_path / 'trace.json')
        plan['trace_sha256'] = manifest['trace_sha256']
    plan['max_budget_lag'] = manifest['config']['max_budget_lag'] = 1.
    save(tmp_path / PLAN_FILE, plan)
    manifest['replay_plan_sha256'] = sha256_file(tmp_path / PLAN_FILE)
    async def scenario():
        alive, budget, version, requests = False, 96., 0, 0
        class GatewayProcess:
            pid = os.getpid()
            returncode = None
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = 0
            def wait(self, timeout=None):
                return self.returncode
        process = GatewayProcess()
        def launch(*args, **kwargs):
            nonlocal alive
            alive = True
            return process
        async def serve(request):
            nonlocal budget, version, requests
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [dict(id='mock-model', max_model_len=8192)]})
            if request.url.path == '/metrics':
                if not alive:
                    raise httpx.ConnectError('Gateway is not running')
                return httpx.Response(200, json=dict(tokenizer_available=True, gateway_diagnostics=False,
                    telemetry_run_id='12345678123456781234567812345678', current_in_flight=0,
                    current_admitted_cost=0., state_sampled_at=time.monotonic(), admission_budget=budget,
                    budget_version=version))
            if request.url.path == '/admin/budget':
                if version and failure == 'control':
                    return httpx.Response(500)
                previous, budget = budget, json.loads(request.content)['budget']
                version += 1
                return httpx.Response(200, json=dict(previous_budget=previous, budget=budget,
                    budget_version=version, current_admitted_cost=0., applied_at=time.monotonic()))
            requests += 1
            if requests > 1:
                await asyncio.Event().wait()
            return sse(request.headers['x-benchmark-request-id'])
        real_client = httpx.AsyncClient
        monkeypatch.setattr(run_trace_pilot.httpx, 'AsyncClient',
                            lambda **kwargs: real_client(transport=httpx.MockTransport(serve), **kwargs))
        monkeypatch.setattr(run_trace_pilot, 'subprocess', SimpleNamespace(Popen=launch, STDOUT=-2))
        args = SimpleNamespace(budget=96., gateway_diagnostics=False, engine_observations=False,
            sampler_mode='thread', vllm_url='http://mock-engine', startup_timeout=2, drain_timeout=2,
            request_timeout=10, warmup_requests=1, max_pending=1 if failure == 'traffic' else 10)
        with pytest.raises(ExceptionGroup):
            await run_trace_pilot.run(args, data, tmp_path, manifest)
        assert process.returncode == 0
        rows = json.loads((tmp_path / 'client.json').read_text())['requests']
        assert len(rows) == len(data['requests']) == 3
        assert rows[0]['error_type'] == 'CancelledError'
        assert all(r['error_type'] == 'NotLaunched' for r in rows[1:])
        assert [r['scheduled_at'] for r in rows] == [
            manifest['measurement_start'] + r['arrival_seconds'] for r in data['requests']]
        changes = json.loads((tmp_path / 'budget_changes.json').read_text())
        assert changes[0]['status'] == 'acknowledged'
        assert changes[1]['status'] == ('failed' if failure == 'control' else 'cancelled')
        assert len(changes) == 2 and version == 1
        assert (tmp_path / 'state_observations.json').is_file()
    asyncio.run(scenario())


def test_joint_goodput_uses_scheduled_deadlines_and_all_offered_denominators():
    rows = [dict(success=True, rejected=False, scheduled_at=10., started_at=10.25,
                 first_content_at=first, finished_at=finished, ttft=first - 10.25,
                 end_to_end=finished - 10.25, stream_complete=True)
            for first, finished in [(10.5, 11.), (10.75, 11.), (10.5, 11.25), (10.75, 11.25)]]
    rows += [dict(success=False, rejected=True), dict(success=False, rejected=False)]
    before = deepcopy(rows)
    result = stats(rows, 2., .5, 1.)
    assert result['attempted'] == 6 and result['success'] == 4
    assert result['rejected'] == result['failed'] == 1
    assert result['slo_successes'] == 1
    assert result['ttft_slo_successes'] == result['e2e_slo_successes'] == 2
    assert result['slo_goodput_per_arrival_second'] == .5
    assert result['slo_success_fraction_of_offered'] == 1 / 6
    assert result['scheduled_p99_e2e'] > result['client_p99_e2e']
    legacy = stats(rows, 2., .5)
    assert legacy['slo_successes'] == 2 and 'scheduled_p99_e2e' not in legacy
    assert 'E2E SLO' not in legacy['slo_definition']
    assert rows == before
    for invalid in (0, -1, True, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            stats(rows, 2., .5, invalid)


@pytest.mark.parametrize('extra', [
    ['--budget', '96', '--budget-schedule', 'unused.json'],
    ['--e2e-slo', '0'], ['--e2e-slo', 'nan'], ['--max-budget-lag', '-1'],
])
def test_cli_rejects_conflicting_or_invalid_controls_before_reading_trace(tmp_path, monkeypatch, extra):
    monkeypatch.setattr('sys.argv', ['run_trace_pilot.py', '--trace', 'not-read.json',
        '--output-dir', str(tmp_path / 'run'), '--ttft-slo', '.5', *extra])
    with pytest.raises(SystemExit) as exc:
        run_trace_pilot.main()
    assert exc.value.code == 2 and not (tmp_path / 'run').exists()
