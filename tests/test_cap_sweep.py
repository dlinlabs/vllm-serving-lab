import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import analyze_cap_sweep as analysis
import run_cap_sweep as sweep
from experiment_protocol import load_protocol
from trace_control import make_plan
from trace_workload import sha256_file


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def good_report():
    return dict(valid=True, **{name: dict(valid=True) for name in sweep.GATES},
                performance=dict(slo_goodput_per_arrival_second=1.), performance_by_input_size={})


def test_frozen_plan_is_full_trace_fixed_caps_mirrored_and_read_only(tmp_path):
    document, _ = load_protocol(sweep.PROTOCOL)
    jobs = sweep.plan(document, tmp_path / 'raw', 'full')
    assert len(jobs) == 9 and jobs[0]['duration_s'] == 30
    assert [job['cap'] for job in jobs[1:]] == [64, 16, 96, 32, 32, 96, 16, 64]
    assert all(job['duration_s'] == 300 for job in jobs[1:])
    assert all('--shared-upstream-client' in job['command'] and '--budget-schedule' not in job['command']
               and '--engine-observations' not in job['command'] for job in jobs)
    assert len({job['name'] for job in jobs}) == 9 and not (tmp_path / 'raw').exists()
    assert all(run['role'] == 'development' for run in document['runs'])
    assert len(sweep.plan(document, tmp_path / 'raw', 'smoke')) == 1
    document['qualification']['order'].append(document['qualification']['order'][0])
    with pytest.raises(ValueError, match='exactly once'):
        sweep.plan(document, tmp_path / 'raw', 'full')


@pytest.mark.parametrize('text', [
    '', 'vllm:num_requests_running 0',
    'vllm:num_requests_running NaN\nvllm:num_requests_waiting 0',
    'vllm:num_requests_running 0\nvllm:num_requests_waiting -1',
])
def test_idle_requires_both_finite_nonnegative_gauges(text):
    with pytest.raises(RuntimeError, match='gauge'):
        sweep.idle_counts(text)


def test_idle_sums_all_engine_instances_including_busy_one():
    text = ('# HELP unrelated ignore\n'
            'vllm:num_requests_running{model_name="q",engine="0"} 0\n'
            'vllm:num_requests_running{model_name="q",engine="1"} 2\n'
            'vllm:num_requests_waiting 0\n')
    assert sweep.idle_counts(text) == {'vllm:num_requests_running': 2, 'vllm:num_requests_waiting': 0}


def test_idle_check_requires_three_consecutive_zero_samples(monkeypatch):
    import httpx
    counts = iter([0, 0, 1, 0, 0, 0])
    calls = []

    def handler(request):
        value = next(counts)
        calls.append(value)
        return httpx.Response(200, text=f'vllm:num_requests_running {value}\nvllm:num_requests_waiting 0')

    real_client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(sweep.time, 'sleep', lambda seconds: None)
    assert sweep.wait_idle()['consecutive'] == 3
    assert calls == [0, 0, 1, 0, 0, 0]


def test_backend_identity_requires_actual_launch_flags_and_accepts_truncated_engine_name(monkeypatch):
    import httpx
    import psutil
    original_read_text = Path.read_text
    monkeypatch.setattr(sweep.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw: 'boot-id' if str(path).endswith(
        '/proc/sys/kernel/random/boot_id') else original_read_text(path, *a, **kw))
    argv = ['vllm', 'serve', 'Qwen/Qwen3-4B-Instruct-2507', '--max-model-len', '8192',
            '--no-enable-prefix-caching']
    child = SimpleNamespace(pid=101, name=lambda: 'VLLM::EngineCor', create_time=lambda: 12.)
    process = SimpleNamespace(pid=100, cmdline=lambda: argv, children=lambda recursive: [child],
                              create_time=lambda: 10.)
    monkeypatch.setattr(psutil, 'Process', lambda pid: process)
    monkeypatch.setattr(psutil, 'net_connections', lambda kind: [SimpleNamespace(
        pid=100, status=psutil.CONN_LISTEN, laddr=SimpleNamespace(port=8000))])
    monkeypatch.setattr(sweep.subprocess, 'check_output', lambda *a, **kw: 'GPU-uuid, RTX 4090, driver, 24GB\n')
    real_client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=dict(data=[dict(
        id='Qwen/Qwen3-4B-Instruct-2507', max_model_len=8192)])))
    monkeypatch.setattr(httpx, 'Client', lambda **kw: real_client(transport=transport, **kw))
    identity = sweep.backend_identity()
    assert identity['backend_pid'] == 100 and identity['engines'] == [(101, 12.)]
    assert identity['prefix_caching'] is False and 'argv' not in identity
    argv.remove('--no-enable-prefix-caching')
    with pytest.raises(RuntimeError, match='prefix-caching'):
        sweep.backend_identity()


def test_native_cpu_evidence_must_match_current_revision_and_complete_junit(tmp_path, monkeypatch):
    from scripts import run_linux_cpu_tests
    path = tmp_path / 'cpu-ci-artifacts' / 'stamp'
    path.mkdir(parents=True)
    write(path / 'acceptance.json', dict(valid=True, commit='old', test_cases=500))
    (path / 'junit.xml').write_text('native record')
    monkeypatch.setattr(sweep, 'ROOT', tmp_path)
    monkeypatch.setattr(run_linux_cpu_tests, 'validate_report', lambda path: dict(test_cases=500))
    with pytest.raises(RuntimeError, match='exact commit'):
        sweep.cpu_acceptance('new')
    write(path / 'acceptance.json', dict(valid=True, commit='new', test_cases=500))
    assert sweep.cpu_acceptance('new')['junit_sha256'] == sha256_file(path / 'junit.xml')
    monkeypatch.setattr(run_linux_cpu_tests, 'validate_report', lambda path: dict(test_cases=499))
    with pytest.raises(RuntimeError, match='exact commit'):
        sweep.cpu_acceptance('new')


@pytest.mark.parametrize('missing', ['valid', *sweep.GATES])
def test_every_measurement_gate_is_required(missing):
    report = good_report()
    report.pop(missing)
    assert sweep.gate_issues(report)


@pytest.fixture
def supervisor(monkeypatch, tmp_path):
    output = tmp_path / 'raw'
    document = dict(protocol='cap-range-v1')
    jobs = [dict(name=f'capture-{i}', cap=16, duration_s=300, command=['mock', str(i)]) for i in range(3)]
    backend = dict(host='same-host', engines=[[1, 2.]])
    monkeypatch.setattr(sweep, 'load_protocol', lambda path: (document, 'sha'))
    monkeypatch.setattr(sweep, 'normalized_sha256', lambda path: 'sha')
    monkeypatch.setattr(sweep, 'plan', lambda *args: jobs)
    monkeypatch.setattr(sweep, 'git', lambda *args: 'revision' if args == ('rev-parse', 'HEAD') else '')
    monkeypatch.setattr(sweep, 'cpu_acceptance', lambda commit: {})
    monkeypatch.setattr(sweep, 'backend_identity', lambda: backend)
    monkeypatch.setattr(sweep, 'wait_idle', lambda: dict(counts={'running': 0, 'waiting': 0}))
    monkeypatch.setattr(sweep, 'qualify_capture', lambda *args: (
        dict(code_sha256={'runtime': 'sha'}, packages={'fastapi': 'test'}), good_report()))
    called = []
    exitcodes = [0, 0, 0]

    class Child:
        def __init__(self, command, **kwargs):
            index = int(command[1])
            called.append(index)
            self.code = exitcodes[index]
            directory = output / jobs[index]['name']
            directory.mkdir()
            write(directory / 'report.json', good_report())
            assert kwargs['env']['INPUT_TOKEN_COST'] == str(sweep.COST['input_token_cost'])

        def wait(self, **kwargs):
            return self.code

    monkeypatch.setattr(sweep.subprocess, 'Popen', Child)
    return SimpleNamespace(output=output, called=called, exitcodes=exitcodes, backend=backend)


def test_supervisor_collects_in_order_and_keeps_artifact_hashes(supervisor):
    sweep.run_suite(supervisor.output, 'full')
    result = sweep.read(supervisor.output / 'suite.json')
    assert result['valid'] and result['status'] == 'collected' and supervisor.called == [0, 1, 2]
    assert all(row['artifact_sha256']['report.json'] and row['backend'] == supervisor.backend for row in result['runs'])


def test_supervisor_stops_after_failed_child_and_refuses_overwrite(supervisor):
    supervisor.exitcodes[1] = 1
    with pytest.raises(RuntimeError, match='no later runs'):
        sweep.run_suite(supervisor.output, 'full')
    result = sweep.read(supervisor.output / 'suite.json')
    assert not result['valid'] and result['status'] == 'failed' and supervisor.called == [0, 1]
    assert [row['status'] for row in result['runs']] == ['passed', 'failed']
    before = (supervisor.output / 'suite.json').read_bytes()
    with pytest.raises(FileExistsError):
        sweep.run_suite(supervisor.output, 'full')
    assert (supervisor.output / 'suite.json').read_bytes() == before


def test_supervisor_stops_on_backend_restart_before_another_run(supervisor, monkeypatch):
    identities = iter([supervisor.backend] * 3 + [dict(host='other')])
    monkeypatch.setattr(sweep, 'backend_identity', lambda: next(identities))
    with pytest.raises(RuntimeError, match='backend changed'):
        sweep.run_suite(supervisor.output, 'full')
    result = sweep.read(supervisor.output / 'suite.json')
    assert not result['valid'] and supervisor.called == [0]


def test_capture_identity_rejects_wrong_revision_cost_mode_and_gate(tmp_path, monkeypatch):
    document, sha = load_protocol(sweep.PROTOCOL)
    job = sweep.plan(document, tmp_path, 'smoke')[0]
    path = tmp_path / job['name']
    path.mkdir()
    trace = sweep.read(job['config']['trace'])
    write(path / 'trace.json', trace)
    # The official trace byte hash must stay identical; copy bytes, not a reserialization.
    (path / 'trace.json').write_bytes(Path(job['config']['trace']).read_bytes())
    trace_sha = sha256_file(path / 'trace.json')
    replay = make_plan(job['config'], trace, trace_sha, [dict(at_seconds=0., budget=96.)])
    write(path / 'replay_plan.json', replay)
    metric = dict(sweep.COST, current_in_flight=0, current_admitted_cost=0,
                  tokenizer_fallback_requests=0, shared_upstream_client=True)
    manifest = dict(config=job['config'], status='collected', git_commit='revision',
                    code_sha256={'gateway.py': sha256_file(sweep.ROOT / 'gateway.py')},
                    experiment_protocol=dict(sha256=sha, run=job['name'], role='development'),
                    initial_metrics=metric, post_warmup_metrics=metric, final_metrics=metric)
    write(path / 'manifest.json', manifest)
    write(path / 'report.json', good_report())
    assert sweep.qualify_capture(path, document, sha, job['name'], 'revision')[1]['valid']
    for change, message in [
        (lambda m: m.update(git_commit='other'), 'revision'),
        (lambda m: m['final_metrics'].update(cost_intercept=1.), 'coefficients'),
        (lambda m: m['initial_metrics'].update(current_in_flight=1), 'drain'),
        (lambda m: m['experiment_protocol'].update(sha256='changed'), 'binding'),
    ]:
        altered = copy.deepcopy(manifest)
        change(altered)
        write(path / 'manifest.json', altered)
        with pytest.raises(RuntimeError, match=message):
            sweep.qualify_capture(path, document, sha, job['name'], 'revision')
    write(path / 'manifest.json', manifest)
    report = good_report()
    report['load_fidelity']['valid'] = False
    write(path / 'report.json', report)
    with pytest.raises(RuntimeError, match='load_fidelity'):
        sweep.qualify_capture(path, document, sha, job['name'], 'revision')


def test_pairing_retains_all_offered_and_does_not_require_both_accepted():
    assert analysis.paired({'a': True, 'b': False, 'c': False}, {'a': False, 'b': True, 'c': False}, 30) == dict(
        offered=3, both_met=0, left_only_met=1, right_only_met=1, left_minus_right_slo_successes=0,
        left_minus_right_goodput=0, interpretation='Matched arrivals, not matched engine state or a causal counterfactual')
    with pytest.raises(ValueError, match='offered trace IDs'):
        analysis.paired({'a': True}, {'b': True}, 30)


def test_audit_uses_private_report_output_and_leaves_raw_unchanged(tmp_path, monkeypatch):
    import analyze_trace_pilot
    write(tmp_path / 'report.json', dict(original=True))
    write(tmp_path / 'manifest.json', dict(capture=True))
    before = (tmp_path / 'report.json').read_bytes()

    def fake_audit(path, plot):
        assert not plot and path != tmp_path and (path / 'manifest.json').is_symlink()
        assert not (path / 'report.json').exists()
        write(path / 'report.json', good_report())
        return good_report()

    monkeypatch.setattr(analyze_trace_pilot, 'analyze', fake_audit)
    assert analysis.audit_copy(tmp_path)['valid']
    assert (tmp_path / 'report.json').read_bytes() == before


@pytest.fixture
def collected_suite(tmp_path, monkeypatch):
    source = tmp_path / 'suite'
    source.mkdir()
    document, _ = load_protocol(sweep.PROTOCOL)
    write(source / 'protocol.json', document)
    protocol_sha = sweep.normalized_sha256(source / 'protocol.json')
    records, manifests = [], {}
    backend = dict(host='same-host')
    for definition in document['runs']:
        name = definition['name']
        path = source / name
        path.mkdir()
        cap = document['schedules'][definition['schedule']]['fixed_budget']
        rows = [dict(trace_id=str(i), scheduled_at=100. + i * 30, first_content_at=100. + i * 30 + .1,
                     finished_at=102. + i * 30, success=True, rejected=False, ttft=.1, end_to_end=2.,
                     expected_input_tokens=800) for i in range(2)]
        trace = dict(duration_seconds=300, requests=[dict(trace_id=str(i), arrival_seconds=i * 30) for i in range(2)])
        write(path / 'trace.json', trace)
        write(path / 'client.json', dict(requests=rows))
        write(path / 'report.json', good_report())
        records.append(dict(name=name, status='passed', exitcode=0, cap=cap, backend=backend,
                            artifact_sha256={p.name: sha256_file(p) for p in path.iterdir()}))
        manifests[name] = dict(host='same-host', code_sha256={}, trace_sha256='same', config=dict(ttft_slo=.5),
                               packages={'fastapi': 'test'})
    suite = dict(stage='full', valid=True, status='collected', protocol_sha256=protocol_sha, commit='revision',
                 planned=document['qualification']['order'], runs=records, backend=backend,
                 cpu_acceptance=dict(acceptance=dict(valid=True, commit='revision')), packages={'fastapi': 'test'})
    write(source / 'suite.json', suite)
    monkeypatch.setattr(analysis, 'qualify_capture', lambda path, *args: (manifests[path.name], good_report()))
    monkeypatch.setattr(analysis, 'audit_copy', lambda path: good_report())
    return source, tmp_path / 'analysis'


def test_analyzer_preserves_whole_run_denominator_and_matched_cohorts(collected_suite):
    source, output = collected_suite
    result = analysis.analyze_suite(source, output, plot=True)
    assert result['valid'] and result['model_readiness'] == 'not_established'
    assert len(result['by_cap']) == 4 and len(result['per_run']) == 8
    assert all(row['slo_goodput_per_arrival_second'] == 2 / 300 for row in result['per_run'])
    assert len(result['matched_30s_cohorts']) == 80 and len(result['paired_to_cap64']) == 6
    assert (output / 'goodput.png').stat().st_size > 0
    with pytest.raises(ValueError, match='fresh analysis'):
        analysis.analyze_suite(source, output, plot=False)


@pytest.mark.parametrize('damage', ['raw', 'cap', 'backend', 'missing', 'incomplete', 'cpu'])
def test_analyzer_refuses_corruption_mislabeling_or_partial_series(collected_suite, damage):
    source, output = collected_suite
    suite = sweep.read(source / 'suite.json')
    if damage == 'raw':
        (source / suite['runs'][0]['name'] / 'client.json').write_text('{}')
    elif damage == 'cap':
        suite['runs'][0]['cap'] = 16
    elif damage == 'backend':
        suite['runs'][0]['backend'] = dict(host='changed')
    elif damage == 'missing':
        suite['runs'].pop()
    elif damage == 'cpu':
        suite['cpu_acceptance']['acceptance']['commit'] = 'other'
    else:
        suite['status'] = 'failed'
    write(source / 'suite.json', suite)
    with pytest.raises(ValueError):
        analysis.analyze_suite(source, output, plot=False)
    assert not output.exists()
