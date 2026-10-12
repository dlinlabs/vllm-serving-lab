from contextlib import nullcontext
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import psutil
import pytest

import diagnostic_sampling
from diagnostic_sampling import ObservationSampler, REQUIRED_METRICS, metric_inventory
from run_trace_pilot import gateway_environment


def metrics_text(value=1, exporter_start=100):
    lines = []
    for kind, names in REQUIRED_METRICS.values():
        name = names[0]
        exposed = name + '_total' if kind == 'counter' else name
        lines.append(f'# TYPE {exposed} {kind}')
        if kind == 'histogram':
            lines.extend([f'{name}_bucket{{le="+Inf"}} {value}',
                          f'{name}_sum {value * .01}', f'{name}_count {value}'])
        else:
            lines.append(f'{name}{"_total" if kind == "counter" else ""} {value}')
    lines.extend(['# TYPE process_start_time_seconds gauge', f'process_start_time_seconds {exporter_start}'])
    return '\n'.join(lines) + '\n'


@pytest.fixture
def linux_procfs(tmp_path, monkeypatch):
    procfs = tmp_path / 'configured-procfs'
    (procfs / 'self').mkdir(parents=True)
    monkeypatch.setattr(diagnostic_sampling, 'sys', SimpleNamespace(platform='linux'))
    monkeypatch.setattr(diagnostic_sampling, 'os', SimpleNamespace(getpid=lambda: 5))
    monkeypatch.setattr(psutil, 'PROCFS_PATH', str(procfs), raising=False)
    return procfs / 'self' / 'status'


@pytest.fixture
def existing_process(monkeypatch):
    process = SimpleNamespace(oneshot=nullcontext, create_time=lambda: 100.,
                              memory_info=lambda: SimpleNamespace(rss=4096),
                              cpu_times=lambda: SimpleNamespace(user=1., system=.5))
    constructor = Mock(return_value=process)
    monkeypatch.setattr(psutil, 'Process', constructor)
    return constructor


@pytest.mark.parametrize('pid', [5, 17])
def test_compatible_linux_procfs_collects_runner_and_child(linux_procfs, existing_process, pid):
    linux_procfs.write_bytes(b'Name:\tpython-\xff\nTgid:\t5\nNStgid:\t5\nNSpid:\t5\n')
    resource = ObservationSampler.process_resources(pid)
    assert resource['available'] and resource['pid'] == pid
    assert resource['process_created_unix'] == 100. and resource['rss_bytes'] == 4096
    assert resource['user_cpu_seconds'] == 1. and resource['system_cpu_seconds'] == .5
    assert resource['sample_start'] <= resource['sample_end']
    existing_process.assert_called_once_with(pid)


@pytest.mark.parametrize('namespace_pids', [b'1569 5', b'5 5', b'6'])
@pytest.mark.parametrize('pid', [5, 17])
def test_linux_namespace_mismatch_rejects_existing_colliding_pid(
        linux_procfs, existing_process, namespace_pids, pid):
    linux_procfs.write_bytes(b'NStgid:\t' + namespace_pids + b'\n')
    resource = ObservationSampler.process_resources(pid)
    assert resource['available'] is False and resource['pid'] == pid
    assert resource['error_reason'] == 'procfs_pid_namespace_unverified'
    assert resource['sample_start'] <= resource['sample_end']
    assert not {'rss_bytes', 'user_cpu_seconds', 'system_cpu_seconds'} & resource.keys()
    existing_process.assert_not_called()


@pytest.mark.parametrize('status', [b'', b'NSpid:\t5\n', b'NStgid:\n',
                                  b'NStgid:\t5\nNStgid:\t5\n', b'NStgid:\tinvalid\n',
                                  b'NStgid:\t5 invalid\n', b'NStgid:\t0\n', b'NStgid:\t-1\n'])
def test_unverifiable_linux_procfs_is_unavailable(linux_procfs, existing_process, status):
    linux_procfs.write_bytes(status)
    resource = ObservationSampler.process_resources(5)
    assert resource['available'] is False
    assert resource['error_reason'] == 'procfs_pid_namespace_unverified'
    existing_process.assert_not_called()


@pytest.mark.parametrize('error', [FileNotFoundError, PermissionError])
def test_unreadable_linux_procfs_is_unavailable(linux_procfs, existing_process, monkeypatch, error):
    original_read = Path.read_bytes
    def read_bytes(path):
        if path == linux_procfs:
            raise error('Cannot read namespace metadata')
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    resource = ObservationSampler.process_resources(5)
    assert resource['available'] is False
    assert resource['error_reason'] == 'procfs_pid_namespace_unverified'
    assert resource['error_type'] == error.__name__
    existing_process.assert_not_called()


def test_linux_procfs_compatibility_is_rechecked(linux_procfs, existing_process):
    linux_procfs.write_bytes(b'NStgid:\t5\n')
    assert ObservationSampler.process_resources(17)['available']
    linux_procfs.write_bytes(b'NStgid:\t1569 5\n')
    assert not ObservationSampler.process_resources(17)['available']
    existing_process.assert_called_once_with(17)


def test_missing_process_failure_is_preserved(linux_procfs, existing_process):
    linux_procfs.write_bytes(b'NStgid:\t5\n')
    existing_process.side_effect = psutil.NoSuchProcess(17)
    resource = ObservationSampler.process_resources(17)
    assert resource['available'] is False and resource['pid'] == 17
    assert resource['error_type'] == 'NoSuchProcess'
    existing_process.assert_called_once_with(17)


@pytest.mark.parametrize('platform', ['win32', 'darwin'])
def test_native_process_collection_does_not_require_procfs(
        linux_procfs, existing_process, monkeypatch, platform):
    monkeypatch.setattr(diagnostic_sampling, 'sys', SimpleNamespace(platform=platform))
    assert not linux_procfs.exists()
    assert ObservationSampler.process_resources(17)['available']
    existing_process.assert_called_once_with(17)


def test_live_metric_types_aliases_zero_and_unavailability():
    inventory, counters, identity = metric_inventory(metrics_text(0))
    assert set(inventory) == set(REQUIRED_METRICS)
    assert all(v['available'] for v in inventory.values())
    assert counters and set(counters.values()) == {0}
    assert identity == {'process_start_time_seconds': 100}
    assert metric_inventory(metrics_text().replace('vllm:num_preemptions_total',
                                                   'vllm:num_preemptions'))[0]['preemptions']['available']
    text = metrics_text().replace('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc')
    assert metric_inventory(text)[0]['kv_cache']['available']
    assert all(not v['available'] and v['reason'] == 'not_exported'
               for v in metric_inventory('')[0].values())
    text = metrics_text().replace('# TYPE vllm:num_requests_waiting gauge',
                                  '# TYPE vllm:num_requests_waiting untyped')
    assert metric_inventory(text)[0]['waiting']['reason'] == 'unexpected_type'
    text = metrics_text().replace('vllm:num_requests_running 1', 'vllm:num_requests_running NaN')
    assert metric_inventory(text)[0]['running']['reason'] == 'nonfinite_samples'
    text = metrics_text().replace('vllm:num_requests_running 1', 'vllm:num_requests_running -1')
    assert metric_inventory(text)[0]['running']['reason'] == 'negative_samples'
    text = metrics_text().replace('vllm:request_queue_time_seconds_count 1\n', '')
    assert metric_inventory(text)[0]['queue_time']['reason'] == 'incomplete_histogram'


def test_external_timing_resources_reset_and_missing_metrics(tmp_path):
    payloads = iter([metrics_text(5), metrics_text(2), metrics_text(3, 200), ''])
    sampler = ObservationSampler(tmp_path, 'http://user:secret@engine:8000', os.getpid(), 'test')
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=next(payloads)))) as client:
        rows = [sampler.sample(client) for _ in range(4)]
    assert rows[0]['source'] == dict(scheme='http', host='engine', port=8000, path='/metrics')
    assert [r['counter_epoch'] for r in rows] == [0, 1, 2, 3]
    assert rows[1]['counter_resets']
    assert rows[2]['exporter_identity_changed']
    assert rows[3]['disappeared_counters'] and rows[3]['unavailable_metrics']
    for row in rows:
        assert row['sent_at'] <= row['received_at'] <= row['finished_at']
        for name in ('gateway', 'runner'):
            resource = row['resources'][name]
            assert resource['available'] and resource['pid'] == os.getpid()
            assert resource['rss_bytes'] > 0
            assert row['received_at'] <= resource['sample_start'] <= resource['sample_end'] <= row['finished_at']
    assert 'secret' not in json.dumps(rows)


@pytest.mark.parametrize('failure', ['transport', 'parse', 'http'])
def test_sample_failure_is_not_zero_engine_activity(tmp_path, failure):
    def serve(request):
        if failure == 'transport':
            raise httpx.ConnectError('unavailable')
        return httpx.Response(503 if failure == 'http' else 200, text='invalid metric text {')
    sampler = ObservationSampler(tmp_path, 'http://engine', os.getpid(), 'test')
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        row = sampler.sample(client)
    assert row['error_type'] is not None
    assert set(row['unavailable_metrics']) == set(REQUIRED_METRICS)
    assert row['resources']['gateway']['available']
    assert row['received_at'] <= row['finished_at']


def test_worker_persists_final_sample_and_explicit_status(tmp_path, monkeypatch):
    real_client = httpx.Client
    sampler = ObservationSampler(tmp_path, 'http://engine', os.getpid(), 'test', interval=.01)
    threads = []
    def serve(request):
        threads.append(threading.get_ident())
        if len(threads) == 1:
            sampler.stop.set()
        return httpx.Response(200, text=metrics_text())
    monkeypatch.setattr('diagnostic_sampling.httpx.Client',
                        lambda **kw: real_client(transport=httpx.MockTransport(serve), **kw))
    worker = threading.Thread(target=sampler.run)
    worker.start()
    assert sampler.ready.wait(5)
    worker.join(5)
    assert not worker.is_alive()
    assert all(t != threading.get_ident() for t in threads)
    rows = [json.loads(line) for line in (tmp_path / 'engine_observations.jsonl').read_text().splitlines()]
    status = json.loads((tmp_path / 'engine_observations_status.json').read_text())
    assert len(rows) == status['samples'] == 2
    assert status['status'] == 'closed'
    assert status['sample_errors'] == status['write_errors'] == 0
    assert rows[-1]['finished_at'] <= status['closed_at']


@pytest.mark.parametrize('limit', ['size', 'time'])
def test_metrics_body_limits_are_unavailable_not_truncated_success(tmp_path, monkeypatch, limit):
    monkeypatch.setattr('diagnostic_sampling.' +
                        ('MAX_METRICS_BYTES' if limit == 'size' else 'METRICS_BODY_TIMEOUT_SECONDS'), 0)
    sampler = ObservationSampler(tmp_path, 'http://engine', os.getpid(), 'test')
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=metrics_text()))) as client:
        row = sampler.sample(client)
    assert row['error_type'] == 'ValueError'
    assert row['error_reason'] == f'metrics_body_{limit}_limit'
    assert row['raw_metrics'] is None
    assert set(row['unavailable_metrics']) == set(REQUIRED_METRICS)


def test_worker_write_failure_is_persistent_and_raised(tmp_path):
    (tmp_path / 'engine_observations.jsonl').mkdir()
    sampler = ObservationSampler(tmp_path, 'http://engine', os.getpid(), 'test')
    with pytest.raises(OSError):
        sampler.run()
    status = json.loads((tmp_path / 'engine_observations_status.json').read_text())
    assert status['status'] == 'failed' and status['write_errors'] == 1
    assert sampler.ready.is_set()


def test_parent_environment_cannot_enable_omitted_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setenv('GATEWAY_DIAGNOSTICS', '1')
    args = SimpleNamespace(budget=96, vllm_url='http://engine', gateway_diagnostics=False)
    selected = gateway_environment(args, {'model': 'mock'}, Path(tmp_path), 'not-logged')
    assert selected['GATEWAY_DIAGNOSTICS'] == '0'
    args.gateway_diagnostics = True
    assert gateway_environment(args, {'model': 'mock'}, Path(tmp_path), 'not-logged')['GATEWAY_DIAGNOSTICS'] == '1'
