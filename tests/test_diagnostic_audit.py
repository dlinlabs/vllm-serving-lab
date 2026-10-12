"""CPU mock collection -> persisted protocol -> audit -> offline joins."""
import asyncio
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
import uuid

import httpx
import pytest

from analyze_trace_dynamics import analyze_run, hashes
from analyze_trace_pilot import analyze
from diagnostic_sampling import ObservationSampler
from run_budget_pilot import save, set_budget
from run_trace_pilot import poll_state, prepare_replay, replay
from sustained_load import consume_request, WorkloadSpec
from telemetry import DIAGNOSTICS_SCHEMA, INVESTIGATION_PROTOCOL
from test_diagnostic_sampling import metrics_text
from test_dynamic_budget import gateway
from test_trace_replay import sse, trace
from trace_workload import sha256_file


@pytest.fixture
def collected_bundle(gateway, monkeypatch, tmp_path):
    async def collect():
        root = Path(__file__).resolve().parents[1]
        real_async, real_sync = httpx.AsyncClient, httpx.Client
        monkeypatch.setattr(gateway, 'GATEWAY_DIAGNOSTICS', True)
        monkeypatch.setattr(gateway, 'BENCHMARK_REQUEST_IDS', True)
        monkeypatch.setattr(gateway, 'TELEMETRY_JSONL', str(tmp_path / 'gateway.jsonl'))
        monkeypatch.setattr(gateway, 'TELEMETRY_STATUS_JSON', str(tmp_path / 'telemetry_status.json'))
        monkeypatch.setattr(gateway, 'estimate_request_cost', lambda payload: (1., 8, 128, False))
        monkeypatch.setattr(gateway.httpx, 'AsyncClient', lambda: real_async(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=sse(r.headers['x-request-id']).content,
                                    headers={'content-type': 'text/event-stream'}))))
        monkeypatch.setattr('diagnostic_sampling.httpx.Client', lambda **kw: real_sync(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text=metrics_text())), **kw))
        data = trace()
        save(tmp_path / 'trace.json', data)
        fingerprints = {p.name: sha256_file(p) for p in root.glob('*.py')}
        fingerprints['requirements-v1.txt'] = sha256_file(root / 'requirements-v1.txt')
        manifest = dict(schema_version=1, status='collected', trace_sha256=sha256_file(tmp_path / 'trace.json'),
            config=dict(warmup_requests=1, budget=96, ttft_slo=1., max_scheduling_lag=1.,
                        cache_policy='unknown', gateway_diagnostics=True, engine_observations=False),
            investigation_protocol=INVESTIGATION_PROTOCOL, diagnostics_schema_version=DIAGNOSTICS_SCHEMA,
            gateway_diagnostics_requested=True, gateway_diagnostics_observed=True, code_sha256=fingerprints,
            sampler=dict(enabled=True, interval_seconds=1.), host=socket.gethostname(), runner_process_id=os.getpid(),
            gateway_launch=dict(process_id=os.getpid()),
            packages={p: importlib.metadata.version(p) for p in
                      ('httpx', 'httpcore', 'fastapi', 'uvicorn', 'psutil', 'prometheus-client')})
        async with gateway.lifespan(gateway.app):
            manifest['run_id'] = gateway.telemetry.run_id
            manifest['initial_metrics'] = await gateway.snapshot_metrics()
            sampler = ObservationSampler(tmp_path, 'http://mock-engine', os.getpid(), manifest['run_id'])
            sampling = asyncio.create_task(asyncio.to_thread(sampler.run))
            try:
                assert await asyncio.to_thread(sampler.ready.wait, 5)
                rows = prepare_replay(data, manifest['run_id'])
                async with real_async(transport=httpx.ASGITransport(app=gateway.app), trust_env=False) as client:
                    manifest['warmup_start'] = time.monotonic()
                    warmup = await consume_request(client, time.monotonic(), data['model'],
                        WorkloadSpec('trace', 'a b c d', 128), client_request_id=uuid.uuid4().hex)
                    assert warmup.success
                    manifest['post_warmup_metrics'] = await gateway.snapshot_metrics()
                    manifest['warmup_end'] = time.monotonic()
                    before = time.monotonic()
                    applied = await set_budget(client, 'test-key', 96)
                    acknowledged = time.monotonic()
                    start = time.monotonic() + .02
                    manifest.update(measurement_start=start, measurement_end=start + data['duration_seconds'])
                    changes = [dict(target=start, sent=before, acknowledged=acknowledged, **applied)]
                    stop, observations = asyncio.Event(), []
                    polling = asyncio.create_task(poll_state(client, stop, observations))
                    try:
                        await replay(client, data, start, manifest['run_id'], 1, 10, rows)
                    finally:
                        stop.set()
                        await polling
                    manifest['final_metrics'] = await gateway.snapshot_metrics()
            finally:
                sampler.stop.set()
                await sampling
        save(tmp_path / 'client.json', dict(requests=rows))
        save(tmp_path / 'warmup.json', dict(requests=[asdict(warmup)]))
        save(tmp_path / 'budget_changes.json', changes)
        save(tmp_path / 'state_observations.json', observations)
        save(tmp_path / 'manifest.json', manifest)
        report = analyze(tmp_path, plot=False)
        assert report['valid'] and report['load_fidelity']['valid'], report
        assert report['diagnostic_integrity']['valid'], report['diagnostic_integrity']
    asyncio.run(collect())
    return tmp_path


def test_collected_protocol_round_trips_read_only(collected_bundle):
    before = hashes(collected_bundle)
    summary, rows, _, _ = analyze_run(collected_bundle, .3)
    assert summary['original_report_reproduced'] and hashes(collected_bundle) == before
    assert len(rows) == 3 and summary['accounting']['outcomes'] == {'successful_within_ttft': 3}
    assert all(r['diagnostics']['summaries'] == 1 for r in rows)
    assert summary['diagnostics']['gc_coverage_complete']
    assert summary['diagnostics']['engine_observations']
    assert summary['diagnostics']['request_linked_prefill'] == 'unavailable'


@pytest.mark.parametrize('damage', ['missing_summary', 'duplicate_summary', 'unfinished_phase',
                                  'cpu_thread', 'cpu_coverage', 'gc_loss', 'missing_engine', 'mode', 'counter_epoch'])
def test_diagnostic_failure_does_not_rewrite_client_integrity(collected_bundle, damage):
    path = collected_bundle / 'gateway.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == 'missing_summary':
        records.remove(next(r for r in records if r['event'] == 'request_diagnostics'))
    elif damage == 'duplicate_summary':
        index = next(i for i, r in enumerate(records) if r['event'] == 'request_diagnostics')
        records.insert(index, dict(records[index]))
    elif damage == 'unfinished_phase':
        row = next(r for r in records if r['event'] == 'request_diagnostics')
        row['phases']['client_close']['end'] = None
    elif damage == 'cpu_thread':
        next(r for r in records if r['event'] == 'diagnostic_cpu')['loop_thread_id'] = -1
    elif damage == 'cpu_coverage':
        records.remove(next(r for r in records if r['event'] == 'diagnostic_cpu'))
        status_path = collected_bundle / 'telemetry_status.json'
        status = json.loads(status_path.read_text())
        status['diagnostics']['cpu_samples'] -= 1
        next(r for r in records if r['event'] == 'diagnostics_stopped')['health'] = status['diagnostics']
        save(status_path, status)
    elif damage == 'gc_loss':
        status_path = collected_bundle / 'telemetry_status.json'
        status = json.loads(status_path.read_text())
        status['diagnostics']['lost_callbacks'] = 1
        next(r for r in records if r['event'] == 'diagnostics_stopped')['health'] = status['diagnostics']
        save(status_path, status)
    elif damage == 'missing_engine':
        (collected_bundle / 'engine_observations.jsonl').write_text('')
    elif damage == 'mode':
        manifest_path = collected_bundle / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['gateway_diagnostics_observed'] = False
        save(manifest_path, manifest)
    elif damage == 'counter_epoch':
        engine_path = collected_bundle / 'engine_observations.jsonl'
        engine = [json.loads(line) for line in engine_path.read_text().splitlines()]
        engine[-1]['counter_epoch'] = 20
        engine_path.write_text('\n'.join(json.dumps(row) for row in engine) + '\n')
    path.write_text('\n'.join(json.dumps(row) for row in records) + '\n')
    report = analyze(collected_bundle, plot=False)
    assert report['valid'] and report['load_fidelity']['valid'], report
    assert not report['diagnostic_integrity']['valid'], report


def test_missing_core_data_retains_failed_diagnostic_verdict(collected_bundle):
    (collected_bundle / 'client.json').write_text('{')
    report = analyze(collected_bundle, plot=False)
    assert not report['valid']
    assert not report['diagnostic_integrity']['valid']
    assert not report['diagnostic_integrity']['gc_coverage_complete']


def test_runner_does_not_persist_backend_credentials(tmp_path, monkeypatch):
    import run_trace_pilot
    source, output = tmp_path / 'trace.json', tmp_path / 'run'
    save(source, trace())
    url = 'http://user-secret:password-secret@127.0.0.1:8000?token=query-secret#fragment-secret'
    observed_urls = []
    async def fail_run(args, data, directory, manifest):
        observed_urls.append(args.vllm_url)
        response = httpx.Response(502, request=httpx.Request('GET', url))
        raise httpx.HTTPStatusError(f'Failure at {url}', request=response.request, response=response)
    monkeypatch.setattr(run_trace_pilot, 'run', fail_run)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: object())))
    monkeypatch.setattr(run_trace_pilot, 'verify_prompts', lambda *a: None)
    monkeypatch.setattr(run_trace_pilot.signal, 'signal', lambda *a: None)
    real_run = run_trace_pilot.subprocess.run
    def cpu_only_run(command, **kwargs):
        if command[0] == 'nvidia-smi':
            raise FileNotFoundError('No GPU probe in this CPU-only unit test')
        return real_run(command, **kwargs)
    monkeypatch.setattr(run_trace_pilot.subprocess, 'run', cpu_only_run)
    monkeypatch.setattr(sys, 'argv', ['run_trace_pilot.py', '--trace', str(source),
                                     '--output-dir', str(output), '--ttft-slo', '.5', '--vllm-url', url])
    with pytest.raises(SystemExit) as exc:
        run_trace_pilot.main()
    assert exc.value.code == 1 and observed_urls == [url]
    raw = (output / 'manifest.json').read_text()
    assert 'secret' not in raw
    manifest = json.loads(raw)
    assert manifest['config']['vllm_url'] == 'http://127.0.0.1:8000'
    assert manifest['status'] == 'failed' and manifest['error_http_status'] == 502
