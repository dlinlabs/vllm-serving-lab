import asyncio
import csv
from datetime import datetime, timezone
import json
import time
import uuid

import httpx
import pytest

from prepare_trace import prepare
from run_trace_pilot import prepare_replay, replay
from trace_workload import synthetic_prompt, validate_trace
from telemetry import Telemetry
from sustained_load import consume_request, WORKLOADS
from test_dynamic_budget import gateway


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(text.strip())]
    def decode(self, ids, **kwargs):
        return ''.join(' ' + chr(i) for i in ids)
    def apply_chat_template(self, messages, **kwargs):
        return {'input_ids': [1] * (4 + len(messages[0]['content'].split()))}


def trace(arrivals=(0, .01, .02), duration=.03):
    return dict(schema_version=1, model='mock-model', max_model_len=8192, output_cap=128,
                duration_seconds=duration, requests=[dict(trace_id=str(i), arrival_seconds=at,
                    input_tokens=8, max_tokens=128, prompt='a b c d') for i, at in enumerate(arrivals)])


def sse(identifier, input_count=8):
    return httpx.Response(200, headers={'x-request-id': identifier}, text=(
        'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        f'data: {{"choices":[],"usage":{{"prompt_tokens":{input_count},"completion_tokens":1,"total_tokens":{input_count + 1}}}}}\n\n'
        'data: [DONE]\n\n'))


def test_prepare_preserves_arrivals_thins_deterministically_and_audits_exclusions(tmp_path):
    source = tmp_path / 'source.csv'
    with source.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['TIMESTAMP', 'ContextTokens', 'GeneratedTokens'])
        for at, count in [(0, 8), (.1, 8), (.1, 8190), (.2, 2), (1, 8)]:
            writer.writerow([datetime.fromtimestamp(at, timezone.utc).isoformat(), count, 5000])
    params = (source, '1970-01-01T00:00:00Z', .5, 1, 7, 128, Tokenizer())
    data, audit = prepare(*params)
    assert [r['arrival_seconds'] for r in data['requests']] == [0, .1]
    assert data['provenance']['selection_counts'] == dict(in_window=4, sampled_out=0,
                                                        context_overflow=1, context_too_short=1, retained=2)
    assert all(r['max_tokens'] == 128 and 'generated_tokens' not in r for r in data['requests'])
    assert all(r['generated_tokens'] == 5000 for r in audit)
    assert prepare(*params) == (data, audit)
    with pytest.raises(ValueError, match='complete requested window'):
        prepare(source, params[1], 2, 1, 7, 128, Tokenizer())
    with pytest.raises(ValueError, match='starts before'):
        prepare(source, '1969-12-31T23:59:59Z', .5, 1, 7, 128, Tokenizer())


def test_exact_chat_template_length_and_trace_guards():
    for length in (4, 8, 100):
        prompt = synthetic_prompt(Tokenizer(), length, 'key')
        assert len(Tokenizer().apply_chat_template([{'content': prompt}])['input_ids']) == length
    data = trace()
    assert validate_trace(data) is data
    data['requests'][0]['generated_tokens'] = 3
    with pytest.raises(ValueError, match='Unexpected replay request'):
        validate_trace(data)
    data = trace((.02, .01))
    with pytest.raises(ValueError, match='sorted'):
        validate_trace(data)
    data = trace()
    data['requests'][0]['input_tokens'] = 8192
    with pytest.raises(ValueError, match='context length'):
        validate_trace(data)


def test_open_loop_overlapping_requests_and_pre_header_failure_accounting():
    async def scenario():
        calls, rows = [], []
        async def handler(request):
            identifier = request.headers['x-benchmark-request-id']
            calls.append((time.monotonic(), identifier, json.loads(request.content)))
            if len(calls) == 2:
                raise httpx.ConnectError('refused')
            await asyncio.sleep(.05)
            return sse(identifier)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            await replay(client, trace(), time.monotonic(), uuid.uuid4().hex, 1, 10, rows)
        assert len(rows) == len(calls) == 3
        assert calls[-1][0] - calls[0][0] < .05
        assert len(set(r['client_request_id'] for r in rows)) == 3
        assert rows[1]['request_id'] is None and rows[1]['client_request_id']
        assert rows[1]['error_type'] == 'ConnectError'
        assert rows[0]['success'] and rows[2]['success']
        assert all(c[2]['ignore_eos'] is False and c[2]['max_tokens'] == 128 for c in calls)
    asyncio.run(scenario())


def test_preassigned_requests_are_reused_without_allocation_during_replay(monkeypatch):
    async def scenario():
        data, run_id = trace(), uuid.uuid4().hex
        rows = prepare_replay(data, run_id)
        identifiers = [r['client_request_id'] for r in rows]
        assert all(r['scheduled_at'] is None and r['error_type'] == 'NotLaunched' for r in rows)
        def forbidden(*args):
            raise AssertionError('Replay must not allocate IDs after the clock origin is chosen')
        monkeypatch.setattr('run_trace_pilot.prepare_replay', forbidden)
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: sse(r.headers['x-benchmark-request-id'])), trust_env=False) as client:
            start = time.monotonic() + .02
            await replay(client, data, start, run_id, 1, 10, rows)
        assert all(r['success'] for r in rows)
        assert [r['client_request_id'] for r in rows] == identifiers
        assert [r['scheduled_at'] - start for r in rows] == pytest.approx([0, .01, .02])
    asyncio.run(scenario())


def test_drain_timeout_retains_started_requests_and_capacity_does_not_silently_skip():
    async def scenario():
        async def hanging(request):
            await asyncio.Event().wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(hanging), trust_env=False) as client:
            rows = []
            await replay(client, trace(), time.monotonic(), uuid.uuid4().hex, .01, 10, rows)
            assert len(rows) == 3 and all(r['error_type'] == 'CancelledError' for r in rows)
            assert all(r['started_at'] is not None and r['finished_at'] is not None for r in rows)
            rows = []
            with pytest.raises(RuntimeError, match='pending limit'):
                await replay(client, trace((0, 0, 0)), time.monotonic(), uuid.uuid4().hex, .01, 1, rows)
            assert len(rows) == 3
            assert sum(r['error_type'] == 'NotLaunched' for r in rows) == 2
    asyncio.run(scenario())


def test_loop_monitor_captures_blocking_without_catch_up_storm():
    async def scenario():
        telemetry, stop = Telemetry(), asyncio.Event()
        telemetry.enabled = True
        task = asyncio.create_task(telemetry.monitor_loop(stop, .01))
        await asyncio.sleep(.005)
        time.sleep(.06)  # Deliberate blocking only in this test.
        await asyncio.sleep(.015)
        stop.set()
        await task
        samples = [r for r in telemetry.records if r['event'] == 'event_loop_lag']
        assert max(r['lag_seconds'] for r in samples) > .04
        assert len(samples) <= 3
    asyncio.run(scenario())


def test_gateway_benchmark_ids_rejection_features_and_application_time(gateway, monkeypatch):
    async def scenario():
        monkeypatch.setattr(gateway, 'BENCHMARK_REQUEST_IDS', True)
        gateway.telemetry.enabled = True
        gateway.state['admission_budget'] = .1
        identifier = uuid.uuid4().hex
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url='http://test', trust_env=False) as client:
            response = await client.post('/v1/chat/completions', json={}, headers={'x-benchmark-request-id': identifier})
            assert response.status_code == 503 and response.headers['x-request-id'] == identifier
            assert (await client.post('/v1/chat/completions', json={}, headers={'x-benchmark-request-id': identifier})).status_code == 409
            assert (await client.post('/v1/chat/completions', json={}, headers={'x-benchmark-request-id': 'bad'})).status_code == 400
            before = time.monotonic()
            response = await client.post('/admin/budget', json={'budget': 96}, headers={'x-admin-api-key': 'test-key'})
            assert before <= response.json()['applied_at'] <= time.monotonic()
        rejected = next(r for r in gateway.telemetry.records if r['event'] == 'rejected')
        assert rejected['estimated_input_tokens'] > 0 and rejected['request_cost'] > 0
        assert gateway.telemetry.totals['received'] == 1
    asyncio.run(scenario())


def test_upstream_503_is_failure_not_admission_rejection():
    async def scenario():
        for admission in (True, False):
            payload = {'detail': 'request rejected: admission capacity exceeded', 'estimated_request_cost': 1} if admission else {'detail': 'backend unavailable'}
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503, json=payload)), trust_env=False) as client:
                result = await consume_request(client, 0, 'model', WORKLOADS[0])
            assert result.rejected == admission and not result.success
    asyncio.run(scenario())


@pytest.mark.parametrize('diagnostics,external,mode', [
    (False, False, 'thread'), (True, True, 'thread'), (True, True, 'process'),
    (False, True, 'thread'), (False, True, 'process'),
])
def test_full_trace_runner_and_adversarial_analysis(tmp_path, diagnostics, external, mode):
    """Real subprocesses, HTTP, SSE, shutdown and plots; explicitly not a GPU benchmark."""
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    from analyze_trace_pilot import analyze
    from analyze_trace_dynamics import analyze_run, hashes
    from test_diagnostic_sampling import metrics_text
    from trace_workload import sha256_file
    root = Path(__file__).resolve().parents[1]
    for port in (8000, 8080):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError:
                pytest.skip(f'Integration port {port} is in use')
    (tmp_path / 'transformers.py').write_text('''class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs): return cls()
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [1] * (4 + len(messages[0]['content'].split()))}
''')
    (tmp_path / 'mock_vllm.py').write_text('''import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
app = FastAPI()
@app.get('/v1/models')
def models(): return {"data": [{"id": "mock-model", "max_model_len": 8192}]}
@app.post('/v1/chat/completions')
async def completion(request: Request):
    payload = await request.json()
    if 'unavailable' in payload['messages'][0]['content']:
        return JSONResponse({'detail': 'backend unavailable'}, status_code=503)
    count = 4 + len(payload['messages'][0]['content'].split())
    async def stream():
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\\n\\n'
        yield ('data: ' + json.dumps({'choices': [], 'usage': {'prompt_tokens': count, 'completion_tokens': 1, 'total_tokens': count+1}}) + '\\n\\n').encode()
        yield b'data: [DONE]\\n\\n'
    return StreamingResponse(stream(), media_type='text/event-stream')
''')
    with (tmp_path / 'mock_vllm.py').open('a') as handle:
        handle.write("\nfrom fastapi.responses import Response\n@app.get('/metrics')\n"
                     f"def metrics(): return Response({metrics_text()!r}, media_type='text/plain')\n")
    env = dict({k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')},
               GATEWAY_DIAGNOSTICS='1',
               PYTHONPATH=str(tmp_path) + os.pathsep + str(root))
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'mock_vllm:app', '--port', '8000'],
                               cwd=tmp_path, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                httpx.get('http://127.0.0.1:8000/v1/models', trust_env=False).raise_for_status()
                break
            except httpx.HTTPError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Mock startup timed out')
                time.sleep(.05)
        for name, budget, unavailable in [('success', 96, False), ('rejected', .75, False), ('failed', 96, True)]:
            data = trace((0, .1, .2), .3)
            if unavailable:
                for row in data['requests']:
                    row['prompt'] = 'unavailable b c d'
            source = tmp_path / f'{name}-trace.json'
            source.write_text(json.dumps(data))
            output = tmp_path / name
            result = subprocess.run([sys.executable, 'run_trace_pilot.py', '--trace', str(source),
                         '--output-dir', str(output), '--budget', str(budget), '--ttft-slo', '1',
                         '--warmup-requests', '1', '--drain-timeout', '2', '--max-scheduling-lag', '1',
                         '--sampler-mode', mode] +
                         (['--gateway-diagnostics'] if diagnostics else ['--engine-observations'] if external else []),
                         cwd=root, env=env, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
            report = json.loads((output / 'report.json').read_text())
            assert report['valid'] and report['performance'][name] == 3
            assert report['performance']['attempted'] == 3
            assert report['diagnostic_integrity']['valid'], report['diagnostic_integrity']
            assert report['diagnostic_integrity']['gateway_mode'] == ('enabled' if diagnostics else 'disabled')
            assert report['diagnostic_integrity']['external_mode'] == ('enabled' if external else 'disabled')
            manifest = json.loads((output / 'manifest.json').read_text())
            assert manifest['sampler']['mode'] == mode
            if external:
                worker = manifest['sampler']['worker']
                assert worker['stopped'] and not worker['forced_termination']
                assert (worker['process_id'] == manifest['runner_process_id']) is (mode == 'thread')
            assert (output / 'timeseries.png').stat().st_size > 1000
            before = hashes(output)
            summary, joined, _, _ = analyze_run(output, .3)
            assert summary['original_report_reproduced'] and hashes(output) == before
            assert len(joined) == summary['accounting']['ledger_rows'] == 3
            assert all((r['diagnostics']['phases'] is not None) == diagnostics for r in joined)
            if diagnostics:
                assert summary['diagnostics']['gc_coverage_complete']
            if external:
                assert summary['diagnostics']['engine_observations']
                assert summary['cpu_analysis']['sampler_mode'] == mode
                engine_path = output / 'engine_observations.jsonl'
                raw_engine = engine_path.read_text()
                engine_path.write_text('')
                damaged = analyze(output, plot=False)
                assert damaged['valid'] and not damaged['diagnostic_integrity']['valid']
                engine_path.write_text(raw_engine)
                analyze(output, plot=False)
        output = tmp_path / 'success'
        report = analyze(output, plot=False)
        assert report['performance']['slo_goodput_per_arrival_second'] == 10
        client_path = output / 'client.json'
        raw_clients = client_path.read_text()
        # Internally consistent client timestamps must still respect gateway causality.
        records = [json.loads(line) for line in (output / 'gateway.jsonl').read_text().splitlines()]
        clients = json.loads(raw_clients)
        first_row = clients['requests'][0]
        gateway_first = next(r['monotonic'] for r in records if r['event'] == 'first_content' and
                             r['request_id'] == first_row['client_request_id'])
        first_row['first_content_at'] = gateway_first - .0001
        first_row['ttft'] = first_row['first_content_at'] - first_row['started_at']
        client_path.write_text(json.dumps(clients))
        assert 'Client first content precedes gateway first content' in analyze(output, plot=False)['issues']
        clients = json.loads(raw_clients)
        first_row = clients['requests'][0]
        gateway_received = next(r['monotonic'] for r in records if r['event'] == 'received' and
                                r['request_id'] == first_row['client_request_id'])
        first_row['started_at'] = gateway_received + .0001
        first_row['ttft'] = first_row['first_content_at'] - first_row['started_at']
        first_row['end_to_end'] = first_row['finished_at'] - first_row['started_at']
        client_path.write_text(json.dumps(clients))
        assert 'Gateway received precedes client start' in analyze(output, plot=False)['issues']
        client_path.write_text(raw_clients)
        clients = json.loads(raw_clients)
        clients['requests'][0]['input_tokens'] += 1
        client_path.write_text(json.dumps(clients))
        assert 'Actual prompt tokens differ from trace' in analyze(output, plot=False)['issues']
        client_path.write_text(raw_clients)
        gateway_path = output / 'gateway.jsonl'
        raw_gateway = gateway_path.read_text()
        records = [json.loads(line) for line in raw_gateway.splitlines()]
        # Losing an entire request must be caught even if all remaining lifecycles look correct.
        identifier = json.loads(raw_clients)['requests'][0]['client_request_id']
        gateway_path.write_text('\n'.join(json.dumps(r) for r in records if r.get('request_id') != identifier))
        assert 'Final cumulative counters disagree with lifecycle records' in analyze(output, plot=False)['issues']
        gateway_path.write_text(raw_gateway)
        manifest_path = output / 'manifest.json'
        original = manifest_path.read_text()
        manifest = json.loads(original)
        manifest['config']['max_scheduling_lag'] = 1e-9
        manifest_path.write_text(json.dumps(manifest))
        report = analyze(output, plot=False)
        assert report['valid'] and not report['load_fidelity']['valid']
        manifest_path.write_text(original)
        shutdown_path = output / 'telemetry_status.json'
        raw_shutdown = shutdown_path.read_text()
        shutdown = json.loads(raw_shutdown)
        shutdown['write_errors'] = 1
        shutdown_path.write_text(json.dumps(shutdown))
        assert 'Telemetry shutdown did not confirm successful persistence' in analyze(output, plot=False)['issues']
        shutdown_path.write_text(raw_shutdown)
        # Change a genuine client-delivery outcome: upstream completion is not client success.
        clients = json.loads(raw_clients)
        clients['requests'][0].update(success=False, request_id=None, status=None,
                                     error_type='ReadTimeout', error_phase='before_response_headers')
        client_path.write_text(json.dumps(clients))
        report = analyze(output, plot=False)
        assert report['valid'] and report['performance']['failed'] == 1
        assert report['performance']['success'] == 2
        client_path.write_text(raw_clients)
        # A genuine connection failure before the handler gets an ID is a complete failure record.
        records = [r for r in records if r.get('request_id') != identifier]
        for record in records:
            if record['event'] == 'snapshot':
                # Only final totals need adjustment to reflect this counterfactual accounting fixture.
                for event in ('received', 'accepted', 'completed'):
                    record['totals'][event] = max(0, record['totals'].get(event, 0) - 1)
        gateway_path.write_text('\n'.join(json.dumps(r) for r in records))
        clients['requests'][0].update(error_type='ConnectError', first_content_at=None, ttft=None,
                                     input_tokens=None, output_tokens=None, stream_complete=False)
        client_path.write_text(json.dumps(clients))
        report = analyze(output, plot=False)
        assert report['valid'] and report['failed_without_gateway_record'] == 1
        gateway_path.write_text(raw_gateway)
        client_path.write_text(raw_clients)
        # Tampering with input must invalidate results even when request count stays the same.
        trace_path = output / 'trace.json'
        trace_path.write_text(trace_path.read_text() + ' ')
        assert 'Replay trace hash mismatch' in analyze(output, plot=False)['issues']
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_prefix_cache_option_keeps_setup_default():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('trace_setup', Path(__file__).parents[1] / 'scripts' / 'setup_v1.py')
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    python = Path('/example/.venv/bin/python')
    default = setup.command(python)
    assert '--no-enable-prefix-caching' not in default
    assert setup.command(python, True) == default + ['--no-enable-prefix-caching']


def test_observations_do_not_consume_snapshot_interval_counters():
    telemetry = Telemetry()
    identifier = telemetry.received()
    telemetry.admitted(identifier)
    for _ in range(5):
        assert telemetry.observed_state()['awaiting_first_content'] == 1
    assert telemetry.sample({})['counts']['received'] == 1
    telemetry.first_content(identifier)
    assert telemetry.observed_state()['streaming_requests'] == 1
    assert telemetry.sample({})['counts']['received'] == 0


def test_shutdown_does_not_hide_final_flush_failure(gateway, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(gateway, 'TELEMETRY_JSONL', str(tmp_path / 'gateway.jsonl'))
    status = tmp_path / 'telemetry_status.json'
    monkeypatch.setattr(gateway, 'TELEMETRY_STATUS_JSON', str(status))
    async def failed_flush(path):
        gateway.telemetry.write_errors += 1
        gateway.telemetry.records.clear()
    monkeypatch.setattr(gateway.telemetry, 'flush', failed_flush)
    with pytest.raises(RuntimeError, match='telemetry was lost'):
        with TestClient(gateway.app):
            pass
    assert json.loads(status.read_text())['write_errors'] == 1


@pytest.mark.parametrize('body,error', [
    ('data: [DONE]\n\n', 'EmptyContentStream'),
    ('data: broken-json\n\ndata: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n', 'InvalidSSE'),
    ('data: {"choices":[{"delta":{"content":{"bad":"object"}}}]}\n\ndata: [DONE]\n\n', 'InvalidSSE'),
    ('data: {"choices":[{"delta":{"content":"hi"}}],"usage":[]}\n\ndata: [DONE]\n\n', 'InvalidSSE'),
    ('data: {"choices":[{"delta":{"content":"hi"}},42]}\n\ndata: [DONE]\n\n', 'InvalidSSE'),
    ('data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n', 'InvalidSSE'),
])
def test_empty_and_corrupt_streams_are_failures(body, error):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)), trust_env=False) as client:
            result = await consume_request(client, 0, 'model', WORKLOADS[0])
        assert not result.success and not result.rejected
        assert result.error_type == error and result.finished_at is not None
    asyncio.run(scenario())
    from telemetry import SSEObserver
    observer = SSEObserver()
    observer.feed(body.encode())
    if error == 'InvalidSSE':
        assert observer.invalid
