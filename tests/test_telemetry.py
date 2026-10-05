import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from test_dynamic_budget import gateway
from telemetry import SSEObserver, Telemetry
from sustained_load import consume_request, WORKLOADS


def test_fragmented_sse_and_empty_role():
    observer = SSEObserver()
    observer.feed(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n')
    assert not observer.has_content
    data = ('data: {"choices":[{"delta":{"content":"你好"}}]}\r\n\r\n' + 'data: [DONE]\n\n').encode()
    for byte in data:
        observer.feed(bytes([byte]))
    assert observer.has_content and observer.done and not observer.invalid


def test_sampler_counters_and_jsonl(tmp_path):
    async def scenario():
        t = Telemetry()
        t.enabled = True
        a, b, c = t.received(), t.received(), t.received()
        t.admitted(a)
        t.admitted(b)
        t.first_content(b)
        t.terminal(c, 'rejected')
        sample = t.sample({'admission_budget': 80, 'budget_version': 0})
        assert sample['awaiting_first_content'] == sample['streaming_requests'] == 1
        assert sample['counts']['received'] == 3
        assert sample['counts']['accepted'] == 2
        assert sample['counts']['rejected'] == 1
        assert sample['oldest_awaiting_first_seconds'] >= 0
        t.terminal(a, 'cancelled')
        t.terminal(b, 'completed')
        t.terminal(b, 'completed')
        sample = t.sample({})
        assert sample['active_requests'] == 0
        assert sample['counts']['completed'] == 1
        assert sample['counts']['received'] == 0
        path = tmp_path / 'trace.jsonl'
        await t.flush(path)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert all('timestamp' in r and 'monotonic' in r and 'run_id' in r for r in records)
        assert len([r for r in records if r['event'] == 'first_content']) == 1
        t.emit('test')
        await t.flush(tmp_path / 'missing' / 'trace.jsonl')
        assert t.write_errors == 1 and t.dropped == 1
    asyncio.run(scenario())


def test_lifespan_periodic_logging(gateway, monkeypatch, tmp_path):
    path = tmp_path / 'logs' / 'trace.jsonl'
    monkeypatch.setattr(gateway, 'TELEMETRY_JSONL', str(path))
    with TestClient(gateway.app) as client:
        response = client.post('/admin/budget', json={'budget': 96}, headers={'x-admin-api-key': 'test-key'})
        assert response.status_code == 200
        # Wait on the app event loop for one periodic sample.
        client.portal.call(asyncio.sleep, 1.1)
        assert any(json.loads(line)['event'] == 'snapshot' for line in path.read_text().splitlines())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(r['event'] == 'budget_updated' and r['budget'] == 96 for r in records)
    assert records[-1]['event'] == 'snapshot'
    assert records[-1]['admission_budget'] == 96


def test_harness_tracks_id_and_incomplete_stream():
    async def scenario():
        for complete in (True, False):
            def handler(request):
                data = 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                if complete:
                    data += 'data: [DONE]\n\n'
                return httpx.Response(200, text=data, headers={'x-request-id': 'request-1'})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await consume_request(client, 0, 'test-model', WORKLOADS[0])
            assert result.request_id == 'request-1'
            assert result.success == result.stream_complete == complete
            assert result.ttft is not None
    asyncio.run(scenario())


def test_gateway_truncated_stream_and_request_id(gateway, monkeypatch):
    async def scenario():
        real_client = httpx.AsyncClient
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        observed = []
        async def upstream(request):
            observed.append(request.headers['x-request-id'])
            assert 'x-admin-api-key' not in request.headers
            return httpx.Response(200, stream=Stream(), headers={'content-type': 'text/event-stream'})
        monkeypatch.setattr(gateway.httpx, 'AsyncClient', lambda: real_client(transport=httpx.MockTransport(upstream)))
        gateway.telemetry.enabled = True
        async with real_client(transport=httpx.ASGITransport(app=gateway.app), base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json={'stream': True}, headers={'x-admin-api-key': 'not-forwarded'})
        assert response.headers['x-request-id'] == observed[0]
        events = [r['event'] for r in gateway.telemetry.records]
        assert events == ['received', 'accepted', 'backend_sent', 'first_content', 'failed']
        assert not gateway.telemetry.active
        assert gateway.state['current_admitted_cost'] == 0
        assert gateway.state['failed_requests'] == 1
    asyncio.run(scenario())
