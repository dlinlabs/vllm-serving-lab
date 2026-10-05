import asyncio
import importlib.util
from pathlib import Path
import sys

import httpx
import pytest
from starlette.requests import Request


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv('ADMISSION_MODE', 'cost_aware')
    monkeypatch.setenv('MAX_ADMITTED_COST', '80')
    monkeypatch.setenv('ADMIN_API_KEY', 'test-key')
    spec = importlib.util.spec_from_file_location('budget_test_gateway', Path(__file__).parents[1] / 'gateway.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_api_validation_and_metrics(gateway):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url='http://test') as client:
            headers = {'x-admin-api-key': 'test-key'}
            assert (await client.post('/admin/budget', json={'budget': 96})).status_code == 401
            for raw in ['{}', '[]', '{', '{"budget":true}', '{"budget":"96"}', '{"budget":0}', '{"budget":-1}', '{"budget":NaN}', '{"budget":Infinity}', '{"budget":1e999}']:
                assert (await client.post('/admin/budget', content=raw, headers=headers)).status_code == 422
            assert gateway.state['admission_budget'] == 80
            assert gateway.state['budget_version'] == 0
            response = await client.post('/admin/budget', json={'budget': 96}, headers=headers)
            assert response.json() == {'previous_budget': 80, 'budget': 96, 'current_admitted_cost': 0, 'budget_version': 1}
            metrics = (await client.get('/metrics')).json()
            assert metrics['max_admitted_cost'] == metrics['admission_budget'] == 96
            assert metrics['initial_admission_budget'] == 80
            gateway.ADMISSION_MODE = 'baseline'
            assert (await client.post('/admin/budget', json={'budget': 90}, headers=headers)).status_code == 409
            gateway.ADMIN_API_KEY = ''
            assert (await client.post('/admin/budget', json={'budget': 90}, headers=headers)).status_code == 503
    asyncio.run(scenario())


def test_lowering_and_concurrent_admission(gateway):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url='http://test') as client:
            async def update(value):
                return await client.post('/admin/budget', json={'budget': value}, headers={'x-admin-api-key': 'test-key'})
            assert (await gateway.begin_request(75, False)).accepted
            await update(50)
            rejected = await gateway.begin_request(1, False)
            assert not rejected.accepted and rejected.budget == 50 and rejected.budget_version == 1
            assert gateway.state['current_admitted_cost'] == 75
            await gateway.finish_request(75)
            await update(10)
            decisions = await asyncio.gather(*(gateway.begin_request(1, False) for _ in range(100)))
            assert sum(d.accepted for d in decisions) == 10
            await asyncio.gather(*(gateway.finish_request(1) for d in decisions if d.accepted))
            await asyncio.gather(update(20), *(gateway.begin_request(1, False) for _ in range(100)))
            assert gateway.state['current_admitted_cost'] <= 20
            assert rejected.budget == 50  # Snapshot survives later updates.
    asyncio.run(scenario())


@pytest.mark.parametrize('outcome', ['complete', 'error', 'cancel', 'connect_error'])
def test_upstream_lifecycle_releases_cost(gateway, monkeypatch, outcome):
    async def scenario():
        first_chunk = asyncio.Event()
        resume = asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                first_chunk.set()
                await resume.wait()
                if outcome == 'error':
                    raise httpx.ReadError('mock failure')
                yield b'data: [DONE]\n\n'

        async def upstream(request):
            if outcome == 'connect_error':
                raise httpx.ConnectError('mock connection failure')
            return httpx.Response(200, stream=Stream(), headers={'content-type': 'text/event-stream'})

        real_client = httpx.AsyncClient
        monkeypatch.setattr(gateway.httpx, 'AsyncClient', lambda: real_client(transport=httpx.MockTransport(upstream)))
        monkeypatch.setattr(gateway, 'estimate_request_cost', lambda payload: (75, 98, 128, False))
        async def receive():
            return {'type': 'http.request', 'body': b'{}', 'more_body': False}
        request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions', 'headers': []}, receive)
        response = await gateway.proxy_chat_completions(request)
        if outcome == 'connect_error':
            assert response.status_code == 502
        else:
            chunks = []
            async def consume():
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
            task = asyncio.create_task(consume())
            await first_chunk.wait()
            async with real_client(transport=httpx.ASGITransport(app=gateway.app), base_url='http://test') as client:
                result = await client.post('/admin/budget', json={'budget': 50}, headers={'x-admin-api-key': 'test-key'})
                assert result.json()['current_admitted_cost'] == 75
                rejected = await client.post('/v1/chat/completions', json={})
                assert rejected.status_code == 503
                assert rejected.json()['max_admitted_cost'] == 50
                assert rejected.json()['budget_version'] == 1
            assert not task.done()
            if outcome == 'cancel':
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                resume.set()
                if outcome == 'error':
                    with pytest.raises(httpx.ReadError):
                        await task
                else:
                    await task
                    assert chunks[-1] == b'data: [DONE]\n\n'
        assert gateway.state['current_admitted_cost'] == 0
        assert gateway.state['current_in_flight'] == 0
    asyncio.run(scenario())
