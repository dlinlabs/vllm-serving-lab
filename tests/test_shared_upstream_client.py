import asyncio
from types import SimpleNamespace

import httpx
import pytest

from test_dynamic_budget import gateway
from test_gateway_diagnostics import request, CONTENT, DONE
from run_trace_pilot import gateway_environment, replay_configuration


def test_full_shared_client_runner_and_audit(tmp_path, monkeypatch):
    """Exercise real gateway subprocesses, HTTP, SSE, errors and budget steps in shared mode."""
    import json
    import subprocess
    from pathlib import Path
    from test_trace_replay import test_full_trace_runner_and_adversarial_analysis
    original = subprocess.run
    checked = []

    def run(command, **kwargs):
        command = list(command)
        shared = 'run_trace_pilot.py' in command
        if shared:
            command.append('--shared-upstream-client')
        result = original(command, **kwargs)
        if shared:
            assert result.returncode == 0, result.stdout + result.stderr
            output = Path(command[command.index('--output-dir') + 1])
            manifest = json.loads((output / 'manifest.json').read_text())
            assert manifest['shared_upstream_client_observed'] is True
            assert manifest['config']['shared_upstream_client'] is True
            assert manifest['gateway_launch']['settings']['SHARED_UPSTREAM_CLIENT'] == '1'
            checked.append(output.name)
        return result

    monkeypatch.setattr(subprocess, 'run', run)
    test_full_trace_runner_and_adversarial_analysis(tmp_path, False, False, 'thread')
    assert checked == ['success', 'rejected', 'failed', 'scheduled']


@pytest.mark.parametrize('outcome', ['complete', 'connect_error', 'stream_error', 'cancel'])
def test_shared_client_survives_request_cleanup_and_closes_at_shutdown(gateway, monkeypatch, tmp_path, outcome):
    async def scenario():
        monkeypatch.setattr(gateway, 'SHARED_UPSTREAM_CLIENT', True)
        monkeypatch.setattr(gateway, 'GATEWAY_DIAGNOSTICS', True)
        monkeypatch.setattr(gateway, 'TELEMETRY_JSONL', str(tmp_path / 'gateway.jsonl'))
        monkeypatch.setattr(gateway, 'TELEMETRY_STATUS_JSON', str(tmp_path / 'status.json'))
        monkeypatch.setattr(gateway, 'estimate_request_cost', lambda payload: (1., 8, 128, False))
        counts = dict(constructed=0, closed=0, response_closed=0, requests=0)
        first, release = asyncio.Event(), asyncio.Event()
        real_client = httpx.AsyncClient

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield CONTENT
                first.set()
                if outcome == 'cancel' and counts['requests'] == 1:
                    await release.wait()
                if outcome == 'stream_error' and counts['requests'] == 1:
                    raise httpx.ReadError('stream failed')
                yield DONE

            async def aclose(self):
                counts['response_closed'] += 1

        def upstream(req):
            counts['requests'] += 1
            # A cookie received on the first response must not leak to the next user.
            assert 'cookie' not in req.headers
            if outcome == 'connect_error' and counts['requests'] == 1:
                raise httpx.ConnectError('connect failed')
            return httpx.Response(200, stream=Stream(), headers={
                'content-type': 'text/event-stream', 'set-cookie': 'private=first-user; Path=/'})

        class Client(real_client):
            async def aclose(self):
                counts['closed'] += 1
                await super().aclose()

        def factory():
            counts['constructed'] += 1
            return Client(transport=httpx.MockTransport(upstream), cookies=gateway.NoUpstreamCookies())

        monkeypatch.setattr(gateway, 'create_upstream_client', factory)
        async with gateway.lifespan(gateway.app):
            client = gateway.app.state.upstream_client
            assert (await gateway.snapshot_metrics())['shared_upstream_client'] is True
            for index in range(2):
                response = await gateway.proxy_chat_completions(request())
                if index == 0 and outcome == 'connect_error':
                    assert response.status_code == 502
                else:
                    async def consume():
                        return [chunk async for chunk in response.body_iterator]
                    task = asyncio.create_task(consume())
                    if index == 0 and outcome == 'cancel':
                        await first.wait()
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                    elif index == 0 and outcome == 'stream_error':
                        with pytest.raises(httpx.ReadError):
                            await task
                    else:
                        assert (await task)[-1] == DONE
                    # StreamingResponse can invoke the same cleanup through BackgroundTask.
                    await response.background()
                assert gateway.state['current_admitted_cost'] == 0
                assert gateway.state['current_in_flight'] == 0
                assert not client.is_closed and counts['closed'] == 0
            assert counts['constructed'] == 1
        assert counts['closed'] == 1 and client.is_closed
        assert counts['response_closed'] == (1 if outcome == 'connect_error' else 2)
        assert gateway.app.state.upstream_client is None
        import json
        events = [json.loads(line) for line in (tmp_path / 'gateway.jsonl').read_text().splitlines()]
        summaries = [row for row in events if row['event'] == 'request_diagnostics']
        assert len(summaries) == 2 and all(row['cleanup_complete'] for row in summaries)
        for row in summaries:
            assert row['phases']['client_construction']['status'] == 'not_reached'
            assert row['phases']['client_close']['status'] == 'not_reached'
            assert row['phases']['reservation_release']['status'] == 'completed'
    asyncio.run(scenario())


def test_cancel_one_shared_stream_preserves_another_stream_and_budget_api(gateway, monkeypatch):
    async def scenario():
        monkeypatch.setattr(gateway, 'SHARED_UPSTREAM_CLIENT', True)
        monkeypatch.setattr(gateway, 'estimate_request_cost', lambda payload: (1., 8, 128, False))
        began = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        streams = []

        class Stream(httpx.AsyncByteStream):
            def __init__(self, index):
                self.index, self.closes = index, 0

            async def __aiter__(self):
                yield CONTENT
                began[self.index].set()
                await release.wait()
                yield DONE

            async def aclose(self):
                self.closes += 1

        def upstream(req):
            stream = Stream(len(streams))
            streams.append(stream)
            return httpx.Response(200, stream=stream, headers={'content-type': 'text/event-stream'})

        client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        monkeypatch.setattr(gateway, 'create_upstream_client', lambda: client)
        async with gateway.lifespan(gateway.app):
            responses = [await gateway.proxy_chat_completions(request()) for _ in range(2)]
            async def consume(response):
                return [chunk async for chunk in response.body_iterator]
            tasks = [asyncio.create_task(consume(response)) for response in responses]
            await asyncio.gather(*(event.wait() for event in began))
            assert gateway.state['current_admitted_cost'] == 2
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert gateway.state['current_admitted_cost'] == 1 and not tasks[1].done() and not client.is_closed
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url='http://test') as admin:
                changed = await admin.post('/admin/budget', json={'budget': .5}, headers={'x-admin-api-key': 'test-key'})
                assert changed.status_code == 200 and changed.json()['current_admitted_cost'] == 1
            release.set()
            assert (await tasks[1])[-1] == DONE
            for response in responses:
                await response.background()
            assert gateway.state['current_admitted_cost'] == gateway.state['current_in_flight'] == 0
            assert [stream.closes for stream in streams] == [1, 1]
            assert not client.is_closed
        assert client.is_closed
    asyncio.run(scenario())


def test_shared_startup_failure_cleans_up_transport(gateway, monkeypatch, tmp_path):
    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
        monkeypatch.setattr(gateway, 'SHARED_UPSTREAM_CLIENT', True)
        monkeypatch.setattr(gateway, 'create_upstream_client', lambda: client)
        # Log initialization fails after transport construction.
        monkeypatch.setattr(gateway, 'TELEMETRY_JSONL', str(tmp_path))
        with pytest.raises(IsADirectoryError):
            async with gateway.lifespan(gateway.app):
                pytest.fail('startup should fail')
        assert client.is_closed and gateway.app.state.upstream_client is None
    asyncio.run(scenario())


def test_runner_records_only_opted_in_mode_and_overrides_inherited_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('SHARED_UPSTREAM_CLIENT', '1')
    args = SimpleNamespace(budget=96, vllm_url='http://localhost:8000', gateway_diagnostics=True,
                           shared_upstream_client=False)
    assert 'shared_upstream_client' not in replay_configuration(args)
    assert gateway_environment(args, {'model': 'm'}, tmp_path, 'key')['SHARED_UPSTREAM_CLIENT'] == '0'
    args.shared_upstream_client = True
    assert replay_configuration(args)['shared_upstream_client'] is True
    assert gateway_environment(args, {'model': 'm'}, tmp_path, 'key')['SHARED_UPSTREAM_CLIENT'] == '1'


def test_qualification_binds_shared_mode_and_keeps_original_protocol_unchanged(tmp_path):
    import json
    from pathlib import Path
    from experiment_protocol import collection_binding, load_protocol, capture_mismatches, capture_identity
    from trace_control import make_plan
    from trace_workload import sha256_file
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / 'experiments/shared-client-qualification-v1.json'
    document, _ = load_protocol(protocol_path)
    assert document['status'] == 'frozen' and 'acceptance' not in document
    assert len(document['runs']) == 2 and all(r['role'] == 'development' for r in document['runs'])
    for run in document['runs']:
        source = root / document['traces'][run['trace']]['path']
        trace = json.loads(source.read_text())
        declaration = document['schedules'][run['schedule']]
        scheduled = 'path' in declaration
        steps = json.loads((root / declaration['path']).read_text()) if scheduled else [dict(at_seconds=0., budget=96.)]
        settings = dict(document['replay'], trace=str(source), output_dir=str(tmp_path / run['name']),
                        budget=96., budget_schedule=declaration.get('path'), vllm_url='http://localhost:8000')
        recorded = collection_binding(protocol_path, run['name'], settings['output_dir'], settings,
                                      trace, sha256_file(source), steps)
        assert recorded['run'] == run['name']
        settings.pop('shared_upstream_client')
        plan = make_plan(settings, trace, sha256_file(source), steps)
        problems = capture_mismatches(document, run['name'], capture_identity(settings, trace, sha256_file(source), plan))
        assert any('shared_upstream_client' in p for p in problems)
    old = root / 'experiments/excitation-v1.json'
    assert sha256_file(old) == '83af37e98799a1850728359d79be06fc832c424da94efde7824867c8641c7993'
