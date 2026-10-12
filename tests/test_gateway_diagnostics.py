import asyncio
import gc
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import httpx
import pytest
from starlette.requests import Request

from test_dynamic_budget import gateway
from telemetry import GatewayDiagnostics, RequestDiagnostics, Telemetry


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = 'a' * 32
CONTENT = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
DONE = b'data: [DONE]\n\n'


def request(body=b'{"stream":true}', receive=None):
    async def body_receive():
        return {'type': 'http.request', 'body': body, 'more_body': False}
    return Request(dict(type='http', method='POST', path='/v1/chat/completions', headers=[
        (b'host', b'client-host'), (b'content-length', str(len(body)).encode()),
        (b'x-admin-api-key', b'not-forwarded'), (b'x-benchmark-request-id', IDENTIFIER.encode()),
        (b'x-test-header', b'preserved')]), receive or body_receive)


def phases_only(gateway):
    gateway.telemetry.enabled = True
    gateway.telemetry.diagnostics = GatewayDiagnostics(gateway.telemetry.emit)


def summaries(gateway):
    return [r for r in gateway.telemetry.records if r['event'] == 'request_diagnostics']


@pytest.mark.parametrize('outcome', ['complete', 'rejected', 'connect_error', 'truncated', 'stream_error'])
def test_off_on_forwarding_and_lifecycle_parity(gateway, monkeypatch, tmp_path, outcome):
    async def scenario():
        original_state = dict(gateway.state)
        real_client = httpx.AsyncClient
        results = []
        for enabled in (False, True):
            gateway.state.update(original_state)
            gateway.benchmark_ids.clear()
            gateway.telemetry = Telemetry()
            monkeypatch.setattr(gateway, 'BENCHMARK_REQUEST_IDS', True)
            monkeypatch.setattr(gateway, 'GATEWAY_DIAGNOSTICS', enabled)
            path = tmp_path / f'{enabled}.jsonl'
            monkeypatch.setattr(gateway, 'TELEMETRY_JSONL', str(path))
            monkeypatch.setattr(gateway, 'TELEMETRY_STATUS_JSON', str(tmp_path / f'{enabled}-status.json'))
            monkeypatch.setattr(gateway, 'estimate_request_cost',
                                lambda payload: (81 if outcome == 'rejected' else 75, 98, 128, False))
            counters = dict(clients=0, client_closes=0, stream_closes=0)
            builds, forwarded = [], []
            chunks = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n', CONTENT[:12], CONTENT[12:]]
            if outcome == 'complete':
                chunks.append(DONE)

            class Stream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    for chunk in chunks:
                        yield chunk
                    if outcome == 'stream_error':
                        raise httpx.ReadError('test stream failure')

                async def aclose(self):
                    counters['stream_closes'] += 1

            def upstream(sent):
                forwarded.append(dict(headers=dict(sent.headers), body=sent.content, method=sent.method))
                if outcome == 'connect_error':
                    raise httpx.ConnectError('test connect failure')
                return httpx.Response(200, stream=Stream(), headers={
                    'content-type': 'text/event-stream', 'content-length': '123',
                    'transfer-encoding': 'chunked', 'content-encoding': 'identity',
                    'x-request-id': 'not-the-client-id', 'x-upstream-test': 'preserved'})

            class Client(real_client):
                def __init__(self):
                    counters['clients'] += 1
                    super().__init__(transport=httpx.MockTransport(upstream))

                def build_request(self, **kwargs):
                    builds.append(kwargs)
                    return super().build_request(**kwargs)

                async def aclose(self):
                    counters['client_closes'] += 1
                    await super().aclose()

            monkeypatch.setattr(gateway.httpx, 'AsyncClient', Client)
            before_callbacks = tuple(gc.callbacks)
            async with gateway.lifespan(gateway.app):
                if not enabled:
                    assert gateway.telemetry.diagnostics is None
                    assert tuple(gc.callbacks) == before_callbacks
                response = await gateway.proxy_chat_completions(request())
                output = []
                if outcome not in {'rejected', 'connect_error'}:
                    assert not summaries(gateway)
                    try:
                        async for chunk in response.body_iterator:
                            output.append(chunk)
                    except httpx.ReadError:
                        assert outcome == 'stream_error'
                    await response.background()
                    assert output == chunks
                phase_rows = summaries(gateway)
                assert len(phase_rows) == int(enabled)
                if enabled:
                    row = phase_rows[0]
                    assert row['cleanup_complete']
                    assert row['outcome'] == ('completed' if outcome == 'complete' else
                                             'rejected' if outcome == 'rejected' else 'failed')
                    assert row['phases']['body_read']['status'] == 'completed'
                    if outcome == 'rejected':
                        assert row['phases']['client_construction']['status'] == 'not_reached'
                    else:
                        assert row['phases']['client_close']['status'] == 'completed'
                        assert row['phases']['reservation_release']['status'] == 'completed'
                    terminal = [r for r in gateway.telemetry.records
                                if r['event'] in {'completed', 'rejected', 'failed'}][0]
                    assert terminal['monotonic'] <= row['span_end'] <= row['monotonic']
                lifecycle = [
                    {k: v for k, v in row.items() if k not in
                     {'monotonic', 'timestamp', 'run_id', 'end_to_end', 'ttft'}}
                    for row in gateway.telemetry.records
                    if row['event'] in {'received', 'accepted', 'backend_sent', 'first_content',
                                        'completed', 'rejected', 'failed', 'cancelled'}]
                if builds:
                    assert len(builds) == 1 and builds[0]['timeout'] == 60.0
                    assert 'x-admin-api-key' not in builds[0]['headers']
                    assert 'x-benchmark-request-id' not in builds[0]['headers']
                    assert 'host' not in builds[0]['headers']
                    assert 'content-length' not in builds[0]['headers']
                    assert builds[0]['headers']['accept-encoding'] == 'identity'
                    assert builds[0]['headers']['x-request-id'] == IDENTIFIER
                assert gateway.state['current_admitted_cost'] == gateway.state['current_in_flight'] == 0
                results.append((response.status_code, dict(response.headers), output,
                                builds, forwarded, counters, lifecycle, dict(gateway.state)))
            assert tuple(gc.callbacks) == before_callbacks
            status = json.loads((tmp_path / f'{enabled}-status.json').read_text())
            assert status['status'] == 'closed'
        assert results[0] == results[1]
    asyncio.run(scenario())


@pytest.mark.parametrize('where', ['body_read', 'json_parse', 'cost_estimation', 'client_construction'])
def test_failed_phases_and_parse_fallback(gateway, monkeypatch, where):
    async def scenario():
        phases_only(gateway)
        async def receive():
            if where == 'body_read':
                raise RuntimeError('body failed')
            return {'type': 'http.request', 'body': b'{', 'more_body': False}
        def estimate(payload):
            if where == 'cost_estimation':
                raise RuntimeError('cost failed')
            assert payload == {}
            return (81 if where == 'json_parse' else 75, 98, 128, False)
        def construct():
            raise RuntimeError('construction failed')
        monkeypatch.setattr(gateway, 'estimate_request_cost', estimate)
        monkeypatch.setattr(gateway.httpx, 'AsyncClient', construct)
        if where == 'json_parse':
            response = await gateway.proxy_chat_completions(request(receive=receive))
            assert response.status_code == 503
        else:
            with pytest.raises(RuntimeError):
                await gateway.proxy_chat_completions(request(receive=receive))
        row, = summaries(gateway)
        assert row['phases'][where]['status'] == 'failed'
        assert row['cleanup_complete']
        assert not gateway.telemetry.active
        assert gateway.state['current_admitted_cost'] == 0
    asyncio.run(scenario())


@pytest.mark.parametrize('where', ['admission_lock_wait', 'http_send_headers'])
def test_cancelled_wait_phase(gateway, monkeypatch, where):
    async def scenario():
        phases_only(gateway)
        entered = asyncio.Event()
        if where == 'admission_lock_wait':
            await gateway.state_lock.acquire()
        else:
            real_client = httpx.AsyncClient
            async def upstream(sent):
                entered.set()
                await asyncio.Event().wait()
            monkeypatch.setattr(gateway.httpx, 'AsyncClient',
                                lambda: real_client(transport=httpx.MockTransport(upstream)))
        task = asyncio.create_task(gateway.proxy_chat_completions(request()))
        if where == 'http_send_headers':
            await entered.wait()
        else:
            await asyncio.sleep(0)
        task.cancel()
        if where == 'admission_lock_wait':
            gateway.state_lock.release()
        with pytest.raises(asyncio.CancelledError):
            await task
        row, = summaries(gateway)
        assert row['phases'][where]['status'] == 'cancelled'
        assert row['outcome'] == 'cancelled' and row['cleanup_complete']
        assert not gateway.telemetry.active
        assert gateway.state['current_admitted_cost'] == 0
    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['repeated_cancellation', 'close_error'])
def test_slow_close_exactly_once_and_terminal_summary(gateway, monkeypatch, failure):
    async def scenario():
        phases_only(gateway)
        closing, resume = asyncio.Event(), asyncio.Event()
        counters = dict(stream_close=0, client_close=0, release=0)
        real_client, real_finish = httpx.AsyncClient, gateway.finish_request

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield CONTENT
                await asyncio.Event().wait()

            async def aclose(self):
                counters['stream_close'] += 1
                closing.set()
                await resume.wait()
                if failure == 'close_error':
                    raise RuntimeError('close failed')

        class Client(real_client):
            async def aclose(self):
                counters['client_close'] += 1
                await super().aclose()

        async def finish(cost):
            counters['release'] += 1
            await real_finish(cost)

        monkeypatch.setattr(gateway, 'finish_request', finish)
        monkeypatch.setattr(gateway.httpx, 'AsyncClient', lambda: Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(
                200, stream=Stream(), headers={'content-type': 'text/event-stream'}))))
        response = await gateway.proxy_chat_completions(request())
        assert await anext(response.body_iterator) == CONTENT
        task = asyncio.create_task(response.body_iterator.aclose())
        await closing.wait()
        assert not summaries(gateway)
        if failure == 'repeated_cancellation':
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
        assert not summaries(gateway)
        resume.set()
        with pytest.raises(asyncio.CancelledError if failure == 'repeated_cancellation' else RuntimeError):
            await task
        if failure == 'repeated_cancellation':
            await response.background()
        else:
            with pytest.raises(RuntimeError):
                await response.background()
        row, = summaries(gateway)
        assert counters == dict(stream_close=1, client_close=1, release=1)
        assert row['outcome'] == 'cancelled'
        assert row['cleanup_complete'] is (failure != 'close_error')
        assert row['phases']['upstream_close']['status'] == ('failed' if failure == 'close_error' else 'completed')
        assert not gateway.telemetry.active and gateway.state['current_admitted_cost'] == 0
    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['sampler', 'flush', 'serialization', 'status'])
def test_shutdown_failure_removes_callback_and_surfaces_failure(gateway, monkeypatch, tmp_path, failure):
    async def scenario():
        path = tmp_path / 'gateway.jsonl'
        status_path = tmp_path / 'status.json'
        monkeypatch.setattr(gateway, 'GATEWAY_DIAGNOSTICS', True)
        monkeypatch.setattr(gateway, 'TELEMETRY_JSONL', str(path))
        monkeypatch.setattr(gateway, 'TELEMETRY_STATUS_JSON', str(status_path))
        before = tuple(gc.callbacks)
        if failure == 'sampler':
            async def fail(*args):
                raise RuntimeError('sampler failure')
            monkeypatch.setattr(gateway.telemetry, 'run', fail)
        with pytest.raises((RuntimeError, OSError, ValueError)):
            async with gateway.lifespan(gateway.app):
                if failure == 'flush':
                    path.unlink()
                    path.mkdir()
                elif failure == 'serialization':
                    gateway.telemetry.emit('invalid_test_record', invalid=float('nan'))
                elif failure == 'status':
                    monkeypatch.setattr(gateway, 'TELEMETRY_STATUS_JSON', str(tmp_path / 'absent' / 'status.json'))
        assert tuple(gc.callbacks) == before
        assert not gateway.telemetry.enabled
        if failure != 'status':
            status = json.loads(status_path.read_text())
            assert status['status'] == 'failed'
            assert status['diagnostics']['closed']
    asyncio.run(scenario())


def test_disabled_no_request_object_or_registration(gateway, monkeypatch):
    async def scenario():
        monkeypatch.setattr(gateway, 'GATEWAY_DIAGNOSTICS', False)
        def forbidden(*args, **kwargs):
            raise AssertionError('Diagnostics allocated while disabled')
        monkeypatch.setattr(gateway, 'GatewayDiagnostics', forbidden)
        monkeypatch.setattr(RequestDiagnostics, '__init__', forbidden)
        monkeypatch.setattr(gateway, 'estimate_request_cost', lambda p: (81, 98, 128, False))
        before = tuple(gc.callbacks)
        tasks_before = asyncio.all_tasks()
        async with gateway.lifespan(gateway.app):
            assert asyncio.all_tasks() == tasks_before
            assert (await gateway.proxy_chat_completions(request())).status_code == 503
        assert tuple(gc.callbacks) == before
    asyncio.run(scenario())


CALLBACK_CASES = {
    'bounded_loss': """
        d = GatewayDiagnostics(t.emit, capacity=4)
        d.start()
        for _ in range(7):
            d.callback('start', info)
            d.callback('stop', info)
        assert len(d.inbox) == 4
        await d.stop()
        h = d.health()
        assert h['callbacks_produced'] == 14 and h['callbacks_received'] == 4
        assert h['lost_callbacks'] == 10 and not h['complete']
        assert h['gc_spans'] == 2
    """,
    'pending_bound': """
        d = GatewayDiagnostics(t.emit, capacity=4)
        d.start()
        for _ in range(10):
            d.callback('start', info)
            d.drain()
        assert d.pending_count == 4
        await d.stop()
        h = d.health()
        assert h['callbacks_received'] == h['callbacks_produced'] == 10
        assert h['lost_callbacks'] == 0 and h['unmatched_boundaries'] == 10
        assert not h['complete']
    """,
    'reentrant': """
        from collections import deque
        class Reentrant(deque):
            inside = False
            def append(self, item):
                if not self.inside:
                    self.inside = True
                    d.callback('start', info)
                    d.callback('stop', info)
                    self.inside = False
                super().append(item)
        d = GatewayDiagnostics(t.emit, capacity=16)
        d.inbox = Reentrant(maxlen=16)
        d.start()
        d.callback('start', info)
        d.callback('stop', info)
        await d.stop()
        h = d.health()
        assert h['callbacks_produced'] == h['callbacks_received'] == 6
        assert h['gc_spans'] == 3 and h['complete']
    """,
    'threads_and_real_gc': """
        d = GatewayDiagnostics(t.emit, capacity=2048)
        d.start()
        def worker():
            for _ in range(100):
                d.callback('start', info)
                d.callback('stop', info)
        workers = [threading.Thread(target=worker) for _ in range(3)]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join(3)
            assert not worker_thread.is_alive()
        assert not any(r['event'] == 'gc_span' for r in t.records)
        d.drain()
        worker_spans = [r for r in t.records if r['event'] == 'gc_span']
        assert len(worker_spans) == 300 and not any(r['on_loop_thread'] for r in worker_spans)
        cycle = []; cycle.append(cycle); del cycle
        gc.collect(2)
        await d.stop()
        assert d.health()['complete']
        assert any(r['event'] == 'gc_span' and r['on_loop_thread'] for r in t.records)
        assert all(r['span_end'] <= r['monotonic'] for r in t.records if r['event'] == 'gc_span')
        cpu = [r for r in t.records if r['event'] == 'diagnostic_cpu']
        assert all(r['loop_thread_id'] == threading.get_ident() for r in cpu)
    """,
    'callback_allocation_failure': """
        from collections import deque
        class Broken(deque):
            def append(self, item):
                raise MemoryError('test allocation failure')
        d = GatewayDiagnostics(t.emit, capacity=4)
        d.inbox = Broken(maxlen=4)
        d.start()
        d.callback('start', info)
        await d.stop()
        h = d.health()
        assert h['callback_errors'] == h['lost_callbacks'] == 1
        assert not h['complete']
    """,
    'quiescence_and_late_delivery': """
        import telemetry
        for timeout in (False, True):
            reached, release = threading.Event(), threading.Event()
            d = GatewayDiagnostics(t.emit)
            d.start()
            real_clock = telemetry.time.thread_time
            calls = 0
            def clock():
                nonlocal calls
                if threading.get_ident() != d.loop_thread_id:
                    calls += 1
                    if calls == 2:
                        reached.set()
                        assert release.wait(3)
                return real_clock()
            telemetry.time.thread_time = clock
            def worker():
                d.callback('start', info)
                d.callback('stop', info)
            thread = threading.Thread(target=worker)
            thread.start()
            assert reached.wait(3)
            stopping = asyncio.create_task(d.stop(timeout=.01 if timeout else 1))
            await asyncio.sleep(.02 if timeout else .005)
            assert stopping.done() is timeout
            release.set()
            thread.join(3)
            assert not thread.is_alive()
            await stopping
            telemetry.time.thread_time = real_clock
            h = d.health()
            assert h['complete'] is not timeout
            if timeout:
                assert not h['quiesced'] and h['lost_callbacks'] is None
                assert h['buffered_callbacks'] == 1 and h['unmatched_boundaries'] == 1
            produced = h['callbacks_produced']
            d.callback('start', info)
            assert d.health()['callbacks_produced'] == produced
            assert d.callback not in gc.callbacks
    """,
}


@pytest.mark.parametrize('case', CALLBACK_CASES)
def test_callback_handoff_in_isolated_process(case):
    script = """\
import asyncio, gc, threading
from telemetry import GatewayDiagnostics, Telemetry
gc.disable()
async def scenario():
    t = Telemetry()
    t.enabled = True
    info = dict(generation=2, collected=0, uncollectable=0)
"""
    script += textwrap.indent(textwrap.dedent(CALLBACK_CASES[case]), '    ')
    script += '\nasyncio.run(scenario())\n'
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, text=True,
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unsupported_runtime_is_explicit(monkeypatch):
    monkeypatch.setattr(sys, '_is_gil_enabled', lambda: False, raising=False)
    with pytest.raises(RuntimeError, match='GIL'):
        GatewayDiagnostics(lambda *a, **kw: None)
